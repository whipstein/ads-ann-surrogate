import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from surrogate_common import (
    DCConductanceModel,
    MDIFBlock,
    _dc_matrix_from_path_conductances,
    _y_matrix_to_s_matrix,
    extract_dc_conductance_samples,
    parse_dc_port_paths,
    train_dc_conductance_model,
    write_ads_hb_mlp_package,
    write_veriloga_package,
)


LABELS_2 = ["S11", "S12", "S21", "S22"]
LABELS_3 = [f"S{row}{col}" for row in range(1, 4) for col in range(1, 4)]


def dc_block(parameter: float, conductances: list[float], paths: str) -> MDIFBlock:
    nports = 3 if len(conductances) == 3 else 2
    labels = LABELS_3 if nports == 3 else LABELS_2
    parsed_paths = parse_dc_port_paths(paths, nports)
    y_matrix = _dc_matrix_from_path_conductances(
        nports,
        parsed_paths,
        conductances,
    ).astype(complex)
    s_matrix = _y_matrix_to_s_matrix(y_matrix, 50.0)
    sparams = {}
    for label in labels:
        row = int(label[1]) - 1
        col = int(label[2]) - 1
        sparams[label] = np.asarray([s_matrix[row, col], 0.1 + 0.02j])
    return MDIFBlock(
        params={"W": str(parameter)},
        freq_hz=np.asarray([0.0, 1.0e9]),
        sparams=sparams,
        source_index=int(parameter),
    )


class DCConductanceModelTests(unittest.TestCase):
    def test_joint_topology_projection_recovers_shared_node_branches(self) -> None:
        blocks = [dc_block(1.0, [0.01, 0.02, 0.03], "1-2,1-3,2-3")]
        _, conductances, metadata = extract_dc_conductance_samples(
            blocks,
            ["W"],
            LABELS_3,
            port_paths="1-2,1-3,2-3",
        )
        np.testing.assert_allclose(
            conductances[0],
            np.asarray([0.01, 0.02, 0.03]),
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        self.assertLess(metadata["dc_topology_s_max_abs_error"], 1.0e-12)

    def test_every_fitted_geometry_requires_an_exact_dc_row(self) -> None:
        block = dc_block(1.0, [0.01], "1-2")
        block.freq_hz = np.asarray([1.0, 1.0e9])
        with self.assertRaisesRegex(ValueError, "Every fitted geometry"):
            extract_dc_conductance_samples(
                [block],
                ["W"],
                LABELS_2,
                port_paths="1-2",
            )

    def test_training_save_load_and_geometry_only_prediction(self) -> None:
        blocks = [
            dc_block(1.0, [0.01], "1-2"),
            dc_block(2.0, [0.02], "1-2"),
            dc_block(3.0, [0.03], "1-2"),
        ]
        model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[6],
            activation="tanh",
            epochs=400,
            batch_size=3,
            learning_rate=0.02,
            patience=100,
            seed=3,
            progress_interval=0,
            port_paths="1-2",
        )
        prediction = model.predict_conductances(np.asarray([[1.0], [3.0]])).reshape(-1)
        np.testing.assert_allclose(prediction, np.asarray([0.01, 0.03]), rtol=0.02)
        self.assertEqual(
            metadata["dc_model_kind"],
            "geometry_dependent_exact_dc_conductance_mlp",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save(Path(temp_dir))
            loaded = DCConductanceModel.load_optional(Path(temp_dir))
            self.assertIsNotNone(loaded)
            np.testing.assert_allclose(
                loaded.predict_conductances(np.asarray([[2.0]])),
                model.predict_conductances(np.asarray([[2.0]])),
            )

    def test_dynamic_dc_is_embedded_in_hb_and_veriloga(self) -> None:
        dc_model = {
            "kind": "geometry_dependent_exact_dc_conductance_mlp",
            "parameter_names": ["W"],
            "sparam_labels": LABELS_2,
            "port_paths": ["p1-p2"],
            "z0": 50.0,
            "activation": "tanh",
            "layer_sizes": [1, 1],
            "weights": [np.asarray([[1.0]])],
            "biases": [np.asarray([0.0])],
            "x_mean": np.asarray([0.0]),
            "x_std": np.asarray([1.0]),
            "y_mean": np.asarray([math.log(0.01)]),
            "y_std": np.asarray([0.1]),
            "log_conductance_min": math.log(1.0e-19),
            "log_conductance_max": math.log(1.0e3),
            "metadata": {},
        }
        common = dict(
            model_kind="DNN",
            module_name="dynamic_dc",
            parameter_names=["W"],
            sparam_labels=LABELS_2,
            freq_transform="log",
            activation="tanh",
            layer_sizes=[2, 8],
            weights=[np.zeros((2, 8))],
            biases=[np.zeros(8)],
            x_mean=np.asarray([1.0, 9.0]),
            x_std=np.ones(2),
            y_mean=np.zeros(8),
            y_std=np.ones(8),
            z0=50.0,
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={"p1-p2": 100.0},
            dc_model=dc_model,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            hb_manifest = write_ads_hb_mlp_package(out_dir=root / "hb", **common)
            hb_text = (root / "hb" / "dynamic_dc.net").read_text()
            self.assertIn("_dc_g0=exp(", hb_text)
            self.assertRegex(hb_text, r"then dynamic_dc_m_dc_y0 else 0\.0")
            self.assertTrue(hb_manifest["dc_geometry_dependent"])

            va_manifest = write_veriloga_package(
                out_dir=root / "va",
                frequency_expression="$freq",
                **common,
            )
            va_text = (root / "va" / "dynamic_dc.va").read_text()
            self.assertIn("dc_g[0] = exp(", va_text)
            self.assertIn("active_yr[0] = dc_g[0]", va_text)
            self.assertTrue(va_manifest["dc_geometry_dependent"])


if __name__ == "__main__":
    unittest.main()
