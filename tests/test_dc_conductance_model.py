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
    extract_average_dc_resistance,
    extract_dc_conductance_samples,
    parse_dc_port_paths,
    resolve_export_dc_conductance_model,
    train_dc_conductance_model,
    validate_dc_model_against_mdif,
    write_mdif,
    write_ads_hb_mlp_package,
    write_veriloga_package,
)


LABELS_2 = ["S11", "S12", "S21", "S22"]
LABELS_4 = [f"S{row}{col}" for row in range(1, 5) for col in range(1, 5)]
LABELS_3 = [f"S{row}{col}" for row in range(1, 4) for col in range(1, 4)]


def dc_block(
    parameter: float,
    conductances: list[float],
    paths: str,
    *,
    nports: int | None = None,
) -> MDIFBlock:
    nports = nports or (3 if len(conductances) == 3 else 2)
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


def dc_y_block(
    parameter: float,
    y_matrix: np.ndarray,
    labels: list[str] = LABELS_2,
) -> MDIFBlock:
    matrix = np.asarray(y_matrix, dtype=float)
    s_matrix = _y_matrix_to_s_matrix(matrix.astype(complex), 50.0)
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


def dc_s_block(parameter: float, s_matrix: np.ndarray) -> MDIFBlock:
    matrix = np.asarray(s_matrix, dtype=complex)
    sparams = {}
    for label in LABELS_2:
        row = int(label[1]) - 1
        col = int(label[2]) - 1
        sparams[label] = np.asarray([matrix[row, col], 0.1 + 0.02j])
    return MDIFBlock(
        params={"W": str(parameter)},
        freq_hz=np.asarray([0.0, 1.0e9]),
        sparams=sparams,
        source_index=int(parameter),
    )


def make_dc_nonpassive(block: MDIFBlock) -> MDIFBlock:
    for label in block.sparams:
        block.sparams[label] = block.sparams[label].copy()
        block.sparams[label][0] = 0.0
    block.sparams["S11"][0] = 2.0
    return block


class DCConductanceModelTests(unittest.TestCase):
    def test_legacy_dc_extraction_counts_contributing_blocks(self) -> None:
        metadata = extract_average_dc_resistance(
            [dc_block(1.0, [0.01], "1-2")],
            LABELS_2,
            port_paths="1-2",
        )
        self.assertEqual(metadata["dc_resistance_block_count"], 1)

    def test_legacy_path_parser_default_is_complete_passive_graph(self) -> None:
        self.assertEqual(
            [path[2] for path in parse_dc_port_paths(None, 2)],
            ["p1-p2", "p1-ground", "p2-ground"],
        )
        block = dc_block(
            1.0,
            [0.01, 0.02, 0.03],
            "1-2,1-ground,2-ground",
            nports=2,
        )
        _, conductances, metadata = extract_dc_conductance_samples(
            [block],
            ["W"],
            LABELS_2,
            port_paths=None,
        )
        np.testing.assert_allclose(
            conductances[0],
            np.asarray([0.01, 0.02, 0.03]),
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        self.assertLess(metadata["dc_topology_s_max_abs_error"], 1.0e-12)
        self.assertFalse(metadata["dc_port_paths_explicit"])
        self.assertEqual(
            metadata["dc_port_path_selection"],
            "automatic_complete_graph",
        )

    def test_unspecified_topology_fits_every_complex_ordered_dc_s_entry(self) -> None:
        y_matrix = np.asarray([[0.02, 0.005], [-0.005, 0.02]])
        blocks = [dc_y_block(1.0, y_matrix), dc_y_block(2.0, y_matrix)]
        model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=20,
            batch_size=2,
            learning_rate=0.01,
            patience=10,
            seed=14,
            progress_interval=0,
            port_paths=None,
        )
        self.assertEqual(model.representation, "full_s_matrix")
        self.assertEqual(model.port_paths, [])
        self.assertEqual(
            metadata["dc_matrix_entries"],
            [
                "S11.real", "S12.real", "S21.real", "S22.real",
                "S11.imag", "S12.imag", "S21.imag", "S22.imag",
            ],
        )
        np.testing.assert_allclose(
            model.conductance_matrix(np.asarray([1.0])),
            y_matrix,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model.predict_s_values(np.asarray([1.0])),
            np.asarray([blocks[0].sparams[label][0] for label in LABELS_2]),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        common = dict(
            model_kind="DNN",
            module_name="full_dc",
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
            dc_equivalent_resistance_ohm=metadata[
                "dc_equivalent_resistance_ohm"
            ],
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm=metadata["dc_port_resistances_ohm"],
            dc_model=model.export_data(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model.save(root / "model")
            loaded = DCConductanceModel.load_optional(root / "model")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.representation, "full_s_matrix")
            np.testing.assert_allclose(
                loaded.conductance_matrix(np.asarray([1.0])),
                y_matrix,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
            hb_manifest = write_ads_hb_mlp_package(out_dir=root / "hb", **common)
            hb_text = (root / "hb" / "full_dc.net").read_text()
            self.assertNotIn("_dc_g0=exp(", hb_text)
            self.assertEqual(
                hb_manifest["dc_matrix_entries"],
                [
                    "S11.real", "S12.real", "S21.real", "S22.real",
                    "S11.imag", "S12.imag", "S21.imag", "S22.imag",
                ],
            )
            self.assertEqual(hb_manifest["dc_port_paths"], [])
            self.assertEqual(
                hb_manifest["dc_sparameter_entries"],
                ["S11", "S12", "S21", "S22"],
            )
            self.assertIsNone(hb_manifest["dc_port_resistances_ohm"])
            self.assertEqual(
                hb_manifest["dc_stamping_representation"],
                "separate_full_ordered_complex_s_to_y_sdd",
            )

            va_manifest = write_veriloga_package(
                out_dir=root / "va",
                frequency_expression="$freq",
                **common,
            )
            va_text = (root / "va" / "full_dc.va").read_text()
            self.assertIn("real dc_sr [0:3];", va_text)
            self.assertIn("real dc_si [0:3];", va_text)
            self.assertIn("active_yr[1] = dc_yr[1]", va_text)
            self.assertEqual(
                va_manifest["dc_matrix_entries"],
                [
                    "S11.real", "S12.real", "S21.real", "S22.real",
                    "S11.imag", "S12.imag", "S21.imag", "S22.imag",
                ],
            )
            self.assertEqual(va_manifest["dc_port_paths"], [])
            self.assertEqual(
                va_manifest["dc_sparameter_entries"],
                ["S11", "S12", "S21", "S22"],
            )
            self.assertIsNone(va_manifest["dc_port_resistances_ohm"])
            readme = (root / "va" / "VERILOGA_README.md").read_text()
            self.assertIn("fits all `8` real/imaginary components", readme)

    def test_default_dc_preserves_complex_s_without_projection_error(self) -> None:
        s_matrix = np.asarray(
            [[0.2 + 0.1j, 0.15 - 0.05j], [0.05 + 0.08j, -0.1 + 0.12j]]
        )
        blocks = [dc_s_block(1.0, s_matrix), dc_s_block(2.0, s_matrix)]
        model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=5,
            batch_size=2,
            learning_rate=0.01,
            patience=5,
            seed=17,
            progress_interval=0,
            port_paths=None,
        )
        self.assertEqual(metadata["dc_topology_s_max_abs_error"], 0.0)
        np.testing.assert_allclose(
            model.predict_s_values(np.asarray([1.0])),
            s_matrix.reshape(-1),
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        validation = validate_dc_model_against_mdif(model, blocks)
        self.assertLess(validation["dc_mdif_model_s_max_abs_error"], 1.0e-12)
        common = dict(
            model_kind="DNN",
            module_name="complex_dc",
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
            dc_equivalent_resistance_ohm=metadata[
                "dc_equivalent_resistance_ohm"
            ],
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={},
            dc_model=model.export_data(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_veriloga_package(
                out_dir=root / "va",
                frequency_expression="$freq",
                **common,
            )
            va_text = (root / "va" / "complex_dc.va").read_text()
            self.assertIn("dc_si[0]", va_text)
            self.assertIn("active_yi[0] = dc_yi[0]", va_text)
            hb_manifest = write_ads_hb_mlp_package(
                out_dir=root / "hb",
                **common,
            )
            hb_text = (root / "hb" / "complex_dc.net").read_text()
            self.assertIn("_dc_s0=complex(", hb_text)
            self.assertEqual(
                hb_manifest["dc_stamping_representation"],
                "separate_full_ordered_complex_s_to_y_sdd",
            )

    def test_unspecified_four_port_model_has_all_sixteen_ordered_entries(self) -> None:
        skew = np.asarray(
            [
                [0.0, 1.0, 2.0, 3.0],
                [-1.0, 0.0, 4.0, 5.0],
                [-2.0, -4.0, 0.0, 6.0],
                [-3.0, -5.0, -6.0, 0.0],
            ]
        )
        y_matrix = 0.02 * np.eye(4) + 0.0002 * skew
        blocks = [
            dc_y_block(1.0, y_matrix, LABELS_4),
            dc_y_block(2.0, y_matrix, LABELS_4),
        ]
        model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_4,
            hidden_layers=[2],
            activation="tanh",
            epochs=5,
            batch_size=2,
            learning_rate=0.01,
            patience=5,
            seed=15,
            progress_interval=0,
            port_paths=None,
        )
        expected_entries = [
            f"S{row}{col}.{component}"
            for component in ("real", "imag")
            for row in range(1, 5)
            for col in range(1, 5)
        ]
        self.assertEqual(metadata["dc_matrix_entries"], expected_entries)
        self.assertEqual(model.mlp.layer_sizes[-1], 32)
        np.testing.assert_allclose(
            model.conductance_matrix(np.asarray([1.0])),
            y_matrix,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

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
        with self.assertRaisesRegex(ValueError, "Every DC-training geometry"):
            extract_dc_conductance_samples(
                [block],
                ["W"],
                LABELS_2,
                port_paths="1-2",
            )

    def test_verification_blocks_without_dc_are_not_required_for_dc_fit(self) -> None:
        train_blocks = [
            dc_block(1.0, [0.01], "1-2"),
            dc_block(2.0, [0.01], "1-2"),
        ]
        verification = dc_block(3.0, [0.01], "1-2")
        verification.freq_hz = np.asarray([1.0, 1.0e9])
        model, _, metadata = train_dc_conductance_model(
            train_blocks,
            [verification],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=20,
            batch_size=2,
            learning_rate=0.01,
            patience=10,
            seed=9,
            progress_interval=0,
            port_paths="1-2",
        )
        self.assertIsNotNone(model)
        self.assertEqual(metadata["dc_model_verification_samples"], 0)

    def test_nonpassive_dc_training_block_is_excluded(self) -> None:
        bad = make_dc_nonpassive(dc_block(1.0, [0.01], "1-2"))
        good = dc_block(2.0, [0.02], "1-2")
        bad.source_index = 0
        good.source_index = 1
        model, _, metadata = train_dc_conductance_model(
            [bad, good],
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=20,
            batch_size=2,
            learning_rate=0.01,
            patience=10,
            seed=11,
            progress_interval=0,
            port_paths="1-2",
        )
        self.assertIsNotNone(model)
        self.assertEqual(metadata["dc_model_training_samples"], 1)
        self.assertEqual(metadata["dc_unusable_block_count"], 1)
        self.assertEqual(metadata["dc_unusable_block_positions"], [1])
        self.assertEqual(metadata["dc_usable_block_positions"], [2])

    def test_dc_fit_fails_only_when_every_dc_sample_is_unusable(self) -> None:
        bad = make_dc_nonpassive(dc_block(1.0, [0.01], "1-2"))
        with self.assertRaisesRegex(ValueError, "a DC model cannot be fitted"):
            train_dc_conductance_model(
                [bad],
                [],
                ["W"],
                LABELS_2,
                hidden_layers=[2],
                activation="tanh",
                epochs=20,
                batch_size=1,
                learning_rate=0.01,
                patience=10,
                seed=12,
                progress_interval=0,
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
            hb_manifest = write_ads_hb_mlp_package(
                out_dir=root / "hb",
                emit_combined_sdd_trial=True,
                **common,
            )
            hb_text = (root / "hb" / "dynamic_dc.net").read_text()
            self.assertIn("_dc_g0=exp(", hb_text)
            self.assertRegex(hb_text, r"then dynamic_dc_m_dc_y0 else 0\.0")
            self.assertTrue(hb_manifest["dc_geometry_dependent"])
            hb_trial_text = (
                root / "hb" / "dynamic_dc_combined_sdd_trial.net"
            ).read_text()
            self.assertIn(
                "SDD:dynamic_dc_combined_sdd_trial_core_combined",
                hb_trial_text,
            )
            self.assertNotIn(
                "SDD:dynamic_dc_combined_sdd_trial_core_dc",
                hb_trial_text,
            )
            self.assertRegex(
                hb_trial_text,
                r"then dynamic_dc_combined_sdd_trial_m_dc_y0 else "
                r"dynamic_dc_combined_sdd_trial_m_stoy_y_0_0",
            )

            va_manifest = write_veriloga_package(
                out_dir=root / "va",
                frequency_expression="$freq",
                **common,
            )
            va_text = (root / "va" / "dynamic_dc.va").read_text()
            self.assertIn("dc_g[0] = exp(", va_text)
            self.assertIn("active_yr[0] = dc_g[0]", va_text)
            self.assertTrue(va_manifest["dc_geometry_dependent"])

    def test_export_dc_mdif_validates_and_keeps_saved_dynamic_model(self) -> None:
        blocks = [
            dc_block(1.0, [0.01], "1-2"),
            dc_block(2.0, [0.01], "1-2"),
        ]
        model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=20,
            batch_size=2,
            learning_rate=0.01,
            patience=10,
            seed=7,
            progress_interval=0,
            port_paths="1-2",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "dc.mdif"
            write_mdif(dc_mdif, blocks, LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                model,
                metadata,
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths="1-2",
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIs(resolved, model)
        self.assertEqual(
            export_metadata["dc_mdif_action"],
            "validated_saved_dc_model",
        )
        self.assertLess(export_metadata["dc_mdif_model_s_max_abs_error"], 1.0e-12)

    def test_export_dc_mdif_fits_dc_only_model_for_legacy_model(self) -> None:
        blocks = [
            dc_block(1.0, [0.02], "1-2"),
            dc_block(2.0, [0.02], "1-2"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "dc.mdif"
            write_mdif(dc_mdif, blocks, LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                None,
                {},
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths="1-2",
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIsNotNone(resolved)
        self.assertEqual(export_metadata["dc_mdif_action"], "fitted_dc_only_model")
        self.assertTrue(export_metadata["dc_model_fitted_during_export"])
        self.assertLess(export_metadata["dc_mdif_model_s_max_abs_error"], 1.0e-12)

    def test_export_without_path_flag_upgrades_pair_only_saved_model_to_full_s(self) -> None:
        old_blocks = [
            dc_block(1.0, [0.01], "1-2"),
            dc_block(2.0, [0.01], "1-2"),
        ]
        old_model, _, old_metadata = train_dc_conductance_model(
            old_blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=20,
            batch_size=2,
            learning_rate=0.01,
            patience=10,
            seed=13,
            progress_interval=0,
            port_paths="1-2",
        )
        generic_blocks = [
            dc_block(
                1.0,
                [0.01, 0.02, 0.03],
                "1-2,1-ground,2-ground",
                nports=2,
            ),
            dc_block(
                2.0,
                [0.01, 0.02, 0.03],
                "1-2,1-ground,2-ground",
                nports=2,
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "generic_dc.mdif"
            write_mdif(dc_mdif, generic_blocks, LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                old_model,
                old_metadata,
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths=None,
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.representation, "full_s_matrix")
        self.assertEqual(resolved.port_paths, [])
        self.assertEqual(
            export_metadata["dc_matrix_entries"],
            [
                "S11.real", "S12.real", "S21.real", "S22.real",
                "S11.imag", "S12.imag", "S21.imag", "S22.imag",
            ],
        )
        self.assertEqual(export_metadata["dc_mdif_action"], "fitted_dc_only_model")
        self.assertLess(export_metadata["dc_mdif_model_s_max_abs_error"], 1.0e-12)

    def test_legacy_automatic_path_model_requires_mdif_for_full_s_upgrade(self) -> None:
        blocks = [
            dc_block(1.0, [0.01], "1-2"),
            dc_block(2.0, [0.01], "1-2"),
        ]
        old_model, _, old_metadata = train_dc_conductance_model(
            blocks,
            [],
            ["W"],
            LABELS_2,
            hidden_layers=[2],
            activation="tanh",
            epochs=5,
            batch_size=2,
            learning_rate=0.01,
            patience=5,
            seed=16,
            progress_interval=0,
            port_paths="1-2",
        )
        old_metadata["dc_port_paths_explicit"] = False
        with self.assertRaisesRegex(ValueError, "Supply --dc-mdif to upgrade"):
            resolve_export_dc_conductance_model(
                old_model,
                old_metadata,
                ["W"],
                LABELS_2,
                dc_mdif=None,
                z0=50.0,
                port_paths=None,
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )

    def test_export_dc_mdif_excludes_nonpassive_training_block(self) -> None:
        blocks = [
            make_dc_nonpassive(dc_block(1.0, [0.02], "1-2")),
            dc_block(2.0, [0.02], "1-2"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "dc.mdif"
            write_mdif(dc_mdif, blocks, LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                None,
                {},
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths="1-2",
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIsNotNone(resolved)
        self.assertEqual(export_metadata["dc_mdif_validation_input_block_count"], 2)
        self.assertEqual(export_metadata["dc_mdif_validation_block_count"], 1)
        self.assertEqual(export_metadata["dc_mdif_excluded_unusable_block_count"], 1)
        self.assertEqual(export_metadata["dc_mdif_excluded_unusable_block_positions"], [1])

    def test_export_dc_mdif_uses_only_training_split_from_combined_file(self) -> None:
        training = [
            dc_block(1.0, [0.02], "1-2"),
            dc_block(2.0, [0.02], "1-2"),
        ]
        verification = [
            dc_block(3.0, [0.02], "1-2"),
            dc_block(4.0, [0.02], "1-2"),
        ]
        for block in training:
            block.params["Dataset"] = "train"
        for block in verification:
            block.params["Dataset"] = "verify"
            block.freq_hz = np.asarray([1.0, 1.0e9])
        combined = [*training, *verification]
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "combined.mdif"
            write_mdif(dc_mdif, combined, LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                None,
                {
                    "split_var": "dataset",
                    "train_values": ["train", "training"],
                    "verify_values": ["verify", "verification"],
                },
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths="1-2",
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIsNotNone(resolved)
        self.assertEqual(export_metadata["dc_mdif_total_block_count"], 4)
        self.assertEqual(export_metadata["dc_mdif_training_block_count"], 2)
        self.assertEqual(
            export_metadata["dc_mdif_excluded_verification_block_count"],
            2,
        )

    def test_export_rejects_dc_topology_that_cannot_match_mdif(self) -> None:
        blocks = [dc_block(1.0, [0.01, 0.02, 0.03], "1-2,1-3,2-3")]
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "dc.mdif"
            write_mdif(dc_mdif, blocks, LABELS_3)
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                resolve_export_dc_conductance_model(
                    None,
                    {},
                    ["W"],
                    LABELS_3,
                    dc_mdif=dc_mdif,
                    z0=50.0,
                    port_paths="1-2",
                    open_threshold_ohm=1.0e12,
                    open_resistance_ohm=1.0e19,
                    activation="tanh",
                    hidden_layers=[2],
                )

    def test_unrestricted_full_s_export_reports_fit_error_without_stopping(self) -> None:
        first = dc_s_block(1.0, np.asarray([[0.2, 0.1], [0.1, 0.2]]))
        second = dc_s_block(1.0, np.asarray([[-0.2, 0.1], [0.1, -0.2]]))
        first.source_index = 0
        second.source_index = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            dc_mdif = Path(temp_dir) / "conflicting_dc.mdif"
            write_mdif(dc_mdif, [first, second], LABELS_2)
            resolved, export_metadata = resolve_export_dc_conductance_model(
                None,
                {},
                ["W"],
                LABELS_2,
                dc_mdif=dc_mdif,
                z0=50.0,
                port_paths=None,
                open_threshold_ohm=1.0e12,
                open_resistance_ohm=1.0e19,
                activation="tanh",
                hidden_layers=[2],
            )
        self.assertIsNotNone(resolved)
        self.assertFalse(export_metadata["dc_mdif_match_within_tolerance"])
        self.assertIn("export continued", export_metadata["dc_mdif_warning"])
        self.assertGreater(export_metadata["dc_mdif_model_s_max_abs_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
