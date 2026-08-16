import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import kbnn
from surrogate_common import (
    MDIFBlock,
    MLP,
    Standardizer,
    write_ads_hb_mlp_package,
    write_mdif,
    write_veriloga_package,
)


LABELS = ["S11", "S12", "S21", "S22"]


class KBNNConditioningTests(unittest.TestCase):
    def test_structural_defaults_and_log_linear_optimize_candidate(self) -> None:
        parser = kbnn.build_arg_parser()
        train = parser.parse_args(
            [
                "train",
                "--mdif",
                "fine.mdif",
                "--coarse-model-dir",
                "coarse",
                "--out-dir",
                "model",
            ]
        )
        self.assertEqual(train.passivity_mode, "auto")
        self.assertEqual(train.reciprocity_mode, "enforce")

        optimize = parser.parse_args(
            [
                "optimize",
                "--mdif",
                "fine.mdif",
                "--coarse-model-dir",
                "coarse",
                "--out-dir",
                "sweep",
                "--freq-transform",
                "log-linear",
            ]
        )
        self.assertEqual(optimize.freq_transform_options, "log-linear")
        candidates = kbnn.sweep_candidate_grid(optimize)
        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate["freq_transform"] == "log-linear" for candidate in candidates)
        )

    def test_physical_weights_cancel_residual_standardization(self) -> None:
        scaler = Standardizer()
        scaler.mean = np.zeros(8)
        scaler.std = np.asarray([0.2, 1.0, 3.0, 0.5, 2.0, 0.4, 0.1, 4.0])
        requested = {"S11": 3.0, "S12": 0.5, "S21": 2.0, "S22": 1.0}
        weights = kbnn.physical_response_output_weights(
            LABELS,
            requested,
            scaler,
        )
        effective = weights / scaler.std**2
        expected = np.asarray([3.0, 0.5, 2.0, 1.0] * 2)
        np.testing.assert_allclose(
            effective / effective.mean(),
            expected / expected.mean(),
        )

    def test_composite_residual_passivity_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(211)
        scaler = Standardizer()
        scaler.mean = rng.normal(size=8) * 0.1
        scaler.std = np.exp(rng.normal(size=8))
        coarse = rng.normal(size=(4, 8)) * 0.15
        callback = kbnn.make_kbnn_composite_passivity_loss_gradient(
            scaler,
            LABELS,
            "residual",
            coarse,
            target_sigma=0.55,
            penalty=2.3,
        )
        values = rng.normal(size=(3, 8))
        indices = np.asarray([3, 0, 2])
        sample_weights = np.asarray([0.8, 1.0, 1.2])
        _loss, gradient = callback(
            values,
            np.zeros_like(values),
            sample_weights,
            indices,
        )
        finite_difference = np.zeros_like(values)
        step = 1e-6
        for row, col in np.ndindex(values.shape):
            plus = values.copy()
            minus = values.copy()
            plus[row, col] += step
            minus[row, col] -= step
            plus_loss, _ = callback(
                plus,
                np.zeros_like(values),
                sample_weights,
                indices,
            )
            minus_loss, _ = callback(
                minus,
                np.zeros_like(values),
                sample_weights,
                indices,
            )
            finite_difference[row, col] = (plus_loss - minus_loss) / (2.0 * step)
        np.testing.assert_allclose(
            gradient,
            finite_difference,
            rtol=3e-5,
            atol=2e-6,
        )

    def test_rf_response_scale_does_not_change_exact_dc(self) -> None:
        mlp = MLP([2, 2], activation="tanh", seed=3)
        mlp.weights[0][...] = 0.0
        mlp.biases[0][...] = 0.0
        x_scaler = Standardizer()
        x_scaler.mean = np.zeros(2)
        x_scaler.std = np.ones(2)
        y_scaler = Standardizer()
        y_scaler.mean = np.asarray([0.8, 0.0])
        y_scaler.std = np.ones(2)
        model = kbnn.KBNN(
            mlp,
            x_scaler,
            y_scaler,
            ["x"],
            ["S11"],
            "plain",
            False,
            "log",
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            rf_response_scale=0.5,
        )
        block = MDIFBlock(
            params={"x": "0"},
            freq_hz=np.asarray([0.0, 1.0e9]),
            sparams={"S11": np.zeros(2, dtype=complex)},
            source_index=0,
        )
        predicted = model.predict_blocks([block], [])[0].sparams["S11"]
        self.assertAlmostEqual(predicted[0].real, 1.0 / 3.0)
        self.assertAlmostEqual(predicted[1].real, 0.4)

        with tempfile.TemporaryDirectory() as temp_dir:
            model.save(Path(temp_dir), metadata={})
            loaded = kbnn.KBNN.load(Path(temp_dir))
            self.assertEqual(loaded.rf_response_scale, 0.5)

    def test_exporters_scale_the_reconstructed_rf_response(self) -> None:
        fine_weights = [np.zeros((2, 8))]
        fine_biases = [np.zeros(8)]
        coarse = {
            "parameter_names": ["W"],
            "sparam_labels": LABELS,
            "freq_transform": "log",
            "activation": "tanh",
            "layer_sizes": [2, 8],
            "weights": [np.zeros((2, 8))],
            "biases": [np.zeros(8)],
            "x_mean": np.asarray([1.0, 9.0]),
            "x_std": np.ones(2),
            "y_mean": np.zeros(8),
            "y_std": np.ones(8),
            "output_domain": "s",
        }
        common = dict(
            model_kind="KBNN",
            parameter_names=["W"],
            sparam_labels=LABELS,
            freq_transform="log",
            activation="tanh",
            layer_sizes=[2, 8],
            weights=fine_weights,
            biases=fine_biases,
            x_mean=np.asarray([1.0, 9.0]),
            x_std=np.ones(2),
            y_mean=np.zeros(8),
            y_std=np.ones(8),
            z0=50.0,
            adds_coarse_to_output=True,
            rf_response_scale=0.5,
            embedded_coarse_model=coarse,
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={"p1-p2": 100.0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            va_manifest = write_veriloga_package(
                out_dir=root / "va",
                module_name="scaled_va",
                frequency_expression="$freq",
                **common,
            )
            va_text = (root / "va" / "scaled_va.va").read_text()
            self.assertIn("*(y[0] + cr[0])", va_text)
            self.assertEqual(va_manifest["rf_response_scale"], 0.5)

            hb_manifest = write_ads_hb_mlp_package(
                out_dir=root / "hb",
                module_name="scaled_hb",
                **common,
            )
            hb_text = (root / "hb" / "scaled_hb.net").read_text()
            self.assertIn(
                "5.00000000000000000e-01)*((scaled_hb_m_coarse_out0)+(scaled_hb_m_fine_out0))",
                hb_text,
            )
            self.assertIn("5.00000000000000000e-01", hb_text)
            self.assertEqual(hb_manifest["rf_response_scale"], 0.5)

    def test_integrated_lossless_line_fit_is_passive_and_reciprocal(self) -> None:
        frequencies = np.concatenate(
            [np.asarray([0.0]), np.linspace(2.0e8, 8.0e9, 31)]
        )
        fine_blocks = []
        coarse_blocks = []
        train_values = np.linspace(0.0, 1.0, 8)
        verify_values = np.asarray([0.2, 0.5, 0.8])
        for source_index, (dataset, x_value) in enumerate(
            [
                *(("train", value) for value in train_values),
                *(("verification", value) for value in verify_values),
            ]
        ):
            fine_delay = 18.0e-12 + 18.0e-12 * float(x_value)
            coarse_delay = 17.0e-12 + 16.0e-12 * float(x_value)
            fine_t = np.exp(-1j * 2.0 * np.pi * frequencies * fine_delay)
            coarse_t = np.exp(-1j * 2.0 * np.pi * frequencies * coarse_delay)
            zero = np.zeros_like(fine_t)
            params = {"dataset": dataset, "x": str(float(x_value))}
            fine_blocks.append(
                MDIFBlock(
                    params=params,
                    freq_hz=frequencies.copy(),
                    sparams={
                        "S11": zero.copy(),
                        "S12": fine_t.copy(),
                        "S21": fine_t.copy(),
                        "S22": zero.copy(),
                    },
                    source_index=source_index,
                )
            )
            coarse_blocks.append(
                MDIFBlock(
                    params=params,
                    freq_hz=frequencies.copy(),
                    sparams={
                        "S11": zero.copy(),
                        "S12": coarse_t.copy(),
                        "S21": coarse_t.copy(),
                        "S22": zero.copy(),
                    },
                    source_index=source_index,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fine_path = root / "fine.mdif"
            coarse_path = root / "coarse.mdif"
            model_dir = root / "model"
            write_mdif(fine_path, fine_blocks, LABELS)
            write_mdif(coarse_path, coarse_blocks, LABELS)
            args = kbnn.build_arg_parser().parse_args(
                [
                    "train",
                    "--mdif",
                    str(fine_path),
                    "--coarse-mdif",
                    str(coarse_path),
                    "--out-dir",
                    str(model_dir),
                    "--parameter-names",
                    "x",
                    "--mode",
                    "residual",
                    "--include-coarse-input",
                    "--freq-transform",
                    "log-linear",
                    "--hidden-layers",
                    "48,48",
                    "--coarse-hidden-layers",
                    "48,48",
                    "--epochs",
                    "350",
                    "--coarse-epochs",
                    "350",
                    "--patience",
                    "120",
                    "--progress-interval",
                    "0",
                    "--worst-plots",
                    "0",
                    "--quiet",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(kbnn.command_train(args), 0)

            metadata = json.loads((model_dir / "metadata.json").read_text())
            summary = json.loads(
                (model_dir / "verification_summary.json").read_text()
            )
            composite = json.loads(
                (model_dir / kbnn.COMPOSITE_MANIFEST_FILENAME).read_text()
            )
            self.assertTrue(metadata["passivity_enforced"])
            self.assertTrue(metadata["reciprocity_enforced"])
            self.assertLessEqual(
                float(
                    metadata["predicted_train_passivity_after_scale"][
                        "max_singular_value"
                    ]
                ),
                float(metadata["passivity_target_sigma"]) + 1e-12,
            )
            self.assertLessEqual(
                float(metadata["predicted_train_reciprocity"]["max_abs_error"]),
                1e-12,
            )
            self.assertLess(float(summary["rmse_abs"]), 0.1)
            self.assertIn(
                "surrogate.py --model kbnn export-veriloga",
                composite["veriloga_export_command"],
            )


if __name__ == "__main__":
    unittest.main()
