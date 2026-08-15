import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import dnn
from surrogate_common import MDIFBlock, MLP, Standardizer, read_mdif, write_mdif


class DNNConditioningTests(unittest.TestCase):
    def test_reciprocity_enforcement_is_the_train_and_optimize_default(self) -> None:
        parser = dnn.build_arg_parser()
        for command in ("train", "optimize"):
            args = parser.parse_args(
                [
                    command,
                    "--mdif",
                    "input.mdif",
                    "--out-dir",
                    "output",
                ]
            )
            self.assertEqual(args.reciprocity_mode, "enforce")

    def test_physical_weights_cancel_output_standardization(self) -> None:
        labels = ["S11", "S12"]
        scaler = Standardizer()
        scaler.mean = np.zeros(4)
        scaler.std = np.asarray([0.5, 2.0, 4.0, 0.25])
        requested = {"S11": 3.0, "S12": 0.5}
        scaled_weights = dnn.physical_response_output_weights(
            labels,
            requested,
            scaler,
        )
        effective_raw_weights = scaled_weights / scaler.std**2
        expected = np.asarray([3.0, 0.5, 3.0, 0.5])
        np.testing.assert_allclose(
            effective_raw_weights / effective_raw_weights.mean(),
            expected / expected.mean(),
        )

    def test_passivity_loss_gradient_matches_finite_difference(self) -> None:
        rng = np.random.default_rng(91)
        scaler = Standardizer()
        scaler.mean = rng.normal(size=8) * 0.1
        scaler.std = np.exp(rng.normal(size=8))
        callback = dnn.make_s_passivity_loss_gradient(
            scaler,
            ["S11", "S12", "S21", "S22"],
            target_sigma=0.8,
            penalty=3.7,
        )
        values = rng.normal(size=(3, 8))
        weights = np.asarray([0.7, 1.1, 1.2])
        _loss, gradient = callback(values, np.zeros_like(values), weights)
        finite_difference = np.zeros_like(values)
        step = 1e-6
        for row, col in np.ndindex(values.shape):
            plus = values.copy()
            minus = values.copy()
            plus[row, col] += step
            minus[row, col] -= step
            plus_loss, _ = callback(plus, np.zeros_like(values), weights)
            minus_loss, _ = callback(minus, np.zeros_like(values), weights)
            finite_difference[row, col] = (plus_loss - minus_loss) / (2.0 * step)
        np.testing.assert_allclose(
            gradient,
            finite_difference,
            rtol=2e-5,
            atol=1e-6,
        )

    def test_folded_reciprocity_projection_preserves_existing_model_format(self) -> None:
        labels = ["S11", "S12", "S21", "S22"]
        projection = dnn.dnn_reciprocity_projection(labels)
        rng = np.random.default_rng(29)
        mlp = MLP([2, 7, 8], activation="tanh", seed=31)
        scaler = Standardizer()
        scaler.mean = rng.normal(size=8)
        scaler.std = np.exp(rng.normal(size=8))
        x = rng.normal(size=(6, 2))
        expected = scaler.inverse_transform(mlp.predict(x)) @ projection
        dnn.fold_raw_output_projection(mlp, scaler, projection)
        actual = scaler.inverse_transform(mlp.predict(x))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual[:, 1], actual[:, 2], rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(actual[:, 5], actual[:, 6], rtol=0.0, atol=1e-14)

    def test_direct_y_guard_identifies_singular_lossless_conversion(self) -> None:
        frequencies = np.concatenate(
            [np.linspace(1.0e8, 12.0e9, 61), np.asarray([40.0e9])]
        )
        transmission = np.exp(-1j * 2.0 * np.pi * frequencies * 25.0e-12)
        zero = np.zeros_like(transmission)
        block = MDIFBlock(
            params={"dataset": "train", "x": "0.5"},
            freq_hz=frequencies,
            sparams={
                "S11": zero.copy(),
                "S12": transmission.copy(),
                "S21": transmission.copy(),
                "S22": zero.copy(),
            },
            source_index=0,
        )
        summary = dnn.direct_y_conditioning_summary(
            [block],
            ["S11", "S12", "S21", "S22"],
        )
        self.assertGreater(
            float(summary["max_condition_number_i_plus_s"]),
            1e10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif_path = root / "singular_y.mdif"
            write_mdif(mdif_path, [block], ["S11", "S12", "S21", "S22"])
            args = dnn.build_arg_parser().parse_args(
                [
                    "train",
                    "--mdif",
                    str(mdif_path),
                    "--out-dir",
                    str(root / "model"),
                    "--parameter-names",
                    "x",
                    "--output-domain",
                    "y",
                    "--epochs",
                    "1",
                    "--progress-interval",
                    "0",
                ]
            )
            with self.assertRaisesRegex(ValueError, r"I \+ S is nearly singular"):
                dnn.train_model(args)

    def test_lossless_distributed_line_trains_as_passive_reciprocal_model(self) -> None:
        frequencies = np.concatenate(
            [np.asarray([0.0]), np.linspace(1.0e8, 12.0e9, 61)]
        )
        blocks = []
        train_values = np.linspace(0.0, 1.0, 10)
        verify_values = np.asarray([0.15, 0.45, 0.85])
        for source_index, (dataset, x_value) in enumerate(
            [
                *(("train", value) for value in train_values),
                *(("verification", value) for value in verify_values),
            ]
        ):
            delay = 15.0e-12 + 20.0e-12 * float(x_value)
            transmission = np.exp(-1j * 2.0 * np.pi * frequencies * delay)
            zero = np.zeros_like(transmission)
            blocks.append(
                MDIFBlock(
                    params={"dataset": dataset, "x": str(float(x_value))},
                    freq_hz=frequencies.copy(),
                    sparams={
                        "S11": zero.copy(),
                        "S12": transmission.copy(),
                        "S21": transmission.copy(),
                        "S22": zero.copy(),
                    },
                    source_index=source_index,
                )
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif_path = root / "lossless_line.mdif"
            model_dir = root / "model"
            write_mdif(mdif_path, blocks, ["S11", "S12", "S21", "S22"])
            args = dnn.build_arg_parser().parse_args(
                [
                    "train",
                    "--mdif",
                    str(mdif_path),
                    "--out-dir",
                    str(model_dir),
                    "--parameter-names",
                    "x",
                    "--freq-transform",
                    "linear",
                    "--hidden-layers",
                    "64,64",
                    "--epochs",
                    "450",
                    "--patience",
                    "150",
                    "--progress-interval",
                    "0",
                    "--worst-plots",
                    "0",
                    "--seed",
                    "1234",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(dnn.command_train(args), 0)

            summary = json.loads(
                (model_dir / "verification_summary.json").read_text()
            )
            metadata = json.loads((model_dir / "metadata.json").read_text())
            self.assertLess(float(summary["rmse_abs"]), 0.1)
            self.assertEqual(summary["passivity"]["violating_points"], 0)
            self.assertLessEqual(
                float(summary["passivity"]["max_singular_value"]),
                1.0 + 1e-6,
            )
            self.assertTrue(metadata["reciprocity_enforced"])
            self.assertTrue(metadata["passivity_enforced"])
            predicted = read_mdif(model_dir / "predicted_verification.mdif")
            for block in predicted:
                np.testing.assert_allclose(
                    block.sparams["S12"],
                    block.sparams["S21"],
                    rtol=0.0,
                    atol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
