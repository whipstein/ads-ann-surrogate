import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import neuro_tf
from surrogate_common import MDIFBlock, MLP, Standardizer, read_mdif, write_mdif


class NeuroTFConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frequencies = np.logspace(7.0, 10.0, 61)
        template = MDIFBlock(
            params={"x": 0.0},
            freq_hz=self.frequencies,
            sparams={"S11": np.zeros_like(self.frequencies, dtype=complex)},
            source_index=0,
        )
        self.blocks = [template]
        self.poles, self.f_scale = neuro_tf.build_fixed_poles(
            self.blocks,
            n_poles=12,
            damping=0.18,
        )

    def test_qr_coordinates_preserve_weighted_response_error(self) -> None:
        weights = np.linspace(0.5, 3.0, len(self.frequencies))
        encoder, decoder, diagnostics = (
            neuro_tf.build_response_conditioning_transform(
                self.blocks,
                self.poles,
                self.f_scale,
                frequency_weights=weights,
            )
        )
        rng = np.random.default_rng(41)
        coefficient_error = rng.normal(size=len(self.poles) + 1) + 1j * rng.normal(
            size=len(self.poles) + 1
        )
        latent_error = coefficient_error @ encoder
        basis = neuro_tf.rational_basis(
            self.frequencies,
            self.poles,
            self.f_scale,
        )
        response_error = basis @ coefficient_error
        self.assertAlmostEqual(
            float(np.linalg.norm(latent_error)),
            float(np.linalg.norm(np.sqrt(weights) * response_error)),
            places=10,
        )
        np.testing.assert_allclose(latent_error @ decoder, coefficient_error)
        self.assertEqual(diagnostics["rank"], len(self.poles) + 1)

    def test_real_decoder_and_folded_output_match_complex_pipeline(self) -> None:
        _encoder, decoder, _diagnostics = (
            neuro_tf.build_response_conditioning_transform(
                self.blocks,
                self.poles,
                self.f_scale,
            )
        )
        n_sparams = 4
        n_outputs = 2 * n_sparams * (len(self.poles) + 1)
        real_decoder = neuro_tf.real_coefficient_transform(decoder, n_sparams)
        rng = np.random.default_rng(7)
        mlp = MLP([2, 8, n_outputs], activation="tanh", seed=19)
        scaler = Standardizer()
        scaler.mean = rng.normal(size=n_outputs)
        scaler.std = np.full(n_outputs, 2.25)
        x = rng.normal(size=(5, 2))
        expected = scaler.inverse_transform(mlp.predict(x)) @ real_decoder
        identity_scaler = neuro_tf.fold_output_transform_into_mlp(
            mlp,
            scaler,
            real_decoder,
        )
        actual = identity_scaler.inverse_transform(mlp.predict(x))
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    def test_reciprocity_projection_ties_every_coefficient(self) -> None:
        labels = ["S11", "S12", "S21", "S22"]
        n_coeffs = 5
        projection = neuro_tf.reciprocity_projection(labels, n_coeffs)
        rng = np.random.default_rng(23)
        row = rng.normal(size=2 * len(labels) * n_coeffs)
        projected = row @ projection
        coeffs = neuro_tf.unflatten_coefficients(
            projected,
            len(labels),
            n_coeffs,
        )
        np.testing.assert_allclose(coeffs[1], coeffs[2], rtol=0.0, atol=0.0)

    def test_uniform_rf_contraction_scales_every_model_output(self) -> None:
        mlp = MLP([2, 6, 10], activation="tanh", seed=29)
        rng = np.random.default_rng(31)
        x = rng.normal(size=(4, 2))
        before = mlp.predict(x)
        neuro_tf.apply_rf_response_scale(mlp, 0.875)
        np.testing.assert_allclose(mlp.predict(x), 0.875 * before)

    def test_lossless_distributed_line_trains_as_passive_reciprocal_model(self) -> None:
        frequencies = np.concatenate(
            [np.asarray([0.0]), np.linspace(1.0e8, 12.0e9, 61)]
        )
        blocks = []
        train_values = np.linspace(0.0, 1.0, 10)
        verify_values = np.asarray([0.15, 0.45, 0.85])
        for source_index, (dataset, x_value) in enumerate(
            [*(('train', value) for value in train_values),
             *(('verification', value) for value in verify_values)]
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
            args = neuro_tf.build_arg_parser().parse_args(
                [
                    "train",
                    "--mdif",
                    str(mdif_path),
                    "--out-dir",
                    str(model_dir),
                    "--parameter-names",
                    "x",
                    "--order",
                    "14",
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
                self.assertEqual(neuro_tf.command_train(args), 0)

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
            self.assertEqual(
                metadata["coefficient_training_representation"],
                "qr_conditioned_rational_response",
            )
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
