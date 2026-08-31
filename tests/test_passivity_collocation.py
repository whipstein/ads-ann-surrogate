import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import dnn
import kbnn
from surrogate_common import (
    MDIFBlock,
    MLP,
    build_passivity_collocation_blocks,
    passivity_columns_summary,
    write_training_markdown,
)


class PassivityCollocationTests(unittest.TestCase):
    def test_cli_defaults_disable_collocation_but_expose_controls(self) -> None:
        dnn_args = dnn.build_arg_parser().parse_args(
            ["train", "--mdif", "input.mdif", "--out-dir", "model"]
        )
        kbnn_args = kbnn.build_arg_parser().parse_args(
            ["train", "--mdif", "input.mdif", "--out-dir", "model"]
        )
        for args in (dnn_args, kbnn_args):
            self.assertEqual(args.passivity_collocation_geometries, 0)
            self.assertEqual(args.passivity_collocation_frequencies, 32)
            self.assertEqual(args.passivity_collocation_candidate_multiplier, 4)
            self.assertEqual(args.passivity_collocation_refresh, 25)

    def test_geometry_json_automatically_matches_bare_mdif_units(self) -> None:
        labels = ["S11"]
        training = [
            MDIFBlock(
                params={"W": value},
                freq_hz=np.asarray([1.0e9, 2.0e9]),
                sparams={"S11": np.zeros(2, dtype=complex)},
                source_index=index,
            )
            for index, value in enumerate(("0.45", "0.75"))
        ]
        geometry = {
            "parameters": [
                {
                    "name": "W",
                    "range": {"lower": 0.4, "upper": 0.8, "unit": "um"},
                    "base_unit_range": {"lower": 4.0e-7, "upper": 8.0e-7},
                    "scale": "linear",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            geometry_path = Path(temp_dir) / "geometries.json"
            geometry_path.write_text(json.dumps(geometry))
            fixed, candidates, metadata = build_passivity_collocation_blocks(
                training,
                ["W"],
                labels,
                geometry_count=4,
                frequency_count=3,
                candidate_multiplier=2,
                frequency_transform="log",
                seed=17,
                geometry_json=geometry_path,
            )
        values = np.asarray(
            [float(block.params["W"]) for block in [*fixed, *candidates]]
        )
        self.assertTrue(np.all(values >= 0.4))
        self.assertTrue(np.all(values <= 0.8))
        self.assertEqual(len(fixed), 4)
        self.assertEqual(len(candidates), 8)
        self.assertEqual(metadata["fixed_sample_count"], 12)
        self.assertEqual(metadata["candidate_sample_count"], 24)
        self.assertFalse(metadata["uses_response_targets"])

    def test_hard_negative_mining_keeps_fixed_and_worst_candidates(self) -> None:
        mlp = MLP([1, 1], activation="tanh", seed=3)
        x_train = np.arange(4, dtype=float)[:, None]
        y_train = np.zeros((4, 1), dtype=float)
        auxiliary_x = np.arange(6, dtype=float)[:, None]
        observed_indices: list[int] = []

        def auxiliary_loss(
            prediction: np.ndarray,
            indices: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            observed_indices.extend(int(index) for index in indices)
            return 0.0, np.zeros_like(prediction)

        def score(_prediction: np.ndarray, indices: np.ndarray) -> np.ndarray:
            return np.asarray(indices, dtype=float)

        mlp.train(
            x_train,
            y_train,
            None,
            None,
            epochs=1,
            batch_size=4,
            learning_rate=0.0,
            patience=0,
            seed=9,
            auxiliary_x=auxiliary_x,
            auxiliary_loss_gradient=auxiliary_loss,
            auxiliary_score=score,
            auxiliary_fixed_count=2,
            auxiliary_hard_count=2,
            auxiliary_refresh_interval=1,
        )
        self.assertEqual(set(observed_indices), {0, 1, 4, 5})

    def test_checkpoint_selection_requires_feasibility_before_error(self) -> None:
        mlp = MLP([1, 1], activation="tanh", seed=11)
        x_train = np.ones((8, 1), dtype=float)
        y_train = np.ones((8, 1), dtype=float)
        x_val = np.ones((4, 1), dtype=float)
        y_val = -np.ones((4, 1), dtype=float)
        calls = 0

        def checkpoint(_network: MLP) -> dict[str, float | int]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "violating_points": 2,
                    "max_singular_value": 1.05,
                    "target_sigma": 0.995,
                }
            return {
                "violating_points": 0,
                "max_singular_value": 0.99 - 0.01 * calls,
                "target_sigma": 0.995,
            }

        history = mlp.train(
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=3,
            batch_size=8,
            learning_rate=0.2,
            patience=0,
            seed=5,
            checkpoint_constraint=checkpoint,
        )
        selected = [
            int(row["epoch"])
            for row in history
            if row["selected_checkpoint"] == 1.0
        ]
        self.assertEqual(selected, [2])
        self.assertGreater(history[1]["val_loss"], history[0]["val_loss"])
        self.assertEqual(history[1]["checkpoint_passivity_violations"], 0.0)

    def test_passivity_column_summary_uses_full_matrix_singular_value(self) -> None:
        # A through connection is lossless/passive with sigma_max exactly one.
        columns = np.asarray([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        summary = passivity_columns_summary(
            columns,
            ["S11", "S12", "S21", "S22"],
            target_sigma=1.0,
        )
        self.assertAlmostEqual(float(summary["max_singular_value"]), 1.0)
        self.assertEqual(summary["violating_points"], 0)

    def test_training_report_identifies_restored_checkpoint(self) -> None:
        history = [
            {
                "epoch": 1.0,
                "train_loss": 0.2,
                "val_loss": 0.1,
                "checkpoint_passivity_violations": 3.0,
                "checkpoint_max_singular_value": 1.02,
                "selected_checkpoint": 0.0,
            },
            {
                "epoch": 2.0,
                "train_loss": 0.25,
                "val_loss": 0.15,
                "checkpoint_passivity_violations": 0.0,
                "checkpoint_max_singular_value": 0.995,
                "selected_checkpoint": 1.0,
            },
            {
                "epoch": 3.0,
                "train_loss": 0.19,
                "val_loss": 0.09,
                "checkpoint_passivity_violations": 1.0,
                "checkpoint_max_singular_value": 1.001,
                "selected_checkpoint": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "training_summary.md"
            write_training_markdown(
                report_path,
                "DNN",
                {},
                {},
                history,
            )
            report = report_path.read_text()
        self.assertIn("## Selected Training Checkpoint", report)
        self.assertIn("| 2 | 0.25 | 0.15 | 0 | 0.995 |", report)
        self.assertIn("The saved model restores this checkpoint", report)


if __name__ == "__main__":
    unittest.main()
