import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dnn
import kbnn
import neuro_tf
import surrogate_common
from surrogate_common import (
    adaptive_candidate_features,
    build_adaptive_candidate_pool,
    run_sweep_command,
    select_adaptive_candidate,
)


class AdaptiveSweepTests(unittest.TestCase):
    def test_numeric_and_hidden_layer_ranges_build_unique_pool(self) -> None:
        candidates, columns, log_parameters = build_adaptive_candidate_pool(
            {
                "freq_transform": "log",
                "hidden_layers": "64,64",
                "activation": "tanh",
                "learning_rate": 0.002,
            },
            [
                "learning_rate=1e-4:1e-2:log",
                "hidden_layers=1:4x32:256:log",
            ],
            {
                "freq_transform": "str",
                "hidden_layers": "hidden_layers",
                "activation": "str",
                "learning_rate": "float",
            },
            max_trials=12,
            candidate_pool=80,
            hidden_width_step=8,
            seed=19,
        )
        self.assertGreaterEqual(len(candidates), 12)
        self.assertEqual(len(candidates), len({str(candidate) for candidate in candidates}))
        self.assertIn("learning_rate", log_parameters)
        self.assertIn("hidden_layers", columns)
        for candidate in candidates:
            self.assertGreaterEqual(float(candidate["learning_rate"]), 1e-4)
            self.assertLessEqual(float(candidate["learning_rate"]), 1e-2)
            layers = [int(value) for value in str(candidate["hidden_layers"]).split(",")]
            self.assertGreaterEqual(len(layers), 1)
            self.assertLessEqual(len(layers), 4)
            self.assertTrue(all(32 <= value <= 256 for value in layers))
            self.assertTrue(all(value % 8 == 0 for value in layers))
        features = adaptive_candidate_features(candidates, columns, log_parameters)
        self.assertEqual(features.shape[0], len(candidates))
        self.assertGreater(features.shape[1], 0)

    def test_adaptive_selector_uses_gp_after_initial_trials(self) -> None:
        candidates = [
            {"learning_rate": value, "hidden_layers": "64,64"}
            for value in (0.001, 0.002, 0.003, 0.004, 0.005)
        ]
        rows = [
            {
                "metric": 0.8,
                "rmse_abs": 0.8,
                "passivity_violating_points": 8,
                "passivity_max_singular_value": 1.4,
            },
            {
                "metric": 0.5,
                "rmse_abs": 0.5,
                "passivity_violating_points": 3,
                "passivity_max_singular_value": 1.15,
            },
            {
                "metric": 0.4,
                "rmse_abs": 0.4,
                "passivity_violating_points": 1,
                "passivity_max_singular_value": 1.04,
            },
        ]
        args = argparse.Namespace(
            selection_metric="rmse_abs",
            require_passive=True,
            max_passivity_violations=None,
            max_passivity_sigma=None,
            adaptive_initial_trials=2,
            adaptive_exploration=1.5,
            adaptive_log_parameters={"learning_rate"},
        )
        index, diagnostics = select_adaptive_candidate(
            candidates,
            remaining_indices=[3, 4],
            selected_indices=[0, 1, 2],
            rows=rows,
            args=args,
            columns=["learning_rate", "hidden_layers"],
        )
        self.assertIn(index, {3, 4})
        self.assertEqual(diagnostics["adaptive_stage"], "gp_lcb")
        self.assertIsNotNone(diagnostics["adaptive_predicted_objective"])
        self.assertIsNotNone(diagnostics["adaptive_uncertainty"])

    def test_all_ineligible_trials_still_write_complete_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "sweep"
            args = argparse.Namespace(
                out_dir=str(out_dir),
                selection_metric="rmse_abs",
                require_passive=True,
                max_passivity_violations=None,
                max_passivity_sigma=None,
                jobs=1,
                seed=1234,
                trial_seed_mode="fixed",
                trial_worst_plots=0,
                worst_plots=0,
                keep_trial_models=True,
                retrain_best=False,
                epochs=10,
                mode="adaptive",
                max_trials=2,
                adaptive_initial_trials=1,
                adaptive_exploration=1.5,
                adaptive_log_parameters={"learning_rate"},
            )
            candidates = [
                {"learning_rate": 0.001, "hidden_layers": "32"},
                {"learning_rate": 0.003, "hidden_layers": "64,64"},
            ]

            def worker(payload):
                _values, candidate, out_text, trial, _plots = payload
                trial_dir = Path(out_text) / "trials" / f"trial_{trial:04d}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                metric = 0.3 if trial == 1 else 0.2
                return {
                    "trial": trial,
                    "candidate": candidate,
                    "summary": {
                        "rmse_abs": metric,
                        "passivity": {
                            "max_singular_value": 1.3 if trial == 1 else 1.1,
                            "violating_points": 8 if trial == 1 else 2,
                        },
                    },
                    "metric": metric,
                    "trial_seed": 1234,
                    "plot_paths": [],
                }

            def namespace_for_trial(_args, _candidate, _out, _trial, plots):
                return argparse.Namespace(seed=1234, worst_plots=plots)

            status = run_sweep_command(
                args,
                candidates,
                worker_func=worker,
                namespace_for_trial_func=namespace_for_trial,
                train_func=lambda _args: 0,
                result_columns=["learning_rate", "hidden_layers"],
                results_filename="results.csv",
                best_config_filename="best.json",
                summary_filename="summary.md",
                diagnostics_prefix="test",
                train_command_prefix=None,
            )
            self.assertEqual(status, 2)
            self.assertTrue((out_dir / "results.csv").is_file())
            self.assertTrue((out_dir / "summary.md").is_file())
            self.assertTrue(
                (out_dir / "sweep_diagnostics" / "test_error_vs_swept_parameters.pdf").is_file()
            )
            self.assertTrue(
                (out_dir / "sweep_diagnostics" / "test_metric_trend.svg").is_file()
            )
            summary_text = (out_dir / "summary.md").read_text()
            self.assertIn("## Selection Status", summary_text)
            self.assertIn("closest available completed trial", summary_text)
            self.assertIn("Diagnostic artifacts", summary_text)
            self.assertIn("Search stage", summary_text)
            self.assertIn("initial_maximin", summary_text)
            self.assertIn("## Sweep Trend Plots", summary_text)
            self.assertIn(
                "![Sweep trend plot: test metric trend]"
                "(sweep_diagnostics/test_metric_trend.svg)",
                summary_text,
            )
            payload = json.loads((out_dir / "best.json").read_text())
            self.assertEqual(payload["status"], "no_eligible_trial")
            self.assertEqual(payload["best_available_trial"], 2)
            stats_path = (
                out_dir
                / "sweep_diagnostics"
                / "test_error_vs_swept_parameters.csv"
            )
            with stats_path.open(newline="") as stream:
                stats = list(csv.DictReader(stream))
            self.assertTrue(stats)
            self.assertTrue(any(row["all_mean"] for row in stats))

    def test_dnn_adaptive_cli_builds_requested_search_space(self) -> None:
        parser = dnn.build_arg_parser()
        args = parser.parse_args(
            [
                "optimize",
                "--mdif",
                "input.mdf",
                "--out-dir",
                "output",
                "--search-mode",
                "adaptive",
                "--max-trials",
                "8",
                "--adaptive-candidate-pool",
                "40",
                "--optimize-parameter",
                "learning_rate=1e-4:5e-3:log",
                "--optimize-parameter",
                "hidden_layers=1:3x32:128",
            ]
        )
        candidates = dnn.sweep_candidate_grid(args)
        self.assertGreaterEqual(len(candidates), 8)
        self.assertEqual(args.adaptive_result_columns[:4], dnn.DNN_SWEEP_RESULT_COLUMNS)
        self.assertIn("learning_rate", args.adaptive_log_parameters)

    def test_kbnn_adaptive_trial_applies_dynamic_training_controls(self) -> None:
        parser = kbnn.build_arg_parser()
        args = parser.parse_args(
            [
                "optimize",
                "--mdif",
                "fine.mdf",
                "--coarse-model-dir",
                "coarse_model",
                "--out-dir",
                "output",
                "--search-mode",
                "adaptive",
                "--max-trials",
                "5",
                "--adaptive-candidate-pool",
                "30",
                "--optimize-parameter",
                "mode=residual,prior-input",
                "--optimize-parameter",
                "batch_size=32:64",
                "--optimize-parameter",
                "hidden_layers=1:3x32:96",
            ]
        )
        args.mode = args.search_mode
        args.mode_options = args.mode_options or "residual,prior-input"
        candidates = kbnn.sweep_candidate_grid(args)
        self.assertGreaterEqual(len(candidates), 5)
        candidate = candidates[0]
        trial_args = kbnn.namespace_for_trial(
            args,
            candidate,
            Path("output/trial"),
            trial_index=1,
            plots=0,
        )
        self.assertEqual(trial_args.batch_size, candidate["batch_size"])
        self.assertEqual(trial_args.hidden_layers, candidate["hidden_layers"])
        self.assertEqual(trial_args.mode, candidate["mode"])

    def test_neurotf_adaptive_trial_applies_dynamic_training_controls(self) -> None:
        parser = neuro_tf.build_arg_parser()
        args = parser.parse_args(
            [
                "optimize",
                "--mdif",
                "input.mdf",
                "--out-dir",
                "output",
                "--search-mode",
                "adaptive",
                "--max-trials",
                "5",
                "--adaptive-candidate-pool",
                "30",
                "--optimize-parameter",
                "order=6:16",
                "--optimize-parameter",
                "batch_size=16:64",
                "--optimize-parameter",
                "hidden_layers=1:3x32:96",
            ]
        )
        candidates = neuro_tf.sweep_candidate_grid(args)
        self.assertGreaterEqual(len(candidates), 5)
        candidate = candidates[0]
        trial_args = neuro_tf.namespace_for_trial(
            args,
            candidate,
            Path("output/trial"),
            trial_index=1,
            plots=0,
        )
        self.assertEqual(trial_args.batch_size, candidate["batch_size"])
        self.assertEqual(trial_args.hidden_layers, candidate["hidden_layers"])
        self.assertEqual(trial_args.order, candidate["order"])

    @mock.patch("surrogate_common.load_matplotlib_modules", return_value=None)
    def test_dependency_free_sweep_plot_is_embeddable_svg(self, _modules) -> None:
        rows = [
            {
                "trial": 1,
                "metric": 0.4,
                "rmse_abs": 0.4,
                "learning_rate": 0.001,
                "passivity_violating_points": 3,
                "passivity_max_singular_value": 1.2,
            },
            {
                "trial": 2,
                "metric": 0.2,
                "rmse_abs": 0.2,
                "learning_rate": 0.003,
                "passivity_violating_points": 0,
                "passivity_max_singular_value": 0.99,
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts, images = surrogate_common.plot_sweep_diagnostics(
                rows,
                Path(temp_dir),
                ["learning_rate"],
                "rmse_abs",
                "fallback",
            )
            self.assertEqual(len(artifacts), 2)
            self.assertTrue(all(path.is_file() for path in artifacts))
            self.assertTrue(images)
            self.assertTrue(all(path.suffix == ".svg" for path in images))
            svg = images[0].read_text()
            self.assertIn("<svg", svg)
            self.assertIn("mean, all trials", svg)
            self.assertIn("passivity fail", svg)


if __name__ == "__main__":
    unittest.main()
