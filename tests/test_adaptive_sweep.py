import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

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
    def test_all_models_rerank_unpack_diagnostic_artifacts_and_images(self) -> None:
        cases = [
            (dnn, "dnn"),
            (kbnn, "kbnn"),
            (neuro_tf, "neurotf"),
        ]
        for module, prefix in cases:
            with self.subTest(model=prefix), tempfile.TemporaryDirectory() as temp_dir:
                sweep_dir = Path(temp_dir) / f"{prefix}_sweep"
                sweep_dir.mkdir()
                results_path = sweep_dir / f"{prefix}_sweep_results.csv"
                with results_path.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=[
                            "trial",
                            "rmse_abs",
                            "passivity_max_singular_value",
                            "passivity_violating_points",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "trial": 1,
                            "rmse_abs": 0.2,
                            "passivity_max_singular_value": 0.99,
                            "passivity_violating_points": 0,
                        }
                    )

                diagnostic_dir = sweep_dir / "sweep_diagnostics"
                diagnostic_dir.mkdir()
                pdf_path = diagnostic_dir / f"{prefix}_reranked_trends.pdf"
                csv_path = diagnostic_dir / f"{prefix}_reranked_trends.csv"
                png_path = diagnostic_dir / f"{prefix}_reranked_rmse_abs_trend.png"
                for path in (pdf_path, csv_path, png_path):
                    path.write_bytes(b"artifact")

                args = argparse.Namespace(
                    sweep_dir=str(sweep_dir),
                    selection_metric="rmse_abs",
                    require_passive=True,
                    max_passivity_violations=None,
                    max_passivity_sigma=1.000001,
                    promote_best=False,
                    replace_current_best=False,
                    best_model_dir=None,
                    overwrite=False,
                )
                with mock.patch.object(
                    module,
                    "plot_sweep_diagnostics",
                    return_value=([pdf_path, csv_path], [png_path]),
                ), mock.patch("builtins.print"):
                    status = module.command_rerank_sweep(args)

                self.assertEqual(status, 0)
                payload = json.loads(
                    (
                        sweep_dir / f"{prefix}_reranked_best_config.json"
                    ).read_text()
                )
                self.assertEqual(
                    payload["diagnostic_artifacts"],
                    [
                        f"sweep_diagnostics/{pdf_path.name}",
                        f"sweep_diagnostics/{csv_path.name}",
                    ],
                )
                self.assertEqual(
                    payload["diagnostic_images"],
                    [f"sweep_diagnostics/{png_path.name}"],
                )
                report = (
                    sweep_dir / f"{prefix}_reranked_sweep_summary.md"
                ).read_text()
                self.assertIn("## Sweep Trend Plots", report)
                self.assertIn(f"](sweep_diagnostics/{png_path.name})", report)

    def test_trial_cleanup_always_retains_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trial_dir = Path(temp_dir) / "trial_0001"
            trial_dir.mkdir()
            metadata = {"model_type": "dnn", "hidden_layers": [64, 64]}
            (trial_dir / "metadata.json").write_text(json.dumps(metadata))
            (trial_dir / "model.npz").write_bytes(b"large model")
            (trial_dir / "dc_model.json").write_text("{}")
            (trial_dir / "verification_summary.json").write_text("{}")

            surrogate_common.cleanup_trial_dir(
                trial_dir,
                keep_trial_models=False,
            )

            self.assertEqual(
                json.loads((trial_dir / "metadata.json").read_text()),
                metadata,
            )
            self.assertFalse((trial_dir / "model.npz").exists())
            self.assertFalse((trial_dir / "dc_model.json").exists())
            self.assertTrue((trial_dir / "verification_summary.json").exists())

    def test_numeric_and_hidden_layer_ranges_build_unique_pool(self) -> None:
        candidates, columns, log_parameters, categorical_values = (
            build_adaptive_candidate_pool(
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
        )
        self.assertGreaterEqual(len(candidates), 12)
        self.assertEqual(len(candidates), len({str(candidate) for candidate in candidates}))
        self.assertIn("learning_rate", log_parameters)
        self.assertIn("hidden_layers", columns)
        self.assertEqual(categorical_values, {})
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

    def test_initial_adaptive_trials_balance_all_categorical_margins(self) -> None:
        candidates = [
            {
                "activation": activation,
                "freq_transform": transform,
                "learning_rate": learning_rate,
            }
            for learning_rate in (0.001, 0.002, 0.003, 0.004)
            for activation in ("tanh", "relu")
            for transform in ("log", "linear", "log-linear")
        ]
        args = argparse.Namespace(
            selection_metric="rmse_abs",
            require_passive=False,
            max_passivity_violations=None,
            max_passivity_sigma=None,
            adaptive_initial_trials=6,
            adaptive_exploration=1.5,
            adaptive_log_parameters={"learning_rate"},
            adaptive_categorical_values={
                "activation": ["tanh", "relu"],
                "freq_transform": ["log", "linear", "log-linear"],
            },
            max_trials=6,
        )
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        rows: list[dict[str, object]] = []
        final_diagnostics: dict[str, object] = {}
        for trial in range(6):
            index, final_diagnostics = select_adaptive_candidate(
                candidates,
                remaining_indices=remaining,
                selected_indices=selected,
                rows=rows,
                args=args,
                columns=["activation", "freq_transform", "learning_rate"],
            )
            remaining.remove(index)
            selected.append(index)
            rows.append({"rmse_abs": 1.0 - 0.05 * trial})

        activation_counts = {
            value: sum(candidates[index]["activation"] == value for index in selected)
            for value in ("tanh", "relu")
        }
        transform_counts = {
            value: sum(
                candidates[index]["freq_transform"] == value for index in selected
            )
            for value in ("log", "linear", "log-linear")
        }
        self.assertEqual(activation_counts, {"tanh": 3, "relu": 3})
        self.assertEqual(
            transform_counts,
            {"log": 2, "linear": 2, "log-linear": 2},
        )
        self.assertIn(
            "activation: tanh=3, relu=3",
            str(final_diagnostics["adaptive_category_coverage"]),
        )

    def test_categorical_levels_expand_too_small_initial_stage(self) -> None:
        candidates = [
            {"activation": activation, "learning_rate": learning_rate}
            for learning_rate in (0.001, 0.002, 0.003)
            for activation in ("tanh", "relu", "sigmoid")
        ]
        args = argparse.Namespace(
            selection_metric="rmse_abs",
            require_passive=False,
            max_passivity_violations=None,
            max_passivity_sigma=None,
            adaptive_initial_trials=2,
            adaptive_exploration=1.5,
            adaptive_log_parameters={"learning_rate"},
            adaptive_categorical_values={
                "activation": ["tanh", "relu", "sigmoid"]
            },
            max_trials=5,
        )
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        rows: list[dict[str, object]] = []
        stages: list[str] = []
        for trial in range(3):
            index, diagnostics = select_adaptive_candidate(
                candidates,
                remaining_indices=remaining,
                selected_indices=selected,
                rows=rows,
                args=args,
                columns=["activation", "learning_rate"],
            )
            remaining.remove(index)
            selected.append(index)
            rows.append({"rmse_abs": 0.5 - 0.05 * trial})
            stages.append(str(diagnostics["adaptive_stage"]))
        self.assertEqual(stages, ["initial_maximin"] * 3)
        self.assertEqual(
            {candidates[index]["activation"] for index in selected},
            {"tanh", "relu", "sigmoid"},
        )

    def test_gp_stage_preserves_configured_categorical_coverage_floor(self) -> None:
        candidates = [
            {"activation": activation, "learning_rate": learning_rate}
            for learning_rate in np.linspace(0.0005, 0.01, 20)
            for activation in ("tanh", "relu")
        ]
        args = argparse.Namespace(
            selection_metric="rmse_abs",
            require_passive=False,
            max_passivity_violations=None,
            max_passivity_sigma=None,
            adaptive_initial_trials=4,
            adaptive_exploration=0.0,
            adaptive_category_balance=0.5,
            adaptive_log_parameters={"learning_rate"},
            adaptive_categorical_values={"activation": ["tanh", "relu"]},
            max_trials=20,
        )
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        rows: list[dict[str, object]] = []
        for _trial in range(20):
            index, _diagnostics = select_adaptive_candidate(
                candidates,
                remaining_indices=remaining,
                selected_indices=selected,
                rows=rows,
                args=args,
                columns=["activation", "learning_rate"],
            )
            remaining.remove(index)
            selected.append(index)
            rows.append(
                {
                    "rmse_abs": (
                        0.01
                        if candidates[index]["activation"] == "relu"
                        else 1.0
                    )
                }
            )
        counts = {
            value: sum(candidates[index]["activation"] == value for index in selected)
            for value in ("tanh", "relu")
        }
        self.assertGreaterEqual(counts["tanh"], 5)
        self.assertGreater(counts["relu"], counts["tanh"])

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
                keep_trial_models=False,
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
                with (trial_dir / "verification_metrics.csv").open(
                    "w", newline=""
                ) as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["source_index", "evm_pct", "width"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "source_index": trial,
                            "evm_pct": metric,
                            "width": candidate["learning_rate"],
                        }
                    )
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
                (out_dir / "sweep_diagnostics" / "test_metric_trend.png").is_file()
            )
            summary_text = (out_dir / "summary.md").read_text()
            self.assertIn("## Selection Status", summary_text)
            self.assertIn("closest available completed trial", summary_text)
            self.assertIn("Diagnostic artifacts", summary_text)
            self.assertIn("Search stage", summary_text)
            self.assertIn("initial_maximin", summary_text)
            self.assertIn("## Sweep Trend Plots", summary_text)
            self.assertIn("## Trial Ranking", summary_text)
            self.assertIn(
                "![Sweep trend plot: test metric trend]"
                "(sweep_diagnostics/test_metric_trend.png)",
                summary_text,
            )
            self.assertLess(
                summary_text.index("## Sweep Trend Plots"),
                summary_text.index("## Trial Ranking"),
            )
            self.assertLess(
                summary_text.index("## Trial Ranking"),
                summary_text.index("| Rank | Trial | Metric |"),
            )
            payload = json.loads((out_dir / "best.json").read_text())
            self.assertEqual(payload["status"], "no_eligible_trial")
            self.assertEqual(payload["best_available_trial"], 2)
            self.assertEqual(payload["point_generation_trial"], 2)
            self.assertFalse(payload["point_generation_export_eligible"])
            fallback_dir = out_dir / "point_generation_fallback"
            self.assertTrue((fallback_dir / "verification_metrics.csv").is_file())
            fallback_source = json.loads(
                (fallback_dir / "point_generation_source.json").read_text()
            )
            self.assertEqual(fallback_source["source_trial"], 2)
            self.assertEqual(
                fallback_source["purpose"], "gp_point_generation_only"
            )
            self.assertFalse(fallback_source["eligible_for_export"])
            self.assertFalse(
                (
                    out_dir
                    / "trials"
                    / "trial_0002"
                    / "verification_metrics.csv"
                ).exists()
            )
            self.assertIn("## GP Point-Generation Fallback", summary_text)
            self.assertIn("--allow-nonpassive", summary_text)
            stats_path = (
                out_dir
                / "sweep_diagnostics"
                / "test_error_vs_swept_parameters.csv"
            )
            with stats_path.open(newline="") as stream:
                stats = list(csv.DictReader(stream))
            self.assertTrue(stats)
            self.assertTrue(any(row["all_mean"] for row in stats))

    def test_passing_sweep_promotes_verification_metrics_before_trial_cleanup(self) -> None:
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
                keep_trial_models=False,
                retrain_best=False,
                epochs=10,
                mode="random",
                mdif="combined.mdif",
                verification_mdif=None,
                split_var="dataset",
                train_values="train,training",
                verify_values="verify,verification,test,validation",
                holdout_fraction=0.2,
            )
            candidates = [{"learning_rate": 0.001, "hidden_layers": "32"}]

            def worker(payload):
                _values, candidate, out_text, trial, _plots = payload
                trial_dir = Path(out_text) / "trials" / f"trial_{trial:04d}"
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "model.npz").write_bytes(b"model")
                (trial_dir / "metadata.json").write_text("{}")
                (trial_dir / "verification_metrics.csv").write_text(
                    "source_index,evm_pct,width\n1,0.2,0.5\n"
                )
                return {
                    "trial": trial,
                    "candidate": candidate,
                    "summary": {
                        "rmse_abs": 0.2,
                        "passivity": {
                            "max_singular_value": 0.99,
                            "violating_points": 0,
                        },
                    },
                    "metric": 0.2,
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
                best_config_filename="best_config.json",
                summary_filename="summary.md",
                diagnostics_prefix="test",
                train_command_prefix=None,
            )

            self.assertEqual(status, 0)
            promoted_metrics = out_dir / "best_model" / "verification_metrics.csv"
            self.assertTrue(promoted_metrics.is_file())
            self.assertIn("0.2", promoted_metrics.read_text())
            self.assertFalse(
                (
                    out_dir
                    / "trials"
                    / "trial_0001"
                    / "verification_metrics.csv"
                ).exists()
            )
            self.assertTrue(
                (out_dir / "trials" / "trial_0001" / "metadata.json").is_file()
            )
            self.assertFalse(
                (out_dir / "trials" / "trial_0001" / "model.npz").exists()
            )
            best_config = json.loads((out_dir / "best_config.json").read_text())
            self.assertEqual(
                best_config["verification_metrics"],
                str(promoted_metrics),
            )

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
                "--optimize-parameter",
                "activation=tanh,relu",
            ]
        )
        candidates = dnn.sweep_candidate_grid(args)
        self.assertGreaterEqual(len(candidates), 8)
        self.assertEqual(args.adaptive_result_columns[:4], dnn.DNN_SWEEP_RESULT_COLUMNS)
        self.assertIn("learning_rate", args.adaptive_log_parameters)
        self.assertEqual(
            args.adaptive_categorical_values,
            {"activation": ["tanh", "relu"]},
        )
        self.assertEqual(args.adaptive_category_balance, 0.5)

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
    def test_dependency_free_sweep_plot_is_embeddable_png(self, _modules) -> None:
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
            self.assertTrue(all(path.suffix == ".png" for path in images))
            with Image.open(images[0]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 800)
                colors = image.convert("RGB").getcolors(
                    maxcolors=image.width * image.height
                )
            self.assertIsNotNone(colors)
            rendered = {color for _count, color in colors or []}
            self.assertIn((31, 119, 180), rendered)
            self.assertIn((214, 39, 40), rendered)


if __name__ == "__main__":
    unittest.main()
