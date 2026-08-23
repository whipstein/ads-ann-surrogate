import argparse
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import debug_model as DEBUG
import surrogate


class ModelDebugTests(unittest.TestCase):
    def test_nonpassive_source_suppresses_enforcement_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "dnn_opt"
            run_dir.mkdir()
            rows = [
                {
                    "trial": 1,
                    "rmse_abs": 0.1,
                    "passivity_violating_points": 5,
                    "passivity_max_singular_value": 1.2,
                    "source_rf_passivity_violating_points": 3,
                }
            ]
            findings = DEBUG.build_findings(
                rows,
                "rmse_abs",
                {"passivity": {"violating_rf_rows": 3}},
                [],
                "dnn",
            )
            suggestions = DEBUG.build_command_suggestions(
                argparse.Namespace(options_json=None),
                run_dir,
                "dnn",
                rows,
                "rmse_abs",
                findings,
                [],
            )

        identifiers = {suggestion["id"] for suggestion in suggestions}
        self.assertIn("audit-source-data", identifiers)
        self.assertNotIn("passivity-feasibility-search", identifiers)
        self.assertTrue(
            all(
                "--passivity-mode enforce" not in suggestion["command"]
                for suggestion in suggestions
            )
        )

    def test_sweep_debug_works_without_trial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "dnn_opt"
            run_dir.mkdir()
            result_rows = []
            metrics = [0.50, 0.42, 0.28, 0.20]
            sigmas = [1.03, 1.015, 1.002, 1.0005]
            violations = [12, 7, 3, 1]
            for trial, (metric, sigma, count) in enumerate(
                zip(metrics, sigmas, violations), start=1
            ):
                result_rows.append(
                    {
                        "trial": trial,
                        "selection_metric": "weighted_evm_pct",
                        "weighted_evm_pct": metric,
                        "passivity_violating_points": count,
                        "passivity_max_singular_value": sigma,
                        "hidden_layers": "64,64",
                    }
                )
                trial_dir = run_dir / "trials" / f"trial_{trial:04d}"
                trial_dir.mkdir(parents=True)
                (trial_dir / "verification_summary.json").write_text(
                    json.dumps(
                        {
                            "weighted_evm_pct": metric,
                            "passivity": {
                                "violating_points": count,
                                "max_singular_value": sigma,
                            },
                            "passivity_enforced": True,
                            "rf_response_scale": 0.999,
                            "source_rf_passivity": {
                                "violating_points": 0,
                                "max_singular_value": 0.99,
                            },
                            "predicted_train_passivity_after_scale": {
                                "violating_points": 0,
                                "max_singular_value": 0.999,
                            },
                        }
                    )
                )
            with (run_dir / "dnn_sweep_results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
                writer.writeheader()
                writer.writerows(result_rows)
            audit_dir = root / "audit"
            audit_dir.mkdir()
            (audit_dir / "dataset_audit.json").write_text(
                json.dumps(
                    {
                        "verdict": "PASS",
                        "passivity": {
                            "violating_rf_rows": 0,
                            "max_singular_value": 0.999,
                        },
                    }
                )
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    DEBUG.main(
                        [
                            "--run-dir",
                            str(run_dir),
                            "--audit",
                            str(audit_dir),
                        ]
                    ),
                    0,
                )
            out_dir = run_dir / "model_debug"
            payload = json.loads((out_dir / "model_debug.json").read_text())
            codes = {finding["code"] for finding in payload["findings"]}
            self.assertIn("MODEL_METADATA_MISSING", codes)
            self.assertIn("TRAIN_PASSIVE_VERIFY_NONPASSIVE", codes)
            self.assertIn("MARGINAL_SIGMA_EXCURSION", codes)
            self.assertIn("ERROR_IMPROVES_WITHOUT_FEASIBILITY", codes)
            self.assertEqual(payload["verification_summaries_found"], 4)
            self.assertEqual(payload["statistics"]["passive_trials"], 0)
            suggestions = {
                suggestion["id"]: suggestion
                for suggestion in payload["suggested_commands"]
            }
            self.assertIn("passivity-feasibility-search", suggestions)
            self.assertIn("refresh-legacy-metadata", suggestions)
            feasibility_command = suggestions["passivity-feasibility-search"][
                "command"
            ]
            self.assertIn("--passivity-mode enforce", feasibility_command)
            self.assertIn("--output-domain s", feasibility_command)
            self.assertIn(
                "--optimize-parameter passivity_penalty=",
                feasibility_command,
            )
            self.assertTrue(
                suggestions["passivity-feasibility-search"]["requires_editing"]
            )
            report = (out_dir / "model_debug.md").read_text()
            self.assertIn(
                "Current optimize runs retain every completed trial's `metadata.json`",
                report,
            )
            self.assertIn("model_debug_passivity.png", report)
            self.assertIn("## Suggested command changes", report)
            self.assertIn("Option or control", report)
            self.assertIn("PATH_TO_TRAINING_MDIF", report)
            plot_path = out_dir / "model_debug_passivity.png"
            with Image.open(plot_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 1000)

    def test_primary_dispatcher_exposes_debug_model(self) -> None:
        self.assertEqual(surrogate.WORKFLOW_SCRIPTS["debug-model"], "debug_model.py")
        help_text = surrogate.build_arg_parser().format_help()
        self.assertIn("debug-model", help_text)

    def test_options_json_can_supply_required_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "dnn_model"
            run_dir.mkdir()
            (run_dir / "verification_summary.json").write_text(
                json.dumps(
                    {
                        "weighted_evm_pct": 0.25,
                        "passivity": {
                            "violating_points": 0,
                            "max_singular_value": 0.999,
                        },
                    }
                )
            )
            config = root / "options.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": {
                            "commands": {
                                "fit": {"mdif": "training.mdif"}
                            }
                        },
                        "workflows": {
                            "debug-model": {
                                "commands": {
                                    "debug-model": {"run_dir": str(run_dir)}
                                }
                            }
                        },
                    }
                )
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DEBUG.main(["--options-json", str(config)]), 0)
            self.assertTrue((run_dir / "model_debug" / "model_debug.md").is_file())
            payload = json.loads(
                (run_dir / "model_debug" / "model_debug.json").read_text()
            )
            commands = [
                suggestion["command"]
                for suggestion in payload["suggested_commands"]
            ]
            self.assertTrue(any("--options-json" in command for command in commands))
            self.assertTrue(all("PATH_TO_" not in command for command in commands))


if __name__ == "__main__":
    unittest.main()
