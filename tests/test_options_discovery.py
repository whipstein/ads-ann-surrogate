import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import surrogate
from options_discovery import DiscoveryAccumulator, discover_options


class OptionsDiscoveryTests(unittest.TestCase):
    def test_scalar_object_path_collision_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            accumulator = DiscoveryAccumulator(root=root)
            source = root / "artifact.json"
            source.write_text("{}", encoding="utf-8")
            accumulator.add(
                ("models", "dnn", "commands", "train", "out_dir"),
                "model",
                source,
                priority=50,
            )
            accumulator.add(
                (
                    "models",
                    "dnn",
                    "commands",
                    "train",
                    "out_dir",
                    "out_dir",
                ),
                "nested-model",
                source,
                priority=40,
            )

            payload = accumulator.payload()

        self.assertEqual(
            payload["models"]["dnn"]["commands"]["train"]["out_dir"],
            "model",
        )
        self.assertEqual(len(accumulator.conflicts), 1)

    def test_options_like_nested_value_is_skipped_without_scalar_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            (root / "nested_state.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "models": {
                            "dnn": {
                                "commands": {
                                    "train": {
                                        "out_dir": {"out_dir": "not-an-option-value"}
                                    }
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "parameter_names": ["W"],
                        "sparam_labels": ["S11"],
                        "layer_sizes": [2, 8, 2],
                        "activation": "tanh",
                        "freq_transform": "log",
                        "output_domain": "s",
                    }
                ),
                encoding="utf-8",
            )

            payload, report = discover_options(root)

        train = payload["models"]["dnn"]["commands"]["train"]
        self.assertIsInstance(train["out_dir"], str)
        self.assertTrue(
            any(
                "nested_state.json" in warning
                and "option value must be" in warning
                for warning in report["warnings"]
            )
        )

    def test_recovers_recursive_geometry_model_and_command_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometry_dir = root / "campaign" / "geometry"
            model_dir = root / "outputs" / "dnn"
            geometry_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            geometry_json = geometry_dir / "geometries.json"
            geometry_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "geometry_file": "geometries.csv",
                        "generation_kind": "generated",
                        "method": "maximin-lhs",
                        "point_count": 24,
                        "split_variable": "dataset",
                        "split_counts": {"train": 18, "verification": 6},
                        "decimal_places": 5,
                        "parameters": [
                            {
                                "name": "W",
                                "range": {
                                    "lower": 0.4,
                                    "upper": 0.8,
                                    "unit": "mm",
                                },
                                "base_unit_range": {
                                    "lower": 4e-4,
                                    "upper": 8e-4,
                                },
                                "scale": "linear",
                            },
                            {
                                "name": "R",
                                "range": {"lower": 1, "upper": 100, "unit": ""},
                                "base_unit_range": {"lower": 1, "upper": 100},
                                "scale": "log",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "parameter_names": ["W", "R"],
                        "sparam_labels": ["S11", "S12", "S21", "S22"],
                        "layer_sizes": [3, 32, 16, 8],
                        "activation": "tanh",
                        "freq_transform": "log-linear",
                        "output_domain": "s",
                        "passivity_mode": "enforce",
                        "passivity_margin": 0.002,
                    }
                ),
                encoding="utf-8",
            )
            (model_dir / "training_summary.md").write_text(
                "# Training\n\n```bash\n"
                "python3 surrogate.py --model dnn train "
                "--mdif data/train_verify.mdif --out-dir outputs/dnn "
                "--hidden-layers 32,16 --activation tanh "
                "--frequency-weights 'default=1;2GHz=4'\n"
                "```\n",
                encoding="utf-8",
            )

            payload, report = discover_options(root)

        generate = payload["workflows"]["points"]["commands"]["generate"]
        self.assertEqual(generate["count"], 24)
        self.assertEqual(generate["verification_count"], 6)
        self.assertEqual(generate["decimal_places"], 5)
        self.assertEqual(generate["parameter"], ["W=0.4mm:0.8mm", "R=1:100:log"])
        self.assertEqual(payload["models"]["commands"]["fit"]["parameter_names"], "W,R")
        suggest = payload["workflows"]["points"]["commands"]["suggest-additional"]
        self.assertIn("parameter_json", suggest)
        self.assertEqual(len(suggest["existing_points"]), 1)
        train = payload["models"]["dnn"]["commands"]["train"]
        self.assertEqual(train["mdif"], "data/train_verify.mdif")
        self.assertEqual(train["out_dir"], "outputs/dnn")
        self.assertEqual(train["hidden_layers"], "32,16")
        self.assertEqual(train["frequency_weights"], "default=1;2GHz=4")
        self.assertIn("model_dir", payload["models"]["dnn"]["commands"]["export-veriloga"])
        self.assertEqual(len(report["recovered_commands"]), 1)
        self.assertGreaterEqual(report["settings_discovered"], 15)

    def test_existing_options_json_has_precedence_and_conflict_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "options-old.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "seed": 77,
                        "models": {
                            "dnn": {
                                "commands": {
                                    "train": {
                                        "activation": "relu",
                                        "mdif": "configured.mdif",
                                    }
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            (root / "training_summary.md").write_text(
                "```bash\npython3 surrogate.py --model dnn train "
                "--mdif report.mdif --out-dir fit --activation tanh\n```\n",
                encoding="utf-8",
            )

            payload, report = discover_options(root)

        train = payload["models"]["dnn"]["commands"]["train"]
        self.assertEqual(payload["seed"], 77)
        self.assertEqual(train["activation"], "relu")
        self.assertEqual(train["mdif"], "configured.mdif")
        conflict_settings = {row["setting"] for row in report["conflicts"]}
        self.assertIn("models.dnn.commands.train.activation", conflict_settings)
        self.assertIn("models.dnn.commands.train.mdif", conflict_settings)

    def test_dispatcher_writes_options_and_provenance_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            (nested / "saved-options.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workflows": {
                            "points": {
                                "commands": {"generate": {"count": 16}}
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            out_path = root / "recovered.json"
            report_path = root / "custom-report.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = surrogate.main(
                    [
                        "options",
                        "discover",
                        str(root),
                        "--out",
                        str(out_path),
                        "--report",
                        str(report_path),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(out_path.read_text())["workflows"]["points"]["commands"][
                    "generate"
                ]["count"],
                16,
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["settings_discovered"], 1)
            self.assertIn(f"wrote {out_path}", stdout.getvalue())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = surrogate.main(
                    ["options", "discover", str(root), "--out", str(out_path)]
                )
            self.assertEqual(status, 2)
            self.assertIn("--overwrite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
