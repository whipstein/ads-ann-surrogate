import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import audit_dataset
import dnn
import generate_points
import kbnn
import neuro_tf

from cli_options import (
    add_options_json_argument,
    parse_args_with_options_json,
    starter_options_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def example_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--mdif", required=True)
    train.add_argument("--frequency-weights")
    train.add_argument("--sparam-weights")
    train.add_argument("--parameter", action="append", default=[])
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--passivity-mode", choices=["auto", "enforce", "off"])
    train.add_argument("--require-passive", action="store_true")
    add_options_json_argument(parser)
    return parser


class OptionsJSONTests(unittest.TestCase):
    def write_config(self, root: Path, payload: dict[str, object]) -> Path:
        path = root / "options.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_structured_defaults_required_values_and_model_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "commands": {
                        "fit": {
                            "frequency_weights": "default=1;1GHz=3",
                            "epochs": 250,
                            "passivity_mode": "auto",
                        }
                    },
                    "models": {
                        "dnn": {
                            "commands": {
                                "train": {
                                    "mdif": "from_json.mdif",
                                    "parameter": ["W=1:2", "L=3:4"],
                                    "require_passive": True,
                                    "sparam_weights": "diag=1;offdiag=0.2",
                                }
                            }
                        }
                    },
                },
            )
            args = parse_args_with_options_json(
                example_parser(),
                ["train", "--options-json", str(config)],
                model="dnn",
            )

        self.assertEqual(args.mdif, "from_json.mdif")
        self.assertEqual(args.frequency_weights, "default=1;1GHz=3")
        self.assertEqual(args.sparam_weights, "diag=1;offdiag=0.2")
        self.assertEqual(args.parameter, ["W=1:2", "L=3:4"])
        self.assertEqual(args.epochs, 250)
        self.assertEqual(args.passivity_mode, "auto")
        self.assertTrue(args.require_passive)
        self.assertEqual(args.options_json, str(config))

    def test_explicit_cli_values_replace_json_defaults_including_append(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "common": {
                        "mdif": "from_json.mdif",
                        "frequency_weights": "default=1;1GHz=3",
                        "parameter": ["W=1:2", "L=3:4"],
                    }
                },
            )
            args = parse_args_with_options_json(
                example_parser(),
                [
                    "--options-json",
                    str(config),
                    "train",
                    "--mdif",
                    "from_cli.mdif",
                    "--frequency-weights",
                    "default=9",
                    "--parameter",
                    "H=5:6",
                ],
                model="dnn",
            )

        self.assertEqual(args.mdif, "from_cli.mdif")
        self.assertEqual(args.frequency_weights, "default=9")
        self.assertEqual(args.parameter, ["H=5:6"])

    def test_model_wide_defaults_and_lower_scope_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "common": {"epochs": 100},
                    "commands": {"fit": {"epochs": 200}},
                    "models": {
                        "common": {
                            "epochs": 300,
                            "frequency_weights": "models-common",
                        },
                        "commands": {
                            "fit": {
                                "epochs": 400,
                                "passivity_mode": "enforce",
                            }
                        },
                        "dnn": {
                            "common": {
                                "frequency_weights": "dnn-common",
                            },
                            "commands": {
                                "train": {
                                    "epochs": 500,
                                }
                            },
                        },
                    },
                },
            )
            dnn_args = parse_args_with_options_json(
                example_parser(),
                ["train", "--options-json", str(config), "--mdif", "x"],
                model="dnn",
            )
            kbnn_args = parse_args_with_options_json(
                example_parser(),
                ["train", "--options-json", str(config), "--mdif", "x"],
                model="kbnn",
            )
            cli_args = parse_args_with_options_json(
                example_parser(),
                [
                    "train",
                    "--options-json",
                    str(config),
                    "--mdif",
                    "x",
                    "--epochs",
                    "600",
                ],
                model="dnn",
            )

        self.assertEqual(dnn_args.epochs, 500)
        self.assertEqual(dnn_args.frequency_weights, "dnn-common")
        self.assertEqual(dnn_args.passivity_mode, "enforce")
        self.assertEqual(kbnn_args.epochs, 400)
        self.assertEqual(kbnn_args.frequency_weights, "models-common")
        self.assertEqual(kbnn_args.passivity_mode, "enforce")
        self.assertEqual(cli_args.epochs, 600)

    def test_invalid_option_is_rejected_for_the_selected_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {"models": {"dnn": {"common": {"not_an_option": 1}}}},
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                parse_args_with_options_json(
                    example_parser(),
                    ["train", "--options-json", str(config), "--mdif", "x"],
                    model="dnn",
                )
        self.assertIn("--not-an-option", stderr.getvalue())
        self.assertIn("is not valid for train", stderr.getvalue())

    def test_repository_example_is_valid_for_every_documented_scope(self) -> None:
        config = str(ROOT / "options.example.json")
        self.assertEqual(
            json.loads(Path(config).read_text()),
            starter_options_payload(),
        )
        model_cases = tuple(
            (
                module.build_arg_parser(),
                [command, "--mdif", "fine.mdif", "--out-dir", model],
                {"model": model},
            )
            for model, module in (
                ("dnn", dnn),
                ("kbnn", kbnn),
                ("neuro-tf", neuro_tf),
            )
            for command in ("train", "optimize")
        )
        cases = (
            *model_cases,
            (
                generate_points.build_generate_parser(),
                ["--parameter", "W=1:2", "--count", "4"],
                {"workflow": "points", "command": "generate"},
            ),
            (
                audit_dataset.build_parser(),
                ["--mdif", "fine.mdif"],
                {"workflow": "audit", "command": "audit"},
            ),
        )
        for parser, arguments, scope in cases:
            with self.subTest(scope=scope):
                args = parse_args_with_options_json(
                    parser,
                    [*arguments, "--options-json", config],
                    **scope,
                )
                self.assertEqual(args.options_json, config)

    def test_readme_project_configuration_runs_documented_commands(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        section = readme.split("### Practical Project Configuration", 1)[1]
        json_text = section.split("```json\n", 1)[1].split("\n```", 1)[0]
        payload = json.loads(json_text)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), payload)
            cases = (
                (
                    generate_points.build_generate_parser(),
                    [],
                    {"workflow": "points", "command": "generate"},
                ),
                (
                    generate_points.build_suggest_parser(),
                    [],
                    {"workflow": "points", "command": "suggest-additional"},
                ),
                (
                    audit_dataset.build_parser(),
                    [],
                    {"workflow": "audit", "command": "audit"},
                ),
                (
                    dnn.build_arg_parser(),
                    ["optimize"],
                    {"model": "dnn"},
                ),
                (
                    dnn.build_arg_parser(),
                    ["export-veriloga"],
                    {"model": "dnn"},
                ),
                (
                    kbnn.build_arg_parser(),
                    ["optimize"],
                    {"model": "kbnn"},
                ),
                (
                    neuro_tf.build_arg_parser(),
                    ["optimize"],
                    {"model": "neuro-tf"},
                ),
            )
            for parser, arguments, scope in cases:
                with self.subTest(scope=scope, arguments=arguments):
                    args = parse_args_with_options_json(
                        parser,
                        [*arguments, "--options-json", str(config)],
                        **scope,
                    )
                    self.assertEqual(args.options_json, str(config))


if __name__ == "__main__":
    unittest.main()
