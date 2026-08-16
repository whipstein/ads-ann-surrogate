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

from cli_options import add_options_json_argument, parse_args_with_options_json

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
        cases = (
            (
                dnn.build_arg_parser(),
                ["optimize", "--mdif", "fine.mdif", "--out-dir", "dnn"],
                {"model": "dnn"},
            ),
            (
                kbnn.build_arg_parser(),
                ["train", "--mdif", "fine.mdif", "--out-dir", "kbnn"],
                {"model": "kbnn"},
            ),
            (
                neuro_tf.build_arg_parser(),
                ["train", "--mdif", "fine.mdif", "--out-dir", "neuro"],
                {"model": "neuro-tf"},
            ),
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


if __name__ == "__main__":
    unittest.main()
