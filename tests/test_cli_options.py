import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit_dataset
import dnn
import generate_points
import kbnn
import neuro_tf
from de_generated_scripts import parse_ads_hb_solver_log

from cli_options import (
    add_options_json_argument,
    finalize_options_json_update,
    fit_shared_option_keys,
    load_options_json_resolution,
    normalize_command_name,
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
    def test_audit_reuses_common_fit_mdif_inputs_from_options_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "commands": {
                            "fit": {
                                "mdif": "data/training.mdif",
                                "verification_mdif": "data/verification.mdif",
                            }
                        }
                    },
                    "workflows": {
                        "audit": {
                            "common": {
                                "mdif": None,
                                "verification_mdif": None,
                            }
                        }
                    },
                },
            )
            args = parse_args_with_options_json(
                audit_dataset.build_parser(),
                ["--options-json", str(config)],
                workflow="audit",
                command="audit",
            )
            defaults, sources = load_options_json_resolution(
                config,
                workflow="audit",
                command="audit",
            )

        self.assertEqual(args.mdif, "data/training.mdif")
        self.assertEqual(args.verification_mdif, "data/verification.mdif")
        self.assertEqual(defaults["mdif"], "data/training.mdif")
        self.assertEqual(defaults["verification_mdif"], "data/verification.mdif")
        self.assertIn("models.commands.fit.mdif", sources["mdif"])
        self.assertIn(
            "models.commands.fit.verification_mdif",
            sources["verification_mdif"],
        )

    def test_audit_specific_mdif_inputs_override_common_fit_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "commands": {
                            "fit": {
                                "mdif": "data/model_train.mdif",
                                "verification_mdif": "data/model_verify.mdif",
                            }
                        }
                    },
                    "workflows": {
                        "audit": {
                            "common": {
                                "mdif": "data/audit_train.mdif",
                                "verification_mdif": "data/audit_verify.mdif",
                            }
                        }
                    },
                },
            )
            args = parse_args_with_options_json(
                audit_dataset.build_parser(),
                ["--options-json", str(config)],
                workflow="audit",
                command="audit",
            )

        self.assertEqual(args.mdif, "data/audit_train.mdif")
        self.assertEqual(args.verification_mdif, "data/audit_verify.mdif")

    def test_audit_reuses_unambiguous_model_specific_fit_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "dnn": {
                            "commands": {
                                "fit": {
                                    "mdif": "data/dnn_train.mdif",
                                    "verification_mdif": "data/dnn_verify.mdif",
                                }
                            }
                        }
                    },
                    "workflows": {"audit": {"common": {"mdif": None}}},
                },
            )
            args = parse_args_with_options_json(
                audit_dataset.build_parser(),
                ["--options-json", str(config)],
                workflow="audit",
                command="audit",
            )

        self.assertEqual(args.mdif, "data/dnn_train.mdif")
        self.assertEqual(args.verification_mdif, "data/dnn_verify.mdif")

    def test_audit_does_not_guess_between_conflicting_model_fit_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "dnn": {
                            "commands": {
                                "fit": {"mdif": "data/dnn_train.mdif"}
                            }
                        },
                        "kbnn": {
                            "commands": {
                                "fit": {"mdif": "data/kbnn_train.mdif"}
                            }
                        },
                    },
                    "workflows": {"audit": {"common": {"mdif": None}}},
                },
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                parse_args_with_options_json(
                    audit_dataset.build_parser(),
                    ["--options-json", str(config)],
                    workflow="audit",
                    command="audit",
                )

        self.assertIn("--mdif", stderr.getvalue())

    def test_boolean_optional_export_setting_preserves_json_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "dnn": {
                            "commands": {
                                "export-ads-ann": {
                                    "include_dc": False,
                                    "mdif": "training.mdif",
                                    "out_dir": "ann_export",
                                }
                            }
                        }
                    }
                },
            )
            args = parse_args_with_options_json(
                dnn.build_arg_parser(),
                ["export-ads-ann", "--options-json", str(config)],
                model="dnn",
            )

        self.assertIs(args.include_dc, False)

    def test_optimize_reuses_fit_compatible_train_settings_when_not_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "dnn": {
                            "commands": {
                                "train": {
                                    "activation": "tanh",
                                    "frequency_weights": "default=1;2GHz=4",
                                    "mdif": "training.mdif",
                                },
                                "optimize": {"out_dir": "optimization"},
                            }
                        }
                    }
                },
            )
            args = parse_args_with_options_json(
                dnn.build_arg_parser(),
                ["optimize", "--options-json", str(config)],
                model="dnn",
            )

        self.assertEqual(args.mdif, "training.mdif")
        self.assertEqual(args.activation_options, "tanh")
        self.assertEqual(args.frequency_weights, "default=1;2GHz=4")
        self.assertEqual(args.out_dir, "optimization")

    def test_required_positional_arguments_can_be_loaded_and_updated_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "workflows": {
                        "hb-report": {
                            "commands": {"hb-report": {"logs": ["saved.log"]}}
                        }
                    }
                },
            )
            loaded = parse_args_with_options_json(
                parse_ads_hb_solver_log.build_arg_parser(),
                ["--options-json", str(config)],
                workflow="hb-report",
                command="hb-report",
            )
            self.assertEqual(loaded.logs, ["saved.log"])

            updated = parse_args_with_options_json(
                parse_ads_hb_solver_log.build_arg_parser(),
                [
                    "new-baseline.log",
                    "new-trial.log",
                    "--options-json",
                    str(config),
                    "--update-options-json",
                ],
                workflow="hb-report",
                command="hb-report",
            )
            self.assertEqual(finalize_options_json_update(updated, 0), 0)
            payload = json.loads(config.read_text())

        self.assertEqual(
            payload["workflows"]["hb-report"]["commands"]["hb-report"]["logs"],
            ["new-baseline.log", "new-trial.log"],
        )

    def test_every_required_repository_argument_is_represented_in_starter_json(self) -> None:
        model_commands = {
            "dnn": (
                dnn.build_arg_parser,
                (
                    "train",
                    "optimize",
                    "rerank-sweep",
                    "predict",
                    "export-ads-mdif",
                    "export-ads-ann",
                    "export-ads-hb",
                    "export-veriloga",
                    "inspect-mdif",
                ),
            ),
            "kbnn": (
                kbnn.build_arg_parser,
                (
                    "train",
                    "optimize",
                    "rerank-sweep",
                    "predict",
                    "export-ads-mdif",
                    "export-ads-ann",
                    "export-ads-hb",
                    "export-veriloga",
                    "inspect-mdif",
                ),
            ),
            "neuro-tf": (
                neuro_tf.build_arg_parser,
                (
                    "train",
                    "optimize",
                    "predict",
                    "export-ads-mdif",
                    "export-ads-hb",
                    "export-veriloga",
                    "inspect-mdif",
                ),
            ),
        }
        workflow_commands = (
            (
                "points",
                "generate",
                generate_points.build_generate_parser,
            ),
            (
                "points",
                "suggest-additional",
                generate_points.build_suggest_parser,
            ),
            ("audit", "audit", audit_dataset.build_parser),
            (
                "hb-report",
                "hb-report",
                parse_ads_hb_solver_log.build_arg_parser,
            ),
        )

        def selected_parser(
            parser: argparse.ArgumentParser, command: str
        ) -> argparse.ArgumentParser:
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for raw_name, child in action.choices.items():
                        if normalize_command_name(raw_name) == command:
                            return child
            return parser

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), starter_options_payload())
            cases: list[tuple[argparse.ArgumentParser, str, dict[str, str]]] = []
            for model, (builder, commands) in model_commands.items():
                cases.extend(
                    (builder(), command, {"model": model})
                    for command in commands
                )
            cases.extend(
                (builder(), command, {"workflow": workflow})
                for workflow, command, builder in workflow_commands
            )
            for parser, command, scope in cases:
                with self.subTest(command=command, scope=scope):
                    defaults, _ = load_options_json_resolution(
                        config,
                        command=command,
                        **scope,
                    )
                    child = selected_parser(parser, command)
                    required = [
                        action
                        for action in child._actions
                        if action.dest != "help"
                        and (
                            bool(getattr(action, "required", False))
                            or (not action.option_strings and action.nargs == "+")
                        )
                    ]
                    for action in required:
                        self.assertIn(
                            action.dest,
                            defaults,
                            f"{scope} {command} omits required {action.dest}",
                        )

    def test_json_can_supply_every_required_repository_argument(self) -> None:
        cases = [
            *(
                (model, command, builder)
                for model, builder, commands in (
                    (
                        "dnn",
                        dnn.build_arg_parser,
                        (
                            "train",
                            "optimize",
                            "rerank-sweep",
                            "predict",
                            "export-ads-mdif",
                            "export-ads-ann",
                            "export-ads-hb",
                            "export-veriloga",
                            "inspect-mdif",
                        ),
                    ),
                    (
                        "kbnn",
                        kbnn.build_arg_parser,
                        (
                            "train",
                            "optimize",
                            "rerank-sweep",
                            "predict",
                            "export-ads-mdif",
                            "export-ads-ann",
                            "export-ads-hb",
                            "export-veriloga",
                            "inspect-mdif",
                        ),
                    ),
                    (
                        "neuro-tf",
                        neuro_tf.build_arg_parser,
                        (
                            "train",
                            "optimize",
                            "predict",
                            "export-ads-mdif",
                            "export-ads-hb",
                            "export-veriloga",
                            "inspect-mdif",
                        ),
                    ),
                )
                for command in commands
            )
        ]

        def child_parser(
            parser: argparse.ArgumentParser, command: str
        ) -> argparse.ArgumentParser:
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    return next(
                        child
                        for raw_name, child in action.choices.items()
                        if normalize_command_name(raw_name) == command
                    )
            return parser

        def required_values(
            parser: argparse.ArgumentParser, command: str
        ) -> dict[str, object]:
            values: dict[str, object] = {}
            for action in child_parser(parser, command)._actions:
                required = bool(getattr(action, "required", False)) or (
                    not action.option_strings and action.nargs == "+"
                )
                if not required or action.dest == "help":
                    continue
                if action.dest in {"count"}:
                    values[action.dest] = 1
                elif isinstance(action, argparse._AppendAction):
                    values[action.dest] = ["W=1:2"]
                elif not action.option_strings:
                    values[action.dest] = ["simulation.log"]
                else:
                    values[action.dest] = f"{action.dest}.value"
            return values

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (model, command, builder) in enumerate(cases):
                parser = builder()
                values = required_values(parser, command)
                config = self.write_config(
                    root / f"model-{index}",
                    {
                        "models": {
                            model: {"commands": {command: values}}
                        }
                    },
                )
                args = parse_args_with_options_json(
                    parser,
                    [command, "--options-json", str(config)],
                    model=model,
                )
                for key, value in values.items():
                    self.assertEqual(getattr(args, key), value)

            workflow_cases = (
                ("points", "generate", generate_points.build_generate_parser),
                ("points", "suggest-additional", generate_points.build_suggest_parser),
                ("audit", "audit", audit_dataset.build_parser),
                (
                    "hb-report",
                    "hb-report",
                    parse_ads_hb_solver_log.build_arg_parser,
                ),
            )
            for index, (workflow, command, builder) in enumerate(workflow_cases):
                parser = builder()
                values = required_values(parser, command)
                config = self.write_config(
                    root / f"workflow-{index}",
                    {
                        "workflows": {
                            workflow: {"commands": {command: values}}
                        }
                    },
                )
                args = parse_args_with_options_json(
                    parser,
                    ["--options-json", str(config)],
                    workflow=workflow,
                    command=command,
                )
                for key, value in values.items():
                    self.assertEqual(getattr(args, key), value)

    def test_all_common_train_optimize_options_are_fit_compatible(self) -> None:
        excluded_run_controls = {
            "explain_options",
            "help",
            "options_json",
            "out_dir",
            "show_options",
            "update_options_json",
        }
        for model, module in (
            ("dnn", dnn),
            ("kbnn", kbnn),
            ("neuro-tf", neuro_tf),
        ):
            parser = module.build_arg_parser()
            children = {
                normalize_command_name(raw_name): child
                for action in parser._actions
                if isinstance(action, argparse._SubParsersAction)
                for raw_name, child in action.choices.items()
            }

            def option_names(command: str) -> set[str]:
                return {
                    option[2:].replace("-", "_")
                    for action in children[command]._actions
                    for option in action.option_strings
                    if option.startswith("--")
                }

            missing = (
                option_names("train")
                & option_names("optimize")
                - fit_shared_option_keys(model)
                - excluded_run_controls
            )
            self.assertEqual(missing, set(), model)

    def test_starter_json_exposes_conditional_sampled_mdif_inputs(self) -> None:
        payload = starter_options_payload()
        for model in ("dnn", "kbnn", "neuro-tf"):
            export = payload["models"][model]["commands"]["export-ads-mdif"]
            self.assertIn("template_mdif", export)
            self.assertIn("parameter_grid", export)
            self.assertIn("freqs", export)

    def write_config(self, root: Path, payload: dict[str, object]) -> Path:
        root.mkdir(parents=True, exist_ok=True)
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

    def test_update_options_json_saves_explicit_values_at_exact_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "schema_version": 1,
                    "models": {
                        "commands": {"fit": {"epochs": 200}},
                        "dnn": {"commands": {}},
                    },
                },
            )
            args = parse_args_with_options_json(
                example_parser(),
                [
                    "train",
                    "--options-json",
                    str(config),
                    "--update-options-json",
                    "--mdif",
                    "data/new.mdif",
                    "--epochs",
                    "500",
                    "--parameter",
                    "W=1:2",
                    "--parameter",
                    "L=3:4",
                    "--require-passive",
                ],
                model="dnn",
            )
            status = finalize_options_json_update(args, 0)
            payload = json.loads(config.read_text())

        self.assertEqual(status, 0)
        self.assertEqual(
            payload["models"]["dnn"]["commands"]["train"],
            {
                "mdif": "data/new.mdif",
                "epochs": 500,
                "parameter": ["W=1:2", "L=3:4"],
                "require_passive": True,
            },
        )
        self.assertEqual(payload["models"]["commands"]["fit"]["epochs"], 200)

    def test_update_options_json_requires_path_and_skips_runtime_failure(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            parse_args_with_options_json(
                example_parser(),
                ["train", "--update-options-json", "--mdif", "x"],
                model="dnn",
            )
        self.assertIn("requires --options-json", stderr.getvalue())

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), {"schema_version": 1})
            original = config.read_text()
            args = parse_args_with_options_json(
                example_parser(),
                [
                    "train",
                    "--options-json",
                    str(config),
                    "--update-options-json",
                    "--mdif",
                    "x",
                ],
                model="dnn",
            )
            status = finalize_options_json_update(args, 2)
            unchanged = config.read_text()

        self.assertEqual(status, 2)
        self.assertEqual(unchanged, original)

    def test_options_json_cannot_enable_its_own_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "dnn": {
                            "commands": {
                                "train": {"update_options_json": True}
                            }
                        }
                    }
                },
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                parse_args_with_options_json(
                    example_parser(),
                    ["train", "--options-json", str(config), "--mdif", "x"],
                    model="dnn",
                )
        self.assertIn("cannot enable --update-options-json", stderr.getvalue())

    def test_explain_options_reports_effective_values_and_exact_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(
                Path(temp_dir),
                {
                    "models": {
                        "commands": {"fit": {"frequency_weights": "default=2"}},
                        "dnn": {
                            "commands": {
                                "train": {"mdif": "data/from_json.mdif"}
                            }
                        },
                    }
                },
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(
                SystemExit
            ) as stopped:
                parse_args_with_options_json(
                    example_parser(),
                    [
                        "train",
                        "--options-json",
                        str(config),
                        "--epochs",
                        "600",
                        "--explain-options",
                    ],
                    model="dnn",
                )

        report = stdout.getvalue()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("command was not executed", report)
        self.assertIn("JSON: models.dnn.commands.train.mdif", report)
        self.assertIn("JSON: models.commands.fit.frequency_weights", report)
        self.assertIn("CLI (--epochs)", report)
        self.assertIn("Required options: OK", report)
        self.assertIn("Validation result: OK", report)

    def test_explain_options_colors_ok_and_can_execute_without_reentry(self) -> None:
        class InteractiveInput(io.StringIO):
            def isatty(self) -> bool:
                return True

        class InteractiveOutput(io.StringIO):
            def isatty(self) -> bool:
                return True

        stdout = InteractiveOutput()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("cli_options.sys.stdin", InteractiveInput()),
            mock.patch("builtins.input", return_value="yes") as confirmation,
            contextlib.redirect_stdout(stdout),
        ):
            args = parse_args_with_options_json(
                example_parser(),
                ["train", "--mdif", "data.mdif", "--explain-options"],
                model="dnn",
            )

        report = stdout.getvalue()
        self.assertEqual(args.mdif, "data.mdif")
        self.assertIn("\033[32m\nRequired options: OK\033[0m", report)
        self.assertIn("\033[32m\nValidation result: OK\033[0m", report)
        self.assertIn("Executing command with the validated options", report)
        confirmation.assert_called_once_with(
            "\nExecute this command with the options above? [y/N] "
        )

    def test_explain_options_colors_missing_and_does_not_prompt(self) -> None:
        class InteractiveInput(io.StringIO):
            def isatty(self) -> bool:
                return True

        stdout = io.StringIO()
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("cli_options.sys.stdin", InteractiveInput()),
            mock.patch("builtins.input") as confirmation,
            contextlib.redirect_stdout(stdout),
            self.assertRaises(SystemExit) as stopped,
        ):
            parse_args_with_options_json(
                example_parser(),
                ["train", "--explain-options"],
                model="dnn",
            )

        report = stdout.getvalue()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("\033[31m\nRequired options: MISSING", report)
        self.assertIn("\033[31m\nValidation result: MISSING", report)
        confirmation.assert_not_called()

    def test_approved_explanation_executes_selected_workflow(self) -> None:
        stdout = io.StringIO()
        with (
            mock.patch(
                "cli_options._confirm_explained_command",
                return_value=True,
            ) as confirmation,
            mock.patch(
                "generate_points.command_generate",
                return_value=0,
            ) as command,
            contextlib.redirect_stdout(stdout),
        ):
            status = generate_points.main(
                [
                    "generate",
                    "--parameter",
                    "W=1:2",
                    "--count",
                    "4",
                    "--explain-options",
                ]
            )

        self.assertEqual(status, 0)
        confirmation.assert_called_once_with()
        command.assert_called_once()

    def test_explain_additional_points_preflights_domain_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metrics = root / "verification_metrics.csv"
            metrics.write_text("W,evm_pct\n1.0,2.0\n", encoding="utf-8")
            config = self.write_config(
                root,
                {
                    "workflows": {
                        "points": {
                            "commands": {
                                "suggest-additional": {
                                    "parameter": ["W=0:2"],
                                    "count": 4,
                                    "verification_metrics": str(metrics),
                                    "metric": "weighted_evm_pct",
                                }
                            }
                        }
                    }
                },
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(SystemExit):
                parse_args_with_options_json(
                    generate_points.build_suggest_parser(),
                    ["--options-json", str(config), "--show-options"],
                    workflow="points",
                    command="suggest-additional",
                )

        report = stdout.getvalue()
        self.assertIn("parameter domain: OK, 1 --parameter value", report)
        self.assertIn("COLUMN NOT FOUND: weighted_evm_pct", report)
        self.assertIn("weighted_* values are fit-summary metrics", report)

    def test_explain_and_update_captures_options_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), {"schema_version": 1})
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), self.assertRaises(
                SystemExit
            ) as stopped:
                parse_args_with_options_json(
                    example_parser(),
                    [
                        "train",
                        "--options-json",
                        str(config),
                        "--mdif",
                        "data/already_used.mdif",
                        "--epochs",
                        "750",
                        "--frequency-weights",
                        "default=1;2GHz=4",
                        "--explain-options",
                        "--update-options-json",
                    ],
                    model="dnn",
                )
            payload = json.loads(config.read_text())

        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(
            payload["models"]["dnn"]["commands"]["train"],
            {
                "mdif": "data/already_used.mdif",
                "epochs": 750,
                "frequency_weights": "default=1;2GHz=4",
            },
        )
        self.assertIn("command was not executed", stdout.getvalue())
        self.assertIn("updated", stdout.getvalue())

    def test_points_main_commits_update_after_command_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self.write_config(Path(temp_dir), {"schema_version": 1})
            with mock.patch(
                "generate_points.command_generate",
                return_value=0,
            ) as command:
                status = generate_points.main(
                    [
                        "generate",
                        "--options-json",
                        str(config),
                        "--update-options-json",
                        "--parameter",
                        "W=1:2",
                        "--count",
                        "12",
                        "--out",
                        "geometries.csv",
                    ]
                )
            payload = json.loads(config.read_text())

        self.assertEqual(status, 0)
        command.assert_called_once()
        self.assertEqual(
            payload["workflows"]["points"]["commands"]["generate"],
            {
                "parameter": ["W=1:2"],
                "count": 12,
                "out": "geometries.csv",
            },
        )

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
