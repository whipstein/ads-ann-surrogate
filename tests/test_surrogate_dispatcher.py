import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

import surrogate
from surrogate_common import build_training_export_commands


ROOT = Path(__file__).resolve().parents[1]


class SurrogateDispatcherTests(unittest.TestCase):
    def test_dispatch_forwards_backend_arguments_and_exit_status(self) -> None:
        completed = mock.Mock(returncode=7)
        with mock.patch("surrogate.subprocess.run", return_value=completed) as run:
            status = surrogate.main(
                [
                    "--model",
                    "kbnn",
                    "train",
                    "--mdif",
                    "relative/input.mdif",
                ]
            )
        self.assertEqual(status, 7)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(
            command,
            [
                sys.executable or "python3",
                str(ROOT / "kbnn.py"),
                "train",
                "--mdif",
                "relative/input.mdif",
            ],
        )
        self.assertFalse(run.call_args.kwargs["check"])
        self.assertEqual(
            run.call_args.kwargs["env"]["ADS_SURROGATE_CLI_PROG"],
            "surrogate.py --model kbnn",
        )

    def test_model_option_can_appear_after_the_backend_subcommand(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch("surrogate.subprocess.run", return_value=completed) as run:
            status = surrogate.main(
                ["inspect-mdif", "--model", "dnn", "--mdif", "input.mdif"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(run.call_args.args[0][2:], ["inspect-mdif", "--mdif", "input.mdif"])

    def test_kbnn_mode_is_not_consumed_as_a_model_abbreviation(self) -> None:
        args, backend_args = surrogate.parse_dispatch_args(
            ["--model", "kbnn", "train", "--mode", "residual"]
        )
        self.assertEqual(args.model_type, "kbnn")
        self.assertEqual(backend_args, ["train", "--mode", "residual"])

    def test_neuro_tf_alias_is_canonicalized(self) -> None:
        completed = mock.Mock(returncode=0)
        with mock.patch("surrogate.subprocess.run", return_value=completed) as run:
            status = surrogate.main(
                ["--model", "neuro_tf", "predict", "--help"]
            )
        self.assertEqual(status, 0)
        self.assertEqual(Path(run.call_args.args[0][1]).name, "neuro_tf.py")
        self.assertEqual(run.call_args.args[0][2:], ["predict", "--help"])

    def test_top_level_help_lists_every_model_type(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = surrogate.main(["--help"])
        self.assertEqual(status, 0)
        for model_type in surrogate.MODEL_SCRIPTS:
            self.assertIn(model_type, output.getvalue())
        for workflow in surrogate.WORKFLOW_SCRIPTS:
            self.assertIn(workflow, output.getvalue())

    def test_every_model_backend_exists(self) -> None:
        for script_name in surrogate.MODEL_SCRIPTS.values():
            self.assertTrue((ROOT / script_name).is_file())

    def test_every_workflow_implementation_exists(self) -> None:
        for script_name in surrogate.WORKFLOW_SCRIPTS.values():
            self.assertTrue((ROOT / script_name).is_file())

    def test_non_model_workflows_forward_arguments_and_primary_prog(self) -> None:
        cases = (
            (
                "points",
                ["generate", "--count", "24"],
                ROOT / "generate_points.py",
            ),
            ("audit", ["--mdif", "data.mdif"], ROOT / "audit_dataset.py"),
            (
                "hb-report",
                ["baseline.log", "trial.log"],
                ROOT / "de_generated_scripts" / "parse_ads_hb_solver_log.py",
            ),
        )
        for workflow, forwarded, implementation in cases:
            with self.subTest(workflow=workflow):
                completed = mock.Mock(returncode=5)
                with mock.patch(
                    "surrogate.subprocess.run", return_value=completed
                ) as run:
                    status = surrogate.main([workflow, *forwarded])
                self.assertEqual(status, 5)
                self.assertEqual(
                    run.call_args.args[0],
                    [sys.executable or "python3", str(implementation), *forwarded],
                )
                self.assertEqual(
                    run.call_args.kwargs["env"]["ADS_SURROGATE_CLI_PROG"],
                    f"surrogate.py {workflow}",
                )

    def test_generated_export_commands_use_the_dispatcher(self) -> None:
        for model_type, script_name in surrogate.MODEL_SCRIPTS.items():
            commands = build_training_export_commands(
                ROOT / script_name,
                ROOT / "outputs" / f"{model_type}_example",
                include_veriloga=True,
                model_type=model_type,
            )
            self.assertTrue(commands)
            for _, command in commands:
                self.assertIn(
                    f"surrogate.py --model {model_type}",
                    command,
                )
                self.assertNotIn(f" {script_name} ", command)


if __name__ == "__main__":
    unittest.main()
