import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import surrogate
from surrogate_common import (
    ADS_EXPORT_TEMPLATE_FILENAME,
    MDIFBlock,
    ads_qt_runtime_helper_script,
    build_ads_export_blocks,
    build_training_export_commands,
    read_mdif,
    write_ads_export_template,
    write_ads_ann_package,
)


ROOT = Path(__file__).resolve().parents[1]


def load_generated_ads_qt_runtime(module_name: str) -> dict[str, object]:
    """Execute the generated helper with a real temporary module identity."""

    module = types.ModuleType(module_name)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(
            compile(ads_qt_runtime_helper_script(), "ads_qt_runtime.py", "exec"),
            module.__dict__,
        )
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return module.__dict__


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
        self.assertIn("options", output.getvalue())
        self.assertIn("--options-json", output.getvalue())
        self.assertIn("--update-options-json", output.getvalue())
        self.assertIn("--explain-options", output.getvalue())

    def test_options_help_lists_recursive_discovery(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                surrogate.main(["options", "--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("discover", output.getvalue())

    def test_options_init_generates_template_and_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "config" / "options.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = surrogate.main(
                    ["options", "init", "--out", str(output_path)]
                )
            self.assertEqual(status, 0)
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                json.loads(output_path.read_text()),
                surrogate.starter_options_payload(),
            )
            self.assertIn(f"wrote {output_path}", stdout.getvalue())

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = surrogate.main(
                    ["options", "generate", "--out", str(output_path)]
                )
            self.assertEqual(status, 2)
            self.assertIn("--overwrite", stderr.getvalue())

            status = surrogate.main(
                [
                    "options",
                    "init",
                    "--out",
                    str(output_path),
                    "--overwrite",
                ]
            )
            self.assertEqual(status, 0)

    def test_options_json_can_precede_model_or_workflow_route(self) -> None:
        cases = (
            (
                ["--options-json", "defaults.json", "--model", "dnn", "train", "--mdif", "x"],
                ["train", "--mdif", "x", "--options-json", "defaults.json"],
            ),
            (
                ["--options-json=defaults.json", "audit", "--mdif", "x"],
                ["--mdif", "x", "--options-json", "defaults.json"],
            ),
            (
                [
                    "--update-options-json",
                    "--options-json",
                    "defaults.json",
                    "points",
                    "generate",
                    "--count",
                    "4",
                ],
                [
                    "generate",
                    "--count",
                    "4",
                    "--options-json",
                    "defaults.json",
                    "--update-options-json",
                ],
            ),
            (
                [
                    "--options-json",
                    "defaults.json",
                    "points",
                    "suggest-additional",
                    "--explain-options",
                ],
                [
                    "suggest-additional",
                    "--options-json",
                    "defaults.json",
                    "--explain-options",
                ],
            ),
        )
        for command, forwarded in cases:
            with self.subTest(command=command):
                completed = mock.Mock(returncode=0)
                with mock.patch(
                    "surrogate.subprocess.run", return_value=completed
                ) as run:
                    status = surrogate.main(command)
                self.assertEqual(status, 0)
                self.assertEqual(run.call_args.args[0][2:], forwarded)

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

    def test_generated_template_is_preferred_and_is_not_used_as_dc_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            source_mdif = root / "source.mdif"
            source_mdif.write_text("source placeholder")
            frequencies = np.asarray([0.0, 1.0e9, 2.0e9])
            blocks = [
                MDIFBlock(
                    params={"dataset": "train", "W": "0.5mm"},
                    freq_hz=frequencies,
                    sparams={"S11": np.ones(3, dtype=complex)},
                    source_index=8,
                )
            ]
            template_path = model_dir / ADS_EXPORT_TEMPLATE_FILENAME
            summary = write_ads_export_template(
                template_path,
                blocks,
                ["W"],
                ["S11"],
            )

            parsed = read_mdif(template_path)
            self.assertEqual(summary["block_count"], 1)
            self.assertEqual(parsed[0].params, {"W": "0.5mm"})
            np.testing.assert_array_equal(
                parsed[0].sparams["S11"],
                np.zeros(3, dtype=complex),
            )
            auto_blocks = build_ads_export_blocks(
                None,
                [],
                None,
                ["W"],
                ["S11"],
                model_dir=model_dir,
            )
            self.assertEqual(auto_blocks[0].params, {"W": "0.5mm"})
            np.testing.assert_array_equal(auto_blocks[0].freq_hz, frequencies)

            commands = dict(
                build_training_export_commands(
                    ROOT / "dnn.py",
                    model_dir,
                    dc_mdif=source_mdif,
                    include_veriloga=True,
                    model_type="dnn",
                )
            )
            sampled = commands["Sampled ADS MDIF"]
            self.assertIn(ADS_EXPORT_TEMPLATE_FILENAME, sampled)
            self.assertIn("--dc-mdif", sampled)
            self.assertIn("source.mdif", sampled)

    def test_ads_ann_package_bootstraps_qt_before_importing_ann(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "ads_ann"
            settings = {
                "seed": 1234,
                "num_hidden_layers": 2,
                "num_neurons_per_layer": 32,
                "neuron_activation_function_type": "TANH",
                "network_training_type": "STANDARD",
                "modeler_optimizer": "LM",
                "max_training_iterations": 100,
                "training_stop_tolerance": 1e-6,
                "output_format": "ALL",
                "output_prefix": "test_ann",
            }
            manifest = write_ads_ann_package(
                out_dir=out_dir,
                model_kind="DNN",
                input_columns=["W", "freq_hz"],
                output_columns=["fine_S11_real", "fine_S11_imag"],
                x_train=np.asarray([[0.5e-3, 1.0e9]]),
                y_train=np.asarray([[0.1, -0.2]]),
                x_verify=None,
                y_verify=None,
                settings=settings,
                parameter_names=["W"],
                sparam_labels=["S11"],
                target_description="test response",
            )

            self.assertEqual(manifest["qt_runtime_helper"], "ads_qt_runtime.py")
            helper_source = (out_dir / "ads_qt_runtime.py").read_text()
            training_source = (out_dir / "train_ads_ann.py").read_text()
            compile(helper_source, "ads_qt_runtime.py", "exec")
            compile(training_source, "train_ads_ann.py", "exec")
            self.assertLess(
                training_source.index("create_or_reuse_qapplication()"),
                training_source.index("import keysight.ads.ann as ann"),
            )
            self.assertIn("finally:", helper_source)
            self.assertIn("QCoreApplication.libraryPaths()", helper_source)
            self.assertIn('"QT_PLUGIN_PATH"', helper_source)
            self.assertIn('("HPEESOF_DIR", "EMPROHOME")', helper_source)
            self.assertIn("resolved_root.rglob(plugin_name)", helper_source)
            self.assertIn('os.environ.get("DISPLAY")', helper_source)
            self.assertIn('os.environ.get("WAYLAND_DISPLAY")', helper_source)
            self.assertIn("environment_was_restored", helper_source)
            self.assertEqual(
                manifest["qt_runtime"]["plugin_discovery"],
                "configured_paths_then_bounded_product_roots",
            )
            self.assertIn("ads_qt_runtime.py", (out_dir / "ADS_ANN_README.md").read_text())

    def test_ads_qt_runtime_helper_restores_platform_plugin_environment(self) -> None:
        helper_globals = load_generated_ads_qt_runtime("ads_qt_runtime_test")
        observed_paths: list[str | None] = []

        class FakeQApplication:
            @classmethod
            def instance(cls):
                return None

            def __init__(self, _argv):
                observed_paths.append(
                    os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
                )

            def platformName(self):
                return "test-platform"

        pyside = types.ModuleType("PySide6")
        pyside.__file__ = str(ROOT / "fake_ads" / "PySide6" / "__init__.py")
        pyside.__version__ = "test"
        qtwidgets = types.ModuleType("PySide6.QtWidgets")
        qtwidgets.QApplication = FakeQApplication

        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "platforms"
            plugin_file = plugin_dir / helper_globals[
                "expected_qt_platform_plugin"
            ]()
            helper_globals["locate_qt_platform_plugin"] = (
                lambda _pyside_file: plugin_file
            )
            helper_globals["validate_linux_plugin"] = lambda _plugin_file: None
            with mock.patch.dict(
                sys.modules,
                {
                    "PySide6": pyside,
                    "PySide6.QtWidgets": qtwidgets,
                },
            ):
                with mock.patch.dict(
                    os.environ,
                    {"QT_QPA_PLATFORM_PLUGIN_PATH": "original-qt-path"},
                    clear=False,
                ):
                    runtime = helper_globals[
                        "create_or_reuse_qapplication"
                    ]()
                    self.assertIsInstance(runtime.application, FakeQApplication)
                    self.assertEqual(
                        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"],
                        "original-qt-path",
                    )
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
                    absent_runtime = helper_globals[
                        "create_or_reuse_qapplication"
                    ]()
                    self.assertNotIn(
                        "QT_QPA_PLATFORM_PLUGIN_PATH",
                        os.environ,
                    )

        self.assertEqual(observed_paths, [str(plugin_dir), str(plugin_dir)])
        self.assertTrue(runtime.application_was_created)
        self.assertTrue(runtime.environment_was_restored)
        self.assertTrue(absent_runtime.environment_was_restored)
        details = helper_globals["qt_runtime_diagnostics"](runtime)
        self.assertEqual(details["qt_platform"], "test-platform")
        self.assertEqual(details["application_ownership"], "created_by_script")

    def test_ads_qt_runtime_locator_uses_qt_configured_library_paths(self) -> None:
        helper_globals = load_generated_ads_qt_runtime(
            "ads_qt_runtime_locator_test"
        )

        class FakeQLibraryInfo:
            class LibraryPath:
                PluginsPath = "plugins"

            @staticmethod
            def path(_library_path):
                return ""

        class FakeQCoreApplication:
            configured_paths: list[str] = []

            @classmethod
            def libraryPaths(cls):
                return cls.configured_paths

        qtcore = types.ModuleType("PySide6.QtCore")
        qtcore.QLibraryInfo = FakeQLibraryInfo
        qtcore.QCoreApplication = FakeQCoreApplication

        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_root = Path(temp_dir) / "qt_plugins"
            plugin_dir = plugin_root / "platforms"
            plugin_dir.mkdir(parents=True)
            plugin_file = plugin_dir / helper_globals[
                "expected_qt_platform_plugin"
            ]()
            plugin_file.touch()
            FakeQCoreApplication.configured_paths = [str(plugin_root)]
            with mock.patch.dict(
                sys.modules,
                {"PySide6.QtCore": qtcore},
            ):
                located = helper_globals["locate_qt_platform_plugin"](
                    Path(temp_dir) / "PySide6" / "__init__.py"
                )

        self.assertEqual(located, plugin_file.resolve())

    def test_ads_qt_runtime_helper_reuses_existing_application_without_search(self) -> None:
        helper_globals = load_generated_ads_qt_runtime(
            "ads_qt_runtime_reuse_test"
        )
        existing_application = object()

        class FakeQApplication:
            @classmethod
            def instance(cls):
                return existing_application

        pyside = types.ModuleType("PySide6")
        pyside.__file__ = str(ROOT / "fake_ads" / "PySide6" / "__init__.py")
        qtwidgets = types.ModuleType("PySide6.QtWidgets")
        qtwidgets.QApplication = FakeQApplication

        def unexpected_search(_pyside_file):
            self.fail("Qt plugin discovery ran despite an existing QApplication")

        helper_globals["locate_qt_platform_plugin"] = unexpected_search
        with mock.patch.dict(
            sys.modules,
            {"PySide6": pyside, "PySide6.QtWidgets": qtwidgets},
        ), mock.patch.dict(
            os.environ,
            {"QT_QPA_PLATFORM_PLUGIN_PATH": "existing-path"},
            clear=False,
        ):
            runtime = helper_globals["create_or_reuse_qapplication"]()
            self.assertEqual(
                os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"),
                "existing-path",
            )

        self.assertIs(runtime.application, existing_application)
        self.assertFalse(runtime.application_was_created)
        self.assertIsNone(runtime.plugin_file)
        self.assertTrue(runtime.environment_was_restored)

    def test_ads_qt_runtime_helper_requires_linux_display_for_new_application(self) -> None:
        helper_globals = load_generated_ads_qt_runtime(
            "ads_qt_runtime_display_test"
        )

        class FakeQApplication:
            @classmethod
            def instance(cls):
                return None

            def __init__(self, _argv):
                raise AssertionError(
                    "QApplication must not start without a Linux display"
                )

        pyside = types.ModuleType("PySide6")
        pyside.__file__ = str(ROOT / "fake_ads" / "PySide6" / "__init__.py")
        qtwidgets = types.ModuleType("PySide6.QtWidgets")
        qtwidgets.QApplication = FakeQApplication
        helper_globals["locate_qt_platform_plugin"] = (
            lambda _pyside_file: Path("/fake/platforms/libqxcb.so")
        )
        helper_globals["validate_linux_plugin"] = lambda _plugin_file: None

        with mock.patch.dict(
            sys.modules,
            {"PySide6": pyside, "PySide6.QtWidgets": qtwidgets},
        ), mock.patch.object(sys, "platform", "linux"), mock.patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "No DISPLAY or WAYLAND_DISPLAY"):
                helper_globals["create_or_reuse_qapplication"]()


if __name__ == "__main__":
    unittest.main()
