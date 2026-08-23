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
    ADS_ANN_NETLIST_PLACEHOLDER,
    ADS_EXPORT_TEMPLATE_FILENAME,
    MDIFBlock,
    ads_ann_netlist_template,
    ads_qt_runtime_helper_script,
    ads_ann_parameter_count,
    build_ads_export_blocks,
    build_training_export_commands,
    read_mdif,
    resolve_ads_ann_layout,
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
    def test_ads_ann_layout_uses_small_independent_defaults(self) -> None:
        self.assertEqual(resolve_ads_ann_layout(None, None), (2, 20))
        self.assertEqual(resolve_ads_ann_layout(3, 48), (3, 48))
        with self.assertRaisesRegex(ValueError, "ads-hidden-layers"):
            resolve_ads_ann_layout(0, 20)
        with self.assertRaisesRegex(ValueError, "ads-neurons-per-layer"):
            resolve_ads_ann_layout(2, 0)

    def test_ads_ann_parameter_count_includes_weights_and_biases(self) -> None:
        # (2 inputs + bias) * 20, one 20-to-20 hidden transition,
        # and (20 hidden + bias) * 2 outputs.
        self.assertEqual(ads_ann_parameter_count(2, 2, 2, 20), 522)

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

    def test_debug_model_reads_required_run_dir_from_options_json_through_dispatcher(self) -> None:
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
            options_path = root / "options.json"
            options_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workflows": {
                            "debug-model": {
                                "commands": {
                                    "debug-model": {
                                        "run_dir": str(run_dir),
                                        "out_dir": str(root / "debug"),
                                    }
                                }
                            }
                        },
                    }
                )
            )

            status = surrogate.main(
                ["--options-json", str(options_path), "debug-model"]
            )

            self.assertEqual(status, 0)
            self.assertTrue((root / "debug" / "model_debug.md").is_file())

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
            self.assertEqual(
                manifest["ads_ann"]["estimated_trainable_parameters"],
                ads_ann_parameter_count(2, 2, 2, 32),
            )
            self.assertIn("--preflight-only", training_source)
            self.assertIn("ann.reset()", training_source)
            self.assertIn("except MemoryError as error", training_source)
            self.assertIn("dense_quasi_newton_state_estimate_gib", training_source)
            self.assertIn("write_sdd_equation_copy", training_source)
            self.assertIn("write_sdd_netlist", training_source)
            self.assertIn("ann_in_", training_source)
            self.assertIn("ann_out_", training_source)
            self.assertIn("ads_qt_runtime.py", (out_dir / "ADS_ANN_README.md").read_text())
            setup_source = (out_dir / "ADS_SDD_SETUP.md").read_text()
            self.assertIn("Preferred: Automatic NetlistInclude/SDD Flow", setup_source)
            self.assertIn("## Manual Schematic Fallback", setup_source)
            for step in range(1, 6):
                self.assertIn(f"### {step}.", setup_source)
                self.assertNotIn(f"\n## {step}.", setup_source)
            self.assertIn("Parameter Entry Mode", setup_source)
            self.assertIn("if (freq equals 0) then 0.0 else Y11 endif", setup_source)
            self.assertIn("`ann_in_1`", setup_source)
            self.assertIn("ann_in_1=W", setup_source)
            self.assertIn("S11=complex(ann_out_1,ann_out_2)", setup_source)
            self.assertEqual(manifest["sdd_setup_guide"], "ADS_SDD_SETUP.md")
            self.assertTrue(manifest["ads_netlist"]["enabled"])
            self.assertEqual(manifest["ads_netlist"]["nports"], 1)
            self.assertEqual(
                manifest["ads_netlist"]["parameter_ranges"]["W"],
                {
                    "model_min": 0.5e-3,
                    "model_max": 0.5e-3,
                    "instance_min": 0.5e-3,
                    "instance_max": 0.5e-3,
                },
            )
            template_path = out_dir / manifest["ads_netlist"]["netlist_template_file"]
            self.assertTrue(template_path.is_file())
            instance_template_path = out_dir / "ADS_ANN_INSTANCE_TEMPLATE.txt"
            self.assertTrue(instance_template_path.is_file())
            instance_template = instance_template_path.read_text()
            self.assertIn("W=ann_sweep_W", instance_template)
            self.assertIn("sweep the exact", instance_template)
            self.assertIn("W=W_A", instance_template)
            netlist_template = template_path.read_text()
            self.assertEqual(netlist_template.count(ADS_ANN_NETLIST_PLACEHOLDER), 1)
            self.assertIn("SDD:test_ann_sdd_core_rf", netlist_template)

            # Exercise the generated post-training finalizer without importing
            # Qt or the licensed ADS ANN module.
            (out_dir / "test_ann.equation").write_text(
                "hidden_1=tanh(_v1)\ny1=hidden_1\ny2=0.0\n"
            )
            fake_pandas = types.ModuleType("pandas")
            fake_runtime = types.ModuleType("ads_qt_runtime")
            fake_runtime.create_or_reuse_qapplication = lambda: None
            fake_runtime.qt_runtime_diagnostics = lambda _runtime: {}
            generated_namespace = {
                "__file__": str(out_dir / "train_ads_ann.py"),
                "__name__": "generated_ads_ann_training",
            }
            with mock.patch.dict(
                sys.modules,
                {"pandas": fake_pandas, "ads_qt_runtime": fake_runtime},
            ):
                exec(training_source, generated_namespace)
            mapped_equation = generated_namespace["write_sdd_equation_copy"](
                manifest
            )
            completed_netlist = generated_namespace["write_sdd_netlist"](
                manifest,
                mapped_equation,
            )
            self.assertEqual(completed_netlist, "test_ann_sdd.net")
            completed_source = (out_dir / completed_netlist).read_text()
            self.assertNotIn(ADS_ANN_NETLIST_PLACEHOLDER, completed_source)
            self.assertIn("hidden_1=tanh(ann_in_1)", completed_source)
            self.assertIn("ann_out_1=hidden_1", completed_source)

    def test_ads_ann_netlist_template_generates_complete_four_port_sdd(self) -> None:
        labels = [f"S{row}{column}" for row in range(1, 5) for column in range(1, 5)]
        outputs = [
            *[f"fine_{label}_real" for label in labels],
            *[f"fine_{label}_imag" for label in labels],
        ]
        template, contract = ads_ann_netlist_template(
            input_columns=["W", "freq_log10_hz"],
            output_columns=outputs,
            parameter_names=["W"],
            sparam_labels=labels,
            module_name="native_ann_4port",
            z0=50.0,
            parameter_input_scales={"W": 1.0e-6},
            parameter_model_defaults=[10.0],
        )

        self.assertEqual(contract["nports"], 4)
        self.assertEqual(contract["module_name"], "native_ann_4port")
        self.assertEqual(template.count(ADS_ANN_NETLIST_PLACEHOLDER), 1)
        self.assertIn("define native_ann_4port (p1 p2 p3 p4)", template)
        self.assertIn("ann_in_1=(W)/(W_input_scale)", template)
        self.assertIn("ann_in_2=log10(max(abs(freq),1.0))", template)
        self.assertIn("native_ann_4port_rf_s44=complex(ann_out_16,ann_out_32)", template)
        self.assertIn("SDD:native_ann_4port_core_rf p1 0 p2 0 p3 0 p4 0", template)
        self.assertIn("I[4,17]=_v4", template)
        self.assertIn("if (freq equals 0) then 0.0 else", template)
        self.assertIn("native_ann_4port_stoy_y_3_3", template)

    def test_ads_ann_netlist_uses_separate_geometry_dependent_exact_dc(self) -> None:
        dc_model = {
            "kind": "geometry_dependent_exact_dc_full_y_mlp",
            "representation": "full_y_matrix",
            "parameter_names": ["W"],
            "sparam_labels": ["S11"],
            "activation": "tanh",
            "layer_sizes": [1, 1],
            "weights": [np.asarray([[0.25]])],
            "biases": [np.asarray([0.0])],
            "x_mean": np.asarray([10.0]),
            "x_std": np.asarray([2.0]),
            "y_mean": np.asarray([0.02]),
            "y_std": np.asarray([0.005]),
        }
        template, contract = ads_ann_netlist_template(
            input_columns=["W", "freq_hz"],
            output_columns=["fine_S11_real", "fine_S11_imag"],
            parameter_names=["W"],
            sparam_labels=["S11"],
            module_name="native_ann_dc",
            z0=50.0,
            parameter_input_scales={"W": 1.0e-6},
            parameter_model_defaults=[10.0],
            dc_model=dc_model,
        )

        self.assertFalse(contract["rf_only"])
        self.assertEqual(
            contract["dc_behavior"],
            "separate_geometry_dependent_exact_dc_model",
        )
        self.assertIn("SDD:native_ann_dc_core_rf", template)
        self.assertIn("SDD:native_ann_dc_core_dc", template)
        self.assertIn("if (freq equals 0) then native_ann_dc_dc_net_out0", template)
        self.assertIn("else 0.0 endif", template)

    def test_ads_ann_netlist_rejects_residual_or_coarse_interface(self) -> None:
        with self.assertRaisesRegex(ValueError, "final fine"):
            ads_ann_netlist_template(
                input_columns=["W", "freq_hz"],
                output_columns=["delta_S11_real", "delta_S11_imag"],
                parameter_names=["W"],
                sparam_labels=["S11"],
                module_name="residual_ann",
                z0=50.0,
                parameter_input_scales={"W": 1.0},
                parameter_model_defaults=[1.0],
            )
        with self.assertRaisesRegex(ValueError, "external coarse"):
            ads_ann_netlist_template(
                input_columns=["W", "freq_hz", "coarse_S11_real"],
                output_columns=["fine_S11_real", "fine_S11_imag"],
                parameter_names=["W"],
                sparam_labels=["S11"],
                module_name="coarse_input_ann",
                z0=50.0,
                parameter_input_scales={"W": 1.0},
                parameter_model_defaults=[1.0],
            )

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
