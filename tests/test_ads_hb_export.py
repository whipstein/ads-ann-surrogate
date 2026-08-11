import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from surrogate_common import (
    build_training_export_commands,
    write_ads_hb_mlp_package,
    write_ads_hb_neurotf_package,
)
from dnn import command_export_ads_hb as command_export_dnn_ads_hb
from kbnn import command_export_ads_hb as command_export_kbnn_ads_hb


LABELS = ["S11", "S12", "S21", "S22"]


class AdsHbExportTests(unittest.TestCase):
    def _assert_rf_only_s_wave(self, netlist: str, module_name: str) -> None:
        self.assertIn(f"SDD:{module_name}_core", netlist)
        z0_text = "5.00000000000000000e+01"
        self.assertIn(f"F[1,0]=_v1-({z0_text})*_i1", netlist)
        self.assertIn(f"F[1,2]=-(_v1+({z0_text})*_i1)", netlist)
        self.assertIn(f"F[1,3]=-(_v2+({z0_text})*_i2)", netlist)
        self.assertNotIn("I[", netlist)
        self.assertNotIn("_stoy_", netlist)
        self.assertNotIn("_core_dc", netlist)
        self.assertNotIn("conductance", netlist.lower())
        self.assertIn(
            "if (freq equals 0) then complex(1.00000000000000000e+00,0.0)",
            netlist,
        )
        self.assertIn("if (freq equals 0) then complex(0.0,0.0)", netlist)

    def _mlp_arguments(self) -> dict[str, object]:
        return {
            "parameter_names": ["W"],
            "sparam_labels": LABELS,
            "freq_transform": "log",
            "activation": "tanh",
            "layer_sizes": [2, 8],
            "weights": [np.zeros((2, 8))],
            "biases": [np.zeros(8)],
            "x_mean": np.array([1.0, 9.0]),
            "x_std": np.ones(2),
            "y_mean": np.zeros(8),
            "y_std": np.ones(8),
            "z0": 50.0,
        }

    def test_s_domain_mlp_export_is_direct_rf_only_s_wave(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_s"
            manifest = write_ads_hb_mlp_package(
                out_dir=out_dir,
                model_kind="DNN",
                module_name=module_name,
                output_domain="s",
                **self._mlp_arguments(),
            )
            netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_rf_only_s_wave(netlist, module_name)
            self.assertEqual(manifest["analysis_scope"], "linear_rf_only")
            self.assertFalse(manifest["dc_model_included"])
            self.assertEqual(manifest["rf_source_conversion"], "none")
            self.assertTrue(manifest["implicit_port_equations"])
            self.assertNotIn("DC", manifest["supported_analyses"])

    def test_direct_y_mlp_export_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires a model trained"):
                write_ads_hb_mlp_package(
                    out_dir=Path(temp_dir),
                    model_kind="DNN",
                    module_name="test_y",
                    output_domain="y",
                    **self._mlp_arguments(),
                )

    def test_neurotf_export_uses_direct_rf_only_s_wave(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_neurotf"
            manifest = write_ads_hb_neurotf_package(
                out_dir=out_dir,
                module_name=module_name,
                parameter_names=["W"],
                sparam_labels=LABELS,
                activation="tanh",
                layer_sizes=[1, 16],
                weights=[np.zeros((1, 16))],
                biases=[np.zeros(16)],
                x_mean=np.array([1.0]),
                x_std=np.ones(1),
                y_mean=np.zeros(16),
                y_std=np.ones(16),
                poles=np.array([-1.0 + 0.0j]),
                f_scale=1.0e9,
                z0=50.0,
            )
            netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_rf_only_s_wave(netlist, module_name)
            self.assertFalse(manifest["dc_model_included"])

    def test_self_contained_kbnn_uses_direct_rf_only_s_wave(self) -> None:
        embedded_coarse = {
            "parameter_names": ["W"],
            "sparam_labels": LABELS,
            "output_domain": "s",
            "freq_transform": "log",
            "activation": "tanh",
            "layer_sizes": [2, 8],
            "weights": [np.zeros((2, 8))],
            "biases": [np.zeros(8)],
            "x_mean": np.array([1.0, 9.0]),
            "x_std": np.ones(2),
            "y_mean": np.zeros(8),
            "y_std": np.ones(8),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_kbnn"
            manifest = write_ads_hb_mlp_package(
                out_dir=out_dir,
                model_kind="KBNN",
                module_name=module_name,
                output_domain="s",
                adds_coarse_to_output=True,
                embedded_coarse_model=embedded_coarse,
                **self._mlp_arguments(),
            )
            netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_rf_only_s_wave(netlist, module_name)
            self.assertIn("_coarse_", netlist)
            self.assertIn("_fine_", netlist)
            self.assertTrue(manifest["embedded_coarse_model"])

    def test_wave_relation_reproduces_supplied_s_matrix(self) -> None:
        rng = np.random.default_rng(42)
        z0 = 50.0
        for nports in (1, 2, 4):
            for _ in range(20):
                s_matrix = 0.2 * (
                    rng.normal(size=(nports, nports))
                    + 1j * rng.normal(size=(nports, nports))
                ) / np.sqrt(nports)
                voltage = rng.normal(size=nports) + 1j * rng.normal(size=nports)
                current = np.linalg.solve(
                    np.eye(nports) + s_matrix,
                    (np.eye(nports) - s_matrix) @ voltage,
                ) / z0
                incident = voltage + z0 * current
                reflected = voltage - z0 * current
                np.testing.assert_allclose(
                    reflected,
                    s_matrix @ incident,
                    rtol=1.0e-12,
                    atol=1.0e-12,
                )

    def test_identity_s_matrix_is_open_at_zero_frequency(self) -> None:
        rng = np.random.default_rng(7)
        for nports in (1, 2, 4):
            voltage = rng.normal(size=nports)
            s_matrix = np.eye(nports)
            current = np.linalg.solve(
                np.eye(nports) + s_matrix,
                (np.eye(nports) - s_matrix) @ voltage,
            ) / 50.0
            np.testing.assert_allclose(current, 0.0, atol=0.0)

    def test_fitting_report_hb_command_is_rf_only_and_s_domain_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()
            metadata_path = model_dir / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "output_domain": "s",
                        "dc_port_path_spec": "1-2",
                    }
                )
            )
            commands = build_training_export_commands(
                Path("dnn.py"),
                model_dir,
                template_mdif=Path(temp_dir) / "with_dc.mdif",
                include_veriloga=True,
            )
            hb_commands = [
                command for label, command in commands if "ADS HB" in label
            ]
            self.assertEqual(len(hb_commands), 1)
            self.assertIn("export-ads-hb", hb_commands[0])
            self.assertIn("--module-name", hb_commands[0])
            self.assertIn("--parameter-input-scales 1.0", hb_commands[0])
            self.assertNotIn("--dc-", hb_commands[0])

            metadata_path.write_text(json.dumps({"output_domain": "y"}))
            commands = build_training_export_commands(
                Path("dnn.py"),
                model_dir,
                include_veriloga=True,
            )
            self.assertFalse(any("ADS HB" in label for label, _ in commands))

    def test_dnn_command_rejects_direct_y_before_writing(self) -> None:
        args = SimpleNamespace(model_dir="unused", out_dir="unused")
        with patch("dnn.DNN.load", return_value=SimpleNamespace(output_domain="y")):
            with self.assertRaisesRegex(ValueError, "only S-domain RF models"):
                command_export_dnn_ads_hb(args)

    def test_plain_kbnn_command_needs_no_dc_arguments(self) -> None:
        model = SimpleNamespace(
            parameter_names=["W"],
            sparam_labels=LABELS,
            freq_transform="log",
            mlp=SimpleNamespace(
                activation="tanh",
                layer_sizes=[2, 8],
                weights=[np.zeros((2, 8))],
                biases=[np.zeros(8)],
            ),
            x_scaler=SimpleNamespace(mean=np.array([1.0, 9.0]), std=np.ones(2)),
            y_scaler=SimpleNamespace(mean=np.zeros(8), std=np.ones(8)),
            mode="plain",
            include_coarse_input=False,
        )
        args = SimpleNamespace(
            model_dir="unused",
            out_dir="unused",
            module_name="plain_kbnn",
            parameter_input_scales=None,
            z0=50.0,
            coarse_model_dir=None,
        )
        manifest = {
            "netlist_file": "plain_kbnn.net",
            "module_name": "plain_kbnn",
            "embedded_coarse_model": False,
            "coarse_model_match_verified": False,
            "linear": True,
            "power_dependent": False,
            "dc_model_included": False,
            "supported_analyses": ["AC", "SP", "HB"],
        }
        with patch("kbnn.KBNN.load", return_value=model):
            with patch("kbnn.write_ads_hb_mlp_package", return_value=manifest):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(command_export_kbnn_ads_hb(args), 0)


if __name__ == "__main__":
    unittest.main()
