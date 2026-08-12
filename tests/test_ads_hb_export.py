import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from surrogate_common import write_ads_hb_mlp_package, write_ads_hb_neurotf_package


LABELS = ["S11", "S12", "S21", "S22"]


class AdsHbExportTests(unittest.TestCase):
    def _assert_explicit_separate_stamps(self, netlist: str, module_name: str) -> None:
        self.assertNotIn("F[", netlist)
        self.assertIn("I[1,0]=0.0", netlist)
        self.assertIn(f"SDD:{module_name}_core_rf", netlist)
        self.assertIn(f"SDD:{module_name}_core_dc", netlist)
        self.assertIn("if (freq equals 0) then 0.0 else", netlist)
        self.assertIn("if (freq equals 0) then 1.00000000000000002e-02", netlist)
        self.assertIn("if (freq equals 0) then -1.00000000000000002e-02", netlist)

    def _assert_combined_stamp(self, netlist: str, module_name: str) -> None:
        self.assertNotIn("F[", netlist)
        self.assertIn("I[1,0]=0.0", netlist)
        self.assertIn(f"SDD:{module_name}_core_combined", netlist)
        self.assertNotIn(f"SDD:{module_name}_core_rf", netlist)
        self.assertNotIn(f"SDD:{module_name}_core_dc", netlist)
        self.assertIn("if (freq equals 0) then 1.00000000000000002e-02 else", netlist)
        self.assertIn("if (freq equals 0) then -1.00000000000000002e-02 else", netlist)

    def test_mlp_s_and_y_exports_use_explicit_y_stamps(self) -> None:
        common = dict(
            parameter_names=["W"],
            sparam_labels=LABELS,
            freq_transform="log",
            activation="tanh",
            layer_sizes=[2, 8],
            weights=[np.zeros((2, 8))],
            biases=[np.zeros(8)],
            x_mean=np.array([1.0, 9.0]),
            x_std=np.ones(2),
            y_mean=np.zeros(8),
            y_std=np.ones(8),
            z0=50.0,
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={"p1-p2": 100.0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for domain in ("s", "y"):
                module_name = f"test_{domain}"
                out_dir = root / domain
                manifest = write_ads_hb_mlp_package(
                    out_dir=out_dir,
                    model_kind="DNN",
                    module_name=module_name,
                    output_domain=domain,
                    **common,
                )
                netlist = (out_dir / f"{module_name}.net").read_text()
                self._assert_explicit_separate_stamps(netlist, module_name)
                self.assertEqual("_stoy_" in netlist, domain == "s")
                self.assertFalse(manifest["implicit_port_equations"])
                self.assertEqual(
                    manifest["dc_stamping_representation"],
                    "separate_explicit_conductance_sdd",
                )

    def test_mlp_export_can_emit_combined_sdd_as_separate_trial(self) -> None:
        common = dict(
            parameter_names=["W"],
            sparam_labels=LABELS,
            freq_transform="log",
            activation="tanh",
            layer_sizes=[2, 8],
            weights=[np.zeros((2, 8))],
            biases=[np.zeros(8)],
            x_mean=np.array([1.0, 9.0]),
            x_std=np.ones(2),
            y_mean=np.zeros(8),
            y_std=np.ones(8),
            z0=50.0,
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={"p1-p2": 100.0},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_dnn"
            manifest = write_ads_hb_mlp_package(
                out_dir=out_dir,
                model_kind="DNN",
                module_name=module_name,
                output_domain="s",
                emit_combined_sdd_trial=True,
                **common,
            )

            default_netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_explicit_separate_stamps(default_netlist, module_name)
            self.assertEqual(default_netlist.count("H["), 2 * len(LABELS))

            trial_module = f"{module_name}_combined_sdd_trial"
            trial_netlist_name = f"{trial_module}.net"
            trial_netlist = (out_dir / trial_netlist_name).read_text()
            self._assert_combined_stamp(trial_netlist, trial_module)
            self.assertEqual(trial_netlist.count(f"SDD:{trial_module}_core"), 1)
            self.assertEqual(trial_netlist.count("H["), len(LABELS))

            trial_exports = manifest["trial_exports"]
            self.assertEqual(len(trial_exports), 1)
            self.assertEqual(trial_exports[0]["netlist_file"], trial_netlist_name)
            self.assertEqual(trial_exports[0]["status"], "trial")
            self.assertEqual(
                trial_exports[0]["sdd_dc_rf_topology"],
                "single_frequency_selected_sdd",
            )

            trial_manifest_path = out_dir / "ads_hb_combined_sdd_trial_manifest.json"
            trial_manifest = json.loads(trial_manifest_path.read_text())
            self.assertEqual(trial_manifest["implementation_status"], "trial")
            self.assertEqual(
                trial_manifest["sdd_dc_rf_topology"],
                "single_frequency_selected_sdd",
            )
            self.assertEqual(
                trial_manifest["dc_stamping_representation"],
                "combined_explicit_conductance_sdd",
            )
            self.assertTrue(
                (out_dir / "ADS_HB_COMBINED_SDD_TRIAL_INSTANCE_TEMPLATE.txt").is_file()
            )
            self.assertTrue(
                (out_dir / "ADS_HB_COMBINED_SDD_TRIAL_README.md").is_file()
            )

    def test_neurotf_export_uses_the_same_explicit_stamps(self) -> None:
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
                dc_equivalent_resistance_ohm=100.0,
                dc_resistance_source_kind="exact_zero_frequency",
                dc_port_resistances_ohm={"p1-p2": 100.0},
            )
            netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_explicit_separate_stamps(netlist, module_name)
            self.assertIn("_stoy_", netlist)
            self.assertFalse(manifest["implicit_port_equations"])

    def test_self_contained_kbnn_uses_explicit_y_stamps(self) -> None:
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
                parameter_names=["W"],
                sparam_labels=LABELS,
                freq_transform="log",
                activation="tanh",
                layer_sizes=[2, 8],
                weights=[np.zeros((2, 8))],
                biases=[np.zeros(8)],
                x_mean=np.array([1.0, 9.0]),
                x_std=np.ones(2),
                y_mean=np.zeros(8),
                y_std=np.ones(8),
                z0=50.0,
                output_domain="s",
                adds_coarse_to_output=True,
                embedded_coarse_model=embedded_coarse,
                dc_equivalent_resistance_ohm=100.0,
                dc_resistance_source_kind="exact_zero_frequency",
                dc_port_resistances_ohm={"p1-p2": 100.0},
            )
            netlist = (out_dir / f"{module_name}.net").read_text()
            self._assert_explicit_separate_stamps(netlist, module_name)
            self.assertIn("_coarse_", netlist)
            self.assertIn("_fine_", netlist)
            self.assertFalse(manifest["implicit_port_equations"])
            self.assertTrue(manifest["embedded_coarse_model"])

    def test_generated_gauss_jordan_relation_matches_numpy(self) -> None:
        rng = np.random.default_rng(42)
        for nports in (1, 2, 4):
            for _ in range(20):
                s_matrix = 0.2 * (
                    rng.normal(size=(nports, nports))
                    + 1j * rng.normal(size=(nports, nports))
                ) / np.sqrt(nports)
                a_matrix = np.eye(nports, dtype=complex) + s_matrix
                b_matrix = np.eye(nports, dtype=complex) - s_matrix
                for pivot in range(nports):
                    pivot_value = a_matrix[pivot, pivot]
                    a_matrix[pivot] /= pivot_value
                    b_matrix[pivot] /= pivot_value
                    for row in range(nports):
                        if row == pivot:
                            continue
                        factor = a_matrix[row, pivot]
                        a_matrix[row] -= factor * a_matrix[pivot]
                        b_matrix[row] -= factor * b_matrix[pivot]
                expected = np.linalg.solve(
                    np.eye(nports) + s_matrix,
                    np.eye(nports) - s_matrix,
                )
                np.testing.assert_allclose(
                    b_matrix,
                    expected,
                    rtol=1.0e-11,
                    atol=1.0e-11,
                )


if __name__ == "__main__":
    unittest.main()
