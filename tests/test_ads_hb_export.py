import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from surrogate_common import (
    _fold_mlp_scalers_into_layers,
    write_ads_hb_mlp_package,
    write_ads_hb_neurotf_package,
)


LABELS = ["S11", "S12", "S21", "S22"]


class AdsHbExportTests(unittest.TestCase):
    @staticmethod
    def _evaluate_mlp(
        values: np.ndarray,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
    ) -> np.ndarray:
        result = np.asarray(values, dtype=float)
        for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
            result = result @ weight + bias
            if layer_idx < len(weights) - 1:
                result = np.tanh(result)
        return result

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

    def test_mlp_scaler_folding_is_algebraically_exact(self) -> None:
        rng = np.random.default_rng(31)
        x_mean = np.array([2.5, -0.75, 10.0])
        x_std = np.array([0.5, 4.0, 2.0])
        y_mean = np.array([-1.25, 3.0])
        y_std = np.array([2.5, 0.125])
        samples = rng.normal(size=(20, 3))

        for layer_sizes in ([3, 2], [3, 5, 4, 2]):
            weights = [
                rng.normal(size=(input_size, output_size))
                for input_size, output_size in zip(
                    layer_sizes[:-1], layer_sizes[1:]
                )
            ]
            biases = [
                rng.normal(size=output_size) for output_size in layer_sizes[1:]
            ]
            expected = self._evaluate_mlp(
                (samples - x_mean) / x_std,
                weights,
                biases,
            )
            expected = expected * y_std + y_mean

            folded_weights, folded_biases = _fold_mlp_scalers_into_layers(
                weights,
                biases,
                x_mean,
                x_std,
                y_mean,
                y_std,
            )
            actual = self._evaluate_mlp(
                samples,
                folded_weights,
                folded_biases,
            )
            np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)

    def test_mlp_export_can_emit_folded_scalers_as_separate_trial(self) -> None:
        rng = np.random.default_rng(19)
        dc_model = {
            "parameter_names": ["W"],
            "sparam_labels": LABELS,
            "representation": "path_conductance",
            "port_paths": ["p1-p2"],
            "activation": "tanh",
            "layer_sizes": [1, 3, 1],
            "weights": [rng.normal(size=(1, 3)), rng.normal(size=(3, 1))],
            "biases": [rng.normal(size=3), rng.normal(size=1)],
            "x_mean": np.array([1.25]),
            "x_std": np.array([0.4]),
            "y_mean": np.array([-2.0]),
            "y_std": np.array([0.75]),
            "log_conductance_min": -40.0,
            "log_conductance_max": 10.0,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_dnn"
            manifest = write_ads_hb_mlp_package(
                out_dir=out_dir,
                model_kind="DNN",
                module_name=module_name,
                parameter_names=["W"],
                sparam_labels=LABELS,
                freq_transform="log",
                activation="tanh",
                layer_sizes=[2, 4, 8],
                weights=[rng.normal(size=(2, 4)), rng.normal(size=(4, 8))],
                biases=[rng.normal(size=4), rng.normal(size=8)],
                x_mean=np.array([1.25, 9.0]),
                x_std=np.array([0.4, 1.5]),
                y_mean=rng.normal(size=8),
                y_std=np.linspace(0.25, 2.0, 8),
                z0=50.0,
                output_domain="s",
                dc_equivalent_resistance_ohm=100.0,
                dc_resistance_source_kind="exact_zero_frequency",
                dc_model=dc_model,
                emit_folded_scalers_trial=True,
            )

            default_netlist = (out_dir / f"{module_name}.net").read_text()
            self.assertIn(f"SDD:{module_name}_core_rf", default_netlist)
            self.assertIn(f"SDD:{module_name}_core_dc", default_netlist)
            self.assertIn("if (freq equals 0) then 0.0 else", default_netlist)
            self.assertIn(
                "if (freq equals 0) then test_dnn_m_dc_y0 else",
                default_netlist,
            )
            self.assertIn(f"{module_name}_m_fine_x0=", default_netlist)
            self.assertIn(f"{module_name}_m_fine_out0=", default_netlist)
            self.assertIn(f"{module_name}_m_dc_net_x0=", default_netlist)
            self.assertIn(f"{module_name}_m_dc_net_out0=", default_netlist)

            trial_module = f"{module_name}_folded_scalers_trial"
            trial_netlist = (out_dir / f"{trial_module}.net").read_text()
            self.assertIn(f"SDD:{trial_module}_core_rf", trial_netlist)
            self.assertIn(f"SDD:{trial_module}_core_dc", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_fine_x0=", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_fine_out0=", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_dc_net_x0=", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_dc_net_out0=", trial_netlist)
            self.assertEqual(default_netlist.count("H["), trial_netlist.count("H["))

            trial_exports = manifest["trial_exports"]
            self.assertEqual(len(trial_exports), 1)
            self.assertEqual(trial_exports[0]["kind"], "folded_mlp_scalers")
            self.assertEqual(
                trial_exports[0]["sdd_dc_rf_topology"],
                "separate_parallel_sdds",
            )
            self.assertEqual(
                trial_exports[0]["mlp_scaler_implementation"],
                "folded_into_first_and_final_layers",
            )

            trial_manifest = json.loads(
                (out_dir / "ads_hb_folded_scalers_trial_manifest.json").read_text()
            )
            self.assertTrue(trial_manifest["folded_input_scaler"])
            self.assertTrue(trial_manifest["folded_output_scaler"])
            self.assertEqual(
                trial_manifest["sdd_dc_rf_topology"],
                "separate_parallel_sdds",
            )
            self.assertTrue(
                (out_dir / "ADS_HB_FOLDED_SCALERS_TRIAL_INSTANCE_TEMPLATE.txt").is_file()
            )
            self.assertTrue(
                (out_dir / "ADS_HB_FOLDED_SCALERS_TRIAL_README.md").is_file()
            )

    def test_mlp_export_can_eliminate_constant_outputs_as_separate_trial(self) -> None:
        rng = np.random.default_rng(23)
        dc_model = {
            "parameter_names": ["W"],
            "sparam_labels": LABELS,
            "representation": "full_s_matrix",
            "port_paths": [],
            "z0": 50.0,
            "activation": "tanh",
            "layer_sizes": [1, 3, 8],
            "weights": [rng.normal(size=(1, 3)), rng.normal(size=(3, 8))],
            "biases": [rng.normal(size=3), rng.normal(size=8)],
            "x_mean": np.array([1.25]),
            "x_std": np.array([0.4]),
            "y_mean": np.array([0.2, 0.8, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]),
            "y_std": np.zeros(8),
        }
        rf_y_std = np.ones(8)
        rf_y_std[0] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            module_name = "test_dnn"
            manifest = write_ads_hb_mlp_package(
                out_dir=out_dir,
                model_kind="DNN",
                module_name=module_name,
                parameter_names=["W"],
                sparam_labels=LABELS,
                freq_transform="log",
                activation="tanh",
                layer_sizes=[2, 4, 8],
                weights=[rng.normal(size=(2, 4)), rng.normal(size=(4, 8))],
                biases=[rng.normal(size=4), rng.normal(size=8)],
                x_mean=np.array([1.25, 9.0]),
                x_std=np.array([0.4, 1.5]),
                y_mean=np.linspace(-0.4, 0.3, 8),
                y_std=rf_y_std,
                z0=50.0,
                output_domain="s",
                dc_equivalent_resistance_ohm=100.0,
                dc_resistance_source_kind="exact_zero_frequency",
                dc_model=dc_model,
                emit_constant_outputs_trial=True,
            )

            default_netlist = (out_dir / f"{module_name}.net").read_text()
            self.assertIn(f"{module_name}_m_fine_l2_0=", default_netlist)
            self.assertIn(f"{module_name}_m_fine_out0=", default_netlist)
            self.assertIn(f"{module_name}_m_dc_net_x0=", default_netlist)
            self.assertIn(f"{module_name}_m_dc_stoy_a0_0_0=", default_netlist)

            trial_module = f"{module_name}_constant_outputs_trial"
            trial_netlist = (out_dir / f"{trial_module}.net").read_text()
            self.assertNotIn(f"{trial_module}_m_fine_l2_0=", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_fine_out0=", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_dc_net_", trial_netlist)
            self.assertNotIn(f"{trial_module}_m_dc_stoy_", trial_netlist)
            self.assertIn("Entire exact-DC MLP is constant", trial_netlist)
            self.assertIn(f"SDD:{trial_module}_core_rf", trial_netlist)
            self.assertIn(f"SDD:{trial_module}_core_dc", trial_netlist)
            self.assertEqual(default_netlist.count("H["), trial_netlist.count("H["))

            trial_exports = manifest["trial_exports"]
            self.assertEqual(len(trial_exports), 1)
            self.assertEqual(
                trial_exports[0]["kind"],
                "eliminated_constant_outputs",
            )
            self.assertTrue(trial_exports[0]["dc_constant_matrix_precomputed"])

            trial_manifest = json.loads(
                (out_dir / "ads_hb_constant_outputs_trial_manifest.json").read_text()
            )
            self.assertTrue(trial_manifest["constant_output_elimination"])
            self.assertTrue(trial_manifest["dc_constant_matrix_precomputed"])
            self.assertEqual(
                trial_manifest["dc_stamping_representation"],
                "separate_precomputed_constant_y_sdd",
            )
            self.assertEqual(
                trial_manifest["constant_output_summary"]["rf"][
                    "constant_output_count"
                ],
                1,
            )
            self.assertEqual(
                trial_manifest["constant_output_summary"]["dc"][
                    "constant_output_count"
                ],
                8,
            )
            self.assertTrue(
                trial_manifest["constant_output_summary"]["dc"][
                    "entire_mlp_constant"
                ]
            )
            self.assertTrue(
                (out_dir / "ADS_HB_CONSTANT_OUTPUTS_TRIAL_INSTANCE_TEMPLATE.txt").is_file()
            )
            self.assertTrue(
                (out_dir / "ADS_HB_CONSTANT_OUTPUTS_TRIAL_README.md").is_file()
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
