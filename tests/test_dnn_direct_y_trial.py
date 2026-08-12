import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dnn import DNN, command_export_ads_hb
from surrogate_common import DCConductanceModel, MLP, Standardizer


LABELS = ["S11", "S12", "S21", "S22"]


class DnnDirectYTrialTests(unittest.TestCase):
    @staticmethod
    def _scaler(mean: np.ndarray, std: np.ndarray) -> Standardizer:
        scaler = Standardizer()
        scaler.mean = np.asarray(mean, dtype=float)
        scaler.std = np.asarray(std, dtype=float)
        return scaler

    def _model(self, output_domain: str, seed: int) -> DNN:
        rf_mlp = MLP([2, 3, 8], activation="tanh", seed=seed)
        dc_mlp = MLP([1, 3, 8], activation="tanh", seed=seed + 10)
        dc_model = DCConductanceModel(
            mlp=dc_mlp,
            x_scaler=self._scaler(np.array([1.0]), np.array([0.5])),
            y_scaler=self._scaler(
                np.array([0.2, 0.8, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]),
                np.zeros(8),
            ),
            parameter_names=["W"],
            sparam_labels=LABELS,
            port_paths=[],
            z0=50.0,
            log_conductance_min=-50.0,
            log_conductance_max=50.0,
            representation="full_s_matrix",
            metadata={
                "dc_model_kind": "geometry_dependent_exact_dc_full_s_mlp",
                "dc_model_representation": "full_s_matrix",
                "dc_resistance_source_kind": "exact_zero_frequency",
                "dc_port_paths": [],
                "dc_port_paths_explicit": False,
                "dc_equivalent_resistance_ohm": 100.0,
                "dc_port_resistances_ohm": {},
            },
        )
        return DNN(
            mlp=rf_mlp,
            x_scaler=self._scaler(np.array([1.0, 9.0]), np.ones(2)),
            y_scaler=self._scaler(np.zeros(8), np.ones(8)),
            parameter_names=["W"],
            sparam_labels=LABELS,
            freq_transform="log",
            output_domain=output_domain,
            target_z0=50.0,
            dc_equivalent_resistance_ohm=100.0,
            dc_resistance_source_kind="exact_zero_frequency",
            dc_port_resistances_ohm={},
            dc_model=dc_model,
        )

    @staticmethod
    def _metadata(output_domain: str) -> dict[str, object]:
        return {
            "output_domain": output_domain,
            "training_blocks": 4,
            "verification_blocks": 1,
            "split_var": "dataset",
            "train_values": ["train", "training"],
            "verify_values": ["test", "validation", "verification", "verify"],
            "normalized_sparam_weights": {label: 1.0 for label in LABELS},
            "frequency_weights": None,
            "dc_model_kind": "geometry_dependent_exact_dc_full_s_mlp",
            "dc_model_representation": "full_s_matrix",
            "dc_resistance_source_kind": "exact_zero_frequency",
            "dc_port_paths": [],
            "dc_port_paths_explicit": False,
            "dc_equivalent_resistance_ohm": 100.0,
            "dc_port_resistances_ohm": {},
        }

    def test_export_pairs_s_baseline_with_validated_direct_y_trial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "baseline"
            trial_dir = root / "direct_y"
            out_dir = root / "export"
            self._model("s", seed=7).save(
                baseline_dir,
                metadata=self._metadata("s"),
            )
            self._model("y", seed=11).save(
                trial_dir,
                metadata=self._metadata("y"),
            )
            args = argparse.Namespace(
                model_dir=str(baseline_dir),
                direct_y_trial_model_dir=str(trial_dir),
                out_dir=str(out_dir),
                module_name="test hb",
                z0=50.0,
                parameter_input_scales="1.0",
                dc_mdif=None,
                dc_port_paths=None,
                dc_open_threshold=1.0e12,
                dc_open_resistance=1.0e19,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(command_export_ads_hb(args), 0)

            manifest = json.loads((out_dir / "ads_hb_manifest.json").read_text())
            self.assertEqual(manifest["response_domain"], "s")
            self.assertEqual(len(manifest["trial_exports"]), 1)
            trial_export = manifest["trial_exports"][0]
            self.assertEqual(trial_export["kind"], "direct_y_rf_dnn")
            self.assertEqual(trial_export["response_domain"], "y")
            self.assertEqual(trial_export["rf_source_conversion"], "none")
            self.assertEqual(
                trial_export["direct_y_comparison"]["training_metadata_differences"],
                {},
            )

            baseline_netlist = (out_dir / "test_hb.net").read_text()
            trial_netlist = (out_dir / "test_hb_direct_y_trial.net").read_text()
            self.assertIn("test_hb_m_stoy_", baseline_netlist)
            self.assertNotIn("test_hb_direct_y_trial_m_stoy_", trial_netlist)
            self.assertIn("test_hb_m_dc_stoy_", baseline_netlist)
            self.assertIn("test_hb_direct_y_trial_m_dc_stoy_", trial_netlist)
            self.assertIn("SDD:test_hb_core_rf", baseline_netlist)
            self.assertIn("SDD:test_hb_direct_y_trial_core_rf", trial_netlist)
            self.assertTrue(
                (out_dir / "ads_hb_direct_y_trial_manifest.json").is_file()
            )
            self.assertTrue(
                (out_dir / "ADS_HB_DIRECT_Y_TRIAL_README.md").is_file()
            )

    def test_direct_y_trial_requires_matching_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_dir = root / "baseline"
            trial_dir = root / "direct_y"
            baseline = self._model("s", seed=7)
            trial = self._model("y", seed=11)
            trial.mlp = MLP([2, 4, 8], activation="tanh", seed=13)
            baseline.save(baseline_dir, metadata=self._metadata("s"))
            trial.save(trial_dir, metadata=self._metadata("y"))
            args = argparse.Namespace(
                model_dir=str(baseline_dir),
                direct_y_trial_model_dir=str(trial_dir),
                out_dir=str(root / "export"),
                module_name="test_hb",
                z0=50.0,
                parameter_input_scales="1.0",
                dc_mdif=None,
                dc_port_paths=None,
                dc_open_threshold=1.0e12,
                dc_open_resistance=1.0e19,
            )
            with self.assertRaisesRegex(ValueError, "layer sizes do not match"):
                command_export_ads_hb(args)


if __name__ == "__main__":
    unittest.main()
