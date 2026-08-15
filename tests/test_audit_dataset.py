import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import audit_dataset as AUDIT
from surrogate_common import MDIFBlock, write_mdif


LABELS = ["S11", "S12", "S21", "S22"]


def passive_block(
    geometry: float,
    role: str,
    *,
    frequencies: tuple[float, ...] = (0.0, 1.0e9, 2.0e9),
    transmission: float = 0.75,
) -> MDIFBlock:
    count = len(frequencies)
    return MDIFBlock(
        params={"dataset": role, "W": str(geometry)},
        freq_hz=np.asarray(frequencies, dtype=float),
        sparams={
            "S11": np.full(count, 0.1 + 0.0j),
            "S12": np.full(count, transmission + 0.0j),
            "S21": np.full(count, transmission + 0.0j),
            "S22": np.full(count, 0.1 + 0.0j),
        },
    )


class DatasetAuditTests(unittest.TestCase):
    def test_passive_consistent_dataset_passes_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "combined.mdif"
            out_dir = root / "audit"
            write_mdif(
                mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train", transmission=0.7),
                    passive_block(1.5, "verification", transmission=0.72),
                ],
                LABELS,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(mdif),
                        "--out-dir",
                        str(out_dir),
                        "--parameter-names",
                        "W",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["block_counts"]["fine_train"], 2)
            self.assertEqual(summary["block_counts"]["fine_verification"], 1)
            self.assertEqual(summary["passivity"]["violating_rows"], 0)
            self.assertTrue((out_dir / "dataset_audit.md").is_file())
            self.assertTrue((out_dir / "dataset_passivity.svg").is_file())

    def test_nonpassive_conflicting_train_verification_overlap_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "combined.mdif"
            out_dir = root / "audit"
            train = passive_block(1.0, "train", transmission=0.75)
            train.sparams["S11"][1] = 1.2 + 0.0j
            verify = passive_block(1.0, "verification", transmission=0.45)
            write_mdif(mdif, [train, verify], LABELS)
            with contextlib.redirect_stdout(io.StringIO()):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(mdif),
                        "--out-dir",
                        str(out_dir),
                        "--parameter-names",
                        "W",
                    ]
                )
            self.assertEqual(status, 1)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            codes = summary["issue_code_counts"]
            self.assertGreater(codes["RAW_NONPASSIVE_DATA"], 0)
            self.assertGreater(codes["TRAIN_VERIFICATION_OVERLAP"], 0)
            self.assertGreater(codes["DUPLICATE_GEOMETRY_RESPONSE_CONFLICT"], 0)

    def test_grid_and_coverage_mismatches_are_reported_as_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "combined.mdif"
            out_dir = root / "audit"
            write_mdif(
                mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train"),
                    passive_block(
                        3.0,
                        "verification",
                        frequencies=(0.0, 1.0e9),
                    ),
                ],
                LABELS,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(mdif),
                        "--out-dir",
                        str(out_dir),
                        "--parameter-names",
                        "W",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertEqual(summary["verdict"], "WARNING")
            codes = summary["issue_code_counts"]
            self.assertGreater(codes["INCONSISTENT_FREQUENCY_GRIDS"], 0)
            self.assertGreater(codes["VERIFICATION_OUTSIDE_TRAIN_RANGE"], 0)

    def test_fine_range_outside_coarse_training_range_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fine_mdif = root / "fine.mdif"
            coarse_mdif = root / "coarse.mdif"
            out_dir = root / "audit"
            write_mdif(
                fine_mdif,
                [
                    passive_block(0.0, "train"),
                    passive_block(3.0, "verification"),
                ],
                LABELS,
            )
            write_mdif(
                coarse_mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train"),
                ],
                LABELS,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(fine_mdif),
                        "--coarse-mdif",
                        str(coarse_mdif),
                        "--out-dir",
                        str(out_dir),
                        "--parameter-names",
                        "W",
                    ]
                )
            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertGreater(
                summary["issue_code_counts"]["FINE_OUTSIDE_COARSE_TRAIN_RANGE"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
