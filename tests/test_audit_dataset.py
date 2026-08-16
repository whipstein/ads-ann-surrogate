import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import audit_dataset as AUDIT
import generate_points as POINTS
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


def generated_geometry_json(root: Path, stem: str = "geometries") -> Path:
    geometry_csv = root / f"{stem}.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        status = POINTS.main(
            [
                "generate",
                "--parameter",
                "W=0:4",
                "--count",
                "6",
                "--verification-count",
                "1",
                "--lhs-candidates",
                "2",
                "--out",
                str(geometry_csv),
            ]
        )
    if status != 0:
        raise AssertionError(f"geometry generation failed with status {status}")
    return geometry_csv.with_suffix(".json")


class DatasetAuditTests(unittest.TestCase):
    def test_terminal_audit_verdict_colors(self) -> None:
        class TerminalStream(io.StringIO):
            def isatty(self) -> bool:
                return True

        expected = {
            "PASS": "\033[32m",
            "WARNING": "\033[33m",
            "FAIL": "\033[31m",
        }
        with mock.patch.object(AUDIT.os, "environ", {"TERM": "xterm-256color"}):
            for verdict, color in expected.items():
                with self.subTest(verdict=verdict):
                    self.assertEqual(
                        AUDIT.format_audit_verdict_line(verdict, TerminalStream()),
                        f"{color}dataset audit: {verdict}\033[0m",
                    )

    def test_nonterminal_and_no_color_audit_verdicts_remain_plain(self) -> None:
        self.assertEqual(
            AUDIT.format_audit_verdict_line("PASS", io.StringIO()),
            "dataset audit: PASS",
        )

        class TerminalStream(io.StringIO):
            def isatty(self) -> bool:
                return True

        with mock.patch.object(AUDIT.os, "environ", {"NO_COLOR": "1"}):
            self.assertEqual(
                AUDIT.format_audit_verdict_line("FAIL", TerminalStream()),
                "dataset audit: FAIL",
            )

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
            self.assertEqual(summary["verdict_reasons"], [])
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
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
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
            output = stdout.getvalue()
            self.assertIn("dataset audit: WARNING", output)
            self.assertIn("verdict reasons:", output)
            self.assertIn("WARNING INCONSISTENT_FREQUENCY_GRIDS", output)
            self.assertIn("fine data uses 2 distinct frequency grids", output)
            self.assertIn("action: Compare dataset_frequency_grids.csv", output)
            self.assertIn("WARNING VERIFICATION_OUTSIDE_TRAIN_RANGE", output)

            reasons = {
                reason["code"]: reason for reason in summary["verdict_reasons"]
            }
            grid_reason = reasons["INCONSISTENT_FREQUENCY_GRIDS"]
            self.assertEqual(grid_reason["severity"], "WARNING")
            self.assertEqual(grid_reason["count"], 1)
            self.assertIn("distinct frequency grids", grid_reason["reason"])
            self.assertIn("dataset_frequency_grids.csv", grid_reason["recommendation"])

            report = (out_dir / "dataset_audit.md").read_text()
            self.assertIn("## Why this verdict was issued", report)
            self.assertIn("### WARNING: `INCONSISTENT_FREQUENCY_GRIDS`", report)
            self.assertIn("**Recommended action:**", report)

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

    def test_geometry_json_defines_verification_coverage_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "combined.mdif"
            out_dir = root / "audit"
            geometry_json = generated_geometry_json(root)
            write_mdif(
                mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train"),
                    passive_block(3.0, "verification"),
                ],
                LABELS,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(mdif),
                        "--geometry-json",
                        str(geometry_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                )

            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(summary["parameter_names"], ["W"])
            self.assertNotIn(
                "VERIFICATION_OUTSIDE_TRAIN_RANGE",
                summary["issue_code_counts"],
            )
            self.assertEqual(
                summary["coverage_domain"]["source"],
                "geometry_generation_json",
            )
            self.assertEqual(summary["coverage_domain"]["selection"], "explicit")
            self.assertIn("coverage domain: geometry JSON (explicit)", stdout.getvalue())

            with (out_dir / "dataset_parameter_coverage.csv").open(newline="") as stream:
                coverage = list(csv.DictReader(stream))
            verification = next(row for row in coverage if row["role"] == "verification")
            self.assertEqual(verification["coverage_basis"], "geometry_generation_json")
            self.assertEqual(verification["verification_outside_observed_training"], "1")
            self.assertEqual(verification["verification_outside_coverage"], "0")

    def test_same_stem_geometry_json_is_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "campaign.mdif"
            out_dir = root / "audit"
            generated_geometry_json(root, stem="campaign")
            write_mdif(
                mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train"),
                    passive_block(3.0, "verification"),
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
                    ]
                )

            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertEqual(summary["verdict"], "PASS")
            self.assertEqual(
                summary["coverage_domain"]["selection"],
                "inferred_same_stem",
            )

    def test_verification_outside_geometry_json_range_still_warns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mdif = root / "combined.mdif"
            out_dir = root / "audit"
            geometry_json = generated_geometry_json(root)
            write_mdif(
                mdif,
                [
                    passive_block(1.0, "train"),
                    passive_block(2.0, "train"),
                    passive_block(5.0, "verification"),
                ],
                LABELS,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = AUDIT.main(
                    [
                        "--mdif",
                        str(mdif),
                        "--geometry-json",
                        str(geometry_json),
                        "--out-dir",
                        str(out_dir),
                    ]
                )

            self.assertEqual(status, 0)
            summary = json.loads((out_dir / "dataset_audit.json").read_text())
            self.assertEqual(summary["verdict"], "WARNING")
            self.assertEqual(
                summary["issue_code_counts"]["VERIFICATION_OUTSIDE_GEOMETRY_RANGE"],
                1,
            )
            self.assertNotIn(
                "VERIFICATION_OUTSIDE_TRAIN_RANGE",
                summary["issue_code_counts"],
            )
            self.assertIn("declared geometry generation range", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
