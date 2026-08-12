import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from de_generated_scripts import parse_ads_hb_solver_log as PARSER


LEVEL5_LOG = """
Frequency = 2 GHz
Input power = -20 dBm
Newton solver:                         Linear solver:
Iter KCL residual Damp % Sol update    Iters Residual
-----------------------------------------------------
1    2.000 mA    100.0                 5     1.000e-03
-----------------------------------------------------
Krylov solver (target tol = 0.001):
Iter Residual
0 1.0
1 0.1
Newton solver:                         Linear solver:
Iter KCL residual Damp % Sol update    Iters Residual
-----------------------------------------------------
2*   4.000 uA    100.0                 3     8.000e-04
Input power = -10 dBm
Newton solver:                         Linear solver:
Iter KCL residual Damp % Sol update    Iters Residual
1    1.000 mA    100.0                 7     2.000e-03
Newton solver:                         Linear solver:
Iter KCL residual Damp % Sol update    Iters Residual
2    5.000 nA    100.0                 4     7.000e-04
"""


class AdsHbLogParserTests(unittest.TestCase):
    def test_parses_summary_rows_and_ignores_inner_krylov_rows(self) -> None:
        result = PARSER.parse_ads_status_text(
            LEVEL5_LOG,
            model="baseline",
            source_file="baseline.log",
        )
        self.assertEqual(len(result.solves), 2)
        first, second = result.solves
        self.assertEqual(first.frequency_hz, 2.0e9)
        self.assertEqual(first.input_power_dbm, -20.0)
        self.assertEqual(len(first.newton), 2)
        self.assertEqual(sum(row.krylov_iterations for row in first.newton), 8)
        self.assertAlmostEqual(first.newton[-1].kcl_residual_a, 4.0e-6)
        self.assertTrue(first.newton[-1].jacobian_rebuilt)
        self.assertEqual(second.input_power_dbm, -10.0)
        self.assertEqual(sum(row.krylov_iterations for row in second.newton), 11)

    def test_newton_counter_reset_splits_unlabelled_solves(self) -> None:
        text = """
Newton solver: Linear solver:
Iter KCL residual Iters Residual
1 1mA 3 1e-2
2 1uA 2 1e-3
1 2mA 5 2e-2
2 2uA 4 2e-3
"""
        result = PARSER.parse_ads_status_text(text, "trial", "trial.log")
        self.assertEqual(len(result.solves), 2)
        self.assertEqual(
            [
                sum(row.krylov_iterations for row in solve.newton)
                for solve in result.solves
            ],
            [5, 9],
        )

    def test_accepts_wrapped_header(self) -> None:
        text = """
Newton solver:
Linear solver:
Iter
KCL residual
Damp % Sol update
Iters Residual
------------------------------------
1 2.0 mA 100.0 6 1.0e-3
"""
        result = PARSER.parse_ads_status_text(text, "trial", "trial.log")
        self.assertEqual(len(result.solves), 1)
        self.assertEqual(result.solves[0].newton[0].krylov_iterations, 6)

    def test_reads_utf16_message_window_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ads_status.txt"
            path.write_text(LEVEL5_LOG, encoding="utf-16")
            text, source_file = PARSER._read_log(str(path))
            result = PARSER.parse_ads_status_text(text, "baseline", source_file)
            self.assertEqual(len(result.solves), 2)
            self.assertEqual(
                sum(
                    row.krylov_iterations
                    for solve in result.solves
                    for row in solve.newton
                ),
                19,
            )

    def test_failure_lists_candidate_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unsupported.log"
            path.write_text("Newton solver:\ncustom columns\ncustom row\n")
            with self.assertRaisesRegex(SystemExit, "Parser candidate lines"):
                PARSER.main([str(path)])

    def test_cli_writes_comparison_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "baseline.log"
            second = root / "trial.log"
            output = root / "report"
            first.write_text(LEVEL5_LOG)
            second.write_text(LEVEL5_LOG.replace("7     2.000e-03", "9     2.000e-03"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    PARSER.main(
                        [
                            str(first),
                            str(second),
                            "--labels",
                            "baseline",
                            "trial",
                            "--out-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            self.assertTrue((output / "ads_hb_solver_points.csv").is_file())
            self.assertTrue((output / "ads_hb_solver_summary.csv").is_file())
            self.assertTrue((output / "ads_hb_solver_summary.json").is_file())
            report = (output / "ads_hb_solver_report.md").read_text()
            self.assertIn("# ADS HB Solver Comparison", report)
            self.assertIn("| baseline | 2 | 4 | 19 |", report)
            self.assertIn("solver_work_totals.svg", report)
            self.assertIn("krylov_per_solve_statistics.svg", report)
            self.assertIn("krylov_by_solve.svg", report)
            self.assertIn("## Results by frequency", report)
            for name in (
                "solver_work_totals.svg",
                "krylov_per_solve_statistics.svg",
                "krylov_by_solve.svg",
            ):
                ElementTree.parse(output / name)
            self.assertIn(
                ">10.5</text>",
                (output / "krylov_per_solve_statistics.svg").read_text(),
            )
            summary_json = json.loads(
                (output / "ads_hb_solver_summary.json").read_text()
            )
            self.assertEqual(
                summary_json["report_artifacts"],
                [
                    "ads_hb_solver_report.md",
                    "solver_work_totals.svg",
                    "krylov_per_solve_statistics.svg",
                    "krylov_by_solve.svg",
                ],
            )


if __name__ == "__main__":
    unittest.main()
