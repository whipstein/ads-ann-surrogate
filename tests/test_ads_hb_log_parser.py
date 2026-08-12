import contextlib
import io
import json
import re
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
Total elapsed time: 12.5 seconds
Total CPU time: 18.75 seconds
"""

RESOURCE_USAGE_LOG = LEVEL5_LOG.replace(
    "Total elapsed time: 12.5 seconds\nTotal CPU time: 18.75 seconds",
    """Resource usage:
Total CPU time = 18.75 seconds.
Simulation stopwatch time = 11.25 seconds.
Total stopwatch time = 12.5 seconds.""",
)


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
        self.assertEqual(result.wall_clock_seconds, 12.5)
        self.assertEqual(result.cpu_time_seconds, 18.75)
        self.assertIn("Total elapsed time", result.wall_clock_source)

    def test_parses_paired_cpu_elapsed_clock_duration(self) -> None:
        text = LEVEL5_LOG.replace(
            "Total elapsed time: 12.5 seconds\nTotal CPU time: 18.75 seconds",
            "Total CPU/Elapsed time: 95.0 s / 0:01:02.5",
        )
        result = PARSER.parse_ads_status_text(text, "trial", "trial.log")
        self.assertEqual(result.wall_clock_seconds, 62.5)
        self.assertEqual(result.cpu_time_seconds, 95.0)

    def test_parses_ads_resource_usage_stopwatch_times(self) -> None:
        result = PARSER.parse_ads_status_text(
            RESOURCE_USAGE_LOG,
            model="baseline",
            source_file="baseline.log",
        )
        self.assertEqual(result.wall_clock_seconds, 12.5)
        self.assertEqual(result.simulation_stopwatch_seconds, 11.25)
        self.assertEqual(result.cpu_time_seconds, 18.75)
        self.assertIn("Total stopwatch time", result.wall_clock_source)
        self.assertIn(
            "Simulation stopwatch time", result.simulation_stopwatch_source
        )
        self.assertIn("Total CPU time", result.cpu_time_source)

    def test_runtime_plot_uses_simulation_stopwatch_without_total(self) -> None:
        simulation_only = RESOURCE_USAGE_LOG.replace(
            "Total stopwatch time = 12.5 seconds.\n", ""
        )
        result = PARSER.parse_ads_status_text(
            simulation_only,
            model="baseline",
            source_file="baseline.log",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "runtime_comparison.svg"
            PARSER._write_runtime_svg(
                path,
                [PARSER.summarize_result(result)],
            )
            plot = path.read_text()
        self.assertNotIn("No ADS stopwatch timing", plot)
        self.assertIn("Simulation stopwatch time", plot)
        self.assertIn(">11.25 s</text>", plot)

    def test_prefers_total_time_and_rejects_stage_elapsed_as_total(self) -> None:
        without_footer = LEVEL5_LOG.replace(
            "Total elapsed time: 12.5 seconds\nTotal CPU time: 18.75 seconds",
            "Matrix solver elapsed time: 3.5 seconds",
        )
        result = PARSER.parse_ads_status_text(
            without_footer, "trial", "trial.log"
        )
        self.assertIsNone(result.wall_clock_seconds)
        with_total = without_footer + "\nTotal time: 0:01:05.5\n"
        result = PARSER.parse_ads_status_text(with_total, "trial", "trial.log")
        self.assertEqual(result.wall_clock_seconds, 65.5)

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
            first.write_text(RESOURCE_USAGE_LOG)
            second.write_text(
                RESOURCE_USAGE_LOG.replace("7     2.000e-03", "9     2.000e-03")
                .replace("12.5 seconds", "15.0 seconds")
                .replace("11.25 seconds", "13.5 seconds")
                .replace("18.75 seconds", "22.5 seconds")
            )
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
            self.assertIn(
                "| baseline | 2 | 12.5 s | 11.25 s | 6.25 s | 18.75 s | 4 | 19 |",
                report,
            )
            self.assertIn("| trial | +20.0% | +20.0% | +20.0% |", report)
            self.assertIn("runtime_comparison.svg", report)
            self.assertRegex(
                report,
                re.compile(r"runtime_comparison\.svg\?v=[0-9a-f]{12}"),
            )
            self.assertIn("solver_work_totals.svg", report)
            self.assertIn("krylov_per_solve_statistics.svg", report)
            self.assertIn("krylov_by_solve.svg", report)
            self.assertIn("## Results by frequency", report)
            for name in (
                "runtime_comparison.svg",
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
                    "runtime_comparison.svg",
                    "solver_work_totals.svg",
                    "krylov_per_solve_statistics.svg",
                    "krylov_by_solve.svg",
                ],
            )
            self.assertEqual(
                summary_json["summaries"][0]["wall_clock_seconds"], 12.5
            )
            self.assertEqual(
                summary_json["summaries"][0]["simulation_stopwatch_seconds"],
                11.25,
            )

    def test_cli_timing_override_replaces_log_timing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log = root / "baseline.log"
            output = root / "report"
            log.write_text(LEVEL5_LOG)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    PARSER.main(
                        [
                            str(log),
                            "--wall-clock-seconds",
                            "9.25",
                            "--cpu-time-seconds",
                            "14.5",
                            "--out-dir",
                            str(output),
                        ]
                    ),
                    0,
                )
            summary = json.loads(
                (output / "ads_hb_solver_summary.json").read_text()
            )["summaries"][0]
            self.assertEqual(summary["wall_clock_seconds"], 9.25)
            self.assertEqual(summary["cpu_time_seconds"], 14.5)
            self.assertEqual(summary["wall_clock_source"], "CLI --wall-clock-seconds")


if __name__ == "__main__":
    unittest.main()
