#!/usr/bin/env python3
"""Summarize Newton/Krylov work from ADS HB and Gain Compression status logs.

The parser targets the Newton/linear-solver summary table emitted by ADS with
StatusLevel=4 or StatusLevel=5.  It deliberately ignores the per-inner-iteration
Krylov table printed at level 5: the Newton summary row already contains the
total Krylov iterations for that Newton step.

No ADS installation or third-party Python package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Pattern, Sequence


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NEWTON_ITERATION_RE = re.compile(r"^(?P<iteration>\d+)(?P<rebuild>\*)?$")
NUMBER_WITH_UNIT_RE = re.compile(
    rf"^(?P<value>{NUMBER})(?P<unit>[A-Za-zµμ]*)$"
)
NEWTON_HEADER_RE = re.compile(
    r"(?:\bNewton(?:[-\s]+Raphson)?\s+solver\b|\bNewton\s+iteration\b)",
    re.IGNORECASE,
)
NEWTON_COLUMNS_RE = re.compile(
    r"(?=.*\bIter(?:ation)?s?\b)(?=.*\bResidual\b)",
    re.IGNORECASE,
)
KRYLOV_HEADER_RE = re.compile(
    r"\b(?:Krylov|GMRES|BiCGStab)\s+solver\b", re.IGNORECASE
)
SEPARATOR_RE = re.compile(r"^\s*[-=_]{3,}\s*$")
FAILURE_RE = re.compile(
    r"(?:fail(?:ed|ure)?\s+(?:to\s+)?converge|did\s+not\s+converge|"
    r"terminat(?:e|ed|ing)\s+due|convergence\s+failure|aborted)",
    re.IGNORECASE,
)
RETRY_RE = re.compile(
    r"(?:retry|re-try|continuation|source\s+stepping|arc[- ]length)",
    re.IGNORECASE,
)

FREQUENCY_PATTERNS = [
    re.compile(
        rf"\b(?:fundamental\s+)?freq(?:uency)?(?:\s*\[\s*\d+\s*\])?"
        rf"\s*(?:=|:)\s*(?P<value>{NUMBER})\s*"
        r"(?P<unit>[kKmMgGtT]?Hz)?\b",
        re.IGNORECASE,
    ),
]
POWER_PATTERNS = [
    re.compile(
        rf"\b(?:input\s*power|inputpower|GC_InputPower|dbm_in|Pin|Pavs)"
        rf"(?:\s*\[\s*\d+\s*\])?\s*(?:=|:)\s*"
        rf"(?P<value>{NUMBER})\s*(?P<unit>dBm|dBW|mW|W)?\b",
        re.IGNORECASE,
    ),
]


@dataclass
class NewtonRecord:
    line_number: int
    iteration: int
    jacobian_rebuilt: bool
    kcl_residual_a: float
    krylov_iterations: int
    krylov_residual: float


@dataclass
class SolveRecord:
    model: str
    source_file: str
    solve_index: int
    frequency_hz: float | None
    frequency_text: str
    input_power_dbm: float | None
    input_power_text: str
    start_line: int
    end_line: int
    newton: list[NewtonRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retry_messages: list[str] = field(default_factory=list)

    def csv_row(self) -> dict[str, object]:
        krylov = [row.krylov_iterations for row in self.newton]
        final = self.newton[-1]
        return {
            "model": self.model,
            "source_file": self.source_file,
            "solve_index": self.solve_index,
            "frequency_hz": _optional_number(self.frequency_hz),
            "frequency_text": self.frequency_text,
            "input_power_dbm": _optional_number(self.input_power_dbm),
            "input_power_text": self.input_power_text,
            "newton_iterations": len(self.newton),
            "first_newton_iteration": self.newton[0].iteration,
            "last_newton_iteration": final.iteration,
            "total_krylov_iterations": sum(krylov),
            "mean_krylov_per_newton": sum(krylov) / len(krylov),
            "max_krylov_in_newton_step": max(krylov),
            "final_kcl_residual_a": final.kcl_residual_a,
            "final_krylov_residual": final.krylov_residual,
            "jacobian_rebuild_rows": sum(
                row.jacobian_rebuilt for row in self.newton
            ),
            "warning_count": len(self.warnings),
            "failure_detected": bool(self.warnings),
            "retry_message_count": len(self.retry_messages),
            "warnings": " | ".join(self.warnings),
            "retry_messages": " | ".join(self.retry_messages),
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass
class ParseResult:
    model: str
    source_file: str
    solves: list[SolveRecord]
    unmatched_failure_messages: list[str]
    unmatched_retry_messages: list[str]
    diagnostic_lines: list[str]


def _optional_number(value: float | None) -> float | str:
    return "" if value is None else value


def _compile_custom_regex(expression: str | None, option: str) -> Pattern[str] | None:
    if not expression:
        return None
    pattern = re.compile(expression, re.IGNORECASE)
    if "value" not in pattern.groupindex:
        raise ValueError(f"{option} must contain a named capture group '(?P<value>...)'")
    return pattern


def _match_value(
    line: str,
    custom: Pattern[str] | None,
    defaults: Sequence[Pattern[str]],
) -> tuple[float, str, str] | None:
    patterns: Iterable[Pattern[str]] = (
        [custom, *defaults] if custom is not None else defaults
    )
    for pattern in patterns:
        match = pattern.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        unit = groups.get("unit") or ""
        return float(groups["value"]), unit, match.group(0).strip()
    return None


def _frequency_hz(value: float, unit: str) -> float:
    multipliers = {
        "": 1.0,
        "hz": 1.0,
        "khz": 1.0e3,
        "mhz": 1.0e6,
        "ghz": 1.0e9,
        "thz": 1.0e12,
    }
    normalized = unit.strip().lower()
    if normalized not in multipliers:
        raise ValueError(f"Unsupported frequency unit {unit!r}")
    return value * multipliers[normalized]


def _power_dbm(value: float, unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in {"", "dbm"}:
        return value
    if normalized == "dbw":
        return value + 30.0
    watts = value * (1.0e-3 if normalized == "mw" else 1.0)
    if watts <= 0.0:
        raise ValueError("A linear input power must be positive")
    return 10.0 * math.log10(watts / 1.0e-3)


def _current_amperes(value: float, unit: str) -> float:
    normalized = unit.strip().replace("μ", "u").replace("µ", "u")
    multipliers = {
        "": 1.0,
        "A": 1.0,
        "fA": 1.0e-15,
        "pA": 1.0e-12,
        "nA": 1.0e-9,
        "uA": 1.0e-6,
        "mA": 1.0e-3,
        "kA": 1.0e3,
        "MA": 1.0e6,
    }
    if normalized not in multipliers:
        # ADS documents this column as a KCL current residual. Preserve a
        # usable value if a release omits or changes the printed unit.
        return value
    return value * multipliers[normalized]


def _parse_newton_row(line: str, line_number: int) -> NewtonRecord | None:
    tokens = [token.strip(",;:()[]{}|") for token in line.strip().split()]
    if len(tokens) < 4:
        return None
    iteration_match = NEWTON_ITERATION_RE.fullmatch(tokens[0])
    if iteration_match is None:
        return None

    linear_summary: tuple[int, float] | None = None
    for index in range(len(tokens) - 2, 1, -1):
        if (
            re.fullmatch(r"\d+", tokens[index]) is not None
            and re.fullmatch(NUMBER, tokens[index + 1]) is not None
        ):
            linear_summary = (int(tokens[index]), float(tokens[index + 1]))
            break
    if linear_summary is None:
        return None

    kcl_match = NUMBER_WITH_UNIT_RE.fullmatch(tokens[1])
    if kcl_match is None:
        return None
    kcl_unit = kcl_match.group("unit")
    if not kcl_unit and len(tokens) > 4 and re.fullmatch(r"[A-Za-zµμ]+", tokens[2]):
        kcl_unit = tokens[2]

    return NewtonRecord(
        line_number=line_number,
        iteration=int(iteration_match.group("iteration")),
        jacobian_rebuilt=bool(iteration_match.group("rebuild")),
        kcl_residual_a=_current_amperes(
            float(kcl_match.group("value")), kcl_unit
        ),
        krylov_iterations=linear_summary[0],
        krylov_residual=linear_summary[1],
    )


def parse_ads_status_text(
    text: str,
    model: str,
    source_file: str,
    frequency_regex: str | None = None,
    power_regex: str | None = None,
) -> ParseResult:
    """Parse one ADS status log into HB solve records."""

    custom_frequency = _compile_custom_regex(frequency_regex, "--frequency-regex")
    custom_power = _compile_custom_regex(power_regex, "--power-regex")
    active_frequency: tuple[float | None, str] = (None, "")
    active_power: tuple[float | None, str] = (None, "")
    solves: list[SolveRecord] = []
    current: SolveRecord | None = None
    in_newton_table = False
    seen_newton_row = False
    header_scan_remaining = 0
    unmatched_failures: list[str] = []
    unmatched_retries: list[str] = []
    diagnostic_lines: list[str] = []

    def finish_current() -> None:
        nonlocal current
        if current is not None and current.newton:
            current.solve_index = len(solves) + 1
            solves.append(current)
        current = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.replace("\x00", "").rstrip()
        frequency_match = _match_value(
            line, custom_frequency, FREQUENCY_PATTERNS
        )
        power_match = _match_value(line, custom_power, POWER_PATTERNS)

        if frequency_match is not None:
            value, unit, label = frequency_match
            normalized = _frequency_hz(value, unit)
            if current is not None and active_frequency[0] != normalized:
                finish_current()
            active_frequency = (normalized, label)
        if power_match is not None:
            value, unit, label = power_match
            normalized = _power_dbm(value, unit)
            if current is not None and active_power[0] != normalized:
                finish_current()
            active_power = (normalized, label)

        if KRYLOV_HEADER_RE.search(line):
            in_newton_table = False
            seen_newton_row = False
            header_scan_remaining = 0
            if len(diagnostic_lines) < 24:
                diagnostic_lines.append(f"{line_number}: {line.strip()}")
            continue
        if NEWTON_HEADER_RE.search(line):
            in_newton_table = True
            seen_newton_row = False
            header_scan_remaining = 20
            if len(diagnostic_lines) < 24:
                diagnostic_lines.append(f"{line_number}: {line.strip()}")
            continue
        if NEWTON_COLUMNS_RE.search(line):
            in_newton_table = True
            seen_newton_row = False
            header_scan_remaining = 20
            if len(diagnostic_lines) < 24:
                diagnostic_lines.append(f"{line_number}: {line.strip()}")
            continue

        if in_newton_table:
            stripped = line.strip()
            if (
                not stripped
                or SEPARATOR_RE.fullmatch(stripped)
            ):
                continue
            newton = _parse_newton_row(line, line_number)
            if newton is not None:
                if (
                    current is not None
                    and current.newton
                    and newton.iteration <= current.newton[-1].iteration
                ):
                    finish_current()
                if current is None:
                    current = SolveRecord(
                        model=model,
                        source_file=source_file,
                        solve_index=0,
                        frequency_hz=active_frequency[0],
                        frequency_text=active_frequency[1],
                        input_power_dbm=active_power[0],
                        input_power_text=active_power[1],
                        start_line=line_number,
                        end_line=line_number,
                    )
                current.newton.append(newton)
                current.end_line = line_number
                seen_newton_row = True
                header_scan_remaining = 20
                if len(diagnostic_lines) < 24:
                    diagnostic_lines.append(f"{line_number}: {line.strip()}")
                continue
            if not seen_newton_row and header_scan_remaining > 0:
                header_scan_remaining -= 1
                if stripped and len(diagnostic_lines) < 24:
                    diagnostic_lines.append(f"{line_number}: {line.strip()}")
                continue
            in_newton_table = False
            seen_newton_row = False
            header_scan_remaining = 0

        if FAILURE_RE.search(line):
            message = line.strip()
            if current is None:
                unmatched_failures.append(message)
            else:
                current.warnings.append(message)
                current.end_line = line_number
        if RETRY_RE.search(line):
            message = line.strip()
            if current is None:
                unmatched_retries.append(message)
            else:
                current.retry_messages.append(message)
                current.end_line = line_number

    finish_current()
    return ParseResult(
        model=model,
        source_file=source_file,
        solves=solves,
        unmatched_failure_messages=unmatched_failures,
        unmatched_retry_messages=unmatched_retries,
        diagnostic_lines=diagnostic_lines,
    )


def summarize_result(result: ParseResult) -> dict[str, object]:
    point_krylov = [
        sum(row.krylov_iterations for row in solve.newton)
        for solve in result.solves
    ]
    point_newton = [len(solve.newton) for solve in result.solves]
    total_krylov = sum(point_krylov)
    total_newton = sum(point_newton)
    worst_index = (
        result.solves[point_krylov.index(max(point_krylov))].solve_index
        if point_krylov
        else ""
    )
    return {
        "model": result.model,
        "source_file": result.source_file,
        "solve_count": len(result.solves),
        "frequency_labelled_solves": sum(
            solve.frequency_hz is not None for solve in result.solves
        ),
        "power_labelled_solves": sum(
            solve.input_power_dbm is not None for solve in result.solves
        ),
        "total_newton_iterations": total_newton,
        "total_krylov_iterations": total_krylov,
        "mean_newton_per_solve": _mean_or_blank(point_newton),
        "mean_krylov_per_solve": _mean_or_blank(point_krylov),
        "median_krylov_per_solve": _median_or_blank(point_krylov),
        "p95_krylov_per_solve": _percentile_or_blank(point_krylov, 0.95),
        "max_krylov_per_solve": max(point_krylov) if point_krylov else "",
        "max_krylov_solve_index": worst_index,
        "mean_krylov_per_newton": (
            total_krylov / total_newton if total_newton else ""
        ),
        "failure_solve_count": sum(bool(solve.warnings) for solve in result.solves),
        "retry_solve_count": sum(
            bool(solve.retry_messages) for solve in result.solves
        ),
        "unmatched_failure_message_count": len(
            result.unmatched_failure_messages
        ),
        "unmatched_retry_message_count": len(result.unmatched_retry_messages),
    }


def _mean_or_blank(values: Sequence[int]) -> float | str:
    return statistics.fmean(values) if values else ""


def _median_or_blank(values: Sequence[int]) -> float | str:
    return statistics.median(values) if values else ""


def _percentile_or_blank(values: Sequence[int], fraction: float) -> int | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_log(path_text: str) -> tuple[str, str]:
    if path_text == "-":
        return sys.stdin.read(), "<stdin>"
    path = Path(path_text)
    return _decode_log_bytes(path.read_bytes()), str(path)


def _decode_log_bytes(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if b"\x00" in data[:4096]:
        even_nuls = data[0::2].count(0)
        odd_nuls = data[1::2].count(0)
        encoding = "utf-16-be" if even_nuls > odd_nuls else "utf-16-le"
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _print_summary(rows: Sequence[dict[str, object]]) -> None:
    headers = [
        "model",
        "solve_count",
        "total_newton_iterations",
        "total_krylov_iterations",
        "mean_krylov_per_solve",
        "max_krylov_per_solve",
    ]
    printable = [[str(row[name]) for name in headers] for row in rows]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in printable))
        for idx in range(len(headers))
    ]
    print("  ".join(name.ljust(widths[idx]) for idx, name in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in printable:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse ADS StatusLevel=4/5 HB or Gain Compression logs and compare "
            "Newton/Krylov solver work."
        )
    )
    parser.add_argument(
        "logs",
        nargs="+",
        metavar="LOG",
        help="ADS status text file. Use - to read one log from standard input.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="NAME",
        help="Model labels in the same order as LOG arguments. Defaults to file stems.",
    )
    parser.add_argument(
        "--out-dir",
        default="ads_hb_solver_report",
        help="Output directory. Default: ads_hb_solver_report",
    )
    parser.add_argument(
        "--frequency-regex",
        help=(
            "Optional release-specific regex with named group (?P<value>...) "
            "and optional (?P<unit>...)."
        ),
    )
    parser.add_argument(
        "--power-regex",
        help=(
            "Optional release-specific regex with named group (?P<value>...) "
            "and optional (?P<unit>...)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise SystemExit("--labels must provide exactly one name for each LOG")
    if args.logs.count("-") > 1:
        raise SystemExit("Standard input (-) can only be used once")

    labels = args.labels or [
        "stdin" if value == "-" else Path(value).stem for value in args.logs
    ]
    results: list[ParseResult] = []
    for path_text, label in zip(args.logs, labels):
        text, source_file = _read_log(path_text)
        result = parse_ads_status_text(
            text,
            model=label,
            source_file=source_file,
            frequency_regex=args.frequency_regex,
            power_regex=args.power_regex,
        )
        if not result.solves:
            candidates = ""
            if result.diagnostic_lines:
                candidates = (
                    "\nParser candidate lines (include these in a bug report):\n  "
                    + "\n  ".join(result.diagnostic_lines)
                )
            raise SystemExit(
                f"No Newton/Krylov summary rows found in {source_file}. "
                "Set the ADS Gain Compression Status level to 4 or 5."
                f"{candidates}"
            )
        results.append(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_rows = [
        solve.csv_row() for result in results for solve in result.solves
    ]
    summary_rows = [summarize_result(result) for result in results]
    _write_csv(out_dir / "ads_hb_solver_points.csv", point_rows)
    _write_csv(out_dir / "ads_hb_solver_summary.csv", summary_rows)
    (out_dir / "ads_hb_solver_summary.json").write_text(
        json.dumps(
            {
                "summaries": summary_rows,
                "unmatched_messages": [
                    {
                        "model": result.model,
                        "failure_messages": result.unmatched_failure_messages,
                        "retry_messages": result.unmatched_retry_messages,
                    }
                    for result in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_summary(summary_rows)
    print(f"\nPoint details: {out_dir / 'ads_hb_solver_points.csv'}")
    print(f"Summary CSV:  {out_dir / 'ads_hb_solver_summary.csv'}")
    print(f"Summary JSON: {out_dir / 'ads_hb_solver_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
