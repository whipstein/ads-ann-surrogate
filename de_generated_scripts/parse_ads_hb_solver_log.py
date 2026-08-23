#!/usr/bin/env python3
"""Summarize Newton/Krylov work from ADS HB and Gain Compression status logs.

The parser targets the Newton/linear-solver summary table emitted by ADS with
StatusLevel=4 or StatusLevel=5.  It deliberately ignores the per-inner-iteration
Krylov table printed at level 5: the Newton summary row already contains the
total Krylov iterations for that Newton step.

No ADS installation is required. Pillow is used for the report's PNG plots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Pattern, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from cli_options import (
    add_options_json_argument,
    finalize_options_json_update,
    parse_args_with_options_json,
)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
DURATION_UNIT = r"(?:sec(?:ond)?s?|min(?:ute)?s?|hours?|hrs?|ms|us|µs|μs|ns|s|h)"
DURATION_TEXT = rf"(?:\d+:\d{{2}}(?::\d{{2}}(?:\.\d+)?)?|{NUMBER}\s*{DURATION_UNIT}?)"
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
WALL_TIME_RE = re.compile(
    rf"(?P<label>(?:(?:total|overall)\s+time|(?:(?:total|overall)\s+)?"
    rf"(?:elapsed(?:\s+wall(?:[- ]?clock)?)?|wall(?:[- ]?clock)?|clock|real|simulation|run)"
    rf"\s+time))\s*(?:\([^)]*\)|\[[^]]*\])?\s*[:=]\s*"
    rf"(?P<duration>{DURATION_TEXT})",
    re.IGNORECASE,
)
CPU_TIME_RE = re.compile(
    rf"(?P<label>(?:(?:total|overall)\s+)?(?:cpu|user|processor)\s+time)"
    rf"\s*(?:\([^)]*\)|\[[^]]*\])?\s*[:=]\s*(?P<duration>{DURATION_TEXT})",
    re.IGNORECASE,
)
CPU_ELAPSED_PAIR_RE = re.compile(
    rf"(?P<label>(?:(?:total|overall)\s+)?cpu\s*/\s*"
    rf"(?:elapsed|wall(?:[- ]?clock)?)\s+time)"
    rf"\s*(?:\([^)]*\)|\[[^]]*\])?\s*[:=]\s*"
    rf"(?P<cpu>{DURATION_TEXT})\s*(?:/|,)\s*(?P<wall>{DURATION_TEXT})",
    re.IGNORECASE,
)
SIMULATION_STOPWATCH_RE = re.compile(
    rf"(?P<label>simulation\s+stopwatch\s+time)\s*[:=]\s*"
    rf"(?P<duration>{DURATION_TEXT})",
    re.IGNORECASE,
)
TOTAL_STOPWATCH_RE = re.compile(
    rf"(?P<label>total\s+stopwatch\s+time)\s*[:=]\s*"
    rf"(?P<duration>{DURATION_TEXT})",
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

PLOT_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
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
class TimingSample:
    kind: str
    seconds: float
    score: int
    line_number: int
    source_text: str


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
    wall_clock_seconds: float | None
    cpu_time_seconds: float | None
    simulation_stopwatch_seconds: float | None
    wall_clock_source: str
    cpu_time_source: str
    simulation_stopwatch_source: str


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


def _duration_seconds(text: str) -> float:
    value = text.strip().replace("μ", "u").replace("µ", "u")
    if ":" in value:
        parts = [float(part) for part in value.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60.0 + parts[1]
        raise ValueError(f"Unsupported clock duration {text!r}")
    match = re.fullmatch(
        rf"(?P<value>{NUMBER})\s*(?P<unit>{DURATION_UNIT})?",
        value,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Unsupported duration {text!r}")
    number = float(match.group("value"))
    unit = (match.group("unit") or "s").lower()
    multipliers = {
        "ns": 1.0e-9,
        "us": 1.0e-6,
        "ms": 1.0e-3,
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "min": 60.0,
        "mins": 60.0,
        "minute": 60.0,
        "minutes": 60.0,
        "h": 3600.0,
        "hr": 3600.0,
        "hrs": 3600.0,
        "hour": 3600.0,
        "hours": 3600.0,
    }
    return number * multipliers[unit]


def _timing_score(label: str, line: str, match_start: int) -> int:
    lowered_label = label.lower()
    lowered_line = line.lower()
    score = 10
    if "total" in lowered_label or "overall" in lowered_label:
        score += 100
    if "simulation" in lowered_label or "simulation finished" in lowered_line:
        score += 70
    if "elapsed" in lowered_label or "wall" in lowered_label:
        score += 20
    if any(word in lowered_label for word in ("cpu", "user", "processor")):
        score += 20
    prefix = line[:match_start]
    if not re.search(r"[A-Za-z0-9]", prefix):
        score += 10
    if re.search(r"\b(?:start|begin|current|end)\b", prefix, re.IGNORECASE):
        score -= 80
    return score


def _timing_samples_from_line(line: str, line_number: int) -> list[TimingSample]:
    samples: list[TimingSample] = []
    paired_spans: list[tuple[int, int]] = []
    for kind, pattern, score in (
        ("simulation_stopwatch", SIMULATION_STOPWATCH_RE, 250),
        ("wall", TOTAL_STOPWATCH_RE, 300),
    ):
        for match in pattern.finditer(line):
            samples.append(
                TimingSample(
                    kind=kind,
                    seconds=_duration_seconds(match.group("duration")),
                    score=score,
                    line_number=line_number,
                    source_text=line.strip(),
                )
            )
    for match in CPU_ELAPSED_PAIR_RE.finditer(line):
        paired_spans.append(match.span())
        score = _timing_score(match.group("label"), line, match.start())
        samples.extend(
            [
                TimingSample(
                    kind="cpu",
                    seconds=_duration_seconds(match.group("cpu")),
                    score=score,
                    line_number=line_number,
                    source_text=line.strip(),
                ),
                TimingSample(
                    kind="wall",
                    seconds=_duration_seconds(match.group("wall")),
                    score=score,
                    line_number=line_number,
                    source_text=line.strip(),
                ),
            ]
        )

    def overlaps_pair(start: int, end: int) -> bool:
        return any(start < pair_end and end > pair_start for pair_start, pair_end in paired_spans)

    for kind, pattern in (("wall", WALL_TIME_RE), ("cpu", CPU_TIME_RE)):
        for match in pattern.finditer(line):
            if overlaps_pair(*match.span()):
                continue
            samples.append(
                TimingSample(
                    kind=kind,
                    seconds=_duration_seconds(match.group("duration")),
                    score=_timing_score(match.group("label"), line, match.start()),
                    line_number=line_number,
                    source_text=line.strip(),
                )
            )
    return samples


def _select_timing_sample(
    samples: Sequence[TimingSample], kind: str
) -> TimingSample | None:
    matching = [sample for sample in samples if sample.kind == kind]
    matching = [sample for sample in matching if sample.score >= 40]
    if not matching:
        return None
    return max(matching, key=lambda sample: (sample.score, sample.line_number))


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
    timing_samples: list[TimingSample] = []

    def finish_current() -> None:
        nonlocal current
        if current is not None and current.newton:
            current.solve_index = len(solves) + 1
            solves.append(current)
        current = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.replace("\x00", "").rstrip()
        timing_samples.extend(_timing_samples_from_line(line, line_number))
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
    wall_timing = _select_timing_sample(timing_samples, "wall")
    cpu_timing = _select_timing_sample(timing_samples, "cpu")
    simulation_stopwatch = _select_timing_sample(
        timing_samples, "simulation_stopwatch"
    )
    return ParseResult(
        model=model,
        source_file=source_file,
        solves=solves,
        unmatched_failure_messages=unmatched_failures,
        unmatched_retry_messages=unmatched_retries,
        diagnostic_lines=diagnostic_lines,
        wall_clock_seconds=(wall_timing.seconds if wall_timing else None),
        cpu_time_seconds=(cpu_timing.seconds if cpu_timing else None),
        simulation_stopwatch_seconds=(
            simulation_stopwatch.seconds if simulation_stopwatch else None
        ),
        wall_clock_source=(wall_timing.source_text if wall_timing else ""),
        cpu_time_source=(cpu_timing.source_text if cpu_timing else ""),
        simulation_stopwatch_source=(
            simulation_stopwatch.source_text if simulation_stopwatch else ""
        ),
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
        "wall_clock_seconds": _optional_number(result.wall_clock_seconds),
        "wall_clock_per_solve_seconds": (
            result.wall_clock_seconds / len(result.solves)
            if result.wall_clock_seconds is not None and result.solves
            else ""
        ),
        "simulation_stopwatch_seconds": _optional_number(
            result.simulation_stopwatch_seconds
        ),
        "simulation_stopwatch_per_solve_seconds": (
            result.simulation_stopwatch_seconds / len(result.solves)
            if result.simulation_stopwatch_seconds is not None and result.solves
            else ""
        ),
        "cpu_time_seconds": _optional_number(result.cpu_time_seconds),
        "cpu_time_per_solve_seconds": (
            result.cpu_time_seconds / len(result.solves)
            if result.cpu_time_seconds is not None and result.solves
            else ""
        ),
        "wall_clock_source": result.wall_clock_source,
        "simulation_stopwatch_source": result.simulation_stopwatch_source,
        "cpu_time_source": result.cpu_time_source,
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


def _metric_number(value: object) -> float:
    if value == "" or value is None:
        return 0.0
    return float(value)


def _format_number(value: object, digits: int = 3) -> str:
    if value == "" or value is None:
        return "—"
    number = float(value)
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _format_axis_number(value: float) -> str:
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.2g}M"
    if abs(value) >= 1.0e3:
        return f"{value / 1.0e3:.2g}k"
    if abs(value) >= 10.0:
        return f"{value:.0f}"
    if abs(value) >= 1.0:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _format_duration(value: object) -> str:
    if value == "" or value is None:
        return "—"
    seconds = float(value)
    if seconds < 60.0:
        formatted = f"{seconds:.3f}".rstrip("0").rstrip(".")
        return f"{formatted} s"
    hours = int(seconds // 3600.0)
    minutes = int((seconds - hours * 3600.0) // 60.0)
    remainder = seconds - hours * 3600.0 - minutes * 60.0
    if hours:
        return f"{hours}h {minutes}m {remainder:.1f}s"
    return f"{minutes}m {remainder:.1f}s"


def _nice_axis_max(value: float) -> float:
    if value <= 0.0:
        return 1.0
    target = value * 1.12
    exponent = 10.0 ** math.floor(math.log10(target))
    fraction = target / exponent
    for nice in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= nice:
            return nice * exponent
    return 10.0 * exponent


def _short_label(value: object, limit: int = 18) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _plot_font(size: int, bold: bool = False):
    try:
        from PIL import ImageFont
    except ImportError as exc:
        raise RuntimeError("ADS HB report PNG output requires Pillow") from exc
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _png_canvas(width: int, height: int, title: str):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("ADS HB report PNG output requires Pillow") from exc
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (width / 2, 28),
        title,
        font=_plot_font(30, bold=True),
        fill="#111827",
        anchor="ma",
    )
    return image, draw


def _draw_png_y_axis(
    draw: object,
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
    y_max: float,
    label: str,
) -> None:
    text_font = _plot_font(15)
    label_font = _plot_font(17, bold=True)
    for tick in range(6):
        value = tick * y_max / 5.0
        y = top + plot_height * (1.0 - tick / 5.0)
        draw.line((left, y, left + plot_width, y), fill="#e5e7eb", width=2)
        draw.text((left - 10, y), _format_number(value), font=text_font, fill="#475569", anchor="rm")
    draw.line((left, top, left, top + plot_height), fill="#64748b", width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="#64748b", width=2)
    draw.text((left - 72, top + plot_height / 2), label, font=label_font, fill="#334155", anchor="mm")


def _write_total_work_png(
    path: Path, summary_rows: Sequence[dict[str, object]]
) -> None:
    width, height = 1800, 760
    image, draw = _png_canvas(width, height, "Total solver work by model")
    models = [str(row["model"]) for row in summary_rows]
    panels = [
        ("Total Newton iterations", "total_newton_iterations"),
        ("Total Krylov iterations", "total_krylov_iterations"),
    ]
    panel_width = 830
    for panel_index, (title, key) in enumerate(panels):
        panel_x = 55 + panel_index * 875
        left, top, plot_width, plot_height = panel_x + 120, 155, 660, 440
        values = [_metric_number(row[key]) for row in summary_rows]
        y_max = _nice_axis_max(max(values, default=0.0))
        draw.text((panel_x + panel_width / 2, 95), title, font=_plot_font(23, bold=True), fill="#1f2937", anchor="ma")
        _draw_png_y_axis(draw, left, top, plot_width, plot_height, y_max, "Iterations")
        group_width = plot_width / max(1, len(models))
        bar_width = min(100.0, group_width * 0.58)
        for index, (model, value) in enumerate(zip(models, values)):
            x = left + group_width * (index + 0.5) - bar_width / 2
            bar_height = plot_height * value / y_max
            y = top + plot_height - bar_height
            color = PLOT_COLORS[index % len(PLOT_COLORS)]
            draw.rounded_rectangle((x, y, x + bar_width, top + plot_height), radius=5, fill=color)
            draw.text((x + bar_width / 2, max(top + 14, y - 12)), _format_number(value), font=_plot_font(15, bold=True), fill="#111827", anchor="ms")
            draw.text((x + bar_width / 2, top + plot_height + 30), _short_label(model), font=_plot_font(15), fill="#334155", anchor="ma")
    draw.text((width / 2, 725), "Totals include every detected HB solve in each Gain Compression log.", font=_plot_font(17), fill="#4b5563", anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)


def _write_runtime_png(
    path: Path, summary_rows: Sequence[dict[str, object]]
) -> None:
    width, height = 2200, 760
    image, draw = _png_canvas(width, height, "ADS Resource usage timing")
    available = any(
        row["wall_clock_seconds"] != ""
        or row["simulation_stopwatch_seconds"] != ""
        for row in summary_rows
    )
    if not available:
        draw.text((width / 2, 325), "No ADS stopwatch timing was found in the supplied logs.", font=_plot_font(28, bold=True), fill="#9a3412", anchor="mm")
        draw.text((width / 2, 380), "Enable ADS Resource usage output or pass --wall-clock-seconds.", font=_plot_font(22), fill="#4b5563", anchor="mm")
        image.save(path, format="PNG", dpi=(144, 144), optimize=True)
        return
    models = [str(row["model"]) for row in summary_rows]
    panels = [
        ("Total stopwatch time", "wall_clock_seconds"),
        ("Simulation stopwatch time", "simulation_stopwatch_seconds"),
        ("Total stopwatch / HB solve", "wall_clock_per_solve_seconds"),
    ]
    panel_width = 690
    for panel_index, (title, key) in enumerate(panels):
        panel_x = 20 + panel_index * 720
        left, top, plot_width, plot_height = panel_x + 120, 155, 530, 440
        raw_values = [row[key] for row in summary_rows]
        numeric_values = [_metric_number(value) for value in raw_values if value != ""]
        y_max = _nice_axis_max(max(numeric_values, default=0.0))
        draw.text((panel_x + panel_width / 2, 95), title, font=_plot_font(22, bold=True), fill="#1f2937", anchor="ma")
        _draw_png_y_axis(draw, left, top, plot_width, plot_height, y_max, "Seconds")
        group_width = plot_width / max(1, len(models))
        bar_width = min(90.0, group_width * 0.58)
        for index, (model, raw_value) in enumerate(zip(models, raw_values)):
            x = left + group_width * (index + 0.5) - bar_width / 2
            color = PLOT_COLORS[index % len(PLOT_COLORS)]
            if raw_value != "":
                value = _metric_number(raw_value)
                bar_height = plot_height * value / y_max
                y = top + plot_height - bar_height
                draw.rounded_rectangle((x, y, x + bar_width, top + plot_height), radius=5, fill=color)
                draw.text((x + bar_width / 2, max(top + 14, y - 12)), _format_duration(value), font=_plot_font(15, bold=True), fill="#111827", anchor="ms")
            else:
                draw.text((x + bar_width / 2, top + plot_height - 10), "n/a", font=_plot_font(16, bold=True), fill="#9a3412", anchor="ms")
            draw.text((x + bar_width / 2, top + plot_height + 30), _short_label(model), font=_plot_font(15), fill="#334155", anchor="ma")
    draw.text((width / 2, 725), "Per-solve time is Total stopwatch time divided by detected HB solves.", font=_plot_font(17), fill="#4b5563", anchor="ma")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)


def _write_krylov_statistics_png(
    path: Path, summary_rows: Sequence[dict[str, object]]
) -> None:
    width, height = 1800, 800
    image, draw = _png_canvas(width, height, "Krylov work per detected HB solve")
    metrics = [
        ("Mean", "mean_krylov_per_solve"),
        ("Median", "median_krylov_per_solve"),
        ("95th percentile", "p95_krylov_per_solve"),
        ("Maximum", "max_krylov_per_solve"),
    ]
    left, top, plot_width, plot_height = 150, 145, 1570, 470
    values = [_metric_number(row[key]) for _, key in metrics for row in summary_rows]
    y_max = _nice_axis_max(max(values, default=0.0))
    _draw_png_y_axis(draw, left, top, plot_width, plot_height, y_max, "Krylov iterations / solve")
    group_width = plot_width / len(metrics)
    model_count = max(1, len(summary_rows))
    bar_width = min(75.0, group_width * 0.72 / model_count)
    for metric_index, (metric_label, key) in enumerate(metrics):
        center = left + group_width * (metric_index + 0.5)
        total_width = bar_width * model_count
        for model_index, row in enumerate(summary_rows):
            value = _metric_number(row[key])
            x = center - total_width / 2 + model_index * bar_width
            y = top + plot_height - plot_height * value / y_max
            color = PLOT_COLORS[model_index % len(PLOT_COLORS)]
            draw.rectangle((x, y, x + bar_width - 4, top + plot_height), fill=color)
            if len(summary_rows) <= 4:
                draw.text((x + (bar_width - 4) / 2, max(top + 12, y - 10)), _format_number(value), font=_plot_font(14), fill="#111827", anchor="ms")
        draw.text((center, top + plot_height + 34), metric_label, font=_plot_font(17), fill="#334155", anchor="ma")
    legend_x = 190
    for index, row in enumerate(summary_rows):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        draw.rectangle((legend_x, 720, legend_x + 20, 740), fill=color)
        draw.text((legend_x + 30, 730), _short_label(row["model"]), font=_plot_font(16), fill="#334155", anchor="lm")
        legend_x += 300
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)


def _write_krylov_by_solve_png(path: Path, results: Sequence[ParseResult]) -> None:
    width, height = 1800, 820
    image, draw = _png_canvas(width, height, "Krylov iterations by HB solve sequence")
    left, top, plot_width, plot_height = 150, 145, 1570, 500
    all_values = [
        sum(row.krylov_iterations for row in solve.newton)
        for result in results
        for solve in result.solves
    ]
    max_solves = max((len(result.solves) for result in results), default=1)
    y_max = _nice_axis_max(max(all_values, default=0.0))
    _draw_png_y_axis(draw, left, top, plot_width, plot_height, y_max, "Krylov iterations")
    tick_step = max(1, math.ceil(max_solves / 10))
    for solve_index in range(1, max_solves + 1, tick_step):
        x = left + plot_width / 2 if max_solves == 1 else left + plot_width * (solve_index - 1) / (max_solves - 1)
        draw.line((x, top, x, top + plot_height), fill="#f3f4f6", width=1)
        draw.text((x, top + plot_height + 25), str(solve_index), font=_plot_font(14), fill="#475569", anchor="ma")
    for model_index, result in enumerate(results):
        color = PLOT_COLORS[model_index % len(PLOT_COLORS)]
        points = []
        for solve in result.solves:
            value = sum(row.krylov_iterations for row in solve.newton)
            x = left + plot_width / 2 if max_solves == 1 else left + plot_width * (solve.solve_index - 1) / (max_solves - 1)
            y = top + plot_height - plot_height * value / y_max
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="white", width=2)
    draw.text((left + plot_width / 2, top + plot_height + 65), "Detected HB solve sequence", font=_plot_font(18, bold=True), fill="#334155", anchor="ma")
    legend_x = 190
    for index, result in enumerate(results):
        color = PLOT_COLORS[index % len(PLOT_COLORS)]
        draw.line((legend_x, 755, legend_x + 35, 755), fill=color, width=5)
        draw.ellipse((legend_x + 12, 748, legend_x + 26, 762), fill=color)
        draw.text((legend_x + 45, 755), _short_label(result.model), font=_plot_font(16), fill="#334155", anchor="lm")
        legend_x += 320
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_markdown_escape(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _percent_change(value: object, reference: object) -> str:
    current = _metric_number(value)
    baseline = _metric_number(reference)
    if baseline == 0.0:
        return "—" if current == 0.0 else "n/a"
    change = 100.0 * (current / baseline - 1.0)
    return f"{change:+.1f}%"


def _format_frequency(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 1.0e9:
        return f"{value / 1.0e9:.6g} GHz"
    if abs(value) >= 1.0e6:
        return f"{value / 1.0e6:.6g} MHz"
    if abs(value) >= 1.0e3:
        return f"{value / 1.0e3:.6g} kHz"
    return f"{value:.6g} Hz"


def _frequency_summary(results: Sequence[ParseResult]) -> list[list[object]]:
    groups: dict[tuple[str, float], list[SolveRecord]] = {}
    model_order = {result.model: index for index, result in enumerate(results)}
    for result in results:
        for solve in result.solves:
            if solve.frequency_hz is not None:
                groups.setdefault((result.model, solve.frequency_hz), []).append(solve)
    rows: list[list[object]] = []
    for (model, frequency), solves in sorted(
        groups.items(), key=lambda item: (model_order[item[0][0]], item[0][1])
    ):
        newton_total = sum(len(solve.newton) for solve in solves)
        krylov_total = sum(
            row.krylov_iterations for solve in solves for row in solve.newton
        )
        rows.append(
            [
                model,
                _format_frequency(frequency),
                len(solves),
                newton_total,
                krylov_total,
                _format_number(krylov_total / len(solves)),
            ]
        )
    return rows


def _write_markdown_report(
    path: Path,
    results: Sequence[ParseResult],
    summary_rows: Sequence[dict[str, object]],
    plot_hrefs: dict[str, str] | None = None,
) -> None:
    plot_hrefs = plot_hrefs or {}

    def plot_href(filename: str) -> str:
        return plot_hrefs.get(filename, filename)

    baseline = summary_rows[0]
    summary_table = [
        [
            row["model"],
            row["solve_count"],
            _format_duration(row["wall_clock_seconds"]),
            _format_duration(row["simulation_stopwatch_seconds"]),
            _format_duration(row["wall_clock_per_solve_seconds"]),
            _format_duration(row["cpu_time_seconds"]),
            row["total_newton_iterations"],
            row["total_krylov_iterations"],
            _format_number(row["mean_krylov_per_solve"]),
            _format_number(row["median_krylov_per_solve"]),
            _format_number(row["p95_krylov_per_solve"]),
            _format_number(row["max_krylov_per_solve"]),
            f"{row['failure_solve_count']} / {row['retry_solve_count']}",
        ]
        for row in summary_rows
    ]
    relative_table = [
        [
            row["model"],
            _percent_change(
                row["wall_clock_seconds"], baseline["wall_clock_seconds"]
            ),
            _percent_change(
                row["simulation_stopwatch_seconds"],
                baseline["simulation_stopwatch_seconds"],
            ),
            _percent_change(
                row["wall_clock_per_solve_seconds"],
                baseline["wall_clock_per_solve_seconds"],
            ),
            _percent_change(
                row["total_newton_iterations"], baseline["total_newton_iterations"]
            ),
            _percent_change(
                row["total_krylov_iterations"], baseline["total_krylov_iterations"]
            ),
            _percent_change(
                row["mean_krylov_per_solve"], baseline["mean_krylov_per_solve"]
            ),
            int(row["solve_count"]) - int(baseline["solve_count"]),
        ]
        for row in summary_rows
    ]
    worst_solves = sorted(
        (
            (
                sum(row.krylov_iterations for row in solve.newton),
                result,
                solve,
            )
            for result in results
            for solve in result.solves
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:12]
    worst_table = [
        [
            result.model,
            solve.solve_index,
            _format_frequency(solve.frequency_hz),
            (
                f"{solve.input_power_dbm:.4g} dBm"
                if solve.input_power_dbm is not None
                else "—"
            ),
            len(solve.newton),
            krylov_total,
            f"{solve.newton[-1].krylov_residual:.3e}",
            "yes" if solve.warnings else "no",
        ]
        for krylov_total, result, solve in worst_solves
    ]
    source_table = [
        [
            result.model,
            result.source_file,
            len(result.solves),
            sum(solve.frequency_hz is not None for solve in result.solves),
            sum(solve.input_power_dbm is not None for solve in result.solves),
        ]
        for result in results
    ]
    timing_source_table = [
        [
            result.model,
            _format_duration(result.wall_clock_seconds),
            result.wall_clock_source or "not found",
            _format_duration(result.simulation_stopwatch_seconds),
            result.simulation_stopwatch_source or "not found",
            _format_duration(result.cpu_time_seconds),
            result.cpu_time_source or "not found",
        ]
        for result in results
    ]
    missing_wall_models = [
        result.model for result in results if result.wall_clock_seconds is None
    ]
    frequency_rows = _frequency_summary(results)
    message_rows = [
        [
            result.model,
            solve.solve_index,
            " | ".join([*solve.warnings, *solve.retry_messages]),
        ]
        for result in results
        for solve in result.solves
        if solve.warnings or solve.retry_messages
    ]
    for result in results:
        for message in result.unmatched_failure_messages:
            message_rows.append([result.model, "unassigned", message])
        for message in result.unmatched_retry_messages:
            message_rows.append([result.model, "unassigned", message])

    lines = [
        "# ADS HB Solver Comparison",
        "",
        (
            "This report compares the Newton/Krylov work parsed from ADS Gain "
            "Compression or harmonic-balance Status/Summary logs. Lower solver "
            "work is generally favorable, but elapsed simulation time remains the "
            "final performance measure."
        ),
        "",
        "## Summary",
        "",
        _markdown_table(
            [
                "Model",
                "HB solves",
                "Total stopwatch",
                "Simulation stopwatch",
                "Total/solve",
                "Total CPU",
                "Newton total",
                "Krylov total",
                "Krylov/solve mean",
                "Median",
                "P95",
                "Maximum",
                "Failures / retries",
            ],
            summary_table,
        ),
        "",
        f"The first model, **{_markdown_escape(baseline['model'])}**, is the comparison reference.",
        "",
        _markdown_table(
            [
                "Model",
                "Δ total stopwatch",
                "Δ simulation stopwatch",
                "Δ total/solve",
                "Δ Newton total",
                "Δ Krylov total",
                "Δ mean Krylov/solve",
                "Δ HB solves",
            ],
            relative_table,
        ),
        "",
        "## Runtime",
        "",
        f"![ADS Resource usage timing by model]({plot_href('runtime_comparison.png')})",
        "",
        _markdown_table(
            [
                "Model",
                "Total stopwatch",
                "Total source",
                "Simulation stopwatch",
                "Simulation source",
                "Total CPU",
                "CPU source",
            ],
            timing_source_table,
        ),
        "",
        *(
            [
                (
                    "> **Timing unavailable for "
                    + ", ".join(_markdown_escape(model) for model in missing_wall_models)
                    + ".** The supplied log did not contain a recognized `Total "
                    "stopwatch time`. Enable ADS Resource usage output or pass "
                    "`--wall-clock-seconds` "
                    "for each log."
                ),
                "",
            ]
            if missing_wall_models
            else []
        ),
        (
            "Total time per solve is derived by dividing `Total stopwatch time` "
            "by the number of detected HB solves; it is not a separately "
            "measured point time."
        ),
        "",
        "## Total solver work",
        "",
        f"![Total Newton and Krylov iterations by model]({plot_href('solver_work_totals.png')})",
        "",
        "Totals are affected by both work per solve and the number of adaptive Gain Compression solves.",
        "",
        "## Normalized Krylov work",
        "",
        f"![Krylov work statistics per HB solve]({plot_href('krylov_per_solve_statistics.png')})",
        "",
        "These statistics normalize for different numbers of adaptive HB solves.",
        "",
        "## Solve sequence",
        "",
        f"![Krylov iterations by detected HB solve]({plot_href('krylov_by_solve.png')})",
        "",
        (
            "Solve indices represent execution order. They are directly comparable "
            "only when the models followed the same frequency and power-point sequence."
        ),
        "",
    ]
    if frequency_rows:
        lines.extend(
            [
                "## Results by frequency",
                "",
                _markdown_table(
                    [
                        "Model",
                        "Frequency",
                        "HB solves",
                        "Newton total",
                        "Krylov total",
                        "Mean Krylov/solve",
                    ],
                    frequency_rows,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Highest-work solves",
            "",
            _markdown_table(
                [
                    "Model",
                    "Solve",
                    "Frequency",
                    "Input power",
                    "Newton",
                    "Krylov",
                    "Final Krylov residual",
                    "Failure",
                ],
                worst_table,
            ),
            "",
            "## Source coverage",
            "",
            _markdown_table(
                [
                    "Model",
                    "Source log",
                    "HB solves",
                    "Frequency-labelled",
                    "Power-labelled",
                ],
                source_table,
            ),
            "",
        ]
    )
    if message_rows:
        lines.extend(
            [
                "## Solver messages",
                "",
                _markdown_table(["Model", "Solve", "Message"], message_rows),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation notes",
            "",
            "- `Krylov total` is the sum of the linear-solver iteration count on every parsed Newton summary row.",
            "- `HB solves` is detected from frequency/power changes or a reset of the Newton counter.",
            "- Gain Compression chooses power points adaptively, so compare both totals and per-solve statistics.",
            "- Use identical StatusLevel, initial-guess policy, solver settings, circuit, and sweep configuration.",
            "- ADS `Total stopwatch time` is the primary end-to-end wall-clock comparison; `Simulation stopwatch time` isolates the simulation portion of that total.",
            "- Timing is parsed from Resource usage or another recognized total-runtime line, or supplied explicitly; it is never inferred from iteration count.",
            "- `Total CPU time` can exceed stopwatch time when ADS uses multiple cores.",
            "- Compare cold and warm stopwatch-time runs separately.",
            "",
            "Machine-readable details are available in `ads_hb_solver_points.csv`, "
            "`ads_hb_solver_summary.csv`, and `ads_hb_solver_summary.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_report_artifacts(
    out_dir: Path,
    results: Sequence[ParseResult],
    summary_rows: Sequence[dict[str, object]],
) -> tuple[list[str], list[str]]:
    artifact_names = [
        "ads_hb_solver_report.md",
        "runtime_comparison.png",
        "solver_work_totals.png",
        "krylov_per_solve_statistics.png",
        "krylov_by_solve.png",
    ]
    _write_runtime_png(out_dir / artifact_names[1], summary_rows)
    _write_total_work_png(out_dir / artifact_names[2], summary_rows)
    _write_krylov_statistics_png(out_dir / artifact_names[3], summary_rows)
    _write_krylov_by_solve_png(out_dir / artifact_names[4], results)
    plot_hrefs: dict[str, str] = {}
    for name in artifact_names[1:]:
        source_path = out_dir / name
        content = source_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()[:12]
        embedded_name = f"{source_path.stem}.{digest}{source_path.suffix}"
        embedded_path = out_dir / embedded_name
        if not embedded_path.is_file() or embedded_path.read_bytes() != content:
            embedded_path.write_bytes(content)
        plot_hrefs[name] = embedded_name
    _write_markdown_report(
        out_dir / artifact_names[0],
        results,
        summary_rows,
        plot_hrefs,
    )
    return artifact_names, list(plot_hrefs.values())


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
        "wall_clock_seconds",
        "simulation_stopwatch_seconds",
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
        prog=os.environ.get("ADS_SURROGATE_CLI_PROG"),
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
    parser.add_argument(
        "--wall-clock-seconds",
        "--elapsed-seconds",
        dest="wall_clock_seconds",
        type=float,
        nargs="+",
        metavar="SECONDS",
        help=(
            "Optional measured wall-clock seconds in the same order as LOG. "
            "Overrides timing parsed from each log."
        ),
    )
    parser.add_argument(
        "--cpu-time-seconds",
        type=float,
        nargs="+",
        metavar="SECONDS",
        help=(
            "Optional CPU seconds in the same order as LOG. Overrides CPU timing "
            "parsed from each log."
        ),
    )
    add_options_json_argument(parser, recursive=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_options_json(
        parser,
        argv,
        workflow="hb-report",
        command="hb-report",
    )
    if args.labels is not None and len(args.labels) != len(args.logs):
        raise SystemExit("--labels must provide exactly one name for each LOG")
    if args.logs.count("-") > 1:
        raise SystemExit("Standard input (-) can only be used once")

    labels = args.labels or [
        "stdin" if value == "-" else Path(value).stem for value in args.logs
    ]
    if len(set(labels)) != len(labels):
        raise SystemExit("Model labels must be unique")
    for option, values in (
        ("--wall-clock-seconds", args.wall_clock_seconds),
        ("--cpu-time-seconds", args.cpu_time_seconds),
    ):
        if values is not None and len(values) != len(args.logs):
            raise SystemExit(f"{option} must provide exactly one value for each LOG")
        if values is not None and any(value < 0.0 for value in values):
            raise SystemExit(f"{option} values must be non-negative")
    results: list[ParseResult] = []
    for log_index, (path_text, label) in enumerate(zip(args.logs, labels)):
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
        if args.wall_clock_seconds is not None:
            result.wall_clock_seconds = args.wall_clock_seconds[log_index]
            result.wall_clock_source = "CLI --wall-clock-seconds"
        if args.cpu_time_seconds is not None:
            result.cpu_time_seconds = args.cpu_time_seconds[log_index]
            result.cpu_time_source = "CLI --cpu-time-seconds"
        results.append(result)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    point_rows = [
        solve.csv_row() for result in results for solve in result.solves
    ]
    summary_rows = [summarize_result(result) for result in results]
    _write_csv(out_dir / "ads_hb_solver_points.csv", point_rows)
    _write_csv(out_dir / "ads_hb_solver_summary.csv", summary_rows)
    report_artifacts, embedded_plot_artifacts = _write_report_artifacts(
        out_dir, results, summary_rows
    )
    (out_dir / "ads_hb_solver_summary.json").write_text(
        json.dumps(
            {
                "summaries": summary_rows,
                "report_artifacts": report_artifacts,
                "embedded_plot_artifacts": embedded_plot_artifacts,
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
    print(f"Markdown:     {out_dir / 'ads_hb_solver_report.md'}")
    print(
        "Plots:        "
        + ", ".join(str(out_dir / name) for name in report_artifacts[1:])
    )
    return finalize_options_json_update(args, 0)


if __name__ == "__main__":
    raise SystemExit(main())
