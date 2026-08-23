#!/usr/bin/env python3
"""Audit raw MDIF training and verification data before surrogate fitting.

The audit is model-independent.  It checks whether the supplied S matrices are
passive and structurally suitable for DNN, KBNN, or Neuro-TF fitting, then
writes detailed machine-readable and human-readable diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from cli_options import (
    add_options_json_argument,
    finalize_options_json_update,
    parse_args_with_options_json,
)
from generate_points import (
    UNIT_SCALES,
    parameter_specs_from_geometry_metadata,
    split_value_token,
)
from surrogate_common import (
    MDIFBlock,
    infer_nports,
    infer_parameter_names,
    normalize_name,
    normalized_mapping_value,
    parse_csv_set,
    parse_number,
    read_mdif,
    sparam_sort_key,
    split_blocks,
)


@dataclass
class SourceHeader:
    data_format: str = "RI"
    frequency_unit: str = "Hz"
    reference_impedance_ohm: float | None = None


@dataclass
class AuditRecord:
    record_id: int
    dataset_kind: str
    role: str
    source_path: Path
    block: MDIFBlock
    header: SourceHeader
    resolved_parameter_values: dict[str, float | None] = field(default_factory=dict)


@dataclass
class GeometryDomain:
    source_paths: list[Path]
    parameters: dict[str, dict[str, object]]
    inferred: bool


AUDIT_VERDICT_COLORS = {
    "PASS": "\033[32m",
    "WARNING": "\033[33m",
    "FAIL": "\033[31m",
    "ERROR": "\033[31m",
}
ANSI_RESET = "\033[0m"


def terminal_color_enabled(
    stream: object,
    color_mode: str = "auto",
) -> bool:
    """Return whether ANSI status colors should be emitted to one stream."""

    if "NO_COLOR" in os.environ or color_mode == "never":
        return False
    if color_mode == "always":
        return True
    return bool(
        color_mode == "auto"
        and hasattr(stream, "isatty")
        and stream.isatty()  # type: ignore[attr-defined]
        and os.environ.get("TERM", "").lower() != "dumb"
    )


def colorize_audit_text(
    text: str,
    severity: str,
    stream: object,
    color_mode: str = "auto",
) -> str:
    if not terminal_color_enabled(stream, color_mode):
        return text
    color = AUDIT_VERDICT_COLORS.get(severity)
    return f"{color}{text}{ANSI_RESET}" if color else text


def format_audit_verdict_line(
    verdict: str,
    stream: object | None = None,
    color_mode: str = "auto",
) -> str:
    """Return the verdict line using the requested ANSI color policy."""

    line = f"dataset audit: {verdict}"
    output = sys.stdout if stream is None else stream
    return colorize_audit_text(line, verdict, output, color_mode)


def issue(
    severity: str,
    code: str,
    message: str,
    *,
    record: AuditRecord | None = None,
    frequency_hz: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "severity": severity,
        "code": code,
        "dataset": record.dataset_kind if record else "",
        "role": record.role if record else "",
        "source_file": str(record.source_path) if record else "",
        "block": record.block.source_index + 1 if record else "",
        "source_index": record.block.source_index if record else "",
        "frequency_hz": frequency_hz if frequency_hz is not None else "",
        "message": message,
    }
    return row


def is_geometry_metadata_json(path: Path) -> bool:
    """Return whether a JSON file has the generated-geometry metadata shape."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and isinstance(payload.get("geometry_file"), str)
        and isinstance(payload.get("parameters"), list)
        and payload["parameters"]
    )


def resolve_geometry_domain(
    explicit_paths: Sequence[str],
    mdif_paths: Sequence[Path],
) -> GeometryDomain | None:
    """Load and merge declared parameter bounds from geometry-generation JSON."""

    inferred = not explicit_paths
    if explicit_paths:
        source_paths = [Path(path).expanduser() for path in explicit_paths]
    else:
        candidates = [path.with_suffix(".json") for path in mdif_paths]
        source_paths = [
            path
            for path in dict.fromkeys(candidates)
            if path.is_file() and is_geometry_metadata_json(path)
        ]
    source_paths = list(dict.fromkeys(source_paths))
    if not source_paths:
        return None

    merged: dict[str, dict[str, object]] = {}
    for source_path in source_paths:
        specs = parameter_specs_from_geometry_metadata(source_path)
        for spec in specs:
            key = normalize_name(spec.name).lower()
            current = merged.get(key)
            if current is None:
                merged[key] = {
                    "name": spec.name,
                    "lower_base_units": float(spec.lower),
                    "upper_base_units": float(spec.upper),
                    "unit": spec.unit,
                    "scale": spec.scale,
                    "source_files": [str(source_path)],
                }
                continue
            if current["scale"] != spec.scale:
                raise ValueError(
                    f"Geometry metadata disagrees on the scale for {spec.name!r}: "
                    f"{current['scale']!r} versus {spec.scale!r} in {source_path}"
                )
            current["lower_base_units"] = min(
                float(current["lower_base_units"]),
                float(spec.lower),
            )
            current["upper_base_units"] = max(
                float(current["upper_base_units"]),
                float(spec.upper),
            )
            sources = list(current["source_files"])
            if str(source_path) not in sources:
                sources.append(str(source_path))
            current["source_files"] = sources
    return GeometryDomain(
        source_paths=source_paths,
        parameters=merged,
        inferred=inferred,
    )


def scan_mdif_headers(path: Path) -> list[SourceHeader]:
    """Collect per-block numeric format, frequency unit, and reference Z0."""

    headers: list[SourceHeader] = []
    current: SourceHeader | None = None
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("BEGIN"):
            current = SourceHeader()
            continue
        if upper.startswith("END") and current is not None:
            headers.append(current)
            current = None
            continue
        if current is None:
            continue
        cleaned = line[1:].strip() if line.startswith("#") else line
        tokens = cleaned.split()
        for token in tokens:
            lowered = token.lower()
            if lowered in {"hz", "khz", "mhz", "ghz", "thz"}:
                current.frequency_unit = token
            token_upper = token.upper()
            if token_upper in {"RI", "MA", "DB"}:
                current.data_format = token_upper
        upper_tokens = [token.upper() for token in tokens]
        if "R" in upper_tokens:
            index = upper_tokens.index("R")
            if index + 1 < len(tokens):
                parsed = parse_number(tokens[index + 1])
                if parsed is not None and math.isfinite(parsed):
                    current.reference_impedance_ohm = float(parsed)
    return headers


def classify_combined_blocks(
    blocks: list[MDIFBlock],
    *,
    split_var: str,
    train_values: set[str],
    verify_values: set[str],
    holdout_fraction: float,
    seed: int,
) -> tuple[dict[int, str], bool]:
    split = split_blocks(
        blocks,
        split_var=split_var,
        train_values=train_values,
        verify_values=verify_values,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    roles = {id(block): "train" for block in split.train}
    roles.update({id(block): "verification" for block in split.verify})
    explicit_train = any(
        (normalized_mapping_value(block.params, split_var) or "").lower()
        in train_values
        for block in blocks
    )
    for block in blocks:
        roles.setdefault(id(block), "unassigned" if explicit_train else "unused")
    return roles, not explicit_train


def load_dataset_records(
    dataset_kind: str,
    mdif_path: Path,
    verification_path: Path | None,
    *,
    split_var: str,
    train_values: set[str],
    verify_values: set[str],
    holdout_fraction: float,
    seed: int,
    starting_record_id: int,
) -> tuple[list[AuditRecord], list[dict[str, object]], int]:
    problems: list[dict[str, object]] = []
    main_blocks = read_mdif(mdif_path)
    main_headers = scan_mdif_headers(mdif_path)
    if len(main_headers) != len(main_blocks):
        main_headers = [SourceHeader() for _ in main_blocks]

    records: list[AuditRecord] = []
    if verification_path is not None:
        roles = {id(block): "train" for block in main_blocks}
        used_random_holdout = False
    else:
        roles, used_random_holdout = classify_combined_blocks(
            main_blocks,
            split_var=split_var,
            train_values=train_values,
            verify_values=verify_values,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
    if used_random_holdout:
        problems.append(
            issue(
                "WARNING",
                "RANDOM_HOLDOUT_USED",
                f"{dataset_kind} MDIF contains no recognized training labels; the audit "
                "reproduced the fitter's random holdout split.",
            )
        )
    for block, header in zip(main_blocks, main_headers):
        records.append(
            AuditRecord(
                record_id=starting_record_id,
                dataset_kind=dataset_kind,
                role=roles[id(block)],
                source_path=mdif_path,
                block=block,
                header=header,
            )
        )
        starting_record_id += 1

    if verification_path is not None:
        verify_blocks = read_mdif(verification_path)
        verify_headers = scan_mdif_headers(verification_path)
        if len(verify_headers) != len(verify_blocks):
            verify_headers = [SourceHeader() for _ in verify_blocks]
        for block, header in zip(verify_blocks, verify_headers):
            records.append(
                AuditRecord(
                    record_id=starting_record_id,
                    dataset_kind=dataset_kind,
                    role="verification",
                    source_path=verification_path,
                    block=block,
                    header=header,
                )
            )
            starting_record_id += 1
    return records, problems, starting_record_id


def _parameter_token_parts(raw: object) -> tuple[float, str] | None:
    text = str(raw).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    text = text.replace(",", "")
    try:
        value, unit = split_value_token(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return float(value), unit


def _domain_interpretation_score(
    values: Sequence[float],
    factor: float,
    lower: float,
    upper: float,
) -> tuple[int, float]:
    """Rank one bare-value interpretation by domain inclusion and distance."""

    span = max(upper - lower, abs(lower), abs(upper), 1.0e-300)
    inside = 0
    outside_distance = 0.0
    for raw_value in values:
        value = raw_value * factor
        if (
            lower <= value <= upper
            or math.isclose(value, lower, rel_tol=1.0e-10, abs_tol=1.0e-15)
            or math.isclose(value, upper, rel_tol=1.0e-10, abs_tol=1.0e-15)
        ):
            inside += 1
        elif value < lower:
            outside_distance += (lower - value) / span
        else:
            outside_distance += (value - upper) / span
    return inside, -outside_distance


def resolve_record_parameter_values(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
    geometry_domain: GeometryDomain | None,
    bare_values: str,
) -> list[dict[str, object]]:
    """Resolve unitless MDIF parameters consistently for every source.

    Explicit unit suffixes always win. In ``auto`` mode, unitless values are
    evaluated both as SI/base-unit values and as values in the unit declared by
    the geometry JSON. The interpretation that places the source's parameter
    values inside or nearest the declared domain is selected consistently for
    every block from that source.
    """

    if bare_values == "parameter-units" and geometry_domain is None:
        raise ValueError(
            "--bare-values parameter-units requires --geometry-json (or valid "
            "same-stem geometry metadata) so declared parameter units are known"
        )

    decisions: list[dict[str, object]] = []
    source_paths = list(dict.fromkeys(record.source_path for record in records))
    for source_path in source_paths:
        source_records = [
            record for record in records if record.source_path == source_path
        ]
        for name in parameter_names:
            key = normalize_name(name).lower()
            domain_parameter = (
                geometry_domain.parameters.get(key)
                if geometry_domain is not None
                else None
            )
            declared_unit = (
                str(domain_parameter.get("unit") or "")
                if domain_parameter is not None
                else ""
            )
            unit_factor = UNIT_SCALES.get(declared_unit, 1.0)
            parsed_rows: list[tuple[AuditRecord, tuple[float, str] | None]] = []
            bare_numbers: list[float] = []
            explicit_unit_count = 0
            for record in source_records:
                raw = normalized_mapping_value(record.block.params, name)
                parsed = _parameter_token_parts(raw) if raw is not None else None
                parsed_rows.append((record, parsed))
                if parsed is None:
                    continue
                value, explicit_unit = parsed
                if explicit_unit:
                    explicit_unit_count += 1
                else:
                    bare_numbers.append(value)

            base_score: tuple[int, float] | None = None
            parameter_score: tuple[int, float] | None = None
            if domain_parameter is not None and bare_numbers:
                lower = float(domain_parameter["lower_base_units"])
                upper = float(domain_parameter["upper_base_units"])
                base_score = _domain_interpretation_score(
                    bare_numbers,
                    1.0,
                    lower,
                    upper,
                )
                parameter_score = _domain_interpretation_score(
                    bare_numbers,
                    unit_factor,
                    lower,
                    upper,
                )

            if not bare_numbers:
                selected_mode = "explicit-units-only"
            elif bare_values == "base-units":
                selected_mode = "base-units"
            elif bare_values == "parameter-units":
                selected_mode = "parameter-units"
            elif domain_parameter is None or not declared_unit:
                selected_mode = "base-units"
            elif parameter_score is not None and base_score is not None:
                # A tie deliberately favors the declared parameter unit. This
                # matches point generation and is the least surprising meaning
                # for a unitless value accompanied by a unit-bearing domain.
                selected_mode = (
                    "parameter-units"
                    if parameter_score >= base_score
                    else "base-units"
                )
            else:
                selected_mode = "base-units"

            bare_factor = unit_factor if selected_mode == "parameter-units" else 1.0
            for record, parsed in parsed_rows:
                if parsed is None:
                    record.resolved_parameter_values[key] = None
                    continue
                value, explicit_unit = parsed
                factor = UNIT_SCALES[explicit_unit] if explicit_unit else bare_factor
                record.resolved_parameter_values[key] = value * factor

            decisions.append(
                {
                    "source_file": str(source_path),
                    "parameter": name,
                    "requested_bare_values": bare_values,
                    "selected_interpretation": selected_mode,
                    "declared_unit": declared_unit,
                    "unitless_value_count": len(bare_numbers),
                    "explicit_unit_value_count": explicit_unit_count,
                    "base_units_inside_geometry": (
                        base_score[0] if base_score is not None else None
                    ),
                    "parameter_units_inside_geometry": (
                        parameter_score[0] if parameter_score is not None else None
                    ),
                }
            )
    return decisions


def record_parameter_values(
    record: AuditRecord,
    parameter_names: Sequence[str],
) -> np.ndarray | None:
    values: list[float] = []
    for name in parameter_names:
        key = normalize_name(name).lower()
        if key in record.resolved_parameter_values:
            resolved = record.resolved_parameter_values[key]
            if resolved is None or not math.isfinite(resolved):
                return None
            values.append(float(resolved))
            continue
        raw = normalized_mapping_value(record.block.params, name)
        if raw is None:
            return None
        parsed = parse_number(raw)
        if parsed is None or not math.isfinite(parsed):
            return None
        values.append(float(parsed))
    return np.asarray(values, dtype=float)


def matrix_at(
    block: MDIFBlock,
    labels: Sequence[str],
    nports: int,
    frequency_index: int,
) -> np.ndarray:
    matrix = np.empty((nports, nports), dtype=complex)
    for label in labels:
        row = int(label[1]) - 1
        col = int(label[2]) - 1
        matrix[row, col] = block.sparams[label][frequency_index]
    return matrix


def grids_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol_hz: float,
) -> bool:
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=rtol, atol=atol_hz, equal_nan=False)
    )


def assign_frequency_grids(
    records: Sequence[AuditRecord],
    *,
    rtol: float,
    atol_hz: float,
) -> tuple[dict[int, int], list[dict[str, object]]]:
    representatives: list[np.ndarray] = []
    assignments: dict[int, int] = {}
    counts: Counter[int] = Counter()
    roles: dict[int, Counter[str]] = {}
    kinds: dict[int, Counter[str]] = {}
    for record in records:
        frequencies = np.asarray(record.block.freq_hz, dtype=float)
        grid_index = next(
            (
                index
                for index, representative in enumerate(representatives)
                if grids_match(
                    frequencies,
                    representative,
                    rtol=rtol,
                    atol_hz=atol_hz,
                )
            ),
            None,
        )
        if grid_index is None:
            representatives.append(frequencies.copy())
            grid_index = len(representatives) - 1
        grid_id = grid_index + 1
        assignments[record.record_id] = grid_id
        counts[grid_id] += 1
        roles.setdefault(grid_id, Counter())[record.role] += 1
        kinds.setdefault(grid_id, Counter())[record.dataset_kind] += 1

    summaries: list[dict[str, object]] = []
    for grid_id, representative in enumerate(representatives, start=1):
        finite = representative[np.isfinite(representative)]
        summaries.append(
            {
                "grid_id": grid_id,
                "blocks": counts[grid_id],
                "rows": int(representative.size),
                "minimum_frequency_hz": float(np.min(finite)) if finite.size else None,
                "maximum_frequency_hz": float(np.max(finite)) if finite.size else None,
                "dc_rows": int(np.count_nonzero(representative == 0.0)),
                "roles": dict(roles[grid_id]),
                "datasets": dict(kinds[grid_id]),
            }
        )
    return assignments, summaries


def parameter_vectors_close(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> bool:
    return bool(np.all(np.isclose(left, right, rtol=rtol, atol=atol)))


def response_difference(
    left: AuditRecord,
    right: AuditRecord,
    labels: Sequence[str],
    *,
    frequency_rtol: float,
    frequency_atol_hz: float,
) -> dict[str, float] | None:
    if not grids_match(
        left.block.freq_hz,
        right.block.freq_hz,
        rtol=frequency_rtol,
        atol_hz=frequency_atol_hz,
    ):
        return None
    if any(label not in left.block.sparams or label not in right.block.sparams for label in labels):
        return None
    left_values = np.column_stack([left.block.sparams[label] for label in labels])
    right_values = np.column_stack([right.block.sparams[label] for label in labels])
    if left_values.shape != right_values.shape:
        return None
    difference = np.abs(left_values - right_values)
    if not np.all(np.isfinite(difference)):
        return None
    rmse = float(np.sqrt(np.mean(difference**2)))
    max_abs = float(np.max(difference))
    reference_rms = float(
        max(
            np.sqrt(np.mean(np.abs(left_values) ** 2)),
            np.sqrt(np.mean(np.abs(right_values) ** 2)),
            1.0e-15,
        )
    )
    return {
        "rmse": rmse,
        "max_abs": max_abs,
        "relative_rmse": rmse / reference_rms,
        "reference_rms": reference_rms,
    }


def duplicate_geometry_analysis(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    parameter_rtol: float,
    parameter_atol: float,
    frequency_rtol: float,
    frequency_atol_hz: float,
    response_rtol: float,
    response_atol: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    problems: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    candidates = [
        (record, record_parameter_values(record, parameter_names))
        for record in records
        if record.role in {"train", "verification"}
    ]
    candidates = [(record, values) for record, values in candidates if values is not None]
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if parameter_vectors_close(
                candidates[left][1],
                candidates[right][1],
                rtol=parameter_rtol,
                atol=parameter_atol,
            ):
                union(left, right)

    groups: dict[int, list[AuditRecord]] = {}
    for index, (record, _values) in enumerate(candidates):
        groups.setdefault(find(index), []).append(record)

    group_number = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group_number += 1
        roles = {record.role for record in group}
        if {"train", "verification"} <= roles:
            problems.append(
                issue(
                    "ERROR",
                    "TRAIN_VERIFICATION_OVERLAP",
                    "The same geometry occurs in both training and verification data; "
                    f"duplicate group {group_number} contains {len(group)} blocks.",
                    record=group[0],
                )
            )
        else:
            problems.append(
                issue(
                    "WARNING",
                    "DUPLICATE_GEOMETRY",
                    f"Duplicate geometry group {group_number} contains {len(group)} "
                    f"{next(iter(roles))} blocks.",
                    record=group[0],
                )
            )

        for right_index in range(1, len(group)):
            left = group[0]
            right = group[right_index]
            difference = response_difference(
                left,
                right,
                labels,
                frequency_rtol=frequency_rtol,
                frequency_atol_hz=frequency_atol_hz,
            )
            row: dict[str, object] = {
                "group": group_number,
                "dataset": left.dataset_kind,
                "left_role": left.role,
                "left_source_file": str(left.source_path),
                "left_block": left.block.source_index + 1,
                "right_role": right.role,
                "right_source_file": str(right.source_path),
                "right_block": right.block.source_index + 1,
                **{
                    name: normalized_mapping_value(left.block.params, name) or ""
                    for name in parameter_names
                },
            }
            if difference is None:
                row.update({"frequency_grid_match": False, "rmse": "", "max_abs": "", "relative_rmse": ""})
                problems.append(
                    issue(
                        "ERROR",
                        "DUPLICATE_GEOMETRY_GRID_MISMATCH",
                        f"Duplicate geometry group {group_number} has incompatible "
                        "frequency grids or S-parameter columns.",
                        record=right,
                    )
                )
            else:
                row.update({"frequency_grid_match": True, **difference})
                scale = float(difference["reference_rms"])
                if float(difference["max_abs"]) > response_atol + response_rtol * scale:
                    problems.append(
                        issue(
                            "ERROR",
                            "DUPLICATE_GEOMETRY_RESPONSE_CONFLICT",
                            f"Duplicate geometry group {group_number} has conflicting S data: "
                            f"maximum absolute difference {difference['max_abs']:.6g}.",
                            record=right,
                        )
                    )
            rows.append(row)
    return rows, problems


def nearest_neighbor_analysis(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    frequency_rtol: float,
    frequency_atol_hz: float,
    outlier_factor: float,
    minimum_relative_jump: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    problems: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for dataset_kind in sorted({record.dataset_kind for record in records}):
        active = [
            record
            for record in records
            if record.dataset_kind == dataset_kind and record.role == "train"
        ]
        vectors = [record_parameter_values(record, parameter_names) for record in active]
        valid = [(record, vector) for record, vector in zip(active, vectors) if vector is not None]
        if len(valid) < 2:
            continue
        matrix = np.vstack([vector for _record, vector in valid])
        lower = np.min(matrix, axis=0)
        span = np.max(matrix, axis=0) - lower
        span[span <= 0.0] = 1.0
        normalized = (matrix - lower) / span
        used_pairs: set[tuple[int, int]] = set()
        for index, (record, _vector) in enumerate(valid):
            distances = np.linalg.norm(normalized - normalized[index], axis=1)
            distances[index] = np.inf
            neighbor_index = int(np.argmin(distances))
            pair = tuple(sorted((index, neighbor_index)))
            if pair in used_pairs or not math.isfinite(float(distances[neighbor_index])):
                continue
            used_pairs.add(pair)
            neighbor = valid[neighbor_index][0]
            difference = response_difference(
                record,
                neighbor,
                labels,
                frequency_rtol=frequency_rtol,
                frequency_atol_hz=frequency_atol_hz,
            )
            if difference is None:
                continue
            rows.append(
                {
                    "dataset": dataset_kind,
                    "left_source_file": str(record.source_path),
                    "left_block": record.block.source_index + 1,
                    "right_source_file": str(neighbor.source_path),
                    "right_block": neighbor.block.source_index + 1,
                    "normalized_geometry_distance": float(distances[neighbor_index]),
                    **difference,
                }
            )

    for dataset_kind in sorted({str(row["dataset"]) for row in rows}):
        selected = [row for row in rows if row["dataset"] == dataset_kind]
        relative = np.asarray([float(row["relative_rmse"]) for row in selected], dtype=float)
        median = float(np.median(relative)) if relative.size else 0.0
        threshold = max(minimum_relative_jump, outlier_factor * max(median, 1.0e-15))
        for row in selected:
            row["outlier_threshold"] = threshold
            row["response_jump_outlier"] = bool(float(row["relative_rmse"]) > threshold)
            if row["response_jump_outlier"]:
                problems.append(
                    issue(
                        "WARNING",
                        "NEIGHBOR_RESPONSE_OUTLIER",
                        f"{dataset_kind} blocks {row['left_block']} and {row['right_block']} "
                        f"are nearest geometry neighbors but have relative S-response "
                        f"RMSE {float(row['relative_rmse']):.6g} versus an outlier threshold "
                        f"of {threshold:.6g}. Inspect for a discontinuity or simulation failure.",
                    )
                )
    rows.sort(key=lambda row: float(row["relative_rmse"]), reverse=True)
    return rows, problems


def write_csv(path: Path, rows: Sequence[dict[str, object]], preferred_fields: Sequence[str] = ()) -> None:
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["message"]
        rows = [{"message": "No rows"}]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def png_passivity_plot(path: Path, block_rows: Sequence[dict[str, object]], limit: float) -> bool:
    """Write the raw-data passivity summary as a document-scale PNG."""

    plotted = [row for row in block_rows if isinstance(row.get("max_singular_value"), (int, float))]
    if not plotted:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Dataset audit PNG output requires Pillow") from exc

    def load_font(size: int, bold: bool = False):
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

    width, height = 1800, 760
    left, right, top, bottom = 150, 50, 100, 115
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row["max_singular_value"]) for row in plotted]
    y_min = min(0.95, min(values) * 0.98, limit * 0.98)
    y_max = max(1.01, max(values) * 1.02, limit * 1.02)
    if y_max <= y_min:
        y_max = y_min + 0.1

    def x_coord(index: int) -> float:
        return left + (index + 0.5) * plot_width / max(len(plotted), 1)

    def y_coord(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    colors = {
        ("fine", "train"): "#2563eb",
        ("fine", "verification"): "#f59e0b",
        ("coarse", "train"): "#059669",
        ("coarse", "verification"): "#a855f7",
    }
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(34, bold=True)
    label_font = load_font(22, bold=True)
    text_font = load_font(18)
    draw.text((width / 2, 35), "Raw MDIF passivity by block", font=title_font, fill="#111827", anchor="ma")
    draw.line((left, top, left, top + plot_height), fill="#444444", width=2)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill="#444444", width=2)
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5.0
        y = y_coord(value)
        draw.line((left, y, left + plot_width, y), fill="#e5e7eb", width=2)
        draw.text((left - 14, y), f"{value:.6g}", font=text_font, fill="#334155", anchor="rm")
    limit_y = y_coord(limit)
    for x in range(left, left + plot_width, 26):
        draw.line((x, limit_y, min(x + 15, left + plot_width), limit_y), fill="#dc2626", width=4)
    for index, row in enumerate(plotted):
        key = (str(row["dataset"]), str(row["role"]))
        color = colors.get(key, "#64748b")
        value = float(row["max_singular_value"])
        x = x_coord(index)
        y = y_coord(value)
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=1)
    draw.text((left + plot_width / 2, height - 48), "Block sequence (fine then coarse; training and verification)", font=label_font, fill="#1f2937", anchor="ma")
    y_label = Image.new("RGBA", (460, 45), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((230, 22), "Maximum singular value", font=label_font, fill="#1f2937", anchor="mm")
    y_label = y_label.rotate(90, expand=True)
    image.paste(y_label, (20, int(top + plot_height / 2 - y_label.height / 2)), y_label)
    draw.text((left + plot_width - 4, limit_y - 10), f"passivity limit = {limit:.9g}", font=text_font, fill="#dc2626", anchor="rs")

    legend_items = [
        ("Fine training", colors[("fine", "train")]),
        ("Fine verification", colors[("fine", "verification")]),
    ]
    present = {(str(row["dataset"]), str(row["role"])) for row in plotted}
    if any(key[0] == "coarse" for key in present):
        legend_items.extend(
            [
                ("Coarse training", colors[("coarse", "train")]),
                ("Coarse verification", colors[("coarse", "verification")]),
            ]
        )
    legend_x = left
    for label, color in legend_items:
        draw.ellipse((legend_x, 72, legend_x + 16, 88), fill=color)
        draw.text((legend_x + 25, 80), label, font=text_font, fill="#334155", anchor="lm")
        legend_x += 260
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)
    return True


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def format_number(value: object) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    return str(value)


ISSUE_RECOMMENDATIONS = {
    "RANDOM_HOLDOUT_USED": (
        "Add explicit train/verification labels through the split VAR, or pass a "
        "separate --verification-mdif. Otherwise confirm that the reproducible "
        "random holdout is intentional."
    ),
    "DUPLICATE_GEOMETRY": (
        "Review dataset_duplicates.csv and remove redundant blocks, or confirm "
        "that the repeated simulations are intentional and equivalent."
    ),
    "NEIGHBOR_RESPONSE_OUTLIER": (
        "Inspect the named block pair in dataset_neighbor_consistency.csv and the "
        "source MDIF. Confirm whether the jump is a real resonance or a failed "
        "simulation, port-order error, or geometry-metadata mismatch."
    ),
    "VERIFICATION_OUTSIDE_TRAIN_RANGE": (
        "Pass the original generated geometry metadata through --geometry-json so "
        "coverage uses the intended domain. If no geometry JSON exists, expand the "
        "training range or treat these metrics explicitly as extrapolation."
    ),
    "VERIFICATION_OUTSIDE_GEOMETRY_RANGE": (
        "Correct the verification geometry if it is invalid, or regenerate/extend "
        "the geometry domain and pass the updated JSON before treating this point "
        "as an in-range verification sample."
    ),
    "FINE_OUTSIDE_COARSE_TRAIN_RANGE": (
        "Add coarse-model training points that cover the full fine-data parameter "
        "range before fitting a residual or prior-input KBNN."
    ),
    "INCONSISTENT_FREQUENCY_GRIDS": (
        "Compare dataset_frequency_grids.csv with the intended sweep setup. Align "
        "the grids when the difference is accidental; otherwise confirm that each "
        "important frequency region still has adequate training coverage."
    ),
    "MISSING_REFERENCE_IMPEDANCE": (
        "Declare the intended R reference impedance in every MDIF ACDATA option "
        "line so conversions and comparisons do not depend on an implicit value."
    ),
    "NO_VERIFICATION_BLOCKS": (
        "Add blocks labeled as verification, or pass --verification-mdif, before "
        "using the audit or fitting metrics to judge generalization."
    ),
}


def issue_location(row: dict[str, object]) -> str:
    """Return a compact human-readable location for one audit issue."""

    parts: list[str] = []
    dataset = str(row.get("dataset") or "")
    role = str(row.get("role") or "")
    if dataset or role:
        parts.append("/".join(value for value in (dataset, role) if value))
    source_file = str(row.get("source_file") or "")
    if source_file:
        parts.append(Path(source_file).name)
    block = row.get("block")
    if block not in (None, ""):
        parts.append(f"block {block}")
    frequency = row.get("frequency_hz")
    if frequency not in (None, ""):
        parts.append(f"{format_number(frequency)} Hz")
    return ", ".join(parts)


def summarize_issue_reasons(
    problems: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Group issue rows into concise verdict reasons with corrective guidance."""

    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in problems:
        key = (str(row.get("severity") or ""), str(row.get("code") or "UNKNOWN"))
        grouped.setdefault(key, []).append(row)

    severity_order = {"ERROR": 0, "WARNING": 1}
    reasons: list[dict[str, object]] = []
    for (severity, code), rows in sorted(
        grouped.items(),
        key=lambda item: (severity_order.get(item[0][0], 2), item[0][1]),
    ):
        messages = list(
            dict.fromkeys(
                str(row.get("message") or "No explanation supplied.")
                for row in rows
            )
        )
        locations = list(
            dict.fromkeys(
                location
                for row in rows
                if (location := issue_location(row))
            )
        )
        reasons.append(
            {
                "severity": severity,
                "code": code,
                "count": len(rows),
                "reason": messages[0],
                "representative_messages": messages[:5],
                "affected_examples": locations[:5],
                "additional_message_count": max(0, len(messages) - 5),
                "additional_location_count": max(0, len(locations) - 5),
                "recommendation": ISSUE_RECOMMENDATIONS.get(
                    code,
                    "Review the listed source rows in dataset_issues.csv and correct "
                    "or explicitly accept the condition before relying on the model.",
                ),
            }
        )
    return reasons


def print_verdict_reasons(
    reasons: Sequence[dict[str, object]],
    *,
    stream: object | None = None,
    color_mode: str = "auto",
) -> None:
    """Print actionable issue details immediately below the CLI verdict."""

    if not reasons:
        return
    output = sys.stdout if stream is None else stream
    print("verdict reasons:", file=output)
    for reason in reasons:
        count = int(reason["count"])
        heading = (
            f"  - {reason['severity']} {reason['code']} "
            f"({count} occurrence{'s' if count != 1 else ''})"
        )
        print(
            colorize_audit_text(
                heading,
                str(reason["severity"]),
                output,
                color_mode,
            ),
            file=output,
        )
        messages = list(reason["representative_messages"])
        for index, message in enumerate(messages):
            label = "reason" if index == 0 else "additional detail"
            print(f"    {label}: {message}", file=output)
        if int(reason["additional_message_count"]):
            print(
                "    additional detail: "
                f"{reason['additional_message_count']} more distinct message(s) in "
                "dataset_issues.csv",
                file=output,
            )
        locations = list(reason["affected_examples"])
        if locations:
            print(
                f"    affected: {'; '.join(str(value) for value in locations)}",
                file=output,
            )
        if int(reason["additional_location_count"]):
            print(
                "    affected: "
                f"{reason['additional_location_count']} more location(s) in "
                "dataset_issues.csv",
                file=output,
            )
        print(f"    action: {reason['recommendation']}", file=output)


def audit_records(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
    expected_labels: Sequence[str],
    *,
    passivity_tolerance: float,
    expect_reciprocal: bool,
    reciprocity_tolerance: float,
    frequency_rtol: float,
    frequency_atol_hz: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    problems: list[dict[str, object]] = []
    passivity_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    nports = infer_nports(expected_labels)
    limit = 1.0 + passivity_tolerance

    for record in records:
        block = record.block
        actual_labels = sorted(block.sparams, key=sparam_sort_key)
        missing = sorted(set(expected_labels) - set(actual_labels), key=sparam_sort_key)
        extra = sorted(set(actual_labels) - set(expected_labels), key=sparam_sort_key)
        if missing or extra:
            problems.append(
                issue(
                    "ERROR",
                    "SPARAM_COLUMN_MISMATCH",
                    f"Expected {list(expected_labels)}; missing {missing or 'none'}, "
                    f"extra {extra or 'none'}.",
                    record=record,
                )
            )
        parameter_values = record_parameter_values(record, parameter_names)
        if parameter_values is None:
            problems.append(
                issue(
                    "ERROR",
                    "INVALID_PARAMETER_VALUE",
                    "One or more required geometry parameters are missing, nonnumeric, or nonfinite.",
                    record=record,
                )
            )

        frequencies = np.asarray(block.freq_hz, dtype=float)
        finite_frequency = bool(np.all(np.isfinite(frequencies)))
        if not finite_frequency:
            problems.append(issue("ERROR", "NONFINITE_FREQUENCY", "Frequency values contain NaN or infinity.", record=record))
        if np.any(frequencies < 0.0):
            problems.append(issue("ERROR", "NEGATIVE_FREQUENCY", "Negative frequency values are not valid fitting data.", record=record))
        if frequencies.size > 1 and np.any(np.diff(frequencies) <= frequency_atol_hz):
            problems.append(issue("ERROR", "NONINCREASING_FREQUENCY", "Frequency rows are duplicated or not strictly increasing.", record=record))

        arrays_valid = True
        for label in actual_labels:
            values = np.asarray(block.sparams[label])
            if values.shape != frequencies.shape:
                arrays_valid = False
                problems.append(
                    issue(
                        "ERROR",
                        "SPARAM_LENGTH_MISMATCH",
                        f"{label} has {values.size} values for {frequencies.size} frequency rows.",
                        record=record,
                    )
                )
            elif not np.all(np.isfinite(values.real) & np.isfinite(values.imag)):
                arrays_valid = False
                problems.append(issue("ERROR", "NONFINITE_SPARAM", f"{label} contains NaN or infinity.", record=record))

        maximum_sigma: float | None = None
        worst_frequency: float | None = None
        violating_points = 0
        maximum_reciprocity_error = 0.0
        can_build_matrix = nports is not None and not missing and arrays_valid and finite_frequency
        if can_build_matrix:
            for frequency_index, frequency in enumerate(frequencies):
                matrix = matrix_at(block, expected_labels, nports, frequency_index)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                sigma = float(singular_values[0])
                passive = sigma <= limit
                if not passive:
                    violating_points += 1
                if maximum_sigma is None or sigma > maximum_sigma:
                    maximum_sigma = sigma
                    worst_frequency = float(frequency)
                reciprocity_error = float(np.max(np.abs(matrix - matrix.T)))
                maximum_reciprocity_error = max(maximum_reciprocity_error, reciprocity_error)
                passivity_rows.append(
                    {
                        "dataset": record.dataset_kind,
                        "role": record.role,
                        "source_file": str(record.source_path),
                        "block": block.source_index + 1,
                        "source_index": block.source_index,
                        "frequency_hz": float(frequency),
                        "frequency_type": "dc" if frequency == 0.0 else "rf",
                        "max_singular_value": sigma,
                        "passivity_limit": limit,
                        "passive": passive,
                        "passivity_excess": max(0.0, sigma - limit),
                        "passivity_margin_db": -20.0 * math.log10(max(sigma, 1.0e-300)),
                        "reciprocity_max_abs_error": reciprocity_error,
                        **{
                            name: normalized_mapping_value(block.params, name) or ""
                            for name in parameter_names
                        },
                    }
                )
            if violating_points:
                problems.append(
                    issue(
                        "ERROR",
                        "RAW_NONPASSIVE_DATA",
                        f"{violating_points} frequency row(s) exceed sigma <= {limit:.9g}; "
                        f"worst sigma is {maximum_sigma:.9g} at {worst_frequency:.9g} Hz.",
                        record=record,
                        frequency_hz=worst_frequency,
                    )
                )
            if expect_reciprocal and maximum_reciprocity_error > reciprocity_tolerance:
                problems.append(
                    issue(
                        "ERROR",
                        "RECIPROCITY_VIOLATION",
                        f"Maximum |Sij-Sji| is {maximum_reciprocity_error:.6g}, exceeding "
                        f"the requested limit {reciprocity_tolerance:.6g}.",
                        record=record,
                    )
                )

        block_rows.append(
            {
                "record_id": record.record_id,
                "dataset": record.dataset_kind,
                "role": record.role,
                "source_file": str(record.source_path),
                "block": block.source_index + 1,
                "source_index": block.source_index,
                "frequency_rows": int(frequencies.size),
                "minimum_frequency_hz": float(np.min(frequencies)) if finite_frequency and frequencies.size else None,
                "maximum_frequency_hz": float(np.max(frequencies)) if finite_frequency and frequencies.size else None,
                "dc_rows": int(np.count_nonzero(frequencies == 0.0)),
                "max_singular_value": maximum_sigma,
                "worst_passivity_frequency_hz": worst_frequency,
                "passivity_violating_rows": violating_points,
                "max_reciprocity_abs_error": maximum_reciprocity_error if can_build_matrix else None,
                "reference_impedance_ohm": record.header.reference_impedance_ohm,
                "source_data_format": record.header.data_format,
                "source_frequency_unit": record.header.frequency_unit,
                **{
                    name: normalized_mapping_value(block.params, name) or ""
                    for name in parameter_names
                },
            }
        )
    return block_rows, passivity_rows, problems


def parameter_coverage(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
    geometry_domain: GeometryDomain | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    for kind in sorted({record.dataset_kind for record in records}):
        kind_records = [record for record in records if record.dataset_kind == kind]
        for name_index, name in enumerate(parameter_names):
            role_values: dict[str, list[float]] = {}
            for role in ("train", "verification"):
                values = []
                for record in kind_records:
                    if record.role != role:
                        continue
                    vector = record_parameter_values(record, parameter_names)
                    if vector is not None:
                        values.append(float(vector[name_index]))
                role_values[role] = values

            train_values = role_values["train"]
            verification_values = role_values["verification"]
            train_bounds = (
                (min(train_values), max(train_values)) if train_values else None
            )
            domain_parameter = (
                geometry_domain.parameters.get(normalize_name(name).lower())
                if geometry_domain is not None
                else None
            )
            if domain_parameter is not None:
                coverage_min = float(domain_parameter["lower_base_units"])
                coverage_max = float(domain_parameter["upper_base_units"])
                coverage_basis = "geometry_generation_json"
                coverage_sources = ", ".join(
                    Path(str(path)).name
                    for path in domain_parameter["source_files"]
                )
            elif train_bounds is not None:
                coverage_min, coverage_max = train_bounds
                coverage_basis = "observed_training_extrema"
                coverage_sources = ""
            else:
                coverage_min = None
                coverage_max = None
                coverage_basis = "unavailable"
                coverage_sources = ""

            def outside_bounds(value: float, lower: float, upper: float) -> bool:
                below = value < lower and not math.isclose(
                    value,
                    lower,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-15,
                )
                above = value > upper and not math.isclose(
                    value,
                    upper,
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-15,
                )
                return below or above

            outside_observed = (
                [
                    value
                    for value in verification_values
                    if outside_bounds(value, train_bounds[0], train_bounds[1])
                ]
                if train_bounds is not None
                else []
            )
            outside_coverage = (
                [
                    value
                    for value in verification_values
                    if outside_bounds(value, coverage_min, coverage_max)
                ]
                if coverage_min is not None and coverage_max is not None
                else []
            )

            for role in ("train", "verification"):
                values = role_values[role]
                if not values:
                    continue
                rows.append(
                    {
                        "dataset": kind,
                        "parameter": name,
                        "role": role,
                        "minimum_base_units": min(values),
                        "maximum_base_units": max(values),
                        "unique_values": len(set(values)),
                        "blocks": len(values),
                        "coverage_basis": coverage_basis,
                        "coverage_source_files": coverage_sources,
                        "coverage_minimum_base_units": coverage_min,
                        "coverage_maximum_base_units": coverage_max,
                        "verification_outside_observed_training": (
                            len(outside_observed) if role == "verification" else ""
                        ),
                        "verification_outside_coverage": (
                            len(outside_coverage) if role == "verification" else ""
                        ),
                    }
                )

            if outside_coverage:
                if domain_parameter is not None:
                    code = "VERIFICATION_OUTSIDE_GEOMETRY_RANGE"
                    observed_min = min(verification_values)
                    observed_max = max(verification_values)
                    declared_unit = str(domain_parameter.get("unit") or "base units")
                    message = (
                        f"{len(outside_coverage)} {kind} verification block(s) have "
                        f"{name} outside the declared geometry generation range "
                        f"[{coverage_min:.9g}, {coverage_max:.9g}] from "
                        f"{coverage_sources}; the interpreted verification range is "
                        f"[{observed_min:.9g}, {observed_max:.9g}] in base units "
                        f"(declared unit: {declared_unit}). Check --bare-values and "
                        "dataset_parameter_coverage.csv."
                    )
                else:
                    code = "VERIFICATION_OUTSIDE_TRAIN_RANGE"
                    message = (
                        f"{len(outside_coverage)} {kind} verification block(s) have "
                        f"{name} outside the observed training range "
                        f"[{coverage_min:.9g}, {coverage_max:.9g}]."
                    )
                problems.append(issue("WARNING", code, message))
    return rows, problems


def coarse_coverage_issues(
    records: Sequence[AuditRecord],
    parameter_names: Sequence[str],
) -> list[dict[str, object]]:
    problems: list[dict[str, object]] = []
    fine = [record for record in records if record.dataset_kind == "fine" and record.role in {"train", "verification"}]
    coarse_train = [record for record in records if record.dataset_kind == "coarse" and record.role == "train"]
    if not fine or not coarse_train:
        return problems
    fine_vectors = [record_parameter_values(record, parameter_names) for record in fine]
    coarse_vectors = [record_parameter_values(record, parameter_names) for record in coarse_train]
    fine_vectors = [vector for vector in fine_vectors if vector is not None]
    coarse_vectors = [vector for vector in coarse_vectors if vector is not None]
    if not fine_vectors or not coarse_vectors:
        return problems
    fine_matrix = np.vstack(fine_vectors)
    coarse_matrix = np.vstack(coarse_vectors)
    for index, name in enumerate(parameter_names):
        fine_min, fine_max = float(np.min(fine_matrix[:, index])), float(np.max(fine_matrix[:, index]))
        coarse_min, coarse_max = float(np.min(coarse_matrix[:, index])), float(np.max(coarse_matrix[:, index]))
        if fine_min < coarse_min or fine_max > coarse_max:
            problems.append(
                issue(
                    "WARNING",
                    "FINE_OUTSIDE_COARSE_TRAIN_RANGE",
                    f"Fine {name} range [{fine_min:.9g}, {fine_max:.9g}] extends outside "
                    f"coarse training range [{coarse_min:.9g}, {coarse_max:.9g}]; the "
                    "fitted KBNN coarse model must extrapolate there.",
                )
            )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=os.environ.get("ADS_SURROGATE_CLI_PROG"),
        description="Audit raw MDIF passivity and train/verification consistency before fitting."
    )
    parser.add_argument("--mdif", required=True, help="Fine or direct-model training/combined MDIF.")
    parser.add_argument("--verification-mdif", help="Optional separate fine/direct verification MDIF.")
    parser.add_argument("--coarse-mdif", help="Optional KBNN coarse training/combined MDIF.")
    parser.add_argument("--coarse-verification-mdif", help="Optional separate KBNN coarse verification MDIF.")
    parser.add_argument(
        "--geometry-json",
        action="append",
        default=[],
        help=(
            "Generated geometry metadata JSON whose declared parameter bounds define "
            "coverage. Repeat for extended campaigns. When omitted, valid same-stem "
            "JSON files beside the supplied MDIFs are used automatically."
        ),
    )
    parser.add_argument(
        "--bare-values",
        choices=("auto", "parameter-units", "base-units"),
        default="auto",
        help=(
            "Interpret unitless MDIF parameter values as declared geometry units, "
            "SI/base units, or choose per source and parameter against the geometry "
            "JSON bounds. Default: auto."
        ),
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="always",
        help=(
            "Color the CLI verdict and reason headings: auto for terminals, always "
            "for captured ADS/IDE consoles, or never. Default: always."
        ),
    )
    parser.add_argument("--out-dir", default="dataset_audit", help="Audit artifact directory. Default: dataset_audit.")
    parser.add_argument("--parameter-names", help="Comma-separated geometry/process VAR names. Default: infer from the fine data.")
    parser.add_argument("--split-var", default="dataset", help="VAR that identifies train and verification blocks. Default: dataset.")
    parser.add_argument("--train-values", default="train,training")
    parser.add_argument("--verify-values", default="verify,verification,test,validation")
    parser.add_argument("--holdout-fraction", type=float, default=0.2, help="Random holdout used only when no recognized train labels exist. Default: 0.2.")
    parser.add_argument("--seed", type=int, default=1234, help="Random-holdout seed. Default: 1234.")
    parser.add_argument("--passivity-tolerance", type=float, default=1e-6, help="Allow sigma <= 1 + tolerance. Default: 1e-6.")
    parser.add_argument("--expect-reciprocal", action="store_true", help="Also require Sij and Sji to agree within --reciprocity-tolerance.")
    parser.add_argument("--reciprocity-tolerance", type=float, default=1e-3, help="Maximum absolute |Sij-Sji| when reciprocity is required. Default: 1e-3.")
    parser.add_argument("--parameter-rel-tolerance", type=float, default=1e-10, help="Relative tolerance for duplicate-geometry detection. Default: 1e-10.")
    parser.add_argument("--parameter-abs-tolerance", type=float, default=1e-15, help="Base-unit absolute tolerance for duplicate-geometry detection. Default: 1e-15.")
    parser.add_argument("--frequency-rel-tolerance", type=float, default=1e-10, help="Relative frequency-grid comparison tolerance. Default: 1e-10.")
    parser.add_argument("--frequency-abs-tolerance-hz", type=float, default=1e-3, help="Absolute frequency-grid tolerance in Hz. Default: 1e-3.")
    parser.add_argument("--response-rel-tolerance", type=float, default=1e-4, help="Relative tolerance for duplicate response conflicts. Default: 1e-4.")
    parser.add_argument("--response-abs-tolerance", type=float, default=1e-6, help="Absolute tolerance for duplicate response conflicts. Default: 1e-6.")
    parser.add_argument("--neighbor-outlier-factor", type=float, default=5.0, help="Nearest-neighbor response jump warning factor above the median. Default: 5.")
    parser.add_argument("--neighbor-min-relative-jump", type=float, default=0.05, help="Minimum relative response RMSE for a neighbor warning. Default: 0.05.")
    parser.add_argument("--fail-on-warnings", action="store_true", help="Return nonzero status for warnings as well as errors.")
    add_options_json_argument(parser, recursive=False)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.coarse_verification_mdif and not args.coarse_mdif:
        parser.error("--coarse-verification-mdif requires --coarse-mdif")
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be between 0 and 1")
    nonnegative = [
        "passivity_tolerance",
        "reciprocity_tolerance",
        "parameter_rel_tolerance",
        "parameter_abs_tolerance",
        "frequency_rel_tolerance",
        "frequency_abs_tolerance_hz",
        "response_rel_tolerance",
        "response_abs_tolerance",
        "neighbor_min_relative_jump",
    ]
    for name in nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not math.isfinite(args.neighbor_outlier_factor) or args.neighbor_outlier_factor <= 0.0:
        parser.error("--neighbor-outlier-factor must be finite and positive")


def run_audit(args: argparse.Namespace) -> tuple[int, dict[str, object]]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_values = parse_csv_set(args.train_values)
    verify_values = parse_csv_set(args.verify_values)
    mdif_paths = [
        Path(path)
        for path in (
            args.mdif,
            args.verification_mdif,
            args.coarse_mdif,
            args.coarse_verification_mdif,
        )
        if path
    ]
    geometry_domain = resolve_geometry_domain(args.geometry_json, mdif_paths)
    records: list[AuditRecord] = []
    problems: list[dict[str, object]] = []
    next_record_id = 1

    fine_records, load_problems, next_record_id = load_dataset_records(
        "fine",
        Path(args.mdif),
        Path(args.verification_mdif) if args.verification_mdif else None,
        split_var=args.split_var,
        train_values=train_values,
        verify_values=verify_values,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        starting_record_id=next_record_id,
    )
    records.extend(fine_records)
    problems.extend(load_problems)
    if args.coarse_mdif:
        coarse_records, load_problems, next_record_id = load_dataset_records(
            "coarse",
            Path(args.coarse_mdif),
            Path(args.coarse_verification_mdif) if args.coarse_verification_mdif else None,
            split_var=args.split_var,
            train_values=train_values,
            verify_values=verify_values,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
            starting_record_id=next_record_id,
        )
        records.extend(coarse_records)
        problems.extend(load_problems)

    requested_names = args.parameter_names
    if requested_names is None and geometry_domain is not None:
        requested_names = ",".join(
            str(parameter["name"])
            for parameter in geometry_domain.parameters.values()
        )
    parameter_names = infer_parameter_names(
        [record.block for record in fine_records],
        requested=requested_names,
        split_var=args.split_var,
    )
    if geometry_domain is not None:
        missing_geometry_parameters = [
            name
            for name in parameter_names
            if normalize_name(name).lower() not in geometry_domain.parameters
        ]
        if missing_geometry_parameters:
            raise ValueError(
                "Geometry metadata does not declare the audited parameter(s): "
                + ", ".join(missing_geometry_parameters)
            )
    parameter_unit_interpretation = resolve_record_parameter_values(
        records,
        parameter_names,
        geometry_domain,
        args.bare_values,
    )
    label_sets = Counter(
        tuple(sorted(record.block.sparams, key=sparam_sort_key)) for record in fine_records
    )
    labels = list(label_sets.most_common(1)[0][0])
    nports = infer_nports(labels)
    if nports is None:
        problems.append(
            issue(
                "ERROR",
                "INCOMPLETE_SPARAM_MATRIX",
                f"The dominant S-parameter columns {labels} do not form a complete square matrix.",
            )
        )

    block_rows, passivity_rows, record_problems = audit_records(
        records,
        parameter_names,
        labels,
        passivity_tolerance=args.passivity_tolerance,
        expect_reciprocal=args.expect_reciprocal,
        reciprocity_tolerance=args.reciprocity_tolerance,
        frequency_rtol=args.frequency_rel_tolerance,
        frequency_atol_hz=args.frequency_abs_tolerance_hz,
    )
    problems.extend(record_problems)

    grid_assignments, grid_rows = assign_frequency_grids(
        records,
        rtol=args.frequency_rel_tolerance,
        atol_hz=args.frequency_abs_tolerance_hz,
    )
    for row in block_rows:
        row["frequency_grid_id"] = grid_assignments[int(row["record_id"])]
    for kind in sorted({record.dataset_kind for record in records}):
        kind_grid_ids = {
            grid_assignments[record.record_id]
            for record in records
            if record.dataset_kind == kind and record.role in {"train", "verification"}
        }
        if len(kind_grid_ids) > 1:
            problems.append(
                issue(
                    "WARNING",
                    "INCONSISTENT_FREQUENCY_GRIDS",
                    f"{kind} data uses {len(kind_grid_ids)} distinct frequency grids. "
                    "Variable grids are supported, but missing or inconsistent sweeps can "
                    "create uneven fitting coverage.",
                )
            )

    z0_values = sorted(
        {
            float(record.header.reference_impedance_ohm)
            for record in records
            if record.header.reference_impedance_ohm is not None
        }
    )
    missing_z0 = sum(record.header.reference_impedance_ohm is None for record in records)
    if len(z0_values) > 1:
        problems.append(issue("ERROR", "INCONSISTENT_REFERENCE_IMPEDANCE", f"MDIF blocks declare multiple reference impedances: {z0_values}."))
    if missing_z0:
        problems.append(issue("WARNING", "MISSING_REFERENCE_IMPEDANCE", f"{missing_z0} block(s) do not explicitly declare an R reference impedance."))

    duplicate_rows: list[dict[str, object]] = []
    for kind in sorted({record.dataset_kind for record in records}):
        kind_records = [record for record in records if record.dataset_kind == kind]
        kind_duplicates, duplicate_problems = duplicate_geometry_analysis(
            kind_records,
            parameter_names,
            labels,
            parameter_rtol=args.parameter_rel_tolerance,
            parameter_atol=args.parameter_abs_tolerance,
            frequency_rtol=args.frequency_rel_tolerance,
            frequency_atol_hz=args.frequency_abs_tolerance_hz,
            response_rtol=args.response_rel_tolerance,
            response_atol=args.response_abs_tolerance,
        )
        duplicate_rows.extend(kind_duplicates)
        problems.extend(duplicate_problems)

    neighbor_rows, neighbor_problems = nearest_neighbor_analysis(
        records,
        parameter_names,
        labels,
        frequency_rtol=args.frequency_rel_tolerance,
        frequency_atol_hz=args.frequency_abs_tolerance_hz,
        outlier_factor=args.neighbor_outlier_factor,
        minimum_relative_jump=args.neighbor_min_relative_jump,
    )
    problems.extend(neighbor_problems)
    coverage_rows, coverage_problems = parameter_coverage(
        records,
        parameter_names,
        geometry_domain,
    )
    problems.extend(coverage_problems)
    problems.extend(coarse_coverage_issues(records, parameter_names))

    role_counts = Counter((record.dataset_kind, record.role) for record in records)
    unassigned = [record for record in records if record.role == "unassigned"]
    if unassigned:
        problems.append(
            issue(
                "ERROR",
                "UNASSIGNED_BLOCKS",
                f"{len(unassigned)} block(s) have unrecognized {args.split_var} values and "
                "will be ignored by the fitter while explicit training labels exist.",
            )
        )
    if not any(record.role == "train" and record.dataset_kind == "fine" for record in records):
        problems.append(issue("ERROR", "NO_TRAINING_BLOCKS", "No fine/direct training blocks were selected."))
    if not any(record.role == "verification" and record.dataset_kind == "fine" for record in records):
        problems.append(issue("WARNING", "NO_VERIFICATION_BLOCKS", "No fine/direct verification blocks were selected."))

    severity_counts = Counter(str(row["severity"]) for row in problems)
    violation_rows = [row for row in passivity_rows if not bool(row["passive"])]
    dc_violation_rows = [
        row for row in violation_rows if row["frequency_type"] == "dc"
    ]
    rf_violation_rows = [
        row for row in violation_rows if row["frequency_type"] == "rf"
    ]
    worst_row = max(
        passivity_rows,
        key=lambda row: float(row["max_singular_value"]),
        default=None,
    )
    if severity_counts["ERROR"]:
        verdict = "FAIL"
    elif severity_counts["WARNING"]:
        verdict = "WARNING"
    else:
        verdict = "PASS"
    verdict_reasons = summarize_issue_reasons(problems)

    issues_path = out_dir / "dataset_issues.csv"
    blocks_path = out_dir / "dataset_blocks.csv"
    passivity_path = out_dir / "dataset_passivity.csv"
    grids_path = out_dir / "dataset_frequency_grids.csv"
    coverage_path = out_dir / "dataset_parameter_coverage.csv"
    duplicates_path = out_dir / "dataset_duplicates.csv"
    neighbors_path = out_dir / "dataset_neighbor_consistency.csv"
    plot_path = out_dir / "dataset_passivity.png"
    json_path = out_dir / "dataset_audit.json"
    markdown_path = out_dir / "dataset_audit.md"
    write_csv(issues_path, problems)
    write_csv(blocks_path, block_rows)
    write_csv(passivity_path, passivity_rows)
    write_csv(grids_path, grid_rows)
    write_csv(coverage_path, coverage_rows)
    write_csv(duplicates_path, duplicate_rows)
    write_csv(neighbors_path, neighbor_rows)
    plot_written = png_passivity_plot(plot_path, block_rows, 1.0 + args.passivity_tolerance)
    geometry_domain_summary = (
        {
            "source": "geometry_generation_json",
            "selection": "inferred_same_stem" if geometry_domain.inferred else "explicit",
            "source_files": [str(path) for path in geometry_domain.source_paths],
            "parameters": list(geometry_domain.parameters.values()),
        }
        if geometry_domain is not None
        else {
            "source": "observed_training_extrema",
            "selection": "no_geometry_json",
            "source_files": [],
            "parameters": [],
        }
    )

    summary: dict[str, object] = {
        "verdict": verdict,
        "parameter_names": list(parameter_names),
        "sparameter_labels": labels,
        "nports": nports,
        "block_counts": {
            f"{kind}_{role}": count
            for (kind, role), count in sorted(role_counts.items())
        },
        "frequency_rows": len(passivity_rows),
        "passivity": {
            "limit": 1.0 + args.passivity_tolerance,
            "violating_rows": len(violation_rows),
            "violating_dc_rows": len(dc_violation_rows),
            "violating_rf_rows": len(rf_violation_rows),
            "violating_blocks": len(
                {
                    (row["dataset"], row["source_file"], row["block"])
                    for row in violation_rows
                }
            ),
            "max_singular_value": (
                float(worst_row["max_singular_value"]) if worst_row else None
            ),
            "worst_row": worst_row,
        },
        "reference_impedance_ohm": z0_values,
        "frequency_grid_count": len(grid_rows),
        "coverage_domain": geometry_domain_summary,
        "parameter_unit_interpretation": parameter_unit_interpretation,
        "issue_counts": dict(severity_counts),
        "issue_code_counts": dict(Counter(str(row["code"]) for row in problems)),
        "verdict_reasons": verdict_reasons,
        "artifacts": {
            "report": markdown_path.name,
            "issues": issues_path.name,
            "blocks": blocks_path.name,
            "passivity": passivity_path.name,
            "frequency_grids": grids_path.name,
            "parameter_coverage": coverage_path.name,
            "duplicates": duplicates_path.name,
            "neighbor_consistency": neighbors_path.name,
            "passivity_plot": plot_path.name if plot_written else None,
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    markdown: list[str] = [
        "# MDIF Dataset Audit",
        "",
        f"**Overall verdict: {verdict}.**",
        "",
        "This report evaluates the supplied raw S-parameter data, before any neural "
        "model is fitted. A raw-data PASS does not guarantee that an unconstrained "
        "neural interpolation will remain passive between samples.",
        "",
        "## Why this verdict was issued",
        "",
    ]
    if verdict_reasons:
        markdown.append(
            f"The `{verdict}` verdict is caused by the following grouped audit "
            "conditions. Each reason includes the practical next step; "
            f"[{issues_path.name}]({issues_path.name}) contains every individual row."
        )
        for reason in verdict_reasons:
            count = int(reason["count"])
            markdown.extend(
                [
                    "",
                    f"### {reason['severity']}: `{reason['code']}`",
                    "",
                    f"**Occurrences:** {count}",
                    "",
                    f"**Reason:** {reason['reason']}",
                ]
            )
            messages = list(reason["representative_messages"])[1:]
            if messages:
                markdown.extend(["", "Additional representative details:", ""])
                markdown.extend(f"- {message}" for message in messages)
            if int(reason["additional_message_count"]):
                markdown.append(
                    f"- {reason['additional_message_count']} more distinct message(s) "
                    f"are recorded in [{issues_path.name}]({issues_path.name})."
                )
            locations = list(reason["affected_examples"])
            if locations:
                markdown.extend(
                    [
                        "",
                        "**Affected examples:** "
                        + "; ".join(str(value) for value in locations),
                    ]
                )
            if int(reason["additional_location_count"]):
                markdown.append(
                    f"{reason['additional_location_count']} more affected location(s) "
                    f"are recorded in [{issues_path.name}]({issues_path.name})."
                )
            markdown.extend(
                [
                    "",
                    f"**Recommended action:** {reason['recommendation']}",
                ]
            )
    else:
        markdown.append(
            "The audit found no error or warning conditions, so the raw dataset "
            "received a `PASS` verdict."
        )
    markdown.extend(
        [
            "",
            "## Summary",
            "",
        ]
    )
    markdown.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ["Parameters", ", ".join(parameter_names)],
                [
                    "Coverage range source",
                    (
                        "geometry-generation JSON: "
                        + ", ".join(
                            Path(str(path)).name
                            for path in geometry_domain_summary["source_files"]
                        )
                        if geometry_domain is not None
                        else "observed training extrema (no geometry JSON supplied or inferred)"
                    ),
                ],
                ["Ports", nports if nports is not None else "incomplete matrix"],
                ["Fine training blocks", role_counts.get(("fine", "train"), 0)],
                ["Fine verification blocks", role_counts.get(("fine", "verification"), 0)],
                ["Coarse training blocks", role_counts.get(("coarse", "train"), 0)],
                ["Coarse verification blocks", role_counts.get(("coarse", "verification"), 0)],
                ["Evaluated frequency rows", len(passivity_rows)],
                ["Raw non-passive rows", len(violation_rows)],
                ["Raw non-passive DC rows", len(dc_violation_rows)],
                ["Raw non-passive RF rows", len(rf_violation_rows)],
                ["Raw non-passive blocks", summary["passivity"]["violating_blocks"]],
                ["Worst raw sigma", format_number(summary["passivity"]["max_singular_value"])],
                ["Distinct frequency grids", len(grid_rows)],
                ["Errors", severity_counts.get("ERROR", 0)],
                ["Warnings", severity_counts.get("WARNING", 0)],
            ],
        )
    )
    markdown.extend(["", "## Raw-data passivity", ""])
    if plot_written:
        markdown.extend([f"![Maximum singular value by block]({plot_path.name})", ""])
    if violation_rows:
        worst_violations = sorted(
            violation_rows,
            key=lambda row: float(row["max_singular_value"]),
            reverse=True,
        )[:20]
        markdown.append(
            f"The raw data itself violates the configured limit at {len(violation_rows)} "
            "frequency rows. Fix or intentionally exclude these samples before diagnosing "
            "the neural architecture. The worst rows are:"
        )
        markdown.append("")
        markdown.extend(
            markdown_table(
                ["Dataset", "Role", "Block", "Frequency (Hz)", "Max sigma", "Excess"],
                [
                    [
                        row["dataset"],
                        row["role"],
                        row["block"],
                        format_number(row["frequency_hz"]),
                        format_number(row["max_singular_value"]),
                        format_number(row["passivity_excess"]),
                    ]
                    for row in worst_violations
                ],
            )
        )
    else:
        markdown.append(
            "Every complete finite raw S-matrix satisfies the configured singular-value "
            f"limit of {1.0 + args.passivity_tolerance:.9g}."
        )

    markdown.extend(["", "## Issues affecting modeling", ""])
    if problems:
        markdown.extend(
            markdown_table(
                ["Severity", "Code", "Dataset", "Role", "Block", "Message"],
                [
                    [
                        row["severity"],
                        row["code"],
                        row["dataset"],
                        row["role"],
                        row["block"],
                        row["message"],
                    ]
                    for row in problems[:80]
                ],
            )
        )
        if len(problems) > 80:
            markdown.extend(["", f"The table is truncated; all {len(problems)} issues are in [{issues_path.name}]({issues_path.name})."])
    else:
        markdown.append("No structural, split, duplicate, coverage, or passivity issues were found.")

    markdown.extend(["", "## Declared geometry domain", ""])
    if geometry_domain is not None:
        markdown.append(
            "Verification coverage is checked against these declared generation "
            "bounds, not against the sampled training-point extrema."
        )
        markdown.append("")
        markdown.extend(
            markdown_table(
                [
                    "Parameter",
                    "Minimum (base units)",
                    "Maximum (base units)",
                    "Scale",
                    "Source JSON",
                ],
                [
                    [
                        parameter["name"],
                        format_number(parameter["lower_base_units"]),
                        format_number(parameter["upper_base_units"]),
                        parameter["scale"],
                        ", ".join(
                            Path(str(path)).name
                            for path in parameter["source_files"]
                        ),
                    ]
                    for parameter in geometry_domain.parameters.values()
                ],
            )
        )
    else:
        markdown.append(
            "No geometry-generation JSON was supplied or inferred. Verification "
            "coverage therefore uses the observed training-point extrema. Pass "
            "`--geometry-json geometries.json` to use the intended generated domain."
        )

    markdown.extend(["", "## Parameter unit interpretation", ""])
    markdown.append(
        "Explicit MDIF unit suffixes are always converted directly. Unitless values "
        "use the selected `--bare-values` policy; `auto` compares base-unit and "
        "declared-parameter-unit interpretations against the geometry JSON for each "
        "source and parameter."
    )
    markdown.append("")
    markdown.extend(
        markdown_table(
            [
                "Source",
                "Parameter",
                "Requested policy",
                "Selected interpretation",
                "Declared unit",
                "Unitless values",
                "Explicit-unit values",
                "Inside as base units",
                "Inside as parameter units",
            ],
            [
                [
                    Path(str(row["source_file"])).name,
                    row["parameter"],
                    row["requested_bare_values"],
                    row["selected_interpretation"],
                    row["declared_unit"] or "base units",
                    row["unitless_value_count"],
                    row["explicit_unit_value_count"],
                    row["base_units_inside_geometry"],
                    row["parameter_units_inside_geometry"],
                ]
                for row in parameter_unit_interpretation
            ],
        )
    )

    markdown.extend(["", "## Parameter coverage", ""])
    markdown.extend(
        markdown_table(
            [
                "Dataset",
                "Parameter",
                "Role",
                "Observed minimum",
                "Observed maximum",
                "Coverage basis",
                "Coverage minimum",
                "Coverage maximum",
                "Verification outside coverage",
            ],
            [
                [
                    row["dataset"],
                    row["parameter"],
                    row["role"],
                    format_number(row["minimum_base_units"]),
                    format_number(row["maximum_base_units"]),
                    row["coverage_basis"],
                    format_number(row["coverage_minimum_base_units"]),
                    format_number(row["coverage_maximum_base_units"]),
                    row["verification_outside_coverage"],
                ]
                for row in coverage_rows
            ],
        )
    )
    markdown.extend(["", "## Largest nearest-neighbor response jumps", ""])
    if neighbor_rows:
        markdown.append(
            "These are diagnostics, not automatic proof of bad data. A legitimate sharp "
            "resonance can create a large jump, but isolated jumps often identify failed "
            "simulations, port-order mistakes, or geometry metadata mismatches."
        )
        markdown.append("")
        markdown.extend(
            markdown_table(
                ["Dataset", "Blocks", "Geometry distance", "Relative S RMSE", "Max abs delta S", "Outlier"],
                [
                    [
                        row["dataset"],
                        f"{row['left_block']} / {row['right_block']}",
                        format_number(row["normalized_geometry_distance"]),
                        format_number(row["relative_rmse"]),
                        format_number(row["max_abs"]),
                        "yes" if row.get("response_jump_outlier") else "no",
                    ]
                    for row in neighbor_rows[:20]
                ],
            )
        )
    else:
        markdown.append("No comparable nearest-neighbor response pairs were available.")

    markdown.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If raw non-passive rows are reported, inspect those exact block/frequency "
            "entries in `dataset_passivity.csv`; the model cannot learn a universally "
            "passive mapping from contradictory non-passive targets.",
            "- Exact-zero-Hz rows are reported separately. The RF fit excludes DC, so "
            "DC-only violations do not explain non-passive positive-frequency predictions; "
            "they instead affect whether a DC sample can be used for DC extraction.",
            "- If the raw data passes but every DNN/KBNN trial is non-passive, the likely "
            "cause is unconstrained interpolation, output-domain conditioning, insufficient "
            "sampling near resonances, or model capacity—not necessarily corrupt data.",
            "- Train/verification overlap invalidates verification metrics. Conflicting "
            "duplicate geometries are stronger evidence of bad simulation or metadata.",
            "- Verification points outside the declared geometry-generation range test "
            "extrapolation rather than interpolation. When no geometry JSON is available, "
            "the observed training extrema are used as a conservative fallback. For KBNN, "
            "fine points outside the coarse training range also force the fitted coarse "
            "model to extrapolate.",
            "- Multiple frequency grids are supported, but a low-coverage grid or missing "
            "resonance band can bias the loss. Review `dataset_frequency_grids.csv`.",
            "",
            "## Detailed artifacts",
            "",
            f"- [Machine-readable summary]({json_path.name})",
            f"- [All issues]({issues_path.name})",
            f"- [Per-block summary]({blocks_path.name})",
            f"- [Every passivity calculation]({passivity_path.name})",
            f"- [Frequency grids]({grids_path.name})",
            f"- [Parameter coverage]({coverage_path.name})",
            f"- [Duplicate geometry comparisons]({duplicates_path.name})",
            f"- [Nearest-neighbor response comparisons]({neighbors_path.name})",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown))

    print(format_audit_verdict_line(verdict, color_mode=args.color))
    print(
        "blocks: "
        f"fine train={role_counts.get(('fine', 'train'), 0)}, "
        f"fine verification={role_counts.get(('fine', 'verification'), 0)}, "
        f"coarse train={role_counts.get(('coarse', 'train'), 0)}, "
        f"coarse verification={role_counts.get(('coarse', 'verification'), 0)}"
    )
    if geometry_domain is not None:
        selection = "inferred" if geometry_domain.inferred else "explicit"
        print(
            f"coverage domain: geometry JSON ({selection}): "
            + ", ".join(str(path) for path in geometry_domain.source_paths)
        )
        for parameter in geometry_domain.parameters.values():
            print(
                f"  {parameter['name']}: "
                f"[{format_number(parameter['lower_base_units'])}, "
                f"{format_number(parameter['upper_base_units'])}] base units"
            )
    else:
        print(
            "coverage domain: observed training extrema; pass --geometry-json "
            "geometries.json to use declared generation bounds"
        )
    unitless_decisions = [
        row
        for row in parameter_unit_interpretation
        if int(row["unitless_value_count"]) > 0
    ]
    if unitless_decisions:
        print(f"unitless parameter interpretation (--bare-values {args.bare_values}):")
        for row in unitless_decisions:
            declared = str(row["declared_unit"] or "base units")
            print(
                f"  {Path(str(row['source_file'])).name}: {row['parameter']} -> "
                f"{row['selected_interpretation']} (declared {declared}, "
                f"{row['unitless_value_count']} unitless value(s))"
            )
    print(
        f"raw passivity: {len(violation_rows)} violating row(s) "
        f"(RF={len(rf_violation_rows)}, DC={len(dc_violation_rows)}), "
        f"worst sigma={format_number(summary['passivity']['max_singular_value'])}, "
        f"limit={1.0 + args.passivity_tolerance:.9g}"
    )
    print(
        f"issues: {severity_counts.get('ERROR', 0)} error(s), "
        f"{severity_counts.get('WARNING', 0)} warning(s)"
    )
    print_verdict_reasons(verdict_reasons, color_mode=args.color)
    print(f"report: {markdown_path}")
    exit_code = 1 if severity_counts["ERROR"] else 0
    if args.fail_on_warnings and severity_counts["WARNING"]:
        exit_code = 1
    return exit_code, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parse_args_with_options_json(
        parser,
        argv,
        workflow="audit",
        command="audit",
    )
    validate_args(parser, args)
    try:
        status = run_audit(args)[0]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return finalize_options_json_update(args, status)


if __name__ == "__main__":
    raise SystemExit(main())
