#!/usr/bin/env python3
"""Audit raw MDIF training and verification data before surrogate fitting.

The audit is model-independent.  It checks whether the supplied S matrices are
passive and structurally suitable for DNN, KBNN, or Neuro-TF fitting, then
writes detailed machine-readable and human-readable diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

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


def record_parameter_values(
    record: AuditRecord,
    parameter_names: Sequence[str],
) -> np.ndarray | None:
    values: list[float] = []
    for name in parameter_names:
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


def svg_passivity_plot(path: Path, block_rows: Sequence[dict[str, object]], limit: float) -> bool:
    plotted = [row for row in block_rows if isinstance(row.get("max_singular_value"), (int, float))]
    if not plotted:
        return False
    width, height = 1000, 420
    left, right, top, bottom = 78, 25, 42, 62
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
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Raw MDIF passivity by block</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#444"/>',
    ]
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5.0
        y = y_coord(value)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.5g}</text>',
            ]
        )
    limit_y = y_coord(limit)
    lines.append(
        f'<line x1="{left}" y1="{limit_y:.2f}" x2="{left + plot_width}" y2="{limit_y:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    for index, row in enumerate(plotted):
        key = (str(row["dataset"]), str(row["role"]))
        color = colors.get(key, "#64748b")
        value = float(row["max_singular_value"])
        title = html.escape(
            f"{row['dataset']} {row['role']} block {row['block']}: sigma={value:.9g}"
        )
        lines.append(
            f'<circle cx="{x_coord(index):.2f}" cy="{y_coord(value):.2f}" r="3.2" fill="{color}"><title>{title}</title></circle>'
        )
    lines.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">Block sequence (fine then coarse; train and verification)</text>',
            f'<text x="18" y="{top + plot_height / 2:.2f}" transform="rotate(-90 18 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="13">Maximum singular value</text>',
            '<text x="790" y="49" font-family="sans-serif" font-size="11" fill="#dc2626">passivity limit</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(lines) + "\n")
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
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    problems: list[dict[str, object]] = []
    for kind in sorted({record.dataset_kind for record in records}):
        kind_records = [record for record in records if record.dataset_kind == kind]
        for name_index, name in enumerate(parameter_names):
            ranges: dict[str, tuple[float, float, int]] = {}
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
                if values:
                    ranges[role] = (min(values), max(values), len(values))
                    rows.append(
                        {
                            "dataset": kind,
                            "parameter": name,
                            "role": role,
                            "minimum_base_units": min(values),
                            "maximum_base_units": max(values),
                            "unique_values": len(set(values)),
                            "blocks": len(values),
                        }
                    )
            if "train" in ranges and role_values["verification"]:
                train_min, train_max, _ = ranges["train"]
                outside = [
                    value
                    for value in role_values["verification"]
                    if value < train_min or value > train_max
                ]
                if outside:
                    problems.append(
                        issue(
                            "WARNING",
                            "VERIFICATION_OUTSIDE_TRAIN_RANGE",
                            f"{len(outside)} {kind} verification block(s) have {name} outside "
                            f"the training range [{train_min:.9g}, {train_max:.9g}].",
                        )
                    )
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
        description="Audit raw MDIF passivity and train/verification consistency before fitting."
    )
    parser.add_argument("--mdif", required=True, help="Fine or direct-model training/combined MDIF.")
    parser.add_argument("--verification-mdif", help="Optional separate fine/direct verification MDIF.")
    parser.add_argument("--coarse-mdif", help="Optional KBNN coarse training/combined MDIF.")
    parser.add_argument("--coarse-verification-mdif", help="Optional separate KBNN coarse verification MDIF.")
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
    parameter_names = infer_parameter_names(
        [record.block for record in fine_records],
        requested=requested_names,
        split_var=args.split_var,
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
    coverage_rows, coverage_problems = parameter_coverage(records, parameter_names)
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

    issues_path = out_dir / "dataset_issues.csv"
    blocks_path = out_dir / "dataset_blocks.csv"
    passivity_path = out_dir / "dataset_passivity.csv"
    grids_path = out_dir / "dataset_frequency_grids.csv"
    coverage_path = out_dir / "dataset_parameter_coverage.csv"
    duplicates_path = out_dir / "dataset_duplicates.csv"
    neighbors_path = out_dir / "dataset_neighbor_consistency.csv"
    plot_path = out_dir / "dataset_passivity.svg"
    json_path = out_dir / "dataset_audit.json"
    markdown_path = out_dir / "dataset_audit.md"
    write_csv(issues_path, problems)
    write_csv(blocks_path, block_rows)
    write_csv(passivity_path, passivity_rows)
    write_csv(grids_path, grid_rows)
    write_csv(coverage_path, coverage_rows)
    write_csv(duplicates_path, duplicate_rows)
    write_csv(neighbors_path, neighbor_rows)
    plot_written = svg_passivity_plot(plot_path, block_rows, 1.0 + args.passivity_tolerance)

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
        "issue_counts": dict(severity_counts),
        "issue_code_counts": dict(Counter(str(row["code"]) for row in problems)),
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
        "## Summary",
        "",
    ]
    markdown.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ["Parameters", ", ".join(parameter_names)],
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

    markdown.extend(["", "## Parameter coverage", ""])
    markdown.extend(
        markdown_table(
            ["Dataset", "Parameter", "Role", "Minimum (base units)", "Maximum (base units)", "Unique values"],
            [
                [
                    row["dataset"],
                    row["parameter"],
                    row["role"],
                    format_number(row["minimum_base_units"]),
                    format_number(row["maximum_base_units"]),
                    row["unique_values"],
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
            "- Verification points outside the training range test extrapolation rather than "
            "interpolation. For KBNN, fine points outside the coarse training range also "
            "force the fitted coarse model to extrapolate.",
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

    print(f"dataset audit: {verdict}")
    print(
        "blocks: "
        f"fine train={role_counts.get(('fine', 'train'), 0)}, "
        f"fine verification={role_counts.get(('fine', 'verification'), 0)}, "
        f"coarse train={role_counts.get(('coarse', 'train'), 0)}, "
        f"coarse verification={role_counts.get(('coarse', 'verification'), 0)}"
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
    print(f"report: {markdown_path}")
    exit_code = 1 if severity_counts["ERROR"] else 0
    if args.fail_on_warnings and severity_counts["WARNING"]:
        exit_code = 1
    return exit_code, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        return run_audit(args)[0]
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
