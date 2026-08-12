#!/usr/bin/env python3
"""Shared training, reporting, and ADS export utilities for all surrogate models.

The implementation is intentionally dependency-light and uses only NumPy for
model calculations. DNN, KBNN, and Neuro-TF entry points import this module
directly from the repository root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import difflib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import textwrap
import traceback
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


VERSION = "0.2.0-rc2"
EPS = 1e-12
DB_MAG_FLOOR = 1e-6
DEFAULT_DC_OPEN_THRESHOLD_OHM = 1e12
DEFAULT_DC_OPEN_RESISTANCE_OHM = 1e19
DEFAULT_DC_PASSIVITY_TOLERANCE = 1e-6
DEFAULT_DC_EXPORT_S_MATCH_TOLERANCE = 1e-3


FREQ_UNITS = {
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
    "thz": 1e12,
}

VALUE_UNITS = {
    "": 1.0,
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "um": 1e-6,
    "micron": 1e-6,
    "microns": 1e-6,
    "m": 1e-3,
    "mil": 25.4e-6,
    "mm": 1e-3,
    "cm": 1e-2,
    "meter": 1.0,
    "meters": 1.0,
    "in": 0.0254,
    "inch": 0.0254,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
}


@dataclass
class MDIFBlock:
    params: dict[str, str]
    freq_hz: np.ndarray
    sparams: dict[str, np.ndarray]
    source_index: int = 0


@dataclass
class SplitData:
    train: list[MDIFBlock]
    verify: list[MDIFBlock]
    all_blocks: list[MDIFBlock]


def normalize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.strip()).strip("_")


def normalized_mapping_value(mapping: dict[str, str], name: str) -> str | None:
    """Read a normalized MDIF VAR name without making its case significant."""

    target = normalize_name(name).lower()
    for key, value in mapping.items():
        if normalize_name(key).lower() == target:
            return value
    return None


def strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_number(value: str) -> float | None:
    """Parse a floating value with an optional engineering/unit suffix."""

    text = strip_quotes(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$",
        text,
    )
    if not match:
        return None
    value_float = float(match.group(1))
    unit = match.group(2).lower()
    if unit in VALUE_UNITS:
        return value_float * VALUE_UNITS[unit]
    if unit in FREQ_UNITS:
        return value_float * FREQ_UNITS[unit]
    return value_float


def parse_scale_number(value: str) -> float:
    """Parse a positive scale value; bare unit names mean one of that unit."""

    text = strip_quotes(value).strip()
    parsed = parse_number(text)
    if parsed is None and re.fullmatch(r"[A-Za-z]+", text):
        parsed = parse_number(f"1{text}")
    if parsed is None:
        raise ValueError(f"Could not parse scale value {value!r}")
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"Scale value must be positive and finite, got {value!r}")
    return float(parsed)


def parse_parameter_scale_spec(
    parameter_names: Sequence[str],
    spec: str | None,
) -> dict[str, float]:
    """Apply one ADS/base-unit-to-model-unit scale to every parameter.

    The returned scale is the ADS-facing parameter value per training-model
    unit. During Verilog-A export the model feature is computed as:

        model_value = ads_instance_parameter / scale

    For example, if every MDIF parameter uses dimensionless micron values but
    ADS passes parameters in meters, use 1um.
    """

    names = list(parameter_names)
    if not spec:
        return {name: 1.0 for name in names}

    text = spec.strip()
    if not text:
        return {name: 1.0 for name in names}
    if "=" in text or "," in text or ";" in text:
        raise ValueError(
            "--parameter-input-scales accepts one positive scale applied to every "
            "model parameter, for example 1.0 or 1um"
        )
    scale = parse_scale_number(text)
    return {name: scale for name in names}


def parse_data_number(token: str) -> float:
    parsed = parse_number(token)
    if parsed is None:
        raise ValueError(f"Could not parse numeric token {token!r}")
    return parsed


def parse_var_line(line: str) -> tuple[str, str] | None:
    body = line.strip()[3:].strip()
    if not body:
        return None
    if "=" in body:
        name, value = body.split("=", 1)
    else:
        parts = body.split(None, 1)
        if len(parts) != 2:
            return None
        name, value = parts
    return normalize_name(name), strip_quotes(value)


def strip_inline_comment(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("!"):
        return ""
    return line.split("!", 1)[0].strip()


def is_numeric_line(line: str) -> bool:
    if not line:
        return False
    first = line.split()[0]
    return parse_number(first) is not None


def sparam_base_name(name: str) -> str | None:
    """Return normalized Sij label from column names such as S11R or S[2,1]."""

    raw = name.strip().strip(",")
    raw = raw.replace("(", "[").replace(")", "]")
    raw = re.sub(r"_(?:re|real|r|im|imag|i)$", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(?:re|real|r|im|imag|i)$", "", raw, flags=re.IGNORECASE)
    match = re.search(r"S\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]", raw, re.IGNORECASE)
    if match:
        return f"S{int(match.group(1))}{int(match.group(2))}"
    match = re.search(r"S\s*[_-]?(\d+)\s*[_,-]\s*(\d+)", raw, re.IGNORECASE)
    if match:
        return f"S{int(match.group(1))}{int(match.group(2))}"
    match = re.search(r"S\s*(\d)(\d)$", raw, re.IGNORECASE)
    if match:
        return f"S{int(match.group(1))}{int(match.group(2))}"
    return None


def column_suffix(name: str) -> str | None:
    lowered = name.strip().lower()
    if re.search(r"(?:_re|_real|r)$", lowered):
        return "real"
    if re.search(r"(?:_im|_imag|i)$", lowered):
        return "imag"
    return None


def infer_sparam_labels(num_pairs: int) -> list[str]:
    nports = int(round(math.sqrt(num_pairs)))
    if nports * nports != num_pairs:
        return [f"S{k + 1}" for k in range(num_pairs)]
    labels = []
    for row in range(1, nports + 1):
        for col in range(1, nports + 1):
            labels.append(f"S{row}{col}")
    return labels


def combine_complex(real: np.ndarray, imag_or_angle: np.ndarray, fmt: str) -> np.ndarray:
    fmt = fmt.upper()
    if fmt == "RI":
        return real + 1j * imag_or_angle
    if fmt == "MA":
        return real * np.exp(1j * np.deg2rad(imag_or_angle))
    if fmt == "DB":
        return 10.0 ** (real / 20.0) * np.exp(1j * np.deg2rad(imag_or_angle))
    raise ValueError(f"Unsupported S-parameter numeric format {fmt!r}")


def parse_table(
    rows: list[list[float]],
    column_names: list[str] | None,
    freq_scale: float,
    data_format: str,
    params: dict[str, str],
    source_index: int,
) -> MDIFBlock:
    if not rows:
        raise ValueError(f"MDIF block {source_index} has no numeric rows")
    width = len(rows[0])
    if width < 3:
        raise ValueError(f"MDIF block {source_index} must have frequency plus S data")
    for row in rows:
        if len(row) != width:
            raise ValueError(
                f"MDIF block {source_index} has ragged data rows: {len(row)} vs {width}"
            )

    data = np.asarray(rows, dtype=float)
    freq_hz = data[:, 0] * freq_scale
    rest = data[:, 1:]

    if column_names:
        names = [normalize_name(x) for x in column_names]
        if names and names[0].lower() in {"freq", "frequency", "f"}:
            names = names[1:]

        # Header has one logical S-parameter name per real/imag pair.
        if len(names) * 2 == rest.shape[1]:
            sparams: dict[str, np.ndarray] = {}
            for idx, name in enumerate(names):
                label = sparam_base_name(name) or f"S{idx + 1}"
                sparams[label] = combine_complex(
                    rest[:, 2 * idx], rest[:, 2 * idx + 1], data_format
                )
            return MDIFBlock(params=params, freq_hz=freq_hz, sparams=sparams, source_index=source_index)

        # Header has explicit real/imag columns such as S11R S11I.
        if len(names) == rest.shape[1]:
            grouped: dict[str, dict[str, np.ndarray]] = {}
            for idx, name in enumerate(names):
                label = sparam_base_name(name)
                suffix = column_suffix(name)
                if label and suffix:
                    grouped.setdefault(label, {})[suffix] = rest[:, idx]
            if grouped and all({"real", "imag"} <= set(parts) for parts in grouped.values()):
                sparams = {
                    label: combine_complex(parts["real"], parts["imag"], data_format)
                    for label, parts in grouped.items()
                }
                return MDIFBlock(params=params, freq_hz=freq_hz, sparams=sparams, source_index=source_index)

    if rest.shape[1] % 2 != 0:
        raise ValueError(
            f"MDIF block {source_index} has {rest.shape[1]} S-data columns; expected real/imag pairs"
        )

    labels = infer_sparam_labels(rest.shape[1] // 2)
    sparams = {}
    for idx, label in enumerate(labels):
        sparams[label] = combine_complex(rest[:, 2 * idx], rest[:, 2 * idx + 1], data_format)
    return MDIFBlock(params=params, freq_hz=freq_hz, sparams=sparams, source_index=source_index)


def update_option_line(line: str, freq_scale: float, data_format: str) -> tuple[float, str]:
    cleaned = line.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:].strip()
    tokens = [tok.strip() for tok in cleaned.split()]
    for token in tokens:
        lowered = token.lower()
        if lowered in FREQ_UNITS:
            freq_scale = FREQ_UNITS[lowered]
        uppered = token.upper()
        if uppered in {"RI", "MA", "DB"}:
            data_format = uppered
    return freq_scale, data_format


def read_mdif(path: Path) -> list[MDIFBlock]:
    """Read a generic S-parameter MDIF with VAR metadata and ACDATA blocks."""

    blocks: list[MDIFBlock] = []
    pending_params: dict[str, str] = {}
    block_params: dict[str, str] | None = None
    column_names: list[str] | None = None
    rows: list[list[float]] = []
    freq_scale = 1.0
    data_format = "RI"
    in_block = False

    for raw in path.read_text(errors="ignore").splitlines():
        line = strip_inline_comment(raw)
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("VAR"):
            parsed = parse_var_line(line)
            if parsed:
                name, value = parsed
                if in_block and block_params is not None:
                    block_params[name] = value
                else:
                    pending_params[name] = value
            continue

        if upper.startswith("BEGIN"):
            in_block = True
            block_params = dict(pending_params)
            column_names = None
            rows = []
            freq_scale = 1.0
            data_format = "RI"
            continue

        if upper.startswith("END") and in_block:
            assert block_params is not None
            blocks.append(
                parse_table(
                    rows=rows,
                    column_names=column_names,
                    freq_scale=freq_scale,
                    data_format=data_format,
                    params=block_params,
                    source_index=len(blocks),
                )
            )
            in_block = False
            block_params = None
            rows = []
            column_names = None
            continue

        if not in_block:
            continue

        if line.startswith("%"):
            column_names = line[1:].strip().split()
            continue

        if line.startswith("#"):
            freq_scale, data_format = update_option_line(line, freq_scale, data_format)
            continue

        first = line.split()[0].lower()
        if first in FREQ_UNITS:
            freq_scale, data_format = update_option_line(line, freq_scale, data_format)
            continue

        if not is_numeric_line(line):
            possible_header = line.split()
            if any(sparam_base_name(tok) for tok in possible_header):
                column_names = possible_header
            continue

        rows.append([parse_data_number(tok) for tok in line.split()])

    if in_block:
        raise ValueError(f"Unterminated MDIF data block in {path}")
    if not blocks:
        raise ValueError(f"No MDIF data blocks found in {path}")
    return blocks


def write_mdif(path: Path, blocks: Sequence[MDIFBlock], sparam_labels: Sequence[str]) -> None:
    lines = [
        "! Neuro-TF predicted S-parameter MDIF",
        f"! Generated by neuro_tf.py {VERSION}",
        "",
    ]
    for block in blocks:
        for name in sorted(block.params):
            value = block.params[name]
            lines.append(f"VAR {name}={value}")
        lines.append("BEGIN ACDATA")
        lines.append("% Freq " + " ".join(sparam_labels))
        lines.append("# Hz S RI R 50")
        for idx, freq in enumerate(block.freq_hz):
            row = [f"{freq:.12g}"]
            for label in sparam_labels:
                value = block.sparams[label][idx]
                row.append(f"{value.real:.16g}")
                row.append(f"{value.imag:.16g}")
            lines.append(" ".join(row))
        lines.append("END")
        lines.append("")
    path.write_text("\n".join(lines))


def split_number_unit(value: str) -> tuple[float, str] | None:
    text = strip_quotes(value).strip()
    match = re.match(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)\s*$",
        text,
    )
    if not match:
        return None
    return float(match.group(1)), match.group(2)


def parse_ads_grid_spec(spec: str) -> tuple[str, list[str]]:
    if "=" not in spec:
        raise ValueError(f"Grid spec must look like NAME=v1,v2 or NAME=start:stop:count, got {spec!r}")
    name, body = spec.split("=", 1)
    name = normalize_name(name)
    body = body.strip()
    if not name or not body:
        raise ValueError(f"Invalid grid spec {spec!r}")
    if ":" in body:
        parts = [part.strip() for part in body.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Range grid spec must look like NAME=start:stop:count, got {spec!r}")
        start_text, stop_text, count_text = parts
        count = int(count_text)
        if count <= 0:
            raise ValueError(f"Range grid count must be positive in {spec!r}")
        start = parse_number(start_text)
        stop = parse_number(stop_text)
        if start is None or stop is None:
            raise ValueError(f"Could not parse range endpoints in {spec!r}")
        start_parts = split_number_unit(start_text)
        stop_parts = split_number_unit(stop_text)
        unit = ""
        scale = 1.0
        if start_parts and stop_parts and start_parts[1] == stop_parts[1]:
            unit = start_parts[1]
            scale = VALUE_UNITS.get(unit.lower(), FREQ_UNITS.get(unit.lower(), 1.0))
        values = np.linspace(start, stop, count)
        return name, [f"{value / scale:.12g}{unit}" for value in values]
    values = [part.strip() for part in body.split(",") if part.strip()]
    if not values:
        raise ValueError(f"Grid spec has no values: {spec!r}")
    for value in values:
        if parse_number(value) is None:
            raise ValueError(f"Could not parse grid value {value!r} in {spec!r}")
    return name, values


def parse_ads_frequency_values(spec: str) -> np.ndarray:
    text = spec.strip()
    if not text:
        raise ValueError("--freqs cannot be empty")
    if ":" in text:
        parts = [part.strip() for part in text.split(":")]
        if len(parts) != 3:
            raise ValueError("--freqs range must look like START:STOP:COUNT")
        start = parse_number(parts[0])
        stop = parse_number(parts[1])
        count = int(parts[2])
        if start is None or stop is None:
            raise ValueError(f"Could not parse frequency range {spec!r}")
        if count <= 0:
            raise ValueError("Frequency count must be positive")
        return np.linspace(start, stop, count)
    values = [parse_number(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(value is None for value in values):
        raise ValueError(f"Could not parse frequency list {spec!r}")
    return np.asarray(values, dtype=float)


def ensure_block_sparams(blocks: Sequence[MDIFBlock], labels: Sequence[str]) -> list[MDIFBlock]:
    fixed = []
    for block in blocks:
        sparams = dict(block.sparams)
        for label in labels:
            if label not in sparams:
                sparams[label] = np.zeros_like(block.freq_hz, dtype=complex)
        fixed.append(
            MDIFBlock(
                params=dict(block.params),
                freq_hz=block.freq_hz.copy(),
                sparams=sparams,
                source_index=block.source_index,
            )
        )
    return fixed


def positive_frequency_blocks(
    blocks: Sequence[MDIFBlock],
    *,
    purpose: str = "model fitting",
) -> list[MDIFBlock]:
    """Copy blocks with DC/nonphysical frequency rows excluded from a fitted model."""

    positive_blocks: list[MDIFBlock] = []
    for block in blocks:
        mask = np.asarray(block.freq_hz, dtype=float) > 0.0
        if not np.any(mask):
            raise ValueError(
                f"MDIF block {block.source_index} has no positive-frequency samples for {purpose}"
            )
        positive_blocks.append(
            MDIFBlock(
                params=dict(block.params),
                freq_hz=np.asarray(block.freq_hz, dtype=float)[mask].copy(),
                sparams={
                    label: np.asarray(values, dtype=complex)[mask].copy()
                    for label, values in block.sparams.items()
                },
                source_index=block.source_index,
            )
        )
    return positive_blocks


def validate_dc_equivalent_resistance(
    value: float | None,
    *,
    context: str = "model",
) -> float:
    """Return a valid stored DC equivalent resistance."""

    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(
            f"{context} does not contain a valid data-derived DC equivalent resistance. "
            "For model export, provide exact-zero-frequency data with --dc-mdif; "
            "the fitted RF model does not need to be retrained."
        )
    return float(value)


def validate_exact_dc_source_kind(
    value: object,
    *,
    context: str = "model",
) -> str:
    """Require proof that DC came only from exact zero-frequency data rows."""

    source_kind = str(value or "").strip()
    if source_kind != "exact_zero_frequency":
        raise ValueError(
            f"{context} does not contain a DC response extracted exclusively from exact "
            "zero-Hz data. For model export, provide that data with --dc-mdif; the fitted "
            "RF model does not need to be retrained. Lowest-positive-frequency fallback "
            "and RF extrapolation are not allowed."
        )
    return source_kind


def _block_s_matrix(
    block: MDIFBlock,
    labels: Sequence[str],
    frequency_index: int,
    nports: int,
) -> np.ndarray:
    matrix = np.zeros((nports, nports), dtype=complex)
    for label in labels:
        indices = sparam_indices(label)
        if indices is None:
            raise ValueError(f"DC resistance extraction requires Sij labels, got {label!r}")
        row, col = indices
        matrix[row - 1, col - 1] = complex(block.sparams[label][frequency_index])
    return matrix


def _dc_pair_resistance_from_s(
    s_matrix: np.ndarray,
    differential_current: np.ndarray,
    z0: float,
) -> float:
    """Return the open-port differential resistance, or infinity if disconnected."""

    identity = np.eye(s_matrix.shape[0], dtype=complex)
    lhs = identity - s_matrix
    rhs = float(z0) * (identity + s_matrix) @ differential_current
    try:
        voltage = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return math.nan
    residual = float(
        np.linalg.norm(lhs @ voltage - rhs)
        / max(np.linalg.norm(rhs), EPS)
    )
    if residual > 1e-7:
        return math.inf
    return float(np.real(np.conjugate(differential_current) @ voltage))


def _resistance_from_mean_conductance(
    conductances: Sequence[float],
    *,
    open_threshold_ohm: float,
    open_resistance_ohm: float,
) -> float:
    """Return a finite equivalent resistance from parallel-path samples."""

    mean_conductance = float(np.mean(np.asarray(conductances, dtype=float)))
    if (
        mean_conductance <= 0.0
        or mean_conductance < 1.0 / float(open_threshold_ohm)
    ):
        return float(open_resistance_ohm)
    return min(1.0 / mean_conductance, float(open_resistance_ohm))


def parse_dc_port_paths(
    spec: object,
    nports: int,
) -> list[tuple[int, int | None, str]]:
    """Parse explicit DC paths into zero-based endpoints and canonical names.

    Port-to-port paths accept forms such as ``1-2`` or ``p1-p2``. A shunt path
    to the simulator reference accepts ``1-ground`` or ``p1-gnd``. When no
    specification is supplied, this legacy path-parser helper selects every port
    pair and every port-to-ground path. Current fitting/export CLIs bypass that
    fallback and fit the full ordered complex S matrix when ``--dc-port-paths``
    is omitted.
    """

    if nports <= 0:
        raise ValueError("DC port-path parsing requires at least one port")
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        raw_paths = [
            *[
                f"{first + 1}-{second + 1}"
                for first in range(nports)
                for second in range(first + 1, nports)
            ],
            *[f"{port + 1}-ground" for port in range(nports)],
        ]
    elif isinstance(spec, str):
        raw_paths = [item.strip() for item in re.split(r"[,;]", spec) if item.strip()]
    elif isinstance(spec, dict):
        raw_paths = [str(item) for item in spec]
    elif isinstance(spec, Sequence):
        raw_paths = [str(item) for item in spec]
    else:
        raise ValueError(
            "DC port paths must be a comma-separated string such as 1-2,3-4"
        )
    if not raw_paths:
        raise ValueError("At least one viable DC port path must be specified")

    parsed: list[tuple[int, int | None, str]] = []
    seen: set[str] = set()
    for raw_path in raw_paths:
        normalized = str(raw_path).strip().lower().replace("port", "p")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = normalized.replace("ground", "gnd")
        match = re.fullmatch(r"p?(\d+)(?:-|:|/)(?:p?(\d+)|(gnd|0))", normalized)
        if match is None:
            raise ValueError(
                f"Invalid DC port path {raw_path!r}; use forms such as 1-2, "
                "p3-p4, or 1-ground"
            )
        first_port = int(match.group(1))
        second_text = match.group(2)
        if first_port < 1 or first_port > nports:
            raise ValueError(
                f"DC port path {raw_path!r} references port {first_port}, but the "
                f"model has ports 1 through {nports}"
            )
        first = first_port - 1
        if second_text is None:
            second = None
            canonical = f"p{first_port}-ground"
        else:
            second_port = int(second_text)
            if second_port < 1 or second_port > nports:
                raise ValueError(
                    f"DC port path {raw_path!r} references port {second_port}, but the "
                    f"model has ports 1 through {nports}"
                )
            if second_port == first_port:
                raise ValueError(f"DC port path {raw_path!r} connects a port to itself")
            low, high = sorted((first_port, second_port))
            first = low - 1
            second = high - 1
            canonical = f"p{low}-p{high}"
        if canonical in seen:
            continue
        seen.add(canonical)
        parsed.append((first, second, canonical))
    return parsed


def dc_port_path_spec(paths: Sequence[str]) -> str:
    """Render canonical stored path names in a compact CLI form."""

    return ",".join(str(path).replace("p", "") for path in paths)


def validate_dc_port_resistances(
    labels: Sequence[str],
    values: object,
    *,
    context: str = "DC response",
) -> dict[str, float] | None:
    """Validate and canonicalize an optional per-path resistance mapping."""

    if values is None:
        return None
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{context} must contain at least one DC port-path resistance")
    nports = infer_complete_sparameter_ports(labels)
    result: dict[str, float] = {}
    for raw_path, raw_resistance in values.items():
        parsed = parse_dc_port_paths([str(raw_path)], nports)
        _, _, canonical = parsed[0]
        result[canonical] = validate_dc_equivalent_resistance(
            float(raw_resistance),
            context=f"{context} path {canonical}",
        )
    return result


def dc_conductance_matrix(
    labels: Sequence[str],
    dc_equivalent_resistance_ohm: float,
    dc_port_resistances_ohm: object = None,
) -> np.ndarray:
    """Return the real DC conductance matrix for legacy or selected-path topology."""

    resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context="DC response",
    )
    nports = infer_complete_sparameter_ports(labels)
    y_matrix = np.zeros((nports, nports), dtype=float)
    path_resistances = validate_dc_port_resistances(
        labels,
        dc_port_resistances_ohm,
    )
    if path_resistances is None:
        if nports == 1:
            y_matrix[0, 0] = 1.0 / resistance
        else:
            branch_conductance = 2.0 / (float(nports) * resistance)
            for first in range(nports):
                for second in range(first + 1, nports):
                    y_matrix[first, first] += branch_conductance
                    y_matrix[second, second] += branch_conductance
                    y_matrix[first, second] -= branch_conductance
                    y_matrix[second, first] -= branch_conductance
        return y_matrix

    for first, second, canonical in parse_dc_port_paths(path_resistances, nports):
        conductance = 1.0 / path_resistances[canonical]
        y_matrix[first, first] += conductance
        if second is not None:
            y_matrix[second, second] += conductance
            y_matrix[first, second] -= conductance
            y_matrix[second, first] -= conductance
    return y_matrix


def extract_average_dc_resistance(
    blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    *,
    z0: float = 50.0,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
    passivity_tolerance: float = DEFAULT_DC_PASSIVITY_TOLERANCE,
    port_paths: object = None,
) -> dict[str, object]:
    """Extract one distinct DC resistance from passive exact-DC data.

    Exact-zero rows are checked for S-matrix passivity before use. Passive rows
    contribute one equivalent conductance for every selected port path.
    Non-passive, non-finite, and electrically invalid results are ignored. Open
    samples contribute zero conductance instead of a large resistance sentinel,
    so they cannot numerically overwhelm connected samples. Each path's
    reciprocal mean conductance becomes its parameter-independent DC resistance.
    A path above ``open_threshold_ohm`` is represented by
    ``open_resistance_ohm``.
    Positive-frequency data is never used as a fallback.
    """

    if not math.isfinite(float(z0)) or float(z0) <= 0.0:
        raise ValueError("DC resistance extraction z0 must be positive and finite")
    if not math.isfinite(float(open_threshold_ohm)) or float(open_threshold_ohm) <= 0.0:
        raise ValueError("DC open-circuit threshold must be positive and finite")
    if (
        not math.isfinite(float(open_resistance_ohm))
        or float(open_resistance_ohm) <= float(open_threshold_ohm)
    ):
        raise ValueError(
            "DC open-circuit resistance must be finite and greater than the threshold"
        )
    if not math.isfinite(float(passivity_tolerance)) or float(passivity_tolerance) < 0.0:
        raise ValueError("DC passivity tolerance must be non-negative and finite")
    if not blocks:
        raise ValueError("DC resistance extraction requires at least one data block")
    nports = infer_complete_sparameter_ports(labels)
    selected_paths = parse_dc_port_paths(port_paths, nports)
    conductances: list[float] = []
    frequencies: list[float] = []
    contributing_blocks: set[int] = set()
    pair_conductances: dict[str, list[float]] = {}
    dc_row_count = 0
    missing_dc_block_count = 0
    ignored_nonpassive_count = 0
    ignored_nonfinite_count = 0
    ignored_invalid_resistance_count = 0
    open_resistance_sample_count = 0
    passivity_limit = 1.0 + float(passivity_tolerance)
    for block_index, block in enumerate(blocks):
        freq = np.asarray(block.freq_hz, dtype=float)
        exact_dc_indices = np.flatnonzero(freq == 0.0)
        if exact_dc_indices.size == 0:
            missing_dc_block_count += 1
            continue
        for raw_frequency_index in exact_dc_indices:
            frequency_index = int(raw_frequency_index)
            dc_row_count += 1
            try:
                s_matrix = _block_s_matrix(block, labels, frequency_index, nports)
            except (KeyError, ValueError):
                ignored_nonfinite_count += 1
                continue
            if not np.all(np.isfinite(s_matrix.real)) or not np.all(np.isfinite(s_matrix.imag)):
                ignored_nonfinite_count += 1
                continue
            try:
                max_sigma = float(np.max(np.linalg.svd(s_matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                ignored_nonfinite_count += 1
                continue
            if not math.isfinite(max_sigma):
                ignored_nonfinite_count += 1
                continue
            if max_sigma > passivity_limit:
                ignored_nonpassive_count += 1
                continue

            block_resistances: list[tuple[str, float]] = []
            for first, second, pair_name in selected_paths:
                differential = np.zeros(nports, dtype=complex)
                differential[first] = 1.0
                if second is not None:
                    differential[second] = -1.0
                resistance = _dc_pair_resistance_from_s(
                    s_matrix,
                    differential,
                    z0,
                )
                if math.isnan(resistance) or resistance <= 0.0:
                    ignored_invalid_resistance_count += 1
                    continue
                block_resistances.append((pair_name, resistance))
            if not block_resistances:
                continue
            for pair_name, resistance in block_resistances:
                is_open = (
                    not math.isfinite(float(resistance))
                    or float(resistance) > float(open_threshold_ohm)
                )
                conductance = 0.0 if is_open else 1.0 / float(resistance)
                conductances.append(conductance)
                pair_conductances.setdefault(pair_name, []).append(conductance)
                if is_open:
                    open_resistance_sample_count += 1
                frequencies.append(float(freq[frequency_index]))
            contributing_blocks.add(block_index)
    if dc_row_count == 0:
        raise ValueError(
            "DC extraction requires at least one exact zero-Hz data row. "
            "Positive-frequency data cannot be substituted for DC."
        )
    if not conductances:
        raise ValueError(
            "No usable passive exact-DC point remains after filtering non-passive, "
            "non-finite, and electrically invalid DC rows"
        )
    missing_selected_paths = [
        canonical
        for _, _, canonical in selected_paths
        if canonical not in pair_conductances
    ]
    if missing_selected_paths:
        raise ValueError(
            "No usable passive exact-DC resistance was found for selected path(s): "
            + ", ".join(missing_selected_paths)
        )
    mean_conductance = float(np.mean(np.asarray(conductances, dtype=float)))
    aggregate_open_circuit_applied = (
        mean_conductance <= 0.0
        or mean_conductance < 1.0 / float(open_threshold_ohm)
    )
    raw_resistance = (
        float(open_resistance_ohm)
        if mean_conductance <= 0.0
        else min(1.0 / mean_conductance, float(open_resistance_ohm))
    )
    resistance = (
        float(open_resistance_ohm)
        if aggregate_open_circuit_applied
        else raw_resistance
    )
    pair_means = {
        pair_name: _resistance_from_mean_conductance(
            values,
            open_threshold_ohm=float(open_threshold_ohm),
            open_resistance_ohm=float(open_resistance_ohm),
        )
        for pair_name, values in sorted(pair_conductances.items())
    }
    open_paths = [
        pair_name
        for pair_name, path_resistance in pair_means.items()
        if path_resistance == float(open_resistance_ohm)
    ]
    dc_values = dc_sparameter_values(
        labels,
        resistance,
        dc_port_resistances_ohm=pair_means,
        z0=z0,
    )
    return {
        "dc_equivalent_resistance_ohm": resistance,
        "dc_equivalent_resistance_raw_mean_ohm": raw_resistance,
        "dc_resistance_sample_count": len(conductances),
        "dc_open_resistance_sample_count": open_resistance_sample_count,
        "dc_mean_conductance_siemens": mean_conductance,
        "dc_resistance_block_count": len(contributing_blocks),
        "dc_row_count": dc_row_count,
        "dc_missing_block_count": missing_dc_block_count,
        "dc_ignored_nonpassive_count": ignored_nonpassive_count,
        "dc_ignored_nonfinite_count": ignored_nonfinite_count,
        "dc_ignored_invalid_resistance_count": ignored_invalid_resistance_count,
        "dc_passivity_tolerance": float(passivity_tolerance),
        "dc_open_threshold_ohm": float(open_threshold_ohm),
        "dc_open_resistance_ohm": float(open_resistance_ohm),
        "dc_open_circuit_applied": bool(open_paths),
        "dc_open_paths": open_paths,
        "dc_open_path_count": len(open_paths),
        "dc_aggregate_open_circuit_applied": aggregate_open_circuit_applied,
        "dc_resistance_source_z0_ohm": float(z0),
        "dc_resistance_source_frequency_min_hz": float(min(frequencies)),
        "dc_resistance_source_frequency_max_hz": float(max(frequencies)),
        "dc_resistance_source_kind": "exact_zero_frequency",
        "dc_port_paths": [canonical for _, _, canonical in selected_paths],
        "dc_port_path_spec": dc_port_path_spec(
            [canonical for _, _, canonical in selected_paths]
        ),
        "dc_port_resistances_ohm": pair_means,
        "dc_resistance_pair_means_ohm": pair_means,
        "dc_resistance_extraction": (
            "Independent reciprocal arithmetic-mean conductance for each selected "
            "exact-zero-Hz open-port path. Non-passive DC rows are ignored; open "
            "samples contribute zero conductance, and path resistances above the "
            "configured threshold use the configured open-circuit resistance"
        ),
        "dc_response_topology": (
            "Parameter-independent selected-path resistor graph. Only paths declared "
            "in dc_port_paths are connected; every undeclared path remains open"
        ),
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_sparameters": {
            label: {
                "real": float(value.real),
                "imag": float(value.imag),
            }
            for label, value in zip(labels, dc_values)
        },
    }


def resolve_export_dc_metadata(
    stored_metadata: dict[str, object],
    labels: Sequence[str],
    *,
    dc_mdif: str | Path | None,
    z0: float,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
    port_paths: object = None,
) -> dict[str, object]:
    """Resolve DC metadata without changing or refitting the RF model."""

    if dc_mdif:
        dc_path = Path(dc_mdif)
        blocks = read_mdif(dc_path)
        selected_paths = (
            port_paths
            if port_paths is not None
            else stored_metadata.get("dc_port_paths")
        )
        metadata = extract_average_dc_resistance(
            blocks,
            labels,
            z0=z0,
            open_threshold_ohm=open_threshold_ohm,
            open_resistance_ohm=open_resistance_ohm,
            port_paths=selected_paths,
        )
        metadata["dc_resistance_source_file"] = str(dc_path)
        metadata["dc_resistance_extracted_during_export"] = True
        return metadata

    validate_exact_dc_source_kind(
        stored_metadata.get("dc_resistance_source_kind"),
        context="Saved surrogate model",
    )
    validate_dc_equivalent_resistance(
        stored_metadata.get("dc_equivalent_resistance_ohm"),
        context="Saved surrogate model",
    )
    if port_paths is not None:
        nports = infer_complete_sparameter_ports(labels)
        requested = [item[2] for item in parse_dc_port_paths(port_paths, nports)]
        stored_paths = stored_metadata.get("dc_port_paths")
        if stored_paths is None:
            raise ValueError(
                "--dc-port-paths changes the saved DC topology. Supply --dc-mdif so "
                "the selected path resistances can be extracted without refitting RF."
            )
        resolved_stored = [
            item[2] for item in parse_dc_port_paths(stored_paths, nports)
        ]
        if set(requested) != set(resolved_stored):
            raise ValueError(
                "--dc-port-paths does not match the saved DC topology. Supply --dc-mdif "
                "to extract the newly selected path resistances without refitting RF."
            )
    return {
        key: value
        for key, value in stored_metadata.items()
        if str(key).startswith("dc_")
    }


def add_dc_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Add export-time exact-DC extraction controls."""

    parser.add_argument(
        "--dc-mdif",
        help=(
            "Exact-zero-Hz MDIF used to validate the saved geometry-dependent DC "
            "network. If it does not match, or the saved model is legacy, fit a new "
            "DC-only full-complex-S or explicit-path model for this export without "
            "refitting RF."
        ),
    )
    add_dc_port_paths_argument(parser)
    add_dc_threshold_arguments(parser)


def add_dc_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the selected-path open-circuit threshold controls."""

    parser.add_argument(
        "--dc-open-threshold",
        type=float,
        default=DEFAULT_DC_OPEN_THRESHOLD_OHM,
        help=(
            "A selected path conductance below the reciprocal of this resistance "
            "is treated as open. "
            f"Default: {DEFAULT_DC_OPEN_THRESHOLD_OHM:g} ohm."
        ),
    )
    parser.add_argument(
        "--dc-open-resistance",
        type=float,
        default=DEFAULT_DC_OPEN_RESISTANCE_OHM,
        help=(
            "Finite resistance used to represent an open DC path. "
            f"Default: {DEFAULT_DC_OPEN_RESISTANCE_OHM:g} ohm."
        ),
    )


def add_dc_fitting_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all exact-DC controls used during model fitting."""

    add_dc_port_paths_argument(parser)
    add_dc_threshold_arguments(parser)


def add_dc_port_paths_argument(parser: argparse.ArgumentParser) -> None:
    """Add the explicit viable-DC-path selector to a CLI parser."""

    parser.add_argument(
        "--dc-port-paths",
        help=(
            "Comma-separated viable DC paths, for example 1-2,3-4. Only declared "
            "paths are extracted and stamped; undeclared paths remain open. Use "
            "1-ground for a port-to-reference path. If omitted, fit both real and "
            "imaginary components of every ordered exact-DC Sij value instead of a "
            "resistor-path graph."
        ),
    )


def dc_sparameter_values(
    labels: Sequence[str],
    dc_equivalent_resistance_ohm: float,
    dc_port_resistances_ohm: object = None,
    *,
    z0: float = 50.0,
) -> np.ndarray:
    """Return the S-vector for the fixed DC resistor topology."""

    nports = infer_complete_sparameter_ports(labels)
    y_matrix = dc_conductance_matrix(
        labels,
        dc_equivalent_resistance_ohm,
        dc_port_resistances_ohm,
    ).astype(complex)
    identity = np.eye(nports, dtype=complex)
    normalized_y = float(z0) * y_matrix
    lhs = identity + normalized_y
    rhs = identity - normalized_y
    try:
        s_matrix = np.linalg.solve(lhs.T, rhs.T).T
    except np.linalg.LinAlgError:
        s_matrix = rhs @ np.linalg.pinv(lhs)
    values = []
    for label in labels:
        row, col = sparam_indices(label) or (0, 0)
        values.append(s_matrix[row - 1, col - 1])
    return np.asarray(values, dtype=complex)


def apply_distinct_dc_response(
    values: np.ndarray,
    freq_hz: np.ndarray,
    labels: Sequence[str],
    dc_equivalent_resistance_ohm: float | None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    *,
    z0: float = 50.0,
) -> np.ndarray:
    """Replace exact-DC rows without evaluating or extrapolating the fitted model."""

    dc_mask = np.asarray(freq_hz, dtype=float) == 0.0
    if not np.any(dc_mask):
        return values
    validate_exact_dc_source_kind(
        dc_resistance_source_kind,
        context="Saved surrogate model",
    )
    resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context="Saved surrogate model",
    )
    result = np.asarray(values, dtype=complex).copy()
    result[dc_mask, :] = dc_sparameter_values(
        labels,
        resistance,
        dc_port_resistances_ohm=dc_port_resistances_ohm,
        z0=z0,
    )[None, :]
    return result


def ensure_dc_frequency_point(blocks: Sequence[MDIFBlock]) -> list[MDIFBlock]:
    """Prepend an exact zero-Hz row to every sampled-export block when absent."""

    result: list[MDIFBlock] = []
    for block in blocks:
        freq = np.asarray(block.freq_hz, dtype=float)
        if np.any(freq == 0.0):
            result.append(block)
            continue
        result.append(
            MDIFBlock(
                params=dict(block.params),
                freq_hz=np.concatenate([np.asarray([0.0]), freq]),
                sparams={
                    label: np.concatenate(
                        [np.asarray([0.0 + 0.0j]), np.asarray(values, dtype=complex)]
                    )
                    for label, values in block.sparams.items()
                },
                source_index=block.source_index,
            )
        )
    return result


def build_ads_export_blocks(
    template_mdif: str | None,
    parameter_grid_specs: Sequence[str],
    freqs_spec: str | None,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
) -> list[MDIFBlock]:
    if template_mdif:
        return ensure_dc_frequency_point(
            ensure_block_sparams(read_mdif(Path(template_mdif)), sparam_labels)
        )
    if not parameter_grid_specs:
        raise ValueError("Provide --template-mdif or at least one --parameter-grid")
    if not freqs_spec:
        raise ValueError("--freqs is required when using --parameter-grid")

    grid: dict[str, list[str]] = {}
    for spec in parameter_grid_specs:
        name, values = parse_ads_grid_spec(spec)
        grid[name] = values
    missing = [name for name in parameter_names if name not in grid]
    if missing:
        raise ValueError(f"Missing --parameter-grid for model parameter(s): {', '.join(missing)}")

    freqs = parse_ads_frequency_values(freqs_spec)
    blocks = []
    source_index = 1
    grid_values = [grid[name] for name in parameter_names]
    for values in itertools.product(*grid_values):
        params = dict(zip(parameter_names, values))
        blocks.append(
            MDIFBlock(
                params=params,
                freq_hz=freqs.copy(),
                sparams={label: np.zeros_like(freqs, dtype=complex) for label in sparam_labels},
                source_index=source_index,
            )
        )
        source_index += 1
    return ensure_dc_frequency_point(blocks)


def ads_export_readme(
    model_kind: str,
    mdif_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    n_blocks: int,
    n_freqs: int,
    extra_notes: Sequence[str] | None = None,
) -> str:
    params = ", ".join(f"`{name}`" for name in parameter_names)
    labels = ", ".join(f"`{label}`" for label in sparam_labels)
    notes = "\n".join(f"- {note}" for note in (extra_notes or []))
    if notes:
        notes = "\n\nNotes:\n\n" + notes + "\n"
    return f"""# ADS Surrogate Export

This package contains an ADS-facing S-parameter MDIF export generated from the
trained {model_kind} surrogate.

## Files

- `{mdif_name}`: predicted S-parameter MDIF table
- `ads_model_manifest.json`: export metadata and model dimensions
- `ADS_README.md`: this usage note

## Contents

- Geometry/process parameters: {params}
- S-parameters: {labels}
- Exported parameter blocks: `{n_blocks}`
- Frequency samples per block: `{n_freqs}`

## ADS Usage

Copy `{mdif_name}` into your ADS workspace, typically under the workspace
`data` directory, then use an ADS data-based n-port / data access component that
can read MDIF S-parameter data. Set the component file to `{mdif_name}` and use
schematic variables with the same names as the MDIF `VAR`s: {params}.

For optimization, expose those same variables in ADS and constrain the optimizer
inside the exported grid. ADS interpolation is only as good as the exported
parameter and frequency sampling density, so use a dense enough grid around the
intended optimization region.

If your ADS flow expects a dataset instead of MDIF, use the ADS data tools or
Data File Tool to convert/import this MDIF into the workspace data format.
{notes}
"""


def write_ads_export_package(
    out_dir: Path,
    model_kind: str,
    model_dir: Path,
    mdif_name: str,
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    extra_manifest: dict[str, object] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dc_point: dict[str, object] | None = None
    if blocks:
        zero_indices = np.flatnonzero(np.asarray(blocks[0].freq_hz, dtype=float) == 0.0)
        if zero_indices.size:
            zero_index = int(zero_indices[0])
            dc_point = {
                "frequency_hz": 0.0,
                "sparameters": {
                    label: {
                        "real": float(complex(blocks[0].sparams[label][zero_index]).real),
                        "imag": float(complex(blocks[0].sparams[label][zero_index]).imag),
                    }
                    for label in sparam_labels
                },
                "parameter_independent": True,
            }
    manifest: dict[str, object] = {
        "format": "ads_sparameter_mdif_surrogate",
        "model_kind": model_kind,
        "model_dir": str(model_dir),
        "mdif": mdif_name,
        "parameter_names": list(parameter_names),
        "sparam_labels": list(sparam_labels),
        "blocks": len(blocks),
        "frequency_points_per_block": int(len(blocks[0].freq_hz)) if blocks else 0,
        "dc_point": dc_point,
        "usage": "Use the MDIF with an ADS data-based n-port/data access component and schematic variables matching the MDIF VAR names.",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / "ads_model_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "ADS_README.md").write_text(
        ads_export_readme(
            model_kind=model_kind,
            mdif_name=mdif_name,
            parameter_names=parameter_names,
            sparam_labels=sparam_labels,
            n_blocks=len(blocks),
            n_freqs=int(len(blocks[0].freq_hz)) if blocks else 0,
            extra_notes=extra_notes,
        )
    )
    return manifest


def sparameter_real_imag_columns(labels: Sequence[str], prefix: str = "") -> list[str]:
    clean_prefix = f"{normalize_name(prefix)}_" if prefix else ""
    return [f"{clean_prefix}{label}_real" for label in labels] + [
        f"{clean_prefix}{label}_imag" for label in labels
    ]


def frequency_feature_columns(freq_transform: str) -> list[str]:
    if freq_transform == "log":
        return ["freq_log10_hz"]
    if freq_transform == "linear":
        return ["freq_hz"]
    if freq_transform == "log-linear":
        return ["freq_log10_hz", "freq_hz"]
    raise ValueError(f"Unsupported frequency transform {freq_transform!r}")


def write_ads_ann_csv(
    path: Path,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    if x_values.shape[1] != len(input_columns):
        raise ValueError(
            f"Input matrix has {x_values.shape[1]} columns, but {len(input_columns)} input names were supplied"
        )
    if y_values.shape[1] != len(output_columns):
        raise ValueError(
            f"Output matrix has {y_values.shape[1]} columns, but {len(output_columns)} output names were supplied"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([x_values, y_values])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*input_columns, *output_columns])
        for row in data:
            writer.writerow([f"{float(value):.17g}" for value in row])


def ads_ann_activation_enum(activation: str) -> str:
    normalized = activation.strip().lower().replace("_", "-")
    if normalized in {"tanh", "hyperbolic-tangent", "hyperbolic tangent"}:
        return "HYPERBOLIC_TANGENT"
    if normalized == "relu":
        return "RELU"
    if normalized == "sigmoid":
        return "SIGMOID"
    raise ValueError(f"Unsupported ADS ANN activation {activation!r}")


def ads_ann_optimizer_enum(optimizer: str) -> str:
    normalized = optimizer.strip().lower().replace("_", "-")
    if normalized in {"quasi-newton", "quasinewton", "qn"}:
        return "QUASI_NEWTON"
    if normalized in {"bayesian-regularization", "bayesian", "br"}:
        return "BAYESIAN_REGULARIZATION"
    raise ValueError(f"Unsupported ADS ANN optimizer {optimizer!r}")


def ads_ann_output_format_enum(output_format: str) -> str:
    normalized = output_format.strip().lower().replace("_", "-")
    aliases = {
        "all": "ALL",
        "c": "C_CODE",
        "c-code": "C_CODE",
        "equation": "TEXT_FORMULA",
        "text-formula": "TEXT_FORMULA",
        "struct-scale": "STRUC_AND_SCALE",
        "struc-scale": "STRUC_AND_SCALE",
        "struc-and-scale": "STRUC_AND_SCALE",
        "veriloga": "VERILOG_A_FORMAT",
        "verilog-a": "VERILOG_A_FORMAT",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported ADS ANN output format {output_format!r}")
    return aliases[normalized]


def ads_ann_training_type_enum(training_type: str) -> str:
    normalized = training_type.strip().lower().replace("_", "-")
    aliases = {
        "adjoint": "ADJOINT",
        "classification": "PATTERN_RECOGNITION_AND_CLASSIFICATION",
        "pattern-recognition-and-classification": "PATTERN_RECOGNITION_AND_CLASSIFICATION",
        "standard": "STANDARD",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported ADS ANN training type {training_type!r}")
    return aliases[normalized]


def ads_ann_expected_suffixes(output_format_enum: str) -> list[str]:
    suffixes = {
        "ALL": [".inc", ".c", ".equation", ".scale", ".struc"],
        "C_CODE": [".c"],
        "STRUC_AND_SCALE": [".scale", ".struc"],
        "TEXT_FORMULA": [".equation"],
        "VERILOG_A_FORMAT": [".inc"],
    }
    if output_format_enum not in suffixes:
        raise ValueError(f"Unsupported ADS ANN output enum {output_format_enum!r}")
    return suffixes[output_format_enum]


def read_model_metadata(model_dir: str | None) -> dict[str, object]:
    if not model_dir:
        return {}
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Could not find metadata.json in model directory {model_dir!r}")
    return json.loads(metadata_path.read_text())


def metadata_hidden_layers(metadata: dict[str, object]) -> str | None:
    layer_sizes = metadata.get("layer_sizes")
    if not isinstance(layer_sizes, list) or len(layer_sizes) < 3:
        return None
    hidden = layer_sizes[1:-1]
    if not all(isinstance(value, int) for value in hidden):
        return None
    return ",".join(str(value) for value in hidden)


def metadata_csv(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, list) and value:
        return ",".join(str(item) for item in value)
    return None


def infer_uniform_hidden_layout(hidden_layers: Sequence[int]) -> tuple[int, int]:
    if not hidden_layers:
        return 1, 20
    if any(layer <= 0 for layer in hidden_layers):
        raise ValueError("Hidden layer sizes must be positive")
    neurons = int(round(float(np.mean(hidden_layers))))
    return len(hidden_layers), max(1, neurons)


def ads_ann_training_script() -> str:
    return """#!/usr/bin/env python3
from pathlib import Path
import json
import os

import pandas as pd
import keysight.ads.ann as ann


PACKAGE_DIR = Path(__file__).resolve().parent
os.chdir(PACKAGE_DIR)


def enum_value(enum_cls, name):
    return getattr(enum_cls, name)


def main():
    manifest = json.loads((PACKAGE_DIR / "ads_ann_manifest.json").read_text())
    settings = manifest["ads_ann"]
    input_columns = manifest["input_columns"]
    output_columns = manifest["output_columns"]
    training_df = pd.read_csv(PACKAGE_DIR / manifest["training_csv"])

    setup = ann.AnnSetup(len(input_columns), len(output_columns))
    setup.seed = int(settings["seed"])
    setup.num_hidden_layers = int(settings["num_hidden_layers"])
    setup.num_neurons_per_layer = int(settings["num_neurons_per_layer"])
    setup.neuron_activation_function_type = enum_value(
        ann.NeuronActivationFunctionType,
        settings["neuron_activation_function_type"],
    )
    setup.output_activation_function_type = ann.OutputActivationFunctionType.LINEAR
    setup.network_training_type = enum_value(
        ann.NetworkTrainingType,
        settings["network_training_type"],
    )
    setup.modeler_optimizer = enum_value(
        ann.ModelerOptimizer,
        settings["modeler_optimizer"],
    )
    setup.max_training_iterations = int(settings["max_training_iterations"])
    setup.training_stop_tolerance = float(settings["training_stop_tolerance"])
    setup.output_format = enum_value(ann.OutputFormat, settings["output_format"])

    ann.configure_setup(setup)
    training_fit = ann.auxiliary_functions.extract_inmemory(
        input_data=training_df,
        input_columns=input_columns,
        output_columns=output_columns,
        ann_saving_names=settings["output_prefix"],
    )
    training_fit.to_csv(PACKAGE_DIR / "ads_ann_training_fit.csv", index=False)

    verification_csv = manifest.get("verification_csv")
    if verification_csv:
        struc_file = PACKAGE_DIR / f"{settings['output_prefix']}.struc"
        if struc_file.exists():
            fresh_setup = ann.AnnSetup(1, 1)
            fresh_setup.existing_file = str(struc_file)
            ann.configure_setup(fresh_setup)
        verify_df = pd.read_csv(PACKAGE_DIR / verification_csv)
        prediction_df = ann.auxiliary_functions.simulate_inmemory(
            verify_df[input_columns],
            input_columns=None,
        )
        if len(prediction_df.columns) == len(output_columns):
            prediction_df.columns = [f"pred_{name}" for name in output_columns]
        truth_df = verify_df[output_columns].rename(columns={name: f"truth_{name}" for name in output_columns})
        result_df = pd.concat(
            [verify_df[input_columns].reset_index(drop=True), truth_df.reset_index(drop=True), prediction_df.reset_index(drop=True)],
            axis=1,
        )
        result_df.to_csv(PACKAGE_DIR / "ads_ann_verification_prediction.csv", index=False)

    expected = [f"{settings['output_prefix']}{suffix}" for suffix in settings["expected_output_suffixes"]]
    print(json.dumps({
        "package_dir": str(PACKAGE_DIR),
        "input_columns": input_columns,
        "output_columns": output_columns,
        "expected_native_outputs": expected,
    }, indent=2))


if __name__ == "__main__":
    main()
"""


def ads_ann_readme(
    model_kind: str,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    settings: dict[str, object],
    training_rows: int,
    verification_rows: int,
    target_description: str,
    extra_notes: Sequence[str] | None = None,
) -> str:
    inputs = ", ".join(f"`{name}`" for name in input_columns)
    outputs = ", ".join(f"`{name}`" for name in output_columns)
    expected_files = ", ".join(
        f"`{settings['output_prefix']}{suffix}`"
        for suffix in settings["expected_output_suffixes"]
    )
    notes = "\n".join(f"- {note}" for note in (extra_notes or []))
    if notes:
        notes = "\n\nNotes:\n\n" + notes + "\n"
    return f"""# ADS ANN Export

This package contains a native ADS ANN training handoff for a {model_kind}
surrogate. It is intended to be run with the ADS Python interpreter on a
licensed ADS machine. The folder is self-contained; create it on any machine
that can parse the MDIF data, then copy the whole folder to the ADS machine.

## Files

- `ads_ann_training.csv`: numeric training table
- `ads_ann_verification.csv`: numeric verification table, when verification data is available
- `ads_ann_manifest.json`: column names, model settings, and target metadata
- `train_ads_ann.py`: ADS Python script that trains and extracts the native ANN
- `ADS_ANN_README.md`: this usage note

## ADS Reference Used

The generated `train_ads_ann.py` follows Keysight's ADS 2026 Update 2.1
ANN Python Documentation example:

- `doc/ann/examples/inmemory_extraction.py`
- HTML page: `doc/ann/html/examples/ex_inmemory_extraction.html`

That example establishes the in-memory workflow used here: create a pandas
DataFrame, configure `keysight.ads.ann.AnnSetup`, call
`ann.configure_setup(setup)`, train/extract with
`ann.auxiliary_functions.extract_inmemory(...)`, and re-load the generated
`.struc`/`.scale` files for verification with
`ann.auxiliary_functions.simulate_inmemory(...)`.

The generated setup fields are cross-checked against the ADS ANN API reference
pages:

- `doc/ann/html/reference/ann/annsetup.html`
- `doc/ann/html/reference/ann/index.html`
- `doc/ann/html/reference/ann/outputformat.html`
- `doc/ann/html/reference/ann/modeleroptimizer.html`
- `doc/ann/html/reference/ann/networktrainingtype.html`
- `doc/ann/html/reference/ann/neuronactivationfunctiontype.html`

If your installed ADS version is different, compare these pages against the
same paths under your installed `$HPEESOF_DIR/doc/ann` tree before production
use.

For weighted fitting context, Keysight's
`doc/ann/examples/training_error_weighting.py` demonstrates sample/row error
weighting. The documented example does not establish direct per-output
S-parameter loss weights for a multi-output ANN, so this package records
S-parameter weights in the manifest but does not apply them inside ADS ANN.

## Model

- Inputs: {inputs}
- Outputs: {outputs}
- Target: {target_description}
- Training rows: `{training_rows}`
- Verification rows: `{verification_rows}`
- ADS hidden layers: `{settings["num_hidden_layers"]}`
- ADS neurons per hidden layer: `{settings["num_neurons_per_layer"]}`
- ADS activation: `{settings["neuron_activation_function_type"]}`
- ADS optimizer: `{settings["modeler_optimizer"]}`
- ADS output format: `{settings["output_format"]}`

## Run in ADS Python

Run:

```bash
python train_ads_ann.py
```

For the selected `output_format`, ADS ANN is expected to write {expected_files}.
The `.inc` file is the Verilog-A-oriented artifact, `.c` is the C-oriented
artifact, `.equation` is the text equation artifact, and `.struc`/`.scale` are
the native ADS ANN structure and scaling files.

If verification data exists, the script also writes
`ads_ann_verification_prediction.csv` using ADS ANN's native simulator.

## Use in an ADS Schematic

The generated ADS ANN files are an ANN evaluator, not by themselves a complete
N-port schematic symbol. In ADS, use them through a small wrapper cell:

1. Run `train_ads_ann.py` on the ADS machine and keep the generated native ANN
   files with the ADS workspace, especially `{settings["output_prefix"]}.inc`.
2. Create an ADS wrapper component. For a Verilog-A wrapper, include/call the
   ANN evaluator from `{settings["output_prefix"]}.inc`. For an SDD or
   equation-based wrapper, use `{settings["output_prefix"]}.equation` as the
   ANN equation source/reference. The wrapper's input contract is the
   `input_columns` list in `ads_ann_manifest.json`; for example `freq_log10_hz`
   means the wrapper must feed `log10(freq_hz)`, while `freq_hz` means it must
   feed the raw simulator frequency in Hz.
3. Interpret the ANN outputs using the `output_columns` list in
   `ads_ann_manifest.json`. Outputs are grouped as all real S-parameter columns
   followed by the matching imaginary columns.
4. For a direct DNN model, or a KBNN exported with `--ads-ann-target fine`, the
   ANN outputs are the final fine S-parameters.
5. For a native residual KBNN export, the ANN outputs are `delta_S*`. The ADS
   wrapper must also evaluate or instantiate the coarse circuit response at the
   same parameter/frequency point, then use `fine_Sij = coarse_Sij + delta_Sij`.
6. To behave as a circuit N-port, the wrapper should convert the final complex
   S-matrix to a circuit relation before driving the pins. For a reference
   impedance `Z0`, a common small-signal conversion is
   `Y = (I - S) * inverse(I + S) / Z0`, then port currents are `Iport = Y * Vport`.

Start by validating the wrapper in an S-parameter or AC simulation against the
verification data before using it in optimization. If you want a simpler KBNN
schematic wrapper, export with `--ads-ann-target fine`; if you want to preserve
the residual KBNN formulation, keep `--ads-ann-target native` and include the
coarse-response addition in the wrapper.
{notes}
"""


def write_ads_ann_package(
    out_dir: Path,
    model_kind: str,
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_verify: np.ndarray | None,
    y_verify: np.ndarray | None,
    settings: dict[str, object],
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    target_description: str,
    extra_manifest: dict[str, object] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(settings)
    settings["expected_output_suffixes"] = ads_ann_expected_suffixes(str(settings["output_format"]))
    training_csv = "ads_ann_training.csv"
    verification_csv = "ads_ann_verification.csv" if x_verify is not None and y_verify is not None else None
    write_ads_ann_csv(out_dir / training_csv, input_columns, output_columns, x_train, y_train)
    if verification_csv:
        write_ads_ann_csv(out_dir / verification_csv, input_columns, output_columns, x_verify, y_verify)

    manifest: dict[str, object] = {
        "format": "ads_ann_training_package",
        "model_kind": model_kind,
        "training_csv": training_csv,
        "verification_csv": verification_csv,
        "input_columns": list(input_columns),
        "output_columns": list(output_columns),
        "parameter_names": list(parameter_names),
        "sparam_labels": list(sparam_labels),
        "training_rows": int(x_train.shape[0]),
        "verification_rows": int(x_verify.shape[0]) if x_verify is not None else 0,
        "target_description": target_description,
        "ads_ann": settings,
        "reference_note": (
            "Generated from the bundled ADS 2026 Update 2.1 ANN Python reference. "
            "Verify against the installed ADS release before production use."
        ),
        "ads_reference_used": {
            "release": "ADS 2026 Update 2.1 ANN Python Documentation",
            "primary_example": "doc/ann/examples/inmemory_extraction.py",
            "primary_example_html": "doc/ann/html/examples/ex_inmemory_extraction.html",
            "api_reference_pages": [
                "doc/ann/html/reference/ann/annsetup.html",
                "doc/ann/html/reference/ann/index.html",
                "doc/ann/html/reference/ann/outputformat.html",
                "doc/ann/html/reference/ann/modeleroptimizer.html",
                "doc/ann/html/reference/ann/networktrainingtype.html",
                "doc/ann/html/reference/ann/neuronactivationfunctiontype.html",
            ],
            "related_weighting_example": "doc/ann/examples/training_error_weighting.py",
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / "ads_ann_manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "train_ads_ann.py").write_text(ads_ann_training_script())
    (out_dir / "ADS_ANN_README.md").write_text(
        ads_ann_readme(
            model_kind=model_kind,
            input_columns=input_columns,
            output_columns=output_columns,
            settings=settings,
            training_rows=int(x_train.shape[0]),
            verification_rows=int(x_verify.shape[0]) if x_verify is not None else 0,
            target_description=target_description,
            extra_notes=extra_notes,
        )
    )
    return manifest


VERILOGA_RESERVED = {
    "above",
    "abs",
    "abstol",
    "access",
    "acos",
    "acosh",
    "aliasparam",
    "analog",
    "analysis",
    "and",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "begin",
    "branch",
    "case",
    "ceil",
    "connectrules",
    "connectmodule",
    "cos",
    "cosh",
    "ddt",
    "discipline",
    "domain",
    "else",
    "end",
    "endcase",
    "endconnectrules",
    "enddiscipline",
    "endfunction",
    "endmodule",
    "endnature",
    "exclude",
    "exp",
    "final_step",
    "flicker_noise",
    "floor",
    "for",
    "from",
    "function",
    "generate",
    "ground",
    "hypot",
    "idt",
    "idtmod",
    "if",
    "initial_step",
    "inout",
    "input",
    "integer",
    "laplace_nd",
    "laplace_np",
    "laplace_zd",
    "laplace_zp",
    "last_crossing",
    "limexp",
    "ln",
    "log",
    "max",
    "min",
    "module",
    "nature",
    "noise_table",
    "or",
    "output",
    "parameter",
    "potential",
    "pow",
    "real",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "while",
    "white_noise",
}


def veriloga_identifier(name: str, fallback: str) -> str:
    ident = normalize_name(name)
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"{fallback}_{ident}"
    if ident.lower() in VERILOGA_RESERVED:
        ident = f"{ident}_p"
    return ident


def unique_veriloga_identifiers(
    names: Sequence[str],
    fallback_prefix: str,
    used_names: set[str] | None = None,
) -> list[str]:
    used: set[str] = set(used_names or set())
    result: list[str] = []
    for idx, name in enumerate(names):
        base = veriloga_identifier(name, f"{fallback_prefix}{idx + 1}")
        ident = base
        suffix = 2
        while ident in used:
            ident = f"{base}_{suffix}"
            suffix += 1
        used.add(ident)
        result.append(ident)
    return result


def veriloga_float(value: float) -> str:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Cannot write non-finite Verilog-A constant {numeric!r}")
    if abs(numeric) < 1e-300:
        return "0.0"
    return f"{numeric:.17e}"


def infer_complete_sparameter_ports(labels: Sequence[str]) -> int:
    pairs: dict[tuple[int, int], str] = {}
    ports: set[int] = set()
    for label in labels:
        ij = sparam_indices(label)
        if ij is None:
            raise ValueError(f"Verilog-A N-port export requires Sij labels, got {label!r}")
        row, col = ij
        if row >= 10 or col >= 10:
            raise ValueError(
                "Verilog-A N-port export currently supports one-digit ADS S-parameter labels "
                f"(S11..S99), got {label!r}"
            )
        pairs[ij] = label
        ports.update({row, col})
    if not ports:
        raise ValueError("No S-parameter labels are available for Verilog-A export")
    nports = max(ports)
    expected_ports = set(range(1, nports + 1))
    if ports != expected_ports:
        raise ValueError(
            f"S-parameter labels must use contiguous ports 1..{nports}; found {sorted(ports)}"
        )
    missing = [
        f"S{row}{col}"
        for row in range(1, nports + 1)
        for col in range(1, nports + 1)
        if (row, col) not in pairs
    ]
    if missing:
        raise ValueError(
            "Verilog-A N-port export requires a complete S-matrix; missing "
            + ", ".join(missing)
        )
    return nports


def _veriloga_layer_assignments(
    lines: list[str],
    source: str,
    dest: str,
    weight: np.ndarray,
    bias: np.ndarray,
    activation: str | None,
) -> None:
    for out_idx in range(weight.shape[1]):
        lines.append(f"    {dest}[{out_idx}] = {veriloga_float(float(bias[out_idx]))};")
        for in_idx in range(weight.shape[0]):
            coeff = float(weight[in_idx, out_idx])
            if abs(coeff) < 1e-300:
                continue
            lines.append(
                f"    {dest}[{out_idx}] = {dest}[{out_idx}] + "
                f"({veriloga_float(coeff)})*{source}[{in_idx}];"
            )
        if activation == "tanh":
            lines.append(f"    if ({dest}[{out_idx}] > 40.0) begin")
            lines.append(f"      {dest}[{out_idx}] = 1.0;")
            lines.append(f"    end else if ({dest}[{out_idx}] < -40.0) begin")
            lines.append(f"      {dest}[{out_idx}] = -1.0;")
            lines.append("    end else begin")
            lines.append(f"      {dest}[{out_idx}] = tanh({dest}[{out_idx}]);")
            lines.append("    end")
        elif activation == "relu":
            lines.append(f"    if ({dest}[{out_idx}] < 0.0) begin")
            lines.append(f"      {dest}[{out_idx}] = 0.0;")
            lines.append("    end")
        elif activation is None:
            pass
        else:
            raise ValueError(f"Unsupported Verilog-A activation {activation!r}")
        lines.append("")


def _append_veriloga_s_to_y_conversion(
    lines: list[str],
    nports: int,
    *,
    s_real: str = "sr",
    s_imag: str = "si",
    y_real: str = "yr",
    y_imag: str = "yi",
) -> None:
    """Append the shared complex Y = (I-S) * inverse(I+S) / Z0 solve."""

    matrix_size = nports * nports
    lines.append(f"    for (i = 0; i < {matrix_size}; i = i + 1) begin")
    lines.append(f"      ar[i] = {s_real}[i];")
    lines.append(f"      ai[i] = {s_imag}[i];")
    lines.append(f"      mr[i] = -{s_real}[i];")
    lines.append(f"      mi[i] = -{s_imag}[i];")
    lines.append("      invr[i] = 0.0;")
    lines.append("      invi[i] = 0.0;")
    lines.append("    end")
    lines.append(f"    for (i = 0; i < {nports}; i = i + 1) begin")
    lines.append(f"      idx = i*{nports} + i;")
    lines.append("      ar[idx] = ar[idx] + 1.0;")
    lines.append("      mr[idx] = mr[idx] + 1.0;")
    lines.append("      invr[idx] = 1.0;")
    lines.append("    end")
    lines.append("")

    lines.append(f"    for (piv = 0; piv < {nports}; piv = piv + 1) begin")
    lines.append("      pivrow = piv;")
    lines.append(f"      idx = piv*{nports} + piv;")
    lines.append("      best_mag = ar[idx]*ar[idx] + ai[idx]*ai[idx];")
    lines.append(f"      for (row = piv + 1; row < {nports}; row = row + 1) begin")
    lines.append(f"        idx = row*{nports} + piv;")
    lines.append("        mag = ar[idx]*ar[idx] + ai[idx]*ai[idx];")
    lines.append("        if (mag > best_mag) begin")
    lines.append("          best_mag = mag;")
    lines.append("          pivrow = row;")
    lines.append("        end")
    lines.append("      end")
    lines.append("      if (pivrow != piv) begin")
    lines.append(f"        for (col = 0; col < {nports}; col = col + 1) begin")
    lines.append(f"          idx = piv*{nports} + col;")
    lines.append(f"          k = pivrow*{nports} + col;")
    lines.append(
        "          tr = ar[idx]; ti = ai[idx]; ar[idx] = ar[k]; "
        "ai[idx] = ai[k]; ar[k] = tr; ai[k] = ti;"
    )
    lines.append(
        "          tr = invr[idx]; ti = invi[idx]; invr[idx] = invr[k]; "
        "invi[idx] = invi[k]; invr[k] = tr; invi[k] = ti;"
    )
    lines.append("        end")
    lines.append("      end")
    lines.append(f"      idx = piv*{nports} + piv;")
    lines.append("      pr = ar[idx];")
    lines.append("      pi = ai[idx];")
    lines.append("      den = pr*pr + pi*pi;")
    lines.append("      if (den < pivot_floor) den = pivot_floor;")
    lines.append(f"      for (col = 0; col < {nports}; col = col + 1) begin")
    lines.append(f"        idx = piv*{nports} + col;")
    lines.append("        tr = (ar[idx]*pr + ai[idx]*pi)/den;")
    lines.append("        ti = (ai[idx]*pr - ar[idx]*pi)/den;")
    lines.append("        ar[idx] = tr; ai[idx] = ti;")
    lines.append("        tr = (invr[idx]*pr + invi[idx]*pi)/den;")
    lines.append("        ti = (invi[idx]*pr - invr[idx]*pi)/den;")
    lines.append("        invr[idx] = tr; invi[idx] = ti;")
    lines.append("      end")
    lines.append(f"      for (row = 0; row < {nports}; row = row + 1) begin")
    lines.append("        if (row != piv) begin")
    lines.append(f"          idx = row*{nports} + piv;")
    lines.append("          fr = ar[idx];")
    lines.append("          fi = ai[idx];")
    lines.append(f"          for (col = 0; col < {nports}; col = col + 1) begin")
    lines.append(f"            idx = row*{nports} + col;")
    lines.append(f"            k = piv*{nports} + col;")
    lines.append("            ar[idx] = ar[idx] - (fr*ar[k] - fi*ai[k]);")
    lines.append("            ai[idx] = ai[idx] - (fr*ai[k] + fi*ar[k]);")
    lines.append("            invr[idx] = invr[idx] - (fr*invr[k] - fi*invi[k]);")
    lines.append("            invi[idx] = invi[idx] - (fr*invi[k] + fi*invr[k]);")
    lines.append("          end")
    lines.append("        end")
    lines.append("      end")
    lines.append("    end")
    lines.append("")

    lines.append(f"    for (row = 0; row < {nports}; row = row + 1) begin")
    lines.append(f"      for (col = 0; col < {nports}; col = col + 1) begin")
    lines.append(f"        idx = row*{nports} + col;")
    lines.append(f"        i = row*{nports};")
    lines.append("        j = col;")
    lines.append(f"        {y_real}[idx] = mr[i]*invr[j] - mi[i]*invi[j];")
    lines.append(f"        {y_imag}[idx] = mr[i]*invi[j] + mi[i]*invr[j];")
    lines.append(f"        for (k = 1; k < {nports}; k = k + 1) begin")
    lines.append(f"          i = row*{nports} + k;")
    lines.append(f"          j = k*{nports} + col;")
    lines.append(
        f"          {y_real}[idx] = {y_real}[idx] + "
        "(mr[i]*invr[j] - mi[i]*invi[j]);"
    )
    lines.append(
        f"          {y_imag}[idx] = {y_imag}[idx] + "
        "(mr[i]*invi[j] + mi[i]*invr[j]);"
    )
    lines.append("        end")
    lines.append(f"        {y_real}[idx] = {y_real}[idx]/z0;")
    lines.append(f"        {y_imag}[idx] = {y_imag}[idx]/z0;")
    lines.append("      end")
    lines.append("    end")
    lines.append("")


def _append_veriloga_dc_or_fitted_y_assignments(
    lines: list[str],
    nports: int,
    fitted_real_by_flat: Sequence[str],
    fitted_imag_by_flat: Sequence[str],
    dc_real_by_flat: Sequence[str] | None = None,
    dc_imag_by_flat: Sequence[str] | None = None,
) -> None:
    """Select fixed DC or fitted Y coefficients without enclosing ``ddt``."""

    matrix_size = nports * nports
    if len(fitted_real_by_flat) != matrix_size or len(fitted_imag_by_flat) != matrix_size:
        raise ValueError("Selected Verilog-A Y coefficient dimensions are inconsistent")
    if dc_real_by_flat is not None and len(dc_real_by_flat) != matrix_size:
        raise ValueError("Selected Verilog-A DC coefficient dimensions are inconsistent")
    if dc_imag_by_flat is not None and len(dc_imag_by_flat) != matrix_size:
        raise ValueError("Selected Verilog-A DC imaginary dimensions are inconsistent")
    lines.append("    if (dc_operating_point != 0) begin")
    for row in range(nports):
        for col in range(nports):
            flat = row * nports + col
            if dc_real_by_flat is not None:
                lines.append(f"      active_yr[{flat}] = {dc_real_by_flat[flat]};")
            else:
                if nports == 1:
                    conductance_factor = 1.0
                elif row == col:
                    conductance_factor = 2.0 * float(nports - 1) / float(nports)
                else:
                    conductance_factor = -2.0 / float(nports)
                lines.append(
                    f"      active_yr[{flat}] = ({veriloga_float(conductance_factor)})/"
                    "dc_equivalent_resistance_ohm;"
                )
            lines.append(
                f"      active_yi[{flat}] = "
                f"{dc_imag_by_flat[flat] if dc_imag_by_flat is not None else '0.0'};"
            )
    lines.append("    end else begin")
    for flat in range(matrix_size):
        lines.append(f"      active_yr[{flat}] = {fitted_real_by_flat[flat]};")
        lines.append(f"      active_yi[{flat}] = {fitted_imag_by_flat[flat]};")
    lines.append("    end")
    lines.append("")


def _veriloga_dc_path_configuration(
    labels: Sequence[str],
    dc_port_resistances_ohm: object,
) -> tuple[list[tuple[str, str, float]], list[str] | None]:
    """Return Verilog-A path parameters and their flattened Y expressions."""

    path_resistances = validate_dc_port_resistances(
        labels,
        dc_port_resistances_ohm,
        context="Verilog-A DC topology",
    )
    if path_resistances is None:
        return [], None
    nports = infer_complete_sparameter_ports(labels)
    parameter_rows: list[tuple[str, str, float]] = []
    terms: list[list[str]] = [[] for _ in range(nports * nports)]
    for first, second, canonical in parse_dc_port_paths(path_resistances, nports):
        identifier = veriloga_identifier(
            f"dc_resistance_{canonical}_ohm",
            "dc_path_resistance_ohm",
        )
        resistance = path_resistances[canonical]
        parameter_rows.append((canonical, identifier, resistance))
        conductance = f"(1.0/{identifier})"
        terms[first * nports + first].append(conductance)
        if second is not None:
            terms[second * nports + second].append(conductance)
            terms[first * nports + second].append(f"(-1.0/{identifier})")
            terms[second * nports + first].append(f"(-1.0/{identifier})")
    expressions = [" + ".join(items) if items else "0.0" for items in terms]
    return parameter_rows, expressions


def _append_veriloga_port_stamps(
    lines: list[str],
    port_ids: Sequence[str],
    real_by_flat: Sequence[str],
    imag_by_flat: Sequence[str],
) -> None:
    """Append the selected DC or fitted small-signal contributions.

    Keep ``ddt`` outside procedural conditionals. ADS and other Verilog-A
    compilers reject analog operators in a runtime ``if`` branch even when the
    condition is derived from simulator frequency.
    """

    lines.append("    omega = 6.2831853071795864769*freq_hz;")
    lines.append("    if (omega < 1.0e-30) omega = 1.0e-30;")
    nports = len(port_ids)
    for row, port_i in enumerate(port_ids):
        for col, port_j in enumerate(port_ids):
            flat = row * nports + col
            lines.append(
                f"    I({port_i}) <+ ({real_by_flat[flat]})*V({port_j}) + "
                f"(({imag_by_flat[flat]})/omega)*ddt(V({port_j}));"
            )


def _veriloga_readme(
    model_kind: str,
    module_name: str,
    va_file: str,
    manifest_name: str,
    nports: int,
    parameter_names: Sequence[str],
    parameter_identifiers: Sequence[str],
    parameter_scale_identifiers: Sequence[str],
    parameter_input_scales: Sequence[float],
    parameter_instance_defaults: Sequence[float],
    input_columns: Sequence[str],
    output_columns: Sequence[str],
    freq_transform: str,
    frequency_expression: str,
    z0: float,
    dc_equivalent_resistance_ohm: float,
    dc_port_resistances_ohm: dict[str, float] | None,
    dc_model_kind: str | None,
    dc_matrix_entries: Sequence[str] | None,
    output_domain: str,
    folded_input_scaler: bool,
    folded_output_scaler: bool,
    uses_coarse_inputs: bool,
    adds_coarse_to_output: bool,
    embedded_coarse_model: bool,
    extra_notes: Sequence[str] | None,
    response_relation_override: str | None = None,
) -> str:
    params = ", ".join(
        f"`{identifier}` from `{name}`"
        for name, identifier in zip(parameter_names, parameter_identifiers)
    )
    if parameter_names:
        scale_rows = "\n".join(
            "- `{param}`: default `{default}`, scale `{scale_param}` = `{scale}`; "
            "model value = `{param}` / `{scale_param}`.".format(
                param=identifier,
                default=veriloga_float(float(default)),
                scale_param=scale_identifier,
                scale=veriloga_float(float(scale)),
            )
            for identifier, scale_identifier, scale, default in zip(
                parameter_identifiers,
                parameter_scale_identifiers,
                parameter_input_scales,
                parameter_instance_defaults,
            )
        )
    else:
        scale_rows = "- No geometry/process parameters were exported."
    if dc_model_kind == "geometry_dependent_exact_dc_full_s_mlp":
        matrix_entries = list(dc_matrix_entries or [])
        dc_topology_text = (
            f"The embedded DC model fits all `{len(matrix_entries)}` real/imaginary "
            "components of the ordered S matrix independently, then converts that "
            "complete DC S matrix to Y for electrical stamping. No Y projection, "
            "resistor-graph constraint, or reciprocity constraint is imposed."
        )
    elif dc_model_kind == "geometry_dependent_exact_dc_full_y_mlp":
        matrix_entries = list(dc_matrix_entries or [])
        dc_topology_text = (
            f"The embedded DC model stamps all `{len(matrix_entries)}` ordered real "
            "Y-matrix entries independently. Thus `Yij` represents the corresponding "
            "ordered `Sij` information without a resistor-graph or reciprocity "
            "constraint. The complete entry list is recorded in the manifest under "
            "`dc_matrix_entries`."
        )
    elif dc_port_resistances_ohm:
        dc_path_rows = "\n".join(
            f"- `{path}`: `{float(resistance):.17g} ohm`"
            for path, resistance in dc_port_resistances_ohm.items()
        )
        dc_topology_text = (
            "Only the explicitly selected paths below are stamped; every undeclared "
            "port pair remains open at DC:\n\n" + dc_path_rows
        )
    else:
        dc_topology_text = (
            "This legacy model has no saved path selection. For a multiport, equal "
            "resistors form a complete graph sized from the saved equivalent resistance."
        )
    notes = "\n".join(f"- {note}" for note in (extra_notes or []))
    if notes:
        notes = "\n\nNotes:\n\n" + notes + "\n"
    if response_relation_override is not None:
        response_relation = response_relation_override
    elif output_domain == "y":
        response_relation = (
            "- Neural output domain: direct Y-parameters in Siemens\n"
            f"- Reference impedance used when generating Y training targets: `{z0:g} ohm`\n"
            "- Runtime conversion: none; the generated module stamps the predicted Y-matrix directly"
        )
    else:
        response_relation = (
            "- Neural output domain: S-parameters\n"
            f"- Reference impedance used for S-to-Y conversion: `{z0:g} ohm`\n"
            "- Current relation: `Y = (I - S) * inverse(I + S) / Z0`, then\n"
            "  `Iport = Y * Vport`"
        )
    scaler_rows = (
        f"- Input standardization folded into first layer: `{str(folded_input_scaler).lower()}`\n"
        f"- Output scaling folded into final layer: `{str(folded_output_scaler).lower()}`"
    )
    coarse = ""
    if embedded_coarse_model:
        coarse = """

## Embedded Coarse-Response Model

This KBNN package is self-contained. A second S-domain DNN evaluates the
coarse response from the same geometry/process parameters and simulator
frequency. The generated module feeds that response into the KBNN and/or adds
it to the residual before converting the final S-matrix to Y. No coarse-response
pins, instance parameters, MDIF files, or schematic subcircuits are required at
runtime.
"""
    elif uses_coarse_inputs or adds_coarse_to_output:
        coarse = """

## Coarse-Response Hooks

This KBNN Verilog-A file contains one editable assignment per coarse
S-parameter real/imaginary value inside the `analog begin` block. The default
assignments copy constant `coarse_Sij_*_default` parameters set to zero. For a
residual or prior-input KBNN to match training, replace those assignment right
hand sides with the real coarse circuit/surrogate response. The expressions can
reference `freq_hz` and the module's geometry/process parameters directly. Set
the defaults only for a fixed frequency/condition sanity check.
"""
    return f"""# Direct Verilog-A Export

This package contains a direct Verilog-A implementation of the trained
{model_kind} model. Unlike `export-ads-ann`, this path does not retrain the
network in ADS ANN; it embeds the saved `model.npz` weights and scaling values
directly into `{va_file}`.

## Files

- `{va_file}`: Verilog-A N-port module `{module_name}`
- `{manifest_name}`: source model metadata, column mapping, and export settings
- `VERILOGA_README.md`: this usage note

## Schematic Use

1. Add `{va_file}` to the ADS workspace as a Verilog-A model and create/place
   the module `{module_name}`.
2. Connect the `{nports}` electrical ports to the schematic ports.
3. Set geometry/process parameters on the instance. Parameter mapping:
   {params if params else "`none`"}.
4. Run an S-parameter or small-signal AC simulation. The model uses
   `{frequency_expression}` for simulator frequency and expects Hz.
5. Compare the simulated S-parameters against the verification data before
   using the model in optimization.

Only the electrical ports and geometry/process parameters need to be provided
for normal ADS use. Generated scale parameters named like `*_input_scale` are
unit-conversion constants with export-time defaults; leave them unchanged unless
you intentionally change the ADS unit convention. Frequency feature columns such
as `freq_hz` and `freq_log10_hz` are computed inside the Verilog-A module from
simulator frequency; do not add external pins or parameters for them.

## Parameter Scaling

ADS instance parameters are assumed to be in the base units you use in the
schematic. The generated model converts those values back to the units seen
during training before applying the neural-network standardization:

{scale_rows}

If the MDIF training values were already in the same units you use in ADS, keep
the scale at `1.0`. If all MDIF parameters used dimensionless micron values and
ADS uses meters, export with `--parameter-input-scales 1um`. The one scale is
applied to every fitted parameter and means "ADS/base-unit value per one
model-training unit".

## Distinct DC Point

At an exact simulator frequency of zero, the fitted neural/rational contribution
is electrically disabled. Current models instead evaluate their embedded
geometry-only DC model, extracted exclusively from exact zero-Hz rows. Legacy
models use their saved fixed path resistances. The compatibility-only aggregate
resistance diagnostic is `{dc_equivalent_resistance_ohm:.17g} ohm`; it does not
replace or modify a full-matrix DC model. Positive-frequency fallback and RF
extrapolation are forbidden.

{dc_topology_text}

Positive frequencies use only the fitted-response contribution.
The module selects the DC or fitted Y coefficients before one unconditional
current contribution, so the `ddt()` analog operator remains outside procedural
conditionals and in a legal Verilog-A context.

## Implementation

- Neural feature columns, computed internally unless they are geometry/process
  parameters: {", ".join(f"`{name}`" for name in input_columns)}
- Outputs: {", ".join(f"`{name}`" for name in output_columns)}
- Frequency transform: `{freq_transform}`
- Export efficiency:
{scaler_rows}
{response_relation}
- Imaginary admittance is contributed as `(B / omega) * ddt(V)`, so this
  direct model is intended for S-parameter/AC use, not as a causal transient
  behavioral model.

If your ADS Verilog-A environment uses a different frequency variable, regenerate
with `--frequency-expression` and the expression supported by that simulator.
{coarse}{notes}
"""


def _validated_dc_model_export(
    dc_model: dict[str, object] | None,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
) -> dict[str, object] | None:
    if dc_model is None:
        return None
    if [str(value) for value in dc_model["parameter_names"]] != list(parameter_names):  # type: ignore[index]
        raise ValueError("DC model parameters do not match the exported RF model")
    if [str(value) for value in dc_model["sparam_labels"]] != list(sparam_labels):  # type: ignore[index]
        raise ValueError("DC model S-parameter order does not match the RF model")
    nports = infer_complete_sparameter_ports(sparam_labels)
    representation = str(
        dc_model.get(
            "representation",
            (
                "full_s_matrix"
                if dc_model.get("kind") == "geometry_dependent_exact_dc_full_s_mlp"
                else "full_y_matrix"
                if dc_model.get("kind") == "geometry_dependent_exact_dc_full_y_mlp"
                else "path_conductance"
            ),
        )
    )
    paths = (
        []
        if representation in {"full_s_matrix", "full_y_matrix"}
        else parse_dc_port_paths(dc_model["port_paths"], nports)
    )
    layer_sizes = [int(value) for value in dc_model["layer_sizes"]]  # type: ignore[index]
    weights = [np.asarray(value, dtype=float) for value in dc_model["weights"]]  # type: ignore[index]
    biases = [np.asarray(value, dtype=float) for value in dc_model["biases"]]  # type: ignore[index]
    x_mean = np.asarray(dc_model["x_mean"], dtype=float)
    x_std = np.asarray(dc_model["x_std"], dtype=float)
    y_mean = np.asarray(dc_model["y_mean"], dtype=float)
    y_std = np.asarray(dc_model["y_std"], dtype=float)
    expected_outputs = (
        2 * nports * nports
        if representation == "full_s_matrix"
        else nports * nports
        if representation == "full_y_matrix"
        else len(paths)
    )
    if layer_sizes[0] != len(parameter_names) or layer_sizes[-1] != expected_outputs:
        raise ValueError("DC model dimensions do not match its selected representation")
    if len(weights) != len(biases) or len(weights) != len(layer_sizes) - 1:
        raise ValueError("DC model layers are inconsistent")
    if x_mean.shape != (layer_sizes[0],) or x_std.shape != (layer_sizes[0],):
        raise ValueError("DC model input scaler is inconsistent")
    if y_mean.shape != (layer_sizes[-1],) or y_std.shape != (layer_sizes[-1],):
        raise ValueError("DC model output scaler is inconsistent")
    return {
        **dc_model,
        "representation": representation,
        "paths_parsed": paths,
        "layer_sizes": layer_sizes,
        "weights": weights,
        "biases": biases,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _veriloga_dc_matrix_expressions(
    nports: int,
    paths: Sequence[tuple[int, int | None, str]],
    conductance_expressions: Sequence[str],
) -> list[str]:
    terms: list[list[str]] = [[] for _ in range(nports * nports)]
    for (first, second, _), expression in zip(paths, conductance_expressions):
        terms[first * nports + first].append(expression)
        if second is not None:
            terms[second * nports + second].append(expression)
            terms[first * nports + second].append(f"-({expression})")
            terms[second * nports + first].append(f"-({expression})")
    return [" + ".join(values) if values else "0.0" for values in terms]


def _dc_export_default_s_values(
    dc_model_data: dict[str, object],
    sparam_labels: Sequence[str],
    z0: float,
) -> np.ndarray:
    """Evaluate exported DC data at its mean geometry for manifest diagnostics."""

    activation = str(dc_model_data["activation"])
    values = np.zeros((1, len(dc_model_data["x_mean"])), dtype=float)  # type: ignore[arg-type]
    weights = dc_model_data["weights"]
    biases = dc_model_data["biases"]
    assert isinstance(weights, list) and isinstance(biases, list)
    for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
        values = values @ np.asarray(weight, dtype=float) + np.asarray(bias, dtype=float)
        if layer_idx < len(weights) - 1:
            values = np.tanh(values) if activation == "tanh" else np.maximum(values, 0.0)
    outputs = (
        values * np.asarray(dc_model_data["y_std"], dtype=float)
        + np.asarray(dc_model_data["y_mean"], dtype=float)
    )
    nports = infer_complete_sparameter_ports(sparam_labels)
    if dc_model_data.get("representation") == "full_s_matrix":
        real = outputs[0, : nports * nports]
        imag = outputs[0, nports * nports :]
        s_matrix = (real + 1j * imag).reshape(nports, nports)
        return np.asarray(
            [
                s_matrix[(sparam_indices(label) or (0, 0))[0] - 1,
                         (sparam_indices(label) or (0, 0))[1] - 1]
                for label in sparam_labels
            ],
            dtype=complex,
        )
    if dc_model_data.get("representation") == "full_y_matrix":
        y_matrix = outputs[0].reshape(nports, nports).astype(complex)
    else:
        conductances = np.exp(
            np.clip(
                outputs[0],
                float(dc_model_data["log_conductance_min"]),
                float(dc_model_data["log_conductance_max"]),
            )
        )
        y_matrix = _dc_matrix_from_path_conductances(
            nports,
            dc_model_data["paths_parsed"],  # type: ignore[arg-type]
            conductances,
        ).astype(complex)
    s_matrix = _y_matrix_to_s_matrix(y_matrix, z0)
    return np.asarray(
        [
            s_matrix[(sparam_indices(label) or (0, 0))[0] - 1, (sparam_indices(label) or (0, 0))[1] - 1]
            for label in sparam_labels
        ],
        dtype=complex,
    )


def veriloga_module_text(
    model_kind: str,
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    freq_transform: str,
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z0: float,
    frequency_expression: str,
    uses_coarse_inputs: bool = False,
    adds_coarse_to_output: bool = False,
    parameter_input_scales: dict[str, float] | None = None,
    output_domain: str = "s",
    fold_input_scaler: bool = False,
    fold_output_scaler: bool = False,
    embedded_coarse_model: dict[str, object] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    output_domain = output_domain.lower().strip()
    if output_domain not in {"s", "y"}:
        raise ValueError("Verilog-A output_domain must be 's' or 'y'")
    if output_domain == "y" and (uses_coarse_inputs or adds_coarse_to_output):
        raise ValueError("Direct-Y Verilog-A export is currently supported only without coarse hooks")
    dc_source_kind = validate_exact_dc_source_kind(
        dc_resistance_source_kind,
        context=f"{model_kind} Verilog-A export",
    )
    dc_resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context=f"{model_kind} model",
    )
    dc_model_data = _validated_dc_model_export(
        dc_model,
        parameter_names,
        sparam_labels,
    )
    if dc_model_data is not None and dc_model_data["representation"] in {
        "full_s_matrix",
        "full_y_matrix",
    }:
        dc_path_resistances = None
        dc_parameter_rows = []
        dc_real_by_flat = ["0.0"] * len(sparam_labels)
    else:
        dc_path_resistances = validate_dc_port_resistances(
            sparam_labels,
            dc_port_resistances_ohm,
            context=f"{model_kind} model",
        )
        dc_parameter_rows, dc_real_by_flat = _veriloga_dc_path_configuration(
            sparam_labels,
            dc_path_resistances,
        )
    nports = infer_complete_sparameter_ports(sparam_labels)
    n_sparams = len(sparam_labels)
    n_outputs = 2 * n_sparams
    if len(layer_sizes) < 2:
        raise ValueError("Layer sizes must include input and output dimensions")
    if layer_sizes[-1] != n_outputs:
        raise ValueError(
            f"Model output dimension {layer_sizes[-1]} does not match "
            f"2 * number of S-parameters ({n_outputs})"
        )
    if len(weights) != len(biases) or len(weights) != len(layer_sizes) - 1:
        raise ValueError("Weights, biases, and layer sizes are inconsistent")

    scale_map = {str(key): float(value) for key, value in (parameter_input_scales or {}).items()}
    unknown_scales = sorted(set(scale_map) - set(parameter_names))
    if unknown_scales:
        raise ValueError(
            "Parameter input scales include names that are not model parameters: "
            + ", ".join(unknown_scales)
        )
    param_ids = unique_veriloga_identifiers(parameter_names, "param")
    scale_ids = unique_veriloga_identifiers(
        [f"{ident}_input_scale" for ident in param_ids],
        "param_input_scale",
        used_names=set(param_ids),
    )
    param_scales: list[float] = []
    for name in parameter_names:
        scale = float(scale_map.get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Parameter input scale for {name!r} must be positive and finite")
        param_scales.append(scale)
    port_ids = [f"p{idx}" for idx in range(1, nports + 1)]
    feature_columns = [*parameter_names, *frequency_feature_columns(freq_transform)]
    if uses_coarse_inputs:
        feature_columns.extend(sparameter_real_imag_columns(sparam_labels, prefix="coarse"))
    if len(feature_columns) != layer_sizes[0]:
        raise ValueError(
            f"Expected {layer_sizes[0]} input columns from the model, but the Verilog-A "
            f"export would generate {len(feature_columns)} columns"
        )
    if x_mean.shape[0] != layer_sizes[0] or x_std.shape[0] != layer_sizes[0]:
        raise ValueError("Input scaler dimensions do not match model input dimension")
    if y_mean.shape[0] != n_outputs or y_std.shape[0] != n_outputs:
        raise ValueError("Output scaler dimensions do not match model output dimension")
    if np.any(np.asarray(x_std, dtype=float) == 0.0):
        raise ValueError("Input scaler standard deviations must be non-zero for Verilog-A export")

    coarse_embedded = embedded_coarse_model is not None
    needs_coarse_response = bool(uses_coarse_inputs or adds_coarse_to_output)
    if coarse_embedded and not needs_coarse_response:
        raise ValueError(
            "An embedded coarse model is only valid when the primary model uses a coarse response"
        )

    coarse_feature_columns: list[str] = []
    coarse_freq_transform = ""
    coarse_activation = ""
    coarse_layer_sizes: list[int] = []
    coarse_weights: list[np.ndarray] = []
    coarse_biases: list[np.ndarray] = []
    coarse_x_mean = np.asarray([], dtype=float)
    coarse_x_std = np.asarray([], dtype=float)
    coarse_y_mean = np.asarray([], dtype=float)
    coarse_y_std = np.asarray([], dtype=float)
    coarse_source_model_dir: str | None = None
    if embedded_coarse_model is not None:
        coarse_parameter_names = [
            str(value) for value in embedded_coarse_model["parameter_names"]  # type: ignore[index]
        ]
        coarse_sparam_labels = [
            str(value) for value in embedded_coarse_model["sparam_labels"]  # type: ignore[index]
        ]
        if coarse_parameter_names != list(parameter_names):
            raise ValueError(
                "Embedded coarse DNN parameter_names must exactly match the KBNN parameter order: "
                f"expected {list(parameter_names)}, got {coarse_parameter_names}"
            )
        if coarse_sparam_labels != list(sparam_labels):
            raise ValueError(
                "Embedded coarse DNN sparam_labels must exactly match the KBNN output order: "
                f"expected {list(sparam_labels)}, got {coarse_sparam_labels}"
            )
        coarse_output_domain = str(
            embedded_coarse_model.get("output_domain", "s")
        ).lower().strip()
        if coarse_output_domain != "s":
            raise ValueError("Embedded coarse DNN must be trained with --output-domain s")
        coarse_freq_transform = str(embedded_coarse_model["freq_transform"])
        coarse_activation = str(embedded_coarse_model["activation"])
        coarse_layer_sizes = [
            int(value) for value in embedded_coarse_model["layer_sizes"]  # type: ignore[index]
        ]
        coarse_weights = [
            np.asarray(value, dtype=float).copy()
            for value in embedded_coarse_model["weights"]  # type: ignore[index]
        ]
        coarse_biases = [
            np.asarray(value, dtype=float).copy()
            for value in embedded_coarse_model["biases"]  # type: ignore[index]
        ]
        coarse_x_mean = np.asarray(embedded_coarse_model["x_mean"], dtype=float)
        coarse_x_std = np.asarray(embedded_coarse_model["x_std"], dtype=float)
        coarse_y_mean = np.asarray(embedded_coarse_model["y_mean"], dtype=float)
        coarse_y_std = np.asarray(embedded_coarse_model["y_std"], dtype=float)
        coarse_source_model_dir = (
            str(embedded_coarse_model.get("source_model_dir") or "") or None
        )
        coarse_feature_columns = [
            *parameter_names,
            *frequency_feature_columns(coarse_freq_transform),
        ]
        if len(coarse_layer_sizes) < 2:
            raise ValueError(
                "Embedded coarse DNN layer sizes must include input and output dimensions"
            )
        if coarse_layer_sizes[0] != len(coarse_feature_columns):
            raise ValueError(
                "Embedded coarse DNN input dimension does not match its parameter/frequency features"
            )
        if coarse_layer_sizes[-1] != n_outputs:
            raise ValueError(
                f"Embedded coarse DNN output dimension must be {n_outputs}, "
                f"got {coarse_layer_sizes[-1]}"
            )
        if (
            len(coarse_weights) != len(coarse_biases)
            or len(coarse_weights) != len(coarse_layer_sizes) - 1
        ):
            raise ValueError(
                "Embedded coarse DNN weights, biases, and layer sizes are inconsistent"
            )
        for layer_idx, (coarse_weight, coarse_bias) in enumerate(
            zip(coarse_weights, coarse_biases)
        ):
            expected_weight_shape = (
                coarse_layer_sizes[layer_idx],
                coarse_layer_sizes[layer_idx + 1],
            )
            if coarse_weight.shape != expected_weight_shape:
                raise ValueError(
                    f"Embedded coarse DNN W{layer_idx} has shape {coarse_weight.shape}; "
                    f"expected {expected_weight_shape}"
                )
            if coarse_bias.shape != (coarse_layer_sizes[layer_idx + 1],):
                raise ValueError(
                    f"Embedded coarse DNN b{layer_idx} has shape {coarse_bias.shape}; "
                    f"expected {(coarse_layer_sizes[layer_idx + 1],)}"
                )
        if (
            coarse_x_mean.shape != (coarse_layer_sizes[0],)
            or coarse_x_std.shape != (coarse_layer_sizes[0],)
        ):
            raise ValueError("Embedded coarse DNN input scaler dimensions are inconsistent")
        if coarse_y_mean.shape != (n_outputs,) or coarse_y_std.shape != (n_outputs,):
            raise ValueError("Embedded coarse DNN output scaler dimensions are inconsistent")
        if np.any(coarse_x_std == 0.0):
            raise ValueError(
                "Embedded coarse DNN input scaler standard deviations must be non-zero"
            )

    va_weights = [np.asarray(weight, dtype=float).copy() for weight in weights]
    va_biases = [np.asarray(bias, dtype=float).copy() for bias in biases]
    folded_input_scaler = bool(fold_input_scaler)
    folded_output_scaler = bool(fold_output_scaler)
    if folded_input_scaler:
        first_scale = np.asarray(x_std, dtype=float)
        first_mean = np.asarray(x_mean, dtype=float)
        va_weights[0] = va_weights[0] / first_scale[:, None]
        va_biases[0] = va_biases[0] - first_mean @ va_weights[0]
    if folded_output_scaler:
        last_scale = np.asarray(y_std, dtype=float)
        last_mean = np.asarray(y_mean, dtype=float)
        va_weights[-1] = va_weights[-1] * last_scale[None, :]
        va_biases[-1] = va_biases[-1] * last_scale + last_mean

    module_id = veriloga_identifier(module_name, "surrogate_va")
    lines: list[str] = [
        "`include \"constants.vams\"",
        "`include \"disciplines.vams\"",
        "",
        f"module {module_id}({', '.join(port_ids)});",
        f"  inout {', '.join(port_ids)};",
        f"  electrical {', '.join(port_ids)};",
        "",
        "  parameter integer clamp_frequency = 1;",
        "  parameter real min_frequency_hz = 1.0;",
        f"  parameter real dc_equivalent_resistance_ohm = {veriloga_float(dc_resistance)}; "
        "// DC summary; selected path parameters below drive stamping when present",
    ]
    if dc_model_data is None:
        for canonical, identifier, path_resistance in dc_parameter_rows:
            lines.append(
                f"  parameter real {identifier} = {veriloga_float(path_resistance)}; "
                f"// selected DC path {canonical}"
            )
    if output_domain == "s":
        lines.append(f"  parameter real z0 = {veriloga_float(z0)};")
        lines.append("  parameter real pivot_floor = 1.0e-24;")

    param_defaults = []
    param_model_defaults = []
    for idx, (name, ident) in enumerate(zip(parameter_names, param_ids)):
        model_default = float(x_mean[idx]) if idx < len(x_mean) else 0.0
        default = model_default * param_scales[idx]
        param_model_defaults.append(model_default)
        param_defaults.append(default)
        lines.append(
            f"  parameter real {ident} = {veriloga_float(default)}; "
            f"// source VAR {name}, ADS/base units"
        )

    if parameter_names:
        lines.append("")
        lines.append("  // ADS/base-unit parameter divided by input_scale equals the model training VAR.")
        for name, ident, scale_ident, scale in zip(parameter_names, param_ids, scale_ids, param_scales):
            lines.append(
                f"  parameter real {scale_ident} = {veriloga_float(scale)}; "
                f"// model VAR {name} = {ident}/{scale_ident}"
            )

    if needs_coarse_response and not coarse_embedded:
        lines.append("")
        lines.append("  // Default coarse S-parameter values. Replace analog assignments for a live coarse model.")
        for label in sparam_labels:
            base = veriloga_identifier(label, "sparam")
            lines.append(f"  parameter real coarse_{base}_real_default = 0.0;")
            lines.append(f"  parameter real coarse_{base}_imag_default = 0.0;")

    matrix_size = nports * nports
    lines.extend(
        [
            "",
            "  real freq_hz;",
            "  real freq_log10_hz;",
            "  real omega;",
            "  integer dc_operating_point;",
        ]
    )
    if output_domain == "s":
        lines.extend(
            [
                "  integer i;",
                "  integer j;",
                "  integer k;",
                "  integer row;",
                "  integer col;",
                "  integer idx;",
                "  integer piv;",
                "  integer pivrow;",
                "  real mag;",
                "  real best_mag;",
                "  real den;",
                "  real pr;",
                "  real pi;",
                "  real fr;",
                "  real fi;",
                "  real tr;",
                "  real ti;",
                f"  real sr [0:{matrix_size - 1}];",
                f"  real si [0:{matrix_size - 1}];",
                f"  real mr [0:{matrix_size - 1}];",
                f"  real mi [0:{matrix_size - 1}];",
                f"  real ar [0:{matrix_size - 1}];",
                f"  real ai [0:{matrix_size - 1}];",
                f"  real invr [0:{matrix_size - 1}];",
                f"  real invi [0:{matrix_size - 1}];",
            ]
        )
    if uses_coarse_inputs or adds_coarse_to_output:
        lines.extend(
            [
                f"  real cr [0:{n_sparams - 1}];",
                f"  real ci [0:{n_sparams - 1}];",
            ]
        )
    if coarse_embedded:
        lines.append(f"  real coarse_y [0:{n_outputs - 1}];")
        for layer_idx, size in enumerate(coarse_layer_sizes):
            lines.append(f"  real c{layer_idx} [0:{size - 1}];")
    if not folded_output_scaler:
        lines.append(f"  real y [0:{n_outputs - 1}];")
    if output_domain == "s":
        lines.extend(
            [
                f"  real yr [0:{matrix_size - 1}];",
                f"  real yi [0:{matrix_size - 1}];",
            ]
        )
    lines.extend(
        [
            f"  real active_yr [0:{matrix_size - 1}];",
            f"  real active_yi [0:{matrix_size - 1}];",
        ]
    )
    for layer_idx, size in enumerate(layer_sizes):
        lines.append(f"  real l{layer_idx} [0:{size - 1}];")
    if dc_model_data is not None:
        dc_layer_sizes = dc_model_data["layer_sizes"]
        assert isinstance(dc_layer_sizes, list)
        if dc_model_data.get("representation") == "full_s_matrix":
            lines.extend(
                [
                    f"  real dc_sr [0:{matrix_size - 1}];",
                    f"  real dc_si [0:{matrix_size - 1}];",
                    f"  real dc_yr [0:{matrix_size - 1}];",
                    f"  real dc_yi [0:{matrix_size - 1}];",
                ]
            )
        elif dc_model_data.get("representation") == "full_y_matrix":
            lines.append(f"  real dc_y [0:{matrix_size - 1}];")
        else:
            lines.append(f"  real dc_log_g [0:{len(dc_model_data['paths_parsed']) - 1}];")  # type: ignore[arg-type]
            lines.append(f"  real dc_g [0:{len(dc_model_data['paths_parsed']) - 1}];")  # type: ignore[arg-type]
        for layer_idx, size in enumerate(dc_layer_sizes):
            lines.append(f"  real dc_l{layer_idx} [0:{size - 1}];")

    lines.extend(["", "  analog begin"])
    lines.append(f"    freq_hz = {frequency_expression};")
    lines.append("    dc_operating_point = (freq_hz == 0.0);")
    lines.append("    if (clamp_frequency != 0 && freq_hz < min_frequency_hz) freq_hz = min_frequency_hz;")
    lines.append("    freq_log10_hz = log(freq_hz)/log(10.0);")
    lines.append("")

    if dc_model_data is not None:
        dc_x_mean = np.asarray(dc_model_data["x_mean"], dtype=float)
        dc_x_std = np.asarray(dc_model_data["x_std"], dtype=float)
        for idx, (ident, scale_ident) in enumerate(zip(param_ids, scale_ids)):
            expression = f"({ident})/({scale_ident})"
            lines.append(
                f"    dc_l0[{idx}] = (({expression}) - "
                f"({veriloga_float(float(dc_x_mean[idx]))}))"
                f"/({veriloga_float(float(dc_x_std[idx]))});"
            )
        dc_weights = dc_model_data["weights"]
        dc_biases = dc_model_data["biases"]
        assert isinstance(dc_weights, list) and isinstance(dc_biases, list)
        for layer_idx, (dc_weight, dc_bias) in enumerate(zip(dc_weights, dc_biases)):
            hidden_activation = (
                str(dc_model_data["activation"])
                if layer_idx < len(dc_weights) - 1
                else None
            )
            _veriloga_layer_assignments(
                lines,
                source=f"dc_l{layer_idx}",
                dest=f"dc_l{layer_idx + 1}",
                weight=np.asarray(dc_weight, dtype=float),
                bias=np.asarray(dc_bias, dtype=float),
                activation=hidden_activation,
            )
        dc_final_layer = f"dc_l{len(dc_model_data['layer_sizes']) - 1}"  # type: ignore[arg-type]
        dc_y_mean = np.asarray(dc_model_data["y_mean"], dtype=float)
        dc_y_std = np.asarray(dc_model_data["y_std"], dtype=float)
        if dc_model_data.get("representation") == "full_s_matrix":
            for idx in range(matrix_size):
                lines.append(
                    f"    dc_sr[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
                imag_idx = matrix_size + idx
                lines.append(
                    f"    dc_si[{idx}] = {dc_final_layer}[{imag_idx}]*"
                    f"({veriloga_float(float(dc_y_std[imag_idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[imag_idx]))});"
                )
            dc_real_by_flat = [f"dc_yr[{idx}]" for idx in range(matrix_size)]
            dc_imag_by_flat = [f"dc_yi[{idx}]" for idx in range(matrix_size)]
        elif dc_model_data.get("representation") == "full_y_matrix":
            for idx in range(len(dc_y_mean)):
                lines.append(
                    f"    dc_y[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
            dc_real_by_flat = [f"dc_y[{idx}]" for idx in range(matrix_size)]
            dc_imag_by_flat = None
        else:
            dc_log_min = veriloga_float(float(dc_model_data["log_conductance_min"]))
            dc_log_max = veriloga_float(float(dc_model_data["log_conductance_max"]))
            for idx in range(len(dc_y_mean)):
                lines.append(
                    f"    dc_log_g[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
                lines.append(
                    f"    dc_g[{idx}] = exp(min(max(dc_log_g[{idx}], {dc_log_min}), "
                    f"{dc_log_max}));"
                )
            dc_real_by_flat = _veriloga_dc_matrix_expressions(
                nports,
                dc_model_data["paths_parsed"],  # type: ignore[arg-type]
                [f"dc_g[{idx}]" for idx in range(len(dc_y_mean))],
            )
            dc_imag_by_flat = None
        lines.append("")
    else:
        dc_imag_by_flat = None

    if coarse_embedded:
        lines.append("    // Self-contained coarse S-parameter DNN.")
        coarse_feature_exprs: list[str] = [
            f"({ident})/({scale_ident})"
            for ident, scale_ident in zip(param_ids, scale_ids)
        ]
        if coarse_freq_transform == "log":
            coarse_feature_exprs.append("freq_log10_hz")
        elif coarse_freq_transform == "linear":
            coarse_feature_exprs.append("freq_hz")
        elif coarse_freq_transform == "log-linear":
            coarse_feature_exprs.extend(["freq_log10_hz", "freq_hz"])
        else:
            raise ValueError(
                f"Unsupported embedded coarse frequency transform {coarse_freq_transform!r}"
            )
        for idx, expr in enumerate(coarse_feature_exprs):
            lines.append(
                f"    c0[{idx}] = (({expr}) - "
                f"({veriloga_float(float(coarse_x_mean[idx]))}))"
                f"/({veriloga_float(float(coarse_x_std[idx]))}); "
                f"// {coarse_feature_columns[idx]}"
            )
        lines.append("")
        for layer_idx, (coarse_weight, coarse_bias) in enumerate(
            zip(coarse_weights, coarse_biases)
        ):
            coarse_hidden_activation = (
                coarse_activation if layer_idx < len(coarse_weights) - 1 else None
            )
            _veriloga_layer_assignments(
                lines,
                source=f"c{layer_idx}",
                dest=f"c{layer_idx + 1}",
                weight=coarse_weight,
                bias=coarse_bias,
                activation=coarse_hidden_activation,
            )
        coarse_final_layer = f"c{len(coarse_layer_sizes) - 1}"
        for idx in range(n_outputs):
            lines.append(
                f"    coarse_y[{idx}] = {coarse_final_layer}[{idx}]"
                f"*({veriloga_float(float(coarse_y_std[idx]))}) "
                f"+ ({veriloga_float(float(coarse_y_mean[idx]))});"
            )
        lines.append("")
        for idx, label in enumerate(sparam_labels):
            lines.append(
                f"    cr[{idx}] = coarse_y[{idx}]; // embedded coarse {label} real"
            )
            lines.append(
                f"    ci[{idx}] = coarse_y[{idx + n_sparams}]; "
                f"// embedded coarse {label} imag"
            )
        lines.append("")
    elif needs_coarse_response:
        lines.append("    // Coarse response values. Replace these assignments for a live coarse model.")
        for idx, label in enumerate(sparam_labels):
            base = veriloga_identifier(label, "sparam")
            lines.append(f"    cr[{idx}] = coarse_{base}_real_default; // {label} real")
            lines.append(f"    ci[{idx}] = coarse_{base}_imag_default; // {label} imag")
        lines.append("")

    feature_exprs: list[str] = []
    for ident, scale_ident in zip(param_ids, scale_ids):
        feature_exprs.append(f"({ident})/({scale_ident})")
    if freq_transform == "log":
        feature_exprs.append("freq_log10_hz")
    elif freq_transform == "linear":
        feature_exprs.append("freq_hz")
    elif freq_transform == "log-linear":
        feature_exprs.extend(["freq_log10_hz", "freq_hz"])
    else:
        raise ValueError(f"Unsupported frequency transform {freq_transform!r}")
    if uses_coarse_inputs:
        feature_exprs.extend([f"cr[{idx}]" for idx in range(n_sparams)])
        feature_exprs.extend([f"ci[{idx}]" for idx in range(n_sparams)])

    for idx, expr in enumerate(feature_exprs):
        if folded_input_scaler:
            lines.append(f"    l0[{idx}] = {expr}; // {feature_columns[idx]}")
        else:
            lines.append(
                f"    l0[{idx}] = (({expr}) - ({veriloga_float(float(x_mean[idx]))}))"
                f"/({veriloga_float(float(x_std[idx]))}); // {feature_columns[idx]}"
            )
    lines.append("")

    for layer_idx, (weight, bias) in enumerate(zip(va_weights, va_biases)):
        hidden_activation = activation if layer_idx < len(va_weights) - 1 else None
        _veriloga_layer_assignments(
            lines,
            source=f"l{layer_idx}",
            dest=f"l{layer_idx + 1}",
            weight=np.asarray(weight, dtype=float),
            bias=np.asarray(bias, dtype=float),
            activation=hidden_activation,
        )

    final_layer = f"l{len(layer_sizes) - 1}"
    if not folded_output_scaler:
        for idx in range(n_outputs):
            lines.append(
                f"    y[{idx}] = {final_layer}[{idx}]*({veriloga_float(float(y_std[idx]))}) "
                f"+ ({veriloga_float(float(y_mean[idx]))});"
            )
        lines.append("")

    def network_output(index: int) -> str:
        return f"{final_layer}[{index}]" if folded_output_scaler else f"y[{index}]"

    direct_y_real_by_flat: list[str | None] = [None] * matrix_size
    direct_y_imag_by_flat: list[str | None] = [None] * matrix_size
    if output_domain == "y":
        for idx, label in enumerate(sparam_labels):
            row, col = sparam_indices(label) or (0, 0)
            flat = (row - 1) * nports + (col - 1)
            direct_y_real_by_flat[flat] = network_output(idx)
            direct_y_imag_by_flat[flat] = network_output(idx + n_sparams)
    else:
        for idx, label in enumerate(sparam_labels):
            row, col = sparam_indices(label) or (0, 0)
            flat = (row - 1) * nports + (col - 1)
            add_re = f" + cr[{idx}]" if adds_coarse_to_output else ""
            add_im = f" + ci[{idx}]" if adds_coarse_to_output else ""
            lines.append(f"    sr[{flat}] = {network_output(idx)}{add_re}; // {label} real")
            lines.append(f"    si[{flat}] = {network_output(idx + n_sparams)}{add_im}; // {label} imag")
        lines.append("")

        _append_veriloga_s_to_y_conversion(lines, nports)

    if output_domain == "y":
        if any(value is None for value in direct_y_real_by_flat + direct_y_imag_by_flat):
            raise ValueError("Internal error: direct-Y output mapping is incomplete")
        stamp_real = [str(value) for value in direct_y_real_by_flat]
        stamp_imag = [str(value) for value in direct_y_imag_by_flat]
    else:
        stamp_real = [f"yr[{flat}]" for flat in range(matrix_size)]
        stamp_imag = [f"yi[{flat}]" for flat in range(matrix_size)]
    if dc_model_data is not None and dc_model_data.get("representation") == "full_s_matrix":
        _append_veriloga_s_to_y_conversion(
            lines,
            nports,
            s_real="dc_sr",
            s_imag="dc_si",
            y_real="dc_yr",
            y_imag="dc_yi",
        )
    _append_veriloga_dc_or_fitted_y_assignments(
        lines,
        nports,
        stamp_real,
        stamp_imag,
        dc_real_by_flat,
        dc_imag_by_flat,
    )
    _append_veriloga_port_stamps(
        lines,
        port_ids,
        [f"active_yr[{flat}]" for flat in range(matrix_size)],
        [f"active_yi[{flat}]" for flat in range(matrix_size)],
    )
    lines.extend(["  end", "endmodule", ""])

    output_columns = (
        sparameter_real_imag_columns(sparam_labels, prefix="fine_y")
        if output_domain == "y"
        else sparameter_real_imag_columns(sparam_labels, prefix="fine")
    )
    if coarse_embedded:
        implementation_note = (
            "Direct Verilog-A embeds an S-domain coarse-response DNN and the trained "
            "KBNN, combines them internally, and converts the final S-parameters to a "
            "small-signal Y-matrix. Intended for AC/SP analyses."
        )
    elif output_domain == "y":
        implementation_note = (
            "Direct Verilog-A embeds the trained NumPy MLP weights and stamps the predicted "
            "Y-parameter matrix directly. Intended for AC/SP analyses."
        )
    else:
        implementation_note = (
            "Direct Verilog-A embeds the trained NumPy MLP weights and converts predicted "
            "S-parameters to a small-signal Y-matrix. Intended for AC/SP analyses."
        )
    manifest = {
        "module_name": module_id,
        "model_kind": model_kind,
        "nports": nports,
        "parameter_names": list(parameter_names),
        "parameter_identifiers": param_ids,
        "parameter_scale_identifiers": scale_ids,
        "parameter_input_scales": {
            name: float(scale) for name, scale in zip(parameter_names, param_scales)
        },
        "parameter_defaults": param_defaults,
        "parameter_instance_defaults": param_defaults,
        "parameter_model_defaults": param_model_defaults,
        "sparam_labels": list(sparam_labels),
        "input_columns": feature_columns,
        "output_columns": output_columns,
        "output_domain": output_domain,
        "freq_transform": freq_transform,
        "frequency_expression": frequency_expression,
        "z0": float(z0),
        "activation": activation,
        "layer_sizes": list(layer_sizes),
        "folded_input_scaler": folded_input_scaler,
        "folded_output_scaler": folded_output_scaler,
        "uses_coarse_inputs": bool(uses_coarse_inputs),
        "adds_coarse_to_output": bool(adds_coarse_to_output),
        "embedded_coarse_model": bool(coarse_embedded),
        "coarse_model": (
            {
                "source_model_dir": coarse_source_model_dir,
                "parameter_names": list(parameter_names),
                "sparam_labels": list(sparam_labels),
                "input_columns": coarse_feature_columns,
                "output_columns": sparameter_real_imag_columns(
                    sparam_labels, prefix="coarse"
                ),
                "output_domain": "s",
                "freq_transform": coarse_freq_transform,
                "activation": coarse_activation,
                "layer_sizes": coarse_layer_sizes,
            }
            if coarse_embedded
            else None
        ),
        "implementation_note": implementation_note,
        "dc_equivalent_resistance_ohm": dc_resistance,
        "dc_port_paths": (
            list(dc_model_data.get("port_paths", []))
            if dc_model_data is not None
            else list(dc_path_resistances)
            if dc_path_resistances is not None
            else None
        ),
        "dc_matrix_entries": (
            list(dc_model_data.get("matrix_entries", []))
            if dc_model_data is not None
            else []
        ),
        "dc_sparameter_entries": (
            list(dc_model_data.get("sparameter_entries", []))
            if dc_model_data is not None
            else []
        ),
        "dc_port_resistances_ohm": (
            None
            if dc_model_data is not None
            and dc_model_data.get("representation") in {"full_s_matrix", "full_y_matrix"}
            else dc_path_resistances
        ),
        "dc_port_parameter_identifiers": {
            canonical: identifier
            for canonical, identifier, _ in dc_parameter_rows
        } if dc_model_data is None else {},
        "dc_resistance_source_kind": dc_source_kind,
        "dc_model_kind": (
            str(dc_model_data.get("kind")) if dc_model_data is not None else None
        ),
        "dc_geometry_dependent": dc_model_data is not None,
        "dc_model_metadata": (
            dc_model_data.get("metadata") if dc_model_data is not None else None
        ),
        "dc_response_topology": (
            "Geometry-dependent full ordered complex S matrix; every real and "
            "imaginary S-parameter component is represented and converted to Y "
            "independently of the fitted RF response at zero Hz"
            if dc_model_data is not None
            and dc_model_data.get("representation") == "full_s_matrix"
            else
            "Geometry-dependent full ordered real Y matrix; every S-parameter entry "
            "is represented and the fitted RF response is bypassed at zero Hz"
            if dc_model_data is not None
            and dc_model_data.get("representation") == "full_y_matrix"
            else "Geometry-dependent selected-path conductance graph; only declared "
            "paths are connected and the fitted RF response is bypassed at zero Hz"
            if dc_model_data is not None
            else "Parameter-independent selected-path resistor graph; only declared "
            "paths are connected and the fitted neural response is bypassed at zero Hz"
            if dc_path_resistances is not None
            else "Legacy parameter-independent equal-resistance complete graph; the "
            "fitted neural response is bypassed at zero Hz"
        ),
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_sparameters_geometry": (
            "parameter_model_defaults" if dc_model_data is not None else "static_legacy"
        ),
        "dc_sparameters": {
            label: {
                "real": float(value.real),
                "imag": float(value.imag),
            }
            for label, value in zip(
                sparam_labels,
                (
                    _dc_export_default_s_values(
                        dc_model_data,
                        sparam_labels,
                        z0,
                    )
                    if dc_model_data is not None
                    else dc_sparameter_values(
                        sparam_labels,
                        dc_resistance,
                        dc_port_resistances_ohm=dc_path_resistances,
                        z0=z0,
                    )
                ),
            )
        },
    }
    return "\n".join(lines), manifest


def write_veriloga_package(
    out_dir: Path,
    model_kind: str,
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    freq_transform: str,
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z0: float = 50.0,
    frequency_expression: str = "$freq",
    uses_coarse_inputs: bool = False,
    adds_coarse_to_output: bool = False,
    parameter_input_scales: dict[str, float] | None = None,
    output_domain: str = "s",
    fold_input_scaler: bool = False,
    fold_output_scaler: bool = False,
    embedded_coarse_model: dict[str, object] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
    source_model_dir: str | None = None,
    extra_manifest: dict[str, object] | None = None,
    extra_notes: Sequence[str] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    module_id = veriloga_identifier(module_name, "surrogate_va")
    va_text, manifest = veriloga_module_text(
        model_kind=model_kind,
        module_name=module_id,
        parameter_names=parameter_names,
        sparam_labels=sparam_labels,
        freq_transform=freq_transform,
        activation=activation,
        layer_sizes=layer_sizes,
        weights=weights,
        biases=biases,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        z0=z0,
        frequency_expression=frequency_expression,
        uses_coarse_inputs=uses_coarse_inputs,
        adds_coarse_to_output=adds_coarse_to_output,
        parameter_input_scales=parameter_input_scales,
        output_domain=output_domain,
        fold_input_scaler=fold_input_scaler,
        fold_output_scaler=fold_output_scaler,
        embedded_coarse_model=embedded_coarse_model,
        dc_equivalent_resistance_ohm=dc_equivalent_resistance_ohm,
        dc_resistance_source_kind=dc_resistance_source_kind,
        dc_port_resistances_ohm=dc_port_resistances_ohm,
        dc_model=dc_model,
    )
    va_name = f"{module_id}.va"
    manifest_name = "veriloga_manifest.json"
    manifest.update(
        {
            "format": "direct_veriloga_surrogate",
            "veriloga_file": va_name,
            "source_model_dir": source_model_dir,
            "reference_note": (
                "Generated from saved local model.npz weights. ADS Verilog-A frequency variable "
                "defaults to $freq; verify against the target ADS release."
            ),
        }
    )
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / va_name).write_text(va_text)
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    (out_dir / "VERILOGA_README.md").write_text(
        _veriloga_readme(
            model_kind=model_kind,
            module_name=module_id,
            va_file=va_name,
            manifest_name=manifest_name,
            nports=int(manifest["nports"]),
            parameter_names=parameter_names,
            parameter_identifiers=manifest["parameter_identifiers"],
            parameter_scale_identifiers=manifest["parameter_scale_identifiers"],
            parameter_input_scales=[
                manifest["parameter_input_scales"][name] for name in parameter_names
            ],
            parameter_instance_defaults=manifest["parameter_instance_defaults"],
            input_columns=manifest["input_columns"],
            output_columns=manifest["output_columns"],
            freq_transform=freq_transform,
            frequency_expression=frequency_expression,
            z0=z0,
            dc_equivalent_resistance_ohm=float(
                manifest["dc_equivalent_resistance_ohm"]
            ),
            dc_port_resistances_ohm=manifest.get("dc_port_resistances_ohm"),
            dc_model_kind=manifest.get("dc_model_kind"),
            dc_matrix_entries=manifest.get("dc_matrix_entries"),
            output_domain=manifest["output_domain"],
            folded_input_scaler=bool(manifest["folded_input_scaler"]),
            folded_output_scaler=bool(manifest["folded_output_scaler"]),
            uses_coarse_inputs=uses_coarse_inputs,
            adds_coarse_to_output=adds_coarse_to_output,
            embedded_coarse_model=bool(manifest["embedded_coarse_model"]),
            extra_notes=extra_notes,
        )
    )
    return manifest


def neurotf_veriloga_module_text(
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    poles: np.ndarray,
    f_scale: float,
    z0: float,
    frequency_expression: str,
    parameter_input_scales: dict[str, float] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    """Generate a direct Neuro-TF Verilog-A implementation."""

    nports = infer_complete_sparameter_ports(sparam_labels)
    n_sparams = len(sparam_labels)
    poles_array = np.asarray(poles, dtype=complex).reshape(-1)
    n_coeffs = len(poles_array) + 1
    n_network_outputs = 2 * n_sparams * n_coeffs
    matrix_size = nports * nports
    dc_source_kind = validate_exact_dc_source_kind(
        dc_resistance_source_kind,
        context="Neuro-TF Verilog-A export",
    )
    dc_resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context="Neuro-TF model",
    )
    dc_model_data = _validated_dc_model_export(
        dc_model,
        parameter_names,
        sparam_labels,
    )
    if dc_model_data is not None and dc_model_data["representation"] in {
        "full_s_matrix",
        "full_y_matrix",
    }:
        dc_path_resistances = None
        dc_parameter_rows = []
        dc_real_by_flat = ["0.0"] * len(sparam_labels)
    else:
        dc_path_resistances = validate_dc_port_resistances(
            sparam_labels,
            dc_port_resistances_ohm,
            context="Neuro-TF model",
        )
        dc_parameter_rows, dc_real_by_flat = _veriloga_dc_path_configuration(
            sparam_labels,
            dc_path_resistances,
        )
    if not parameter_names:
        raise ValueError("Neuro-TF Verilog-A export requires at least one model parameter")
    if len(layer_sizes) < 2:
        raise ValueError("Layer sizes must include input and output dimensions")
    if layer_sizes[0] != len(parameter_names):
        raise ValueError(
            f"Neuro-TF input dimension {layer_sizes[0]} does not match its "
            f"{len(parameter_names)} model parameters"
        )
    if layer_sizes[-1] != n_network_outputs:
        raise ValueError(
            f"Neuro-TF output dimension {layer_sizes[-1]} does not match "
            f"2 * {n_sparams} S-parameters * {n_coeffs} coefficients "
            f"({n_network_outputs})"
        )
    if len(weights) != len(biases) or len(weights) != len(layer_sizes) - 1:
        raise ValueError("Weights, biases, and layer sizes are inconsistent")
    numeric_weights = [np.asarray(value, dtype=float) for value in weights]
    numeric_biases = [np.asarray(value, dtype=float) for value in biases]
    for layer_idx, (weight, bias) in enumerate(zip(numeric_weights, numeric_biases)):
        expected_weight_shape = (layer_sizes[layer_idx], layer_sizes[layer_idx + 1])
        if weight.shape != expected_weight_shape:
            raise ValueError(
                f"Neuro-TF W{layer_idx} has shape {weight.shape}; "
                f"expected {expected_weight_shape}"
            )
        if bias.shape != (layer_sizes[layer_idx + 1],):
            raise ValueError(
                f"Neuro-TF b{layer_idx} has shape {bias.shape}; "
                f"expected {(layer_sizes[layer_idx + 1],)}"
            )
    x_mean_array = np.asarray(x_mean, dtype=float).reshape(-1)
    x_std_array = np.asarray(x_std, dtype=float).reshape(-1)
    y_mean_array = np.asarray(y_mean, dtype=float).reshape(-1)
    y_std_array = np.asarray(y_std, dtype=float).reshape(-1)
    if x_mean_array.shape != (layer_sizes[0],) or x_std_array.shape != (layer_sizes[0],):
        raise ValueError("Neuro-TF input scaler dimensions do not match its input layer")
    if y_mean_array.shape != (n_network_outputs,) or y_std_array.shape != (n_network_outputs,):
        raise ValueError("Neuro-TF output scaler dimensions do not match its coefficient rows")
    if np.any(x_std_array == 0.0):
        raise ValueError("Neuro-TF input scaler standard deviations must be non-zero")
    if not np.all(np.isfinite(poles_array.real)) or not np.all(np.isfinite(poles_array.imag)):
        raise ValueError("Neuro-TF poles must be finite")
    if not math.isfinite(float(f_scale)) or float(f_scale) <= 0.0:
        raise ValueError("Neuro-TF f_scale must be positive and finite")
    if not math.isfinite(float(z0)) or float(z0) <= 0.0:
        raise ValueError("Reference impedance z0 must be positive and finite")

    scale_map = {
        str(key): float(value) for key, value in (parameter_input_scales or {}).items()
    }
    unknown_scales = sorted(set(scale_map) - set(parameter_names))
    if unknown_scales:
        raise ValueError(
            "Parameter input scales include names that are not model parameters: "
            + ", ".join(unknown_scales)
        )
    param_ids = unique_veriloga_identifiers(parameter_names, "param")
    scale_ids = unique_veriloga_identifiers(
        [f"{ident}_input_scale" for ident in param_ids],
        "param_input_scale",
        used_names=set(param_ids),
    )
    param_scales: list[float] = []
    param_defaults: list[float] = []
    param_model_defaults: list[float] = []
    for idx, name in enumerate(parameter_names):
        scale = float(scale_map.get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Parameter input scale for {name!r} must be positive and finite")
        model_default = float(x_mean_array[idx])
        param_scales.append(scale)
        param_model_defaults.append(model_default)
        param_defaults.append(model_default * scale)

    module_id = veriloga_identifier(module_name, "neuro_tf_va")
    port_ids = [f"p{idx}" for idx in range(1, nports + 1)]
    lines: list[str] = [
        "`include \"constants.vams\"",
        "`include \"disciplines.vams\"",
        "",
        f"module {module_id}({', '.join(port_ids)});",
        f"  inout {', '.join(port_ids)};",
        f"  electrical {', '.join(port_ids)};",
        "",
        "  parameter integer clamp_frequency = 1;",
        "  parameter real min_frequency_hz = 1.0;",
        f"  parameter real dc_equivalent_resistance_ohm = {veriloga_float(dc_resistance)}; "
        "// DC summary; selected path parameters below drive stamping when present",
        f"  parameter real z0 = {veriloga_float(z0)};",
        "  parameter real pivot_floor = 1.0e-24;",
    ]
    if dc_model_data is None:
        for canonical, identifier, path_resistance in dc_parameter_rows:
            lines.append(
                f"  parameter real {identifier} = {veriloga_float(path_resistance)}; "
                f"// selected DC path {canonical}"
            )
    for name, ident, default in zip(parameter_names, param_ids, param_defaults):
        lines.append(
            f"  parameter real {ident} = {veriloga_float(default)}; "
            f"// source VAR {name}, ADS/base units"
        )
    lines.extend(
        [
            "",
            "  // ADS/base-unit parameter divided by input_scale equals the model training VAR.",
        ]
    )
    for name, ident, scale_ident, scale in zip(
        parameter_names, param_ids, scale_ids, param_scales
    ):
        lines.append(
            f"  parameter real {scale_ident} = {veriloga_float(scale)}; "
            f"// model VAR {name} = {ident}/{scale_ident}"
        )

    lines.extend(
        [
            "",
            "  integer i;",
            "  integer j;",
            "  integer k;",
            "  integer row;",
            "  integer col;",
            "  integer idx;",
            "  integer piv;",
            "  integer pivrow;",
            "  integer dc_operating_point;",
            "  real freq_hz;",
            "  real normalized_frequency;",
            "  real omega;",
            "  real mag;",
            "  real best_mag;",
            "  real den;",
            "  real pr;",
            "  real pi;",
            "  real fr;",
            "  real fi;",
            "  real tr;",
            "  real ti;",
            f"  real basis_r [0:{n_coeffs - 1}];",
            f"  real basis_i [0:{n_coeffs - 1}];",
            f"  real coeff [0:{n_network_outputs - 1}];",
            f"  real sr [0:{matrix_size - 1}];",
            f"  real si [0:{matrix_size - 1}];",
            f"  real mr [0:{matrix_size - 1}];",
            f"  real mi [0:{matrix_size - 1}];",
            f"  real ar [0:{matrix_size - 1}];",
            f"  real ai [0:{matrix_size - 1}];",
            f"  real invr [0:{matrix_size - 1}];",
            f"  real invi [0:{matrix_size - 1}];",
            f"  real yr [0:{matrix_size - 1}];",
            f"  real yi [0:{matrix_size - 1}];",
            f"  real active_yr [0:{matrix_size - 1}];",
            f"  real active_yi [0:{matrix_size - 1}];",
        ]
    )
    for layer_idx, size in enumerate(layer_sizes):
        lines.append(f"  real l{layer_idx} [0:{size - 1}];")
    if dc_model_data is not None:
        dc_layer_sizes = dc_model_data["layer_sizes"]
        assert isinstance(dc_layer_sizes, list)
        if dc_model_data.get("representation") == "full_s_matrix":
            lines.extend(
                [
                    f"  real dc_sr [0:{matrix_size - 1}];",
                    f"  real dc_si [0:{matrix_size - 1}];",
                    f"  real dc_yr [0:{matrix_size - 1}];",
                    f"  real dc_yi [0:{matrix_size - 1}];",
                ]
            )
        elif dc_model_data.get("representation") == "full_y_matrix":
            lines.append(f"  real dc_y [0:{matrix_size - 1}];")
        else:
            dc_path_count = len(dc_model_data["paths_parsed"])  # type: ignore[arg-type]
            lines.append(f"  real dc_log_g [0:{dc_path_count - 1}];")
            lines.append(f"  real dc_g [0:{dc_path_count - 1}];")
        for layer_idx, size in enumerate(dc_layer_sizes):
            lines.append(f"  real dc_l{layer_idx} [0:{size - 1}];")

    lines.extend(["", "  analog begin"])
    lines.append(f"    freq_hz = {frequency_expression};")
    lines.append("    dc_operating_point = (freq_hz == 0.0);")
    lines.append(
        "    if (clamp_frequency != 0 && freq_hz < min_frequency_hz) "
        "freq_hz = min_frequency_hz;"
    )
    lines.append(f"    normalized_frequency = freq_hz/({veriloga_float(float(f_scale))});")
    lines.append("")
    if dc_model_data is not None:
        dc_x_mean = np.asarray(dc_model_data["x_mean"], dtype=float)
        dc_x_std = np.asarray(dc_model_data["x_std"], dtype=float)
        for idx, (ident, scale_ident) in enumerate(zip(param_ids, scale_ids)):
            lines.append(
                f"    dc_l0[{idx}] = ((({ident})/({scale_ident})) - "
                f"({veriloga_float(float(dc_x_mean[idx]))}))"
                f"/({veriloga_float(float(dc_x_std[idx]))});"
            )
        dc_weights = dc_model_data["weights"]
        dc_biases = dc_model_data["biases"]
        assert isinstance(dc_weights, list) and isinstance(dc_biases, list)
        for layer_idx, (dc_weight, dc_bias) in enumerate(zip(dc_weights, dc_biases)):
            hidden_activation = (
                str(dc_model_data["activation"])
                if layer_idx < len(dc_weights) - 1
                else None
            )
            _veriloga_layer_assignments(
                lines,
                source=f"dc_l{layer_idx}",
                dest=f"dc_l{layer_idx + 1}",
                weight=np.asarray(dc_weight, dtype=float),
                bias=np.asarray(dc_bias, dtype=float),
                activation=hidden_activation,
            )
        dc_final_layer = f"dc_l{len(dc_model_data['layer_sizes']) - 1}"  # type: ignore[arg-type]
        dc_y_mean = np.asarray(dc_model_data["y_mean"], dtype=float)
        dc_y_std = np.asarray(dc_model_data["y_std"], dtype=float)
        if dc_model_data.get("representation") == "full_s_matrix":
            for idx in range(matrix_size):
                lines.append(
                    f"    dc_sr[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
                imag_idx = matrix_size + idx
                lines.append(
                    f"    dc_si[{idx}] = {dc_final_layer}[{imag_idx}]*"
                    f"({veriloga_float(float(dc_y_std[imag_idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[imag_idx]))});"
                )
            dc_real_by_flat = [f"dc_yr[{idx}]" for idx in range(matrix_size)]
            dc_imag_by_flat = [f"dc_yi[{idx}]" for idx in range(matrix_size)]
        elif dc_model_data.get("representation") == "full_y_matrix":
            for idx in range(len(dc_y_mean)):
                lines.append(
                    f"    dc_y[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
            dc_real_by_flat = [f"dc_y[{idx}]" for idx in range(matrix_size)]
            dc_imag_by_flat = None
        else:
            dc_log_min = veriloga_float(float(dc_model_data["log_conductance_min"]))
            dc_log_max = veriloga_float(float(dc_model_data["log_conductance_max"]))
            for idx in range(len(dc_y_mean)):
                lines.append(
                    f"    dc_log_g[{idx}] = {dc_final_layer}[{idx}]*"
                    f"({veriloga_float(float(dc_y_std[idx]))}) + "
                    f"({veriloga_float(float(dc_y_mean[idx]))});"
                )
                lines.append(
                    f"    dc_g[{idx}] = exp(min(max(dc_log_g[{idx}], {dc_log_min}), "
                    f"{dc_log_max}));"
                )
            dc_real_by_flat = _veriloga_dc_matrix_expressions(
                nports,
                dc_model_data["paths_parsed"],  # type: ignore[arg-type]
                [f"dc_g[{idx}]" for idx in range(len(dc_y_mean))],
            )
            dc_imag_by_flat = None
        lines.append("")
    else:
        dc_imag_by_flat = None
    for idx, (ident, scale_ident) in enumerate(zip(param_ids, scale_ids)):
        lines.append(
            f"    l0[{idx}] = ((({ident})/({scale_ident})) - "
            f"({veriloga_float(float(x_mean_array[idx]))}))"
            f"/({veriloga_float(float(x_std_array[idx]))}); // {parameter_names[idx]}"
        )
    lines.append("")

    for layer_idx, (weight, bias) in enumerate(zip(numeric_weights, numeric_biases)):
        hidden_activation = activation if layer_idx < len(numeric_weights) - 1 else None
        _veriloga_layer_assignments(
            lines,
            source=f"l{layer_idx}",
            dest=f"l{layer_idx + 1}",
            weight=weight,
            bias=bias,
            activation=hidden_activation,
        )
    final_layer = f"l{len(layer_sizes) - 1}"
    for idx in range(n_network_outputs):
        lines.append(
            f"    coeff[{idx}] = {final_layer}[{idx}]"
            f"*({veriloga_float(float(y_std_array[idx]))}) "
            f"+ ({veriloga_float(float(y_mean_array[idx]))});"
        )
    lines.extend(["", "    // Fixed-pole rational basis: 1, 1/(j*f/f_scale - pole_k)."])
    lines.append("    basis_r[0] = 1.0;")
    lines.append("    basis_i[0] = 0.0;")
    for pole_idx, pole in enumerate(poles_array, start=1):
        real_part = -float(pole.real)
        imag_part = float(pole.imag)
        lines.append(
            f"    den = ({veriloga_float(real_part)})*({veriloga_float(real_part)}) + "
            f"(normalized_frequency - ({veriloga_float(imag_part)}))"
            f"*(normalized_frequency - ({veriloga_float(imag_part)}));"
        )
        lines.append("    if (den < 1.0e-30) den = 1.0e-30;")
        lines.append(
            f"    basis_r[{pole_idx}] = ({veriloga_float(real_part)})/den;"
        )
        lines.append(
            f"    basis_i[{pole_idx}] = "
            f"-(normalized_frequency - ({veriloga_float(imag_part)}))/den;"
        )
    lines.append("")

    coefficient_half = n_sparams * n_coeffs
    for sparam_idx, label in enumerate(sparam_labels):
        row_number, col_number = sparam_indices(label) or (0, 0)
        flat = (row_number - 1) * nports + (col_number - 1)
        lines.append(f"    sr[{flat}] = 0.0; // {label} real")
        lines.append(f"    si[{flat}] = 0.0; // {label} imag")
        for coeff_idx in range(n_coeffs):
            real_index = sparam_idx * n_coeffs + coeff_idx
            imag_index = coefficient_half + real_index
            lines.append(
                f"    sr[{flat}] = sr[{flat}] + coeff[{real_index}]"
                f"*basis_r[{coeff_idx}] - coeff[{imag_index}]"
                f"*basis_i[{coeff_idx}];"
            )
            lines.append(
                f"    si[{flat}] = si[{flat}] + coeff[{real_index}]"
                f"*basis_i[{coeff_idx}] + coeff[{imag_index}]"
                f"*basis_r[{coeff_idx}];"
            )
        lines.append("")

    _append_veriloga_s_to_y_conversion(lines, nports)
    if dc_model_data is not None and dc_model_data.get("representation") == "full_s_matrix":
        _append_veriloga_s_to_y_conversion(
            lines,
            nports,
            s_real="dc_sr",
            s_imag="dc_si",
            y_real="dc_yr",
            y_imag="dc_yi",
        )
    _append_veriloga_dc_or_fitted_y_assignments(
        lines,
        nports,
        [f"yr[{flat}]" for flat in range(matrix_size)],
        [f"yi[{flat}]" for flat in range(matrix_size)],
        dc_real_by_flat,
        dc_imag_by_flat,
    )
    _append_veriloga_port_stamps(
        lines,
        port_ids,
        [f"active_yr[{flat}]" for flat in range(matrix_size)],
        [f"active_yi[{flat}]" for flat in range(matrix_size)],
    )
    lines.extend(["  end", "endmodule", ""])

    coefficient_columns = [
        f"{label}_c{coeff_idx}_real"
        for label in sparam_labels
        for coeff_idx in range(n_coeffs)
    ] + [
        f"{label}_c{coeff_idx}_imag"
        for label in sparam_labels
        for coeff_idx in range(n_coeffs)
    ]
    manifest: dict[str, object] = {
        "module_name": module_id,
        "model_kind": "Neuro-TF",
        "nports": nports,
        "parameter_names": list(parameter_names),
        "parameter_identifiers": param_ids,
        "parameter_scale_identifiers": scale_ids,
        "parameter_input_scales": {
            name: float(scale) for name, scale in zip(parameter_names, param_scales)
        },
        "parameter_defaults": param_defaults,
        "parameter_instance_defaults": param_defaults,
        "parameter_model_defaults": param_model_defaults,
        "sparam_labels": list(sparam_labels),
        "input_columns": list(parameter_names),
        "output_columns": coefficient_columns,
        "output_domain": "s",
        "representation": "fixed-pole rational transfer function",
        "frequency_expression": frequency_expression,
        "z0": float(z0),
        "activation": activation,
        "layer_sizes": list(layer_sizes),
        "folded_input_scaler": False,
        "folded_output_scaler": False,
        "uses_coarse_inputs": False,
        "adds_coarse_to_output": False,
        "embedded_coarse_model": False,
        "fully_self_contained": True,
        "n_poles": int(len(poles_array)),
        "n_coeffs_per_sparam": int(n_coeffs),
        "poles_real": [float(value) for value in poles_array.real],
        "poles_imag": [float(value) for value in poles_array.imag],
        "f_scale": float(f_scale),
        "implementation_note": (
            "Direct Verilog-A embeds the trained geometry-to-coefficient MLP and fixed "
            "rational poles, evaluates the S-matrix at simulator frequency, and converts "
            "it to a small-signal Y-matrix. Intended for AC/SP analyses."
        ),
        "dc_equivalent_resistance_ohm": dc_resistance,
        "dc_port_paths": (
            list(dc_model_data.get("port_paths", []))
            if dc_model_data is not None
            else list(dc_path_resistances)
            if dc_path_resistances is not None
            else None
        ),
        "dc_matrix_entries": (
            list(dc_model_data.get("matrix_entries", []))
            if dc_model_data is not None
            else []
        ),
        "dc_sparameter_entries": (
            list(dc_model_data.get("sparameter_entries", []))
            if dc_model_data is not None
            else []
        ),
        "dc_port_resistances_ohm": (
            None
            if dc_model_data is not None
            and dc_model_data.get("representation") in {"full_s_matrix", "full_y_matrix"}
            else dc_path_resistances
        ),
        "dc_port_parameter_identifiers": {
            canonical: identifier
            for canonical, identifier, _ in dc_parameter_rows
        } if dc_model_data is None else {},
        "dc_resistance_source_kind": dc_source_kind,
        "dc_model_kind": (
            str(dc_model_data.get("kind")) if dc_model_data is not None else None
        ),
        "dc_geometry_dependent": dc_model_data is not None,
        "dc_model_metadata": (
            dc_model_data.get("metadata") if dc_model_data is not None else None
        ),
        "dc_response_topology": (
            "Geometry-dependent full ordered complex S matrix; every real and "
            "imaginary S-parameter component is represented and converted to Y "
            "independently of the fitted coefficient response at zero Hz"
            if dc_model_data is not None
            and dc_model_data.get("representation") == "full_s_matrix"
            else
            "Geometry-dependent full ordered real Y matrix; every S-parameter entry "
            "is represented and the fitted coefficient response is bypassed at zero Hz"
            if dc_model_data is not None
            and dc_model_data.get("representation") == "full_y_matrix"
            else "Geometry-dependent selected-path conductance graph; only declared "
            "paths are connected and the fitted coefficient response is bypassed at zero Hz"
            if dc_model_data is not None
            else "Parameter-independent selected-path resistor graph; only declared "
            "paths are connected and the fitted coefficient response is bypassed at zero Hz"
            if dc_path_resistances is not None
            else "Legacy parameter-independent equal-resistance complete graph; the "
            "fitted coefficient response is bypassed at zero Hz"
        ),
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_sparameters_geometry": (
            "parameter_model_defaults" if dc_model_data is not None else "static_legacy"
        ),
        "dc_sparameters": {
            label: {
                "real": float(value.real),
                "imag": float(value.imag),
            }
            for label, value in zip(
                sparam_labels,
                (
                    _dc_export_default_s_values(
                        dc_model_data,
                        sparam_labels,
                        z0,
                    )
                    if dc_model_data is not None
                    else dc_sparameter_values(
                        sparam_labels,
                        dc_resistance,
                        dc_port_resistances_ohm=dc_path_resistances,
                        z0=z0,
                    )
                ),
            )
        },
    }
    return "\n".join(lines), manifest


def write_neurotf_veriloga_package(
    out_dir: Path,
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    poles: np.ndarray,
    f_scale: float,
    z0: float = 50.0,
    frequency_expression: str = "$freq",
    parameter_input_scales: dict[str, float] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
    source_model_dir: str | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a self-contained Neuro-TF Verilog-A package."""

    out_dir.mkdir(parents=True, exist_ok=True)
    module_id = veriloga_identifier(module_name, "neuro_tf_va")
    va_text, manifest = neurotf_veriloga_module_text(
        module_name=module_id,
        parameter_names=parameter_names,
        sparam_labels=sparam_labels,
        activation=activation,
        layer_sizes=layer_sizes,
        weights=weights,
        biases=biases,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        poles=poles,
        f_scale=f_scale,
        z0=z0,
        frequency_expression=frequency_expression,
        parameter_input_scales=parameter_input_scales,
        dc_equivalent_resistance_ohm=dc_equivalent_resistance_ohm,
        dc_resistance_source_kind=dc_resistance_source_kind,
        dc_port_resistances_ohm=dc_port_resistances_ohm,
        dc_model=dc_model,
    )
    va_name = f"{module_id}.va"
    manifest_name = "veriloga_manifest.json"
    manifest.update(
        {
            "format": "direct_veriloga_neurotf",
            "veriloga_file": va_name,
            "source_model_dir": source_model_dir,
            "reference_note": (
                "Generated from saved local model.npz weights and fixed poles. ADS "
                "Verilog-A frequency defaults to $freq; verify against the target ADS release."
            ),
        }
    )
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / va_name).write_text(va_text)
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    response_relation = (
        "- Neural output: complex fixed-pole rational coefficients\n"
        "- Runtime response: `Sij(f) = c0 + sum(c_k / (j*f/f_scale - pole_k))`\n"
        f"- Reference impedance used for S-to-Y conversion: `{z0:g} ohm`\n"
        "- Current relation: `Y = (I - S) * inverse(I + S) / Z0`, then "
        "`Iport = Y * Vport`"
    )
    (out_dir / "VERILOGA_README.md").write_text(
        _veriloga_readme(
            model_kind="Neuro-TF",
            module_name=module_id,
            va_file=va_name,
            manifest_name=manifest_name,
            nports=int(manifest["nports"]),
            parameter_names=parameter_names,
            parameter_identifiers=manifest["parameter_identifiers"],
            parameter_scale_identifiers=manifest["parameter_scale_identifiers"],
            parameter_input_scales=[
                manifest["parameter_input_scales"][name] for name in parameter_names
            ],
            parameter_instance_defaults=manifest["parameter_instance_defaults"],
            input_columns=manifest["input_columns"],
            output_columns=manifest["output_columns"],
            freq_transform="fixed-pole rational basis using freq_hz/f_scale",
            frequency_expression=frequency_expression,
            z0=z0,
            dc_equivalent_resistance_ohm=float(
                manifest["dc_equivalent_resistance_ohm"]
            ),
            dc_port_resistances_ohm=manifest.get("dc_port_resistances_ohm"),
            dc_model_kind=manifest.get("dc_model_kind"),
            dc_matrix_entries=manifest.get("dc_matrix_entries"),
            output_domain="s",
            folded_input_scaler=False,
            folded_output_scaler=False,
            uses_coarse_inputs=False,
            adds_coarse_to_output=False,
            embedded_coarse_model=False,
            extra_notes=[
                "The model is self-contained: no MDIF table, Python runtime, or external "
                "coarse model is required by the generated component.",
                "The rational frequency response and admittance stamping target AC/SP "
                "analysis, not causal transient behavior.",
            ],
            response_relation_override=response_relation,
        )
    )
    return manifest


def _ads_hb_activation(expression: str, activation: str | None) -> str:
    """Return an ADS simulator-expression activation."""

    if activation is None:
        return expression
    normalized = activation.strip().lower()
    if normalized == "tanh":
        return f"tanh({expression})"
    if normalized == "relu":
        return f"max(({expression}),0.0)"
    raise ValueError(f"Unsupported ADS HB activation {activation!r}")


ADS_HB_RESERVED = {
    *{name.lower() for name in VERILOGA_RESERVED},
    "freq",
    "j",
    "omega",
    "pi",
    "temp",
    "time",
    "z0",
}


def _ads_hb_identifier(name: str, fallback: str) -> str:
    identifier = veriloga_identifier(name, fallback)
    if identifier.lower() in ADS_HB_RESERVED:
        identifier = f"{identifier}_p"
    return identifier


def _unique_ads_hb_identifiers(
    names: Sequence[str],
    fallback_prefix: str,
    used_names: set[str] | None = None,
) -> list[str]:
    used = {value.lower() for value in (used_names or set())}
    result: list[str] = []
    for idx, name in enumerate(names):
        base = _ads_hb_identifier(name, f"{fallback_prefix}{idx + 1}")
        identifier = base
        suffix = 2
        while identifier.lower() in used:
            identifier = f"{base}_{suffix}"
            suffix += 1
        used.add(identifier.lower())
        result.append(identifier)
    return result


def _ads_hb_weighted_sum(
    bias: float,
    source_names: Sequence[str],
    weights: np.ndarray,
) -> str:
    terms = [f"({veriloga_float(float(bias))})"]
    for source_name, weight in zip(source_names, np.asarray(weights, dtype=float)):
        if float(weight) == 0.0:
            continue
        terms.append(f"({veriloga_float(float(weight))})*({source_name})")
    return "+".join(terms)


def _fold_mlp_scalers_into_layers(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Fold affine input/output standardizers into an MLP exactly."""

    folded_weights = [np.asarray(value, dtype=float).copy() for value in weights]
    folded_biases = [np.asarray(value, dtype=float).copy() for value in biases]
    if not folded_weights or len(folded_weights) != len(folded_biases):
        raise ValueError("Scaler folding requires matching non-empty MLP layers")
    x_mean_array = np.asarray(x_mean, dtype=float)
    x_std_array = np.asarray(x_std, dtype=float)
    y_mean_array = np.asarray(y_mean, dtype=float)
    y_std_array = np.asarray(y_std, dtype=float)
    if np.any(x_std_array == 0.0):
        raise ValueError("Input scaler standard deviations must be non-zero")

    folded_weights[0] = folded_weights[0] / x_std_array[:, None]
    folded_biases[0] = folded_biases[0] - x_mean_array @ folded_weights[0]
    folded_weights[-1] = folded_weights[-1] * y_std_array[None, :]
    folded_biases[-1] = folded_biases[-1] * y_std_array + y_mean_array
    return folded_weights, folded_biases


def _append_ads_hb_mlp_equations(
    lines: list[str],
    prefix: str,
    feature_expressions: Sequence[str],
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    fold_scalers: bool = False,
) -> list[str]:
    """Append shared ADS equations and return physical-output expression names."""

    numeric_weights = [np.asarray(value, dtype=float) for value in weights]
    numeric_biases = [np.asarray(value, dtype=float) for value in biases]
    x_mean_array = np.asarray(x_mean, dtype=float)
    x_std_array = np.asarray(x_std, dtype=float)
    y_mean_array = np.asarray(y_mean, dtype=float)
    y_std_array = np.asarray(y_std, dtype=float)
    sizes = [int(value) for value in layer_sizes]
    if len(sizes) < 2 or len(numeric_weights) != len(sizes) - 1:
        raise ValueError("ADS HB MLP weights and layer sizes are inconsistent")
    if len(numeric_biases) != len(numeric_weights):
        raise ValueError("ADS HB MLP weights and biases are inconsistent")
    if sizes[0] != len(feature_expressions):
        raise ValueError(
            f"ADS HB MLP expects {sizes[0]} features, got {len(feature_expressions)}"
        )
    if x_mean_array.shape != (sizes[0],) or x_std_array.shape != (sizes[0],):
        raise ValueError("ADS HB MLP input scaler dimensions are inconsistent")
    if y_mean_array.shape != (sizes[-1],) or y_std_array.shape != (sizes[-1],):
        raise ValueError("ADS HB MLP output scaler dimensions are inconsistent")
    if np.any(x_std_array == 0.0):
        raise ValueError("ADS HB MLP input scaler standard deviations must be non-zero")

    for layer_idx, (weight, bias) in enumerate(zip(numeric_weights, numeric_biases)):
        expected = (sizes[layer_idx], sizes[layer_idx + 1])
        if weight.shape != expected or bias.shape != (sizes[layer_idx + 1],):
            raise ValueError(
                f"ADS HB MLP layer {layer_idx} has shapes {weight.shape}/{bias.shape}; "
                f"expected {expected}/{(sizes[layer_idx + 1],)}"
            )

    if fold_scalers:
        numeric_weights, numeric_biases = _fold_mlp_scalers_into_layers(
            numeric_weights,
            numeric_biases,
            x_mean_array,
            x_std_array,
            y_mean_array,
            y_std_array,
        )

    if fold_scalers:
        source_names = list(feature_expressions)
    else:
        source_names = []
        for idx, expression in enumerate(feature_expressions):
            name = f"{prefix}_x{idx}"
            lines.append(
                f"{name}=(({expression})-({veriloga_float(float(x_mean_array[idx]))}))"
                f"/({veriloga_float(float(x_std_array[idx]))})"
            )
            source_names.append(name)

    for layer_idx, (weight, bias) in enumerate(zip(numeric_weights, numeric_biases)):
        dest_names: list[str] = []
        hidden_activation = activation if layer_idx < len(numeric_weights) - 1 else None
        for out_idx in range(sizes[layer_idx + 1]):
            dest = f"{prefix}_l{layer_idx + 1}_{out_idx}"
            expression = _ads_hb_weighted_sum(
                float(bias[out_idx]),
                source_names,
                weight[:, out_idx],
            )
            lines.append(f"{dest}={_ads_hb_activation(expression, hidden_activation)}")
            dest_names.append(dest)
        source_names = dest_names

    if fold_scalers:
        return source_names

    outputs: list[str] = []
    for idx, source in enumerate(source_names):
        name = f"{prefix}_out{idx}"
        lines.append(
            f"{name}=({source})*({veriloga_float(float(y_std_array[idx]))})+"
            f"({veriloga_float(float(y_mean_array[idx]))})"
        )
        outputs.append(name)
    return outputs


def _ads_hb_frequency_features(freq_transform: str) -> list[str]:
    safe_frequency = "max(abs(freq),1.0)"
    if freq_transform == "log":
        return [f"log10({safe_frequency})"]
    if freq_transform == "linear":
        return [safe_frequency]
    if freq_transform == "log-linear":
        return [f"log10({safe_frequency})", safe_frequency]
    raise ValueError(f"Unsupported ADS HB frequency transform {freq_transform!r}")


def _ads_hb_parameter_configuration(
    parameter_names: Sequence[str],
    x_mean: np.ndarray,
    parameter_input_scales: dict[str, float] | None,
) -> tuple[list[str], list[str], list[float], list[float], list[str]]:
    scale_map = {
        str(key): float(value) for key, value in (parameter_input_scales or {}).items()
    }
    unknown_scales = sorted(set(scale_map) - set(parameter_names))
    if unknown_scales:
        raise ValueError(
            "Parameter input scales include names that are not model parameters: "
            + ", ".join(unknown_scales)
        )
    used_names = {"z0", *(f"p{idx}" for idx in range(1, 100))}
    param_ids = _unique_ads_hb_identifiers(
        parameter_names,
        "param",
        used_names=used_names,
    )
    scale_ids = _unique_ads_hb_identifiers(
        [f"{identifier}_input_scale" for identifier in param_ids],
        "param_input_scale",
        used_names={*used_names, *param_ids},
    )
    x_mean_array = np.asarray(x_mean, dtype=float)
    scales: list[float] = []
    model_defaults: list[float] = []
    instance_defaults: list[float] = []
    feature_expressions: list[str] = []
    for idx, (name, identifier, scale_identifier) in enumerate(
        zip(parameter_names, param_ids, scale_ids)
    ):
        scale = float(scale_map.get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Parameter input scale for {name!r} must be positive and finite")
        model_default = float(x_mean_array[idx])
        scales.append(scale)
        model_defaults.append(model_default)
        instance_defaults.append(model_default * scale)
        feature_expressions.append(f"({identifier})/({scale_identifier})")
    return param_ids, scale_ids, scales, instance_defaults, feature_expressions


def _append_ads_hb_sdd(
    lines: list[str],
    instance_name: str,
    port_ids: Sequence[str],
    response_names: Sequence[str],
    response_domain: str,
    dc_conductance: np.ndarray | None = None,
    dc_conductance_names: Sequence[str] | None = None,
    combine_dc_rf: bool = False,
) -> None:
    nports = len(port_ids)
    if len(response_names) != nports * nports:
        raise ValueError("ADS HB response matrix does not match the electrical port count")
    nodes = " ".join(f"{port} 0" for port in port_ids)

    def append_tokens(tokens: Sequence[str]) -> None:
        for idx, token in enumerate(tokens):
            suffix = " \\" if idx < len(tokens) - 1 else ""
            lines.append(f"  {token}{suffix}")

    if response_domain != "y":
        raise ValueError(
            "ADS HB circuit stamping requires Y-domain response weights; "
            "convert fitted S responses before creating the SDD"
        )
    dc_matrix = (
        None
        if dc_conductance is None
        else np.asarray(dc_conductance, dtype=float)
    )
    if dc_matrix is not None and (
        dc_matrix.shape != (nports, nports) or not np.all(np.isfinite(dc_matrix))
    ):
        raise ValueError("ADS HB DC conductance matrix is invalid")
    if dc_conductance_names is not None and len(dc_conductance_names) != nports * nports:
        raise ValueError("ADS HB dynamic DC conductance matrix has invalid dimensions")
    if dc_matrix is not None and dc_conductance_names is not None:
        raise ValueError("ADS HB DC stamping must use either static values or expressions")

    if combine_dc_rf:
        # Trial topology: use one explicit-current SDD and select its complete
        # frequency weight between the exact-DC and fitted-RF admittances.  This
        # is algebraically identical to summing the two mutually exclusive
        # parallel SDDs below, but presents only one dense control/stamp graph to
        # ADS.  Keep the established two-SDD implementation as the default until
        # circuit-level timing and response equivalence have been verified.
        combined_tokens = [f"SDD:{instance_name}_combined {nodes}"]
        weight_index = 2
        for row in range(nports):
            port_number = row + 1
            combined_tokens.append(f"I[{port_number},0]=0.0")
            for col in range(nports):
                control_number = col + 1
                flat = row * nports + col
                response_name = response_names[flat]
                if dc_conductance_names is not None:
                    conductance = str(dc_conductance_names[flat])
                elif dc_matrix is not None:
                    conductance = veriloga_float(float(dc_matrix[row, col]))
                else:
                    conductance = "0.0"
                combined_tokens.append(
                    f"I[{port_number},{weight_index}]=_v{control_number}"
                )
                combined_tokens.append(
                    f"H[{weight_index}]=if (freq equals 0) then {conductance} "
                    f"else {response_name} endif"
                )
                weight_index += 1
        append_tokens(combined_tokens)
        return

    # Explicit-current SDDs use ordinary nodal analysis and introduce no
    # branch-current unknowns. Keep RF and DC in separate parallel branches:
    # the RF branch is exactly open at zero frequency, while the DC branch is
    # exactly open at every non-zero frequency.
    rf_tokens = [f"SDD:{instance_name}_rf {nodes}"]
    weight_index = 2
    for row in range(nports):
        port_number = row + 1
        rf_tokens.append(f"I[{port_number},0]=0.0")
        for col in range(nports):
            control_number = col + 1
            response_name = response_names[row * nports + col]
            rf_tokens.append(f"I[{port_number},{weight_index}]=_v{control_number}")
            rf_tokens.append(
                f"H[{weight_index}]=if (freq equals 0) then 0.0 "
                f"else {response_name} endif"
            )
            weight_index += 1
    append_tokens(rf_tokens)
    if dc_matrix is None and dc_conductance_names is None:
        return
    lines.append("")

    # Stamp DC as the extracted resistor graph itself, never as an S-matrix
    # approximation and never through the fitted RF response. At RF every
    # weight is zero, so this parallel branch contributes no current.
    dc_tokens = [f"SDD:{instance_name}_dc {nodes}"]
    weight_index = 2
    for row in range(nports):
        port_number = row + 1
        dc_tokens.append(f"I[{port_number},0]=0.0")
        for col in range(nports):
            control_number = col + 1
            conductance = (
                str(dc_conductance_names[row * nports + col])
                if dc_conductance_names is not None
                else veriloga_float(float(dc_matrix[row, col]))
            )
            dc_tokens.append(f"I[{port_number},{weight_index}]=_v{control_number}")
            dc_tokens.append(
                f"H[{weight_index}]=if (freq equals 0) then {conductance} else 0.0 endif"
            )
            weight_index += 1
    append_tokens(dc_tokens)


def _append_ads_hb_dc_conductance_model(
    lines: list[str],
    prefix: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    parameter_features: Sequence[str],
    dc_model: dict[str, object],
    fold_scalers: bool = False,
) -> list[str]:
    """Append a geometry-only DC MLP and return row-major Y expressions."""

    if [str(value) for value in dc_model["parameter_names"]] != list(parameter_names):  # type: ignore[index]
        raise ValueError("DC model parameters do not match the exported RF model")
    if [str(value) for value in dc_model["sparam_labels"]] != list(sparam_labels):  # type: ignore[index]
        raise ValueError("DC model S-parameter order does not match the RF model")
    nports = infer_complete_sparameter_ports(sparam_labels)
    representation = str(dc_model.get("representation", "path_conductance"))
    paths = (
        []
        if representation in {"full_s_matrix", "full_y_matrix"}
        else parse_dc_port_paths(dc_model["port_paths"], nports)
    )
    outputs = _append_ads_hb_mlp_equations(
        lines,
        f"{prefix}_net",
        parameter_features,
        str(dc_model["activation"]),
        dc_model["layer_sizes"],  # type: ignore[arg-type]
        dc_model["weights"],  # type: ignore[arg-type]
        dc_model["biases"],  # type: ignore[arg-type]
        np.asarray(dc_model["x_mean"], dtype=float),
        np.asarray(dc_model["x_std"], dtype=float),
        np.asarray(dc_model["y_mean"], dtype=float),
        np.asarray(dc_model["y_std"], dtype=float),
        fold_scalers=fold_scalers,
    )
    if representation == "full_s_matrix":
        matrix_size = nports * nports
        if len(outputs) != 2 * matrix_size:
            raise ValueError("Full-S DC model output count is not complex N-port S")
        s_names: list[str] = []
        for idx in range(matrix_size):
            name = f"{prefix}_s{idx}"
            lines.append(
                f"{name}=complex({outputs[idx]},{outputs[matrix_size + idx]})"
            )
            s_names.append(name)
        return _append_ads_hb_s_to_y_equations(
            lines,
            f"{prefix}_stoy",
            s_names,
            nports,
            float(dc_model["z0"]),
        )
    if representation == "full_y_matrix":
        if len(outputs) != nports * nports:
            raise ValueError("Full-matrix DC model output count is not N-port Y")
        return outputs
    if len(outputs) != len(paths):
        raise ValueError("DC model output count does not match selected path count")
    log_min = veriloga_float(float(dc_model["log_conductance_min"]))
    log_max = veriloga_float(float(dc_model["log_conductance_max"]))
    conductance_names: list[str] = []
    for idx, output_name in enumerate(outputs):
        name = f"{prefix}_g{idx}"
        lines.append(f"{name}=exp(min(max({output_name},{log_min}),{log_max}))")
        conductance_names.append(name)
    entry_terms: list[list[str]] = [[] for _ in range(nports * nports)]
    for (first, second, _), conductance_name in zip(paths, conductance_names):
        entry_terms[first * nports + first].append(conductance_name)
        if second is not None:
            entry_terms[second * nports + second].append(conductance_name)
            entry_terms[first * nports + second].append(f"-({conductance_name})")
            entry_terms[second * nports + first].append(f"-({conductance_name})")
    matrix_names: list[str] = []
    for flat, terms in enumerate(entry_terms):
        name = f"{prefix}_y{flat}"
        lines.append(f"{name}=" + ("+".join(terms) if terms else "0.0"))
        matrix_names.append(name)
    return matrix_names


def _append_ads_hb_s_to_y_equations(
    lines: list[str],
    prefix: str,
    s_response_names: Sequence[str],
    nports: int,
    z0: float,
) -> list[str]:
    """Append frequency-only equations for Y=(I+S)^-1(I-S)/z0.

    The ADS SDD then remains an explicit voltage-controlled current device.
    ``I+S`` and ``I-S`` commute because both are polynomials in the same S
    matrix, so this solve is equivalent to the conventional right-side form.
    """

    matrix_size = nports * nports
    if len(s_response_names) != matrix_size:
        raise ValueError("ADS HB S-to-Y conversion matrix dimensions are inconsistent")
    equation_prefix = _ads_hb_identifier(prefix, "hb_stoy")
    a_names: list[list[str]] = [[""] * nports for _ in range(nports)]
    b_names: list[list[str]] = [[""] * nports for _ in range(nports)]
    for row in range(nports):
        for col in range(nports):
            s_name = s_response_names[row * nports + col]
            a_name = f"{equation_prefix}_a0_{row}_{col}"
            b_name = f"{equation_prefix}_b0_{row}_{col}"
            if row == col:
                lines.append(f"{a_name}=1.0+({s_name})")
                lines.append(f"{b_name}=1.0-({s_name})")
            else:
                lines.append(f"{a_name}={s_name}")
                lines.append(f"{b_name}=-({s_name})")
            a_names[row][col] = a_name
            b_names[row][col] = b_name

    # Gauss-Jordan elimination is expressed through named scalar equations so
    # ADS can share intermediate results across all matrix elements. For a
    # passive finite-Y network, I+S and its elimination pivots are non-singular.
    for pivot in range(nports):
        stage = pivot + 1
        pivot_name = a_names[pivot][pivot]
        next_a: list[list[str]] = [[""] * nports for _ in range(nports)]
        next_b: list[list[str]] = [[""] * nports for _ in range(nports)]
        for col in range(nports):
            a_name = f"{equation_prefix}_a{stage}_{pivot}_{col}"
            b_name = f"{equation_prefix}_b{stage}_{pivot}_{col}"
            lines.append(f"{a_name}=({a_names[pivot][col]})/({pivot_name})")
            lines.append(f"{b_name}=({b_names[pivot][col]})/({pivot_name})")
            next_a[pivot][col] = a_name
            next_b[pivot][col] = b_name
        for row in range(nports):
            if row == pivot:
                continue
            factor = a_names[row][pivot]
            for col in range(nports):
                a_name = f"{equation_prefix}_a{stage}_{row}_{col}"
                b_name = f"{equation_prefix}_b{stage}_{row}_{col}"
                lines.append(
                    f"{a_name}=({a_names[row][col]})-({factor})*"
                    f"({next_a[pivot][col]})"
                )
                lines.append(
                    f"{b_name}=({b_names[row][col]})-({factor})*"
                    f"({next_b[pivot][col]})"
                )
                next_a[row][col] = a_name
                next_b[row][col] = b_name
        a_names = next_a
        b_names = next_b

    y_names: list[str] = []
    z0_text = veriloga_float(z0)
    for row in range(nports):
        for col in range(nports):
            name = f"{equation_prefix}_y_{row}_{col}"
            lines.append(f"{name}=({b_names[row][col]})/({z0_text})")
            y_names.append(name)
    return y_names


def _ads_hb_instance_call(
    module_name: str,
    instance_name: str,
    nports: int,
    parameter_ids: Sequence[str],
    value_suffix: str,
) -> str:
    nodes = " ".join(
        f"{instance_name.lower()}_p{idx}" for idx in range(1, nports + 1)
    )
    call = f"{module_name}:{instance_name} {nodes}"
    if parameter_ids:
        call += " " + " ".join(
            f"{identifier}={identifier}_{value_suffix}" for identifier in parameter_ids
        )
    return call


def _ads_hb_instance_template(
    module_name: str,
    netlist_name: str,
    nports: int,
    parameter_ids: Sequence[str],
) -> str:
    call_a = _ads_hb_instance_call(module_name, "X1", nports, parameter_ids, "A")
    call_b = _ads_hb_instance_call(module_name, "X2", nports, parameter_ids, "B")
    return f"""; ADS HB instance-call template -- documentation only
; Do not include this file unchanged. Copy the calls you need into a separate
; top-level ADS netlist fragment after {netlist_name}, then replace the node
; labels and parameter expressions. Values such as W_A may be top-level VARs.
;
; {call_a}
; {call_b}
"""


def _ads_hb_readme(
    model_kind: str,
    module_name: str,
    netlist_name: str,
    manifest_name: str,
    nports: int,
    parameter_names: Sequence[str],
    parameter_ids: Sequence[str],
    parameter_scale_ids: Sequence[str],
    parameter_scales: Sequence[float],
    parameter_instance_defaults: Sequence[float],
    response_domain: str,
    instance_template_name: str = "ADS_HB_INSTANCE_TEMPLATE.txt",
    readme_name: str = "ADS_HB_README.md",
    combined_sdd: bool = False,
    fold_mlp_scalers: bool = False,
    extra_notes: Sequence[str] | None = None,
) -> str:
    parameter_rows = "\n".join(
        f"| `{source}` | `{identifier}` | `{default:.12g}` | "
        f"`{scale_identifier}` | `{scale:.12g}` |"
        for source, identifier, default, scale_identifier, scale in zip(
            parameter_names,
            parameter_ids,
            parameter_instance_defaults,
            parameter_scale_ids,
            parameter_scales,
        )
    )
    if not parameter_rows:
        parameter_rows = "| _(none)_ | _(none)_ | _(none)_ | _(none)_ | _(none)_ |"
    instance_a_call = _ads_hb_instance_call(
        module_name, "X1", nports, parameter_ids, "A"
    )
    instance_b_call = _ads_hb_instance_call(
        module_name, "X2", nports, parameter_ids, "B"
    )
    notes = "\n".join(f"- {note}" for note in (extra_notes or []))
    if notes:
        notes = f"\n## Notes\n\n{notes}\n"
    if combined_sdd:
        dc_rf_behavior = """The export contains one explicit-current SDD. Each
frequency weight selects the separately extracted geometry-dependent DC
admittance at `freq=0` and the fitted RF admittance at every non-zero spectral
frequency. This is algebraically equivalent to the default package's two
mutually exclusive parallel SDDs, but exposes only one dense SDD control/stamp
graph to ADS for performance evaluation."""
    else:
        dc_rf_behavior = """The export contains two explicit-current SDD branches.
At `freq=0`, the RF branch contributes exactly zero current and the DC branch
directly stamps the separately extracted conductance matrix. For current models
this matrix is evaluated from geometry only; legacy models use a fixed matrix.
At every non-zero spectral frequency, the DC branch contributes exactly zero
current and only the fitted RF surrogate is used."""
    scaler_behavior = (
        "The trial folds every embedded MLP input standardizer into its first "
        "affine layer and every output inverse-standardizer into its final affine "
        "layer. No separate neural input/output scaling equations are emitted. "
        "The transformation is algebraically exact and does not change the trained "
        "network response."
        if fold_mlp_scalers
        else "The package emits the trained MLP input standardization and output "
        "inverse-standardization as explicit ADS equations."
    )
    return f"""# ADS Harmonic-Balance Passive Network

This package contains a self-contained ADS Symbolically Defined Device (SDD)
subnetwork for the trained {model_kind} passive structure. It is linear and
power independent. ADS evaluates every complex {response_domain.upper()}-matrix
weight at the frequency of each HB spectral component, so active devices around
this network can compress normally without making the passive structure itself
power dependent.

## Files

- `{netlist_name}`: self-contained ADS simulator subnetwork
- `{manifest_name}`: model contract and generation metadata
- `{instance_template_name}`: copyable two-instance native ADS call example
- `{readme_name}`: this file

## Use in ADS

The include and the model instances have different jobs:

```text
top-level HB testbench
|- NetlistInclude -> loads {netlist_name} once
|- {module_name}:X1 -> geometry/process set A
|- {module_name}:X2 -> geometry/process set B
`- HB controller and the rest of the circuit
```

1. Copy this complete export directory under the ADS workspace, for example
   `./hb_models/{module_name}/`.
2. On the **top-level simulation schematic** that contains the HB controller,
   place one `NetlistInclude` from the **Data Items** palette. Do not place one
   in every model instance or inside an ordinary hierarchical wrapper because
   this file contains a `define` subnetwork declaration.
3. Configure the include approximately as follows; use **Browse** if the
   installed ADS release formats the fields differently:

   ```text
   IncludePath="./hb_models/{module_name}"
   IncludeFiles[1]="{netlist_name}"
   UsePreprocessor=yes
   ```

   `IncludePath` is the directory and `IncludeFiles` contains the filename.
   Keep the path relative for local simulation. A remote simulator must see the
   same path or an equivalent archived/absolute path.
4. Instantiate the subnetwork `{module_name}` as many times as needed. The
   `NetlistInclude` receives **no geometry values**. Geometry/process values are
   overrides on each subnetwork call, after the electrical nodes.
5. Use the component in HB exactly as a passive S-parameter network. No input
   power parameter is present or required.

`NetlistInclude` makes the subnetwork definition available to the simulator; it
is not the electrical component instance. This export is a native ADS netlist
package and does not contain an OpenAccess schematic symbol. For graphical
schematic placement, create one {nports}-pin custom/adapter component in ADS
whose generated ADS netlist line has the call format below. Expose the listed
instance parameters on that component. The symbol can then be placed repeatedly
while the single top-level include remains unchanged. Do not parse
`{netlist_name}` as SPICE; its contents are already native ADS syntax.

Two instances have the following native ADS call form. Replace the node names
and the `_A`/`_B` variables with values or top-level ADS `VAR` expressions:

```text
{instance_a_call}
{instance_b_call}
```

For a quick netlist-only hookup, label the corresponding top-level schematic
nets, copy the calls from `{instance_template_name}` into a separate `.net`
fragment, and add that fragment to the same top-level `NetlistInclude` **after**
`{netlist_name}`, for example as
`IncludeFiles[2]="my_model_instances.net"`. This creates electrically connected
but visually hidden instances. A custom adapter symbol is preferable for a
maintained schematic.

The electrical node order is `p1` through `p{nports}`, matching the order in
the first `define {module_name} (...)` line of `{netlist_name}`. Parameters that
are omitted from a call use the generated defaults below.

### Instance parameters

| Training VAR | Instance parameter | Default instance value | Input scale parameter | Default scale |
| --- | --- | ---: | --- | ---: |
{parameter_rows}

### Exactly what value to pass

The ADS-facing value and the trained-model value are related by:

```text
model_value = ADS_instance_parameter / input_scale_parameter
```

Pass the **physical ADS-side value**, including an ADS unit suffix when
appropriate. Do not pre-divide it yourself.

- If the MDIF trained on dimensionless micron counts such as `W=10`, export
  with `--parameter-input-scales 1um`, leave `W_input_scale=1um`, and pass
  `W=10um` on the instance. The embedded network receives `10`.
- If the MDIF value already parsed to SI units, such as `W=0.40mm` becoming
  `0.0004`, export with `--parameter-input-scales 1.0` and pass `W=0.40mm`.
  The embedded network receives `0.0004`.

Normally, override only the geometry/process parameters on each instance.
Leave every `*_input_scale` parameter at its generated default; it describes
the training-to-ADS unit convention and is not a geometry setting. For an
S-domain model, the `--z0` value is fixed into the generated S-to-Y conversion
at export time and is recorded in the manifest.

The exact sanitized parameter names, scales, and defaults are also recorded in
`{manifest_name}` under `parameter_identifiers`,
`parameter_scale_identifiers`, `parameter_input_scales`, and
`parameter_instance_defaults`.

## DC and RF behavior

{dc_rf_behavior} DC is never extrapolated from the RF fit.

## MLP scaler implementation

{scaler_behavior}

Negative frequencies use the complex conjugate of the surrogate at the
matching positive frequency, preserving the conjugate symmetry required for
real voltage and current waveforms. The selection occurs only in SDD frequency
weighting functions.

## Why this works in harmonic balance

The model implements the linear multiport relation directly in the frequency
domain as `I = Y(f) * V`. A direct-Y model supplies those weights immediately.
For an S-domain model, generated frequency-only equations first calculate
`Y = inverse(I + S) * (I - S) / Z0`; the circuit stamp is still explicit Y.
ADS evaluates those weights independently at the fundamental, harmonics, and
mixing frequencies requested by the HB controller.

No electrical port uses an implicit SDD equation. This avoids the extra branch
current and modified-nodal equation that ADS otherwise adds for every implicit
port. A sampled `export-ads-mdif` model can still be faster when its parameter
grid is practical because ADS does not need to evaluate the embedded neural
network.

This export targets ADS simulator expressions and the SDD netlist syntax. It
does not require Python, MDIF, or the original model files during simulation.

The implementation follows Keysight's ADS documentation for
[SDD frequency weighting](https://edadownload.software.keysight.com/eedl/ads/2011/pdf/modbuild.pdf),
[simulator expressions](https://edadownload.software.keysight.com/eedl/ads/2011_01/pdf/expsim.pdf),
and [subnetwork/netlist syntax](https://edadownload.software.keysight.com/eedl/ads/2011_01/pdf/cktsim.pdf).

Linear and power independent do not by themselves guarantee that an
unconstrained neural fit is passive at every interpolated or extrapolated
point. Validate the exported component over the complete HB frequency and
parameter range, and use passivity-aware sweep selection when required.
{notes}"""


def write_ads_hb_mlp_package(
    out_dir: Path,
    model_kind: str,
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    freq_transform: str,
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    z0: float,
    parameter_input_scales: dict[str, float] | None = None,
    output_domain: str = "s",
    uses_coarse_inputs: bool = False,
    adds_coarse_to_output: bool = False,
    embedded_coarse_model: dict[str, object] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
    source_model_dir: str | None = None,
    extra_manifest: dict[str, object] | None = None,
    extra_notes: Sequence[str] | None = None,
    combined_sdd: bool = False,
    fold_mlp_scalers: bool = False,
    emit_combined_sdd_trial: bool = False,
    emit_folded_scalers_trial: bool = False,
    artifact_variant: str | None = None,
) -> dict[str, object]:
    """Write a self-contained, linear ADS SDD model for HB/AC/SP analyses."""

    response_domain = output_domain.strip().lower()
    if response_domain not in {"s", "y"}:
        raise ValueError("ADS HB output domain must be 's' or 'y'")
    if response_domain == "y" and (uses_coarse_inputs or adds_coarse_to_output):
        raise ValueError("Direct-Y ADS HB export does not support KBNN coarse hooks")
    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError("ADS HB reference impedance must be positive and finite")
    if (combined_sdd or fold_mlp_scalers) and (
        emit_combined_sdd_trial or emit_folded_scalers_trial
    ):
        raise ValueError(
            "A trial package cannot recursively emit additional ADS HB trials"
        )
    nports = infer_complete_sparameter_ports(sparam_labels)
    n_sparams = len(sparam_labels)
    matrix_size = nports * nports
    if n_sparams != matrix_size:
        raise ValueError("ADS HB export requires a complete S-parameter matrix")
    dc_source_kind = validate_exact_dc_source_kind(
        dc_resistance_source_kind,
        context=f"{model_kind} ADS HB export",
    )
    dc_resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context=f"{model_kind} model",
    )
    full_dc_model = dc_model is not None and str(
        dc_model.get(
            "representation",
            "full_s_matrix"
            if dc_model.get("kind") == "geometry_dependent_exact_dc_full_s_mlp"
            else "full_y_matrix"
            if dc_model.get("kind") == "geometry_dependent_exact_dc_full_y_mlp"
            else "path_conductance",
        )
    ) in {"full_s_matrix", "full_y_matrix"}
    dc_path_resistances = (
        None
        if full_dc_model
        else validate_dc_port_resistances(
            sparam_labels,
            dc_port_resistances_ohm,
            context=f"{model_kind} model",
        )
    )
    module_id = _ads_hb_identifier(module_name, "surrogate_hb")
    prefix = _ads_hb_identifier(f"{module_id}_m", "hb_model")
    param_ids, scale_ids, param_scales, param_defaults, parameter_features = (
        _ads_hb_parameter_configuration(
            parameter_names,
            np.asarray(x_mean, dtype=float),
            parameter_input_scales,
        )
    )
    port_ids = [f"p{idx}" for idx in range(1, nports + 1)]
    lines = [
        "; Self-contained linear ADS SDD surrogate for harmonic balance",
        "; Generated by ads-ann-surrogate",
        f"define {module_id} ({' '.join(port_ids)})",
    ]
    parameter_tokens = [f"{identifier}={veriloga_float(default)}" for identifier, default in zip(param_ids, param_defaults)]
    parameter_tokens.extend(
        f"{scale_id}={veriloga_float(scale)}"
        for scale_id, scale in zip(scale_ids, param_scales)
    )
    if parameter_tokens:
        lines.append("parameters " + " ".join(parameter_tokens))
    lines.append("")

    coarse_outputs: list[str] | None = None
    if embedded_coarse_model is not None:
        coarse_parameter_names = [str(value) for value in embedded_coarse_model["parameter_names"]]  # type: ignore[index]
        coarse_labels = [str(value) for value in embedded_coarse_model["sparam_labels"]]  # type: ignore[index]
        if coarse_parameter_names != list(parameter_names) or coarse_labels != list(sparam_labels):
            raise ValueError("Embedded coarse DNN parameters and S-parameter order must match the KBNN")
        if str(embedded_coarse_model.get("output_domain", "s")).lower() != "s":
            raise ValueError("Embedded coarse DNN must use the S output domain")
        coarse_features = [
            *parameter_features,
            *_ads_hb_frequency_features(str(embedded_coarse_model["freq_transform"])),
        ]
        coarse_outputs = _append_ads_hb_mlp_equations(
            lines,
            f"{prefix}_coarse",
            coarse_features,
            str(embedded_coarse_model["activation"]),
            embedded_coarse_model["layer_sizes"],  # type: ignore[arg-type]
            embedded_coarse_model["weights"],  # type: ignore[arg-type]
            embedded_coarse_model["biases"],  # type: ignore[arg-type]
            np.asarray(embedded_coarse_model["x_mean"], dtype=float),
            np.asarray(embedded_coarse_model["x_std"], dtype=float),
            np.asarray(embedded_coarse_model["y_mean"], dtype=float),
            np.asarray(embedded_coarse_model["y_std"], dtype=float),
            fold_scalers=fold_mlp_scalers,
        )
        lines.append("")
    elif uses_coarse_inputs or adds_coarse_to_output:
        raise ValueError(
            "A self-contained residual/prior-input ADS HB export requires the frozen coarse DNN"
        )

    primary_features = [*parameter_features, *_ads_hb_frequency_features(freq_transform)]
    if uses_coarse_inputs:
        assert coarse_outputs is not None
        primary_features.extend(coarse_outputs)
    primary_outputs = _append_ads_hb_mlp_equations(
        lines,
        f"{prefix}_fine",
        primary_features,
        activation,
        layer_sizes,
        weights,
        biases,
        np.asarray(x_mean, dtype=float),
        np.asarray(x_std, dtype=float),
        np.asarray(y_mean, dtype=float),
        np.asarray(y_std, dtype=float),
        fold_scalers=fold_mlp_scalers,
    )
    lines.append("")
    if len(primary_outputs) != 2 * n_sparams:
        raise ValueError("ADS HB neural output count does not match the response matrix")

    fitted_response_names: list[str] = []
    for idx, label in enumerate(sparam_labels):
        real_name = primary_outputs[idx]
        imag_name = primary_outputs[idx + n_sparams]
        if adds_coarse_to_output:
            assert coarse_outputs is not None
            real_expression = f"({coarse_outputs[idx]})+({real_name})"
            imag_expression = f"({coarse_outputs[idx + n_sparams]})+({imag_name})"
        else:
            real_expression = real_name
            imag_expression = imag_name
        fitted_name = f"{prefix}_rf_{label.lower()}"
        lines.append(f"{fitted_name}=complex(({real_expression}),({imag_expression}))")
        fitted_response_names.append(fitted_name)

    dc_matrix = None
    dc_matrix_names = None
    if dc_model is not None:
        lines.append("")
        lines.append("; Geometry-dependent exact-DC conductance surrogate")
        dc_matrix_names = _append_ads_hb_dc_conductance_model(
            lines,
            f"{prefix}_dc",
            parameter_names,
            sparam_labels,
            parameter_features,
            dc_model,
            fold_scalers=fold_mlp_scalers,
        )
    else:
        dc_matrix = dc_conductance_matrix(
            sparam_labels,
            dc_resistance,
            dc_path_resistances,
        )
    rf_response_names: list[str] = [""] * matrix_size
    for idx, label in enumerate(sparam_labels):
        row, col = sparam_indices(label) or (0, 0)
        flat = (row - 1) * nports + (col - 1)
        active_name = f"{prefix}_{label.lower()}"
        lines.append(
            f"{active_name}=if (freq < 0) then conj({fitted_response_names[idx]}) "
            f"else {fitted_response_names[idx]} endif"
        )
        rf_response_names[flat] = active_name
    if response_domain == "s":
        lines.append("")
        rf_y_names = _append_ads_hb_s_to_y_equations(
            lines,
            f"{prefix}_stoy",
            rf_response_names,
            nports,
            z0,
        )
    else:
        rf_y_names = rf_response_names
    lines.append("")
    _append_ads_hb_sdd(
        lines,
        f"{module_id}_core",
        port_ids,
        rf_y_names,
        "y",
        dc_conductance=dc_matrix,
        dc_conductance_names=dc_matrix_names,
        combine_dc_rf=combined_sdd,
    )
    lines.extend([f"end {module_id}", ""])

    out_dir.mkdir(parents=True, exist_ok=True)
    netlist_name = f"{module_id}.net"
    artifact_token = (
        _ads_hb_identifier(artifact_variant, "trial").lower()
        if artifact_variant
        else None
    )
    manifest_name = (
        f"ads_hb_{artifact_token}_manifest.json"
        if artifact_token
        else "ads_hb_manifest.json"
    )
    readme_name = (
        f"ADS_HB_{artifact_token.upper()}_README.md"
        if artifact_token
        else "ADS_HB_README.md"
    )
    (out_dir / netlist_name).write_text("\n".join(lines))
    instance_template_name = (
        f"ADS_HB_{artifact_token.upper()}_INSTANCE_TEMPLATE.txt"
        if artifact_token
        else "ADS_HB_INSTANCE_TEMPLATE.txt"
    )
    (out_dir / instance_template_name).write_text(
        _ads_hb_instance_template(module_id, netlist_name, nports, param_ids)
    )
    manifest: dict[str, object] = {
        "format": "ads_hb_sdd_linear_multiport",
        "implementation_status": "trial" if artifact_token else "default",
        "artifact_variant": artifact_token,
        "model_kind": model_kind,
        "module_name": module_id,
        "netlist_file": netlist_name,
        "manifest_file": manifest_name,
        "readme_file": readme_name,
        "nports": nports,
        "parameter_names": list(parameter_names),
        "parameter_identifiers": param_ids,
        "parameter_scale_identifiers": scale_ids,
        "parameter_input_scales": {
            name: scale for name, scale in zip(parameter_names, param_scales)
        },
        "parameter_instance_defaults": param_defaults,
        "instance_template_file": instance_template_name,
        "instance_call_example": _ads_hb_instance_call(
            module_id, "X1", nports, param_ids, "A"
        ),
        "parameter_value_relation": (
            "model_value = ADS_instance_parameter / input_scale_parameter"
        ),
        "netlist_include_contract": {
            "placement": "top-level simulation schematic",
            "include_file": netlist_name,
            "receives_instance_parameters": False,
        },
        "sparam_labels": list(sparam_labels),
        "response_domain": response_domain,
        "z0": z0,
        "folded_input_scaler": bool(fold_mlp_scalers),
        "folded_output_scaler": bool(fold_mlp_scalers),
        "mlp_scaler_implementation": (
            "folded_into_first_and_final_layers"
            if fold_mlp_scalers
            else "explicit_input_and_output_equations"
        ),
        "linear": True,
        "power_dependent": False,
        "passivity_enforced_by_export": False,
        "hb_frequency_weighted": True,
        "negative_frequency_response": "conjugate_of_positive_frequency_surrogate",
        "fully_self_contained": True,
        "uses_coarse_inputs": bool(uses_coarse_inputs),
        "adds_coarse_to_output": bool(adds_coarse_to_output),
        "embedded_coarse_model": embedded_coarse_model is not None,
        "source_model_dir": source_model_dir,
        "dc_equivalent_resistance_ohm": dc_resistance,
        "dc_port_paths": (
            list(dc_model.get("port_paths", []))
            if dc_model is not None
            else list(dc_path_resistances)
            if dc_path_resistances is not None
            else None
        ),
        "dc_matrix_entries": (
            list(dc_model.get("matrix_entries", [])) if dc_model is not None else []
        ),
        "dc_sparameter_entries": (
            list(dc_model.get("sparameter_entries", [])) if dc_model is not None else []
        ),
        "dc_port_resistances_ohm": (
            None
            if dc_model is not None
            and dc_model.get("representation") in {"full_s_matrix", "full_y_matrix"}
            else dc_path_resistances
        ),
        "dc_resistance_source_kind": dc_source_kind,
        "dc_model_kind": (
            str(dc_model.get("kind")) if dc_model is not None else None
        ),
        "dc_geometry_dependent": dc_model is not None,
        "dc_model_metadata": (
            dc_model.get("metadata") if dc_model is not None else None
        ),
        "dc_is_separate_from_fitted_response": True,
        "dc_stamping_representation": (
            "combined_full_ordered_complex_s_to_y_sdd"
            if combined_sdd
            and dc_model is not None
            and dc_model.get("representation") == "full_s_matrix"
            else "combined_full_ordered_y_sdd"
            if combined_sdd
            and dc_model is not None
            and dc_model.get("representation") == "full_y_matrix"
            else "combined_explicit_conductance_sdd"
            if combined_sdd
            else "separate_full_ordered_complex_s_to_y_sdd"
            if dc_model is not None
            and dc_model.get("representation") == "full_s_matrix"
            else "separate_full_ordered_y_sdd"
            if dc_model is not None
            and dc_model.get("representation") == "full_y_matrix"
            else "separate_explicit_conductance_sdd"
        ),
        "rf_stamping_representation": (
            "combined_explicit_current_y_sdd"
            if combined_sdd
            else "explicit_current_y_sdd"
        ),
        "sdd_dc_rf_topology": (
            "single_frequency_selected_sdd"
            if combined_sdd
            else "separate_parallel_sdds"
        ),
        "rf_source_conversion": (
            "runtime_frequency_only_s_to_y" if response_domain == "s" else "none"
        ),
        "implicit_port_equations": False,
        "supported_analyses": ["DC", "AC", "SP", "HB"],
        "implementation_note": (
            "One explicit-current SDD selects the extracted DC admittance at zero "
            "frequency and the fitted RF admittance otherwise. This trial is "
            "algebraically equivalent to the separate branches while exposing one "
            "dense SDD control/stamp graph."
            if combined_sdd
            else "The default separate DC/RF SDD topology is retained while every "
            "embedded MLP's input standardizer and output inverse-standardizer are "
            "folded into its first and final affine layers. This trial is "
            "algebraically equivalent to the explicit scaler equations."
            if fold_mlp_scalers
            else "Separate explicit-current SDD branches stamp the extracted DC "
            "conductance and fitted RF admittance. S-output fits are converted to Y "
            "in frequency-only equations before stamping; no implicit port-current "
            "unknowns are introduced."
        ),
        "reference_note": (
            "Generated against Keysight's documented ADS SDD frequency-weighting and "
            "simulator-expression syntax; validate with the installed ADS release."
        ),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    resolved_notes = list(extra_notes or [])
    trial_exports: list[dict[str, object]] = []
    if emit_combined_sdd_trial:
        trial_module_name = f"{module_id}_combined_sdd_trial"
        trial_extra_manifest = dict(extra_manifest or {})
        trial_extra_manifest.update(
            {
                "trial_parent_module_name": module_id,
                "trial_purpose": (
                    "Compare one frequency-selected DC/RF SDD against the default "
                    "two mutually exclusive parallel SDDs"
                ),
            }
        )
        trial_manifest = write_ads_hb_mlp_package(
            out_dir=out_dir,
            model_kind=model_kind,
            module_name=trial_module_name,
            parameter_names=parameter_names,
            sparam_labels=sparam_labels,
            freq_transform=freq_transform,
            activation=activation,
            layer_sizes=layer_sizes,
            weights=weights,
            biases=biases,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            z0=z0,
            parameter_input_scales=parameter_input_scales,
            output_domain=output_domain,
            uses_coarse_inputs=uses_coarse_inputs,
            adds_coarse_to_output=adds_coarse_to_output,
            embedded_coarse_model=embedded_coarse_model,
            dc_equivalent_resistance_ohm=dc_equivalent_resistance_ohm,
            dc_resistance_source_kind=dc_resistance_source_kind,
            dc_port_resistances_ohm=dc_port_resistances_ohm,
            dc_model=dc_model,
            source_model_dir=source_model_dir,
            extra_manifest=trial_extra_manifest,
            extra_notes=[
                *resolved_notes,
                "Trial implementation: one SDD selects exact DC or fitted RF in each frequency weight.",
                f"The unchanged default implementation remains `{netlist_name}` with module `{module_id}`.",
            ],
            combined_sdd=True,
            fold_mlp_scalers=False,
            emit_combined_sdd_trial=False,
            emit_folded_scalers_trial=False,
            artifact_variant="combined_sdd_trial",
        )
        trial_export = {
            "kind": "combined_dc_rf_sdd",
            "status": "trial",
            "module_name": trial_manifest["module_name"],
            "netlist_file": trial_manifest["netlist_file"],
            "manifest_file": trial_manifest["manifest_file"],
            "readme_file": trial_manifest["readme_file"],
            "instance_template_file": trial_manifest["instance_template_file"],
            "sdd_dc_rf_topology": trial_manifest["sdd_dc_rf_topology"],
        }
        trial_exports.append(trial_export)
        resolved_notes.extend(
            [
                "The default two-SDD implementation in this package is unchanged.",
                f"A comparison trial is also exported as `{trial_export['netlist_file']}` with module `{trial_export['module_name']}`.",
                f"Use `{trial_export['instance_template_file']}` and `{trial_export['readme_file']}` for the trial; benchmark the default and trial as separate instances.",
            ]
        )
    if emit_folded_scalers_trial:
        trial_module_name = f"{module_id}_folded_scalers_trial"
        trial_extra_manifest = dict(extra_manifest or {})
        trial_extra_manifest.update(
            {
                "trial_parent_module_name": module_id,
                "trial_purpose": (
                    "Compare algebraically folded MLP input/output scalers against "
                    "the default explicit scaler equations"
                ),
            }
        )
        trial_manifest = write_ads_hb_mlp_package(
            out_dir=out_dir,
            model_kind=model_kind,
            module_name=trial_module_name,
            parameter_names=parameter_names,
            sparam_labels=sparam_labels,
            freq_transform=freq_transform,
            activation=activation,
            layer_sizes=layer_sizes,
            weights=weights,
            biases=biases,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            z0=z0,
            parameter_input_scales=parameter_input_scales,
            output_domain=output_domain,
            uses_coarse_inputs=uses_coarse_inputs,
            adds_coarse_to_output=adds_coarse_to_output,
            embedded_coarse_model=embedded_coarse_model,
            dc_equivalent_resistance_ohm=dc_equivalent_resistance_ohm,
            dc_resistance_source_kind=dc_resistance_source_kind,
            dc_port_resistances_ohm=dc_port_resistances_ohm,
            dc_model=dc_model,
            source_model_dir=source_model_dir,
            extra_manifest=trial_extra_manifest,
            extra_notes=[
                *resolved_notes,
                "Trial implementation: all embedded MLP scalers are folded into their first and final affine layers.",
                f"The unchanged default implementation remains `{netlist_name}` with module `{module_id}`.",
            ],
            combined_sdd=False,
            fold_mlp_scalers=True,
            emit_combined_sdd_trial=False,
            emit_folded_scalers_trial=False,
            artifact_variant="folded_scalers_trial",
        )
        trial_export = {
            "kind": "folded_mlp_scalers",
            "status": "trial",
            "module_name": trial_manifest["module_name"],
            "netlist_file": trial_manifest["netlist_file"],
            "manifest_file": trial_manifest["manifest_file"],
            "readme_file": trial_manifest["readme_file"],
            "instance_template_file": trial_manifest["instance_template_file"],
            "sdd_dc_rf_topology": trial_manifest["sdd_dc_rf_topology"],
            "mlp_scaler_implementation": trial_manifest[
                "mlp_scaler_implementation"
            ],
        }
        trial_exports.append(trial_export)
        resolved_notes.extend(
            [
                "The default two-SDD topology and explicit scaler equations remain unchanged.",
                f"The active scaler-folding comparison trial is `{trial_export['netlist_file']}` with module `{trial_export['module_name']}`.",
                f"Use `{trial_export['instance_template_file']}` and `{trial_export['readme_file']}` for the trial; benchmark it independently from other trial implementations.",
            ]
        )
    if trial_exports:
        manifest["trial_exports"] = trial_exports
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    (out_dir / readme_name).write_text(
        _ads_hb_readme(
            model_kind=model_kind,
            module_name=module_id,
            netlist_name=netlist_name,
            manifest_name=manifest_name,
            nports=nports,
            parameter_names=parameter_names,
            parameter_ids=param_ids,
            parameter_scale_ids=scale_ids,
            parameter_scales=param_scales,
            parameter_instance_defaults=param_defaults,
            response_domain=response_domain,
            instance_template_name=instance_template_name,
            readme_name=readme_name,
            combined_sdd=combined_sdd,
            fold_mlp_scalers=fold_mlp_scalers,
            extra_notes=resolved_notes,
        )
    )
    return manifest


def write_ads_hb_neurotf_package(
    out_dir: Path,
    module_name: str,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
    activation: str,
    layer_sizes: Sequence[int],
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    poles: np.ndarray,
    f_scale: float,
    z0: float,
    parameter_input_scales: dict[str, float] | None = None,
    dc_equivalent_resistance_ohm: float | None = None,
    dc_resistance_source_kind: object = None,
    dc_port_resistances_ohm: object = None,
    dc_model: dict[str, object] | None = None,
    source_model_dir: str | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write a fixed-pole Neuro-TF as a linear ADS SDD HB subnetwork."""

    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError("ADS HB reference impedance must be positive and finite")
    if not math.isfinite(f_scale) or f_scale <= 0.0:
        raise ValueError("Neuro-TF ADS HB frequency scale must be positive and finite")
    nports = infer_complete_sparameter_ports(sparam_labels)
    n_sparams = len(sparam_labels)
    n_coeffs = len(poles) + 1
    expected_outputs = 2 * n_sparams * n_coeffs
    if int(layer_sizes[-1]) != expected_outputs:
        raise ValueError(
            f"Neuro-TF ADS HB export expected {expected_outputs} coefficient outputs, "
            f"got {layer_sizes[-1]}"
        )
    dc_source_kind = validate_exact_dc_source_kind(
        dc_resistance_source_kind,
        context="Neuro-TF ADS HB export",
    )
    dc_resistance = validate_dc_equivalent_resistance(
        dc_equivalent_resistance_ohm,
        context="Neuro-TF model",
    )
    full_dc_model = dc_model is not None and str(
        dc_model.get(
            "representation",
            "full_s_matrix"
            if dc_model.get("kind") == "geometry_dependent_exact_dc_full_s_mlp"
            else "full_y_matrix"
            if dc_model.get("kind") == "geometry_dependent_exact_dc_full_y_mlp"
            else "path_conductance",
        )
    ) in {"full_s_matrix", "full_y_matrix"}
    dc_path_resistances = (
        None
        if full_dc_model
        else validate_dc_port_resistances(
            sparam_labels,
            dc_port_resistances_ohm,
            context="Neuro-TF model",
        )
    )
    module_id = _ads_hb_identifier(module_name, "neuro_tf_hb")
    prefix = _ads_hb_identifier(f"{module_id}_m", "hb_model")
    param_ids, scale_ids, param_scales, param_defaults, parameter_features = (
        _ads_hb_parameter_configuration(
            parameter_names,
            np.asarray(x_mean, dtype=float),
            parameter_input_scales,
        )
    )
    port_ids = [f"p{idx}" for idx in range(1, nports + 1)]
    lines = [
        "; Self-contained linear Neuro-TF ADS SDD surrogate for harmonic balance",
        "; Generated by ads-ann-surrogate",
        f"define {module_id} ({' '.join(port_ids)})",
    ]
    parameter_tokens = [f"{identifier}={veriloga_float(default)}" for identifier, default in zip(param_ids, param_defaults)]
    parameter_tokens.extend(
        f"{scale_id}={veriloga_float(scale)}"
        for scale_id, scale in zip(scale_ids, param_scales)
    )
    lines.append("parameters " + " ".join(parameter_tokens))
    lines.append("")
    coefficient_outputs = _append_ads_hb_mlp_equations(
        lines,
        f"{prefix}_coeff",
        parameter_features,
        activation,
        layer_sizes,
        weights,
        biases,
        np.asarray(x_mean, dtype=float),
        np.asarray(x_std, dtype=float),
        np.asarray(y_mean, dtype=float),
        np.asarray(y_std, dtype=float),
    )
    lines.append("")
    coefficient_half = n_sparams * n_coeffs
    fitted_names: list[str] = []
    for sparam_idx, label in enumerate(sparam_labels):
        terms: list[str] = []
        for coeff_idx in range(n_coeffs):
            real_idx = sparam_idx * n_coeffs + coeff_idx
            imag_idx = coefficient_half + real_idx
            coefficient = (
                f"complex({coefficient_outputs[real_idx]},{coefficient_outputs[imag_idx]})"
            )
            if coeff_idx == 0:
                terms.append(coefficient)
            else:
                pole = complex(np.asarray(poles, dtype=complex)[coeff_idx - 1])
                denominator = (
                    f"j*(abs(freq)/({veriloga_float(f_scale)}))-"
                    f"complex({veriloga_float(pole.real)},{veriloga_float(pole.imag)})"
                )
                terms.append(f"({coefficient})/({denominator})")
        fitted_name = f"{prefix}_rf_{label.lower()}"
        lines.append(f"{fitted_name}=" + "+".join(terms))
        fitted_names.append(fitted_name)

    dc_matrix = None
    dc_matrix_names = None
    if dc_model is not None:
        lines.append("")
        lines.append("; Geometry-dependent exact-DC conductance surrogate")
        dc_matrix_names = _append_ads_hb_dc_conductance_model(
            lines,
            f"{prefix}_dc",
            parameter_names,
            sparam_labels,
            parameter_features,
            dc_model,
        )
    else:
        dc_matrix = dc_conductance_matrix(
            sparam_labels,
            dc_resistance,
            dc_path_resistances,
        )
    active_names: list[str] = [""] * (nports * nports)
    for idx, label in enumerate(sparam_labels):
        row, col = sparam_indices(label) or (0, 0)
        flat = (row - 1) * nports + (col - 1)
        active_name = f"{prefix}_{label.lower()}"
        lines.append(
            f"{active_name}=if (freq < 0) then conj({fitted_names[idx]}) "
            f"else {fitted_names[idx]} endif"
        )
        active_names[flat] = active_name
    lines.append("")
    rf_y_names = _append_ads_hb_s_to_y_equations(
        lines,
        f"{prefix}_stoy",
        active_names,
        nports,
        z0,
    )
    lines.append("")
    _append_ads_hb_sdd(
        lines,
        f"{module_id}_core",
        port_ids,
        rf_y_names,
        "y",
        dc_conductance=dc_matrix,
        dc_conductance_names=dc_matrix_names,
    )
    lines.extend([f"end {module_id}", ""])

    out_dir.mkdir(parents=True, exist_ok=True)
    netlist_name = f"{module_id}.net"
    manifest_name = "ads_hb_manifest.json"
    (out_dir / netlist_name).write_text("\n".join(lines))
    instance_template_name = "ADS_HB_INSTANCE_TEMPLATE.txt"
    (out_dir / instance_template_name).write_text(
        _ads_hb_instance_template(module_id, netlist_name, nports, param_ids)
    )
    manifest: dict[str, object] = {
        "format": "ads_hb_sdd_linear_multiport",
        "model_kind": "Neuro-TF",
        "module_name": module_id,
        "netlist_file": netlist_name,
        "nports": nports,
        "parameter_names": list(parameter_names),
        "parameter_identifiers": param_ids,
        "parameter_scale_identifiers": scale_ids,
        "parameter_input_scales": {
            name: scale for name, scale in zip(parameter_names, param_scales)
        },
        "parameter_instance_defaults": param_defaults,
        "instance_template_file": instance_template_name,
        "instance_call_example": _ads_hb_instance_call(
            module_id, "X1", nports, param_ids, "A"
        ),
        "parameter_value_relation": (
            "model_value = ADS_instance_parameter / input_scale_parameter"
        ),
        "netlist_include_contract": {
            "placement": "top-level simulation schematic",
            "include_file": netlist_name,
            "receives_instance_parameters": False,
        },
        "sparam_labels": list(sparam_labels),
        "response_domain": "s",
        "z0": z0,
        "n_poles": len(poles),
        "f_scale": f_scale,
        "linear": True,
        "power_dependent": False,
        "passivity_enforced_by_export": False,
        "hb_frequency_weighted": True,
        "negative_frequency_response": "conjugate_of_positive_frequency_surrogate",
        "fully_self_contained": True,
        "source_model_dir": source_model_dir,
        "dc_equivalent_resistance_ohm": dc_resistance,
        "dc_port_paths": (
            list(dc_model.get("port_paths", []))
            if dc_model is not None
            else list(dc_path_resistances)
            if dc_path_resistances is not None
            else None
        ),
        "dc_matrix_entries": (
            list(dc_model.get("matrix_entries", [])) if dc_model is not None else []
        ),
        "dc_sparameter_entries": (
            list(dc_model.get("sparameter_entries", [])) if dc_model is not None else []
        ),
        "dc_port_resistances_ohm": (
            None
            if dc_model is not None
            and dc_model.get("representation") in {"full_s_matrix", "full_y_matrix"}
            else dc_path_resistances
        ),
        "dc_resistance_source_kind": dc_source_kind,
        "dc_model_kind": (
            str(dc_model.get("kind")) if dc_model is not None else None
        ),
        "dc_geometry_dependent": dc_model is not None,
        "dc_model_metadata": (
            dc_model.get("metadata") if dc_model is not None else None
        ),
        "dc_is_separate_from_fitted_response": True,
        "dc_stamping_representation": (
            "separate_full_ordered_complex_s_to_y_sdd"
            if dc_model is not None
            and dc_model.get("representation") == "full_s_matrix"
            else "separate_full_ordered_y_sdd"
            if dc_model is not None
            and dc_model.get("representation") == "full_y_matrix"
            else "separate_explicit_conductance_sdd"
        ),
        "rf_stamping_representation": "explicit_current_y_sdd",
        "rf_source_conversion": "runtime_frequency_only_s_to_y",
        "implicit_port_equations": False,
        "supported_analyses": ["DC", "AC", "SP", "HB"],
        "implementation_note": (
            "The fixed-pole rational S-matrix is converted to admittance in frequency-only "
            "equations. Separate explicit-current SDD branches stamp RF Y and exact DC "
            "conductance without implicit port-current unknowns."
        ),
        "reference_note": (
            "Generated against Keysight's documented ADS SDD frequency-weighting and "
            "simulator-expression syntax; validate with the installed ADS release."
        ),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2))
    (out_dir / "ADS_HB_README.md").write_text(
        _ads_hb_readme(
            model_kind="Neuro-TF",
            module_name=module_id,
            netlist_name=netlist_name,
            manifest_name=manifest_name,
            nports=nports,
            parameter_names=parameter_names,
            parameter_ids=param_ids,
            parameter_scale_ids=scale_ids,
            parameter_scales=param_scales,
            parameter_instance_defaults=param_defaults,
            response_domain="s",
            extra_notes=[
                "The fixed-pole rational response is evaluated directly as a complex ADS expression.",
            ],
        )
    )
    return manifest


def split_blocks(
    blocks: list[MDIFBlock],
    split_var: str,
    train_values: set[str],
    verify_values: set[str],
    holdout_fraction: float,
    seed: int,
) -> SplitData:
    split_key = normalize_name(split_var)
    train: list[MDIFBlock] = []
    verify: list[MDIFBlock] = []

    for block in blocks:
        value = normalized_mapping_value(block.params, split_key)
        lowered = value.lower() if value is not None else ""
        if lowered in train_values:
            train.append(block)
        elif lowered in verify_values:
            verify.append(block)

    if train:
        return SplitData(train=train, verify=verify, all_blocks=blocks)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(blocks))
    rng.shuffle(indices)
    n_verify = int(round(len(blocks) * holdout_fraction))
    n_verify = min(max(n_verify, 1 if len(blocks) > 1 else 0), max(len(blocks) - 1, 0))
    verify_indices = set(indices[:n_verify].tolist())
    train = [block for idx, block in enumerate(blocks) if idx not in verify_indices]
    verify = [block for idx, block in enumerate(blocks) if idx in verify_indices]
    return SplitData(train=train, verify=verify, all_blocks=blocks)


def parse_csv_set(text: str) -> set[str]:
    return {strip_quotes(part).strip().lower() for part in text.split(",") if part.strip()}


def infer_parameter_names(
    blocks: Sequence[MDIFBlock],
    requested: str | None,
    split_var: str,
) -> list[str]:
    if not blocks:
        raise ValueError("No MDIF blocks were available for parameter inference")

    excluded = {normalize_name(split_var).lower(), "dataset", "set", "split"}
    common = set(blocks[0].params)
    for block in blocks[1:]:
        common &= set(block.params)

    numeric_common = []
    for name in sorted(common):
        if name.lower() in excluded:
            continue
        if all(parse_number(block.params[name]) is not None for block in blocks):
            numeric_common.append(name)

    if requested:
        raw_names = [part.strip() for part in requested.split(",") if part.strip()]
        if not raw_names:
            raise ValueError("--parameter-names did not contain any parameter names")

        lookup: dict[str, str] = {}
        for name in sorted(common):
            for key in {name, name.lower(), normalize_name(name), normalize_name(name).lower()}:
                lookup.setdefault(key, name)

        resolved = []
        for raw_name in raw_names:
            normalized = normalize_name(raw_name)
            target = (
                lookup.get(normalized)
                or lookup.get(normalized.lower())
                or lookup.get(raw_name)
                or lookup.get(raw_name.lower())
            )
            if target is None:
                candidates = numeric_common or sorted(common)
                suggestion = difflib.get_close_matches(normalized, candidates, n=1)
                suggestion_text = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
                available = ", ".join(numeric_common) if numeric_common else "none"
                raise ValueError(
                    f"Requested parameter {raw_name!r} was not found as a common MDIF VAR. "
                    f"Common numeric VARs: {available}.{suggestion_text}"
                )
            bad_blocks = [
                block.source_index
                for block in blocks
                if parse_number(block.params[target]) is None
            ]
            if bad_blocks:
                sample_value = next(
                    block.params[target]
                    for block in blocks
                    if parse_number(block.params[target]) is None
                )
                available = ", ".join(numeric_common) if numeric_common else "none"
                raise ValueError(
                    f"Requested parameter {raw_name!r} matched MDIF VAR {target!r}, "
                    f"but it is not numeric in block(s) {bad_blocks[:8]} "
                    f"(example value {sample_value!r}). Common numeric VARs: {available}."
                )
            resolved.append(target)
        return resolved

    if not numeric_common:
        raise ValueError(
            "Could not infer numeric geometry parameters. Pass --parameter-names w,l,..."
        )
    return numeric_common


def parameter_matrix(blocks: Sequence[MDIFBlock], parameter_names: Sequence[str]) -> np.ndarray:
    matrix = []
    for block in blocks:
        row = []
        for name in parameter_names:
            if name not in block.params:
                raise ValueError(f"Block {block.source_index} is missing parameter {name!r}")
            value = parse_number(block.params[name])
            if value is None:
                raise ValueError(
                    f"Block {block.source_index} parameter {name!r} is not numeric: {block.params[name]!r}"
                )
            row.append(value)
        matrix.append(row)
    return np.asarray(matrix, dtype=float)


def common_sparameter_labels(blocks: Sequence[MDIFBlock]) -> list[str]:
    common = set(blocks[0].sparams)
    for block in blocks[1:]:
        common &= set(block.sparams)
    if not common:
        raise ValueError("No common S-parameter labels found across MDIF blocks")
    return sorted(common, key=sparam_sort_key)


def sparam_sort_key(label: str) -> tuple[int, int, str]:
    match = re.match(r"S(\d+)(\d+)$", label)
    if match:
        return int(match.group(1)), int(match.group(2)), label
    return 999, 999, label


def sparam_indices(label: str) -> tuple[int, int] | None:
    match = re.match(r"S(\d+)(\d+)$", label.upper())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def split_sparam_weight_rules(spec: str) -> list[str]:
    text = spec.strip()
    if not text:
        return []
    if ";" in text:
        return [part.strip() for part in text.split(";") if part.strip()]
    comma_parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(comma_parts) > 1 and all("=" in part for part in comma_parts):
        return comma_parts
    return [text]


def canonical_sparam_label(token: str) -> str:
    raw = token.strip().upper()
    match = re.fullmatch(r"S\[(\d+),(\d+)\]", raw)
    if match:
        return f"S{int(match.group(1))}{int(match.group(2))}"
    match = re.fullmatch(r"S(\d+)[,_-](\d+)", raw)
    if match:
        return f"S{int(match.group(1))}{int(match.group(2))}"
    return raw


def wildcard_to_regex(pattern: str) -> re.Pattern[str]:
    escaped = re.escape(pattern.upper())
    escaped = escaped.replace(r"\*", ".*").replace(r"\?", ".")
    return re.compile(f"^{escaped}$")


def expand_sparam_weight_token(token: str, labels: Sequence[str]) -> list[str]:
    raw = token.strip()
    if not raw:
        return []
    normalized = raw.lower().replace("_", "-")
    indexed = [(label, sparam_indices(label)) for label in labels]

    if normalized in {"*", "all", "default"}:
        return list(labels)
    if normalized in {"diag", "diagonal", "return", "reflection", "reflections"}:
        return [label for label, ij in indexed if ij is not None and ij[0] == ij[1]]
    if normalized in {"offdiag", "off-diag", "off-diagonal", "transmission", "transmissions"}:
        return [label for label, ij in indexed if ij is not None and ij[0] != ij[1]]
    if normalized == "upper":
        return [label for label, ij in indexed if ij is not None and ij[0] < ij[1]]
    if normalized == "lower":
        return [label for label, ij in indexed if ij is not None and ij[0] > ij[1]]

    match = re.fullmatch(r"(?:row|out|output)(\d+)", normalized)
    if match:
        row = int(match.group(1))
        return [label for label, ij in indexed if ij is not None and ij[0] == row]
    match = re.fullmatch(r"(?:col|column|in|input)(\d+)", normalized)
    if match:
        col = int(match.group(1))
        return [label for label, ij in indexed if ij is not None and ij[1] == col]

    canonical = canonical_sparam_label(raw)
    if "*" in canonical or "?" in canonical:
        regex = wildcard_to_regex(canonical)
        return [label for label in labels if regex.fullmatch(label.upper())]
    return [label for label in labels if label.upper() == canonical]


def parse_sparam_weights(labels: Sequence[str], spec: str | None) -> dict[str, float]:
    weights = {label: 1.0 for label in labels}
    if not spec:
        return weights

    for rule in split_sparam_weight_rules(spec):
        if "=" not in rule:
            raise ValueError(
                f"S-parameter weight rule {rule!r} must look like selector=weight"
            )
        left, right = rule.rsplit("=", 1)
        try:
            weight = float(right.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid S-parameter weight in {rule!r}") from exc
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"S-parameter weights must be finite and non-negative, got {weight}")

        selectors = [part.strip() for part in re.split(r"[,+\s]+", left.strip()) if part.strip()]
        if not selectors:
            raise ValueError(f"S-parameter weight rule {rule!r} has no selector")
        matched: list[str] = []
        for selector in selectors:
            matched.extend(expand_sparam_weight_token(selector, labels))
        matched = sorted(set(matched), key=sparam_sort_key)
        if not matched:
            raise ValueError(f"S-parameter weight selector {left!r} did not match any labels")
        for label in matched:
            weights[label] = weight

    if sum(weights.values()) <= EPS:
        raise ValueError("At least one S-parameter weight must be greater than zero")
    return weights


def sparam_weight_mean(labels: Sequence[str], weights: dict[str, float]) -> float:
    values = np.asarray([float(weights[label]) for label in labels], dtype=float)
    mean = float(np.mean(values))
    if mean <= EPS:
        raise ValueError("At least one S-parameter weight must be greater than zero")
    return mean


def normalize_sparam_weights(
    labels: Sequence[str],
    weights: dict[str, float],
) -> dict[str, float]:
    mean = sparam_weight_mean(labels, weights)
    return {label: float(weights[label] / mean) for label in labels}


def output_weights_from_sparam_weights(
    labels: Sequence[str],
    weights: dict[str, float],
    normalize: bool = True,
) -> np.ndarray:
    label_weights = normalize_sparam_weights(labels, weights) if normalize else weights
    values = np.asarray([label_weights[label] for label in labels] * 2, dtype=float)
    if normalize:
        mean = float(np.mean(values))
        if not np.isclose(mean, 1.0):
            values = values / max(mean, EPS)
    return values


def frequency_weights_for_values(
    freq_hz: np.ndarray,
    spec: str | None,
    *,
    require_all_rules_match: bool = True,
) -> np.ndarray:
    """Return raw per-frequency weights from exact-frequency and range rules.

    Rules use ``selector=weight`` entries separated by semicolons. A selector
    is ``all``/``default``, one frequency, or an inclusive ``start:stop``
    range. Commas may combine selectors on the left side, and later rules
    override earlier rules.
    """

    frequencies = np.asarray(freq_hz, dtype=float).reshape(-1)
    weights = np.ones(frequencies.shape, dtype=float)
    if not spec:
        return weights

    rules = [rule.strip() for rule in str(spec).split(";") if rule.strip()]
    if not rules:
        return weights
    for rule in rules:
        if "=" not in rule:
            raise ValueError(
                f"Frequency weight rule {rule!r} must look like selector=weight"
            )
        left, right = rule.rsplit("=", 1)
        try:
            weight = float(right.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid frequency weight in {rule!r}") from exc
        if weight < 0.0 or not math.isfinite(weight):
            raise ValueError(
                f"Frequency weights must be finite and non-negative, got {weight}"
            )
        selectors = [part.strip() for part in left.split(",") if part.strip()]
        if not selectors:
            raise ValueError(f"Frequency weight rule {rule!r} has no selector")
        rule_mask = np.zeros(frequencies.shape, dtype=bool)
        for selector in selectors:
            normalized = selector.lower().replace(" ", "")
            if normalized in {"all", "default", "*"}:
                selector_mask = np.ones(frequencies.shape, dtype=bool)
            elif ":" in selector:
                parts = [part.strip() for part in selector.split(":")]
                if len(parts) != 2:
                    raise ValueError(
                        f"Frequency range {selector!r} must look like start:stop"
                    )
                start = parse_number(parts[0])
                stop = parse_number(parts[1])
                if start is None or stop is None:
                    raise ValueError(f"Could not parse frequency range {selector!r}")
                if start > stop:
                    raise ValueError(
                        f"Frequency range start must not exceed stop in {selector!r}"
                    )
                selector_mask = (frequencies >= float(start)) & (
                    frequencies <= float(stop)
                )
            else:
                target = parse_number(selector)
                if target is None:
                    raise ValueError(f"Could not parse frequency selector {selector!r}")
                selector_mask = np.isclose(
                    frequencies,
                    float(target),
                    rtol=1.0e-12,
                    atol=max(1.0e-12, abs(float(target)) * 1.0e-12),
                )
            rule_mask |= selector_mask
        if require_all_rules_match and not np.any(rule_mask):
            raise ValueError(
                f"Frequency weight selector {left!r} did not match any fitted frequency"
            )
        weights[rule_mask] = weight

    if weights.size and float(np.sum(weights)) <= EPS:
        raise ValueError("At least one fitted frequency weight must be greater than zero")
    return weights


def frequency_weights_from_blocks(
    blocks: Sequence[MDIFBlock],
    spec: str | None,
    *,
    require_all_rules_match: bool = True,
) -> np.ndarray:
    """Return weights aligned with the concatenated sample rows of MDIF blocks."""

    frequencies = (
        np.concatenate([np.asarray(block.freq_hz, dtype=float) for block in blocks])
        if blocks
        else np.asarray([], dtype=float)
    )
    return frequency_weights_for_values(
        frequencies,
        spec,
        require_all_rules_match=require_all_rules_match,
    )


def normalize_frequency_weights(
    weights: np.ndarray,
    *,
    mean: float | None = None,
) -> tuple[np.ndarray, float]:
    """Normalize per-sample frequency weights and return the applied raw mean."""

    values = np.asarray(weights, dtype=float).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Frequency weights must be finite and non-negative")
    applied_mean = float(np.mean(values)) if mean is None else float(mean)
    if not math.isfinite(applied_mean) or applied_mean <= EPS:
        raise ValueError("At least one fitted frequency weight must be greater than zero")
    return values / applied_mean, applied_mean




class Standardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "Standardizer":
        self.mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < EPS] = 1.0
        self.std = std
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return data * self.std + self.mean


class MLP:
    def __init__(self, layer_sizes: Sequence[int], activation: str, seed: int) -> None:
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    def activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        raise ValueError(f"Unsupported activation {self.activation!r}")

    def activation_grad(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            a = np.tanh(z)
            return 1.0 - a * a
        if self.activation == "relu":
            return (z > 0.0).astype(float)
        raise ValueError(f"Unsupported activation {self.activation!r}")

    def activation_grad_from_activation(self, activation: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return 1.0 - activation * activation
        if self.activation == "relu":
            return (activation > 0.0).astype(float)
        raise ValueError(f"Unsupported activation {self.activation!r}")

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [x]
        preacts = []
        a = x
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = a @ weight + bias
            preacts.append(z)
            if idx == len(self.weights) - 1:
                a = z
            else:
                a = self.activate(z)
            activations.append(a)
        return a, activations, preacts

    def forward_training(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [x]
        a = x
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = a @ weight + bias
            if idx == len(self.weights) - 1:
                a = z
            else:
                a = self.activate(z)
            activations.append(a)
        return a, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = a @ weight + bias
            if idx == len(self.weights) - 1:
                a = z
            else:
                a = self.activate(z)
        return a

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None,
        y_val: np.ndarray | None,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        patience: int,
        seed: int,
        output_weights: np.ndarray | None = None,
        loss_interval: int = 1,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        progress_interval: int = 25,
        sample_weights: np.ndarray | None = None,
        val_sample_weights: np.ndarray | None = None,
    ) -> list[dict[str, float]]:
        if output_weights is None:
            output_weights = np.ones(y_train.shape[1], dtype=float)
        else:
            output_weights = np.asarray(output_weights, dtype=float)
            if output_weights.shape != (y_train.shape[1],):
                raise ValueError(
                    f"Expected {y_train.shape[1]} output weights, got {output_weights.shape}"
                )
        weighted_outputs = bool(np.any(np.abs(output_weights - 1.0) > 1e-15))
        if sample_weights is None:
            sample_weights = np.ones(len(x_train), dtype=float)
        else:
            sample_weights = np.asarray(sample_weights, dtype=float).reshape(-1)
            if sample_weights.shape != (len(x_train),):
                raise ValueError(
                    f"Expected {len(x_train)} training sample weights, got "
                    f"{sample_weights.shape}"
                )
            if np.any(~np.isfinite(sample_weights)) or np.any(sample_weights < 0.0):
                raise ValueError("Training sample weights must be finite and non-negative")
            if float(np.sum(sample_weights)) <= EPS:
                raise ValueError("At least one training sample weight must be greater than zero")
        weighted_samples = bool(np.any(np.abs(sample_weights - 1.0) > 1e-15))
        if x_val is None or y_val is None:
            val_sample_weights = None
        elif val_sample_weights is None:
            val_sample_weights = np.ones(len(x_val), dtype=float)
        else:
            val_sample_weights = np.asarray(val_sample_weights, dtype=float).reshape(-1)
            if val_sample_weights.shape != (len(x_val),):
                raise ValueError(
                    f"Expected {len(x_val)} validation sample weights, got "
                    f"{val_sample_weights.shape}"
                )
            if np.any(~np.isfinite(val_sample_weights)) or np.any(
                val_sample_weights < 0.0
            ):
                raise ValueError(
                    "Validation sample weights must be finite and non-negative"
                )
            if float(np.sum(val_sample_weights)) <= EPS:
                raise ValueError(
                    "At least one validation sample weight must be greater than zero"
                )
        rng = np.random.default_rng(seed)
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-8
        step = 0
        beta1_power = 1.0
        beta2_power = 1.0
        history: list[dict[str, float]] = []
        best_val = float("inf")
        best_weights = [w.copy() for w in self.weights]
        best_biases = [b.copy() for b in self.biases]
        best_epoch = 0
        stale = 0

        batch_size = max(1, min(batch_size, len(x_train)))
        loss_interval = max(1, int(loss_interval))
        progress_interval = max(0, int(progress_interval))
        progress_enabled = progress_callback is not None and progress_interval > 0
        order = np.arange(len(x_train))
        n_layers = len(self.weights)
        scale_base = 2.0 / y_train.shape[1]
        for epoch in range(1, epochs + 1):
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                xb = x_train[indices]
                yb = y_train[indices]
                pred, activations = self.forward_training(xb)
                delta = pred - yb
                if weighted_outputs:
                    delta *= output_weights[None, :]
                if weighted_samples:
                    delta *= sample_weights[indices, None]
                delta *= scale_base / len(xb)
                grad_w: list[np.ndarray | None] = [None] * n_layers
                grad_b: list[np.ndarray | None] = [None] * n_layers
                for layer in reversed(range(n_layers)):
                    a_prev = activations[layer]
                    grad_w[layer] = a_prev.T @ delta
                    grad_b[layer] = np.sum(delta, axis=0)
                    if layer > 0:
                        delta = (delta @ self.weights[layer].T) * self.activation_grad_from_activation(
                            activations[layer]
                        )

                step += 1
                beta1_power *= beta1
                beta2_power *= beta2
                sqrt_bias_correction2 = math.sqrt(1.0 - beta2_power)
                step_scale = learning_rate * sqrt_bias_correction2 / (1.0 - beta1_power)
                eps_hat = adam_eps * sqrt_bias_correction2
                for idx in range(n_layers):
                    weight_gradient = grad_w[idx]
                    bias_gradient = grad_b[idx]
                    assert weight_gradient is not None and bias_gradient is not None
                    m_w[idx] *= beta1
                    m_w[idx] += (1.0 - beta1) * weight_gradient
                    v_w[idx] *= beta2
                    v_w[idx] += (1.0 - beta2) * (weight_gradient * weight_gradient)
                    m_b[idx] *= beta1
                    m_b[idx] += (1.0 - beta1) * bias_gradient
                    v_b[idx] *= beta2
                    v_b[idx] += (1.0 - beta2) * (bias_gradient * bias_gradient)
                    self.weights[idx] -= step_scale * m_w[idx] / (np.sqrt(v_w[idx]) + eps_hat)
                    self.biases[idx] -= step_scale * m_b[idx] / (np.sqrt(v_b[idx]) + eps_hat)

            loss_epoch = epoch == 1 or epoch == epochs or epoch % loss_interval == 0
            progress_epoch = progress_enabled and (
                epoch == 1 or epoch == epochs or epoch % progress_interval == 0
            )
            if not loss_epoch:
                if progress_epoch:
                    progress_callback(
                        {
                            "epoch": epoch,
                            "epochs": epochs,
                            "train_loss": None,
                            "val_loss": None,
                            "best_val": best_val if math.isfinite(best_val) else None,
                            "best_epoch": best_epoch,
                            "stale": stale,
                            "stopped": False,
                        }
                    )
                continue

            train_loss = mse(
                self.predict(x_train),
                y_train,
                output_weights=output_weights,
                sample_weights=sample_weights,
            )
            val_loss = (
                mse(
                    self.predict(x_val),
                    y_val,
                    output_weights=output_weights,
                    sample_weights=val_sample_weights,
                )
                if x_val is not None and y_val is not None
                else train_loss
            )
            history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val - 1e-10:
                best_val = val_loss
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                best_epoch = epoch
                stale = 0
            else:
                stale = epoch - best_epoch

            stopped = patience > 0 and stale >= patience
            if progress_enabled and (progress_epoch or stopped):
                progress_callback(
                    {
                        "epoch": epoch,
                        "epochs": epochs,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "best_epoch": best_epoch,
                        "stale": stale,
                        "stopped": stopped,
                    }
                )

            if stopped:
                break

        self.weights = best_weights
        self.biases = best_biases
        return history


class DCConductanceModel:
    """Geometry-only neural surrogate for an exact-zero-frequency network.

    Explicit path mode fits logarithms of selected non-negative branch
    conductances. The generic full-S mode fits the real and imaginary component
    of every ordered Sij directly. The former full-Y representation remains
    loadable for backward compatibility. Frequency is deliberately not an input,
    so no representation can influence the positive-frequency RF fit.
    """

    def __init__(
        self,
        mlp: MLP,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        parameter_names: Sequence[str],
        sparam_labels: Sequence[str],
        port_paths: Sequence[str],
        z0: float,
        log_conductance_min: float,
        log_conductance_max: float,
        representation: str = "path_conductance",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = list(parameter_names)
        self.sparam_labels = list(sparam_labels)
        self.port_paths = list(port_paths)
        self.z0 = float(z0)
        self.log_conductance_min = float(log_conductance_min)
        self.log_conductance_max = float(log_conductance_max)
        self.representation = str(representation)
        self.metadata = dict(metadata or {})
        nports = infer_complete_sparameter_ports(self.sparam_labels)
        if self.mlp.layer_sizes[0] != len(self.parameter_names):
            raise ValueError("DC model input dimension does not match parameter names")
        if self.representation == "path_conductance":
            parsed = parse_dc_port_paths(self.port_paths, nports)
            if [item[2] for item in parsed] != self.port_paths:
                raise ValueError("DC model path order is not canonical")
            if self.mlp.layer_sizes[-1] != len(self.port_paths):
                raise ValueError("DC model output dimension does not match port paths")
        elif self.representation == "full_y_matrix":
            if self.port_paths:
                raise ValueError("Full-matrix DC model must not contain resistor paths")
            if self.mlp.layer_sizes[-1] != nports * nports:
                raise ValueError("Full-matrix DC model output dimension is not N-port Y")
        elif self.representation == "full_s_matrix":
            if self.port_paths:
                raise ValueError("Full-S DC model must not contain resistor paths")
            if self.mlp.layer_sizes[-1] != 2 * nports * nports:
                raise ValueError(
                    "Full-S DC model output dimension is not complex N-port S"
                )
        else:
            raise ValueError(f"Unsupported DC model representation {self.representation!r}")

    @property
    def kind(self) -> str:
        if self.representation == "full_s_matrix":
            return "geometry_dependent_exact_dc_full_s_mlp"
        if self.representation == "full_y_matrix":
            return "geometry_dependent_exact_dc_full_y_mlp"
        return "geometry_dependent_exact_dc_conductance_mlp"

    def predict_outputs(self, parameter_values: np.ndarray) -> np.ndarray:
        values = np.asarray(parameter_values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        scaled = self.x_scaler.transform(values)
        predicted_scaled = self.mlp.predict(scaled)
        return self.y_scaler.inverse_transform(predicted_scaled)

    def predict_log_conductances(self, parameter_values: np.ndarray) -> np.ndarray:
        if self.representation != "path_conductance":
            raise ValueError("Full-matrix DC model does not contain path conductances")
        predicted = self.predict_outputs(parameter_values)
        return np.clip(
            predicted,
            self.log_conductance_min,
            self.log_conductance_max,
        )

    def predict_conductances(self, parameter_values: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_conductances(parameter_values))

    def predict_matrix_entries(self, parameter_values: np.ndarray) -> np.ndarray:
        if self.representation != "full_y_matrix":
            raise ValueError("Path DC model does not contain direct matrix entries")
        return self.predict_outputs(parameter_values)

    def predict_s_components(self, parameter_values: np.ndarray) -> np.ndarray:
        if self.representation != "full_s_matrix":
            raise ValueError("This DC model does not contain direct complex S entries")
        return self.predict_outputs(parameter_values)

    def predict_s_matrices(self, parameter_values: np.ndarray) -> np.ndarray:
        components = self.predict_s_components(parameter_values)
        nports = infer_complete_sparameter_ports(self.sparam_labels)
        matrix_size = nports * nports
        return (
            components[:, :matrix_size] + 1j * components[:, matrix_size:]
        ).reshape(-1, nports, nports)

    def conductance_matrix(self, parameter_values: np.ndarray) -> np.ndarray:
        nports = infer_complete_sparameter_ports(self.sparam_labels)
        if self.representation == "full_s_matrix":
            return _s_matrix_to_y_matrix(
                self.predict_s_matrices(parameter_values)[0],
                self.z0,
            )
        if self.representation == "full_y_matrix":
            return self.predict_matrix_entries(parameter_values)[0].reshape(
                nports,
                nports,
            )
        conductances = self.predict_conductances(parameter_values)[0]
        return _dc_matrix_from_path_conductances(
            nports,
            parse_dc_port_paths(self.port_paths, nports),
            conductances,
        )

    def predict_s_values(self, parameter_values: np.ndarray) -> np.ndarray:
        if self.representation == "full_s_matrix":
            matrix = self.predict_s_matrices(parameter_values)[0]
            return np.asarray(
                [
                    matrix[(sparam_indices(label) or (0, 0))[0] - 1,
                           (sparam_indices(label) or (0, 0))[1] - 1]
                    for label in self.sparam_labels
                ],
                dtype=complex,
            )
        y_matrix = self.conductance_matrix(parameter_values).astype(complex)
        s_matrix = _y_matrix_to_s_matrix(y_matrix, self.z0)
        values: list[complex] = []
        for label in self.sparam_labels:
            row, col = sparam_indices(label) or (0, 0)
            values.append(s_matrix[row - 1, col - 1])
        return np.asarray(values, dtype=complex)

    def predict_block_s_values(self, block: MDIFBlock) -> np.ndarray:
        return self.predict_s_values(parameter_matrix([block], self.parameter_names))

    def export_data(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "representation": self.representation,
            "parameter_names": list(self.parameter_names),
            "sparam_labels": list(self.sparam_labels),
            "port_paths": list(self.port_paths),
            "matrix_entries": list(self.metadata.get("dc_matrix_entries", [])),
            "sparameter_entries": list(
                self.metadata.get("dc_sparameter_entries", [])
            ),
            "z0": self.z0,
            "activation": self.mlp.activation,
            "layer_sizes": list(self.mlp.layer_sizes),
            "weights": [value.copy() for value in self.mlp.weights],
            "biases": [value.copy() for value in self.mlp.biases],
            "x_mean": np.asarray(self.x_scaler.mean, dtype=float).copy(),
            "x_std": np.asarray(self.x_scaler.std, dtype=float).copy(),
            "y_mean": np.asarray(self.y_scaler.mean, dtype=float).copy(),
            "y_std": np.asarray(self.y_scaler.std, dtype=float).copy(),
            "log_conductance_min": self.log_conductance_min,
            "log_conductance_max": self.log_conductance_max,
            "metadata": dict(self.metadata),
        }

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "x_mean": np.asarray(self.x_scaler.mean, dtype=float),
            "x_std": np.asarray(self.x_scaler.std, dtype=float),
            "y_mean": np.asarray(self.y_scaler.mean, dtype=float),
            "y_std": np.asarray(self.y_scaler.std, dtype=float),
        }
        for idx, (weight, bias) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
            arrays[f"W{idx}"] = weight
            arrays[f"b{idx}"] = bias
        np.savez_compressed(out_dir / "dc_model.npz", **arrays)
        payload = {
            "version": VERSION,
            "kind": self.kind,
            "representation": self.representation,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "port_paths": self.port_paths,
            "z0": self.z0,
            "activation": self.mlp.activation,
            "layer_sizes": self.mlp.layer_sizes,
            "log_conductance_min": self.log_conductance_min,
            "log_conductance_max": self.log_conductance_max,
            **self.metadata,
        }
        (out_dir / "dc_model.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def load_optional(model_dir: Path) -> "DCConductanceModel | None":
        metadata_path = model_dir / "dc_model.json"
        arrays_path = model_dir / "dc_model.npz"
        if not metadata_path.exists() and not arrays_path.exists():
            return None
        if not metadata_path.exists() or not arrays_path.exists():
            raise ValueError(
                f"Incomplete DC surrogate in {model_dir}: dc_model.json and "
                "dc_model.npz must both be present"
            )
        metadata = json.loads(metadata_path.read_text())
        data = np.load(arrays_path)
        mlp = MLP(metadata["layer_sizes"], metadata["activation"], seed=1)
        for idx in range(len(mlp.weights)):
            mlp.weights[idx] = data[f"W{idx}"]
            mlp.biases[idx] = data[f"b{idx}"]
        x_scaler = Standardizer()
        x_scaler.mean = data["x_mean"]
        x_scaler.std = data["x_std"]
        y_scaler = Standardizer()
        y_scaler.mean = data["y_mean"]
        y_scaler.std = data["y_std"]
        structural_keys = {
            "version",
            "kind",
            "representation",
            "parameter_names",
            "sparam_labels",
            "port_paths",
            "z0",
            "activation",
            "layer_sizes",
            "log_conductance_min",
            "log_conductance_max",
        }
        representation = str(
            metadata.get(
                "representation",
                (
                    "full_s_matrix"
                    if metadata.get("kind")
                    == "geometry_dependent_exact_dc_full_s_mlp"
                    else "full_y_matrix"
                    if metadata.get("kind")
                    == "geometry_dependent_exact_dc_full_y_mlp"
                    else "path_conductance"
                ),
            )
        )
        return DCConductanceModel(
            mlp=mlp,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            parameter_names=metadata["parameter_names"],
            sparam_labels=metadata["sparam_labels"],
            port_paths=metadata.get("port_paths", []),
            z0=float(metadata["z0"]),
            log_conductance_min=float(metadata.get("log_conductance_min", -50.0)),
            log_conductance_max=float(metadata.get("log_conductance_max", 50.0)),
            representation=representation,
            metadata={
                key: value for key, value in metadata.items() if key not in structural_keys
            },
        )


def _y_matrix_to_s_matrix(y_matrix: np.ndarray, z0: float) -> np.ndarray:
    identity = np.eye(y_matrix.shape[0], dtype=complex)
    normalized_y = float(z0) * np.asarray(y_matrix, dtype=complex)
    lhs = identity + normalized_y
    rhs = identity - normalized_y
    try:
        return np.linalg.solve(lhs.T, rhs.T).T
    except np.linalg.LinAlgError:
        return rhs @ np.linalg.pinv(lhs)


def _s_matrix_to_y_matrix(s_matrix: np.ndarray, z0: float) -> np.ndarray:
    """Convert a complete complex S matrix to Y at the given reference impedance."""

    identity = np.eye(s_matrix.shape[0], dtype=complex)
    lhs = identity + np.asarray(s_matrix, dtype=complex)
    rhs = identity - np.asarray(s_matrix, dtype=complex)
    try:
        return np.linalg.solve(lhs.T, rhs.T).T / float(z0)
    except np.linalg.LinAlgError:
        return rhs @ np.linalg.pinv(lhs) / float(z0)


def _dc_matrix_from_path_conductances(
    nports: int,
    paths: Sequence[tuple[int, int | None, str]],
    conductances: Sequence[float],
) -> np.ndarray:
    matrix = np.zeros((nports, nports), dtype=float)
    for (first, second, _), raw_conductance in zip(paths, conductances):
        conductance = max(0.0, float(raw_conductance))
        matrix[first, first] += conductance
        if second is not None:
            matrix[second, second] += conductance
            matrix[first, second] -= conductance
            matrix[second, first] -= conductance
    return matrix


def _dc_path_basis(
    nports: int,
    paths: Sequence[tuple[int, int | None, str]],
) -> np.ndarray:
    columns = []
    for path_index in range(len(paths)):
        conductances = np.zeros(len(paths), dtype=float)
        conductances[path_index] = 1.0
        columns.append(
            _dc_matrix_from_path_conductances(nports, paths, conductances).reshape(-1)
        )
    return np.column_stack(columns)


def _nonnegative_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 2000,
) -> np.ndarray:
    """Small dependency-free projected-gradient NNLS solver."""

    coefficients = np.maximum(
        np.linalg.lstsq(matrix, target, rcond=None)[0],
        0.0,
    )
    spectral_norm = float(np.linalg.norm(matrix, ord=2))
    if not math.isfinite(spectral_norm) or spectral_norm <= EPS:
        raise ValueError("Selected DC path topology has no usable conductance basis")
    step = 1.0 / (spectral_norm * spectral_norm)
    for _ in range(max_iterations):
        updated = np.maximum(
            coefficients - step * (matrix.T @ (matrix @ coefficients - target)),
            0.0,
        )
        if np.linalg.norm(updated - coefficients) <= 1e-12 * max(
            1.0, np.linalg.norm(coefficients)
        ):
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def extract_dc_full_s_samples(
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    z0: float = 50.0,
    passivity_tolerance: float = DEFAULT_DC_PASSIVITY_TOLERANCE,
    require_every_block: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract both components of every ordered exact-DC Sij entry."""

    if not blocks:
        raise ValueError("DC full-S fitting requires at least one MDIF block")
    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError("DC reference impedance must be positive and finite")
    nports = infer_complete_sparameter_ports(labels)
    passivity_limit = 1.0 + float(passivity_tolerance)
    sample_blocks: list[MDIFBlock] = []
    component_rows: list[np.ndarray] = []
    missing_blocks: list[int] = []
    unusable_blocks: list[int] = []
    ignored_nonpassive = 0
    ignored_nonfinite = 0
    dc_row_count = 0

    for block in blocks:
        exact_indices = np.flatnonzero(np.asarray(block.freq_hz, dtype=float) == 0.0)
        if exact_indices.size == 0:
            missing_blocks.append(int(block.source_index) + 1)
            continue
        block_s: list[np.ndarray] = []
        for raw_index in exact_indices:
            dc_row_count += 1
            s_matrix = _block_s_matrix(block, labels, int(raw_index), nports)
            if not np.all(np.isfinite(s_matrix)):
                ignored_nonfinite += 1
                continue
            try:
                max_sigma = float(np.max(np.linalg.svd(s_matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                ignored_nonfinite += 1
                continue
            if not math.isfinite(max_sigma):
                ignored_nonfinite += 1
                continue
            if max_sigma > passivity_limit:
                ignored_nonpassive += 1
                continue
            block_s.append(s_matrix.reshape(-1))
        if not block_s:
            unusable_blocks.append(int(block.source_index) + 1)
            continue
        mean_s = np.mean(np.asarray(block_s, dtype=complex), axis=0)
        sample_blocks.append(block)
        component_rows.append(np.concatenate([mean_s.real, mean_s.imag]))

    if missing_blocks and require_every_block:
        raise ValueError(
            "Every DC-training geometry must contain an exact zero-Hz row; missing "
            "one-based ACDATA block position(s) in the source MDIF: "
            + ", ".join(map(str, missing_blocks[:20]))
        )
    if not component_rows:
        if require_every_block:
            raise ValueError(
                "No usable passive exact-zero-Hz rows remain after excluding "
                "non-passive or non-finite DC samples; a DC model cannot be fitted"
            )
        x_values = np.empty((0, len(parameter_names)), dtype=float)
        component_values = np.empty((0, 2 * nports * nports), dtype=float)
    else:
        x_values = parameter_matrix(sample_blocks, parameter_names)
        component_values = np.asarray(component_rows, dtype=float)
    matrix_entries = [
        f"S{row}{col}.{component}"
        for component in ("real", "imag")
        for row in range(1, nports + 1)
        for col in range(1, nports + 1)
    ]
    metadata: dict[str, object] = {
        "dc_model_kind": "geometry_dependent_exact_dc_full_s_mlp",
        "dc_model_representation": "full_s_matrix",
        "dc_resistance_source_kind": "exact_zero_frequency",
        "dc_port_paths": [],
        "dc_port_path_spec": "",
        "dc_port_paths_explicit": False,
        "dc_port_path_selection": "full_ordered_complex_s_matrix",
        "dc_matrix_entries": matrix_entries,
        "dc_sparameter_entries": list(labels),
        "dc_row_count": dc_row_count,
        "dc_resistance_block_count": len(sample_blocks),
        "dc_missing_block_count": len(missing_blocks),
        "dc_missing_block_positions": missing_blocks,
        "dc_unusable_block_count": len(unusable_blocks),
        "dc_unusable_block_positions": unusable_blocks,
        "dc_usable_block_positions": [
            int(block.source_index) + 1 for block in sample_blocks
        ],
        "dc_ignored_nonpassive_count": ignored_nonpassive,
        "dc_ignored_nonfinite_count": ignored_nonfinite,
        "dc_ignored_invalid_resistance_count": 0,
        "dc_passivity_tolerance": float(passivity_tolerance),
        "dc_resistance_source_z0_ohm": float(z0),
        "dc_resistance_source_frequency_min_hz": 0.0,
        "dc_resistance_source_frequency_max_hz": 0.0,
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_topology_s_rmse": 0.0,
        "dc_topology_s_max_abs_error": 0.0,
        "dc_resistance_extraction": (
            "Both real and imaginary components of every ordered Sij value are "
            "taken directly from passive exact-zero-Hz rows; no Y projection, "
            "resistor-path projection, or reciprocity constraint is applied"
        ),
        "dc_response_topology": (
            "Geometry-dependent full ordered complex S matrix evaluated only at zero Hz"
        ),
    }
    return x_values, component_values, metadata


def extract_dc_full_y_samples(
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    z0: float = 50.0,
    passivity_tolerance: float = DEFAULT_DC_PASSIVITY_TOLERANCE,
    require_every_block: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract every ordered real DC Y entry without a resistor-path projection."""

    if not blocks:
        raise ValueError("DC full-matrix fitting requires at least one MDIF block")
    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError("DC reference impedance must be positive and finite")
    nports = infer_complete_sparameter_ports(labels)
    passivity_limit = 1.0 + float(passivity_tolerance)
    identity = np.eye(nports, dtype=complex)
    sample_blocks: list[MDIFBlock] = []
    y_rows: list[np.ndarray] = []
    measured_s_rows: list[np.ndarray] = []
    reconstructed_s_rows: list[np.ndarray] = []
    y_projection_errors: list[float] = []
    missing_blocks: list[int] = []
    unusable_blocks: list[int] = []
    ignored_nonpassive = 0
    ignored_nonfinite = 0
    dc_row_count = 0

    for block in blocks:
        exact_indices = np.flatnonzero(np.asarray(block.freq_hz, dtype=float) == 0.0)
        if exact_indices.size == 0:
            missing_blocks.append(int(block.source_index) + 1)
            continue
        block_y: list[np.ndarray] = []
        block_s: list[np.ndarray] = []
        block_reconstructed_s: list[np.ndarray] = []
        for raw_index in exact_indices:
            dc_row_count += 1
            s_matrix = _block_s_matrix(block, labels, int(raw_index), nports)
            if not np.all(np.isfinite(s_matrix)):
                ignored_nonfinite += 1
                continue
            try:
                max_sigma = float(np.max(np.linalg.svd(s_matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                ignored_nonfinite += 1
                continue
            if not math.isfinite(max_sigma):
                ignored_nonfinite += 1
                continue
            if max_sigma > passivity_limit:
                ignored_nonpassive += 1
                continue
            lhs = identity + s_matrix
            rhs = identity - s_matrix
            try:
                y_matrix = np.linalg.solve(lhs.T, rhs.T).T / float(z0)
            except np.linalg.LinAlgError:
                y_matrix = rhs @ np.linalg.pinv(lhs) / float(z0)
            if not np.all(np.isfinite(y_matrix)):
                ignored_nonfinite += 1
                continue
            real_y = np.real(y_matrix)
            reconstructed_s = _y_matrix_to_s_matrix(real_y.astype(complex), z0)
            y_projection_errors.append(
                float(np.sqrt(np.mean(np.abs(real_y - y_matrix) ** 2)))
            )
            block_y.append(real_y.reshape(-1))
            block_s.append(s_matrix)
            block_reconstructed_s.append(reconstructed_s)
        if not block_y:
            unusable_blocks.append(int(block.source_index) + 1)
            continue
        sample_blocks.append(block)
        y_rows.append(np.mean(np.asarray(block_y), axis=0))
        measured_s_rows.append(np.mean(np.asarray(block_s), axis=0))
        reconstructed_s_rows.append(
            np.mean(np.asarray(block_reconstructed_s), axis=0)
        )

    if missing_blocks and require_every_block:
        raise ValueError(
            "Every DC-training geometry must contain an exact zero-Hz row; missing "
            "one-based ACDATA block position(s) in the source MDIF: "
            + ", ".join(map(str, missing_blocks[:20]))
        )
    if not y_rows:
        if require_every_block:
            raise ValueError(
                "No usable passive exact-zero-Hz rows remain after excluding "
                "non-passive or non-finite DC samples; a DC model cannot be fitted"
            )
        x_values = np.empty((0, len(parameter_names)), dtype=float)
        y_values = np.empty((0, nports * nports), dtype=float)
        s_error = np.empty((0, nports, nports), dtype=float)
    else:
        x_values = parameter_matrix(sample_blocks, parameter_names)
        y_values = np.asarray(y_rows, dtype=float)
        s_error = np.abs(
            np.asarray(reconstructed_s_rows, dtype=complex)
            - np.asarray(measured_s_rows, dtype=complex)
        )
    matrix_entries = [
        f"Y{row}{col}"
        for row in range(1, nports + 1)
        for col in range(1, nports + 1)
    ]
    metadata: dict[str, object] = {
        "dc_model_kind": "geometry_dependent_exact_dc_full_y_mlp",
        "dc_model_representation": "full_y_matrix",
        "dc_resistance_source_kind": "exact_zero_frequency",
        "dc_port_paths": [],
        "dc_port_path_spec": "",
        "dc_port_paths_explicit": False,
        "dc_port_path_selection": "full_ordered_y_matrix",
        "dc_matrix_entries": matrix_entries,
        "dc_row_count": dc_row_count,
        "dc_resistance_block_count": len(sample_blocks),
        "dc_missing_block_count": len(missing_blocks),
        "dc_missing_block_positions": missing_blocks,
        "dc_unusable_block_count": len(unusable_blocks),
        "dc_unusable_block_positions": unusable_blocks,
        "dc_usable_block_positions": [
            int(block.source_index) + 1 for block in sample_blocks
        ],
        "dc_ignored_nonpassive_count": ignored_nonpassive,
        "dc_ignored_nonfinite_count": ignored_nonfinite,
        "dc_ignored_invalid_resistance_count": 0,
        "dc_passivity_tolerance": float(passivity_tolerance),
        "dc_resistance_source_z0_ohm": float(z0),
        "dc_resistance_source_frequency_min_hz": 0.0,
        "dc_resistance_source_frequency_max_hz": 0.0,
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_topology_y_rmse_siemens": (
            float(np.sqrt(np.mean(np.square(y_projection_errors))))
            if y_projection_errors
            else None
        ),
        "dc_topology_s_rmse": (
            float(np.sqrt(np.mean(s_error * s_error))) if s_error.size else None
        ),
        "dc_topology_s_max_abs_error": (
            float(np.max(s_error)) if s_error.size else None
        ),
        "dc_resistance_extraction": (
            "Every ordered real Y-matrix entry converted from each passive exact-zero-Hz "
            "S-matrix; no resistor-path projection or reciprocity constraint is applied"
        ),
        "dc_response_topology": (
            "Geometry-dependent full ordered real Y matrix evaluated only at zero Hz"
        ),
    }
    return x_values, y_values, metadata


def extract_dc_conductance_samples(
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    z0: float = 50.0,
    port_paths: object = None,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
    passivity_tolerance: float = DEFAULT_DC_PASSIVITY_TOLERANCE,
    require_every_block: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract one selected-path conductance vector per geometry at exact DC."""

    if not blocks:
        raise ValueError("DC conductance fitting requires at least one MDIF block")
    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError("DC reference impedance must be positive and finite")
    if open_threshold_ohm <= 0.0 or not math.isfinite(open_threshold_ohm):
        raise ValueError("DC open threshold must be positive and finite")
    if open_resistance_ohm <= open_threshold_ohm or not math.isfinite(
        open_resistance_ohm
    ):
        raise ValueError("DC open resistance must exceed the open threshold")
    nports = infer_complete_sparameter_ports(labels)
    port_paths_explicit = not (
        port_paths is None
        or (isinstance(port_paths, str) and not port_paths.strip())
    )
    paths = parse_dc_port_paths(port_paths, nports)
    basis = _dc_path_basis(nports, paths)
    passivity_limit = 1.0 + float(passivity_tolerance)
    minimum_conductance = 1.0 / float(open_resistance_ohm)
    open_threshold_conductance = 1.0 / float(open_threshold_ohm)
    sample_blocks: list[MDIFBlock] = []
    conductance_rows: list[np.ndarray] = []
    measured_s_rows: list[np.ndarray] = []
    topology_s_rows: list[np.ndarray] = []
    missing_blocks: list[int] = []
    unusable_blocks: list[int] = []
    ignored_nonpassive = 0
    ignored_nonfinite = 0
    dc_row_count = 0
    topology_y_errors: list[float] = []

    for block_index, block in enumerate(blocks):
        exact_indices = np.flatnonzero(np.asarray(block.freq_hz, dtype=float) == 0.0)
        if exact_indices.size == 0:
            missing_blocks.append(int(block.source_index) + 1)
            continue
        block_conductances: list[np.ndarray] = []
        block_measured_s: list[np.ndarray] = []
        block_topology_s: list[np.ndarray] = []
        for frequency_index in exact_indices:
            dc_row_count += 1
            s_matrix = _block_s_matrix(block, labels, int(frequency_index), nports)
            if not np.all(np.isfinite(s_matrix)):
                ignored_nonfinite += 1
                continue
            try:
                max_sigma = float(np.max(np.linalg.svd(s_matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                ignored_nonfinite += 1
                continue
            if not math.isfinite(max_sigma):
                ignored_nonfinite += 1
                continue
            if max_sigma > passivity_limit:
                ignored_nonpassive += 1
                continue
            identity = np.eye(nports, dtype=complex)
            lhs = identity + s_matrix
            rhs = identity - s_matrix
            try:
                y_matrix = np.linalg.solve(lhs.T, rhs.T).T / float(z0)
            except np.linalg.LinAlgError:
                y_matrix = rhs @ np.linalg.pinv(lhs) / float(z0)
            if not np.all(np.isfinite(y_matrix)):
                ignored_nonfinite += 1
                continue
            target_matrix = np.real(0.5 * (y_matrix + y_matrix.T))
            conductances = _nonnegative_least_squares(basis, target_matrix.reshape(-1))
            conductances[conductances < open_threshold_conductance] = minimum_conductance
            topology_y = _dc_matrix_from_path_conductances(nports, paths, conductances)
            topology_s = _y_matrix_to_s_matrix(topology_y.astype(complex), z0)
            topology_y_errors.append(
                float(np.sqrt(np.mean(np.abs(topology_y - y_matrix) ** 2)))
            )
            block_conductances.append(conductances)
            block_measured_s.append(s_matrix)
            block_topology_s.append(topology_s)
        if not block_conductances:
            unusable_blocks.append(int(block.source_index) + 1)
            continue
        sample_blocks.append(block)
        conductance_rows.append(np.mean(np.asarray(block_conductances), axis=0))
        measured_s_rows.append(np.mean(np.asarray(block_measured_s), axis=0))
        topology_s_rows.append(np.mean(np.asarray(block_topology_s), axis=0))

    if missing_blocks and require_every_block:
        raise ValueError(
            "Every DC-training geometry must contain an exact zero-Hz row; missing "
            "one-based ACDATA block position(s) in the source MDIF: "
            + ", ".join(map(str, missing_blocks[:20]))
        )
    if not conductance_rows:
        if require_every_block:
            raise ValueError(
                "No usable passive exact-zero-Hz rows remain after excluding "
                "non-passive or non-finite DC samples; a DC model cannot be fitted"
            )
        x_values = np.empty((0, len(parameter_names)), dtype=float)
        conductances = np.empty((0, len(paths)), dtype=float)
        s_error = np.empty((0, nports, nports), dtype=float)
    else:
        x_values = parameter_matrix(sample_blocks, parameter_names)
        conductances = np.asarray(conductance_rows, dtype=float)
        measured_s = np.asarray(measured_s_rows, dtype=complex)
        topology_s = np.asarray(topology_s_rows, dtype=complex)
        s_error = np.abs(topology_s - measured_s)
    metadata: dict[str, object] = {
        "dc_model_kind": "geometry_dependent_exact_dc_conductance_mlp",
        "dc_resistance_source_kind": "exact_zero_frequency",
        "dc_port_paths": [item[2] for item in paths],
        "dc_port_path_spec": dc_port_path_spec([item[2] for item in paths]),
        "dc_port_paths_explicit": port_paths_explicit,
        "dc_port_path_selection": (
            "explicit" if port_paths_explicit else "automatic_complete_graph"
        ),
        "dc_row_count": dc_row_count,
        "dc_resistance_block_count": len(sample_blocks),
        "dc_missing_block_count": len(missing_blocks),
        "dc_missing_block_positions": missing_blocks,
        "dc_unusable_block_count": len(unusable_blocks),
        "dc_unusable_block_positions": unusable_blocks,
        "dc_usable_block_positions": [
            int(block.source_index) + 1 for block in sample_blocks
        ],
        "dc_ignored_nonpassive_count": ignored_nonpassive,
        "dc_ignored_nonfinite_count": ignored_nonfinite,
        "dc_ignored_invalid_resistance_count": 0,
        "dc_passivity_tolerance": float(passivity_tolerance),
        "dc_open_threshold_ohm": float(open_threshold_ohm),
        "dc_open_resistance_ohm": float(open_resistance_ohm),
        "dc_resistance_source_z0_ohm": float(z0),
        "dc_resistance_source_frequency_min_hz": 0.0,
        "dc_resistance_source_frequency_max_hz": 0.0,
        "dc_is_separate_from_fitted_response": True,
        "dc_requires_exact_zero_frequency": True,
        "dc_rf_fallback_allowed": False,
        "dc_topology_y_rmse_siemens": (
            float(np.sqrt(np.mean(np.square(topology_y_errors))))
            if topology_y_errors
            else None
        ),
        "dc_topology_s_rmse": (
            float(np.sqrt(np.mean(s_error * s_error))) if s_error.size else None
        ),
        "dc_topology_s_max_abs_error": (
            float(np.max(s_error)) if s_error.size else None
        ),
        "dc_resistance_extraction": (
            "Each passive exact-zero-Hz S-matrix is converted to Y and projected onto "
            "the declared non-negative branch-conductance graph before geometry-only fitting"
        ),
        "dc_response_topology": (
            "Geometry-dependent selected-path conductance graph evaluated only at exactly zero Hz"
        ),
    }
    return x_values, conductances, metadata


def train_dc_conductance_model(
    train_blocks: Sequence[MDIFBlock],
    verify_blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    hidden_layers: Sequence[int],
    activation: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    loss_interval: int = 1,
    progress_interval: int = 25,
    progress_label: str = "DC fit",
    z0: float = 50.0,
    port_paths: object = None,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
) -> tuple[DCConductanceModel, list[dict[str, float]], dict[str, object]]:
    if port_paths is None or (isinstance(port_paths, str) and not port_paths.strip()):
        return train_dc_full_s_model(
            train_blocks,
            verify_blocks,
            parameter_names,
            labels,
            hidden_layers=hidden_layers,
            activation=activation,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            seed=seed,
            loss_interval=loss_interval,
            progress_interval=progress_interval,
            progress_label=progress_label,
            z0=z0,
            open_threshold_ohm=open_threshold_ohm,
            open_resistance_ohm=open_resistance_ohm,
        )
    x_train, conductance_train, metadata = extract_dc_conductance_samples(
        train_blocks,
        parameter_names,
        labels,
        z0=z0,
        port_paths=port_paths,
        open_threshold_ohm=open_threshold_ohm,
        open_resistance_ohm=open_resistance_ohm,
    )
    if verify_blocks:
        x_verify, conductance_verify, verify_metadata = extract_dc_conductance_samples(
            verify_blocks,
            parameter_names,
            labels,
            z0=z0,
            port_paths=metadata["dc_port_paths"],
            open_threshold_ohm=open_threshold_ohm,
            open_resistance_ohm=open_resistance_ohm,
            require_every_block=False,
        )
        if x_verify.shape[0] == 0:
            x_verify = None
            conductance_verify = None
    else:
        x_verify = None
        conductance_verify = None
        verify_metadata = None
    log_min = math.log(1.0 / float(open_resistance_ohm))
    observed_log_max = float(np.max(np.log(np.maximum(conductance_train, math.exp(log_min)))))
    log_max = min(observed_log_max + math.log(100.0), 40.0)
    y_train = np.log(np.maximum(conductance_train, math.exp(log_min)))
    y_verify = (
        np.log(np.maximum(conductance_verify, math.exp(log_min)))
        if conductance_verify is not None
        else None
    )
    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(y_train)
    constant_output_mask = np.std(y_train, axis=0) < EPS
    mlp = MLP(
        [x_train.shape[1], *list(hidden_layers), y_train.shape[1]],
        activation=activation,
        seed=seed + 101,
    )
    history = mlp.train(
        x_scaler.transform(x_train),
        y_scaler.transform(y_train),
        x_scaler.transform(x_verify) if x_verify is not None else None,
        y_scaler.transform(y_verify) if y_verify is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed + 103,
        loss_interval=loss_interval,
        progress_callback=make_training_progress_callback(
            progress_label,
            epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    # A path that is constant across every training geometry is represented
    # exactly by its stored mean.  Leaving Standardizer's unit fallback scale
    # in place would unnecessarily make that constant depend on neural-fit
    # convergence and could badly perturb DC after a short run.
    assert y_scaler.std is not None
    y_scaler.std[constant_output_mask] = 0.0
    model = DCConductanceModel(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=labels,
        port_paths=metadata["dc_port_paths"],  # type: ignore[arg-type]
        z0=z0,
        log_conductance_min=log_min,
        log_conductance_max=log_max,
    )
    train_prediction = model.predict_conductances(x_train)
    train_log_error = model.predict_log_conductances(x_train) - y_train

    nports = infer_complete_sparameter_ports(labels)
    parsed_paths = parse_dc_port_paths(model.port_paths, nports)

    def branch_s_matrices(rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _y_matrix_to_s_matrix(
                    _dc_matrix_from_path_conductances(
                        nports,
                        parsed_paths,
                        row,
                    ).astype(complex),
                    z0,
                )
                for row in np.asarray(rows, dtype=float)
            ],
            dtype=complex,
        )

    train_s_error = np.abs(
        branch_s_matrices(train_prediction) - branch_s_matrices(conductance_train)
    )
    metadata.update(
        {
            "dc_model_training_samples": int(len(x_train)),
            "dc_model_verification_samples": int(len(x_verify)) if x_verify is not None else 0,
            "dc_model_activation": activation,
            "dc_model_layer_sizes": list(model.mlp.layer_sizes),
            "dc_model_log_conductance_min": log_min,
            "dc_model_log_conductance_max": log_max,
            "dc_model_train_log_rmse": float(np.sqrt(np.mean(train_log_error**2))),
            "dc_model_train_conductance_rmse_siemens": float(
                np.sqrt(np.mean((train_prediction - conductance_train) ** 2))
            ),
            "dc_model_train_s_rmse": float(
                np.sqrt(np.mean(train_s_error * train_s_error))
            ),
            "dc_model_train_s_max_abs_error": float(np.max(train_s_error)),
            "dc_model_constant_paths": [
                path
                for path, is_constant in zip(model.port_paths, constant_output_mask)
                if bool(is_constant)
            ],
        }
    )
    if verify_metadata is not None:
        metadata["dc_model_verification_extraction"] = verify_metadata
    if x_verify is not None and y_verify is not None and conductance_verify is not None:
        verify_prediction = model.predict_conductances(x_verify)
        verify_log_error = model.predict_log_conductances(x_verify) - y_verify
        verify_s_error = np.abs(
            branch_s_matrices(verify_prediction)
            - branch_s_matrices(conductance_verify)
        )
        metadata.update(
            {
                "dc_model_verify_log_rmse": float(
                    np.sqrt(np.mean(verify_log_error**2))
                ),
                "dc_model_verify_conductance_rmse_siemens": float(
                    np.sqrt(np.mean((verify_prediction - conductance_verify) ** 2))
                ),
                "dc_model_verify_s_rmse": float(
                    np.sqrt(np.mean(verify_s_error * verify_s_error))
                ),
                "dc_model_verify_s_max_abs_error": float(np.max(verify_s_error)),
            }
        )
    mean_conductances = np.mean(conductance_train, axis=0)
    path_resistances = {
        path: min(1.0 / max(float(value), math.exp(log_min)), float(open_resistance_ohm))
        for path, value in zip(model.port_paths, mean_conductances)
    }
    aggregate_mean = float(np.mean(mean_conductances))
    metadata.update(
        {
            # Compatibility summaries for older reports/export readers.  These values
            # do not drive a newly saved model when dc_model.npz is present.
            "dc_equivalent_resistance_ohm": min(
                1.0 / max(aggregate_mean, math.exp(log_min)),
                float(open_resistance_ohm),
            ),
            "dc_equivalent_resistance_raw_mean_ohm": min(
                1.0 / max(aggregate_mean, math.exp(log_min)),
                float(open_resistance_ohm),
            ),
            "dc_resistance_sample_count": int(conductance_train.size),
            "dc_open_resistance_sample_count": int(
                np.sum(conductance_train <= math.exp(log_min) * (1.0 + 1e-12))
            ),
            "dc_mean_conductance_siemens": aggregate_mean,
            "dc_port_resistances_ohm": path_resistances,
            "dc_resistance_pair_means_ohm": path_resistances,
            "dc_open_paths": [
                path
                for path, value in path_resistances.items()
                if value >= float(open_resistance_ohm)
            ],
            "dc_open_path_count": sum(
                value >= float(open_resistance_ohm)
                for value in path_resistances.values()
            ),
            "dc_open_circuit_applied": any(
                value >= float(open_resistance_ohm)
                for value in path_resistances.values()
            ),
            "dc_aggregate_open_circuit_applied": aggregate_mean < 1.0 / open_threshold_ohm,
        }
    )
    model.metadata = dict(metadata)
    return model, history, metadata


def train_dc_full_s_model(
    train_blocks: Sequence[MDIFBlock],
    verify_blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    hidden_layers: Sequence[int],
    activation: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    loss_interval: int = 1,
    progress_interval: int = 25,
    progress_label: str = "DC fit",
    z0: float = 50.0,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
) -> tuple[DCConductanceModel, list[dict[str, float]], dict[str, object]]:
    """Fit both components of every ordered exact-DC Sij entry."""

    x_train, s_train, metadata = extract_dc_full_s_samples(
        train_blocks,
        parameter_names,
        labels,
        z0=z0,
    )
    if verify_blocks:
        x_verify, s_verify, verify_metadata = extract_dc_full_s_samples(
            verify_blocks,
            parameter_names,
            labels,
            z0=z0,
            require_every_block=False,
        )
        if x_verify.shape[0] == 0:
            x_verify = None
            s_verify = None
    else:
        x_verify = None
        s_verify = None
        verify_metadata = None

    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(s_train)
    constant_output_mask = np.std(s_train, axis=0) < EPS
    mlp = MLP(
        [x_train.shape[1], *list(hidden_layers), s_train.shape[1]],
        activation=activation,
        seed=seed + 101,
    )
    history = mlp.train(
        x_scaler.transform(x_train),
        y_scaler.transform(s_train),
        x_scaler.transform(x_verify) if x_verify is not None else None,
        y_scaler.transform(s_verify) if s_verify is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed + 103,
        loss_interval=loss_interval,
        progress_callback=make_training_progress_callback(
            progress_label,
            epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    assert y_scaler.std is not None
    y_scaler.std[constant_output_mask] = 0.0
    model = DCConductanceModel(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=labels,
        port_paths=[],
        z0=z0,
        log_conductance_min=-50.0,
        log_conductance_max=50.0,
        representation="full_s_matrix",
    )
    train_prediction = model.predict_s_components(x_train)
    train_component_error = train_prediction - s_train
    nports = infer_complete_sparameter_ports(labels)
    matrix_size = nports * nports

    def components_to_s(rows: np.ndarray) -> np.ndarray:
        values = np.asarray(rows, dtype=float)
        return (
            values[:, :matrix_size] + 1j * values[:, matrix_size:]
        ).reshape(-1, nports, nports)

    train_s_error = np.abs(
        components_to_s(train_prediction) - components_to_s(s_train)
    )
    metadata.update(
        {
            "dc_model_training_samples": int(len(x_train)),
            "dc_model_verification_samples": (
                int(len(x_verify)) if x_verify is not None else 0
            ),
            "dc_model_activation": activation,
            "dc_model_layer_sizes": list(model.mlp.layer_sizes),
            "dc_model_train_component_rmse": float(
                np.sqrt(np.mean(train_component_error**2))
            ),
            "dc_model_train_s_rmse": float(
                np.sqrt(np.mean(train_s_error * train_s_error))
            ),
            "dc_model_train_s_max_abs_error": float(np.max(train_s_error)),
            "dc_model_constant_matrix_entries": [
                entry
                for entry, is_constant in zip(
                    metadata["dc_matrix_entries"],
                    constant_output_mask,
                )
                if bool(is_constant)
            ],
            "dc_open_threshold_ohm": float(open_threshold_ohm),
            "dc_open_resistance_ohm": float(open_resistance_ohm),
        }
    )
    if verify_metadata is not None:
        metadata["dc_model_verification_extraction"] = verify_metadata
    if x_verify is not None and s_verify is not None:
        verify_prediction = model.predict_s_components(x_verify)
        verify_component_error = verify_prediction - s_verify
        verify_s_error = np.abs(
            components_to_s(verify_prediction) - components_to_s(s_verify)
        )
        metadata.update(
            {
                "dc_model_verify_component_rmse": float(
                    np.sqrt(np.mean(verify_component_error**2))
                ),
                "dc_model_verify_s_rmse": float(
                    np.sqrt(np.mean(verify_s_error * verify_s_error))
                ),
                "dc_model_verify_s_max_abs_error": float(np.max(verify_s_error)),
            }
        )

    mean_s = np.mean(components_to_s(s_train), axis=0)
    mean_y = _s_matrix_to_y_matrix(mean_s, z0)
    minimum_conductance = 1.0 / float(open_resistance_ohm)
    diagonal = np.maximum(np.real(np.diag(mean_y)), minimum_conductance)
    aggregate_conductance = float(np.mean(diagonal))
    equivalent_resistance = min(
        1.0 / max(aggregate_conductance, minimum_conductance),
        float(open_resistance_ohm),
    )
    metadata.update(
        {
            "dc_equivalent_resistance_ohm": equivalent_resistance,
            "dc_equivalent_resistance_raw_mean_ohm": equivalent_resistance,
            "dc_resistance_sample_count": int(s_train.size),
            "dc_open_resistance_sample_count": int(
                np.sum(diagonal <= minimum_conductance * (1.0 + 1e-12))
            ),
            "dc_mean_conductance_siemens": aggregate_conductance,
            "dc_port_resistances_ohm": {},
            "dc_resistance_pair_means_ohm": {},
            "dc_open_paths": [],
            "dc_open_path_count": 0,
            "dc_open_circuit_applied": False,
            "dc_aggregate_open_circuit_applied": (
                aggregate_conductance < 1.0 / float(open_threshold_ohm)
            ),
        }
    )
    model.metadata = dict(metadata)
    return model, history, metadata


def train_dc_full_y_model(
    train_blocks: Sequence[MDIFBlock],
    verify_blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    hidden_layers: Sequence[int],
    activation: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
    loss_interval: int = 1,
    progress_interval: int = 25,
    progress_label: str = "DC fit",
    z0: float = 50.0,
    open_threshold_ohm: float = DEFAULT_DC_OPEN_THRESHOLD_OHM,
    open_resistance_ohm: float = DEFAULT_DC_OPEN_RESISTANCE_OHM,
) -> tuple[DCConductanceModel, list[dict[str, float]], dict[str, object]]:
    """Fit every ordered real DC Y entry when no branch topology is requested."""

    x_train, y_train, metadata = extract_dc_full_y_samples(
        train_blocks,
        parameter_names,
        labels,
        z0=z0,
    )
    if verify_blocks:
        x_verify, y_verify, verify_metadata = extract_dc_full_y_samples(
            verify_blocks,
            parameter_names,
            labels,
            z0=z0,
            require_every_block=False,
        )
        if x_verify.shape[0] == 0:
            x_verify = None
            y_verify = None
    else:
        x_verify = None
        y_verify = None
        verify_metadata = None

    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(y_train)
    constant_output_mask = np.std(y_train, axis=0) < EPS
    mlp = MLP(
        [x_train.shape[1], *list(hidden_layers), y_train.shape[1]],
        activation=activation,
        seed=seed + 101,
    )
    history = mlp.train(
        x_scaler.transform(x_train),
        y_scaler.transform(y_train),
        x_scaler.transform(x_verify) if x_verify is not None else None,
        y_scaler.transform(y_verify) if y_verify is not None else None,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        seed=seed + 103,
        loss_interval=loss_interval,
        progress_callback=make_training_progress_callback(
            progress_label,
            epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    assert y_scaler.std is not None
    y_scaler.std[constant_output_mask] = 0.0
    model = DCConductanceModel(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=labels,
        port_paths=[],
        z0=z0,
        log_conductance_min=-50.0,
        log_conductance_max=50.0,
        representation="full_y_matrix",
    )
    train_prediction = model.predict_matrix_entries(x_train)
    train_error = train_prediction - y_train
    nports = infer_complete_sparameter_ports(labels)

    def matrices_to_s(rows: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _y_matrix_to_s_matrix(row.reshape(nports, nports).astype(complex), z0)
                for row in np.asarray(rows, dtype=float)
            ],
            dtype=complex,
        )

    train_s_error = np.abs(
        matrices_to_s(train_prediction) - matrices_to_s(y_train)
    )
    metadata.update(
        {
            "dc_model_training_samples": int(len(x_train)),
            "dc_model_verification_samples": (
                int(len(x_verify)) if x_verify is not None else 0
            ),
            "dc_model_activation": activation,
            "dc_model_layer_sizes": list(model.mlp.layer_sizes),
            "dc_model_train_y_rmse_siemens": float(
                np.sqrt(np.mean(train_error**2))
            ),
            "dc_model_train_s_rmse": float(
                np.sqrt(np.mean(train_s_error * train_s_error))
            ),
            "dc_model_train_s_max_abs_error": float(np.max(train_s_error)),
            "dc_model_constant_matrix_entries": [
                entry
                for entry, is_constant in zip(
                    metadata["dc_matrix_entries"],
                    constant_output_mask,
                )
                if bool(is_constant)
            ],
            "dc_open_threshold_ohm": float(open_threshold_ohm),
            "dc_open_resistance_ohm": float(open_resistance_ohm),
        }
    )
    if verify_metadata is not None:
        metadata["dc_model_verification_extraction"] = verify_metadata
    if x_verify is not None and y_verify is not None:
        verify_prediction = model.predict_matrix_entries(x_verify)
        verify_error = verify_prediction - y_verify
        verify_s_error = np.abs(
            matrices_to_s(verify_prediction) - matrices_to_s(y_verify)
        )
        metadata.update(
            {
                "dc_model_verify_y_rmse_siemens": float(
                    np.sqrt(np.mean(verify_error**2))
                ),
                "dc_model_verify_s_rmse": float(
                    np.sqrt(np.mean(verify_s_error * verify_s_error))
                ),
                "dc_model_verify_s_max_abs_error": float(np.max(verify_s_error)),
            }
        )

    mean_y = np.mean(y_train, axis=0).reshape(nports, nports)
    minimum_conductance = 1.0 / float(open_resistance_ohm)
    diagonal = np.maximum(np.real(np.diag(mean_y)), minimum_conductance)
    aggregate_conductance = float(np.mean(diagonal))
    equivalent_resistance = min(
        1.0 / max(aggregate_conductance, minimum_conductance),
        float(open_resistance_ohm),
    )
    metadata.update(
        {
            "dc_equivalent_resistance_ohm": equivalent_resistance,
            "dc_equivalent_resistance_raw_mean_ohm": equivalent_resistance,
            "dc_resistance_sample_count": int(y_train.size),
            "dc_open_resistance_sample_count": int(
                np.sum(diagonal <= minimum_conductance * (1.0 + 1e-12))
            ),
            "dc_mean_conductance_siemens": aggregate_conductance,
            # These legacy path summaries stay empty in full-matrix mode. The
            # independently fitted entries in dc_matrix_entries drive stamping.
            "dc_port_resistances_ohm": {},
            "dc_resistance_pair_means_ohm": {},
            "dc_open_paths": [],
            "dc_open_path_count": 0,
            "dc_open_circuit_applied": False,
            "dc_aggregate_open_circuit_applied": (
                aggregate_conductance < 1.0 / float(open_threshold_ohm)
            ),
        }
    )
    model.metadata = dict(metadata)
    return model, history, metadata


def select_dc_export_training_blocks(
    blocks: Sequence[MDIFBlock],
    stored_metadata: dict[str, object],
) -> tuple[list[MDIFBlock], dict[str, object]]:
    """Select only the fitted split from a combined MDIF used during export."""

    split_var = str(stored_metadata.get("split_var") or "dataset")
    split_key = normalize_name(split_var)

    def value_set(key: str, defaults: set[str]) -> set[str]:
        raw = stored_metadata.get(key)
        if isinstance(raw, str):
            values = raw.split(",")
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            return defaults
        parsed = {
            strip_quotes(str(value)).strip().lower()
            for value in values
            if str(value).strip()
        }
        return parsed or defaults

    train_values = value_set("train_values", {"train", "training"})
    verify_values = value_set(
        "verify_values",
        {"verify", "verification", "test", "validation"},
    )
    training_blocks: list[MDIFBlock] = []
    verification_count = 0
    unclassified_count = 0
    for block in blocks:
        raw_value = normalized_mapping_value(block.params, split_key)
        value = strip_quotes(str(raw_value)).strip().lower() if raw_value is not None else ""
        if value in train_values:
            training_blocks.append(block)
        elif value in verify_values:
            verification_count += 1
        else:
            unclassified_count += 1

    if training_blocks:
        selected = training_blocks
        selection = "metadata_training_split"
    elif verification_count:
        raise ValueError(
            f"The DC MDIF contains {verification_count} recognized verification block(s) "
            f"but no {split_var!r} value matching the model's training values "
            f"{sorted(train_values)}. Supply the training MDIF or correct its split VAR."
        )
    else:
        selected = list(blocks)
        selection = "all_blocks_no_recognized_split"

    return selected, {
        "dc_mdif_selection": selection,
        "dc_mdif_split_var": split_var,
        "dc_mdif_train_values": sorted(train_values),
        "dc_mdif_total_block_count": len(blocks),
        "dc_mdif_training_block_count": len(selected),
        "dc_mdif_excluded_verification_block_count": verification_count,
        "dc_mdif_excluded_unclassified_block_count": (
            unclassified_count if training_blocks else 0
        ),
    }


def validate_dc_model_against_mdif(
    model: DCConductanceModel,
    blocks: Sequence[MDIFBlock],
) -> dict[str, object]:
    """Compare a saved/export DC model directly with exact-DC MDIF rows."""

    if model.representation == "full_s_matrix":
        x_values, target_outputs, extraction = extract_dc_full_s_samples(
            blocks,
            model.parameter_names,
            model.sparam_labels,
            z0=model.z0,
        )
        predicted_outputs = model.predict_s_components(x_values)
    elif model.representation == "full_y_matrix":
        x_values, target_outputs, extraction = extract_dc_full_y_samples(
            blocks,
            model.parameter_names,
            model.sparam_labels,
            z0=model.z0,
        )
        predicted_outputs = model.predict_matrix_entries(x_values)
    else:
        x_values, target_outputs, extraction = extract_dc_conductance_samples(
            blocks,
            model.parameter_names,
            model.sparam_labels,
            z0=model.z0,
            port_paths=model.port_paths,
            open_threshold_ohm=float(
                model.metadata.get(
                    "dc_open_threshold_ohm",
                    DEFAULT_DC_OPEN_THRESHOLD_OHM,
                )
            ),
            open_resistance_ohm=float(
                model.metadata.get(
                    "dc_open_resistance_ohm",
                    DEFAULT_DC_OPEN_RESISTANCE_OHM,
                )
            ),
        )
        predicted_outputs = model.predict_conductances(x_values)
    output_error = predicted_outputs - target_outputs

    direct_errors: list[float] = []
    passivity_limit = 1.0 + float(
        model.metadata.get(
            "dc_passivity_tolerance",
            DEFAULT_DC_PASSIVITY_TOLERANCE,
        )
    )
    nports = infer_complete_sparameter_ports(model.sparam_labels)
    usable_positions = {
        int(value) for value in extraction["dc_usable_block_positions"]
    }
    usable_blocks = [
        block
        for block in blocks
        if int(block.source_index) + 1 in usable_positions
    ]
    for block, parameter_row in zip(usable_blocks, x_values):
        predicted = model.predict_s_values(parameter_row)
        for raw_index in np.flatnonzero(
            np.asarray(block.freq_hz, dtype=float) == 0.0
        ):
            frequency_index = int(raw_index)
            s_matrix = _block_s_matrix(
                block,
                model.sparam_labels,
                frequency_index,
                nports,
            )
            if not np.all(np.isfinite(s_matrix)):
                continue
            try:
                max_sigma = float(np.max(np.linalg.svd(s_matrix, compute_uv=False)))
            except np.linalg.LinAlgError:
                continue
            if not math.isfinite(max_sigma) or max_sigma > passivity_limit:
                continue
            measured = np.asarray(
                [
                    complex(block.sparams[label][frequency_index])
                    for label in model.sparam_labels
                ],
                dtype=complex,
            )
            direct_errors.extend(np.abs(predicted - measured).tolist())
    if not direct_errors:
        raise ValueError("No usable passive exact-DC rows remain for export validation")
    direct_error_array = np.asarray(direct_errors, dtype=float)
    return {
        "dc_mdif_validation_input_block_count": len(blocks),
        "dc_mdif_validation_block_count": len(usable_blocks),
        "dc_mdif_excluded_unusable_block_count": extraction[
            "dc_unusable_block_count"
        ],
        "dc_mdif_excluded_unusable_block_positions": extraction[
            "dc_unusable_block_positions"
        ],
        "dc_mdif_model_output_rmse": float(
            np.sqrt(np.mean(output_error**2))
        ),
        "dc_mdif_model_conductance_rmse_siemens": (
            float(np.sqrt(np.mean(output_error**2)))
            if model.representation == "path_conductance"
            else None
        ),
        "dc_mdif_model_y_rmse_siemens": (
            float(np.sqrt(np.mean(output_error**2)))
            if model.representation == "full_y_matrix"
            else None
        ),
        "dc_mdif_model_s_component_rmse": (
            float(np.sqrt(np.mean(output_error**2)))
            if model.representation == "full_s_matrix"
            else None
        ),
        "dc_mdif_model_s_rmse": float(
            np.sqrt(np.mean(direct_error_array**2))
        ),
        "dc_mdif_model_s_max_abs_error": float(np.max(direct_error_array)),
        "dc_mdif_topology_s_rmse": extraction["dc_topology_s_rmse"],
        "dc_mdif_topology_s_max_abs_error": extraction[
            "dc_topology_s_max_abs_error"
        ],
    }


def resolve_export_dc_conductance_model(
    saved_model: DCConductanceModel | None,
    stored_metadata: dict[str, object],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    *,
    dc_mdif: str | Path | None,
    z0: float,
    port_paths: object,
    open_threshold_ohm: float,
    open_resistance_ohm: float,
    activation: str,
    hidden_layers: Sequence[int],
) -> tuple[DCConductanceModel | None, dict[str, object]]:
    """Resolve dynamic DC for export without changing the fitted RF model.

    A saved geometry-dependent DC model is used by default. If ``--dc-mdif``
    is supplied, it is first checked against that saved model. A mismatching or
    legacy model triggers a DC-only fit from the supplied MDIF; RF weights and
    poles are never retrained.
    """

    requested_paths = port_paths
    if requested_paths is None and dc_mdif is None:
        requested_paths = (
            saved_model.port_paths
            if saved_model is not None
            else stored_metadata.get("dc_port_paths")
        )
    if saved_model is not None and dc_mdif is None:
        if (
            port_paths is None
            and saved_model.representation != "full_s_matrix"
            and (
                saved_model.representation == "full_y_matrix"
                or stored_metadata.get("dc_port_paths_explicit") is False
            )
        ):
            raise ValueError(
                "The saved DC model uses an older lossy DC representation. "
                "Supply --dc-mdif to upgrade it to the full ordered complex-S model, "
                "or pass --dc-port-paths explicitly to retain a restricted graph."
            )
        nports = infer_complete_sparameter_ports(labels)
        if port_paths is not None:
            requested = [item[2] for item in parse_dc_port_paths(port_paths, nports)]
            if requested != saved_model.port_paths:
                raise ValueError(
                    "--dc-port-paths does not match the saved geometry-dependent DC "
                    "model. Supply --dc-mdif to fit the requested DC topology without "
                    "refitting RF."
                )
        metadata = {
            key: value
            for key, value in stored_metadata.items()
            if str(key).startswith("dc_")
        }
        return saved_model, metadata

    if dc_mdif is not None:
        dc_path = Path(dc_mdif)
        all_blocks = read_mdif(dc_path)
        blocks, selection_metadata = select_dc_export_training_blocks(
            all_blocks,
            stored_metadata,
        )
        if saved_model is not None:
            if port_paths is None:
                saved_representation_matches = (
                    saved_model.representation == "full_s_matrix"
                )
            else:
                requested_canonical = [
                    item[2]
                    for item in parse_dc_port_paths(
                        requested_paths,
                        infer_complete_sparameter_ports(labels),
                    )
                ]
                saved_representation_matches = (
                    saved_model.representation == "path_conductance"
                    and list(saved_model.port_paths) == requested_canonical
                )
            if saved_representation_matches:
                validation = validate_dc_model_against_mdif(saved_model, blocks)
                if float(validation["dc_mdif_model_s_max_abs_error"]) <= 1.0e-4:
                    metadata = {
                        key: value
                        for key, value in stored_metadata.items()
                        if str(key).startswith("dc_")
                    }
                    metadata.update(validation)
                    metadata.update(selection_metadata)
                    metadata["dc_resistance_source_file"] = str(dc_path)
                    metadata["dc_mdif_action"] = "validated_saved_dc_model"
                    metadata["dc_mdif_match_within_tolerance"] = True
                    return saved_model, metadata

        resolved_hidden_layers = [int(value) for value in hidden_layers]
        if not resolved_hidden_layers:
            resolved_hidden_layers = [32, 32]
        if requested_paths is None:
            # The automatic model carries two components for every ordered Sij.
            # Give this export-only fit enough capacity and convergence time instead
            # of inheriting an undersized RF/DC architecture from an older model.
            minimum_width = max(
                64,
                4 * len(parameter_names),
                2 * len(labels),
            )
            resolved_width = max(minimum_width, *resolved_hidden_layers)
            resolved_hidden_layers = [resolved_width, resolved_width]
        fitted_model, _, metadata = train_dc_conductance_model(
            blocks,
            [],
            parameter_names,
            labels,
            hidden_layers=resolved_hidden_layers,
            activation=activation,
            epochs=8000 if requested_paths is None else 4000,
            batch_size=min(256, max(1, len(blocks))),
            learning_rate=2.0e-3,
            patience=800 if requested_paths is None else 400,
            seed=1234,
            loss_interval=5,
            progress_interval=0,
            progress_label="DC export fit",
            z0=z0,
            port_paths=requested_paths,
            open_threshold_ohm=open_threshold_ohm,
            open_resistance_ohm=open_resistance_ohm,
        )
        validation = validate_dc_model_against_mdif(fitted_model, blocks)
        within_tolerance = (
            float(validation["dc_mdif_model_s_max_abs_error"])
            <= DEFAULT_DC_EXPORT_S_MATCH_TOLERANCE
        )
        if not within_tolerance and port_paths is not None:
            topology_description = (
                "automatic full ordered complex S-matrix"
                if port_paths is None
                else "declared --dc-port-paths resistor graph"
            )
            raise ValueError(
                "The DC-only export model does not reproduce the supplied exact-DC "
                "S-parameters closely enough: maximum absolute error is "
                f"{float(validation['dc_mdif_model_s_max_abs_error']):.6g}, "
                f"limit {DEFAULT_DC_EXPORT_S_MATCH_TOLERANCE:.6g}. The "
                f"{topology_description} topology error is "
                f"{float(validation['dc_mdif_topology_s_max_abs_error']):.6g}. "
                + (
                    "Fitted matrix components: "
                    + ", ".join(
                        str(value)
                        for value in fitted_model.metadata.get(
                            "dc_matrix_entries",
                            [],
                        )
                    )
                    if fitted_model.representation in {"full_s_matrix", "full_y_matrix"}
                    else "Selected paths: " + ", ".join(fitted_model.port_paths)
                )
                + ". Check the passive zero-Hz data; export was stopped."
            )
        metadata.update(validation)
        metadata["dc_mdif_match_within_tolerance"] = within_tolerance
        if not within_tolerance:
            metadata["dc_mdif_warning"] = (
                "The unrestricted full-complex-S DC model exceeded the preferred "
                f"maximum S-error {DEFAULT_DC_EXPORT_S_MATCH_TOLERANCE:g}, but export "
                "continued because every Sij component is represented and there is no "
                "topology/projection error. The measured fit error is recorded in the "
                "manifest."
            )
        metadata.update(selection_metadata)
        metadata["dc_resistance_source_file"] = str(dc_path)
        metadata["dc_model_fitted_during_export"] = True
        metadata["dc_mdif_action"] = "fitted_dc_only_model"
        fitted_model.metadata = dict(metadata)
        return fitted_model, metadata

    metadata = resolve_export_dc_metadata(
        stored_metadata,
        labels,
        dc_mdif=None,
        z0=z0,
        open_threshold_ohm=open_threshold_ohm,
        open_resistance_ohm=open_resistance_ohm,
        port_paths=port_paths,
    )
    return None, metadata


def mse(
    pred: np.ndarray | None,
    truth: np.ndarray | None,
    output_weights: np.ndarray | None = None,
    sample_weights: np.ndarray | None = None,
) -> float:
    if pred is None or truth is None:
        return float("nan")
    err2 = (pred - truth) ** 2
    if output_weights is not None:
        err2 = err2 * np.asarray(output_weights, dtype=float)[None, :]
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        if weights.shape != (err2.shape[0],):
            raise ValueError(
                f"Expected {err2.shape[0]} sample weights, got {weights.shape}"
            )
        err2 = err2 * weights[:, None]
    return float(np.mean(err2))


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fit_terminal_status_line(text: str, columns: int) -> str:
    """Keep a redrawable status below the terminal's automatic-wrap width."""

    limit = max(1, int(columns) - 1)
    if len(text) <= limit:
        return text
    if limit <= 3:
        return "." * limit
    return text[: limit - 3].rstrip() + "..."


def terminal_status_line(text: str, stream: object | None = None) -> str:
    """Fit a status line to the attached terminal, with a stable fallback."""

    output = stream if stream is not None else sys.stderr
    fileno = getattr(output, "fileno", None)
    try:
        columns = os.get_terminal_size(fileno()).columns if fileno else 120
    except (OSError, TypeError, ValueError):
        columns = 120
    return fit_terminal_status_line(text, columns)


def progress_interval_from_args(args: argparse.Namespace, default: int = 25) -> int:
    raw_value = getattr(args, "progress_interval", default)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"--progress-interval must be an integer, got {raw_value!r}") from None


def make_training_progress_callback(
    label: str,
    epochs: int,
    progress_interval: int | None,
) -> Callable[[dict[str, object]], None] | None:
    if progress_interval is None or int(progress_interval) <= 0:
        return None
    total_epochs = max(1, int(epochs))
    start_time = time.monotonic()

    def number_text(value: object) -> str | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        return f"{numeric:.6g}"

    def callback(event: dict[str, object]) -> None:
        epoch = int(event.get("epoch") or 0)
        pct = min(100.0, max(0.0, 100.0 * epoch / total_epochs))
        is_final = bool(event.get("stopped")) or epoch >= total_epochs
        parts = [
            f"{label}: epoch {epoch}/{total_epochs}",
            f"({pct:.1f}%, elapsed {format_duration(time.monotonic() - start_time)})",
        ]
        train_loss = number_text(event.get("train_loss"))
        val_loss = number_text(event.get("val_loss"))
        best_val = number_text(event.get("best_val"))
        if train_loss is not None:
            parts.append(f"train_loss={train_loss}")
        if val_loss is not None:
            parts.append(f"val_loss={val_loss}")
        if best_val is not None:
            best_epoch = int(event.get("best_epoch") or 0)
            parts.append(f"best_val={best_val}@{best_epoch}")
        if event.get("stopped"):
            parts.append("early_stop")
        if is_final:
            sys.stderr.write("\r\033[2K")
            sys.stderr.flush()
            return
        sys.stderr.write("\r\033[2K" + terminal_status_line(" ".join(parts)))
        sys.stderr.flush()

    return callback




def infer_nports(labels: Sequence[str]) -> int | None:
    ports = set()
    for label in labels:
        match = re.match(r"S(\d+)(\d+)$", label)
        if not match:
            return None
        ports.add(int(match.group(1)))
        ports.add(int(match.group(2)))
    nports = max(ports) if ports else None
    if nports is None:
        return None
    expected = {f"S{i}{j}" for i in range(1, nports + 1) for j in range(1, nports + 1)}
    if set(labels) != expected:
        return None
    return nports


def passivity_summary(blocks: Sequence[MDIFBlock], labels: Sequence[str]) -> dict[str, float | int | None]:
    nports = infer_nports(labels)
    if nports is None:
        return {"nports": None, "max_singular_value": None, "violating_points": None}
    max_sigma = 0.0
    violating = 0
    for block in blocks:
        for idx in range(len(block.freq_hz)):
            matrix = np.zeros((nports, nports), dtype=complex)
            for label in labels:
                match = re.match(r"S(\d+)(\d+)$", label)
                assert match
                row = int(match.group(1)) - 1
                col = int(match.group(2)) - 1
                matrix[row, col] = block.sparams[label][idx]
            sigma = float(np.linalg.svd(matrix, compute_uv=False)[0])
            max_sigma = max(max_sigma, sigma)
            if sigma > 1.0 + 1e-6:
                violating += 1
    return {
        "nports": nports,
        "max_singular_value": max_sigma,
        "violating_points": violating,
    }


def evm_values(abs_error: np.ndarray, truth_magnitude: np.ndarray) -> tuple[float | None, float | None, float | None]:
    rmse_abs = float(np.sqrt(np.mean(abs_error**2)))
    ref_rms = float(np.sqrt(np.mean(truth_magnitude**2)))
    if ref_rms <= EPS:
        return None, None, None
    evm_rms = rmse_abs / ref_rms
    evm_pct = 100.0 * evm_rms
    evm_db = 20.0 * math.log10(max(evm_rms, EPS))
    return evm_rms, evm_pct, evm_db


def weighted_evm_values(
    abs_error: np.ndarray,
    truth_magnitude: np.ndarray,
    weights: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    numerator = float(np.sum(weights * abs_error**2))
    denominator = float(np.sum(weights * truth_magnitude**2))
    if denominator <= EPS:
        return None, None, None
    evm_rms = math.sqrt(max(numerator, 0.0) / denominator)
    evm_pct = 100.0 * evm_rms
    evm_db = 20.0 * math.log10(max(evm_rms, EPS))
    return evm_rms, evm_pct, evm_db


def verification_metrics(
    truth_blocks: Sequence[MDIFBlock],
    pred_blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    parameter_names: Sequence[str],
    sparam_weights: dict[str, float] | None = None,
    frequency_weights: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    label_weights = sparam_weights or {label: 1.0 for label in labels}
    normalized_label_weights = normalize_sparam_weights(labels, label_weights)
    weight_mean = sparam_weight_mean(labels, label_weights)
    rows: list[dict[str, object]] = []
    all_abs_errors = []
    all_truth_magnitudes = []
    all_db_errors = []
    all_weights = []
    all_normalized_weights = []
    all_normalized_db_weights = []
    for truth, pred in zip(truth_blocks, pred_blocks):
        raw_frequency_weights = frequency_weights_for_values(
            truth.freq_hz,
            frequency_weights,
            require_all_rules_match=False,
        )
        normalized_frequency_weights, block_frequency_weight_mean = (
            normalize_frequency_weights(raw_frequency_weights)
        )
        base: dict[str, object] = {"source_index": truth.source_index}
        for name in parameter_names:
            base[name] = truth.params.get(name, "")
        for label in labels:
            weight = float(label_weights.get(label, 1.0))
            normalized_weight = float(normalized_label_weights.get(label, 1.0))
            err = pred.sparams[label] - truth.sparams[label]
            abs_err = np.abs(err)
            pred_mag = np.abs(pred.sparams[label])
            truth_mag = np.abs(truth.sparams[label])
            db_mask = (pred_mag > DB_MAG_FLOOR) & (truth_mag > DB_MAG_FLOOR)
            db_err = (
                20 * np.log10(pred_mag[db_mask]) - 20 * np.log10(truth_mag[db_mask])
                if np.any(db_mask)
                else np.asarray([], dtype=float)
            )
            all_abs_errors.append(abs_err)
            all_truth_magnitudes.append(truth_mag)
            all_weights.append(raw_frequency_weights * weight)
            all_normalized_weights.append(
                normalized_frequency_weights * normalized_weight
            )
            if db_err.size:
                all_db_errors.append(db_err)
                all_normalized_db_weights.append(
                    normalized_frequency_weights[db_mask] * normalized_weight
                )
                rmse_db: float | None = float(np.sqrt(np.mean(db_err**2)))
                max_abs_db: float | None = float(np.max(np.abs(db_err)))
            else:
                rmse_db = None
                max_abs_db = None
            evm_rms, evm_pct, evm_db = evm_values(abs_err, truth_mag)
            rows.append(
                {
                    **base,
                    "sparam": label,
                    "sparam_weight": weight,
                    "normalized_sparam_weight": normalized_weight,
                    "frequency_weight_mean": block_frequency_weight_mean,
                    "frequency_weight_min": float(np.min(raw_frequency_weights)),
                    "frequency_weight_max": float(np.max(raw_frequency_weights)),
                    "rmse_abs": float(np.sqrt(np.mean(abs_err**2))),
                    "mean_abs": float(np.mean(abs_err)),
                    "max_abs": float(np.max(abs_err)),
                    "evm_rms": evm_rms,
                    "evm_pct": evm_pct,
                    "evm_db": evm_db,
                    "rmse_db": rmse_db,
                    "max_abs_db": max_abs_db,
                }
            )
    if all_abs_errors:
        abs_concat = np.concatenate(all_abs_errors)
        truth_concat = np.concatenate(all_truth_magnitudes)
        weights_concat = np.concatenate(all_weights)
        normalized_weights = np.concatenate(all_normalized_weights)
        db_concat = np.concatenate(all_db_errors) if all_db_errors else np.asarray([])
        normalized_db_weights = (
            np.concatenate(all_normalized_db_weights)
            if all_normalized_db_weights
            else np.asarray([])
        )
        evm_rms, evm_pct, evm_db = evm_values(abs_concat, truth_concat)
        weighted_evm_rms, weighted_evm_pct, weighted_evm_db = weighted_evm_values(
            abs_concat,
            truth_concat,
            weights_concat,
        )
        summary: dict[str, object] = {
            "rmse_abs": float(np.sqrt(np.mean(abs_concat**2))),
            "mean_abs": float(np.mean(abs_concat)),
            "max_abs": float(np.max(abs_concat)),
            "evm_rms": evm_rms,
            "evm_pct": evm_pct,
            "evm_db": evm_db,
            "weighted_rmse_abs": float(np.sqrt(np.mean(normalized_weights * abs_concat**2))),
            "weighted_mean_abs": float(np.mean(normalized_weights * abs_concat)),
            "weighted_max_abs": float(np.max(normalized_weights * abs_concat)),
            "weighted_evm_rms": weighted_evm_rms,
            "weighted_evm_pct": weighted_evm_pct,
            "weighted_evm_db": weighted_evm_db,
            "rmse_db": float(np.sqrt(np.mean(db_concat**2))) if db_concat.size else None,
            "max_abs_db": float(np.max(np.abs(db_concat))) if db_concat.size else None,
            "weighted_rmse_db": (
                float(np.sqrt(np.mean(normalized_db_weights * db_concat**2)))
                if db_concat.size
                else None
            ),
            "weighted_max_abs_db": (
                float(np.max(normalized_db_weights * np.abs(db_concat)))
                if db_concat.size
                else None
            ),
            "db_magnitude_floor": DB_MAG_FLOOR,
            "evm_definition": "sqrt(mean(|pred-truth|^2) / mean(|truth|^2))",
            "weighted_evm_definition": "sqrt(sum(weight[Sij]*weight[f]*|pred-truth|^2) / sum(weight[Sij]*weight[f]*|truth|^2))",
            "sparam_weights": {label: float(label_weights.get(label, 1.0)) for label in labels},
            "normalized_sparam_weights": normalized_label_weights,
            "sparam_weight_mean": weight_mean,
            "sparam_weight_normalization": "Raw S-parameter weights are divided by their mean before training and scale-sensitive weighted RMSE/MAE metrics, so the average normalized weight is 1.0.",
            "frequency_weights": frequency_weights,
            "frequency_weight_normalization": "Frequency weights are normalized to mean 1 over each verification block before scale-sensitive weighted metrics are calculated.",
        }
    else:
        summary = {
            "rmse_abs": None,
            "mean_abs": None,
            "max_abs": None,
            "evm_rms": None,
            "evm_pct": None,
            "evm_db": None,
            "weighted_rmse_abs": None,
            "weighted_mean_abs": None,
            "weighted_max_abs": None,
            "weighted_evm_rms": None,
            "weighted_evm_pct": None,
            "weighted_evm_db": None,
            "rmse_db": None,
            "max_abs_db": None,
            "weighted_rmse_db": None,
            "weighted_max_abs_db": None,
        }
    summary["passivity"] = passivity_summary(pred_blocks, labels)
    return rows, summary


def pdf_escape(text: object) -> str:
    cleaned = str(text).encode("latin-1", errors="replace").decode("latin-1")
    return cleaned.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return safe or "plot"


def expanded_range(values: Sequence[np.ndarray]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    concat = np.concatenate([np.asarray(v, dtype=float).ravel() for v in values])
    concat = concat[np.isfinite(concat)]
    if concat.size == 0:
        return 0.0, 1.0
    lo = float(np.min(concat))
    hi = float(np.max(concat))
    if abs(hi - lo) < EPS:
        pad = max(abs(hi) * 0.05, 1.0)
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def plot_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    left, top, width, height = rect
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    x_den = max(x_hi - x_lo, EPS)
    y_den = max(y_hi - y_lo, EPS)
    points: list[tuple[float, float]] = []
    for x_val, y_val in zip(x_values, y_values):
        if not np.isfinite(x_val) or not np.isfinite(y_val):
            continue
        px = left + width * (float(x_val) - x_lo) / x_den
        py = top + height * (1.0 - (float(y_val) - y_lo) / y_den)
        points.append((px, py))
    return points


class PdfCanvas:
    def __init__(self, width: float = 930.0, height: float = 625.0) -> None:
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def y(self, top_y: float) -> float:
        return self.height - top_y

    def color(self, stroke: tuple[float, float, float] | None = None, fill: tuple[float, float, float] | None = None) -> None:
        if stroke is not None:
            self.commands.append(f"{stroke[0]:.4f} {stroke[1]:.4f} {stroke[2]:.4f} RG")
        if fill is not None:
            self.commands.append(f"{fill[0]:.4f} {fill[1]:.4f} {fill[2]:.4f} rg")

    def text(
        self,
        x: float,
        top_y: float,
        text: object,
        size: float = 12.0,
        font: str = "F1",
        align: str = "left",
        rotate: float = 0.0,
        fill: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    ) -> None:
        escaped = pdf_escape(text)
        estimated_width = 0.5 * size * len(str(text))
        y = self.y(top_y)
        x_adjusted = x
        y_adjusted = y
        align_offset = 0.0
        if align == "center":
            align_offset = estimated_width / 2.0
        elif align == "right":
            align_offset = estimated_width
        if fill is not None:
            self.color(fill=fill)
        if rotate:
            angle = math.radians(rotate)
            c = math.cos(angle)
            s = math.sin(angle)
            x_adjusted -= c * align_offset
            y_adjusted -= s * align_offset
            self.commands.append(
                f"BT /{font} {size:.2f} Tf {c:.6f} {s:.6f} {-s:.6f} {c:.6f} {x_adjusted:.2f} {y_adjusted:.2f} Tm ({escaped}) Tj ET"
            )
        else:
            x_adjusted -= align_offset
            self.commands.append(f"BT /{font} {size:.2f} Tf {x_adjusted:.2f} {y_adjusted:.2f} Td ({escaped}) Tj ET")

    def line(
        self,
        x1: float,
        top_y1: float,
        x2: float,
        top_y2: float,
        color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        self.color(stroke=color)
        self.commands.append(f"{width:.2f} w")
        self.commands.append(f"{dash or '[] 0'} d")
        self.commands.append(f"{x1:.2f} {self.y(top_y1):.2f} m {x2:.2f} {self.y(top_y2):.2f} l S")
        if dash:
            self.commands.append("[] 0 d")

    def rect(
        self,
        x: float,
        top_y: float,
        width: float,
        height: float,
        stroke: tuple[float, float, float] | None = None,
        fill: tuple[float, float, float] | None = None,
        line_width: float = 1.0,
    ) -> None:
        self.color(stroke=stroke, fill=fill)
        self.commands.append(f"{line_width:.2f} w")
        op = "B" if stroke and fill else "S" if stroke else "f"
        self.commands.append(f"{x:.2f} {self.y(top_y + height):.2f} {width:.2f} {height:.2f} re {op}")

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: tuple[float, float, float],
        width: float = 2.0,
        dash: str | None = None,
    ) -> None:
        clean = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
        if len(clean) < 2:
            return
        self.color(stroke=color)
        self.commands.append(f"{width:.2f} w")
        self.commands.append(f"{dash or '[] 0'} d")
        first_x, first_y = clean[0]
        pieces = [f"{first_x:.2f} {self.y(first_y):.2f} m"]
        for x, y in clean[1:]:
            pieces.append(f"{x:.2f} {self.y(y):.2f} l")
        pieces.append("S")
        self.commands.append(" ".join(pieces))
        if dash:
            self.commands.append("[] 0 d")

    def save(self, path: Path) -> None:
        content = canvas_content_bytes(self)
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] "
                f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>"
            ).encode("latin-1"),
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for idx, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{idx} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref_offset = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        out.extend(
            (
                f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(bytes(out))


def canvas_content_bytes(canvas: PdfCanvas) -> bytes:
    commands = ["1 1 1 rg", f"0 0 {canvas.width:.2f} {canvas.height:.2f} re f"]
    commands.extend(canvas.commands)
    return "\n".join(commands).encode("latin-1", errors="replace")


def save_pdf_pages(path: Path, canvases: Sequence[PdfCanvas]) -> None:
    if not canvases:
        raise ValueError("Cannot write a PDF with no pages")
    n_pages = len(canvases)
    font1_id = 3 + 2 * n_pages
    font2_id = font1_id + 1
    kids = " ".join(f"{3 + 2 * idx} 0 R" for idx in range(n_pages))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("latin-1"),
    ]
    for idx, canvas in enumerate(canvases):
        page_id = 3 + 2 * idx
        content_id = page_id + 1
        content = canvas_content_bytes(canvas)
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {canvas.width:.0f} {canvas.height:.0f}] "
                f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{idx} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref_offset = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(out))


def pdf_ticks(
    canvas: PdfCanvas,
    rect: tuple[float, float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    y_label: str,
    x_label: str | None = None,
    tick_size: float = 12.0,
    label_size: float = 13.0,
    x_label_offset: float = 44.0,
    y_label_offset: float = 62.0,
) -> None:
    left, top, width, height = rect
    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    canvas.rect(left, top, width, height, stroke=(0.2, 0.2, 0.2), fill=(1.0, 1.0, 1.0))
    for frac in np.linspace(0.0, 1.0, 5):
        x = left + frac * width
        value = x_lo + frac * (x_hi - x_lo)
        canvas.line(x, top + height, x, top + height + 5, color=(0.2, 0.2, 0.2))
        canvas.text(x, top + height + 22, f"{value:.4g}", size=tick_size, align="center")
    for frac in np.linspace(0.0, 1.0, 5):
        y = top + height * (1.0 - frac)
        value = y_lo + frac * (y_hi - y_lo)
        canvas.line(left - 5, y, left, y, color=(0.2, 0.2, 0.2))
        canvas.text(left - 10, y + 4, f"{value:.4g}", size=tick_size, align="right")
        if 0.0 < frac < 1.0:
            canvas.line(left, y, left + width, y, color=(0.9137, 0.9255, 0.9373))
    canvas.text(left - y_label_offset, top + height / 2, y_label, size=label_size, align="center", rotate=90)
    if x_label:
        canvas.text(left + width / 2, top + height + x_label_offset, x_label, size=label_size, align="center")


def clipped_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def wrapped_text_lines(text: object, max_chars: int) -> list[str]:
    raw = str(text)
    lines: list[str] = []
    for part in raw.splitlines() or [""]:
        wrapped = textwrap.wrap(
            part,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def draw_wrapped_centered_text(
    canvas: PdfCanvas,
    lines: Sequence[str],
    x: float,
    top_y: float,
    size: float,
    line_height: float,
    font: str = "F1",
) -> float:
    for idx, line in enumerate(lines):
        canvas.text(x, top_y + idx * line_height, line, size=size, font=font, align="center")
    return top_y + max(len(lines) - 1, 0) * line_height


def complete_sparam_grid(labels: Sequence[str]) -> tuple[int, list[str]] | None:
    nports = infer_nports(labels)
    if nports is None:
        return None
    ordered = [f"S{i}{j}" for i in range(1, nports + 1) for j in range(1, nports + 1)]
    return nports, ordered


def subplot_grid(labels: Sequence[str]) -> tuple[int, int, list[str]]:
    grid = complete_sparam_grid(labels)
    if grid is not None:
        nports, ordered = grid
        return nports, nports, ordered
    cols = int(math.ceil(math.sqrt(len(labels))))
    rows = int(math.ceil(len(labels) / cols))
    return rows, cols, list(labels)


def mag_db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(values), DB_MAG_FLOOR))


def unwrapped_phase_deg(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.angle(values)))


def block_values(block: MDIFBlock, labels: Sequence[str]) -> np.ndarray:
    return np.column_stack([block.sparams[label] for label in labels])


def sparam_matrix_index_arrays(labels: Sequence[str]) -> tuple[int, np.ndarray, np.ndarray]:
    try:
        nports = infer_complete_sparameter_ports(labels)
    except ValueError as exc:
        raise ValueError(f"Y-parameter conversion requires a complete S-matrix: {exc}") from exc
    rows = []
    cols = []
    for label in labels:
        indices = sparam_indices(label)
        if indices is None:
            raise ValueError(f"Label {label!r} is not an S-parameter name")
        row, col = indices
        rows.append(row - 1)
        cols.append(col - 1)
    return nports, np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def s_values_to_y_values(values: np.ndarray, labels: Sequence[str], z0: float) -> np.ndarray:
    nports, rows, cols = sparam_matrix_index_arrays(labels)
    identity = np.eye(nports, dtype=complex)
    smatrix = np.zeros((values.shape[0], nports, nports), dtype=complex)
    smatrix[:, rows, cols] = values
    lhs = identity[None, :, :] + smatrix
    rhs = identity[None, :, :] - smatrix
    try:
        ymatrix = np.swapaxes(
            np.linalg.solve(np.swapaxes(lhs, -1, -2), np.swapaxes(rhs, -1, -2)),
            -1,
            -2,
        ) / z0
    except np.linalg.LinAlgError:
        ymatrix = rhs @ np.linalg.pinv(lhs) / z0
    return ymatrix[:, rows, cols]


def blocks_s_to_y_blocks(
    blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    z0: float,
) -> list[MDIFBlock]:
    if not math.isfinite(z0) or z0 <= 0.0:
        raise ValueError(f"Y-parameter plot reference impedance must be positive, got {z0!r}")
    converted: list[MDIFBlock] = []
    for block in blocks:
        values = block_values(block, labels)
        y_values = s_values_to_y_values(values, labels, z0)
        sparams = dict(block.sparams)
        for idx, label in enumerate(labels):
            sparams[label] = y_values[:, idx]
        converted.append(
            MDIFBlock(
                params=dict(block.params),
                freq_hz=np.asarray(block.freq_hz, dtype=float).copy(),
                sparams=sparams,
                source_index=block.source_index,
            )
        )
    return converted


def response_plot_label(label: str, response_kind: str) -> str:
    kind = str(response_kind or "S").strip().upper()
    if kind and kind != "S" and label.upper().startswith("S"):
        return kind + label[1:]
    return label


def response_magnitude_axis_label(response_kind: str) -> str:
    return f"|{str(response_kind or 'S').strip().upper()}ij| (dB)"


def response_component_axis_label(response_kind: str, component: str) -> str:
    kind = str(response_kind or "S").strip().upper()
    component_name = "Real" if component == "real" else "Imaginary"
    if kind == "Y":
        return f"{component_name}({kind}ij) (siemens)"
    return f"{component_name}({kind}ij)"


def square_plot_rect(rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, top, width, height = rect
    size = min(width, height)
    return (left + (width - size) / 2.0, top + (height - size) / 2.0, size, size)


def complex_points_to_rect(
    values: np.ndarray,
    rect: tuple[float, float, float, float],
    limit: float = 1.08,
) -> list[tuple[float, float]]:
    left, top, width, height = rect
    points: list[tuple[float, float]] = []
    for value in values:
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            continue
        px = left + width * (float(value.real) + limit) / (2.0 * limit)
        py = top + height * (1.0 - (float(value.imag) + limit) / (2.0 * limit))
        points.append((px, py))
    return points


def smith_chart_curves() -> list[tuple[np.ndarray, tuple[float, float, float], float, str | None]]:
    curves: list[tuple[np.ndarray, tuple[float, float, float], float, str | None]] = []
    grid_color = (0.78, 0.80, 0.82)
    axis_color = (0.58, 0.60, 0.62)
    boundary_color = (0.34, 0.36, 0.38)
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    curves.append((np.exp(1j * theta), boundary_color, 1.0, None))

    reactance_values = np.linspace(-20.0, 20.0, 600)
    for resistance in [0.2, 0.5, 1.0, 2.0, 5.0]:
        z_norm = resistance + 1j * reactance_values
        gamma = (z_norm - 1.0) / (z_norm + 1.0)
        curves.append((gamma, grid_color, 0.55, None))

    resistance_values = np.linspace(0.0, 20.0, 600)
    for reactance in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [-1.0, 1.0]:
            z_norm = resistance_values + 1j * sign * reactance
            gamma = (z_norm - 1.0) / (z_norm + 1.0)
            curves.append((gamma, grid_color, 0.55, None))

    curves.append((np.linspace(-1.0, 1.0, 241).astype(complex), axis_color, 0.7, "[2 3] 0"))
    curves.append((1j * np.linspace(-1.0, 1.0, 241), axis_color, 0.7, "[2 3] 0"))
    return curves


def draw_smith_chart_axes(
    canvas: PdfCanvas,
    rect: tuple[float, float, float, float],
) -> None:
    canvas.rect(rect[0], rect[1], rect[2], rect[3], stroke=(0.88, 0.88, 0.88), fill=(1.0, 1.0, 1.0))
    for curve, color, width, dash in smith_chart_curves():
        canvas.polyline(complex_points_to_rect(curve, rect), color=color, width=width, dash=dash)


def draw_smith_subplot(
    canvas: PdfCanvas,
    rect: tuple[float, float, float, float],
    truth_values: np.ndarray,
    pred_values: np.ndarray,
    label: str,
    show_legend: bool,
) -> None:
    left, top, width, _height = rect
    chart_rect = square_plot_rect((rect[0], rect[1] + 4.0, rect[2], max(rect[3] - 10.0, 20.0)))
    canvas.text(left + width / 2, top - 12, label, size=13, font="F2", align="center")
    draw_smith_chart_axes(canvas, chart_rect)
    canvas.polyline(
        complex_points_to_rect(pred_values, chart_rect),
        color=(0.1216, 0.4667, 0.7059),
        width=1.8,
    )
    canvas.polyline(
        complex_points_to_rect(truth_values, chart_rect),
        color=(1.0, 0.498, 0.0549),
        width=1.8,
        dash="[5 3] 0",
    )
    if show_legend:
        draw_legend(canvas, chart_rect, location="upper_left")


def chunks(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[idx : idx + size]) for idx in range(0, len(values), size)]


def case_metrics(
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    include_passivity: bool = True,
) -> dict[str, object]:
    abs_errors = []
    truth_values = []
    mag_errors = []
    phase_errors = []
    max_mag_error_by_label: dict[str, float] = {}
    for label in labels:
        truth_s = truth.sparams[label]
        pred_s = pred.sparams[label]
        abs_err = np.abs(pred_s - truth_s)
        mag_err = mag_db(pred_s) - mag_db(truth_s)
        phase_err = unwrapped_phase_deg(pred_s) - unwrapped_phase_deg(truth_s)
        abs_errors.append(abs_err)
        truth_values.append(np.abs(truth_s))
        mag_errors.append(mag_err)
        phase_errors.append(phase_err)
        max_mag_error_by_label[label] = float(np.max(np.abs(mag_err)))

    abs_concat = np.concatenate(abs_errors)
    truth_concat = np.concatenate(truth_values)
    mag_concat = np.concatenate(mag_errors)
    phase_concat = np.concatenate(phase_errors)
    rmse_abs = float(np.sqrt(np.mean(abs_concat**2)))
    evm_rms, evm_pct, evm_db = evm_values(abs_concat, truth_concat)
    if evm_rms is None or evm_pct is None or evm_db is None:
        evm_rms, evm_pct, evm_db = float("nan"), float("nan"), float("nan")
    worst_label = max(max_mag_error_by_label, key=max_mag_error_by_label.get)
    metrics: dict[str, object] = {
        "rmse_abs": rmse_abs,
        "rel_rmse": evm_rms,
        "evm_rms": evm_rms,
        "evm_pct": evm_pct,
        "evm_db": evm_db,
        "mag_rmse_db": float(np.sqrt(np.mean(mag_concat**2))),
        "phase_rmse_deg": float(np.sqrt(np.mean(phase_concat**2))),
        "max_abs": float(np.max(abs_concat)),
        "max_abs_db": float(np.max(np.abs(mag_concat))),
        "worst_sparam": worst_label,
        "max_mag_error_by_label": max_mag_error_by_label,
    }
    if include_passivity:
        metrics["passivity"] = passivity_summary([pred], labels)
    return metrics


def draw_page_header(
    canvas: PdfCanvas,
    title: str,
    case_name: str,
    metrics: dict[str, object],
    page_label: str,
) -> float:
    title_lines = wrapped_text_lines(title, 108)
    detail_lines = wrapped_text_lines(
        f"{case_name} | EVM={float(metrics['evm_pct']):.4g}%, "
        f"mag RMSE={float(metrics['mag_rmse_db']):.4g} dB",
        122,
    )
    y = draw_wrapped_centered_text(canvas, title_lines, canvas.width / 2, 22, 17, 19, font="F2")
    y = draw_wrapped_centered_text(canvas, detail_lines, canvas.width / 2, y + 22, 14, 17)
    canvas.text(canvas.width / 2, y + 22, page_label, size=15, font="F2", align="center")
    return y + 22


def draw_legend(canvas: PdfCanvas, rect: tuple[float, float, float, float], location: str = "upper_right") -> None:
    left, top, width, _height = rect
    legend_left = left + 10 if location == "upper_left" else left + width - 110
    legend_top = top + 12
    canvas.rect(legend_left - 8, legend_top - 10, 104, 42, stroke=(0.82, 0.82, 0.82), fill=(1.0, 1.0, 1.0))
    canvas.line(legend_left, legend_top, legend_left + 24, legend_top, color=(0.1216, 0.4667, 0.7059), width=1.8)
    canvas.text(legend_left + 30, legend_top + 4, "modeled", size=10)
    canvas.line(legend_left, legend_top + 19, legend_left + 24, legend_top + 19, color=(1.0, 0.498, 0.0549), width=1.8, dash="[5 3] 0")
    canvas.text(legend_left + 30, legend_top + 23, "measured", size=10)


def draw_component_legend(canvas: PdfCanvas, rect: tuple[float, float, float, float]) -> None:
    left, top, _width, _height = rect
    legend_left = left + 10
    legend_top = top + 12
    entries = [
        ("modeled real", (0.1216, 0.4667, 0.7059), None),
        ("measured real", (1.0, 0.498, 0.0549), "[5 3] 0"),
        ("modeled imag", (0.1725, 0.6275, 0.1725), None),
        ("measured imag", (0.8392, 0.1529, 0.1569), "[5 3] 0"),
    ]
    canvas.rect(legend_left - 8, legend_top - 10, 136, 82, stroke=(0.82, 0.82, 0.82), fill=(1.0, 1.0, 1.0))
    for idx, (label, color, dash) in enumerate(entries):
        y = legend_top + 19 * idx
        canvas.line(legend_left, y, legend_left + 24, y, color=color, width=1.8, dash=dash)
        canvas.text(legend_left + 30, y + 4, label, size=9)


def draw_series_subplot(
    canvas: PdfCanvas,
    rect: tuple[float, float, float, float],
    freq_ghz: np.ndarray,
    truth_y: np.ndarray,
    pred_y: np.ndarray,
    label: str,
    y_label: str,
    show_legend: bool,
) -> None:
    left, top, width, _height = rect
    x_range = expanded_range([freq_ghz])
    y_range = expanded_range([truth_y, pred_y])
    canvas.text(left + width / 2, top - 12, label, size=13, font="F2", align="center")
    pdf_ticks(canvas, rect, x_range, y_range, y_label, "Frequency (GHz)")
    canvas.polyline(
        plot_points(freq_ghz, pred_y, x_range, y_range, rect),
        color=(0.1216, 0.4667, 0.7059),
        width=1.8,
    )
    canvas.polyline(
        plot_points(freq_ghz, truth_y, x_range, y_range, rect),
        color=(1.0, 0.498, 0.0549),
        width=1.8,
        dash="[5 3] 0",
    )
    if show_legend:
        draw_legend(canvas, rect)


def draw_grid_page(
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    page_label: str,
    quantity: str,
    response_kind: str = "S",
) -> PdfCanvas:
    rows, cols, ordered = subplot_grid(labels)
    canvas = PdfCanvas(width=1133.0, height=830.0)
    header_bottom = draw_page_header(canvas, title, case_name, metrics, page_label)
    margin_x = 70.0 if cols >= 4 else 86.0
    margin_bottom = 58.0
    grid_top = max(142.0, header_bottom + 48.0)
    gap_x = 52.0 if cols <= 2 else 84.0 if cols >= 4 else 65.0
    if rows <= 2:
        gap_y = 82.0
    elif rows == 3:
        gap_y = 62.0
    else:
        gap_y = 50.0
    plot_w = (canvas.width - 2 * margin_x - gap_x * (cols - 1)) / cols
    plot_h = (canvas.height - grid_top - margin_bottom - gap_y * (rows - 1)) / rows
    freq_ghz = truth.freq_hz / 1e9

    for idx, label in enumerate(ordered):
        if label not in truth.sparams or label not in pred.sparams:
            continue
        row = idx // cols
        col = idx % cols
        rect = (
            margin_x + col * (plot_w + gap_x),
            grid_top + row * (plot_h + gap_y),
            plot_w,
            plot_h,
        )
        if quantity == "magnitude":
            truth_y = mag_db(truth.sparams[label])
            pred_y = mag_db(pred.sparams[label])
            y_label = response_magnitude_axis_label(response_kind)
        elif quantity == "phase":
            truth_y = unwrapped_phase_deg(truth.sparams[label])
            pred_y = unwrapped_phase_deg(pred.sparams[label])
            y_label = "Phase (deg)"
        elif quantity == "real":
            truth_y = np.real(truth.sparams[label])
            pred_y = np.real(pred.sparams[label])
            y_label = response_component_axis_label(response_kind, "real")
        elif quantity == "imag":
            truth_y = np.imag(truth.sparams[label])
            pred_y = np.imag(pred.sparams[label])
            y_label = response_component_axis_label(response_kind, "imag")
        else:
            raise ValueError(f"Unsupported plot quantity {quantity!r}")
        draw_series_subplot(
            canvas,
            rect,
            freq_ghz,
            truth_y,
            pred_y,
            response_plot_label(label, response_kind),
            y_label,
            show_legend=idx == 0,
        )
    return canvas


def draw_smith_grid_pages(
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    response_kind: str = "S",
) -> list[PdfCanvas]:
    _rows, _cols, ordered = subplot_grid(labels)
    present = [label for label in ordered if label in truth.sparams and label in pred.sparams]
    pages: list[PdfCanvas] = []
    groups = chunks(present, 4)
    for page_index, group in enumerate(groups, start=1):
        canvas = PdfCanvas(width=1133.0, height=830.0)
        page_suffix = f" page {page_index}/{len(groups)}" if len(groups) > 1 else ""
        header_bottom = draw_page_header(
            canvas,
            title,
            case_name,
            metrics,
            f"Smith / complex plane{page_suffix}",
        )
        margin_x = 92.0
        margin_bottom = 58.0
        grid_top = max(142.0, header_bottom + 42.0)
        gap_x = 86.0
        gap_y = 74.0
        plot_w = (canvas.width - 2 * margin_x - gap_x) / 2.0
        plot_h = (canvas.height - grid_top - margin_bottom - gap_y) / 2.0

        for idx, label in enumerate(group):
            row = idx // 2
            col = idx % 2
            rect = (
                margin_x + col * (plot_w + gap_x),
                grid_top + row * (plot_h + gap_y),
                plot_w,
                plot_h,
            )
            draw_smith_subplot(
                canvas,
                rect,
                truth.sparams[label],
                pred.sparams[label],
                response_plot_label(label, response_kind),
                show_legend=page_index == 1 and idx == 0,
            )
        pages.append(canvas)
    return pages


def heat_color(value: float, vmin: float, vmax: float) -> tuple[float, float, float]:
    palette = [
        (0.0157, 0.0118, 0.0706),
        (0.2549, 0.0549, 0.4078),
        (0.5922, 0.1686, 0.4549),
        (0.9020, 0.2902, 0.3176),
        (0.9922, 0.6000, 0.3686),
        (0.9882, 0.9961, 0.6431),
    ]
    if vmax <= vmin + EPS:
        t = 1.0
    else:
        t = min(max((value - vmin) / (vmax - vmin), 0.0), 1.0)
    scaled = t * (len(palette) - 1)
    idx = min(int(math.floor(scaled)), len(palette) - 2)
    frac = scaled - idx
    c0 = palette[idx]
    c1 = palette[idx + 1]
    return tuple(c0[channel] * (1.0 - frac) + c1[channel] * frac for channel in range(3))


def draw_error_heatmap(
    canvas: PdfCanvas,
    rect: tuple[float, float, float, float],
    labels: Sequence[str],
    metrics: dict[str, object],
    response_kind: str = "S",
) -> None:
    grid = complete_sparam_grid(labels)
    max_by_label = metrics["max_mag_error_by_label"]
    assert isinstance(max_by_label, dict)
    if grid is None:
        canvas.text(
            rect[0],
            rect[1] - 16,
            f"Max magnitude error by {str(response_kind or 'S').strip().upper()}-parameter (dB)",
            size=15,
            font="F2",
        )
        values = [float(max_by_label[label]) for label in labels]
        y_range = expanded_range([np.asarray(values), np.asarray([0.0])])
        bar_w = rect[2] / max(len(labels), 1)
        for idx, (label, value) in enumerate(zip(labels, values)):
            height = rect[3] * value / max(y_range[1], EPS)
            x = rect[0] + idx * bar_w
            canvas.rect(x + 2, rect[1] + rect[3] - height, max(bar_w - 4, 1), height, fill=heat_color(value, y_range[0], y_range[1]))
            canvas.text(x + bar_w / 2, rect[1] + rect[3] + 16, response_plot_label(label, response_kind), size=9, align="center")
        return

    nports, ordered = grid
    values = np.asarray([float(max_by_label[label]) for label in ordered]).reshape(nports, nports)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    canvas.text(
        rect[0],
        rect[1] - 16,
        f"Max magnitude error by {str(response_kind or 'S').strip().upper()}ij (dB)",
        size=15,
        font="F2",
    )
    cell_w = rect[2] / nports
    cell_h = rect[3] / nports
    for row in range(nports):
        for col in range(nports):
            value = float(values[row, col])
            x = rect[0] + col * cell_w
            y = rect[1] + row * cell_h
            canvas.rect(x, y, cell_w, cell_h, stroke=(1.0, 1.0, 1.0), fill=heat_color(value, vmin, vmax), line_width=0.5)
            text_color = (1.0, 1.0, 1.0) if value < (vmin + vmax) / 2 else (0.0, 0.0, 0.0)
            canvas.text(x + cell_w / 2, y + cell_h / 2 + 4, f"{value:.3g}", size=12, align="center", fill=text_color)
    for idx in range(nports):
        canvas.text(rect[0] - 15, rect[1] + idx * cell_h + cell_h / 2 + 4, str(idx + 1), size=12, align="right")
        canvas.text(rect[0] + idx * cell_w + cell_w / 2, rect[1] + rect[3] + 20, str(idx + 1), size=12, align="center")
    canvas.text(rect[0] - 24, rect[1] + rect[3] / 2, "i", size=13, align="center")
    canvas.text(rect[0] + rect[2] / 2, rect[1] + rect[3] + 30, "j", size=13, align="center")

    bar_left = rect[0] + rect[2] + 22
    steps = 40
    for step in range(steps):
        frac = step / (steps - 1)
        value = vmin + frac * (vmax - vmin)
        color = heat_color(value, vmin, vmax)
        canvas.rect(bar_left, rect[1] + rect[3] * (1.0 - frac), 12, rect[3] / steps + 1, fill=color)
    for frac in np.linspace(0.0, 1.0, 5):
        value = vmin + frac * (vmax - vmin)
        y = rect[1] + rect[3] * (1.0 - frac)
        canvas.line(bar_left + 12, y, bar_left + 17, y, color=(0.0, 0.0, 0.0))
        canvas.text(bar_left + 22, y + 4, f"{value:.3g}", size=11)


def draw_focus_page(
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    response_kind: str = "S",
) -> PdfCanvas:
    canvas = PdfCanvas(width=669.0, height=325.0)
    title_lines = wrapped_text_lines(f"{title}: error focus for {case_name}", 82)
    header_bottom = draw_wrapped_centered_text(
        canvas,
        title_lines,
        canvas.width / 2,
        16,
        14,
        16,
        font="F2",
    )
    grid = complete_sparam_grid(labels)
    nports = grid[0] if grid is not None else 0
    focus_top_offset = max(0.0, header_bottom - 32.0)
    if nports >= 4:
        heat_rect = (40.0, 114.0 + focus_top_offset, 184.0, 174.0 - focus_top_offset)
        focus_rect = (350.0, 114.0 + focus_top_offset, 313.0, 174.0 - focus_top_offset)
    else:
        heat_rect = (45.0, 110.0 + focus_top_offset, 160.0, 160.0 - focus_top_offset)
        focus_rect = (346.0, 96.0 + focus_top_offset, 295.0, 180.0 - focus_top_offset)
    draw_error_heatmap(canvas, heat_rect, labels, metrics, response_kind=response_kind)

    worst_label = str(metrics["worst_sparam"])
    worst_display = response_plot_label(worst_label, response_kind)
    freq_ghz = truth.freq_hz / 1e9
    is_y_response = str(response_kind or "S").strip().upper() == "Y"
    x_range = expanded_range([freq_ghz])
    canvas.text(focus_rect[0] + focus_rect[2] / 2, focus_rect[1] - 18, f"Worst {worst_display}", size=14, font="F2", align="center")
    if is_y_response:
        truth_real = np.real(truth.sparams[worst_label])
        pred_real = np.real(pred.sparams[worst_label])
        truth_imag = np.imag(truth.sparams[worst_label])
        pred_imag = np.imag(pred.sparams[worst_label])
        y_range = expanded_range([truth_real, pred_real, truth_imag, pred_imag])
        pdf_ticks(
            canvas,
            focus_rect,
            x_range,
            y_range,
            "Admittance (siemens)",
            "Frequency (GHz)",
            tick_size=9.0,
            label_size=10.0,
            x_label_offset=27.0,
            y_label_offset=42.0,
        )
        canvas.polyline(plot_points(freq_ghz, pred_real, x_range, y_range, focus_rect), color=(0.1216, 0.4667, 0.7059), width=2.0)
        canvas.polyline(plot_points(freq_ghz, truth_real, x_range, y_range, focus_rect), color=(1.0, 0.498, 0.0549), width=2.0, dash="[5 3] 0")
        canvas.polyline(plot_points(freq_ghz, pred_imag, x_range, y_range, focus_rect), color=(0.1725, 0.6275, 0.1725), width=2.0)
        canvas.polyline(plot_points(freq_ghz, truth_imag, x_range, y_range, focus_rect), color=(0.8392, 0.1529, 0.1569), width=2.0, dash="[5 3] 0")
        draw_component_legend(canvas, focus_rect)
    else:
        truth_mag = mag_db(truth.sparams[worst_label])
        pred_mag = mag_db(pred.sparams[worst_label])
        y_range = expanded_range([truth_mag, pred_mag])
        pdf_ticks(
            canvas,
            focus_rect,
            x_range,
            y_range,
            response_magnitude_axis_label(response_kind),
            "Frequency (GHz)",
            tick_size=9.0,
            label_size=10.0,
            x_label_offset=27.0,
            y_label_offset=42.0,
        )
        canvas.polyline(plot_points(freq_ghz, pred_mag, x_range, y_range, focus_rect), color=(0.1216, 0.4667, 0.7059), width=2.0)
        canvas.polyline(plot_points(freq_ghz, truth_mag, x_range, y_range, focus_rect), color=(1.0, 0.498, 0.0549), width=2.0, dash="[5 3] 0")
        draw_legend(canvas, focus_rect, location="upper_left")
    passivity = metrics.get("passivity")
    max_sigma = None
    if isinstance(passivity, dict):
        max_sigma = passivity.get("max_singular_value")
    info = [
        f"EVM={float(metrics['evm_pct']):.4g}% ({float(metrics['evm_db']):.4g} dB)",
        f"mag RMSE={float(metrics['mag_rmse_db']):.4g} dB",
        f"phase RMSE={float(metrics['phase_rmse_deg']):.4g} deg",
    ]
    if max_sigma is not None:
        passive = bool(float(max_sigma) <= 1.0 + 1e-6)
        info.extend(
            [
                f"passive={passive}",
                f"max sigma={float(max_sigma):.4g}",
            ]
        )
    info_start = focus_rect[1] + (124.0 if nports >= 4 else 112.0)
    for idx, line in enumerate(info):
        canvas.text(focus_rect[0] + 12, info_start + 12 * idx, line, size=9)
    return canvas


def load_matplotlib_modules() -> tuple[object, object] | None:
    try:
        cache_root = Path(tempfile.gettempdir()) / "ads_surrogate_matplotlib"
        cache_root.mkdir(parents=True, exist_ok=True)
        mpl_config = cache_root / "config"
        xdg_cache = cache_root / "xdg"
        mpl_config.mkdir(parents=True, exist_ok=True)
        xdg_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        return None
    return plt, PdfPages


def matplotlib_header(title: str, case_name: str, metrics: dict[str, object], page_label: str) -> str:
    lines = []
    lines.extend(wrapped_text_lines(title, 118))
    lines.extend(
        wrapped_text_lines(
            f"{case_name} | EVM={float(metrics['evm_pct']):.4g}%, "
            f"mag RMSE={float(metrics['mag_rmse_db']):.4g} dB",
            118,
        )
    )
    lines.append(page_label)
    return "\n".join(lines)


def compact_plot_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ",".join(compact_plot_value(item) for item in value)
    return str(value)


def model_settings_title(
    model_kind: str,
    config: dict[str, object],
    label: object | None = None,
) -> str:
    display_label = str(label or model_kind).strip() or model_kind
    parts = [display_label]
    fields = [
        ("freq_transform", "ft"),
        ("output_domain", "out"),
        ("target_z0", "z0"),
        ("mode", "mode"),
        ("include_coarse_input", "coarse_in"),
        ("order", "order"),
        ("pole_damping", "damp"),
        ("ridge", "ridge"),
        ("hidden_layers", "layers"),
        ("activation", "act"),
        ("learning_rate", "lr"),
        ("batch_size", "bs"),
        ("epochs", "ep"),
        ("patience", "pat"),
        ("seed", "seed"),
    ]
    for key, short_name in fields:
        if key not in config:
            continue
        value = config.get(key)
        if value is None or value == "":
            continue
        parts.append(f"{short_name}={compact_plot_value(value)}")
    return " | ".join(parts)


def matplotlib_grid_page(
    plt: object,
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    page_label: str,
    quantity: str,
    response_kind: str = "S",
) -> object:
    rows, cols, ordered = subplot_grid(labels)
    fig, axes = plt.subplots(rows, cols, figsize=(15.74, 11.53), squeeze=False)
    header = matplotlib_header(title, case_name, metrics, page_label)
    header_lines = header.count("\n") + 1
    top = max(0.58, 0.82 - 0.028 * max(header_lines - 3, 0))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.06, top=top, wspace=0.30, hspace=0.58)
    fig.suptitle(header, fontsize=12, y=0.985)
    freq_ghz = truth.freq_hz / 1e9

    for idx, ax in enumerate(axes.ravel()):
        if idx >= len(ordered):
            ax.axis("off")
            continue
        label = ordered[idx]
        if label not in truth.sparams or label not in pred.sparams:
            ax.axis("off")
            continue
        if quantity == "magnitude":
            truth_y = mag_db(truth.sparams[label])
            pred_y = mag_db(pred.sparams[label])
            y_label = response_magnitude_axis_label(response_kind)
        elif quantity == "phase":
            truth_y = unwrapped_phase_deg(truth.sparams[label])
            pred_y = unwrapped_phase_deg(pred.sparams[label])
            y_label = "Phase (deg)"
        elif quantity == "real":
            truth_y = np.real(truth.sparams[label])
            pred_y = np.real(pred.sparams[label])
            y_label = response_component_axis_label(response_kind, "real")
        elif quantity == "imag":
            truth_y = np.imag(truth.sparams[label])
            pred_y = np.imag(pred.sparams[label])
            y_label = response_component_axis_label(response_kind, "imag")
        else:
            raise ValueError(f"Unsupported plot quantity {quantity!r}")
        ax.plot(freq_ghz, pred_y, color="#1f77b4", linewidth=1.5, label="modeled")
        ax.plot(freq_ghz, truth_y, color="#ff7f0e", linestyle="--", linewidth=1.25, label="measured")
        ax.set_title(response_plot_label(label, response_kind), fontsize=12)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.28)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9, frameon=True)
    return fig


def matplotlib_draw_smith_axes(ax: object) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")
    for curve, color, width, dash in smith_chart_curves():
        linestyle = ":" if dash else "-"
        ax.plot(np.real(curve), np.imag(curve), color=color, linewidth=width, linestyle=linestyle, zorder=1)


def matplotlib_smith_grid_pages(
    plt: object,
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    response_kind: str = "S",
) -> list[object]:
    _rows, _cols, ordered = subplot_grid(labels)
    present = [label for label in ordered if label in truth.sparams and label in pred.sparams]
    figures = []
    groups = chunks(present, 4)
    for page_index, group in enumerate(groups, start=1):
        fig, axes = plt.subplots(2, 2, figsize=(15.74, 11.53), squeeze=False)
        page_suffix = f" page {page_index}/{len(groups)}" if len(groups) > 1 else ""
        header = matplotlib_header(title, case_name, metrics, f"Smith / complex plane{page_suffix}")
        header_lines = header.count("\n") + 1
        top = max(0.58, 0.82 - 0.028 * max(header_lines - 3, 0))
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=top, wspace=0.18, hspace=0.34)
        fig.suptitle(header, fontsize=12, y=0.985)

        for idx, ax in enumerate(axes.ravel()):
            if idx >= len(group):
                ax.axis("off")
                continue
            label = group[idx]
            matplotlib_draw_smith_axes(ax)
            ax.plot(
                np.real(pred.sparams[label]),
                np.imag(pred.sparams[label]),
                color="#1f77b4",
                linewidth=1.8,
                label="modeled",
                zorder=3,
            )
            ax.plot(
                np.real(truth.sparams[label]),
                np.imag(truth.sparams[label]),
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.45,
                label="measured",
                zorder=4,
            )
            ax.set_title(response_plot_label(label, response_kind), fontsize=13)
            if page_index == 1 and idx == 0:
                ax.legend(loc="upper left", fontsize=10, frameon=True)
        figures.append(fig)
    if not figures:
        fig, ax = plt.subplots(figsize=(15.74, 11.53))
        ax.axis("off")
        fig.suptitle(matplotlib_header(title, case_name, metrics, "Smith / complex plane"), fontsize=12, y=0.985)
        figures.append(fig)
    return figures


def matplotlib_focus_page(
    plt: object,
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    metrics: dict[str, object],
    title: str,
    response_kind: str = "S",
) -> object:
    fig, (heat_ax, focus_ax) = plt.subplots(
        1,
        2,
        figsize=(9.285, 4.5),
        gridspec_kw={"width_ratios": [1.05, 1.65]},
    )
    title_lines = wrapped_text_lines(f"{title}: error focus for {case_name}", 78)
    top = max(0.60, 0.76 - 0.055 * max(len(title_lines) - 1, 0))
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.16, top=top, wspace=0.42)
    fig.suptitle("\n".join(title_lines), fontsize=12, y=0.97)

    grid = complete_sparam_grid(labels)
    max_by_label = metrics["max_mag_error_by_label"]
    assert isinstance(max_by_label, dict)
    if grid is not None:
        nports, ordered = grid
        values = np.asarray([float(max_by_label[label]) for label in ordered]).reshape(nports, nports)
        image = heat_ax.imshow(values, cmap="magma", origin="upper")
        heat_ax.set_xticks(np.arange(nports), labels=[str(idx) for idx in range(1, nports + 1)])
        heat_ax.set_yticks(np.arange(nports), labels=[str(idx) for idx in range(1, nports + 1)])
        heat_ax.set_xlabel("j")
        heat_ax.set_ylabel("i")
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        for row in range(nports):
            for col in range(nports):
                value = float(values[row, col])
                text_color = "white" if value < (vmin + vmax) / 2 else "black"
                heat_ax.text(col, row, f"{value:.3g}", ha="center", va="center", color=text_color)
        colorbar = fig.colorbar(image, ax=heat_ax, fraction=0.046, pad=0.06, format="%.3g")
        colorbar.ax.yaxis.get_offset_text().set_visible(False)
        heat_ax.set_title(f"Max magnitude error by {str(response_kind or 'S').strip().upper()}ij (dB)", fontsize=11)
    else:
        values = np.asarray([float(max_by_label[label]) for label in labels])
        heat_ax.bar(np.arange(len(labels)), values, color="#7b3294")
        heat_ax.set_xticks(
            np.arange(len(labels)),
            labels=[response_plot_label(label, response_kind) for label in labels],
            rotation=45,
            ha="right",
        )
        heat_ax.set_title(
            f"Max magnitude error by {str(response_kind or 'S').strip().upper()}-parameter (dB)",
            fontsize=11,
        )
        heat_ax.set_ylabel(f"{response_magnitude_axis_label(response_kind)} error")

    worst_label = str(metrics["worst_sparam"])
    worst_display = response_plot_label(worst_label, response_kind)
    freq_ghz = truth.freq_hz / 1e9
    is_y_response = str(response_kind or "S").strip().upper() == "Y"
    if is_y_response:
        focus_ax.plot(
            freq_ghz,
            np.real(pred.sparams[worst_label]),
            color="#1f77b4",
            linewidth=1.5,
            label="modeled real",
        )
        focus_ax.plot(
            freq_ghz,
            np.real(truth.sparams[worst_label]),
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.25,
            label="measured real",
        )
        focus_ax.plot(
            freq_ghz,
            np.imag(pred.sparams[worst_label]),
            color="#2ca02c",
            linewidth=1.5,
            label="modeled imag",
        )
        focus_ax.plot(
            freq_ghz,
            np.imag(truth.sparams[worst_label]),
            color="#d62728",
            linestyle="--",
            linewidth=1.25,
            label="measured imag",
        )
        focus_ax.set_ylabel("Admittance (siemens)")
    else:
        truth_mag = mag_db(truth.sparams[worst_label])
        pred_mag = mag_db(pred.sparams[worst_label])
        focus_ax.plot(freq_ghz, pred_mag, color="#1f77b4", linewidth=1.5, label="modeled")
        focus_ax.plot(freq_ghz, truth_mag, color="#ff7f0e", linestyle="--", linewidth=1.25, label="measured")
        focus_ax.set_ylabel(response_magnitude_axis_label(response_kind))
    focus_ax.set_title(f"Worst {worst_display}", fontsize=12)
    focus_ax.set_xlabel("Frequency (GHz)")
    focus_ax.grid(True, alpha=0.28)
    focus_ax.legend(loc="upper left", fontsize=9, frameon=True)

    passivity = metrics.get("passivity")
    max_sigma = None
    if isinstance(passivity, dict):
        max_sigma = passivity.get("max_singular_value")
    info_lines = [
        f"EVM={float(metrics['evm_pct']):.4g}% ({float(metrics['evm_db']):.4g} dB)",
        f"mag RMSE={float(metrics['mag_rmse_db']):.4g} dB",
        f"phase RMSE={float(metrics['phase_rmse_deg']):.4g} deg",
    ]
    if max_sigma is not None:
        passive = bool(float(max_sigma) <= 1.0 + 1e-6)
        info_lines.extend(
            [
                f"passive={passive}",
                f"max sigma={float(max_sigma):.4g}",
            ]
        )
    info = "\n".join(info_lines)
    focus_ax.text(0.04, 0.07, info, transform=focus_ax.transAxes, fontsize=9, va="bottom")
    return fig


def write_case_pdf_matplotlib(
    path: Path,
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    case_name: str,
    title: str,
    metrics: dict[str, object],
    response_kind: str = "S",
    plot_quantities: Sequence[str] | None = None,
    include_smith: bool = False,
) -> bool:
    modules = load_matplotlib_modules()
    if modules is None:
        return False
    plt, PdfPages = modules
    quantities = list(plot_quantities or ["magnitude", "phase"])
    figures = []
    if include_smith:
        figures.extend(matplotlib_smith_grid_pages(plt, truth, pred, labels, case_name, metrics, title, response_kind))
    page_labels = {
        "magnitude": "Magnitude",
        "phase": "Unwrapped phase",
        "real": "Real",
        "imag": "Imaginary",
    }
    for quantity in quantities:
        figures.append(
            matplotlib_grid_page(
                plt,
                truth,
                pred,
                labels,
                case_name,
                metrics,
                title,
                page_labels.get(quantity, quantity.title()),
                quantity,
                response_kind,
            )
        )
    figures.append(matplotlib_focus_page(plt, truth, pred, labels, case_name, metrics, title, response_kind))
    with PdfPages(path) as pdf:
        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)
    return True


def write_case_pdf(
    path: Path,
    truth: MDIFBlock,
    pred: MDIFBlock,
    labels: Sequence[str],
    parameter_names: Sequence[str],
    rank: int,
    metrics: dict[str, object],
    title: str | None = None,
    response_kind: str = "S",
    plot_quantities: Sequence[str] | None = None,
    include_smith: bool | None = None,
) -> None:
    param_text = ", ".join(f"{name}={truth.params.get(name, '')}" for name in parameter_names)
    case_name = f"block_{truth.source_index}" + (f" | {param_text}" if param_text else "")
    plot_title = title or f"Worst verification case {rank}"
    kind = str(response_kind or "S").strip().upper()
    quantities = list(plot_quantities or (["real", "imag"] if kind == "Y" else ["magnitude", "phase"]))
    add_smith = bool(kind == "S" if include_smith is None else include_smith)
    if write_case_pdf_matplotlib(
        path,
        truth,
        pred,
        labels,
        case_name,
        plot_title,
        metrics,
        response_kind,
        plot_quantities=quantities,
        include_smith=add_smith,
    ):
        return
    pages = []
    if add_smith:
        pages.extend(draw_smith_grid_pages(truth, pred, labels, case_name, metrics, plot_title, response_kind))
    page_labels = {
        "magnitude": "Magnitude",
        "phase": "Unwrapped phase",
        "real": "Real",
        "imag": "Imaginary",
    }
    for quantity in quantities:
        pages.append(
            draw_grid_page(
                truth,
                pred,
                labels,
                case_name,
                metrics,
                plot_title,
                page_labels.get(quantity, quantity.title()),
                quantity,
                response_kind,
            )
        )
    pages.append(draw_focus_page(truth, pred, labels, case_name, metrics, plot_title, response_kind))
    save_pdf_pages(path, pages)


def plot_worst_case_fits(
    truth_blocks: Sequence[MDIFBlock],
    pred_blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    parameter_names: Sequence[str],
    out_dir: Path,
    max_plots: int,
    plot_subdir: str = "worst_case_plots",
    csv_name: str = "worst_case_plots.csv",
    title_prefix: str = "Worst verification case",
    response_kind: str = "S",
    include_passivity: bool = True,
) -> list[Path]:
    if max_plots <= 0:
        return []
    plot_dir = out_dir / plot_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for truth, pred in zip(truth_blocks, pred_blocks):
        metrics = case_metrics(truth, pred, labels, include_passivity=include_passivity)
        cases.append({"source_index": truth.source_index, "truth": truth, "pred": pred, "metrics": metrics})
    cases.sort(
        key=lambda item: (
            float(item["metrics"]["max_abs_db"]),  # type: ignore[index]
            float(item["metrics"]["max_abs"]),  # type: ignore[index]
        ),
        reverse=True,
    )
    written: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    for rank, case in enumerate(cases[:max_plots], start=1):
        metrics = case["metrics"]
        assert isinstance(metrics, dict)
        stem = safe_filename(f"{rank:02d}_block_{case['source_index']}_worst_case")
        path = plot_dir / f"{stem}.pdf"
        write_case_pdf(
            path,
            truth=case["truth"],  # type: ignore[arg-type]
            pred=case["pred"],  # type: ignore[arg-type]
            labels=labels,
            parameter_names=parameter_names,
            rank=rank,
            metrics=metrics,
            title=f"{title_prefix} {rank}",
            response_kind=response_kind,
        )
        written.append(path)
        summary_rows.append(
            {
                "rank": rank,
                "source_index": case["source_index"],
                "worst_sparam": metrics["worst_sparam"],
                "rmse_abs": metrics["rmse_abs"],
                "rel_rmse": metrics["rel_rmse"],
                "evm_rms": metrics["evm_rms"],
                "evm_pct": metrics["evm_pct"],
                "evm_db": metrics["evm_db"],
                "mag_rmse_db": metrics["mag_rmse_db"],
                "phase_rmse_deg": metrics["phase_rmse_deg"],
                "max_abs": metrics["max_abs"],
                "max_abs_db": metrics["max_abs_db"],
                "plot": str(path.name),
            }
        )
    write_csv(plot_dir / csv_name, summary_rows)
    return written


def plot_worst_case_y_fits(
    truth_blocks: Sequence[MDIFBlock],
    pred_blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    parameter_names: Sequence[str],
    out_dir: Path,
    max_plots: int,
    z0: float = 50.0,
    title_context: str | None = None,
) -> tuple[list[Path], str | None]:
    if max_plots <= 0:
        return [], None
    try:
        truth_y_blocks = blocks_s_to_y_blocks(truth_blocks, labels, z0)
        pred_y_blocks = blocks_s_to_y_blocks(pred_blocks, labels, z0)
    except ValueError as exc:
        return [], str(exc)
    return (
        plot_worst_case_fits(
            truth_y_blocks,
            pred_y_blocks,
            labels,
            parameter_names,
            out_dir,
            max_plots=max_plots,
            plot_subdir="worst_case_y_plots",
            csv_name="worst_case_y_plots.csv",
            title_prefix=(
                f"{title_context} | Worst verification Y-parameter case"
                if title_context
                else "Worst verification Y-parameter case"
            ),
            response_kind="Y",
            include_passivity=False,
        ),
        None,
    )


def write_training_verification_artifacts(
    out_dir: Path,
    truth_blocks: Sequence[MDIFBlock],
    pred_blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
    parameter_names: Sequence[str],
    max_worst_plots: int,
    sparam_weights: dict[str, float] | None = None,
    y_z0: float = 50.0,
    title_context: str | None = None,
    frequency_weights: str | None = None,
) -> dict[str, object]:
    write_mdif(out_dir / "predicted_verification.mdif", pred_blocks, labels)
    metric_rows, summary = verification_metrics(
        truth_blocks,
        pred_blocks,
        labels,
        parameter_names,
        sparam_weights=sparam_weights,
        frequency_weights=frequency_weights,
    )
    write_csv(out_dir / "verification_metrics.csv", metric_rows)
    plot_paths = plot_worst_case_fits(
        truth_blocks,
        pred_blocks,
        labels,
        parameter_names,
        out_dir,
        max_plots=max_worst_plots,
        title_prefix=(
            f"{title_context} | Worst verification case"
            if title_context
            else "Worst verification case"
        ),
    )
    summary["worst_case_plots"] = [str(path.relative_to(out_dir)) for path in plot_paths]
    y_plot_paths, y_plot_warning = plot_worst_case_y_fits(
        truth_blocks,
        pred_blocks,
        labels,
        parameter_names,
        out_dir,
        max_plots=max_worst_plots,
        z0=y_z0,
        title_context=title_context,
    )
    if y_plot_paths:
        summary["worst_case_y_plots"] = [str(path.relative_to(out_dir)) for path in y_plot_paths]
        summary["worst_case_y_z0"] = y_z0
    if y_plot_warning:
        summary["worst_case_y_plot_warning"] = y_plot_warning
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    seen = set(fields)
    for row in rows[1:]:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_history(
    path: Path,
    history: Sequence[dict[str, float]],
    plot_title: str = "Model performance vs epoch",
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)
    write_training_history_plot(path.with_suffix(".pdf"), history, title=plot_title)


def history_series(
    history: Sequence[dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs: list[float] = []
    train_loss: list[float] = []
    val_loss: list[float] = []
    for row in history:
        epoch = csv_number(row.get("epoch"))
        train = csv_number(row.get("train_loss"))
        val = csv_number(row.get("val_loss"))
        if epoch is None or train is None or val is None:
            continue
        epochs.append(epoch)
        train_loss.append(train)
        val_loss.append(val)
    return (
        np.asarray(epochs, dtype=float),
        np.asarray(train_loss, dtype=float),
        np.asarray(val_loss, dtype=float),
    )


def write_training_history_plot_matplotlib(
    path: Path,
    history: Sequence[dict[str, float]],
    title: str = "Model performance vs epoch",
) -> bool:
    modules = load_matplotlib_modules()
    if modules is None:
        return False
    epochs, train_loss, val_loss = history_series(history)
    if epochs.size == 0:
        return False
    plt, PdfPages = modules
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=1.6, label="train loss")
    ax.plot(epochs, val_loss, color="#ff7f0e", linestyle="--", linewidth=1.4, label="validation loss")
    finite_losses = np.concatenate([train_loss[np.isfinite(train_loss)], val_loss[np.isfinite(val_loss)]])
    if finite_losses.size and np.all(finite_losses > 0.0):
        ax.set_yscale("log")
    ax.set_title("\n".join(wrapped_text_lines(title, 92)), fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    return True


def draw_training_history_legend(canvas: PdfCanvas, rect: tuple[float, float, float, float]) -> None:
    left, top, width, _height = rect
    legend_left = left + width - 148
    legend_top = top + 12
    canvas.rect(legend_left - 8, legend_top - 10, 140, 42, stroke=(0.82, 0.82, 0.82), fill=(1.0, 1.0, 1.0))
    canvas.line(legend_left, legend_top, legend_left + 24, legend_top, color=(0.1216, 0.4667, 0.7059), width=1.8)
    canvas.text(legend_left + 30, legend_top + 4, "train loss", size=10)
    canvas.line(legend_left, legend_top + 19, legend_left + 24, legend_top + 19, color=(1.0, 0.498, 0.0549), width=1.8, dash="[5 3] 0")
    canvas.text(legend_left + 30, legend_top + 23, "validation loss", size=10)


def write_training_history_plot_fallback(
    path: Path,
    history: Sequence[dict[str, float]],
    title: str = "Model performance vs epoch",
) -> bool:
    epochs, train_loss, val_loss = history_series(history)
    if epochs.size == 0:
        return False
    canvas = PdfCanvas(width=900.0, height=520.0)
    title_lines = wrapped_text_lines(title, 96)
    header_bottom = draw_wrapped_centered_text(
        canvas,
        title_lines,
        canvas.width / 2,
        24,
        16,
        18,
        font="F2",
    )
    title_offset = max(0.0, header_bottom - 42.0)
    rect = (86.0, 82.0 + title_offset, 742.0, max(260.0, 338.0 - title_offset))
    finite_losses = np.concatenate([train_loss[np.isfinite(train_loss)], val_loss[np.isfinite(val_loss)]])
    use_log = bool(finite_losses.size and np.all(finite_losses > 0.0))
    if use_log:
        train_y = np.log10(train_loss)
        val_y = np.log10(val_loss)
        y_label = "log10(loss)"
    else:
        train_y = train_loss
        val_y = val_loss
        y_label = "Loss"
    x_range = expanded_range([epochs])
    y_range = expanded_range([train_y, val_y])
    pdf_ticks(canvas, rect, x_range, y_range, y_label, "Epoch")
    canvas.polyline(
        plot_points(epochs, train_y, x_range, y_range, rect),
        color=(0.1216, 0.4667, 0.7059),
        width=2.0,
    )
    canvas.polyline(
        plot_points(epochs, val_y, x_range, y_range, rect),
        color=(1.0, 0.498, 0.0549),
        width=2.0,
        dash="[5 3] 0",
    )
    draw_training_history_legend(canvas, rect)
    save_pdf_pages(path, [canvas])
    return True


def write_training_history_plot(
    path: Path,
    history: Sequence[dict[str, float]],
    title: str = "Model performance vs epoch",
) -> Path | None:
    if not history:
        if path.exists():
            path.unlink()
        return None
    if write_training_history_plot_matplotlib(path, history, title=title):
        return path
    if write_training_history_plot_fallback(path, history, title=title):
        return path
    if path.exists():
        path.unlink()
    return None


def cleanup_trial_dir(trial_dir: Path, keep_trial_models: bool) -> None:
    if keep_trial_models:
        return
    for name in [
        "model.npz",
        "metadata.json",
        "dc_model.npz",
        "dc_model.json",
        "composite_model_manifest.json",
        "predicted_verification.mdif",
        "training_history.csv",
        "dc_training_history.csv",
        "training_summary.md",
        "verification_metrics.csv",
    ]:
        path = trial_dir / name
        if path.exists():
            path.unlink()


def markdown_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def metric_text(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return markdown_escape(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6g}"


def metric_text_fixed(value: object, decimals: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return markdown_escape(value)
    if not math.isfinite(numeric):
        return ""
    places = max(0, int(decimals))
    fixed = f"{numeric:.{places}f}"
    if numeric != 0.0 and csv_number(fixed) == 0.0:
        return f"{numeric:.{places}e}"
    return fixed


def cli_metric_label(metric_name: object) -> str:
    text = str(metric_name or "metric")
    labels = {
        "evm_db": "EVMdB",
        "evm_pct": "EVM%",
        "evm_rms": "EVM",
        "max_abs": "maxAbs",
        "max_abs_db": "maxdB",
        "passivity.max_singular_value": "sigma",
        "passivity.violating_points": "pv",
        "rmse_abs": "RMSE",
        "rmse_db": "RMSEdB",
        "weighted_evm_db": "wEVMdB",
        "weighted_evm_pct": "wEVM%",
        "weighted_evm_rms": "wEVM",
        "weighted_max_abs": "wMaxAbs",
        "weighted_max_abs_db": "wMaxdB",
        "weighted_rmse_abs": "wRMSE",
        "weighted_rmse_db": "wRMSEdB",
    }
    if text in labels:
        return labels[text]
    compact = (
        text.replace("passivity.", "p.")
        .replace("weighted_", "w_")
        .replace("max_singular_value", "sigma")
        .replace("violating_points", "pv")
    )
    return compact if len(compact) <= 14 else compact[:13] + "~"


def cli_color_enabled(stream: object) -> bool:
    if os.environ.get("CLICOLOR_FORCE", "0") != "0":
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("CLICOLOR") == "0":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def cli_color_text(text: str, color: str, stream: object | None = None) -> str:
    if stream is None:
        stream = sys.stdout
    if not cli_color_enabled(stream):
        return text
    codes = {
        "green": "\033[32m",
        "red": "\033[31m",
    }
    prefix = codes.get(color)
    if not prefix:
        return text
    return f"{prefix}{text}\033[0m"


def compact_cli_text(text: object, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if max_chars <= 0 or len(compact) <= max_chars:
        return compact
    if max_chars <= 1:
        return "~"
    return compact[: max_chars - 1] + "~"


def plot_links_cell(raw_paths: object) -> str:
    if not raw_paths:
        return ""
    paths = [part.strip() for part in str(raw_paths).split(";") if part.strip()]
    return plot_links_for_paths(paths)


def plot_links_for_paths(paths: Sequence[str]) -> str:
    links = []
    s_count = 0
    y_count = 0
    for path in paths:
        lower = path.lower()
        if lower.endswith("training_history.pdf"):
            label = "Loss vs epoch"
        elif "worst_case_y_plots" in lower:
            y_count += 1
            label = f"Y plot {y_count}"
        else:
            s_count += 1
            label = f"S plot {s_count}"
        links.append(f"[{label}]({path})")
    return "<br>".join(links)


def sweep_plot_links_cell(raw_paths: object, trial_value: object, sweep_dir: Path) -> str:
    paths = [part.strip() for part in str(raw_paths or "").split(";") if part.strip()]
    trial_number = csv_number(trial_value)
    if trial_number is not None:
        history_rel = f"trials/trial_{int(trial_number):04d}/training_history.pdf"
        if (sweep_dir / history_rel).exists() and history_rel not in paths:
            paths.insert(0, history_rel)
    return plot_links_for_paths(paths)


def trial_plot_paths(summary: dict[str, object], trial_dir: Path, out_dir: Path) -> list[str]:
    paths = []
    history_plot = trial_dir / "training_history.pdf"
    if history_plot.exists():
        try:
            paths.append(str(history_plot.relative_to(out_dir)))
        except ValueError:
            paths.append(str(history_plot))
    for key in ["worst_case_plots", "worst_case_y_plots"]:
        raw = summary.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            try:
                path = (trial_dir / str(item)).relative_to(out_dir)
            except ValueError:
                path = trial_dir / str(item)
            paths.append(str(path))
    return paths


def markdown_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return markdown_escape(json.dumps(value, sort_keys=True))
    return metric_text(value)


def single_model_train_command(
    command_prefix: Sequence[str],
    train_args: argparse.Namespace,
    out_dir: Path,
) -> str:
    """Build a shell-copyable train command from a selected sweep trial."""

    argv = [str(part) for part in command_prefix]
    internal_names = {
        "command",
        "func",
        "progress_label",
        "debug_label",
        "quiet",
    }
    for name, raw_value in vars(train_args).items():
        if name in internal_names or raw_value is None:
            continue
        value = str(out_dir) if name == "out_dir" else raw_value
        flag = f"--{name.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        argv.extend([flag, str(value)])
    return shell_command(argv)


def shell_command(argv: Sequence[object]) -> str:
    """Render command arguments as a shell-copyable command line."""

    return " ".join(shlex.quote(str(part)) for part in argv)


def repository_relative_path(path: str | Path, repository_root: Path) -> str:
    """Return a command path relative to the repository root."""

    return os.path.relpath(
        Path(path).expanduser().resolve(),
        repository_root.expanduser().resolve(),
    )


def veriloga_command_defaults(
    script_path: Path,
    model_dir: Path,
) -> tuple[str, str]:
    """Return an explicit module name and one unity scale for all parameters."""

    resolved_model_dir = model_dir.resolve()
    fallback_name = normalize_name(script_path.stem) or "model"
    module_name = f"{normalize_name(resolved_model_dir.name) or fallback_name}_va"
    return module_name, "1.0"


def build_training_export_commands(
    script_path: Path,
    model_dir: Path,
    template_mdif: str | Path | None = None,
    *,
    include_veriloga: bool = False,
) -> list[tuple[str, str]]:
    """Build export commands whose paths are relative to the repository root."""

    resolved_script_path = script_path.resolve()
    repository_root = resolved_script_path.parent
    resolved_model_dir = model_dir.resolve()
    command_script_path = repository_relative_path(
        resolved_script_path,
        repository_root,
    )
    command_python = Path(sys.executable).name or "python3"
    command_model_dir = repository_relative_path(
        resolved_model_dir,
        repository_root,
    )
    dc_path = Path(template_mdif).resolve() if template_mdif else None
    dc_port_paths_spec: str | None = None
    saved_dynamic_dc_model = False
    saved_full_matrix_dc_model = False
    saved_paths_were_explicit = False
    metadata_path = resolved_model_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            saved_metadata = json.loads(metadata_path.read_text())
            saved_dynamic_dc_model = bool(saved_metadata.get("dc_model_kind"))
            saved_full_matrix_dc_model = (
                saved_metadata.get("dc_model_representation") == "full_s_matrix"
                or saved_metadata.get("dc_model_kind")
                == "geometry_dependent_exact_dc_full_s_mlp"
            )
            saved_spec = saved_metadata.get("dc_port_path_spec")
            paths_were_explicit = (
                saved_metadata.get("dc_port_paths_explicit") is not False
            )
            saved_paths_were_explicit = paths_were_explicit
            if saved_spec and paths_were_explicit:
                dc_port_paths_spec = str(saved_spec)
            elif saved_metadata.get("dc_port_paths") and paths_were_explicit:
                dc_port_paths_spec = dc_port_path_spec(
                    [str(path) for path in saved_metadata["dc_port_paths"]]
                )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            dc_port_paths_spec = None
    needs_dc_source = not saved_dynamic_dc_model or (
        not saved_full_matrix_dc_model and not saved_paths_were_explicit
    )
    commands: list[tuple[str, str]] = []
    if include_veriloga:
        module_name, parameter_scale_spec = veriloga_command_defaults(
            resolved_script_path,
            resolved_model_dir,
        )
        hb_module_name = f"{normalize_name(resolved_model_dir.name) or resolved_script_path.stem}_hb"
        ads_hb_argv = [
            command_python,
            command_script_path,
            "export-ads-hb",
            "--model-dir",
            command_model_dir,
            "--out-dir",
            repository_relative_path(
                resolved_model_dir / "ads_hb_export",
                repository_root,
            ),
            "--module-name",
            hb_module_name,
            "--parameter-input-scales",
            parameter_scale_spec,
        ]
        if needs_dc_source:
            ads_hb_argv.extend(
                [
                    "--dc-open-threshold",
                    f"{DEFAULT_DC_OPEN_THRESHOLD_OHM:g}",
                    "--dc-open-resistance",
                    f"{DEFAULT_DC_OPEN_RESISTANCE_OHM:g}",
                ]
            )
        if dc_path is not None and needs_dc_source:
            ads_hb_argv.extend(
                ["--dc-mdif", repository_relative_path(dc_path, repository_root)]
            )
        if dc_port_paths_spec:
            ads_hb_argv.extend(["--dc-port-paths", dc_port_paths_spec])
        commands.append(
            (
                "Self-contained ADS HB passive network",
                shell_command(ads_hb_argv),
            )
        )
        veriloga_argv = [
            command_python,
            command_script_path,
            "export-veriloga",
            "--model-dir",
            command_model_dir,
            "--out-dir",
            repository_relative_path(
                resolved_model_dir / "veriloga_export",
                repository_root,
            ),
            "--module-name",
            module_name,
            "--parameter-input-scales",
            parameter_scale_spec,
        ]
        if needs_dc_source:
            veriloga_argv.extend(
                [
                    "--dc-open-threshold",
                    f"{DEFAULT_DC_OPEN_THRESHOLD_OHM:g}",
                    "--dc-open-resistance",
                    f"{DEFAULT_DC_OPEN_RESISTANCE_OHM:g}",
                ]
            )
        if dc_path is not None and needs_dc_source:
            veriloga_argv.extend(
                ["--dc-mdif", repository_relative_path(dc_path, repository_root)]
            )
        if dc_port_paths_spec:
            veriloga_argv.extend(["--dc-port-paths", dc_port_paths_spec])
        commands.append(
            (
                "Self-contained Verilog-A",
                shell_command(veriloga_argv),
            )
        )
    template_path = Path(template_mdif).resolve() if template_mdif else None
    if template_path is None:
        candidate = resolved_model_dir / "predicted_verification.mdif"
        template_path = candidate if candidate.exists() else None
    if template_path is not None:
        sampled_argv = [
            command_python,
            command_script_path,
            "export-ads-mdif",
            "--model-dir",
            command_model_dir,
            "--out-dir",
            repository_relative_path(
                resolved_model_dir / "ads_mdif_export",
                repository_root,
            ),
            "--template-mdif",
            repository_relative_path(template_path, repository_root),
        ]
        if needs_dc_source:
            sampled_argv.extend(
                [
                    "--dc-open-threshold",
                    f"{DEFAULT_DC_OPEN_THRESHOLD_OHM:g}",
                    "--dc-open-resistance",
                    f"{DEFAULT_DC_OPEN_RESISTANCE_OHM:g}",
                ]
            )
        if dc_path is not None and needs_dc_source:
            sampled_argv.extend(
                ["--dc-mdif", repository_relative_path(dc_path, repository_root)]
            )
        if dc_port_paths_spec:
            sampled_argv.extend(["--dc-port-paths", dc_port_paths_spec])
        commands.append(
            (
                "Sampled ADS MDIF",
                shell_command(sampled_argv),
            )
        )
    return commands


def training_export_section(
    export_commands: Sequence[tuple[str, str]],
) -> list[str]:
    """Render copyable model-export commands for a training report."""

    if not export_commands:
        return []
    lines = [
        "## Export Model",
        "",
        "Run any of these commands from the repository root:",
        "",
    ]
    for label, command in export_commands:
        lines.extend(
            [
                f"### {markdown_escape(label)}",
                "",
                "```bash",
                command,
                "```",
                "",
            ]
        )
    return lines


def update_training_export_commands(
    path: Path,
    export_commands: Sequence[tuple[str, str]],
) -> None:
    """Replace the export section after a sweep promotes a trial model."""

    if not path.exists():
        raise FileNotFoundError(f"Training report not found: {path}")
    heading = "## Export Model"
    text = path.read_text()
    prefix = text.split(f"\n{heading}\n", 1)[0].rstrip()
    section = training_export_section(export_commands)
    if section:
        path.write_text(f"{prefix}\n\n" + "\n".join(section))
    else:
        path.write_text(f"{prefix}\n")


def write_training_markdown(
    path: Path,
    model_kind: str,
    config: dict[str, object],
    summary: dict[str, object],
    history: Sequence[dict[str, float]],
    export_commands: Sequence[tuple[str, str]] | None = None,
) -> None:
    lines = [
        "# Training Summary",
        "",
        f"Model: `{markdown_escape(model_kind)}`",
        "",
    ]
    warning = summary.get("warning")
    if warning:
        lines.extend([f"Warning: {markdown_escape(warning)}", ""])

    lines.extend(["## Configuration", "", "| Setting | Value |", "| --- | --- |"])
    for key, value in config.items():
        lines.append(f"| `{markdown_escape(key)}` | {markdown_value(value)} |")
    lines.append("")

    if history:
        final = history[-1]
        lines.extend(
            [
                "## Final Training Loss",
                "",
                "| Epoch | Train Loss | Validation Loss |",
                "| ---: | ---: | ---: |",
                "| "
                + " | ".join(
                    [
                        metric_text(final.get("epoch")),
                        metric_text(final.get("train_loss")),
                        metric_text(final.get("val_loss")),
                    ]
                )
                + " |",
                "",
            ]
        )

    metric_rows = [
        ("RMSE abs", summary.get("rmse_abs")),
        ("Mean abs", summary.get("mean_abs")),
        ("Max abs", summary.get("max_abs")),
        ("EVM %", summary.get("evm_pct")),
        ("EVM dB", summary.get("evm_db")),
        ("Weighted RMSE abs", summary.get("weighted_rmse_abs")),
        ("Weighted mean abs", summary.get("weighted_mean_abs")),
        ("Weighted max abs", summary.get("weighted_max_abs")),
        ("Weighted EVM %", summary.get("weighted_evm_pct")),
        ("Weighted EVM dB", summary.get("weighted_evm_db")),
        ("RMSE dB", summary.get("rmse_db")),
        ("Max abs dB", summary.get("max_abs_db")),
        ("Weighted RMSE dB", summary.get("weighted_rmse_db")),
        ("Weighted max abs dB", summary.get("weighted_max_abs_db")),
    ]
    present_metric_rows = [(name, value) for name, value in metric_rows if value is not None]
    if present_metric_rows:
        lines.extend(["## Verification Metrics", "", "| Metric | Value |", "| --- | ---: |"])
        for name, value in present_metric_rows:
            lines.append(f"| {markdown_escape(name)} | {metric_text(value)} |")
        lines.append("")

    passivity = summary.get("passivity")
    if isinstance(passivity, dict):
        lines.extend(["## Passivity", "", "| Metric | Value |", "| --- | ---: |"])
        for key in ["nports", "max_singular_value", "violating_points"]:
            if key in passivity:
                lines.append(f"| `{markdown_escape(key)}` | {metric_text(passivity.get(key))} |")
        lines.append("")

    raw_plots = summary.get("worst_case_plots")
    raw_y_plots = summary.get("worst_case_y_plots")
    has_plots = isinstance(raw_plots, list) and bool(raw_plots)
    has_y_plots = isinstance(raw_y_plots, list) and bool(raw_y_plots)
    y_plot_warning = summary.get("worst_case_y_plot_warning")
    artifacts = [
        ("RF model weights", "model.npz"),
        ("Metadata", "metadata.json"),
        ("DC model weights", "dc_model.npz"),
        ("DC model metadata", "dc_model.json"),
        ("RF training history", "training_history.csv"),
        ("DC training history", "dc_training_history.csv"),
        ("Verification summary JSON", "verification_summary.json"),
    ]
    if (path.parent / "training_history.pdf").exists():
        artifacts.insert(3, ("Training loss plot", "training_history.pdf"))
    if (path.parent / "dc_training_history.pdf").exists():
        artifacts.append(("DC training loss plot", "dc_training_history.pdf"))
    if (path.parent / "kbnn_training_debug.json").exists():
        artifacts.append(("KBNN debug diagnostics", "kbnn_training_debug.json"))
    if not warning:
        artifacts.extend(
            [
                ("Predicted verification MDIF", "predicted_verification.mdif"),
                ("Detailed verification metrics", "verification_metrics.csv"),
            ]
        )
        if has_plots:
            artifacts.append(("Worst-case plot index", "worst_case_plots/worst_case_plots.csv"))
        if has_y_plots:
            artifacts.append(("Worst-case Y-parameter plot index", "worst_case_y_plots/worst_case_y_plots.csv"))
    lines.extend(["## Artifacts", ""])
    for label, artifact in artifacts:
        lines.append(f"- [{markdown_escape(label)}]({artifact})")
    lines.append("")

    if y_plot_warning:
        lines.extend(
            [
                "Y-parameter implementation plots were skipped:",
                "",
                f"> {markdown_escape(y_plot_warning)}",
                "",
            ]
        )

    if has_plots:
        lines.extend(["## Worst-Case Plots", ""])
        assert isinstance(raw_plots, list)
        for idx, plot_path in enumerate(raw_plots, start=1):
            lines.append(f"- [plot {idx}]({markdown_escape(plot_path)})")
        lines.append("")

    if has_y_plots:
        lines.extend(["## Worst-Case Y-Parameter Plots", ""])
        y_z0 = summary.get("worst_case_y_z0")
        if y_z0 is not None:
            lines.extend(
                [
                    f"Reference impedance for S-to-Y conversion: `{metric_text(y_z0)}` ohms.",
                    "",
                ]
            )
        assert isinstance(raw_y_plots, list)
        for idx, plot_path in enumerate(raw_y_plots, start=1):
            lines.append(f"- [plot {idx}]({markdown_escape(plot_path)})")
        lines.append("")

    lines.extend(training_export_section(export_commands or []))

    path.write_text("\n".join(lines))


def write_sweep_markdown(
    path: Path,
    rows: Sequence[dict[str, object]],
    selection_metric: str,
    best_config: dict[str, object] | None,
    best_metric: float | None,
    reproduction_command: str | None = None,
    diagnostic_artifacts: Sequence[str] | None = None,
) -> None:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            bool(str(row.get("error") or "").strip()),
            row.get("metric") is None,
            float(row.get("metric")) if row.get("metric") is not None else float("inf"),
        ),
    )
    lines = [
        "# Sweep Summary",
        "",
        f"Selection metric: `{selection_metric}`",
        "",
    ]
    if best_config is not None and best_metric is not None:
        config_text = ", ".join(f"`{key}={value}`" for key, value in best_config.items())
        lines.extend([f"Best metric: `{best_metric:.6g}`", f"Best configuration: {config_text}", ""])
    if reproduction_command:
        lines.extend(
            [
                "## Reproduce the Best Model",
                "",
                "Run this command from the repository root to train the selected configuration by itself:",
                "",
                "```bash",
                reproduction_command,
                "```",
                "",
            ]
        )
    if diagnostic_artifacts:
        lines.extend(["Diagnostic artifacts:", ""])
        for artifact in diagnostic_artifacts:
            label = Path(artifact).name
            lines.append(f"- [{markdown_escape(label)}]({artifact})")
        lines.append("")
    lines.extend(
        [
            "| Rank | Trial | Metric | RMSE abs | Max abs | EVM % | EVM dB | Weighted RMSE | Weighted EVM % | Weighted EVM dB | RMSE dB | Max dB | Weighted RMSE dB | Max sigma | Violations | Configuration | Trial plots | Error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    known = {
        "trial",
        "metric",
        "selection_metric",
        "rmse_abs",
        "max_abs",
        "evm_rms",
        "evm_pct",
        "evm_db",
        "weighted_rmse_abs",
        "weighted_max_abs",
        "weighted_evm_rms",
        "weighted_evm_pct",
        "weighted_evm_db",
        "rmse_db",
        "max_abs_db",
        "weighted_rmse_db",
        "weighted_max_abs_db",
        "error",
        "passivity_max_singular_value",
        "passivity_violating_points",
        "trial_seed",
        "trial_seed_mode",
        "worst_case_plots",
    }
    for rank, row in enumerate(sorted_rows, start=1):
        config = ", ".join(
            f"`{markdown_escape(key)}={markdown_escape(value)}`"
            for key, value in row.items()
            if key not in known
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    metric_text(row.get("trial")),
                    metric_text(row.get("metric")),
                    metric_text(row.get("rmse_abs")),
                    metric_text(row.get("max_abs")),
                    metric_text(row.get("evm_pct")),
                    metric_text(row.get("evm_db")),
                    metric_text(row.get("weighted_rmse_abs")),
                    metric_text(row.get("weighted_evm_pct")),
                    metric_text(row.get("weighted_evm_db")),
                    metric_text(row.get("rmse_db")),
                    metric_text(row.get("max_abs_db")),
                    metric_text(row.get("weighted_rmse_db")),
                    metric_text(row.get("passivity_max_singular_value")),
                    metric_text(row.get("passivity_violating_points")),
                    config,
                    sweep_plot_links_cell(row.get("worst_case_plots"), row.get("trial"), path.parent),
                    markdown_escape(row.get("error", "")),
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


SWEEP_DIAGNOSTIC_METRICS = [
    "metric",
    "rmse_abs",
    "max_abs",
    "evm_pct",
    "evm_db",
    "weighted_rmse_abs",
    "weighted_max_abs",
    "weighted_evm_pct",
    "weighted_evm_db",
    "rmse_db",
    "max_abs_db",
    "weighted_rmse_db",
    "weighted_max_abs_db",
    "passivity_max_singular_value",
    "passivity_violating_points",
]


def sweep_diagnostic_metric_names(
    rows: Sequence[dict[str, object]],
    selection_metric: str,
) -> list[str]:
    names = ["metric"]
    mapped_selection = selection_metric.replace("passivity.", "passivity_")
    if mapped_selection not in names:
        names.append(mapped_selection)
    for name in SWEEP_DIAGNOSTIC_METRICS:
        if name not in names:
            names.append(name)
    available: list[str] = []
    for name in names:
        if any(csv_number(row.get(name)) is not None for row in rows):
            available.append(name)
    return available


def sweep_diagnostic_metric_label(metric_name: str, selection_metric: str) -> str:
    if metric_name == "metric":
        return f"selection metric ({selection_metric})"
    if metric_name == "passivity_max_singular_value":
        return "passivity.max_singular_value"
    if metric_name == "passivity_violating_points":
        return "passivity.violating_points"
    return metric_name


def sort_category_key(value: object) -> tuple[int, float | str]:
    numeric = csv_number(value)
    if numeric is not None:
        return (0, numeric)
    return (1, str(value))


def sweep_row_fails_passivity(row: dict[str, object]) -> bool:
    violations = csv_number(row.get("passivity_violating_points"))
    if violations is not None:
        return violations > 0.0
    max_sigma = csv_number(row.get("passivity_max_singular_value"))
    if max_sigma is not None:
        return max_sigma > 1.0 + 1e-6
    return False


def finite_metric_pairs(
    rows: Sequence[dict[str, object]],
    parameter_name: str,
    metric_name: str,
) -> list[tuple[object, float, int | None, bool]]:
    pairs: list[tuple[object, float, int | None, bool]] = []
    for row in rows:
        y = csv_number(row.get(metric_name))
        if y is None:
            continue
        value = row.get(parameter_name)
        if value is None or value == "":
            continue
        trial_value = csv_number(row.get("trial"))
        pairs.append(
            (
                value,
                y,
                int(trial_value) if trial_value is not None else None,
                sweep_row_fails_passivity(row),
            )
        )
    return pairs


def write_sweep_diagnostic_stats(
    path: Path,
    rows: Sequence[dict[str, object]],
    swept_columns: Sequence[str],
    metric_names: Sequence[str],
    selection_metric: str,
) -> None:
    stat_rows: list[dict[str, object]] = []
    for metric_name in metric_names:
        metric_label = sweep_diagnostic_metric_label(metric_name, selection_metric)
        for parameter_name in swept_columns:
            groups: dict[str, dict[str, object]] = {}
            for value, y, _trial, passivity_failed in finite_metric_pairs(rows, parameter_name, metric_name):
                group = groups.setdefault(str(value), {"values": [], "total_count": 0, "failed_count": 0})
                group["total_count"] = int(group["total_count"]) + 1
                if passivity_failed:
                    group["failed_count"] = int(group["failed_count"]) + 1
                else:
                    values = group["values"]
                    assert isinstance(values, list)
                    values.append(y)
            for value in sorted(groups, key=sort_category_key):
                group = groups[value]
                values = np.asarray(group["values"], dtype=float)
                total_count = int(group["total_count"])
                failed_count = int(group["failed_count"])
                stat_rows.append(
                    {
                        "metric": metric_label,
                        "swept_parameter": parameter_name,
                        "value": value,
                        "count": int(values.size),
                        "passivity_pass_count": int(values.size),
                        "passivity_failed_count": failed_count,
                        "total_count": total_count,
                        "excluded_from_average": failed_count,
                        "best": float(np.min(values)) if values.size else "",
                        "median": float(np.median(values)) if values.size else "",
                        "mean": float(np.mean(values)) if values.size else "",
                        "worst": float(np.max(values)) if values.size else "",
                    }
                )
    write_csv(path, stat_rows)


def plot_sweep_diagnostics_matplotlib(
    path: Path,
    rows: Sequence[dict[str, object]],
    swept_columns: Sequence[str],
    metric_names: Sequence[str],
    selection_metric: str,
) -> bool:
    modules = load_matplotlib_modules()
    if modules is None:
        return False
    plt, PdfPages = modules
    with PdfPages(path) as pdf:
        for metric_name in metric_names:
            label = sweep_diagnostic_metric_label(metric_name, selection_metric)
            cols = 2 if len(swept_columns) <= 4 else 3
            plot_rows = int(math.ceil(len(swept_columns) / cols))
            fig, axes = plt.subplots(
                plot_rows,
                cols,
                figsize=(5.8 * cols, max(3.8, 3.35 * plot_rows)),
                squeeze=False,
            )
            title = f"Sweep error diagnostics: {label}"
            title_lines = wrapped_text_lines(title, 110)
            top = max(0.76, 0.90 - 0.03 * max(len(title_lines) - 1, 0))
            fig.subplots_adjust(left=0.08, right=0.975, bottom=0.12, top=top, wspace=0.30, hspace=0.46)
            fig.suptitle("\n".join(title_lines), fontsize=14, y=0.985)

            for idx, ax in enumerate(axes.ravel()):
                if idx >= len(swept_columns):
                    ax.axis("off")
                    continue
                parameter_name = swept_columns[idx]
                pairs = finite_metric_pairs(rows, parameter_name, metric_name)
                if not pairs:
                    ax.axis("off")
                    continue
                y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
                failed_mask = np.asarray([pair[3] for pair in pairs], dtype=bool)
                pass_mask = ~failed_mask
                x_numeric = [csv_number(pair[0]) for pair in pairs]
                if all(value is not None for value in x_numeric):
                    x_values = np.asarray([float(value) for value in x_numeric if value is not None], dtype=float)
                    if np.any(pass_mask):
                        ax.scatter(
                            x_values[pass_mask],
                            y_values[pass_mask],
                            color="#1f77b4",
                            alpha=0.78,
                            s=28,
                            label="passivity OK",
                        )
                    if np.any(failed_mask):
                        ax.scatter(
                            x_values[failed_mask],
                            y_values[failed_mask],
                            color="#d62728",
                            alpha=0.88,
                            s=38,
                            marker="x",
                            linewidths=1.25,
                            label="passivity fail",
                        )
                    grouped: dict[float, list[float]] = {}
                    for x_value, y_value, passivity_failed in zip(x_values, y_values, failed_mask):
                        if passivity_failed:
                            continue
                        grouped.setdefault(float(x_value), []).append(float(y_value))
                    xs = sorted(grouped)
                    means = [float(np.mean(grouped[x])) for x in xs]
                    if len(xs) > 1:
                        ax.plot(
                            xs,
                            means,
                            color="#ff7f0e",
                            linewidth=1.8,
                            marker="o",
                            label="mean, passive trials",
                        )
                    elif xs:
                        ax.scatter(
                            xs,
                            means,
                            color="#ff7f0e",
                            s=50,
                            marker="D",
                            label="mean, passive trials",
                        )
                    ax.set_xlabel(parameter_name)
                else:
                    categories = sorted({str(pair[0]) for pair in pairs}, key=sort_category_key)
                    index = {category: pos for pos, category in enumerate(categories)}
                    offsets_by_category: dict[str, int] = {category: 0 for category in categories}
                    counts_by_category = {
                        category: sum(1 for pair in pairs if str(pair[0]) == category)
                        for category in categories
                    }
                    x_values = []
                    for value, _y, _trial, _passivity_failed in pairs:
                        category = str(value)
                        count = counts_by_category[category]
                        rank = offsets_by_category[category]
                        offsets_by_category[category] += 1
                        jitter = 0.0
                        if count > 1:
                            jitter = (rank - (count - 1) / 2.0) * min(0.08, 0.35 / count)
                        x_values.append(index[category] + jitter)
                    x_values_array = np.asarray(x_values, dtype=float)
                    if np.any(pass_mask):
                        ax.scatter(
                            x_values_array[pass_mask],
                            y_values[pass_mask],
                            color="#1f77b4",
                            alpha=0.78,
                            s=28,
                            label="passivity OK",
                        )
                    if np.any(failed_mask):
                        ax.scatter(
                            x_values_array[failed_mask],
                            y_values[failed_mask],
                            color="#d62728",
                            alpha=0.88,
                            s=38,
                            marker="x",
                            linewidths=1.25,
                            label="passivity fail",
                        )
                    for category in categories:
                        grouped_values = [
                            pair[1]
                            for pair in pairs
                            if str(pair[0]) == category and not pair[3]
                        ]
                        if not grouped_values:
                            continue
                        mean = float(np.mean(grouped_values))
                        xpos = index[category]
                        ax.plot(
                            [xpos - 0.22, xpos + 0.22],
                            [mean, mean],
                            color="#ff7f0e",
                            linewidth=2.2,
                            label="mean, passive trials" if category == categories[0] else None,
                        )
                    ax.set_xticks(range(len(categories)), labels=categories, rotation=25, ha="right")
                    ax.set_xlabel(parameter_name)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.28)
                ax.set_title(parameter_name, fontsize=12)
                handles, _labels = ax.get_legend_handles_labels()
                if idx == 0 and handles:
                    ax.legend(loc="best", fontsize=9, frameon=True)
            pdf.savefig(fig)
            plt.close(fig)
    return True


def plot_sweep_diagnostics_fallback_pdf(
    path: Path,
    rows: Sequence[dict[str, object]],
    swept_columns: Sequence[str],
    metric_names: Sequence[str],
    selection_metric: str,
) -> None:
    canvases: list[PdfCanvas] = []
    for metric_name in metric_names:
        metric_label = sweep_diagnostic_metric_label(metric_name, selection_metric)
        canvas = PdfCanvas(width=930.0, height=625.0)
        title_lines = wrapped_text_lines(f"Sweep error diagnostics: {metric_label}", 98)
        y = draw_wrapped_centered_text(canvas, title_lines, canvas.width / 2, 24, 16, 18, font="F2") + 36
        canvas.text(52, y, "Best and mean values by swept parameter; passivity failures excluded", size=13, font="F2")
        y += 28
        for parameter_name in swept_columns:
            groups: dict[str, dict[str, object]] = {}
            for value, metric_value, _trial, passivity_failed in finite_metric_pairs(rows, parameter_name, metric_name):
                group = groups.setdefault(str(value), {"values": [], "failed_count": 0, "total_count": 0})
                group["total_count"] = int(group["total_count"]) + 1
                if passivity_failed:
                    group["failed_count"] = int(group["failed_count"]) + 1
                else:
                    values = group["values"]
                    assert isinstance(values, list)
                    values.append(metric_value)
            if not groups:
                continue
            canvas.text(52, y, parameter_name, size=12, font="F2")
            y += 18
            for value in sorted(groups, key=sort_category_key)[:8]:
                group = groups[value]
                values = np.asarray(group["values"], dtype=float)
                failed_count = int(group["failed_count"])
                total_count = int(group["total_count"])
                if values.size:
                    stats_text = (
                        f"best={float(np.min(values)):.6g}, "
                        f"mean={float(np.mean(values)):.6g}, "
                        f"n={values.size}"
                    )
                else:
                    stats_text = "no passive trials"
                canvas.text(
                    70,
                    y,
                    f"{value}: {stats_text}, passivity failed/excluded={failed_count}/{total_count}",
                    size=10,
                )
                y += 14
            y += 8
            if y > canvas.height - 42:
                canvases.append(canvas)
                canvas = PdfCanvas(width=930.0, height=625.0)
                y = 42
        canvases.append(canvas)
    save_pdf_pages(path, canvases)


def plot_sweep_diagnostics(
    rows: Sequence[dict[str, object]],
    out_dir: Path,
    swept_columns: Sequence[str],
    selection_metric: str,
    prefix: str,
) -> list[Path]:
    usable_rows = list(rows)
    if not usable_rows or not swept_columns:
        return []
    metric_names = sweep_diagnostic_metric_names(usable_rows, selection_metric)
    if not metric_names:
        return []
    plot_dir = out_dir / "sweep_diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = plot_dir / f"{prefix}_error_vs_swept_parameters.pdf"
    stats_path = plot_dir / f"{prefix}_error_vs_swept_parameters.csv"
    write_sweep_diagnostic_stats(
        stats_path,
        usable_rows,
        swept_columns,
        metric_names,
        selection_metric,
    )
    if not plot_sweep_diagnostics_matplotlib(
        pdf_path,
        usable_rows,
        swept_columns,
        metric_names,
        selection_metric,
    ):
        plot_sweep_diagnostics_fallback_pdf(
            pdf_path,
            usable_rows,
            swept_columns,
            metric_names,
            selection_metric,
        )
    return [pdf_path, stats_path]


def parse_hidden_layers(text: str) -> list[int]:
    layers = [int(part) for part in text.split(",") if part.strip()]
    if any(layer <= 0 for layer in layers):
        raise ValueError("--hidden-layers must contain positive integers")
    return layers


def parse_int_options(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer option")
    return values


def parse_float_options(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one floating-point option")
    return values


def parse_text_options(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one option")
    return values


def parse_hidden_layer_options(text: str) -> list[str]:
    options = [part.strip() for part in text.split(";") if part.strip()]
    if not options:
        raise ValueError("Expected at least one hidden-layer option")
    for option in options:
        parse_hidden_layers(option)
    return options


def summary_metric(summary: dict[str, object], metric_name: str) -> float | None:
    if metric_name.startswith("passivity."):
        passivity = summary.get("passivity")
        if not isinstance(passivity, dict):
            return None
        value = passivity.get(metric_name.split(".", 1)[1])
    else:
        value = summary.get(metric_name)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def add_debug_argument(
    parser: argparse.ArgumentParser,
    help_text: str | None = None,
) -> None:
    parser.add_argument(
        "--debug",
        action="store_true",
        help=help_text
        or "Print common sweep diagnostics and show tracebacks for failed commands/trials.",
    )


def debug_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "debug", False))


def debug_print(
    args: argparse.Namespace,
    message: str,
    label: str | None = None,
) -> None:
    if not debug_enabled(args):
        return
    resolved_label = (
        label
        or getattr(args, "progress_label", None)
        or getattr(args, "debug_label", None)
        or "surrogate"
    )
    print(f"debug: {resolved_label}: {message}", file=sys.stderr, flush=True)


def debug_traceback(args: argparse.Namespace) -> str | None:
    traceback_text = traceback.format_exc()
    if debug_enabled(args):
        print(traceback_text, file=sys.stderr, flush=True)
        return traceback_text
    return None


def print_cli_error(args: argparse.Namespace, exc: Exception) -> None:
    print(f"error: {exc}", file=sys.stderr)
    debug_traceback(args)


def load_or_write_trial_summary(
    summary_path: Path,
    status: int,
    error_message: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, object]:
    if status == 0 and summary_path.exists():
        return json.loads(summary_path.read_text())

    summary: dict[str, object] = {"error": error_message or "trial failed"}
    if traceback_text:
        summary["traceback"] = traceback_text
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def csv_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def training_history_epochs(path: Path) -> int | None:
    if not path.exists():
        return None
    last_epoch: float | None = None
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                epoch = csv_number(row.get("epoch"))
                if epoch is not None:
                    last_epoch = epoch
    except OSError:
        return None
    if last_epoch is None:
        return None
    return int(round(last_epoch))


def sweep_row_metric(row: dict[str, object], metric_name: str) -> float | None:
    if metric_name == "passivity.max_singular_value":
        return csv_number(row.get("passivity_max_singular_value"))
    if metric_name == "passivity.violating_points":
        return csv_number(row.get("passivity_violating_points"))
    return csv_number(row.get(metric_name))


def sweep_row_exclusion_reasons(
    row: dict[str, object],
    selection_metric: str,
    max_passivity_violations: int | None = None,
    max_passivity_sigma: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    metric = sweep_row_metric(row, selection_metric)
    if metric is None:
        reasons.append(f"missing {selection_metric}")
    if max_passivity_violations is not None:
        violations = sweep_row_metric(row, "passivity.violating_points")
        if violations is None or violations > max_passivity_violations:
            reasons.append(f"passivity violations > {max_passivity_violations}")
    if max_passivity_sigma is not None:
        sigma = sweep_row_metric(row, "passivity.max_singular_value")
        if sigma is None or sigma > max_passivity_sigma:
            reasons.append(f"max sigma > {max_passivity_sigma:g}")
    return reasons


def update_sweep_row_from_summary(
    row: dict[str, object],
    summary: dict[str, object],
) -> None:
    for key in [
        "rmse_abs",
        "max_abs",
        "evm_rms",
        "evm_pct",
        "evm_db",
        "weighted_rmse_abs",
        "weighted_max_abs",
        "weighted_evm_rms",
        "weighted_evm_pct",
        "weighted_evm_db",
        "rmse_db",
        "max_abs_db",
        "weighted_rmse_db",
        "weighted_max_abs_db",
    ]:
        value = summary.get(key)
        if value is not None:
            row[key] = value
    passivity = summary.get("passivity")
    if isinstance(passivity, dict):
        row["passivity_max_singular_value"] = passivity.get("max_singular_value")
        row["passivity_violating_points"] = passivity.get("violating_points")


def load_sweep_rows(
    sweep_dir: Path,
    results_filename: str,
) -> list[dict[str, object]]:
    rows = read_csv_rows(sweep_dir / results_filename)
    for row in rows:
        trial_value = csv_number(row.get("trial"))
        if trial_value is None:
            continue
        summary_path = sweep_dir / "trials" / f"trial_{int(trial_value):04d}" / "verification_summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        update_sweep_row_from_summary(row, summary)
    return rows


def rerank_sweep_rows(
    rows: Sequence[dict[str, object]],
    selection_metric: str,
    require_passive: bool = False,
    max_passivity_violations: int | None = None,
    max_passivity_sigma: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None, float | None]:
    if require_passive and max_passivity_violations is None:
        max_passivity_violations = 0

    reranked: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_metric: float | None = None
    for raw_row in rows:
        row = dict(raw_row)
        metric = sweep_row_metric(row, selection_metric)
        row["metric"] = metric
        row["selection_metric"] = selection_metric
        excluded_reasons = sweep_row_exclusion_reasons(
            row,
            selection_metric=selection_metric,
            max_passivity_violations=max_passivity_violations,
            max_passivity_sigma=max_passivity_sigma,
        )
        if excluded_reasons:
            existing_error = str(row.get("error") or "").strip()
            row["error"] = "; ".join(
                [part for part in [existing_error, *excluded_reasons] if part]
            )
        elif best_metric is None or (metric is not None and metric < best_metric):
            best_metric = metric
            best_row = row
        reranked.append(row)
    return reranked, best_row, best_metric


def copy_trial_model(
    sweep_dir: Path,
    trial: int,
    best_model_dir: Path,
    overwrite: bool = False,
) -> tuple[bool, str | None]:
    trial_dir = sweep_dir / "trials" / f"trial_{trial:04d}"
    required_names = ["model.npz", "metadata.json"]
    try:
        trial_metadata = json.loads((trial_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        trial_metadata = {}
    if trial_metadata.get("dc_model_kind"):
        required_names.extend(["dc_model.npz", "dc_model.json"])
    missing = [name for name in required_names if not (trial_dir / name).exists()]
    if missing:
        return (
            False,
            f"Trial {trial} does not contain {', '.join(missing)}. "
            "Re-run that single configuration or run future sweeps with --keep-trial-models.",
        )
    if best_model_dir.exists():
        if not overwrite:
            return False, f"{best_model_dir} already exists; use --overwrite to replace it."
        shutil.rmtree(best_model_dir)
    shutil.copytree(trial_dir, best_model_dir)
    return True, None


def configure_parallel_numeric_threads(jobs: int) -> int:
    jobs = max(1, int(jobs))
    if jobs > 1:
        for name in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ.setdefault(name, "1")
    return jobs


def sweep_trial_seed(base_seed: int, trial_index: int, mode: str = "fixed") -> int:
    normalized = str(mode or "fixed").strip().lower()
    if normalized == "fixed":
        return int(base_seed)
    if normalized == "indexed":
        return int(base_seed) + int(trial_index)
    raise ValueError(f"Unsupported trial seed mode {mode!r}")


def sweep_arg_values(args: argparse.Namespace) -> dict[str, object]:
    values = dict(vars(args))
    values.pop("func", None)
    return values


def sweep_result_row(
    args: argparse.Namespace,
    result: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    summary = dict(result["summary"])  # type: ignore[arg-type]
    row: dict[str, object] = {
        "trial": int(result["trial"]),
        "trial_seed": result.get("trial_seed"),
        "trial_seed_mode": getattr(args, "trial_seed_mode", "fixed"),
        "metric": result.get("metric"),
        "selection_metric": args.selection_metric,
        **candidate,
        "rmse_abs": summary.get("rmse_abs"),
        "max_abs": summary.get("max_abs"),
        "evm_rms": summary.get("evm_rms"),
        "evm_pct": summary.get("evm_pct"),
        "evm_db": summary.get("evm_db"),
        "weighted_rmse_abs": summary.get("weighted_rmse_abs"),
        "weighted_max_abs": summary.get("weighted_max_abs"),
        "weighted_evm_rms": summary.get("weighted_evm_rms"),
        "weighted_evm_pct": summary.get("weighted_evm_pct"),
        "weighted_evm_db": summary.get("weighted_evm_db"),
        "rmse_db": summary.get("rmse_db"),
        "max_abs_db": summary.get("max_abs_db"),
        "weighted_rmse_db": summary.get("weighted_rmse_db"),
        "weighted_max_abs_db": summary.get("weighted_max_abs_db"),
        "worst_case_plots": "; ".join(result.get("plot_paths") or []),
        "error": summary.get("error"),
    }
    passivity = summary.get("passivity")
    if isinstance(passivity, dict):
        row["passivity_max_singular_value"] = passivity.get("max_singular_value")
        row["passivity_violating_points"] = passivity.get("violating_points")
    return row


def run_sweep_command(
    args: argparse.Namespace,
    candidates: Sequence[dict[str, object]],
    *,
    worker_func: Callable[[tuple[dict[str, object], dict[str, object], str, int, int]], dict[str, object]],
    namespace_for_trial_func: Callable[
        [argparse.Namespace, dict[str, object], Path, int, int],
        argparse.Namespace,
    ],
    train_func: Callable[[argparse.Namespace], int],
    result_columns: Sequence[str],
    results_filename: str,
    best_config_filename: str,
    summary_filename: str,
    diagnostics_prefix: str,
    train_command_prefix: Sequence[str] | None = None,
) -> int:
    out_dir = Path(args.out_dir)
    trials_dir = out_dir / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        raise ValueError("No sweep candidates were generated")

    rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    best_metric: float | None = None
    require_passive = bool(getattr(args, "require_passive", False))
    max_passivity_violations = getattr(args, "max_passivity_violations", None)
    max_passivity_sigma = getattr(args, "max_passivity_sigma", None)
    if require_passive and max_passivity_violations is None:
        max_passivity_violations = 0
    best_dir = out_dir / "best_model"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    live_best_trial: int | None = None
    live_promotion_warning: str | None = None
    jobs = configure_parallel_numeric_threads(getattr(args, "jobs", 1))
    if debug_enabled(args):
        debug_label = f"{diagnostics_prefix} sweep"
        debug_print(
            args,
            f"candidates={len(candidates)} jobs={jobs} out_dir={out_dir}",
            label=debug_label,
        )
        if jobs != 1:
            debug_print(
                args,
                "parallel trial debug output may interleave; use --jobs 1 for the cleanest trace",
                label=debug_label,
            )
        for idx, candidate in enumerate(candidates, start=1):
            debug_print(args, f"candidate {idx}: {candidate}", label=debug_label)
    payloads = [
        (sweep_arg_values(args), candidate, str(out_dir), trial_index, args.trial_worst_plots)
        for trial_index, candidate in enumerate(candidates, start=1)
    ]
    trial_width = max(1, len(str(len(candidates))))
    max_epoch_value = csv_number(getattr(args, "epochs", None))
    epoch_width = max(
        len("unknown"),
        len(str(int(max_epoch_value))) if max_epoch_value is not None else 0,
    )
    metric_label = cli_metric_label(args.selection_metric)
    metric_label_width = max(6, len(metric_label))
    metric_width = 12
    passivity_violations_width = 8
    passivity_sigma_width = 12

    def handle_result(result: dict[str, object]) -> None:
        nonlocal best_row, best_metric, live_best_trial, live_promotion_warning
        trial_index = int(result["trial"])
        candidate = dict(result["candidate"])  # type: ignore[arg-type]
        metric_value = result.get("metric")
        row = sweep_result_row(args, result, candidate)
        rows.append(row)
        _, current_best_row, current_best_metric = rerank_sweep_rows(
            rows,
            selection_metric=args.selection_metric,
            require_passive=require_passive,
            max_passivity_violations=max_passivity_violations,
            max_passivity_sigma=max_passivity_sigma,
        )
        if current_best_row is not None and current_best_metric is not None:
            best_row = current_best_row
            best_metric = current_best_metric
            best_trial_value = csv_number(current_best_row.get("trial"))
            if best_trial_value is not None:
                current_best_trial = int(best_trial_value)
                if current_best_trial != live_best_trial or not (best_dir / "model.npz").exists():
                    promoted, warning = copy_trial_model(
                        out_dir,
                        current_best_trial,
                        best_dir,
                        overwrite=True,
                    )
                    if promoted:
                        live_best_trial = current_best_trial
                        live_promotion_warning = None
                    else:
                        live_promotion_warning = warning
                        print(f"warning: {warning}", file=sys.stderr, flush=True)
        metric_display = metric_text_fixed(metric_value, 2) if metric_value is not None else "failed"
        epochs_display = training_history_epochs(
            trials_dir / f"trial_{trial_index:04d}" / "training_history.csv"
        )
        epoch_text = str(epochs_display if epochs_display is not None else "unknown")
        passivity_violations = metric_text(row.get("passivity_violating_points")) or "n/a"
        passivity_max_sigma = metric_text(row.get("passivity_max_singular_value")) or "n/a"
        line = (
            f"trial complete {trial_index:>{trial_width}}/{len(candidates):>{trial_width}} "
            f"ep={epoch_text:>{epoch_width}} "
            f"{metric_label:>{metric_label_width}}={metric_display:>{metric_width}} "
            f"pv={passivity_violations:>{passivity_violations_width}} "
            f"sigma={passivity_max_sigma:>{passivity_sigma_width}}"
        )
        trial_error = str(row.get("error") or "").strip()
        exclusion_reasons = sweep_row_exclusion_reasons(
            row,
            selection_metric=args.selection_metric,
            max_passivity_violations=max_passivity_violations,
            max_passivity_sigma=max_passivity_sigma,
        )
        failed = bool(trial_error) or bool(exclusion_reasons)
        if failed:
            reason = trial_error or "; ".join(exclusion_reasons)
            suffix = f"err={compact_cli_text(reason, 36)}"
            available = max(0, 116 - len(line) - 1)
            if available >= 5:
                line = f"{line} {compact_cli_text(suffix, available)}"
        print(
            cli_color_text(line, "red" if failed else "green"),
            flush=True,
        )
        cleanup_trial_dir(trials_dir / f"trial_{trial_index:04d}", args.keep_trial_models)

    if jobs == 1:
        for payload in payloads:
            handle_result(worker_func(payload))
    else:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
                future_payloads = {
                    executor.submit(worker_func, payload): payload
                    for payload in payloads
                }
                for future in concurrent.futures.as_completed(future_payloads):
                    payload = future_payloads[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        summary: dict[str, object] = {"error": str(exc)}
                        traceback_text = debug_traceback(args)
                        if traceback_text:
                            summary["traceback"] = traceback_text
                        result = {
                            "trial": payload[3],
                            "candidate": payload[1],
                            "summary": summary,
                            "metric": None,
                            "plot_paths": [],
                        }
                    handle_result(result)
        except OSError as exc:
            print(f"warning: parallel sweep unavailable ({exc}); retrying serially", file=sys.stderr)
            rows.clear()
            best_row = None
            best_metric = float("inf")
            jobs = 1
            for payload in payloads:
                handle_result(worker_func(payload))

    rows.sort(key=lambda row: int(row["trial"]))
    rows, best_row, best_metric = rerank_sweep_rows(
        rows,
        selection_metric=args.selection_metric,
        require_passive=require_passive,
        max_passivity_violations=max_passivity_violations,
        max_passivity_sigma=max_passivity_sigma,
    )
    write_csv(out_dir / results_filename, rows)
    if best_row is None or best_metric is None:
        raise ValueError("No successful sweep trial satisfied the selected metric and passivity criteria")
    best_candidate = {
        key: best_row[key]
        for key in result_columns
        if key in best_row and best_row[key] not in {None, ""}
    }

    best_args = namespace_for_trial_func(
        args,
        best_candidate,
        best_dir,
        int(best_row["trial"]),
        plots=args.worst_plots,
    )
    original_label = getattr(best_args, "progress_label", None)
    if original_label:
        best_args.progress_label = f"best_model from {original_label}"
    best_model_source = "promoted_trial"
    final_best_trial = int(best_row["trial"])
    retrain_best = bool(getattr(args, "retrain_best", False))
    if retrain_best:
        if best_dir.exists():
            shutil.rmtree(best_dir)
        train_func(best_args)
        best_model_source = "retrained"
        live_promotion_warning = None
    elif live_best_trial != final_best_trial or not (best_dir / "model.npz").exists():
        promoted, warning = copy_trial_model(out_dir, final_best_trial, best_dir, overwrite=True)
        if promoted:
            live_best_trial = final_best_trial
            live_promotion_warning = None
        else:
            live_promotion_warning = warning
            if best_dir.exists():
                shutil.rmtree(best_dir)
            print(
                f"warning: {warning}; retraining best_model instead",
                file=sys.stderr,
                flush=True,
            )
            train_func(best_args)
            best_model_source = "retrained_after_promotion_failure"
    reproduction_command = (
        single_model_train_command(
            train_command_prefix,
            best_args,
            out_dir / "reproduced_model",
        )
        if train_command_prefix
        else None
    )
    (out_dir / best_config_filename).write_text(
        json.dumps(
            {
                "selection_metric": args.selection_metric,
                "require_passive": require_passive,
                "max_passivity_violations": max_passivity_violations,
                "max_passivity_sigma": max_passivity_sigma,
                "metric": best_metric,
                "trial": best_row["trial"],
                "trial_seed": best_args.seed,
                "trial_seed_mode": getattr(args, "trial_seed_mode", "fixed"),
                "config": best_candidate,
                "best_model_dir": str(best_dir),
                "best_model_source": best_model_source,
                "promotion_warning": live_promotion_warning,
                "reproduction_command": reproduction_command,
            },
            indent=2,
        )
    )
    diagnostic_artifacts = [
        str(path.relative_to(out_dir))
        for path in plot_sweep_diagnostics(
            rows,
            out_dir,
            result_columns,
            args.selection_metric,
            prefix=diagnostics_prefix,
        )
    ]
    write_sweep_markdown(
        out_dir / summary_filename,
        rows,
        selection_metric=args.selection_metric,
        best_config=best_candidate,
        best_metric=best_metric,
        reproduction_command=reproduction_command,
        diagnostic_artifacts=diagnostic_artifacts,
    )
    if reproduction_command:
        print("reproduce best model:", flush=True)
        print(reproduction_command, flush=True)
    return 0
