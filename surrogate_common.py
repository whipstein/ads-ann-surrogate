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
    """Parse ADS/base-unit-to-model-unit parameter scales.

    The returned scale is the ADS-facing parameter value per training-model
    unit. During Verilog-A export the model feature is computed as:

        model_value = ads_instance_parameter / scale

    For example, if the MDIF used W=0.4 to mean 0.4 um but ADS will pass W in
    meters, use W=1um.
    """

    names = list(parameter_names)
    scales = {name: 1.0 for name in names}
    if not spec:
        return scales

    text = spec.strip()
    if not text:
        return scales
    if "=" not in text:
        scale = parse_scale_number(text)
        return {name: scale for name in names}

    lookup: dict[str, str] = {}
    for name in names:
        keys = {
            name,
            name.lower(),
            normalize_name(name),
            normalize_name(name).lower(),
        }
        for key in keys:
            lookup.setdefault(key, name)

    assigned: set[str] = set()
    for raw_part in re.split(r"[;,]", text):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                "Parameter scale mappings must use NAME=SCALE entries, "
                f"got {part!r}"
            )
        raw_name, raw_value = part.split("=", 1)
        key = raw_name.strip()
        if not key:
            raise ValueError(f"Parameter scale mapping is missing a parameter name: {part!r}")
        scale = parse_scale_number(raw_value)
        if key.strip().lower() in {"*", "all"}:
            for name in names:
                scales[name] = scale
            continue
        target = (
            lookup.get(key)
            or lookup.get(key.lower())
            or lookup.get(normalize_name(key))
            or lookup.get(normalize_name(key).lower())
        )
        if target is None:
            raise ValueError(
                f"Unknown parameter {key!r} in scale spec. Available parameters: "
                + ", ".join(names)
            )
        if target in assigned:
            raise ValueError(f"Duplicate scale mapping for parameter {target!r}")
        scales[target] = scale
        assigned.add(target)
    return scales


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


def build_ads_export_blocks(
    template_mdif: str | None,
    parameter_grid_specs: Sequence[str],
    freqs_spec: str | None,
    parameter_names: Sequence[str],
    sparam_labels: Sequence[str],
) -> list[MDIFBlock]:
    if template_mdif:
        return ensure_block_sparams(read_mdif(Path(template_mdif)), sparam_labels)
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
    return blocks


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
    manifest: dict[str, object] = {
        "format": "ads_sparameter_mdif_surrogate",
        "model_kind": model_kind,
        "model_dir": str(model_dir),
        "mdif": mdif_name,
        "parameter_names": list(parameter_names),
        "sparam_labels": list(sparam_labels),
        "blocks": len(blocks),
        "frequency_points_per_block": int(len(blocks[0].freq_hz)) if blocks else 0,
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
    output_domain: str,
    folded_input_scaler: bool,
    folded_output_scaler: bool,
    uses_coarse_inputs: bool,
    adds_coarse_to_output: bool,
    embedded_coarse_model: bool,
    extra_notes: Sequence[str] | None,
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
    notes = "\n".join(f"- {note}" for note in (extra_notes or []))
    if notes:
        notes = "\n\nNotes:\n\n" + notes + "\n"
    if output_domain == "y":
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
the scale at `1.0`. If the MDIF used dimensionless micron values and ADS uses
meters, export with a scale such as `--parameter-input-scales W=1um,L=1um`.
The scale means "ADS/base-unit value per one model-training unit".

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
) -> tuple[str, dict[str, object]]:
    output_domain = output_domain.lower().strip()
    if output_domain not in {"s", "y"}:
        raise ValueError("Verilog-A output_domain must be 's' or 'y'")
    if output_domain == "y" and (uses_coarse_inputs or adds_coarse_to_output):
        raise ValueError("Direct-Y Verilog-A export is currently supported only without coarse hooks")
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
    ]
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
    for layer_idx, size in enumerate(layer_sizes):
        lines.append(f"  real l{layer_idx} [0:{size - 1}];")

    lines.extend(["", "  analog begin"])
    lines.append(f"    freq_hz = {frequency_expression};")
    lines.append("    if (clamp_frequency != 0 && freq_hz < min_frequency_hz) freq_hz = min_frequency_hz;")
    lines.append("    freq_log10_hz = log(freq_hz)/log(10.0);")
    lines.append("")

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

        lines.append(f"    for (i = 0; i < {matrix_size}; i = i + 1) begin")
        lines.append("      ar[i] = sr[i];")
        lines.append("      ai[i] = si[i];")
        lines.append("      mr[i] = -sr[i];")
        lines.append("      mi[i] = -si[i];")
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
        lines.append("          tr = ar[idx]; ti = ai[idx]; ar[idx] = ar[k]; ai[idx] = ai[k]; ar[k] = tr; ai[k] = ti;")
        lines.append("          tr = invr[idx]; ti = invi[idx]; invr[idx] = invr[k]; invi[idx] = invi[k]; invr[k] = tr; invi[k] = ti;")
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
        lines.append("        yr[idx] = mr[i]*invr[j] - mi[i]*invi[j];")
        lines.append("        yi[idx] = mr[i]*invi[j] + mi[i]*invr[j];")
        lines.append(f"        for (k = 1; k < {nports}; k = k + 1) begin")
        lines.append(f"          i = row*{nports} + k;")
        lines.append(f"          j = k*{nports} + col;")
        lines.append("          yr[idx] = yr[idx] + (mr[i]*invr[j] - mi[i]*invi[j]);")
        lines.append("          yi[idx] = yi[idx] + (mr[i]*invi[j] + mi[i]*invr[j]);")
        lines.append("        end")
        lines.append("        yr[idx] = yr[idx]/z0;")
        lines.append("        yi[idx] = yi[idx]/z0;")
        lines.append("      end")
        lines.append("    end")
        lines.append("")
    lines.append("    omega = 6.2831853071795864769*freq_hz;")
    lines.append("    if (omega < 1.0e-30) omega = 1.0e-30;")
    for row, port_i in enumerate(port_ids):
        for col, port_j in enumerate(port_ids):
            flat = row * nports + col
            real_expr = (
                direct_y_real_by_flat[flat]
                if output_domain == "y"
                else f"yr[{flat}]"
            )
            imag_expr = (
                direct_y_imag_by_flat[flat]
                if output_domain == "y"
                else f"yi[{flat}]"
            )
            if real_expr is None or imag_expr is None:
                raise ValueError("Internal error: direct-Y output mapping is incomplete")
            lines.append(
                f"    I({port_i}) <+ ({real_expr})*V({port_j}) + "
                f"(({imag_expr})/omega)*ddt(V({port_j}));"
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
        value = block.params.get(split_key)
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

            train_loss = mse(self.predict(x_train), y_train, output_weights=output_weights)
            val_loss = (
                mse(self.predict(x_val), y_val, output_weights=output_weights)
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


def mse(
    pred: np.ndarray | None,
    truth: np.ndarray | None,
    output_weights: np.ndarray | None = None,
) -> float:
    if pred is None or truth is None:
        return float("nan")
    err2 = (pred - truth) ** 2
    if output_weights is not None:
        err2 = err2 * np.asarray(output_weights, dtype=float)[None, :]
    return float(np.mean(err2))


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
        sys.stderr.write("\r\033[2K" + " ".join(parts))
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
            all_weights.append(np.full_like(abs_err, weight, dtype=float))
            all_normalized_weights.append(np.full_like(abs_err, normalized_weight, dtype=float))
            if db_err.size:
                all_db_errors.append(db_err)
                all_normalized_db_weights.append(
                    np.full_like(db_err, normalized_weight, dtype=float)
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
            "weighted_evm_definition": "sqrt(sum(weight[Sij]*|pred-truth|^2) / sum(weight[Sij]*|truth|^2))",
            "sparam_weights": {label: float(label_weights.get(label, 1.0)) for label in labels},
            "normalized_sparam_weights": normalized_label_weights,
            "sparam_weight_mean": weight_mean,
            "sparam_weight_normalization": "Raw S-parameter weights are divided by their mean before training and scale-sensitive weighted RMSE/MAE metrics, so the average normalized weight is 1.0.",
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
) -> dict[str, object]:
    write_mdif(out_dir / "predicted_verification.mdif", pred_blocks, labels)
    metric_rows, summary = verification_metrics(
        truth_blocks,
        pred_blocks,
        labels,
        parameter_names,
        sparam_weights=sparam_weights,
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
        "predicted_verification.mdif",
        "training_history.csv",
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
    return " ".join(shlex.quote(part) for part in argv)


def write_training_markdown(
    path: Path,
    model_kind: str,
    config: dict[str, object],
    summary: dict[str, object],
    history: Sequence[dict[str, float]],
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
        ("Model weights", "model.npz"),
        ("Metadata", "metadata.json"),
        ("Training history", "training_history.csv"),
        ("Verification summary JSON", "verification_summary.json"),
    ]
    if (path.parent / "training_history.pdf").exists():
        artifacts.insert(3, ("Training loss plot", "training_history.pdf"))
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
    missing = [
        name
        for name in ["model.npz", "metadata.json"]
        if not (trial_dir / name).exists()
    ]
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
