#!/usr/bin/env python3
"""Generate and adapt geometry/process points for surrogate-model extraction.

The default method is a maximin Latin hypercube because finite EM sample sets
usually benefit from stratification plus good point separation. Sobol is also
available when SciPy is installed. After a fit, GP-UCB acquisition can use a
Matérn-5/2 posterior over geometry-level model error to select the next EM
batch while retaining the original error-distance selector as an alternative.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cli_options import (
    add_options_json_argument,
    finalize_options_json_update,
    parse_args_with_options_json,
)


UNIT_SCALES = {
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
    "hz": 1.0,
    "khz": 1e3,
    "mhz": 1e6,
    "ghz": 1e9,
    "thz": 1e12,
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    unit: str
    scale: str = "linear"


@dataclass(frozen=True)
class RangeExtensionPlan:
    parameter_name: str
    side: str
    original_parameters: list[ParameterSpec]
    overall_parameters: list[ParameterSpec]
    sampling_parameters: list[ParameterSpec]
    added_volume_ratio: float


@dataclass
class ErrorRegion:
    source_index: str
    unit_point: list[float]
    score: float
    worst_sparam: str
    worst_sparam_score: float
    row_count: int


@dataclass
class SuggestedPoint:
    unit_point: list[float]
    acquisition_score: float
    distance_to_existing: float
    nearest_error_source_index: str
    nearest_error_score: float
    nearest_error_distance: float
    predicted_error: float | None = None
    gp_log_uncertainty: float | None = None
    gp_upper_confidence_error: float | None = None


@dataclass
class GaussianProcessModel:
    observation_points: list[list[float]]
    log_error_mean: float
    log_error_scale: float
    length_scale: float
    noise_variance: float
    cholesky: list[list[float]]
    alpha: list[float]
    log_marginal_likelihood: float


def split_value_token(token: object) -> tuple[float, str]:
    text = str(token).strip()
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)",
        text,
    )
    if not match:
        raise ValueError(f"Could not parse numeric value {token!r}")
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit not in UNIT_SCALES:
        raise ValueError(f"Unsupported unit suffix {unit!r} in {token!r}")
    return value, unit


def parse_value_token(token: object) -> tuple[float, str]:
    value, unit = split_value_token(token)
    return value * UNIT_SCALES[unit], unit


def parse_observed_value(
    token: object,
    spec: ParameterSpec,
    bare_values: str = "parameter-units",
) -> float:
    value, unit = split_value_token(token)
    if unit:
        return value * UNIT_SCALES[unit]
    if bare_values == "parameter-units" and spec.unit:
        return value * UNIT_SCALES[spec.unit]
    return value


def parse_parameter_spec(raw: str) -> ParameterSpec:
    if "=" not in raw:
        raise ValueError(f"Parameter spec must look like NAME=LOW:HIGH[:linear|log], got {raw!r}")
    name, body = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Parameter spec is missing a name: {raw!r}")

    parts = [part.strip() for part in body.split(":")]
    scale = "linear"
    if len(parts) >= 3 and parts[-1].lower() in {"linear", "log"}:
        scale = parts.pop(-1).lower()
    if len(parts) != 2:
        raise ValueError(f"Parameter spec must look like NAME=LOW:HIGH[:linear|log], got {raw!r}")

    lower, lower_unit = parse_value_token(parts[0])
    upper, upper_unit = parse_value_token(parts[1])
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"Parameter bounds must be finite: {raw!r}")
    if upper <= lower:
        raise ValueError(f"Upper bound must be greater than lower bound: {raw!r}")
    if scale == "log" and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(f"Log-scaled bounds must be positive: {raw!r}")

    output_unit = lower_unit if lower_unit == upper_unit else ""
    return ParameterSpec(name=name, lower=lower, upper=upper, unit=output_unit, scale=scale)


def parse_range_factor(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        raise ValueError(f"Range factor must look like NAME=FACTOR, got {raw!r}")
    name, factor_text = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Range factor is missing a parameter name: {raw!r}")
    try:
        factor = float(factor_text.strip())
    except ValueError as exc:
        raise ValueError(f"Range factor must be numeric in {raw!r}") from exc
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError(f"Range factor must be finite and greater than 1 in {raw!r}")
    return name, factor


def apply_range_factors(
    parameters: Sequence[ParameterSpec],
    raw_factors: Sequence[str],
) -> list[ParameterSpec]:
    by_name: dict[str, ParameterSpec] = {}
    for parameter in parameters:
        if parameter.name in by_name:
            raise ValueError(f"Parameter {parameter.name!r} was specified more than once")
        by_name[parameter.name] = parameter

    factors: dict[str, float] = {}
    for raw in raw_factors:
        name, factor = parse_range_factor(raw)
        if name not in by_name:
            raise ValueError(
                f"Range factor refers to unknown parameter {name!r}; define it "
                "with --parameter or in the selected parameter metadata JSON"
            )
        if name in factors:
            raise ValueError(f"Range factor for parameter {name!r} was specified more than once")
        factors[name] = factor

    expanded: list[ParameterSpec] = []
    for parameter in parameters:
        factor = factors.get(parameter.name)
        if factor is None:
            expanded.append(parameter)
            continue
        if parameter.scale == "log":
            lower = math.log(parameter.lower)
            upper = math.log(parameter.upper)
            center = 0.5 * (lower + upper)
            half_span = 0.5 * (upper - lower) * factor
            try:
                new_lower = math.exp(center - half_span)
                new_upper = math.exp(center + half_span)
            except OverflowError as exc:
                raise ValueError(
                    f"Range factor for parameter {parameter.name!r} produces non-finite bounds"
                ) from exc
        else:
            center = 0.5 * (parameter.lower + parameter.upper)
            half_span = 0.5 * (parameter.upper - parameter.lower) * factor
            new_lower = center - half_span
            new_upper = center + half_span
        if not math.isfinite(new_lower) or not math.isfinite(new_upper):
            raise ValueError(
                f"Range factor for parameter {parameter.name!r} produces non-finite bounds"
            )
        expanded.append(
            ParameterSpec(
                name=parameter.name,
                lower=new_lower,
                upper=new_upper,
                unit=parameter.unit,
                scale=parameter.scale,
            )
        )
    return expanded


def parameter_span(spec: ParameterSpec) -> float:
    if spec.scale == "log":
        return math.log(spec.upper) - math.log(spec.lower)
    return spec.upper - spec.lower


def build_range_extension_plan(
    parameters: Sequence[ParameterSpec],
    raw_extension: str,
) -> RangeExtensionPlan:
    by_name = {parameter.name: parameter for parameter in parameters}
    if len(by_name) != len(parameters):
        raise ValueError("Each --parameter name must be unique")

    requested = parse_parameter_spec(raw_extension)
    original = by_name.get(requested.name)
    if original is None:
        raise ValueError(
            f"Range extension refers to unknown parameter {requested.name!r}; "
            "add its original range with --parameter first"
        )

    extension_body = raw_extension.split("=", 1)[1]
    extension_parts = [part.strip() for part in extension_body.split(":")]
    explicit_scale = (
        extension_parts[-1].lower()
        if extension_parts and extension_parts[-1].lower() in {"linear", "log"}
        else None
    )
    if explicit_scale is not None and explicit_scale != original.scale:
        raise ValueError(
            f"Range extension scale for {original.name!r} must remain {original.scale!r}"
        )
    requested = ParameterSpec(
        name=requested.name,
        lower=requested.lower,
        upper=requested.upper,
        unit=original.unit,
        scale=original.scale,
    )
    if requested.scale == "log" and requested.lower <= 0.0:
        raise ValueError(f"Extended log range for {requested.name!r} must remain positive")

    lower_same = math.isclose(requested.lower, original.lower, rel_tol=1e-10, abs_tol=1e-18)
    upper_same = math.isclose(requested.upper, original.upper, rel_tol=1e-10, abs_tol=1e-18)
    lower_extension = requested.lower < original.lower and upper_same
    upper_extension = lower_same and requested.upper > original.upper
    if not lower_extension and not upper_extension:
        raise ValueError(
            f"--extend-range must keep one {original.name!r} bound unchanged and move "
            "the other outward; the new range must contain the original range"
        )

    if lower_extension:
        side = "lower"
        slab = ParameterSpec(
            original.name,
            requested.lower,
            original.lower,
            original.unit,
            original.scale,
        )
    else:
        side = "upper"
        slab = ParameterSpec(
            original.name,
            original.upper,
            requested.upper,
            original.unit,
            original.scale,
        )

    overall_parameters: list[ParameterSpec] = []
    sampling_parameters: list[ParameterSpec] = []
    for parameter in parameters:
        if parameter.name == original.name:
            overall_parameters.append(requested)
            sampling_parameters.append(slab)
        else:
            overall_parameters.append(parameter)
            sampling_parameters.append(parameter)

    return RangeExtensionPlan(
        parameter_name=original.name,
        side=side,
        original_parameters=list(parameters),
        overall_parameters=overall_parameters,
        sampling_parameters=sampling_parameters,
        added_volume_ratio=parameter_span(slab) / parameter_span(original),
    )


def parameter_decimal_grid(
    parameter: ParameterSpec,
    decimal_places: int,
) -> tuple[int, int, int]:
    unit_scale = UNIT_SCALES.get(parameter.unit, 1.0)
    decimal_scale = 10**decimal_places
    lower_grid = parameter.lower / unit_scale * decimal_scale
    upper_grid = parameter.upper / unit_scale * decimal_scale
    lower_tolerance = 1e-12 * max(1.0, abs(lower_grid))
    upper_tolerance = 1e-12 * max(1.0, abs(upper_grid))
    first = math.ceil(lower_grid - lower_tolerance)
    last = math.floor(upper_grid + upper_tolerance)
    return first, last, decimal_scale


def round_parameter_value(
    value: float,
    parameter: ParameterSpec,
    decimal_places: int | None,
) -> float:
    if decimal_places is None:
        return value
    first, last, decimal_scale = parameter_decimal_grid(parameter, decimal_places)
    unit_scale = UNIT_SCALES.get(parameter.unit, 1.0)
    grid_value = round(value / unit_scale * decimal_scale)
    grid_value = min(last, max(first, grid_value))
    return grid_value / decimal_scale * unit_scale


def format_value(
    value: float,
    unit: str,
    decimal_places: int | None = None,
) -> str:
    scale = UNIT_SCALES.get(unit, 1.0)
    if decimal_places is not None:
        text = f"{value / scale:.{decimal_places}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text == "-0":
            text = "0"
        return f"{text}{unit}"
    return f"{value / scale:.12g}{unit}"


def map_unit_point(value: float, spec: ParameterSpec) -> float:
    if spec.scale == "log":
        log_lo = math.log(spec.lower)
        log_hi = math.log(spec.upper)
        return math.exp(log_lo + value * (log_hi - log_lo))
    return spec.lower + value * (spec.upper - spec.lower)


def unit_coordinate_for_value(value: float, spec: ParameterSpec) -> float:
    if spec.scale == "log":
        return (math.log(value) - math.log(spec.lower)) / (math.log(spec.upper) - math.log(spec.lower))
    return (value - spec.lower) / (spec.upper - spec.lower)


def latin_hypercube_points(count: int, dimensions: int, rng: random.Random) -> list[list[float]]:
    points = [[0.0 for _ in range(dimensions)] for _ in range(count)]
    for dim in range(dimensions):
        values = [(idx + rng.random()) / count for idx in range(count)]
        rng.shuffle(values)
        for row, value in enumerate(values):
            points[row][dim] = value
    return points


def min_pairwise_distance(
    points: Sequence[Sequence[float]],
    abandon_at_or_below: float | None = None,
) -> float:
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for idx, lhs in enumerate(points[:-1]):
        for rhs in points[idx + 1 :]:
            distance2 = sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs))
            if distance2 < best:
                best = distance2
                if (
                    abandon_at_or_below is not None
                    and math.sqrt(best) <= abandon_at_or_below
                ):
                    return math.sqrt(best)
    return math.sqrt(best)


def maximin_lhs_points(
    count: int,
    dimensions: int,
    rng: random.Random,
    candidates: int,
) -> list[list[float]]:
    best_points: list[list[float]] | None = None
    best_score = -1.0
    for _ in range(max(1, candidates)):
        trial = latin_hypercube_points(count, dimensions, rng)
        score = min_pairwise_distance(
            trial,
            abandon_at_or_below=best_score if best_score >= 0.0 else None,
        )
        if score > best_score:
            best_score = score
            best_points = trial
    assert best_points is not None
    return best_points


def first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        root = int(math.sqrt(candidate))
        if all(candidate % prime for prime in primes if prime <= root):
            primes.append(candidate)
        candidate += 1
    return primes


def radical_inverse(index: int, base: int) -> float:
    value = 0.0
    inv_base = 1.0 / base
    factor = inv_base
    while index > 0:
        digit = index % base
        value += digit * factor
        index //= base
        factor *= inv_base
    return value


def halton_points(count: int, dimensions: int, skip: int) -> list[list[float]]:
    bases = first_primes(dimensions)
    return [
        [radical_inverse(row + skip + 1, base) for base in bases]
        for row in range(count)
    ]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def sobol_points(
    count: int,
    dimensions: int,
    seed: int,
    scramble: bool,
    skip: int,
) -> list[list[float]]:
    try:
        from scipy.stats import qmc  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Sobol generation requires scipy. Install scipy or use "
            "--method maximin-lhs, --method latin-hypercube, or --method halton."
        ) from exc

    sampler = qmc.Sobol(d=dimensions, scramble=scramble, seed=seed)
    if skip > 0:
        sampler.fast_forward(skip)
    if skip == 0 and is_power_of_two(count):
        points = sampler.random_base2(int(math.log2(count)))
    else:
        if not is_power_of_two(count):
            warnings.warn(
                "Sobol balance properties are best when --count is a power of two.",
                RuntimeWarning,
                stacklevel=2,
            )
        points = sampler.random(count)
    return [[float(value) for value in row] for row in points]


def generate_unit_points(
    method: str,
    count: int,
    dimensions: int,
    seed: int,
    lhs_candidates: int,
    scramble: bool,
    skip: int,
) -> list[list[float]]:
    if method == "minimax-lhs":
        method = "maximin-lhs"
    rng = random.Random(seed)
    if method == "latin-hypercube":
        return latin_hypercube_points(count, dimensions, rng)
    if method == "maximin-lhs":
        return maximin_lhs_points(count, dimensions, rng, lhs_candidates)
    if method == "halton":
        return halton_points(count, dimensions, skip)
    if method == "sobol":
        return sobol_points(count, dimensions, seed, scramble, skip)
    raise ValueError(f"Unsupported method {method!r}")


def safe_method_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", method).strip("_")


def output_path_for_method(path: Path, method: str, multiple_methods: bool) -> Path:
    if not multiple_methods:
        return path
    safe = safe_method_name(method)
    if "{method}" in str(path):
        return Path(str(path).replace("{method}", safe))
    return path.with_name(f"{path.stem}_{safe}{path.suffix or '.csv'}")


def write_rows_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def geometry_metadata_path(path: Path) -> Path:
    metadata_path = path.with_suffix(".json")
    if metadata_path == path:
        return path.with_name(f"{path.stem}_metadata.json")
    return metadata_path


def geometry_coverage_plot_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_parameter_coverage.png")


def coverage_axis_label(parameter: ParameterSpec) -> str:
    details = [value for value in (parameter.unit, parameter.scale if parameter.scale == "log" else "") if value]
    return (
        f"{parameter.name} ({', '.join(details)})"
        if details
        else parameter.name
    )


def coverage_tick_label(parameter: ParameterSpec, coordinate: float) -> str:
    value = map_unit_point(coordinate, parameter)
    unit_scale = UNIT_SCALES.get(parameter.unit, 1.0)
    displayed = value / unit_scale
    if math.isclose(displayed, 0.0, abs_tol=1.0e-15):
        displayed = 0.0
    return f"{displayed:.4g}"


def coverage_split_group(value: object) -> str:
    token = normalize_key(value)
    if token in {"verification", "verify", "validation", "test"}:
        return "verification"
    return "training"


def write_parameter_coverage_png(
    geometry_path: Path,
    parameters: Sequence[ParameterSpec],
    rows: Sequence[dict[str, object]],
    split_var: str,
    *,
    bare_values: str = "parameter-units",
) -> Path:
    """Write a PNG scatter/histogram matrix for generated geometry points."""

    if not parameters:
        raise ValueError("A parameter coverage plot requires at least one parameter")
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "PNG parameter coverage plots require Pillow. Install it with "
            "'python3 -m pip install pillow'."
        ) from exc

    grouped_points: dict[str, list[list[float]]] = {
        "training": [],
        "verification": [],
    }
    for row_index, row in enumerate(rows, start=1):
        coordinates: list[float] = []
        for parameter in parameters:
            raw_value = lookup_row_value(row, parameter.name)
            if raw_value is None or str(raw_value).strip() == "":
                raise ValueError(
                    f"Could not plot geometry row {row_index}: missing parameter "
                    f"{parameter.name!r}"
                )
            try:
                value = parse_observed_value(
                    raw_value,
                    parameter,
                    bare_values=bare_values,
                )
                coordinate = unit_coordinate_for_value(value, parameter)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Could not plot geometry row {row_index} parameter "
                    f"{parameter.name!r}: {raw_value!r}"
                ) from exc
            if not math.isfinite(coordinate) or not -1.0e-8 <= coordinate <= 1.0 + 1.0e-8:
                raise ValueError(
                    f"Could not plot geometry row {row_index}: parameter "
                    f"{parameter.name!r} is outside the declared range"
                )
            coordinates.append(min(1.0, max(0.0, coordinate)))
        split_value = lookup_row_value(row, split_var) or "train"
        grouped_points[coverage_split_group(split_value)].append(coordinates)

    dimension_count = len(parameters)
    render_scale = 2

    def px(value: float) -> int:
        return int(round(render_scale * value))

    cell_size = px(174)
    left_margin = px(105)
    right_margin = px(24)
    top_margin = px(104)
    bottom_margin = px(60)
    width = max(px(520), left_margin + dimension_count * cell_size + right_margin)
    height = top_margin + dimension_count * cell_size + bottom_margin
    plot_inset_left = px(20)
    plot_inset_right = px(10)
    plot_inset_top = px(10)
    plot_inset_bottom = px(25)
    training_color = (37, 99, 235, 255)
    verification_color = (249, 115, 22, 255)
    background_color = (248, 250, 252, 255)
    cell_color = (255, 255, 255, 255)
    border_color = (203, 213, 225, 255)
    grid_color = (226, 232, 240, 255)
    axis_color = (148, 163, 184, 255)
    title_color = (15, 23, 42, 255)
    text_color = (51, 65, 85, 255)
    tick_color = (100, 116, 139, 255)
    output_path = geometry_coverage_plot_path(geometry_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        title_font = ImageFont.load_default(size=px(19))
        label_font = ImageFont.load_default(size=px(11))
        tick_font = ImageFont.load_default(size=px(9))
    except TypeError:  # Pillow versions before scalable built-in fonts.
        title_font = ImageFont.load_default()
        label_font = title_font
        tick_font = title_font

    image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text(
        (left_margin, px(12)),
        f"Parameter coverage: {geometry_path.name}",
        font=title_font,
        fill=title_color,
    )
    draw.text(
        (left_margin, px(42)),
        f"{len(grouped_points['training'])} training point(s), "
        f"{len(grouped_points['verification'])} verification point(s)",
        font=label_font,
        fill=tick_color,
    )
    draw.ellipse(
        (left_margin - px(4), px(68), left_margin + px(4), px(76)),
        fill=training_color,
    )
    draw.text(
        (left_margin + px(10), px(66)),
        "Training",
        font=label_font,
        fill=text_color,
    )
    draw.ellipse(
        (
            left_margin + px(78),
            px(68),
            left_margin + px(86),
            px(76),
        ),
        fill=verification_color,
    )
    draw.text(
        (left_margin + px(92), px(66)),
        "Verification",
        font=label_font,
        fill=text_color,
    )

    for column, parameter in enumerate(parameters):
        center_x = left_margin + column * cell_size + cell_size / 2
        draw.text(
            (center_x, px(92)),
            coverage_axis_label(parameter),
            font=label_font,
            fill=title_color,
            anchor="ms",
        )
    for row_index, parameter in enumerate(parameters):
        center_y = top_margin + row_index * cell_size + cell_size / 2
        draw.text(
            (left_margin - px(11), center_y),
            coverage_axis_label(parameter),
            font=label_font,
            fill=title_color,
            anchor="rm",
        )

    for row_index, y_parameter in enumerate(parameters):
        for column_index, x_parameter in enumerate(parameters):
            cell_x = left_margin + column_index * cell_size
            cell_y = top_margin + row_index * cell_size
            plot_left = cell_x + plot_inset_left
            plot_right = cell_x + cell_size - plot_inset_right
            plot_top = cell_y + plot_inset_top
            plot_bottom = cell_y + cell_size - plot_inset_bottom
            plot_width = plot_right - plot_left
            plot_height = plot_bottom - plot_top
            draw.rectangle(
                (cell_x, cell_y, cell_x + cell_size, cell_y + cell_size),
                fill=cell_color,
                outline=border_color,
                width=px(1),
            )
            draw.line(
                (
                    plot_left + plot_width / 2,
                    plot_top,
                    plot_left + plot_width / 2,
                    plot_bottom,
                ),
                fill=grid_color,
                width=px(1),
            )
            draw.line(
                (
                    plot_left,
                    plot_top + plot_height / 2,
                    plot_right,
                    plot_top + plot_height / 2,
                ),
                fill=grid_color,
                width=px(1),
            )
            draw.rectangle(
                (plot_left, plot_top, plot_right, plot_bottom),
                outline=axis_color,
                width=px(1),
            )

            if row_index == column_index:
                largest_group = max(len(points) for points in grouped_points.values())
                bin_count = max(5, min(18, int(math.ceil(math.sqrt(max(1, largest_group))))))
                histograms: dict[str, list[int]] = {}
                maximum_count = 1
                for group_name, points in grouped_points.items():
                    counts = [0] * bin_count
                    for point in points:
                        bin_index = min(
                            bin_count - 1,
                            int(point[column_index] * bin_count),
                        )
                        counts[bin_index] += 1
                    histograms[group_name] = counts
                    maximum_count = max(maximum_count, *counts)
                bar_width = plot_width / bin_count
                for group_name, color in (
                    ("training", training_color),
                    ("verification", verification_color),
                ):
                    histogram_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
                    histogram_draw = ImageDraw.Draw(histogram_layer)
                    for bin_index, count in enumerate(histograms[group_name]):
                        if count == 0:
                            continue
                        bar_height = plot_height * count / maximum_count
                        histogram_draw.rectangle(
                            (
                                plot_left + bin_index * bar_width + px(1),
                                plot_bottom - bar_height,
                                plot_left + (bin_index + 1) * bar_width - px(1),
                                plot_bottom,
                            ),
                            fill=(*color[:3], 132),
                        )
                    image = Image.alpha_composite(image, histogram_layer)
                    draw = ImageDraw.Draw(image)
            else:
                for group_name, color in (
                    ("training", training_color),
                    ("verification", verification_color),
                ):
                    for point in grouped_points[group_name]:
                        point_x = plot_left + point[column_index] * plot_width
                        point_y = plot_bottom - point[row_index] * plot_height
                        draw.ellipse(
                            (
                                point_x - px(3),
                                point_y - px(3),
                                point_x + px(3),
                                point_y + px(3),
                            ),
                            fill=color,
                            outline=(255, 255, 255, 255),
                            width=px(1),
                        )

            if row_index == dimension_count - 1:
                draw.text(
                    (plot_left, cell_y + cell_size - px(9)),
                    coverage_tick_label(x_parameter, 0.0),
                    font=tick_font,
                    fill=tick_color,
                    anchor="ls",
                )
                draw.text(
                    (plot_right, cell_y + cell_size - px(9)),
                    coverage_tick_label(x_parameter, 1.0),
                    font=tick_font,
                    fill=tick_color,
                    anchor="rs",
                )
            if column_index == 0 and row_index != column_index:
                draw.text(
                    (plot_left - px(4), plot_bottom),
                    coverage_tick_label(y_parameter, 0.0),
                    font=tick_font,
                    fill=tick_color,
                    anchor="rm",
                )
                draw.text(
                    (plot_left - px(4), plot_top),
                    coverage_tick_label(y_parameter, 1.0),
                    font=tick_font,
                    fill=tick_color,
                    anchor="rm",
                )

    image.convert("RGB").save(output_path, format="PNG", optimize=True, dpi=(144, 144))
    return output_path


def metadata_number(value: float) -> float:
    return float(f"{value:.12g}")


def parameter_range_metadata(parameter: ParameterSpec) -> dict[str, object]:
    unit_scale = UNIT_SCALES.get(parameter.unit, 1.0)
    return {
        "name": parameter.name,
        "range": {
            "lower": metadata_number(parameter.lower / unit_scale),
            "upper": metadata_number(parameter.upper / unit_scale),
            "unit": parameter.unit,
        },
        "base_unit_range": {
            "lower": metadata_number(parameter.lower),
            "upper": metadata_number(parameter.upper),
        },
        "scale": parameter.scale,
    }


def write_geometry_metadata(
    geometry_path: Path,
    parameters: Sequence[ParameterSpec],
    rows: Sequence[dict[str, object]],
    split_var: str,
    generation_kind: str,
    method: str,
    extra: dict[str, object] | None = None,
    decimal_places: int | None = None,
    bare_values: str = "parameter-units",
) -> Path:
    split_counts: dict[str, int] = {}
    for row in rows:
        split_value = str(row.get(split_var, "")).strip()
        if split_value:
            split_counts[split_value] = split_counts.get(split_value, 0) + 1

    metadata: dict[str, object] = {
        "schema_version": 1,
        "geometry_file": geometry_path.name,
        "generation_kind": generation_kind,
        "method": method,
        "point_count": len(rows),
        "split_variable": split_var,
        "split_counts": split_counts,
        "parameters": [parameter_range_metadata(parameter) for parameter in parameters],
    }
    if decimal_places is not None:
        metadata["decimal_places"] = decimal_places
    if extra:
        metadata.update(extra)

    coverage_plot_path = write_parameter_coverage_png(
        geometry_path,
        parameters,
        rows,
        split_var,
        bare_values=bare_values,
    )
    metadata["parameter_coverage_plot"] = coverage_plot_path.name

    metadata_path = geometry_metadata_path(geometry_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def parameter_specs_from_geometry_metadata(path: Path) -> list[ParameterSpec]:
    if not path.exists():
        raise ValueError(f"Parameter metadata JSON does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read parameter metadata JSON {path}: {exc}") from exc
    raw_parameters = payload.get("parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError(
            f"Parameter metadata JSON {path} has no non-empty 'parameters' list"
        )

    parameters: list[ParameterSpec] = []
    seen: set[str] = set()
    for index, raw_parameter in enumerate(raw_parameters, start=1):
        if not isinstance(raw_parameter, dict):
            raise ValueError(
                f"Parameter entry {index} in {path} must be a JSON object"
            )
        name = str(raw_parameter.get("name") or "").strip()
        if not name:
            raise ValueError(f"Parameter entry {index} in {path} has no name")
        if name in seen:
            raise ValueError(f"Parameter {name!r} appears more than once in {path}")
        seen.add(name)
        raw_range = raw_parameter.get("range")
        if not isinstance(raw_range, dict):
            raise ValueError(f"Parameter {name!r} in {path} has no range object")
        try:
            lower_declared = float(raw_range["lower"])
            upper_declared = float(raw_range["upper"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Parameter {name!r} in {path} needs numeric range.lower and range.upper"
            ) from exc
        unit = str(raw_range.get("unit") or "").strip().lower()
        if unit not in UNIT_SCALES:
            raise ValueError(
                f"Parameter {name!r} in {path} uses unsupported unit {unit!r}"
            )
        scale = str(raw_parameter.get("scale") or "linear").strip().lower()
        if scale not in {"linear", "log"}:
            raise ValueError(
                f"Parameter {name!r} in {path} has unsupported scale {scale!r}"
            )
        lower = lower_declared * UNIT_SCALES[unit]
        upper = upper_declared * UNIT_SCALES[unit]
        if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
            raise ValueError(
                f"Parameter {name!r} in {path} needs finite ordered bounds"
            )
        if scale == "log" and lower <= 0.0:
            raise ValueError(
                f"Log-scaled parameter {name!r} in {path} needs positive bounds"
            )
        parameters.append(
            ParameterSpec(
                name=name,
                lower=lower,
                upper=upper,
                unit=unit,
                scale=scale,
            )
        )
    return parameters


def companion_geometry_metadata_candidates(csv_path: Path) -> list[Path]:
    candidates = [geometry_metadata_path(csv_path)]
    for suffix in ("_train", "_verification"):
        if csv_path.stem.endswith(suffix):
            combined_path = csv_path.with_name(
                f"{csv_path.stem[:-len(suffix)]}{csv_path.suffix or '.csv'}"
            )
            candidates.append(geometry_metadata_path(combined_path))
    return list(dict.fromkeys(candidates))


def parameter_specs_equal(
    lhs: Sequence[ParameterSpec],
    rhs: Sequence[ParameterSpec],
) -> bool:
    if len(lhs) != len(rhs):
        return False
    for left, right in zip(lhs, rhs):
        if (
            left.name != right.name
            or left.unit != right.unit
            or left.scale != right.scale
            or not math.isclose(left.lower, right.lower, rel_tol=1e-10, abs_tol=1e-15)
            or not math.isclose(left.upper, right.upper, rel_tol=1e-10, abs_tol=1e-15)
        ):
            return False
    return True


def generated_point_rows(
    method: str,
    unit_points: Sequence[Sequence[float]],
    parameters: Sequence[ParameterSpec],
    verification_count: int,
    split_var: str,
    include_normalized: bool,
    decimal_places: int | None = None,
) -> tuple[list[dict[str, object]], list[str]]:
    fields = [
        "point_index",
        split_var,
        "split_sequence",
        "train_sequence",
        "verification_sequence",
        "method",
    ]
    if include_normalized:
        fields.extend(f"u_{parameter.name}" for parameter in parameters)
    fields.extend(parameter.name for parameter in parameters)

    train_count = len(unit_points) - verification_count
    rows: list[dict[str, object]] = []
    for idx, point in enumerate(unit_points, start=1):
        is_train = idx <= train_count
        split_sequence = idx if is_train else idx - train_count
        row: dict[str, object] = {
            "point_index": idx,
            split_var: "train" if is_train else "verification",
            "split_sequence": split_sequence,
            "train_sequence": split_sequence if is_train else "",
            "verification_sequence": "" if is_train else split_sequence,
            "method": method,
        }
        rounded_values = [
            round_parameter_value(
                map_unit_point(unit_value, parameter),
                parameter,
                decimal_places,
            )
            for parameter, unit_value in zip(parameters, point)
        ]
        if include_normalized:
            for parameter, value, original_unit_value in zip(
                parameters,
                rounded_values,
                point,
            ):
                unit_value = (
                    original_unit_value
                    if decimal_places is None
                    else unit_coordinate_for_value(value, parameter)
                )
                row[f"u_{parameter.name}"] = f"{unit_value:.16g}"
        for parameter, value in zip(parameters, rounded_values):
            row[parameter.name] = format_value(value, parameter.unit, decimal_places)
        rows.append(row)
    return rows, fields


def split_output_path(path: Path, split_name: str) -> Path:
    return path.with_name(f"{path.stem}_{safe_method_name(split_name)}{path.suffix or '.csv'}")


def write_points_csv(
    path: Path,
    method: str,
    unit_points: Sequence[Sequence[float]],
    parameters: Sequence[ParameterSpec],
    verification_count: int,
    split_var: str,
    include_normalized: bool,
    write_split_files: bool,
    decimal_places: int | None = None,
) -> list[Path]:
    rows, fields = generated_point_rows(
        method,
        unit_points,
        parameters,
        verification_count,
        split_var,
        include_normalized,
        decimal_places,
    )
    write_rows_csv(path, rows, fields)
    metadata_path = write_geometry_metadata(
        path,
        parameters,
        rows,
        split_var,
        generation_kind="generated",
        method=method,
        decimal_places=decimal_places,
    )
    written = [path, metadata_path, geometry_coverage_plot_path(path)]
    if write_split_files:
        split_values = ["train", "verification"] if verification_count else ["train"]
        for split_value in split_values:
            split_rows = [row for row in rows if row.get(split_var) == split_value]
            split_path = split_output_path(path, split_value)
            write_rows_csv(split_path, split_rows, fields)
            written.append(split_path)
    return written


def read_csv_table(path: Path) -> tuple[list[str], list[dict[str, object]]]:
    if not path.exists():
        raise ValueError(f"Existing-points CSV does not exist: {path}")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    if not fields:
        raise ValueError(f"Existing-points CSV has no header: {path}")
    if not rows:
        raise ValueError(f"Existing-points CSV has no data rows: {path}")
    return fields, rows


def range_extension_recommendation(
    rows: Sequence[dict[str, object]],
    split_var: str,
    dimensions: int,
    added_volume_ratio: float,
) -> tuple[int, int]:
    def stable_ceil(value: float) -> int:
        tolerance = 1e-12 * max(1.0, abs(value))
        return int(math.ceil(value - tolerance))

    verification_rows = sum(
        str(lookup_row_value(row, split_var) or "train").strip().lower() == "verification"
        for row in rows
    )
    training_rows = len(rows) - verification_rows
    recommended_training = max(
        stable_ceil(training_rows * added_volume_ratio),
        4 * dimensions,
    )
    recommended_verification = 0
    if verification_rows:
        recommended_verification = max(
            stable_ceil(verification_rows * added_volume_ratio),
            2 * dimensions,
        )
    return recommended_training, recommended_verification


def validate_existing_parameter_rows(
    rows: Sequence[dict[str, object]],
    parameters: Sequence[ParameterSpec],
    bare_values: str,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        for parameter in parameters:
            raw = lookup_row_value(row, parameter.name)
            if raw is None or str(raw).strip() == "":
                raise ValueError(
                    f"Existing-points row {row_index} is missing parameter {parameter.name!r}"
                )
            try:
                value = parse_observed_value(raw, parameter, bare_values=bare_values)
                coordinate = unit_coordinate_for_value(value, parameter)
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Could not parse parameter {parameter.name!r} in existing-points "
                    f"row {row_index}: {raw!r}"
                ) from exc
            if not math.isfinite(coordinate) or not -1e-9 <= coordinate <= 1.0 + 1e-9:
                raise ValueError(
                    f"Existing-points row {row_index} value {raw!r} for {parameter.name!r} "
                    "is outside its original --parameter range"
                )


def write_range_extension_csv(
    path: Path,
    existing_fields: Sequence[str],
    existing_rows: Sequence[dict[str, object]],
    method: str,
    unit_points: Sequence[Sequence[float]],
    plan: RangeExtensionPlan,
    verification_count: int,
    split_var: str,
    include_normalized: bool,
    bare_values: str,
    write_split_files: bool,
    decimal_places: int | None = None,
) -> list[Path]:
    rows = [dict(row) for row in existing_rows]
    train_count = len(unit_points) - verification_count
    for offset, point in enumerate(unit_points):
        is_train = offset < train_count
        row: dict[str, object] = {
            split_var: "train" if is_train else "verification",
            "method": f"range-extension-{method}",
        }
        for parameter, unit_value in zip(plan.sampling_parameters, point):
            value = round_parameter_value(
                map_unit_point(unit_value, parameter),
                parameter,
                decimal_places,
            )
            row[parameter.name] = format_value(
                value,
                parameter.unit,
                decimal_places,
            )
        rows.append(row)

    normalized_fields = [f"u_{parameter.name}" for parameter in plan.overall_parameters]
    include_any_normalized = include_normalized or any(
        field in existing_fields for field in normalized_fields
    )
    split_counts: dict[str, int] = {}
    for point_index, row in enumerate(rows, start=1):
        split_value = str(lookup_row_value(row, split_var) or "train").strip() or "train"
        if split_value.lower() in {"train", "verification"}:
            split_value = split_value.lower()
        row[split_var] = split_value
        split_counts[split_value] = split_counts.get(split_value, 0) + 1
        split_sequence = split_counts[split_value]
        row["point_index"] = point_index
        row["split_sequence"] = split_sequence
        row["train_sequence"] = split_sequence if split_value.lower() == "train" else ""
        row["verification_sequence"] = (
            split_sequence if split_value.lower() == "verification" else ""
        )
        if include_any_normalized:
            for parameter in plan.overall_parameters:
                raw = lookup_row_value(row, parameter.name)
                assert raw is not None
                value = parse_observed_value(raw, parameter, bare_values=bare_values)
                coordinate = unit_coordinate_for_value(value, parameter)
                row[f"u_{parameter.name}"] = f"{coordinate:.16g}"

    canonical_fields = [
        "point_index",
        split_var,
        "split_sequence",
        "train_sequence",
        "verification_sequence",
        "method",
    ]
    if include_any_normalized:
        canonical_fields.extend(normalized_fields)
    canonical_fields.extend(parameter.name for parameter in plan.overall_parameters)
    fields = list(dict.fromkeys([*canonical_fields, *existing_fields]))
    write_rows_csv(path, rows, fields)
    extension_metadata: dict[str, object] = {
        "range_extension": {
            "parameter": plan.parameter_name,
            "side": plan.side,
            "added_volume_ratio": metadata_number(plan.added_volume_ratio),
            "original_point_count": len(existing_rows),
            "added_point_count": len(unit_points),
            "original_parameters": [
                parameter_range_metadata(parameter)
                for parameter in plan.original_parameters
            ],
            "new_point_sampling_parameters": [
                parameter_range_metadata(parameter)
                for parameter in plan.sampling_parameters
            ],
        }
    }
    metadata_path = write_geometry_metadata(
        path,
        plan.overall_parameters,
        rows,
        split_var,
        generation_kind="range_extension",
        method=method,
        decimal_places=decimal_places,
        bare_values=bare_values,
        extra=extension_metadata,
    )
    written = [path, metadata_path, geometry_coverage_plot_path(path)]
    if write_split_files:
        for split_value in split_counts:
            split_rows = [
                row for row in rows if str(row.get(split_var, "")) == split_value
            ]
            split_path = split_output_path(path, split_value)
            write_rows_csv(split_path, split_rows, fields)
            written.append(split_path)
    return written


def normalize_key(name: object) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip()).strip("_").lower()


def lookup_row_value(row: dict[str, object], name: str) -> object | None:
    if name in row:
        return row[name]
    wanted = normalize_key(name)
    for key, value in row.items():
        if normalize_key(key) == wanted:
            return value
    return None


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


def row_unit_point(
    row: dict[str, object],
    parameters: Sequence[ParameterSpec],
    bare_values: str,
) -> list[float] | None:
    point: list[float] = []
    for parameter in parameters:
        raw = lookup_row_value(row, parameter.name)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = parse_observed_value(raw, parameter, bare_values=bare_values)
            unit_value = unit_coordinate_for_value(value, parameter)
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(unit_value):
            return None
        point.append(unit_value)
    return point


def in_unit_cube(point: Sequence[float], tolerance: float = 1e-9) -> bool:
    return all(-tolerance <= value <= 1.0 + tolerance for value in point)


def clamp_unit_point(point: Sequence[float]) -> list[float]:
    return [min(1.0, max(0.0, float(value))) for value in point]


def squared_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs))


def point_distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return math.sqrt(squared_distance(lhs, rhs))


def nearest_distance(point: Sequence[float], others: Sequence[Sequence[float]]) -> float:
    if not others:
        return math.sqrt(len(point))
    return min(point_distance(point, other) for other in others)


def dedupe_points(points: Sequence[Sequence[float]], tolerance: float = 1e-10) -> list[list[float]]:
    seen: set[tuple[int, ...]] = set()
    unique: list[list[float]] = []
    scale = 1.0 / tolerance
    for point in points:
        key = tuple(int(round(value * scale)) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append([float(value) for value in point])
    return unique


def parse_var_assignment(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text.upper().startswith("VAR"):
        return None
    body = text[3:].strip()
    if not body:
        return None
    if "=" in body:
        name, value = body.split("=", 1)
    else:
        parts = body.split(None, 1)
        if len(parts) != 2:
            return None
        name, value = parts
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return name.strip(), value


def read_mdif_parameter_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pending: dict[str, object] = {}
    current: dict[str, object] | None = None
    in_block = False
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.split("!", 1)[0].strip()
        if not line:
            continue
        parsed = parse_var_assignment(line)
        if parsed:
            name, value = parsed
            target = current if in_block and current is not None else pending
            target[name] = value
            continue
        upper = line.upper()
        if upper.startswith("BEGIN"):
            in_block = True
            current = dict(pending)
            continue
        if upper.startswith("END") and in_block:
            if current is not None:
                rows.append(current)
            current = None
            in_block = False
    return rows


def load_existing_points(
    csv_paths: Sequence[str],
    mdif_paths: Sequence[str],
    parameters: Sequence[ParameterSpec],
    bare_values: str,
) -> list[list[float]]:
    points: list[list[float]] = []
    for raw_path in csv_paths:
        path = Path(raw_path)
        for row in read_csv_rows(path):
            point = row_unit_point(row, parameters, bare_values=bare_values)
            if point is not None and in_unit_cube(point):
                points.append(clamp_unit_point(point))
    for raw_path in mdif_paths:
        path = Path(raw_path)
        for row in read_mdif_parameter_rows(path):
            point = row_unit_point(row, parameters, bare_values=bare_values)
            if point is not None and in_unit_cube(point):
                points.append(clamp_unit_point(point))
    return dedupe_points(points)


def metric_score_value(metric_name: str, value: float) -> float:
    lowered = metric_name.lower()
    if lowered in {"evm_db", "weighted_evm_db"}:
        return 10.0 ** (value / 20.0)
    return abs(value)


def load_error_regions(
    metrics_path: Path,
    parameters: Sequence[ParameterSpec],
    metric_name: str,
    bare_values: str,
) -> tuple[list[ErrorRegion], str]:
    rows = read_csv_rows(metrics_path)
    if not rows:
        raise ValueError(f"{metrics_path} is empty")
    if metric_name == "auto":
        for candidate in ["evm_pct", "rmse_abs", "max_abs", "rmse_db", "max_abs_db"]:
            if any(str(row.get(candidate) or "").strip() for row in rows):
                metric_name = candidate
                break
        else:
            raise ValueError(f"{metrics_path} does not contain a usable error metric")

    groups: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        point = row_unit_point(row, parameters, bare_values=bare_values)
        value = csv_number(row.get(metric_name))
        if point is None or value is None or not in_unit_cube(point):
            continue
        point = clamp_unit_point(point)
        source_index = str(row.get("source_index") or "")
        key = (source_index, *(round(coord, 12) for coord in point))
        bucket = groups.setdefault(
            key,
            {
                "source_index": source_index,
                "point": point,
                "weighted_sum": 0.0,
                "weight_sum": 0.0,
                "worst_sparam": "",
                "worst_score": -1.0,
                "row_count": 0,
            },
        )
        score = metric_score_value(metric_name, value)
        weight = csv_number(row.get("normalized_sparam_weight"))
        if weight is None or weight <= 0.0:
            weight = csv_number(row.get("sparam_weight")) or 1.0
        bucket["weighted_sum"] = float(bucket["weighted_sum"]) + weight * score * score
        bucket["weight_sum"] = float(bucket["weight_sum"]) + weight
        bucket["row_count"] = int(bucket["row_count"]) + 1
        if score > float(bucket["worst_score"]):
            bucket["worst_score"] = score
            bucket["worst_sparam"] = str(row.get("sparam") or "")

    regions: list[ErrorRegion] = []
    for bucket in groups.values():
        weight_sum = float(bucket["weight_sum"])
        if weight_sum <= 0.0:
            continue
        regions.append(
            ErrorRegion(
                source_index=str(bucket["source_index"]),
                unit_point=list(bucket["point"]),  # type: ignore[arg-type]
                score=math.sqrt(float(bucket["weighted_sum"]) / weight_sum),
                worst_sparam=str(bucket["worst_sparam"]),
                worst_sparam_score=float(bucket["worst_score"]),
                row_count=int(bucket["row_count"]),
            )
        )
    regions.sort(key=lambda region: region.score, reverse=True)
    if not regions:
        raise ValueError(
            f"No rows in {metrics_path} had {metric_name!r} and all requested parameters"
        )
    return regions, metric_name


def error_region_fields(parameters: Sequence[ParameterSpec]) -> list[str]:
    return [
        "rank",
        "source_index",
        "error_score",
        "worst_sparam",
        "worst_sparam_score",
        "metric_rows",
        *[parameter.name for parameter in parameters],
    ]


def write_error_regions_csv(
    path: Path,
    regions: Sequence[ErrorRegion],
    parameters: Sequence[ParameterSpec],
) -> None:
    rows: list[dict[str, object]] = []
    for rank, region in enumerate(regions, start=1):
        row: dict[str, object] = {
            "rank": rank,
            "source_index": region.source_index,
            "error_score": f"{region.score:.12g}",
            "worst_sparam": region.worst_sparam,
            "worst_sparam_score": f"{region.worst_sparam_score:.12g}",
            "metric_rows": region.row_count,
        }
        for parameter, unit_value in zip(parameters, region.unit_point):
            row[parameter.name] = format_value(map_unit_point(unit_value, parameter), parameter.unit)
        rows.append(row)
    write_rows_csv(path, rows, error_region_fields(parameters))


def analysis_output_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_fit_error_regions{out_path.suffix or '.csv'}")


def candidate_focus(
    point: Sequence[float],
    regions: Sequence[ErrorRegion],
    focus_radius: float,
    focus_power: float,
) -> tuple[float, ErrorRegion | None, float]:
    if not regions:
        return 1.0, None, math.sqrt(len(point))
    sigma2 = max(focus_radius, 1e-12) ** 2
    total = 0.0
    best_contribution = -1.0
    best_region: ErrorRegion | None = None
    best_distance = math.sqrt(len(point))
    for region in regions:
        distance2 = squared_distance(point, region.unit_point)
        weight = max(region.score, 0.0) ** focus_power
        contribution = weight * math.exp(-0.5 * distance2 / sigma2)
        total += contribution
        if contribution > best_contribution:
            best_contribution = contribution
            best_region = region
            best_distance = math.sqrt(distance2)
    return total, best_region, best_distance


def _matern52_kernel(
    lhs: Sequence[float],
    rhs: Sequence[float],
    length_scale: float,
) -> float:
    distance2 = sum(
        ((float(a) - float(b)) / length_scale) ** 2
        for a, b in zip(lhs, rhs)
    )
    if distance2 <= 0.0:
        return 1.0
    distance = math.sqrt(distance2)
    root_five_distance = math.sqrt(5.0) * distance
    return (
        1.0 + root_five_distance + (5.0 / 3.0) * distance2
    ) * math.exp(-root_five_distance)


def _cholesky_factor(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    size = len(matrix)
    factor = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(row + 1):
            value = float(matrix[row][col]) - sum(
                factor[row][idx] * factor[col][idx] for idx in range(col)
            )
            if row == col:
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("Gaussian-process covariance is not positive definite")
                factor[row][col] = math.sqrt(value)
            else:
                factor[row][col] = value / factor[col][col]
    return factor


def _solve_lower(factor: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    result: list[float] = []
    for row in range(len(factor)):
        value = float(values[row]) - sum(
            float(factor[row][col]) * result[col] for col in range(row)
        )
        result.append(value / float(factor[row][row]))
    return result


def _solve_upper_from_lower(
    factor: Sequence[Sequence[float]], values: Sequence[float]
) -> list[float]:
    size = len(factor)
    result = [0.0] * size
    for row in range(size - 1, -1, -1):
        value = float(values[row]) - sum(
            float(factor[col][row]) * result[col]
            for col in range(row + 1, size)
        )
        result[row] = value / float(factor[row][row])
    return result


def _unique_gp_observations(
    regions: Sequence[ErrorRegion],
    error_floor: float,
) -> tuple[list[list[float]], list[float]]:
    grouped: dict[tuple[float, ...], tuple[list[float], float]] = {}
    for region in regions:
        key = tuple(round(float(value), 12) for value in region.unit_point)
        score = max(float(region.score), error_floor)
        previous = grouped.get(key)
        if previous is None or score > previous[1]:
            grouped[key] = (list(region.unit_point), score)
    points = [value[0] for value in grouped.values()]
    log_errors = [math.log(value[1]) for value in grouped.values()]
    return points, log_errors


def _factor_gp_candidate(
    points: Sequence[Sequence[float]],
    normalized_targets: Sequence[float],
    length_scale: float,
    noise_variance: float,
) -> tuple[list[list[float]], list[float], float]:
    covariance = [
        [
            _matern52_kernel(lhs, rhs, length_scale)
            + (noise_variance if row == col else 0.0)
            for col, rhs in enumerate(points)
        ]
        for row, lhs in enumerate(points)
    ]
    jitter = max(1e-12, noise_variance * 1e-6)
    for _ in range(8):
        try:
            factor = _cholesky_factor(covariance)
            break
        except ValueError:
            for idx in range(len(covariance)):
                covariance[idx][idx] += jitter
            jitter *= 10.0
    else:
        raise ValueError(
            "Could not stabilize the Gaussian-process covariance; increase "
            "--gp-noise-variance"
        )
    intermediate = _solve_lower(factor, normalized_targets)
    alpha = _solve_upper_from_lower(factor, intermediate)
    quadratic = sum(
        float(value) * coefficient
        for value, coefficient in zip(normalized_targets, alpha)
    )
    log_determinant_half = sum(math.log(row[idx]) for idx, row in enumerate(factor))
    log_marginal_likelihood = (
        -0.5 * quadratic
        - log_determinant_half
        - 0.5 * len(points) * math.log(2.0 * math.pi)
    )
    return factor, alpha, log_marginal_likelihood


def fit_error_gaussian_process(
    regions: Sequence[ErrorRegion],
    length_scale: float | None,
    noise_variance: float,
    error_floor: float,
) -> GaussianProcessModel:
    points, log_errors = _unique_gp_observations(regions, error_floor)
    if len(points) < 2:
        raise ValueError(
            "GP-UCB acquisition requires errors at at least two distinct geometries"
        )
    log_error_mean = sum(log_errors) / len(log_errors)
    variance = sum((value - log_error_mean) ** 2 for value in log_errors) / max(
        1, len(log_errors) - 1
    )
    # Retain meaningful posterior uncertainty when the first few measured
    # errors happen to be almost equal. A 0.25 natural-log scale corresponds
    # to about a 28% multiplicative one-sigma uncertainty.
    log_error_scale = max(math.sqrt(max(variance, 0.0)), 0.25)
    normalized_targets = [
        (value - log_error_mean) / log_error_scale for value in log_errors
    ]
    candidates = (
        [float(length_scale)]
        if length_scale is not None
        else [0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.35]
    )
    best: tuple[list[list[float]], list[float], float, float] | None = None
    for candidate in candidates:
        factor, alpha, likelihood = _factor_gp_candidate(
            points,
            normalized_targets,
            candidate,
            noise_variance,
        )
        if best is None or likelihood > best[2]:
            best = (factor, alpha, likelihood, candidate)
    assert best is not None
    return GaussianProcessModel(
        observation_points=points,
        log_error_mean=log_error_mean,
        log_error_scale=log_error_scale,
        length_scale=best[3],
        noise_variance=noise_variance,
        cholesky=best[0],
        alpha=best[1],
        log_marginal_likelihood=best[2],
    )


def predict_error_gaussian_process(
    model: GaussianProcessModel,
    point: Sequence[float],
) -> tuple[float, float]:
    covariance = [
        _matern52_kernel(point, observed, model.length_scale)
        for observed in model.observation_points
    ]
    normalized_mean = sum(
        value * coefficient for value, coefficient in zip(covariance, model.alpha)
    )
    projected = _solve_lower(model.cholesky, covariance)
    normalized_variance = max(
        0.0,
        1.0 - sum(value * value for value in projected),
    )
    mean_log_error = (
        model.log_error_mean + model.log_error_scale * normalized_mean
    )
    std_log_error = model.log_error_scale * math.sqrt(normalized_variance)
    return mean_log_error, std_log_error


def _nearest_error_region(
    point: Sequence[float], regions: Sequence[ErrorRegion]
) -> tuple[ErrorRegion | None, float]:
    if not regions:
        return None, math.sqrt(len(point))
    region = min(
        regions,
        key=lambda value: squared_distance(point, value.unit_point),
    )
    return region, math.sqrt(squared_distance(point, region.unit_point))


def select_gp_ucb_points(
    candidate_points: Sequence[Sequence[float]],
    regions: Sequence[ErrorRegion],
    existing_points: Sequence[Sequence[float]],
    count: int,
    exploration_weight: float,
    novelty_power: float,
    min_distance: float,
    length_scale: float | None,
    noise_variance: float,
    error_floor: float,
) -> tuple[list[SuggestedPoint], GaussianProcessModel]:
    model = fit_error_gaussian_process(
        regions,
        length_scale=length_scale,
        noise_variance=noise_variance,
        error_floor=error_floor,
    )
    selected: list[SuggestedPoint] = []
    occupied = [list(point) for point in existing_points]
    diag = max(
        math.sqrt(len(candidate_points[0])) if candidate_points else 1.0,
        1e-12,
    )
    unused: list[tuple[list[float], float, float, float]] = []
    for raw_point in candidate_points:
        point = list(raw_point)
        mean_log_error, std_log_error = predict_error_gaussian_process(model, point)
        predicted_error = math.exp(min(700.0, mean_log_error))
        upper_error = math.exp(
            min(700.0, mean_log_error + exploration_weight * std_log_error)
        )
        unused.append((point, predicted_error, std_log_error, upper_error))

    while len(selected) < count and unused:
        best_idx: int | None = None
        best_point: SuggestedPoint | None = None
        for idx, (point, predicted_error, std_log_error, upper_error) in enumerate(unused):
            distance_existing = nearest_distance(point, occupied)
            if distance_existing < min_distance:
                continue
            novelty = min(1.0, distance_existing / diag)
            acquisition = upper_error * (max(novelty, 1e-12) ** novelty_power)
            region, region_distance = _nearest_error_region(point, regions)
            candidate = SuggestedPoint(
                unit_point=point,
                acquisition_score=acquisition,
                distance_to_existing=distance_existing,
                nearest_error_source_index=(region.source_index if region else ""),
                nearest_error_score=(region.score if region else 0.0),
                nearest_error_distance=region_distance,
                predicted_error=predicted_error,
                gp_log_uncertainty=std_log_error,
                gp_upper_confidence_error=upper_error,
            )
            if best_point is None or candidate.acquisition_score > best_point.acquisition_score:
                best_idx = idx
                best_point = candidate
        if best_idx is None or best_point is None:
            break
        selected.append(best_point)
        occupied.append(best_point.unit_point)
        unused.pop(best_idx)
    return selected, model


def select_targeted_points(
    candidate_points: Sequence[Sequence[float]],
    regions: Sequence[ErrorRegion],
    existing_points: Sequence[Sequence[float]],
    count: int,
    focus_radius: float,
    focus_power: float,
    novelty_power: float,
    min_distance: float,
) -> list[SuggestedPoint]:
    selected: list[SuggestedPoint] = []
    occupied = [list(point) for point in existing_points]
    unused = [list(point) for point in candidate_points]
    diag = max(math.sqrt(len(candidate_points[0])) if candidate_points else 1.0, 1e-12)

    while len(selected) < count and unused:
        best_idx: int | None = None
        best_point: SuggestedPoint | None = None
        for idx, point in enumerate(unused):
            distance_existing = nearest_distance(point, occupied)
            if distance_existing < min_distance:
                continue
            focus, region, region_distance = candidate_focus(
                point,
                regions,
                focus_radius=focus_radius,
                focus_power=focus_power,
            )
            novelty = min(1.0, distance_existing / diag)
            acquisition = focus * (max(novelty, 1e-12) ** novelty_power)
            source_index = region.source_index if region is not None else ""
            source_score = region.score if region is not None else 0.0
            candidate = SuggestedPoint(
                unit_point=point,
                acquisition_score=acquisition,
                distance_to_existing=distance_existing,
                nearest_error_source_index=source_index,
                nearest_error_score=source_score,
                nearest_error_distance=region_distance,
            )
            if best_point is None or candidate.acquisition_score > best_point.acquisition_score:
                best_idx = idx
                best_point = candidate
        if best_idx is None or best_point is None:
            break
        selected.append(best_point)
        occupied.append(best_point.unit_point)
        unused.pop(best_idx)
    return selected


def write_suggested_points_csv(
    path: Path,
    suggestions: Sequence[SuggestedPoint],
    parameters: Sequence[ParameterSpec],
    split_var: str,
    target_dataset: str,
    candidate_method: str,
    acquisition_method: str,
    metric_name: str,
    include_normalized: bool,
    decimal_places: int | None = None,
    acquisition_metadata: dict[str, object] | None = None,
) -> Path:
    method_name = (
        f"targeted-{candidate_method}"
        if acquisition_method == "error-distance"
        else f"{acquisition_method}-{candidate_method}"
    )
    fields = [
        "point_index",
        split_var,
        "additional_sequence",
        "method",
        "acquisition_method",
        "analysis_metric",
        "fit_error_score",
        "nearest_error_source_index",
        "nearest_error_distance",
        "distance_to_existing",
        "acquisition_score",
        "predicted_error",
        "gp_log_uncertainty",
        "gp_upper_confidence_error",
    ]
    if include_normalized:
        fields.extend(f"u_{parameter.name}" for parameter in parameters)
    fields.extend(parameter.name for parameter in parameters)

    rows: list[dict[str, object]] = []
    for idx, suggestion in enumerate(suggestions, start=1):
        row: dict[str, object] = {
            "point_index": idx,
            split_var: target_dataset,
            "additional_sequence": idx,
            "method": method_name,
            "acquisition_method": acquisition_method,
            "analysis_metric": metric_name,
            "fit_error_score": f"{suggestion.nearest_error_score:.12g}",
            "nearest_error_source_index": suggestion.nearest_error_source_index,
            "nearest_error_distance": f"{suggestion.nearest_error_distance:.12g}",
            "distance_to_existing": f"{suggestion.distance_to_existing:.12g}",
            "acquisition_score": f"{suggestion.acquisition_score:.12g}",
            "predicted_error": (
                ""
                if suggestion.predicted_error is None
                else f"{suggestion.predicted_error:.12g}"
            ),
            "gp_log_uncertainty": (
                ""
                if suggestion.gp_log_uncertainty is None
                else f"{suggestion.gp_log_uncertainty:.12g}"
            ),
            "gp_upper_confidence_error": (
                ""
                if suggestion.gp_upper_confidence_error is None
                else f"{suggestion.gp_upper_confidence_error:.12g}"
            ),
        }
        rounded_values = [
            round_parameter_value(
                map_unit_point(unit_value, parameter),
                parameter,
                decimal_places,
            )
            for parameter, unit_value in zip(parameters, suggestion.unit_point)
        ]
        if include_normalized:
            for parameter, value, original_unit_value in zip(
                parameters,
                rounded_values,
                suggestion.unit_point,
            ):
                unit_value = (
                    original_unit_value
                    if decimal_places is None
                    else unit_coordinate_for_value(value, parameter)
                )
                row[f"u_{parameter.name}"] = f"{unit_value:.16g}"
        for parameter, value in zip(parameters, rounded_values):
            row[parameter.name] = format_value(value, parameter.unit, decimal_places)
        rows.append(row)
    write_rows_csv(path, rows, fields)
    return write_geometry_metadata(
        path,
        parameters,
        rows,
        split_var,
        generation_kind="targeted_additional",
        method=method_name,
        decimal_places=decimal_places,
        extra={
            "analysis_metric": metric_name,
            "acquisition_method": acquisition_method,
            "candidate_method": candidate_method,
            **(acquisition_metadata or {}),
        },
    )


def parse_methods(raw_methods: Sequence[str]) -> list[str]:
    methods: list[str] = []
    for raw in raw_methods:
        for part in raw.split(","):
            method = part.strip().lower()
            if method == "minimax-lhs":
                method = "maximin-lhs"
            if method:
                methods.append(method)
    return methods


VALID_METHODS = {
    "maximin-lhs",
    "minimax-lhs",
    "latin-hypercube",
    "sobol",
    "halton",
}


def add_parameter_arguments(
    parser: argparse.ArgumentParser,
    *,
    parameter_required: bool = True,
) -> None:
    parser.add_argument(
        "--parameter",
        action="append",
        required=parameter_required,
        default=[],
        metavar="NAME=LOW:HIGH[:linear|log]",
        help=(
            "Repeat once per geometry/process variable, e.g. W=0.40mm:0.80mm "
            "or R=1:100:log."
            if parameter_required
            else "Optional explicit parameter-domain override. When omitted, "
            "suggest-additional loads parameters from --parameter-json or the "
            "companion JSON beside --existing-points."
        ),
    )
    parser.add_argument(
        "--range-factor",
        action="append",
        default=[],
        metavar="NAME=FACTOR",
        help=(
            "Increase an existing parameter-domain span around its center by this "
            "factor. Repeat for multiple parameters, e.g. W=1.5."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for randomized methods.")
    parser.add_argument(
        "--lhs-candidates",
        type=int,
        default=64,
        help="Number of Latin-hypercube candidate designs tried by maximin-lhs. Default: 64.",
    )
    parser.add_argument("--skip", type=int, default=0, help="Skip this many leading Sobol/Halton points.")
    parser.add_argument(
        "--no-scramble",
        action="store_true",
        help="Disable Sobol scrambling. Scrambling is enabled by default.",
    )
    parser.add_argument(
        "--split-var",
        default="dataset",
        help="CSV column used to label point groups. Default: dataset.",
    )
    parser.add_argument(
        "--decimal-places",
        type=int,
        help=(
            "Round generated parameter values to this many decimal places in their "
            "declared units. Must be between 0 and 15."
        ),
    )
    parser.add_argument(
        "--include-normalized",
        action="store_true",
        help="Include u_<name> columns with the underlying [0, 1] coordinates.",
    )


def build_generate_parser() -> argparse.ArgumentParser:
    dispatcher_prog = os.environ.get("ADS_SURROGATE_CLI_PROG")
    parser = argparse.ArgumentParser(
        prog=f"{dispatcher_prog} generate" if dispatcher_prog else None,
        description="Generate geometry/process sample points for ADS surrogate extraction.",
    )
    add_parameter_arguments(parser)
    parser.add_argument(
        "--bare-values",
        choices=["parameter-units", "base-units"],
        default="parameter-units",
        help=(
            "How to interpret unitless values in --existing-points. "
            "Default: parameter-units."
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        help=(
            "Number of new points. Required for normal generation; a range extension "
            "uses the density-based recommendation when omitted."
        ),
    )
    parser.add_argument(
        "--existing-points",
        help="Original geometry CSV to retain and append to when using --extend-range.",
    )
    parser.add_argument(
        "--extend-range",
        metavar="NAME=NEW_LOW:NEW_HIGH",
        help=(
            "Extend one parameter on one side, sample only the added slab, and append "
            "the new points to --existing-points."
        ),
    )
    parser.add_argument(
        "--verification-count",
        type=int,
        help=(
            "Number of new tail points labeled verification. Default: 0 normally; "
            "a range extension preserves the original split ratio."
        ),
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help=(
            "Point-set method. Repeat or comma-separate values. Choices: maximin-lhs, "
            "latin-hypercube, sobol, halton. Default: maximin-lhs."
        ),
    )
    parser.add_argument(
        "--out",
        help=(
            "Output CSV path; a same-stem parameter-range JSON is also written. "
            "Use {method} for multiple methods. Default: generated_points.csv, "
            "or <existing>_extended.csv for a range extension."
        ),
    )
    parser.add_argument(
        "--write-split-files",
        action="store_true",
        help="Also write *_train.csv and *_verification.csv files beside the combined CSV.",
    )
    add_options_json_argument(parser, recursive=False)
    return parser


def build_suggest_parser() -> argparse.ArgumentParser:
    dispatcher_prog = os.environ.get("ADS_SURROGATE_CLI_PROG")
    parser = argparse.ArgumentParser(
        prog=(
            f"{dispatcher_prog} suggest-additional"
            if dispatcher_prog
            else None
        ),
        description=(
            "Analyze current verification error and suggest targeted additional "
            "geometry/process points for the next EM batch."
        ),
    )
    add_parameter_arguments(parser, parameter_required=False)
    parser.add_argument(
        "--parameter-json",
        help=(
            "Generated geometry metadata JSON containing parameter names, bounds, "
            "units, and scales. When omitted with no --parameter options, infer the "
            "same-stem JSON from --existing-points."
        ),
    )
    parser.add_argument("--count", type=int, required=True, help="Number of additional points to suggest.")
    parser.add_argument(
        "--fit-dir",
        help=(
            "Training/model directory containing verification_metrics.csv, or an "
            "optimize/sweep directory containing best_model."
        ),
    )
    parser.add_argument(
        "--allow-nonpassive",
        action="store_true",
        help=(
            "When an optimize/sweep run has no passivity-eligible best_model, use "
            "its retained point_generation_fallback verification errors for point "
            "selection only. This does not make that model export-eligible."
        ),
    )
    parser.add_argument(
        "--verification-metrics",
        help="Path to verification_metrics.csv. Overrides --fit-dir.",
    )
    parser.add_argument(
        "--existing-points",
        action="append",
        default=[],
        help=(
            "CSV containing already simulated points. Repeat for multiple files. "
            "When --parameter and --parameter-json are omitted, load the parameter "
            "domain from its generated companion JSON."
        ),
    )
    parser.add_argument(
        "--existing-mdif",
        action="append",
        default=[],
        help="MDIF containing already simulated training/verification blocks. Repeat for multiple files.",
    )
    parser.add_argument(
        "--metric",
        default="evm_pct",
        help="Column in verification_metrics.csv used to target errors. Use 'auto' to pick a known metric.",
    )
    parser.add_argument(
        "--candidate-method",
        default="maximin-lhs",
        choices=sorted(VALID_METHODS),
        help=(
            "Method used to create the candidate pool before targeted selection. "
            "minimax-lhs is accepted as an alias for maximin-lhs. Default: maximin-lhs."
        ),
    )
    parser.add_argument(
        "--acquisition",
        choices=["error-distance", "gp-ucb"],
        default="gp-ucb",
        help=(
            "Candidate scoring method. gp-ucb fits a Matérn-5/2 Gaussian process "
            "to log geometry-level error; error-distance retains the original "
            "radial error-focus and novelty selector without fitting a GP. "
            "Default: gp-ucb."
        ),
    )
    parser.add_argument(
        "--exploration-weight",
        type=float,
        default=2.0,
        help=(
            "GP-UCB standard-deviation multiplier. Larger values explore uncertain "
            "regions more strongly. Default: 2.0."
        ),
    )
    parser.add_argument(
        "--gp-length-scale",
        type=float,
        help=(
            "Optional Matérn-5/2 length scale in normalized geometry space. If "
            "omitted, select it by log marginal likelihood."
        ),
    )
    parser.add_argument(
        "--gp-noise-variance",
        type=float,
        default=1e-6,
        help="Non-negative normalized GP covariance nugget. Default: 1e-6.",
    )
    parser.add_argument(
        "--gp-error-floor",
        type=float,
        default=1e-12,
        help="Positive error floor applied before the GP log transform. Default: 1e-12.",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        help="Number of candidate points to score. Default: max(1000, count * candidate-factor).",
    )
    parser.add_argument(
        "--candidate-factor",
        type=int,
        default=200,
        help="Candidate multiplier used when --candidate-count is omitted. Default: 200.",
    )
    parser.add_argument(
        "--focus-radius",
        type=float,
        default=0.25,
        help="Normalized unit-cube radius around high-error verification points. Default: 0.25.",
    )
    parser.add_argument(
        "--focus-power",
        type=float,
        default=1.0,
        help="Exponent applied to verification error scores. Default: 1.0.",
    )
    parser.add_argument(
        "--novelty-power",
        type=float,
        default=1.0,
        help="Exponent applied to distance from existing/suggested points. Default: 1.0.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.0,
        help="Reject candidates closer than this normalized distance to existing/suggested points.",
    )
    parser.add_argument(
        "--bare-values",
        choices=["parameter-units", "base-units"],
        default="parameter-units",
        help=(
            "How to interpret unitless values in metrics/MDIF/CSV rows. "
            "Default: parameter-units."
        ),
    )
    parser.add_argument(
        "--target-dataset",
        default="targeted",
        help="Dataset label assigned to suggested points. Default: targeted.",
    )
    parser.add_argument(
        "--out",
        default="targeted_additional_points.csv",
        help="Suggested-point CSV path; a same-stem parameter-range JSON is also written.",
    )
    parser.add_argument(
        "--analysis-out",
        help="Ranked current-fit error-region CSV path. Default: <out>_fit_error_regions.csv.",
    )
    add_options_json_argument(parser, recursive=False)
    return parser


def parse_parameters_or_error(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    apply_factors: bool = True,
) -> list[ParameterSpec]:
    try:
        parameters = [parse_parameter_spec(raw) for raw in args.parameter]
        return apply_range_factors(parameters, args.range_factor if apply_factors else [])
    except ValueError as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable")


def resolve_suggest_parameters(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> list[ParameterSpec]:
    if args.parameter and args.parameter_json:
        parser.error("Use either --parameter or --parameter-json, not both")
    if args.parameter:
        args.parameter_metadata_source = None
        return parse_parameters_or_error(parser, args)

    metadata_paths: list[Path] = []
    if args.parameter_json:
        metadata_paths.append(Path(args.parameter_json))
    else:
        for raw_path in args.existing_points:
            for candidate in companion_geometry_metadata_candidates(Path(raw_path)):
                if candidate.exists() and candidate not in metadata_paths:
                    metadata_paths.append(candidate)
                    break
    if not metadata_paths:
        parser.error(
            "No parameter domain was supplied. Add --existing-points with its "
            "same-stem generated JSON, pass --parameter-json explicitly, or use "
            "repeatable --parameter overrides."
        )

    try:
        parameters = parameter_specs_from_geometry_metadata(metadata_paths[0])
        for metadata_path in metadata_paths[1:]:
            other_parameters = parameter_specs_from_geometry_metadata(metadata_path)
            if not parameter_specs_equal(parameters, other_parameters):
                raise ValueError(
                    "Existing-point metadata files describe different parameter "
                    f"domains: {metadata_paths[0]} and {metadata_path}. Use "
                    "--parameter-json to select the intended complete domain."
                )
        parameters = apply_range_factors(parameters, args.range_factor)
    except ValueError as exc:
        parser.error(str(exc))
    args.parameter_metadata_source = str(metadata_paths[0])
    return parameters


def validate_shared_sampling_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.decimal_places is not None and not 0 <= args.decimal_places <= 15:
        parser.error("--decimal-places must be between 0 and 15")
    if args.lhs_candidates <= 0:
        parser.error("--lhs-candidates must be positive")
    if args.skip < 0:
        parser.error("--skip must be non-negative")


def validate_parameter_decimal_places(
    parser: argparse.ArgumentParser,
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> None:
    if decimal_places is None:
        return
    for parameter in parameters:
        first, last, _ = parameter_decimal_grid(parameter, decimal_places)
        if first > last:
            unit_label = parameter.unit or "base units"
            parser.error(
                f"--decimal-places {decimal_places} cannot represent any value inside "
                f"the {parameter.name!r} range in {unit_label}; increase the precision"
            )


def command_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_shared_sampling_args(parser, args)
    extending = args.extend_range is not None
    if extending != (args.existing_points is not None):
        parser.error("--extend-range and --existing-points must be used together")
    if extending and args.range_factor:
        parser.error("--extend-range cannot be combined with --range-factor")
    parameters = parse_parameters_or_error(parser, args, apply_factors=not extending)
    validate_parameter_decimal_places(parser, parameters, args.decimal_places)

    extension_plan: RangeExtensionPlan | None = None
    existing_fields: list[str] = []
    existing_rows: list[dict[str, object]] = []
    if extending:
        try:
            extension_plan = build_range_extension_plan(parameters, args.extend_range)
            validate_parameter_decimal_places(
                parser,
                extension_plan.sampling_parameters,
                args.decimal_places,
            )
            existing_fields, existing_rows = read_csv_table(Path(args.existing_points))
            validate_existing_parameter_rows(
                existing_rows,
                extension_plan.original_parameters,
                bare_values=args.bare_values,
            )
        except ValueError as exc:
            parser.error(str(exc))
        recommended_train, recommended_verification = range_extension_recommendation(
            existing_rows,
            args.split_var,
            len(parameters),
            extension_plan.added_volume_ratio,
        )
        recommended_total = recommended_train + recommended_verification
        original_parameter = next(
            parameter
            for parameter in extension_plan.original_parameters
            if parameter.name == extension_plan.parameter_name
        )
        overall_parameter = next(
            parameter
            for parameter in extension_plan.overall_parameters
            if parameter.name == extension_plan.parameter_name
        )
        print(
            f"extending {extension_plan.parameter_name} on the {extension_plan.side} side: "
            f"{format_value(original_parameter.lower, original_parameter.unit)}:"
            f"{format_value(original_parameter.upper, original_parameter.unit)} -> "
            f"{format_value(overall_parameter.lower, overall_parameter.unit)}:"
            f"{format_value(overall_parameter.upper, overall_parameter.unit)}"
        )
        print(
            f"range extension adds {extension_plan.added_volume_ratio * 100.0:.1f}% "
            "of the original transformed design-space volume"
        )
        print(
            "point guidance: "
            f"{recommended_total} new points "
            f"({recommended_train} train, {recommended_verification} verification); "
            "this preserves sampling density with minimum boundary coverage",
            flush=True,
        )
        if args.count is None:
            verification_count = (
                recommended_verification
                if args.verification_count is None
                else args.verification_count
            )
            count = recommended_train + verification_count
            print(f"using recommended --count {count}")
        else:
            count = args.count
            if args.verification_count is None:
                old_verification = sum(
                    str(lookup_row_value(row, args.split_var) or "train").strip().lower()
                    == "verification"
                    for row in existing_rows
                )
                verification_count = int(round(count * old_verification / len(existing_rows)))
                if count == 1:
                    verification_count = 0
                elif old_verification:
                    verification_count = min(count - 1, max(1, verification_count))
            else:
                verification_count = args.verification_count
            if count < recommended_total:
                print(
                    f"warning: --count {count} is below the recommended {recommended_total} "
                    "points for this extension",
                    file=sys.stderr,
                )
    else:
        if args.count is None:
            parser.error("--count is required unless --extend-range is used")
        count = args.count
        verification_count = args.verification_count or 0

    if count <= 0:
        parser.error("--count must be positive")
    if verification_count < 0 or verification_count >= count:
        parser.error("--verification-count must be non-negative and smaller than --count")

    methods = parse_methods(args.method) or ["maximin-lhs"]
    unknown = [method for method in methods if method not in VALID_METHODS]
    if unknown:
        parser.error("Unknown method(s): " + ", ".join(unknown))

    multiple_methods = len(methods) > 1
    if args.out:
        base_out_path = Path(args.out)
    elif extending:
        existing_path = Path(args.existing_points)
        base_out_path = existing_path.with_name(
            f"{existing_path.stem}_extended{existing_path.suffix or '.csv'}"
        )
    else:
        base_out_path = Path("generated_points.csv")
    written_paths: list[Path] = []
    for offset, method in enumerate(methods):
        try:
            unit_points = generate_unit_points(
                method,
                count=count,
                dimensions=len(parameters),
                seed=args.seed + offset,
                lhs_candidates=args.lhs_candidates,
                scramble=not args.no_scramble,
                skip=args.skip,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        out_path = output_path_for_method(base_out_path, method, multiple_methods)
        if extension_plan is not None:
            written_paths.extend(
                write_range_extension_csv(
                    out_path,
                    existing_fields,
                    existing_rows,
                    method,
                    unit_points,
                    extension_plan,
                    verification_count=verification_count,
                    split_var=args.split_var,
                    include_normalized=args.include_normalized,
                    decimal_places=args.decimal_places,
                    bare_values=args.bare_values,
                    write_split_files=args.write_split_files,
                )
            )
        else:
            written_paths.extend(write_points_csv(
                out_path,
                method,
                unit_points,
                parameters,
                verification_count=verification_count,
                split_var=args.split_var,
                include_normalized=args.include_normalized,
                decimal_places=args.decimal_places,
                write_split_files=args.write_split_files,
            ))
    for path in written_paths:
        print(f"wrote {path}")
    return 0


def verification_metrics_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    fallback_manifest: Path | None = None
    if args.verification_metrics:
        path = Path(args.verification_metrics)
        sibling_manifest = path.parent / "point_generation_source.json"
        if sibling_manifest.is_file():
            fallback_manifest = sibling_manifest
    elif args.fit_dir:
        fit_dir = Path(args.fit_dir)
        direct_path = fit_dir / "verification_metrics.csv"
        best_model_path = fit_dir / "best_model" / "verification_metrics.csv"
        fallback_path = (
            fit_dir / "point_generation_fallback" / "verification_metrics.csv"
        )
        parent_fallback_path = (
            fit_dir.parent
            / "point_generation_fallback"
            / "verification_metrics.csv"
        )
        if direct_path.is_file():
            path = direct_path
            sibling_manifest = fit_dir / "point_generation_source.json"
            if sibling_manifest.is_file():
                fallback_manifest = sibling_manifest
        elif best_model_path.is_file():
            path = best_model_path
        elif fallback_path.is_file():
            path = fallback_path
            fallback_manifest = fallback_path.parent / "point_generation_source.json"
        elif fit_dir.name == "best_model" and parent_fallback_path.is_file():
            path = parent_fallback_path
            fallback_manifest = (
                parent_fallback_path.parent / "point_generation_source.json"
            )
        else:
            path = direct_path
    else:
        parser.error("Either --fit-dir or --verification-metrics is required")
    if not path.exists():
        parser.error(
            f"Verification metrics file does not exist: {path}. For an optimize "
            "run, pass the sweep directory or its best_model directory."
        )
    if fallback_manifest is not None:
        if not getattr(args, "allow_nonpassive", False):
            parser.error(
                "Only a passivity-ineligible point-generation fallback is "
                "available. Add --allow-nonpassive to use its verification errors "
                "for point selection; it will remain ineligible for model export."
            )
        try:
            source_payload = json.loads(fallback_manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Could not read point-generation fallback metadata: {exc}")
        args.nonpassive_source = source_payload
        source_trial = source_payload.get("source_trial", "unknown")
        print(
            "warning: using verification errors from passivity-ineligible trial "
            f"{source_trial} for point selection only; this model is not eligible "
            "for export",
            file=sys.stderr,
        )
    else:
        args.nonpassive_source = None
    return path


def command_suggest_additional(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.candidate_factor <= 0:
        parser.error("--candidate-factor must be positive")
    if args.candidate_count is not None and args.candidate_count <= 0:
        parser.error("--candidate-count must be positive")
    if args.focus_radius <= 0.0:
        parser.error("--focus-radius must be positive")
    if args.focus_power < 0.0:
        parser.error("--focus-power must be non-negative")
    if args.novelty_power < 0.0:
        parser.error("--novelty-power must be non-negative")
    if args.min_distance < 0.0:
        parser.error("--min-distance must be non-negative")
    if args.exploration_weight < 0.0:
        parser.error("--exploration-weight must be non-negative")
    if args.gp_length_scale is not None and args.gp_length_scale <= 0.0:
        parser.error("--gp-length-scale must be positive")
    if args.gp_noise_variance < 0.0:
        parser.error("--gp-noise-variance must be non-negative")
    if args.gp_error_floor <= 0.0:
        parser.error("--gp-error-floor must be positive")
    validate_shared_sampling_args(parser, args)
    parameters = resolve_suggest_parameters(parser, args)
    validate_parameter_decimal_places(parser, parameters, args.decimal_places)

    metrics_path = verification_metrics_path(args, parser)
    try:
        regions, metric_name = load_error_regions(
            metrics_path,
            parameters,
            metric_name=args.metric,
            bare_values=args.bare_values,
        )
    except ValueError as exc:
        parser.error(str(exc))

    existing_points = [region.unit_point for region in regions]
    existing_points.extend(
        load_existing_points(
            args.existing_points,
            args.existing_mdif,
            parameters,
            bare_values=args.bare_values,
        )
    )
    existing_points = dedupe_points(existing_points)

    candidate_count = args.candidate_count or max(1000, args.count * args.candidate_factor)
    candidate_method = (
        "maximin-lhs"
        if args.candidate_method == "minimax-lhs"
        else args.candidate_method
    )
    try:
        candidates = generate_unit_points(
            candidate_method,
            count=candidate_count,
            dimensions=len(parameters),
            seed=args.seed,
            lhs_candidates=args.lhs_candidates,
            scramble=not args.no_scramble,
            skip=args.skip,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    acquisition_metadata: dict[str, object] = {}
    acquisition_metadata["verification_metrics_source"] = str(metrics_path)
    if getattr(args, "nonpassive_source", None):
        acquisition_metadata["nonpassive_point_generation_source"] = (
            args.nonpassive_source
        )
    if getattr(args, "parameter_metadata_source", None):
        acquisition_metadata["parameter_metadata_source"] = (
            args.parameter_metadata_source
        )
    if args.acquisition == "gp-ucb":
        try:
            suggestions, gp_model = select_gp_ucb_points(
                candidates,
                regions,
                existing_points,
                count=args.count,
                exploration_weight=args.exploration_weight,
                novelty_power=args.novelty_power,
                min_distance=args.min_distance,
                length_scale=args.gp_length_scale,
                noise_variance=args.gp_noise_variance,
                error_floor=args.gp_error_floor,
            )
        except ValueError as exc:
            parser.error(str(exc))
        acquisition_metadata["gp"] = {
            "kernel": "matern52_isotropic",
            "target_transform": "natural_log_error",
            "observation_count": len(gp_model.observation_points),
            "length_scale": gp_model.length_scale,
            "length_scale_selection": (
                "user" if args.gp_length_scale is not None else "log_marginal_likelihood"
            ),
            "noise_variance": gp_model.noise_variance,
            "exploration_weight": args.exploration_weight,
            "log_marginal_likelihood": gp_model.log_marginal_likelihood,
        }
        if len(gp_model.observation_points) < len(parameters) + 1:
            print(
                "warning: GP-UCB has fewer distinct error observations than "
                "dimensions + 1; selections will emphasize posterior uncertainty "
                "until more simulated error observations are available",
                file=sys.stderr,
            )
    else:
        suggestions = select_targeted_points(
            candidates,
            regions,
            existing_points,
            count=args.count,
            focus_radius=args.focus_radius,
            focus_power=args.focus_power,
            novelty_power=args.novelty_power,
            min_distance=args.min_distance,
        )
    if len(suggestions) < args.count:
        print(
            f"warning: selected {len(suggestions)} of {args.count} requested points; "
            "increase --candidate-count or lower --min-distance",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    analysis_path = Path(args.analysis_out) if args.analysis_out else analysis_output_path(out_path)
    write_error_regions_csv(analysis_path, regions, parameters)
    metadata_path = write_suggested_points_csv(
        out_path,
        suggestions,
        parameters,
        split_var=args.split_var,
        target_dataset=args.target_dataset,
        candidate_method=candidate_method,
        acquisition_method=args.acquisition,
        metric_name=metric_name,
        include_normalized=args.include_normalized,
        decimal_places=args.decimal_places,
        acquisition_metadata=acquisition_metadata,
    )
    print(f"analyzed {len(regions)} verification error region(s) from {metrics_path}")
    print(
        f"considered {len(existing_points)} existing point(s) and "
        f"{candidate_count} {candidate_method} candidate point(s)"
    )
    print(f"acquisition method: {args.acquisition}")
    if getattr(args, "parameter_metadata_source", None):
        print(f"parameter domain: {args.parameter_metadata_source}")
    if args.acquisition == "gp-ucb":
        print(
            "GP observations: "
            f"{len(gp_model.observation_points)}, "
            f"Matérn-5/2 length scale: {gp_model.length_scale:.6g}, "
            f"exploration weight: {args.exploration_weight:.6g}"
        )
    print(f"wrote {out_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {geometry_coverage_plot_path(out_path)}")
    print(f"wrote {analysis_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    command = "generate"
    if raw_args and raw_args[0] in {"generate", "suggest-additional"}:
        command = raw_args.pop(0)

    if command == "suggest-additional":
        parser = build_suggest_parser()
        args = parse_args_with_options_json(
            parser,
            raw_args,
            workflow="points",
            command="suggest-additional",
        )
        status = command_suggest_additional(args, parser)
        return finalize_options_json_update(args, status)

    parser = build_generate_parser()
    args = parse_args_with_options_json(
        parser,
        raw_args,
        workflow="points",
        command="generate",
    )
    status = command_generate(args, parser)
    return finalize_options_json_update(args, status)


if __name__ == "__main__":
    raise SystemExit(main())
