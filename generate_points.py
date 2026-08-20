#!/usr/bin/env python3
"""Generate and adapt geometry/process points for surrogate-model extraction.

The default initial method is a maximin Latin hypercube because finite EM
sample sets usually benefit from stratification plus good point separation.
Sobol is also available when SciPy is installed. After a fit, the default
hybrid acquisition combines predicted-error exploitation, Gaussian-process
uncertainty, and maximin coverage while retaining GP-UCB and the original
error-distance selector as explicit alternatives.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
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
    selection_component: str = ""


@dataclass
class GaussianProcessModel:
    observation_points: list[list[float]]
    log_error_mean: float
    log_error_scale: float
    length_scales: list[float]
    noise_variance: float
    cholesky: list[list[float]]
    alpha: list[float]
    log_marginal_likelihood: float
    normalized_targets: list[float]
    length_scale_selection: str = "isotropic"

    @property
    def length_scale(self) -> float:
        """Geometric-mean length scale retained for backward compatibility."""

        if not self.length_scales:
            return 1.0
        return math.exp(
            sum(math.log(max(value, 1e-300)) for value in self.length_scales)
            / len(self.length_scales)
        )


@dataclass(frozen=True)
class PointCountRecommendation:
    recommended_count: int
    dimensions: int
    current_error_rms: float
    current_error_median: float
    current_error_p90: float
    current_error_max: float
    target_error: float | None
    target_ratio: float | None
    previous_error_rms: list[float]
    latest_improvement_fraction: float | None
    existing_training_count: int
    verification_observation_count: int
    stage: str
    rationale: list[str]


@dataclass(frozen=True)
class HybridAllocation:
    exploitation: int
    uncertainty: int
    coverage: int
    regime: str


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


TRAIN_DATASET_TOKENS = {"train", "training"}
VERIFICATION_DATASET_TOKENS = {
    "verification",
    "verify",
    "validation",
    "test",
}
LEGACY_TRAIN_DATASET_TOKENS = {
    "targeted",
    "additional",
    "added",
    "new",
    "acquisition",
}


def canonical_dataset_label(value: object, *, default: str | None = None) -> str:
    """Return the only two dataset labels accepted by geometry workflows.

    ``targeted`` and the other acquisition-origin labels were emitted by older
    suggest-additional versions.  They describe provenance, not a holdout
    split, so they are migrated to ``train``.  Newness remains represented by
    ``point_origin=additional`` and ``method``.
    """

    token = normalize_key(value) if str(value or "").strip() else ""
    if not token and default is not None:
        token = normalize_key(default)
    if token in TRAIN_DATASET_TOKENS or token in LEGACY_TRAIN_DATASET_TOKENS:
        return "train"
    if token in VERIFICATION_DATASET_TOKENS:
        return "verification"
    raise ValueError(
        f"Unrecognized geometry dataset value {value!r}; expected train or "
        "verification"
    )


def coverage_split_group(value: object) -> str:
    return (
        "verification"
        if canonical_dataset_label(value) == "verification"
        else "training"
    )


def geometry_file_split_group(path: Path) -> str | None:
    """Infer a geometry-file role from train/verification words in its name.

    Generated split files use the explicit ``_training`` and
    ``_verification`` suffixes. Legacy ``train``/``verify`` spellings remain
    accepted as final suffixes. Complete ``training`` or ``verification``
    words are recognized anywhere, so a file such as
    ``round_2_training_geometries.csv`` cannot be mistaken for a combined
    inventory.
    """

    ordered_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", path.stem.lower())
        if token
    ]
    tokens = set(ordered_tokens)
    final_token = ordered_tokens[-1] if ordered_tokens else ""
    has_training = "training" in tokens or final_token == "train"
    has_verification = "verification" in tokens or final_token in {
        "verify",
        "validation",
        "test",
    }
    if has_training and has_verification:
        raise ValueError(
            f"Geometry filename {path.name!r} contains both training and "
            "verification role words"
        )
    if has_training:
        return "training"
    if has_verification:
        return "verification"
    return None


def require_combined_geometry_path(path: Path, option_name: str) -> None:
    """Reject a split-looking path for an output that contains both roles."""

    role = geometry_file_split_group(path)
    if role is not None:
        raise ValueError(
            f"{option_name} is a combined geometry output, but {path.name!r} "
            f"is named like a {role} split. Use a basename without train, "
            "training, verify, verification, validation, or test; the command "
            "writes explicit _training.csv and _verification.csv views."
        )


def combined_geometry_stem(path: Path) -> str:
    """Remove split-role words when deriving a new combined output name."""

    tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", path.stem)
        if token
    ]
    filtered = [
        token
        for token in tokens
        if token.lower() not in {"train", "training", "verification", "verify"}
    ]
    return "_".join(filtered) or "geometries"


def coverage_point_group(
    row: dict[str, object],
    split_var: str,
    default_dataset: object = "train",
) -> str:
    origin = lookup_row_value(row, "point_origin")
    if origin is not None and str(origin).strip():
        origin_token = normalize_key(origin)
        if origin_token in {
            "additional",
            "added",
            "new",
            "targeted",
            "acquisition",
        }:
            return "additional"
        if origin_token in {"existing", "original", "prior"}:
            return coverage_split_group(
                lookup_row_value(row, split_var) or default_dataset or "train"
            )
    split_value = lookup_row_value(row, split_var) or default_dataset or "train"
    if normalize_key(split_value) in {
        "additional",
        "added",
        "new",
        "targeted",
        "acquisition",
    }:
        return "additional"
    split_group = coverage_split_group(split_value)
    if split_group != "training":
        return split_group
    method = normalize_key(lookup_row_value(row, "method") or "")
    if method.startswith("targeted_") or method.startswith("gp_ucb_"):
        return "additional"
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
        "additional": [],
    }
    default_dataset = geometry_file_split_group(geometry_path) or "train"
    dataset_counts = {"training": 0, "verification": 0}
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
        grouped_points[
            coverage_point_group(
                row,
                split_var,
                default_dataset=default_dataset,
            )
        ].append(coordinates)
        dataset_group = coverage_split_group(
            lookup_row_value(row, split_var) or default_dataset
        )
        dataset_counts[dataset_group] += 1

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
    additional_color = (22, 163, 74, 255)
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
    present_dataset_groups = [
        name for name in ("training", "verification")
        if dataset_counts[name]
    ]
    coverage_scope = (
        "combined"
        if len(present_dataset_groups) > 1
        else (
            f"{present_dataset_groups[0]}-only"
            if present_dataset_groups
            else "empty"
        )
    )
    draw.text(
        (left_margin, px(12)),
        f"Parameter coverage: {coverage_scope.replace('-', ' ')}",
        font=title_font,
        fill=title_color,
    )
    group_labels = {
        "training": "Training",
        "verification": "Verification",
        "additional": "Added",
    }
    group_colors = {
        "training": training_color,
        "verification": verification_color,
        "additional": additional_color,
    }
    present_groups = [
        name for name in ("training", "verification", "additional")
        if grouped_points[name]
    ]
    draw.text(
        (left_margin, px(42)),
        (
            "Dataset: "
            + ", ".join(
                f"{dataset_counts[name]} {name}"
                for name in present_dataset_groups
            )
            + " | Visual: "
            + (
                ", ".join(
                    f"{len(grouped_points[name])} {group_labels[name].lower()}"
                    for name in present_groups
                )
                or "0 points"
            )
        ),
        font=label_font,
        fill=tick_color,
    )
    legend_x = left_margin
    for group_name in present_groups:
        draw.ellipse(
            (legend_x - px(4), px(68), legend_x + px(4), px(76)),
            fill=group_colors[group_name],
        )
        draw.text(
            (legend_x + px(10), px(66)),
            group_labels[group_name],
            font=label_font,
            fill=text_color,
        )
        legend_x += px(112 if group_name == "verification" else 82)

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
                for group_name in present_groups:
                    color = group_colors[group_name]
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
                for group_name in present_groups:
                    color = group_colors[group_name]
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
    validate_geometry_output_rows(
        geometry_path,
        rows,
        parameters,
        split_var,
        bare_values=bare_values,
        decimal_places=decimal_places,
    )
    split_counts: dict[str, int] = {}
    split_geometry_keys: dict[str, set[tuple[str, ...]]] = {
        "train": set(),
        "verification": set(),
    }
    inferred_split = geometry_file_split_group(geometry_path)
    for row in rows:
        raw_split_value = row.get(split_var, "")
        if not str(raw_split_value or "").strip() and inferred_split is not None:
            raw_split_value = "train" if inferred_split == "training" else "verification"
        if str(raw_split_value or "").strip():
            split_value = canonical_dataset_label(raw_split_value)
            split_counts[split_value] = split_counts.get(split_value, 0) + 1
            key = geometry_output_key_from_row(
                row,
                parameters,
                decimal_places,
                bare_values=bare_values,
            )
            assert key is not None
            split_geometry_keys[split_value].add(key)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "geometry_file": geometry_path.name,
        "generation_kind": generation_kind,
        "method": method,
        "point_count": len(rows),
        "split_variable": split_var,
        "split_counts": split_counts,
        "geometry_integrity": {
            "unique_point_count": len(
                split_geometry_keys["train"]
                | split_geometry_keys["verification"]
            ),
            "training_point_count": len(split_geometry_keys["train"]),
            "verification_point_count": len(
                split_geometry_keys["verification"]
            ),
            "training_verification_overlap_count": len(
                split_geometry_keys["train"]
                & split_geometry_keys["verification"]
            ),
            "duplicates_present": False,
        },
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
    for suffix in (
        "_train",
        "_training",
        "_verification",
        "_verify",
        "_validation",
        "_test",
    ):
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
    canonical_name = canonical_dataset_label(split_name)
    suffix = "training" if canonical_name == "train" else "verification"
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.csv'}")


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
    require_combined_geometry_path(path, "--out")
    validate_geometry_output_rows(
        path,
        rows,
        parameters,
        split_var,
        bare_values="parameter-units",
        decimal_places=decimal_places,
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
            validate_geometry_output_rows(
                split_path,
                split_rows,
                parameters,
                split_var,
                bare_values="parameter-units",
                decimal_places=decimal_places,
            )
            write_rows_csv(split_path, split_rows, fields)
            split_plot_path = write_parameter_coverage_png(
                split_path,
                parameters,
                split_rows,
                split_var,
            )
            written.extend([split_path, split_plot_path])
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
    input_cleanup: dict[str, int] | None = None,
) -> list[Path]:
    require_combined_geometry_path(path, "--out")
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
        split_value = canonical_dataset_label(
            lookup_row_value(row, split_var),
            default="train",
        )
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
    validate_geometry_output_rows(
        path,
        rows,
        plan.overall_parameters,
        split_var,
        bare_values=bare_values,
        decimal_places=decimal_places,
    )
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
        },
        "input_geometry_cleanup": dict(input_cleanup or {}),
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
            validate_geometry_output_rows(
                split_path,
                split_rows,
                plan.overall_parameters,
                split_var,
                bare_values=bare_values,
                decimal_places=decimal_places,
            )
            write_rows_csv(split_path, split_rows, fields)
            split_plot_path = write_parameter_coverage_png(
                split_path,
                plan.overall_parameters,
                split_rows,
                split_var,
                bare_values=bare_values,
            )
            written.extend([split_path, split_plot_path])
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
    values = row_parameter_values(row, parameters, bare_values=bare_values)
    if values is None:
        return None
    point: list[float] = []
    for parameter, value in zip(parameters, values):
        try:
            unit_value = unit_coordinate_for_value(value, parameter)
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(unit_value):
            return None
        point.append(unit_value)
    return point


def row_parameter_values(
    row: dict[str, object],
    parameters: Sequence[ParameterSpec],
    *,
    bare_values: str,
) -> list[float] | None:
    """Read one row into base-unit parameter values without normalizing it."""

    values: list[float] = []
    for parameter in parameters:
        raw = lookup_row_value(row, parameter.name)
        if raw is None or str(raw).strip() == "":
            return None
        try:
            value = parse_observed_value(raw, parameter, bare_values=bare_values)
        except (ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return values


def geometry_output_key_from_values(
    values: Sequence[float],
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> tuple[str, ...]:
    """Return the identity of the parameter values as they appear in a CSV.

    Geometry duplication is an output concern: values that resolve to the same
    declared-unit digits are the same simulation point even when their original
    floating-point or normalized coordinates differ.  Building the key from the
    canonical formatted values keeps generation, validation, and accumulation
    on exactly the same decimal grid.
    """

    if len(values) != len(parameters):
        raise ValueError("A geometry key requires one value per parameter")
    return tuple(
        format_value(
            round_parameter_value(float(value), parameter, decimal_places),
            parameter.unit,
            decimal_places,
        )
        for parameter, value in zip(parameters, values)
    )


def geometry_output_key_from_unit_point(
    point: Sequence[float],
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> tuple[str, ...]:
    """Return the emitted-value identity for a normalized geometry point."""

    if len(point) != len(parameters):
        raise ValueError("A geometry key requires one coordinate per parameter")
    values = [
        map_unit_point(float(coordinate), parameter)
        for parameter, coordinate in zip(parameters, point)
    ]
    return geometry_output_key_from_values(values, parameters, decimal_places)


def geometry_output_key_from_row(
    row: dict[str, object],
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
    *,
    bare_values: str,
) -> tuple[str, ...] | None:
    """Return the target-digit identity of a parsed geometry CSV row."""

    values = row_parameter_values(row, parameters, bare_values=bare_values)
    if values is None:
        return None
    return geometry_output_key_from_values(values, parameters, decimal_places)


def round_unit_point_for_output(
    point: Sequence[float],
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> list[float]:
    """Move a normalized point onto the exact grid used by the output CSV."""

    if decimal_places is None:
        return [float(value) for value in point]
    values = [
        round_parameter_value(
            map_unit_point(float(coordinate), parameter),
            parameter,
            decimal_places,
        )
        for parameter, coordinate in zip(parameters, point)
    ]
    return [
        unit_coordinate_for_value(value, parameter)
        for parameter, value in zip(parameters, values)
    ]


def canonicalize_sampled_point(
    point: Sequence[float],
    sampling_parameters: Sequence[ParameterSpec],
    output_parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> tuple[list[float], tuple[str, ...]]:
    """Round a sampled point and return its final output-space identity."""

    if len(sampling_parameters) != len(output_parameters):
        raise ValueError("Sampling and output parameter domains must align")
    if any(
        sampling.name != output.name or sampling.unit != output.unit
        for sampling, output in zip(sampling_parameters, output_parameters)
    ):
        raise ValueError(
            "Sampling and output parameters must have matching names and units"
        )
    values = [
        round_parameter_value(
            map_unit_point(float(coordinate), sampling_parameter),
            sampling_parameter,
            decimal_places,
        )
        for sampling_parameter, coordinate in zip(sampling_parameters, point)
    ]
    rounded_point = (
        [float(value) for value in point]
        if decimal_places is None
        else [
            unit_coordinate_for_value(value, sampling_parameter)
            for sampling_parameter, value in zip(sampling_parameters, values)
        ]
    )
    key = geometry_output_key_from_values(
        values,
        output_parameters,
        decimal_places,
    )
    return rounded_point, key


def target_grid_capacity(
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
) -> int | None:
    """Return the finite point capacity of a requested decimal grid."""

    if decimal_places is None:
        return None
    capacity = 1
    for parameter in parameters:
        first, last, _ = parameter_decimal_grid(parameter, decimal_places)
        capacity *= max(0, last - first + 1)
    return capacity


def generate_unique_output_points(
    method: str,
    *,
    count: int,
    sampling_parameters: Sequence[ParameterSpec],
    output_parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
    seed: int,
    lhs_candidates: int,
    scramble: bool,
    skip: int,
    excluded_keys: set[tuple[str, ...]] | None = None,
) -> list[list[float]]:
    """Generate exactly ``count`` points unique at the requested output digits.

    The first design is unchanged when it is already unique.  If rounding
    collapses points, deterministic follow-up batches fill only the missing
    slots.  A finite-grid or retry failure is reported before any CSV is
    written.
    """

    capacity = target_grid_capacity(sampling_parameters, decimal_places)
    occupied = set(excluded_keys or ())
    if capacity is not None and count > capacity:
        raise ValueError(
            f"--decimal-places {decimal_places} provides only {capacity} unique "
            f"point(s) in the sampling range, fewer than --count {count}; "
            "increase --decimal-places or reduce --count"
        )

    selected: list[list[float]] = []
    next_skip = skip
    max_attempts = 32
    for attempt in range(max_attempts):
        remaining = count - len(selected)
        if remaining <= 0:
            return selected
        batch_count = (
            count
            if attempt == 0
            else max(remaining, min(count, max(8, 2 * len(sampling_parameters))))
        )
        batch = generate_unit_points(
            method,
            count=batch_count,
            dimensions=len(sampling_parameters),
            seed=seed + attempt * 1009,
            lhs_candidates=lhs_candidates,
            scramble=scramble,
            skip=next_skip,
        )
        next_skip += batch_count
        for point in batch:
            rounded_point, key = canonicalize_sampled_point(
                point,
                sampling_parameters,
                output_parameters,
                decimal_places,
            )
            if key in occupied:
                continue
            occupied.add(key)
            selected.append(rounded_point)
            if len(selected) == count:
                return selected

    available = len(selected)
    precision = (
        "the default output precision"
        if decimal_places is None
        else f"--decimal-places {decimal_places}"
    )
    raise ValueError(
        f"Could generate only {available} of {count} unique point(s) using "
        f"{precision} after excluding existing geometries. Increase "
        "--decimal-places, increase the parameter range, or reduce --count."
    )


def filter_unique_output_candidates(
    candidate_points: Sequence[Sequence[float]],
    parameters: Sequence[ParameterSpec],
    decimal_places: int | None,
    *,
    excluded_keys: set[tuple[str, ...]],
) -> list[list[float]]:
    """Collapse an acquisition pool onto unique, unoccupied output points."""

    seen = set(excluded_keys)
    unique: list[list[float]] = []
    for point in candidate_points:
        rounded_point = round_unit_point_for_output(
            point,
            parameters,
            decimal_places,
        )
        key = geometry_output_key_from_unit_point(
            rounded_point,
            parameters,
            decimal_places,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(rounded_point)
    return unique


def validate_geometry_output_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    parameters: Sequence[ParameterSpec],
    split_var: str,
    *,
    bare_values: str,
    decimal_places: int | None = None,
) -> None:
    """Validate role naming and prohibit every duplicate output geometry."""

    filename_role = geometry_file_split_group(path)
    seen: dict[tuple[str, ...], tuple[str, int]] = {}
    for row_number, row in enumerate(rows, start=2):
        raw_dataset = lookup_row_value(row, split_var)
        try:
            dataset = canonical_dataset_label(raw_dataset)
        except ValueError as exc:
            raise ValueError(f"{path} row {row_number}: {exc}") from exc
        row[split_var] = dataset
        if filename_role is not None and coverage_split_group(dataset) != filename_role:
            raise ValueError(
                f"Geometry file {path.name!r} is named as {filename_role}, but "
                f"row {row_number} is labeled {dataset!r}"
            )
        point = row_unit_point(row, parameters, bare_values=bare_values)
        if point is None or not in_unit_cube(point):
            raise ValueError(
                f"Could not validate every parameter in {path} row {row_number}"
            )
        key = geometry_output_key_from_row(
            row,
            parameters,
            decimal_places,
            bare_values=bare_values,
        )
        assert key is not None
        prior = seen.get(key)
        if prior is not None:
            prior_dataset, prior_row = prior
            relation = (
                "across training and verification"
                if prior_dataset != dataset
                else f"within {dataset}"
            )
            raise ValueError(
                f"Duplicate geometry {relation} in {path}: rows {prior_row} "
                f"and {row_number}. Increase --decimal-places or correct the "
                "input geometry inventory."
            )
        seen[key] = (dataset, row_number)


def clean_existing_geometry_rows(
    path: Path,
    rows: Sequence[dict[str, object]],
    parameters: Sequence[ParameterSpec],
    split_var: str,
    *,
    bare_values: str,
    decimal_places: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Normalize a legacy inventory and remove train/verification leakage.

    If a geometry was ever assigned to training, it cannot remain an
    independent verification point.  Training therefore wins a cross-split
    conflict.  This is deliberately conservative: it prevents an already
    trained-on response from inflating verification quality.
    """

    filename_role = geometry_file_split_group(path)
    default_dataset = (
        "train"
        if filename_role == "training"
        else "verification" if filename_role == "verification" else None
    )
    cleaned: list[dict[str, object]] = []
    seen: dict[tuple[str, ...], int] = {}
    stats = {
        "legacy_dataset_rows_normalized": 0,
        "same_split_duplicates_removed": 0,
        "cross_split_duplicates_removed": 0,
        "filename_role_mismatches": 0,
    }
    for row_number, source_row in enumerate(rows, start=2):
        row = dict(source_row)
        raw_dataset = lookup_row_value(row, split_var)
        if not str(raw_dataset or "").strip() and default_dataset is None:
            raise ValueError(
                f"Combined geometry file {path} row {row_number} has no "
                f"{split_var!r} value. Combined files must explicitly identify "
                "every row as train or verification."
            )
        try:
            dataset = canonical_dataset_label(raw_dataset, default=default_dataset)
        except ValueError as exc:
            raise ValueError(f"{path} row {row_number}: {exc}") from exc
        if normalize_key(raw_dataset) in LEGACY_TRAIN_DATASET_TOKENS:
            stats["legacy_dataset_rows_normalized"] += 1
        if filename_role is not None and coverage_split_group(dataset) != filename_role:
            # Older versions emitted mixed cumulative inventories with
            # "training" in the filename. Accept them as migration inputs, but
            # never reproduce that mismatch in a new output.
            stats["filename_role_mismatches"] += 1
        row[split_var] = dataset
        point = row_unit_point(row, parameters, bare_values=bare_values)
        if point is None or not in_unit_cube(point):
            raise ValueError(
                f"Could not validate every parameter in {path} row {row_number}"
            )
        key = geometry_output_key_from_row(
            row,
            parameters,
            decimal_places,
            bare_values=bare_values,
        )
        assert key is not None
        prior_index = seen.get(key)
        if prior_index is None:
            seen[key] = len(cleaned)
            cleaned.append(row)
            continue
        prior_dataset = str(cleaned[prior_index][split_var])
        if prior_dataset == dataset:
            stats["same_split_duplicates_removed"] += 1
            continue
        stats["cross_split_duplicates_removed"] += 1
        if dataset == "train":
            cleaned[prior_index] = row
    return cleaned, stats


def resolve_bare_values_for_rows(
    rows: Sequence[dict[str, object]],
    parameters: Sequence[ParameterSpec],
    requested_mode: str,
) -> str:
    """Choose a unitless-value convention independently for one input source."""

    if requested_mode != "auto":
        return requested_mode
    counts: dict[str, int] = {}
    for mode in ("parameter-units", "base-units"):
        counts[mode] = sum(
            1
            for row in rows
            if (
                (point := row_unit_point(row, parameters, bare_values=mode))
                is not None
                and in_unit_cube(point)
            )
        )
    return max(
        counts,
        key=lambda mode: (counts[mode], mode == "parameter-units"),
    )


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
        rows = read_csv_rows(path)
        source_mode = resolve_bare_values_for_rows(
            rows,
            parameters,
            bare_values,
        )
        for row in rows:
            point = row_unit_point(row, parameters, bare_values=source_mode)
            if point is not None and in_unit_cube(point):
                points.append(clamp_unit_point(point))
    for raw_path in mdif_paths:
        path = Path(raw_path)
        rows = read_mdif_parameter_rows(path)
        source_mode = resolve_bare_values_for_rows(
            rows,
            parameters,
            bare_values,
        )
        for row in rows:
            point = row_unit_point(row, parameters, bare_values=source_mode)
            if point is not None and in_unit_cube(point):
                points.append(clamp_unit_point(point))
    return dedupe_points(points)


def existing_csv_dataset_assignments(
    csv_paths: Sequence[str],
    parameters: Sequence[ParameterSpec],
    split_var: str,
    bare_values: str,
    decimal_places: int | None = None,
) -> dict[tuple[str, ...], str]:
    """Resolve one leakage-safe dataset assignment per existing geometry."""

    assignments: dict[tuple[str, ...], str] = {}
    for raw_path in csv_paths:
        path = Path(raw_path)
        rows = read_csv_rows(path)
        source_mode = resolve_bare_values_for_rows(
            rows,
            parameters,
            bare_values,
        )
        rows, _cleanup = clean_existing_geometry_rows(
            path,
            rows,
            parameters,
            split_var,
            bare_values=source_mode,
            decimal_places=decimal_places,
        )
        for row in rows:
            point = row_unit_point(row, parameters, bare_values=source_mode)
            if point is None or not in_unit_cube(point):
                continue
            key = geometry_output_key_from_row(
                row,
                parameters,
                decimal_places,
                bare_values=source_mode,
            )
            assert key is not None
            group = coverage_split_group(lookup_row_value(row, split_var))
            # Any geometry exposed to training cannot be counted as an
            # independent verification geometry.
            if assignments.get(key) != "training" or group == "training":
                assignments[key] = group
    return assignments


def existing_csv_dataset_counts(
    csv_paths: Sequence[str],
    parameters: Sequence[ParameterSpec],
    split_var: str,
    bare_values: str,
    decimal_places: int | None = None,
) -> dict[str, int]:
    assignments = existing_csv_dataset_assignments(
        csv_paths,
        parameters,
        split_var,
        bare_values,
        decimal_places,
    )
    return {
        "training": sum(group == "training" for group in assignments.values()),
        "verification": sum(
            group == "verification" for group in assignments.values()
        ),
    }


def automatic_verification_plan(
    *,
    dimensions: int,
    existing_training_count: int,
    verification_observation_count: int,
    requested_training_count: int,
    enabled: bool,
    interval: int | None = None,
    batch: int | None = None,
    max_add: int | None = None,
) -> dict[str, object]:
    """Plan milestone-based acquisition-verification growth for adaptive GP use."""

    # Preserve the intentionally lean adaptive-campaign seed. The error GP can
    # start sparse; hybrid uncertainty/coverage roles and this milestone policy
    # grow its verification observations over later rounds.
    initial_training = max(4 * dimensions, 12)
    initial_verification = max(dimensions + 2, 6)
    effective_interval = interval if interval is not None else max(2 * dimensions, 1)
    effective_batch = batch if batch is not None else max(
        2,
        int(math.ceil(2 * dimensions / 3.0)),
    )
    effective_max_add = max_add if max_add is not None else initial_verification
    if effective_interval <= 0:
        raise ValueError("--verification-interval must be positive")
    if effective_batch <= 0:
        raise ValueError("--verification-batch must be positive")
    if effective_max_add <= 0:
        raise ValueError("--verification-max-add must be positive")

    projected_training = existing_training_count + requested_training_count
    completed_milestones = max(
        0,
        (projected_training - initial_training) // effective_interval,
    )
    target_verification = (
        initial_verification + completed_milestones * effective_batch
    )
    needed = max(0, target_verification - verification_observation_count)
    added = (
        min(needed, effective_max_add)
        if enabled and projected_training >= initial_training
        else 0
    )
    next_trigger = (
        initial_training
        if projected_training < initial_training
        else initial_training + (completed_milestones + 1) * effective_interval
    )
    return {
        "enabled": bool(enabled),
        "dimensions": dimensions,
        "initial_training_anchor": initial_training,
        "initial_verification_target": initial_verification,
        "existing_training_count": existing_training_count,
        "requested_training_count": requested_training_count,
        "projected_training_count": projected_training,
        "verification_observation_count": verification_observation_count,
        "training_interval": effective_interval,
        "verification_batch": effective_batch,
        "maximum_additional_verification_per_command": effective_max_add,
        "completed_growth_milestones": completed_milestones,
        "target_verification_count": target_verification,
        "needed_verification_count": needed,
        "additional_verification_count": added,
        "next_training_trigger": next_trigger,
    }


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
) -> tuple[list[ErrorRegion], str, str]:
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

    available_columns = list(rows[0])
    available_normalized = {normalize_key(column) for column in available_columns}
    missing_parameter_columns = [
        parameter.name
        for parameter in parameters
        if normalize_key(parameter.name) not in available_normalized
    ]
    metric_column_missing = normalize_key(metric_name) not in available_normalized

    def attempt(mode: str) -> tuple[list[ErrorRegion], dict[str, object]]:
        groups: dict[tuple[object, ...], dict[str, object]] = {}
        usable_metric_rows = 0
        complete_parameter_rows = 0
        parsed_parameter_rows = 0
        accepted_rows = 0
        invalid_samples: list[str] = []
        outside_samples: list[str] = []
        for row in rows:
            value = csv_number(lookup_row_value(row, metric_name))
            if value is None:
                continue
            usable_metric_rows += 1
            raw_values = [lookup_row_value(row, parameter.name) for parameter in parameters]
            if any(raw is None or not str(raw).strip() for raw in raw_values):
                continue
            complete_parameter_rows += 1
            point: list[float] = []
            invalid = False
            for parameter, raw in zip(parameters, raw_values):
                try:
                    observed = parse_observed_value(raw, parameter, bare_values=mode)
                    unit_value = unit_coordinate_for_value(observed, parameter)
                except (ValueError, OverflowError) as exc:
                    if len(invalid_samples) < 3:
                        invalid_samples.append(
                            f"{parameter.name}={raw!r} ({exc})"
                        )
                    invalid = True
                    break
                if not math.isfinite(unit_value):
                    if len(invalid_samples) < 3:
                        invalid_samples.append(
                            f"{parameter.name}={raw!r} (non-finite normalized value)"
                        )
                    invalid = True
                    break
                point.append(unit_value)
            if invalid:
                continue
            parsed_parameter_rows += 1
            if not in_unit_cube(point):
                if len(outside_samples) < 3:
                    outside = [
                        f"{parameter.name}={raw!r} -> u={coordinate:.6g}"
                        for parameter, raw, coordinate in zip(
                            parameters, raw_values, point
                        )
                        if coordinate < -1e-9 or coordinate > 1.0 + 1e-9
                    ]
                    outside_samples.append(", ".join(outside))
                continue
            accepted_rows += 1
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
            bucket["weighted_sum"] = (
                float(bucket["weighted_sum"]) + weight * score * score
            )
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
        return regions, {
            "mode": mode,
            "usable_metric_rows": usable_metric_rows,
            "complete_parameter_rows": complete_parameter_rows,
            "parsed_parameter_rows": parsed_parameter_rows,
            "accepted_rows": accepted_rows,
            "invalid_samples": invalid_samples,
            "outside_samples": outside_samples,
        }

    requested_mode = bare_values
    modes = ["parameter-units", "base-units"]
    attempts = [attempt(mode) for mode in modes]
    if requested_mode == "auto":
        regions, diagnostics = max(
            attempts,
            key=lambda item: (
                int(item[1]["accepted_rows"]),
                len(item[0]),
                item[1]["mode"] == "parameter-units",
            ),
        )
    else:
        regions, diagnostics = next(
            item for item in attempts if item[1]["mode"] == requested_mode
        )
    effective_mode = str(diagnostics["mode"])
    if regions:
        return regions, metric_name, effective_mode

    details = [
        f"Could not use any row in {metrics_path} for metric {metric_name!r}."
    ]
    if metric_column_missing:
        details.append(f"The metric column {metric_name!r} is missing.")
    elif int(diagnostics["usable_metric_rows"]) == 0:
        details.append(
            f"Column {metric_name!r} exists, but every value is blank, non-numeric, or non-finite."
        )
    if missing_parameter_columns:
        details.append(
            "Missing requested parameter column(s): "
            + ", ".join(missing_parameter_columns)
            + "."
        )
    for _, item in attempts:
        details.append(
            f"With unitless values interpreted as {item['mode']}: "
            f"{item['usable_metric_rows']} row(s) had a usable metric, "
            f"{item['complete_parameter_rows']} also had every parameter, "
            f"{item['parsed_parameter_rows']} parsed, and "
            f"{item['accepted_rows']} were inside the declared domain."
        )
        invalid_samples = item["invalid_samples"]
        outside_samples = item["outside_samples"]
        if invalid_samples:
            details.append("Example invalid value(s): " + "; ".join(invalid_samples))
        if outside_samples:
            details.append(
                "Example out-of-range value(s): " + "; ".join(outside_samples)
            )
    if requested_mode != "auto":
        alternate_regions, alternate_diagnostics = next(
            item for item in attempts if item[1]["mode"] != requested_mode
        )
        if alternate_regions:
            details.append(
                f"The alternate --bare-values {alternate_diagnostics['mode']} "
                f"interpretation would accept "
                f"{alternate_diagnostics['accepted_rows']} row(s). Use that mode "
                "or set --bare-values auto."
            )
    details.append("Available columns: " + ", ".join(available_columns) + ".")
    details.append(
        "Requested domains: "
        + ", ".join(
            f"{parameter.name}=[{format_value(parameter.lower, parameter.unit)}, "
            f"{format_value(parameter.upper, parameter.unit)}]"
            for parameter in parameters
        )
        + "."
    )
    raise ValueError("\n".join(details))


def _linear_quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = min(max(float(fraction), 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def error_region_summary(regions: Sequence[ErrorRegion]) -> dict[str, float]:
    scores = [max(float(region.score), 0.0) for region in regions]
    if not scores:
        return {"rms": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "rms": math.sqrt(sum(value * value for value in scores) / len(scores)),
        "median": _linear_quantile(scores, 0.5),
        "p90": _linear_quantile(scores, 0.9),
        "max": max(scores),
    }


def recommend_additional_point_count(
    *,
    dimensions: int,
    regions: Sequence[ErrorRegion],
    existing_training_count: int,
    target_error: float | None,
    previous_error_rms: Sequence[float] = (),
) -> PointCountRecommendation:
    """Recommend a dimension-scaled primary batch from accuracy and progress."""

    if dimensions <= 0:
        raise ValueError("Point-count recommendation requires at least one dimension")
    summary = error_region_summary(regions)
    current_error = summary["rms"]
    target_ratio = (
        current_error / target_error
        if target_error is not None and target_error > 0.0
        else None
    )
    history = [
        float(value)
        for value in previous_error_rms
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]
    latest_improvement = None
    if history and history[-1] > 0.0:
        latest_improvement = (history[-1] - current_error) / history[-1]

    minimum = max(dimensions, 4)
    moderate = max(2 * dimensions, minimum)
    aggressive = max(3 * dimensions, moderate)
    rationale: list[str] = []

    if target_ratio is not None and target_ratio <= 1.0:
        count = 0
        stage = "target-met"
        rationale.append(
            f"current RMS error {current_error:.6g} meets target {target_error:.6g}"
        )
    elif target_ratio is not None and target_ratio >= 4.0:
        count = aggressive
        stage = "far-from-target"
        rationale.append(
            f"current RMS error is {target_ratio:.3g}x the requested target"
        )
    elif target_ratio is not None and target_ratio >= 2.0:
        count = moderate
        stage = "above-target"
        rationale.append(
            f"current RMS error is {target_ratio:.3g}x the requested target"
        )
    elif target_ratio is not None:
        count = max(minimum, int(math.ceil(1.5 * dimensions)))
        stage = "near-target"
        rationale.append(
            f"current RMS error is {target_ratio:.3g}x the requested target"
        )
    else:
        count = moderate
        stage = "coverage-guided"
        rationale.append(
            "no --target-error was supplied, so the recommendation uses geometry "
            "and error-observation density"
        )

    lean_training = max(4 * dimensions, 12)
    if count and existing_training_count < lean_training:
        count = max(count, moderate)
        rationale.append(
            f"training inventory {existing_training_count} is below the lean "
            f"{lean_training}-point dimension-scaled anchor"
        )

    observation_target = max(3 * dimensions, 12)
    if count and len(regions) < observation_target:
        count = max(count, moderate)
        rationale.append(
            f"only {len(regions)} GP error observations are available; "
            f"approximately {observation_target} are preferred for {dimensions}D"
        )

    if count and latest_improvement is not None:
        if latest_improvement < -0.02:
            count = minimum
            stage = "regressed"
            rationale.append(
                f"RMS error regressed by {-100.0 * latest_improvement:.1f}% in the "
                "latest recorded round; use a diagnostic-sized batch"
            )
        elif latest_improvement < 0.05:
            count = min(max(count, moderate), moderate)
            stage = "plateau"
            rationale.append(
                f"latest RMS improvement was only {100.0 * latest_improvement:.1f}%; "
                "a larger blind batch is unlikely to be point-efficient"
            )
        elif latest_improvement >= 0.20:
            count = max(minimum, min(count, moderate))
            stage = "improving"
            rationale.append(
                f"latest RMS improvement was {100.0 * latest_improvement:.1f}%; "
                "continue with a moderate batch and remeasure"
            )

    count = min(max(count, 0), aggressive)
    return PointCountRecommendation(
        recommended_count=count,
        dimensions=dimensions,
        current_error_rms=current_error,
        current_error_median=summary["median"],
        current_error_p90=summary["p90"],
        current_error_max=summary["max"],
        target_error=target_error,
        target_ratio=target_ratio,
        previous_error_rms=history,
        latest_improvement_fraction=latest_improvement,
        existing_training_count=existing_training_count,
        verification_observation_count=len(regions),
        stage=stage,
        rationale=rationale,
    )


def hybrid_component_allocation(
    count: int,
    *,
    dimensions: int,
    observation_count: int,
    target_ratio: float | None,
    latest_improvement_fraction: float | None,
) -> HybridAllocation:
    """Allocate a hybrid batch across exploitation, uncertainty, and coverage."""

    if count <= 0:
        return HybridAllocation(0, 0, 0, "empty")
    observation_target = max(3 * dimensions, 12)
    if observation_count < observation_target:
        fractions = (0.35, 0.35, 0.30)
        regime = "sparse-error-observations"
    elif latest_improvement_fraction is not None and latest_improvement_fraction < 0.05:
        fractions = (0.40, 0.25, 0.35)
        regime = "plateau"
    elif target_ratio is not None and target_ratio >= 2.0:
        fractions = (0.60, 0.20, 0.20)
        regime = "far-from-target"
    elif target_ratio is not None and target_ratio < 1.5:
        fractions = (0.45, 0.25, 0.30)
        regime = "near-target"
    else:
        fractions = (0.50, 0.25, 0.25)
        regime = "balanced"

    raw = [count * fraction for fraction in fractions]
    allocated = [int(math.floor(value)) for value in raw]
    if count >= 3:
        allocated = [max(1, value) for value in allocated]
    while sum(allocated) > count:
        index = max(range(3), key=lambda idx: allocated[idx] - raw[idx])
        if allocated[index] <= (1 if count >= 3 else 0):
            break
        allocated[index] -= 1
    while sum(allocated) < count:
        index = max(range(3), key=lambda idx: raw[idx] - allocated[idx])
        allocated[index] += 1
    return HybridAllocation(
        exploitation=allocated[0],
        uncertainty=allocated[1],
        coverage=allocated[2],
        regime=regime,
    )


def adaptive_exploration_weight(
    recommendation: PointCountRecommendation,
) -> float:
    """Return a conservative-to-exploitative GP schedule for the current round."""

    preferred_observations = max(3 * recommendation.dimensions, 12)
    if recommendation.verification_observation_count < preferred_observations:
        return 2.5
    if recommendation.stage in {"plateau", "regressed", "near-target"}:
        return 0.75
    if recommendation.stage in {"far-from-target", "above-target", "improving"}:
        return 1.0
    return 1.5


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
    return out_path.with_name(
        f"{out_path.stem}_verification_error_regions{out_path.suffix or '.csv'}"
    )


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
    length_scale: float | Sequence[float],
) -> float:
    if isinstance(length_scale, (int, float)):
        scales = [float(length_scale)] * len(lhs)
    else:
        scales = [float(value) for value in length_scale]
        if len(scales) != len(lhs):
            raise ValueError(
                f"Expected {len(lhs)} GP length scales, received {len(scales)}"
            )
    if any(not math.isfinite(value) or value <= 0.0 for value in scales):
        raise ValueError("Every GP length scale must be positive and finite")
    distance2 = sum(
        ((float(a) - float(b)) / scale) ** 2
        for a, b, scale in zip(lhs, rhs, scales)
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
    length_scale: float | Sequence[float],
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
    length_scale: float | Sequence[float] | None,
    noise_variance: float,
    error_floor: float,
    ard_mode: str = "auto",
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
    dimensions = len(points[0])
    scale_grid = [0.08, 0.12, 0.18, 0.27, 0.4, 0.6, 0.9, 1.35]
    explicit_scales: list[float] | None = None
    if length_scale is not None:
        if isinstance(length_scale, (int, float)):
            explicit_scales = [float(length_scale)] * dimensions
        else:
            explicit_scales = [float(value) for value in length_scale]
            if len(explicit_scales) != dimensions:
                raise ValueError(
                    f"--gp-length-scale supplied {len(explicit_scales)} values "
                    f"for a {dimensions}-dimensional geometry"
                )
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in explicit_scales
        ):
            raise ValueError("Every --gp-length-scale value must be positive")

    isotropic_candidates = (
        [explicit_scales[0]]
        if explicit_scales is not None and len(set(explicit_scales)) == 1
        else ([] if explicit_scales is not None else scale_grid)
    )
    best: tuple[list[list[float]], list[float], float, list[float]] | None = None
    if explicit_scales is not None and len(set(explicit_scales)) > 1:
        factor, alpha, likelihood = _factor_gp_candidate(
            points,
            normalized_targets,
            explicit_scales,
            noise_variance,
        )
        best = (factor, alpha, likelihood, explicit_scales)
    for candidate in isotropic_candidates:
        candidate_scales = [float(candidate)] * dimensions
        factor, alpha, likelihood = _factor_gp_candidate(
            points,
            normalized_targets,
            candidate_scales,
            noise_variance,
        )
        if best is None or likelihood > best[2]:
            best = (factor, alpha, likelihood, candidate_scales)
    assert best is not None

    use_ard = (
        explicit_scales is None
        and ard_mode in {"auto", "on"}
        and (
            ard_mode == "on"
            or len(points) >= max(3 * dimensions, 12)
        )
    )
    selection = "user" if explicit_scales is not None else "isotropic-likelihood"
    if use_ard:
        # Coordinate-wise marginal-likelihood refinement avoids an exponential
        # Cartesian search. A weak log-scale shrinkage penalty prevents sparse
        # error observations from forcing unrelated dimensions to opposite
        # extremes of the search grid.
        scales = list(best[3])
        base_scale = math.exp(sum(math.log(value) for value in scales) / dimensions)
        penalty_weight = 0.15

        def selection_score(likelihood: float, values: Sequence[float]) -> float:
            shrinkage = sum(
                math.log(value / base_scale) ** 2 for value in values
            )
            return likelihood - penalty_weight * shrinkage

        current_score = selection_score(best[2], scales)
        for dimension in range(dimensions):
            dimension_best = best
            dimension_score = current_score
            local_grid = sorted(
                {
                    *scale_grid,
                    max(0.05, min(2.0, base_scale * 0.5)),
                    max(0.05, min(2.0, base_scale * 1.5)),
                    max(0.05, min(2.0, base_scale * 2.0)),
                }
            )
            for candidate in local_grid:
                trial_scales = list(scales)
                trial_scales[dimension] = candidate
                factor, alpha, likelihood = _factor_gp_candidate(
                    points,
                    normalized_targets,
                    trial_scales,
                    noise_variance,
                )
                score = selection_score(likelihood, trial_scales)
                if score > dimension_score + 1e-12:
                    dimension_best = (
                        factor,
                        alpha,
                        likelihood,
                        trial_scales,
                    )
                    dimension_score = score
            best = dimension_best
            scales = list(best[3])
            current_score = dimension_score
        selection = "ard-coordinate-likelihood"

    return GaussianProcessModel(
        observation_points=points,
        log_error_mean=log_error_mean,
        log_error_scale=log_error_scale,
        length_scales=list(best[3]),
        noise_variance=noise_variance,
        cholesky=best[0],
        alpha=best[1],
        log_marginal_likelihood=best[2],
        normalized_targets=normalized_targets,
        length_scale_selection=selection,
    )


def predict_error_gaussian_process(
    model: GaussianProcessModel,
    point: Sequence[float],
) -> tuple[float, float]:
    covariance = [
        _matern52_kernel(point, observed, model.length_scales)
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


def condition_gp_on_fantasy_mean(
    model: GaussianProcessModel,
    point: Sequence[float],
) -> GaussianProcessModel:
    """Condition on the current posterior mean to reduce batch uncertainty."""

    mean_log_error, _ = predict_error_gaussian_process(model, point)
    normalized_target = (
        (mean_log_error - model.log_error_mean) / model.log_error_scale
        if model.log_error_scale > 0.0
        else 0.0
    )
    observation_points = [*model.observation_points, list(point)]
    normalized_targets = [*model.normalized_targets, normalized_target]
    factor, alpha, likelihood = _factor_gp_candidate(
        observation_points,
        normalized_targets,
        model.length_scales,
        model.noise_variance,
    )
    return GaussianProcessModel(
        observation_points=observation_points,
        log_error_mean=model.log_error_mean,
        log_error_scale=model.log_error_scale,
        length_scales=list(model.length_scales),
        noise_variance=model.noise_variance,
        cholesky=factor,
        alpha=alpha,
        log_marginal_likelihood=likelihood,
        normalized_targets=normalized_targets,
        length_scale_selection=model.length_scale_selection,
    )


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
    length_scale: float | Sequence[float] | None,
    noise_variance: float,
    error_floor: float,
    ard_mode: str = "auto",
) -> tuple[list[SuggestedPoint], GaussianProcessModel]:
    model = fit_error_gaussian_process(
        regions,
        length_scale=length_scale,
        noise_variance=noise_variance,
        error_floor=error_floor,
        ard_mode=ard_mode,
    )
    working_model = model
    selected: list[SuggestedPoint] = []
    occupied = [list(point) for point in existing_points]
    diag = max(
        math.sqrt(len(candidate_points[0])) if candidate_points else 1.0,
        1e-12,
    )
    unused = [list(point) for point in candidate_points]

    while len(selected) < count and unused:
        best_idx: int | None = None
        best_point: SuggestedPoint | None = None
        for idx, point in enumerate(unused):
            mean_log_error, std_log_error = predict_error_gaussian_process(
                working_model,
                point,
            )
            predicted_error = math.exp(min(700.0, mean_log_error))
            upper_error = math.exp(
                min(700.0, mean_log_error + exploration_weight * std_log_error)
            )
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
                selection_component="gp-ucb",
            )
            if best_point is None or candidate.acquisition_score > best_point.acquisition_score:
                best_idx = idx
                best_point = candidate
        if best_idx is None or best_point is None:
            break
        selected.append(best_point)
        occupied.append(best_point.unit_point)
        unused.pop(best_idx)
        working_model = condition_gp_on_fantasy_mean(
            working_model,
            best_point.unit_point,
        )
    return selected, model


def _hybrid_component_schedule(allocation: HybridAllocation) -> list[str]:
    targets = {
        "exploitation": allocation.exploitation,
        "uncertainty": allocation.uncertainty,
        "coverage": allocation.coverage,
    }
    total = sum(targets.values())
    selected = {name: 0 for name in targets}
    schedule: list[str] = []
    order = ("exploitation", "uncertainty", "coverage")
    for step in range(total):
        available = [name for name in order if selected[name] < targets[name]]
        component = max(
            available,
            key=lambda name: (
                targets[name] * (step + 1) / max(total, 1) - selected[name],
                -order.index(name),
            ),
        )
        schedule.append(component)
        selected[component] += 1
    return schedule


def select_hybrid_points(
    candidate_points: Sequence[Sequence[float]],
    regions: Sequence[ErrorRegion],
    existing_points: Sequence[Sequence[float]],
    count: int,
    allocation: HybridAllocation,
    exploration_weight: float,
    novelty_power: float,
    min_distance: float,
    length_scale: float | Sequence[float] | None,
    noise_variance: float,
    error_floor: float,
    ard_mode: str = "auto",
) -> tuple[list[SuggestedPoint], GaussianProcessModel, dict[str, object]]:
    """Select an adaptive mixture of error, uncertainty, and coverage points."""

    model = fit_error_gaussian_process(
        regions,
        length_scale=length_scale,
        noise_variance=noise_variance,
        error_floor=error_floor,
        ard_mode=ard_mode,
    )
    working_model = model
    selected: list[SuggestedPoint] = []
    occupied = [list(point) for point in existing_points]
    unused = [list(point) for point in candidate_points]
    diag = max(
        math.sqrt(len(candidate_points[0])) if candidate_points else 1.0,
        1e-12,
    )
    schedule = _hybrid_component_schedule(allocation)

    initial_predictions: list[float] = []
    initial_uncertainties: list[float] = []
    for point in unused:
        mean_log_error, std_log_error = predict_error_gaussian_process(model, point)
        initial_predictions.append(math.exp(min(700.0, mean_log_error)))
        initial_uncertainties.append(std_log_error)

    for component in schedule:
        best_idx: int | None = None
        best_point: SuggestedPoint | None = None
        for idx, point in enumerate(unused):
            distance_existing = nearest_distance(point, occupied)
            if distance_existing < min_distance:
                continue
            novelty = min(1.0, distance_existing / diag)
            mean_log_error, std_log_error = predict_error_gaussian_process(
                working_model,
                point,
            )
            predicted_error = math.exp(min(700.0, mean_log_error))
            upper_error = math.exp(
                min(700.0, mean_log_error + exploration_weight * std_log_error)
            )
            if component == "exploitation":
                component_novelty = min(novelty_power, 0.5)
                score = predicted_error * (
                    max(novelty, 1e-12) ** component_novelty
                )
            elif component == "uncertainty":
                component_novelty = max(0.5, min(novelty_power, 1.5))
                score = std_log_error * (
                    max(novelty, 1e-12) ** component_novelty
                )
            else:
                score = novelty
            region, region_distance = _nearest_error_region(point, regions)
            candidate = SuggestedPoint(
                unit_point=point,
                acquisition_score=score,
                distance_to_existing=distance_existing,
                nearest_error_source_index=(region.source_index if region else ""),
                nearest_error_score=(region.score if region else 0.0),
                nearest_error_distance=region_distance,
                predicted_error=predicted_error,
                gp_log_uncertainty=std_log_error,
                gp_upper_confidence_error=upper_error,
                selection_component=component,
            )
            if best_point is None or candidate.acquisition_score > best_point.acquisition_score:
                best_idx = idx
                best_point = candidate
        if best_idx is None or best_point is None:
            break
        selected.append(best_point)
        occupied.append(best_point.unit_point)
        unused.pop(best_idx)
        working_model = condition_gp_on_fantasy_mean(
            working_model,
            best_point.unit_point,
        )

    prediction_median = _linear_quantile(initial_predictions, 0.5)
    prediction_p10 = _linear_quantile(initial_predictions, 0.1)
    prediction_p90 = _linear_quantile(initial_predictions, 0.9)
    prediction_spread_ratio = (
        prediction_p90 / prediction_p10
        if prediction_p10 > 0.0
        else None
    )
    diagnostics: dict[str, object] = {
        "allocation": {
            "exploitation": allocation.exploitation,
            "uncertainty": allocation.uncertainty,
            "coverage": allocation.coverage,
            "regime": allocation.regime,
        },
        "selected_components": {
            name: sum(item.selection_component == name for item in selected)
            for name in ("exploitation", "uncertainty", "coverage")
        },
        "candidate_prediction": {
            "median": prediction_median,
            "p10": prediction_p10,
            "p90": prediction_p90,
            "p90_to_p10_ratio": prediction_spread_ratio,
            "median_log_uncertainty": _linear_quantile(
                initial_uncertainties,
                0.5,
            ),
        },
        "batch_posterior_update": "kriging-believer-posterior-mean",
    }
    return selected, model, diagnostics


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
                selection_component="error-distance",
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
    target_datasets: Sequence[str] | None,
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
        "point_origin",
        "method",
        "acquisition_method",
        "selection_component",
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

    if target_datasets is not None and len(target_datasets) != len(suggestions):
        raise ValueError(
            "One target dataset label is required for every suggested point"
        )
    rows: list[dict[str, object]] = []
    for idx, suggestion in enumerate(suggestions, start=1):
        point_dataset = (
            str(target_datasets[idx - 1])
            if target_datasets is not None
            else target_dataset
        )
        point_dataset = canonical_dataset_label(point_dataset)
        row: dict[str, object] = {
            "point_index": idx,
            split_var: point_dataset,
            "additional_sequence": idx,
            "point_origin": "additional",
            "method": method_name,
            "acquisition_method": acquisition_method,
            "selection_component": suggestion.selection_component,
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
    require_combined_geometry_path(path, "--out")
    validate_geometry_output_rows(
        path,
        rows,
        parameters,
        split_var,
        bare_values="parameter-units",
        decimal_places=decimal_places,
    )
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


def accumulated_geometry_path(path: Path) -> Path:
    """Return the combined cumulative geometry CSV beside a new-point CSV."""

    return path.with_name(f"{path.stem}_all_geometries.csv")


def write_dataset_split_geometry_views(
    source_path: Path,
    parameters: Sequence[ParameterSpec],
    split_var: str,
    *,
    bare_values: str = "parameter-units",
    decimal_places: int | None = None,
    coverage_rows: Sequence[dict[str, object]] | None = None,
) -> list[Path]:
    """Write RFPro queues with coverage plots placed in full prior context."""

    fields, rows = read_csv_table(source_path)
    contextual_rows = list(coverage_rows) if coverage_rows is not None else rows
    written: list[Path] = []
    for group, suffix in (("training", "train"), ("verification", "verification")):
        split_rows = [
            row
            for row in rows
            if coverage_split_group(
                lookup_row_value(row, split_var) or "train"
            )
            == group
        ]
        if not split_rows:
            continue
        split_coverage_rows = [
            row
            for row in contextual_rows
            if coverage_split_group(
                lookup_row_value(row, split_var) or "train"
            )
            == group
        ]
        split_path = split_output_path(source_path, suffix)
        validate_geometry_output_rows(
            split_path,
            split_rows,
            parameters,
            split_var,
            bare_values=bare_values,
            decimal_places=decimal_places,
        )
        write_rows_csv(split_path, split_rows, fields)
        plot_path = write_parameter_coverage_png(
            split_path,
            parameters,
            split_coverage_rows or split_rows,
            split_var,
            bare_values=bare_values,
        )
        written.extend([split_path, plot_path])
    return written


def write_accumulated_geometries(
    path: Path,
    parameters: Sequence[ParameterSpec],
    split_var: str,
    existing_csv_paths: Sequence[str],
    observed_points: Sequence[tuple[Sequence[float], str, str, object]],
    additional_path: Path,
    *,
    include_normalized: bool,
    decimal_places: int | None,
    bare_values: str,
    method: str,
    metadata_extra: dict[str, object] | None = None,
) -> Path:
    """Write a deduplicated union suitable for the next GP acquisition round."""

    require_combined_geometry_path(path, "--combined-out")

    fields = [
        "point_index",
        split_var,
        "split_sequence",
        "train_sequence",
        "verification_sequence",
        "point_origin",
        "method",
        "geometry_source",
        "source_point_index",
    ]
    if include_normalized:
        fields.extend(f"u_{parameter.name}" for parameter in parameters)
    fields.extend(parameter.name for parameter in parameters)

    records: list[tuple[list[float], str, str, str, str, object]] = []
    seen: dict[tuple[str, ...], int] = {}
    duplicate_count = 0
    same_split_duplicate_count = 0
    cross_split_duplicate_count = 0
    legacy_dataset_rows_normalized = 0
    filename_role_mismatch_count = 0
    cross_split_conflicts: list[dict[str, object]] = []

    def append_point(
        point: Sequence[float],
        dataset: object,
        point_origin: object,
        source_method: object,
        source: str,
        source_index: object,
    ) -> None:
        nonlocal duplicate_count
        nonlocal same_split_duplicate_count
        nonlocal cross_split_duplicate_count
        nonlocal legacy_dataset_rows_normalized
        raw_dataset = dataset
        dataset = canonical_dataset_label(dataset, default="train")
        if normalize_key(raw_dataset) in LEGACY_TRAIN_DATASET_TOKENS:
            legacy_dataset_rows_normalized += 1
        if len(point) != len(parameters) or not in_unit_cube(point):
            coordinates = ", ".join(
                f"{parameter.name}:u={float(coordinate):.6g} "
                f"for [{format_value(parameter.lower, parameter.unit)}, "
                f"{format_value(parameter.upper, parameter.unit)}]"
                for parameter, coordinate in zip(parameters, point)
            )
            raise ValueError(
                f"Geometry source {source} row {source_index} is outside the "
                f"declared parameter domain ({coordinates})"
            )
        rounded_values = [
            round_parameter_value(
                map_unit_point(float(coordinate), parameter),
                parameter,
                decimal_places,
            )
            for parameter, coordinate in zip(parameters, point)
        ]
        key = geometry_output_key_from_values(
            rounded_values,
            parameters,
            decimal_places,
        )
        if key in seen:
            duplicate_count += 1
            existing_index = seen[key]
            existing_record = records[existing_index]
            existing_dataset = existing_record[1]
            new_dataset = str(dataset)
            if str(point_origin or "").lower() == "additional":
                raise ValueError(
                    "A newly suggested geometry becomes a duplicate after "
                    f"rounding: {source} row {source_index} matches "
                    f"{existing_record[4]} row {existing_record[5]}. Increase "
                    "--decimal-places or --min-distance."
                )
            if existing_dataset == new_dataset:
                same_split_duplicate_count += 1
            else:
                cross_split_duplicate_count += 1
                cross_split_conflicts.append(
                    {
                        "geometry": {
                            parameter.name: format_value(
                                value,
                                parameter.unit,
                                decimal_places,
                            )
                            for parameter, value in zip(parameters, rounded_values)
                        },
                        "first_dataset": existing_dataset,
                        "first_source": existing_record[4],
                        "first_source_point_index": existing_record[5],
                        "conflicting_dataset": new_dataset,
                        "conflicting_source": source,
                        "conflicting_source_point_index": source_index,
                        "retained_dataset": "train",
                    }
                )
            # A response used for training is no longer an independent
            # verification response. Training therefore wins every conflict.
            if existing_dataset == "verification" and new_dataset == "train":
                records[existing_index] = (
                    rounded_values,
                    "train",
                    str(point_origin or "existing"),
                    str(source_method or "existing"),
                    source,
                    source_index,
                )
            return
        seen[key] = len(records)
        records.append(
            (
                rounded_values,
                str(dataset),
                str(point_origin or "existing"),
                str(source_method or "existing"),
                source,
                source_index,
            )
        )

    for raw_path in existing_csv_paths:
        source_path = Path(raw_path)
        _, source_rows = read_csv_table(source_path)
        source_filename_role = geometry_file_split_group(source_path)
        default_source_dataset = (
            "train"
            if source_filename_role == "training"
            else "verification" if source_filename_role == "verification" else None
        )
        source_mode = resolve_bare_values_for_rows(
            source_rows,
            parameters,
            bare_values,
        )
        for row_number, row in enumerate(source_rows, start=2):
            point = row_unit_point(row, parameters, bare_values=source_mode)
            if point is None:
                raise ValueError(
                    f"Could not read every parameter from {source_path} row "
                    f"{row_number} while building the cumulative geometry CSV"
                )
            raw_dataset = lookup_row_value(row, split_var)
            if not str(raw_dataset or "").strip() and default_source_dataset is None:
                raise ValueError(
                    f"Combined geometry file {source_path} row {row_number} has "
                    f"no {split_var!r} value. Combined files must explicitly "
                    "identify every row as train or verification."
                )
            dataset = canonical_dataset_label(
                raw_dataset,
                default=default_source_dataset,
            )
            if (
                source_filename_role is not None
                and coverage_split_group(dataset) != source_filename_role
            ):
                filename_role_mismatch_count += 1
            append_point(
                point,
                raw_dataset or default_source_dataset,
                "existing",
                lookup_row_value(row, "method") or "existing",
                str(lookup_row_value(row, "geometry_source") or source_path),
                lookup_row_value(row, "source_point_index")
                or lookup_row_value(row, "point_index")
                or row_number - 1,
            )

    for point, dataset, source, source_index in observed_points:
        append_point(
            point,
            dataset,
            "existing",
            "existing-observation",
            source,
            source_index,
        )

    additional_rows = read_csv_rows(additional_path)
    additional_mode = resolve_bare_values_for_rows(
        additional_rows,
        parameters,
        bare_values,
    )
    for row_number, row in enumerate(additional_rows, start=2):
        point = row_unit_point(row, parameters, bare_values=additional_mode)
        if point is None:
            raise ValueError(
                f"Could not read every parameter from {additional_path} row "
                f"{row_number} while building the cumulative geometry CSV"
            )
        append_point(
            point,
            lookup_row_value(row, split_var) or "train",
            lookup_row_value(row, "point_origin") or "additional",
            lookup_row_value(row, "method") or method,
            str(additional_path),
            lookup_row_value(row, "point_index") or row_number - 1,
        )

    split_sequences = {"training": 0, "verification": 0}
    rows: list[dict[str, object]] = []
    for point_index, record in enumerate(records, start=1):
        values, dataset, point_origin, source_method, source, source_index = record
        split_group = coverage_split_group(dataset)
        split_sequences[split_group] += 1
        split_sequence = split_sequences[split_group]
        row: dict[str, object] = {
            "point_index": point_index,
            split_var: dataset,
            "split_sequence": split_sequence,
            "train_sequence": split_sequence if split_group == "training" else "",
            "verification_sequence": (
                split_sequence if split_group == "verification" else ""
            ),
            "point_origin": point_origin,
            "method": source_method,
            "geometry_source": source,
            "source_point_index": source_index,
        }
        if include_normalized:
            for parameter, value in zip(parameters, values):
                row[f"u_{parameter.name}"] = (
                    f"{unit_coordinate_for_value(value, parameter):.16g}"
                )
        for parameter, value in zip(parameters, values):
            row[parameter.name] = format_value(
                value,
                parameter.unit,
                decimal_places,
            )
        rows.append(row)

    validate_geometry_output_rows(
        path,
        rows,
        parameters,
        split_var,
        bare_values=bare_values,
        decimal_places=decimal_places,
    )
    write_rows_csv(path, rows, fields)
    if legacy_dataset_rows_normalized:
        print(
            "warning: normalized "
            f"{legacy_dataset_rows_normalized} legacy targeted/additional "
            "dataset label(s) to train in the cumulative geometry inventory",
            file=sys.stderr,
        )
    if cross_split_duplicate_count:
        print(
            "warning: removed "
            f"{cross_split_duplicate_count} geometry duplicate(s) shared by "
            "training and verification; retained each as training to prevent "
            "verification leakage",
            file=sys.stderr,
        )
    if filename_role_mismatch_count:
        print(
            "warning: migrated "
            f"{filename_role_mismatch_count} row(s) from legacy mixed geometry "
            "files whose names indicated a single split",
            file=sys.stderr,
        )
    return write_geometry_metadata(
        path,
        parameters,
        rows,
        split_var,
        generation_kind="accumulated_geometries",
        method=method,
        decimal_places=decimal_places,
        bare_values=bare_values,
        extra={
            "additional_points_file": str(additional_path),
            "existing_geometry_files": list(existing_csv_paths),
            "deduplicated_input_rows": duplicate_count,
            "same_split_duplicates_removed": same_split_duplicate_count,
            "cross_split_duplicates_removed": cross_split_duplicate_count,
            "cross_split_conflict_resolution": "training_wins",
            "cross_split_conflicts": cross_split_conflicts,
            "legacy_dataset_rows_normalized": legacy_dataset_rows_normalized,
            "legacy_filename_role_mismatches": filename_role_mismatch_count,
            "next_gp_existing_points": str(path),
            **(metadata_extra or {}),
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


def parse_auto_point_count(value: object) -> int | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        count = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must be a positive integer or 'auto'") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("count must be positive")
    return count


def parse_gp_length_scale(value: object) -> float | list[float]:
    text = str(value).strip()
    try:
        values = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GP length scale must be a positive number or comma-separated list"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("every GP length scale must be positive")
    return values[0] if len(values) == 1 else values


def parse_auto_nonnegative_float(value: object) -> float | str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        number = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative number or 'auto'"
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative and finite")
    return number


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
        help=(
            "Also write *_training.csv and *_verification.csv files beside "
            "the combined CSV."
        ),
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
    parser.add_argument(
        "--count",
        type=parse_auto_point_count,
        default="auto",
        help=(
            "Number of primary additional points, or auto for a dimension-, "
            "accuracy-, and progress-scaled recommendation. Default: auto. "
            "Automatic verification points are appended beyond this count when triggered."
        ),
    )
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
        "--target-error",
        type=float,
        help=(
            "Desired RMS geometry-level value of --metric. With --count auto, "
            "the current-to-target ratio scales the recommended primary batch."
        ),
    )
    parser.add_argument(
        "--previous-verification-metrics",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Prior-round verification_metrics.csv used to measure improvement. "
            "Repeat in oldest-to-newest order."
        ),
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
        choices=["error-distance", "gp-ucb", "hybrid"],
        default="hybrid",
        help=(
            "Candidate scoring method. hybrid divides a dimension-scaled batch "
            "among exploitation, GP uncertainty, and maximin coverage and updates "
            "posterior uncertainty after each selection. gp-ucb retains one-score "
            "UCB acquisition; error-distance uses measured error hotspots. "
            "Default: hybrid."
        ),
    )
    parser.add_argument(
        "--exploration-weight",
        type=parse_auto_nonnegative_float,
        default="auto",
        help=(
            "GP-UCB standard-deviation multiplier, or auto to reduce exploration "
            "as error observations mature and accuracy approaches its target. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--gp-length-scale",
        type=parse_gp_length_scale,
        help=(
            "Optional Matérn-5/2 length scale in normalized geometry space. "
            "Supply one value for isotropic behavior or one comma-separated value "
            "per parameter. If omitted, likelihood fitting and --gp-ard apply."
        ),
    )
    parser.add_argument(
        "--gp-ard",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Per-dimension GP length-scale fitting: auto enables it at roughly "
            "3*d error observations, on forces it, and off retains one isotropic "
            "scale. Default: auto."
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
        help=(
            "Number of candidate points to score. Default: max(1000, "
            "planned-primary-and-verification-count * candidate-factor)."
        ),
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
        choices=["auto", "parameter-units", "base-units"],
        default="auto",
        help=(
            "How to interpret unitless values in metrics/MDIF/CSV rows. auto "
            "tests declared parameter units and SI base units against the saved "
            "geometry domain. Default: auto."
        ),
    )
    parser.add_argument(
        "--target-dataset",
        default="train",
        help=(
            "Dataset assigned to primary suggested points: train or "
            "verification. Legacy targeted/additional values are migrated to "
            "train. Default: train."
        ),
    )
    parser.add_argument(
        "--verification-policy",
        choices=["auto", "off"],
        default="auto",
        help=(
            "Automatically add acquisition-verification points as cumulative "
            "training count crosses dimension-based milestones. Applies to "
            "hybrid and GP-UCB training batches. Default: auto."
        ),
    )
    parser.add_argument(
        "--verification-interval",
        type=int,
        help=(
            "Training-point growth between automatic verification milestones. "
            "Default: 2*d."
        ),
    )
    parser.add_argument(
        "--verification-batch",
        type=int,
        help=(
            "Verification points added at each crossed milestone. Default: "
            "max(2, ceil(2*d/3))."
        ),
    )
    parser.add_argument(
        "--verification-max-add",
        type=int,
        help=(
            "Maximum automatic verification points added by one command, "
            "including catch-up. Default: max(d+2, 6)."
        ),
    )
    parser.add_argument(
        "--out",
        default="targeted_additional_points.csv",
        help=(
            "New-points-only CSV path. A same-stem JSON/coverage plot and a "
            "combined cumulative all-geometries CSV/JSON plus strict split "
            "views are also written."
        ),
    )
    parser.add_argument(
        "--combined-out",
        help=(
            "Cumulative CSV containing existing and newly suggested geometries for "
            "the next GP round. Its basename must not contain a training or "
            "verification role word. Default: <out>_all_geometries.csv."
        ),
    )
    parser.add_argument(
        "--analysis-out",
        help=(
            "Ranked verification-error-region CSV path. Its basename must "
            "contain verification. Default: "
            "<out>_verification_error_regions.csv."
        ),
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
    extension_cleanup_stats: dict[str, int] | None = None
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
            existing_rows, cleanup_stats = clean_existing_geometry_rows(
                Path(args.existing_points),
                existing_rows,
                extension_plan.original_parameters,
                args.split_var,
                bare_values=args.bare_values,
                decimal_places=args.decimal_places,
            )
            extension_cleanup_stats = cleanup_stats
            if cleanup_stats["legacy_dataset_rows_normalized"]:
                print(
                    "warning: normalized "
                    f"{cleanup_stats['legacy_dataset_rows_normalized']} legacy "
                    "targeted/additional dataset label(s) to train",
                    file=sys.stderr,
                )
            removed_duplicates = (
                cleanup_stats["same_split_duplicates_removed"]
                + cleanup_stats["cross_split_duplicates_removed"]
            )
            if removed_duplicates:
                print(
                    "warning: removed "
                    f"{removed_duplicates} duplicate existing geometries before "
                    "range extension; training retained every cross-split conflict",
                    file=sys.stderr,
                )
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
            f"{combined_geometry_stem(existing_path)}_extended"
            f"{existing_path.suffix or '.csv'}"
        )
    else:
        base_out_path = Path("generated_points.csv")
    sampling_parameters = (
        extension_plan.sampling_parameters
        if extension_plan is not None
        else parameters
    )
    output_parameters = (
        extension_plan.overall_parameters
        if extension_plan is not None
        else parameters
    )
    excluded_output_keys: set[tuple[str, ...]] = set()
    if extension_plan is not None:
        for row in existing_rows:
            key = geometry_output_key_from_row(
                row,
                output_parameters,
                args.decimal_places,
                bare_values=args.bare_values,
            )
            if key is not None:
                excluded_output_keys.add(key)
    written_paths: list[Path] = []
    for offset, method in enumerate(methods):
        try:
            unit_points = generate_unique_output_points(
                method,
                count=count,
                sampling_parameters=sampling_parameters,
                output_parameters=output_parameters,
                decimal_places=args.decimal_places,
                seed=args.seed + offset,
                lhs_candidates=args.lhs_candidates,
                scramble=not args.no_scramble,
                skip=args.skip,
                excluded_keys=excluded_output_keys,
            )
        except (RuntimeError, ValueError) as exc:
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
                    input_cleanup=extension_cleanup_stats,
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


def _resolved_existing_path(raw_path: object, config_path: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    supplied = Path(text).expanduser()
    candidates = [supplied] if supplied.is_absolute() else [
        Path.cwd() / supplied,
        Path(__file__).resolve().parent / supplied,
        config_path.parent / supplied,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _command_options(command: object) -> dict[str, str]:
    if not command:
        return {}
    try:
        tokens = shlex.split(str(command))
    except ValueError:
        return {}
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if "=" in token:
            name, value = token.split("=", 1)
            options[name] = value
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            options[token] = tokens[index + 1]
            index += 2
            continue
        options[token] = "true"
        index += 1
    return options


def _best_config_records(
    sweep_dir: Path,
    target_model_dir: Path,
) -> list[tuple[Path, dict[str, object]]]:
    records: list[tuple[Path, dict[str, object]]] = []
    config_paths = sorted(
        sweep_dir.glob("*_best_config.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    target_resolved = target_model_dir.resolve()
    for config_path in config_paths:
        try:
            payload = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") == "no_eligible_trial":
            continue
        raw_model_dir = payload.get("best_model_dir")
        if raw_model_dir:
            raw_path = Path(str(raw_model_dir)).expanduser()
            model_candidates = [raw_path] if raw_path.is_absolute() else [
                Path.cwd() / raw_path,
                Path(__file__).resolve().parent / raw_path,
                config_path.parent / raw_path,
            ]
            if not any(candidate.resolve() == target_resolved for candidate in model_candidates):
                continue
        elif "reranked" in config_path.name and target_model_dir.name == "best_model":
            # A rerank report is not necessarily promoted over the primary model.
            continue
        records.append((config_path, payload))
    return records


def _selected_trial_metrics_path(
    sweep_dir: Path,
    target_model_dir: Path,
) -> Path | None:
    for _config_path, payload in _best_config_records(sweep_dir, target_model_dir):
        raw_trial = payload.get("trial", payload.get("best_trial"))
        try:
            trial = int(raw_trial)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        candidate = (
            sweep_dir
            / "trials"
            / f"trial_{trial:04d}"
            / "verification_metrics.csv"
        )
        if candidate.is_file():
            return candidate
    return None


def _recover_promoted_verification_metrics(
    sweep_dir: Path,
    target_model_dir: Path,
) -> tuple[Path | None, str | None]:
    """Rebuild metrics for sweep output created before promotion retained them."""

    predicted_path = target_model_dir / "predicted_verification.mdif"
    metadata_path = target_model_dir / "metadata.json"
    if not predicted_path.is_file() or not metadata_path.is_file():
        return None, None

    records = _best_config_records(sweep_dir, target_model_dir)
    if not records:
        return None, None
    config_path, payload = records[0]
    command_options = _command_options(payload.get("reproduction_command"))
    raw_fit_data = payload.get("fit_data")
    fit_data = raw_fit_data if isinstance(raw_fit_data, dict) else {}

    def setting(json_name: str, cli_name: str, default: object = None) -> object:
        value = fit_data.get(json_name)
        if value is not None and value != "":
            return value
        return command_options.get(cli_name, default)

    mdif_path = _resolved_existing_path(setting("mdif", "--mdif"), config_path)
    verification_mdif_path = _resolved_existing_path(
        setting("verification_mdif", "--verification-mdif"),
        config_path,
    )
    if mdif_path is None and verification_mdif_path is None:
        return (
            None,
            "the selected model artifacts do not identify a readable source MDIF",
        )

    try:
        from surrogate_common import (
            parse_csv_set,
            parse_sparam_weights,
            positive_frequency_blocks,
            read_mdif,
            split_blocks,
            verification_metrics,
            write_csv,
        )

        metadata = json.loads(metadata_path.read_text())
        parameter_names = [str(name) for name in metadata["parameter_names"]]
        labels = [str(label) for label in metadata["sparam_labels"]]
        if verification_mdif_path is not None:
            truth_blocks = read_mdif(verification_mdif_path)
        else:
            assert mdif_path is not None
            source_blocks = read_mdif(mdif_path)
            split = split_blocks(
                source_blocks,
                split_var=str(setting("split_var", "--split-var", "dataset")),
                train_values=parse_csv_set(
                    str(setting("train_values", "--train-values", "train,training"))
                ),
                verify_values=parse_csv_set(
                    str(
                        setting(
                            "verify_values",
                            "--verify-values",
                            "verify,verification,test,validation",
                        )
                    )
                ),
                holdout_fraction=float(
                    setting("holdout_fraction", "--holdout-fraction", 0.2)
                ),
                seed=int(setting("seed", "--seed", 1234)),
            )
            truth_blocks = split.verify
        predicted_blocks = read_mdif(predicted_path)
        truth_rf = positive_frequency_blocks(
            truth_blocks,
            purpose="verification-metrics recovery",
        )
        predicted_rf = positive_frequency_blocks(
            predicted_blocks,
            purpose="verification-metrics recovery",
        )
        if len(truth_rf) != len(predicted_rf):
            raise ValueError(
                f"truth has {len(truth_rf)} verification block(s), but the saved "
                f"prediction has {len(predicted_rf)}"
            )
        for block_index, (truth, predicted) in enumerate(
            zip(truth_rf, predicted_rf),
            start=1,
        ):
            if len(truth.freq_hz) != len(predicted.freq_hz) or any(
                not math.isclose(
                    float(truth_frequency),
                    float(predicted_frequency),
                    rel_tol=1e-10,
                    abs_tol=1e-6,
                )
                for truth_frequency, predicted_frequency in zip(
                    truth.freq_hz,
                    predicted.freq_hz,
                )
            ):
                raise ValueError(
                    "the saved prediction frequency grid does not match source "
                    f"verification block {block_index}"
                )
            mismatched_parameters = [
                name
                for name in parameter_names
                if str(truth.params.get(name, ""))
                != str(predicted.params.get(name, ""))
            ]
            if mismatched_parameters:
                raise ValueError(
                    "the saved prediction geometry does not match source "
                    f"verification block {block_index} for parameter(s): "
                    + ", ".join(mismatched_parameters)
                )
        raw_sparam_weights = metadata.get("sparam_weights")
        if isinstance(raw_sparam_weights, dict):
            sparam_weights = {
                str(name): float(value)
                for name, value in raw_sparam_weights.items()
            }
        else:
            sparam_weights = parse_sparam_weights(
                labels,
                command_options.get("--sparam-weights"),
            )
        raw_frequency_weights = metadata.get("frequency_weights")
        frequency_weights = (
            str(raw_frequency_weights)
            if raw_frequency_weights is not None and raw_frequency_weights != ""
            else command_options.get("--frequency-weights")
        )
        metric_rows, _summary = verification_metrics(
            truth_rf,
            predicted_rf,
            labels,
            parameter_names,
            sparam_weights=sparam_weights,
            frequency_weights=frequency_weights,
        )
        if not metric_rows:
            raise ValueError("the recovered verification comparison produced no rows")
        recovered_path = target_model_dir / "verification_metrics.csv"
        write_csv(recovered_path, metric_rows)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, str(exc)

    print(
        "recovered missing promoted verification metrics from the saved "
        f"verification prediction and source data: {recovered_path}",
        file=sys.stderr,
    )
    return recovered_path, None


def verification_metrics_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    fallback_manifest: Path | None = None
    recovery_error: str | None = None
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
            sweep_dir = (
                fit_dir.parent if fit_dir.name.startswith("best_model") else fit_dir
            )
            target_model_dir = (
                fit_dir
                if fit_dir.name.startswith("best_model")
                else fit_dir / "best_model"
            )
            selected_trial_path = _selected_trial_metrics_path(
                sweep_dir,
                target_model_dir,
            )
            if selected_trial_path is not None:
                path = selected_trial_path
            else:
                recovered_path, recovery_error = _recover_promoted_verification_metrics(
                    sweep_dir,
                    target_model_dir,
                )
                path = recovered_path or direct_path
    else:
        parser.error("Either --fit-dir or --verification-metrics is required")
    if not path.exists():
        recovery_detail = (
            f" Automatic recovery was attempted but failed: {recovery_error}."
            if recovery_error
            else ""
        )
        parser.error(
            f"Verification metrics file does not exist: {path}. For an optimize "
            "run, pass the sweep directory or its best_model directory."
            f"{recovery_detail}"
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
    try:
        canonical_dataset_label(args.target_dataset)
    except ValueError as exc:
        parser.error(f"--target-dataset: {exc}")
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
    if args.exploration_weight != "auto" and args.exploration_weight < 0.0:
        parser.error("--exploration-weight must be non-negative")
    if args.target_error is not None and (
        not math.isfinite(args.target_error) or args.target_error <= 0.0
    ):
        parser.error("--target-error must be positive and finite")
    if args.gp_noise_variance < 0.0:
        parser.error("--gp-noise-variance must be non-negative")
    if args.gp_error_floor <= 0.0:
        parser.error("--gp-error-floor must be positive")
    for option_name in (
        "verification_interval",
        "verification_batch",
        "verification_max_add",
    ):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be positive")
    validate_shared_sampling_args(parser, args)
    parameters = resolve_suggest_parameters(parser, args)
    validate_parameter_decimal_places(parser, parameters, args.decimal_places)
    if isinstance(args.gp_length_scale, list) and len(args.gp_length_scale) not in {
        1,
        len(parameters),
    }:
        parser.error(
            "--gp-length-scale must contain one value or exactly one value per "
            f"parameter ({len(parameters)})"
        )

    metrics_path = verification_metrics_path(args, parser)
    try:
        regions, metric_name, effective_bare_values = load_error_regions(
            metrics_path,
            parameters,
            metric_name=args.metric,
            bare_values=args.bare_values,
        )
    except ValueError as exc:
        parser.error(str(exc))

    csv_existing_points = load_existing_points(
        args.existing_points,
        [],
        parameters,
        bare_values=args.bare_values,
    )
    mdif_observed_points: list[tuple[list[float], str, str, object]] = []
    for raw_path in args.existing_mdif:
        mdif_path = Path(raw_path)
        mdif_rows = read_mdif_parameter_rows(mdif_path)
        source_mode = resolve_bare_values_for_rows(
            mdif_rows,
            parameters,
            args.bare_values,
        )
        try:
            filename_role = geometry_file_split_group(mdif_path)
        except ValueError:
            # MDIF names such as train_verify.mdif commonly describe a
            # combined file; row-level split variables remain authoritative.
            filename_role = None
        default_dataset = (
            "train"
            if filename_role != "verification"
            else "verification"
        )
        for source_index, row in enumerate(mdif_rows, start=1):
            point = row_unit_point(row, parameters, bare_values=source_mode)
            if point is None or not in_unit_cube(point):
                continue
            try:
                dataset = canonical_dataset_label(
                    lookup_row_value(row, args.split_var),
                    default=default_dataset,
                )
            except ValueError as exc:
                parser.error(f"{mdif_path} block {source_index}: {exc}")
            mdif_observed_points.append(
                (clamp_unit_point(point), dataset, str(raw_path), source_index)
            )
    existing_points = [region.unit_point for region in regions]
    existing_points.extend(csv_existing_points)
    existing_points.extend(point for point, _, _, _ in mdif_observed_points)
    existing_points = dedupe_points(existing_points)

    try:
        existing_dataset_assignments = existing_csv_dataset_assignments(
            args.existing_points,
            parameters,
            args.split_var,
            args.bare_values,
            args.decimal_places,
        )
    except ValueError as exc:
        parser.error(str(exc))
    for point, dataset, _source, _source_index in mdif_observed_points:
        key = geometry_output_key_from_unit_point(
            point,
            parameters,
            args.decimal_places,
        )
        group = coverage_split_group(dataset)
        if existing_dataset_assignments.get(key) != "training" or group == "training":
            existing_dataset_assignments[key] = group
    training_geometry_keys = {
        key
        for key, group in existing_dataset_assignments.items()
        if group == "training"
    }
    existing_verification_geometry_keys = {
        key
        for key, group in existing_dataset_assignments.items()
        if group == "verification"
    }
    metrics_geometry_keys = {
        geometry_output_key_from_unit_point(
            region.unit_point,
            parameters,
            args.decimal_places,
        )
        for region in regions
    }
    leaked_metrics_geometry_keys = metrics_geometry_keys & training_geometry_keys
    verification_inventory_keys = (
        existing_verification_geometry_keys | metrics_geometry_keys
    ) - training_geometry_keys
    existing_dataset_counts = {
        "training": len(training_geometry_keys),
        "verification": len(existing_verification_geometry_keys),
    }
    verification_metrics_geometry_count = len(
        metrics_geometry_keys - training_geometry_keys
    )
    existing_verification_geometry_count = len(existing_verification_geometry_keys)
    effective_verification_count = len(verification_inventory_keys)
    if leaked_metrics_geometry_keys:
        print(
            "warning: "
            f"{len(leaked_metrics_geometry_keys)} verification-metrics "
            "geometry/ies also occur in training; they remain useful GP error "
            "observations but are excluded from the independent verification "
            "inventory",
            file=sys.stderr,
        )
    primary_is_training = (
        coverage_split_group(args.target_dataset) == "training"
    )

    previous_error_rms: list[float] = []
    for raw_history_path in args.previous_verification_metrics:
        history_path = Path(raw_history_path)
        try:
            history_regions, _, _ = load_error_regions(
                history_path,
                parameters,
                metric_name=metric_name,
                bare_values=args.bare_values,
            )
        except ValueError as exc:
            parser.error(f"--previous-verification-metrics {history_path}: {exc}")
        previous_error_rms.append(error_region_summary(history_regions)["rms"])

    recommendation = recommend_additional_point_count(
        dimensions=len(parameters),
        regions=regions,
        existing_training_count=existing_dataset_counts["training"],
        target_error=args.target_error,
        previous_error_rms=previous_error_rms,
    )
    count_is_automatic = args.count == "auto"
    primary_count = (
        recommendation.recommended_count
        if count_is_automatic
        else int(args.count)
    )
    exploration_is_automatic = args.exploration_weight == "auto"
    effective_exploration_weight = (
        adaptive_exploration_weight(recommendation)
        if exploration_is_automatic
        else float(args.exploration_weight)
    )
    print(
        "point recommendation: "
        f"{recommendation.recommended_count} primary "
        f"{canonical_dataset_label(args.target_dataset)} point(s) for "
        f"{len(parameters)}D; current {metric_name} RMS "
        f"{recommendation.current_error_rms:.6g}, p90 "
        f"{recommendation.current_error_p90:.6g}, max "
        f"{recommendation.current_error_max:.6g}; stage "
        f"{recommendation.stage}"
    )
    if args.target_error is not None:
        print(
            f"accuracy target: {args.target_error:.6g}; current/target "
            f"{recommendation.target_ratio:.6g}x"
        )
    if recommendation.latest_improvement_fraction is not None:
        print(
            "latest recorded RMS improvement: "
            f"{100.0 * recommendation.latest_improvement_fraction:.2f}%"
        )
    for reason in recommendation.rationale:
        print(f"  recommendation basis: {reason}")
    print(
        "exploration weight: "
        f"{effective_exploration_weight:.6g}"
        + (" (auto)" if exploration_is_automatic else " (explicit)")
    )
    if not count_is_automatic and primary_count != recommendation.recommended_count:
        print(
            f"explicit --count {primary_count} overrides the recommended "
            f"{recommendation.recommended_count}"
        )
    if count_is_automatic and primary_count == 0:
        print(
            "no additional points were generated because the requested accuracy "
            "target is met"
        )
        return 0

    automatic_verification_enabled = (
        args.acquisition in {"gp-ucb", "hybrid"}
        and args.verification_policy == "auto"
        and primary_is_training
    )
    preliminary_verification_plan = automatic_verification_plan(
        dimensions=len(parameters),
        existing_training_count=existing_dataset_counts["training"],
        verification_observation_count=effective_verification_count,
        requested_training_count=primary_count if primary_is_training else 0,
        enabled=automatic_verification_enabled,
        interval=args.verification_interval,
        batch=args.verification_batch,
        max_add=args.verification_max_add,
    )
    preliminary_verification_plan["policy"] = args.verification_policy
    preliminary_verification_plan["existing_verification_geometry_count"] = (
        existing_verification_geometry_count
    )
    preliminary_verification_plan["verification_metrics_geometry_count"] = (
        verification_metrics_geometry_count
    )
    preliminary_verification_plan["verification_count_basis"] = (
        "deduplicated union of existing verification geometry inventory and "
        "verification-metrics geometries, excluding every training geometry"
    )
    preliminary_verification_plan["training_verification_overlap_count"] = len(
        leaked_metrics_geometry_keys
    )
    planned_total_count = primary_count + int(
        preliminary_verification_plan["additional_verification_count"]
    )
    candidate_count = args.candidate_count or max(
        1000,
        planned_total_count * args.candidate_factor,
    )
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

    occupied_output_keys = {
        geometry_output_key_from_unit_point(
            point,
            parameters,
            args.decimal_places,
        )
        for point in existing_points
    }
    raw_candidate_count = len(candidates)
    candidates = filter_unique_output_candidates(
        candidates,
        parameters,
        args.decimal_places,
        excluded_keys=occupied_output_keys,
    )
    if not candidates:
        parser.error(
            "No unoccupied candidate remains at the requested output digits. "
            "Increase --decimal-places, increase --candidate-count, or expand "
            "the parameter range."
        )

    acquisition_metadata: dict[str, object] = {}
    acquisition_metadata["candidate_output_grid"] = {
        "requested_candidate_count": raw_candidate_count,
        "unique_unoccupied_candidate_count": len(candidates),
        "decimal_places": args.decimal_places,
    }
    acquisition_metadata["verification_metrics_source"] = str(metrics_path)
    acquisition_metadata["bare_values_mode"] = args.bare_values
    acquisition_metadata["bare_values_interpretation"] = effective_bare_values
    acquisition_metadata["verification_metrics_bare_values_interpretation"] = (
        effective_bare_values
    )
    acquisition_metadata["point_count_recommendation"] = {
        "requested_count": args.count,
        "count_mode": "auto" if count_is_automatic else "explicit",
        "recommended_primary_count": recommendation.recommended_count,
        "resolved_primary_count": primary_count,
        "dimensions": recommendation.dimensions,
        "current_error_rms": recommendation.current_error_rms,
        "current_error_median": recommendation.current_error_median,
        "current_error_p90": recommendation.current_error_p90,
        "current_error_max": recommendation.current_error_max,
        "target_error": recommendation.target_error,
        "target_ratio": recommendation.target_ratio,
        "previous_error_rms": recommendation.previous_error_rms,
        "previous_verification_metrics": list(
            args.previous_verification_metrics
        ),
        "latest_improvement_fraction": (
            recommendation.latest_improvement_fraction
        ),
        "existing_training_count": recommendation.existing_training_count,
        "verification_observation_count": (
            recommendation.verification_observation_count
        ),
        "stage": recommendation.stage,
        "rationale": recommendation.rationale,
    }
    if getattr(args, "nonpassive_source", None):
        acquisition_metadata["nonpassive_point_generation_source"] = (
            args.nonpassive_source
        )
    if getattr(args, "parameter_metadata_source", None):
        acquisition_metadata["parameter_metadata_source"] = (
            args.parameter_metadata_source
        )
    target_datasets: list[str]
    automatic_verification_suggestions: list[SuggestedPoint] = []
    hybrid_diagnostics: dict[str, object] = {}
    if args.acquisition in {"gp-ucb", "hybrid"}:
        try:
            if args.acquisition == "hybrid":
                hybrid_allocation = hybrid_component_allocation(
                    primary_count,
                    dimensions=len(parameters),
                    observation_count=len(regions),
                    target_ratio=recommendation.target_ratio,
                    latest_improvement_fraction=(
                        recommendation.latest_improvement_fraction
                    ),
                )
                suggestions, gp_model, hybrid_diagnostics = select_hybrid_points(
                    candidates,
                    regions,
                    existing_points,
                    count=primary_count,
                    allocation=hybrid_allocation,
                    exploration_weight=effective_exploration_weight,
                    novelty_power=args.novelty_power,
                    min_distance=args.min_distance,
                    length_scale=args.gp_length_scale,
                    noise_variance=args.gp_noise_variance,
                    error_floor=args.gp_error_floor,
                    ard_mode=args.gp_ard,
                )
            else:
                suggestions, gp_model = select_gp_ucb_points(
                    candidates,
                    regions,
                    existing_points,
                    count=primary_count,
                    exploration_weight=effective_exploration_weight,
                    novelty_power=args.novelty_power,
                    min_distance=args.min_distance,
                    length_scale=args.gp_length_scale,
                    noise_variance=args.gp_noise_variance,
                    error_floor=args.gp_error_floor,
                    ard_mode=args.gp_ard,
                )
        except ValueError as exc:
            parser.error(str(exc))
        verification_plan = automatic_verification_plan(
            dimensions=len(parameters),
            existing_training_count=existing_dataset_counts["training"],
            verification_observation_count=effective_verification_count,
            requested_training_count=(
                len(suggestions) if primary_is_training else 0
            ),
            enabled=automatic_verification_enabled,
            interval=args.verification_interval,
            batch=args.verification_batch,
            max_add=args.verification_max_add,
        )
        verification_plan["policy"] = args.verification_policy
        verification_plan["existing_verification_geometry_count"] = (
            existing_verification_geometry_count
        )
        verification_plan["verification_metrics_geometry_count"] = (
            verification_metrics_geometry_count
        )
        verification_plan["verification_count_basis"] = (
            "deduplicated union of existing verification geometry inventory and "
            "verification-metrics geometries, excluding every training geometry"
        )
        verification_plan["training_verification_overlap_count"] = len(
            leaked_metrics_geometry_keys
        )
        if not automatic_verification_enabled:
            verification_plan["reason"] = (
                "policy set to off"
                if args.verification_policy == "off"
                else "the requested primary batch is verification rather than training"
            )
        automatic_verification_count = int(
            verification_plan["additional_verification_count"]
        )
        if automatic_verification_count:
            verification_existing_points = [
                *existing_points,
                *(suggestion.unit_point for suggestion in suggestions),
            ]
            try:
                automatic_verification_suggestions, _verification_gp_model = (
                    select_gp_ucb_points(
                        candidates,
                        regions,
                        verification_existing_points,
                        count=automatic_verification_count,
                        exploration_weight=max(effective_exploration_weight, 3.0),
                        novelty_power=max(args.novelty_power, 2.0),
                        min_distance=max(args.min_distance, 1.0e-9),
                        length_scale=args.gp_length_scale,
                        noise_variance=args.gp_noise_variance,
                        error_floor=args.gp_error_floor,
                        ard_mode=args.gp_ard,
                    )
                )
                for suggestion in automatic_verification_suggestions:
                    suggestion.selection_component = "verification-uncertainty"
            except ValueError as exc:
                parser.error(str(exc))
            if len(automatic_verification_suggestions) < automatic_verification_count:
                print(
                    "warning: selected "
                    f"{len(automatic_verification_suggestions)} of "
                    f"{automatic_verification_count} automatic verification "
                    "points; increase --candidate-count or lower --min-distance",
                    file=sys.stderr,
                )
        verification_plan["selected_additional_verification_count"] = len(
            automatic_verification_suggestions
        )
        verification_plan["selection_exploration_weight"] = max(
            effective_exploration_weight,
            3.0,
        )
        verification_plan["selection_novelty_power"] = max(
            args.novelty_power,
            2.0,
        )
        acquisition_metadata["automatic_verification"] = verification_plan
        acquisition_metadata["gp"] = {
            "kernel": (
                "matern52_ard"
                if len(set(round(value, 12) for value in gp_model.length_scales)) > 1
                else "matern52_isotropic"
            ),
            "target_transform": "natural_log_error",
            "observation_count": len(gp_model.observation_points),
            "length_scale": gp_model.length_scale,
            "length_scales": gp_model.length_scales,
            "length_scale_by_parameter": {
                parameter.name: scale
                for parameter, scale in zip(parameters, gp_model.length_scales)
            },
            "length_scale_selection": gp_model.length_scale_selection,
            "ard_mode": args.gp_ard,
            "noise_variance": gp_model.noise_variance,
            "exploration_weight": effective_exploration_weight,
            "exploration_weight_requested": args.exploration_weight,
            "exploration_weight_selection": (
                "auto" if exploration_is_automatic else "user"
            ),
            "log_marginal_likelihood": gp_model.log_marginal_likelihood,
            "batch_posterior_update": "kriging-believer-posterior-mean",
        }
        if hybrid_diagnostics:
            acquisition_metadata["hybrid"] = hybrid_diagnostics
        if len(gp_model.observation_points) < max(3 * len(parameters), 12):
            print(
                "warning: the adaptive GP has fewer distinct error observations "
                "than the preferred max(3*d, 12); the hybrid allocation reserves "
                "extra uncertainty and coverage points until observations grow",
                file=sys.stderr,
            )
        target_datasets = [args.target_dataset] * len(suggestions)
        suggestions = [*suggestions, *automatic_verification_suggestions]
        target_datasets.extend(
            ["verification"] * len(automatic_verification_suggestions)
        )
    else:
        suggestions = select_targeted_points(
            candidates,
            regions,
            existing_points,
            count=primary_count,
            focus_radius=args.focus_radius,
            focus_power=args.focus_power,
            novelty_power=args.novelty_power,
            min_distance=args.min_distance,
        )
        target_datasets = [args.target_dataset] * len(suggestions)
        acquisition_metadata["automatic_verification"] = {
            **preliminary_verification_plan,
            "enabled": False,
            "reason": "automatic verification applies only to hybrid or GP-UCB training batches",
            "selected_additional_verification_count": 0,
        }
    primary_suggestion_count = len(suggestions) - len(
        automatic_verification_suggestions
    )
    if primary_suggestion_count < primary_count:
        print(
            f"warning: selected {primary_suggestion_count} of {primary_count} requested points; "
            "increase --candidate-count or lower --min-distance",
            file=sys.stderr,
        )

    selected_output_keys: dict[tuple[str, ...], tuple[int, str]] = {}
    for suggestion_index, (suggestion, raw_dataset) in enumerate(
        zip(suggestions, target_datasets),
        start=1,
    ):
        key = geometry_output_key_from_unit_point(
            suggestion.unit_point,
            parameters,
            args.decimal_places,
        )
        dataset = canonical_dataset_label(raw_dataset)
        if key in occupied_output_keys:
            parser.error(
                f"Suggested point {suggestion_index} becomes identical to an "
                "existing geometry after output rounding. Increase "
                "--decimal-places or --min-distance."
            )
        prior = selected_output_keys.get(key)
        if prior is not None:
            prior_index, prior_dataset = prior
            relation = (
                "across training and verification"
                if prior_dataset != dataset
                else f"within {dataset}"
            )
            parser.error(
                f"Suggested points {prior_index} and {suggestion_index} become "
                f"duplicates {relation} after output rounding. Increase "
                "--decimal-places or --min-distance."
            )
        selected_output_keys[key] = (suggestion_index, dataset)

    out_path = Path(args.out)
    combined_path = (
        Path(args.combined_out)
        if args.combined_out
        else accumulated_geometry_path(out_path)
    )
    analysis_path = (
        Path(args.analysis_out)
        if args.analysis_out
        else analysis_output_path(out_path)
    )
    if combined_path.resolve() == out_path.resolve():
        parser.error("--combined-out must be different from --out")
    if combined_path.resolve() == analysis_path.resolve():
        parser.error("--combined-out must be different from --analysis-out")
    if geometry_metadata_path(combined_path).resolve() == geometry_metadata_path(
        out_path
    ).resolve():
        parser.error(
            "--combined-out must produce a different companion JSON from --out"
        )
    try:
        require_combined_geometry_path(out_path, "--out")
        require_combined_geometry_path(combined_path, "--combined-out")
        if geometry_file_split_group(analysis_path) != "verification":
            raise ValueError(
                f"--analysis-out contains verification information only, so "
                f"{analysis_path.name!r} must include the word verification"
            )
    except ValueError as exc:
        parser.error(str(exc))
    normalized_target_dataset = canonical_dataset_label(args.target_dataset)
    if normalized_target_dataset != normalize_key(args.target_dataset):
        print(
            f"warning: normalized legacy --target-dataset {args.target_dataset!r} "
            f"to {normalized_target_dataset!r}; use point_origin=additional to "
            "identify the new batch",
            file=sys.stderr,
        )
    batch_dataset_labels = {
        canonical_dataset_label(value) for value in target_datasets
    }
    acquisition_metadata["split_files"] = {
        **(
            {"training": str(split_output_path(out_path, "train"))}
            if "train" in batch_dataset_labels
            else {}
        ),
        **(
            {"verification": str(split_output_path(out_path, "verification"))}
            if "verification" in batch_dataset_labels
            else {}
        ),
        "companion_json": str(geometry_metadata_path(out_path)),
        "separate_split_json_files": False,
    }
    cumulative_dataset_labels = set(batch_dataset_labels)
    if training_geometry_keys:
        cumulative_dataset_labels.add("train")
    if verification_inventory_keys:
        cumulative_dataset_labels.add("verification")
    acquisition_metadata["cumulative_split_files"] = {
        **(
            {"training": str(split_output_path(combined_path, "train"))}
            if "train" in cumulative_dataset_labels
            else {}
        ),
        **(
            {
                "verification": str(
                    split_output_path(combined_path, "verification")
                )
            }
            if "verification" in cumulative_dataset_labels
            else {}
        ),
        "combined": str(combined_path),
        "companion_json": str(geometry_metadata_path(combined_path)),
        "separate_split_json_files": False,
    }
    acquisition_metadata["parameter_coverage_context"] = {
        "source": str(combined_path),
        "includes_existing_points": True,
        "current_batch_origin": "additional",
        "existing_point_origin": "existing",
    }
    write_error_regions_csv(analysis_path, regions, parameters)
    metadata_path = write_suggested_points_csv(
        out_path,
        suggestions,
        parameters,
        split_var=args.split_var,
        target_dataset=args.target_dataset,
        target_datasets=target_datasets,
        candidate_method=candidate_method,
        acquisition_method=args.acquisition,
        metric_name=metric_name,
        include_normalized=args.include_normalized,
        decimal_places=args.decimal_places,
        acquisition_metadata=acquisition_metadata,
    )
    method_name = (
        f"targeted-{candidate_method}"
        if args.acquisition == "error-distance"
        else f"{args.acquisition}-{candidate_method}"
    )
    observed_points: list[tuple[Sequence[float], str, str, object]] = [
        (
            region.unit_point,
            "verification",
            str(metrics_path),
            region.source_index,
        )
        for region in regions
    ]
    observed_points.extend(mdif_observed_points)
    try:
        combined_metadata_path = write_accumulated_geometries(
            combined_path,
            parameters,
            args.split_var,
            args.existing_points,
            observed_points,
            out_path,
            include_normalized=args.include_normalized,
            decimal_places=args.decimal_places,
            bare_values=args.bare_values,
            method=method_name,
            metadata_extra={
                "analysis_metric": metric_name,
                "acquisition_method": args.acquisition,
                "candidate_method": candidate_method,
                "verification_metrics_source": str(metrics_path),
                "existing_mdif_files": list(args.existing_mdif),
                "acquisition_occupied_point_count": len(existing_points),
                "new_point_count": len(suggestions),
                **acquisition_metadata,
            },
        )
    except ValueError as exc:
        parser.error(str(exc))
    _, coverage_context_rows = read_csv_table(combined_path)
    cumulative_training_count = sum(
        canonical_dataset_label(lookup_row_value(row, args.split_var)) == "train"
        for row in coverage_context_rows
    )
    cumulative_verification_count = (
        len(coverage_context_rows) - cumulative_training_count
    )
    write_parameter_coverage_png(
        out_path,
        parameters,
        coverage_context_rows,
        args.split_var,
        bare_values=args.bare_values,
    )
    split_view_paths = write_dataset_split_geometry_views(
        out_path,
        parameters,
        args.split_var,
        bare_values=args.bare_values,
        decimal_places=args.decimal_places,
        coverage_rows=coverage_context_rows,
    )
    cumulative_split_view_paths = write_dataset_split_geometry_views(
        combined_path,
        parameters,
        args.split_var,
        bare_values=args.bare_values,
        decimal_places=args.decimal_places,
        coverage_rows=coverage_context_rows,
    )
    print(f"analyzed {len(regions)} verification error region(s) from {metrics_path}")
    print(
        f"considered {len(existing_points)} existing point(s) and "
        f"{candidate_count} {candidate_method} candidate point(s)"
    )
    print(f"acquisition method: {args.acquisition}")
    print(
        "verification-metrics unitless input interpretation: "
        f"{effective_bare_values}"
        + (" (auto-detected)" if args.bare_values == "auto" else "")
    )
    if args.bare_values == "auto":
        print(
            "other geometry/MDIF inputs: unitless values are detected "
            "independently for each source"
        )
    if getattr(args, "parameter_metadata_source", None):
        print(f"parameter domain: {args.parameter_metadata_source}")
    if args.acquisition in {"gp-ucb", "hybrid"}:
        scale_text = ", ".join(
            f"{parameter.name}={scale:.4g}"
            for parameter, scale in zip(parameters, gp_model.length_scales)
        )
        print(
            "GP observations: "
            f"{len(gp_model.observation_points)}, "
            f"Matérn-5/2 length scales: {scale_text}, "
            f"exploration weight: {effective_exploration_weight:.6g}"
        )
        if args.acquisition == "hybrid":
            allocation = hybrid_diagnostics.get("allocation", {})
            print(
                "hybrid batch: "
                f"{allocation.get('exploitation', 0)} exploitation + "
                f"{allocation.get('uncertainty', 0)} uncertainty + "
                f"{allocation.get('coverage', 0)} coverage "
                f"({allocation.get('regime', 'unknown')})"
            )
        verification_plan = acquisition_metadata["automatic_verification"]
        assert isinstance(verification_plan, dict)
        added_verification = int(
            verification_plan.get("selected_additional_verification_count", 0)
        )
        if verification_plan.get("enabled"):
            print(
                "automatic verification: "
                f"{added_verification} point(s) added; projected training "
                f"count {verification_plan['projected_training_count']}, "
                "current acquisition-verification inventory "
                f"{verification_plan['verification_observation_count']}, "
                f"target {verification_plan['target_verification_count']}, "
                f"interval {verification_plan['training_interval']} training "
                "point(s), next scheduled trigger at "
                f"{verification_plan['next_training_trigger']} training point(s)"
            )
        else:
            reason = verification_plan.get("reason")
            print(
                "automatic verification: disabled"
                + (f" ({reason})" if reason else "")
            )
    print(f"wrote {out_path}")
    print(f"wrote {metadata_path}")
    print(f"wrote {geometry_coverage_plot_path(out_path)}")
    for split_view_path in split_view_paths:
        print(f"wrote {split_view_path}")
    print(f"wrote {analysis_path}")
    print(f"wrote {combined_path}")
    print(f"wrote {combined_metadata_path}")
    print(f"wrote {geometry_coverage_plot_path(combined_path)}")
    for split_view_path in cumulative_split_view_paths:
        print(f"wrote {split_view_path}")
    print(
        "geometry integrity: "
        f"{len(coverage_context_rows)} unique combined point(s) = "
        f"{cumulative_training_count} training + "
        f"{cumulative_verification_count} verification; "
        "training/verification overlap: 0"
    )
    print(
        "next GP round: --existing-points "
        f"{shlex.quote(str(combined_path))}"
    )
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
