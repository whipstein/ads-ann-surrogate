#!/usr/bin/env python3
"""Generate geometry/process sample points for surrogate-model extraction.

The default method is a maximin Latin hypercube because finite EM sample sets
usually benefit from stratification plus good point separation. Sobol is also
available when SciPy is installed.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


def parse_value_token(token: str) -> tuple[float, str]:
    text = token.strip()
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
    return value * UNIT_SCALES[unit], unit


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


def format_value(value: float, unit: str) -> str:
    scale = UNIT_SCALES.get(unit, 1.0)
    return f"{value / scale:.12g}{unit}"


def map_unit_point(value: float, spec: ParameterSpec) -> float:
    if spec.scale == "log":
        log_lo = math.log(spec.lower)
        log_hi = math.log(spec.upper)
        return math.exp(log_lo + value * (log_hi - log_lo))
    return spec.lower + value * (spec.upper - spec.lower)


def latin_hypercube_points(count: int, dimensions: int, rng: random.Random) -> list[list[float]]:
    points = [[0.0 for _ in range(dimensions)] for _ in range(count)]
    for dim in range(dimensions):
        values = [(idx + rng.random()) / count for idx in range(count)]
        rng.shuffle(values)
        for row, value in enumerate(values):
            points[row][dim] = value
    return points


def min_pairwise_distance(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return float("inf")
    best = float("inf")
    for idx, lhs in enumerate(points[:-1]):
        for rhs in points[idx + 1 :]:
            distance2 = sum((float(a) - float(b)) ** 2 for a, b in zip(lhs, rhs))
            if distance2 < best:
                best = distance2
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
        score = min_pairwise_distance(trial)
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


def write_points_csv(
    path: Path,
    method: str,
    unit_points: Sequence[Sequence[float]],
    parameters: Sequence[ParameterSpec],
    verification_count: int,
    split_var: str,
    include_normalized: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["point_index", split_var, "method"]
    if include_normalized:
        fields.extend(f"u_{parameter.name}" for parameter in parameters)
    fields.extend(parameter.name for parameter in parameters)

    train_count = len(unit_points) - verification_count
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, point in enumerate(unit_points, start=1):
            row: dict[str, object] = {
                "point_index": idx,
                split_var: "train" if idx <= train_count else "verification",
                "method": method,
            }
            if include_normalized:
                for parameter, unit_value in zip(parameters, point):
                    row[f"u_{parameter.name}"] = f"{unit_value:.16g}"
            for parameter, unit_value in zip(parameters, point):
                row[parameter.name] = format_value(map_unit_point(unit_value, parameter), parameter.unit)
            writer.writerow(row)


def parse_methods(raw_methods: Sequence[str]) -> list[str]:
    methods: list[str] = []
    for raw in raw_methods:
        for part in raw.split(","):
            method = part.strip().lower()
            if method:
                methods.append(method)
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate geometry/process sample points for ADS surrogate extraction.",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        required=True,
        metavar="NAME=LOW:HIGH[:linear|log]",
        help="Repeat once per geometry/process variable, e.g. W=0.40mm:0.80mm or R=1:100:log.",
    )
    parser.add_argument("--count", type=int, required=True, help="Total number of points to write.")
    parser.add_argument(
        "--verification-count",
        type=int,
        default=0,
        help="Number of tail points labeled verification. Default: 0.",
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
    parser.add_argument("--out", default="generated_points.csv", help="Output CSV path. Use {method} for multiple methods.")
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
        help="CSV column used to label train vs verification points. Default: dataset.",
    )
    parser.add_argument(
        "--include-normalized",
        action="store_true",
        help="Include u_<name> columns with the underlying [0, 1] coordinates.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.verification_count < 0 or args.verification_count >= args.count:
        parser.error("--verification-count must be non-negative and smaller than --count")
    if args.lhs_candidates <= 0:
        parser.error("--lhs-candidates must be positive")
    if args.skip < 0:
        parser.error("--skip must be non-negative")

    try:
        parameters = [parse_parameter_spec(raw) for raw in args.parameter]
    except ValueError as exc:
        parser.error(str(exc))

    methods = parse_methods(args.method) or ["maximin-lhs"]
    valid_methods = {"maximin-lhs", "latin-hypercube", "sobol", "halton"}
    unknown = [method for method in methods if method not in valid_methods]
    if unknown:
        parser.error("Unknown method(s): " + ", ".join(unknown))

    multiple_methods = len(methods) > 1
    for offset, method in enumerate(methods):
        try:
            unit_points = generate_unit_points(
                method,
                count=args.count,
                dimensions=len(parameters),
                seed=args.seed + offset,
                lhs_candidates=args.lhs_candidates,
                scramble=not args.no_scramble,
                skip=args.skip,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        out_path = output_path_for_method(Path(args.out), method, multiple_methods)
        write_points_csv(
            out_path,
            method,
            unit_points,
            parameters,
            verification_count=args.verification_count,
            split_var=args.split_var,
            include_normalized=args.include_normalized,
        )
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
