#!/usr/bin/env python3
"""Knowledge-based neural-network trainer for parameterized S-parameter MDIF.

The supported KBNN forms are:

    plain        : NN(geometry, frequency) -> fine S
    residual    : coarse S + NN(geometry, frequency[, coarse S]) -> fine S
    prior-input : NN(geometry, frequency, coarse S) -> fine S

The residual mode is the classic difference-method KBNN: a cheap coarse model
captures most of the physics, and the neural network only learns the correction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import shutil
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

RC2_ROOT = Path(__file__).resolve().parents[1]
if str(RC2_ROOT) not in sys.path:
    sys.path.insert(0, str(RC2_ROOT))

from common.surrogate_common import (  # noqa: E402
    DB_MAG_FLOOR,
    EPS,
    MDIFBlock,
    MLP,
    Standardizer,
    ads_ann_activation_enum,
    ads_ann_optimizer_enum,
    ads_ann_output_format_enum,
    ads_ann_training_type_enum,
    build_ads_export_blocks,
    cleanup_trial_dir,
    common_sparameter_labels,
    configure_parallel_numeric_threads,
    copy_trial_model,
    csv_number,
    frequency_feature_columns,
    infer_parameter_names,
    load_sweep_rows,
    infer_uniform_hidden_layout,
    make_training_progress_callback,
    metadata_csv,
    metadata_hidden_layers,
    model_settings_title,
    mse,
    normalize_name,
    normalize_sparam_weights,
    output_weights_from_sparam_weights,
    parse_csv_set,
    parse_number,
    parse_parameter_scale_spec,
    parse_sparam_weights,
    progress_interval_from_args,
    plot_sweep_diagnostics,
    plot_worst_case_fits,
    plot_worst_case_y_fits,
    read_mdif,
    read_model_metadata,
    rerank_sweep_rows,
    run_sweep_command,
    sparam_sort_key,
    sparam_weight_mean,
    sparameter_real_imag_columns,
    split_blocks,
    sweep_arg_values,
    sweep_trial_seed,
    trial_plot_paths,
    verification_metrics,
    write_training_verification_artifacts,
    write_ads_ann_package,
    write_ads_export_package,
    write_csv,
    write_history,
    write_mdif,
    write_sweep_markdown,
    write_training_markdown,
    write_veriloga_package,
)


VERSION = "0.2.0-rc2"
KBNN_SWEEP_RESULT_COLUMNS = ["mode", "include_coarse_input", "hidden_layers", "activation", "learning_rate"]
_MDIF_BLOCK_CACHE: dict[tuple[str, int, int], list[MDIFBlock]] = {}
_BLOCK_PARAMETER_CACHE: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
_ALIGN_COARSE_CACHE: dict[tuple[object, ...], list[MDIFBlock]] = {}
_FEATURE_CACHE: dict[tuple[object, ...], np.ndarray] = {}
_SAMPLE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}
_FINE_TARGET_CACHE: dict[tuple[object, ...], np.ndarray] = {}


def mdif_cache_key(path_text: str) -> tuple[str, int, int]:
    path = Path(path_text).expanduser().resolve()
    stat = path.stat()
    return str(path), int(stat.st_mtime_ns), int(stat.st_size)


def read_mdif_cached(path_text: str) -> list[MDIFBlock]:
    key = mdif_cache_key(path_text)
    cached = _MDIF_BLOCK_CACHE.get(key)
    if cached is None:
        cached = read_mdif(Path(key[0]))
        _MDIF_BLOCK_CACHE[key] = cached
    return cached


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    aliases = {
        "difference": "residual",
        "diff": "residual",
        "pki": "prior-input",
        "prior": "prior-input",
        "priorinput": "prior-input",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"plain", "residual", "prior-input"}:
        raise ValueError(f"Unsupported KBNN mode {mode!r}")
    return normalized


def parse_bool_option(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


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


def parse_hidden_layers(text: str) -> list[int]:
    layers = [int(part) for part in text.split(",") if part.strip()]
    if any(layer <= 0 for layer in layers):
        raise ValueError("Hidden layers must be positive integers")
    return layers


def parse_hidden_layer_options(text: str) -> list[str]:
    options = [part.strip() for part in text.split(";") if part.strip()]
    if not options:
        raise ValueError("Expected at least one hidden-layer option")
    for option in options:
        parse_hidden_layers(option)
    return options


def split_fine_blocks(args: argparse.Namespace) -> tuple[list[MDIFBlock], list[MDIFBlock], list[MDIFBlock]]:
    fine_blocks = read_mdif_cached(args.mdif)
    if args.verification_mdif:
        train_blocks = fine_blocks
        verify_blocks = read_mdif_cached(args.verification_mdif)
        return train_blocks, verify_blocks, train_blocks + verify_blocks
    split_data = split_blocks(
        fine_blocks,
        split_var=args.split_var,
        train_values=parse_csv_set(args.train_values),
        verify_values=parse_csv_set(args.verify_values),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    return split_data.train, split_data.verify, split_data.all_blocks


def split_coarse_blocks(
    args: argparse.Namespace,
    train_fine: Sequence[MDIFBlock],
    verify_fine: Sequence[MDIFBlock],
) -> tuple[list[MDIFBlock] | None, list[MDIFBlock] | None]:
    if not args.coarse_mdif:
        return None, None
    coarse_blocks = read_mdif_cached(args.coarse_mdif)
    if args.coarse_verification_mdif:
        return coarse_blocks, read_mdif_cached(args.coarse_verification_mdif)

    split_data = split_blocks(
        coarse_blocks,
        split_var=args.split_var,
        train_values=parse_csv_set(args.train_values),
        verify_values=parse_csv_set(args.verify_values),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    if split_data.train and split_data.verify:
        return split_data.train, split_data.verify
    return coarse_blocks, coarse_blocks if verify_fine else []


def block_key(block: MDIFBlock, parameter_names: Sequence[str]) -> tuple[float, ...]:
    return tuple(round(float(value), 15) for value in block_parameter_values(block, parameter_names))


def block_parameter_values(block: MDIFBlock, parameter_names: Sequence[str]) -> np.ndarray:
    key = (id(block), tuple(parameter_names))
    cached = _BLOCK_PARAMETER_CACHE.get(key)
    if cached is not None:
        return cached
    values = []
    for name in parameter_names:
        if name not in block.params:
            raise ValueError(f"Block {block.source_index} is missing parameter {name!r}")
        value = parse_number(block.params[name])
        if value is None:
            raise ValueError(
                f"Block {block.source_index} parameter {name!r} is not numeric: {block.params[name]!r}"
            )
        values.append(float(value))
    result = np.asarray(values, dtype=float)
    _BLOCK_PARAMETER_CACHE[key] = result
    return result


def block_values(block: MDIFBlock, labels: Sequence[str]) -> np.ndarray:
    return np.column_stack([block.sparams[label] for label in labels])


def complex_to_columns(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values.real, values.imag], axis=1)


def columns_to_complex(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    return values[:, :half] + 1j * values[:, half:]


def fit_output_standardizer(
    data: np.ndarray,
    labels: Sequence[str],
) -> tuple[Standardizer, list[str], float]:
    mean = np.mean(data, axis=0)
    raw_std = np.std(data, axis=0)
    varying = raw_std >= EPS
    if np.any(varying):
        floor = float(np.median(raw_std[varying]))
        if not math.isfinite(floor) or floor < EPS:
            floor = 1.0
    else:
        floor = 1.0

    std = raw_std.copy()
    std[~varying] = floor
    scaler = Standardizer()
    scaler.mean = mean
    scaler.std = std
    floored_columns = [
        name
        for name, is_floored in zip(sparameter_real_imag_columns(labels), ~varying)
        if bool(is_floored)
    ]
    return scaler, floored_columns, floor


def kbnn_feature_columns(
    parameter_names: Sequence[str],
    labels: Sequence[str],
    mode: str,
    include_coarse_input: bool,
    freq_transform: str,
) -> list[str]:
    columns = [*parameter_names, *frequency_feature_columns(freq_transform)]
    include_coarse = bool(include_coarse_input or normalize_mode(mode) == "prior-input")
    if normalize_mode(mode) == "plain":
        include_coarse = False
    if include_coarse:
        columns.extend(f"coarse_{label}_real" for label in labels)
        columns.extend(f"coarse_{label}_imag" for label in labels)
    return list(columns)


def finite_array_stats(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    stats: dict[str, object] = {
        "shape": [int(size) for size in array.shape],
        "finite_values": int(finite.size),
        "nonfinite_values": int(array.size - finite.size),
    }
    if finite.size:
        stats.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "rms": float(np.sqrt(np.mean(finite * finite))),
            }
        )
    return stats


def vector_stats(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"finite_values": 0}
    return {
        "finite_values": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def finite_loss(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def compact_list(values: Sequence[object], limit: int = 12) -> list[object]:
    items = list(values)
    if len(items) <= limit:
        return items
    return [*items[:limit], f"... ({len(items) - limit} more)"]


def debug_print(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "debug", False):
        label = str(getattr(args, "progress_label", "KBNN debug"))
        print(f"debug: {label}: {message}", file=sys.stderr, flush=True)


def build_training_debug_info(
    args: argparse.Namespace,
    mode: str,
    include_coarse_input: bool,
    parameter_names: Sequence[str],
    labels: Sequence[str],
    train_fine: Sequence[MDIFBlock],
    verify_fine: Sequence[MDIFBlock],
    train_coarse: Sequence[MDIFBlock],
    verify_coarse: Sequence[MDIFBlock],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_verify: np.ndarray | None,
    y_verify: np.ndarray | None,
    x_scaler: Standardizer,
    y_scaler: Standardizer,
    floored_output_columns: Sequence[str],
    output_std_floor: float,
    initial_train_loss: float,
    initial_verify_loss: float | None,
    final_train_loss: float,
    final_verify_loss: float | None,
    history: Sequence[dict[str, float]],
) -> dict[str, object]:
    feature_columns = kbnn_feature_columns(
        parameter_names,
        labels,
        mode,
        include_coarse_input,
        args.freq_transform,
    )
    target_columns = sparameter_real_imag_columns(labels)
    raw_x_std = np.std(x_train, axis=0)
    raw_y_std = np.std(y_train, axis=0)
    constant_features = [
        name
        for name, std in zip(feature_columns, raw_x_std)
        if math.isfinite(float(std)) and float(std) < EPS
    ]
    constant_targets = [
        name
        for name, std in zip(target_columns, raw_y_std)
        if math.isfinite(float(std)) and float(std) < EPS
    ]
    best_history = None
    if history:
        def history_val_loss(row: dict[str, float]) -> float:
            value = finite_loss(row.get("val_loss"))
            return value if value is not None else float("inf")

        best_history = min(
            history,
            key=history_val_loss,
        )
    coarse_train_keys = [block_key(block, parameter_names) for block in train_coarse] if train_coarse else []
    fine_train_keys = [block_key(block, parameter_names) for block in train_fine]
    alignment_mismatches = 0
    if train_coarse:
        alignment_mismatches += sum(
            int(fine_key != coarse_key)
            for fine_key, coarse_key in zip(fine_train_keys, coarse_train_keys)
        )
    if verify_fine and verify_coarse:
        alignment_mismatches += sum(
            int(block_key(fine, parameter_names) != block_key(coarse, parameter_names))
            for fine, coarse in zip(verify_fine, verify_coarse)
        )
    return {
        "mode": mode,
        "include_coarse_input": bool(include_coarse_input),
        "coarse_mdif_supplied": bool(args.coarse_mdif),
        "coarse_verification_mdif_supplied": bool(args.coarse_verification_mdif),
        "freq_transform": args.freq_transform,
        "parameter_names": list(parameter_names),
        "sparam_labels": list(labels),
        "blocks": {
            "train_fine": len(train_fine),
            "verify_fine": len(verify_fine),
            "train_coarse": len(train_coarse),
            "verify_coarse": len(verify_coarse),
            "coarse_alignment_mismatches": int(alignment_mismatches),
        },
        "samples": {
            "x_train": finite_array_stats(x_train),
            "y_train": finite_array_stats(y_train),
            "x_verify": finite_array_stats(x_verify) if x_verify is not None else None,
            "y_verify": finite_array_stats(y_verify) if y_verify is not None else None,
        },
        "feature_scaling": {
            "columns": feature_columns,
            "raw_std": vector_stats(raw_x_std),
            "scaler_std": vector_stats(x_scaler.std if x_scaler.std is not None else np.asarray([])),
            "constant_columns": constant_features,
        },
        "target_scaling": {
            "columns": target_columns,
            "raw_std": vector_stats(raw_y_std),
            "scaler_std": vector_stats(y_scaler.std if y_scaler.std is not None else np.asarray([])),
            "constant_columns": constant_targets,
            "floored_output_columns": list(floored_output_columns),
            "output_std_floor": float(output_std_floor),
        },
        "loss_scaled": {
            "initial_train": float(initial_train_loss),
            "initial_verify": finite_loss(initial_verify_loss),
            "final_best_train": float(final_train_loss),
            "final_best_verify": finite_loss(final_verify_loss),
            "improvement_train": (
                float(initial_train_loss / final_train_loss)
                if final_train_loss > 0.0
                else None
            ),
            "improvement_verify": (
                float(initial_verify_loss / final_verify_loss)
                if initial_verify_loss is not None and final_verify_loss is not None and final_verify_loss > 0.0
                else None
            ),
        },
        "history": {
            "rows": len(history),
            "first": dict(history[0]) if history else None,
            "last_recorded": dict(history[-1]) if history else None,
            "best_recorded": dict(best_history) if best_history is not None else None,
        },
    }


def emit_training_debug(args: argparse.Namespace, info: dict[str, object]) -> None:
    if not getattr(args, "debug", False):
        return
    blocks = info["blocks"] if isinstance(info.get("blocks"), dict) else {}
    samples = info["samples"] if isinstance(info.get("samples"), dict) else {}
    feature_scaling = info["feature_scaling"] if isinstance(info.get("feature_scaling"), dict) else {}
    target_scaling = info["target_scaling"] if isinstance(info.get("target_scaling"), dict) else {}
    loss_scaled = info["loss_scaled"] if isinstance(info.get("loss_scaled"), dict) else {}
    debug_print(
        args,
        (
            f"mode={info.get('mode')} coarse_input={info.get('include_coarse_input')} "
            f"coarse_mdif={info.get('coarse_mdif_supplied')}"
        ),
    )
    debug_print(
        args,
        (
            "blocks "
            f"train_fine={blocks.get('train_fine')} verify_fine={blocks.get('verify_fine')} "
            f"train_coarse={blocks.get('train_coarse')} verify_coarse={blocks.get('verify_coarse')} "
            f"align_mismatch={blocks.get('coarse_alignment_mismatches')}"
        ),
    )
    x_train_stats = samples.get("x_train") if isinstance(samples.get("x_train"), dict) else {}
    y_train_stats = samples.get("y_train") if isinstance(samples.get("y_train"), dict) else {}
    debug_print(
        args,
        (
            f"x_train shape={x_train_stats.get('shape')} nonfinite={x_train_stats.get('nonfinite_values')} "
            f"y_train shape={y_train_stats.get('shape')} nonfinite={y_train_stats.get('nonfinite_values')}"
        ),
    )
    debug_print(
        args,
        (
            f"feature std={feature_scaling.get('raw_std')} "
            f"constant_features={compact_list(feature_scaling.get('constant_columns', []))}"
        ),
    )
    debug_print(
        args,
        (
            f"target std={target_scaling.get('raw_std')} "
            f"floored_targets={compact_list(target_scaling.get('floored_output_columns', []))}"
        ),
    )
    debug_print(
        args,
        (
            "scaled loss "
            f"train {loss_scaled.get('initial_train')} -> {loss_scaled.get('final_best_train')} "
            f"verify {loss_scaled.get('initial_verify')} -> {loss_scaled.get('final_best_verify')}"
        ),
    )


def interpolate_coarse_to_fine(
    coarse: MDIFBlock,
    fine: MDIFBlock,
    labels: Sequence[str],
) -> MDIFBlock:
    if np.array_equal(coarse.freq_hz, fine.freq_hz):
        return MDIFBlock(
            params=dict(fine.params),
            freq_hz=fine.freq_hz,
            sparams={label: coarse.sparams[label] for label in labels},
            source_index=fine.source_index,
        )
    if np.min(fine.freq_hz) < np.min(coarse.freq_hz) or np.max(fine.freq_hz) > np.max(coarse.freq_hz):
        raise ValueError(
            f"Coarse block for fine block {fine.source_index} does not cover the fine frequency range"
        )
    sparams = {}
    for label in labels:
        coarse_values = coarse.sparams[label]
        real = np.interp(fine.freq_hz, coarse.freq_hz, coarse_values.real)
        imag = np.interp(fine.freq_hz, coarse.freq_hz, coarse_values.imag)
        sparams[label] = real + 1j * imag
    return MDIFBlock(params=dict(fine.params), freq_hz=fine.freq_hz, sparams=sparams, source_index=fine.source_index)


def zero_coarse_blocks(fine_blocks: Sequence[MDIFBlock], labels: Sequence[str]) -> list[MDIFBlock]:
    zeros = []
    for fine in fine_blocks:
        zeros.append(
            MDIFBlock(
                params=dict(fine.params),
                freq_hz=fine.freq_hz,
                sparams={label: np.zeros_like(fine.sparams[label]) for label in labels},
                source_index=fine.source_index,
            )
        )
    return zeros


def align_coarse_blocks(
    fine_blocks: Sequence[MDIFBlock],
    coarse_blocks: Sequence[MDIFBlock] | None,
    parameter_names: Sequence[str],
    labels: Sequence[str],
) -> list[MDIFBlock]:
    cache_key = (
        tuple(id(block) for block in fine_blocks),
        None if coarse_blocks is None else tuple(id(block) for block in coarse_blocks),
        tuple(parameter_names),
        tuple(labels),
    )
    cached = _ALIGN_COARSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if coarse_blocks is None:
        aligned_zero = zero_coarse_blocks(fine_blocks, labels)
        _ALIGN_COARSE_CACHE[cache_key] = aligned_zero
        return aligned_zero

    if len(coarse_blocks) == len(fine_blocks):
        aligned_by_order = True
        for fine, coarse in zip(fine_blocks, coarse_blocks):
            if block_key(fine, parameter_names) != block_key(coarse, parameter_names):
                aligned_by_order = False
                break
        if aligned_by_order:
            aligned_by_order_blocks = [
                interpolate_coarse_to_fine(coarse, fine, labels)
                for fine, coarse in zip(fine_blocks, coarse_blocks)
            ]
            _ALIGN_COARSE_CACHE[cache_key] = aligned_by_order_blocks
            return aligned_by_order_blocks

    buckets: dict[tuple[float, ...], list[MDIFBlock]] = {}
    for coarse in coarse_blocks:
        buckets.setdefault(block_key(coarse, parameter_names), []).append(coarse)
    aligned = []
    for fine in fine_blocks:
        key = block_key(fine, parameter_names)
        choices = buckets.get(key)
        if not choices:
            raise ValueError(f"No matching coarse block for fine block {fine.source_index} with key {key}")
        aligned.append(interpolate_coarse_to_fine(choices.pop(0), fine, labels))
    _ALIGN_COARSE_CACHE[cache_key] = aligned
    return aligned


def frequency_feature(freq_hz: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log10(np.maximum(freq_hz, 1.0))[:, None]
    if transform == "linear":
        return freq_hz[:, None]
    raise ValueError(f"Unsupported frequency transform {transform!r}")


def make_feature_target_samples(
    fine_blocks: Sequence[MDIFBlock],
    coarse_blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    mode: str,
    include_coarse_input: bool,
    freq_transform: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode = normalize_mode(mode)
    include_coarse_input = bool(include_coarse_input or mode == "prior-input")
    if mode == "plain":
        include_coarse_input = False
    needs_coarse = mode == "residual" or include_coarse_input
    if needs_coarse and len(coarse_blocks) != len(fine_blocks):
        raise ValueError("KBNN coarse blocks must align one-to-one with fine blocks")
    cache_key = (
        tuple(id(block) for block in fine_blocks),
        tuple(id(block) for block in coarse_blocks) if needs_coarse else (),
        tuple(parameter_names),
        tuple(labels),
        mode,
        include_coarse_input,
        freq_transform,
    )
    cached = _SAMPLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    total_rows = sum(len(block.freq_hz) for block in fine_blocks)
    n_params = len(parameter_names)
    n_freq_features = len(frequency_feature_columns(freq_transform))
    n_labels = len(labels)
    n_coarse_columns = 2 * n_labels if include_coarse_input else 0
    features = np.empty((total_rows, n_params + n_freq_features + n_coarse_columns), dtype=float)
    target_matrix = np.empty((total_rows, n_labels), dtype=complex)

    offset = 0
    for block_idx, fine in enumerate(fine_blocks):
        coarse = coarse_blocks[block_idx] if needs_coarse else None
        nfreq = len(fine.freq_hz)
        end = offset + nfreq
        if n_params:
            features[offset:end, :n_params] = block_parameter_values(fine, parameter_names)
        features[offset:end, n_params : n_params + n_freq_features] = frequency_feature(
            fine.freq_hz,
            freq_transform,
        )
        fine_values = np.empty((nfreq, n_labels), dtype=complex)
        for label_idx, label in enumerate(labels):
            fine_values[:, label_idx] = fine.sparams[label]
        if mode == "residual":
            assert coarse is not None
            coarse_values = np.empty((nfreq, n_labels), dtype=complex)
            for label_idx, label in enumerate(labels):
                coarse_values[:, label_idx] = coarse.sparams[label]
            block_target_values = fine_values - coarse_values
        else:
            block_target_values = fine_values

        if include_coarse_input:
            if mode == "residual":
                input_coarse_values = coarse_values
            else:
                assert coarse is not None
                input_coarse_values = np.empty((nfreq, n_labels), dtype=complex)
                for label_idx, label in enumerate(labels):
                    input_coarse_values[:, label_idx] = coarse.sparams[label]
            coarse_start = n_params + n_freq_features
            features[offset:end, coarse_start : coarse_start + n_labels] = input_coarse_values.real
            features[offset:end, coarse_start + n_labels : coarse_start + 2 * n_labels] = (
                input_coarse_values.imag
            )
        target_matrix[offset:end, :] = block_target_values
        offset = end

    result = features, complex_to_columns(target_matrix)
    _SAMPLE_CACHE[cache_key] = result
    _FEATURE_CACHE[cache_key] = features
    return result


def make_feature_samples(
    fine_blocks: Sequence[MDIFBlock],
    coarse_blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    mode: str,
    include_coarse_input: bool,
    freq_transform: str,
) -> np.ndarray:
    mode = normalize_mode(mode)
    include_coarse_input = bool(include_coarse_input or mode == "prior-input")
    if mode == "plain":
        include_coarse_input = False
    needs_coarse = mode == "residual" or include_coarse_input
    if needs_coarse and len(coarse_blocks) != len(fine_blocks):
        raise ValueError("KBNN coarse blocks must align one-to-one with fine blocks")
    sample_key = (
        tuple(id(block) for block in fine_blocks),
        tuple(id(block) for block in coarse_blocks) if needs_coarse else (),
        tuple(parameter_names),
        tuple(labels),
        mode,
        include_coarse_input,
        freq_transform,
    )
    cached_sample = _SAMPLE_CACHE.get(sample_key)
    if cached_sample is not None:
        return cached_sample[0]
    cached_feature = _FEATURE_CACHE.get(sample_key)
    if cached_feature is not None:
        return cached_feature

    total_rows = sum(len(block.freq_hz) for block in fine_blocks)
    n_params = len(parameter_names)
    n_freq_features = len(frequency_feature_columns(freq_transform))
    n_labels = len(labels)
    n_coarse_columns = 2 * n_labels if include_coarse_input else 0
    features = np.empty((total_rows, n_params + n_freq_features + n_coarse_columns), dtype=float)

    offset = 0
    for block_idx, fine in enumerate(fine_blocks):
        coarse = coarse_blocks[block_idx] if needs_coarse else None
        nfreq = len(fine.freq_hz)
        end = offset + nfreq
        if n_params:
            features[offset:end, :n_params] = block_parameter_values(fine, parameter_names)
        features[offset:end, n_params : n_params + n_freq_features] = frequency_feature(
            fine.freq_hz,
            freq_transform,
        )
        if include_coarse_input:
            assert coarse is not None
            coarse_start = n_params + n_freq_features
            for label_idx, label in enumerate(labels):
                coarse_values = coarse.sparams[label]
                features[offset:end, coarse_start + label_idx] = coarse_values.real
                features[offset:end, coarse_start + n_labels + label_idx] = coarse_values.imag
        offset = end

    _FEATURE_CACHE[sample_key] = features
    return features


class KBNN:
    def __init__(
        self,
        mlp: MLP,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        parameter_names: list[str],
        sparam_labels: list[str],
        mode: str,
        include_coarse_input: bool,
        freq_transform: str,
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = parameter_names
        self.sparam_labels = sparam_labels
        self.mode = mode
        self.include_coarse_input = include_coarse_input
        self.freq_transform = freq_transform

    def predict_blocks(
        self,
        fine_shape_blocks: Sequence[MDIFBlock],
        coarse_blocks: Sequence[MDIFBlock],
    ) -> list[MDIFBlock]:
        x = make_feature_samples(
            fine_shape_blocks,
            coarse_blocks,
            self.parameter_names,
            self.sparam_labels,
            self.mode,
            self.include_coarse_input,
            self.freq_transform,
        )
        y_scaled = self.mlp.predict(self.x_scaler.transform(x))
        y_columns = self.y_scaler.inverse_transform(y_scaled)
        y_complex = columns_to_complex(y_columns)
        predicted = []
        offset = 0
        needs_coarse_output = self.mode == "residual"
        if needs_coarse_output and len(coarse_blocks) != len(fine_shape_blocks):
            raise ValueError("Residual KBNN prediction requires one coarse block per fine block")
        for block_idx, fine in enumerate(fine_shape_blocks):
            nfreq = len(fine.freq_hz)
            end = offset + nfreq
            block_y_complex = y_complex[offset:end]
            if self.mode == "residual":
                coarse_values = block_values(coarse_blocks[block_idx], self.sparam_labels)
                pred_values = coarse_values + block_y_complex
            else:
                pred_values = block_y_complex
            sparams = {
                label: pred_values[:, idx]
                for idx, label in enumerate(self.sparam_labels)
            }
            predicted.append(
                MDIFBlock(
                    params=dict(fine.params),
                    freq_hz=fine.freq_hz.copy(),
                    sparams=sparams,
                    source_index=fine.source_index,
                )
            )
            offset = end
        return predicted

    def save(self, out_dir: Path, metadata: dict[str, object]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "x_mean": self.x_scaler.mean,
            "x_std": self.x_scaler.std,
            "y_mean": self.y_scaler.mean,
            "y_std": self.y_scaler.std,
        }
        for idx, (weight, bias) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
            arrays[f"W{idx}"] = weight
            arrays[f"b{idx}"] = bias
        np.savez_compressed(out_dir / "model.npz", **arrays)
        combined_metadata = {
            "version": VERSION,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation,
            "mode": self.mode,
            "include_coarse_input": self.include_coarse_input,
            "freq_transform": self.freq_transform,
            **metadata,
        }
        (out_dir / "metadata.json").write_text(json.dumps(combined_metadata, indent=2))

    @staticmethod
    def load(model_dir: Path) -> "KBNN":
        metadata = json.loads((model_dir / "metadata.json").read_text())
        data = np.load(model_dir / "model.npz")
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
        return KBNN(
            mlp=mlp,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            parameter_names=list(metadata["parameter_names"]),
            sparam_labels=list(metadata["sparam_labels"]),
            mode=metadata["mode"],
            include_coarse_input=bool(metadata["include_coarse_input"]),
            freq_transform=metadata["freq_transform"],
        )


def determine_labels(all_fine: Sequence[MDIFBlock], coarse_sets: Sequence[MDIFBlock] | None) -> list[str]:
    if coarse_sets:
        return common_sparameter_labels(list(all_fine) + list(coarse_sets))
    return common_sparameter_labels(all_fine)


def train_model(args: argparse.Namespace) -> tuple[KBNN, list[MDIFBlock], list[MDIFBlock], list[str], list[str], list[dict[str, float]], dict[str, object] | None]:
    mode = normalize_mode(args.mode)
    include_coarse_input = parse_bool_option(args.include_coarse_input)
    if mode == "plain":
        include_coarse_input = False
    if mode == "prior-input":
        include_coarse_input = True
    if mode == "prior-input" and not args.coarse_mdif:
        raise ValueError("--mode prior-input requires --coarse-mdif")
    if include_coarse_input and not args.coarse_mdif:
        raise ValueError("--include-coarse-input requires --coarse-mdif")

    train_fine, verify_fine, all_fine = split_fine_blocks(args)
    if not train_fine:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(all_fine, requested=args.parameter_names, split_var=args.split_var)
    if mode == "plain":
        coarse_train_raw = None
        coarse_verify_raw = None
        all_coarse_raw: list[MDIFBlock] = []
        labels = determine_labels(all_fine, None)
        train_coarse: list[MDIFBlock] = []
        verify_coarse: list[MDIFBlock] = []
    else:
        coarse_train_raw, coarse_verify_raw = split_coarse_blocks(args, train_fine, verify_fine)
        all_coarse_raw = []
        if coarse_train_raw:
            all_coarse_raw.extend(coarse_train_raw)
        if coarse_verify_raw:
            all_coarse_raw.extend(coarse_verify_raw)
        labels = determine_labels(all_fine, all_coarse_raw or None)
        train_coarse = align_coarse_blocks(train_fine, coarse_train_raw, parameter_names, labels)
        verify_coarse = align_coarse_blocks(verify_fine, coarse_verify_raw, parameter_names, labels) if verify_fine else []

    x_train, y_train = make_feature_target_samples(
        train_fine,
        train_coarse,
        parameter_names,
        labels,
        mode,
        include_coarse_input,
        args.freq_transform,
    )
    if verify_fine:
        x_verify, y_verify = make_feature_target_samples(
            verify_fine,
            verify_coarse,
            parameter_names,
            labels,
            mode,
            include_coarse_input,
            args.freq_transform,
        )
    else:
        x_verify = None
        y_verify = None

    x_scaler = Standardizer().fit(x_train)
    y_scaler, floored_output_columns, output_std_floor = fit_output_standardizer(y_train, labels)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_verify_scaled = x_scaler.transform(x_verify) if x_verify is not None else None
    y_verify_scaled = y_scaler.transform(y_verify) if y_verify is not None else None

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    layer_sizes = [x_train.shape[1], *hidden_layers, y_train.shape[1]]
    mlp = MLP(layer_sizes, activation=args.activation, seed=args.seed)
    sparam_weights = parse_sparam_weights(labels, getattr(args, "sparam_weights", None))
    normalized_sparam_weights = normalize_sparam_weights(labels, sparam_weights)
    output_weights = output_weights_from_sparam_weights(labels, sparam_weights)
    progress_interval = progress_interval_from_args(args)
    initial_train_loss = mse(mlp.predict(x_train_scaled), y_train_scaled, output_weights=output_weights)
    initial_verify_loss = (
        mse(mlp.predict(x_verify_scaled), y_verify_scaled, output_weights=output_weights)
        if x_verify_scaled is not None and y_verify_scaled is not None
        else None
    )
    history = mlp.train(
        x_train_scaled,
        y_train_scaled,
        x_verify_scaled,
        y_verify_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed + 19,
        output_weights=output_weights,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_callback=make_training_progress_callback(
            getattr(args, "progress_label", "KBNN fit"),
            args.epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    final_train_loss = mse(mlp.predict(x_train_scaled), y_train_scaled, output_weights=output_weights)
    final_verify_loss = (
        mse(mlp.predict(x_verify_scaled), y_verify_scaled, output_weights=output_weights)
        if x_verify_scaled is not None and y_verify_scaled is not None
        else None
    )
    model = KBNN(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=labels,
        mode=mode,
        include_coarse_input=include_coarse_input,
        freq_transform=args.freq_transform,
    )
    metadata = {
        "training_blocks": len(train_fine),
        "verification_blocks": len(verify_fine),
        "training_samples": int(x_train.shape[0]),
        "mode": mode,
        "include_coarse_input": include_coarse_input,
        "freq_transform": args.freq_transform,
        "coarse_mdif": bool(args.coarse_mdif),
        "split_var": args.split_var,
        "train_values": sorted(parse_csv_set(args.train_values)),
        "verify_values": sorted(parse_csv_set(args.verify_values)),
        "sparam_weights": sparam_weights,
        "normalized_sparam_weights": normalized_sparam_weights,
        "sparam_weight_mean": sparam_weight_mean(labels, sparam_weights),
        "sparam_weight_normalization": "Raw S-parameter weights are divided by their mean before training, so the average normalized weight is 1.0.",
        "output_scaler_floor": output_std_floor,
        "floored_output_columns": floored_output_columns,
    }
    if getattr(args, "debug", False):
        debug_info = build_training_debug_info(
            args,
            mode,
            include_coarse_input,
            parameter_names,
            labels,
            train_fine,
            verify_fine,
            train_coarse,
            verify_coarse,
            x_train,
            y_train,
            x_verify,
            y_verify,
            x_scaler,
            y_scaler,
            floored_output_columns,
            output_std_floor,
            initial_train_loss,
            initial_verify_loss,
            final_train_loss,
            final_verify_loss,
            history,
        )
        metadata["training_debug"] = debug_info
        emit_training_debug(args, debug_info)
    return model, verify_fine, verify_coarse, parameter_names, labels, history, metadata


def command_train(args: argparse.Namespace) -> int:
    model, verify_fine, verify_coarse, parameter_names, labels, history, metadata = train_model(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assert metadata is not None
    model.save(out_dir, metadata=metadata)
    debug_info = metadata.get("training_debug")
    if getattr(args, "debug", False) and isinstance(debug_info, dict):
        debug_path = out_dir / "kbnn_training_debug.json"
        debug_path.write_text(json.dumps(debug_info, indent=2))
        debug_print(args, f"wrote {debug_path}")
    training_config = {
        "training_blocks": metadata["training_blocks"],
        "verification_blocks": metadata["verification_blocks"],
        "training_samples": metadata["training_samples"],
        "parameters": parameter_names,
        "sparameters": labels,
        "mode": model.mode,
        "include_coarse_input": model.include_coarse_input,
        "freq_transform": model.freq_transform,
        "coarse_mdif": metadata["coarse_mdif"],
        "hidden_layers": args.hidden_layers,
        "activation": model.mlp.activation,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "loss_interval": getattr(args, "loss_interval", 1),
        "progress_interval": progress_interval_from_args(args),
        "seed": args.seed,
        "sparam_weights": metadata["sparam_weights"],
        "normalized_sparam_weights": metadata["normalized_sparam_weights"],
        "output_scaler_floor": metadata["output_scaler_floor"],
        "floored_output_columns": metadata["floored_output_columns"],
    }
    plot_context = model_settings_title(
        "KBNN",
        training_config,
        getattr(args, "progress_label", "KBNN fit"),
    )
    write_history(
        out_dir / "training_history.csv",
        history,
        plot_title=f"Model performance vs epoch | {plot_context}",
    )

    if verify_fine:
        pred_blocks = model.predict_blocks(verify_fine, verify_coarse)
        summary = write_training_verification_artifacts(
            out_dir,
            verify_fine,
            pred_blocks,
            labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            sparam_weights=parse_sparam_weights(labels, getattr(args, "sparam_weights", None)),
            y_z0=50.0,
            title_context=plot_context,
        )
    else:
        summary = {"warning": "No verification blocks were available"}
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="KBNN",
        config=training_config,
        summary=summary,
        history=history,
    )

    if not getattr(args, "quiet", False):
        print(json.dumps({
            "out_dir": str(out_dir),
            "training_summary": str(out_dir / "training_summary.md"),
            "mode": model.mode,
            "include_coarse_input": model.include_coarse_input,
            "parameters": parameter_names,
            "sparameters": labels,
            "sparam_weights": metadata["sparam_weights"],
            "normalized_sparam_weights": metadata["normalized_sparam_weights"],
            "sparam_weight_mean": metadata["sparam_weight_mean"],
            "output_scaler_floor": metadata["output_scaler_floor"],
            "floored_output_columns": metadata["floored_output_columns"],
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
        }, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model = KBNN.load(Path(args.model_dir))
    blocks = read_mdif_cached(args.mdif)
    if model.mode == "prior-input" and not args.coarse_mdif:
        raise ValueError("This model requires --coarse-mdif for prediction")
    if model.mode == "plain":
        coarse = []
    else:
        coarse_raw = read_mdif_cached(args.coarse_mdif) if args.coarse_mdif else None
        coarse = align_coarse_blocks(blocks, coarse_raw, model.parameter_names, model.sparam_labels)
    pred_blocks = model.predict_blocks(blocks, coarse)
    out_path = Path(args.out_mdif)
    write_mdif(out_path, pred_blocks, model.sparam_labels)
    print(f"Wrote {out_path}")
    return 0


def command_export_ads(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = KBNN.load(model_dir)
    if model.mode == "prior-input" and not args.coarse_mdif:
        raise ValueError("This KBNN prior-input model requires --coarse-mdif for ADS export")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mdif_name = args.output_name
    blocks = build_ads_export_blocks(
        template_mdif=args.template_mdif,
        parameter_grid_specs=args.parameter_grid,
        freqs_spec=args.freqs,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
    )
    if model.mode == "plain":
        coarse = []
        coarse_raw = None
    else:
        coarse_raw = read_mdif_cached(args.coarse_mdif) if args.coarse_mdif else None
        coarse = align_coarse_blocks(blocks, coarse_raw, model.parameter_names, model.sparam_labels)
    pred_blocks = model.predict_blocks(blocks, coarse)
    write_mdif(out_dir / mdif_name, pred_blocks, model.sparam_labels)

    notes = []
    if args.coarse_mdif:
        notes.append(
            "The coarse/prior response was evaluated during export; ADS only needs the final exported MDIF."
        )
    elif model.mode == "residual":
        notes.append("No coarse/prior MDIF was supplied for export, so the residual model used a zero coarse response.")
    manifest = write_ads_export_package(
        out_dir=out_dir,
        model_kind="KBNN",
        model_dir=model_dir,
        mdif_name=mdif_name,
        blocks=pred_blocks,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        extra_manifest={
            "mode": model.mode,
            "include_coarse_input": model.include_coarse_input,
            "freq_transform": model.freq_transform,
            "layer_sizes": model.mlp.layer_sizes,
            "coarse_mdif": args.coarse_mdif,
        },
        extra_notes=notes,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mdif": str(out_dir / mdif_name),
        "manifest": str(out_dir / "ads_model_manifest.json"),
        "blocks": manifest["blocks"],
        "frequency_points_per_block": manifest["frequency_points_per_block"],
    }, indent=2))
    return 0


def fine_target_matrix(fine_blocks: Sequence[MDIFBlock], labels: Sequence[str]) -> np.ndarray:
    cache_key = (tuple(id(block) for block in fine_blocks), tuple(labels), "fine_target")
    cached = _FINE_TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached
    total_rows = sum(len(block.freq_hz) for block in fine_blocks)
    values = np.empty((total_rows, len(labels)), dtype=complex)
    offset = 0
    for block in fine_blocks:
        nfreq = len(block.freq_hz)
        end = offset + nfreq
        for label_idx, label in enumerate(labels):
            values[offset:end, label_idx] = block.sparams[label]
        offset = end
    result = complex_to_columns(values)
    _FINE_TARGET_CACHE[cache_key] = result
    return result


def command_export_ads_ann(args: argparse.Namespace) -> int:
    model_metadata = read_model_metadata(args.model_dir)
    args = argparse.Namespace(**vars(args))
    args.split_var = args.split_var or str(model_metadata.get("split_var", "dataset"))
    args.train_values = args.train_values or metadata_csv(model_metadata, "train_values") or "train,training"
    args.verify_values = (
        args.verify_values
        or metadata_csv(model_metadata, "verify_values")
        or "verify,verification,test,validation"
    )
    args.parameter_names = args.parameter_names or metadata_csv(model_metadata, "parameter_names")
    args.freq_transform = args.freq_transform or str(model_metadata.get("freq_transform", "log"))
    args.hidden_layers = args.hidden_layers or metadata_hidden_layers(model_metadata) or "64,64"
    args.activation = args.activation or str(model_metadata.get("activation", "tanh"))
    args.mode = args.mode or str(model_metadata.get("mode", "residual"))
    if args.include_coarse_input is None:
        args.include_coarse_input = bool(model_metadata.get("include_coarse_input", False))
    args.seed = args.seed if args.seed is not None else 1234
    if not getattr(args, "sparam_weights", None):
        metadata_weights = model_metadata.get("sparam_weights")
        if isinstance(metadata_weights, dict):
            args.sparam_weights = ";".join(
                f"{label}={metadata_weights[label]}" for label in sorted(metadata_weights, key=sparam_sort_key)
            )

    mode = normalize_mode(args.mode)
    include_coarse_input = parse_bool_option(args.include_coarse_input)
    if mode == "plain":
        include_coarse_input = False
    if mode == "prior-input":
        include_coarse_input = True
    if mode == "prior-input" and not args.coarse_mdif:
        raise ValueError("--mode prior-input requires --coarse-mdif for ADS ANN export")

    train_fine, verify_fine, all_fine = split_fine_blocks(args)
    if not train_fine:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(all_fine, requested=args.parameter_names, split_var=args.split_var)
    if mode == "plain":
        coarse_train_raw = None
        coarse_verify_raw = None
        all_coarse_raw: list[MDIFBlock] = []
    else:
        coarse_train_raw, coarse_verify_raw = split_coarse_blocks(args, train_fine, verify_fine)
        all_coarse_raw = []
        if coarse_train_raw:
            all_coarse_raw.extend(coarse_train_raw)
        if coarse_verify_raw:
            all_coarse_raw.extend(coarse_verify_raw)
    metadata_labels = model_metadata.get("sparam_labels")
    if isinstance(metadata_labels, list) and metadata_labels:
        labels = [str(label) for label in metadata_labels]
        label_blocks = list(all_fine) + all_coarse_raw
        missing = sorted(
            {label for label in labels for block in label_blocks if label not in block.sparams}
        )
        if missing:
            raise ValueError(
                f"Model metadata requested S-parameters not present in the MDIF data: {', '.join(missing)}"
            )
    else:
        labels = determine_labels(all_fine, all_coarse_raw or None)
    if mode == "plain":
        train_coarse = []
        verify_coarse = []
    else:
        train_coarse = align_coarse_blocks(train_fine, coarse_train_raw, parameter_names, labels)
        verify_coarse = align_coarse_blocks(verify_fine, coarse_verify_raw, parameter_names, labels) if verify_fine else []

    x_train, native_y_train = make_feature_target_samples(
        train_fine,
        train_coarse,
        parameter_names,
        labels,
        mode,
        include_coarse_input,
        args.freq_transform,
    )
    if verify_fine:
        x_verify, native_y_verify = make_feature_target_samples(
            verify_fine,
            verify_coarse,
            parameter_names,
            labels,
            mode,
            include_coarse_input,
            args.freq_transform,
        )
    else:
        x_verify = None
        native_y_verify = None

    target = args.ads_ann_target
    if target == "native":
        y_train = native_y_train
        y_verify = native_y_verify
        output_prefix = "delta" if mode == "residual" else "fine"
        if mode == "residual":
            target_description = (
                "KBNN residual correction, stored as fine minus coarse S-parameter real columns "
                "followed by imaginary columns. Final fine response is coarse plus ANN output."
            )
        else:
            target_description = (
                "Fine S-parameter response from the native KBNN formulation, stored as real columns "
                "followed by imaginary columns."
            )
    elif target == "fine":
        y_train = fine_target_matrix(train_fine, labels)
        y_verify = fine_target_matrix(verify_fine, labels) if verify_fine else None
        output_prefix = "fine"
        target_description = (
            "Direct fine S-parameter response, stored as real columns followed by imaginary columns. "
            "This is easiest to place in ADS, but residual mode no longer trains a delta target."
        )
    else:
        raise ValueError(f"Unsupported ADS ANN target {target!r}")

    requested_hidden_layers = parse_hidden_layers(args.hidden_layers)
    default_hidden_layers, default_neurons = infer_uniform_hidden_layout(requested_hidden_layers)
    ads_hidden_layers = args.ads_hidden_layers if args.ads_hidden_layers is not None else default_hidden_layers
    ads_neurons = args.ads_neurons_per_layer if args.ads_neurons_per_layer is not None else default_neurons
    if ads_hidden_layers <= 0:
        raise ValueError("--ads-hidden-layers must be positive")
    if ads_neurons <= 0:
        raise ValueError("--ads-neurons-per-layer must be positive")

    input_columns = [*parameter_names, *frequency_feature_columns(args.freq_transform)]
    if include_coarse_input or mode == "prior-input":
        input_columns.extend(sparameter_real_imag_columns(labels, prefix="coarse"))
    output_columns = sparameter_real_imag_columns(labels, prefix=output_prefix)
    settings = {
        "seed": args.seed,
        "num_hidden_layers": ads_hidden_layers,
        "num_neurons_per_layer": ads_neurons,
        "neuron_activation_function_type": ads_ann_activation_enum(args.activation),
        "network_training_type": ads_ann_training_type_enum(args.ads_network_training_type),
        "modeler_optimizer": ads_ann_optimizer_enum(args.ads_optimizer),
        "max_training_iterations": args.ads_iterations,
        "training_stop_tolerance": args.ads_training_stop_tolerance,
        "output_format": ads_ann_output_format_enum(args.ads_output_format),
        "output_prefix": normalize_name(args.output_prefix) or "kbnn_ads_ann",
    }

    notes = [
        "This export retrains the ANN in ADS; it does not import NumPy model.npz weights.",
        "When --model-dir is supplied, metadata from that trained or optimized model is used to seed the ADS ANN architecture/settings.",
    ]
    if mode == "residual" and target == "native":
        notes.append(
            "The native residual export predicts delta S. In ADS, add these outputs to the coarse/prior S-parameter response to reconstruct the fine response."
        )
    if include_coarse_input or mode == "prior-input":
        notes.append(
            "The exported ADS ANN inputs include coarse S-parameter real/imaginary values at each parameter/frequency sample."
        )
    if getattr(args, "sparam_weights", None):
        notes.append(
            "The package records S-parameter weights in the manifest. ADS ANN's documented Python API does not expose direct per-output loss weights, so the included ADS training script does not apply those weights."
        )
    if target == "fine" and mode == "residual":
        notes.append(
            "The fine-target export is simpler to consume in ADS, but it does not preserve the residual delta target that usually gives the KBNN its sample-efficiency advantage."
        )

    out_dir = Path(args.out_dir)
    manifest = write_ads_ann_package(
        out_dir=out_dir,
        model_kind="KBNN",
        input_columns=input_columns,
        output_columns=output_columns,
        x_train=x_train,
        y_train=y_train,
        x_verify=x_verify,
        y_verify=y_verify,
        settings=settings,
        parameter_names=parameter_names,
        sparam_labels=labels,
        target_description=target_description,
        extra_manifest={
            "source_model_dir": args.model_dir,
            "source_fine_mdif": args.mdif,
            "source_coarse_mdif": args.coarse_mdif,
            "verification_mdif": args.verification_mdif,
            "coarse_verification_mdif": args.coarse_verification_mdif,
            "mode": mode,
            "include_coarse_input": include_coarse_input,
            "ads_ann_target": target,
            "freq_transform": args.freq_transform,
            "sparam_weights": parse_sparam_weights(labels, getattr(args, "sparam_weights", None)),
            "requested_hidden_layers": requested_hidden_layers,
            "ads_layout_note": (
                "ADS ANN exposes a uniform hidden-layer width in the documented API. "
                "The package uses --ads-hidden-layers/--ads-neurons-per-layer, or derives "
                "them from --hidden-layers when those overrides are omitted."
            ),
        },
        extra_notes=notes,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "manifest": str(out_dir / "ads_ann_manifest.json"),
        "training_csv": str(out_dir / manifest["training_csv"]),
        "verification_csv": str(out_dir / manifest["verification_csv"]) if manifest.get("verification_csv") else None,
        "ads_script": str(out_dir / "train_ads_ann.py"),
        "input_columns": manifest["input_columns"],
        "output_columns": manifest["output_columns"],
        "ads_ann": manifest["ads_ann"],
        "mode": mode,
        "ads_ann_target": target,
    }, indent=2))
    return 0


def command_export_veriloga(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = KBNN.load(model_dir)
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'kbnn'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    uses_coarse_inputs = bool(model.include_coarse_input or model.mode == "prior-input")
    adds_coarse_to_output = model.mode == "residual"
    notes = [
        "This direct Verilog-A export embeds the saved local model.npz weights; it does not retrain in ADS ANN.",
        "The generated N-port is intended for S-parameter and small-signal AC simulation. It is not a causal transient model.",
        "The default frequency expression is $freq. If your ADS Verilog-A environment uses a different frequency variable, regenerate with --frequency-expression.",
    ]
    if uses_coarse_inputs or adds_coarse_to_output:
        notes.append(
            "This KBNN formulation requires a coarse response at runtime. The generated Verilog-A file includes coarse response assignment hooks with zero defaults inside the analog block; replace those assignment right hand sides with the actual coarse circuit/surrogate response before relying on the model."
        )
    if model.mode == "residual":
        notes.append(
            "Residual KBNN output is delta S internally; the generated N-port adds the coarse hook response before converting final S to Y."
        )
    elif model.mode == "prior-input":
        notes.append(
            "Prior-input KBNN output is final fine S, but the ANN inputs still require the coarse hook response."
        )

    manifest = write_veriloga_package(
        out_dir=out_dir,
        model_kind="KBNN",
        module_name=module_name,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        freq_transform=model.freq_transform,
        activation=model.mlp.activation,
        layer_sizes=model.mlp.layer_sizes,
        weights=model.mlp.weights,
        biases=model.mlp.biases,
        x_mean=np.asarray(model.x_scaler.mean, dtype=float),
        x_std=np.asarray(model.x_scaler.std, dtype=float),
        y_mean=np.asarray(model.y_scaler.mean, dtype=float),
        y_std=np.asarray(model.y_scaler.std, dtype=float),
        z0=args.z0,
        frequency_expression=args.frequency_expression,
        uses_coarse_inputs=uses_coarse_inputs,
        adds_coarse_to_output=adds_coarse_to_output,
        parameter_input_scales=parameter_input_scales,
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "knowledge_based_neural_network",
            "mode": model.mode,
            "include_coarse_input": model.include_coarse_input,
            "requires_coarse_hooks": bool(uses_coarse_inputs or adds_coarse_to_output),
        },
        extra_notes=notes,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "veriloga": str(out_dir / manifest["veriloga_file"]),
        "manifest": str(out_dir / "veriloga_manifest.json"),
        "readme": str(out_dir / "VERILOGA_README.md"),
        "module_name": manifest["module_name"],
        "nports": manifest["nports"],
        "parameters": manifest["parameter_identifiers"],
        "parameter_input_scales": manifest["parameter_input_scales"],
        "requires_coarse_hooks": manifest["requires_coarse_hooks"],
    }, indent=2))
    return 0


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
    return numeric if math.isfinite(numeric) else None


def sweep_candidate_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    axes = {
        "mode": [normalize_mode(value) for value in parse_text_options(args.mode_options)],
        "include_coarse_input": [parse_bool_option(value) for value in parse_text_options(args.include_coarse_input_options)],
        "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options),
        "activation": parse_text_options(args.activation_options),
        "learning_rate": parse_float_options(args.learning_rates),
    }
    candidates = []
    keys = list(axes)
    skipped_without_coarse = 0
    for values in itertools.product(*(axes[key] for key in keys)):
        candidate = dict(zip(keys, values))
        if candidate["mode"] == "plain" and candidate["include_coarse_input"]:
            continue
        if candidate["mode"] == "prior-input" and not candidate["include_coarse_input"]:
            continue
        if not args.coarse_mdif and (
            candidate["mode"] == "prior-input" or bool(candidate["include_coarse_input"])
        ):
            skipped_without_coarse += 1
            continue
        candidates.append(candidate)
    if skipped_without_coarse:
        print(
            "warning: skipped "
            f"{skipped_without_coarse} KBNN candidate(s) that require --coarse-mdif",
            file=sys.stderr,
        )
    if candidates and not args.coarse_mdif and any(candidate["mode"] == "residual" for candidate in candidates):
        print(
            "warning: --coarse-mdif was not supplied; residual KBNN candidates "
            "will use a zero coarse response",
            file=sys.stderr,
        )
    if not candidates:
        raise ValueError(
            "No valid KBNN sweep candidates. prior-input mode and coarse-input "
            "residual candidates require --coarse-mdif."
        )
    if args.mode == "random" and args.max_trials and args.max_trials < len(candidates):
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(candidates), size=args.max_trials, replace=False)
        candidates = [candidates[int(idx)] for idx in chosen]
    elif args.max_trials and args.max_trials < len(candidates):
        candidates = candidates[: args.max_trials]
    return candidates


def namespace_for_trial(
    args: argparse.Namespace,
    candidate: dict[str, object],
    out_dir: Path,
    trial_index: int,
    plots: int,
) -> argparse.Namespace:
    trial_seed = sweep_trial_seed(args.seed, trial_index, getattr(args, "trial_seed_mode", "fixed"))
    return argparse.Namespace(
        mdif=args.mdif,
        verification_mdif=args.verification_mdif,
        coarse_mdif=args.coarse_mdif,
        coarse_verification_mdif=args.coarse_verification_mdif,
        out_dir=str(out_dir),
        split_var=args.split_var,
        train_values=args.train_values,
        verify_values=args.verify_values,
        parameter_names=args.parameter_names,
        holdout_fraction=args.holdout_fraction,
        mode=str(candidate["mode"]),
        include_coarse_input=bool(candidate["include_coarse_input"]),
        freq_transform=args.freq_transform,
        hidden_layers=str(candidate["hidden_layers"]),
        activation=str(candidate["activation"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=float(candidate["learning_rate"]),
        patience=args.patience,
        loss_interval=args.loss_interval,
        progress_interval=args.progress_interval,
        progress_label=f"KBNN trial {trial_index}",
        seed=trial_seed,
        worst_plots=plots,
        sparam_weights=args.sparam_weights,
        debug=bool(getattr(args, "debug", False)),
        quiet=True,
    )


def kbnn_sweep_trial_worker(payload: tuple[dict[str, object], dict[str, object], str, int, int]) -> dict[str, object]:
    args_values, candidate, out_dir_text, trial_index, plots = payload
    args = argparse.Namespace(**args_values)
    out_dir = Path(out_dir_text)
    trial_dir = out_dir / "trials" / f"trial_{trial_index:04d}"
    error_message = None
    error_traceback = None
    trial_seed = sweep_trial_seed(args.seed, trial_index, getattr(args, "trial_seed_mode", "fixed"))
    try:
        trial_args = namespace_for_trial(args, candidate, trial_dir, trial_index, plots=plots)
        status = command_train(trial_args)
    except Exception as exc:
        status = 2
        error_message = str(exc)
        error_traceback = traceback.format_exc()
        if getattr(args, "debug", False):
            print(error_traceback, file=sys.stderr, flush=True)
    summary_path = trial_dir / "verification_summary.json"
    if status != 0 or not summary_path.exists():
        summary: dict[str, object] = {"error": error_message or "trial failed"}
        if error_traceback:
            summary["traceback"] = error_traceback
        metric_value = None
    else:
        summary = json.loads(summary_path.read_text())
        metric_value = summary_metric(summary, args.selection_metric)
    return {
        "trial": trial_index,
        "candidate": candidate,
        "summary": summary,
        "metric": metric_value,
        "trial_seed": trial_seed,
        "plot_paths": trial_plot_paths(summary, trial_dir, out_dir),
    }


def command_sweep(args: argparse.Namespace) -> int:
    candidates = sweep_candidate_grid(args)
    if getattr(args, "debug", False):
        print(
            f"debug: KBNN sweep: candidates={len(candidates)} jobs={args.jobs} "
            f"out_dir={args.out_dir}",
            file=sys.stderr,
            flush=True,
        )
        if args.jobs != 1:
            print(
                "debug: KBNN sweep: parallel trial debug output may interleave; "
                "use --jobs 1 for the cleanest trace",
                file=sys.stderr,
                flush=True,
            )
        for idx, candidate in enumerate(candidates, start=1):
            print(f"debug: KBNN sweep: candidate {idx}: {candidate}", file=sys.stderr, flush=True)
    return run_sweep_command(
        args,
        candidates,
        worker_func=kbnn_sweep_trial_worker,
        namespace_for_trial_func=namespace_for_trial,
        train_func=command_train,
        result_columns=KBNN_SWEEP_RESULT_COLUMNS,
        results_filename="kbnn_sweep_results.csv",
        best_config_filename="kbnn_best_config.json",
        summary_filename="kbnn_sweep_summary.md",
        diagnostics_prefix="kbnn",
    )


def command_rerank_sweep(args: argparse.Namespace) -> int:
    sweep_dir = Path(args.sweep_dir)
    results_filename = (
        "kbnn_sweep_results.csv"
        if (sweep_dir / "kbnn_sweep_results.csv").exists()
        else "sweep_results.csv"
    )
    rows = load_sweep_rows(sweep_dir, results_filename)
    if not rows:
        raise ValueError(f"No sweep rows found in {sweep_dir / results_filename}")

    reranked, best_row, best_metric = rerank_sweep_rows(
        rows,
        selection_metric=args.selection_metric,
        require_passive=args.require_passive,
        max_passivity_violations=args.max_passivity_violations,
        max_passivity_sigma=args.max_passivity_sigma,
    )
    if best_row is None or best_metric is None:
        raise ValueError("No sweep trial satisfied the rerank criteria")

    trial_value = csv_number(best_row.get("trial"))
    if trial_value is None:
        raise ValueError("Selected row does not have a numeric trial number")
    best_trial = int(trial_value)
    best_config = {
        key: best_row[key]
        for key in KBNN_SWEEP_RESULT_COLUMNS
        if key in best_row and best_row[key] not in {None, ""}
    }

    results_path = sweep_dir / "kbnn_reranked_sweep_results.csv"
    summary_path = sweep_dir / "kbnn_reranked_sweep_summary.md"
    best_config_path = sweep_dir / "kbnn_reranked_best_config.json"
    write_csv(results_path, reranked)
    diagnostic_artifacts = [
        str(path.relative_to(sweep_dir))
        for path in plot_sweep_diagnostics(
            reranked,
            sweep_dir,
            KBNN_SWEEP_RESULT_COLUMNS,
            args.selection_metric,
            prefix="kbnn_reranked",
        )
    ]
    write_sweep_markdown(
        summary_path,
        reranked,
        selection_metric=args.selection_metric,
        best_config=best_config,
        best_metric=best_metric,
        diagnostic_artifacts=diagnostic_artifacts,
    )

    promoted = False
    promotion_warning = None
    best_model_dir = None
    if args.promote_best or args.replace_current_best:
        if args.replace_current_best:
            best_model_dir = sweep_dir / "best_model"
            overwrite = True
        else:
            best_model_dir = (
                Path(args.best_model_dir)
                if args.best_model_dir
                else sweep_dir / "best_model_reranked"
            )
            overwrite = args.overwrite
        promoted, promotion_warning = copy_trial_model(
            sweep_dir,
            best_trial,
            best_model_dir,
            overwrite=overwrite,
        )

    payload = {
        "sweep_dir": str(sweep_dir),
        "selection_metric": args.selection_metric,
        "require_passive": bool(args.require_passive),
        "max_passivity_violations": args.max_passivity_violations,
        "max_passivity_sigma": args.max_passivity_sigma,
        "best_trial": best_trial,
        "best_metric": best_metric,
        "best_config": best_config,
        "reranked_results": str(results_path),
        "reranked_summary": str(summary_path),
        "diagnostic_artifacts": diagnostic_artifacts,
        "promoted": promoted,
        "best_model_dir": str(best_model_dir) if best_model_dir is not None else None,
        "promotion_warning": promotion_warning,
    }
    best_config_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    blocks = read_mdif_cached(args.mdif)
    labels = common_sparameter_labels(blocks)
    numeric_params = infer_parameter_names(blocks, requested=None, split_var=args.split_var)
    datasets = {}
    split_key = normalize_name(args.split_var)
    for block in blocks:
        value = block.params.get(split_key, "")
        datasets[value] = datasets.get(value, 0) + 1
    print(json.dumps({
        "blocks": len(blocks),
        "sparameters": labels,
        "inferred_numeric_parameters": numeric_params,
        "split_counts": datasets,
        "freq_min_hz": float(min(np.min(block.freq_hz) for block in blocks)),
        "freq_max_hz": float(max(np.max(block.freq_hz) for block in blocks)),
    }, indent=2))
    return 0


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mdif", required=True, help="Fine/target S-parameter MDIF")
    parser.add_argument("--verification-mdif", help="Optional separate fine/target verification MDIF")
    parser.add_argument("--coarse-mdif", help="Optional coarse/prior S-parameter MDIF")
    parser.add_argument("--coarse-verification-mdif", help="Optional separate coarse/prior verification MDIF")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--split-var", default="dataset")
    parser.add_argument("--train-values", default="train,training")
    parser.add_argument("--verify-values", default="verify,verification,test,validation")
    parser.add_argument("--parameter-names", help="Comma-separated geometry/process VAR names")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--freq-transform", choices=["log", "linear"], default="log")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss-interval", type=int, default=1, help="Full train/validation loss check interval in epochs")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Console progress update interval in epochs. Use 0 to disable progress updates.",
    )
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print KBNN data/loss diagnostics and write kbnn_training_debug.json "
            "in each training output directory."
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a knowledge-based neural network from fine/coarse S-parameter MDIF data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one KBNN model")
    add_common_train_args(train)
    train.add_argument("--mode", choices=["plain", "residual", "prior-input"], default="residual")
    train.add_argument("--include-coarse-input", action="store_true")
    train.add_argument("--hidden-layers", default="64,64")
    train.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument(
        "--sparam-weights",
        help="S-parameter loss weights. Examples: 'diag=1;offdiag=0.2' or 'S11,S22=1;S12,S21=0.1'. Later rules override earlier ones.",
    )
    train.add_argument("--worst-plots", type=int, default=6)
    train.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    train.set_defaults(func=command_train)

    sweep = sub.add_parser(
        "sweep",
        aliases=["optimize"],
        help="Try multiple KBNN configurations and retrain the best one",
    )
    add_common_train_args(sweep)
    sweep.add_argument("--mode-options", default="residual,prior-input")
    sweep.add_argument("--include-coarse-input-options", default="false,true")
    sweep.add_argument("--hidden-layer-options", default="32;64;64,64")
    sweep.add_argument("--activation-options", default="tanh,relu")
    sweep.add_argument("--learning-rates", default="0.001,0.002,0.005")
    sweep.add_argument("--jobs", type=int, default=1, help="Number of sweep trials to train in parallel")
    sweep.add_argument(
        "--sparam-weights",
        help="S-parameter loss/selection weights. Examples: 'diag=1;offdiag=0.2' or 'all=0.2;S21=1'.",
    )
    sweep.add_argument("--mode", choices=["grid", "random"], default="random")
    sweep.add_argument("--max-trials", type=int, default=24)
    sweep.add_argument(
        "--trial-seed-mode",
        choices=["fixed", "indexed"],
        default="fixed",
        help="Per-trial seed policy. fixed uses --seed for every trial; indexed uses the older --seed + trial_number behavior",
    )
    sweep.add_argument(
        "--selection-metric",
        default="rmse_abs",
        choices=[
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
            "passivity.max_singular_value",
            "passivity.violating_points",
        ],
    )
    sweep.add_argument(
        "--require-passive",
        action="store_true",
        help="Only consider trials with zero passivity-violating frequency points when selecting best_model",
    )
    sweep.add_argument(
        "--max-passivity-violations",
        type=int,
        help="Only consider trials at or below this number of passivity-violating frequency points when selecting best_model",
    )
    sweep.add_argument(
        "--max-passivity-sigma",
        type=float,
        help="Only consider trials whose worst S-matrix singular value is at or below this value when selecting best_model",
    )
    sweep.add_argument("--worst-plots", type=int, default=6)
    sweep.add_argument("--trial-worst-plots", type=int, default=1)
    sweep.add_argument("--keep-trial-models", action="store_true")
    sweep.add_argument(
        "--retrain-best",
        action="store_true",
        help="Retrain the selected best configuration at the end instead of using the best completed trial model",
    )
    sweep.set_defaults(func=command_sweep)

    rerank = sub.add_parser(
        "rerank-sweep",
        help="Re-rank an existing KBNN sweep using saved trial summaries without rerunning all trials",
    )
    rerank.add_argument("--sweep-dir", required=True, help="Existing KBNN sweep/optimize output directory")
    rerank.add_argument(
        "--selection-metric",
        default="rmse_abs",
        choices=[
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
            "passivity.max_singular_value",
            "passivity.violating_points",
        ],
    )
    rerank.add_argument(
        "--require-passive",
        action="store_true",
        help="Only consider trials with zero passivity-violating frequency points",
    )
    rerank.add_argument(
        "--max-passivity-violations",
        type=int,
        help="Only consider trials at or below this number of passivity-violating frequency points",
    )
    rerank.add_argument(
        "--max-passivity-sigma",
        type=float,
        help="Only consider trials whose worst S-matrix singular value is at or below this value",
    )
    rerank.add_argument(
        "--promote-best",
        action="store_true",
        help="Copy the selected trial model to --best-model-dir if trial model files were kept",
    )
    rerank.add_argument(
        "--best-model-dir",
        help="Destination for --promote-best. Default: <sweep-dir>/best_model_reranked",
    )
    rerank.add_argument(
        "--replace-current-best",
        action="store_true",
        help="Overwrite <sweep-dir>/best_model with the selected trial model if available",
    )
    rerank.add_argument("--overwrite", action="store_true", help="Allow --best-model-dir replacement")
    rerank.set_defaults(func=command_rerank_sweep)

    predict = sub.add_parser("predict", help="Predict S-parameters for MDIF parameter blocks")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--mdif", required=True)
    predict.add_argument("--coarse-mdif", help="Coarse/prior MDIF for residual or prior-input models")
    predict.add_argument("--out-mdif", required=True)
    predict.set_defaults(func=command_predict)

    export_ads = sub.add_parser(
        "export-ads-mdif",
        aliases=["export-ads"],
        help="Export a trained KBNN as an ADS-ready parameterized S-parameter MDIF package",
    )
    export_ads.add_argument("--model-dir", required=True, help="Directory containing a trained model.npz and metadata.json")
    export_ads.add_argument("--out-dir", required=True, help="Output directory for the ADS MDIF package")
    export_ads.add_argument(
        "--template-mdif",
        help="MDIF containing the exact parameter/frequency blocks to evaluate; S-data is ignored",
    )
    export_ads.add_argument(
        "--coarse-mdif",
        help="Coarse/prior MDIF evaluated during export for residual or prior-input models",
    )
    export_ads.add_argument(
        "--parameter-grid",
        action="append",
        default=[],
        help="Parameter grid item such as W=0.4mm:0.8mm:9 or W=0.4mm,0.5mm. Repeat once per model parameter.",
    )
    export_ads.add_argument(
        "--freqs",
        help="Frequency grid such as 1GHz:20GHz:401 or 1GHz,2GHz,4GHz. Required with --parameter-grid.",
    )
    export_ads.add_argument("--output-name", default="surrogate_ads.mdif", help="Output MDIF file name")
    export_ads.set_defaults(func=command_export_ads)

    export_ann = sub.add_parser(
        "export-ads-ann",
        help="Create an ADS ANN training/extraction package for a native ADS KBNN model",
    )
    export_ann.add_argument("--mdif", required=True, help="Fine/target S-parameter MDIF")
    export_ann.add_argument("--verification-mdif", help="Optional separate fine/target verification MDIF")
    export_ann.add_argument("--coarse-mdif", help="Optional coarse/prior S-parameter MDIF")
    export_ann.add_argument("--coarse-verification-mdif", help="Optional separate coarse/prior verification MDIF")
    export_ann.add_argument(
        "--model-dir",
        help="Optional trained model directory, or sweep/optimize best_model directory, used for architecture metadata",
    )
    export_ann.add_argument("--out-dir", required=True, help="Output directory for the ADS ANN package")
    export_ann.add_argument("--split-var")
    export_ann.add_argument("--train-values")
    export_ann.add_argument("--verify-values")
    export_ann.add_argument("--parameter-names", help="Comma-separated geometry/process VAR names")
    export_ann.add_argument("--holdout-fraction", type=float, default=0.2)
    export_ann.add_argument("--freq-transform", choices=["log", "linear"])
    export_ann.add_argument("--mode", choices=["plain", "residual", "prior-input"])
    export_ann.add_argument("--include-coarse-input", action="store_true", default=None)
    export_ann.add_argument(
        "--ads-ann-target",
        choices=["native", "fine"],
        default="native",
        help="native preserves the KBNN target; fine trains ADS ANN to output final fine S directly",
    )
    export_ann.add_argument("--hidden-layers")
    export_ann.add_argument("--activation", choices=["tanh", "relu"])
    export_ann.add_argument(
        "--sparam-weights",
        help="Optional S-parameter weights to record in the ADS ANN manifest; defaults to model metadata when --model-dir is supplied",
    )
    export_ann.add_argument("--ads-hidden-layers", type=int, help="Override ADS AnnSetup.num_hidden_layers")
    export_ann.add_argument("--ads-neurons-per-layer", type=int, help="Override ADS AnnSetup.num_neurons_per_layer")
    export_ann.add_argument(
        "--ads-optimizer",
        choices=["quasi-newton", "bayesian-regularization"],
        default="quasi-newton",
    )
    export_ann.add_argument("--ads-iterations", type=int, default=500)
    export_ann.add_argument("--ads-training-stop-tolerance", type=float, default=0.0)
    export_ann.add_argument(
        "--ads-network-training-type",
        choices=["standard", "adjoint", "classification"],
        default="standard",
    )
    export_ann.add_argument(
        "--ads-output-format",
        choices=["all", "verilog-a", "c-code", "equation", "struct-scale"],
        default="all",
    )
    export_ann.add_argument("--output-prefix", default="kbnn_ads_ann")
    export_ann.add_argument("--seed", type=int)
    export_ann.set_defaults(func=command_export_ads_ann)

    export_va = sub.add_parser(
        "export-veriloga",
        help="Export a trained KBNN directly as a Verilog-A N-port using saved model.npz weights",
    )
    export_va.add_argument("--model-dir", required=True, help="Directory containing trained model.npz and metadata.json")
    export_va.add_argument("--out-dir", required=True, help="Output directory for the Verilog-A package")
    export_va.add_argument("--module-name", help="Verilog-A module name. Defaults to the model directory name plus _va")
    export_va.add_argument("--z0", type=float, default=50.0, help="Reference impedance for S-to-Y conversion")
    export_va.add_argument(
        "--frequency-expression",
        default="$freq",
        help="Verilog-A expression for simulator frequency in Hz. Default: $freq",
    )
    export_va.add_argument(
        "--parameter-input-scales",
        help=(
            "Optional NAME=SCALE mappings converting ADS/base-unit instance parameters "
            "to model-training units. Example: W=1um,L=1um or all=1um"
        ),
    )
    export_va.set_defaults(func=command_export_veriloga)

    inspect = sub.add_parser("inspect-mdif", help="Inspect parsed MDIF blocks")
    inspect.add_argument("--mdif", required=True)
    inspect.add_argument("--split-var", default="dataset")
    inspect.set_defaults(func=command_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
