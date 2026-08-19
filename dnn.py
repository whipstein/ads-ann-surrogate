#!/usr/bin/env python3
"""Deep neural-network trainer for parameterized S-parameter MDIF data.

The model is a direct response surrogate:

    geometry/process VARs + frequency features -> DNN -> complex S-parameters

Unlike Neuro-TF, this does not constrain the frequency response with rational
poles. Unlike KBNN, this does not need a coarse circuit prior. It is therefore
simple and flexible, but usually needs more EM samples for robust extrapolation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from cli_options import (
    add_options_json_argument,
    finalize_options_json_update,
    parse_args_with_options_json,
)
from surrogate_common import (  # noqa: E402
    ADS_EXPORT_TEMPLATE_FILENAME,
    EPS,
    DEFAULT_DC_OPEN_RESISTANCE_OHM,
    DEFAULT_DC_OPEN_THRESHOLD_OHM,
    DCConductanceModel,
    MDIFBlock,
    MLP,
    Standardizer,
    add_dc_export_arguments,
    add_dc_fitting_arguments,
    add_dc_port_paths_argument,
    add_adaptive_search_arguments,
    add_debug_argument,
    ads_ann_activation_enum,
    ads_ann_optimizer_enum,
    ads_ann_output_format_enum,
    ads_ann_training_type_enum,
    apply_distinct_dc_response,
    apply_candidate_overrides,
    build_ads_export_blocks,
    build_adaptive_candidate_pool,
    build_training_export_commands,
    cleanup_trial_dir,
    common_sparameter_labels,
    configure_parallel_numeric_threads,
    copy_trial_model,
    csv_number,
    extract_average_dc_resistance,
    frequency_feature_columns,
    frequency_weights_from_blocks,
    infer_complete_sparameter_ports,
    infer_parameter_names,
    load_sweep_rows,
    load_or_write_trial_summary,
    make_training_progress_callback,
    metadata_csv,
    metadata_hidden_layers,
    model_settings_title,
    normalize_name,
    normalize_frequency_weights,
    normalize_sparam_weights,
    output_weights_from_sparam_weights,
    parse_csv_set,
    debug_traceback,
    parse_float_options,
    parse_hidden_layer_options,
    parse_hidden_layers,
    parse_number,
    parse_parameter_scale_spec,
    parse_sparam_weights,
    parse_text_options,
    passivity_summary,
    progress_interval_from_args,
    plot_sweep_diagnostics,
    plot_worst_case_fits,
    plot_worst_case_y_fits,
    positive_frequency_blocks,
    print_cli_error,
    read_mdif,
    read_model_metadata,
    resolve_export_dc_conductance_model,
    resolve_ads_ann_layout,
    rerank_sweep_rows,
    run_sweep_command,
    sparam_sort_key,
    sparam_indices,
    sparam_weight_mean,
    sparameter_real_imag_columns,
    split_blocks,
    sweep_arg_values,
    sweep_trial_seed,
    trial_plot_paths,
    train_dc_conductance_model,
    verification_metrics,
    write_training_verification_artifacts,
    write_ads_ann_package,
    write_ads_export_template,
    write_ads_export_package,
    write_csv,
    write_history,
    write_mdif,
    write_sweep_markdown,
    write_training_markdown,
    update_training_export_commands,
    write_ads_hb_mlp_package,
    write_veriloga_package,
)


VERSION = "0.2.0-rc3"
DNN_SWEEP_RESULT_COLUMNS = ["freq_transform", "hidden_layers", "activation", "learning_rate"]
_MDIF_BLOCK_CACHE: dict[tuple[str, int, int], list[MDIFBlock]] = {}
_SAMPLE_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray]] = {}


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


def split_data(args: argparse.Namespace) -> tuple[list[MDIFBlock], list[MDIFBlock], list[MDIFBlock]]:
    blocks = read_mdif_cached(args.mdif)
    if args.verification_mdif:
        verify_blocks = read_mdif_cached(args.verification_mdif)
        return blocks, verify_blocks, blocks + verify_blocks

    split = split_blocks(
        blocks,
        split_var=args.split_var,
        train_values=parse_csv_set(args.train_values),
        verify_values=parse_csv_set(args.verify_values),
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    return split.train, split.verify, split.all_blocks


def frequency_feature(freq_hz: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log":
        return np.log10(np.maximum(freq_hz, 1.0))[:, None]
    if transform == "linear":
        return freq_hz[:, None]
    if transform == "log-linear":
        return np.column_stack([np.log10(np.maximum(freq_hz, 1.0)), freq_hz])
    raise ValueError(f"Unsupported frequency transform {transform!r}")


def block_features(
    block: MDIFBlock,
    parameter_names: Sequence[str],
    freq_transform: str,
) -> np.ndarray:
    nfreq = len(block.freq_hz)
    params = block_parameter_values(block, parameter_names)
    param_features = np.repeat(params[None, :], nfreq, axis=0)
    return np.column_stack([param_features, frequency_feature(block.freq_hz, freq_transform)])


def block_parameter_values(block: MDIFBlock, parameter_names: Sequence[str]) -> np.ndarray:
    params = []
    for name in parameter_names:
        if name not in block.params:
            raise ValueError(f"Block {block.source_index} is missing parameter {name!r}")
        value = parse_number(block.params[name])
        if value is None:
            raise ValueError(
                f"Block {block.source_index} parameter {name!r} is not numeric: {block.params[name]!r}"
            )
        params.append(float(value))
    return np.asarray(params, dtype=float)


def validate_output_domain(value: str) -> str:
    output_domain = value.lower().strip()
    if output_domain not in {"s", "y"}:
        raise ValueError("Output domain must be 's' or 'y'")
    return output_domain


def block_sparameter_values(block: MDIFBlock, labels: Sequence[str]) -> np.ndarray:
    return np.column_stack([block.sparams[label] for label in labels])


def sparam_matrix_index_arrays(labels: Sequence[str]) -> tuple[int, np.ndarray, np.ndarray]:
    nports = infer_complete_sparameter_ports(labels)
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


def values_to_matrix(values: np.ndarray, labels: Sequence[str], nports: int) -> np.ndarray:
    matrix = np.zeros((values.shape[0], nports, nports), dtype=complex)
    _, rows, cols = sparam_matrix_index_arrays(labels)
    matrix[:, rows, cols] = values
    return matrix


def matrix_to_values(matrix: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    _, rows, cols = sparam_matrix_index_arrays(labels)
    return matrix[:, rows, cols]


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


def y_values_to_s_values(values: np.ndarray, labels: Sequence[str], z0: float) -> np.ndarray:
    nports, rows, cols = sparam_matrix_index_arrays(labels)
    identity = np.eye(nports, dtype=complex)
    ymatrix = np.zeros((values.shape[0], nports, nports), dtype=complex)
    ymatrix[:, rows, cols] = values
    normalized_y = z0 * ymatrix
    lhs = identity[None, :, :] + normalized_y
    rhs = identity[None, :, :] - normalized_y
    try:
        smatrix = np.swapaxes(
            np.linalg.solve(np.swapaxes(lhs, -1, -2), np.swapaxes(rhs, -1, -2)),
            -1,
            -2,
        )
    except np.linalg.LinAlgError:
        smatrix = rhs @ np.linalg.pinv(lhs)
    return smatrix[:, rows, cols]


def complex_values_to_columns(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values.real, values.imag], axis=1)


def output_column_names(labels: Sequence[str]) -> list[str]:
    return [f"{label}.real" for label in labels] + [f"{label}.imag" for label in labels]


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
        for name, is_floored in zip(output_column_names(labels), ~varying)
        if bool(is_floored)
    ]
    return scaler, floored_columns, floor


def physical_response_output_weights(
    labels: Sequence[str],
    sparam_weights: dict[str, float],
    output_scaler: Standardizer,
) -> np.ndarray:
    """Preserve requested raw-response weights after output standardization."""

    if output_scaler.std is None:
        raise ValueError("Output standardizer must be fitted before weighting")
    requested = output_weights_from_sparam_weights(labels, sparam_weights)
    weights = requested * np.asarray(output_scaler.std, dtype=float) ** 2
    mean = float(np.mean(weights))
    if not math.isfinite(mean) or mean <= EPS:
        raise ValueError("Physical response output weights must have a positive mean")
    return weights / mean


def dnn_reciprocity_summary(
    blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
) -> dict[str, float | int | bool | None]:
    """Measure relative Sij/Sji disagreement for a complete S-matrix."""

    try:
        nports = infer_complete_sparameter_ports(labels)
    except ValueError:
        return {
            "nports": None,
            "comparable": False,
            "max_abs_error": None,
            "relative_error": None,
        }
    max_error = 0.0
    max_reference = 0.0
    for block in blocks:
        for row in range(1, nports + 1):
            for col in range(row + 1, nports + 1):
                forward = np.asarray(block.sparams[f"S{row}{col}"], dtype=complex)
                reverse = np.asarray(block.sparams[f"S{col}{row}"], dtype=complex)
                max_error = max(max_error, float(np.max(np.abs(forward - reverse))))
                max_reference = max(
                    max_reference,
                    float(np.max(np.abs(forward))),
                    float(np.max(np.abs(reverse))),
                )
    return {
        "nports": int(nports),
        "comparable": True,
        "max_abs_error": max_error,
        "relative_error": max_error / max(max_reference, EPS),
    }


def dnn_reciprocity_projection(
    labels: Sequence[str],
) -> np.ndarray:
    """Return a raw real/imag output projection that ties reciprocal entries."""

    nports = infer_complete_sparameter_ports(labels)
    output_count = 2 * len(labels)
    projection = np.eye(output_count, dtype=float)
    indices = {label: index for index, label in enumerate(labels)}
    for row in range(1, nports + 1):
        for col in range(row + 1, nports + 1):
            first = indices[f"S{row}{col}"]
            second = indices[f"S{col}{row}"]
            for offset in (0, len(labels)):
                a = first + offset
                b = second + offset
                projection[a, a] = 0.5
                projection[b, a] = 0.5
                projection[a, b] = 0.5
                projection[b, b] = 0.5
    return projection


def fold_raw_output_projection(
    mlp: MLP,
    output_scaler: Standardizer,
    projection: np.ndarray,
) -> None:
    """Fold a raw-domain linear projection into the scaled MLP output layer."""

    if output_scaler.mean is None or output_scaler.std is None:
        raise ValueError("Output standardizer must be fitted before projection")
    matrix = np.asarray(projection, dtype=float)
    output_count = mlp.weights[-1].shape[1]
    if matrix.shape != (output_count, output_count):
        raise ValueError(
            f"Expected a {output_count}x{output_count} output projection, got "
            f"{matrix.shape}"
        )
    std = np.asarray(output_scaler.std, dtype=float)
    mean = np.asarray(output_scaler.mean, dtype=float)
    scaled_projection = std[:, None] * matrix / std[None, :]
    scaled_offset = (mean @ matrix - mean) / std
    mlp.weights[-1] = mlp.weights[-1] @ scaled_projection
    mlp.biases[-1] = mlp.biases[-1] @ scaled_projection + scaled_offset


def make_s_passivity_loss_gradient(
    output_scaler: Standardizer,
    labels: Sequence[str],
    target_sigma: float,
    penalty: float,
):
    """Build a differentiable sampled S-matrix passivity penalty callback."""

    if output_scaler.mean is None or output_scaler.std is None:
        raise ValueError("Output standardizer must be fitted before passivity loss")
    nports, rows, cols = sparam_matrix_index_arrays(labels)
    mean = np.asarray(output_scaler.mean, dtype=float)
    std = np.asarray(output_scaler.std, dtype=float)
    penalty_value = float(penalty)
    target_value = float(target_sigma)

    def callback(
        predicted_scaled: np.ndarray,
        _truth_scaled: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        predicted = np.asarray(predicted_scaled, dtype=float)
        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        raw = predicted * std[None, :] + mean[None, :]
        values = columns_to_complex(raw)
        matrices = np.zeros((len(values), nports, nports), dtype=complex)
        matrices[:, rows, cols] = values
        left, singular_values, right_h = np.linalg.svd(
            matrices,
            full_matrices=False,
        )
        sigma = singular_values[:, 0]
        excess = np.maximum(0.0, sigma - target_value)
        weighted_excess_squared = weights * excess**2
        # The mean term shapes every violating sample.  The RMS-of-squared-
        # excess term emphasizes narrow singular-value spikes without the
        # noisy, discontinuous batch-to-batch behavior of a hard maximum.
        excess_rms = float(np.sqrt(np.mean(weighted_excess_squared**2)))
        loss = penalty_value * float(
            np.mean(weighted_excess_squared) + excess_rms
        )
        gradient = np.zeros_like(predicted)
        active = excess > 0.0
        if not np.any(active) or penalty_value <= 0.0:
            return loss, gradient
        top_gradient = np.einsum(
            "bi,bj->bij",
            left[:, :, 0],
            right_h[:, 0, :],
        )
        factor = 2.0 * penalty_value * weights * excess * (
            1.0 / len(predicted)
            + weighted_excess_squared
            / (len(predicted) * max(excess_rms, np.finfo(float).tiny))
        )
        top_gradient *= factor[:, None, None]
        value_gradient = top_gradient[:, rows, cols]
        raw_gradient = np.concatenate(
            [value_gradient.real, value_gradient.imag],
            axis=1,
        )
        gradient = raw_gradient * std[None, :]
        return loss, gradient

    return callback


def direct_y_conditioning_summary(
    blocks: Sequence[MDIFBlock],
    labels: Sequence[str],
) -> dict[str, object]:
    """Assess conditioning of the S-to-Y matrix inverse over RF source rows."""

    nports = infer_complete_sparameter_ports(labels)
    identity = np.eye(nports, dtype=complex)
    maximum_condition = 0.0
    minimum_singular = float("inf")
    worst: dict[str, object] | None = None
    for block in blocks:
        values = block_sparameter_values(block, labels)
        matrices = values_to_matrix(values, labels, nports)
        singular_values = np.linalg.svd(
            identity[None, :, :] + matrices,
            compute_uv=False,
        )
        conditions = singular_values[:, 0] / np.maximum(
            singular_values[:, -1],
            np.finfo(float).tiny,
        )
        index = int(np.argmax(conditions))
        condition = float(conditions[index])
        minimum_singular = min(
            minimum_singular,
            float(np.min(singular_values[:, -1])),
        )
        if condition > maximum_condition:
            maximum_condition = condition
            worst = {
                "source_block": int(block.source_index) + 1,
                "frequency_hz": float(block.freq_hz[index]),
                "parameters": dict(block.params),
            }
    return {
        "max_condition_number_i_plus_s": maximum_condition,
        "min_singular_value_i_plus_s": minimum_singular,
        "worst_case": worst,
    }


def block_targets(
    block: MDIFBlock,
    labels: Sequence[str],
    output_domain: str,
    target_z0: float,
) -> np.ndarray:
    output_domain = validate_output_domain(output_domain)
    values = np.column_stack([block.sparams[label] for label in labels])
    if output_domain == "y":
        values = s_values_to_y_values(values, labels, target_z0)
    return complex_values_to_columns(values)


def columns_to_complex(values: np.ndarray) -> np.ndarray:
    half = values.shape[1] // 2
    return values[:, :half] + 1j * values[:, half:]


def sample_cache_key(
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    freq_transform: str,
    output_domain: str,
    target_z0: float,
) -> tuple[object, ...]:
    return (
        tuple(id(block) for block in blocks),
        tuple(parameter_names),
        tuple(labels),
        freq_transform,
        output_domain,
        float(target_z0),
    )


def make_feature_target_samples(
    blocks: Sequence[MDIFBlock],
    parameter_names: Sequence[str],
    labels: Sequence[str],
    freq_transform: str,
    output_domain: str,
    target_z0: float,
) -> tuple[np.ndarray, np.ndarray]:
    output_domain = validate_output_domain(output_domain)
    key = sample_cache_key(blocks, parameter_names, labels, freq_transform, output_domain, target_z0)
    cached = _SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    total_rows = sum(len(block.freq_hz) for block in blocks)
    n_param_features = len(parameter_names)
    n_freq_features = len(frequency_feature_columns(freq_transform))
    features = np.empty((total_rows, n_param_features + n_freq_features), dtype=float)
    values = np.empty((total_rows, len(labels)), dtype=complex)
    offset = 0
    for block in blocks:
        nfreq = len(block.freq_hz)
        end = offset + nfreq
        if n_param_features:
            features[offset:end, :n_param_features] = block_parameter_values(
                block,
                parameter_names,
            )
        features[offset:end, n_param_features:] = frequency_feature(block.freq_hz, freq_transform)
        for label_idx, label in enumerate(labels):
            values[offset:end, label_idx] = block.sparams[label]
        offset = end
    if output_domain == "y":
        values = s_values_to_y_values(values, labels, target_z0)
    result = features, complex_values_to_columns(values)
    _SAMPLE_CACHE[key] = result
    return result


class DNN:
    def __init__(
        self,
        mlp: MLP,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        parameter_names: list[str],
        sparam_labels: list[str],
        freq_transform: str,
        output_domain: str = "s",
        target_z0: float = 50.0,
        dc_equivalent_resistance_ohm: float | None = None,
        dc_resistance_source_kind: str | None = None,
        dc_port_resistances_ohm: dict[str, float] | None = None,
        dc_model: DCConductanceModel | None = None,
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = parameter_names
        self.sparam_labels = sparam_labels
        self.freq_transform = freq_transform
        self.output_domain = validate_output_domain(output_domain)
        self.target_z0 = float(target_z0)
        self.dc_equivalent_resistance_ohm = (
            None
            if dc_equivalent_resistance_ohm is None
            else float(dc_equivalent_resistance_ohm)
        )
        self.dc_resistance_source_kind = dc_resistance_source_kind
        self.dc_port_resistances_ohm = (
            None
            if dc_port_resistances_ohm is None
            else {
                str(path): float(resistance)
                for path, resistance in dc_port_resistances_ohm.items()
            }
        )
        self.dc_model = dc_model

    def predict_blocks(self, blocks: Sequence[MDIFBlock]) -> list[MDIFBlock]:
        predicted = []
        for block in blocks:
            x = block_features(block, self.parameter_names, self.freq_transform)
            y_scaled = self.mlp.predict(self.x_scaler.transform(x))
            y_columns = self.y_scaler.inverse_transform(y_scaled)
            values = columns_to_complex(y_columns)
            if self.output_domain == "y":
                values = y_values_to_s_values(values, self.sparam_labels, self.target_z0)
            dc_mask = np.asarray(block.freq_hz, dtype=float) == 0.0
            if self.dc_model is not None and np.any(dc_mask):
                values[dc_mask, :] = self.dc_model.predict_block_s_values(block)[None, :]
            else:
                values = apply_distinct_dc_response(
                    values,
                    block.freq_hz,
                    self.sparam_labels,
                    self.dc_equivalent_resistance_ohm,
                    self.dc_resistance_source_kind,
                    self.dc_port_resistances_ohm,
                    z0=self.target_z0,
                )
            sparams = {label: values[:, idx] for idx, label in enumerate(self.sparam_labels)}
            predicted.append(
                MDIFBlock(
                    params=dict(block.params),
                    freq_hz=block.freq_hz.copy(),
                    sparams=sparams,
                    source_index=block.source_index,
                )
            )
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
        if self.dc_model is not None:
            self.dc_model.save(out_dir)
        combined_metadata = {
            "version": VERSION,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation,
            "freq_transform": self.freq_transform,
            "output_domain": self.output_domain,
            "target_z0": self.target_z0,
            "dc_equivalent_resistance_ohm": self.dc_equivalent_resistance_ohm,
            "dc_resistance_source_kind": self.dc_resistance_source_kind,
            "dc_port_resistances_ohm": self.dc_port_resistances_ohm,
            **metadata,
        }
        (out_dir / "metadata.json").write_text(json.dumps(combined_metadata, indent=2))

    @staticmethod
    def load(model_dir: Path) -> "DNN":
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
        return DNN(
            mlp=mlp,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            parameter_names=list(metadata["parameter_names"]),
            sparam_labels=list(metadata["sparam_labels"]),
            freq_transform=metadata["freq_transform"],
            output_domain=metadata.get("output_domain", "s"),
            target_z0=float(metadata.get("target_z0", 50.0)),
            dc_equivalent_resistance_ohm=metadata.get("dc_equivalent_resistance_ohm"),
            dc_resistance_source_kind=metadata.get("dc_resistance_source_kind"),
            dc_port_resistances_ohm=metadata.get("dc_port_resistances_ohm"),
            dc_model=DCConductanceModel.load_optional(model_dir),
        )


def train_model(args: argparse.Namespace) -> tuple[DNN, list[MDIFBlock], list[str], list[str], list[dict[str, float]], list[dict[str, float]], dict[str, object]]:
    train_blocks, verify_blocks, all_blocks = split_data(args)
    if not train_blocks:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(all_blocks, requested=args.parameter_names, split_var=args.split_var)
    labels = common_sparameter_labels(all_blocks)
    output_domain = validate_output_domain(getattr(args, "output_domain", "s"))
    target_z0 = float(getattr(args, "target_z0", 50.0))
    if not math.isfinite(target_z0) or target_z0 <= 0.0:
        raise ValueError("--target-z0 must be positive and finite")
    if output_domain == "y":
        infer_complete_sparameter_ports(labels)
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    progress_interval = progress_interval_from_args(args)
    fit_train_blocks = positive_frequency_blocks(train_blocks)
    fit_verify_blocks = positive_frequency_blocks(verify_blocks) if verify_blocks else []
    source_rf_passivity = passivity_summary(fit_train_blocks, labels)
    source_rf_reciprocity = dnn_reciprocity_summary(fit_train_blocks, labels)

    passivity_mode = str(getattr(args, "passivity_mode", "auto"))
    passivity_margin = float(getattr(args, "passivity_margin", 1e-3))
    passivity_penalty = float(getattr(args, "passivity_penalty", 10.0))
    if passivity_mode not in {"auto", "enforce", "off"}:
        raise ValueError(f"Unsupported passivity mode {passivity_mode!r}")
    if (
        not math.isfinite(passivity_margin)
        or passivity_margin < 0.0
        or passivity_margin >= 1.0
    ):
        raise ValueError("--passivity-margin must be finite and in [0, 1)")
    if not math.isfinite(passivity_penalty) or passivity_penalty < 0.0:
        raise ValueError("--passivity-penalty must be finite and non-negative")
    source_passivity_available = source_rf_passivity["nports"] is not None
    source_is_passive = (
        source_passivity_available
        and source_rf_passivity["violating_points"] == 0
    )
    passivity_requested = passivity_mode == "enforce" or (
        passivity_mode == "auto" and source_is_passive
    )
    if passivity_mode == "enforce" and not source_passivity_available:
        raise ValueError(
            "--passivity-mode enforce requires a complete S-parameter matrix"
        )
    if passivity_mode == "enforce" and output_domain != "s":
        raise ValueError(
            "--passivity-mode enforce is only available for --output-domain s; "
            "direct-Y output cannot fold an exact S-domain safeguard into the "
            "saved linear output layer"
        )
    passivity_enforced = passivity_requested and output_domain == "s"
    passivity_unavailable_reason = (
        "Direct-Y output does not support the S-domain passivity penalty and "
        "folded contraction; verification passivity is still reported"
        if passivity_requested and output_domain != "s"
        else None
    )
    passivity_target_sigma = 1.0 - passivity_margin

    reciprocity_mode = str(getattr(args, "reciprocity_mode", "enforce"))
    reciprocity_tolerance = float(getattr(args, "reciprocity_tolerance", 1e-6))
    if reciprocity_mode not in {"auto", "enforce", "off"}:
        raise ValueError(f"Unsupported reciprocity mode {reciprocity_mode!r}")
    if not math.isfinite(reciprocity_tolerance) or reciprocity_tolerance < 0.0:
        raise ValueError("--reciprocity-tolerance must be finite and non-negative")
    reciprocity_comparable = bool(source_rf_reciprocity["comparable"])
    source_reciprocity_error = source_rf_reciprocity["relative_error"]
    reciprocity_enforced = reciprocity_mode == "enforce" or (
        reciprocity_mode == "auto"
        and reciprocity_comparable
        and source_reciprocity_error is not None
        and float(source_reciprocity_error) <= reciprocity_tolerance
    )
    if reciprocity_mode == "enforce" and not reciprocity_comparable:
        raise ValueError(
            "--reciprocity-mode enforce requires a complete S-parameter matrix"
        )

    max_y_condition = float(getattr(args, "max_y_condition", 1e10))
    if not math.isfinite(max_y_condition) or max_y_condition <= 1.0:
        raise ValueError("--max-y-condition must be finite and greater than 1")
    y_conditioning = None
    if output_domain == "y":
        y_conditioning = direct_y_conditioning_summary(fit_train_blocks, labels)
        observed_condition = float(
            y_conditioning["max_condition_number_i_plus_s"]
        )
        if observed_condition > max_y_condition:
            worst = y_conditioning.get("worst_case")
            raise ValueError(
                "Direct-Y fitting is numerically unsafe because I + S is nearly "
                f"singular: maximum condition number {observed_condition:.6g} "
                f"exceeds --max-y-condition {max_y_condition:.6g}; worst case "
                f"{worst}. Use --output-domain s or raise the limit only after "
                "checking the resulting Y-target dynamic range."
            )
    dc_model, dc_history, dc_metadata = train_dc_conductance_model(
        train_blocks,
        verify_blocks,
        parameter_names,
        labels,
        hidden_layers=hidden_layers,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_interval=progress_interval,
        progress_label=f"{getattr(args, 'progress_label', 'DNN fit')} DC",
        z0=target_z0,
        port_paths=getattr(args, "dc_port_paths", None),
        open_threshold_ohm=float(
            getattr(args, "dc_open_threshold", DEFAULT_DC_OPEN_THRESHOLD_OHM)
        ),
        open_resistance_ohm=float(
            getattr(args, "dc_open_resistance", DEFAULT_DC_OPEN_RESISTANCE_OHM)
        ),
    )
    x_train, y_train = make_feature_target_samples(
        fit_train_blocks,
        parameter_names,
        labels,
        args.freq_transform,
        output_domain,
        target_z0,
    )
    if fit_verify_blocks:
        x_verify, y_verify = make_feature_target_samples(
            fit_verify_blocks,
            parameter_names,
            labels,
            args.freq_transform,
            output_domain,
            target_z0,
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

    layer_sizes = [x_train.shape[1], *hidden_layers, y_train.shape[1]]
    mlp = MLP(layer_sizes, activation=args.activation, seed=args.seed)
    sparam_weights = parse_sparam_weights(labels, getattr(args, "sparam_weights", None))
    normalized_sparam_weights = normalize_sparam_weights(labels, sparam_weights)
    output_weights = physical_response_output_weights(
        labels,
        sparam_weights,
        y_scaler,
    )
    frequency_weight_spec = getattr(args, "frequency_weights", None)
    raw_frequency_weights = frequency_weights_from_blocks(
        fit_train_blocks,
        frequency_weight_spec,
    )
    normalized_frequency_weights, frequency_weight_mean = (
        normalize_frequency_weights(raw_frequency_weights)
    )
    if fit_verify_blocks:
        raw_verify_frequency_weights = frequency_weights_from_blocks(
            fit_verify_blocks,
            frequency_weight_spec,
            require_all_rules_match=False,
        )
        normalized_verify_frequency_weights, _ = normalize_frequency_weights(
            raw_verify_frequency_weights,
            mean=frequency_weight_mean,
        )
    else:
        normalized_verify_frequency_weights = None
    passivity_loss_gradient = (
        make_s_passivity_loss_gradient(
            y_scaler,
            labels,
            passivity_target_sigma,
            passivity_penalty,
        )
        if passivity_enforced and passivity_penalty > 0.0
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
        seed=args.seed + 29,
        output_weights=output_weights,
        sample_weights=normalized_frequency_weights,
        val_sample_weights=normalized_verify_frequency_weights,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_callback=make_training_progress_callback(
            getattr(args, "progress_label", "DNN fit"),
            args.epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
        extra_loss_gradient=passivity_loss_gradient,
    )
    if reciprocity_enforced:
        fold_raw_output_projection(
            mlp,
            y_scaler,
            dnn_reciprocity_projection(labels),
        )
    model = DNN(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=labels,
        freq_transform=args.freq_transform,
        output_domain=output_domain,
        target_z0=target_z0,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=str(dc_metadata["dc_resistance_source_kind"]),
        dc_port_resistances_ohm=dict(dc_metadata["dc_port_resistances_ohm"]),
        dc_model=dc_model,
    )
    predicted_train_before_scale = model.predict_blocks(fit_train_blocks)
    predicted_train_passivity_before_scale = passivity_summary(
        predicted_train_before_scale,
        labels,
    )
    rf_response_scale = 1.0
    if passivity_enforced:
        predicted_sigma = predicted_train_passivity_before_scale[
            "max_singular_value"
        ]
        if predicted_sigma is None or not math.isfinite(float(predicted_sigma)):
            raise ValueError("Could not assess the fitted DNN response for passivity")
        if float(predicted_sigma) > passivity_target_sigma:
            rf_response_scale = passivity_target_sigma / float(predicted_sigma)
            fold_raw_output_projection(
                mlp,
                y_scaler,
                rf_response_scale * np.eye(2 * len(labels), dtype=float),
            )
    predicted_train_passivity_after_scale = passivity_summary(
        model.predict_blocks(fit_train_blocks),
        labels,
    )
    max_abs_training_target = float(np.max(np.abs(y_train)))
    metadata = {
        "training_blocks": len(train_blocks),
        "verification_blocks": len(verify_blocks),
        "training_samples": int(x_train.shape[0]),
        "verification_samples": int(x_verify.shape[0]) if x_verify is not None else 0,
        "output_domain": output_domain,
        "target_z0": target_z0,
        "split_var": args.split_var,
        "train_values": sorted(parse_csv_set(args.train_values)),
        "verify_values": sorted(parse_csv_set(args.verify_values)),
        "sparam_weights": sparam_weights,
        "normalized_sparam_weights": normalized_sparam_weights,
        "scaled_output_loss_weights": {
            name: float(weight)
            for name, weight in zip(output_column_names(labels), output_weights)
        },
        "sparam_weight_mean": sparam_weight_mean(labels, sparam_weights),
        "sparam_weight_normalization": "Requested response weights are combined with squared output standard deviations and renormalized before training, so standardized-coordinate MSE is proportional to the requested physical response-domain MSE.",
        "response_loss_domain": f"physical_{output_domain}_components",
        "frequency_weights": frequency_weight_spec,
        "frequency_weight_mean": frequency_weight_mean,
        "frequency_weight_min": float(np.min(raw_frequency_weights)),
        "frequency_weight_max": float(np.max(raw_frequency_weights)),
        "frequency_weight_normalization": "Raw frequency weights are divided by their mean over fitted training samples, so the average normalized weight is 1.0.",
        "output_scaler_floor": output_std_floor,
        "floored_output_columns": floored_output_columns,
        "max_abs_training_target": max_abs_training_target,
        "direct_y_conditioning": y_conditioning,
        "max_y_condition": max_y_condition,
        "reciprocity_mode": reciprocity_mode,
        "reciprocity_tolerance": reciprocity_tolerance,
        "reciprocity_enforced": reciprocity_enforced,
        "source_rf_reciprocity": source_rf_reciprocity,
        "passivity_mode": passivity_mode,
        "passivity_margin": passivity_margin,
        "passivity_penalty": passivity_penalty,
        "passivity_target_sigma": passivity_target_sigma,
        "passivity_requested": passivity_requested,
        "passivity_enforced": passivity_enforced,
        "passivity_unavailable_reason": passivity_unavailable_reason,
        "source_rf_passivity": source_rf_passivity,
        "predicted_train_passivity_before_scale": predicted_train_passivity_before_scale,
        "rf_response_scale": rf_response_scale,
        "predicted_train_passivity_after_scale": predicted_train_passivity_after_scale,
        "passivity_assessment_scope": "positive-frequency training blocks only",
        **dc_metadata,
        "dc_model_history_rows": len(dc_history),
    }
    return model, verify_blocks, parameter_names, labels, history, dc_history, metadata


def dnn_export_commands(
    model_dir: Path,
    template_mdif: str | Path | None = None,
    *,
    dc_mdif: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Build runnable export commands for a fitted DNN report."""

    return build_training_export_commands(
        Path(__file__),
        model_dir,
        template_mdif,
        dc_mdif=dc_mdif,
        include_veriloga=True,
        model_type="dnn",
    )


def resolve_dnn_export_dc(
    model: DNN,
    source_metadata: dict[str, object],
    args: argparse.Namespace,
    z0: float,
) -> tuple[DCConductanceModel | None, dict[str, object]]:
    return resolve_export_dc_conductance_model(
        model.dc_model,
        source_metadata,
        model.parameter_names,
        model.sparam_labels,
        dc_mdif=args.dc_mdif,
        z0=z0,
        port_paths=args.dc_port_paths,
        open_threshold_ohm=args.dc_open_threshold,
        open_resistance_ohm=args.dc_open_resistance,
        activation=(
            model.dc_model.mlp.activation
            if model.dc_model is not None
            else model.mlp.activation
        ),
        hidden_layers=(
            model.dc_model.mlp.layer_sizes[1:-1]
            if model.dc_model is not None
            else model.mlp.layer_sizes[1:-1]
        ),
    )


def command_train(args: argparse.Namespace) -> int:
    model, verify_blocks, parameter_names, labels, history, dc_history, metadata = train_model(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, _, all_blocks = split_data(args)
    template_path = out_dir / ADS_EXPORT_TEMPLATE_FILENAME
    metadata["ads_export_template"] = write_ads_export_template(
        template_path,
        all_blocks,
        parameter_names,
        labels,
    )
    model.save(out_dir, metadata=metadata)
    write_history(
        out_dir / "dc_training_history.csv",
        dc_history,
        plot_title="Separate exact-DC conductance model performance vs epoch",
    )
    training_config = {
        "training_blocks": metadata["training_blocks"],
        "verification_blocks": metadata["verification_blocks"],
        "training_samples": metadata["training_samples"],
        "verification_samples": metadata["verification_samples"],
        "parameters": parameter_names,
        "sparameters": labels,
        "freq_transform": model.freq_transform,
        "output_domain": model.output_domain,
        "target_z0": model.target_z0,
        "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
        "dc_resistance_source_kind": metadata["dc_resistance_source_kind"],
        "dc_port_paths": metadata["dc_port_paths"],
        "dc_matrix_entries": metadata.get("dc_matrix_entries", []),
        "dc_sparameter_entries": metadata.get("dc_sparameter_entries", []),
        "dc_port_resistances_ohm": metadata["dc_port_resistances_ohm"],
        "dc_resistance_pair_means_ohm": metadata["dc_resistance_pair_means_ohm"],
        "dc_resistance_extraction": metadata["dc_resistance_extraction"],
        "dc_model_kind": metadata["dc_model_kind"],
        "dc_model_representation": metadata.get("dc_model_representation"),
        "dc_model_layer_sizes": metadata["dc_model_layer_sizes"],
        "dc_model_train_log_rmse": metadata.get("dc_model_train_log_rmse"),
        "dc_model_train_y_rmse_siemens": metadata.get(
            "dc_model_train_y_rmse_siemens"
        ),
        "dc_model_train_s_component_rmse": metadata.get(
            "dc_model_train_component_rmse"
        ),
        "dc_model_train_s_rmse": metadata["dc_model_train_s_rmse"],
        "dc_model_train_s_max_abs_error": metadata["dc_model_train_s_max_abs_error"],
        "dc_topology_s_rmse": metadata["dc_topology_s_rmse"],
        "dc_topology_s_max_abs_error": metadata["dc_topology_s_max_abs_error"],
        "dc_resistance_filtering": {
            "raw_mean_ohm": metadata["dc_equivalent_resistance_raw_mean_ohm"],
            "mean_conductance_siemens": metadata["dc_mean_conductance_siemens"],
            "open_resistance_samples": metadata[
                "dc_open_resistance_sample_count"
            ],
            "dc_rows": metadata["dc_row_count"],
            "ignored_nonpassive": metadata["dc_ignored_nonpassive_count"],
            "ignored_nonfinite": metadata["dc_ignored_nonfinite_count"],
            "ignored_invalid_resistance": metadata[
                "dc_ignored_invalid_resistance_count"
            ],
            "open_threshold_ohm": metadata["dc_open_threshold_ohm"],
            "open_resistance_ohm": metadata["dc_open_resistance_ohm"],
            "open_circuit_applied": metadata["dc_open_circuit_applied"],
        },
        "dc_is_separate_from_fitted_response": True,
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
        "scaled_output_loss_weights": metadata["scaled_output_loss_weights"],
        "response_loss_domain": metadata["response_loss_domain"],
        "frequency_weights": metadata["frequency_weights"],
        "frequency_weight_mean": metadata["frequency_weight_mean"],
        "output_scaler_floor": metadata["output_scaler_floor"],
        "floored_output_columns": metadata["floored_output_columns"],
        "max_abs_training_target": metadata["max_abs_training_target"],
        "direct_y_conditioning": metadata["direct_y_conditioning"],
        "max_y_condition": metadata["max_y_condition"],
        "reciprocity_mode": metadata["reciprocity_mode"],
        "reciprocity_tolerance": metadata["reciprocity_tolerance"],
        "reciprocity_enforced": metadata["reciprocity_enforced"],
        "source_rf_reciprocity": metadata["source_rf_reciprocity"],
        "passivity_mode": metadata["passivity_mode"],
        "passivity_margin": metadata["passivity_margin"],
        "passivity_penalty": metadata["passivity_penalty"],
        "passivity_target_sigma": metadata["passivity_target_sigma"],
        "passivity_requested": metadata["passivity_requested"],
        "passivity_enforced": metadata["passivity_enforced"],
        "passivity_unavailable_reason": metadata[
            "passivity_unavailable_reason"
        ],
        "source_rf_passivity": metadata["source_rf_passivity"],
        "rf_response_scale": metadata["rf_response_scale"],
        "predicted_train_passivity_before_scale": metadata[
            "predicted_train_passivity_before_scale"
        ],
        "predicted_train_passivity_after_scale": metadata[
            "predicted_train_passivity_after_scale"
        ],
    }
    plot_context = model_settings_title(
        "DNN",
        training_config,
        getattr(args, "progress_label", "DNN fit"),
    )
    write_history(
        out_dir / "training_history.csv",
        history,
        plot_title=f"Model performance vs epoch | {plot_context}",
    )

    if verify_blocks:
        pred_blocks = model.predict_blocks(verify_blocks)
        rf_verify_blocks = positive_frequency_blocks(
            verify_blocks,
            purpose="RF verification",
        )
        rf_pred_blocks = positive_frequency_blocks(
            pred_blocks,
            purpose="RF verification",
        )
        summary = write_training_verification_artifacts(
            out_dir,
            rf_verify_blocks,
            rf_pred_blocks,
            labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            sparam_weights=parse_sparam_weights(labels, getattr(args, "sparam_weights", None)),
            frequency_weights=getattr(args, "frequency_weights", None),
            y_z0=model.target_z0,
            title_context=plot_context,
        )
        # Keep the complete deployment prediction, including the independently
        # evaluated DC point, while RF metrics and sweep ranking remain DC-free.
        write_mdif(out_dir / "predicted_verification.mdif", pred_blocks, labels)
        summary.update(
            {
                "verification_frequency_scope": "positive_frequency_rf_only",
                "dc_model_train_s_rmse": metadata["dc_model_train_s_rmse"],
                "dc_model_train_s_max_abs_error": metadata[
                    "dc_model_train_s_max_abs_error"
                ],
                "dc_model_verify_s_rmse": metadata.get("dc_model_verify_s_rmse"),
                "dc_model_verify_s_max_abs_error": metadata.get(
                    "dc_model_verify_s_max_abs_error"
                ),
                "response_loss_domain": metadata["response_loss_domain"],
                "reciprocity_enforced": metadata["reciprocity_enforced"],
                "source_rf_reciprocity": metadata["source_rf_reciprocity"],
                "passivity_enforced": metadata["passivity_enforced"],
                "passivity_unavailable_reason": metadata[
                    "passivity_unavailable_reason"
                ],
                "source_rf_passivity": metadata["source_rf_passivity"],
                "rf_response_scale": metadata["rf_response_scale"],
                "predicted_train_passivity_before_scale": metadata[
                    "predicted_train_passivity_before_scale"
                ],
                "predicted_train_passivity_after_scale": metadata[
                    "predicted_train_passivity_after_scale"
                ],
            }
        )
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    else:
        summary = {
            "warning": "No verification blocks were available",
            "response_loss_domain": metadata["response_loss_domain"],
            "reciprocity_enforced": metadata["reciprocity_enforced"],
            "source_rf_reciprocity": metadata["source_rf_reciprocity"],
            "passivity_enforced": metadata["passivity_enforced"],
            "passivity_unavailable_reason": metadata[
                "passivity_unavailable_reason"
            ],
            "source_rf_passivity": metadata["source_rf_passivity"],
            "rf_response_scale": metadata["rf_response_scale"],
            "predicted_train_passivity_before_scale": metadata[
                "predicted_train_passivity_before_scale"
            ],
            "predicted_train_passivity_after_scale": metadata[
                "predicted_train_passivity_after_scale"
            ],
        }
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    export_commands = dnn_export_commands(out_dir, dc_mdif=args.mdif)
    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="DNN",
        config=training_config,
        summary=summary,
        history=history,
        export_commands=export_commands,
    )

    if not getattr(args, "quiet", False):
        print(json.dumps({
            "out_dir": str(out_dir),
            "training_summary": str(out_dir / "training_summary.md"),
            "ads_export_template": str(template_path),
            "parameters": parameter_names,
            "sparameters": labels,
            "freq_transform": model.freq_transform,
            "output_domain": model.output_domain,
            "target_z0": model.target_z0,
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_resistance_source_kind": metadata["dc_resistance_source_kind"],
            "dc_model_kind": metadata["dc_model_kind"],
            "dc_model_train_s_rmse": metadata["dc_model_train_s_rmse"],
            "dc_port_paths": metadata["dc_port_paths"],
            "dc_matrix_entries": metadata.get("dc_matrix_entries", []),
            "dc_sparameter_entries": metadata.get("dc_sparameter_entries", []),
            "dc_port_resistances_ohm": metadata["dc_port_resistances_ohm"],
            "dc_resistance_pair_means_ohm": metadata["dc_resistance_pair_means_ohm"],
            "layer_sizes": model.mlp.layer_sizes,
            "sparam_weights": metadata["sparam_weights"],
            "normalized_sparam_weights": metadata["normalized_sparam_weights"],
            "scaled_output_loss_weights": metadata[
                "scaled_output_loss_weights"
            ],
            "sparam_weight_mean": metadata["sparam_weight_mean"],
            "frequency_weights": metadata["frequency_weights"],
            "frequency_weight_mean": metadata["frequency_weight_mean"],
            "output_scaler_floor": metadata["output_scaler_floor"],
            "floored_output_columns": metadata["floored_output_columns"],
            "reciprocity_enforced": metadata["reciprocity_enforced"],
            "passivity_enforced": metadata["passivity_enforced"],
            "rf_response_scale": metadata["rf_response_scale"],
            "export_commands": dict(export_commands),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
        }, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model = DNN.load(Path(args.model_dir))
    blocks = read_mdif(Path(args.mdif))
    pred_blocks = model.predict_blocks(blocks)
    out_path = Path(args.out_mdif)
    write_mdif(out_path, pred_blocks, model.sparam_labels)
    print(f"Wrote {out_path}")
    return 0


def command_export_ads(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = DNN.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    export_dc_model, dc_metadata = resolve_dnn_export_dc(
        model,
        source_metadata,
        args,
        model.target_z0,
    )
    model.dc_equivalent_resistance_ohm = float(
        dc_metadata["dc_equivalent_resistance_ohm"]
    )
    model.dc_resistance_source_kind = str(dc_metadata["dc_resistance_source_kind"])
    model.dc_port_resistances_ohm = (
        dict(dc_metadata["dc_port_resistances_ohm"])
        if isinstance(dc_metadata.get("dc_port_resistances_ohm"), dict)
        else None
    )
    model.dc_model = export_dc_model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mdif_name = args.output_name
    blocks = build_ads_export_blocks(
        template_mdif=args.template_mdif,
        parameter_grid_specs=args.parameter_grid,
        freqs_spec=args.freqs,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        model_dir=model_dir,
    )
    pred_blocks = model.predict_blocks(blocks)
    write_mdif(out_dir / mdif_name, pred_blocks, model.sparam_labels)
    manifest = write_ads_export_package(
        out_dir=out_dir,
        model_kind="DNN",
        model_dir=model_dir,
        mdif_name=mdif_name,
        blocks=pred_blocks,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        extra_manifest={
            "freq_transform": model.freq_transform,
            "layer_sizes": model.mlp.layer_sizes,
            "output_domain": model.output_domain,
            "target_z0": model.target_z0,
            "source_model_passivity_enforced": source_metadata.get(
                "passivity_enforced"
            ),
            "source_model_rf_response_scale": source_metadata.get(
                "rf_response_scale"
            ),
            "source_model_reciprocity_enforced": source_metadata.get(
                "reciprocity_enforced"
            ),
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_metadata": dc_metadata,
            "dc_is_separate_from_fitted_response": True,
        },
        extra_notes=[
            "Every exported block includes a zero-Hz point from the selected passive "
            "exact-DC port paths; it is not an extrapolation of the fitted DNN."
        ],
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mdif": str(out_dir / mdif_name),
        "manifest": str(out_dir / "ads_model_manifest.json"),
        "blocks": manifest["blocks"],
        "frequency_points_per_block": manifest["frequency_points_per_block"],
        "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
        "dc_port_paths": dc_metadata.get("dc_port_paths"),
        "dc_matrix_entries": dc_metadata.get("dc_matrix_entries"),
        "dc_sparameter_entries": dc_metadata.get("dc_sparameter_entries"),
        "dc_model_kind": dc_metadata.get("dc_model_kind"),
        "dc_port_resistances_ohm": dc_metadata.get("dc_port_resistances_ohm"),
        "dc_ignored_nonpassive_count": dc_metadata.get(
            "dc_ignored_nonpassive_count"
        ),
        "dc_open_circuit_applied": dc_metadata.get("dc_open_circuit_applied"),
        "dc_mdif_action": dc_metadata.get("dc_mdif_action"),
        "dc_mdif_training_blocks": dc_metadata.get("dc_mdif_training_block_count"),
        "dc_mdif_excluded_verification_blocks": dc_metadata.get(
            "dc_mdif_excluded_verification_block_count"
        ),
        "dc_mdif_excluded_unusable_blocks": dc_metadata.get(
            "dc_mdif_excluded_unusable_block_count"
        ),
        "dc_mdif_model_s_rmse": dc_metadata.get("dc_mdif_model_s_rmse"),
        "dc_mdif_model_s_max_abs_error": dc_metadata.get(
            "dc_mdif_model_s_max_abs_error"
        ),
        "dc_mdif_match_within_tolerance": dc_metadata.get(
            "dc_mdif_match_within_tolerance"
        ),
        "dc_mdif_warning": dc_metadata.get("dc_mdif_warning"),
    }, indent=2))
    return 0


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
    args.hidden_layers = args.hidden_layers or metadata_hidden_layers(model_metadata) or "128,128,64"
    args.activation = args.activation or str(model_metadata.get("activation", "tanh"))
    args.seed = args.seed if args.seed is not None else 1234
    if not getattr(args, "sparam_weights", None):
        metadata_weights = model_metadata.get("sparam_weights")
        if isinstance(metadata_weights, dict):
            args.sparam_weights = ";".join(
                f"{label}={metadata_weights[label]}" for label in sorted(metadata_weights, key=sparam_sort_key)
            )

    train_blocks, verify_blocks, all_blocks = split_data(args)
    if not train_blocks:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(all_blocks, requested=args.parameter_names, split_var=args.split_var)
    metadata_labels = model_metadata.get("sparam_labels")
    if isinstance(metadata_labels, list) and metadata_labels:
        labels = [str(label) for label in metadata_labels]
        missing = sorted(
            {label for label in labels for block in all_blocks if label not in block.sparams}
        )
        if missing:
            raise ValueError(
                f"Model metadata requested S-parameters not present in the MDIF data: {', '.join(missing)}"
            )
    else:
        labels = common_sparameter_labels(all_blocks)
    train_blocks = positive_frequency_blocks(train_blocks, purpose="ADS ANN fitting")
    verify_blocks = (
        positive_frequency_blocks(verify_blocks, purpose="ADS ANN verification")
        if verify_blocks
        else []
    )
    x_train, y_train = make_feature_target_samples(
        train_blocks,
        parameter_names,
        labels,
        args.freq_transform,
        "s",
        50.0,
    )
    if verify_blocks:
        x_verify, y_verify = make_feature_target_samples(
            verify_blocks,
            parameter_names,
            labels,
            args.freq_transform,
            "s",
            50.0,
        )
    else:
        x_verify = None
        y_verify = None

    requested_hidden_layers = parse_hidden_layers(args.hidden_layers)
    ads_hidden_layers, ads_neurons = resolve_ads_ann_layout(
        args.ads_hidden_layers,
        args.ads_neurons_per_layer,
    )

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
        "output_prefix": normalize_name(args.output_prefix) or "dnn_ads_ann",
        "netlist_module_name": args.module_name
        or f"{normalize_name(args.output_prefix) or 'dnn_ads_ann'}_sdd",
        "parameter_input_scales": parse_parameter_scale_spec(
            parameter_names,
            args.parameter_input_scales,
        ),
        "z0": float(args.z0),
    }
    input_columns = [*parameter_names, *frequency_feature_columns(args.freq_transform)]
    output_columns = sparameter_real_imag_columns(labels, prefix="fine")
    out_dir = Path(args.out_dir)
    manifest = write_ads_ann_package(
        out_dir=out_dir,
        model_kind="DNN",
        input_columns=input_columns,
        output_columns=output_columns,
        x_train=x_train,
        y_train=y_train,
        x_verify=x_verify,
        y_verify=y_verify,
        settings=settings,
        parameter_names=parameter_names,
        sparam_labels=labels,
        target_description="Direct fine S-parameter response, stored as real columns followed by imaginary columns.",
        extra_manifest={
            "source_model_dir": args.model_dir,
            "source_mdif": args.mdif,
            "verification_mdif": args.verification_mdif,
            "freq_transform": args.freq_transform,
            "sparam_weights": parse_sparam_weights(labels, getattr(args, "sparam_weights", None)),
            "source_frequency_weights": model_metadata.get("frequency_weights"),
            "requested_hidden_layers": requested_hidden_layers,
            "ads_layout_note": (
                "ADS ANN exposes a uniform hidden-layer width in the documented API. "
                "The package uses explicit --ads-hidden-layers/--ads-neurons-per-layer "
                "values, or the ADS-safe documented-example defaults of 2 layers and "
                "20 neurons per layer. Local model layers are recorded but not inherited."
            ),
        },
        extra_notes=[
            "This export retrains the DNN in ADS ANN; it does not import NumPy model.npz weights.",
            "When --model-dir is supplied, its labels, transforms, activation, and other compatible settings seed the export; local hidden-layer sizes are recorded but not inherited by ADS ANN.",
            "The package records S-parameter weights in the manifest. ADS ANN's documented Python API does not expose direct per-output loss weights, so the included ADS training script does not apply those weights.",
            "Any source-model frequency weights are recorded in the manifest but are not applied because the documented ADS ANN API does not expose per-sample loss weights.",
            "The native ADS ANN output predicts fine S-parameter real/imaginary values directly.",
            "Zero-Hz rows are excluded from ADS ANN fitting. Use the self-contained Verilog-A or sampled-MDIF export when the distinct saved DC resistance is required.",
        ],
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
        "ads_netlist": manifest["ads_netlist"],
    }, indent=2))
    return 0


def command_export_veriloga(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = DNN.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'dnn'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    fold_scalers = not bool(getattr(args, "no_fold_scalers", False))
    export_z0 = float(model.target_z0 if model.output_domain == "y" else args.z0)
    export_dc_model, dc_metadata = resolve_dnn_export_dc(
        model,
        source_metadata,
        args,
        export_z0,
    )
    if model.output_domain == "y" and not math.isclose(float(args.z0), export_z0, rel_tol=1e-12, abs_tol=1e-12):
        print(
            f"warning: direct-Y model was trained with target_z0={export_z0:g}; "
            "--z0 is ignored for direct-Y Verilog-A stamping",
            file=sys.stderr,
        )
    export_notes = [
        "This direct Verilog-A export embeds the saved local model.npz weights; it does not retrain in ADS ANN.",
        "The generated N-port is intended for S-parameter and small-signal AC simulation. It is not a causal transient model.",
        "The default frequency expression is $freq. If your ADS Verilog-A environment uses a different frequency variable, regenerate with --frequency-expression.",
        "At exactly zero frequency, the fitted DNN is bypassed and only the selected data-derived DC port paths are stamped.",
    ]
    if model.output_domain == "y":
        export_notes.append(
            "This DNN was trained with --output-domain y, so the Verilog-A stamps predicted admittance directly and skips runtime S-to-Y matrix inversion."
        )
    elif fold_scalers:
        export_notes.append(
            "Input standardization and output scaling were folded into the first and final neural layers to reduce per-evaluation Verilog-A arithmetic."
        )
    manifest = write_veriloga_package(
        out_dir=out_dir,
        model_kind="DNN",
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
        z0=export_z0,
        frequency_expression=args.frequency_expression,
        parameter_input_scales=parameter_input_scales,
        output_domain=model.output_domain,
        fold_input_scaler=fold_scalers,
        fold_output_scaler=fold_scalers,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=dc_metadata.get("dc_resistance_source_kind"),
        dc_port_resistances_ohm=dc_metadata.get("dc_port_resistances_ohm"),
        dc_model=(export_dc_model.export_data() if export_dc_model is not None else None),
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "direct_dnn",
            "fully_self_contained": True,
            "training_output_domain": model.output_domain,
            "training_target_z0": model.target_z0,
            "source_model_passivity_enforced": source_metadata.get(
                "passivity_enforced"
            ),
            "source_model_rf_response_scale": source_metadata.get(
                "rf_response_scale"
            ),
            "source_model_reciprocity_enforced": source_metadata.get(
                "reciprocity_enforced"
            ),
            "dc_resistance_source_kind": dc_metadata.get(
                "dc_resistance_source_kind"
            ),
            "dc_resistance_pair_means_ohm": dc_metadata.get(
                "dc_resistance_pair_means_ohm"
            ),
            "dc_metadata": dc_metadata,
        },
        extra_notes=export_notes,
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
        "output_domain": manifest["output_domain"],
        "folded_input_scaler": manifest["folded_input_scaler"],
        "folded_output_scaler": manifest["folded_output_scaler"],
        "dc_equivalent_resistance_ohm": manifest["dc_equivalent_resistance_ohm"],
        "dc_port_paths": manifest.get("dc_port_paths"),
        "dc_matrix_entries": manifest.get("dc_matrix_entries"),
        "dc_sparameter_entries": manifest.get("dc_sparameter_entries"),
        "dc_model_kind": manifest.get("dc_model_kind"),
        "dc_port_resistances_ohm": manifest.get("dc_port_resistances_ohm"),
        "dc_resistance_source_kind": manifest["dc_resistance_source_kind"],
        "dc_resistance_pair_means_ohm": manifest["dc_resistance_pair_means_ohm"],
        "dc_ignored_nonpassive_count": dc_metadata.get(
            "dc_ignored_nonpassive_count"
        ),
        "dc_open_circuit_applied": dc_metadata.get("dc_open_circuit_applied"),
        "dc_mdif_action": dc_metadata.get("dc_mdif_action"),
        "dc_mdif_training_blocks": dc_metadata.get("dc_mdif_training_block_count"),
        "dc_mdif_excluded_verification_blocks": dc_metadata.get(
            "dc_mdif_excluded_verification_block_count"
        ),
        "dc_mdif_excluded_unusable_blocks": dc_metadata.get(
            "dc_mdif_excluded_unusable_block_count"
        ),
        "dc_mdif_model_s_rmse": dc_metadata.get("dc_mdif_model_s_rmse"),
        "dc_mdif_model_s_max_abs_error": dc_metadata.get(
            "dc_mdif_model_s_max_abs_error"
        ),
        "dc_mdif_match_within_tolerance": dc_metadata.get(
            "dc_mdif_match_within_tolerance"
        ),
        "dc_mdif_warning": dc_metadata.get("dc_mdif_warning"),
    }, indent=2))
    return 0


def command_export_ads_hb(args: argparse.Namespace) -> int:
    """Export a trained DNN as a linear ADS SDD network for harmonic balance."""

    model_dir = Path(args.model_dir)
    model = DNN.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    direct_y_trial_dir = (
        Path(args.direct_y_trial_model_dir)
        if args.direct_y_trial_model_dir
        else None
    )
    direct_y_trial = (
        DNN.load(direct_y_trial_dir) if direct_y_trial_dir is not None else None
    )
    direct_y_trial_metadata = (
        read_model_metadata(str(direct_y_trial_dir))
        if direct_y_trial_dir is not None
        else None
    )
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'dnn'}_hb"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    export_z0 = float(model.target_z0 if model.output_domain == "y" else args.z0)
    direct_y_comparison: dict[str, object] | None = None
    if direct_y_trial is not None:
        if model.output_domain != "s":
            raise ValueError(
                "--direct-y-trial-model-dir requires an S-domain baseline model; "
                "the selected baseline already stamps Y directly"
            )
        if direct_y_trial.output_domain != "y":
            raise ValueError(
                "--direct-y-trial-model-dir must contain a model trained with "
                "--output-domain y"
            )
        if direct_y_trial.parameter_names != model.parameter_names:
            raise ValueError(
                "The direct-Y trial parameter names/order do not match the baseline"
            )
        if direct_y_trial.sparam_labels != model.sparam_labels:
            raise ValueError(
                "The direct-Y trial S-parameter labels/order do not match the baseline"
            )
        if direct_y_trial.freq_transform != model.freq_transform:
            raise ValueError(
                "The direct-Y trial frequency transform does not match the baseline"
            )
        if direct_y_trial.mlp.activation != model.mlp.activation:
            raise ValueError(
                "The direct-Y trial activation does not match the baseline"
            )
        if direct_y_trial.mlp.layer_sizes != model.mlp.layer_sizes:
            raise ValueError(
                "The direct-Y trial layer sizes do not match the baseline. Keep the "
                "same architecture so this trial isolates the response domain."
            )
        if not math.isclose(
            direct_y_trial.target_z0,
            export_z0,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "The direct-Y trial target_z0 does not match the baseline export "
                f"reference impedance ({direct_y_trial.target_z0:g} versus "
                f"{export_z0:g})"
            )
        comparison_fields = [
            "training_blocks",
            "verification_blocks",
            "split_var",
            "train_values",
            "verify_values",
            "normalized_sparam_weights",
            "frequency_weights",
        ]
        training_metadata_differences = {
            field: {
                "baseline": source_metadata.get(field),
                "direct_y_trial": direct_y_trial_metadata.get(field),
            }
            for field in comparison_fields
            if direct_y_trial_metadata is not None
            and field in source_metadata
            and field in direct_y_trial_metadata
            and source_metadata.get(field) != direct_y_trial_metadata.get(field)
        }
        direct_y_comparison = {
            "baseline_output_domain": "s",
            "trial_output_domain": "y",
            "target_z0": export_z0,
            "same_parameter_order": True,
            "same_sparameter_order": True,
            "same_frequency_transform": True,
            "same_activation": True,
            "same_layer_sizes": True,
            "training_metadata_differences": training_metadata_differences,
        }
    export_dc_model, dc_metadata = resolve_dnn_export_dc(
        model,
        source_metadata,
        args,
        export_z0,
    )
    if model.output_domain == "y" and not math.isclose(
        float(args.z0), export_z0, rel_tol=1e-12, abs_tol=1e-12
    ):
        print(
            f"warning: direct-Y model was trained with target_z0={export_z0:g}; "
            "--z0 is ignored for direct-Y ADS HB stamping",
            file=sys.stderr,
        )
    default_notes = [
        "The fitted RF response is evaluated at each HB spectral frequency.",
        "The passive network has no input-power parameter and introduces no compression.",
    ]
    if direct_y_trial is not None:
        default_notes.extend(
            [
                "The default S-domain implementation in this package is unchanged.",
                "A separately trained direct-Y comparison package is exported beside this baseline.",
                "The direct-Y trial reuses this baseline package's exact-DC model and differs only in its separately fitted RF response domain.",
            ]
        )
    manifest = write_ads_hb_mlp_package(
        out_dir=out_dir,
        model_kind="DNN",
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
        z0=export_z0,
        parameter_input_scales=parameter_input_scales,
        output_domain=model.output_domain,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=dc_metadata.get("dc_resistance_source_kind"),
        dc_port_resistances_ohm=dc_metadata.get("dc_port_resistances_ohm"),
        dc_model=(export_dc_model.export_data() if export_dc_model is not None else None),
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "direct_dnn",
            "training_output_domain": model.output_domain,
            "training_target_z0": model.target_z0,
            "source_model_passivity_enforced": source_metadata.get(
                "passivity_enforced"
            ),
            "source_model_rf_response_scale": source_metadata.get(
                "rf_response_scale"
            ),
            "source_model_reciprocity_enforced": source_metadata.get(
                "reciprocity_enforced"
            ),
            "dc_metadata": dc_metadata,
        },
        extra_notes=default_notes,
    )
    if direct_y_trial is not None:
        assert direct_y_trial_dir is not None
        assert direct_y_trial_metadata is not None
        assert direct_y_comparison is not None
        trial_module_name = f"{manifest['module_name']}_direct_y_trial"
        trial_manifest = write_ads_hb_mlp_package(
            out_dir=out_dir,
            model_kind="DNN",
            module_name=trial_module_name,
            parameter_names=direct_y_trial.parameter_names,
            sparam_labels=direct_y_trial.sparam_labels,
            freq_transform=direct_y_trial.freq_transform,
            activation=direct_y_trial.mlp.activation,
            layer_sizes=direct_y_trial.mlp.layer_sizes,
            weights=direct_y_trial.mlp.weights,
            biases=direct_y_trial.mlp.biases,
            x_mean=np.asarray(direct_y_trial.x_scaler.mean, dtype=float),
            x_std=np.asarray(direct_y_trial.x_scaler.std, dtype=float),
            y_mean=np.asarray(direct_y_trial.y_scaler.mean, dtype=float),
            y_std=np.asarray(direct_y_trial.y_scaler.std, dtype=float),
            z0=float(direct_y_trial.target_z0),
            parameter_input_scales=parameter_input_scales,
            output_domain="y",
            dc_equivalent_resistance_ohm=float(
                dc_metadata["dc_equivalent_resistance_ohm"]
            ),
            dc_resistance_source_kind=dc_metadata.get(
                "dc_resistance_source_kind"
            ),
            dc_port_resistances_ohm=dc_metadata.get(
                "dc_port_resistances_ohm"
            ),
            dc_model=(
                export_dc_model.export_data()
                if export_dc_model is not None
                else None
            ),
            source_model_dir=str(direct_y_trial_dir),
            extra_manifest={
                "model_family": "direct_dnn",
                "training_output_domain": "y",
                "training_target_z0": direct_y_trial.target_z0,
                "source_model_passivity_enforced": direct_y_trial_metadata.get(
                    "passivity_enforced"
                ),
                "source_model_rf_response_scale": direct_y_trial_metadata.get(
                    "rf_response_scale"
                ),
                "source_model_reciprocity_enforced": direct_y_trial_metadata.get(
                    "reciprocity_enforced"
                ),
                "trial_parent_module_name": manifest["module_name"],
                "trial_parent_model_dir": str(model_dir),
                "trial_purpose": (
                    "Compare a separately trained direct-Y RF DNN against the "
                    "S-domain baseline without runtime RF S-to-Y conversion"
                ),
                "direct_y_comparison": direct_y_comparison,
                "dc_metadata": dc_metadata,
                "dc_source_model_dir": str(model_dir),
            },
            extra_notes=[
                "Trial implementation: the RF DNN was trained directly against Y-parameters.",
                "The trial stamps predicted RF admittance without the baseline runtime S-to-Y equation graph.",
                "The baseline package's separately extracted exact-DC model is reused unchanged.",
                f"The unchanged baseline remains `{manifest['netlist_file']}` with module `{manifest['module_name']}`.",
            ],
            artifact_variant="direct_y_trial",
        )
        trial_export = {
            "kind": "direct_y_rf_dnn",
            "status": "trial",
            "module_name": trial_manifest["module_name"],
            "netlist_file": trial_manifest["netlist_file"],
            "manifest_file": trial_manifest["manifest_file"],
            "readme_file": trial_manifest["readme_file"],
            "instance_template_file": trial_manifest["instance_template_file"],
            "response_domain": trial_manifest["response_domain"],
            "rf_source_conversion": trial_manifest["rf_source_conversion"],
            "sdd_dc_rf_topology": trial_manifest["sdd_dc_rf_topology"],
            "source_model_dir": str(direct_y_trial_dir),
            "direct_y_comparison": direct_y_comparison,
        }
        manifest["trial_exports"] = [trial_export]
        (out_dir / str(manifest["manifest_file"])).write_text(
            json.dumps(manifest, indent=2)
        )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "netlist": str(out_dir / str(manifest["netlist_file"])),
                "manifest": str(out_dir / "ads_hb_manifest.json"),
                "module_name": manifest["module_name"],
                "linear": manifest["linear"],
                "power_dependent": manifest["power_dependent"],
                "supported_analyses": manifest["supported_analyses"],
                "trial_exports": manifest.get("trial_exports", []),
                "dc_matrix_entries": manifest.get("dc_matrix_entries"),
                "dc_sparameter_entries": manifest.get("dc_sparameter_entries"),
                "dc_model_kind": manifest.get("dc_model_kind"),
                "dc_mdif_action": dc_metadata.get("dc_mdif_action"),
                "dc_mdif_training_blocks": dc_metadata.get(
                    "dc_mdif_training_block_count"
                ),
                "dc_mdif_excluded_verification_blocks": dc_metadata.get(
                    "dc_mdif_excluded_verification_block_count"
                ),
                "dc_mdif_excluded_unusable_blocks": dc_metadata.get(
                    "dc_mdif_excluded_unusable_block_count"
                ),
                "dc_mdif_model_s_rmse": dc_metadata.get("dc_mdif_model_s_rmse"),
                "dc_mdif_model_s_max_abs_error": dc_metadata.get(
                    "dc_mdif_model_s_max_abs_error"
                ),
                "dc_mdif_match_within_tolerance": dc_metadata.get(
                    "dc_mdif_match_within_tolerance"
                ),
                "dc_mdif_warning": dc_metadata.get("dc_mdif_warning"),
            },
            indent=2,
        )
    )
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
    if args.mode == "adaptive":
        base_config = {
            "freq_transform": parse_text_options(args.freq_transform_options)[0],
            "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options)[0],
            "activation": parse_text_options(args.activation_options)[0],
            "learning_rate": parse_float_options(args.learning_rates)[0],
            "passivity_penalty": float(args.passivity_penalty),
        }
        candidates, columns, log_parameters, categorical_values = (
            build_adaptive_candidate_pool(
                base_config,
                args.optimize_parameter,
                {
                    "activation": "str",
                    "batch_size": "int",
                    "epochs": "int",
                    "freq_transform": "str",
                    "hidden_layers": "hidden_layers",
                    "learning_rate": "float",
                    "output_domain": "str",
                    "passivity_penalty": "float",
                    "patience": "int",
                    "target_z0": "float",
                },
                max_trials=args.max_trials,
                candidate_pool=args.adaptive_candidate_pool,
                hidden_width_step=args.adaptive_hidden_width_step,
                seed=args.seed,
            )
        )
        args.adaptive_result_columns = columns
        args.adaptive_log_parameters = log_parameters
        args.adaptive_categorical_values = categorical_values
        return candidates
    axes = {
        "freq_transform": parse_text_options(args.freq_transform_options),
        "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options),
        "activation": parse_text_options(args.activation_options),
        "learning_rate": parse_float_options(args.learning_rates),
    }
    candidates = []
    keys = list(axes)
    for values in itertools.product(*(axes[key] for key in keys)):
        candidate = dict(zip(keys, values))
        if candidate["freq_transform"] not in {"log", "linear", "log-linear"}:
            raise ValueError(f"Unsupported frequency transform {candidate['freq_transform']!r}")
        candidates.append(candidate)
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
    return apply_candidate_overrides(argparse.Namespace(
        mdif=args.mdif,
        verification_mdif=args.verification_mdif,
        out_dir=str(out_dir),
        split_var=args.split_var,
        train_values=args.train_values,
        verify_values=args.verify_values,
        parameter_names=args.parameter_names,
        holdout_fraction=args.holdout_fraction,
        dc_port_paths=getattr(args, "dc_port_paths", None),
        dc_open_threshold=args.dc_open_threshold,
        dc_open_resistance=args.dc_open_resistance,
        freq_transform=str(candidate["freq_transform"]),
        hidden_layers=str(candidate["hidden_layers"]),
        activation=str(candidate["activation"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=float(candidate["learning_rate"]),
        patience=args.patience,
        loss_interval=args.loss_interval,
        progress_interval=args.progress_interval,
        progress_label=f"DNN trial {trial_index}",
        seed=trial_seed,
        output_domain=args.output_domain,
        target_z0=args.target_z0,
        max_y_condition=args.max_y_condition,
        passivity_mode=args.passivity_mode,
        passivity_margin=args.passivity_margin,
        passivity_penalty=args.passivity_penalty,
        reciprocity_mode=args.reciprocity_mode,
        reciprocity_tolerance=args.reciprocity_tolerance,
        worst_plots=plots,
        sparam_weights=args.sparam_weights,
        frequency_weights=args.frequency_weights,
        debug=bool(getattr(args, "debug", False)),
        quiet=True,
    ), candidate)


def dnn_sweep_trial_worker(payload: tuple[dict[str, object], dict[str, object], str, int, int]) -> dict[str, object]:
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
        error_traceback = debug_traceback(args)
    summary_path = trial_dir / "verification_summary.json"
    summary = load_or_write_trial_summary(
        summary_path,
        status=status,
        error_message=error_message,
        traceback_text=error_traceback,
    )
    metric_value = summary_metric(summary, args.selection_metric) if status == 0 else None
    return {
        "trial": trial_index,
        "candidate": candidate,
        "summary": summary,
        "metric": metric_value,
        "trial_seed": trial_seed,
        "plot_paths": trial_plot_paths(summary, trial_dir, out_dir),
    }


def command_sweep(args: argparse.Namespace) -> int:
    status = run_sweep_command(
        args,
        sweep_candidate_grid(args),
        worker_func=dnn_sweep_trial_worker,
        namespace_for_trial_func=namespace_for_trial,
        train_func=command_train,
        result_columns=DNN_SWEEP_RESULT_COLUMNS,
        results_filename="dnn_sweep_results.csv",
        best_config_filename="dnn_best_config.json",
        summary_filename="dnn_sweep_summary.md",
        diagnostics_prefix="dnn",
        train_command_prefix=[
            sys.executable,
            "surrogate.py",
            "--model",
            "dnn",
            "train",
        ],
    )
    best_dir = Path(args.out_dir) / "best_model"
    if status == 0:
        update_training_export_commands(
            best_dir / "training_summary.md",
            dnn_export_commands(best_dir, dc_mdif=args.mdif),
        )
        update_training_export_commands(
            Path(args.out_dir) / "dnn_sweep_summary.md",
            dnn_export_commands(best_dir, dc_mdif=args.mdif),
        )
    return status


def command_rerank_sweep(args: argparse.Namespace) -> int:
    sweep_dir = Path(args.sweep_dir)
    results_filename = (
        "dnn_sweep_results.csv"
        if (sweep_dir / "dnn_sweep_results.csv").exists()
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
        for key in DNN_SWEEP_RESULT_COLUMNS
        if key in best_row and best_row[key] not in {None, ""}
    }

    results_path = sweep_dir / "dnn_reranked_sweep_results.csv"
    summary_path = sweep_dir / "dnn_reranked_sweep_summary.md"
    best_config_path = sweep_dir / "dnn_reranked_best_config.json"
    write_csv(results_path, reranked)
    diagnostic_artifacts = [
        str(path.relative_to(sweep_dir))
        for path in plot_sweep_diagnostics(
            reranked,
            sweep_dir,
            DNN_SWEEP_RESULT_COLUMNS,
            args.selection_metric,
            prefix="dnn_reranked",
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
        if promoted and best_model_dir is not None:
            export_commands = dnn_export_commands(best_model_dir)
            update_training_export_commands(
                best_model_dir / "training_summary.md",
                export_commands,
            )
            update_training_export_commands(
                summary_path,
                export_commands,
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
    blocks = read_mdif(Path(args.mdif))
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


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mdif", required=True, help="Training/verification MDIF path")
    parser.add_argument("--verification-mdif", help="Optional separate verification MDIF")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--split-var", default="dataset")
    parser.add_argument("--train-values", default="train,training")
    parser.add_argument("--verify-values", default="verify,verification,test,validation")
    parser.add_argument("--parameter-names", help="Comma-separated geometry/process VAR names")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    add_dc_fitting_arguments(parser)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss-interval", type=int, default=1, help="Full train/validation loss check interval in epochs")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Console progress update interval in epochs. Use 0 to disable progress updates.",
    )
    parser.add_argument(
        "--output-domain",
        choices=["s", "y"],
        default="s",
        help="Training target domain: S-parameters (s) or direct admittance Y-parameters (y)",
    )
    parser.add_argument(
        "--max-y-condition",
        type=float,
        default=1e10,
        help=(
            "Reject direct-Y fitting when cond(I + S) exceeds this limit; "
            "near-singular conversion targets are not numerically learnable "
            "(default: 1e10)"
        ),
    )
    parser.add_argument(
        "--passivity-mode",
        choices=["auto", "enforce", "off"],
        default="auto",
        help=(
            "S-domain passivity handling. auto protects a passive training set, "
            "enforce always protects it, and off disables the protection"
        ),
    )
    parser.add_argument(
        "--passivity-margin",
        type=float,
        default=1e-3,
        help="Target margin below unit maximum singular value (default: 0.001)",
    )
    parser.add_argument(
        "--passivity-penalty",
        type=float,
        default=10.0,
        help=(
            "Weight of the differentiable S-matrix passivity loss; may also be "
            "an adaptive --optimize-parameter (default: 10)"
        ),
    )
    parser.add_argument(
        "--reciprocity-mode",
        choices=["auto", "enforce", "off"],
        default="enforce",
        help=(
            "Reciprocity handling. enforce always ties Sij/Sji, auto ties them "
            "only for reciprocal training data, and off leaves them independent "
            "(default: enforce)"
        ),
    )
    parser.add_argument(
        "--reciprocity-tolerance",
        type=float,
        default=1e-6,
        help="Maximum relative training-data mismatch accepted by reciprocity auto mode",
    )
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--target-z0",
        type=float,
        default=50.0,
        help="Reference impedance used when --output-domain y converts S-parameter MDIF data to Y targets",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=os.environ.get("ADS_SURROGATE_CLI_PROG"),
        description="Train and evaluate a direct deep neural-network S-parameter surrogate from MDIF data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one DNN model")
    add_data_args(train)
    train.add_argument("--freq-transform", choices=["log", "linear", "log-linear"], default="log")
    train.add_argument("--hidden-layers", default="128,128,64")
    train.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument(
        "--sparam-weights",
        help="S-parameter loss weights. Examples: 'diag=1;offdiag=0.2' or 'S11,S22=1;S12,S21=0.1'. Later rules override earlier ones.",
    )
    train.add_argument(
        "--frequency-weights",
        help="Frequency loss weights, e.g. 'default=1;1GHz=5;2GHz:4GHz=3'. Exact frequencies and inclusive ranges are supported; later rules override earlier ones.",
    )
    train.add_argument("--worst-plots", type=int, default=6)
    add_debug_argument(train)
    train.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    train.set_defaults(func=command_train)

    sweep = sub.add_parser(
        "sweep",
        aliases=["optimize"],
        help="Try multiple DNN configurations and retrain the best one",
    )
    add_data_args(sweep)
    sweep.add_argument(
        "--freq-transforms",
        "--freq-transform-options",
        "--freq-transform",
        dest="freq_transform_options",
        default="log,linear,log-linear",
        help="Comma-separated frequency transforms; --freq-transform is the single-value train-compatible form.",
    )
    sweep.add_argument(
        "--hidden-layers",
        "--hidden-layer-layouts",
        "--hidden-layer-options",
        dest="hidden_layer_options",
        default="64,64;128,128,64;128,128,128;256,128,64",
        help="One train-style layout or semicolon-separated hidden-layer layouts.",
    )
    sweep.add_argument(
        "--activations",
        "--activation-options",
        "--activation",
        dest="activation_options",
        default="tanh,relu",
        help="Comma-separated activations; --activation is the single-value train-compatible form.",
    )
    sweep.add_argument(
        "--learning-rates",
        "--learning-rate",
        dest="learning_rates",
        default="0.001,0.002,0.005",
        help="Comma-separated learning rates; --learning-rate accepts one value as in train.",
    )
    sweep.add_argument("--jobs", type=int, default=1, help="Number of sweep trials to train in parallel")
    sweep.add_argument(
        "--sparam-weights",
        help="S-parameter loss/selection weights. Examples: 'diag=1;offdiag=0.2' or 'all=0.2;S21=1'.",
    )
    sweep.add_argument(
        "--frequency-weights",
        help="Frequency loss/selection weights, e.g. 'default=1;1GHz=5;2GHz:4GHz=3'.",
    )
    sweep.add_argument(
        "--search-mode",
        "--mode",
        dest="mode",
        choices=["adaptive", "grid", "random"],
        default="random",
        help="Sweep search strategy. --mode remains a backward-compatible alias.",
    )
    sweep.add_argument("--max-trials", type=int, default=24)
    add_adaptive_search_arguments(sweep)
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
    add_debug_argument(sweep)
    sweep.add_argument(
        "--retrain-best",
        action="store_true",
        help="Retrain the selected best configuration at the end instead of using the best completed trial model",
    )
    sweep.set_defaults(func=command_sweep)

    rerank = sub.add_parser(
        "rerank-sweep",
        help="Re-rank an existing DNN sweep using saved trial summaries without rerunning all trials",
    )
    rerank.add_argument("--sweep-dir", required=True, help="Existing DNN sweep/optimize output directory")
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
    predict.add_argument("--out-mdif", required=True)
    predict.set_defaults(func=command_predict)

    export_ads = sub.add_parser(
        "export-ads-mdif",
        aliases=["export-ads"],
        help="Export a trained DNN as an ADS-ready parameterized S-parameter MDIF package",
    )
    export_ads.add_argument("--model-dir", required=True, help="Directory containing a trained model.npz and metadata.json")
    export_ads.add_argument("--out-dir", required=True, help="Output directory for the ADS MDIF package")
    export_ads.add_argument(
        "--template-mdif",
        help="MDIF containing the exact parameter/frequency blocks to evaluate; S-data is ignored",
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
    add_dc_export_arguments(export_ads)
    export_ads.set_defaults(func=command_export_ads)

    export_ann = sub.add_parser(
        "export-ads-ann",
        help="Create an ADS ANN training/extraction package for a native ADS DNN model",
    )
    export_ann.add_argument("--mdif", required=True, help="Training/verification MDIF path")
    export_ann.add_argument("--verification-mdif", help="Optional separate verification MDIF")
    export_ann.add_argument(
        "--model-dir",
        help="Optional trained model or best_model directory used for labels and compatible export metadata; ADS hidden sizes remain independent",
    )
    export_ann.add_argument("--out-dir", required=True, help="Output directory for the ADS ANN package")
    export_ann.add_argument("--split-var")
    export_ann.add_argument("--train-values")
    export_ann.add_argument("--verify-values")
    export_ann.add_argument("--parameter-names", help="Comma-separated geometry/process VAR names")
    export_ann.add_argument("--holdout-fraction", type=float, default=0.2)
    export_ann.add_argument("--freq-transform", choices=["log", "linear", "log-linear"])
    export_ann.add_argument("--hidden-layers")
    export_ann.add_argument("--activation", choices=["tanh", "relu"])
    export_ann.add_argument(
        "--sparam-weights",
        help="Optional S-parameter weights to record in the ADS ANN manifest; defaults to model metadata when --model-dir is supplied",
    )
    export_ann.add_argument("--ads-hidden-layers", type=int, help="Override ADS AnnSetup.num_hidden_layers; default 2")
    export_ann.add_argument("--ads-neurons-per-layer", type=int, help="Override ADS AnnSetup.num_neurons_per_layer; default 20")
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
    export_ann.add_argument("--output-prefix", default="dnn_ads_ann")
    export_ann.add_argument(
        "--module-name",
        help="Generated ADS ANN SDD subnetwork name. Defaults to <output-prefix>_sdd",
    )
    export_ann.add_argument(
        "--parameter-input-scales",
        default="1.0",
        help="One ADS-side input scale applied to every geometry parameter, such as 1.0 or 1um",
    )
    export_ann.add_argument(
        "--z0",
        type=float,
        default=50.0,
        help="S-parameter reference impedance used by the generated ANN SDD netlist",
    )
    export_ann.add_argument("--seed", type=int)
    export_ann.set_defaults(func=command_export_ads_ann)

    export_hb = sub.add_parser(
        "export-ads-hb",
        help="Export a trained DNN as a self-contained linear ADS SDD network for harmonic balance",
    )
    export_hb.add_argument("--model-dir", required=True, help="Directory containing trained model.npz and metadata.json")
    export_hb.add_argument(
        "--direct-y-trial-model-dir",
        help=(
            "Optional separately trained direct-Y DNN with the same parameters, "
            "S-parameter order, frequency transform, activation, layer sizes, and "
            "target z0. Exports it beside the unchanged S-domain baseline as the "
            "next ADS HB timing trial."
        ),
    )
    export_hb.add_argument("--out-dir", required=True, help="Output directory for the ADS HB package")
    export_hb.add_argument("--module-name", help="ADS subnetwork name. Defaults to the model directory name plus _hb")
    export_hb.add_argument("--z0", type=float, default=50.0, help="S-parameter reference impedance")
    export_hb.add_argument(
        "--parameter-input-scales",
        metavar="SCALE",
        help=(
            "Common positive ADS-side denominator for every instance parameter: "
            "model_value = instance_value / scale. Example: 1um"
        ),
    )
    add_dc_export_arguments(export_hb)
    export_hb.set_defaults(func=command_export_ads_hb)

    export_va = sub.add_parser(
        "export-veriloga",
        help="Export a trained DNN directly as a Verilog-A N-port using saved model.npz weights",
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
        metavar="SCALE",
        help=(
            "Common positive ADS-side denominator for every instance parameter: "
            "model_value = instance_value / scale. Example: 1um"
        ),
    )
    export_va.add_argument(
        "--no-fold-scalers",
        action="store_true",
        help="Debug option: keep input/output standardization as explicit Verilog-A arithmetic instead of folding it into the neural layers",
    )
    add_dc_export_arguments(export_va)
    export_va.set_defaults(func=command_export_veriloga)

    inspect = sub.add_parser("inspect-mdif", help="Inspect parsed MDIF blocks")
    inspect.add_argument("--mdif", required=True)
    inspect.add_argument("--split-var", default="dataset")
    inspect.set_defaults(func=command_inspect)
    add_options_json_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parse_args_with_options_json(parser, argv, model="dnn")
    try:
        status = int(args.func(args))
    except Exception as exc:
        print_cli_error(args, exc)
        return 2
    return finalize_options_json_update(args, status)


if __name__ == "__main__":
    raise SystemExit(main())
