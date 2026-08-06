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
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

RC2_ROOT = Path(__file__).resolve().parents[1]
if str(RC2_ROOT) not in sys.path:
    sys.path.insert(0, str(RC2_ROOT))

from common.surrogate_common import (  # noqa: E402
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
    infer_complete_sparameter_ports,
    infer_parameter_names,
    load_sweep_rows,
    infer_uniform_hidden_layout,
    make_training_progress_callback,
    metadata_csv,
    metadata_hidden_layers,
    normalize_name,
    normalize_sparam_weights,
    output_weights_from_sparam_weights,
    parse_csv_set,
    parse_float_options,
    parse_hidden_layer_options,
    parse_hidden_layers,
    parse_number,
    parse_parameter_scale_spec,
    parse_sparam_weights,
    parse_text_options,
    progress_interval_from_args,
    plot_sweep_diagnostics,
    plot_worst_case_fits,
    plot_worst_case_y_fits,
    read_mdif,
    read_model_metadata,
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
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = parameter_names
        self.sparam_labels = sparam_labels
        self.freq_transform = freq_transform
        self.output_domain = validate_output_domain(output_domain)
        self.target_z0 = float(target_z0)

    def predict_blocks(self, blocks: Sequence[MDIFBlock]) -> list[MDIFBlock]:
        predicted = []
        for block in blocks:
            x = block_features(block, self.parameter_names, self.freq_transform)
            y_scaled = self.mlp.predict(self.x_scaler.transform(x))
            y_columns = self.y_scaler.inverse_transform(y_scaled)
            values = columns_to_complex(y_columns)
            if self.output_domain == "y":
                values = y_values_to_s_values(values, self.sparam_labels, self.target_z0)
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
        combined_metadata = {
            "version": VERSION,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation,
            "freq_transform": self.freq_transform,
            "output_domain": self.output_domain,
            "target_z0": self.target_z0,
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
        )


def train_model(args: argparse.Namespace) -> tuple[DNN, list[MDIFBlock], list[str], list[str], list[dict[str, float]], dict[str, object]]:
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
    x_train, y_train = make_feature_target_samples(
        train_blocks,
        parameter_names,
        labels,
        args.freq_transform,
        output_domain,
        target_z0,
    )
    if verify_blocks:
        x_verify, y_verify = make_feature_target_samples(
            verify_blocks,
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

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    layer_sizes = [x_train.shape[1], *hidden_layers, y_train.shape[1]]
    mlp = MLP(layer_sizes, activation=args.activation, seed=args.seed)
    sparam_weights = parse_sparam_weights(labels, getattr(args, "sparam_weights", None))
    normalized_sparam_weights = normalize_sparam_weights(labels, sparam_weights)
    output_weights = output_weights_from_sparam_weights(labels, sparam_weights)
    progress_interval = progress_interval_from_args(args)
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
        loss_interval=getattr(args, "loss_interval", 1),
        progress_callback=make_training_progress_callback(
            getattr(args, "progress_label", "DNN fit"),
            args.epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
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
    )
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
        "sparam_weight_mean": sparam_weight_mean(labels, sparam_weights),
        "sparam_weight_normalization": "Raw S-parameter weights are divided by their mean before training, so the average normalized weight is 1.0.",
        "output_scaler_floor": output_std_floor,
        "floored_output_columns": floored_output_columns,
    }
    return model, verify_blocks, parameter_names, labels, history, metadata


def command_train(args: argparse.Namespace) -> int:
    model, verify_blocks, parameter_names, labels, history, metadata = train_model(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir, metadata=metadata)
    write_history(out_dir / "training_history.csv", history)
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

    if verify_blocks:
        pred_blocks = model.predict_blocks(verify_blocks)
        summary = write_training_verification_artifacts(
            out_dir,
            verify_blocks,
            pred_blocks,
            labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            sparam_weights=parse_sparam_weights(labels, getattr(args, "sparam_weights", None)),
            y_z0=model.target_z0,
        )
    else:
        summary = {"warning": "No verification blocks were available"}
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="DNN",
        config=training_config,
        summary=summary,
        history=history,
    )

    if not getattr(args, "quiet", False):
        print(json.dumps({
            "out_dir": str(out_dir),
            "training_summary": str(out_dir / "training_summary.md"),
            "parameters": parameter_names,
            "sparameters": labels,
            "freq_transform": model.freq_transform,
            "output_domain": model.output_domain,
            "target_z0": model.target_z0,
            "layer_sizes": model.mlp.layer_sizes,
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
        },
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mdif": str(out_dir / mdif_name),
        "manifest": str(out_dir / "ads_model_manifest.json"),
        "blocks": manifest["blocks"],
        "frequency_points_per_block": manifest["frequency_points_per_block"],
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
    default_hidden_layers, default_neurons = infer_uniform_hidden_layout(requested_hidden_layers)
    ads_hidden_layers = args.ads_hidden_layers if args.ads_hidden_layers is not None else default_hidden_layers
    ads_neurons = args.ads_neurons_per_layer if args.ads_neurons_per_layer is not None else default_neurons
    if ads_hidden_layers <= 0:
        raise ValueError("--ads-hidden-layers must be positive")
    if ads_neurons <= 0:
        raise ValueError("--ads-neurons-per-layer must be positive")

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
            "requested_hidden_layers": requested_hidden_layers,
            "ads_layout_note": (
                "ADS ANN exposes a uniform hidden-layer width in the documented API. "
                "The package uses --ads-hidden-layers/--ads-neurons-per-layer, or derives "
                "them from --hidden-layers when those overrides are omitted."
            ),
        },
        extra_notes=[
            "This export retrains the DNN in ADS ANN; it does not import NumPy model.npz weights.",
            "When --model-dir is supplied, metadata from that trained or optimized model is used to seed the ADS ANN architecture/settings.",
            "The package records S-parameter weights in the manifest. ADS ANN's documented Python API does not expose direct per-output loss weights, so the included ADS training script does not apply those weights.",
            "The native ADS ANN output predicts fine S-parameter real/imaginary values directly.",
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
    }, indent=2))
    return 0


def command_export_veriloga(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = DNN.load(model_dir)
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'dnn'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    fold_scalers = not bool(getattr(args, "no_fold_scalers", False))
    export_z0 = float(model.target_z0 if model.output_domain == "y" else args.z0)
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
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "direct_dnn",
            "fully_self_contained": True,
            "training_output_domain": model.output_domain,
            "training_target_z0": model.target_z0,
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
    return argparse.Namespace(
        mdif=args.mdif,
        verification_mdif=args.verification_mdif,
        out_dir=str(out_dir),
        split_var=args.split_var,
        train_values=args.train_values,
        verify_values=args.verify_values,
        parameter_names=args.parameter_names,
        holdout_fraction=args.holdout_fraction,
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
        worst_plots=plots,
        sparam_weights=args.sparam_weights,
        quiet=True,
    )


def dnn_sweep_trial_worker(payload: tuple[dict[str, object], dict[str, object], str, int, int]) -> dict[str, object]:
    args_values, candidate, out_dir_text, trial_index, plots = payload
    args = argparse.Namespace(**args_values)
    out_dir = Path(out_dir_text)
    trial_dir = out_dir / "trials" / f"trial_{trial_index:04d}"
    error_message = None
    trial_seed = sweep_trial_seed(args.seed, trial_index, getattr(args, "trial_seed_mode", "fixed"))
    try:
        trial_args = namespace_for_trial(args, candidate, trial_dir, trial_index, plots=plots)
        status = command_train(trial_args)
    except Exception as exc:
        status = 2
        error_message = str(exc)
    summary_path = trial_dir / "verification_summary.json"
    if status != 0 or not summary_path.exists():
        summary: dict[str, object] = {"error": error_message or "trial failed"}
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
    return run_sweep_command(
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
    )


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
    train.add_argument("--worst-plots", type=int, default=6)
    train.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    train.set_defaults(func=command_train)

    sweep = sub.add_parser(
        "sweep",
        aliases=["optimize"],
        help="Try multiple DNN configurations and retrain the best one",
    )
    add_data_args(sweep)
    sweep.add_argument("--freq-transform-options", default="log,log-linear")
    sweep.add_argument("--hidden-layer-options", default="64,64;128,128,64;128,128,128;256,128,64")
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
    export_ads.set_defaults(func=command_export_ads)

    export_ann = sub.add_parser(
        "export-ads-ann",
        help="Create an ADS ANN training/extraction package for a native ADS DNN model",
    )
    export_ann.add_argument("--mdif", required=True, help="Training/verification MDIF path")
    export_ann.add_argument("--verification-mdif", help="Optional separate verification MDIF")
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
    export_ann.add_argument("--freq-transform", choices=["log", "linear", "log-linear"])
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
    export_ann.add_argument("--output-prefix", default="dnn_ads_ann")
    export_ann.add_argument("--seed", type=int)
    export_ann.set_defaults(func=command_export_ads_ann)

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
        help=(
            "Optional NAME=SCALE mappings converting ADS/base-unit instance parameters "
            "to model-training units. Example: W=1um,L=1um or all=1um"
        ),
    )
    export_va.add_argument(
        "--no-fold-scalers",
        action="store_true",
        help="Debug option: keep input/output standardization as explicit Verilog-A arithmetic instead of folding it into the neural layers",
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
