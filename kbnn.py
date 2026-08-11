#!/usr/bin/env python3
"""Knowledge-based neural-network trainer for parameterized S-parameter MDIF.

The supported KBNN forms are:

    plain        : NN(geometry, frequency) -> fine S
    residual    : coarse S + NN(geometry, frequency[, coarse S]) -> fine S
    prior-input : NN(geometry, frequency, coarse S) -> fine S

The integrated workflow fits and saves an S-domain DNN for the coarse response,
freezes it, and uses its predictions for fine KBNN fitting.  The same fitted
coarse network drives verification, prediction, and self-contained Verilog-A
export.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from surrogate_common import (  # noqa: E402
    DB_MAG_FLOOR,
    EPS,
    MDIFBlock,
    MLP,
    Standardizer,
    add_dc_export_arguments,
    add_debug_argument,
    ads_ann_activation_enum,
    ads_ann_optimizer_enum,
    ads_ann_output_format_enum,
    ads_ann_training_type_enum,
    apply_distinct_dc_response,
    build_ads_export_blocks,
    build_training_export_commands,
    cleanup_trial_dir,
    common_sparameter_labels,
    configure_parallel_numeric_threads,
    copy_trial_model,
    csv_number,
    debug_print,
    debug_traceback,
    extract_average_dc_resistance,
    frequency_feature_columns,
    frequency_weights_from_blocks,
    infer_parameter_names,
    load_sweep_rows,
    infer_uniform_hidden_layout,
    load_or_write_trial_summary,
    make_training_progress_callback,
    metadata_csv,
    metadata_hidden_layers,
    model_settings_title,
    mse,
    normalize_frequency_weights,
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
    positive_frequency_blocks,
    print_cli_error,
    read_mdif,
    read_model_metadata,
    repository_relative_path,
    resolve_export_dc_metadata,
    rerank_sweep_rows,
    run_sweep_command,
    single_model_train_command,
    sparam_sort_key,
    sparam_weight_mean,
    sparameter_real_imag_columns,
    split_blocks,
    sweep_arg_values,
    sweep_trial_seed,
    terminal_status_line,
    trial_plot_paths,
    verification_metrics,
    veriloga_command_defaults,
    write_training_verification_artifacts,
    write_ads_ann_package,
    write_ads_export_package,
    write_csv,
    write_history,
    write_mdif,
    write_sweep_markdown,
    write_training_markdown,
    update_training_export_commands,
    write_veriloga_package,
)
from dnn import DNN, command_train as command_train_dnn, dnn_export_commands  # noqa: E402


VERSION = "0.2.0-rc3"
COARSE_MODEL_DIRNAME = "coarse_model"
COMPOSITE_MANIFEST_FILENAME = "composite_model_manifest.json"
KBNN_SWEEP_RESULT_COLUMNS = [
    "mode",
    "include_coarse_input",
    "freq_transform",
    "hidden_layers",
    "activation",
    "learning_rate",
]
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


def coarse_dnn_train_namespace(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    coarse_epochs = getattr(args, "coarse_epochs", None)
    coarse_batch_size = getattr(args, "coarse_batch_size", None)
    coarse_patience = getattr(args, "coarse_patience", None)
    coarse_loss_interval = getattr(args, "coarse_loss_interval", None)
    coarse_progress_interval = getattr(args, "coarse_progress_interval", None)
    coarse_seed = getattr(args, "coarse_seed", None)
    coarse_worst_plots = getattr(args, "coarse_worst_plots", None)
    return argparse.Namespace(
        mdif=args.coarse_mdif,
        verification_mdif=getattr(args, "coarse_verification_mdif", None),
        out_dir=str(out_dir),
        split_var=args.split_var,
        train_values=args.train_values,
        verify_values=args.verify_values,
        parameter_names=args.parameter_names,
        holdout_fraction=args.holdout_fraction,
        freq_transform=getattr(args, "coarse_freq_transform", None) or args.freq_transform,
        hidden_layers=getattr(args, "coarse_hidden_layers", "64,64"),
        activation=getattr(args, "coarse_activation", "tanh"),
        epochs=args.epochs if coarse_epochs is None else coarse_epochs,
        batch_size=args.batch_size if coarse_batch_size is None else coarse_batch_size,
        learning_rate=getattr(args, "coarse_learning_rate", 2e-3),
        patience=args.patience if coarse_patience is None else coarse_patience,
        loss_interval=args.loss_interval if coarse_loss_interval is None else coarse_loss_interval,
        progress_interval=(
            args.progress_interval
            if coarse_progress_interval is None
            else coarse_progress_interval
        ),
        progress_label="Coarse DNN fit",
        seed=args.seed if coarse_seed is None else coarse_seed,
        output_domain="s",
        target_z0=50.0,
        worst_plots=(
            getattr(args, "worst_plots", 6)
            if coarse_worst_plots is None
            else coarse_worst_plots
        ),
        sparam_weights=(
            getattr(args, "coarse_sparam_weights", None)
            or getattr(args, "sparam_weights", None)
        ),
        frequency_weights=(
            getattr(args, "coarse_frequency_weights", None)
            or getattr(args, "frequency_weights", None)
        ),
        debug=bool(getattr(args, "debug", False)),
        # The joint KBNN command owns CLI reporting. Suppress the standalone
        # DNN command's multi-line JSON and emit one compact coarse-stage line.
        quiet=True,
    )


def coarse_fit_status_line(model_dir: Path) -> str:
    summary_path = model_dir / "verification_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    passivity = summary.get("passivity")
    passivity = passivity if isinstance(passivity, dict) else {}

    def metric(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{number:.6g}" if math.isfinite(number) else "n/a"

    return " ".join(
        [
            "Coarse DNN fit complete",
            f"RMSE={metric(summary.get('rmse_abs'))}",
            f"pv={metric(passivity.get('violating_points'))}",
            f"sigma={metric(passivity.get('max_singular_value'))}",
        ]
    )


def finish_coarse_fit_status(model_dir: Path) -> None:
    """Replace the live coarse progress line with its final metrics."""

    status = terminal_status_line(coarse_fit_status_line(model_dir))
    sys.stderr.write(f"\r\033[2K{status}\n")
    sys.stderr.flush()


def prepare_fitted_coarse_model(
    args: argparse.Namespace,
    output_root: Path,
) -> argparse.Namespace:
    """Fit the coarse DNN once, or retain an explicitly supplied frozen model."""

    prepared = argparse.Namespace(**vars(args))
    coarse_mdif = getattr(prepared, "coarse_mdif", None)
    coarse_model_dir = getattr(prepared, "coarse_model_dir", None)
    coarse_verification_mdif = getattr(prepared, "coarse_verification_mdif", None)
    if coarse_mdif and coarse_model_dir:
        raise ValueError("Use either --coarse-mdif or --coarse-model-dir, not both")
    if coarse_verification_mdif and not coarse_mdif:
        raise ValueError("--coarse-verification-mdif requires --coarse-mdif")
    if not coarse_mdif:
        prepared.coarse_model_packaged = False
        return prepared

    fitted_dir = (output_root / COARSE_MODEL_DIRNAME).resolve()
    status = command_train_dnn(coarse_dnn_train_namespace(prepared, fitted_dir))
    if status != 0:
        raise RuntimeError(f"Coarse DNN fitting failed with status {status}")
    if not getattr(prepared, "quiet", False):
        finish_coarse_fit_status(fitted_dir)
    prepared.coarse_model_dir = str(fitted_dir)
    prepared.coarse_model_packaged = True
    return prepared


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coarse_dnn_identity(model_dir: Path, model: DNN) -> dict[str, object]:
    resolved = model_dir.expanduser().resolve()
    return {
        "source_model_dir": str(resolved),
        "model_npz_sha256": file_sha256(resolved / "model.npz"),
        "metadata_sha256": file_sha256(resolved / "metadata.json"),
        "output_domain": model.output_domain,
        "parameter_names": list(model.parameter_names),
        "sparam_labels": list(model.sparam_labels),
    }


def load_frozen_coarse_dnn(
    model_dir: Path,
    parameter_names: Sequence[str],
    labels: Sequence[str],
    expected_identity: dict[str, object] | None = None,
) -> tuple[DNN, dict[str, object]]:
    resolved = model_dir.expanduser().resolve()
    model = DNN.load(resolved)
    if model.output_domain != "s":
        raise ValueError(
            "The fitted coarse DNN must be trained with --output-domain s"
        )
    if model.parameter_names != list(parameter_names):
        raise ValueError(
            "The fitted coarse DNN parameter names/order must match the KBNN: "
            f"expected {list(parameter_names)}, got {model.parameter_names}"
        )
    if model.sparam_labels != list(labels):
        raise ValueError(
            "The fitted coarse DNN S-parameter labels/order must match the KBNN: "
            f"expected {list(labels)}, got {model.sparam_labels}"
        )
    identity = coarse_dnn_identity(resolved, model)
    if expected_identity is not None:
        for key in ("model_npz_sha256", "metadata_sha256"):
            expected = expected_identity.get(key)
            actual = identity[key]
            if not expected or str(expected) != str(actual):
                raise ValueError(
                    "The supplied coarse DNN does not match the model used during KBNN "
                    f"training ({key}: expected {expected!r}, got {actual!r})"
                )
    return model, identity


def kbnn_metadata(model_dir: Path) -> dict[str, object]:
    return json.loads((model_dir / "metadata.json").read_text())


def load_matching_coarse_dnn(
    kbnn_model_dir: Path,
    model: "KBNN",
    coarse_model_dir: str | None,
) -> tuple[DNN, dict[str, object]]:
    metadata = kbnn_metadata(kbnn_model_dir)
    expected = metadata.get("coarse_model")
    if not isinstance(expected, dict):
        raise ValueError(
            "This KBNN was not trained against a fitted coarse DNN. Retrain it with "
            "--coarse-mdif or --coarse-model-dir before creating a deployment-matched export."
        )
    selected = coarse_model_dir
    if not selected:
        packaged_relative = expected.get("packaged_relative_model_dir")
        if packaged_relative:
            packaged_candidate = (
                kbnn_model_dir.expanduser().resolve() / str(packaged_relative)
            ).resolve()
            if (
                (packaged_candidate / "model.npz").is_file()
                and (packaged_candidate / "metadata.json").is_file()
            ):
                selected = str(packaged_candidate)
    if not selected:
        selected = str(expected.get("source_model_dir") or "")
    if not selected:
        raise ValueError(
            "A matching fitted coarse DNN is required; pass --coarse-model-dir"
        )
    return load_frozen_coarse_dnn(
        Path(selected),
        model.parameter_names,
        model.sparam_labels,
        expected_identity=expected,
    )


def write_composite_model_manifest(
    model_dir: Path,
    model: "KBNN",
) -> Path:
    resolved_model_dir = model_dir.expanduser().resolve()
    repository_root = Path(__file__).resolve().parent
    metadata = kbnn_metadata(resolved_model_dir)
    coarse_identity = metadata.get("coarse_model")
    coarse_payload: dict[str, object] | None = None
    module_name, parameter_scale_spec = veriloga_command_defaults(
        Path(__file__),
        resolved_model_dir,
    )
    export_argv = [
        Path(sys.executable).name or "python3",
        repository_relative_path(Path(__file__), repository_root),
        "export-veriloga",
        "--model-dir",
        repository_relative_path(resolved_model_dir, repository_root),
        "--out-dir",
        repository_relative_path(resolved_model_dir / "veriloga", repository_root),
        "--module-name",
        module_name,
        "--parameter-input-scales",
        parameter_scale_spec,
    ]
    if isinstance(coarse_identity, dict):
        coarse_path = Path(str(coarse_identity.get("source_model_dir") or ""))
        packaged_relative = coarse_identity.get("packaged_relative_model_dir")
        if packaged_relative:
            packaged_candidate = (resolved_model_dir / str(packaged_relative)).resolve()
            if packaged_candidate.is_dir():
                coarse_path = packaged_candidate
        coarse_metadata = json.loads((coarse_path / "metadata.json").read_text())
        coarse_payload = {
            "role": "frozen_coarse_s_domain_dnn",
            "model_dir": str(coarse_path),
            "packaged_relative_model_dir": packaged_relative,
            "required_files": ["model.npz", "metadata.json"],
            "model_npz_sha256": coarse_identity.get("model_npz_sha256"),
            "metadata_sha256": coarse_identity.get("metadata_sha256"),
            "training_summary": str(coarse_path / "training_summary.md"),
            "verification_summary": str(coarse_path / "verification_summary.json"),
            "dc_equivalent_resistance_ohm": coarse_metadata.get(
                "dc_equivalent_resistance_ohm"
            ),
            "dc_is_separate_from_fitted_response": coarse_metadata.get(
                "dc_is_separate_from_fitted_response",
                False,
            ),
        }
        export_argv.extend(
            [
                "--coarse-model-dir",
                repository_relative_path(coarse_path, repository_root),
            ]
        )
    manifest = {
        "version": VERSION,
        "model_family": "composite_kbnn",
        "mode": model.mode,
        "fit_order": ["coarse_dnn", "fine_kbnn"],
        "fit_contract": (
            "The coarse S-domain DNN is fitted first and frozen. Its predictions, "
            "not raw coarse MDIF samples, are used to fit and verify the fine KBNN."
        ),
        "fine_model": {
            "role": "fine_kbnn_correction_or_prior_network",
            "model_dir": str(resolved_model_dir),
            "required_files": ["model.npz", "metadata.json"],
            "model_npz_sha256": file_sha256(resolved_model_dir / "model.npz"),
            "metadata_sha256": file_sha256(resolved_model_dir / "metadata.json"),
            "training_summary": str(resolved_model_dir / "training_summary.md"),
            "verification_summary": str(resolved_model_dir / "verification_summary.json"),
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_is_separate_from_fitted_response": True,
        },
        "coarse_model": coarse_payload,
        "veriloga_ready": bool(model.mode == "plain" or coarse_payload is not None),
        "veriloga_export_command": shlex.join(export_argv),
    }
    path = resolved_model_dir / COMPOSITE_MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2))
    return path


def set_packaged_coarse_reference(model_dir: Path, coarse_model_dir: Path) -> None:
    """Copy the frozen coarse package beside a promoted fine model and relink it."""

    resolved_model_dir = model_dir.expanduser().resolve()
    resolved_coarse_dir = coarse_model_dir.expanduser().resolve()
    metadata_path = resolved_model_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text())
    coarse_identity = metadata.get("coarse_model")
    if not isinstance(coarse_identity, dict):
        return
    packaged_coarse_dir = resolved_model_dir / COARSE_MODEL_DIRNAME
    copied_coarse_package = False
    if packaged_coarse_dir.resolve() != resolved_coarse_dir:
        if packaged_coarse_dir.exists():
            shutil.rmtree(packaged_coarse_dir)
        shutil.copytree(resolved_coarse_dir, packaged_coarse_dir)
        copied_coarse_package = True
    coarse_report = packaged_coarse_dir / "training_summary.md"
    if copied_coarse_package and coarse_report.is_file():
        update_training_export_commands(
            coarse_report,
            dnn_export_commands(packaged_coarse_dir),
        )
    coarse_identity["source_model_dir"] = str(packaged_coarse_dir)
    coarse_identity["packaged_relative_model_dir"] = os.path.relpath(
        packaged_coarse_dir,
        resolved_model_dir,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2))
    write_composite_model_manifest(resolved_model_dir, KBNN.load(resolved_model_dir))


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
        "coarse_source": "fitted_dnn" if mode != "plain" else None,
        "coarse_model_dir": getattr(args, "coarse_model_dir", None),
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
            f"coarse_model={info.get('coarse_model_dir')}"
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
        dc_equivalent_resistance_ohm: float | None = None,
        dc_resistance_source_kind: str | None = None,
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = parameter_names
        self.sparam_labels = sparam_labels
        self.mode = mode
        self.include_coarse_input = include_coarse_input
        self.freq_transform = freq_transform
        self.dc_equivalent_resistance_ohm = (
            None
            if dc_equivalent_resistance_ohm is None
            else float(dc_equivalent_resistance_ohm)
        )
        self.dc_resistance_source_kind = dc_resistance_source_kind

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
            pred_values = apply_distinct_dc_response(
                pred_values,
                fine.freq_hz,
                self.sparam_labels,
                self.dc_equivalent_resistance_ohm,
                self.dc_resistance_source_kind,
                z0=50.0,
            )
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
            "dc_equivalent_resistance_ohm": self.dc_equivalent_resistance_ohm,
            "dc_resistance_source_kind": self.dc_resistance_source_kind,
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
            dc_equivalent_resistance_ohm=metadata.get("dc_equivalent_resistance_ohm"),
            dc_resistance_source_kind=metadata.get("dc_resistance_source_kind"),
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
    coarse_model_dir = getattr(args, "coarse_model_dir", None)
    if mode == "plain" and coarse_model_dir:
        raise ValueError("--coarse-model-dir is only valid for residual or prior-input KBNNs")
    if mode != "plain" and not coarse_model_dir:
        raise ValueError(
            "Residual and prior-input KBNN training requires --coarse-mdif or "
            "--coarse-model-dir. "
            "The frozen fitted DNN is evaluated on the fine-model grids so training "
            "matches the self-contained export."
        )

    train_fine, verify_fine, all_fine = split_fine_blocks(args)
    if not train_fine:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(all_fine, requested=args.parameter_names, split_var=args.split_var)
    labels = determine_labels(all_fine, None)
    dc_metadata = extract_average_dc_resistance(train_fine, labels, z0=50.0)
    fit_train_fine = positive_frequency_blocks(train_fine)
    fit_verify_fine = positive_frequency_blocks(verify_fine) if verify_fine else []
    coarse_identity: dict[str, object] | None = None
    if mode == "plain":
        train_coarse: list[MDIFBlock] = []
        fit_verify_coarse: list[MDIFBlock] = []
        verify_coarse: list[MDIFBlock] = []
    else:
        coarse_model, coarse_identity = load_frozen_coarse_dnn(
            Path(str(coarse_model_dir)),
            parameter_names,
            labels,
        )
        if getattr(args, "coarse_model_packaged", False):
            coarse_identity["packaged_relative_model_dir"] = os.path.relpath(
                Path(str(coarse_model_dir)).expanduser().resolve(),
                Path(args.out_dir).expanduser().resolve(),
            )
        train_coarse = coarse_model.predict_blocks(fit_train_fine)
        fit_verify_coarse = (
            coarse_model.predict_blocks(fit_verify_fine) if fit_verify_fine else []
        )
        verify_coarse = coarse_model.predict_blocks(verify_fine) if verify_fine else []

    x_train, y_train = make_feature_target_samples(
        fit_train_fine,
        train_coarse,
        parameter_names,
        labels,
        mode,
        include_coarse_input,
        args.freq_transform,
    )
    if fit_verify_fine:
        x_verify, y_verify = make_feature_target_samples(
            fit_verify_fine,
            fit_verify_coarse,
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
    frequency_weight_spec = getattr(args, "frequency_weights", None)
    raw_frequency_weights = frequency_weights_from_blocks(
        fit_train_fine,
        frequency_weight_spec,
    )
    normalized_frequency_weights, frequency_weight_mean = (
        normalize_frequency_weights(raw_frequency_weights)
    )
    if fit_verify_fine:
        raw_verify_frequency_weights = frequency_weights_from_blocks(
            fit_verify_fine,
            frequency_weight_spec,
            require_all_rules_match=False,
        )
        normalized_verify_frequency_weights, _ = normalize_frequency_weights(
            raw_verify_frequency_weights,
            mean=frequency_weight_mean,
        )
    else:
        normalized_verify_frequency_weights = None
    progress_interval = progress_interval_from_args(args)
    initial_train_loss = mse(
        mlp.predict(x_train_scaled),
        y_train_scaled,
        output_weights=output_weights,
        sample_weights=normalized_frequency_weights,
    )
    initial_verify_loss = (
        mse(
            mlp.predict(x_verify_scaled),
            y_verify_scaled,
            output_weights=output_weights,
            sample_weights=normalized_verify_frequency_weights,
        )
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
        sample_weights=normalized_frequency_weights,
        val_sample_weights=normalized_verify_frequency_weights,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_callback=make_training_progress_callback(
            getattr(args, "progress_label", "KBNN fit"),
            args.epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    final_train_loss = mse(
        mlp.predict(x_train_scaled),
        y_train_scaled,
        output_weights=output_weights,
        sample_weights=normalized_frequency_weights,
    )
    final_verify_loss = (
        mse(
            mlp.predict(x_verify_scaled),
            y_verify_scaled,
            output_weights=output_weights,
            sample_weights=normalized_verify_frequency_weights,
        )
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
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=str(dc_metadata["dc_resistance_source_kind"]),
    )
    metadata = {
        "training_blocks": len(train_fine),
        "verification_blocks": len(verify_fine),
        "training_samples": int(x_train.shape[0]),
        "mode": mode,
        "include_coarse_input": include_coarse_input,
        "freq_transform": args.freq_transform,
        "coarse_source": "fitted_dnn" if coarse_identity is not None else None,
        "coarse_model": coarse_identity,
        "split_var": args.split_var,
        "train_values": sorted(parse_csv_set(args.train_values)),
        "verify_values": sorted(parse_csv_set(args.verify_values)),
        "sparam_weights": sparam_weights,
        "normalized_sparam_weights": normalized_sparam_weights,
        "sparam_weight_mean": sparam_weight_mean(labels, sparam_weights),
        "sparam_weight_normalization": "Raw S-parameter weights are divided by their mean before training, so the average normalized weight is 1.0.",
        "frequency_weights": frequency_weight_spec,
        "frequency_weight_mean": frequency_weight_mean,
        "frequency_weight_min": float(np.min(raw_frequency_weights)),
        "frequency_weight_max": float(np.max(raw_frequency_weights)),
        "frequency_weight_normalization": "Raw frequency weights are divided by their mean over fitted training samples, so the average normalized weight is 1.0.",
        "output_scaler_floor": output_std_floor,
        "floored_output_columns": floored_output_columns,
        **dc_metadata,
    }
    if getattr(args, "debug", False):
        debug_info = build_training_debug_info(
            args,
            mode,
            include_coarse_input,
            parameter_names,
            labels,
            fit_train_fine,
            fit_verify_fine,
            train_coarse,
            fit_verify_coarse,
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


def kbnn_export_commands(
    model_dir: Path,
    template_mdif: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Build runnable export commands for a fitted composite KBNN report."""

    return build_training_export_commands(
        Path(__file__),
        model_dir,
        template_mdif,
        include_veriloga=True,
    )


def command_train(args: argparse.Namespace) -> int:
    if normalize_mode(args.mode) == "plain" and (
        getattr(args, "coarse_mdif", None) or getattr(args, "coarse_model_dir", None)
    ):
        raise ValueError("Coarse-model fitting is only valid for residual or prior-input KBNNs")
    args = prepare_fitted_coarse_model(args, Path(args.out_dir))
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
        "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
        "dc_resistance_source_kind": metadata["dc_resistance_source_kind"],
        "dc_resistance_pair_means_ohm": metadata["dc_resistance_pair_means_ohm"],
        "dc_resistance_extraction": metadata["dc_resistance_extraction"],
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
        "coarse_source": metadata["coarse_source"],
        "coarse_model": metadata["coarse_model"],
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
        "frequency_weights": metadata["frequency_weights"],
        "frequency_weight_mean": metadata["frequency_weight_mean"],
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
            frequency_weights=getattr(args, "frequency_weights", None),
            y_z0=50.0,
            title_context=plot_context,
        )
    else:
        summary = {"warning": "No verification blocks were available"}
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    export_commands = kbnn_export_commands(out_dir, args.mdif)
    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="KBNN",
        config=training_config,
        summary=summary,
        history=history,
        export_commands=export_commands,
    )
    composite_manifest = write_composite_model_manifest(out_dir, model)

    if not getattr(args, "quiet", False):
        print(json.dumps({
            "out_dir": str(out_dir),
            "training_summary": str(out_dir / "training_summary.md"),
            "composite_manifest": str(composite_manifest),
            "mode": model.mode,
            "include_coarse_input": model.include_coarse_input,
            "coarse_source": metadata["coarse_source"],
            "coarse_model": metadata["coarse_model"],
            "parameters": parameter_names,
            "sparameters": labels,
            "sparam_weights": metadata["sparam_weights"],
            "normalized_sparam_weights": metadata["normalized_sparam_weights"],
            "sparam_weight_mean": metadata["sparam_weight_mean"],
            "frequency_weights": metadata["frequency_weights"],
            "frequency_weight_mean": metadata["frequency_weight_mean"],
            "output_scaler_floor": metadata["output_scaler_floor"],
            "floored_output_columns": metadata["floored_output_columns"],
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_resistance_source_kind": metadata["dc_resistance_source_kind"],
            "dc_resistance_pair_means_ohm": metadata["dc_resistance_pair_means_ohm"],
            "export_commands": dict(export_commands),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
        }, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = KBNN.load(model_dir)
    blocks = read_mdif_cached(args.mdif)
    if model.mode == "plain":
        if args.coarse_model_dir:
            raise ValueError("--coarse-model-dir is only valid for residual or prior-input KBNN models")
        coarse = []
    else:
        coarse_model, _ = load_matching_coarse_dnn(
            model_dir,
            model,
            args.coarse_model_dir,
        )
        coarse = coarse_model.predict_blocks(blocks)
    pred_blocks = model.predict_blocks(blocks, coarse)
    out_path = Path(args.out_mdif)
    write_mdif(out_path, pred_blocks, model.sparam_labels)
    print(f"Wrote {out_path}")
    return 0


def command_export_ads(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = KBNN.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    dc_metadata = resolve_export_dc_metadata(
        source_metadata,
        model.sparam_labels,
        dc_mdif=args.dc_mdif,
        z0=50.0,
        open_threshold_ohm=args.dc_open_threshold,
        open_resistance_ohm=args.dc_open_resistance,
    )
    model.dc_equivalent_resistance_ohm = float(
        dc_metadata["dc_equivalent_resistance_ohm"]
    )
    model.dc_resistance_source_kind = str(dc_metadata["dc_resistance_source_kind"])

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
        if args.coarse_model_dir:
            raise ValueError("--coarse-model-dir is only valid for residual or prior-input KBNN models")
        coarse = []
        coarse_identity = None
    else:
        coarse_model, coarse_identity = load_matching_coarse_dnn(
            model_dir,
            model,
            args.coarse_model_dir,
        )
        coarse_model.dc_equivalent_resistance_ohm = model.dc_equivalent_resistance_ohm
        coarse_model.dc_resistance_source_kind = model.dc_resistance_source_kind
        coarse = coarse_model.predict_blocks(blocks)
    pred_blocks = model.predict_blocks(blocks, coarse)
    write_mdif(out_dir / mdif_name, pred_blocks, model.sparam_labels)

    notes = []
    notes.append(
        "Every exported block includes a zero-Hz point from the resolved passive "
        "exact-DC average resistance; it bypasses both fitted networks."
    )
    if coarse_identity is not None:
        notes.append(
            "The same frozen coarse DNN used during KBNN training was evaluated during export; ADS only needs the final exported MDIF."
        )
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
            "coarse_source": "fitted_dnn" if coarse_identity is not None else None,
            "coarse_model": coarse_identity,
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_metadata": dc_metadata,
            "dc_is_separate_from_fitted_response": True,
        },
        extra_notes=notes,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mdif": str(out_dir / mdif_name),
        "manifest": str(out_dir / "ads_model_manifest.json"),
        "blocks": manifest["blocks"],
        "frequency_points_per_block": manifest["frequency_points_per_block"],
        "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
        "dc_ignored_nonpassive_count": dc_metadata.get(
            "dc_ignored_nonpassive_count"
        ),
        "dc_open_circuit_applied": dc_metadata.get("dc_open_circuit_applied"),
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
    train_fine = positive_frequency_blocks(train_fine, purpose="ADS ANN fitting")
    verify_fine = (
        positive_frequency_blocks(verify_fine, purpose="ADS ANN verification")
        if verify_fine
        else []
    )
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
        "Zero-Hz rows are excluded from ADS ANN fitting. Use the self-contained Verilog-A or sampled-MDIF export when the distinct saved DC resistance is required.",
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
    if model_metadata.get("frequency_weights"):
        notes.append(
            "The source model's frequency weights are recorded in the manifest but are not applied because the documented ADS ANN API does not expose per-sample loss weights."
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
            "source_frequency_weights": model_metadata.get("frequency_weights"),
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
    source_metadata = read_model_metadata(str(model_dir))
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'kbnn'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    dc_metadata = resolve_export_dc_metadata(
        source_metadata,
        model.sparam_labels,
        dc_mdif=args.dc_mdif,
        z0=args.z0,
        open_threshold_ohm=args.dc_open_threshold,
        open_resistance_ohm=args.dc_open_resistance,
    )
    uses_coarse_inputs = bool(model.include_coarse_input or model.mode == "prior-input")
    adds_coarse_to_output = model.mode == "residual"
    needs_coarse_response = bool(uses_coarse_inputs or adds_coarse_to_output)
    allow_coarse_hooks = bool(getattr(args, "allow_coarse_hooks", False))
    if args.coarse_model_dir and not needs_coarse_response:
        raise ValueError(
            "--coarse-model-dir is only valid for residual or prior-input KBNN models"
        )
    coarse_model: DNN | None = None
    coarse_identity: dict[str, object] | None = None
    if needs_coarse_response and (args.coarse_model_dir or not allow_coarse_hooks):
        coarse_model, coarse_identity = load_matching_coarse_dnn(
            model_dir,
            model,
            args.coarse_model_dir,
        )
    embedded_coarse_model = None
    if coarse_model is not None and coarse_identity is not None:
        embedded_coarse_model = {
            "source_model_dir": str(coarse_identity["source_model_dir"]),
            "parameter_names": coarse_model.parameter_names,
            "sparam_labels": coarse_model.sparam_labels,
            "freq_transform": coarse_model.freq_transform,
            "activation": coarse_model.mlp.activation,
            "layer_sizes": coarse_model.mlp.layer_sizes,
            "weights": coarse_model.mlp.weights,
            "biases": coarse_model.mlp.biases,
            "x_mean": np.asarray(coarse_model.x_scaler.mean, dtype=float),
            "x_std": np.asarray(coarse_model.x_scaler.std, dtype=float),
            "y_mean": np.asarray(coarse_model.y_scaler.mean, dtype=float),
            "y_std": np.asarray(coarse_model.y_scaler.std, dtype=float),
            "output_domain": coarse_model.output_domain,
        }
    notes = [
        "This direct Verilog-A export embeds the saved local model.npz weights; it does not retrain in ADS ANN.",
        "The generated N-port is intended for S-parameter and small-signal AC simulation. It is not a causal transient model.",
        "The default frequency expression is $freq. If your ADS Verilog-A environment uses a different frequency variable, regenerate with --frequency-expression.",
        "At exactly zero frequency, both the fine KBNN and embedded coarse DNN are bypassed and the fine-data average equivalent resistance is used instead.",
    ]
    if coarse_model is not None:
        notes.append(
            "This package embeds the exact frozen coarse S-domain DNN used during KBNN training and is fully self-contained; ADS only supplies geometry/process parameters and simulator frequency."
        )
    elif needs_coarse_response:
        notes.append(
            "Legacy coarse-response hooks were explicitly requested. This package is not self-contained and the zero defaults are only suitable for a fixed-point diagnostic."
        )
    if model.mode == "residual":
        notes.append(
            "Residual KBNN output is delta S internally; the generated N-port adds the embedded coarse response before converting final S to Y."
            if coarse_model is not None
            else "Residual KBNN output is delta S internally; the generated N-port adds the coarse hook response before converting final S to Y."
        )
    elif model.mode == "prior-input":
        notes.append(
            "Prior-input KBNN output is final fine S; the generated model feeds the embedded coarse response into the KBNN."
            if coarse_model is not None
            else "Prior-input KBNN output is final fine S, but the ANN inputs still require the coarse hook response."
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
        embedded_coarse_model=embedded_coarse_model,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=dc_metadata.get("dc_resistance_source_kind"),
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "knowledge_based_neural_network",
            "mode": model.mode,
            "include_coarse_input": model.include_coarse_input,
            "coarse_source": "fitted_dnn" if coarse_identity is not None else None,
            "coarse_model": coarse_identity,
            "coarse_model_dir": (
                str(coarse_identity["source_model_dir"])
                if coarse_identity is not None
                else None
            ),
            "coarse_model_match_verified": coarse_identity is not None,
            "fully_self_contained": bool(not needs_coarse_response or coarse_model is not None),
            "requires_coarse_hooks": bool(needs_coarse_response and coarse_model is None),
            "dc_resistance_source_kind": dc_metadata.get(
                "dc_resistance_source_kind"
            ),
            "dc_resistance_pair_means_ohm": dc_metadata.get(
                "dc_resistance_pair_means_ohm"
            ),
            "dc_metadata": dc_metadata,
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
        "coarse_model_dir": manifest["coarse_model_dir"],
        "coarse_model_match_verified": manifest["coarse_model_match_verified"],
        "fully_self_contained": manifest["fully_self_contained"],
        "requires_coarse_hooks": manifest["requires_coarse_hooks"],
        "dc_equivalent_resistance_ohm": manifest["dc_equivalent_resistance_ohm"],
        "dc_resistance_source_kind": manifest["dc_resistance_source_kind"],
        "dc_resistance_pair_means_ohm": manifest["dc_resistance_pair_means_ohm"],
        "dc_ignored_nonpassive_count": dc_metadata.get(
            "dc_ignored_nonpassive_count"
        ),
        "dc_open_circuit_applied": dc_metadata.get("dc_open_circuit_applied"),
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
    freq_transform_options = (
        parse_text_options(args.freq_transform_options)
        if getattr(args, "freq_transform_options", None)
        else [args.freq_transform]
    )
    for freq_transform in freq_transform_options:
        if freq_transform not in {"log", "linear"}:
            raise ValueError(f"Unsupported frequency transform {freq_transform!r}")
    axes = {
        "mode": [
            normalize_mode(value)
            for value in parse_text_options(
                getattr(args, "mode_options", None) or "residual,prior-input"
            )
        ],
        "include_coarse_input": [parse_bool_option(value) for value in parse_text_options(args.include_coarse_input_options)],
        "freq_transform": freq_transform_options,
        "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options),
        "activation": parse_text_options(args.activation_options),
        "learning_rate": parse_float_options(args.learning_rates),
    }
    candidates = []
    keys = list(axes)
    skipped_without_coarse_model = 0
    for values in itertools.product(*(axes[key] for key in keys)):
        candidate = dict(zip(keys, values))
        if candidate["mode"] == "plain" and candidate["include_coarse_input"]:
            continue
        if candidate["mode"] == "prior-input" and not candidate["include_coarse_input"]:
            continue
        if not args.coarse_model_dir and candidate["mode"] != "plain":
            skipped_without_coarse_model += 1
            continue
        candidates.append(candidate)
    if skipped_without_coarse_model:
        print(
            "warning: skipped "
            f"{skipped_without_coarse_model} KBNN candidate(s) that require "
            "--coarse-mdif or --coarse-model-dir",
            file=sys.stderr,
        )
    if not candidates:
        raise ValueError(
            "No valid KBNN sweep candidates. Residual and prior-input modes require "
            "--coarse-mdif or --coarse-model-dir so the sweep uses the fitted "
            "deployment model."
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
        coarse_model_dir=args.coarse_model_dir,
        out_dir=str(out_dir),
        split_var=args.split_var,
        train_values=args.train_values,
        verify_values=args.verify_values,
        parameter_names=args.parameter_names,
        holdout_fraction=args.holdout_fraction,
        mode=str(candidate["mode"]),
        include_coarse_input=bool(candidate["include_coarse_input"]),
        freq_transform=str(candidate["freq_transform"]),
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
        frequency_weights=args.frequency_weights,
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
    compatibility_mode = getattr(args, "mode", None)
    model_modes = getattr(args, "mode_options", None)
    search_mode = getattr(args, "search_mode", "random")
    if compatibility_mode in {"grid", "random"}:
        search_mode = compatibility_mode
    elif compatibility_mode is not None:
        if model_modes is not None:
            raise ValueError("Use either --mode for one KBNN mode or --modes for several, not both")
        model_modes = compatibility_mode
    args = argparse.Namespace(**vars(args))
    args.mode = search_mode
    args.mode_options = model_modes or "residual,prior-input"
    integrated_coarse_fit = bool(getattr(args, "coarse_mdif", None))
    prepared_args = prepare_fitted_coarse_model(args, Path(args.out_dir))
    # Trial directories are transient, so package the shared coarse model only
    # after the winning fine model has been promoted into best_model/.
    prepared_args.coarse_model_packaged = False
    status = run_sweep_command(
        prepared_args,
        sweep_candidate_grid(prepared_args),
        worker_func=kbnn_sweep_trial_worker,
        namespace_for_trial_func=namespace_for_trial,
        train_func=command_train,
        result_columns=KBNN_SWEEP_RESULT_COLUMNS,
        results_filename="kbnn_sweep_results.csv",
        best_config_filename="kbnn_best_config.json",
        summary_filename="kbnn_sweep_summary.md",
        diagnostics_prefix="kbnn",
        train_command_prefix=None,
    )
    if integrated_coarse_fit and prepared_args.coarse_model_dir:
        set_packaged_coarse_reference(
            Path(prepared_args.out_dir) / "best_model",
            Path(prepared_args.coarse_model_dir),
        )
    best_dir = Path(prepared_args.out_dir) / "best_model"
    update_training_export_commands(
        best_dir / "training_summary.md",
        kbnn_export_commands(best_dir, args.mdif),
    )
    best_config_path = Path(prepared_args.out_dir) / "kbnn_best_config.json"
    best_payload = json.loads(best_config_path.read_text())
    best_candidate = dict(best_payload["config"])
    best_uses_coarse = normalize_mode(str(best_candidate["mode"])) != "plain"
    reproduction_args = namespace_for_trial(
        prepared_args,
        best_candidate,
        Path(prepared_args.out_dir) / "reproduced_model",
        int(best_payload["trial"]),
        plots=prepared_args.worst_plots,
    )
    if not best_uses_coarse:
        reproduction_args.coarse_model_dir = None
    elif integrated_coarse_fit:
        reproduction_args.coarse_model_dir = None
        reproduction_args.coarse_mdif = args.coarse_mdif
        reproduction_args.coarse_verification_mdif = getattr(
            args,
            "coarse_verification_mdif",
            None,
        )
        for name in (
            "coarse_hidden_layers",
            "coarse_activation",
            "coarse_freq_transform",
            "coarse_learning_rate",
            "coarse_epochs",
            "coarse_batch_size",
            "coarse_patience",
            "coarse_loss_interval",
            "coarse_progress_interval",
            "coarse_seed",
            "coarse_worst_plots",
            "coarse_sparam_weights",
            "coarse_frequency_weights",
        ):
            setattr(reproduction_args, name, getattr(args, name, None))
    reproduction_command = single_model_train_command(
        [sys.executable, "kbnn.py", "train"],
        reproduction_args,
        Path(prepared_args.out_dir) / "reproduced_model",
    )
    best_payload["reproduction_command"] = reproduction_command
    best_config_path.write_text(json.dumps(best_payload, indent=2))
    summary_path = Path(prepared_args.out_dir) / "kbnn_sweep_summary.md"
    summary_text = summary_path.read_text().rstrip()
    summary_path.write_text(
        f"{summary_text}\n\n## Reproduce Best Model\n\n```bash\n"
        f"{reproduction_command}\n```\n"
    )
    update_training_export_commands(
        summary_path,
        kbnn_export_commands(best_dir, args.mdif),
    )
    print("reproduce best model:", flush=True)
    print(reproduction_command, flush=True)
    return status


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
        packaged_coarse_dir = sweep_dir / COARSE_MODEL_DIRNAME
        if promoted and best_model_dir is not None and packaged_coarse_dir.is_dir():
            set_packaged_coarse_reference(best_model_dir, packaged_coarse_dir)
        if promoted and best_model_dir is not None:
            export_commands = kbnn_export_commands(best_model_dir)
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
    coarse_source = parser.add_mutually_exclusive_group()
    coarse_source.add_argument(
        "--coarse-mdif",
        help=(
            "Coarse/prior S-parameter MDIF. Fits and saves an S-domain DNN under "
            "<out-dir>/coarse_model before fitting the KBNN."
        ),
    )
    coarse_source.add_argument(
        "--coarse-model-dir",
        help=(
            "Reuse an existing frozen S-domain DNN instead of fitting --coarse-mdif. "
            "Residual and prior-input modes require one of these two coarse sources."
        ),
    )
    parser.add_argument(
        "--coarse-verification-mdif",
        help="Optional separate verification MDIF used only while fitting --coarse-mdif",
    )
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
    coarse_fit = parser.add_argument_group("integrated coarse DNN fitting")
    coarse_fit.add_argument("--coarse-hidden-layers", default="64,64")
    coarse_fit.add_argument("--coarse-activation", choices=["tanh", "relu"], default="tanh")
    coarse_fit.add_argument(
        "--coarse-freq-transform",
        choices=["log", "linear", "log-linear"],
        help="Coarse-DNN frequency transform. Defaults to --freq-transform.",
    )
    coarse_fit.add_argument("--coarse-learning-rate", type=float, default=2e-3)
    coarse_fit.add_argument("--coarse-epochs", type=int, help="Defaults to --epochs")
    coarse_fit.add_argument("--coarse-batch-size", type=int, help="Defaults to --batch-size")
    coarse_fit.add_argument("--coarse-patience", type=int, help="Defaults to --patience")
    coarse_fit.add_argument("--coarse-loss-interval", type=int, help="Defaults to --loss-interval")
    coarse_fit.add_argument(
        "--coarse-progress-interval",
        type=int,
        help="Defaults to --progress-interval",
    )
    coarse_fit.add_argument("--coarse-seed", type=int, help="Defaults to --seed")
    coarse_fit.add_argument(
        "--coarse-worst-plots",
        type=int,
        help="Defaults to --worst-plots",
    )
    coarse_fit.add_argument(
        "--coarse-sparam-weights",
        help="Optional coarse-DNN S-parameter weights. Defaults to --sparam-weights.",
    )
    coarse_fit.add_argument(
        "--coarse-frequency-weights",
        help="Optional coarse-DNN frequency weights. Defaults to --frequency-weights.",
    )
    add_debug_argument(
        parser,
        (
            "Print common sweep diagnostics plus KBNN data/loss diagnostics, "
            "and write kbnn_training_debug.json in each training output directory."
        ),
    )
    parser.set_defaults(debug_label="KBNN debug")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a coarse S-domain DNN and fine knowledge-based neural network as one Verilog-A-ready workflow."
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
    train.add_argument(
        "--frequency-weights",
        help="Frequency loss weights, e.g. 'default=1;1GHz=5;2GHz:4GHz=3'. Exact frequencies and inclusive ranges are supported; later rules override earlier ones.",
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
    sweep.add_argument(
        "--modes",
        "--mode-options",
        dest="mode_options",
        help="Comma-separated KBNN model modes. Use train-compatible --mode for one value.",
    )
    sweep.add_argument(
        "--mode",
        choices=["plain", "residual", "prior-input", "grid", "random"],
        help="One train-compatible KBNN model mode; grid/random remain accepted for legacy search-mode commands.",
    )
    sweep.add_argument(
        "--include-coarse-inputs",
        "--include-coarse-input-options",
        dest="include_coarse_input_options",
        default="false,true",
        help="Comma-separated false/true candidate values.",
    )
    sweep.add_argument(
        "--include-coarse-input",
        dest="include_coarse_input_options",
        action="store_const",
        const="true",
        default=argparse.SUPPRESS,
        help="Train-compatible single true value for the optimize candidate set.",
    )
    sweep.add_argument(
        "--freq-transforms",
        "--freq-transform-options",
        dest="freq_transform_options",
        help=(
            "Comma-separated frequency transforms to try. Available values are log and linear. "
            "If omitted, the sweep uses --freq-transform."
        ),
    )
    sweep.add_argument(
        "--hidden-layers",
        "--hidden-layer-layouts",
        "--hidden-layer-options",
        dest="hidden_layer_options",
        default="32;64;64,64",
        help="One train-style layout or semicolon-separated hidden-layer layouts.",
    )
    sweep.add_argument(
        "--activations",
        "--activation-options",
        "--activation",
        dest="activation_options",
        default="tanh,relu",
        help="Comma-separated activations; --activation accepts one value as in train.",
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
        choices=["grid", "random"],
        default="random",
        help="Sweep search strategy. Legacy --mode grid/random is still accepted.",
    )
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
    predict.add_argument(
        "--coarse-model-dir",
        help="Matching fitted coarse DNN; defaults to the path recorded during KBNN training",
    )
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
        "--coarse-model-dir",
        help="Matching fitted coarse DNN; defaults to the path recorded during KBNN training",
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
    export_va.add_argument(
        "--coarse-model-dir",
        help=(
            "Matching frozen S-domain DNN used during KBNN training. Defaults to the "
            "recorded training path; hashes must match for a self-contained export."
        ),
    )
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
            "Optional positive scale applied to every ADS/base-unit instance parameter "
            "before conversion to model-training units. Example: 1um"
        ),
    )
    export_va.add_argument(
        "--allow-coarse-hooks",
        action="store_true",
        help=(
            "Allow the legacy non-self-contained residual/prior-input export with "
            "zero-default coarse response hooks when --coarse-model-dir is omitted"
        ),
    )
    add_dc_export_arguments(export_va)
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
        print_cli_error(args, exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
