#!/usr/bin/env python3
"""Neuro-TF trainer, predictor, sweep, and ADS export CLI."""

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

from surrogate_common import *  # noqa: F401,F403,E402

VERSION = "0.2.0-rc2"

def build_fixed_poles(
    blocks: Sequence[MDIFBlock],
    n_poles: int,
    damping: float,
) -> tuple[np.ndarray, float]:
    all_freq = np.concatenate([block.freq_hz for block in blocks])
    positive = all_freq[all_freq > 0]
    if positive.size == 0:
        raise ValueError("Frequencies must include at least one positive value")
    f_min = float(np.min(positive))
    f_max = float(np.max(positive))
    f_scale = math.sqrt(f_min * f_max)
    x_min = max(f_min / f_scale, 1e-6)
    x_max = max(f_max / f_scale, x_min * 1.01)

    poles: list[complex] = []
    n_pairs = n_poles // 2
    if n_pairs:
        centers = np.logspace(math.log10(x_min * 0.75), math.log10(x_max * 1.25), n_pairs)
        for center in centers:
            poles.append(complex(-damping * center, center))
            poles.append(complex(-damping * center, -center))
    if len(poles) < n_poles:
        poles.append(complex(-math.sqrt(x_min * x_max), 0.0))
    return np.asarray(poles[:n_poles], dtype=complex), f_scale


def rational_basis(freq_hz: np.ndarray, poles: np.ndarray, f_scale: float) -> np.ndarray:
    s = 1j * (freq_hz / f_scale)
    columns = [np.ones_like(s, dtype=complex)]
    for pole in poles:
        columns.append(1.0 / (s - pole))
    return np.column_stack(columns)


def fit_rational_coeffs(
    freq_hz: np.ndarray,
    values: np.ndarray,
    poles: np.ndarray,
    f_scale: float,
    ridge: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    basis = rational_basis(freq_hz, poles, f_scale)
    weighted_values = np.asarray(values, dtype=complex)
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float).reshape(-1)
        if weights.shape != (basis.shape[0],):
            raise ValueError(
                f"Expected {basis.shape[0]} rational-fit frequency weights, got "
                f"{weights.shape}"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "Rational-fit frequency weights must be finite and non-negative"
            )
        root_weights = np.sqrt(weights)
        basis = basis * root_weights[:, None]
        weighted_values = weighted_values * root_weights
    if ridge > 0:
        reg = math.sqrt(ridge) * np.eye(basis.shape[1], dtype=complex)
        rhs = np.concatenate(
            [weighted_values, np.zeros(basis.shape[1], dtype=complex)]
        )
        lhs = np.vstack([basis, reg])
    else:
        lhs = basis
        rhs = weighted_values
    coeffs, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return coeffs


def fit_all_coefficients(
    blocks: Sequence[MDIFBlock],
    sparam_labels: Sequence[str],
    poles: np.ndarray,
    f_scale: float,
    ridge: float,
    frequency_weights: np.ndarray | None = None,
) -> np.ndarray:
    rows = []
    offset = 0
    for block in blocks:
        end = offset + len(block.freq_hz)
        block_weights = (
            None
            if frequency_weights is None
            else np.asarray(frequency_weights, dtype=float)[offset:end]
        )
        complex_coeffs = []
        for label in sparam_labels:
            coeffs = fit_rational_coeffs(
                block.freq_hz,
                block.sparams[label],
                poles,
                f_scale,
                ridge,
                sample_weights=block_weights,
            )
            complex_coeffs.append(coeffs)
        flat = np.concatenate(complex_coeffs)
        rows.append(np.concatenate([flat.real, flat.imag]))
        offset = end
    if frequency_weights is not None and offset != len(frequency_weights):
        raise ValueError(
            f"Expected {offset} rational-fit frequency weights, got "
            f"{len(frequency_weights)}"
        )
    return np.asarray(rows, dtype=float)


def unflatten_coefficients(row: np.ndarray, n_sparams: int, n_coeffs: int) -> np.ndarray:
    half = n_sparams * n_coeffs
    complex_flat = row[:half] + 1j * row[half : 2 * half]
    return complex_flat.reshape(n_sparams, n_coeffs)


def evaluate_coefficients(
    coeffs: np.ndarray,
    freq_hz: np.ndarray,
    poles: np.ndarray,
    f_scale: float,
) -> np.ndarray:
    basis = rational_basis(freq_hz, poles, f_scale)
    return coeffs @ basis.T



class NeuroTF:
    def __init__(
        self,
        mlp: MLP,
        x_scaler: Standardizer,
        y_scaler: Standardizer,
        parameter_names: list[str],
        sparam_labels: list[str],
        poles: np.ndarray,
        f_scale: float,
        dc_equivalent_resistance_ohm: float | None = None,
    ) -> None:
        self.mlp = mlp
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.parameter_names = parameter_names
        self.sparam_labels = sparam_labels
        self.poles = poles
        self.f_scale = f_scale
        self.dc_equivalent_resistance_ohm = (
            None
            if dc_equivalent_resistance_ohm is None
            else float(dc_equivalent_resistance_ohm)
        )

    @property
    def n_coeffs(self) -> int:
        return len(self.poles) + 1

    def predict_coeff_rows(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.x_scaler.transform(x)
        y_scaled = self.mlp.predict(x_scaled)
        return self.y_scaler.inverse_transform(y_scaled)

    def predict_blocks(self, blocks: Sequence[MDIFBlock]) -> list[MDIFBlock]:
        x = parameter_matrix(blocks, self.parameter_names)
        coeff_rows = self.predict_coeff_rows(x)
        predicted_blocks = []
        for block, row in zip(blocks, coeff_rows):
            coeffs = unflatten_coefficients(row, len(self.sparam_labels), self.n_coeffs)
            values_by_label = evaluate_coefficients(coeffs, block.freq_hz, self.poles, self.f_scale)
            values_by_label = apply_distinct_dc_response(
                values_by_label.T,
                block.freq_hz,
                self.sparam_labels,
                self.dc_equivalent_resistance_ohm,
                z0=50.0,
            ).T
            sparams = {
                label: values_by_label[idx, :]
                for idx, label in enumerate(self.sparam_labels)
            }
            predicted_blocks.append(
                MDIFBlock(
                    params=dict(block.params),
                    freq_hz=block.freq_hz.copy(),
                    sparams=sparams,
                    source_index=block.source_index,
                )
            )
        return predicted_blocks

    def save(self, out_dir: Path, metadata: dict[str, object]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "x_mean": self.x_scaler.mean,
            "x_std": self.x_scaler.std,
            "y_mean": self.y_scaler.mean,
            "y_std": self.y_scaler.std,
            "poles_real": self.poles.real,
            "poles_imag": self.poles.imag,
            "f_scale": np.asarray([self.f_scale], dtype=float),
        }
        for idx, (weight, bias) in enumerate(zip(self.mlp.weights, self.mlp.biases)):
            arrays[f"W{idx}"] = weight
            arrays[f"b{idx}"] = bias
        np.savez_compressed(out_dir / "model.npz", **arrays)
        combined_metadata = {
            "version": VERSION,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "n_poles": int(len(self.poles)),
            "n_coeffs_per_sparam": int(self.n_coeffs),
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation,
            "dc_equivalent_resistance_ohm": self.dc_equivalent_resistance_ohm,
            **metadata,
        }
        (out_dir / "metadata.json").write_text(json.dumps(combined_metadata, indent=2))

    @staticmethod
    def load(model_dir: Path) -> "NeuroTF":
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
        poles = data["poles_real"] + 1j * data["poles_imag"]
        f_scale = float(data["f_scale"][0])
        return NeuroTF(
            mlp=mlp,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            parameter_names=list(metadata["parameter_names"]),
            sparam_labels=list(metadata["sparam_labels"]),
            poles=poles,
            f_scale=f_scale,
            dc_equivalent_resistance_ohm=metadata.get("dc_equivalent_resistance_ohm"),
        )


def sweep_candidate_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    axes = {
        "order": parse_int_options(args.orders),
        "pole_damping": parse_float_options(args.pole_dampings),
        "ridge": parse_float_options(args.ridge_values),
        "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options),
        "activation": parse_text_options(args.activation_options),
        "learning_rate": parse_float_options(args.learning_rates),
    }
    candidates = []
    keys = list(axes)
    for values in itertools.product(*(axes[key] for key in keys)):
        candidates.append(dict(zip(keys, values)))
    if args.mode == "random" and args.max_trials and args.max_trials < len(candidates):
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(candidates), size=args.max_trials, replace=False)
        candidates = [candidates[int(idx)] for idx in chosen]
    elif args.max_trials and args.max_trials < len(candidates):
        candidates = candidates[: args.max_trials]
    return candidates


def namespace_for_trial(args: argparse.Namespace, candidate: dict[str, object], out_dir: Path, trial_index: int, plots: int) -> argparse.Namespace:
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
        order=int(candidate["order"]),
        pole_damping=float(candidate["pole_damping"]),
        ridge=float(candidate["ridge"]),
        hidden_layers=str(candidate["hidden_layers"]),
        activation=str(candidate["activation"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=float(candidate["learning_rate"]),
        patience=args.patience,
        loss_interval=args.loss_interval,
        progress_interval=args.progress_interval,
        progress_label=f"Neuro-TF trial {trial_index}",
        seed=trial_seed,
        worst_plots=plots,
        frequency_weights=args.frequency_weights,
        debug=bool(getattr(args, "debug", False)),
        quiet=True,
    )


def neurotf_sweep_trial_worker(payload: tuple[dict[str, object], dict[str, object], str, int, int]) -> dict[str, object]:
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
        worker_func=neurotf_sweep_trial_worker,
        namespace_for_trial_func=namespace_for_trial,
        train_func=command_train,
        result_columns=["order", "pole_damping", "ridge", "hidden_layers", "activation", "learning_rate"],
        results_filename="neurotf_sweep_results.csv",
        best_config_filename="neurotf_best_config.json",
        summary_filename="neurotf_sweep_summary.md",
        diagnostics_prefix="neurotf",
        train_command_prefix=[sys.executable, "neuro_tf.py", "train"],
    )
    best_dir = Path(args.out_dir) / "best_model"
    update_training_export_commands(
        best_dir / "training_summary.md",
        neurotf_export_commands(best_dir, args.mdif),
    )
    update_training_export_commands(
        Path(args.out_dir) / "neurotf_sweep_summary.md",
        neurotf_export_commands(best_dir, args.mdif),
    )
    return status


def neurotf_export_commands(
    model_dir: Path,
    template_mdif: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Build runnable direct-Verilog-A and sampled-MDIF Neuro-TF commands."""

    return build_training_export_commands(
        Path(__file__),
        model_dir,
        template_mdif,
        include_veriloga=True,
    )


def command_train(args: argparse.Namespace) -> int:
    mdif_blocks = read_mdif(Path(args.mdif))
    if args.verification_mdif:
        train_blocks = mdif_blocks
        verify_blocks = read_mdif(Path(args.verification_mdif))
        split_data = SplitData(train=train_blocks, verify=verify_blocks, all_blocks=train_blocks + verify_blocks)
    else:
        split_data = split_blocks(
            mdif_blocks,
            split_var=args.split_var,
            train_values=parse_csv_set(args.train_values),
            verify_values=parse_csv_set(args.verify_values),
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
        train_blocks = split_data.train
        verify_blocks = split_data.verify

    if not train_blocks:
        raise ValueError("No training blocks found")

    parameter_names = infer_parameter_names(
        split_data.all_blocks,
        requested=args.parameter_names,
        split_var=args.split_var,
    )
    sparam_labels = common_sparameter_labels(split_data.all_blocks)
    dc_metadata = extract_average_dc_resistance(train_blocks, sparam_labels, z0=50.0)
    fit_train_blocks = positive_frequency_blocks(train_blocks)
    fit_verify_blocks = positive_frequency_blocks(verify_blocks) if verify_blocks else []
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
    poles, f_scale = build_fixed_poles(fit_train_blocks, args.order, args.pole_damping)
    x_train = parameter_matrix(fit_train_blocks, parameter_names)
    y_train = fit_all_coefficients(
        fit_train_blocks,
        sparam_labels,
        poles,
        f_scale,
        args.ridge,
        frequency_weights=normalized_frequency_weights,
    )

    if fit_verify_blocks:
        x_verify = parameter_matrix(fit_verify_blocks, parameter_names)
        y_verify = fit_all_coefficients(
            fit_verify_blocks,
            sparam_labels,
            poles,
            f_scale,
            args.ridge,
            frequency_weights=normalized_verify_frequency_weights,
        )
    else:
        x_verify = None
        y_verify = None

    x_scaler = Standardizer().fit(x_train)
    y_scaler = Standardizer().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = y_scaler.transform(y_train)
    x_verify_scaled = x_scaler.transform(x_verify) if x_verify is not None else None
    y_verify_scaled = y_scaler.transform(y_verify) if y_verify is not None else None

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    layer_sizes = [x_train.shape[1], *hidden_layers, y_train.shape[1]]
    mlp = MLP(layer_sizes, activation=args.activation, seed=args.seed)
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
        seed=args.seed + 17,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_callback=make_training_progress_callback(
            getattr(args, "progress_label", "Neuro-TF fit"),
            args.epochs,
            progress_interval,
        ),
        progress_interval=progress_interval,
    )
    model = NeuroTF(
        mlp=mlp,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        parameter_names=parameter_names,
        sparam_labels=sparam_labels,
        poles=poles,
        f_scale=f_scale,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "training_blocks": len(train_blocks),
        "verification_blocks": len(verify_blocks),
        "ridge": args.ridge,
        "pole_damping": args.pole_damping,
        "split_var": args.split_var,
        "train_values": sorted(parse_csv_set(args.train_values)),
        "verify_values": sorted(parse_csv_set(args.verify_values)),
        "frequency_weights": frequency_weight_spec,
        "frequency_weight_mean": frequency_weight_mean,
        "frequency_weight_min": float(np.min(raw_frequency_weights)),
        "frequency_weight_max": float(np.max(raw_frequency_weights)),
        "frequency_weight_normalization": "Raw frequency weights are divided by their mean over fitted training samples before weighted rational least squares.",
        **dc_metadata,
    }
    model.save(
        out_dir,
        metadata=metadata,
    )
    training_config = {
        "training_blocks": len(train_blocks),
        "verification_blocks": len(verify_blocks),
        "training_samples": int(x_train.shape[0]),
        "verification_samples": int(x_verify.shape[0]) if x_verify is not None else 0,
        "parameters": parameter_names,
        "sparameters": sparam_labels,
        "order": args.order,
        "pole_damping": args.pole_damping,
        "ridge": args.ridge,
        "f_scale": f_scale,
        "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
        "dc_resistance_extraction": metadata["dc_resistance_extraction"],
        "dc_is_separate_from_fitted_response": True,
        "hidden_layers": args.hidden_layers,
        "activation": mlp.activation,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "loss_interval": getattr(args, "loss_interval", 1),
        "progress_interval": progress_interval_from_args(args),
        "seed": args.seed,
        "frequency_weights": metadata["frequency_weights"],
        "frequency_weight_mean": metadata["frequency_weight_mean"],
    }
    plot_context = model_settings_title(
        "Neuro-TF",
        training_config,
        getattr(args, "progress_label", "Neuro-TF fit"),
    )
    write_history(
        out_dir / "training_history.csv",
        history,
        plot_title=f"Model performance vs epoch | {plot_context}",
    )

    if verify_blocks:
        pred_blocks = model.predict_blocks(verify_blocks)
        summary = write_training_verification_artifacts(
            out_dir,
            verify_blocks,
            pred_blocks,
            sparam_labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            frequency_weights=getattr(args, "frequency_weights", None),
            y_z0=50.0,
            title_context=plot_context,
        )
    else:
        summary = {"warning": "No verification blocks were available"}
        (out_dir / "verification_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
    export_commands = neurotf_export_commands(out_dir, args.mdif)
    write_training_markdown(
        out_dir / "training_summary.md",
        model_kind="Neuro-TF",
        config=training_config,
        summary=summary,
        history=history,
        export_commands=export_commands,
    )

    if not getattr(args, "quiet", False):
        print(json.dumps({
            "out_dir": str(out_dir),
            "training_summary": str(out_dir / "training_summary.md"),
            "training_blocks": len(train_blocks),
            "verification_blocks": len(verify_blocks),
            "parameters": parameter_names,
            "sparameters": sparam_labels,
            "n_poles": args.order,
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "frequency_weights": metadata["frequency_weights"],
            "frequency_weight_mean": metadata["frequency_weight_mean"],
            "export_commands": dict(export_commands),
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
        }, indent=2))
    return 0


def command_predict(args: argparse.Namespace) -> int:
    model = NeuroTF.load(Path(args.model_dir))
    blocks = read_mdif(Path(args.mdif))
    pred_blocks = model.predict_blocks(blocks)
    out_path = Path(args.out_mdif)
    write_mdif(out_path, pred_blocks, model.sparam_labels)
    print(f"Wrote {out_path}")
    return 0


def command_export_ads(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = NeuroTF.load(model_dir)
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
        model_kind="Neuro-TF",
        model_dir=model_dir,
        mdif_name=mdif_name,
        blocks=pred_blocks,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        extra_manifest={
            "order": int(len(model.poles)),
            "f_scale": model.f_scale,
            "layer_sizes": model.mlp.layer_sizes,
            "representation": "fixed-pole rational transfer function",
            "dc_equivalent_resistance_ohm": model.dc_equivalent_resistance_ohm,
            "dc_is_separate_from_fitted_response": True,
        },
        extra_notes=[
            "The exported MDIF samples the fitted fixed-pole Neuro-TF response; ADS does not execute the neural network or rational basis directly.",
            "Every exported block includes a zero-Hz point from the saved average equivalent resistance; the coefficient network and rational basis are bypassed there.",
        ],
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "mdif": str(out_dir / mdif_name),
        "manifest": str(out_dir / "ads_model_manifest.json"),
        "blocks": manifest["blocks"],
        "frequency_points_per_block": manifest["frequency_points_per_block"],
    }, indent=2))
    return 0


def command_export_veriloga(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = NeuroTF.load(model_dir)
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'neuro_tf'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    manifest = write_neurotf_veriloga_package(
        out_dir=out_dir,
        module_name=module_name,
        parameter_names=model.parameter_names,
        sparam_labels=model.sparam_labels,
        activation=model.mlp.activation,
        layer_sizes=model.mlp.layer_sizes,
        weights=model.mlp.weights,
        biases=model.mlp.biases,
        x_mean=np.asarray(model.x_scaler.mean, dtype=float),
        x_std=np.asarray(model.x_scaler.std, dtype=float),
        y_mean=np.asarray(model.y_scaler.mean, dtype=float),
        y_std=np.asarray(model.y_scaler.std, dtype=float),
        poles=model.poles,
        f_scale=model.f_scale,
        z0=args.z0,
        frequency_expression=args.frequency_expression,
        parameter_input_scales=parameter_input_scales,
        dc_equivalent_resistance_ohm=model.dc_equivalent_resistance_ohm,
        source_model_dir=str(model_dir),
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
        "n_poles": manifest["n_poles"],
        "f_scale": manifest["f_scale"],
        "fully_self_contained": manifest["fully_self_contained"],
        "dc_equivalent_resistance_ohm": manifest["dc_equivalent_resistance_ohm"],
    }, indent=2))
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Neuro-TF surrogate from generic S-parameter MDIF data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a Neuro-TF model")
    train.add_argument("--mdif", required=True, help="Training/verification MDIF path")
    train.add_argument("--verification-mdif", help="Optional separate verification MDIF")
    train.add_argument("--out-dir", required=True, help="Directory for model and reports")
    train.add_argument("--split-var", default="dataset", help="VAR name that marks train/verification blocks")
    train.add_argument("--train-values", default="train,training", help="Comma-separated split values for training")
    train.add_argument(
        "--verify-values",
        default="verify,verification,test,validation",
        help="Comma-separated split values for verification",
    )
    train.add_argument("--parameter-names", help="Comma-separated geometry parameter VAR names")
    train.add_argument("--holdout-fraction", type=float, default=0.2)
    train.add_argument("--order", type=int, default=10, help="Number of fixed rational poles")
    train.add_argument("--pole-damping", type=float, default=0.18)
    train.add_argument("--ridge", type=float, default=1e-8, help="Least-squares ridge for TF fitting")
    train.add_argument("--hidden-layers", default="64,64")
    train.add_argument("--activation", choices=["tanh", "relu"], default="tanh")
    train.add_argument("--epochs", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument(
        "--frequency-weights",
        help="Frequency loss weights for rational fitting, e.g. 'default=1;1GHz=5;2GHz:4GHz=3'. Exact frequencies and inclusive ranges are supported; later rules override earlier ones.",
    )
    train.add_argument("--loss-interval", type=int, default=1, help="Full train/validation loss check interval in epochs")
    train.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Console progress update interval in epochs. Use 0 to disable progress updates.",
    )
    train.add_argument("--patience", type=int, default=200)
    train.add_argument("--worst-plots", type=int, default=6, help="Number of worst verification fits to plot as PDF")
    train.add_argument("--seed", type=int, default=1234)
    add_debug_argument(train)
    train.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    train.set_defaults(func=command_train)

    sweep = sub.add_parser(
        "sweep",
        aliases=["optimize"],
        help="Try multiple Neuro-TF/NN configurations and retrain the best one",
    )
    sweep.add_argument("--mdif", required=True, help="Training/verification MDIF path")
    sweep.add_argument("--verification-mdif", help="Optional separate verification MDIF")
    sweep.add_argument("--out-dir", required=True, help="Directory for sweep reports and best model")
    sweep.add_argument("--split-var", default="dataset")
    sweep.add_argument("--train-values", default="train,training")
    sweep.add_argument("--verify-values", default="verify,verification,test,validation")
    sweep.add_argument("--parameter-names", help="Comma-separated geometry parameter VAR names")
    sweep.add_argument("--holdout-fraction", type=float, default=0.2)
    sweep.add_argument(
        "--orders",
        "--order",
        dest="orders",
        default="6,10,14",
        help="Comma-separated rational pole counts; --order accepts one value as in train.",
    )
    sweep.add_argument(
        "--pole-dampings",
        "--pole-damping",
        dest="pole_dampings",
        default="0.12,0.18,0.28",
        help="Comma-separated damping values; --pole-damping accepts one value as in train.",
    )
    sweep.add_argument(
        "--ridges",
        "--ridge-values",
        "--ridge",
        dest="ridge_values",
        default="1e-10,1e-8,1e-6",
        help="Comma-separated ridge values; --ridge accepts one value as in train.",
    )
    sweep.add_argument(
        "--hidden-layers",
        "--hidden-layer-layouts",
        "--hidden-layer-options",
        dest="hidden_layer_options",
        default="32;64;64,64",
        help="One train-style layout or semicolon-separated hidden-layer layouts, e.g. '32;64,64;128,64'.",
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
    sweep.add_argument(
        "--frequency-weights",
        help="Frequency loss/selection weights, e.g. 'default=1;1GHz=5;2GHz:4GHz=3'.",
    )
    sweep.add_argument("--jobs", type=int, default=1, help="Number of sweep trials to train in parallel")
    sweep.add_argument(
        "--search-mode",
        "--mode",
        dest="mode",
        choices=["grid", "random"],
        default="random",
        help="Sweep search strategy. --mode remains a backward-compatible alias.",
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
    sweep.add_argument("--epochs", type=int, default=1200)
    sweep.add_argument("--batch-size", type=int, default=64)
    sweep.add_argument("--loss-interval", type=int, default=1, help="Full train/validation loss check interval in epochs")
    sweep.add_argument(
        "--progress-interval",
        type=int,
        default=25,
        help="Console progress update interval in epochs. Use 0 to disable progress updates.",
    )
    sweep.add_argument("--patience", type=int, default=150)
    sweep.add_argument("--worst-plots", type=int, default=6)
    sweep.add_argument("--trial-worst-plots", type=int, default=1)
    sweep.add_argument("--keep-trial-models", action="store_true")
    add_debug_argument(sweep)
    sweep.add_argument(
        "--retrain-best",
        action="store_true",
        help="Retrain the selected best configuration at the end instead of using the best completed trial model",
    )
    sweep.add_argument("--seed", type=int, default=1234)
    sweep.set_defaults(func=command_sweep)

    predict = sub.add_parser("predict", help="Predict S-parameters for MDIF parameter blocks")
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--mdif", required=True)
    predict.add_argument("--out-mdif", required=True)
    predict.set_defaults(func=command_predict)

    export_ads = sub.add_parser(
        "export-ads-mdif",
        aliases=["export-ads"],
        help="Export a trained Neuro-TF model as an ADS-ready parameterized S-parameter MDIF package",
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

    export_va = sub.add_parser(
        "export-veriloga",
        help=(
            "Export a trained Neuro-TF directly as a self-contained Verilog-A "
            "N-port using its saved coefficient network and fixed poles"
        ),
    )
    export_va.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing trained model.npz and metadata.json",
    )
    export_va.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the Verilog-A package",
    )
    export_va.add_argument(
        "--module-name",
        help="Verilog-A module name. Defaults to the model directory name plus _va",
    )
    export_va.add_argument(
        "--z0",
        type=float,
        default=50.0,
        help="Reference impedance for S-to-Y conversion",
    )
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
