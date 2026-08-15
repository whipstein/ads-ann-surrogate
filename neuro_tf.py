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

VERSION = "0.2.0-rc3"

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


def flatten_coefficients(coeffs: np.ndarray) -> np.ndarray:
    """Flatten complex S-parameter coefficients into the persisted real layout."""

    complex_flat = np.asarray(coeffs, dtype=complex).reshape(-1)
    return np.concatenate([complex_flat.real, complex_flat.imag])


def unflatten_coefficients(row: np.ndarray, n_sparams: int, n_coeffs: int) -> np.ndarray:
    half = n_sparams * n_coeffs
    complex_flat = row[:half] + 1j * row[half : 2 * half]
    return complex_flat.reshape(n_sparams, n_coeffs)


def build_response_conditioning_transform(
    blocks: Sequence[MDIFBlock],
    poles: np.ndarray,
    f_scale: float,
    frequency_weights: np.ndarray | None = None,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build a coefficient transform whose Euclidean error is response error.

    If ``B`` is the weighted rational basis, its ridge-augmented form is
    ``B_aug = [B; sqrt(ridge) I]`` and ``B_aug = Q R``. A raw coefficient
    column ``c`` is represented during NN training as ``R c``. Consequently,
    ``||R dc||_2 == ||B_aug dc||_2`` while the orthonormal ``Q`` prevents small
    latent errors from being amplified by an ill-conditioned pole/residue
    basis. Coefficients use row-major storage in this module, so the returned
    encoder and decoder operate on the right of a coefficient row.
    """

    frequencies = np.concatenate(
        [np.asarray(block.freq_hz, dtype=float).reshape(-1) for block in blocks]
    )
    basis = rational_basis(frequencies, poles, f_scale)
    if frequency_weights is not None:
        weights = np.asarray(frequency_weights, dtype=float).reshape(-1)
        if weights.shape != (basis.shape[0],):
            raise ValueError(
                f"Expected {basis.shape[0]} conditioning frequency weights, got "
                f"{weights.shape}"
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(
                "Conditioning frequency weights must be finite and non-negative"
            )
        basis = basis * np.sqrt(weights)[:, None]
    if not math.isfinite(float(ridge)) or float(ridge) < 0.0:
        raise ValueError("Conditioning ridge must be finite and non-negative")
    n_coeffs = basis.shape[1]
    response_basis = basis
    if ridge > 0.0:
        basis = np.vstack(
            [
                basis,
                math.sqrt(float(ridge)) * np.eye(n_coeffs, dtype=complex),
            ]
        )
    if basis.shape[0] < n_coeffs:
        raise ValueError(
            "Neuro-TF needs at least order + 1 fitted RF frequency rows to "
            f"condition an order-{len(poles)} rational basis; got "
            f"{basis.shape[0]} rows. Reduce --order or add RF frequencies."
        )
    rank = int(np.linalg.matrix_rank(basis))
    if rank < n_coeffs:
        raise ValueError(
            "The Neuro-TF rational basis is rank deficient on the fitted RF "
            f"frequency grid (rank {rank} of {n_coeffs}). Reduce --order or "
            "provide a broader frequency grid."
        )
    _q, upper = np.linalg.qr(basis, mode="reduced")
    encoder = upper.T
    decoder = np.linalg.solve(encoder, np.eye(n_coeffs, dtype=complex))
    diagnostics: dict[str, float | int] = {
        "frequency_rows": int(response_basis.shape[0]),
        "conditioning_rows": int(basis.shape[0]),
        "coefficient_count": int(n_coeffs),
        "rank": rank,
        "ridge": float(ridge),
        "basis_condition_number": float(np.linalg.cond(response_basis)),
        "conditioning_matrix_condition_number": float(np.linalg.cond(basis)),
        "decoder_condition_number": float(np.linalg.cond(decoder)),
    }
    return encoder, decoder, diagnostics


def transform_coefficient_rows(
    rows: np.ndarray,
    n_sparams: int,
    n_coeffs: int,
    transform: np.ndarray,
) -> np.ndarray:
    """Apply one complex right-side transform to every S-parameter row."""

    values = np.asarray(rows, dtype=float)
    transformed = []
    for row in values:
        coeffs = unflatten_coefficients(row, n_sparams, n_coeffs)
        transformed.append(flatten_coefficients(coeffs @ transform))
    return np.asarray(transformed, dtype=float)


def real_coefficient_transform(
    complex_transform: np.ndarray,
    n_sparams: int,
) -> np.ndarray:
    """Return the real matrix equivalent of a repeated complex transform."""

    transform = np.asarray(complex_transform, dtype=complex)
    if transform.ndim != 2 or transform.shape[0] != transform.shape[1]:
        raise ValueError("Coefficient transform must be a square matrix")
    n_coeffs = transform.shape[0]
    complex_matrix = np.zeros(
        (n_sparams * n_coeffs, n_sparams * n_coeffs),
        dtype=complex,
    )
    for index in range(n_sparams):
        start = index * n_coeffs
        complex_matrix[start : start + n_coeffs, start : start + n_coeffs] = transform
    return np.block(
        [
            [complex_matrix.real, complex_matrix.imag],
            [-complex_matrix.imag, complex_matrix.real],
        ]
    )


def response_equivalent_standardizer(data: np.ndarray) -> Standardizer:
    """Center each latent output but apply one scale to preserve its metric."""

    values = np.asarray(data, dtype=float)
    scaler = Standardizer()
    scaler.mean = np.mean(values, axis=0)
    scale = float(np.sqrt(np.mean((values - scaler.mean) ** 2)))
    if not math.isfinite(scale) or scale < EPS:
        scale = 1.0
    scaler.std = np.full(values.shape[1], scale, dtype=float)
    return scaler


def fold_output_transform_into_mlp(
    mlp: MLP,
    output_scaler: Standardizer,
    real_transform: np.ndarray,
) -> Standardizer:
    """Fold inverse scaling and a linear decoder into the MLP output layer.

    After this operation the network directly emits decoded raw coefficients.
    An identity standardizer is returned so all existing model and export code
    continues to consume the established raw coefficient representation.
    """

    if output_scaler.mean is None or output_scaler.std is None:
        raise ValueError("Output standardizer must be fitted before folding")
    transform = np.asarray(real_transform, dtype=float)
    output_count = mlp.weights[-1].shape[1]
    if transform.shape != (output_count, output_count):
        raise ValueError(
            f"Expected a {output_count}x{output_count} output transform, got "
            f"{transform.shape}"
        )
    scaled_transform = output_scaler.std[:, None] * transform
    mlp.weights[-1] = mlp.weights[-1] @ scaled_transform
    mlp.biases[-1] = mlp.biases[-1] @ scaled_transform + output_scaler.mean @ transform
    identity = Standardizer()
    identity.mean = np.zeros(output_count, dtype=float)
    identity.std = np.ones(output_count, dtype=float)
    return identity


def reciprocity_summary(
    blocks: Sequence[MDIFBlock],
    sparam_labels: Sequence[str],
) -> dict[str, float | int | bool | None]:
    """Measure Sij/Sji agreement for a complete S-parameter matrix."""

    nports = infer_nports(sparam_labels)
    if nports is None:
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
    relative_error = max_error / max(max_reference, EPS)
    return {
        "nports": int(nports),
        "comparable": True,
        "max_abs_error": max_error,
        "relative_error": relative_error,
    }


def reciprocity_projection(
    sparam_labels: Sequence[str],
    n_coeffs: int,
) -> np.ndarray:
    """Build a real output projection that exactly ties reciprocal S entries."""

    nports = infer_nports(sparam_labels)
    if nports is None:
        raise ValueError(
            "Reciprocity enforcement requires a complete S-parameter matrix"
        )
    n_complex = len(sparam_labels) * n_coeffs
    projection = np.eye(2 * n_complex, dtype=float)
    label_indices = {label: index for index, label in enumerate(sparam_labels)}
    for row in range(1, nports + 1):
        for col in range(row + 1, nports + 1):
            first_label = label_indices[f"S{row}{col}"]
            second_label = label_indices[f"S{col}{row}"]
            for coefficient in range(n_coeffs):
                first = first_label * n_coeffs + coefficient
                second = second_label * n_coeffs + coefficient
                for offset in (0, n_complex):
                    a = first + offset
                    b = second + offset
                    projection[a, a] = 0.5
                    projection[b, a] = 0.5
                    projection[a, b] = 0.5
                    projection[b, b] = 0.5
    return projection


def apply_output_projection(mlp: MLP, projection: np.ndarray) -> None:
    """Fold a raw-output linear projection into an already decoded MLP."""

    matrix = np.asarray(projection, dtype=float)
    output_count = mlp.weights[-1].shape[1]
    if matrix.shape != (output_count, output_count):
        raise ValueError(
            f"Expected a {output_count}x{output_count} output projection, got "
            f"{matrix.shape}"
        )
    mlp.weights[-1] = mlp.weights[-1] @ matrix
    mlp.biases[-1] = mlp.biases[-1] @ matrix


def apply_rf_response_scale(mlp: MLP, scale: float) -> None:
    """Fold a uniform RF S-matrix contraction into the coefficient network."""

    value = float(scale)
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError("RF response scale must be finite and in (0, 1]")
    mlp.weights[-1] *= value
    mlp.biases[-1] *= value


def blocks_from_coefficient_rows(
    template_blocks: Sequence[MDIFBlock],
    coefficient_rows: np.ndarray,
    sparam_labels: Sequence[str],
    poles: np.ndarray,
    f_scale: float,
) -> list[MDIFBlock]:
    """Evaluate fitted coefficient rows on matching template frequency grids."""

    if len(template_blocks) != len(coefficient_rows):
        raise ValueError(
            "Coefficient-row count does not match the template block count"
        )
    n_coeffs = len(poles) + 1
    predicted: list[MDIFBlock] = []
    for block, row in zip(template_blocks, coefficient_rows):
        coeffs = unflatten_coefficients(row, len(sparam_labels), n_coeffs)
        values = evaluate_coefficients(coeffs, block.freq_hz, poles, f_scale)
        predicted.append(
            MDIFBlock(
                params=dict(block.params),
                freq_hz=np.asarray(block.freq_hz, dtype=float).copy(),
                sparams={
                    label: np.asarray(values[index], dtype=complex)
                    for index, label in enumerate(sparam_labels)
                },
                source_index=block.source_index,
            )
        )
    return predicted


def compact_response_summary(
    truth_blocks: Sequence[MDIFBlock],
    predicted_blocks: Sequence[MDIFBlock],
    sparam_labels: Sequence[str],
) -> dict[str, object]:
    """Summarize the rational stage without creating report artifacts."""

    errors: list[np.ndarray] = []
    for truth, predicted in zip(truth_blocks, predicted_blocks):
        for label in sparam_labels:
            errors.append(
                np.abs(
                    np.asarray(predicted.sparams[label], dtype=complex)
                    - np.asarray(truth.sparams[label], dtype=complex)
                )
            )
    flat_error = np.concatenate(errors) if errors else np.asarray([], dtype=float)
    return {
        "rmse_abs": (
            float(np.sqrt(np.mean(flat_error**2))) if flat_error.size else None
        ),
        "max_abs": float(np.max(flat_error)) if flat_error.size else None,
        "passivity": passivity_summary(predicted_blocks, sparam_labels),
    }


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
        dc_resistance_source_kind: str | None = None,
        dc_port_resistances_ohm: dict[str, float] | None = None,
        dc_model: DCConductanceModel | None = None,
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
            row_values = values_by_label.T
            dc_mask = np.asarray(block.freq_hz, dtype=float) == 0.0
            if self.dc_model is not None and np.any(dc_mask):
                row_values[dc_mask, :] = self.dc_model.predict_block_s_values(block)[None, :]
            else:
                row_values = apply_distinct_dc_response(
                    row_values,
                    block.freq_hz,
                    self.sparam_labels,
                    self.dc_equivalent_resistance_ohm,
                    self.dc_resistance_source_kind,
                    self.dc_port_resistances_ohm,
                    z0=50.0,
                )
            values_by_label = row_values.T
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
        if self.dc_model is not None:
            self.dc_model.save(out_dir)
        combined_metadata = {
            "version": VERSION,
            "parameter_names": self.parameter_names,
            "sparam_labels": self.sparam_labels,
            "n_poles": int(len(self.poles)),
            "n_coeffs_per_sparam": int(self.n_coeffs),
            "layer_sizes": self.mlp.layer_sizes,
            "activation": self.mlp.activation,
            "dc_equivalent_resistance_ohm": self.dc_equivalent_resistance_ohm,
            "dc_resistance_source_kind": self.dc_resistance_source_kind,
            "dc_port_resistances_ohm": self.dc_port_resistances_ohm,
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
            dc_resistance_source_kind=metadata.get("dc_resistance_source_kind"),
            dc_port_resistances_ohm=metadata.get("dc_port_resistances_ohm"),
            dc_model=DCConductanceModel.load_optional(model_dir),
        )


def sweep_candidate_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.mode == "adaptive":
        base_config = {
            "order": parse_int_options(args.orders)[0],
            "pole_damping": parse_float_options(args.pole_dampings)[0],
            "ridge": parse_float_options(args.ridge_values)[0],
            "hidden_layers": parse_hidden_layer_options(args.hidden_layer_options)[0],
            "activation": parse_text_options(args.activation_options)[0],
            "learning_rate": parse_float_options(args.learning_rates)[0],
        }
        candidates, columns, log_parameters = build_adaptive_candidate_pool(
            base_config,
            args.optimize_parameter,
            {
                "activation": "str",
                "batch_size": "int",
                "epochs": "int",
                "hidden_layers": "hidden_layers",
                "learning_rate": "float",
                "order": "int",
                "patience": "int",
                "pole_damping": "float",
                "ridge": "float",
            },
            max_trials=args.max_trials,
            candidate_pool=args.adaptive_candidate_pool,
            hidden_width_step=args.adaptive_hidden_width_step,
            seed=args.seed,
        )
        args.adaptive_result_columns = columns
        args.adaptive_log_parameters = log_parameters
        return candidates
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
        passivity_mode=args.passivity_mode,
        passivity_margin=args.passivity_margin,
        reciprocity_mode=args.reciprocity_mode,
        reciprocity_tolerance=args.reciprocity_tolerance,
        debug=bool(getattr(args, "debug", False)),
        quiet=True,
    ), candidate)


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
        train_command_prefix=[
            sys.executable,
            "surrogate.py",
            "--model",
            "neuro-tf",
            "train",
        ],
    )
    best_dir = Path(args.out_dir) / "best_model"
    if status == 0:
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
        model_type="neuro-tf",
    )


def resolve_neurotf_export_dc(
    model: NeuroTF,
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
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    progress_interval = progress_interval_from_args(args)
    dc_model, dc_history, dc_metadata = train_dc_conductance_model(
        train_blocks,
        verify_blocks,
        parameter_names,
        sparam_labels,
        hidden_layers=hidden_layers,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        loss_interval=getattr(args, "loss_interval", 1),
        progress_interval=progress_interval,
        progress_label=f"{getattr(args, 'progress_label', 'Neuro-TF fit')} DC",
        z0=50.0,
        port_paths=getattr(args, "dc_port_paths", None),
        open_threshold_ohm=float(
            getattr(args, "dc_open_threshold", DEFAULT_DC_OPEN_THRESHOLD_OHM)
        ),
        open_resistance_ohm=float(
            getattr(args, "dc_open_resistance", DEFAULT_DC_OPEN_RESISTANCE_OHM)
        ),
    )
    fit_train_blocks = positive_frequency_blocks(train_blocks)
    fit_verify_blocks = positive_frequency_blocks(verify_blocks) if verify_blocks else []
    source_rf_passivity = passivity_summary(fit_train_blocks, sparam_labels)
    source_rf_reciprocity = reciprocity_summary(fit_train_blocks, sparam_labels)
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
    raw_y_train = fit_all_coefficients(
        fit_train_blocks,
        sparam_labels,
        poles,
        f_scale,
        args.ridge,
        frequency_weights=normalized_frequency_weights,
    )

    if fit_verify_blocks:
        x_verify = parameter_matrix(fit_verify_blocks, parameter_names)
        raw_y_verify = fit_all_coefficients(
            fit_verify_blocks,
            sparam_labels,
            poles,
            f_scale,
            args.ridge,
            frequency_weights=normalized_verify_frequency_weights,
        )
    else:
        x_verify = None
        raw_y_verify = None

    rational_fit_train_summary = compact_response_summary(
        fit_train_blocks,
        blocks_from_coefficient_rows(
            fit_train_blocks,
            raw_y_train,
            sparam_labels,
            poles,
            f_scale,
        ),
        sparam_labels,
    )
    rational_fit_verify_summary = (
        compact_response_summary(
            fit_verify_blocks,
            blocks_from_coefficient_rows(
                fit_verify_blocks,
                raw_y_verify,
                sparam_labels,
                poles,
                f_scale,
            ),
            sparam_labels,
        )
        if raw_y_verify is not None
        else None
    )

    coefficient_encoder, coefficient_decoder, conditioning_diagnostics = (
        build_response_conditioning_transform(
            fit_train_blocks,
            poles,
            f_scale,
            frequency_weights=normalized_frequency_weights,
            ridge=args.ridge,
        )
    )
    n_coeffs = len(poles) + 1
    n_sparams = len(sparam_labels)
    y_train = transform_coefficient_rows(
        raw_y_train,
        n_sparams,
        n_coeffs,
        coefficient_encoder,
    )
    y_verify = (
        transform_coefficient_rows(
            raw_y_verify,
            n_sparams,
            n_coeffs,
            coefficient_encoder,
        )
        if raw_y_verify is not None
        else None
    )

    x_scaler = Standardizer().fit(x_train)
    latent_y_scaler = response_equivalent_standardizer(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    y_train_scaled = latent_y_scaler.transform(y_train)
    x_verify_scaled = x_scaler.transform(x_verify) if x_verify is not None else None
    y_verify_scaled = (
        latent_y_scaler.transform(y_verify) if y_verify is not None else None
    )

    layer_sizes = [x_train.shape[1], *hidden_layers, y_train.shape[1]]
    mlp = MLP(layer_sizes, activation=args.activation, seed=args.seed)
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
    real_decoder = real_coefficient_transform(coefficient_decoder, n_sparams)
    y_scaler = fold_output_transform_into_mlp(
        mlp,
        latent_y_scaler,
        real_decoder,
    )
    reciprocity_mode = str(getattr(args, "reciprocity_mode", "auto"))
    reciprocity_tolerance = float(getattr(args, "reciprocity_tolerance", 1e-6))
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
    if reciprocity_enforced:
        apply_output_projection(
            mlp,
            reciprocity_projection(sparam_labels, n_coeffs),
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
        dc_resistance_source_kind=str(dc_metadata["dc_resistance_source_kind"]),
        dc_port_resistances_ohm=dict(dc_metadata["dc_port_resistances_ohm"]),
        dc_model=dc_model,
    )

    passivity_mode = str(getattr(args, "passivity_mode", "auto"))
    passivity_margin = float(getattr(args, "passivity_margin", 1e-3))
    if (
        not math.isfinite(passivity_margin)
        or passivity_margin < 0.0
        or passivity_margin >= 1.0
    ):
        raise ValueError("--passivity-margin must be finite and in [0, 1)")
    source_passivity_available = source_rf_passivity["nports"] is not None
    source_is_passive = (
        source_passivity_available
        and source_rf_passivity["violating_points"] == 0
    )
    passivity_enforced = passivity_mode == "enforce" or (
        passivity_mode == "auto" and source_is_passive
    )
    if passivity_mode == "enforce" and not source_passivity_available:
        raise ValueError(
            "--passivity-mode enforce requires a complete S-parameter matrix"
        )
    predicted_train_before_scale = model.predict_blocks(fit_train_blocks)
    predicted_rf_passivity_before_scale = passivity_summary(
        predicted_train_before_scale,
        sparam_labels,
    )
    rf_response_scale = 1.0
    passivity_target_sigma = 1.0 - passivity_margin
    if passivity_enforced:
        predicted_sigma = predicted_rf_passivity_before_scale["max_singular_value"]
        if predicted_sigma is None or not math.isfinite(float(predicted_sigma)):
            raise ValueError(
                "Could not assess the fitted Neuro-TF response for passivity"
            )
        if float(predicted_sigma) > passivity_target_sigma:
            rf_response_scale = passivity_target_sigma / float(predicted_sigma)
            apply_rf_response_scale(mlp, rf_response_scale)
    predicted_rf_passivity_after_scale = passivity_summary(
        model.predict_blocks(fit_train_blocks),
        sparam_labels,
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
        "coefficient_training_representation": "qr_conditioned_rational_response",
        "coefficient_training_loss_domain": "weighted_complex_sparameter_response",
        "coefficient_training_output_scale": float(latent_y_scaler.std[0]),
        "coefficient_decoder_folded_into_output_layer": True,
        "coefficient_conditioning": conditioning_diagnostics,
        "rational_fit_train_summary": rational_fit_train_summary,
        "rational_fit_verification_summary": rational_fit_verify_summary,
        "reciprocity_mode": reciprocity_mode,
        "reciprocity_tolerance": reciprocity_tolerance,
        "reciprocity_enforced": reciprocity_enforced,
        "source_rf_reciprocity": source_rf_reciprocity,
        "passivity_mode": passivity_mode,
        "passivity_margin": passivity_margin,
        "passivity_target_sigma": passivity_target_sigma,
        "passivity_enforced": passivity_enforced,
        "source_rf_passivity": source_rf_passivity,
        "predicted_train_passivity_before_scale": predicted_rf_passivity_before_scale,
        "rf_response_scale": rf_response_scale,
        "predicted_train_passivity_after_scale": predicted_rf_passivity_after_scale,
        "passivity_assessment_scope": "positive-frequency training blocks only",
        **dc_metadata,
        "dc_model_history_rows": len(dc_history),
    }
    model.save(
        out_dir,
        metadata=metadata,
    )
    write_history(
        out_dir / "dc_training_history.csv",
        dc_history,
        plot_title="Separate exact-DC conductance model performance vs epoch",
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
        "coefficient_training_representation": metadata[
            "coefficient_training_representation"
        ],
        "coefficient_training_loss_domain": metadata[
            "coefficient_training_loss_domain"
        ],
        "coefficient_training_output_scale": metadata[
            "coefficient_training_output_scale"
        ],
        "coefficient_conditioning": metadata["coefficient_conditioning"],
        "rational_fit_train_summary": metadata["rational_fit_train_summary"],
        "rational_fit_verification_summary": metadata[
            "rational_fit_verification_summary"
        ],
        "reciprocity_mode": metadata["reciprocity_mode"],
        "reciprocity_enforced": metadata["reciprocity_enforced"],
        "source_rf_reciprocity": metadata["source_rf_reciprocity"],
        "passivity_mode": metadata["passivity_mode"],
        "passivity_margin": metadata["passivity_margin"],
        "passivity_enforced": metadata["passivity_enforced"],
        "source_rf_passivity": metadata["source_rf_passivity"],
        "rf_response_scale": metadata["rf_response_scale"],
        "predicted_train_passivity_before_scale": metadata[
            "predicted_train_passivity_before_scale"
        ],
        "predicted_train_passivity_after_scale": metadata[
            "predicted_train_passivity_after_scale"
        ],
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
            sparam_labels,
            parameter_names,
            max_worst_plots=getattr(args, "worst_plots", 6),
            frequency_weights=getattr(args, "frequency_weights", None),
            y_z0=50.0,
            title_context=plot_context,
        )
        write_mdif(
            out_dir / "predicted_verification.mdif",
            pred_blocks,
            sparam_labels,
        )
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
                "coefficient_training_representation": metadata[
                    "coefficient_training_representation"
                ],
                "coefficient_conditioning": metadata["coefficient_conditioning"],
                "rational_fit_verification_summary": metadata[
                    "rational_fit_verification_summary"
                ],
                "reciprocity_enforced": metadata["reciprocity_enforced"],
                "source_rf_reciprocity": metadata["source_rf_reciprocity"],
                "passivity_enforced": metadata["passivity_enforced"],
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
            "coefficient_training_representation": metadata[
                "coefficient_training_representation"
            ],
            "coefficient_conditioning": metadata["coefficient_conditioning"],
            "rational_fit_train_summary": metadata["rational_fit_train_summary"],
            "reciprocity_enforced": metadata["reciprocity_enforced"],
            "passivity_enforced": metadata["passivity_enforced"],
            "rf_response_scale": metadata["rf_response_scale"],
            "predicted_train_passivity_after_scale": metadata[
                "predicted_train_passivity_after_scale"
            ],
        }
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
            "dc_resistance_source_kind": metadata["dc_resistance_source_kind"],
            "dc_model_kind": metadata["dc_model_kind"],
            "dc_model_train_s_rmse": metadata["dc_model_train_s_rmse"],
            "dc_port_paths": metadata["dc_port_paths"],
            "dc_matrix_entries": metadata.get("dc_matrix_entries", []),
            "dc_sparameter_entries": metadata.get("dc_sparameter_entries", []),
            "dc_port_resistances_ohm": metadata["dc_port_resistances_ohm"],
            "dc_resistance_pair_means_ohm": metadata[
                "dc_resistance_pair_means_ohm"
            ],
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
    source_metadata = read_model_metadata(str(model_dir))
    export_dc_model, dc_metadata = resolve_neurotf_export_dc(
        model,
        source_metadata,
        args,
        50.0,
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
            "dc_metadata": dc_metadata,
            "dc_is_separate_from_fitted_response": True,
        },
        extra_notes=[
            "The exported MDIF samples the fitted fixed-pole Neuro-TF response; ADS does not execute the neural network or rational basis directly.",
            "Every exported block includes a zero-Hz point from the selected passive exact-DC port paths; the coefficient network and rational basis are bypassed there.",
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
        "dc_mdif_model_s_max_abs_error": dc_metadata.get(
            "dc_mdif_model_s_max_abs_error"
        ),
        "dc_mdif_match_within_tolerance": dc_metadata.get(
            "dc_mdif_match_within_tolerance"
        ),
        "dc_mdif_warning": dc_metadata.get("dc_mdif_warning"),
    }, indent=2))
    return 0


def command_export_veriloga(args: argparse.Namespace) -> int:
    model_dir = Path(args.model_dir)
    model = NeuroTF.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'neuro_tf'}_va"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    export_dc_model, dc_metadata = resolve_neurotf_export_dc(
        model,
        source_metadata,
        args,
        args.z0,
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
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=dc_metadata.get("dc_resistance_source_kind"),
        dc_port_resistances_ohm=dc_metadata.get("dc_port_resistances_ohm"),
        dc_model=(export_dc_model.export_data() if export_dc_model is not None else None),
        source_model_dir=str(model_dir),
        extra_manifest={
            "dc_resistance_source_kind": dc_metadata.get(
                "dc_resistance_source_kind"
            ),
            "dc_resistance_pair_means_ohm": dc_metadata.get(
                "dc_resistance_pair_means_ohm"
            ),
            "dc_metadata": dc_metadata,
        },
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
    """Export a fixed-pole Neuro-TF as a linear ADS HB subnetwork."""

    model_dir = Path(args.model_dir)
    model = NeuroTF.load(model_dir)
    source_metadata = read_model_metadata(str(model_dir))
    out_dir = Path(args.out_dir)
    module_name = args.module_name or f"{normalize_name(model_dir.name) or 'neuro_tf'}_hb"
    parameter_input_scales = parse_parameter_scale_spec(
        model.parameter_names,
        args.parameter_input_scales,
    )
    export_dc_model, dc_metadata = resolve_neurotf_export_dc(
        model,
        source_metadata,
        args,
        args.z0,
    )
    manifest = write_ads_hb_neurotf_package(
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
        parameter_input_scales=parameter_input_scales,
        dc_equivalent_resistance_ohm=float(
            dc_metadata["dc_equivalent_resistance_ohm"]
        ),
        dc_resistance_source_kind=dc_metadata.get("dc_resistance_source_kind"),
        dc_port_resistances_ohm=dc_metadata.get("dc_port_resistances_ohm"),
        dc_model=(export_dc_model.export_data() if export_dc_model is not None else None),
        source_model_dir=str(model_dir),
        extra_manifest={
            "model_family": "neuro_transfer_function",
            "dc_metadata": dc_metadata,
        },
    )
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "netlist": str(out_dir / str(manifest["netlist_file"])),
                "manifest": str(out_dir / "ads_hb_manifest.json"),
                "module_name": manifest["module_name"],
                "n_poles": manifest["n_poles"],
                "linear": manifest["linear"],
                "power_dependent": manifest["power_dependent"],
                "supported_analyses": manifest["supported_analyses"],
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
        prog=os.environ.get("ADS_SURROGATE_CLI_PROG"),
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
    add_dc_fitting_arguments(train)
    train.add_argument("--order", type=int, default=10, help="Number of fixed rational poles")
    train.add_argument("--pole-damping", type=float, default=0.18)
    train.add_argument("--ridge", type=float, default=1e-8, help="Least-squares ridge for TF fitting")
    train.add_argument(
        "--passivity-mode",
        choices=["auto", "enforce", "off"],
        default="auto",
        help=(
            "RF passivity handling. auto contracts the fitted RF response only "
            "when the positive-frequency training data is passive; enforce "
            "always contracts it; off leaves the fitted response unchanged."
        ),
    )
    train.add_argument(
        "--passivity-margin",
        type=float,
        default=1e-3,
        help="Target margin below sigma_max=1 when passivity is enforced. Default: 0.001",
    )
    train.add_argument(
        "--reciprocity-mode",
        choices=["auto", "enforce", "off"],
        default="auto",
        help=(
            "Reciprocity handling. auto ties Sij/Sji when the training data is "
            "reciprocal; enforce always ties them; off trains them independently."
        ),
    )
    train.add_argument(
        "--reciprocity-tolerance",
        type=float,
        default=1e-6,
        help="Maximum relative Sij/Sji disagreement accepted by auto mode.",
    )
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
    add_dc_fitting_arguments(sweep)
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
    sweep.add_argument(
        "--passivity-mode",
        choices=["auto", "enforce", "off"],
        default="auto",
        help="Passivity handling passed to every trial; see train --passivity-mode.",
    )
    sweep.add_argument(
        "--passivity-margin",
        type=float,
        default=1e-3,
        help="Target margin below sigma_max=1 when passivity is enforced.",
    )
    sweep.add_argument(
        "--reciprocity-mode",
        choices=["auto", "enforce", "off"],
        default="auto",
        help="Reciprocity handling passed to every trial.",
    )
    sweep.add_argument(
        "--reciprocity-tolerance",
        type=float,
        default=1e-6,
        help="Maximum relative Sij/Sji disagreement accepted by auto mode.",
    )
    sweep.add_argument("--jobs", type=int, default=1, help="Number of sweep trials to train in parallel")
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
    add_dc_export_arguments(export_ads)
    export_ads.set_defaults(func=command_export_ads)

    export_hb = sub.add_parser(
        "export-ads-hb",
        help="Export a trained Neuro-TF as a self-contained linear ADS SDD network for harmonic balance",
    )
    export_hb.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing trained model.npz and metadata.json",
    )
    export_hb.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for the ADS HB package",
    )
    export_hb.add_argument(
        "--module-name",
        help="ADS subnetwork name. Defaults to the model directory name plus _hb",
    )
    export_hb.add_argument(
        "--z0",
        type=float,
        default=50.0,
        help="S-parameter reference impedance",
    )
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
            "Common positive ADS-side denominator for every instance parameter: "
            "model_value = instance_value / scale. Example: 1um"
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
