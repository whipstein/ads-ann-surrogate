#!/usr/bin/env python3
"""Diagnose model-fit and passivity behavior from retained run artifacts.

Optimization removes heavyweight per-trial model weights unless full trial
retention is requested, but current runs always keep metadata.json. This command
also supports legacy runs whose metadata was cleaned by treating the sweep CSV
and each trial's verification_summary.json as authoritative diagnostic records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

from cli_options import (
    OptionsJSONError,
    add_options_json_argument,
    finalize_options_json_update,
    load_options_json_resolution,
    parse_args_with_options_json,
)


RESULT_FILENAMES = (
    "dnn_sweep_results.csv",
    "kbnn_sweep_results.csv",
    "sweep_results.csv",
    "dnn_reranked_sweep_results.csv",
    "kbnn_reranked_sweep_results.csv",
)
KNOWN_SELECTION_METRICS = (
    "weighted_evm_pct",
    "evm_pct",
    "weighted_rmse_abs",
    "rmse_abs",
    "max_abs",
)


def number(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: object) -> int | None:
    numeric = number(value)
    return int(round(numeric)) if numeric is not None else None


def read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["message"]
        rows = [{"message": "No trial rows were available"}]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_results_file(run_dir: Path) -> Path | None:
    for name in RESULT_FILENAMES:
        path = run_dir / name
        if path.is_file():
            return path
    candidates = sorted(run_dir.glob("*_sweep_results.csv"))
    return candidates[0] if candidates else None


def infer_model(run_dir: Path, results_path: Path | None) -> str:
    text = " ".join(
        [run_dir.name.lower(), results_path.name.lower() if results_path else ""]
    )
    if "kbnn" in text:
        return "kbnn"
    if "dnn" in text:
        return "dnn"
    for metadata_path in (
        run_dir / "metadata.json",
        run_dir / "best_model" / "metadata.json",
    ):
        metadata = read_json(metadata_path)
        if not metadata:
            continue
        if metadata.get("mode") is not None or metadata.get("coarse_model") is not None:
            return "kbnn"
        return "dnn"
    return "unknown"


def selection_metric(rows: Sequence[Mapping[str, object]]) -> str | None:
    for row in rows:
        raw = str(row.get("selection_metric") or "").strip()
        if raw:
            return raw
    for name in KNOWN_SELECTION_METRICS:
        if any(number(row.get(name)) is not None for row in rows):
            return name
    if any(number(row.get("metric")) is not None for row in rows):
        return "metric"
    return None


def nested_dict(payload: Mapping[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else {}


def trial_summary_path(run_dir: Path, trial: int) -> Path:
    return run_dir / "trials" / f"trial_{trial:04d}" / "verification_summary.json"


def enrich_trial_rows(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
) -> tuple[list[dict[str, object]], int]:
    enriched: list[dict[str, object]] = []
    summaries_found = 0
    for position, raw_row in enumerate(rows, start=1):
        row = dict(raw_row)
        trial = integer(row.get("trial")) or position
        summary_path = trial_summary_path(run_dir, trial)
        summary = read_json(summary_path)
        if summary:
            summaries_found += 1
            passivity = nested_dict(summary, "passivity")
            row["passivity_violating_points"] = passivity.get(
                "violating_points", row.get("passivity_violating_points", "")
            )
            row["passivity_max_singular_value"] = passivity.get(
                "max_singular_value", row.get("passivity_max_singular_value", "")
            )
            for key in (
                "passivity_enforced",
                "passivity_unavailable_reason",
                "rf_response_scale",
            ):
                if key in summary:
                    row[key] = summary[key]
            source = nested_dict(summary, "source_rf_passivity")
            train_after = nested_dict(
                summary, "predicted_train_passivity_after_scale"
            )
            row["source_rf_passivity_violating_points"] = source.get(
                "violating_points", ""
            )
            row["source_rf_passivity_max_sigma"] = source.get(
                "max_singular_value", ""
            )
            row["train_after_scale_violating_points"] = train_after.get(
                "violating_points", ""
            )
            row["train_after_scale_max_sigma"] = train_after.get(
                "max_singular_value", ""
            )
            row["verification_summary"] = os.path.relpath(summary_path, run_dir)
            if metric_name and number(row.get(metric_name)) is None:
                row[metric_name] = summary.get(metric_name, "")
        row["trial"] = trial
        row["passive"] = (
            integer(row.get("passivity_violating_points")) == 0
            if integer(row.get("passivity_violating_points")) is not None
            else ""
        )
        enriched.append(row)
    return enriched, summaries_found


def single_model_rows(run_dir: Path) -> tuple[list[dict[str, object]], int]:
    summary_path = run_dir / "verification_summary.json"
    if not summary_path.is_file() and (run_dir / "best_model").is_dir():
        summary_path = run_dir / "best_model" / "verification_summary.json"
    summary = read_json(summary_path)
    if not summary:
        return [], 0
    passivity = nested_dict(summary, "passivity")
    source = nested_dict(summary, "source_rf_passivity")
    train_after = nested_dict(summary, "predicted_train_passivity_after_scale")
    row: dict[str, object] = {
        "trial": 1,
        "passivity_violating_points": passivity.get("violating_points", ""),
        "passivity_max_singular_value": passivity.get("max_singular_value", ""),
        "passivity_enforced": summary.get("passivity_enforced", ""),
        "passivity_unavailable_reason": summary.get(
            "passivity_unavailable_reason", ""
        ),
        "rf_response_scale": summary.get("rf_response_scale", ""),
        "source_rf_passivity_violating_points": source.get("violating_points", ""),
        "source_rf_passivity_max_sigma": source.get("max_singular_value", ""),
        "train_after_scale_violating_points": train_after.get(
            "violating_points", ""
        ),
        "train_after_scale_max_sigma": train_after.get(
            "max_singular_value", ""
        ),
        "verification_summary": str(summary_path),
    }
    for metric_name in KNOWN_SELECTION_METRICS:
        if metric_name in summary:
            row[metric_name] = summary[metric_name]
    row["passive"] = integer(row["passivity_violating_points"]) == 0
    return [row], 1


def resolve_audit_json(raw_path: str | None, run_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if raw_path:
        supplied = Path(raw_path).expanduser()
        candidates.append(
            supplied / "dataset_audit.json" if supplied.is_dir() else supplied
        )
    candidates.extend(
        [
            run_dir / "audit" / "dataset_audit.json",
            run_dir.parent / "audit" / "dataset_audit.json",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def surviving_metadata(run_dir: Path) -> list[tuple[Path, dict[str, object]]]:
    paths = [
        run_dir / "metadata.json",
        run_dir / "best_model" / "metadata.json",
        run_dir / "coarse_model" / "metadata.json",
        run_dir / "best_model" / "coarse_model" / "metadata.json",
    ]
    paths.extend(sorted((run_dir / "trials").glob("trial_*/metadata.json")))
    paths.extend(
        sorted((run_dir / "trials").glob("trial_*/coarse_model/metadata.json"))
    )
    records: list[tuple[Path, dict[str, object]]] = []
    for path in paths:
        payload = read_json(path)
        if payload is not None:
            records.append((path, payload))
    return records


def ranked_trial_rows(
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
) -> list[dict[str, object]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            integer(row.get("passivity_violating_points"))
            if integer(row.get("passivity_violating_points")) is not None
            else 10**18,
            number(row.get("passivity_max_singular_value"))
            if number(row.get("passivity_max_singular_value")) is not None
            else float("inf"),
            number(row.get(metric_name))
            if metric_name and number(row.get(metric_name)) is not None
            else float("inf"),
        ),
    )


def representative_metadata(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
    metadata_records: Sequence[tuple[Path, Mapping[str, object]]],
) -> dict[str, object]:
    ranked = ranked_trial_rows(rows, metric_name)
    if ranked:
        trial = integer(ranked[0].get("trial"))
        if trial is not None:
            preferred = read_json(
                run_dir / "trials" / f"trial_{trial:04d}" / "metadata.json"
            )
            if preferred:
                return preferred
    for path, payload in metadata_records:
        if path.name == "metadata.json" and "coarse_model" not in path.parts:
            return dict(payload)
    return {}


def command_path(path: str | Path) -> str:
    expanded = Path(path).expanduser()
    try:
        return os.path.relpath(expanded.resolve(), Path.cwd())
    except OSError:
        return str(expanded)


def numeric_text(value: float) -> str:
    return f"{float(value):.6g}"


def observed_setting(
    row: Mapping[str, object],
    metadata: Mapping[str, object],
    key: str,
    fallback: object = "unknown",
) -> object:
    value = row.get(key)
    if value not in (None, ""):
        return value
    value = metadata.get(key)
    return fallback if value in (None, "") else value


def model_optimize_command_base(
    args: argparse.Namespace,
    model: str,
) -> tuple[list[str], bool]:
    command = ["python3", "surrogate.py"]
    options_json = getattr(args, "options_json", None)
    defaults: dict[str, object] = {}
    if options_json:
        command.extend(["--options-json", command_path(options_json)])
        try:
            defaults, _sources = load_options_json_resolution(
                options_json,
                model=model,
                command="optimize",
            )
        except (OSError, OptionsJSONError):
            defaults = {}
    command.extend(["--model", model, "optimize"])
    requires_editing = not bool(defaults.get("mdif"))
    if not defaults.get("mdif"):
        command.extend(["--mdif", "PATH_TO_TRAINING_MDIF"])
    if model == "kbnn":
        mode = str(defaults.get("mode") or "residual").strip().lower()
        coarse_available = bool(
            defaults.get("coarse_mdif") or defaults.get("coarse_model_dir")
        )
        if mode != "plain" and not coarse_available:
            command.extend(["--coarse-mdif", "PATH_TO_COARSE_MDIF"])
            requires_editing = True
    return command, requires_editing


def add_command_suggestion(
    suggestions: list[dict[str, object]],
    *,
    identifier: str,
    title: str,
    triggered_by: Sequence[str],
    rationale: str,
    command: Sequence[str],
    changes: Sequence[Mapping[str, object]],
    notes: Sequence[str] = (),
    requires_editing: bool = False,
) -> None:
    suggestions.append(
        {
            "id": identifier,
            "title": title,
            "triggered_by": list(triggered_by),
            "rationale": rationale,
            "command": shlex.join([str(value) for value in command]),
            "changes": [dict(change) for change in changes],
            "notes": list(notes),
            "requires_editing": bool(requires_editing),
        }
    )


def build_command_suggestions(
    args: argparse.Namespace,
    run_dir: Path,
    model: str,
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
    findings: Sequence[Mapping[str, str]],
    metadata_records: Sequence[tuple[Path, Mapping[str, object]]],
) -> list[dict[str, object]]:
    """Translate diagnostic findings into ordered, copyable follow-up commands."""

    suggestions: list[dict[str, object]] = []
    codes = {str(finding.get("code") or "") for finding in findings}
    ranked = ranked_trial_rows(rows, metric_name)
    reference_row: Mapping[str, object] = ranked[0] if ranked else {}
    metadata = representative_metadata(
        run_dir,
        rows,
        metric_name,
        metadata_records,
    )
    options_json = getattr(args, "options_json", None)
    audit_out = run_dir.parent / f"{run_dir.name}_audit_debug"

    if codes & {
        "RAW_DATA_AUDIT_MISSING",
        "RAW_RF_DATA_NONPASSIVE",
        "TRAINING_SOURCE_NONPASSIVE",
    }:
        audit_command = ["python3", "surrogate.py"]
        audit_defaults: dict[str, object] = {}
        if options_json:
            audit_command.extend(["--options-json", command_path(options_json)])
            try:
                audit_defaults, _sources = load_options_json_resolution(
                    options_json,
                    workflow="audit",
                    command="audit",
                )
            except (OSError, OptionsJSONError):
                audit_defaults = {}
        audit_command.append("audit")
        requires_editing = not bool(audit_defaults.get("mdif"))
        if not audit_defaults.get("mdif"):
            audit_command.extend(["--mdif", "PATH_TO_TRAINING_MDIF"])
        audit_command.extend(
            [
                "--passivity-tolerance",
                "1e-6",
                "--out-dir",
                command_path(audit_out),
            ]
        )
        raw_nonpassive = bool(
            codes & {"RAW_RF_DATA_NONPASSIVE", "TRAINING_SOURCE_NONPASSIVE"}
        )
        add_command_suggestion(
            suggestions,
            identifier="audit-source-data",
            title=(
                "Re-audit and isolate non-passive RF rows"
                if raw_nonpassive
                else "Audit the exact data used by this run"
            ),
            triggered_by=sorted(
                codes
                & {
                    "RAW_DATA_AUDIT_MISSING",
                    "RAW_RF_DATA_NONPASSIVE",
                    "TRAINING_SOURCE_NONPASSIVE",
                }
            ),
            rationale=(
                "Passivity enforcement cannot reconcile contradictory non-passive training targets."
                if raw_nonpassive
                else "The fitting recommendations should be applied only after the positive-frequency source data is confirmed passive."
            ),
            command=audit_command,
            changes=[
                {
                    "option": "--passivity-tolerance",
                    "observed": "not independently confirmed" if not raw_nonpassive else "violations reported",
                    "suggested": "1e-6",
                    "reason": "List the exact block/frequency rows that exceed the passive limit.",
                }
            ],
            notes=(
                "Inspect dataset_passivity.csv before changing neural-network settings.",
                "Do not force passivity until erroneous or intentionally active source rows are resolved.",
            ),
            requires_editing=requires_editing,
        )

    raw_data_conflict = bool(
        codes & {"RAW_RF_DATA_NONPASSIVE", "TRAINING_SOURCE_NONPASSIVE"}
    )
    passivity_problem = bool(
        codes
        & {
            "NO_PASSIVE_TRIAL",
            "PASSIVITY_ENFORCEMENT_DISABLED",
            "TRAIN_PASSIVE_VERIFY_NONPASSIVE",
            "TRAINING_SAFEGUARD_NOT_PASSIVE",
            "MARGINAL_SIGMA_EXCURSION",
            "MODERATE_SIGMA_EXCURSION",
            "MATERIAL_SIGMA_EXCURSION",
            "ERROR_IMPROVES_WITHOUT_FEASIBILITY",
            "LARGE_RF_CONTRACTION",
        }
    )
    if passivity_problem and not raw_data_conflict and model in {"dnn", "kbnn"}:
        base, requires_editing = model_optimize_command_base(args, model)
        observed_penalty = number(
            observed_setting(reference_row, metadata, "passivity_penalty", 10.0)
        )
        current_penalty = 10.0 if observed_penalty is None else observed_penalty
        observed_learning_rate = number(
            observed_setting(reference_row, metadata, "learning_rate", 0.002)
        )
        current_learning_rate = (
            0.002 if observed_learning_rate is None else observed_learning_rate
        )
        observed_margin = number(
            observed_setting(reference_row, metadata, "passivity_margin", 0.001)
        )
        current_margin = 0.001 if observed_margin is None else observed_margin
        penalty_low = max(0.1, current_penalty / 10.0)
        penalty_high = max(30.0, current_penalty * 10.0)
        learning_low = max(1.0e-5, current_learning_rate / 10.0)
        learning_high = max(learning_low * 2.0, current_learning_rate)
        if "LARGE_RF_CONTRACTION" in codes:
            suggested_margin = min(current_margin, 5.0e-4)
        elif "MARGINAL_SIGMA_EXCURSION" in codes:
            suggested_margin = max(current_margin, 2.0e-3)
        elif "MODERATE_SIGMA_EXCURSION" in codes:
            suggested_margin = max(current_margin, 5.0e-3)
        else:
            suggested_margin = current_margin
        selection = (
            metric_name
            if metric_name
            in {
                "rmse_abs",
                "max_abs",
                "evm_pct",
                "weighted_rmse_abs",
                "weighted_evm_pct",
            }
            else "rmse_abs"
        )
        output_dir = run_dir.parent / f"{run_dir.name}_passivity_search"
        command = [*base]
        if model == "dnn":
            command.extend(["--output-domain", "s"])
        command.extend(
            [
                "--passivity-mode",
                "enforce",
                "--passivity-margin",
                numeric_text(suggested_margin),
                "--search-mode",
                "adaptive",
                "--selection-metric",
                selection,
                "--require-passive",
                "--max-passivity-sigma",
                "1.000001",
                "--optimize-parameter",
                f"passivity_penalty={numeric_text(penalty_low)}:{numeric_text(penalty_high)}:log",
                "--optimize-parameter",
                f"learning_rate={numeric_text(learning_low)}:{numeric_text(learning_high)}:log",
            ]
        )
        if "MATERIAL_SIGMA_EXCURSION" in codes:
            command.extend(
                [
                    "--optimize-parameter",
                    "hidden_layers=1:4x64:256:log",
                ]
            )
        command.extend(
            [
                "--adaptive-initial-trials",
                "8",
                "--max-trials",
                "24",
                "--out-dir",
                command_path(output_dir),
            ]
        )
        changes: list[dict[str, object]] = [
            {
                "option": "--passivity-mode",
                "observed": observed_setting(reference_row, metadata, "passivity_mode"),
                "suggested": "enforce",
                "reason": "Make passivity protection explicit during this feasibility search.",
            },
            {
                "option": "--passivity-penalty",
                "observed": numeric_text(current_penalty),
                "suggested": f"adaptive {numeric_text(penalty_low)} to {numeric_text(penalty_high)} (log)",
                "reason": "Determine whether the penalty is too weak or overwhelms response-error learning.",
            },
            {
                "option": "--learning-rate",
                "observed": numeric_text(current_learning_rate),
                "suggested": f"adaptive {numeric_text(learning_low)} to {numeric_text(learning_high)} (log)",
                "reason": "Search lower update sizes where the passivity gradient is less likely to destabilize fitting.",
            },
            {
                "option": "--passivity-margin",
                "observed": numeric_text(current_margin),
                "suggested": numeric_text(suggested_margin),
                "reason": (
                    "Reduce avoidable global response contraction."
                    if "LARGE_RF_CONTRACTION" in codes
                    else "Add a controlled verification buffer without excessive RF loss."
                ),
            },
            {
                "option": "passivity eligibility",
                "observed": "no fully passive eligible trial" if "NO_PASSIVE_TRIAL" in codes else "constraint problems reported",
                "suggested": "--require-passive --max-passivity-sigma 1.000001",
                "reason": "Teach adaptive optimization to prefer feasible trials before comparing response error.",
            },
        ]
        if model == "dnn":
            changes.insert(
                0,
                {
                    "option": "--output-domain",
                    "observed": observed_setting(reference_row, metadata, "output_domain"),
                    "suggested": "s",
                    "reason": "DNN passivity enforcement operates on a complete S-domain output.",
                },
            )
        notes = [
            "Keep the original data, split, seed policy, weights, and parameter names unchanged so the comparison isolates these settings.",
            "The adaptive objective already includes the requested passivity constraints; the response metric ranks trials once they become feasible.",
        ]
        if "RAW_DATA_AUDIT_MISSING" in codes:
            notes.insert(0, "Run the suggested audit first and continue only if its RF rows pass.")
        if "MATERIAL_SIGMA_EXCURSION" in codes:
            notes.append("The hidden-layer range is included because a material excursion can indicate insufficient capacity, not just a weak penalty.")
        add_command_suggestion(
            suggestions,
            identifier="passivity-feasibility-search",
            title="Run a constrained adaptive passivity search",
            triggered_by=sorted(
                codes
                & {
                    "NO_PASSIVE_TRIAL",
                    "PASSIVITY_ENFORCEMENT_DISABLED",
                    "TRAIN_PASSIVE_VERIFY_NONPASSIVE",
                    "TRAINING_SAFEGUARD_NOT_PASSIVE",
                    "MARGINAL_SIGMA_EXCURSION",
                    "MODERATE_SIGMA_EXCURSION",
                    "MATERIAL_SIGMA_EXCURSION",
                    "ERROR_IMPROVES_WITHOUT_FEASIBILITY",
                    "LARGE_RF_CONTRACTION",
                }
            ),
            rationale="Search response accuracy and passivity controls together instead of continuing an error-only trajectory.",
            command=command,
            changes=changes,
            notes=notes,
            requires_editing=requires_editing,
        )

    if "PASSIVE_TRIAL_AVAILABLE" in codes and model in {"dnn", "kbnn"}:
        selection = metric_name or "rmse_abs"
        command = [
            "python3",
            "surrogate.py",
            "--model",
            model,
            "rerank-sweep",
            "--sweep-dir",
            command_path(run_dir),
            "--selection-metric",
            selection,
            "--require-passive",
        ]
        add_command_suggestion(
            suggestions,
            identifier="rerank-passive-trials",
            title="Re-rank only the passive trials",
            triggered_by=["PASSIVE_TRIAL_AVAILABLE"],
            rationale="Once feasibility exists, choose the lowest-error member of the passive subset.",
            command=command,
            changes=[
                {
                    "option": "selection filter",
                    "observed": "mixed passive and non-passive trials",
                    "suggested": "--require-passive",
                    "reason": "Prevent a lower-error non-passive model from winning the ranking.",
                }
            ],
            notes=(
                "Reranking does not retrain. Promotion still requires retained model.npz files, but the ranking report works from saved summaries.",
            ),
        )

    if "MODEL_METADATA_MISSING" in codes and model in {"dnn", "kbnn"}:
        base, requires_editing = model_optimize_command_base(args, model)
        output_dir = run_dir.parent / f"{run_dir.name}_metadata_refresh"
        command = [
            *base,
            "--max-trials",
            "4",
            "--out-dir",
            command_path(output_dir),
        ]
        add_command_suggestion(
            suggestions,
            identifier="refresh-legacy-metadata",
            title="Refresh a legacy run with retained trial metadata",
            triggered_by=["MODEL_METADATA_MISSING"],
            rationale="Current runs retain metadata.json automatically, so a small controlled rerun restores architecture and training-setting evidence.",
            command=command,
            changes=[
                {
                    "option": "--max-trials",
                    "observed": "legacy metadata unavailable",
                    "suggested": "4",
                    "reason": "A small diagnostic run is usually sufficient to restore metadata without repeating the full search.",
                }
            ],
            notes=(
                "Add --keep-trial-models only if the corresponding model.npz weights must also survive.",
            ),
            requires_editing=requires_editing,
        )

    return suggestions


def improvement_fraction(values: Sequence[float]) -> float | None:
    if len(values) < 4:
        return None
    midpoint = max(1, len(values) // 2)
    early = min(values[:midpoint])
    late = min(values[midpoint:])
    return (early - late) / max(abs(early), 1.0e-300)


def add_finding(
    findings: list[dict[str, str]],
    severity: str,
    code: str,
    reason: str,
    action: str,
) -> None:
    findings.append(
        {"severity": severity, "code": code, "reason": reason, "action": action}
    )


def build_findings(
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
    audit: Mapping[str, object] | None,
    metadata_records: Sequence[tuple[Path, Mapping[str, object]]],
    model: str,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if audit is None:
        add_finding(
            findings,
            "WARNING",
            "RAW_DATA_AUDIT_MISSING",
            "No dataset_audit.json was supplied or found beside the run.",
            "Run `python3 surrogate.py audit ...` and pass its directory with --audit.",
        )
    else:
        raw_passivity = nested_dict(audit, "passivity")
        raw_rf_violations = integer(raw_passivity.get("violating_rf_rows")) or 0
        if raw_rf_violations:
            add_finding(
                findings,
                "ERROR",
                "RAW_RF_DATA_NONPASSIVE",
                f"The audit reports {raw_rf_violations} non-passive positive-frequency row(s).",
                "Inspect dataset_passivity.csv first; auto enforcement is disabled by non-passive training targets.",
            )
        else:
            add_finding(
                findings,
                "OK",
                "RAW_RF_DATA_PASSIVE",
                "The supplied audit reports no non-passive RF rows.",
                "Continue by comparing sampled-training passivity with verification passivity.",
            )

    if not rows:
        add_finding(
            findings,
            "ERROR",
            "NO_VERIFICATION_RESULTS",
            "No sweep CSV or verification_summary.json could be read.",
            "Point --run-dir at a completed train/optimize output directory.",
        )
        return findings

    violations = [
        value
        for row in rows
        if (value := integer(row.get("passivity_violating_points"))) is not None
    ]
    sigmas = [
        value
        for row in rows
        if (value := number(row.get("passivity_max_singular_value"))) is not None
    ]
    passive_count = sum(value == 0 for value in violations)
    if violations and passive_count == 0:
        add_finding(
            findings,
            "ERROR",
            "NO_PASSIVE_TRIAL",
            f"All {len(violations)} assessed trial(s) violate verification passivity; the best has {min(violations)} violating row(s).",
            "Use the following findings to distinguish disabled enforcement, raw-data conflict, and unseen-domain excursions.",
        )
    elif passive_count:
        add_finding(
            findings,
            "OK",
            "PASSIVE_TRIAL_AVAILABLE",
            f"{passive_count} of {len(violations)} assessed trial(s) have zero verification passivity violations.",
            "Rank the passive subset by the desired response-error metric.",
        )

    enforcement = [row.get("passivity_enforced") for row in rows]
    known_enforcement = [value for value in enforcement if isinstance(value, bool)]
    if known_enforcement and not any(known_enforcement):
        reasons = sorted(
            {
                str(row.get("passivity_unavailable_reason") or "").strip()
                for row in rows
                if str(row.get("passivity_unavailable_reason") or "").strip()
            }
        )
        add_finding(
            findings,
            "ERROR",
            "PASSIVITY_ENFORCEMENT_DISABLED",
            "Every retained trial summary reports passivity_enforced=false."
            + (f" Reason: {'; '.join(reasons)}" if reasons else ""),
            "Confirm --passivity-mode enforce/auto and DNN --output-domain s with --explain-options.",
        )

    source_violations = [
        value
        for row in rows
        if (value := integer(row.get("source_rf_passivity_violating_points")))
        is not None
    ]
    if source_violations and any(value > 0 for value in source_violations):
        add_finding(
            findings,
            "ERROR",
            "TRAINING_SOURCE_NONPASSIVE",
            "At least one trial reports non-passive positive-frequency training targets.",
            "Auto mode will not enforce passivity; audit the exact training split or explicitly enforce only after resolving contradictory targets.",
        )

    train_after = [
        value
        for row in rows
        if (value := integer(row.get("train_after_scale_violating_points")))
        is not None
    ]
    if train_after and max(train_after) == 0 and violations and passive_count == 0:
        add_finding(
            findings,
            "WARNING",
            "TRAIN_PASSIVE_VERIFY_NONPASSIVE",
            "The saved responses are passive on sampled training rows but every trial leaves the passive set on verification rows.",
            "This is an interpolation/collocation problem; more EVM-targeted points alone may not remove the worst singular-value excursion.",
        )
    elif train_after and any(value > 0 for value in train_after):
        add_finding(
            findings,
            "ERROR",
            "TRAINING_SAFEGUARD_NOT_PASSIVE",
            "At least one retained trial remains non-passive on sampled training rows after its final safeguard.",
            "Inspect that trial directly; an enforced complete S-domain model should not have this result.",
        )

    if sigmas and passive_count == 0:
        minimum_sigma = min(sigmas)
        if minimum_sigma <= 1.001:
            add_finding(
                findings,
                "WARNING",
                "MARGINAL_SIGMA_EXCURSION",
                f"The best verification maximum singular value is {minimum_sigma:.9g}, only marginally above one.",
                "Try a controlled 0.005 passivity margin and inspect the RF response scale/loss tradeoff.",
            )
        elif minimum_sigma <= 1.01:
            add_finding(
                findings,
                "WARNING",
                "MODERATE_SIGMA_EXCURSION",
                f"The best verification maximum singular value is {minimum_sigma:.9g}.",
                "Tune the passivity penalty and margin, then add constraint points around the violating geometries/frequencies.",
            )
        else:
            add_finding(
                findings,
                "ERROR",
                "MATERIAL_SIGMA_EXCURSION",
                f"The best verification maximum singular value is {minimum_sigma:.9g}, too large to treat as numerical tolerance.",
                "Check raw data, model capacity/conditioning, and passivity-critical coverage before increasing global contraction.",
            )

    metric_values = [
        value
        for row in rows
        if metric_name and (value := number(row.get(metric_name))) is not None
    ]
    improvement = improvement_fraction(metric_values)
    if improvement is not None and improvement > 0.05 and passive_count == 0:
        add_finding(
            findings,
            "WARNING",
            "ERROR_IMPROVES_WITHOUT_FEASIBILITY",
            f"The best late-run {metric_name} improves by {100.0 * improvement:.2f}% over the best early-run value, but no passive trial appears.",
            "Response error and worst-case matrix passivity are decoupled; use passivity-aware checkpointing/acquisition rather than adding only EVM-targeted points.",
        )

    scales = [
        value
        for row in rows
        if (value := number(row.get("rf_response_scale"))) is not None
    ]
    if scales and min(scales) < 0.98:
        add_finding(
            findings,
            "WARNING",
            "LARGE_RF_CONTRACTION",
            f"At least one model uses rf_response_scale={min(scales):.6g}.",
            "Avoid solving verification passivity only by scaling; this directly adds RF loss.",
        )

    if model == "kbnn" and not any(
        "coarse" in str(path).lower() for path, _payload in metadata_records
    ):
        add_finding(
            findings,
            "INFO",
            "COARSE_METADATA_NOT_RETAINED",
            "No retained coarse-model metadata was found at the run root.",
            "Use --keep-trial-models for a diagnostic optimization if coarse versus fine passivity must be separated per trial.",
        )
    if not metadata_records:
        add_finding(
            findings,
            "INFO",
            "MODEL_METADATA_MISSING",
            "No metadata.json was found. Current optimize runs retain it for every completed model trial; this is expected only for a legacy cleaned run or a trial that failed before saving a model.",
            "This report can still use retained verification summaries; rerun only the affected configuration if model architecture details are required.",
        )
    return findings


def font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "DejaVuSans.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def write_diagnostic_plot(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    metric_name: str | None,
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Model debug PNG output requires Pillow") from exc
    plot_rows = [
        row
        for row in rows
        if integer(row.get("trial")) is not None
        and number(row.get("passivity_max_singular_value")) is not None
    ]
    if not plot_rows:
        return False
    width, height = 1400, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(30, bold=True)
    label_font = font(18, bold=True)
    text_font = font(15)
    draw.text((width / 2, 28), "Model error and verification passivity by trial", font=title_font, fill="#111827", anchor="ma")
    panels = [
        (90, 110, width - 150, 280, metric_name or "selection metric"),
        (90, 500, width - 150, 280, "maximum S-matrix singular value"),
    ]
    trial_values = [integer(row.get("trial")) or 0 for row in plot_rows]
    metric_values = [
        number(row.get(metric_name)) if metric_name else None for row in plot_rows
    ]
    sigma_values = [
        float(number(row.get("passivity_max_singular_value")) or 0.0)
        for row in plot_rows
    ]
    series = [metric_values, sigma_values]
    for panel_index, (left, top, panel_width, panel_height, label) in enumerate(panels):
        draw.rectangle((left, top, left + panel_width, top + panel_height), outline="#94a3b8", width=2)
        draw.text((left, top - 34), label, font=label_font, fill="#1f2937")
        values = [value for value in series[panel_index] if value is not None]
        if not values:
            draw.text((left + panel_width / 2, top + panel_height / 2), "No values available", font=text_font, fill="#64748b", anchor="mm")
            continue
        low = min(values)
        high = max(values)
        if panel_index == 1:
            low = min(low, 1.0)
            high = max(high, 1.000001)
        pad = 0.08 * max(high - low, abs(high) * 1e-4, 1e-9)
        low -= pad
        high += pad

        def x_pos(trial: int) -> float:
            lo = min(trial_values)
            hi = max(trial_values)
            return left + panel_width / 2 if hi == lo else left + panel_width * (trial - lo) / (hi - lo)

        def y_pos(value: float) -> float:
            return top + panel_height * (high - value) / (high - low)

        for tick in range(5):
            value = low + tick * (high - low) / 4.0
            y = y_pos(value)
            draw.line((left, y, left + panel_width, y), fill="#e5e7eb", width=1)
            draw.text((left - 10, y), f"{value:.6g}", font=text_font, fill="#475569", anchor="rm")
        if panel_index == 1 and low <= 1.000001 <= high:
            y_limit = y_pos(1.000001)
            for x in range(left, left + panel_width, 18):
                draw.line((x, y_limit, min(x + 10, left + panel_width), y_limit), fill="#dc2626", width=2)
            draw.text((left + panel_width - 4, y_limit - 6), "passivity limit", font=text_font, fill="#dc2626", anchor="rs")
        points: list[tuple[float, float]] = []
        for row, trial, raw_value in zip(plot_rows, trial_values, series[panel_index]):
            if raw_value is None:
                continue
            x = x_pos(trial)
            y = y_pos(float(raw_value))
            points.append((x, y))
            passive = integer(row.get("passivity_violating_points")) == 0
            color = "#16a34a" if passive else "#dc2626"
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=1)
        if len(points) > 1:
            draw.line(points, fill="#64748b", width=2)
            for row, trial, raw_value in zip(plot_rows, trial_values, series[panel_index]):
                if raw_value is None:
                    continue
                x = x_pos(trial)
                y = y_pos(float(raw_value))
                passive = integer(row.get("passivity_violating_points")) == 0
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="#16a34a" if passive else "#dc2626", outline="white", width=1)
        draw.text((left + panel_width / 2, top + panel_height + 34), "optimization trial", font=label_font, fill="#1f2937", anchor="ma")
    draw.ellipse((100, 835, 116, 851), fill="#16a34a")
    draw.text((126, 843), "passivity OK", font=text_font, fill="#334155", anchor="lm")
    draw.ellipse((280, 835, 296, 851), fill="#dc2626")
    draw.text((306, 843), "passivity violation", font=text_font, fill="#334155", anchor="lm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", dpi=(144, 144), optimize=True)
    return True


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row)
        + " |"
        for row in rows
    )
    return "\n".join(lines)


def suggested_commands_markdown(
    suggestions: Sequence[Mapping[str, object]],
) -> list[str]:
    lines = [
        "## Suggested command changes",
        "",
        "These commands are generated from the findings above. Paths are relative to the current working directory. Review the change table before running a command.",
        "",
    ]
    if not suggestions:
        lines.extend(
            [
                "No command change is justified by the available evidence. Resolve missing inputs or inspect the detailed trial table first.",
                "",
            ]
        )
        return lines
    for index, suggestion in enumerate(suggestions, start=1):
        lines.extend(
            [
                f"### {index}. {suggestion['title']}",
                "",
                f"Triggered by: `{', '.join(str(value) for value in suggestion.get('triggered_by', []))}`",
                "",
                str(suggestion.get("rationale") or ""),
                "",
            ]
        )
        changes = suggestion.get("changes")
        if isinstance(changes, list) and changes:
            lines.extend(
                [
                    markdown_table(
                        ["Option or control", "Observed", "Suggested", "Why"],
                        [
                            [
                                change.get("option", "") if isinstance(change, dict) else "",
                                change.get("observed", "") if isinstance(change, dict) else "",
                                change.get("suggested", "") if isinstance(change, dict) else "",
                                change.get("reason", "") if isinstance(change, dict) else "",
                            ]
                            for change in changes
                        ],
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                "```bash",
                str(suggestion.get("command") or ""),
                "```",
                "",
            ]
        )
        if suggestion.get("requires_editing"):
            lines.extend(
                [
                    "> Replace the `PATH_TO_*` placeholder(s) before running this command. Passing the project options JSON to `debug-model` allows future reports to reuse the configured data paths automatically.",
                    "",
                ]
            )
        notes = suggestion.get("notes")
        if isinstance(notes, list):
            lines.extend(f"- {note}" for note in notes)
            if notes:
                lines.append("")
    return lines


def build_parser() -> argparse.ArgumentParser:
    dispatcher_prog = os.environ.get("ADS_SURROGATE_CLI_PROG")
    parser = argparse.ArgumentParser(
        prog=dispatcher_prog or None,
        description=(
            "Diagnose DNN/KBNN fitting and passivity from retained train or "
            "optimization artifacts; per-trial metadata.json is optional."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Completed train, sweep, or optimize output directory.")
    parser.add_argument("--audit", help="dataset_audit.json or its containing audit directory.")
    parser.add_argument("--model", choices=["auto", "dnn", "kbnn"], default="auto", help="Model family. Default: infer from artifacts.")
    parser.add_argument("--out-dir", help="Diagnostic output directory. Default: <run-dir>/model_debug.")
    parser.add_argument("--top", type=int, default=12, help="Number of lowest-violation trials in the Markdown table. Default: 12.")
    add_options_json_argument(parser, recursive=False)
    return parser


def command_debug(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_dir():
        parser.error(f"Run directory does not exist: {run_dir}")
    if args.top <= 0:
        parser.error("--top must be positive")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else run_dir / "model_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = find_results_file(run_dir)
    rows = read_csv(results_path) if results_path else []
    metric_name = selection_metric(rows)
    if rows:
        rows, summaries_found = enrich_trial_rows(run_dir, rows, metric_name)
    else:
        rows, summaries_found = single_model_rows(run_dir)
        metric_name = selection_metric(rows)
    model = infer_model(run_dir, results_path) if args.model == "auto" else args.model
    audit_path = resolve_audit_json(args.audit, run_dir)
    audit = read_json(audit_path) if audit_path else None
    metadata_records = surviving_metadata(run_dir)
    findings = build_findings(rows, metric_name, audit, metadata_records, model)
    ordered = ranked_trial_rows(rows, metric_name)
    suggestions = build_command_suggestions(
        args,
        run_dir,
        model,
        rows,
        metric_name,
        findings,
        metadata_records,
    )

    trials_csv = out_dir / "model_debug_trials.csv"
    report_json = out_dir / "model_debug.json"
    report_md = out_dir / "model_debug.md"
    plot_path = out_dir / "model_debug_passivity.png"
    write_csv(trials_csv, rows)
    plot_written = write_diagnostic_plot(plot_path, rows, metric_name)
    violations = [value for row in rows if (value := integer(row.get("passivity_violating_points"))) is not None]
    sigmas = [value for row in rows if (value := number(row.get("passivity_max_singular_value"))) is not None]
    metrics = [value for row in rows if metric_name and (value := number(row.get(metric_name))) is not None]
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "model": model,
        "results_file": str(results_path) if results_path else None,
        "selection_metric": metric_name,
        "trial_rows": len(rows),
        "verification_summaries_found": summaries_found,
        "metadata_files_found": [str(path) for path, _payload in metadata_records],
        "metadata_policy": "metadata.json is retained for every completed model trial; --keep-trial-models controls heavyweight model files.",
        "audit_file": str(audit_path) if audit_path else None,
        "statistics": {
            "passive_trials": sum(value == 0 for value in violations),
            "assessed_trials": len(violations),
            "minimum_violating_points": min(violations) if violations else None,
            "minimum_max_singular_value": min(sigmas) if sigmas else None,
            "median_max_singular_value": statistics.median(sigmas) if sigmas else None,
            "best_selection_metric": min(metrics) if metrics else None,
            "late_vs_early_best_metric_improvement_fraction": improvement_fraction(metrics),
        },
        "findings": findings,
        "suggested_commands": suggestions,
        "artifacts": {
            "report": report_md.name,
            "trials": trials_csv.name,
            "plot": plot_path.name if plot_written else None,
        },
    }
    report_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Model Fit and Passivity Debug Report",
        "",
        f"- Run: `{run_dir}`",
        f"- Model: `{model}`",
        f"- Results: `{results_path}`" if results_path else "- Results: single-model verification summary",
        f"- Audit: `{audit_path}`" if audit_path else "- Audit: not found",
        f"- Per-trial verification summaries found: `{summaries_found}`",
        "",
        "> Current optimize runs retain every completed trial's `metadata.json`, even without `--keep-trial-models`. Legacy cleaned runs remain supported through `verification_summary.json` and the sweep CSV.",
        "",
        "## Findings and recommended actions",
        "",
        markdown_table(
            ["Status", "Code", "Reason", "Recommended action"],
            [[item["severity"], item["code"], item["reason"], item["action"]] for item in findings],
        ),
        "",
    ]
    lines.extend(suggested_commands_markdown(suggestions))
    if plot_written:
        lines.extend(["## Error and passivity by trial", "", f"![Model passivity diagnostics]({plot_path.name})", ""])
    lines.extend(
        [
            "## Lowest-passivity-error trials",
            "",
            markdown_table(
                ["Trial", metric_name or "Metric", "Violating points", "Max sigma", "Enforced", "Train-after violations", "RF scale"],
                [
                    [
                        row.get("trial", ""),
                        row.get(metric_name, "") if metric_name else "",
                        row.get("passivity_violating_points", ""),
                        row.get("passivity_max_singular_value", ""),
                        row.get("passivity_enforced", ""),
                        row.get("train_after_scale_violating_points", ""),
                        row.get("rf_response_scale", ""),
                    ]
                    for row in ordered[: args.top]
                ],
            ),
            "",
            "## Artifact inventory",
            "",
            f"- Machine-readable diagnosis: `{report_json.name}`",
            f"- Enriched trial table: `{trials_csv.name}`",
            f"- Surviving metadata files: `{len(metadata_records)}`",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"model debug report: {report_md}")
    print(f"model debug data: {report_json}")
    print(f"model debug trials: {trials_csv}")
    if plot_written:
        print(f"model debug plot: {plot_path}")
    error_count = sum(item["severity"] == "ERROR" for item in findings)
    warning_count = sum(item["severity"] == "WARNING" for item in findings)
    print(f"model debug result: {error_count} error finding(s), {warning_count} warning finding(s)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parse_args_with_options_json(
        parser,
        raw_args,
        workflow="debug-model",
        command="debug-model",
    )
    status = command_debug(args, parser)
    return finalize_options_json_update(args, status)


if __name__ == "__main__":
    raise SystemExit(main())
