"""Recover reusable CLI options from an existing surrogate project tree.

Discovery is intentionally read-only.  It inventories durable JSON/CSV/report
artifacts, recovers primary-CLI commands embedded in those artifacts, and
returns a normal options document plus a separate provenance report.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from cli_options import (
    COMMAND_GROUPS,
    COMMAND_NAMES,
    MODEL_NAMES,
    NODE_HEADINGS,
    ROOT_HEADINGS,
    WORKFLOW_NAMES,
    _explicit_option_updates,
    _relax_parser_requirements_for_explanation,
    fit_shared_option_keys,
    normalize_command_name,
    normalize_model_name,
    starter_options_payload,
)


TEXT_COMMAND_SUFFIXES = {".md", ".txt", ".log", ".sh", ".command"}
MODEL_EXPORT_COMMANDS = (
    "predict",
    "export-ads-mdif",
    "export-ads-ann",
    "export-ads-hb",
    "export-veriloga",
)
SKIPPED_DOCUMENT_NAMES = {
    "readme.md",
    "sandbox_handoff.md",
    "options.example.json",
}


def json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_compatible(item) for item in value]
    if isinstance(value, list):
        return [json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    return value


def same_value(left: object, right: object) -> bool:
    return json_compatible(left) == json_compatible(right)


@dataclass
class SettingCandidate:
    value: object
    source: str
    priority: int
    modified_time: float


@dataclass
class DiscoveryAccumulator:
    root: Path
    settings: dict[tuple[str, ...], SettingCandidate] = field(default_factory=dict)
    setting_sources: dict[tuple[str, ...], list[str]] = field(default_factory=dict)
    artifacts: list[dict[str, object]] = field(default_factory=list)
    commands: list[dict[str, object]] = field(default_factory=list)
    conflicts: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def cli_path(self, path: Path) -> str:
        """Return a path usable from the process working directory."""

        try:
            return os.path.relpath(path.resolve(), Path.cwd().resolve())
        except (OSError, ValueError):
            return str(path)

    def add_artifact(self, path: Path, kind: str) -> None:
        entry = {"path": self.display_path(path), "kind": kind}
        if entry not in self.artifacts:
            self.artifacts.append(entry)

    def add(
        self,
        location: Sequence[str],
        value: object,
        source: Path | str,
        *,
        priority: int,
    ) -> None:
        if value is None:
            return
        key = tuple(str(part) for part in location)
        source_path = Path(source) if not isinstance(source, Path) else source
        source_text = self.display_path(source_path)
        try:
            modified_time = source_path.stat().st_mtime
        except OSError:
            modified_time = 0.0
        incoming = SettingCandidate(
            value=json_compatible(value),
            source=source_text,
            priority=priority,
            modified_time=modified_time,
        )
        incoming_rank = (incoming.priority, incoming.modified_time, incoming.source)
        prefix_conflicts = [
            existing_key
            for existing_key in self.settings
            if existing_key != key
            and (
                existing_key[: len(key)] == key
                or key[: len(existing_key)] == existing_key
            )
        ]
        if prefix_conflicts:
            strongest_key = max(
                prefix_conflicts,
                key=lambda item: (
                    self.settings[item].priority,
                    self.settings[item].modified_time,
                    self.settings[item].source,
                ),
            )
            strongest = self.settings[strongest_key]
            strongest_rank = (
                strongest.priority,
                strongest.modified_time,
                strongest.source,
            )
            if incoming_rank <= strongest_rank:
                self.conflicts.append(
                    {
                        "setting": ".".join(key),
                        "selected_setting": ".".join(strongest_key),
                        "selected_value": strongest.value,
                        "selected_source": strongest.source,
                        "other_value": incoming.value,
                        "other_source": incoming.source,
                        "resolution": "scalar/object path collision; higher-ranked setting retained",
                    }
                )
                return
            for existing_key in prefix_conflicts:
                rejected = self.settings.pop(existing_key)
                self.setting_sources.pop(existing_key, None)
                self.conflicts.append(
                    {
                        "setting": ".".join(key),
                        "selected_setting": ".".join(key),
                        "selected_value": incoming.value,
                        "selected_source": incoming.source,
                        "other_setting": ".".join(existing_key),
                        "other_value": rejected.value,
                        "other_source": rejected.source,
                        "resolution": "scalar/object path collision; higher-ranked setting retained",
                    }
                )
        existing = self.settings.get(key)
        sources = self.setting_sources.setdefault(key, [])
        if source_text not in sources:
            sources.append(source_text)
        if existing is None:
            self.settings[key] = incoming
            return
        if same_value(existing.value, incoming.value):
            if (incoming.priority, incoming.modified_time, incoming.source) > (
                existing.priority,
                existing.modified_time,
                existing.source,
            ):
                self.settings[key] = incoming
            return

        existing_rank = (existing.priority, existing.modified_time, existing.source)
        winner, rejected = (
            (incoming, existing) if incoming_rank > existing_rank else (existing, incoming)
        )
        self.settings[key] = winner
        self.conflicts.append(
            {
                "setting": ".".join(key),
                "selected_value": winner.value,
                "selected_source": winner.source,
                "other_value": rejected.value,
                "other_source": rejected.source,
                "resolution": (
                    "higher-confidence artifact"
                    if winner.priority != rejected.priority
                    else "newer artifact of equal confidence"
                ),
            }
        )

    def payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"schema_version": 1}
        for location, candidate in sorted(self.settings.items()):
            node = payload
            for part in location[:-1]:
                child = node.setdefault(part, {})
                if not isinstance(child, dict):
                    raise ValueError(
                        f"Discovered setting {'.'.join(location)} collides with "
                        f"the scalar setting {part!r}"
                    )
                node = child
            node[location[-1]] = candidate.value
        return payload


def option_location(
    *, model: str | None, workflow: str | None, command: str, key: str
) -> tuple[str, ...]:
    if model is not None:
        return (
            "models",
            normalize_model_name(model),
            "commands",
            normalize_command_name(command),
            key,
        )
    if workflow is not None:
        return (
            "workflows",
            workflow,
            "commands",
            normalize_command_name(command),
            key,
        )
    raise ValueError("A discovered command needs a model or workflow scope")


def iter_json_leaves(
    value: object, prefix: tuple[str, ...] = ()
) -> Iterable[tuple[tuple[str, ...], object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from iter_json_leaves(child, (*prefix, str(key)))
        return
    yield prefix, value


def _options_leaf_mapping(value: object, context: str) -> str | None:
    if not isinstance(value, dict):
        return f"{context} must be an object"
    for key, option_value in value.items():
        if isinstance(option_value, dict):
            return (
                f"{context}.{key} is an object, but an option value must be a "
                "scalar, array, boolean, or null"
            )
    return None


def _options_commands_mapping(value: object, context: str) -> str | None:
    if not isinstance(value, dict):
        return f"{context} must be an object"
    for raw_command, options in value.items():
        command = normalize_command_name(str(raw_command))
        if command not in COMMAND_NAMES | COMMAND_GROUPS:
            return f"{context} contains unknown command heading {raw_command!r}"
        problem = _options_leaf_mapping(options, f"{context}.{raw_command}")
        if problem:
            return problem
    return None


def _options_scope_node(value: object, context: str) -> str | None:
    if not isinstance(value, dict):
        return f"{context} must be an object"
    for heading in ("generic", "common"):
        if heading in value:
            problem = _options_leaf_mapping(value[heading], f"{context}.{heading}")
            if problem:
                return problem
    if "commands" in value:
        problem = _options_commands_mapping(value["commands"], f"{context}.commands")
        if problem:
            return problem
    for key, option_value in value.items():
        if key not in NODE_HEADINGS and isinstance(option_value, dict):
            return (
                f"{context}.{key} is an object, but a flat option value must be "
                "a scalar, array, boolean, or null"
            )
    return None


def options_document_problem(payload: object) -> tuple[bool, str | None]:
    """Return whether JSON looks like options and why it is not valid, if so."""

    if not isinstance(payload, dict):
        return False, None
    headings = set(payload) - {"schema_version"}
    candidate = bool(headings & ROOT_HEADINGS) and not {
        "generation_kind",
        "geometry_file",
    }.intersection(payload)
    if not candidate:
        return False, None
    if payload.get("schema_version", 1) != 1:
        return True, f"unsupported schema_version {payload.get('schema_version')!r}"
    for heading in ("generic", "common"):
        if heading in payload:
            problem = _options_leaf_mapping(payload[heading], heading)
            if problem:
                return True, problem
    if "commands" in payload:
        problem = _options_commands_mapping(payload["commands"], "commands")
        if problem:
            return True, problem
    models = payload.get("models")
    if models is not None:
        if not isinstance(models, dict):
            return True, "models must be an object"
        for raw_model, node in models.items():
            if raw_model in {"generic", "common"}:
                problem = _options_leaf_mapping(node, f"models.{raw_model}")
            elif raw_model == "commands":
                problem = _options_commands_mapping(node, "models.commands")
            elif normalize_model_name(str(raw_model)) in MODEL_NAMES:
                problem = _options_scope_node(node, f"models.{raw_model}")
            else:
                return True, f"models contains unknown model heading {raw_model!r}"
            if problem:
                return True, problem
    workflows = payload.get("workflows")
    if workflows is not None:
        if not isinstance(workflows, dict):
            return True, "workflows must be an object"
        for raw_workflow, node in workflows.items():
            workflow = str(raw_workflow).strip().lower().replace("_", "-")
            if workflow not in WORKFLOW_NAMES:
                return True, (
                    f"workflows contains unknown workflow heading {raw_workflow!r}"
                )
            problem = _options_scope_node(node, f"workflows.{raw_workflow}")
            if problem:
                return True, problem
    for key, option_value in payload.items():
        if key not in ROOT_HEADINGS and isinstance(option_value, dict):
            return True, (
                f"{key} is an object, but a root option value must be a scalar, "
                "array, boolean, or null"
            )
    return True, None


def recover_options_document(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    accumulator.add_artifact(path, "options_json")
    for location, value in iter_json_leaves(payload):
        if not location or location == ("schema_version",) or value is None:
            continue
        accumulator.add(location, value, path, priority=100)


def format_number(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


def parameter_specs(payload: Mapping[str, object]) -> list[str]:
    specs: list[str] = []
    parameters = payload.get("parameters")
    if not isinstance(parameters, list):
        return specs
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        name = str(parameter.get("name") or "").strip()
        declared_range = parameter.get("range")
        if not name or not isinstance(declared_range, dict):
            continue
        lower = declared_range.get("lower")
        upper = declared_range.get("upper")
        if lower is None or upper is None:
            continue
        unit = str(declared_range.get("unit") or "")
        scale = str(parameter.get("scale") or "linear").lower()
        spec = (
            f"{name}={format_number(lower)}{unit}:"
            f"{format_number(upper)}{unit}"
        )
        if scale != "linear":
            spec += f":{scale}"
        specs.append(spec)
    return specs


def geometry_file_path(metadata_path: Path, payload: Mapping[str, object]) -> Path:
    raw = str(payload.get("geometry_file") or "").strip()
    if raw:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else metadata_path.parent / candidate
    return metadata_path.with_suffix(".csv")


def recover_geometry_metadata(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    generation_kind = str(payload.get("generation_kind") or "")
    if not generation_kind or not isinstance(payload.get("parameters"), list):
        return
    accumulator.add_artifact(path, f"geometry_{generation_kind}")
    csv_path = geometry_file_path(path, payload)
    specs = parameter_specs(payload)
    parameter_names = [spec.split("=", 1)[0] for spec in specs]
    if parameter_names:
        accumulator.add(
            ("models", "commands", "fit", "parameter_names"),
            ",".join(parameter_names),
            path,
            priority=20,
        )
        accumulator.add(
            ("workflows", "audit", "commands", "audit", "parameter_names"),
            ",".join(parameter_names),
            path,
            priority=20,
        )
    accumulator.add(
        ("workflows", "audit", "commands", "audit", "geometry_json"),
        [accumulator.cli_path(path)],
        path,
        priority=20,
    )
    if generation_kind in {"generated", "range_extension"}:
        base = ("workflows", "points", "commands", "generate")
        accumulator.add((*base, "parameter"), specs, path, priority=35)
        accumulator.add(
            (*base, "out"), accumulator.cli_path(csv_path), path, priority=35
        )
        accumulator.add((*base, "count"), payload.get("point_count"), path, priority=35)
        split_counts = payload.get("split_counts")
        if isinstance(split_counts, dict):
            verification_count = sum(
                int(value)
                for key, value in split_counts.items()
                if str(key).strip().lower()
                in {"verify", "verification", "test", "validation"}
            )
            accumulator.add(
                (*base, "verification_count"),
                verification_count,
                path,
                priority=35,
            )
        method = payload.get("method")
        if method:
            accumulator.add((*base, "method"), method, path, priority=35)
        accumulator.add(
            (*base, "decimal_places"),
            payload.get("decimal_places"),
            path,
            priority=35,
        )
        split_var = payload.get("split_variable")
        if split_var:
            accumulator.add((*base, "split_var"), split_var, path, priority=35)
        if generation_kind == "range_extension":
            accumulator.warnings.append(
                f"{accumulator.display_path(path)} records the resulting extended "
                "domain, but not every original --extend-range invocation detail; "
                "the discovered generate settings reproduce the complete domain."
            )

        suggest_base = (
            "workflows",
            "points",
            "commands",
            "suggest-additional",
        )
        accumulator.add(
            (*suggest_base, "parameter_json"),
            accumulator.cli_path(path),
            path,
            priority=20,
        )
        accumulator.add(
            (*suggest_base, "existing_points"),
            [accumulator.cli_path(csv_path)],
            path,
            priority=20,
        )

    if generation_kind not in {
        "targeted_additional",
        "accumulated_training_geometries",
    }:
        return
    base = ("workflows", "points", "commands", "suggest-additional")
    accumulator.add(
        (*base, "parameter_json"), accumulator.cli_path(path), path, priority=45
    )
    accumulator.add((*base, "metric"), payload.get("analysis_metric"), path, priority=45)
    accumulator.add(
        (*base, "acquisition"), payload.get("acquisition_method"), path, priority=45
    )
    accumulator.add(
        (*base, "candidate_method"), payload.get("candidate_method"), path, priority=45
    )
    accumulator.add(
        (*base, "bare_values"),
        payload.get("bare_values_mode") or (
            "auto" if payload.get("bare_values_interpretation") else None
        ),
        path,
        priority=45,
    )
    accumulator.add(
        (*base, "decimal_places"), payload.get("decimal_places"), path, priority=45
    )
    accumulator.add(
        (*base, "verification_metrics"),
        payload.get("verification_metrics_source"),
        path,
        priority=45,
    )
    gp = payload.get("gp")
    if isinstance(gp, dict):
        accumulator.add(
            (*base, "exploration_weight"),
            gp.get("exploration_weight"),
            path,
            priority=45,
        )
        if gp.get("length_scale_selection") == "user":
            accumulator.add(
                (*base, "gp_length_scale"),
                gp.get("length_scale"),
                path,
                priority=45,
            )
        accumulator.add(
            (*base, "gp_noise_variance"),
            gp.get("noise_variance"),
            path,
            priority=45,
        )
    if generation_kind == "targeted_additional":
        accumulator.add((*base, "out"), accumulator.cli_path(csv_path), path, priority=40)
        accumulator.add((*base, "count"), payload.get("point_count"), path, priority=40)
        return

    accumulator.add(
        (*base, "existing_points"),
        [accumulator.cli_path(csv_path)],
        path,
        priority=55,
    )
    accumulator.add(
        (*base, "combined_out"), accumulator.cli_path(csv_path), path, priority=50
    )
    additional = payload.get("additional_points_file")
    if additional:
        additional_path = Path(str(additional))
        if not additional_path.is_absolute():
            additional_path = path.parent / additional_path
        accumulator.add(
            (*base, "out"), accumulator.cli_path(additional_path), path, priority=50
        )
    accumulator.add(
        (*base, "count"), payload.get("new_point_count"), path, priority=50
    )


def infer_model_type(payload: Mapping[str, object]) -> str | None:
    if "n_poles" in payload and "n_coeffs_per_sparam" in payload:
        return "neuro-tf"
    if "mode" in payload and "include_coarse_input" in payload:
        return "kbnn"
    if "output_domain" in payload and "layer_sizes" in payload:
        return "dnn"
    return None


def recover_model_metadata(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    model = infer_model_type(payload)
    if model is None:
        return
    accumulator.add_artifact(path, f"{model}_model_metadata")
    base = ("models", model, "commands", "train")
    accumulator.add((*base, "out_dir"), accumulator.cli_path(path.parent), path, priority=30)
    names = payload.get("parameter_names")
    if isinstance(names, list) and names:
        accumulator.add(
            (*base, "parameter_names"),
            ",".join(map(str, names)),
            path,
            priority=30,
        )
    layer_sizes = payload.get("layer_sizes")
    if isinstance(layer_sizes, list) and len(layer_sizes) >= 3:
        accumulator.add(
            (*base, "hidden_layers"),
            ",".join(str(value) for value in layer_sizes[1:-1]),
            path,
            priority=30,
        )
    mapping = {
        "activation": "activation",
        "freq_transform": "freq_transform",
        "output_domain": "output_domain",
        "mode": "mode",
        "include_coarse_input": "include_coarse_input",
        "frequency_weights": "frequency_weights",
        "sparam_weights": "sparam_weights",
        "passivity_mode": "passivity_mode",
        "passivity_margin": "passivity_margin",
        "passivity_penalty": "passivity_penalty",
        "reciprocity_mode": "reciprocity_mode",
        "reciprocity_tolerance": "reciprocity_tolerance",
        "split_var": "split_var",
        "target_z0": "target_z0",
        "pole_damping": "pole_damping",
        "ridge": "ridge",
    }
    for metadata_key, option_key in mapping.items():
        accumulator.add(
            (*base, option_key), payload.get(metadata_key), path, priority=30
        )
    if model == "neuro-tf":
        accumulator.add((*base, "order"), payload.get("n_poles"), path, priority=30)
    inferred_outputs = {
        "predict": ("out_mdif", path.parent / "predicted.mdif"),
        "export-ads-mdif": ("out_dir", path.parent / "ads_mdif_export"),
        "export-ads-ann": ("out_dir", path.parent / "ads_ann_export"),
        "export-ads-hb": ("out_dir", path.parent / "ads_hb_export"),
        "export-veriloga": ("out_dir", path.parent / "veriloga_export"),
    }
    for command in MODEL_EXPORT_COMMANDS:
        if command == "export-ads-ann" and model == "neuro-tf":
            continue
        accumulator.add(
            ("models", model, "commands", command, "model_dir"),
            accumulator.cli_path(path.parent),
            path,
            priority=25,
        )
        output_key, output_path = inferred_outputs[command]
        accumulator.add(
            ("models", model, "commands", command, output_key),
            accumulator.cli_path(output_path),
            path,
            priority=15,
        )


def recover_audit_summary(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.name != "dataset_audit.json" or "verdict" not in payload:
        return
    accumulator.add_artifact(path, "dataset_audit")
    base = ("workflows", "audit", "commands", "audit")
    accumulator.add((*base, "out_dir"), accumulator.cli_path(path.parent), path, priority=30)
    parameter_names = payload.get("parameter_names")
    if isinstance(parameter_names, list) and parameter_names:
        accumulator.add(
            (*base, "parameter_names"),
            ",".join(map(str, parameter_names)),
            path,
            priority=30,
        )
    passivity = payload.get("passivity")
    if isinstance(passivity, dict) and passivity.get("limit") is not None:
        try:
            tolerance = float(passivity["limit"]) - 1.0
        except (TypeError, ValueError):
            tolerance = None
        accumulator.add(
            (*base, "passivity_tolerance"), tolerance, path, priority=30
        )
    coverage = payload.get("coverage_domain")
    if isinstance(coverage, dict):
        source_files = coverage.get("source_files")
        if isinstance(source_files, list) and source_files:
            accumulator.add((*base, "geometry_json"), source_files, path, priority=30)

    blocks_path = path.parent / "dataset_blocks.csv"
    if not blocks_path.is_file():
        return
    try:
        with blocks_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        accumulator.warnings.append(f"Could not read {blocks_path}: {exc}")
        return
    sources: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (
            str(row.get("dataset_kind") or row.get("dataset") or "fine").lower(),
            str(row.get("role") or "").lower(),
        )
        source = str(row.get("source_file") or "").strip()
        if source and source not in sources.setdefault(key, []):
            sources[key].append(source)
    fine_train = sources.get(("fine", "train"), [])
    fine_verify = sources.get(("fine", "verification"), [])
    coarse_train = sources.get(("coarse", "train"), [])
    coarse_verify = sources.get(("coarse", "verification"), [])
    if fine_train:
        accumulator.add((*base, "mdif"), fine_train[0], blocks_path, priority=35)
        accumulator.add(
            ("models", "commands", "fit", "mdif"),
            fine_train[0],
            blocks_path,
            priority=25,
        )
    if fine_verify and fine_verify[0] not in fine_train:
        accumulator.add(
            (*base, "verification_mdif"), fine_verify[0], blocks_path, priority=35
        )
        accumulator.add(
            ("models", "commands", "fit", "verification_mdif"),
            fine_verify[0],
            blocks_path,
            priority=25,
        )
    if coarse_train:
        accumulator.add((*base, "coarse_mdif"), coarse_train[0], blocks_path, priority=35)
        accumulator.add(
            ("models", "kbnn", "commands", "fit", "coarse_mdif"),
            coarse_train[0],
            blocks_path,
            priority=25,
        )
    if coarse_verify and coarse_verify[0] not in coarse_train:
        accumulator.add(
            (*base, "coarse_verification_mdif"),
            coarse_verify[0],
            blocks_path,
            priority=35,
        )
        accumulator.add(
            (
                "models",
                "kbnn",
                "commands",
                "fit",
                "coarse_verification_mdif",
            ),
            coarse_verify[0],
            blocks_path,
            priority=25,
        )


def model_type_from_artifact_name(path: Path) -> str | None:
    name = path.name.lower()
    if name.startswith("dnn_"):
        return "dnn"
    if name.startswith("kbnn_"):
        return "kbnn"
    if name.startswith("neurotf_") or name.startswith("neuro_tf_"):
        return "neuro-tf"
    return None


def recover_best_config(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if "best_config" not in path.name.lower():
        return
    model = model_type_from_artifact_name(path)
    if model is None:
        return
    accumulator.add_artifact(path, "optimize_best_config")
    base = ("models", model, "commands", "optimize")
    accumulator.add((*base, "out_dir"), accumulator.cli_path(path.parent), path, priority=50)
    if model in {"dnn", "kbnn"}:
        accumulator.add(
            ("models", model, "commands", "rerank-sweep", "sweep_dir"),
            accumulator.cli_path(path.parent),
            path,
            priority=50,
        )
    for key in (
        "selection_metric",
        "require_passive",
        "max_passivity_violations",
        "max_passivity_sigma",
        "trial_seed_mode",
    ):
        accumulator.add((*base, key), payload.get(key), path, priority=50)
    config = payload.get("config")
    if not isinstance(config, dict):
        config = payload.get("best_available_config")
    if not isinstance(config, dict):
        return
    option_keys = {
        "freq_transform": "freq_transform",
        "hidden_layers": "hidden_layers",
        "activation": "activation",
        "learning_rate": "learning_rate",
        "order": "order",
        "pole_damping": "pole_damping",
        "ridge": "ridge",
        "mode": "mode",
    }
    for candidate_key, option_key in option_keys.items():
        accumulator.add(
            (*base, option_key), config.get(candidate_key), path, priority=50
        )
    if "include_coarse_input" in config:
        include_coarse = config["include_coarse_input"]
        if isinstance(include_coarse, str):
            normalized = include_coarse.strip().lower()
            include_coarse = normalized in {"1", "true", "yes", "on"}
        if include_coarse:
            accumulator.add(
                (*base, "include_coarse_input"), True, path, priority=50
            )
        else:
            accumulator.add(
                (*base, "include_coarse_inputs"), "false", path, priority=50
            )


def recover_hb_report_summary(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.name != "ads_hb_solver_summary.json":
        return
    summaries = payload.get("summaries")
    if not isinstance(summaries, list) or not summaries:
        return
    logs: list[str] = []
    labels: list[str] = []
    for row in summaries:
        if not isinstance(row, dict):
            continue
        source_file = str(row.get("source_file") or "").strip()
        label = str(row.get("model") or "").strip()
        if source_file:
            logs.append(source_file)
            labels.append(label or Path(source_file).stem)
    if not logs:
        return
    accumulator.add_artifact(path, "hb_solver_report")
    base = ("workflows", "hb-report", "commands", "hb-report")
    accumulator.add((*base, "logs"), logs, path, priority=35)
    accumulator.add((*base, "labels"), labels, path, priority=35)
    accumulator.add(
        (*base, "out_dir"), accumulator.cli_path(path.parent), path, priority=35
    )


def recover_sweep_results(
    accumulator: DiscoveryAccumulator,
    path: Path,
) -> None:
    if not path.name.lower().endswith("_sweep_results.csv"):
        return
    model = model_type_from_artifact_name(path)
    if model is None:
        return
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        accumulator.warnings.append(f"Could not read {path}: {exc}")
        return
    if not rows:
        return
    accumulator.add_artifact(path, "optimize_trial_results")
    base = ("models", model, "commands", "optimize")
    accumulator.add((*base, "out_dir"), accumulator.cli_path(path.parent), path, priority=40)
    accumulator.add((*base, "max_trials"), len(rows), path, priority=40)
    column_options = {
        "freq_transform": ("freq_transforms", ","),
        "hidden_layers": ("hidden_layers", ";"),
        "activation": ("activations", ","),
        "learning_rate": ("learning_rates", ","),
        "mode": ("modes", ","),
        "include_coarse_input": ("include_coarse_inputs", ","),
        "order": ("orders", ","),
        "pole_damping": ("pole_dampings", ","),
        "ridge": ("ridges", ","),
    }
    for column, (option_key, delimiter) in column_options.items():
        values: list[str] = []
        for row in rows:
            value = str(row.get(column) or "").strip()
            if value and value not in values:
                values.append(value)
        if values:
            accumulator.add(
                (*base, option_key), delimiter.join(values), path, priority=40
            )
    accumulator.warnings.append(
        f"{accumulator.display_path(path)} reconstructs optimize candidate choices "
        "from completed trial rows; configured choices that were never attempted "
        "cannot be recovered from this CSV."
    )


def command_strings_from_json(value: object) -> Iterable[str]:
    if isinstance(value, str):
        if "surrogate.py" in value:
            yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from command_strings_from_json(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from command_strings_from_json(child)


def command_strings_from_text(text: str) -> Iterable[str]:
    joined = text.replace("\\\r\n", " ").replace("\\\n", " ")
    for line in joined.splitlines():
        if "surrogate.py" in line:
            yield line.strip().strip("`")


def tokenize_surrogate_command(text: str) -> list[str] | None:
    cleaned = text.strip().lstrip("$> ").strip().strip("`")
    try:
        tokens = shlex.split(cleaned)
    except ValueError:
        return None
    start = next(
        (
            index
            for index, token in enumerate(tokens)
            if Path(token).name == "surrogate.py"
        ),
        None,
    )
    if start is None:
        return None
    command = tokens[start + 1 :]
    for separator in ("&&", "||", ";", "|"):
        if separator in command:
            command = command[: command.index(separator)]
    return command or None


def remove_meta_options(tokens: Sequence[str]) -> list[str]:
    clean: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--update-options-json", "--explain-options", "--show-options"}:
            index += 1
            continue
        if token == "--options-json":
            index += 2
            continue
        if token.startswith("--options-json="):
            index += 1
            continue
        clean.append(token)
        index += 1
    return clean


def parser_for_route(
    *, model: str | None, workflow: str | None, command: str
) -> argparse.ArgumentParser:
    if model == "dnn":
        import dnn

        return dnn.build_arg_parser()
    if model == "kbnn":
        import kbnn

        return kbnn.build_arg_parser()
    if model == "neuro-tf":
        import neuro_tf

        return neuro_tf.build_arg_parser()
    if workflow == "points":
        import generate_points

        return (
            generate_points.build_suggest_parser()
            if command == "suggest-additional"
            else generate_points.build_generate_parser()
        )
    if workflow == "audit":
        import audit_dataset

        return audit_dataset.build_parser()
    if workflow == "hb-report":
        from de_generated_scripts import parse_ads_hb_solver_log

        return parse_ads_hb_solver_log.build_arg_parser()
    raise ValueError("Unsupported discovered command route")


def decode_route(
    raw_tokens: Sequence[str],
) -> tuple[str | None, str | None, str, list[str]] | None:
    tokens = remove_meta_options(raw_tokens)
    model: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--model" and index + 1 < len(tokens):
            model = normalize_model_name(tokens[index + 1])
            del tokens[index : index + 2]
            continue
        if token.startswith("--model="):
            model = normalize_model_name(token.split("=", 1)[1])
            del tokens[index]
            continue
        index += 1
    if model is not None:
        if not tokens:
            return None
        command = normalize_command_name(tokens[0])
        return model, None, command, tokens
    if not tokens:
        return None
    workflow = tokens.pop(0).strip().lower()
    if workflow == "points":
        command = (
            normalize_command_name(tokens.pop(0))
            if tokens and tokens[0] in {"generate", "suggest-additional"}
            else "generate"
        )
        return None, workflow, command, tokens
    if workflow in {"audit", "hb-report"}:
        return None, workflow, workflow, tokens
    return None


def recover_command(
    accumulator: DiscoveryAccumulator,
    source: Path,
    command_text: str,
    *,
    priority: int,
) -> None:
    raw_tokens = tokenize_surrogate_command(command_text)
    if raw_tokens is None:
        return
    route = decode_route(raw_tokens)
    if route is None:
        return
    model, workflow, command, parser_tokens = route
    try:
        parser = parser_for_route(model=model, workflow=workflow, command=command)
        _relax_parser_requirements_for_explanation(parser, command)
        with contextlib.redirect_stderr(io.StringIO()):
            args, unknown = parser.parse_known_args(parser_tokens)
        if unknown:
            accumulator.warnings.append(
                f"Ignored unrecognized tokens in a command from "
                f"{accumulator.display_path(source)}: {' '.join(unknown)}"
            )
        updates, _ = _explicit_option_updates(
            parser,
            parser_tokens,
            args,
            command=command,
        )
    except (Exception, SystemExit) as exc:
        accumulator.warnings.append(
            f"Could not parse a surrogate command from "
            f"{accumulator.display_path(source)}: {exc}"
        )
        return
    for key, value in updates.items():
        if key in {"options_json", "update_options_json", "explain_options"}:
            continue
        accumulator.add(
            option_location(
                model=model,
                workflow=workflow,
                command=command,
                key=key,
            ),
            value,
            source,
            priority=priority,
        )
        if model is not None and command in {"train", "optimize"}:
            if key in fit_shared_option_keys(model):
                accumulator.add(
                    ("models", model, "commands", "fit", key),
                    value,
                    source,
                    priority=max(1, priority - 1),
                )
    if model is not None and command.startswith("export-"):
        inferred_training_mdif = updates.get("template_mdif")
        if inferred_training_mdif is None and command == "export-ads-ann":
            inferred_training_mdif = updates.get("mdif")
        if inferred_training_mdif is not None:
            accumulator.add(
                ("models", model, "commands", "fit", "mdif"),
                inferred_training_mdif,
                source,
                priority=max(1, priority - 15),
            )
    accumulator.commands.append(
        {
            "source": accumulator.display_path(source),
            "model": model,
            "workflow": workflow,
            "command": command,
            "settings_recovered": len(updates),
            "text": " ".join(["surrogate.py", *raw_tokens]),
        }
    )


def inspect_json(
    accumulator: DiscoveryAccumulator,
    path: Path,
    payload: object,
) -> None:
    if not isinstance(payload, dict):
        return
    if {
        "recognized_artifacts",
        "recovered_commands",
        "setting_sources",
    }.issubset(payload):
        accumulator.add_artifact(path, "prior_discovery_report")
        return
    options_like, options_problem = options_document_problem(payload)
    if options_like and options_problem is None:
        recover_options_document(accumulator, path, payload)
    elif options_like:
        accumulator.warnings.append(
            f"Skipped options-like JSON {accumulator.display_path(path)}: "
            f"{options_problem}."
        )
    recover_geometry_metadata(accumulator, path, payload)
    recover_model_metadata(accumulator, path, payload)
    recover_audit_summary(accumulator, path, payload)
    recover_best_config(accumulator, path, payload)
    recover_hb_report_summary(accumulator, path, payload)
    saved_commands = list(command_strings_from_json(payload))
    if saved_commands:
        accumulator.add_artifact(
            path,
            "optimize_best_config"
            if "best_config" in path.name
            else "saved_command_json",
        )
    command_priority = 85 if "best_config" in path.name else 70
    for command_text in saved_commands:
        recover_command(
            accumulator,
            path,
            command_text,
            priority=command_priority,
        )


def candidate_source_path(
    accumulator: DiscoveryAccumulator, candidate: SettingCandidate
) -> Path:
    source = Path(candidate.source)
    return source if source.is_absolute() else accumulator.root / source


def add_from_candidate(
    accumulator: DiscoveryAccumulator,
    destination: tuple[str, ...],
    candidate: SettingCandidate,
    *,
    priority_adjustment: int = -1,
) -> None:
    accumulator.add(
        destination,
        candidate.value,
        candidate_source_path(accumulator, candidate),
        priority=max(1, candidate.priority + priority_adjustment),
    )


def first_setting_candidate(
    accumulator: DiscoveryAccumulator,
    locations: Sequence[tuple[str, ...]],
) -> SettingCandidate | None:
    for location in locations:
        candidate = accumulator.settings.get(location)
        if candidate is not None:
            return candidate
    return None


def complete_discovered_command_inputs(
    accumulator: DiscoveryAccumulator,
) -> None:
    """Reuse recovered inputs across commands that consume the same artifact."""

    for model in ("dnn", "kbnn", "neuro-tf"):
        for key in fit_shared_option_keys(model):
            fit_location = ("models", model, "commands", "fit", key)
            if fit_location in accumulator.settings:
                continue
            candidate = first_setting_candidate(
                accumulator,
                [
                    ("models", model, "commands", "optimize", key),
                    ("models", model, "commands", "train", key),
                ],
            )
            if candidate is not None:
                add_from_candidate(accumulator, fit_location, candidate)

        mdif_candidate = first_setting_candidate(
            accumulator,
            [
                ("models", model, "commands", "fit", "mdif"),
                ("models", "commands", "fit", "mdif"),
            ],
        )
        if mdif_candidate is not None:
            dependent_commands = ["inspect-mdif", "predict"]
            if model in {"dnn", "kbnn"}:
                dependent_commands.append("export-ads-ann")
            for command in dependent_commands:
                destination = ("models", model, "commands", command, "mdif")
                if destination not in accumulator.settings:
                    add_from_candidate(accumulator, destination, mdif_candidate)

        optimize_out = (
            "models",
            model,
            "commands",
            "optimize",
            "out_dir",
        )
        train_out_candidate = accumulator.settings.get(
            ("models", model, "commands", "train", "out_dir")
        )
        if optimize_out not in accumulator.settings and train_out_candidate is not None:
            train_path = Path(str(train_out_candidate.value))
            optimize_path = train_path.with_name(f"{train_path.name}_optimize")
            accumulator.add(
                optimize_out,
                str(optimize_path),
                candidate_source_path(accumulator, train_out_candidate),
                priority=max(1, train_out_candidate.priority - 10),
            )

        if model in {"dnn", "kbnn"}:
            rerank_location = (
                "models",
                model,
                "commands",
                "rerank-sweep",
                "sweep_dir",
            )
            optimize_candidate = accumulator.settings.get(optimize_out)
            if (
                rerank_location not in accumulator.settings
                and optimize_candidate is not None
            ):
                add_from_candidate(
                    accumulator,
                    rerank_location,
                    optimize_candidate,
                )

    audit_mdif = (
        "workflows",
        "audit",
        "commands",
        "audit",
        "mdif",
    )
    if audit_mdif not in accumulator.settings:
        candidate = accumulator.settings.get(
            ("models", "commands", "fit", "mdif")
        )
        if candidate is None:
            candidate = first_setting_candidate(
                accumulator,
                [
                    ("models", model, "commands", "fit", "mdif")
                    for model in ("dnn", "kbnn", "neuro-tf")
                ],
            )
        if candidate is not None:
            add_from_candidate(accumulator, audit_mdif, candidate)


def overlay_options_payload(
    target: dict[str, object], source: Mapping[str, object]
) -> None:
    for key, value in source.items():
        existing = target.get(str(key))
        if isinstance(existing, dict) and isinstance(value, Mapping):
            overlay_options_payload(existing, value)
        else:
            target[str(key)] = json_compatible(value)


def discover_options(
    directory: Path,
    *,
    excluded_paths: Iterable[Path] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    root = directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Discovery directory does not exist or is not a directory: {directory}")
    excluded = {path.expanduser().resolve() for path in excluded_paths}
    accumulator = DiscoveryAccumulator(root=root)
    scanned_files = 0
    unreadable_files: list[str] = []
    seen_commands: set[tuple[str, str]] = set()

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        if any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        scanned_files += 1
        if path.suffix.lower() == ".json":
            if path.name.lower() in SKIPPED_DOCUMENT_NAMES:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                unreadable_files.append(f"{accumulator.display_path(path)}: {exc}")
                continue
            inspect_json(accumulator, path, payload)
            continue
        if path.suffix.lower() == ".csv":
            recover_sweep_results(accumulator, path)
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_COMMAND_SUFFIXES:
            continue
        if path.name.lower() in SKIPPED_DOCUMENT_NAMES or path.name.lower().startswith("readme"):
            continue
        if suffix == ".md" and not (
            path.name.lower().endswith("_summary.md")
            or path.name.lower().endswith("_report.md")
        ):
            # Hand-authored documentation often contains illustrative commands,
            # not a record of work actually performed.  Generated summaries and
            # reports use these stable suffixes.
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            unreadable_files.append(f"{accumulator.display_path(path)}: {exc}")
            continue
        saved_commands = list(command_strings_from_text(text))
        if saved_commands:
            accumulator.add_artifact(path, "saved_command_report")
        for command_text in saved_commands:
            command_key = (str(path.resolve()), command_text)
            if command_key in seen_commands:
                continue
            seen_commands.add(command_key)
            recover_command(accumulator, path, command_text, priority=60)

    complete_discovered_command_inputs(accumulator)
    discovered_payload = accumulator.payload()
    payload = starter_options_payload()
    overlay_options_payload(payload, discovered_payload)
    source_report = {
        ".".join(location): {
            "selected_source": accumulator.settings[location].source,
            "all_sources": sorted(sources),
        }
        for location, sources in sorted(accumulator.setting_sources.items())
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "directory": str(root),
        "files_scanned": scanned_files,
        "settings_discovered": len(accumulator.settings),
        "starter_defaults_included": True,
        "recognized_artifacts": sorted(
            accumulator.artifacts,
            key=lambda item: (str(item["kind"]), str(item["path"])),
        ),
        "recovered_commands": accumulator.commands,
        "setting_sources": source_report,
        "conflicts": accumulator.conflicts,
        "warnings": [*accumulator.warnings, *unreadable_files],
    }
    return payload, report
