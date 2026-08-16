"""Reusable JSON defaults for the repository command-line interfaces.

The JSON file supplies argparse defaults rather than synthesizing shell text.
Explicit command-line options therefore retain precedence, and the selected
command's existing types, choices, and required-option rules remain the source
of truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Mapping, Sequence


OPTIONS_JSON_HELP = (
    "JSON file containing reusable option defaults. Supports common, commands, "
    "model/workflow-wide defaults, and narrower overrides; explicit command-line "
    "options take precedence."
)
UPDATE_OPTIONS_JSON_HELP = (
    "After a successful command, save explicitly supplied CLI options into "
    "the exact model/workflow command section of --options-json."
)
EXPLAIN_OPTIONS_HELP = (
    "Show the effective options, their CLI/JSON/default sources, and conditional "
    "input checks, then exit without running the command."
)

MODEL_NAMES = {"dnn", "kbnn", "neuro-tf"}
MODEL_ALIASES = {"neuro_tf": "neuro-tf", "neurotf": "neuro-tf"}
WORKFLOW_NAMES = {"points", "audit", "hb-report"}
COMMAND_ALIASES = {
    "sweep": "optimize",
    "export-ads": "export-ads-mdif",
}
COMMAND_NAMES = {
    "generate",
    "suggest-additional",
    "audit",
    "hb-report",
    "inspect-mdif",
    "train",
    "optimize",
    "rerank-sweep",
    "predict",
    "export-ads-mdif",
    "export-ads-ann",
    "export-ads-hb",
    "export-veriloga",
}
COMMAND_GROUPS = {"all", "fit", "export"}
NODE_HEADINGS = {"generic", "common", "commands"}
ROOT_HEADINGS = NODE_HEADINGS | {"schema_version", "models", "workflows"}


def starter_options_payload() -> dict[str, object]:
    """Return a ready-to-edit options document using every supported scope."""

    return {
        "schema_version": 1,
        "common": {},
        "commands": {},
        "models": {
            "common": {},
            "commands": {
                "fit": {
                    "frequency_weights": "default=1;1GHz=3;2GHz:4GHz=2",
                    "mdif": None,
                    "parameter_names": None,
                    "passivity_margin": 0.001,
                    "passivity_mode": "auto",
                    "progress_interval": 25,
                    "reciprocity_mode": "enforce",
                    "reciprocity_tolerance": 1e-6,
                    "seed": 1234,
                    "verification_mdif": None,
                },
                "optimize": {
                    "adaptive_category_balance": 0.5,
                    "require_passive": True,
                    "selection_metric": "weighted_evm_pct",
                    "trial_seed_mode": "fixed",
                    "trial_worst_plots": 0,
                },
            },
            "dnn": {
                "commands": {
                    "fit": {
                        "output_domain": "s",
                        "sparam_weights": "diag=1;offdiag=0.2",
                    },
                    "optimize": {"out_dir": None},
                    "train": {"out_dir": None},
                }
            },
            "kbnn": {
                "commands": {
                    "fit": {
                        "coarse_mdif": None,
                        "coarse_sparam_weights": "diag=1;offdiag=0.2",
                        "mode": "residual",
                        "sparam_weights": "diag=1;offdiag=0.2",
                    },
                    "optimize": {"out_dir": None},
                    "train": {"out_dir": None},
                }
            },
            "neuro-tf": {
                "commands": {
                    "fit": {"order": 10},
                    "optimize": {"out_dir": None},
                    "train": {"out_dir": None},
                }
            },
        },
        "workflows": {
            "audit": {
                "common": {
                    "expect_reciprocal": True,
                    "geometry_json": None,
                    "mdif": None,
                    "out_dir": None,
                    "parameter_names": None,
                    "passivity_tolerance": 1e-6,
                }
            },
            "points": {
                "commands": {
                    "generate": {
                        "count": None,
                        "decimal_places": 6,
                        "method": "maximin-lhs",
                        "out": None,
                        "parameter": None,
                        "verification_count": None,
                    },
                    "suggest-additional": {
                        "acquisition": "gp-ucb",
                        "allow_nonpassive": False,
                        "count": None,
                        "existing_points": None,
                        "exploration_weight": 2.0,
                        "fit_dir": None,
                        "metric": "auto",
                        "min_distance": 0.05,
                        "out": None,
                        "parameter_json": None,
                        "verification_metrics": None,
                    },
                }
            },
        },
    }


class OptionsJSONError(ValueError):
    """Raised when an options JSON file cannot be applied safely."""


def normalize_model_name(value: str) -> str:
    normalized = value.strip().lower()
    return MODEL_ALIASES.get(normalized, normalized)


def normalize_command_name(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    return COMMAND_ALIASES.get(normalized, normalized)


def normalize_workflow_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def add_options_json_argument(
    parser: argparse.ArgumentParser,
    *,
    recursive: bool = True,
) -> None:
    """Expose ``--options-json`` in root and selected-command help output."""

    if not any(
        "--options-json" in action.option_strings for action in parser._actions
    ):
        parser.add_argument(
            "--options-json",
            metavar="PATH",
            help=OPTIONS_JSON_HELP,
        )
    if not any(
        "--update-options-json" in action.option_strings
        for action in parser._actions
    ):
        parser.add_argument(
            "--update-options-json",
            action="store_true",
            help=UPDATE_OPTIONS_JSON_HELP,
        )
    if not any(
        "--explain-options" in action.option_strings
        for action in parser._actions
    ):
        parser.add_argument(
            "--explain-options",
            "--show-options",
            action="store_true",
            help=EXPLAIN_OPTIONS_HELP,
        )
    if not recursive:
        return
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        seen: set[int] = set()
        for child in action.choices.values():
            if id(child) in seen:
                continue
            seen.add(id(child))
            add_options_json_argument(child, recursive=True)


def extract_options_json_argument(
    argv: Sequence[str],
) -> tuple[str | None, list[str]]:
    """Remove one ``--options-json`` occurrence from an argument sequence."""

    clean: list[str] = []
    paths: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        token = values[index]
        if token == "--options-json":
            if index + 1 >= len(values):
                raise OptionsJSONError("--options-json requires a PATH")
            paths.append(values[index + 1])
            index += 2
            continue
        if token.startswith("--options-json="):
            paths.append(token.split("=", 1)[1])
            index += 1
            continue
        clean.append(token)
        index += 1
    if len(paths) > 1:
        raise OptionsJSONError("--options-json may be supplied only once")
    if paths and not paths[0].strip():
        raise OptionsJSONError("--options-json requires a non-empty PATH")
    return (paths[0] if paths else None), clean


def extract_update_options_json_argument(
    argv: Sequence[str],
) -> tuple[bool, list[str]]:
    """Remove the opt-in JSON-update flag from an argument sequence."""

    clean: list[str] = []
    requested = False
    for token in argv:
        if token == "--update-options-json":
            requested = True
            continue
        if token.startswith("--update-options-json="):
            raise OptionsJSONError("--update-options-json does not take a value")
        clean.append(token)
    return requested, clean


def extract_explain_options_argument(
    argv: Sequence[str],
) -> tuple[bool, list[str]]:
    """Remove the non-executing effective-option diagnostic flag."""

    clean: list[str] = []
    requested = False
    for token in argv:
        if token in {"--explain-options", "--show-options"}:
            requested = True
            continue
        if token.startswith("--explain-options=") or token.startswith(
            "--show-options="
        ):
            raise OptionsJSONError(f"{token.split('=', 1)[0]} does not take a value")
        clean.append(token)
    return requested, clean


def _mapping(value: object, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise OptionsJSONError(f"{context} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _merge_options(target: dict[str, object], source: Mapping[str, object]) -> None:
    for key, value in source.items():
        target[str(key)] = value


def _node_defaults(node: Mapping[str, object], context: str) -> dict[str, object]:
    defaults = {
        str(key): value
        for key, value in node.items()
        if key not in NODE_HEADINGS
    }
    for heading in ("generic", "common"):
        if heading in node:
            _merge_options(
                defaults,
                _mapping(node[heading], f"{context}.{heading}"),
            )
    return defaults


def _command_group(command: str) -> str | None:
    if command in {"train", "optimize"}:
        return "fit"
    if command.startswith("export-"):
        return "export"
    return None


def _validate_commands(commands: Mapping[str, object], context: str) -> None:
    seen: set[str] = set()
    for raw_name, raw_options in commands.items():
        name = normalize_command_name(raw_name)
        if name not in COMMAND_NAMES | COMMAND_GROUPS:
            available = ", ".join(sorted(COMMAND_NAMES | COMMAND_GROUPS))
            raise OptionsJSONError(
                f"Unknown command heading {raw_name!r} in {context}; "
                f"available headings: {available}"
            )
        if name in seen:
            raise OptionsJSONError(
                f"Duplicate normalized command heading {name!r} in {context}"
            )
        seen.add(name)
        _mapping(raw_options, f"{context}.{raw_name}")


def _command_defaults(
    commands: Mapping[str, object],
    command: str,
    context: str,
) -> dict[str, object]:
    normalized: dict[str, dict[str, object]] = {}
    for raw_name, raw_options in commands.items():
        normalized[normalize_command_name(raw_name)] = _mapping(
            raw_options,
            f"{context}.{raw_name}",
        )
    defaults: dict[str, object] = {}
    for key in ("all", _command_group(command), command):
        if key and key in normalized:
            _merge_options(defaults, normalized[key])
    return defaults


def _record_locations(
    target: dict[str, str],
    source: Mapping[str, object],
    context: str,
) -> None:
    for key in source:
        target[str(key)] = f"{context}.{key}" if context else str(key)


def _command_locations(
    commands: Mapping[str, object],
    command: str,
    context: str,
) -> dict[str, str]:
    normalized: dict[str, tuple[str, dict[str, object]]] = {}
    for raw_name, raw_options in commands.items():
        normalized[normalize_command_name(raw_name)] = (
            str(raw_name),
            _mapping(raw_options, f"options JSON {context}.{raw_name}"),
        )
    locations: dict[str, str] = {}
    for key in ("all", _command_group(command), command):
        if key and key in normalized:
            raw_name, options = normalized[key]
            _record_locations(locations, options, f"{context}.{raw_name}")
    return locations


def _node_locations(
    node: Mapping[str, object],
    context: str,
) -> dict[str, str]:
    locations: dict[str, str] = {}
    _record_locations(
        locations,
        {
            str(key): value
            for key, value in node.items()
            if key not in NODE_HEADINGS
        },
        context,
    )
    for heading in ("generic", "common"):
        if heading in node:
            _record_locations(
                locations,
                _mapping(node[heading], f"options JSON {context}.{heading}"),
                f"{context}.{heading}",
            )
    return locations


def load_options_json_resolution(
    path: str | Path,
    *,
    model: str | None = None,
    workflow: str | None = None,
    command: str,
) -> tuple[dict[str, object], dict[str, str]]:
    """Load applicable defaults and the exact JSON source of every raw key."""

    source = Path(path).expanduser()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OptionsJSONError(f"Could not read options JSON {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OptionsJSONError(
            f"Could not parse options JSON {source}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    root = _mapping(payload, "options JSON root")
    schema_version = root.get("schema_version", 1)
    if schema_version != 1:
        raise OptionsJSONError(
            f"Unsupported options JSON schema_version {schema_version!r}; expected 1"
        )

    commands = _mapping(root.get("commands"), "options JSON commands")
    _validate_commands(commands, "options JSON commands")
    models = _mapping(root.get("models"), "options JSON models")
    workflows = _mapping(root.get("workflows"), "options JSON workflows")
    selected_model_name = normalize_model_name(model) if model is not None else None
    allowed_models = set(MODEL_NAMES)
    if selected_model_name:
        allowed_models.add(selected_model_name)
    for heading in ("generic", "common"):
        if heading in models:
            _mapping(models[heading], f"options JSON models.{heading}")
    model_container_commands = _mapping(
        models.get("commands"),
        "options JSON models.commands",
    )
    _validate_commands(model_container_commands, "options JSON models.commands")
    for raw_name, node in models.items():
        if raw_name in NODE_HEADINGS:
            continue
        normalized = normalize_model_name(raw_name)
        if normalized not in allowed_models:
            raise OptionsJSONError(
                f"Unknown model heading {raw_name!r}; available models: "
                + ", ".join(sorted(allowed_models))
            )
        model_node = _mapping(node, f"options JSON models.{raw_name}")
        model_commands = _mapping(
            model_node.get("commands"),
            f"options JSON models.{raw_name}.commands",
        )
        _validate_commands(
            model_commands,
            f"options JSON models.{raw_name}.commands",
        )
    for raw_name, node in workflows.items():
        normalized = normalize_workflow_name(raw_name)
        if normalized not in WORKFLOW_NAMES:
            raise OptionsJSONError(
                f"Unknown workflow heading {raw_name!r}; available workflows: "
                + ", ".join(sorted(WORKFLOW_NAMES))
            )
        workflow_node = _mapping(node, f"options JSON workflows.{raw_name}")
        workflow_commands = _mapping(
            workflow_node.get("commands"),
            f"options JSON workflows.{raw_name}.commands",
        )
        _validate_commands(
            workflow_commands,
            f"options JSON workflows.{raw_name}.commands",
        )

    canonical_command = normalize_command_name(command)
    if canonical_command not in COMMAND_NAMES:
        raise OptionsJSONError(f"Unknown selected command {command!r}")

    defaults = {
        str(key): value
        for key, value in root.items()
        if key not in ROOT_HEADINGS
    }
    for heading in ("generic", "common"):
        if heading in root:
            _merge_options(
                defaults,
                _mapping(root[heading], f"options JSON {heading}"),
            )
    _merge_options(
        defaults,
        _command_defaults(commands, canonical_command, "options JSON commands"),
    )

    selected_scope: dict[str, object] | None = None
    scope_context = ""
    if model is not None:
        for heading in ("generic", "common"):
            if heading in models:
                _merge_options(
                    defaults,
                    _mapping(
                        models[heading],
                        f"options JSON models.{heading}",
                    ),
                )
        _merge_options(
            defaults,
            _command_defaults(
                model_container_commands,
                canonical_command,
                "options JSON models.commands",
            ),
        )
        canonical_model = selected_model_name
        for raw_name, node in models.items():
            if raw_name in NODE_HEADINGS:
                continue
            if normalize_model_name(raw_name) == canonical_model:
                selected_scope = _mapping(
                    node,
                    f"options JSON models.{raw_name}",
                )
                scope_context = f"options JSON models.{raw_name}"
                break
    elif workflow is not None:
        canonical_workflow = normalize_workflow_name(workflow)
        for raw_name, node in workflows.items():
            if normalize_workflow_name(raw_name) == canonical_workflow:
                selected_scope = _mapping(
                    node,
                    f"options JSON workflows.{raw_name}",
                )
                scope_context = f"options JSON workflows.{raw_name}"
                break

    if selected_scope is not None:
        _merge_options(defaults, _node_defaults(selected_scope, scope_context))
        scope_commands = _mapping(
            selected_scope.get("commands"),
            f"{scope_context}.commands",
        )
        _merge_options(
            defaults,
            _command_defaults(
                scope_commands,
                canonical_command,
                f"{scope_context}.commands",
            ),
        )
    locations: dict[str, str] = {}
    _record_locations(
        locations,
        {
            str(key): value
            for key, value in root.items()
            if key not in ROOT_HEADINGS
        },
        "",
    )
    for heading in ("generic", "common"):
        if heading in root:
            _record_locations(
                locations,
                _mapping(root[heading], f"options JSON {heading}"),
                heading,
            )
    locations.update(
        _command_locations(commands, canonical_command, "commands")
    )
    if model is not None:
        for heading in ("generic", "common"):
            if heading in models:
                _record_locations(
                    locations,
                    _mapping(models[heading], f"options JSON models.{heading}"),
                    f"models.{heading}",
                )
        locations.update(
            _command_locations(
                model_container_commands,
                canonical_command,
                "models.commands",
            )
        )
    if selected_scope is not None:
        locations.update(
            _node_locations(
                selected_scope,
                scope_context.removeprefix("options JSON "),
            )
        )
        scope_commands = _mapping(
            selected_scope.get("commands"),
            f"{scope_context}.commands",
        )
        locations.update(
            _command_locations(
                scope_commands,
                canonical_command,
                f"{scope_context.removeprefix('options JSON ')}.commands",
            )
        )
    return defaults, locations


def load_options_json_defaults(
    path: str | Path,
    *,
    model: str | None = None,
    workflow: str | None = None,
    command: str,
) -> dict[str, object]:
    """Load and merge the defaults applicable to one selected command."""

    defaults, _ = load_options_json_resolution(
        path,
        model=model,
        workflow=workflow,
        command=command,
    )
    return defaults


def _selected_parser(
    parser: argparse.ArgumentParser,
    command: str,
) -> argparse.ArgumentParser:
    canonical = normalize_command_name(command)
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, child in action.choices.items():
            if normalize_command_name(name) == canonical:
                return child
        raise OptionsJSONError(
            f"Command {command!r} is not registered by this parser"
        )
    return parser


def _option_name(key: str) -> str:
    normalized = key.strip().lstrip("-").replace("_", "-")
    if not normalized:
        raise OptionsJSONError("An options JSON option name is empty")
    return f"--{normalized}"


def _coerce_scalar(action: argparse.Action, value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        raise OptionsJSONError(
            f"{action.option_strings[-1]} expects one scalar JSON value"
        )
    try:
        converted = action.type(str(value)) if action.type is not None else str(value)
    except (TypeError, ValueError) as exc:
        raise OptionsJSONError(
            f"Invalid value {value!r} for {action.option_strings[-1]}: {exc}"
        ) from exc
    if action.choices is not None and converted not in action.choices:
        choices = ", ".join(map(str, action.choices))
        raise OptionsJSONError(
            f"Invalid value {converted!r} for {action.option_strings[-1]}; "
            f"choose from {choices}"
        )
    return converted


_SKIP = object()


def _coerce_default(action: argparse.Action, value: object) -> object:
    if value is None:
        return _SKIP
    if isinstance(action, argparse._StoreTrueAction):
        if not isinstance(value, bool):
            raise OptionsJSONError(
                f"{action.option_strings[-1]} expects true or false in JSON"
            )
        return value
    if isinstance(action, argparse._StoreFalseAction):
        if not isinstance(value, bool):
            raise OptionsJSONError(
                f"{action.option_strings[-1]} expects true or false in JSON"
            )
        if value:
            return action.const
        return action.default if action.default is not argparse.SUPPRESS else _SKIP
    if isinstance(action, argparse._StoreConstAction):
        if not isinstance(value, bool):
            raise OptionsJSONError(
                f"{action.option_strings[-1]} expects true or false in JSON"
            )
        if value:
            return action.const
        return action.default if action.default is not argparse.SUPPRESS else _SKIP
    if isinstance(action, argparse._AppendAction):
        values = value if isinstance(value, list) else [value]
        return [_coerce_scalar(action, item) for item in values]
    if action.nargs in ("+", "*") or isinstance(action.nargs, int):
        if not isinstance(value, list):
            raise OptionsJSONError(
                f"{action.option_strings[-1]} expects a JSON array"
            )
        return [_coerce_scalar(action, item) for item in value]
    return _coerce_scalar(action, value)


def apply_options_json_defaults(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    defaults: Mapping[str, object],
    *,
    command: str,
    source_path: str | Path,
    default_sources: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate and apply JSON values to the selected argparse command."""

    selected = _selected_parser(parser, command)
    actions = [
        action
        for owner in (parser, selected)
        for action in owner._actions
        if action.option_strings
    ]
    option_actions: dict[str, argparse.Action] = {}
    for action in actions:
        for option in action.option_strings:
            if option.startswith("--"):
                option_actions[option] = action

    explicit_destinations: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            explicit_destinations.add(action.dest)

    applied: dict[str, object] = {"options_json": str(source_path)}
    applied_sources: dict[str, str] = {}
    supplied_actions: list[argparse.Action] = []
    for raw_key, raw_value in defaults.items():
        option = _option_name(raw_key)
        if option == "--options-json":
            raise OptionsJSONError(
                "The options JSON cannot set its own --options-json path"
            )
        if option == "--update-options-json":
            raise OptionsJSONError(
                "The options JSON cannot enable --update-options-json; "
                "request updates explicitly on the command line"
            )
        if option in {"--explain-options", "--show-options"}:
            raise OptionsJSONError(
                "The options JSON cannot enable --explain-options; "
                "request diagnostics explicitly on the command line"
            )
        action = option_actions.get(option)
        if action is None:
            available = sorted(option_actions)
            close = get_close_matches(option, available, n=3)
            suggestion = f" Did you mean {', '.join(close)}?" if close else ""
            raise OptionsJSONError(
                f"Option {option!r} from {source_path} is not valid for "
                f"{command}.{suggestion}"
            )
        if action.dest in explicit_destinations:
            continue
        converted = _coerce_default(action, raw_value)
        if converted is _SKIP:
            if default_sources is not None:
                applied_sources[action.dest] = (
                    "null ignored at "
                    + default_sources.get(raw_key, str(source_path))
                )
            continue
        applied[action.dest] = converted
        if default_sources is not None:
            applied_sources[action.dest] = default_sources.get(
                raw_key,
                str(source_path),
            )
        supplied_actions.append(action)

    selected.set_defaults(**applied)
    for action in supplied_actions:
        if action.required:
            action.required = False
    return applied_sources


def _json_compatible(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise OptionsJSONError(
        f"Cannot save command option value of type {type(value).__name__} to JSON"
    )


def _explicit_option_updates(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    args: argparse.Namespace,
    *,
    command: str,
) -> tuple[dict[str, object], dict[str, argparse.Action]]:
    selected = _selected_parser(parser, command)
    actions = [
        action
        for owner in (parser, selected)
        for action in owner._actions
        if action.option_strings
    ]
    option_actions = {
        option: action
        for action in actions
        for option in action.option_strings
        if option.startswith("-")
    }
    selected_options: dict[str, tuple[str, argparse.Action]] = {}
    for token in argv:
        if not token.startswith("-"):
            continue
        raw_option = token.split("=", 1)[0]
        action = option_actions.get(raw_option)
        if action is None or raw_option in {
            "--options-json",
            "--update-options-json",
        }:
            continue
        saved_option = raw_option
        if not saved_option.startswith("--"):
            saved_option = next(
                (
                    option
                    for option in action.option_strings
                    if option.startswith("--")
                ),
                saved_option,
            )
        selected_options[action.dest] = (
            saved_option.lstrip("-").replace("-", "_"),
            action,
        )

    updates: dict[str, object] = {}
    updated_actions: dict[str, argparse.Action] = {}
    for destination, (key, action) in selected_options.items():
        if isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse._StoreConstAction,
            ),
        ):
            value: object = True
        else:
            value = getattr(args, destination)
        updates[key] = _json_compatible(value)
        updated_actions[key] = action
    return updates, updated_actions


def _exact_command_node(
    payload: dict[str, object],
    *,
    model: str | None,
    workflow: str | None,
    command: str,
) -> tuple[dict[str, object], str]:
    if model is not None:
        scope_name = normalize_model_name(model)
        container_name = "models"
    elif workflow is not None:
        scope_name = normalize_workflow_name(workflow)
        container_name = "workflows"
    else:
        raise OptionsJSONError(
            "Cannot update options JSON without a selected model or workflow"
        )

    container = payload.setdefault(container_name, {})
    if not isinstance(container, dict):
        raise OptionsJSONError(f"options JSON {container_name} must be an object")

    scope_key = scope_name
    for existing_key in container:
        normalized = (
            normalize_model_name(existing_key)
            if model is not None
            else normalize_workflow_name(existing_key)
        )
        if normalized == scope_name:
            scope_key = existing_key
            break
    scope = container.setdefault(scope_key, {})
    if not isinstance(scope, dict):
        raise OptionsJSONError(
            f"options JSON {container_name}.{scope_key} must be an object"
        )
    commands = scope.setdefault("commands", {})
    if not isinstance(commands, dict):
        raise OptionsJSONError(
            f"options JSON {container_name}.{scope_key}.commands must be an object"
        )

    canonical_command = normalize_command_name(command)
    command_key = canonical_command
    for existing_key in commands:
        if normalize_command_name(existing_key) == canonical_command:
            command_key = existing_key
            break
    command_node = commands.setdefault(command_key, {})
    if not isinstance(command_node, dict):
        raise OptionsJSONError(
            f"options JSON {container_name}.{scope_key}.commands.{command_key} "
            "must be an object"
        )
    location = f"{container_name}.{scope_key}.commands.{command_key}"
    return command_node, location


def _write_options_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    resolved = path.resolve(strict=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, resolved.stat().st_mode & 0o777)
        temporary.replace(resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize_options_json_update(args: argparse.Namespace, status: int) -> int:
    """Persist an opted-in invocation after, and only after, successful work."""

    requested = bool(getattr(args, "_options_json_update_requested", False))
    # Status 1 is a completed command with an unfavorable domain result for
    # workflows such as dataset audit.  Its invocation is still useful project
    # state.  Status 2+ represents a CLI/runtime failure and is not persisted.
    if status >= 2 or not requested:
        return int(status)
    try:
        parser = getattr(args, "_options_json_update_parser")
        argv = getattr(args, "_options_json_update_argv")
        model = getattr(args, "_options_json_update_model")
        workflow = getattr(args, "_options_json_update_workflow")
        command = getattr(args, "_options_json_update_command")
        source = Path(args.options_json).expanduser()
        updates, updated_actions = _explicit_option_updates(
            parser,
            argv,
            args,
            command=command,
        )
        payload = json.loads(source.read_text(encoding="utf-8"))
        root = _mapping(payload, "options JSON root")
        command_node, location = _exact_command_node(
            root,
            model=model,
            workflow=workflow,
            command=command,
        )

        for key, action in updated_actions.items():
            aliases = {
                option
                for option in action.option_strings
                if option.startswith("--")
            }
            for existing_key in list(command_node):
                if _option_name(existing_key) in aliases:
                    del command_node[existing_key]
            command_node[key] = updates[key]

        if not updates:
            print(
                f"options JSON unchanged: no explicit command options to save "
                f"in {location}"
            )
            return int(status)
        _write_options_json_atomic(source, root)
        saved = ", ".join(f"--{key.replace('_', '-')}" for key in updates)
        print(f"updated {source}: {location} ({saved})")
        return int(status)
    except (OSError, TypeError, ValueError, OptionsJSONError) as exc:
        print(f"error: could not update options JSON: {exc}", file=sys.stderr)
        return 2


def _parser_actions(
    parser: argparse.ArgumentParser,
    command: str,
) -> list[argparse.Action]:
    selected = _selected_parser(parser, command)
    actions: list[argparse.Action] = []
    seen: set[int] = set()
    for owner in (parser, selected):
        for action in owner._actions:
            if id(action) in seen or isinstance(action, argparse._SubParsersAction):
                continue
            seen.add(id(action))
            actions.append(action)
    return actions


def _explicit_option_sources(
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    *,
    command: str,
) -> dict[str, str]:
    option_actions = {
        option: action
        for action in _parser_actions(parser, command)
        for option in action.option_strings
    }
    sources: dict[str, str] = {}
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        action = option_actions.get(option)
        if action is not None:
            sources[action.dest] = option
    return sources


def _display_value(value: object) -> str:
    if value is argparse.SUPPRESS:
        return "<unset>"
    try:
        rendered = json.dumps(_json_compatible(value), ensure_ascii=False)
    except OptionsJSONError:
        rendered = repr(value)
    return rendered


def _point_parameter_metadata_candidates(path: Path) -> list[Path]:
    candidates = [path.with_suffix(".json")]
    for suffix in ("_train", "_verification"):
        if path.stem.endswith(suffix):
            combined = path.with_name(
                f"{path.stem[:-len(suffix)]}{path.suffix or '.csv'}"
            )
            candidates.append(combined.with_suffix(".json"))
    return list(dict.fromkeys(candidates))


def _resolved_point_metrics_path(args: argparse.Namespace) -> Path | None:
    if getattr(args, "verification_metrics", None):
        return Path(args.verification_metrics)
    if not getattr(args, "fit_dir", None):
        return None
    fit_dir = Path(args.fit_dir)
    candidates = (
        fit_dir / "verification_metrics.csv",
        fit_dir / "best_model" / "verification_metrics.csv",
        fit_dir / "point_generation_fallback" / "verification_metrics.csv",
        fit_dir.parent / "point_generation_fallback" / "verification_metrics.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _print_point_suggestion_checks(args: argparse.Namespace) -> None:
    print("\nAdditional-point input checks:")
    parameters = list(getattr(args, "parameter", None) or [])
    parameter_json = getattr(args, "parameter_json", None)
    existing_points = list(getattr(args, "existing_points", None) or [])
    if parameters:
        print(f"  parameter domain: OK, {len(parameters)} --parameter value(s)")
    elif parameter_json:
        path = Path(parameter_json)
        status = "found" if path.is_file() else "NOT FOUND"
        print(f"  parameter domain: --parameter-json {path} ({status})")
    elif existing_points:
        found: list[Path] = []
        checked: list[Path] = []
        for raw_path in existing_points:
            for candidate in _point_parameter_metadata_candidates(Path(raw_path)):
                checked.append(candidate)
                if candidate.is_file():
                    found.append(candidate)
                    break
        if found:
            print(
                "  parameter domain: OK, inferred companion JSON: "
                + ", ".join(map(str, found))
            )
        else:
            print(
                "  parameter domain: MISSING; no companion JSON found (checked "
                + ", ".join(map(str, checked))
                + ")"
            )
    else:
        print(
            "  parameter domain: MISSING; set --parameter, --parameter-json, or "
            "--existing-points with its companion JSON"
        )

    metrics_path = _resolved_point_metrics_path(args)
    metric = str(getattr(args, "metric", "evm_pct"))
    if metrics_path is None:
        print(
            "  verification metrics: MISSING; set --fit-dir or "
            "--verification-metrics"
        )
    elif not metrics_path.is_file():
        print(f"  verification metrics: NOT FOUND at {metrics_path}")
    else:
        try:
            with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                rows = list(reader)
        except OSError as exc:
            print(f"  verification metrics: could not read {metrics_path}: {exc}")
        else:
            def usable_count(name: str) -> int:
                count = 0
                for row in rows:
                    try:
                        value = float(str(row.get(name) or "").strip())
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        count += 1
                return count

            if metric == "auto":
                candidates = (
                    "evm_pct",
                    "rmse_abs",
                    "max_abs",
                    "rmse_db",
                    "max_abs_db",
                )
                selected = next(
                    (name for name in candidates if usable_count(name)),
                    None,
                )
                status = f"auto -> {selected}" if selected else "auto found no known column"
            else:
                if metric not in columns:
                    status = "COLUMN NOT FOUND"
                else:
                    usable = usable_count(metric)
                    status = (
                        f"{usable} usable numeric row(s)"
                        if usable
                        else "COLUMN PRESENT BUT NO USABLE NUMERIC VALUES"
                    )
            print(f"  verification metrics: {metrics_path} ({status}: {metric})")
            if (
                "point_generation_fallback" in metrics_path.parts
                and not getattr(args, "allow_nonpassive", False)
            ):
                print(
                    "  passivity fallback: BLOCKED; add --allow-nonpassive to use "
                    "these errors for point selection"
                )
            if metric.startswith("weighted_"):
                print(
                    "  metric guidance: weighted_* values are fit-summary metrics, "
                    "not geometry-row columns; use --metric evm_pct or --metric auto. "
                    "Per-row normalized_sparam_weight still applies S-parameter weights."
                )


def print_effective_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: Sequence[str],
    *,
    model: str | None,
    workflow: str | None,
    command: str,
    json_sources: Mapping[str, str],
    required_actions: Mapping[int, bool],
) -> None:
    """Print the values a backend would receive without executing it."""

    route = (
        f"model={normalize_model_name(model)}"
        if model
        else f"workflow={workflow}"
    )
    print("Effective command options (command was not executed)")
    print(f"  route: {route}, command={normalize_command_name(command)}")
    options_path = getattr(args, "options_json", None)
    print(f"  options JSON: {options_path or '<none>'}")
    print("  precedence: explicit CLI > narrow JSON > broad JSON > parser default")
    explicit = _explicit_option_sources(parser, argv, command=command)
    missing: list[str] = []
    rows: list[tuple[str, str, str]] = []
    displayed_destinations: set[str] = set()
    for action in _parser_actions(parser, command):
        if not action.option_strings:
            value = getattr(args, action.dest, argparse.SUPPRESS)
            label = str(action.metavar or action.dest).upper()
            is_missing = value is argparse.SUPPRESS or value is None or (
                isinstance(value, (list, tuple)) and not value
            )
            source = "CLI positional argument"
            if required_actions.get(id(action), False) and is_missing:
                source = "MISSING REQUIRED ARGUMENT"
                missing.append(label)
            rows.append((label, _display_value(value), source))
            continue
        long_options = [
            item for item in action.option_strings if item.startswith("--")
        ]
        option = long_options[0] if long_options else action.option_strings[0]
        if option in {
            "--help",
            "--options-json",
            "--update-options-json",
            "--explain-options",
        }:
            continue
        if action.dest in displayed_destinations:
            continue
        displayed_destinations.add(action.dest)
        value = getattr(args, action.dest, argparse.SUPPRESS)
        if action.dest in explicit:
            source = f"CLI ({explicit[action.dest]})"
        elif action.dest in json_sources:
            json_source = json_sources[action.dest]
            if json_source.startswith("null ignored at "):
                source = f"parser default; JSON {json_source}"
            else:
                source = f"JSON: {json_source}"
        else:
            source = "parser default"
        is_missing = value is argparse.SUPPRESS or value is None or (
            isinstance(value, (list, tuple)) and not value
        )
        if required_actions.get(id(action), False) and is_missing:
            ignored_source = json_sources.get(action.dest, "")
            source = "MISSING REQUIRED OPTION"
            if ignored_source.startswith("null ignored at "):
                source += f"; JSON {ignored_source}"
            missing.append(option)
        rows.append((option, _display_value(value), source))
    width = max((len(row[0]) for row in rows), default=6)
    for option, value, source in rows:
        print(f"  {option:<{width}} = {value}  [{source}]")
    if missing:
        print("\nRequired options still missing: " + ", ".join(missing))
    else:
        print("\nRequired argparse options: complete")
    if (
        workflow == "points"
        and normalize_command_name(command) == "suggest-additional"
    ):
        _print_point_suggestion_checks(args)


def _relax_parser_requirements_for_explanation(
    parser: argparse.ArgumentParser,
    command: str,
) -> dict[int, bool]:
    required_actions: dict[int, bool] = {}
    selected = _selected_parser(parser, command)
    owners = (parser, selected) if selected is not parser else (parser,)
    for action in _parser_actions(parser, command):
        required_actions[id(action)] = bool(getattr(action, "required", False))
        if action.option_strings:
            action.required = False
        elif action.nargs is None:
            action.nargs = "?"
        elif action.nargs == "+":
            action.nargs = "*"
    for owner in owners:
        for group in owner._mutually_exclusive_groups:
            group.required = False
    return required_actions


def parse_args_with_options_json(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    model: str | None = None,
    workflow: str | None = None,
    command: str | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments after applying any selected JSON defaults."""

    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        options_path, without_options_path = extract_options_json_argument(raw_args)
        update_requested, without_update = extract_update_options_json_argument(
            without_options_path
        )
        explain_requested, clean_args = extract_explain_options_argument(
            without_update
        )
        if update_requested and options_path is None:
            raise OptionsJSONError(
                "--update-options-json requires --options-json PATH"
            )
        if update_requested and explain_requested:
            raise OptionsJSONError(
                "--update-options-json cannot be combined with --explain-options "
                "because the diagnostic does not run the command"
            )
        selected_command = command
        if selected_command is None:
            registered: set[str] = set()
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    registered.update(action.choices)
            selected_command = next(
                (token for token in clean_args if token in registered),
                None,
            )
        json_sources: dict[str, str] = {}
        if options_path is not None:
            if selected_command is None:
                raise OptionsJSONError(
                    "Could not determine the command used with --options-json"
                )
            defaults, default_sources = load_options_json_resolution(
                options_path,
                model=model,
                workflow=workflow,
                command=selected_command,
            )
            json_sources = apply_options_json_defaults(
                parser,
                clean_args,
                defaults,
                command=selected_command,
                source_path=options_path,
                default_sources=default_sources,
            )
        if explain_requested:
            if selected_command is None:
                raise OptionsJSONError(
                    "Could not determine the command used with --explain-options"
                )
            required_actions = _relax_parser_requirements_for_explanation(
                parser,
                selected_command,
            )
        args = parser.parse_args(clean_args)
        if explain_requested:
            print_effective_options(
                parser,
                args,
                clean_args,
                model=model,
                workflow=workflow,
                command=selected_command,
                json_sources=json_sources,
                required_actions=required_actions,
            )
            parser.exit(0)
        args.update_options_json = update_requested
        args._options_json_update_requested = update_requested
        args._options_json_update_parser = parser
        args._options_json_update_argv = clean_args
        args._options_json_update_model = model
        args._options_json_update_workflow = workflow
        args._options_json_update_command = normalize_command_name(
            selected_command or command or ""
        )
        return args
    except OptionsJSONError as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.parser.error should not return")
