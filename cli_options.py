"""Reusable JSON defaults for the repository command-line interfaces.

The JSON file supplies argparse defaults rather than synthesizing shell text.
Explicit command-line options therefore retain precedence, and the selected
command's existing types, choices, and required-option rules remain the source
of truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from difflib import get_close_matches
from pathlib import Path
from typing import Mapping, Sequence


OPTIONS_JSON_HELP = (
    "JSON file containing reusable option defaults. Supports common, commands, "
    "model/workflow-wide defaults, and narrower overrides; explicit command-line "
    "options take precedence."
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
                        "count": None,
                        "existing_points": None,
                        "exploration_weight": 2.0,
                        "fit_dir": None,
                        "min_distance": 0.05,
                        "out": None,
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


def load_options_json_defaults(
    path: str | Path,
    *,
    model: str | None = None,
    workflow: str | None = None,
    command: str,
) -> dict[str, object]:
    """Load and merge the defaults applicable to one selected command."""

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
) -> None:
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
    supplied_actions: list[argparse.Action] = []
    for raw_key, raw_value in defaults.items():
        option = _option_name(raw_key)
        if option == "--options-json":
            raise OptionsJSONError(
                "The options JSON cannot set its own --options-json path"
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
            continue
        applied[action.dest] = converted
        supplied_actions.append(action)

    selected.set_defaults(**applied)
    for action in supplied_actions:
        if action.required:
            action.required = False


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
        options_path, clean_args = extract_options_json_argument(raw_args)
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
        if options_path is not None:
            if selected_command is None:
                raise OptionsJSONError(
                    "Could not determine the command used with --options-json"
                )
            defaults = load_options_json_defaults(
                options_path,
                model=model,
                workflow=workflow,
                command=selected_command,
            )
            apply_options_json_defaults(
                parser,
                clean_args,
                defaults,
                command=selected_command,
                source_path=options_path,
            )
        return parser.parse_args(clean_args)
    except OptionsJSONError as exc:
        parser.error(str(exc))
    raise AssertionError("argparse.parser.error should not return")
