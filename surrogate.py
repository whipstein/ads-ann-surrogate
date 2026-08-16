#!/usr/bin/env python3
"""Primary command-line entry point for the complete surrogate workflow."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cli_options import (
    OptionsJSONError,
    add_options_json_argument,
    extract_options_json_argument,
)


MODEL_SCRIPTS = {
    "dnn": "dnn.py",
    "kbnn": "kbnn.py",
    "neuro-tf": "neuro_tf.py",
}

MODEL_TYPE_ALIASES = {
    "neuro_tf": "neuro-tf",
    "neurotf": "neuro-tf",
}

WORKFLOW_SCRIPTS = {
    "points": "generate_points.py",
    "audit": "audit_dataset.py",
    "hb-report": "de_generated_scripts/parse_ads_hb_solver_log.py",
}

WORKFLOW_DESCRIPTIONS = {
    "points": "generate initial points or suggest adaptive additions",
    "audit": "audit training and verification MDIF data",
    "hb-report": "compare ADS HB Newton/Krylov logs and runtimes",
}


def normalize_model_type(value: str) -> str:
    """Return the canonical command-line name for a model family."""

    normalized = value.strip().lower()
    return MODEL_TYPE_ALIASES.get(normalized, normalized)


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run point generation, data auditing, model fitting/export, and ADS "
            "HB log reporting through one stable entry point. Arguments after "
            "the selected workflow are forwarded unchanged to its implementation."
        ),
        add_help=add_help,
        usage=(
            "%(prog)s {points,audit,hb-report} [COMMAND] [OPTIONS]\n"
            "       %(prog)s --model {dnn,kbnn,neuro-tf} COMMAND [OPTIONS]"
        ),
        epilog=(
            "Examples:\n"
            "  python3 surrogate.py points generate --parameter W=1mm:2mm "
            "--count 24 --out geometries.csv\n"
            "  python3 surrogate.py audit --mdif data.mdif\n"
            "  python3 surrogate.py --model dnn train "
            "--mdif data.mdif --out-dir outputs/dnn\n"
            "  python3 surrogate.py --model neuro-tf export-veriloga --help\n"
            "  python3 surrogate.py hb-report baseline.log trial.log"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="model_type",
        required=False,
        type=normalize_model_type,
        choices=tuple(MODEL_SCRIPTS),
        metavar="{dnn,kbnn,neuro-tf}",
        help="Model family whose backend should receive the command",
    )
    add_options_json_argument(parser, recursive=False)
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=tuple(WORKFLOW_SCRIPTS),
        metavar="{points,audit,hb-report}",
        help=(
            "Non-model workflow command: "
            + "; ".join(
                f"{name} = {WORKFLOW_DESCRIPTIONS[name]}"
                for name in WORKFLOW_SCRIPTS
            )
        ),
    )
    return parser


def build_model_arg_parser() -> argparse.ArgumentParser:
    """Build the parser used only for model-backend selection."""

    # Backend options are intentionally opaque to this small dispatcher.  In
    # particular, KBNN's ``--mode`` must not be treated as an abbreviation of
    # this parser's ``--model`` option.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--model",
        dest="model_type",
        required=True,
        type=normalize_model_type,
        choices=tuple(MODEL_SCRIPTS),
        metavar="{dnn,kbnn,neuro-tf}",
    )
    return parser


def parse_dispatch_args(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse only dispatcher arguments and preserve all backend arguments."""

    parser = build_model_arg_parser()
    args, backend_args = parser.parse_known_args(list(argv))
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    return args, backend_args


def dispatch_script(
    script_name: str,
    backend_args: Sequence[str],
    *,
    cli_prog: str,
) -> int:
    """Execute one internal implementation script and return its exit status."""

    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.is_file():
        print(
            f"error: command implementation was not found: {script_path}",
            file=sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    environment["ADS_SURROGATE_CLI_PROG"] = cli_prog
    completed = subprocess.run(
        [sys.executable or "python3", str(script_path), *map(str, backend_args)],
        check=False,
        env=environment,
    )
    return int(completed.returncode)


def dispatch(model_type: str, backend_args: Sequence[str]) -> int:
    """Execute the selected model backend with this interpreter."""

    canonical_model_type = normalize_model_type(model_type)
    return dispatch_script(
        MODEL_SCRIPTS[canonical_model_type],
        backend_args,
        cli_prog=f"surrogate.py --model {canonical_model_type}",
    )


def dispatch_workflow(workflow: str, workflow_args: Sequence[str]) -> int:
    """Execute a non-model workflow through the primary CLI."""

    return dispatch_script(
        WORKFLOW_SCRIPTS[workflow],
        workflow_args,
        cli_prog=f"surrogate.py {workflow}",
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        options_json, routing_args = extract_options_json_argument(raw_args)
    except OptionsJSONError as exc:
        build_arg_parser().error(str(exc))
    options_args = (
        ["--options-json", options_json] if options_json is not None else []
    )
    if not routing_args or routing_args in (["-h"], ["--help"]):
        build_arg_parser().print_help()
        return 0
    if routing_args[0] in WORKFLOW_SCRIPTS:
        return dispatch_workflow(
            routing_args[0],
            [*routing_args[1:], *options_args],
        )
    args, backend_args = parse_dispatch_args(routing_args)
    return dispatch(args.model_type, [*backend_args, *options_args])


if __name__ == "__main__":
    raise SystemExit(main())
