#!/usr/bin/env python3
"""Unified command-line dispatcher for the surrogate model backends."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


MODEL_SCRIPTS = {
    "dnn": "dnn.py",
    "kbnn": "kbnn.py",
    "neuro-tf": "neuro_tf.py",
}

MODEL_TYPE_ALIASES = {
    "neuro_tf": "neuro-tf",
    "neurotf": "neuro-tf",
}


def normalize_model_type(value: str) -> str:
    """Return the canonical command-line name for a model family."""

    normalized = value.strip().lower()
    return MODEL_TYPE_ALIASES.get(normalized, normalized)


def build_arg_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a DNN, KBNN, or Neuro-TF command through one stable entry point. "
            "All arguments other than --model are forwarded unchanged to the "
            "selected backend."
        ),
        add_help=add_help,
        epilog=(
            "Examples:\n"
            "  python3 surrogate.py --model dnn train "
            "--mdif data.mdif --out-dir outputs/dnn\n"
            "  python3 surrogate.py --model kbnn optimize --help\n"
            "  python3 surrogate.py --model neuro-tf export-veriloga --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        dest="model_type",
        required=True,
        type=normalize_model_type,
        choices=tuple(MODEL_SCRIPTS),
        metavar="{dnn,kbnn,neuro-tf}",
        help="Model family whose backend should receive the command",
    )
    return parser


def parse_dispatch_args(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, list[str]]:
    """Parse only dispatcher arguments and preserve all backend arguments."""

    parser = build_arg_parser(add_help=False)
    args, backend_args = parser.parse_known_args(list(argv))
    if backend_args[:1] == ["--"]:
        backend_args = backend_args[1:]
    return args, backend_args


def dispatch(model_type: str, backend_args: Sequence[str]) -> int:
    """Execute the selected backend with this interpreter and return its status."""

    canonical_model_type = normalize_model_type(model_type)
    script_name = MODEL_SCRIPTS[canonical_model_type]
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.is_file():
        print(
            f"error: backend script for {model_type!r} was not found: {script_path}",
            file=sys.stderr,
        )
        return 2
    environment = os.environ.copy()
    environment["ADS_SURROGATE_CLI_PROG"] = (
        f"surrogate.py --model {canonical_model_type}"
    )
    completed = subprocess.run(
        [sys.executable or "python3", str(script_path), *map(str, backend_args)],
        check=False,
        env=environment,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or raw_args in (["-h"], ["--help"]):
        build_arg_parser().print_help()
        return 0
    args, backend_args = parse_dispatch_args(raw_args)
    return dispatch(args.model_type, backend_args)


if __name__ == "__main__":
    raise SystemExit(main())
