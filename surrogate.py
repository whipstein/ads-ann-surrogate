#!/usr/bin/env python3
"""Primary command-line entry point for the complete surrogate workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from cli_options import (
    OptionsJSONError,
    add_options_json_argument,
    extract_explain_options_argument,
    extract_options_json_argument,
    extract_update_options_json_argument,
    starter_options_payload,
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
    "debug-model": "debug_model.py",
    "hb-report": "de_generated_scripts/parse_ads_hb_solver_log.py",
}

WORKFLOW_DESCRIPTIONS = {
    "points": "generate initial points or suggest adaptive additions",
    "audit": "audit training and verification MDIF data",
    "debug-model": "diagnose fitting and passivity from retained run artifacts",
    "hb-report": "compare ADS HB Newton/Krylov logs and runtimes",
}

LOCAL_WORKFLOWS = {
    "options": "generate or discover a reusable options JSON",
}


def workflow_names() -> tuple[str, ...]:
    return (*LOCAL_WORKFLOWS, *WORKFLOW_SCRIPTS)


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
            "%(prog)s {options,points,audit,debug-model,hb-report} [COMMAND] [OPTIONS]\n"
            "       %(prog)s --model {dnn,kbnn,neuro-tf} COMMAND [OPTIONS]"
        ),
        epilog=(
            "Examples:\n"
            "  python3 surrogate.py options init --out options.json\n"
            "  python3 surrogate.py options discover . --out recovered-options.json\n"
            "  python3 surrogate.py points generate --parameter W=1mm:2mm "
            "--count 24 --out geometries.csv\n"
            "  python3 surrogate.py audit --mdif data.mdif\n"
            "  python3 surrogate.py debug-model --run-dir outputs/dnn_opt --audit audit\n"
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
        choices=workflow_names(),
        metavar="{options,points,audit,debug-model,hb-report}",
        help=(
            "Non-model workflow command: "
            + "; ".join(
                f"{name} = {description}"
                for name, description in {
                    **LOCAL_WORKFLOWS,
                    **WORKFLOW_DESCRIPTIONS,
                }.items()
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


def build_options_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surrogate.py options",
        description="Generate a starter options JSON or discover one from project artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser(
        "init",
        aliases=["generate"],
        help="Write a documented starter options JSON",
    )
    init.add_argument(
        "--out",
        default="options.json",
        help="Output JSON path. Default: options.json.",
    )
    init.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    discover = commands.add_parser(
        "discover",
        help="Recursively recover settings from an existing project directory",
    )
    discover.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to inspect recursively. Default: current directory.",
    )
    discover.add_argument(
        "--out",
        default="options.json",
        help="Recovered options JSON path. Default: options.json.",
    )
    discover.add_argument(
        "--report",
        help=(
            "Discovery provenance JSON path. Default: "
            "<out-stem>_discovery.json beside --out."
        ),
    )
    discover.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing options and discovery-report files.",
    )
    return parser


def command_options(argv: Sequence[str]) -> int:
    parser = build_options_parser()
    args = parser.parse_args(list(argv))
    out_path = Path(args.out)
    report_path: Path | None = None
    if args.command == "discover":
        report_path = (
            Path(args.report)
            if args.report
            else out_path.with_name(f"{out_path.stem}_discovery.json")
        )
    existing_targets = [
        path for path in (out_path, report_path) if path is not None and path.exists()
    ]
    if existing_targets and not args.overwrite:
        print(
            "error: output already exists: "
            + ", ".join(map(str, existing_targets))
            + "; use --overwrite to replace it",
            file=sys.stderr,
        )
        return 2
    if args.command == "discover":
        from options_discovery import discover_options

        assert report_path is not None
        if out_path.resolve() == report_path.resolve():
            parser.error("--out and --report must name different files")
        try:
            payload, report = discover_options(
                Path(args.directory),
                excluded_paths=(out_path, report_path),
            )
        except ValueError as exc:
            parser.error(str(exc))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out_path}")
        print(f"wrote {report_path}")
        print(
            f"discovered {report['settings_discovered']} setting(s) from "
            f"{len(report['recognized_artifacts'])} recognized artifact(s) and "
            f"{len(report['recovered_commands'])} recovered command(s)"
        )
        if report["conflicts"]:
            print(
                f"warning: resolved {len(report['conflicts'])} conflicting setting(s); "
                f"review {report_path}",
                file=sys.stderr,
            )
        if report["warnings"]:
            print(
                f"warning: discovery recorded {len(report['warnings'])} warning(s); "
                f"review {report_path}",
                file=sys.stderr,
            )
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(starter_options_payload(), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    try:
        options_json, without_options_path = extract_options_json_argument(raw_args)
        update_options_json, without_update = extract_update_options_json_argument(
            without_options_path
        )
        explain_options, routing_args = extract_explain_options_argument(
            without_update
        )
    except OptionsJSONError as exc:
        build_arg_parser().error(str(exc))
    if update_options_json and options_json is None:
        build_arg_parser().error(
            "--update-options-json requires --options-json PATH"
        )
    options_args = (
        ["--options-json", options_json] if options_json is not None else []
    )
    update_args = ["--update-options-json"] if update_options_json else []
    explain_args = ["--explain-options"] if explain_options else []
    if not routing_args or routing_args in (["-h"], ["--help"]):
        build_arg_parser().print_help()
        return 0
    if routing_args[0] in LOCAL_WORKFLOWS:
        if options_json is not None or update_options_json or explain_options:
            build_options_parser().error(
                "--options-json, --update-options-json, and --explain-options cannot "
                "be used while generating an options JSON"
            )
        return command_options(routing_args[1:])
    if routing_args[0] in WORKFLOW_SCRIPTS:
        return dispatch_workflow(
            routing_args[0],
            [*routing_args[1:], *options_args, *update_args, *explain_args],
        )
    args, backend_args = parse_dispatch_args(routing_args)
    return dispatch(
        args.model_type,
        [*backend_args, *options_args, *update_args, *explain_args],
    )


if __name__ == "__main__":
    raise SystemExit(main())
