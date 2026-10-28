"""Command-line entry point: `trackvision` or `python -m trackvision`.

Exit codes:
    0  success (for `config validate`: structurally valid, including UNRESOLVED)
    1  smoke fixture check failed
    2  INVALID configuration
    3  UNRESOLVED: a research choice the experiment uses is still null
    4  run directory already exists (use --force to rewrite its files)
    5  no runner exists yet for this experiment kind
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from trackvision import __version__, config, repro
from trackvision.experiments import smoke

EXIT_OK, EXIT_CHECK_FAILED, EXIT_INVALID, EXIT_UNRESOLVED, EXIT_EXISTS, EXIT_NO_RUNNER = range(6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _print_result(result: config.ValidationResult) -> None:
    print(f"status: {result.status}")
    for error in result.errors:
        print(f"  error: {error}")
    if result.config is not None:
        print(f"config_hash: {result.config.hash}")
        if result.unresolved:
            print("unresolved research choices (research runs that use them are blocked):")
            for name in result.unresolved:
                marker = "  (used by this experiment)" if name in result.unresolved_used else ""
                print(f"  - {name}{marker}")


def _cmd_config(args: argparse.Namespace) -> int:
    result = config.validate(args.config, args.base)
    if args.action == "validate":
        _print_result(result)
        return EXIT_INVALID if result.status == config.INVALID else EXIT_OK
    if result.status == config.INVALID:
        _print_result(result)
        return EXIT_INVALID
    if args.action == "hash":
        print(result.config.hash)
    else:
        print(yaml.safe_dump(result.config.canonical_dict(), sort_keys=True, allow_unicode=True), end="")
    return EXIT_OK


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def _cmd_run(args: argparse.Namespace) -> int:
    result = config.validate(args.config, args.base)
    if result.status != config.RUNNABLE:
        _print_result(result)
        return EXIT_INVALID if result.status == config.INVALID else EXIT_UNRESOLVED
    cfg = result.config
    if cfg.experiment["kind"] != "smoke":
        print(f"no runner exists yet for experiment kind {cfg.experiment['kind']!r} (Phase 0A)")
        return EXIT_NO_RUNNER

    # Identity is fixed here, from configuration alone, before any provenance is collected.
    experiment_id = repro.make_experiment_id(cfg.experiment["name"], cfg.hash, None)
    run_dir = Path(args.runs_dir) / experiment_id
    if run_dir.exists() and not args.force:
        print(f"run directory exists: {run_dir}; use --force to rewrite its files")
        return EXIT_EXISTS

    try:
        summary = smoke.run(cfg)
    except smoke.SmokeCheckError as exc:
        print(f"smoke check failed: {exc}")
        return EXIT_CHECK_FAILED

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg.canonical_dict(), sort_keys=True, allow_unicode=True), encoding="utf-8")
    _write_json(run_dir / "fixture_summary.json", summary)
    _write_json(run_dir / "manifest.json", {
        "experiment_id": experiment_id,
        "experiment_name": cfg.experiment["name"],
        "kind": cfg.experiment["kind"],
        "scientific_validity": "none - scaffold check only",
        "config_hash": cfg.hash,
        "seed": None,
        "research_choices_used": list(cfg.experiment["uses_research_choices"]),
        "protocol_complete": cfg.protocol_complete(),
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": _now(),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "artifacts": ["config.resolved.yaml", "fixture_summary.json", "manifest.json"],
    })
    print(f"experiment_id: {experiment_id}")
    print(f"config_hash: {cfg.hash}")
    print(f"wrote: {run_dir}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trackvision", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"trackvision {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    cfg_parser = sub.add_parser("config", help="validate, hash or show the configuration")
    cfg_parser.add_argument("action", choices=["validate", "hash", "show"])
    cfg_parser.add_argument("--config", type=Path, help="experiment file (default: base.yaml only)")
    cfg_parser.add_argument("--base", type=Path, help="base config (default: configs/base.yaml)")
    cfg_parser.set_defaults(handler=_cmd_config)

    run_parser = sub.add_parser("run", help="run an experiment")
    run_parser.add_argument("--config", type=Path, required=True, help="experiment file")
    run_parser.add_argument("--base", type=Path, help="base config (default: configs/base.yaml)")
    run_parser.add_argument("--runs-dir", type=Path, default=config.REPO_ROOT / "runs")
    run_parser.add_argument("--force", action="store_true",
                            help="rewrite the files of an existing run directory")
    run_parser.set_defaults(handler=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
