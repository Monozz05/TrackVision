"""Command-line entry point: `trackvision` or `python -m trackvision`.

Exit codes:
    0  success (for `config validate`: structurally valid, including UNRESOLVED)
    1  smoke fixture check failed
    2  INVALID configuration
    3  UNRESOLVED: a research choice the experiment uses is still null
    4  run directory already exists (use --force to rewrite its files)
    5  no runner exists yet for this experiment kind
    6  the requested experiment is blocked (`sweep`)
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

EXIT_OK, EXIT_CHECK_FAILED, EXIT_INVALID, EXIT_UNRESOLVED, EXIT_EXISTS, EXIT_NO_RUNNER, EXIT_BLOCKED = range(7)


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


def _scenes(scope: str, cfg: config.Config) -> list:
    from trackvision.experiments import learned_feasibility, sweep
    from trackvision.io import ewap
    if scope == sweep.SYNTHETIC:
        return learned_feasibility.synthetic_scenes(cfg)
    return [ewap.load_scene(name, cfg) for name in ("ETH", "HOTEL")]


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Imported here so that `config` and `run` do not load PyTorch."""
    from trackvision.experiments import analysis, store, sweep
    from trackvision.viz.curves import export_figures

    cfg = config.validate().config
    if args.action == "plan":
        print(json.dumps(sweep.plan(cfg), indent=2))
        return EXIT_OK
    if args.action == "run":
        blocked = sweep.blocked_experiments(cfg)
        if args.experiment in blocked:
            print(f"BLOCKED {args.experiment}: {blocked[args.experiment]}")
            return EXIT_BLOCKED
    if args.out is None:
        args.parser.error(f"sweep {args.action} requires --out DIRECTORY")
    if args.action == "run":
        scope = {"synthetic": sweep.SYNTHETIC, "ethhotel": sweep.ETHHOTEL}[args.scope]
        spec = sweep.synthetic_spec(cfg) if scope == sweep.SYNTHETIC else sweep.ethhotel_spec(cfg)
        if args.experiment:
            try:
                spec = sweep.experiment_spec(spec, args.experiment, cfg)
            except sweep.ScopeError as exc:
                args.parser.error(str(exc))
        if args.reuse_from and not spec.common_training:
            args.parser.error(f"--reuse-from applies only to --experiment {sweep.COMPANION}")
        status = sweep.run_sweep(args.out, spec, _scenes(scope, cfg), cfg, resume=args.resume, reuse_from=args.reuse_from)
        print(json.dumps({k: status[k] for k in ("scope", "scope_complete", "research_complete", "expected_cells",
                                                 "finished_cells", "counts", "by_status")}, indent=2))
        print(f"blocked: {sorted(status['blocked'])}")
        return EXIT_OK if status["scope_complete"] else EXIT_CHECK_FAILED
    run = store.RunStore.open(args.out)
    if args.action == "status":
        manifest = run.manifest()
        print(json.dumps({"label": manifest["spec"]["label"], "scope": manifest["spec"]["scope"],
                          "config_hash": manifest["spec"]["config_hash"], "source_sha256": manifest["spec"]["source_sha256"],
                          "created_utc": manifest["created_utc"], "git": manifest["git"], "research_complete": False,
                          "blocked": sweep.blocked_experiments(cfg), **run.summary()}, indent=2))
        return EXIT_OK
    doc = analysis.write_analysis(run, cfg)
    names = export_figures(doc, Path(args.out) / "figures")
    print(f"wrote analysis.json and {len(names)} figures under {args.out}")
    return EXIT_OK


def _cmd_engineering(args: argparse.Namespace) -> int:
    """The limited Phase 5 and Phase 6 scope (decision 0006). Imported here so that `config` does not load PyTorch."""
    from trackvision.experiments import (association_provenance, engineering_analysis, mot17_engineering,
                                         synthetic_tracker_engineering)

    if args.action == "status":
        print(json.dumps(engineering_analysis.status(), indent=2))
        return EXIT_OK
    if args.out is None:
        args.parser.error(f"engineering {args.action} requires --out DIRECTORY")
    cfg = config.validate().config
    needed = {"mot17": ("trackeval_python", "trackeval_source"),
              "synthetic": ("models", "trackeval_python", "trackeval_source"),
              "analysis": ("synthetic_analysis", "ethhotel_analysis", "e4_results"),
              "provenance": ("models", "ethhotel_run", "synthetic_analysis", "ethhotel_analysis")}[args.action]
    missing = [name for name in needed if getattr(args, name) is None]
    if missing:
        args.parser.error(f"engineering {args.action} requires " + ", ".join("--" + m.replace("_", "-") for m in missing))
    if args.action == "provenance":
        from trackvision.experiments import sweep
        entries = association_provenance.write_all(args.out, cfg, args.models, args.ethhotel_run, args.synthetic_analysis,
                                                   args.ethhotel_analysis, _scenes(sweep.SYNTHETIC, cfg),
                                                   _scenes(sweep.ETHHOTEL, cfg))
        print(json.dumps({run: [(e["experiment"], e["association_mechanism"], e["status"]) for e in items]
                          for run, items in entries.items()}, indent=2))
        return EXIT_OK
    if args.action == "mot17":
        doc = mot17_engineering.run(args.out, cfg, args.trackeval_python, args.trackeval_source,
                                    tuple(args.variants.split(",")))
    elif args.action == "synthetic":
        doc = synthetic_tracker_engineering.run(args.out, cfg, args.models, args.trackeval_python, args.trackeval_source,
                                                args.scenes.split(",") if args.scenes else None, args.resume)
    else:
        doc = engineering_analysis.run(args.out, cfg, args.synthetic_analysis, args.ethhotel_analysis, args.e4_results)
    print(f"{doc['label']}\nwrote results.json under {args.out}")
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

    sweep_parser = sub.add_parser("sweep", help="run, resume, inspect or export the eligible Phase 4 experiments")
    sweep_parser.add_argument("action", choices=["plan", "run", "status", "export"])
    sweep_parser.add_argument("--scope", choices=["synthetic", "ethhotel"], default="synthetic")
    sweep_parser.add_argument("--experiment", help="run only this eligible experiment, for example P-4 (default: the "
                                                   "whole eligible scope); a blocked one is refused with exit code 6")
    sweep_parser.add_argument("--out", type=Path, help="run directory, outside the repository or under runs/")
    sweep_parser.add_argument("--resume", action="store_true", help="continue a compatible interrupted run")
    sweep_parser.add_argument("--reuse-from", type=Path, help="earlier run whose H = 12 models the common-population "
                                                              "companion may adopt, once each is proved identical")
    sweep_parser.set_defaults(handler=_cmd_sweep, parser=sweep_parser)

    eng = sub.add_parser("engineering", help="the limited Phase 5 and 6 scope; closes no research gate (decision 0006)")
    eng.add_argument("action", choices=["status", "mot17", "synthetic", "analysis", "provenance"])
    eng.add_argument("--out", type=Path, help="a new run directory, outside the repository or under runs/; for "
                                              "provenance, the folder holding the stored Phase 5 and 6 runs")
    eng.add_argument("--variants", default="FRCNN", help="mot17: detection sets, FRCNN first, for example FRCNN,DPM,SDP")
    eng.add_argument("--scenes", help="synthetic: limit to these scenes for a pilot, for example SYN-501")
    eng.add_argument("--models", type=Path, help="synthetic: the Phase 4 synthetic run, read only")
    eng.add_argument("--resume", action="store_true", help="synthetic: continue a compatible interrupted run")
    eng.add_argument("--trackeval-python", type=Path, help="python.exe of the isolated TrackEval environment")
    eng.add_argument("--trackeval-source", type=Path, help="the pinned TrackEval source tree")
    eng.add_argument("--synthetic-analysis", type=Path, help="analysis: the corrected Phase 4 synthetic analysis.json")
    eng.add_argument("--ethhotel-analysis", type=Path, help="analysis: the ETH and HOTEL descriptive analysis.json")
    eng.add_argument("--e4-results", type=Path, help="analysis: the E4 results.json")
    eng.add_argument("--ethhotel-run", type=Path, help="provenance: the Phase 4 ETH and HOTEL run, read only")
    eng.set_defaults(handler=_cmd_engineering, parser=eng)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
