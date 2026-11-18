"""Phase 4 sweep runner for the currently eligible experiments. Resumable, and never a research result.

Two scopes (decision 0001 F; decision 0004):
  synthetic_engineering   P-3 to P-7, P-12 and P-15 with CV, CA, LSTM and GRU on synthetic scenes, on nested
                          scene-level splits with a separate fixture rotation. ENGINEERING FEASIBILITY ONLY.
  ethhotel_descriptive    P-3 to P-7 with CV and CA on ETH and HOTEL, one scene at a time, never pooled.
                          DESCRIPTIVE AND LOCAL ONLY. Learned models are refused on real scenes.

P-8, P-10, P-11 and the five-scene static research protocol are recorded as blocked in every run, so a finished
sweep can never read as a complete study.

One cell is one predictor, condition, requested level, seed, history length and population on one held-out scene.
The same seed drives training and injection. Windows and clean reference errors are built once per unit, CV and
CA clean errors once per population, and injections once per (level, seed) and only if a cell needs them.
Learned models are trained per fold, seed and history (`predict.training.load_or_train`).

P-15 evaluates two populations: `per_history`, the anchors eligible at each history (section 3.2), and `common_h12`,
the anchors eligible at H = 12, evaluated at every history. Models are trained on the per-history training windows
in both; only the evaluation population differs, so `common_h12` is the supplementary common EVALUATION population
and does not isolate history length for LSTM or GRU.

The revised companion (decision 0005, section 2) is a separate run with `common_training` set, and its population is
`common_h12_trained`: the H = 12-eligible anchors are used for training, validation and evaluation at every history,
so only the input length differs. `common_preflight` proves that before anything is trained. A model trained at
H = 12 by an earlier run may be adopted with `reuse_from`, only after its stored split, configuration and settings
are shown to equal this run's.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trackvision.config import Config, UnresolvedChoiceError
from trackvision.experiments import evaluation, injection_feasibility as inj
from trackvision.experiments.learned_feasibility import fixture_rotation
from trackvision.experiments.mot17_trackeval import check_output_location
from trackvision.experiments.store import (BLOCKED, COMPLETE, FAILED, UNREALIZED, RunStore, failure_info, identity_key,
                                           source_sha256)
from trackvision.io.atomic import atomic_write_bytes, atomic_write_text
from trackvision.predict import learned
from trackvision.predict.folds import NestedSplit, nested_split
from trackvision.predict.leakage import (NORMALIZATION_NOT_USED, array_hash, require_no_leakage, run_leakage_checks,
                                         temporal_ordering, window_boundary)
from trackvision.predict.training import (CheckpointError, TrainingSettings, checkpoint_mismatches, load_checkpoint,
                                          load_or_train)
from trackvision.predict.windowing import Windows, make_windows

SYNTHETIC, ETHHOTEL = "synthetic_engineering", "ethhotel_descriptive"
PER_HISTORY, COMMON, COMMON_TRAINED = "per_history", "common_h12", "common_h12_trained"
COMPANION = "P-15-common-companion"
POPULATION_LABELS = {
    PER_HISTORY: "Per-history population: the anchors eligible at each history length. Primary.",
    COMMON: ("Common evaluation population \u2014 training and validation remain per-history. Supplementary and "
             "descriptive: it does not isolate history length for LSTM or GRU."),
    COMMON_TRAINED: ("Common training, validation and evaluation population: the H = 12-eligible anchors at every "
                     "history length, so only the input length differs."),
}
EXPERIMENT_CONDITIONS = {"P-3": (), "P-4": ("A",), "P-5": ("B1", "B2"), "P-6": ("C",), "P-7": ("D",)}
CONDITION_EXPERIMENT = {"clean": "P-3", "A": "P-4", "B1": "P-5", "B2": "P-5", "C": "P-6", "D": "P-7"}
DETERMINISTIC = ("CV", "CA")
LABELS = {
    SYNTHETIC: ("SYNTHETIC ENGINEERING FEASIBILITY ONLY. Synthetic scenes from one generator; not real-world "
                "generalisation, not the five-scene research protocol, no statistically supported conclusion. "
                "v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen."),
    ETHHOTEL: ("DESCRIPTIVE AND LOCAL ONLY. CV and CA on ETH and HOTEL, one scene at a time. Dataset permissions "
               "are unconfirmed: do not report, publish or commit. Not the five-scene research protocol, not a "
               "benchmark. v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen."),
}


class ScopeError(ValueError):
    """The requested run would exceed what is currently eligible."""


class IntegrityError(RuntimeError):
    """Clean tracks or targets changed during a unit."""


@dataclass(frozen=True)
class SweepSpec:
    scope: str
    predictors: tuple
    conditions: tuple
    histories: tuple
    seeds: tuple
    settings: TrainingSettings | None = None
    common_training: bool = False

    def as_dict(self) -> dict:
        return {"scope": self.scope, "predictors": list(self.predictors), "conditions": list(self.conditions),
                "histories": list(self.histories), "seeds": list(self.seeds),
                "settings": dataclasses.asdict(self.settings) if self.settings else None,
                "common_training": self.common_training}


def synthetic_spec(cfg: Config, histories=None, settings: TrainingSettings | None = None) -> SweepSpec:
    lens = histories or cfg.frozen["analysis"]["history_ablation"]["history_lens"]
    return SweepSpec(SYNTHETIC, (*DETERMINISTIC, *learned.CELLS), evaluation.CONDITIONS, tuple(lens),
                     tuple(cfg.research_choice("seeds")), settings or TrainingSettings.from_config(cfg))


def ethhotel_spec(cfg: Config) -> SweepSpec:
    return SweepSpec(ETHHOTEL, DETERMINISTIC, evaluation.CONDITIONS,
                     (cfg.frozen["protocol"]["static"]["history_len"],), tuple(cfg.research_choice("seeds")))


def blocked_experiments(cfg: Config) -> dict:
    """Experiments that cannot run, with the reason. Nothing here is skipped silently."""
    try:
        moving = f"moving_dataset is {cfg.research_choice('moving_dataset')!r}"
    except UnresolvedChoiceError:
        moving = "moving_dataset is unresolved"
    return {
        "P-8": f"{moving}; the E injectors are not implemented",
        "P-10": ("protocol decisions are open (decision 0004, sections 5 and 6): target severity, reseeding, C x D "
                 "mechanics, bootstrap constants, record shape, and the v4.15 six-pair versus v4.16 selected-pair "
                 "conflict; h8_generation_attempts is null"),
        "P-11": f"{moving}, so the moving-camera arm is unavailable, and the static five-scene arm is incomplete",
        "five-scene-static-research": ("only ETH and HOTEL have usable world coordinates; UCY needs calibration "
                                       "provenance and permission (decision 0001 D)"),
    }


def plan(cfg: Config) -> dict:
    return {"eligible": {SYNTHETIC: ["P-3", "P-4", "P-5", "P-6", "P-7", "P-12", "P-15", COMPANION],
                         ETHHOTEL: ["P-3", "P-4", "P-5", "P-6", "P-7"]},
            "blocked": blocked_experiments(cfg)}


def experiment_spec(spec: SweepSpec, experiment: str, cfg: Config) -> SweepSpec:
    """`spec` narrowed to one eligible experiment. P-3 to P-7 and P-12 use the primary history; P-15 needs every
    history and every condition, and clean cells are always included because degradation is relative to them."""
    if experiment not in plan(cfg)["eligible"][spec.scope]:
        raise ScopeError(f"{experiment} is not eligible in scope {spec.scope}")
    if experiment == COMPANION:
        return dataclasses.replace(spec, common_training=True)
    if experiment == "P-15":
        return spec
    primary = dataclasses.replace(spec, histories=(cfg.frozen["protocol"]["static"]["history_len"],))
    if experiment == "P-12":
        return primary
    return dataclasses.replace(primary, conditions=EXPERIMENT_CONDITIONS[experiment])


def populations_of(spec: SweepSpec) -> tuple:
    if spec.common_training:
        return (COMMON_TRAINED,)
    return (PER_HISTORY, COMMON) if len(spec.histories) > 1 else (PER_HISTORY,)


def _check_scope(spec: SweepSpec, scenes, cfg: Config, reuse_from=None) -> None:
    real = set(cfg.frozen["protocol"]["static"]["scenes"])
    names = {s.name for s in scenes}
    if spec.common_training and spec.scope != SYNTHETIC:
        raise ScopeError("the common-population companion is synthetic engineering only")
    if reuse_from is not None and not spec.common_training:
        raise ScopeError("only the common-population companion can adopt models from an earlier run")
    if spec.scope == SYNTHETIC:
        if names & real:
            raise ScopeError(f"synthetic_engineering refuses real scenes: {sorted(names & real)}")
    elif spec.scope == ETHHOTEL:
        if set(spec.predictors) - set(DETERMINISTIC):
            raise ScopeError("learned models are never run on ETH or HOTEL (decision 0001 F)")
        if names - {"ETH", "HOTEL"} or not names:
            raise ScopeError(f"ethhotel_descriptive takes ETH and HOTEL only, got {sorted(names)}")
    else:
        raise ScopeError(f"unknown scope {spec.scope!r}")


def common_anchors(tracks, frame_step: int, history: int, horizon: int, stride: int) -> set:
    """(track id, last observed frame) of every window eligible at `history`."""
    win = make_windows("anchors", tracks, frame_step, history, horizon, stride)
    return {(int(g), int(f[history - 1])) for g, f in zip(win.track_ids, win.frames)}


def restrict_windows(windows: Windows, anchors: set) -> Windows:
    """The windows whose (track, last observed frame) is in `anchors`, in their original order."""
    h = windows.history_len
    keep = np.array([(int(g), int(f[h - 1])) in anchors for g, f in zip(windows.track_ids, windows.frames)], dtype=bool)
    return dataclasses.replace(windows, observed=windows.observed[keep], target=windows.target[keep],
                               track_ids=windows.track_ids[keep], frames=windows.frames[keep])


def common_windows(scenes, history: int, common_h: int, horizon: int, stride: int) -> list:
    """Every scene's windows at `history`, on that scene's own anchors eligible at `common_h`."""
    return [restrict_windows(make_windows(s.name, s.tracks, s.frame_step, history, horizon, stride),
                             common_anchors(s.tracks, s.frame_step, common_h, horizon, stride)) for s in scenes]


def _require_only_input_length_differs(windows: Windows, reference: Windows) -> None:
    """Against the windows at the longest history: same anchors in the same order, same targets and target frames,
    and observed points that are the last `history_len` observed points of the same track."""
    same = (len(windows) == len(reference) and np.array_equal(windows.track_ids, reference.track_ids)
            and np.array_equal(windows.target, reference.target)
            and np.array_equal(windows.frames[:, windows.history_len - 1:], reference.frames[:, reference.history_len - 1:])
            and np.array_equal(windows.observed, reference.observed[:, -windows.history_len:]))
    if not same:
        raise IntegrityError(f"{windows.scene}: H = {windows.history_len} differs from H = {reference.history_len} "
                             "in more than the input length")


def common_preflight(spec: SweepSpec, scenes, cfg: Config) -> dict:
    """Decision 0005, section 2: run before any training, and nothing is written if a check fails.

    Returns the anchor construction, the counts per fold and history, and the leakage-check results. Anchors are
    read from frame numbers alone: they are recomputed with every position zeroed and must not change.
    """
    static = cfg.frozen["protocol"]["static"]
    horizon, stride = static["horizon_len"], static["window_stride"]
    common_h = max(cfg.frozen["analysis"]["history_ablation"]["history_lens"])
    anchors = {}
    for s in scenes:
        anchors[s.name] = common_anchors(s.tracks, s.frame_step, common_h, horizon, stride)
        blank = tuple(dataclasses.replace(t, positions=np.zeros_like(t.positions)) for t in s.tracks)
        if common_anchors(blank, s.frame_step, common_h, horizon, stride) != anchors[s.name]:
            raise IntegrityError(f"{s.name}: anchors depend on positions")
    reference = {w.scene: w for w in common_windows(scenes, common_h, common_h, horizon, stride)}
    rotation = fixture_rotation(s.name for s in scenes)
    folds = []
    for history in spec.histories:
        windows = common_windows(scenes, history, common_h, horizon, stride)
        for w in windows:
            _require_only_input_length_differs(w, reference[w.scene])
        for s in scenes:
            split = nested_split(windows, s.name, rotation)
            checks = run_leakage_checks(split, scenes, learned.window_inputs, None, NORMALIZATION_NOT_USED)
            require_no_leakage(checks)
            folds.append({"history_len": history, "test_scene": split.test_scene,
                          "validation_scene": split.validation_scene, "train_scenes": list(split.train_scenes),
                          "train_windows": sum(len(w) for w in split.train), "validation_windows": len(split.validation),
                          "test_windows": len(split.test), "leakage_checks": {r.name: r.passed for r in checks}})
    return {"construction": (f"For each scene, the anchors are (track id, last observed frame) of every window eligible "
                             f"at H = {common_h}, F = {horizon}, from frame numbers only. At each history the scene's "
                             "windows are cut as usual and kept only if their anchor is in that set, in original order."),
            "common_history": common_h, "anchors_by_scene": {name: len(a) for name, a in anchors.items()},
            "input_length_is_the_only_difference": True, "folds": folds}


def scene_window_checks(scene, windows: Windows) -> None:
    """Ordering and boundary checks for an unsplit scene. Every partition holds the same windows because these
    two checks look at the windows only; the checks about partitions do not apply to a scene with no split."""
    probe = NestedSplit(scene.name, scene.name, (scene.name,), windows, windows, (windows,))
    problems = [p for r in (temporal_ordering(probe), window_boundary(probe, [scene])) for p in r.problems]
    if problems:
        raise IntegrityError(f"{scene.name}: {problems[:3]}")


@dataclass
class _Context:
    store: RunStore
    spec: SweepSpec
    cfg: Config
    out_dir: Path
    source: str
    scene_hash: dict
    counts: Counter
    reuse_from: Path | None = None


def _identity(ctx: _Context, scene, fold, validation, predictor, condition, level, seed, history, population,
              windows_hash, checkpoint=None) -> dict:
    primary = ctx.cfg.frozen["protocol"]["static"]["history_len"]
    experiment = (CONDITION_EXPERIMENT[condition] if history == primary and population == PER_HISTORY else "P-15")
    return {"scope": ctx.spec.scope, "experiment": experiment, "scene": scene, "fold": fold,
            "validation_scene": validation, "predictor": predictor, "condition": condition,
            "requested_level": level, "seed": seed, "history_len": history,
            "horizon_len": ctx.cfg.frozen["protocol"]["static"]["horizon_len"], "population": population,
            "config_hash": ctx.cfg.hash, "source_sha256": ctx.source, "scene_sha256": ctx.scene_hash[scene],
            "windows_sha256": windows_hash, "checkpoint_sha256": checkpoint,
            "settings": dataclasses.asdict(ctx.spec.settings) if predictor in learned.CELLS else None}


def _adopt_checkpoint(ctx: _Context, split: NestedSplit, cell: str, seed: int, history: int, folder: Path) -> str:
    """Copy a model from an earlier run, only once it is shown to be the run this fold would do itself: file hash,
    state hash, configuration, settings, and the hash of every training, validation and test scene's windows."""
    name = f"{cell.lower()}-seed{seed}.pt"
    source = Path(ctx.reuse_from) / "checkpoints" / f"h{history}" / split.test_scene / name
    sidecar = source.with_suffix(".pt.json")
    if not sidecar.exists():
        raise CheckpointError(f"no checkpoint to adopt: {sidecar}")
    _, meta = load_checkpoint(source, json.loads(sidecar.read_text(encoding="utf-8"))["checkpoint_sha256"])
    differing = checkpoint_mismatches(meta, split, cell, seed, ctx.cfg, ctx.spec.settings)
    if differing:
        raise CheckpointError(f"{source} cannot be adopted: it differs in {differing}")
    target = folder / name
    if not target.with_suffix(".pt.json").exists():
        atomic_write_bytes(target, source.read_bytes())
        atomic_write_bytes(target.with_suffix(".pt.json"), sidecar.read_bytes())      # last: it is the commit marker
        ctx.counts["trainings_adopted"] += 1
    return str(source)


def _models(ctx: _Context, split: NestedSplit, scenes, history: int) -> dict:
    """Train or reload every learned model of one fold. Failures are recorded and block only their own cells."""
    models = {}
    for cell in (p for p in ctx.spec.predictors if p in learned.CELLS):
        for seed in ctx.spec.seeds:
            identity = {"scope": ctx.spec.scope, "experiment": "training", "fold": split.test_scene,
                        "validation_scene": split.validation_scene, "predictor": cell, "seed": seed,
                        "history_len": history, "config_hash": ctx.cfg.hash, "source_sha256": ctx.source}
            try:
                folder = ctx.out_dir / "checkpoints" / f"h{history}" / split.test_scene
                adopted = None
                if ctx.reuse_from is not None and history == max(ctx.spec.histories):
                    adopted = _adopt_checkpoint(ctx, split, cell, seed, history, folder)
                result, reused = load_or_train(split, scenes, cell, seed, ctx.cfg, folder, LABELS[ctx.spec.scope],
                                               ctx.spec.settings)
            except Exception as exc:                  # includes LeakageError: nothing is written for this model
                ctx.store.put(identity, FAILED, failure=failure_info(exc))
                ctx.counts["trainings_failed"] += 1
                continue
            ctx.counts["trainings_reused" if reused else "trainings_run"] += 1
            models[(cell, seed)] = (result.model, result.checkpoint_sha256)
            m = result.metadata
            ctx.store.put(identity, COMPLETE, {"checkpoint_sha256": result.checkpoint_sha256,
                                               "state_sha256": m["state_sha256"], "best_epoch": m["best_epoch"],
                                               "epochs_run": m["epochs_run"], "stop_reason": m["stop_reason"],
                                               "best_validation_ade_m": m["best_validation_ade_m"],
                                               "adopted_from": adopted})
    return models


def _write_cell(ctx: _Context, identity: dict, predictor, corrupted, windows, clean_per_step, requested) -> None:
    achieved = None
    if corrupted.condition != "clean":
        achieved = {"measure": corrupted.measure, "value": corrupted.value, "detail": corrupted.achieved}
    metrics = evaluation.score(predictor, corrupted, windows, clean_per_step)
    status = UNREALIZED if achieved is not None and achieved["value"] is None else COMPLETE
    ctx.store.put(identity, status, {"requested": requested, "achieved": achieved, "metrics": metrics,
                                     "extra": corrupted.extra})
    ctx.counts["cells_computed"] += 1


def _evaluate(ctx: _Context, unit, history: int, population: str, windows: Windows, models: dict) -> None:
    scene, fold, validation = unit["scene"], unit["fold"], unit["validation"]
    cfg, spec = ctx.cfg, ctx.spec
    pop = evaluation.scene_population(scene.tracks, windows)
    windows_hash = inj._digest(windows.observed, windows.target, windows.track_ids, windows.frames)
    tracks_before = inj.tracks_digest(scene.tracks)
    target_before = array_hash(windows.target)
    fixed = {"CV": evaluation.cv_predictor(cfg, windows.horizon_len), "CA": evaluation.ca_predictor(cfg, windows.horizon_len)}
    clean_cache: dict = {}

    def predictor_for(name, seed):
        if name in fixed:
            return fixed[name], None
        model, checkpoint = models[(name, seed)]
        return evaluation.learned_predictor(model), checkpoint

    def clean_errors(name, seed, predictor):
        key = (name, None if name in fixed else seed)        # CV and CA are the same for every seed
        if key not in clean_cache:
            clean_cache[key] = evaluation.clean_errors(predictor, windows)
        return clean_cache[key]

    grid = [("clean", 0.0)] + [(c, level) for c in spec.conditions for level in inj.levels(cfg, c)]
    for seed in spec.seeds:
        names = [p for p in spec.predictors if p in fixed or (p, seed) in models]
        for condition, level in grid:
            pending = []
            for name in names:
                predictor, checkpoint = predictor_for(name, seed)
                identity = _identity(ctx, scene.name, fold, validation, name, condition,
                                     None if condition == "clean" else level, seed, history, population,
                                     windows_hash, checkpoint)
                record = ctx.store.get(identity)
                if record is not None and record.status in (COMPLETE, UNREALIZED):
                    ctx.counts["cells_reused"] += 1
                else:
                    pending.append((name, predictor, identity))
            if not pending:
                continue
            try:
                corrupted = (evaluation.clean_inputs(windows) if condition == "clean" else
                             evaluation.corrupt(scene.tracks, pop, condition, level, seed, cfg, scene.frame_step))
            except Exception as exc:
                for _, _, identity in pending:
                    ctx.store.put(identity, FAILED, failure=failure_info(exc))
                    ctx.counts["cells_failed"] += 1
                continue
            requested = None if condition == "clean" else inj.requested(cfg, condition, level)
            for name, predictor, identity in pending:
                try:
                    _write_cell(ctx, identity, predictor, corrupted, windows,
                                clean_errors(name, seed, predictor), requested)
                except Exception as exc:
                    ctx.store.put(identity, FAILED, failure=failure_info(exc))
                    ctx.counts["cells_failed"] += 1
    if inj.tracks_digest(scene.tracks) != tracks_before or array_hash(windows.target) != target_before:
        raise IntegrityError(f"{scene.name}: clean tracks or targets changed during evaluation")


def expected_cells(spec: SweepSpec, cfg: Config, n_units: int) -> int:
    per_seed = 1 + sum(len(inj.levels(cfg, c)) for c in spec.conditions)
    return n_units * len(spec.histories) * len(populations_of(spec)) * len(spec.seeds) * len(spec.predictors) * per_seed


def run_sweep(out_dir: Path, spec: SweepSpec, scenes, cfg: Config, resume: bool = False, reuse_from: Path | None = None) -> dict:
    """Run or resume `spec` on `scenes`. Complete cells are reused only if their whole identity matches."""
    _check_scope(spec, scenes, cfg, reuse_from)
    check_output_location(out_dir)
    static = cfg.frozen["protocol"]["static"]
    horizon, stride = static["horizon_len"], static["window_stride"]
    source = source_sha256()
    scene_hash = {s.name: inj.tracks_digest(s.tracks) for s in scenes}
    run_spec = {**spec.as_dict(), "config_hash": cfg.hash, "source_sha256": source, "scenes": scene_hash,
                "horizon_len": horizon, "window_stride": stride, "label": LABELS[spec.scope],
                "torch": str(torch.__version__), "torch_num_threads": torch.get_num_threads()}
    preflight = None
    if spec.common_training:
        preflight = common_preflight(spec, scenes, cfg)
        run_spec.update(preflight_sha256=identity_key(preflight), adopts_models_from=reuse_from and str(reuse_from))
    store = RunStore.create(out_dir, run_spec, resume)
    if preflight:
        atomic_write_text(Path(out_dir) / "preflight.json", json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    ctx = _Context(store, spec, cfg, Path(out_dir), source, scene_hash, Counter(), reuse_from)

    for experiment, reason in blocked_experiments(cfg).items():
        store.put({"scope": spec.scope, "experiment": experiment, "config_hash": cfg.hash}, BLOCKED,
                  {"reason": reason})

    uses_models = any(p in learned.CELLS for p in spec.predictors)
    common_h = max(cfg.frozen["analysis"]["history_ablation"]["history_lens"])
    populations = populations_of(spec)
    for history in spec.histories:
        if spec.common_training:
            all_windows = common_windows(scenes, history, common_h, horizon, stride)
        else:
            all_windows = [make_windows(s.name, s.tracks, s.frame_step, history, horizon, stride) for s in scenes]
        for w in all_windows:
            w.target.setflags(write=False)
        rotation = fixture_rotation(s.name for s in scenes) if uses_models else None
        for scene, windows in zip(scenes, all_windows):
            unit = {"scene": scene, "fold": None, "validation": None}
            models: dict = {}
            try:
                if uses_models:
                    split = nested_split(all_windows, scene.name, rotation)
                    unit.update(fold=split.test_scene, validation=split.validation_scene)
                    checks = run_leakage_checks(split, scenes, learned.window_inputs, None, NORMALIZATION_NOT_USED)
                    require_no_leakage(checks)
                    models = _models(ctx, split, scenes, history)
                else:
                    scene_window_checks(scene, windows)
            except Exception as exc:
                store.put({"scope": spec.scope, "experiment": "unit", "scene": scene.name, "history_len": history,
                           "config_hash": cfg.hash, "source_sha256": source}, FAILED, failure=failure_info(exc))
                ctx.counts["units_failed"] += 1
                continue
            for population in populations:
                chosen = windows
                if population == COMMON:
                    anchors = common_anchors(scene.tracks, scene.frame_step, common_h, horizon, stride)
                    chosen = restrict_windows(windows, anchors)
                _evaluate(ctx, unit, history, population, chosen, models)

    summary = store.summary()
    expected = expected_cells(spec, cfg, len(scenes))
    finished = sum(v.get(COMPLETE, 0) + v.get(UNREALIZED, 0) for k, v in summary["by_experiment"].items()
                   if k.startswith("P-") and k not in blocked_experiments(cfg))
    status = {"scope": spec.scope, "label": LABELS[spec.scope], "populations": {p: POPULATION_LABELS[p] for p in populations},
              "expected_cells": expected, "finished_cells": finished,
              "scope_complete": finished >= expected and summary["by_status"][FAILED] == 0,
              "research_complete": False, "blocked": blocked_experiments(cfg), "counts": dict(sorted(ctx.counts.items())),
              "by_status": summary["by_status"], "by_experiment": summary["by_experiment"]}
    atomic_write_text(Path(out_dir) / "run_status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status
