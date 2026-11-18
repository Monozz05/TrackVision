"""Feasibility sweep of identity-switch injection (D) on clean synthetic ground truth, with CV
prediction under window policy P, coverage and saved results (U8).

FEASIBILITY ONLY: not a v4.15 section 11.3 record, not gate evidence, no hypothesis test; descriptive
only (overlapping, correlated windows). Injection seeds (0, 1, 2 by default) are feasibility values for
this sweep. The research choice `seeds` is decided separately (docs/decisions/0001-p2d-protocol-decisions.md,
B) and governs research runs, which this is not. A parallel module to U7's injection_feasibility, whose
A/B1/B2/C sweep is unchanged; U7's population, integrity and summary helpers are reused.

Parameters. D's levels, D_MAX (`injection.d_max_m`) and the cooldown (`injection.d_cooldown`) are read
from the configuration. Only the primary cooldown `locked_until_end_of_segment` is implemented; the
segment is the whole synthetic scene.

Population. U2's clean ground-truth windows, identical for every level and seed. Every injection starts
from a fresh copy of the clean trajectories, which are switched once per (level, seed), so overlapping
windows share the same relabelling. Targets always come from the clean windows; their hash and that of
the clean trajectories are checked unchanged after the sweep.

Window policy P (approved). For the window of pedestrian g with history timesteps t_1 .. t_H, the CV
input is the history of the identity label that holds g at t_H, at t_1 .. t_H; the target is g's clean
future. If that label lacks a sample at any history timestep, the window is unpredicted
(`label_history_incomplete`). A switch at or before t_1, or after t_H, leaves the input unchanged; one at
t_2 .. t_H gives a mixed history; CV reads only the last two observed points, so its input changes only
when the history's last two points are not both g's. This differs from U6, where identity-mixed tracker
histories were unscorable; the two are reported separately and are not comparable.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments.injection_feasibility import (PROVISIONAL_SEEDS, InjectionIntegrityError, Population,
                                                           _accuracy, _digest, _json, _seed_aggregate, _stats,
                                                           fresh_copy, levels, population, requested,
                                                           tracks_digest)
from trackvision.experiments.synthetic_feasibility import evaluate_ground_truth_cv, protocol
from trackvision.inject.identity_switch import identity_switch
from trackvision.io.manifest import file_sha256
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks

NAME = "synthetic-switch-feasibility"
CONDITION = "D"
PRIMARY_COOLDOWN = "locked_until_end_of_segment"
LOST_REASON = "label_history_incomplete"
LABEL = ("Feasibility identity-switch injection sweep (D) on clean synthetic ground truth (controlled proxy). "
         "Not a v4.15 section 11.3 record; not gate evidence; descriptive only. Injection seeds are "
         "provisional feasibility values. v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
NAMING_NOTE = ("switch_event_rate is the achieved severity of the synthetic identity-switch injector (spec "
               "section 3.5 naming rule). It is not the TrackEval identity-switch metric and is never stored "
               "under that metric's name.")
POLICY_NOTE = ("Policy P: CV input = history of the label holding the window's pedestrian at the last history "
               "timestep; target = that pedestrian's clean future. Mixed histories are predicted, unlike U6.")


def check_cooldown(cfg: Config) -> str:
    rule = cfg.frozen["injection"]["d_cooldown"]
    if rule != PRIMARY_COOLDOWN:
        raise ValueError(f"only the primary cooldown {PRIMARY_COOLDOWN!r} is implemented, config has {rule!r}")
    return rule


@dataclass(frozen=True)
class SwitchWindows:
    predicted: np.ndarray            # (N,) bool
    reason: tuple                    # (N,) str | None
    label: np.ndarray                # (N,) label holding the window's pedestrian at t_H
    mixed: np.ndarray                # (N,) bool: a history sample is occupied by another pedestrian
    cv_changed: np.ndarray           # (N,) bool: the points CV reads differ from the clean history's
    inputs: np.ndarray               # (P, history_len, 2) for predicted windows, in window order


def switch_windows(pop: Population, tracks, occupant: dict, cv_points: int) -> SwitchWindows:
    """Policy P for every eligible window, given relabelled tracks and their occupant map."""
    win, n, h = pop.windows, len(pop.windows), pop.windows.history_len
    by_label = {int(t.ped_id): t for t in tracks}
    index = {label: {int(f): k for k, f in enumerate(t.frames)} for label, t in by_label.items()}
    holder = {(int(o), int(f)): label for label, t in by_label.items() for f, o in zip(t.frames, occupant[label])}
    predicted = np.zeros(n, dtype=bool)
    mixed, changed = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    label_used = np.zeros(n, dtype=np.int64)
    reason: list = [None] * n
    inputs = []
    for w, (g, frames) in enumerate(zip(win.track_ids, win.frames)):
        hist = [int(f) for f in frames[:h]]
        label = holder[(int(g), hist[-1])]
        label_used[w] = label
        if any(f not in index[label] for f in hist):
            reason[w] = LOST_REASON
            continue
        k = [index[label][f] for f in hist]
        x = by_label[label].positions[k]
        predicted[w] = True
        mixed[w] = bool(np.any(occupant[label][k] != g))
        changed[w] = not np.array_equal(x[-cv_points:], win.observed[w][-cv_points:])
        inputs.append(x)
    return SwitchWindows(predicted, tuple(reason), label_used, mixed, changed,
                         np.asarray(inputs, dtype=np.float64).reshape(-1, h, 2))


@dataclass(frozen=True)
class SwitchResult:
    level: float
    seed: int
    events: tuple
    windows: SwitchWindows
    per_step: np.ndarray             # (P, horizon_len)
    summary: dict


def evaluate_switch(clean_tracks, pop: Population, level: float, seed: int, cfg: Config,
                    frame_step: int) -> SwitchResult:
    p = protocol(cfg)
    d_max = float(cfg.frozen["injection"]["d_max_m"])
    win = pop.windows
    n = len(win)
    sw = identity_switch(fresh_copy(clean_tracks), level, seed, d_max, frame_step)
    sw_windows = switch_windows(pop, sw.tracks, sw.occupant, p["cv_min_final_observations"])
    idx = np.flatnonzero(sw_windows.predicted)
    per_step = displacement_errors(constant_velocity(sw_windows.inputs, p["horizon_len"],
                                                     p["cv_min_final_observations"]), win.target[idx])
    injected, clean_same = _accuracy(per_step), _accuracy(pop.clean_per_step[idx])
    summary = {
        "condition": CONDITION,
        "seed": int(seed),
        "requested": {**requested(cfg, CONDITION, level), "nominal_per_100_track_frames": 100.0 * level},
        "achieved": sw.achieved,
        "achieved_primary": {"name": "switch_event_rate", "value": sw.achieved["switch_event_rate"]},
        "events": [{"frame": e.frame, "initiator": e.initiator, "partner": e.partner, "distance_m": e.distance_m}
                   for e in sw.events],
        "coverage": {"eligible_windows": int(n), "predicted_windows": int(len(idx)),
                     "prediction_coverage": float(len(idx) / n) if n else None, "lost_windows": int(n - len(idx)),
                     "lost_by_reason": {LOST_REASON: int(n - len(idx))}},
        "history": {"windows_with_mixed_history": int(sw_windows.mixed.sum()),
                    "windows_with_changed_cv_input": int(sw_windows.cv_changed.sum()),
                    "policy": POLICY_NOTE},
        "accuracy_on_predicted_windows": {
            "injected": injected,
            "clean_same_windows": clean_same,
            "mean_ade_difference_injected_minus_clean_m": (
                injected["mean_ade_m"] - clean_same["mean_ade_m"] if injected["mean_ade_m"] is not None else None),
            "mean_fde_difference_injected_minus_clean_m": (
                injected["mean_fde_m"] - clean_same["mean_fde_m"] if injected["mean_fde_m"] is not None else None),
        },
    }
    return SwitchResult(float(level), int(seed), sw.events, sw_windows, per_step, summary)


def _switch_aggregate(results) -> dict:
    agg = _seed_aggregate(results)
    collect = lambda get: [v for v in (get(r.summary) for r in results) if v is not None]
    agg.update({
        "switch_events": _stats(collect(lambda s: s["achieved"]["switch_events"])),
        "suppressed_triggers": _stats(collect(lambda s: s["achieved"]["suppressed_triggers"])),
        "lock_aware_exposure": _stats(collect(lambda s: s["achieved"]["lock_aware_exposure"])),
        "windows_with_mixed_history": _stats(collect(lambda s: s["history"]["windows_with_mixed_history"])),
        "windows_with_changed_cv_input": _stats(collect(lambda s: s["history"]["windows_with_changed_cv_input"])),
        "injected_minus_clean_mean_ade_m": _stats(collect(
            lambda s: s["accuracy_on_predicted_windows"]["mean_ade_difference_injected_minus_clean_m"])),
    })
    return agg


@dataclass(frozen=True)
class SwitchRun:
    experiment_id: str
    settings: dict
    settings_hash: str
    config_hash: str
    scene_content_hash: str
    clean_tracks_sha256: str
    targets_sha256: str
    eligible_track_frames: int
    reference: dict                  # U2 clean CV summary on all eligible windows
    results: tuple                   # SwitchResult, ordered by level, seed
    population: Population


def run_settings(cfg: Config, scene_config: SceneConfig, seeds) -> dict:
    p = protocol(cfg)
    return {
        "scene": scene_config.as_dict(),
        "scene_seed": scene_config.seed,
        "injection_seeds": [int(s) for s in seeds],
        "injection_seeds_status": "provisional feasibility values; not the research choice 'seeds'",
        "condition": {"name": CONDITION, "grid": cfg.frozen["injection"]["conditions"][CONDITION]["grid"],
                      "levels": levels(cfg, CONDITION)},
        "d_max_m": float(cfg.frozen["injection"]["d_max_m"]),
        "d_cooldown": check_cooldown(cfg),
        "segment": "the whole synthetic scene (approved U8-1a)",
        "exchange": "labels exchanged from the event timestep onwards, inclusively",
        "eligibility": "both tracks sampled at k - 1 and k; distance at k <= d_max_m",
        "denominator": "pre-injection proximity-eligible track frames (approved U8-2a); lock-aware exposure supplementary",
        "random_numbers": "SeedSequence([seed, 4, ped_id]); one uniform per track sample, shared across levels",
        "order": "timesteps ascending, initiating ped_id ascending; partner nearest unlocked, ties lower ped_id",
        "window_policy": POLICY_NOTE,
        "naming": NAMING_NOTE,
        "base": "clean U1 ground-truth grid trajectories (whole scene)",
        "predictor": {"name": "CV", "cv_min_final_observations": p["cv_min_final_observations"]},
        "protocol": p,
    }


def run_switch(cfg: Config, scene_config: SceneConfig, seeds=PROVISIONAL_SEEDS) -> SwitchRun:
    settings = run_settings(cfg, scene_config, seeds)
    settings_hash = config.config_hash({"config_hash": cfg.hash, "settings": settings})
    scene = generate(scene_config, settings["protocol"]["sampling_interval_s"])
    clean_eval = evaluate_ground_truth_cv(scene, cfg)
    clean = ground_truth_grid_tracks(scene)
    for t in clean:
        t.frames.setflags(write=False)
        t.positions.setflags(write=False)
    win = clean_eval.windows
    win.target.setflags(write=False)
    clean_digest, target_digest = tracks_digest(clean), _digest(win.target)
    pop = population(clean, win, clean_eval.per_step_error)

    results = tuple(evaluate_switch(clean, pop, level, s, cfg, scene.frames_per_step)
                    for level in levels(cfg, CONDITION) for s in seeds)
    if tracks_digest(clean) != clean_digest:
        raise InjectionIntegrityError("clean trajectories changed during the sweep")
    if _digest(win.target) != target_digest:
        raise InjectionIntegrityError("clean targets changed during the sweep")
    denominators = {r.summary["achieved"]["eligible_track_frames"] for r in results}
    if len(denominators) != 1:
        raise InjectionIntegrityError(f"pre-injection denominator varies across levels/seeds: {denominators}")
    return SwitchRun(repro.make_experiment_id(NAME, settings_hash, scene_config.seed), settings, settings_hash,
                     cfg.hash, scene.content_hash(), clean_digest, target_digest, denominators.pop(),
                     clean_eval.summary, results, pop)


def results_document(run: SwitchRun) -> dict:
    by_level = [[r for r in run.results if r.level == lv] for lv in sorted({r.level for r in run.results})]
    return {
        "label": LABEL,
        "naming": NAMING_NOTE,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "scene_content_hash": run.scene_content_hash,
        "clean_tracks_sha256": run.clean_tracks_sha256,
        "targets_sha256": run.targets_sha256,
        "eligible_track_frames": run.eligible_track_frames,
        "clean_reference_all_eligible": run.reference,
        "levels": [{"requested": rows[0].summary["requested"], "per_seed": [r.summary for r in rows],
                    "across_seeds": _switch_aggregate(rows)} for rows in by_level],
    }


def windows_csv(run: SwitchRun) -> str:
    win, h = run.population.windows, run.population.windows.history_len
    clean_ade, clean_fde = ade_fde(run.population.clean_per_step) if len(win) else ([], [])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["condition", "level", "seed", "ped_id", "anchor_frame", "status", "lost_reason", "label_used",
                     "mixed_history", "cv_input_changed", "ade_m", "fde_m", "clean_ade_m", "clean_fde_m"])
    for r in run.results:
        sw = r.windows
        position = {w: k for k, w in enumerate(np.flatnonzero(sw.predicted))}
        for w in range(len(win)):
            row = [CONDITION, repr(r.level), r.seed, int(win.track_ids[w]), int(win.frames[w, h - 1]),
                   "predicted" if sw.predicted[w] else "lost", sw.reason[w] or "", int(sw.label[w]),
                   int(sw.mixed[w]), int(sw.cv_changed[w])]
            if w in position:
                k = position[w]
                row += [repr(float(r.per_step[k].mean())), repr(float(r.per_step[k, -1]))]
            else:
                row += ["", ""]
            row += [repr(float(clean_ade[w])), repr(float(clean_fde[w]))]
            writer.writerow(row)
    return buffer.getvalue()


def check_output_location(out_dir: Path) -> None:
    """Refuse any location inside the repository except the Git-ignored runs/."""
    out, repo = Path(out_dir).resolve(), config.REPO_ROOT.resolve()
    if out.is_relative_to(repo) and not out.is_relative_to(repo / "runs"):
        raise ValueError(f"{out} is inside the repository; outputs go outside it or under runs/")


def save_run(out_dir: Path, run: SwitchRun, cfg: Config) -> Path:
    """Write the run to a new directory `out_dir`; raise FileExistsError if it exists."""
    out = Path(out_dir)
    check_output_location(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    files = {
        "results.json": _json(results_document(run)),
        "windows.csv": windows_csv(run),
        "config.resolved.yaml": yaml.safe_dump(cfg.canonical_dict(), sort_keys=True, allow_unicode=True),
    }
    out.mkdir(parents=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "experiment_id": run.experiment_id,
        "experiment_name": NAME,
        "label": LABEL,
        "scene_seed": run.settings["scene_seed"],
        "n_frames": run.settings["scene"]["n_frames"],
        "injection_seeds": run.settings["injection_seeds"],
        "condition": run.settings["condition"],
        "d_max_m": run.settings["d_max_m"],
        "d_cooldown": run.settings["d_cooldown"],
        "eligible_track_frames": run.eligible_track_frames,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "scene_content_hash": run.scene_content_hash,
        "clean_tracks_sha256": run.clean_tracks_sha256,
        "targets_sha256": run.targets_sha256,
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "files": {name: file_sha256(out / name) for name in files},
    }
    (out / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    return out
