"""Feasibility sweep of controlled injection (A, B1, B2, C) on clean synthetic ground truth, with CV
prediction, coverage and saved results.

FEASIBILITY ONLY: not a v4.15 section 11.3 record, not gate evidence, no hypothesis test. D, E,
severity normalization, D(s), AUDC and other predictors are not part of it. Injection seeds (0, 1, 2
by default) are feasibility values, not the research choice `seeds` (decision 0001 B).

Population. The eligible windows are U2's clean ground-truth windows (configured history_len,
horizon_len and stride); they are the same for every condition, level and seed. Corruption applies to
the observed history only: targets always come from the clean windows (v4.15 section 5.2), and their
hash is checked unchanged after the sweep. Every injection starts from a fresh copy of the clean
trajectories, whose hash is also checked unchanged. Each trajectory is corrupted once per
(condition, level, seed), so overlapping windows share corrupted observations.

History validity (v4.15 section 3.2) and the CV input, per condition:
  A   always valid (structure preserved)            noisy positions
  B2  all history observations survive              surviving positions; never shortened
  B1  first and last history points survive         per-window interpolation of interior gaps
                                                    (OFFLINE, NON-CAUSAL; never the target period)
  C   all history samples carry one fragment id     fragment positions (unchanged)
Unpredicted windows reduce coverage. Requested and achieved severity are reported separately;
achieved severity uses the whole clean base, with predictor-visible counts (distinct observations in
eligible histories) as a supplement for A and B. Accuracy on predicted windows is paired with clean
CV on the same windows, and clean CV on all eligible windows is reported for survivorship.
Overlapping windows are correlated: descriptive summaries only.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments.synthetic_feasibility import evaluate_ground_truth_cv, protocol
from trackvision.inject.trajectory import dropout_masks, fragment, localization_noise, reconstruct_history
from trackvision.io.manifest import file_sha256
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

NAME = "synthetic-injection-feasibility"
CONDITIONS = ("A", "B1", "B2", "C")
PROVISIONAL_SEEDS = (0, 1, 2)
PRIMARY_ACHIEVED = {"A": "rmse_pos_m", "B1": "rate", "B2": "rate", "C": "fragmentation_rate"}
LOST_REASON = {"B1": "boundary_gap", "B2": "missing_observation", "C": "split_in_history"}
LABEL = ("Feasibility injection sweep on clean synthetic ground truth (controlled proxies). Not a v4.15 "
         "section 11.3 record; not gate evidence; descriptive only. Injection seeds are provisional "
         "feasibility values, not the research seed decision. v4.15 Phase 1 remains BLOCKED and v4.16 "
         "is unfrozen.")
B1_NOTE = ("B1 is an OFFLINE, NON-CAUSAL reconstruction: linear interpolation of interior gaps within each "
           "window's own observed history, using later history points but never the target period.")
B_COVERAGE_NOTE = ("prediction_coverage_pre applies the B2 rule (no reconstruction) and _post the B1 rule, to "
                   "the same dropout masks; for B2 they are identical by definition (v4.15 section 3.6).")


class InjectionIntegrityError(RuntimeError):
    """Raised if clean trajectories or targets change during the sweep."""


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}:{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tracks_digest(tracks) -> str:
    return _digest(*[x for t in tracks for x in (np.array([t.ped_id]), t.frames, t.positions)])


def fresh_copy(tracks) -> tuple[GridTrack, ...]:
    return tuple(GridTrack(int(t.ped_id), t.frames.copy(), t.positions.copy()) for t in tracks)


def _stats(values) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def _accuracy(per_step: np.ndarray) -> dict:
    n = len(per_step)
    ade, fde = ade_fde(per_step) if n else (np.empty(0), np.empty(0))
    return {"n_windows": int(n), "mean_ade_m": float(ade.mean()) if n else None,
            "mean_fde_m": float(fde.mean()) if n else None,
            "mean_error_by_step_m": per_step.mean(axis=0).tolist() if n else None}


def levels(cfg: Config, condition: str) -> list[float]:
    grid = cfg.frozen["injection"]["grids"][cfg.frozen["injection"]["conditions"][condition]["grid"]]
    return [float(v) for v in grid["levels"]]


def requested(cfg: Config, condition: str, level: float) -> dict:
    grid = cfg.frozen["injection"]["grids"][cfg.frozen["injection"]["conditions"][condition]["grid"]]
    return {"parameter": grid["parameter"], "unit": grid["unit"], "value": level}


@dataclass(frozen=True)
class Population:
    """Eligible windows, as sample indices into the clean tracks."""
    windows: object                  # U2 Windows (clean): track_ids, frames, observed, target
    clean_per_step: np.ndarray       # (N, horizon_len) clean CV errors
    track_index: dict                # ped_id -> position in the clean track tuple
    history_samples: np.ndarray      # (N, history_len) sample indices into each window's clean track
    visible: dict                    # ped_id -> sorted distinct sample indices seen in eligible histories


def population(clean_tracks, windows, clean_per_step) -> Population:
    h = windows.history_len
    index = {int(t.ped_id): i for i, t in enumerate(clean_tracks)}
    samples = np.zeros((len(windows), h), dtype=np.int64)
    visible: dict[int, set] = {}
    for w, (g, frames) in enumerate(zip(windows.track_ids, windows.frames)):
        t = clean_tracks[index[int(g)]]
        s = np.searchsorted(t.frames, frames[:h])
        if not np.array_equal(t.frames[s], frames[:h]):
            raise ValueError(f"window {w} frames are not samples of track {g}")
        samples[w] = s
        visible.setdefault(int(g), set()).update(s.tolist())
    return Population(windows, clean_per_step, index, samples,
                      {g: np.array(sorted(v), dtype=np.int64) for g, v in visible.items()})


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    level: float
    seed: int
    predicted: np.ndarray            # (N,) bool
    reason: tuple                    # (N,) str | None
    reconstructed: np.ndarray        # (N,) interior points filled (B1 only, else 0)
    inputs: np.ndarray               # (P, history_len, 2)
    per_step: np.ndarray             # (P, horizon_len)
    summary: dict


@dataclass(frozen=True)
class WindowInputs:
    predicted: np.ndarray            # (N,) bool
    reason: tuple                    # (N,) str | None
    reconstructed: np.ndarray        # (N,) int
    inputs: np.ndarray               # (P, history_len, 2) for predicted windows, in window order
    extra: dict


def dropout_windows(tracks, pop: Population, present, reconstruct: bool) -> WindowInputs:
    """B2 (reconstruct=False) or B1 (True) for every eligible window, given one survival mask per track.

    Each window sees only its own history: missing points enter as NaN, and B1 interpolation runs on
    that history alone (`reconstruct_history`), so the target period is never used."""
    win, n = pop.windows, len(pop.windows)
    predicted = np.zeros(n, dtype=bool)
    reason: list = [None] * n
    reconstructed = np.zeros(n, dtype=np.int64)
    inputs, pre, boundary, distinct = [], 0, 0, set()
    for w, (g, s) in enumerate(zip(win.track_ids, pop.history_samples)):
        i = pop.track_index[int(g)]
        alive = np.asarray(present[i], dtype=bool)[s]
        history = np.where(alive[:, None], tracks[i].positions[s], np.nan)
        pre += bool(alive.all())
        if not reconstruct:
            if alive.all():
                predicted[w] = True
                inputs.append(history)
            else:
                reason[w] = LOST_REASON["B2"]
            continue
        rec = reconstruct_history(history, alive)
        boundary += rec.boundary_gap
        if rec.positions is None:
            reason[w] = LOST_REASON["B1"]
        else:
            predicted[w] = True
            reconstructed[w] = rec.reconstructed
            distinct.update((int(g), int(k)) for k in s[~alive])
            inputs.append(rec.positions)
    extra = {"prediction_coverage_pre": pre / n if n else None,
             "prediction_coverage_post": float(predicted.sum() / n) if n else None,
             "coverage_note": B_COVERAGE_NOTE}
    if reconstruct:
        extra["reconstruction"] = {
            "note": B1_NOTE,
            "windows_with_reconstruction": int(np.sum(reconstructed > 0)),
            "interior_reconstructed_window_observations": int(reconstructed.sum()),
            "distinct_observations_reconstructed": len(distinct),
            "windows_with_unresolved_boundary_gap": int(boundary),
        }
    h = win.history_len
    return WindowInputs(predicted, tuple(reason), reconstructed,
                        np.asarray(inputs, dtype=np.float64).reshape(-1, h, 2), extra)


def fragment_windows(pop: Population, frag) -> WindowInputs:
    """C for every eligible window: predicted only if its whole history lies in one fragment."""
    win, n = pop.windows, len(pop.windows)
    tracks = {int(t.ped_id): t for t in frag.tracks}
    label = {}                                   # (ped_id, sample) -> (fragment id, index in fragment)
    for fid, (g, a, b) in frag.origin.items():
        for k in range(a, b):
            label[(g, k)] = (fid, k - a)
    predicted = np.zeros(n, dtype=bool)
    reason: list = [None] * n
    inputs = []
    for w, (g, s) in enumerate(zip(win.track_ids, pop.history_samples)):
        hits = [label[(int(g), int(k))] for k in s]
        ids = {fid for fid, _ in hits}
        if len(ids) == 1:
            predicted[w] = True
            inputs.append(tracks[ids.pop()].positions[[j for _, j in hits]])
        else:
            reason[w] = LOST_REASON["C"]
    return WindowInputs(predicted, tuple(reason), np.zeros(n, dtype=np.int64),
                        np.asarray(inputs, dtype=np.float64).reshape(-1, win.history_len, 2), {})


def evaluate_condition(clean_tracks, pop: Population, condition: str, level: float, seed: int,
                       cfg: Config) -> ConditionResult:
    p = protocol(cfg)
    h, horizon = p["history_len"], p["horizon_len"]
    win = pop.windows
    n = len(win)
    work = fresh_copy(clean_tracks)
    supplement = None
    if condition == "A":
        inj = localization_noise(work, level, seed)
        achieved = inj.achieved
        wi = WindowInputs(np.ones(n, dtype=bool), (None,) * n, np.zeros(n, dtype=np.int64),
                          np.array([inj.tracks[pop.track_index[int(g)]].positions[s]
                                    for g, s in zip(win.track_ids, pop.history_samples)]).reshape(-1, h, 2), {})
        sq = np.concatenate([np.sum(inj.offsets[pop.track_index[g]][s] ** 2, axis=1)
                             for g, s in pop.visible.items()] or [np.empty(0)])
        supplement = {"rmse_pos_m": float(np.sqrt(sq.mean())) if len(sq) else None, "n_observations": int(len(sq))}
    elif condition in ("B1", "B2"):
        drop = dropout_masks(work, level, seed)
        achieved = drop.achieved
        wi = dropout_windows(work, pop, drop.present, reconstruct=condition == "B1")
        removed = sum(int(np.sum(~drop.present[pop.track_index[g]][s])) for g, s in pop.visible.items())
        seen = sum(len(s) for s in pop.visible.values())
        supplement = {"removed_observations": removed, "eligible_observations": seen,
                      "rate": removed / seen if seen else None}
    elif condition == "C":
        frag = fragment(work, level, seed)
        achieved = frag.achieved
        wi = fragment_windows(pop, frag)
    else:
        raise ValueError(f"unknown condition {condition!r}; U7 implements {CONDITIONS}")

    predicted, reason, reconstructed, inputs, extra = wi.predicted, wi.reason, wi.reconstructed, wi.inputs, wi.extra
    idx = np.flatnonzero(predicted)
    per_step = displacement_errors(constant_velocity(inputs, horizon, p["cv_min_final_observations"]),
                                   win.target[idx])
    injected, clean_same = _accuracy(per_step), _accuracy(pop.clean_per_step[idx])
    summary = {
        "condition": condition,
        "seed": int(seed),
        "requested": requested(cfg, condition, level),
        "achieved": achieved,
        "achieved_primary": {"name": PRIMARY_ACHIEVED[condition], "value": achieved[PRIMARY_ACHIEVED[condition]]},
        "achieved_predictor_visible": supplement,
        "coverage": {"eligible_windows": int(n), "predicted_windows": int(len(idx)),
                     "prediction_coverage": float(len(idx) / n) if n else None, "lost_windows": int(n - len(idx)),
                     "lost_by_reason": {r: sum(1 for x in reason if x == r) for r in sorted(set(LOST_REASON.values()))}},
        "accuracy_on_predicted_windows": {
            "injected": injected,
            "clean_same_windows": clean_same,
            "mean_ade_difference_injected_minus_clean_m": (
                injected["mean_ade_m"] - clean_same["mean_ade_m"] if injected["mean_ade_m"] is not None else None),
            "mean_fde_difference_injected_minus_clean_m": (
                injected["mean_fde_m"] - clean_same["mean_fde_m"] if injected["mean_fde_m"] is not None else None),
        },
        **extra,
    }
    return ConditionResult(condition, float(level), int(seed), predicted, tuple(reason), reconstructed, inputs,
                           per_step, summary)


def _seed_aggregate(results) -> dict:
    def collect(get):
        return [v for v in (get(r.summary) for r in results) if v is not None]
    acc = lambda key, metric: (lambda s: s["accuracy_on_predicted_windows"][key][metric])
    return {
        "n_seeds": len(results),
        "seeds": [r.seed for r in results],
        "achieved_primary": _stats(collect(lambda s: s["achieved_primary"]["value"])),
        "prediction_coverage": _stats(collect(lambda s: s["coverage"]["prediction_coverage"])),
        "injected_mean_ade_m": _stats(collect(acc("injected", "mean_ade_m"))),
        "injected_mean_fde_m": _stats(collect(acc("injected", "mean_fde_m"))),
        "clean_same_windows_mean_ade_m": _stats(collect(acc("clean_same_windows", "mean_ade_m"))),
        "n_seeds_with_predictions": len(collect(acc("injected", "mean_ade_m"))),
        "note": "mean, min and max across injection seeds; descriptive, not a confidence interval.",
    }


@dataclass(frozen=True)
class InjectionRun:
    experiment_id: str
    settings: dict
    settings_hash: str
    config_hash: str
    scene_content_hash: str
    clean_tracks_sha256: str
    targets_sha256: str
    reference: dict                  # U2 clean CV summary on all eligible windows
    results: tuple                   # ConditionResult, ordered by condition, level, seed
    population: Population


def run_settings(cfg: Config, scene_config: SceneConfig, seeds) -> dict:
    p = protocol(cfg)
    return {
        "scene": scene_config.as_dict(),
        "scene_seed": scene_config.seed,
        "injection_seeds": [int(s) for s in seeds],
        "injection_seeds_status": "provisional feasibility values; not the research choice 'seeds'",
        "conditions": {c: {"grid": cfg.frozen["injection"]["conditions"][c]["grid"], "levels": levels(cfg, c)}
                       for c in CONDITIONS},
        "base": "clean U1 ground-truth grid trajectories (whole scene)",
        "random_numbers": "SeedSequence([seed, family, ped_id]); shared across levels (nested masks, common draws); "
                          "B1 and B2 share dropout masks",
        "fragmentation_convention": "split after sample k (1 <= k <= n - 2); new identity from k + 1",
        "b1_reconstruction": B1_NOTE,
        "predictor": {"name": "CV", "cv_min_final_observations": p["cv_min_final_observations"]},
        "protocol": p,
    }


def run_injection(cfg: Config, scene_config: SceneConfig, seeds=PROVISIONAL_SEEDS) -> InjectionRun:
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

    results = tuple(evaluate_condition(clean, pop, c, level, s, cfg)
                    for c in CONDITIONS for level in levels(cfg, c) for s in seeds)
    if tracks_digest(clean) != clean_digest:
        raise InjectionIntegrityError("clean trajectories changed during the sweep")
    if _digest(win.target) != target_digest:
        raise InjectionIntegrityError("clean targets changed during the sweep")
    return InjectionRun(repro.make_experiment_id(NAME, settings_hash, scene_config.seed), settings, settings_hash,
                        cfg.hash, scene.content_hash(), clean_digest, target_digest, clean_eval.summary, results, pop)


def results_document(run: InjectionRun) -> dict:
    conditions = {}
    for c in CONDITIONS:
        rows = [r for r in run.results if r.condition == c]
        conditions[c] = [{"requested": rows_l[0].summary["requested"],
                          "per_seed": [r.summary for r in rows_l],
                          "across_seeds": _seed_aggregate(rows_l)}
                         for rows_l in ([r for r in rows if r.level == lv] for lv in sorted({r.level for r in rows}))]
    return {
        "label": LABEL,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "scene_content_hash": run.scene_content_hash,
        "clean_tracks_sha256": run.clean_tracks_sha256,
        "targets_sha256": run.targets_sha256,
        "clean_reference_all_eligible": run.reference,
        "conditions": conditions,
    }


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _json(data) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_native) + "\n"


def windows_csv(run: InjectionRun) -> str:
    win, h = run.population.windows, run.population.windows.history_len
    clean_ade, clean_fde = ade_fde(run.population.clean_per_step) if len(win) else ([], [])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["condition", "level", "seed", "ped_id", "anchor_frame", "status", "lost_reason",
                     "b1_reconstructed_points", "ade_m", "fde_m", "clean_ade_m", "clean_fde_m"])
    for r in run.results:
        position = {w: k for k, w in enumerate(np.flatnonzero(r.predicted))}
        for w in range(len(win)):
            row = [r.condition, repr(r.level), r.seed, int(win.track_ids[w]), int(win.frames[w, h - 1]),
                   "predicted" if r.predicted[w] else "lost", r.reason[w] or "", int(r.reconstructed[w])]
            if w in position:
                k = position[w]
                row += [repr(float(r.per_step[k].mean())), repr(float(r.per_step[k, -1]))]
            else:
                row += ["", ""]
            row += [repr(float(clean_ade[w])), repr(float(clean_fde[w]))]
            writer.writerow(row)
    return buffer.getvalue()


def save_run(out_dir: Path, run: InjectionRun, cfg: Config) -> Path:
    """Write the run to a new directory `out_dir`; raise FileExistsError if it exists."""
    out = Path(out_dir)
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
        "conditions": run.settings["conditions"],
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
