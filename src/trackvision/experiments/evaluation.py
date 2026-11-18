"""Predictor-agnostic evaluation of clean and corrupted observed histories.

`corrupt` applies one condition at one level and seed by calling the accepted U7 and U8 injectors and window
builders unchanged, on a fresh copy of the clean tracks. `score` evaluates any predictor on the resulting inputs
against the clean targets. A predictor is only a name and a function from observed histories (P, H, 2) to
positions (P, F, 2); CV, CA, LSTM and GRU keep their own definitions.

Coverage and accuracy are separate fields. ADE, FDE and RMSE are conditional on the predicted windows, and
coverage says how many that was; when none were predicted the metrics are None with a reason, never a number.
Both clean references are kept for each cell: all eligible windows (the primary ADE0, decision 0004) is the clean
cell of the same population, and `clean_same_windows` is the paired sensitivity check.

Cells are per-seed summaries over overlapping, correlated windows: descriptive values, not independent samples.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from trackvision.config import Config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments import switch_feasibility as sf
from trackvision.inject.identity_switch import identity_switch
from trackvision.inject.trajectory import dropout_masks, fragment, localization_noise
from trackvision.metrics.displacement import ade_fde, displacement_errors, rmse
from trackvision.predict import learned
from trackvision.predict.baselines import constant_acceleration, constant_velocity
from trackvision.predict.windowing import Windows

CONDITIONS = ("A", "B1", "B2", "C", "D")
MEASURE = {**inj.PRIMARY_ACHIEVED, "D": "switch_event_rate"}


@dataclass(frozen=True)
class Predictor:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]


def cv_predictor(cfg: Config, horizon: int) -> Predictor:
    minimum = cfg.frozen["predictors"]["cv_min_final_observations"]
    return Predictor("CV", lambda x: constant_velocity(x, horizon, minimum))


def ca_predictor(cfg: Config, horizon: int) -> Predictor:
    minimum = cfg.frozen["predictors"]["ca_min_final_observations"]
    return Predictor("CA", lambda x: constant_acceleration(x, horizon, minimum))


def learned_predictor(model: learned.DisplacementPredictor) -> Predictor:
    return Predictor(model.cell, lambda x: learned.predict_positions(model, x))


@dataclass(frozen=True)
class Corrupted:
    condition: str
    level: float
    seed: int
    predicted: np.ndarray            # (N,) bool
    reason: tuple                    # (N,) str | None
    inputs: np.ndarray               # (P, H, 2)
    achieved: dict                   # the injector's full achieved-severity report
    measure: str
    value: float | None              # the primary achieved severity, over the whole clean base
    extra: dict


def clean_inputs(windows: Windows) -> Corrupted:
    """The clean reference: every eligible window, observed history unchanged."""
    n = len(windows)
    return Corrupted("clean", 0.0, 0, np.ones(n, dtype=bool), (None,) * n, windows.observed, {}, "", None, {})


def scene_population(tracks, windows: Windows) -> inj.Population:
    """U7's window indexing. Its CV reference errors are not used by this evaluator."""
    return inj.population(tracks, windows, np.zeros((len(windows), windows.horizon_len)))


def corrupt(tracks, pop: inj.Population, condition: str, level: float, seed: int, cfg: Config,
            frame_step: int) -> Corrupted:
    """One condition, level and seed on a fresh copy of the clean tracks. Targets are never touched."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    win, n, work = pop.windows, len(pop.windows), inj.fresh_copy(tracks)
    extra: dict = {}
    if condition == "A":
        noise = localization_noise(work, level, seed)
        achieved = noise.achieved
        wi = inj.WindowInputs(np.ones(n, dtype=bool), (None,) * n, np.zeros(n, dtype=np.int64), np.array(
            [noise.tracks[pop.track_index[int(g)]].positions[s] for g, s in zip(win.track_ids, pop.history_samples)]
        ).reshape(-1, win.history_len, 2), {})
    elif condition in ("B1", "B2"):
        drop = dropout_masks(work, level, seed)
        achieved = drop.achieved
        wi = inj.dropout_windows(work, pop, drop.present, reconstruct=condition == "B1")
        extra = dict(wi.extra)
    elif condition == "C":
        frag = fragment(work, level, seed)
        achieved = frag.achieved
        wi = inj.fragment_windows(pop, frag)
    else:
        sf.check_cooldown(cfg)
        p = cfg.frozen["predictors"]
        sw = identity_switch(work, level, seed, float(cfg.frozen["injection"]["d_max_m"]), frame_step)
        achieved = sw.achieved
        windows = sf.switch_windows(pop, sw.tracks, sw.occupant, p["cv_min_final_observations"])
        idx = np.flatnonzero(windows.predicted)
        changed = {}
        for name, k in (("CV", p["cv_min_final_observations"]), ("CA", p["ca_min_final_observations"]),
                        ("learned", win.history_len)):
            differs = np.any(windows.inputs[:, -k:] != win.observed[idx][:, -k:], axis=(1, 2))
            changed[name] = int(differs.sum())
        extra = {"windows_with_mixed_history": int(windows.mixed.sum()), "windows_with_changed_input": changed}
        wi = inj.WindowInputs(windows.predicted, windows.reason, np.zeros(n, dtype=np.int64), windows.inputs, {})
    return Corrupted(condition, float(level), int(seed), wi.predicted, wi.reason, wi.inputs, achieved,
                     MEASURE[condition], achieved[MEASURE[condition]], extra)


def clean_errors(predictor: Predictor, windows: Windows) -> np.ndarray:
    """Per-step clean errors (N, F) on every eligible window."""
    if len(windows) == 0:
        return np.empty((0, windows.horizon_len))
    return displacement_errors(predictor.predict(windows.observed), windows.target)


def _means(per_step: np.ndarray) -> dict:
    ade, fde = ade_fde(per_step)
    return {"ade_m": float(ade.mean()), "fde_m": float(fde.mean()), "rmse_m": float(rmse(per_step).mean()),
            "de_by_step_m": per_step.mean(axis=0).tolist()}


def score(predictor: Predictor, corrupted: Corrupted, windows: Windows, clean_per_step: np.ndarray) -> dict:
    """Coverage and conditional accuracy of one predictor on one set of inputs, against clean targets."""
    n, idx = len(corrupted.predicted), np.flatnonzero(corrupted.predicted)
    out = {"eligible_windows": n, "predicted_windows": int(len(idx)),
           "prediction_coverage": float(len(idx) / n) if n else None,
           "lost_by_reason": dict(sorted(Counter(r for r in corrupted.reason if r is not None).items()))}
    if len(idx) == 0:
        reason = "no_eligible_windows" if n == 0 else "no_predicted_windows"
        return {**out, "ade_m": None, "fde_m": None, "rmse_m": None, "de_by_step_m": None, "undefined_reason": reason,
                "clean_same_windows": None}
    per_step = displacement_errors(predictor.predict(corrupted.inputs), windows.target[idx])
    return {**out, **_means(per_step), "undefined_reason": None, "clean_same_windows": _means(clean_per_step[idx])}
