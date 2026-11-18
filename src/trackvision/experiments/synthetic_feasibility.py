"""Feasibility evaluation of the constant-velocity baseline on synthetic ground-truth trajectories.

FEASIBILITY ONLY: no research hypothesis is tested and no gate is passed. Nothing is written to disk.

Windows are cut per pedestrian on the sampling grid (history_len observed + horizon_len future steps,
stride from config), so a window never crosses identities, gaps or scene boundaries. CV sees only the
observed part; targets are ground-truth positions.

Aggregation is descriptive. Stride-1 windows overlap and windows from one pedestrian are correlated,
so the scene summary is one descriptive value for one scene unit, never a set of independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.config import Config
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.predict.windowing import Windows, make_windows
from trackvision.synthetic.scene import SceneGroundTruth
from trackvision.synthetic.tracks import ground_truth_grid_tracks

NOTE = ("Feasibility baseline on synthetic ground truth. Overlapping stride-1 windows from correlated "
        "tracks: descriptive values for one scene unit, not independent samples; no inference.")


@dataclass(frozen=True)
class CvEvaluation:
    windows: Windows
    predicted: np.ndarray        # (N, horizon_len, 2)
    per_step_error: np.ndarray   # (N, horizon_len) metres
    ade: np.ndarray              # (N,)
    fde: np.ndarray              # (N,)
    summary: dict


def protocol(cfg: Config) -> dict:
    """The protocol values U2 uses, all read from the frozen configuration."""
    static = cfg.frozen["protocol"]["static"]
    return {
        "history_len": static["history_len"],
        "horizon_len": static["horizon_len"],
        "window_stride": static["window_stride"],
        "sampling_interval_s": static["sampling_interval_s"],
        "cv_min_final_observations": cfg.frozen["predictors"]["cv_min_final_observations"],
    }


def _stats(values: np.ndarray) -> dict | None:
    if len(values) == 0:
        return None
    return {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())}


def evaluate_cv_on_tracks(tracks, frame_step: int, cfg: Config, scene_name: str = "synthetic") -> CvEvaluation:
    """Window the tracks, predict with CV and score against the windows' ground-truth targets."""
    p = protocol(cfg)
    windows = make_windows(scene_name, tracks, frame_step, p["history_len"], p["horizon_len"], p["window_stride"])
    predicted = constant_velocity(windows.observed, p["horizon_len"], p["cv_min_final_observations"])
    per_step = displacement_errors(predicted, windows.target)
    ade, fde = ade_fde(per_step)

    ids = np.unique(windows.track_ids)
    track_ade = np.array([ade[windows.track_ids == i].mean() for i in ids])
    track_fde = np.array([fde[windows.track_ids == i].mean() for i in ids])
    n = len(windows)
    summary = {
        "note": NOTE,
        "protocol": p,
        "n_tracks": len(tracks),
        "n_tracks_contributing": int(len(ids)),
        "n_windows": n,
        # Every eligible ground-truth window is predicted by CV; undefined when there are none.
        "coverage": 1.0 if n else None,
        "mean_ade_m": float(ade.mean()) if n else None,
        "mean_fde_m": float(fde.mean()) if n else None,
        "per_track_mean_ade_m": _stats(track_ade),
        "per_track_mean_fde_m": _stats(track_fde),
        "mean_error_by_step_m": per_step.mean(axis=0).tolist() if n else None,
    }
    return CvEvaluation(windows, predicted, per_step, ade, fde, summary)


def evaluate_ground_truth_cv(scene: SceneGroundTruth, cfg: Config) -> CvEvaluation:
    """CV on a U1 scene's ground-truth grid tracks; the scene must use the configured interval."""
    expected = protocol(cfg)["sampling_interval_s"]
    if abs(scene.sampling_interval_s - expected) > 1e-12:
        raise ValueError(f"scene sampled at {scene.sampling_interval_s} s, configuration requires {expected} s")
    return evaluate_cv_on_tracks(ground_truth_grid_tracks(scene), scene.frames_per_step, cfg,
                                 scene_name=f"synthetic-seed{scene.config.seed}")
