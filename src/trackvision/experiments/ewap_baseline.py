"""Trajectory-only constant-velocity baseline on the real ETH and HOTEL ground truth (EWAP), one scene
per run (R1).

TRAJECTORY-ONLY and LOCAL ONLY. CV is applied to the dataset's own annotated world trajectories; no
video is read, and nothing here evaluates a detector or a tracker. The dataset licences are unresolved
(Phase 0B, G3), so runs are saved outside the repository (or under the Git-ignored runs/) and are not
reported or published. Not a v4.15 section 11.3 record, not gate evidence, no inference: stride-1
windows overlap and windows of one pedestrian are correlated, so each scene summary is a descriptive
value for one scene unit. No leave-one-scene-out fold is involved: CV has no trained parameters, so
each scene is evaluated whole and ETH and HOTEL are never pooled.

Annotations are used unchanged: positions are obsmat pos_x, pos_y in metres and frames are the obsmat
frame numbers. Windows come from U2's `evaluate_cv_on_tracks` (predict.windowing with the configured
history_len, horizon_len and stride) on the scene's own annotation step; they never cross a gap or a
pedestrian. Every eligible window is predicted, so coverage is 1.0 by construction. `ewap.load_scene`
checks that the documented timestep equals the configured sampling interval and, with a manifest, the
SHA-256 of every file it reads. It also loads H.txt and map.png, which this evaluation does not use.

Timing. The ETH annotation-step / video-container discrepancy is recorded, not resolved (TIMING_NOTES).
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
from trackvision.experiments.geometry_validation import LOCAL_NOTICE
from trackvision.experiments.synthetic_feasibility import CvEvaluation, evaluate_cv_on_tracks, protocol
from trackvision.io import ewap
from trackvision.io.manifest import file_sha256
from trackvision.predict.windowing import consecutive_runs

NAME = "ewap-cv-baseline"
EVALUATION_TYPE = "trajectory_only"
LABEL = ("Trajectory-only CV baseline on EWAP ground-truth world trajectories (R1). No detector or tracker "
         "is evaluated and no video is read. LOCAL ONLY: dataset licences are unresolved. Not a v4.15 "
         "section 11.3 record; not gate evidence; descriptive only (overlapping, correlated windows). "
         "v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
TIMING_NOTES = {
    "ETH": ("UNRESOLVED, not corrected. obsmat.txt annotations are 6 video frames apart and info.txt documents "
            "a 0.4 s timestep (implying 15 fps), while the seq_eth.avi container declares 25 fps (Phase 0B, "
            "ffprobe), at which 6 frames would be 0.24 s. This run uses the documented timestep and the "
            "original annotation frame numbers; the video is not read and no timestamp is altered."),
    "HOTEL": ("Consistent: 10 video frames per annotation step at the 25 fps declared by the seq_hotel.avi "
              "container (Phase 0B, ffprobe) equals the documented 0.4 s timestep. The video is not read."),
}
COVERAGE_NOTE = ("Every eligible ground-truth window is predicted by CV (no tracker, no missing history), so "
                 "coverage is 1.0 by construction; undefined when there are no windows.")
NOT_USED = {
    "homography": "H.txt is loaded and checksum-verified by ewap.load_scene but not used: positions are already world metres",
    "obstacle_map": "map.png is loaded and checksum-verified by ewap.load_scene but not used",
    "video": "not read",
    "obsmat_velocities": "not used; CV derives velocity from the last two observed positions",
}


@dataclass(frozen=True)
class BaselineRun:
    experiment_id: str
    scene_name: str
    settings: dict
    settings_hash: str
    config_hash: str
    scene: ewap.Scene
    evaluation: CvEvaluation


def _distribution(values) -> dict | None:
    """Descriptive quantiles (numpy's default linear interpolation); None when empty."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None
    p25, median, p75, p90 = np.percentile(v, [25, 50, 75, 90])
    return {"n": int(len(v)), "min": float(v.min()), "p25": float(p25), "median": float(median),
            "p75": float(p75), "p90": float(p90), "max": float(v.max()), "mean": float(v.mean())}


def track_lengths(tracks, frame_step: int, sampling_interval_s: float, history_len: int, window_len: int) -> dict:
    """Track lengths in annotation samples, consecutive runs and time span; bins follow the protocol."""
    samples = np.array([len(t.frames) for t in tracks], dtype=np.int64)
    runs = [consecutive_runs(t.frames, frame_step) for t in tracks]
    longest = np.array([max((b - a for a, b in r), default=0) for r in runs], dtype=np.int64)
    span_s = np.array([(t.frames[-1] - t.frames[0]) / frame_step * sampling_interval_s for t in tracks])
    edges = [1, history_len, window_len, 2 * window_len, 4 * window_len]
    labels = [f"[{a},{b})" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]}"]
    bins = np.digitize(samples, edges[1:])
    return {
        "unit": "annotation samples (one per documented timestep)",
        "samples": _distribution(samples),
        "longest_consecutive_run": _distribution(longest),
        "span_s": _distribution(span_s),
        "histogram_samples": {label: int(np.sum(bins == i)) for i, label in enumerate(labels)},
        "tracks_with_internal_gaps": int(sum(1 for r in runs if len(r) > 1)),
        "tracks_with_a_window_length_run": int(np.sum(longest >= window_len)),
        "window_length_samples": window_len,
    }


def per_pedestrian(ev: CvEvaluation, tracks, frame_step: int) -> list[dict]:
    rows = []
    for t in tracks:
        sel = ev.windows.track_ids == t.ped_id
        n = int(np.sum(sel))
        rows.append({"ped_id": int(t.ped_id), "samples": int(len(t.frames)),
                     "runs": len(consecutive_runs(t.frames, frame_step)), "windows": n,
                     "mean_ade_m": float(ev.ade[sel].mean()) if n else None,
                     "mean_fde_m": float(ev.fde[sel].mean()) if n else None})
    return rows


def run_settings(cfg: Config, scene: ewap.Scene, source_sha256: dict) -> dict:
    p = protocol(cfg)
    return {
        "dataset": ewap.DATASET,
        "scene": scene.name,
        "evaluation_type": EVALUATION_TYPE,
        "base": "dataset world positions (obsmat pos_x, pos_y, metres) and frame numbers, unchanged",
        "predictor": {"name": "CV", "cv_min_final_observations": p["cv_min_final_observations"]},
        "protocol": p,
        "frames_per_step": scene.frame_step,
        "split": "none: CV has no trained parameters; the whole scene is evaluated on its own",
        "dataset_manifest_hash": scene.provenance["dataset_manifest_hash"],
        "manifest_verified": scene.provenance["manifest_verified"],
        "source_files": dict(scene.provenance["source_files"]),
        "source_sha256": source_sha256,
    }


def run_baseline(cfg: Config, scene_name: str, raw_root: Path = ewap.DEFAULT_RAW_ROOT,
                 manifest_path: Path | None = ewap.DEFAULT_MANIFEST) -> BaselineRun:
    """CV on one EWAP scene's ground-truth trajectories. The source files are never modified."""
    raw_root = Path(raw_root)
    scene = ewap.load_scene(scene_name, cfg, raw_root, manifest_path)
    source = {rel: file_sha256(raw_root / rel) for rel in scene.provenance["source_files"].values()}
    if scene.provenance["manifest_verified"] and source != scene.provenance["source_sha256"]:
        raise RuntimeError(f"{scene_name}: source files changed after manifest verification")
    for t in scene.tracks:
        for a in (t.frames, t.positions, t.velocities):
            a.setflags(write=False)
    ev = evaluate_cv_on_tracks(scene.tracks, scene.frame_step, cfg, scene_name=scene.name)
    settings = run_settings(cfg, scene, source)
    settings_hash = config.config_hash({"config_hash": cfg.hash, "settings": settings})
    return BaselineRun(repro.make_experiment_id(f"{NAME}-{scene_name.lower()}", settings_hash, None), scene_name,
                       settings, settings_hash, cfg.hash, scene, ev)


def results_document(run: BaselineRun) -> dict:
    ev, scene, p = run.evaluation, run.scene, run.settings["protocol"]
    s = ev.summary
    n = s["n_windows"]
    diag = scene.diagnostics
    return {
        "label": LABEL,
        "notice": LOCAL_NOTICE,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "scene": run.scene_name,
        "evaluation_type": EVALUATION_TYPE,
        "summary": {
            "n_tracks": s["n_tracks"],
            "n_windows": n,
            "n_pedestrians_contributing": s["n_tracks_contributing"],
            "coverage": {"eligible_windows": n, "predicted_windows": n, "prediction_coverage": s["coverage"],
                         "note": COVERAGE_NOTE},
            "mean_ade_m": s["mean_ade_m"],
            "mean_fde_m": s["mean_fde_m"],
            "ade_m": _distribution(ev.ade),
            "fde_m": _distribution(ev.fde),
            "per_pedestrian_mean_ade_m": s["per_track_mean_ade_m"],
            "per_pedestrian_mean_fde_m": s["per_track_mean_fde_m"],
            "mean_error_by_step_m": s["mean_error_by_step_m"],
            "median_error_by_step_m": np.median(ev.per_step_error, axis=0).tolist() if n else None,
            "note": "descriptive values for one scene unit; overlapping stride-1 windows; no inference",
        },
        "track_lengths": track_lengths(scene.tracks, scene.frame_step, scene.sampling_interval_s,
                                       p["history_len"], p["history_len"] + p["horizon_len"]),
        "per_pedestrian": per_pedestrian(ev, scene.tracks, scene.frame_step),
        "timing": {
            "documented_timestep_s": diag["documented_timestep_s"],
            "configured_sampling_interval_s": p["sampling_interval_s"],
            "frames_per_step": diag["frames_per_step"],
            "implied_video_fps": diag["implied_video_fps"],
            "velocity_timestep": diag["velocity_timestep"],
            "frame_grid_phases": diag["frame_grid_phases"],
            "timestamps_used": "original annotation frame numbers and the documented timestep; the video is not read",
            "note": TIMING_NOTES[run.scene_name],
        },
        "assumptions": dict(scene.assumptions),
        "not_used": NOT_USED,
    }


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _json(data) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_native) + "\n"


def windows_csv(run: BaselineRun) -> str:
    """One row per window: frames and errors only (no dataset positions are copied out)."""
    ev = run.evaluation
    h = ev.windows.history_len
    header = (["ped_id", "first_frame", "anchor_frame", "last_frame", "ade_m", "fde_m"]
              + [f"de_step{t}_m" for t in range(1, ev.windows.horizon_len + 1)])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for w in range(len(ev.windows)):
        frames = ev.windows.frames[w]
        writer.writerow([int(ev.windows.track_ids[w]), int(frames[0]), int(frames[h - 1]), int(frames[-1]),
                         repr(float(ev.ade[w])), repr(float(ev.fde[w]))]
                        + [repr(float(v)) for v in ev.per_step_error[w]])
    return buffer.getvalue()


def check_output_location(out_dir: Path) -> None:
    """Local-only results: refuse any location inside the repository except the Git-ignored runs/."""
    out, repo = Path(out_dir).resolve(), config.REPO_ROOT.resolve()
    if out.is_relative_to(repo) and not out.is_relative_to(repo / "runs"):
        raise ValueError(f"{out} is inside the repository; local-only results go outside it or under runs/")


def save_run(out_dir: Path, run: BaselineRun, cfg: Config) -> Path:
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
        "notice": LOCAL_NOTICE,
        "evaluation_type": EVALUATION_TYPE,
        "dataset": ewap.DATASET,
        "scene": run.scene_name,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "dataset_manifest_hash": run.settings["dataset_manifest_hash"],
        "manifest_verified": run.settings["manifest_verified"],
        "source_sha256": run.settings["source_sha256"],
        "timing_note": TIMING_NOTES[run.scene_name],
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
    }
    (out / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    return out
