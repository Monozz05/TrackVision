"""End-to-end synthetic feasibility run: scene -> render -> detect -> track -> CV prediction -> ADE/FDE
and coverage, for every detection source, plus saved results.

FEASIBILITY ONLY. Not a v4.15 section 11.3 experiment record, not gate evidence, and no research
choice (detector, tracker, renderer, motion model) is made by it.

Compared, each reported separately:
  - ground-truth reference: CV on U1 foot points (U2; no detector or tracker; coverage 1 by construction);
  - ground-truth boxes as detections -> tracker (DIAGNOSTIC);
  - detector with the oracle background -> tracker (DIAGNOSTIC);
  - detector with the median-estimated background -> tracker (PRACTICAL BASELINE).
Frames are rendered in memory by the U3 renderer (not decoded from FFV1; U3 checked that round trip
is lossless). A SHA-256 over all frame bytes is recorded.

`save_run` writes a NEW directory chosen by the caller and refuses an existing one:
  results.json          deterministic results (no timestamps)
  windows.csv           one row per eligible ground-truth window per tracker source
  config.resolved.yaml  the merged configuration
  manifest.json         run ID, provenance (timestamp, Git state, environment, spec) and file hashes
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments.detection_feasibility import ESTIMATED, MIN_VISIBLE_FRACTION, ORACLE
from trackvision.experiments.prediction_feasibility import LEVELS, PredictionEvaluation, evaluate_prediction_on_scene
from trackvision.experiments.synthetic_feasibility import evaluate_ground_truth_cv, protocol
from trackvision.experiments.tracking_feasibility import (GT_BOXES, detector_detections, evaluate_tracking_on_scene,
                                                          ground_truth_detections)
from trackvision.io.manifest import file_sha256
from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import SceneConfig, SceneGroundTruth, generate
from trackvision.tracking.background_subtraction import BgSubConfig, detect, median_background
from trackvision.tracking.tracker import TrackerConfig

NAME = "synthetic-cv-feasibility"
GROUND_TRUTH_REFERENCE = "ground_truth_reference (CV on U1 foot points; no detector or tracker)"
SOURCES = (GT_BOXES, ORACLE, ESTIMATED)
ROLES = {GROUND_TRUTH_REFERENCE: "reference", GT_BOXES: "diagnostic", ORACLE: "diagnostic",
         ESTIMATED: "practical baseline"}
LABEL = ("Feasibility run on one synthetic scene. Not a v4.15 section 11.3 experiment record; not gate "
         "evidence; v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
U5_WINDOWS_NOTE = ("tracking_u5.grid.windows is U5's count of windows with 20 consecutive tracker observations "
                   "(history and future both from the tracker). U6 prediction coverage needs only history_len "
                   "(8) consecutive tracker observations, because the future target comes from ground truth. "
                   "The two numbers measure different things; U5's is reported unchanged.")
TRACKING_COVERAGE_NOTE = ("tp / evaluable ground-truth observations over all frames, from U5's metrics; a "
                          "reading of v4.15 section 3.2 'matched_observations / expected_GT_observations'.")


@dataclass(frozen=True)
class FeasibilityRun:
    experiment_id: str
    settings: dict
    settings_hash: str
    config_hash: str
    scene: SceneGroundTruth
    scene_content_hash: str
    frames_sha256: str
    reference: dict                  # U2 summary on ground truth
    tracking: dict                   # source -> U5 tracking summary
    prediction: dict                 # source -> PredictionEvaluation


def run_settings(cfg: Config, scene_config: SceneConfig, render_config: RenderConfig, det_config: BgSubConfig,
                 tracker_config: TrackerConfig, min_visible_fraction: float) -> dict:
    p = protocol(cfg)
    return {
        "scene": scene_config.as_dict(),
        "seed": scene_config.seed,
        "render": asdict(render_config),
        "frame_source": "rendered in memory by the U3 renderer (not decoded from FFV1)",
        "detector": {"name": "background_subtraction (feasibility)", **asdict(det_config)},
        "tracker": {"name": "IouTracker (feasibility)", **asdict(tracker_config)},
        "predictor": {"name": "CV", "cv_min_final_observations": p["cv_min_final_observations"]},
        "protocol": p,
        "evaluation_iou_threshold": float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"]),
        "min_visible_fraction": min_visible_fraction,
    }


def run_feasibility(cfg: Config, scene_config: SceneConfig, render_config: RenderConfig = RenderConfig(),
                    det_config: BgSubConfig = BgSubConfig(), tracker_config: TrackerConfig = TrackerConfig(),
                    min_visible_fraction: float = MIN_VISIBLE_FRACTION) -> FeasibilityRun:
    settings = run_settings(cfg, scene_config, render_config, det_config, tracker_config, min_visible_fraction)
    settings_hash = config.config_hash({"config_hash": cfg.hash, "settings": settings})
    scene = generate(scene_config, settings["protocol"]["sampling_interval_s"])
    n = scene_config.n_frames

    # One rendering pass: visible fractions, frame digest and oracle-background detections.
    renderer = SceneRenderer(scene, render_config)
    vf = np.zeros(len(scene.frame))
    digest = hashlib.sha256()
    oracle = []
    for rendered in renderer:
        vf[rendered.rows] = rendered.visible_fraction
        digest.update(rendered.image.tobytes())
        oracle.append(detect(rendered.image, renderer.background, det_config).boxes)
    frames = lambda: (renderer.frame(k).image for k in range(n))
    estimated = detector_detections(frames(), median_background(frames(), n, det_config.background_frames),
                                    det_config)
    detections = {GT_BOXES: ground_truth_detections(scene, vf), ORACLE: oracle, ESTIMATED: estimated}

    tracking, prediction = {}, {}
    for source in SOURCES:
        tev = evaluate_tracking_on_scene(scene, detections[source], vf, cfg, source, tracker_config,
                                         min_visible_fraction)
        tracking[source] = tev.summary
        prediction[source] = evaluate_prediction_on_scene(scene, tev.tracker, detections[source], vf, cfg, source,
                                                          min_visible_fraction)
    return FeasibilityRun(
        experiment_id=repro.make_experiment_id(NAME, settings_hash, scene_config.seed), settings=settings,
        settings_hash=settings_hash, config_hash=cfg.hash, scene=scene, scene_content_hash=scene.content_hash(),
        frames_sha256=digest.hexdigest(), reference=evaluate_ground_truth_cv(scene, cfg).summary,
        tracking=tracking, prediction=prediction)


def results_document(run: FeasibilityRun) -> dict:
    sources = {}
    for source in SOURCES:
        m = run.tracking[source]["metrics"]
        sources[source] = {
            "role": ROLES[source],
            "tracking_u5": run.tracking[source],
            "u5_grid_windows_note": U5_WINDOWS_NOTE,
            "tracking_coverage": {"value": m["tp"] / m["evaluable_observations"] if m["evaluable_observations"]
                                  else None, "note": TRACKING_COVERAGE_NOTE},
            "prediction": run.prediction[source].summary,
        }
    return {
        "label": LABEL,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "scene_content_hash": run.scene_content_hash,
        "frames_sha256": run.frames_sha256,
        "roles": ROLES,
        "ground_truth_reference": {"role": ROLES[GROUND_TRUTH_REFERENCE], **run.reference},
        "sources": sources,
    }


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _json(data) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_native) + "\n"


def _num(value) -> str:
    return repr(float(value))


def window_rows(source: str, ev: PredictionEvaluation, history_len: int) -> list[list]:
    horizon = ev.windows.horizon_len
    l0_ade, l0_fde = (ev.l0_all_per_step.mean(axis=1), ev.l0_all_per_step[:, -1]) if len(ev.windows) else ([], [])
    position = {w: k for k, w in enumerate(ev.predicted_index)}
    rows = []
    for w in range(len(ev.windows)):
        frames = ev.windows.frames[w]
        row = [source, int(ev.windows.track_ids[w]), int(frames[history_len - 1]), int(frames[0]),
               "predicted" if ev.predicted[w] else "lost", ev.primary_reason[w] or "", ";".join(ev.cause_flags[w]),
               int(ev.tracker_id[w]) if ev.predicted[w] else "", _num(l0_ade[w]), _num(l0_fde[w])]
        if w in position:
            k = position[w]
            for level in ("L1", "L2"):
                row += [_num(ev.per_step[level][k].mean()), _num(ev.per_step[level][k, -1])]
            row += [_num(ev.history_error[k].mean()), _num(ev.history_error[k, -1])]
            row += [_num(v) for v in ev.per_step["L2"][k]]
        else:
            row += [""] * (6 + horizon)
        rows.append(row)
    return rows


def windows_csv(run: FeasibilityRun) -> str:
    h = run.settings["protocol"]["history_len"]
    horizon = run.settings["protocol"]["horizon_len"]
    header = ["source", "ped_id", "anchor_frame", "first_frame", "status", "primary_reason", "cause_flags",
              "tracker_id", "ade_L0_m", "fde_L0_m", "ade_L1_m", "fde_L1_m", "ade_L2_m", "fde_L2_m",
              "history_error_mean_m", "history_error_last_m"] + [f"de_L2_step{t}_m" for t in range(1, horizon + 1)]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for source in SOURCES:
        writer.writerows(window_rows(source, run.prediction[source], h))
    return buffer.getvalue()


def save_run(out_dir: Path, run: FeasibilityRun, cfg: Config) -> Path:
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
        "seed": run.settings["seed"],
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "scene_content_hash": run.scene_content_hash,
        "frames_sha256": run.frames_sha256,
        "sources": {source: ROLES[source] for source in (GROUND_TRUTH_REFERENCE,) + SOURCES},
        "levels": LEVELS,
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "files": {name: file_sha256(out / name) for name in files},
    }
    (out / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    return out
