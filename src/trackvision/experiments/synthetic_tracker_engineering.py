"""E4: a synthetic detector and tracker pipeline, end to end (decision 0006).

  synthetic scene -> render -> blob detector -> existing IoU tracker -> TrackEval
                  -> diagnostics on the 0.4 s grid -> CV, CA, LSTM and GRU on tracker histories

SYNTHETIC ENGINEERING ONLY. NOT P-9 AND NOT A REAL-TRACKER EVALUATION. The scenes are the Phase 4 scenes SYN-501 to
SYN-505, with ground-truth boxes that exist only because they were generated. The detector is the background
subtraction blob detector with the estimated background, not YOLO; ground-truth boxes are never used as detections.
The tracker is the existing IoU tracker. Identity comes from the generator's `ped_id` through box matching on these
synthetic boxes. `ground_truth_lineage` records where identities come from; it is not a section 3.7 mechanism. The
run's `association_mechanism` is null and unresolved, for the reason in `experiments.association_provenance`.

Predictors are the clean-trained Phase 4 models, each used only on the scene it never saw: the H = 8 checkpoint of
that scene's own fold, verified by hash. Nothing is retrained. A checkpoint from another fold is refused.

Official TrackEval values and project diagnostics sit in separate sections. The official implementation scoring a
synthetic pipeline is still not a benchmark. Severities are diagnostics, not `real_switch_event_rate`: the name is
reserved for real trackers. Each scene is one unit; windows are never independent observations.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments import association_provenance, evaluation, injection_feasibility as inj
from trackvision.experiments import mot17_trackeval as mt
from trackvision.experiments.detection_feasibility import MIN_VISIBLE_FRACTION, categorize
from trackvision.experiments.learned_feasibility import N_FRAMES, N_PEDESTRIANS, SCENE_SEEDS, scene_name
from trackvision.experiments.mot17_trackeval import check_output_location
from trackvision.experiments.prediction_feasibility import (PredictionEvaluation, associate_grid,
                                                            box_bottom_centre_grid_tracks, evaluate_prediction)
from trackvision.experiments.store import COMPLETE, RunStore, source_sha256
from trackvision.experiments.tracking_feasibility import evaluate_tracking_on_scene
from trackvision.inject.identity_switch import proximity
from trackvision.io import mot17
from trackvision.metrics.detection import clip_boxes
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.metrics.error_profile import MatchedFrame, error_profile
from trackvision.predict import learned
from trackvision.predict.training import load_checkpoint
from trackvision.stats.aggregate import aggregate, relative_degradation
from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import SceneConfig, SceneGroundTruth, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks
from trackvision.tracking.background_subtraction import BgSubConfig, detect, median_background, sample_indices
from trackvision.tracking.tracker import TrackerConfig, TrackerResult

SCOPE = "synthetic_tracker_engineering"
SOURCE = "synthetic_blob_detector_estimated_background"
GROUND_TRUTH_LINEAGE = "synthetic_generator_ped_id"
BENCHMARK, SPLIT = "SYNTHETIC", "engineering"
GT_SET = f"{BENCHMARK}-{SPLIT}"
TRACKER = "trackvision-iou-on-synthetic-blobs"
DISTRACTOR_CLASS = 8                    # MOT class for a partly visible pedestrian: neither TP nor FP
LABEL = ("SYNTHETIC ENGINEERING ONLY (E4, decision 0006). Generated scenes, a blob detector and the existing IoU "
         "tracker. Not P-9, not a real-tracker evaluation, not ETH/UCY, no real-world generalisation. Official "
         "TrackEval values here score a synthetic pipeline and are not benchmark results. v4.15 Phase 1 remains "
         "BLOCKED and v4.16 is unfrozen.")
PROVENANCE = {"detector": "synthetic blob detector (background subtraction, estimated background); not YOLO",
              "tracker": "existing IoU tracker; not ByteTrack or BoT-SORT",
              "ground_truth_lineage": GROUND_TRUTH_LINEAGE, "research_results": False}


def scene_ground_truth(seed: int, cfg: Config) -> SceneGroundTruth:
    """The Phase 4 scene of this generator seed."""
    dt = cfg.frozen["protocol"]["static"]["sampling_interval_s"]
    return generate(SceneConfig(seed=seed, n_frames=N_FRAMES, n_pedestrians=N_PEDESTRIANS), dt)


def blob_detections(scene: SceneGroundTruth, render: RenderConfig, det: BgSubConfig) -> tuple[list, np.ndarray, str]:
    """Blob detections with the background estimated from the video itself, per-row visible fractions, frame digest."""
    renderer = SceneRenderer(scene, render)
    n = scene.config.n_frames
    picked = sample_indices(n, det.background_frames)
    background = median_background((renderer.frame(int(k)).image for k in picked), len(picked), len(picked))
    visible, digest, boxes = np.zeros(len(scene.frame)), hashlib.sha256(), []
    for rendered in renderer:
        visible[rendered.rows] = rendered.visible_fraction
        digest.update(rendered.image.tobytes())
        boxes.append(detect(rendered.image, background, det).boxes)
    return boxes, visible, digest.hexdigest()


def tracker_mot_text(result: TrackerResult) -> str:
    """MOT rows for the tracker output, frames 1-based, one line per confirmed observation."""
    rows = sorted((int(f) + 1, t.track_id, b) for t in result.tracks for f, b in zip(t.frames, t.boxes))
    lines = []
    for f, tid, box in rows:
        left, top, w, h = mot17.to_mot_box(box)[0]
        lines.append(f"{f},{tid},{left:.3f},{top:.3f},{w:.3f},{h:.3f},1,-1,-1,-1\n")
    return "".join(lines)


def ground_truth_mot_text(scene: SceneGroundTruth, visible: np.ndarray) -> str:
    """MOT rows for the generated boxes. Partly visible pedestrians are distractors; invisible ones are left out."""
    cam = scene.config.camera
    boxes = mot17.to_mot_box(clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height))
    categories = categorize(visible, MIN_VISIBLE_FRACTION)
    lines = []
    for k in range(len(scene.frame)):
        if categories[k] == "invisible":
            continue
        left, top, w, h = boxes[k]
        consider, cls = (1, mot17.PEDESTRIAN) if categories[k] == "evaluable" else (0, DISTRACTOR_CLASS)
        lines.append(f"{int(scene.frame[k]) + 1},{int(scene.ped_id[k])},{left:.3f},{top:.3f},{w:.3f},{h:.3f},{consider},"
                     f"{cls},{visible[k]:.6f}\n")
    return "".join(lines)


def grid_profile(scene: SceneGroundTruth, association, cfg: Config) -> dict:
    """Diagnostics on the 0.4 s grid. Eligible observations are the clean ground-truth samples (section 3.7 B)."""
    gt_tracks = ground_truth_grid_tracks(scene)
    step, assoc = scene.frames_per_step, cfg.frozen["association"]
    position = {(t.ped_id, int(f)): p for t in gt_tracks for f, p in zip(t.frames, t.positions)}
    assigned: dict = {}
    for t in association.tracks:
        for f, g, p in zip(t.frames, t.labels, t.positions):
            if g != 0:
                assigned.setdefault(int(f), {})[int(g)] = (t.track_id, p)
    present: dict = {}
    for t in gt_tracks:
        for f in t.frames:
            present.setdefault(int(f), []).append(t.ped_id)
    frames = []
    for index, f in enumerate(range(0, scene.config.n_frames, step)):
        here = assigned.get(f, {})
        frames.append(MatchedFrame(index, tuple(present.get(f, ())), {g: v[0] for g, v in here.items()},
                                   {g: float(np.linalg.norm(v[1] - position[(g, f)])) for g, v in here.items()}))
    near = proximity(gt_tracks, float(cfg.frozen["injection"]["d_max_m"]), step)
    profile = error_profile(frames, assoc["frag_gap_max_steps"], assoc["switch_gap_max_steps"], "metres",
                            sum(max(len(t.frames) - 2, 0) for t in gt_tracks), near.eligible_track_frames,
                            "eligible_track_frames (section 3.5 D proximity, D_MAX, no injection locks)")
    return {"note": "project diagnostic on the synthetic 0.4 s grid; not a section 3.7 real-tracker severity",
            "uses_synthetic_camera_for_metres": True, **profile}


def load_fold_models(models_run: Path, scene: str, seeds, cfg: Config) -> dict:
    """The clean-trained H = 8 models of this scene's fold, verified. Refuses any model that saw the scene."""
    history = cfg.frozen["protocol"]["static"]["history_len"]
    models = {}
    for cell in learned.CELLS:
        for seed in seeds:
            path = Path(models_run) / "checkpoints" / f"h{history}" / scene / f"{cell.lower()}-seed{seed}.pt"
            digest = json.loads(path.with_suffix(".pt.json").read_text(encoding="utf-8"))["checkpoint_sha256"]
            model, meta = load_checkpoint(path, digest)
            split = meta["split"]
            if split["test_scene"] != scene or scene in (split["validation_scene"], *split["train_scenes"]):
                raise ValueError(f"{path} was not held out on {scene}")
            if meta["config_hash"] != cfg.hash or meta["model"]["architecture"]["history_len"] != history:
                raise ValueError(f"{path} does not match this configuration")
            models[(cell, seed)] = (model, digest)
    return models


def _ade_fde(predictor, observed, target) -> tuple[float, float]:
    ade, fde = ade_fde(displacement_errors(predictor.predict(observed), target))
    return float(ade.mean()), float(fde.mean())


def score_predictors(ev: PredictionEvaluation, models: dict, seeds, cfg: Config) -> dict:
    """Clean and tracker-history accuracy per predictor, seeds first. Coverage is shared and reported beside it."""
    idx = ev.predicted_index
    w, tracked, surviving = ev.windows, ev.inputs["L2"], ev.inputs["L0"]
    horizon = w.horizon_len
    fixed = {"CV": evaluation.cv_predictor(cfg, horizon), "CA": evaluation.ca_predictor(cfg, horizon)}
    out = {}
    for name in ("CV", "CA", *learned.CELLS):
        runs = {}
        for seed in ((None,) if name in fixed else seeds):
            predictor = fixed[name] if name in fixed else evaluation.learned_predictor(models[(name, seed)][0])
            ade0, fde0 = _ade_fde(predictor, w.observed, w.target)
            row = {"clean_all_eligible_ade_m": ade0, "clean_all_eligible_fde_m": fde0, "tracker_ade_m": None,
                   "tracker_fde_m": None, "clean_paired_ade_m": None, "checkpoint_sha256": None}
            if len(idx):
                row["tracker_ade_m"], row["tracker_fde_m"] = _ade_fde(predictor, tracked, w.target[idx])
                row["clean_paired_ade_m"] = _ade_fde(predictor, surviving, w.target[idx])[0]
            if name not in fixed:
                row["checkpoint_sha256"] = models[(name, seed)][1]
            row["d_ade_all_eligible"] = relative_degradation(row["tracker_ade_m"], ade0).value
            row["d_ade_paired"] = relative_degradation(row["tracker_ade_m"], row["clean_paired_ade_m"]).value
            runs[str(seed)] = row
        fields = ("clean_all_eligible_ade_m", "tracker_ade_m", "tracker_fde_m", "d_ade_all_eligible", "d_ade_paired")
        out[name] = {"by_seed": runs, "seed_mean": {f: aggregate({s: r[f] for s, r in runs.items()}).as_dict() for f in fields}}
    return out


def scene_record(scene: SceneGroundTruth, name: str, models: dict, seeds, cfg: Config, render: RenderConfig,
                 det: BgSubConfig, tracker: TrackerConfig, out_dir: Path) -> dict:
    """Run one scene and write its raw tracker output and ground truth under `out_dir`."""
    detections, visible, frames_sha = blob_detections(scene, render, det)
    tev = evaluate_tracking_on_scene(scene, detections, visible, cfg, SOURCE, tracker)
    threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    association = associate_grid(scene, tev.tracker, detections, visible, threshold)
    ev = evaluate_prediction(ground_truth_grid_tracks(scene), box_bottom_centre_grid_tracks(scene), association,
                             scene.frames_per_step, cfg, SOURCE)
    tracker_text, gt_text = tracker_mot_text(tev.tracker), ground_truth_mot_text(scene, visible)
    for folder, text in (("tracker_output", tracker_text), ("ground_truth", gt_text)):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
        (out_dir / folder / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
    coverage = {k: ev.summary["coverage"][k] for k in ("eligible_ground_truth_windows", "predicted_windows",
                                                       "prediction_coverage", "lost_by_primary_reason")}
    return {"scene": name, "frames_sha256": frames_sha, "scene_content_hash": scene.content_hash(),
            "tracker_output_sha256": hashlib.sha256(tracker_text.encode()).hexdigest(),
            "ground_truth_sha256": hashlib.sha256(gt_text.encode()).hexdigest(),
            "project_diagnostics": {"note": "project-specific counts; not TrackEval", **tev.summary["metrics"],
                                    "tracks": tev.summary["tracks"], "error_profile": grid_profile(scene, association, cfg)},
            "prediction": {"coverage": coverage, "predictors": score_predictors(ev, models, seeds, cfg)}}


def seqinfo(name: str, scene: SceneGroundTruth) -> mot17.SequenceInfo:
    cam = scene.config.camera
    return mot17.SequenceInfo(name, "img1", scene.config.fps, scene.config.n_frames, cam.image_width, cam.image_height, ".jpg")


def run_official(out_dir: Path, scenes: dict, cfg: Config, python: Path, source: Path) -> dict:
    """Official TrackEval on the stored outputs, one sequence per scene, in its isolated environment."""
    eval_dir = out_dir / "trackeval"
    root = eval_dir / "trackeval_input"
    exports = []
    for name, scene in scenes.items():
        gt = (out_dir / "ground_truth" / f"{name}.txt").read_text(encoding="utf-8")
        tracks = (out_dir / "tracker_output" / f"{name}.txt").read_text(encoding="utf-8")
        mt.write_sequence(root, seqinfo(name, scene), scene.config.n_frames, gt, tracks, GT_SET, TRACKER)
        exports.append({"folder": name, "seq_length": scene.config.n_frames, "tracker_rows": tracks.count("\n")})
    mt.write_seqmap(root, list(scenes), GT_SET)
    (eval_dir / "export_manifest.json").write_text(json.dumps({"sequences": exports}, indent=2) + "\n", encoding="utf-8",
                                                   newline="\n")
    code = mt.run_trackeval(eval_dir, python, source, cfg, benchmark=BENCHMARK, split=SPLIT, tracker=TRACKER)
    if code != 0:
        raise RuntimeError(f"TrackEval exited with code {code}; see {eval_dir / 'logs'}")
    return {s: mt.headline(r) for s, r in mt.validate_results(eval_dir, GT_SET, TRACKER)["sequences"].items()}


def run(out_dir: Path, cfg: Config, models_run: Path, python: Path, source: Path, scenes=None, resume: bool = False) -> dict:
    """Run E4 in a new run directory. `scenes` limits the run, for a pilot; `models_run` is only read."""
    out_dir = Path(out_dir)
    check_output_location(out_dir)
    render, det, tracker = RenderConfig(), BgSubConfig(), TrackerConfig()
    wanted = [scene_name(s) for s in SCENE_SEEDS] if scenes is None else list(scenes)
    by_seed = {scene_name(s): s for s in SCENE_SEEDS}
    if not wanted or set(wanted) - set(by_seed):
        raise ValueError(f"scenes must come from {sorted(by_seed)}, got {wanted}")
    models_spec = RunStore.open(models_run).manifest()["spec"]
    seeds = list(models_spec["seeds"])
    spec = {"scope": SCOPE, "label": LABEL, "provenance": PROVENANCE, "scenes": wanted, "config_hash": cfg.hash,
            "source_sha256": source_sha256(), "render": asdict(render), "detector": asdict(det),
            "tracker": asdict(tracker), "models_run": {"scope": models_spec["scope"], "config_hash": models_spec["config_hash"],
                                                       "source_sha256": models_spec["source_sha256"],
                                                       "scene_sha256": models_spec["scenes"]}, "seeds": seeds,
            "min_visible_fraction": MIN_VISIBLE_FRACTION, "trackeval_commit": mt.TRACKEVAL_COMMIT}
    store = RunStore.create(out_dir, spec, resume)
    generated, records = {}, {}
    for name in wanted:
        scene = generated[name] = scene_ground_truth(by_seed[name], cfg)
        digest = inj.tracks_digest(ground_truth_grid_tracks(scene))
        if digest != models_spec["scenes"][name]:
            raise ValueError(f"{name} is not the scene the Phase 4 models were evaluated on")
        identity = {"scope": SCOPE, "experiment": "E4", "scene": name, "scene_sha256": digest, "config_hash": cfg.hash,
                    "source_sha256": spec["source_sha256"]}
        record = store.get(identity)
        if record is None or record.status != COMPLETE:
            start = time.perf_counter()
            result = scene_record(scene, name, load_fold_models(models_run, name, seeds, cfg), seeds, cfg, render, det,
                                  tracker, out_dir)
            record = store.put(identity, COMPLETE, result, provenance={"seconds": round(time.perf_counter() - start, 1)})
        records[name] = record.result
    official = run_official(out_dir, generated, cfg, Path(python), Path(source))
    doc = {"label": LABEL, "experiment": "E4", "scope": SCOPE, "research_results": False, "not_p9": True,
           "provenance": PROVENANCE, "scenes": wanted, **association_provenance.synthetic_tracker_field(),
           "official_trackeval": {"note": "official implementation on synthetic outputs; not a benchmark", **official},
           "by_scene": records}
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    (out_dir / "results.json").write_text(text, encoding="utf-8", newline="\n")
    (out_dir / "run_manifest.json").write_text(json.dumps(
        {"label": LABEL, "results_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "config_hash": cfg.hash,
         "trackvision_version": __version__, "git": repro.git_state(config.REPO_ROOT),
         "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "environment": repro.environment_info()},
        indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return doc
