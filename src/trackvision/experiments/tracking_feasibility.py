"""Feasibility pipeline: detections -> tracker -> ground-plane trajectories on the protocol grid.

FEASIBILITY ONLY. Metrics are project-specific CLEAR-MOT-style counts and a simplified IDF1
(metrics/tracking.py): not MOTChallenge-compliant, not TrackEval results. Nothing is written to disk.

Detection sources, reported separately:
  - ground-truth boxes as detections (DIAGNOSTIC: isolates the tracker from the detector);
  - U4 detector with the oracle background (DIAGNOSTIC);
  - U4 detector with the median-estimated background (PRACTICAL BASELINE).

Two IoU thresholds with distinct purposes:
  - tracker `match_iou` (default 0.3): internal track-to-detection association;
  - configured `tracker_gt_assoc_iou_threshold` (0.5): evaluation against ground truth only.

Ground-plane position of an observation: box bottom-centre ((x1 + x2) / 2, y2) (v4.15 section 4.2)
mapped through the known synthetic camera's image-to-ground homography. Its systematic bias is
measured against U1's foot points (`geometry_bias`), not corrected. Only matched observations become
trajectory samples; coasting predictions never do. Grid samples are observed frames that are
multiples of `frames_per_step`, so missed grid frames leave gaps that windowing does not bridge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from trackvision.config import Config
from trackvision.experiments.detection_feasibility import MIN_VISIBLE_FRACTION, categorize
from trackvision.metrics.detection import clip_boxes
from trackvision.metrics.tracking import FrameTracking, evaluate_tracking
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.camera import PinholeCamera
from trackvision.synthetic.scene import SceneGroundTruth
from trackvision.synthetic.tracks import GridTrack
from trackvision.tracking.background_subtraction import BgSubConfig, detect
from trackvision.tracking.tracker import TrackerConfig, TrackerResult, run_tracker

GT_BOXES = "ground_truth_boxes_as_detections (diagnostic)"
NOTE = ("Feasibility pipeline. Project-specific CLEAR-MOT-style counts and simplified IDF1; not "
        "MOTChallenge-compliant and not TrackEval results. Descriptive for one scene.")


@dataclass(frozen=True)
class TrackingEvaluation:
    mode: str
    tracker: TrackerResult
    grid_tracks: tuple
    summary: dict


def bottom_centre_world(boxes, camera: PinholeCamera) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    pixels = np.column_stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]])
    return camera.ground_transform().image_to_world(pixels) if len(b) else np.empty((0, 2))


def geometry_bias(scene: SceneGroundTruth) -> dict:
    """Bottom-centre of the exact (unclipped) ground-truth box, mapped to ground, minus the foot point."""
    offset = bottom_centre_world(scene.bbox_xyxy, scene.config.camera) - scene.foot_world
    dist = np.linalg.norm(offset, axis=1)
    return {"n": int(len(dist)), "mean_offset_m": offset.mean(axis=0).tolist() if len(dist) else None,
            "mean_distance_m": float(dist.mean()) if len(dist) else None,
            "max_distance_m": float(dist.max()) if len(dist) else None}


def grid_tracks(result: TrackerResult, camera: PinholeCamera, frames_per_step: int) -> tuple[GridTrack, ...]:
    """Observed grid samples per confirmed track; `ped_id` holds the TRACKER id."""
    out = []
    for t in result.tracks:
        on_grid = t.frames % frames_per_step == 0
        out.append(GridTrack(t.track_id, t.frames[on_grid].copy(), bottom_centre_world(t.boxes[on_grid], camera)))
    return tuple(out)


def ground_truth_detections(scene: SceneGroundTruth, visible_fraction: np.ndarray) -> list[np.ndarray]:
    """Per frame, the clipped ground-truth boxes of pedestrians with any visible pixel."""
    cam = scene.config.camera
    boxes = clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height)
    out = []
    for k in range(scene.config.n_frames):
        start, stop = np.searchsorted(scene.frame, [k, k + 1])
        out.append(boxes[start:stop][visible_fraction[start:stop] > 0])
    return out


def detector_detections(frames, background: np.ndarray, det_config: BgSubConfig) -> list[np.ndarray]:
    return [detect(frame, background, det_config).boxes for frame in frames]


def evaluate_tracking_on_scene(scene: SceneGroundTruth, detections_per_frame, visible_fraction: np.ndarray,
                               cfg: Config, mode: str, tracker_config: TrackerConfig = TrackerConfig(),
                               min_visible_fraction: float = MIN_VISIBLE_FRACTION) -> TrackingEvaluation:
    cam = scene.config.camera
    detections_per_frame = list(detections_per_frame)
    if len(detections_per_frame) != scene.config.n_frames:
        raise ValueError("one detection array per scene frame is required")
    iou_threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    result = run_tracker(detections_per_frame, tracker_config)
    by_frame = result.frame_index()

    categories = categorize(visible_fraction, min_visible_fraction)
    gt_boxes = clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height)
    frames = []
    for k in range(scene.config.n_frames):
        start, stop = np.searchsorted(scene.frame, [k, k + 1])
        obs = by_frame.get(k, [])
        boxes = np.array([b for _, b in obs]).reshape(-1, 4)
        frames.append(FrameTracking(
            track_ids=np.array([i for i, _ in obs], dtype=np.int64), track_boxes=boxes,
            track_positions=bottom_centre_world(boxes, cam),
            gt_ids=scene.ped_id[start:stop], gt_boxes=gt_boxes[start:stop],
            gt_categories=categories[start:stop], gt_positions=scene.foot_world[start:stop]))
    metrics = evaluate_tracking(frames, iou_threshold)

    tracks = grid_tracks(result, cam, scene.frames_per_step)
    static = cfg.frozen["protocol"]["static"]
    windows = make_windows(mode, tracks, scene.frames_per_step, static["history_len"], static["horizon_len"],
                           static["window_stride"])
    summary = {
        "mode": mode,
        "note": NOTE,
        "parameters": {**asdict(tracker_config), "evaluation_iou_threshold": iou_threshold,
                       "min_visible_fraction": min_visible_fraction},
        "n_frames": scene.config.n_frames,
        "frames_without_detections": int(sum(len(d) == 0 for d in detections_per_frame)),
        "tracks": {"created": result.n_created, "confirmed": result.n_confirmed,
                   "deleted_tentative": result.n_deleted_tentative, "terminated": result.n_terminated},
        "metrics": metrics,
        "geometry_bias_of_bottom_centre": geometry_bias(scene),
        "grid": {"tracks": len(tracks), "samples": int(sum(len(t.frames) for t in tracks)),
                 "windows": len(windows)},
    }
    return TrackingEvaluation(mode, result, tracks, summary)
