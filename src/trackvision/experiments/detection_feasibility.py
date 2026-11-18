"""Feasibility evaluation of the background-subtraction detector on rendered synthetic scenes.

FEASIBILITY ONLY: an abstract renderer and a blob detector. The numbers describe this setup, not
realistic detector performance, and no research gate is passed. Nothing is written to disk.

Two separately reported modes:
  - oracle background (DIAGNOSTIC): the renderer's clean background, isolating the detector from
    background estimation;
  - estimated background (PRACTICAL BASELINE): the per-pixel median of sampled frames of the same
    video, as the detector would run without ground truth.

Ground-truth categories per row, from the renderer's visible fraction (truncation and occlusion):
  evaluable          visible_fraction >= min_visible_fraction (provisional, predeclared 0.5)
  ignore (partial)   0 < visible_fraction < min_visible_fraction: never FN; matched detections are
                     ignored matches, not FP
  invisible          visible_fraction == 0: excluded entirely
Ground-truth boxes are U1's boxes clipped to the image. The IoU threshold is the configured
`frozen.association.tracker_gt_assoc_iou_threshold`. Counts are micro-summed over correlated frames:
descriptive values for one scene, not independent samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np

from trackvision.config import Config
from trackvision.metrics.detection import clip_boxes, evaluate_frame, precision_recall_f1
from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import SceneGroundTruth
from trackvision.tracking.background_subtraction import BgSubConfig, detect, median_background

MIN_VISIBLE_FRACTION = 0.5          # provisional, predeclared feasibility parameter
ORACLE = "oracle_background (diagnostic)"
ESTIMATED = "estimated_median_background (practical baseline)"
NOTE = ("Feasibility baseline: abstract grey renderer and blob detector; not realistic detector "
        "performance. Counts are micro-summed over correlated frames of one scene; descriptive only.")


@dataclass(frozen=True)
class DetectionEvaluation:
    mode: str
    detections: tuple            # Detections per frame
    frame_counts: tuple          # FrameCounts per frame
    categories: np.ndarray       # per U1 row: "evaluable" | "ignore" | "invisible"
    summary: dict


def categorize(visible_fraction: np.ndarray, min_visible_fraction: float) -> np.ndarray:
    vf = np.asarray(visible_fraction, dtype=np.float64)
    if not 0.0 < min_visible_fraction <= 1.0:
        raise ValueError("min_visible_fraction must lie in (0, 1]")
    out = np.full(vf.shape, "ignore", dtype=object)
    out[vf >= min_visible_fraction] = "evaluable"
    out[vf <= 0.0] = "invisible"
    return out


def rendered_visible_fraction(scene: SceneGroundTruth, render_config: RenderConfig) -> np.ndarray:
    """Per-row visible fraction (aligned with U1 rows), from one rendering pass."""
    vf = np.zeros(len(scene.frame))
    for rendered in SceneRenderer(scene, render_config):
        vf[rendered.rows] = rendered.visible_fraction
    return vf


def evaluate_detector(scene: SceneGroundTruth, frames: Iterable[np.ndarray], background: np.ndarray,
                      visible_fraction: np.ndarray, det_config: BgSubConfig, cfg: Config, mode: str,
                      min_visible_fraction: float = MIN_VISIBLE_FRACTION) -> DetectionEvaluation:
    cam = scene.config.camera
    iou_threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    if len(visible_fraction) != len(scene.frame):
        raise ValueError("visible_fraction must be aligned with the scene's ground-truth rows")
    categories = categorize(visible_fraction, min_visible_fraction)
    gt_boxes = clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height)

    detections, counts = [], []
    n = 0
    for k, frame in enumerate(frames):
        start, stop = np.searchsorted(scene.frame, [k, k + 1])
        det = detect(frame, background, det_config)
        cat = categories[start:stop]
        counts.append(evaluate_frame(det.boxes, gt_boxes[start:stop], cat == "evaluable", cat == "ignore",
                                     iou_threshold))
        detections.append(det)
        n += 1
    if n != scene.config.n_frames:
        raise ValueError(f"expected {scene.config.n_frames} frames, got {n}")

    total = {f: int(sum(getattr(c, f) for c in counts))
             for f in ("n_detections", "tp", "fn", "ignored_matches", "fp_overlapping_visible", "fp_background")}
    total["fp"] = total["fp_overlapping_visible"] + total["fp_background"]
    summary = {
        "mode": mode,
        "note": NOTE,
        "parameters": {**asdict(det_config), "min_visible_fraction": min_visible_fraction,
                       "iou_threshold": iou_threshold, "matching": "greedy one-to-one, IoU >= threshold"},
        "n_frames": n,
        "ground_truth": {
            "rows": int(len(categories)),
            "evaluable": int(np.sum(categories == "evaluable")),
            "evaluable_truncated": int(np.sum((categories == "evaluable") & scene.truncated)),
            "ignore_partial": int(np.sum(categories == "ignore")),
            "invisible": int(np.sum(categories == "invisible")),
        },
        **total,
        **precision_recall_f1(total["tp"], total["fp"], total["fn"]),
    }
    return DetectionEvaluation(mode, tuple(detections), tuple(counts), categories, summary)


def evaluate_with_oracle_background(scene: SceneGroundTruth, render_config: RenderConfig,
                                    det_config: BgSubConfig, cfg: Config) -> DetectionEvaluation:
    """DIAGNOSTIC: detector run against the renderer's clean background."""
    renderer = SceneRenderer(scene, render_config)
    vf = rendered_visible_fraction(scene, render_config)
    frames = (renderer.frame(k).image for k in range(scene.config.n_frames))
    return evaluate_detector(scene, frames, renderer.background, vf, det_config, cfg, ORACLE)


def evaluate_with_estimated_background(scene: SceneGroundTruth, frames_factory: Callable[[], Iterable[np.ndarray]],
                                       visible_fraction: np.ndarray, det_config: BgSubConfig,
                                       cfg: Config) -> DetectionEvaluation:
    """PRACTICAL BASELINE: median background from the frames themselves. `frames_factory` must return a
    fresh iterable of the scene's frames on each call (it is read twice)."""
    background = median_background(frames_factory(), scene.config.n_frames, det_config.background_frames)
    return evaluate_detector(scene, frames_factory(), background, visible_fraction, det_config, cfg, ESTIMATED)
