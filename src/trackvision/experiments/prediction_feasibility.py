"""Feasibility evaluation of constant-velocity prediction on tracker trajectories, with coverage.

FEASIBILITY ONLY: no research hypothesis is tested, no gate is passed, and nothing is written to
disk. This is not a spec v4.15 section 11.3 experiment record.

Identity of a tracker observation (grid frames only). At each grid frame the tracker's output boxes
are associated one-to-one with ground truth by `gated_iou_assignment` at the configured
`tracker_gt_assoc_iou_threshold` (0.5), in the same two stages as metrics/tracking.py: evaluable
ground truth first, then low-visibility (ignore) ground truth. A matched observation is labelled
with that pedestrian's ped_id; an unmatched one is unassociated (label 0).

Four counts, never interchanged:
  - eligible ground-truth windows: U2's windows (`make_windows` on ground-truth grid tracks with
    the configured history_len, horizon_len and stride). Eligibility depends on ground truth alone
    (v4.15 section 3.2), so occluded pedestrians stay eligible;
  - available tracker histories: history_len consecutive grid samples of ONE tracker ID, cut with
    `consecutive_runs` and the configured stride. They never bridge a missed grid frame or join IDs;
  - predicted windows: eligible windows (g, f_H) whose history_len history frames are all matched
    to g by the same tracker ID. One-to-one association makes that tracker history unique;
  - unscorable tracker histories: available histories with no eligible target (counted, never scored).
Prediction coverage = predicted / eligible (None when nothing is eligible). Available tracker
histories are never the numerator.

Lost windows (eligible, not predicted). Each unmatched history frame has a cause, by priority:
invisible (visible fraction 0) > low_visibility (below min_visible_fraction) > no_detection (no raw
detection with IoU >= threshold against the pedestrian's box) > not_in_tracker_output (a detection
existed, but no confirmed tracker observation was matched to the pedestrian). If all history frames
are matched but to more than one tracker ID, the cause is identity_change. The primary reason is the
highest-priority cause; non-exclusive flags list every cause present.

Accuracy on predicted windows. The unchanged CV predictor is run on three inputs, each scored against
the SAME pedestrian's ground-truth future (U1 foot points at the future grid frames):
  L0  the pedestrian's ground-truth foot points          (predictor error alone)
  L1  bottom-centre of its exact, unclipped ground-truth box mapped to the ground
                                                         (L0 plus the projection bias)
  L2  the tracker's observed positions                   (the system)
The levels are paired on the same windows. Euclidean errors are not additive, so the levels are
descriptive and not a decomposition. Survivorship: L0 on all eligible windows versus L0 on
predicted windows only. Overlapping windows are correlated: summaries are descriptive, with no
confidence intervals or tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from trackvision.config import Config
from trackvision.experiments.detection_feasibility import MIN_VISIBLE_FRACTION, categorize
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.experiments.tracking_feasibility import bottom_centre_world, grid_tracks
from trackvision.metrics.detection import clip_boxes, iou_matrix
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.predict.windowing import Windows, consecutive_runs, make_windows
from trackvision.synthetic.scene import SceneGroundTruth
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks
from trackvision.tracking.assignment import gated_iou_assignment
from trackvision.tracking.tracker import TrackerResult

INVISIBLE = "invisible"
LOW_VISIBILITY = "low_visibility"
NO_DETECTION = "no_detection"
NOT_IN_TRACKER_OUTPUT = "not_in_tracker_output"
IDENTITY_CHANGE = "identity_change"
FRAME_CAUSES = (INVISIBLE, LOW_VISIBILITY, NO_DETECTION, NOT_IN_TRACKER_OUTPUT)     # priority order
LOST_REASONS = FRAME_CAUSES + (IDENTITY_CHANGE,)

UNASSOCIATED_OBSERVATION = "contains_unassociated_observation"
IDENTITY_MIXED = "identity_mixed"
GT_FUTURE_INCOMPLETE = "gt_future_incomplete"
NOT_ON_WINDOW_STRIDE = "not_on_window_stride"     # cannot occur at the configured stride of 1
UNSCORABLE = (UNASSOCIATED_OBSERVATION, IDENTITY_MIXED, GT_FUTURE_INCOMPLETE, NOT_ON_WINDOW_STRIDE)

LEVELS = {
    "L0": "ground-truth foot points as CV input (predictor error alone)",
    "L1": "exact ground-truth box bottom-centre as CV input (L0 plus projection bias)",
    "L2": "tracker observed positions as CV input (the system)",
}
NOTE = ("Feasibility evaluation on one synthetic scene; not a v4.15 section 11.3 record and not gate "
        "evidence. Overlapping stride-1 windows are correlated: descriptive summaries only, no inference.")
COVERAGE_NOTE = ("prediction_coverage = predicted eligible ground-truth windows / eligible ground-truth "
                 "windows (v4.15 section 3.2). Available tracker histories are never the numerator. "
                 "Conditional accuracy (ADE/FDE on predicted windows) must be read together with "
                 "coverage; on its own it is not system performance.")
LEVELS_NOTE = ("L0, L1 and L2 are paired on the same predicted windows and scored against the same "
               "pedestrian's ground-truth future. Euclidean errors are not additive; the levels are "
               "descriptive, not a decomposition.")


@dataclass(frozen=True)
class LabelledTrack:
    """A tracker track's observed grid samples with their ground-truth identity labels."""
    track_id: int
    frames: np.ndarray       # (n,) int64 grid frames
    positions: np.ndarray    # (n, 2) metres, observed box bottom-centre on the ground
    labels: np.ndarray       # (n,) int64 ped_id, 0 = unassociated


@dataclass(frozen=True)
class GridAssociation:
    tracks: tuple            # LabelledTrack per confirmed tracker track
    causes: Mapping          # (ped_id, frame) -> frame cause, for ground truth present but unmatched


@dataclass(frozen=True)
class TrackerHistory:
    track_id: int
    frames: np.ndarray       # (history_len,)
    labels: np.ndarray       # (history_len,)
    ped_id: int              # the single associated pedestrian, 0 if none or several
    category: str | None     # None = scorable (it predicts an eligible window), else an UNSCORABLE value


@dataclass(frozen=True)
class PredictionEvaluation:
    source: str
    windows: Windows                 # eligible ground-truth windows (L0 inputs and targets)
    predicted: np.ndarray            # (N,) bool per eligible window
    tracker_id: np.ndarray           # (N,) int64, 0 if lost
    primary_reason: tuple            # (N,) str | None
    cause_flags: tuple               # (N,) tuple of reasons present
    l0_all_per_step: np.ndarray      # (N, horizon_len) L0 errors on every eligible window
    inputs: dict                     # level -> (P, history_len, 2) for predicted windows
    predictions: dict                # level -> (P, horizon_len, 2)
    per_step: dict                   # level -> (P, horizon_len) metres
    history_error: np.ndarray        # (P, history_len) || L2 input - L0 input ||
    histories: tuple                 # TrackerHistory
    summary: dict

    @property
    def predicted_index(self) -> np.ndarray:
        return np.flatnonzero(self.predicted)


def _stats(values) -> dict | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    return {"mean": float(values.mean()), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())}


def _accuracy(per_step: np.ndarray) -> dict:
    n = len(per_step)
    ade, fde = ade_fde(per_step) if n else (np.empty(0), np.empty(0))
    return {"n_windows": int(n), "mean_ade_m": float(ade.mean()) if n else None,
            "mean_fde_m": float(fde.mean()) if n else None, "ade_m": _stats(ade), "fde_m": _stats(fde),
            "mean_error_by_step_m": per_step.mean(axis=0).tolist() if n else None}


def box_bottom_centre_grid_tracks(scene: SceneGroundTruth) -> tuple[GridTrack, ...]:
    """Per pedestrian, the bottom-centre of its exact (unclipped) ground-truth box on the grid, mapped
    to the ground through the known camera: the L1 input."""
    on_grid = scene.frame % scene.frames_per_step == 0
    out = []
    for pid in range(1, scene.config.n_pedestrians + 1):
        rows = on_grid & (scene.ped_id == pid)
        out.append(GridTrack(pid, scene.frame[rows].copy(),
                             bottom_centre_world(scene.bbox_xyxy[rows], scene.config.camera)))
    return tuple(out)


def associate_grid(scene: SceneGroundTruth, tracker_result: TrackerResult, detections_per_frame,
                   visible_fraction: np.ndarray, iou_threshold: float,
                   min_visible_fraction: float = MIN_VISIBLE_FRACTION) -> GridAssociation:
    """Label every grid observation of the tracker and record a cause for unmatched ground truth."""
    cam, step = scene.config.camera, scene.frames_per_step
    detections_per_frame = list(detections_per_frame)
    if len(detections_per_frame) != scene.config.n_frames:
        raise ValueError("one detection array per scene frame is required")
    visible_fraction = np.asarray(visible_fraction, dtype=np.float64)
    categories = categorize(visible_fraction, min_visible_fraction)
    gt_boxes = clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height)
    by_frame = tracker_result.frame_index()
    label_of: dict[tuple[int, int], int] = {}
    causes: dict[tuple[int, int], str] = {}

    for f in range(0, scene.config.n_frames, step):
        start, stop = np.searchsorted(scene.frame, [f, f + 1])
        obs = by_frame.get(f, [])
        tids = [i for i, _ in obs]
        tb = np.array([b for _, b in obs]).reshape(-1, 4)
        cats, boxes = categories[start:stop], gt_boxes[start:stop]
        ev, ig = np.flatnonzero(cats == "evaluable"), np.flatnonzero(cats == "ignore")

        pairs = [(i, int(ev[g])) for i, g, _ in gated_iou_assignment(tb, boxes[ev], iou_threshold)]
        left = [i for i in range(len(tb)) if i not in {i for i, _ in pairs}]
        if left:
            pairs += [(left[i], int(ig[g])) for i, g, _ in gated_iou_assignment(tb[left], boxes[ig], iou_threshold)]
        matched_rows = set()
        for i, r in pairs:
            label_of[(tids[i], f)] = int(scene.ped_id[start + r])
            matched_rows.add(r)

        dets = np.asarray(detections_per_frame[f], dtype=np.float64).reshape(-1, 4)
        for r in range(stop - start):
            if r in matched_rows:
                continue
            vf = visible_fraction[start + r]
            if vf <= 0.0:
                cause = INVISIBLE
            elif vf < min_visible_fraction:
                cause = LOW_VISIBILITY
            elif not (len(dets) and np.any(iou_matrix(dets, boxes[[r]]) >= iou_threshold)):
                cause = NO_DETECTION
            else:
                cause = NOT_IN_TRACKER_OUTPUT
            causes[(int(scene.ped_id[start + r]), f)] = cause

    tracks = tuple(
        LabelledTrack(g.ped_id, g.frames, g.positions,
                      np.array([label_of.get((g.ped_id, int(f)), 0) for f in g.frames], dtype=np.int64))
        for g in grid_tracks(tracker_result, cam, step))
    return GridAssociation(tracks, causes)


def _matched_index(tracks) -> dict[tuple[int, int], tuple[int, int]]:
    """(ped_id, frame) -> (track_id, sample index); one-to-one is enforced."""
    out = {}
    for t in tracks:
        for k, (f, g) in enumerate(zip(t.frames, t.labels)):
            if g == 0:
                continue
            key = (int(g), int(f))
            if key in out:
                raise ValueError(f"pedestrian {key[0]} is claimed twice at frame {key[1]}")
            out[key] = (t.track_id, k)
    return out


def evaluate_prediction(gt_tracks, bias_tracks, association: GridAssociation, frame_step: int, cfg: Config,
                        source: str) -> PredictionEvaluation:
    """Coverage and paired L0/L1/L2 accuracy of CV for one detection source (see module docstring)."""
    p = protocol(cfg)
    h, horizon, stride = p["history_len"], p["horizon_len"], p["window_stride"]
    eligible = make_windows(source, gt_tracks, frame_step, h, horizon, stride)
    biased = make_windows(source, bias_tracks, frame_step, h, horizon, stride)
    if not (np.array_equal(eligible.frames, biased.frames) and np.array_equal(eligible.track_ids, biased.track_ids)):
        raise ValueError("ground-truth and box bottom-centre tracks must have the same grid samples")
    tracks = {t.track_id: t for t in association.tracks}
    matched = _matched_index(association.tracks)

    n = len(eligible)
    predicted = np.zeros(n, dtype=bool)
    tracker_id = np.zeros(n, dtype=np.int64)
    primary, flags, l2_inputs = [], [], []
    for w in range(n):
        g = int(eligible.track_ids[w])
        hits = [matched.get((g, int(f))) for f in eligible.frames[w, :h]]
        ids = {hit[0] for hit in hits if hit is not None}
        missing = []
        for f, hit in zip(eligible.frames[w, :h], hits):
            if hit is None:
                key = (g, int(f))
                if key not in association.causes:
                    raise ValueError(f"no cause recorded for unmatched pedestrian {g} at frame {int(f)}")
                missing.append(association.causes[key])
        present = [c for c in FRAME_CAUSES if c in missing] + ([IDENTITY_CHANGE] if len(ids) > 1 else [])
        if not missing and len(ids) == 1:
            tid = ids.pop()
            predicted[w], tracker_id[w] = True, tid
            l2_inputs.append(tracks[tid].positions[[hit[1] for hit in hits]])
            primary.append(None)
            flags.append(())
        else:
            primary.append(present[0])
            flags.append(tuple(present))

    idx = np.flatnonzero(predicted)
    cv = lambda obs: constant_velocity(obs, horizon, p["cv_min_final_observations"])
    targets = eligible.target[idx]
    inputs = {"L0": eligible.observed[idx], "L1": biased.observed[idx],
              "L2": np.asarray(l2_inputs, dtype=np.float64).reshape(-1, h, 2)}
    predictions = {level: cv(obs) for level, obs in inputs.items()}
    per_step = {level: displacement_errors(pred, targets) for level, pred in predictions.items()}
    l0_all = displacement_errors(cv(eligible.observed), eligible.target)
    history_error = np.linalg.norm(inputs["L2"] - inputs["L0"], axis=-1)

    eligible_keys = {(int(g), int(fr[h - 1])): w for w, (g, fr) in enumerate(zip(eligible.track_ids, eligible.frames))}
    gt_frames = {t.ped_id: set(int(f) for f in t.frames) for t in gt_tracks}
    histories = []
    for t in association.tracks:
        for start, stop in consecutive_runs(t.frames, frame_step):
            for s in range(start, stop - h + 1, stride):
                labels = t.labels[s:s + h]
                peds = set(int(v) for v in labels)
                ped = peds.pop() if len(peds) == 1 else 0
                anchor = int(t.frames[s + h - 1])
                if np.any(labels == 0):
                    category, ped = UNASSOCIATED_OBSERVATION, 0
                elif ped == 0:
                    category = IDENTITY_MIXED
                elif (ped, anchor) in eligible_keys:
                    w = eligible_keys[(ped, anchor)]
                    if not (predicted[w] and tracker_id[w] == t.track_id):
                        raise AssertionError("a consistent history must predict its eligible window")
                    category = None
                elif all(anchor + k * frame_step in gt_frames.get(ped, ()) for k in range(1, horizon + 1)):
                    category = NOT_ON_WINDOW_STRIDE
                else:
                    category = GT_FUTURE_INCOMPLETE
                histories.append(TrackerHistory(t.track_id, t.frames[s:s + h].copy(), labels.copy(), ped, category))

    summary = _summary(source, p, eligible, predicted, tracker_id, primary, flags, l0_all, per_step,
                       history_error, histories, association.tracks)
    return PredictionEvaluation(source, eligible, predicted, tracker_id, tuple(primary), tuple(flags), l0_all,
                                inputs, predictions, per_step, history_error, tuple(histories), summary)


def _summary(source, p, eligible, predicted, tracker_id, primary, flags, l0_all, per_step, history_error,
             histories, tracks) -> dict:
    n, idx = len(eligible), np.flatnonzero(predicted)
    n_pred = len(idx)
    scorable = sum(hh.category is None for hh in histories)
    l0_all_acc, l0_pred_acc = _accuracy(l0_all), _accuracy(per_step["L0"])

    per_ped = []
    for g in np.unique(eligible.track_ids):
        mine = eligible.track_ids == g
        mine_pred = mine[idx]
        entry = {"ped_id": int(g), "eligible_windows": int(mine.sum()), "predicted_windows": int(mine_pred.sum()),
                 "prediction_coverage": float(mine_pred.sum() / mine.sum()),
                 "mean_ade_L0_all_eligible_m": float(ade_fde(l0_all[mine])[0].mean()),
                 "lost_by_primary_reason": {r: sum(1 for w in np.flatnonzero(mine) if primary[w] == r)
                                            for r in LOST_REASONS}}
        for level in LEVELS:
            ade, fde = ade_fde(per_step[level][mine_pred]) if mine_pred.any() else (np.empty(0), np.empty(0))
            entry[f"mean_ade_{level}_m"] = float(ade.mean()) if len(ade) else None
            entry[f"mean_fde_{level}_m"] = float(fde.mean()) if len(fde) else None
        per_ped.append(entry)

    per_track = []
    for t in tracks:
        mine = [hh for hh in histories if hh.track_id == t.track_id]
        mine_pred = tracker_id[idx] == t.track_id
        ade, fde = ade_fde(per_step["L2"][mine_pred]) if mine_pred.any() else (np.empty(0), np.empty(0))
        per_track.append({
            "track_id": int(t.track_id), "grid_samples": int(len(t.frames)),
            "associated_ped_ids": sorted(int(v) for v in set(t.labels.tolist()) - {0}),
            "unassociated_samples": int(np.sum(t.labels == 0)),
            "histories": len(mine), "scorable_histories": sum(hh.category is None for hh in mine),
            "unscorable_histories": {c: sum(hh.category == c for hh in mine) for c in UNSCORABLE},
            "predicted_windows": int(mine_pred.sum()),
            "mean_ade_L2_m": float(ade.mean()) if len(ade) else None,
            "mean_fde_L2_m": float(fde.mean()) if len(fde) else None})

    return {
        "source": source,
        "note": NOTE,
        "protocol": p,
        "predictor": {"name": "CV", "cv_min_final_observations": p["cv_min_final_observations"]},
        "coverage": {
            "note": COVERAGE_NOTE,
            "eligible_ground_truth_windows": int(n),
            "available_tracker_histories": len(histories),
            "predicted_windows": int(n_pred),
            "prediction_coverage": float(n_pred / n) if n else None,
            "lost_windows": int(n - n_pred),
            "lost_by_primary_reason": {r: sum(1 for x in primary if x == r) for r in LOST_REASONS},
            "lost_windows_with_cause": {r: sum(1 for fl in flags if r in fl) for r in LOST_REASONS},
            "scorable_tracker_histories": int(scorable),
            "unscorable_tracker_histories": {"total": len(histories) - scorable,
                                             **{c: sum(hh.category == c for hh in histories) for c in UNSCORABLE}},
        },
        "accuracy_on_predicted_windows": {
            "note": LEVELS_NOTE,
            "levels": LEVELS,
            **{level: _accuracy(per_step[level]) for level in LEVELS},
            "history_input_error_m": {"mean_over_history": _stats(history_error.mean(axis=1)) if n_pred else None,
                                      "at_last_observation": _stats(history_error[:, -1]) if n_pred else None},
        },
        "survivorship_L0": {
            "note": "L0 (ground-truth input) on all eligible windows versus on the predicted windows only.",
            "all_eligible": l0_all_acc,
            "predicted_only": l0_pred_acc,
            "mean_ade_difference_predicted_minus_all_m": (
                l0_pred_acc["mean_ade_m"] - l0_all_acc["mean_ade_m"]
                if l0_pred_acc["mean_ade_m"] is not None and l0_all_acc["mean_ade_m"] is not None else None),
        },
        "per_pedestrian": per_ped,
        "per_tracker_track": per_track,
    }


def evaluate_prediction_on_scene(scene: SceneGroundTruth, tracker_result: TrackerResult, detections_per_frame,
                                 visible_fraction: np.ndarray, cfg: Config, source: str,
                                 min_visible_fraction: float = MIN_VISIBLE_FRACTION) -> PredictionEvaluation:
    expected = protocol(cfg)["sampling_interval_s"]
    if abs(scene.sampling_interval_s - expected) > 1e-12:
        raise ValueError(f"scene sampled at {scene.sampling_interval_s} s, configuration requires {expected} s")
    iou_threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    association = associate_grid(scene, tracker_result, detections_per_frame, visible_fraction, iou_threshold,
                                 min_visible_fraction)
    return evaluate_prediction(ground_truth_grid_tracks(scene), box_bottom_centre_grid_tracks(scene), association,
                               scene.frames_per_step, cfg, source)
