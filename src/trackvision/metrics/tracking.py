"""Project-specific tracking metrics for the feasibility pipeline.

These are CLEAR-MOT-style counts and a simplified IDF1 implemented here. They are NOT
MOTChallenge-compliant and NOT TrackEval results, and are not comparable to published benchmarks.

Ground truth per frame is split into evaluable, ignore and invisible rows (as in U4). Per frame:
  - tracker boxes are associated one-to-one with evaluable ground truth by `gated_iou_assignment` at
    the configured evaluation threshold (IoU >= 0.5; v4.15 section 3.7): true positives;
  - remaining tracker boxes are associated with ignore regions: ignored (neither TP nor FP);
  - remaining tracker boxes are FP; unmatched evaluable ground truth are FN.
Across frames, per ground-truth pedestrian (over its evaluable frames only):
  - IDSW: a match to a different track ID than at its previous match (CLEAR-MOT definition);
  - Frag: a match after one or more unmatched evaluable frames that followed an earlier match.
MOTA = 1 - (FN + FP + IDSW) / evaluable ground-truth observations (None if there are none).
Simplified IDF1: for every (pedestrian, track) pair count the frames where the track's box overlaps
the pedestrian's evaluable box with IoU >= threshold; a one-to-one pedestrian-track matching
maximising that count gives IDTP; IDFN = evaluable observations - IDTP; IDFP = non-ignored tracker
observations - IDTP; IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN).
Position error: Euclidean distance (metres) between a TP tracker position and the ground-truth foot point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.metrics.detection import iou_matrix
from trackvision.tracking.assignment import gated_iou_assignment, linear_assignment


@dataclass(frozen=True)
class FrameTracking:
    """One frame: tracker observations and ground truth (boxes xyxy, positions in metres)."""
    track_ids: np.ndarray        # (k,)
    track_boxes: np.ndarray      # (k, 4)
    track_positions: np.ndarray  # (k, 2)
    gt_ids: np.ndarray           # (g,)
    gt_boxes: np.ndarray         # (g, 4) clipped to the image
    gt_categories: np.ndarray    # (g,) "evaluable" | "ignore" | "invisible"
    gt_positions: np.ndarray     # (g, 2) ground-truth foot points


def evaluate_tracking(frames, iou_threshold: float) -> dict:
    tp = fp = fn = ignored = n_eval = n_track_obs = 0
    last_track: dict[int, int] = {}
    seen_match: set[int] = set()
    in_gap: set[int] = set()
    idsw = frag = 0
    pair_counts: dict[tuple[int, int], int] = {}
    errors = []
    track_ids_per_gt: dict[int, set] = {}

    for fr in frames:
        tb = np.asarray(fr.track_boxes, dtype=np.float64).reshape(-1, 4)
        cats = np.asarray(fr.gt_categories)
        ev, ig = np.flatnonzero(cats == "evaluable"), np.flatnonzero(cats == "ignore")
        n_eval += len(ev)
        n_track_obs += len(tb)

        tp_pairs = [(i, int(ev[g])) for i, g, _ in gated_iou_assignment(tb, fr.gt_boxes[ev], iou_threshold)]
        left = [i for i in range(len(tb)) if i not in {i for i, _ in tp_pairs}]
        ign = gated_iou_assignment(tb[left], fr.gt_boxes[ig], iou_threshold) if left else []
        tp += len(tp_pairs)
        ignored += len(ign)
        fp += len(left) - len(ign)
        fn += len(ev) - len(tp_pairs)

        matched_gt = {}
        for i, g in tp_pairs:
            gid, tid = int(fr.gt_ids[g]), int(fr.track_ids[i])
            matched_gt[gid] = tid
            errors.append(float(np.linalg.norm(fr.track_positions[i] - fr.gt_positions[g])))
            track_ids_per_gt.setdefault(gid, set()).add(tid)
        for g in ev:
            gid = int(fr.gt_ids[g])
            if gid in matched_gt:
                tid = matched_gt[gid]
                if gid in last_track and last_track[gid] != tid:
                    idsw += 1
                if gid in in_gap:
                    frag += 1
                    in_gap.discard(gid)
                last_track[gid] = tid
                seen_match.add(gid)
            elif gid in seen_match:
                in_gap.add(gid)

        if len(tb) and len(ev):
            overlap = iou_matrix(tb, fr.gt_boxes[ev]) >= iou_threshold
            for i, g in zip(*np.nonzero(overlap)):
                key = (int(fr.gt_ids[ev[g]]), int(fr.track_ids[i]))
                pair_counts[key] = pair_counts.get(key, 0) + 1

    idtp = 0
    if pair_counts:
        gts = sorted({g for g, _ in pair_counts})
        trs = sorted({t for _, t in pair_counts})
        counts = np.zeros((len(gts), len(trs)))
        for (g, t), c in pair_counts.items():
            counts[gts.index(g), trs.index(t)] = c
        idtp = int(sum(counts[r, c] for r, c in linear_assignment(-counts)))
    idfn = n_eval - idtp
    idfp = (n_track_obs - ignored) - idtp
    id_den = 2 * idtp + idfp + idfn
    errors = np.array(errors)
    per_gt = [len(s) for s in track_ids_per_gt.values()]
    return {
        "evaluable_observations": n_eval, "tracker_observations": n_track_obs,
        "tp": tp, "fp": fp, "fn": fn, "ignored": ignored, "idsw": idsw, "frag": frag,
        "mota": 1.0 - (fn + fp + idsw) / n_eval if n_eval else None,
        "idtp": idtp, "idfp": idfp, "idfn": idfn, "idf1": 2 * idtp / id_den if id_den else None,
        "position_rmse_m": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
        "position_median_error_m": float(np.median(errors)) if len(errors) else None,
        "track_ids_per_matched_pedestrian": ({"mean": float(np.mean(per_gt)), "max": int(np.max(per_gt))}
                                             if per_gt else None),
    }
