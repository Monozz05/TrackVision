"""Detection evaluation against ground-truth boxes (evaluation only; not a tracking assignment).

Matching rule, per frame:
  1. IoU between every detection and every ground-truth box (xyxy, continuous pixels).
  2. Greedy one-to-one matching: candidate pairs with IoU >= threshold are taken in descending IoU
     order (ties: lower detection index, then lower ground-truth index); each detection and each
     ground-truth box is used at most once.
  3. Detections are matched first to evaluable ground truth (true positives), then the remaining
     detections to ignore regions (ignored matches: neither TP nor FP).
  4. Remaining detections are false positives, split into those overlapping any visible ground-truth
     box (IoU > 0: typically merged or partial blobs) and background false positives.
  5. Unmatched evaluable ground truth are false negatives. Ignore regions never count as FN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _boxes(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if np.any(b[:, 2] < b[:, 0]) or np.any(b[:, 3] < b[:, 1]):
        raise ValueError("boxes must satisfy x2 >= x1 and y2 >= y1")
    return b


def clip_boxes(boxes, width: int, height: int) -> np.ndarray:
    b = _boxes(boxes).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, width)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, height)
    return b


def iou_matrix(a, b) -> np.ndarray:
    """(n, m) IoU; 0 where the union is empty."""
    a, b = _boxes(a), _boxes(b)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def greedy_match(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """One-to-one (row, column) pairs with IoU >= threshold, in the order of the rule above."""
    rows, cols = np.nonzero(iou >= threshold)
    order = np.lexsort((cols, rows, -iou[rows, cols]))
    used_r, used_c, pairs = set(), set(), []
    for k in order:
        r, c = int(rows[k]), int(cols[k])
        if r not in used_r and c not in used_c:
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
    return pairs


@dataclass(frozen=True)
class FrameCounts:
    n_detections: int
    n_evaluable: int
    n_ignore: int
    tp: int
    fn: int
    ignored_matches: int
    fp_overlapping_visible: int
    fp_background: int
    matches: tuple            # (detection index, ground-truth index) for true positives

    @property
    def fp(self) -> int:
        return self.fp_overlapping_visible + self.fp_background


def evaluate_frame(det_boxes, gt_boxes, evaluable, ignore, iou_threshold: float) -> FrameCounts:
    """Score one frame. `evaluable` and `ignore` are disjoint boolean masks over gt_boxes; ground truth
    in neither (invisible) is excluded entirely."""
    det = _boxes(det_boxes)
    gt = _boxes(gt_boxes)
    evaluable, ignore = np.asarray(evaluable, bool), np.asarray(ignore, bool)
    if np.any(evaluable & ignore):
        raise ValueError("evaluable and ignore must be disjoint")
    ev, ig = np.flatnonzero(evaluable), np.flatnonzero(ignore)

    tp_pairs = [(d, int(ev[g])) for d, g in greedy_match(iou_matrix(det, gt[ev]), iou_threshold)]
    left = np.array(sorted(set(range(len(det))) - {d for d, _ in tp_pairs}), dtype=np.int64)
    ignored = greedy_match(iou_matrix(det[left], gt[ig]), iou_threshold) if len(left) else []
    unmatched = np.array(sorted(set(left.tolist()) - {int(left[d]) for d, _ in ignored}), dtype=np.int64)

    visible = np.concatenate([ev, ig])
    if len(unmatched) and len(visible):
        overlaps = np.any(iou_matrix(det[unmatched], gt[visible]) > 0, axis=1)
    else:
        overlaps = np.zeros(len(unmatched), dtype=bool)
    return FrameCounts(
        n_detections=len(det), n_evaluable=len(ev), n_ignore=len(ig), tp=len(tp_pairs),
        fn=len(ev) - len(tp_pairs), ignored_matches=len(ignored),
        fp_overlapping_visible=int(overlaps.sum()), fp_background=int((~overlaps).sum()),
        matches=tuple(tp_pairs),
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """Each value is None when its denominator is zero."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
