"""One-to-one assignment (Hungarian algorithm, numpy only) and gated IoU assignment.

`linear_assignment(cost)` returns a minimum-total-cost one-to-one matching of rows to columns for any
rectangular matrix (min(n, m) pairs). It is deterministic: rows are processed in index order and
column scans pick the lowest index among equal candidates.

`gated_iou_assignment(a, b, min_iou)` maximises the total *gated* IoU, where a pair's gated IoU is its
IoU if IoU >= min_iou and 0 otherwise, then drops pairs whose gated IoU is 0. A pair below the gate
is therefore never matched, and one strong match is preferred to two weak ones (the common SORT-style
convention), rather than maximising the number of matched pairs.
"""

from __future__ import annotations

import numpy as np

from trackvision.metrics.detection import iou_matrix


def linear_assignment(cost) -> list[tuple[int, int]]:
    """Minimum-cost one-to-one (row, column) pairs, sorted by row."""
    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {c.shape}")
    if c.size == 0:
        return []
    if not np.all(np.isfinite(c)):
        raise ValueError("cost must be finite")
    transposed = c.shape[0] > c.shape[1]
    a = c.T if transposed else c
    n, m = a.shape                                   # n <= m
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)              # p[j]: row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = a[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            candidates = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1      # lowest index among equal minima
            delta = candidates[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(col, row) for row, col in pairs]
    return sorted(pairs)


def gated_assignment(iou, min_iou: float) -> list[tuple[int, int, float]]:
    """(row, column, IoU) for the matching maximising total gated IoU; only IoU >= min_iou kept."""
    iou = np.asarray(iou, dtype=np.float64)
    if iou.size == 0:
        return []
    gated = np.where(iou >= min_iou, iou, 0.0)
    return [(i, j, float(iou[i, j])) for i, j in linear_assignment(1.0 - gated) if gated[i, j] > 0]


def gated_iou_assignment(boxes_a, boxes_b, min_iou: float) -> list[tuple[int, int, float]]:
    return gated_assignment(iou_matrix(boxes_a, boxes_b), min_iou)
