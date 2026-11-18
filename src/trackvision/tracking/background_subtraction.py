"""Background-subtraction blob detector for a static camera (feasibility baseline; not a person detector).

Pipeline per frame:
  1. foreground = |frame - background| > diff_threshold   (grey levels, strict)
  2. connected components of the foreground (8- or 4-connectivity)
  3. one detection per component with at least min_area_px pixels

Boxes are xyxy in the camera pixel convention (pixel x covers [x, x+1)): a component spanning columns
x_min..x_max and rows y_min..y_max has box [x_min, y_min, x_max + 1, y_max + 1]. No scores are
produced. Overlapping pedestrians merge into one component; nothing splits them.

Background models: `median_background` estimates the static background as the per-pixel median of
evenly spaced frames of the same video. It assumes every pixel is uncovered in more than half of the
sampled frames; otherwise a stationary object leaves a ghost in the background.

Default thresholds are fixed in advance from the U3 rendering design (noise sigma 2, pedestrian
contrast >= 38 grey levels), not tuned on any evaluation score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BgSubConfig:
    diff_threshold: float = 20.0      # grey levels: ~9 sigma of a noisy difference, below the >= 38 contrast
    min_area_px: int = 50             # half-visible distant pedestrians are >= ~500 px
    background_frames: int = 50       # frames sampled evenly for the median background
    connectivity: int = 8

    def __post_init__(self) -> None:
        if not self.diff_threshold > 0:
            raise ValueError("diff_threshold must be positive")
        if not (isinstance(self.min_area_px, int) and self.min_area_px >= 1):
            raise ValueError("min_area_px must be a positive integer")
        if not (isinstance(self.background_frames, int) and self.background_frames >= 1):
            raise ValueError("background_frames must be a positive integer")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass(frozen=True)
class Detections:
    boxes: np.ndarray     # (k, 4) float64 xyxy
    areas: np.ndarray     # (k,) int64 foreground pixels per component


def sample_indices(frame_count: int, n_samples: int) -> np.ndarray:
    """Evenly spaced, unique frame indices in [0, frame_count - 1]."""
    if frame_count < 1 or n_samples < 1:
        raise ValueError("frame_count and n_samples must be positive")
    return np.unique(np.round(np.linspace(0, frame_count - 1, min(n_samples, frame_count))).astype(np.int64))


def median_background(frames: Iterable[np.ndarray], frame_count: int, n_samples: int) -> np.ndarray:
    """Per-pixel median (float64) of the evenly spaced sample of a frame sequence of length frame_count."""
    wanted = set(sample_indices(frame_count, n_samples).tolist())
    picked = [np.asarray(frame, dtype=np.float64) for k, frame in enumerate(frames) if k in wanted]
    if len(picked) != len(wanted):
        raise ValueError(f"expected {len(wanted)} sampled frames, got {len(picked)}; is frame_count right?")
    return np.median(np.stack(picked), axis=0)


def foreground_mask(frame: np.ndarray, background: np.ndarray, threshold: float) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float64)
    if frame.shape != background.shape:
        raise ValueError(f"frame {frame.shape} and background {background.shape} differ in shape")
    return np.abs(frame - background) > threshold


def connected_components(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (k, 4) xyxy and areas (k,) of the mask's components, sorted by (y_min, x_min)."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.zeros((height, width + 2), dtype=np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    run_row, run_start = np.nonzero(d == 1)          # row-major order: starts and ends pair up
    _, run_end = np.nonzero(d == -1)                 # exclusive end column
    n = len(run_row)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reach = 1 if connectivity == 8 else 0            # 8-connectivity also joins diagonal neighbours
    row_first = np.searchsorted(run_row, np.arange(height + 1))
    for r in range(1, height):
        prev = range(row_first[r - 1], row_first[r])
        for i in range(row_first[r], row_first[r + 1]):
            for j in prev:
                if run_start[i] < run_end[j] + reach and run_start[j] < run_end[i] + reach:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[max(a, b)] = min(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    boxes, areas = [], []
    for members in groups.values():
        m = np.array(members)
        boxes.append([run_start[m].min(), run_row[m].min(), run_end[m].max(), run_row[m].max() + 1])
        areas.append(int((run_end[m] - run_start[m]).sum()))
    if not boxes:
        return np.empty((0, 4)), np.empty(0, dtype=np.int64)
    boxes = np.array(boxes, dtype=np.float64)
    areas = np.array(areas, dtype=np.int64)
    order = np.lexsort((boxes[:, 0], boxes[:, 1]))
    return boxes[order], areas[order]


def detect(frame: np.ndarray, background: np.ndarray, config: BgSubConfig = BgSubConfig()) -> Detections:
    boxes, areas = connected_components(foreground_mask(frame, background, config.diff_threshold),
                                        config.connectivity)
    keep = areas >= config.min_area_px
    return Detections(boxes[keep], areas[keep])
