"""Ground-truth tracks sampled on the protocol grid.

The grid is the one defined by scene.py: a single global grid anchored at frame 0, with step
`frames_per_step` (frame f is on the grid if f % frames_per_step == 0). Each pedestrian's active
interval is contiguous, so its grid samples are consecutive grid steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.synthetic.scene import SceneGroundTruth


@dataclass(frozen=True)
class GridTrack:
    """One pedestrian on the sampling grid, in the form `predict.windowing` expects."""

    ped_id: int
    frames: np.ndarray       # (n,) int64 frame numbers, multiples of frames_per_step
    positions: np.ndarray    # (n, 2) world metres (ground-contact point)


def ground_truth_grid_tracks(scene: SceneGroundTruth) -> tuple[GridTrack, ...]:
    """Every pedestrian's ground-truth grid samples, ordered by ped_id (empty if none on the grid)."""
    on_grid = scene.frame % scene.frames_per_step == 0
    tracks = []
    for pid in range(1, scene.config.n_pedestrians + 1):
        rows = on_grid & (scene.ped_id == pid)          # rows are already sorted by frame
        tracks.append(GridTrack(pid, scene.frame[rows].copy(), scene.foot_world[rows].copy()))
    return tuple(tracks)
