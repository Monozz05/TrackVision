"""Prediction windows: history_len observed + horizon_len future points, cut from tracks sampled
on the 0.4 s grid (spec v4.15 sections 5.3 and 7.2).

A window is valid only if all history_len + horizon_len points belong to one track and are
consecutive samples (each frame difference equals the scene's frame step). Gaps split a track
into runs; windows never span a gap. Within each run, windows start at the run's first sample
and advance by the stride. Protocol lengths and the stride come from configs/base.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from trackvision.config import Config


@dataclass(frozen=True)
class Windows:
    scene: str
    history_len: int
    horizon_len: int
    stride: int
    observed: np.ndarray      # (N, history_len, 2) world metres
    target: np.ndarray        # (N, horizon_len, 2) world metres
    track_ids: np.ndarray     # (N,) int64
    frames: np.ndarray        # (N, history_len + horizon_len) int64 source frame numbers

    def __len__(self) -> int:
        return len(self.track_ids)


def consecutive_runs(frames: np.ndarray, frame_step: int) -> list[tuple[int, int]]:
    """Half-open index ranges [start, stop) of maximal runs with frame difference == frame_step."""
    frames = np.asarray(frames)
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != frame_step) + 1
    edges = np.concatenate([[0], breaks, [len(frames)]])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def make_windows(scene: str, tracks: Iterable, frame_step: int, history_len: int,
                 horizon_len: int, stride: int) -> Windows:
    """Cut windows from tracks with `ped_id`, `frames` (n,) and `positions` (n, 2)."""
    for name, value in (("history_len", history_len), ("horizon_len", horizon_len),
                        ("stride", stride), ("frame_step", frame_step)):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    length = history_len + horizon_len
    observed, target, ids, frames = [], [], [], []
    for track in tracks:
        for start, stop in consecutive_runs(track.frames, frame_step):
            for s in range(start, stop - length + 1, stride):
                pts = track.positions[s:s + length]
                observed.append(pts[:history_len])
                target.append(pts[history_len:])
                ids.append(track.ped_id)
                frames.append(track.frames[s:s + length])
    return Windows(
        scene=scene,
        history_len=history_len,
        horizon_len=horizon_len,
        stride=stride,
        observed=np.asarray(observed, dtype=np.float64).reshape(-1, history_len, 2),
        target=np.asarray(target, dtype=np.float64).reshape(-1, horizon_len, 2),
        track_ids=np.asarray(ids, dtype=np.int64),
        frames=np.asarray(frames, dtype=np.int64).reshape(-1, length),
    )


def static_windows(scene, config: Config) -> Windows:
    """Windows for a static-arm scene under the frozen static primary protocol (8/12, stride 1)."""
    static = config.frozen["protocol"]["static"]
    return make_windows(scene.name, scene.tracks, scene.frame_step, static["history_len"],
                        static["horizon_len"], static["window_stride"])
