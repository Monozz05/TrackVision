"""Small synthetic multi-scene collections for the fold builder and the leakage checks (P2-c).

Each scene holds a few straight-line tracks on its own sampling grid, with positions offset per scene
so that a window spliced from another scene is detectable. Track ids repeat across scenes on purpose:
trajectory identity is (scene, track id). Scene names are synthetic labels, not dataset scenes.

`cyclic_rotation` is a TEST FIXTURE mapping (each scene validates on the next name in sorted order). It
is deliberately separate from the project's inner-validation rotation (`inner_validation_rotation` in
configs/base.yaml, decision 0001 C), which maps the five real static scenes only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.predict.windowing import static_windows
from trackvision.synthetic.tracks import GridTrack

FRAME_STEP = 10


@dataclass(frozen=True)
class FakeScene:
    name: str
    tracks: tuple
    frame_step: int = FRAME_STEP


def make_scene(name: str, index: int, lengths=(26, 23, 30), frame_step: int = FRAME_STEP) -> FakeScene:
    tracks = []
    for j, n in enumerate(lengths):
        k = np.arange(n, dtype=float)[:, None]
        origin = np.array([100.0 * index + 3.0 * j, -50.0 * index + j])
        velocity = np.array([0.4 + 0.1 * j, 0.05 * (index + 1)])
        start = frame_step * (5 * j + index)
        tracks.append(GridTrack(j + 1, start + frame_step * np.arange(n), origin + k * velocity))
    return FakeScene(name, tuple(tracks), frame_step)


def make_scenes(n: int) -> list[FakeScene]:
    return [make_scene(f"S{i + 1}", i) for i in range(n)]


def windows_of(scenes, cfg):
    return [static_windows(s, cfg) for s in scenes]


def cyclic_rotation(names) -> dict[str, str]:
    order = sorted(names)
    return {name: order[(i + 1) % len(order)] for i, name in enumerate(order)}
