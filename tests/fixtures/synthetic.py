"""Known-answer synthetic trajectories for tests.

Geometry comes from the frozen sampling protocol in base.yaml. The walker speed is a property
of this fixture, not a protocol constant. Expected values are closed-form, so code under test
is checked against analytic answers rather than against itself.
"""

import numpy as np

from trackvision.config import Config
from trackvision.experiments import smoke

OFFSET_M = 1.0


def build(config: Config) -> dict:
    n_points, dt = smoke.static_window_shape(config)
    walker = smoke.constant_velocity_walker(n_points, dt)
    return {
        "n_points": n_points,
        "dt": dt,
        "walker": walker,
        "stationary": smoke.stationary_point(n_points),
        # Reserved for Phase 2 metric tests: every point is exactly 1 m from the walker, so
        # ADE = FDE = RMSE = 1.0 against it (spec v4.15 section 11.2, "Metric correctness").
        "walker_offset_1m": walker + np.array([0.0, OFFSET_M]),
        "expected": {
            "step_m": smoke.WALKER_SPEED_MPS * dt,
            "speed_mps": smoke.WALKER_SPEED_MPS,
            "heading_rad": smoke.WALKER_HEADING_RAD,
            "offset_m": OFFSET_M,
        },
    }
