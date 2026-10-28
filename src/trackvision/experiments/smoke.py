"""Synthetic smoke experiment: an end-to-end scaffold check with no scientific validity.

It builds deterministic synthetic trajectories whose properties are known analytically, using
only the frozen sampling protocol (history_len, horizon_len, sampling interval), and checks
those properties. It runs no research algorithm and reads no research choice.
"""

from __future__ import annotations

import hashlib

import numpy as np

from trackvision.config import Config

# Properties of the synthetic fixture only; these are not protocol constants.
WALKER_SPEED_MPS = 1.25
WALKER_HEADING_RAD = 0.0
STATIONARY_POSITION_M = (3.0, 4.0)
ANALYTIC_TOLERANCE = 1e-12


class SmokeCheckError(AssertionError):
    """Raised when a synthetic trajectory does not have its analytic properties."""


def static_window_shape(config: Config) -> tuple[int, float]:
    """Number of points in one static primary window, and the sampling interval."""
    static = config.frozen["protocol"]["static"]
    return static["history_len"] + static["horizon_len"], static["sampling_interval_s"]


def constant_velocity_walker(n_points: int, dt: float, speed: float = WALKER_SPEED_MPS,
                             heading: float = WALKER_HEADING_RAD) -> np.ndarray:
    """Positions (n_points, 2) in metres of a walker starting at the origin."""
    t = np.arange(n_points, dtype=np.float64) * dt
    direction = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
    return t[:, None] * speed * direction[None, :]


def stationary_point(n_points: int, position: tuple[float, float] = STATIONARY_POSITION_M) -> np.ndarray:
    return np.tile(np.asarray(position, dtype=np.float64), (n_points, 1))


def array_checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest()


def run(config: Config) -> dict:
    """Build the synthetic trajectories, check them, and return the fixture summary."""
    n_points, dt = static_window_shape(config)
    walker = constant_velocity_walker(n_points, dt)
    still = stationary_point(n_points)

    steps = np.linalg.norm(np.diff(walker, axis=0), axis=1)
    expected_step = WALKER_SPEED_MPS * dt
    recovered_speed = float(steps.mean() / dt)
    headings = np.arctan2(np.diff(walker[:, 1]), np.diff(walker[:, 0]))
    still_velocity = np.linalg.norm(np.diff(still, axis=0), axis=1) / dt

    checks = {
        "point_count_equals_history_plus_horizon": walker.shape == (n_points, 2),
        "constant_step_length": bool(np.all(np.abs(steps - expected_step) <= ANALYTIC_TOLERANCE)),
        "recovered_speed_equals_constructed": abs(recovered_speed - WALKER_SPEED_MPS) <= ANALYTIC_TOLERANCE,
        "constant_heading": bool(np.all(np.abs(headings - WALKER_HEADING_RAD) <= ANALYTIC_TOLERANCE)),
        "stationary_point_has_zero_velocity": bool(np.all(still_velocity == 0.0)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SmokeCheckError(f"synthetic fixture checks failed: {failed}")

    return {
        "point_count": n_points,
        "sampling_interval_s": dt,
        "walker_speed_mps": WALKER_SPEED_MPS,
        "expected_step_m": expected_step,
        "recovered_speed_mps": recovered_speed,
        "walker_checksum_sha256": array_checksum(walker),
        "stationary_checksum_sha256": array_checksum(still),
        "checks": checks,
    }
