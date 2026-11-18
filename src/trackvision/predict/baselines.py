"""Deterministic analytical predictors (spec v4.15 section 5.4).

Constant velocity (CV), using only the final two observed positions p_{H-1}, p_H:

    v_H = (p_H - p_{H-1}) / dt,   p_hat_{H+k} = p_H + k * dt * v_H = p_H + k * (p_H - p_{H-1}),
    k = 1 ... horizon_len.

Constant acceleration (CA), using only the final three observed positions p_{H-2}, p_{H-1}, p_H:

    v_H = (p_H - p_{H-1}) / dt,   a_H = (p_H - 2 p_{H-1} + p_{H-2}) / dt^2,
    p_hat_{H+k} = p_H + k dt v_H + 0.5 (k dt)^2 a_H = p_H + k d1 + 0.5 k^2 d2,
    d1 = p_H - p_{H-1},   d2 = p_H - 2 p_{H-1} + p_{H-2},   k = 1 ... horizon_len.

dt cancels, so the predictors need only the observed array. They never receive future positions.
"""

from __future__ import annotations

import numpy as np


def constant_velocity(observed, horizon_len: int, min_observations: int) -> np.ndarray:
    """Predict (N, horizon_len, 2) from observed histories (N, H, 2), H >= min_observations.

    `horizon_len` and `min_observations` come from the protocol configuration.
    """
    if not isinstance(min_observations, (int, np.integer)) or min_observations < 2:
        raise ValueError("constant velocity needs min_observations >= 2")
    if not isinstance(horizon_len, (int, np.integer)) or horizon_len < 1:
        raise ValueError(f"horizon_len must be a positive integer, got {horizon_len!r}")
    obs = np.asarray(observed, dtype=np.float64)
    if obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError(f"observed must have shape (N, H, 2), got {obs.shape}")
    if obs.shape[1] < min_observations:
        raise ValueError(f"history of {obs.shape[1]} points is shorter than the required {min_observations}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observed positions must be finite")
    last = obs[:, -1, :]
    step = obs[:, -1, :] - obs[:, -2, :]
    k = np.arange(1, horizon_len + 1, dtype=np.float64)
    return last[:, None, :] + k[None, :, None] * step[:, None, :]


def constant_acceleration(observed, horizon_len: int, min_observations: int) -> np.ndarray:
    """Predict (N, horizon_len, 2) from observed histories (N, H, 2), H >= min_observations.

    `horizon_len` and `min_observations` come from the protocol configuration.
    """
    if not isinstance(min_observations, (int, np.integer)) or min_observations < 3:
        raise ValueError("constant acceleration needs min_observations >= 3")
    if not isinstance(horizon_len, (int, np.integer)) or horizon_len < 1:
        raise ValueError(f"horizon_len must be a positive integer, got {horizon_len!r}")
    obs = np.asarray(observed, dtype=np.float64)
    if obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError(f"observed must have shape (N, H, 2), got {obs.shape}")
    if obs.shape[1] < min_observations:
        raise ValueError(f"history of {obs.shape[1]} points is shorter than the required {min_observations}")
    if not np.all(np.isfinite(obs)):
        raise ValueError("observed positions must be finite")
    last = obs[:, -1, :]
    d1 = obs[:, -1, :] - obs[:, -2, :]
    d2 = obs[:, -1, :] - 2.0 * obs[:, -2, :] + obs[:, -3, :]
    k = np.arange(1, horizon_len + 1, dtype=np.float64)
    return last[:, None, :] + k[None, :, None] * d1[:, None, :] + 0.5 * (k * k)[None, :, None] * d2[:, None, :]
