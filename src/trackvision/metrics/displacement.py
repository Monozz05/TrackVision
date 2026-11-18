"""Displacement error metrics (spec v4.15 section 3.1), in metres.

    DE[n, t] = || p_hat[n, t] - p[n, t] ||_2          per predicted step t = 1 ... T
    ADE[n]   = (1 / T) * sum_t DE[n, t]
    FDE[n]   = DE[n, T]                               the final predicted step
    RMSE[n]  = sqrt( (1 / T) * sum_t DE[n, t]^2 )      Euclidean form; never the per-coordinate form
"""

from __future__ import annotations

import numpy as np


def displacement_errors(predicted, target) -> np.ndarray:
    """Per-step Euclidean errors (N, T) for predictions and targets of identical shape (N, T, 2)."""
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape or pred.ndim != 3 or pred.shape[-1] != 2:
        raise ValueError(f"predicted {pred.shape} and target {true.shape} must both be (N, T, 2)")
    if pred.shape[1] < 1:
        raise ValueError("the horizon must contain at least one step")
    if not (np.all(np.isfinite(pred)) and np.all(np.isfinite(true))):
        raise ValueError("predicted and target positions must be finite")
    return np.linalg.norm(pred - true, axis=-1)


def ade_fde(per_step) -> tuple[np.ndarray, np.ndarray]:
    """ADE (N,) and FDE (N,) from per-step errors (N, T)."""
    de = np.asarray(per_step, dtype=np.float64)
    if de.ndim != 2 or de.shape[1] < 1:
        raise ValueError(f"per-step errors must have shape (N, T) with T >= 1, got {de.shape}")
    return de.mean(axis=1), de[:, -1].copy()


def rmse(per_step) -> np.ndarray:
    """Euclidean RMSE (N,) from per-step errors (N, T): the root of the mean squared Euclidean error."""
    de = np.asarray(per_step, dtype=np.float64)
    if de.ndim != 2 or de.shape[1] < 1:
        raise ValueError(f"per-step errors must have shape (N, T) with T >= 1, got {de.shape}")
    if not np.all(np.isfinite(de)) or np.any(de < 0):
        raise ValueError("per-step errors must be finite and non-negative")
    return np.sqrt(np.mean(de * de, axis=1))
