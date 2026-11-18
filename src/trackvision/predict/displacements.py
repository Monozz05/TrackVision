"""Frame-to-frame displacement encoding and reconstruction for the learned predictors (spec v4.15
section 5.4, 'Learning target - frame-to-frame future displacements').

With observed positions p_1 ... p_H and future positions p_{H+1} ... p_{H+F} (H = history_len,
F = horizon_len, both from the protocol configuration):

    encoder input    do_k = p_{k+1} - p_k              k = 1 ... H-1      (N, H-1, 2)
    decoder target   df_1 = p_{H+1} - p_H                                 (N, F, 2)
                     df_j = p_{H+j} - p_{H+j-1}       j = 2 ... F
    reconstruction   p_hat_{H+k} = p_H + sum_{j<=k} df_hat_j              (N, F, 2)

The first target bridges the observation boundary. Offsets from the last observed position
(p_{H+k} - p_H) are not used. Errors are always computed on reconstructed positions, never on
displacements. Inputs are never modified; invalid inputs raise and are never repaired.
"""

from __future__ import annotations

import numpy as np


def _points(name: str, value, min_len: int) -> np.ndarray:
    """A finite float64 copy of an (N, T, 2) array with T >= min_len (N = 0 is allowed)."""
    arr = np.array(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (N, T, 2), got {arr.shape}")
    if arr.shape[1] < min_len:
        raise ValueError(f"{name} needs at least {min_len} points per window, got {arr.shape[1]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def observed_displacements(observed) -> np.ndarray:
    """Encoder input (N, H-1, 2): p_{k+1} - p_k over the observed history (N, H, 2), H >= 2."""
    obs = _points("observed", observed, 2)
    return obs[:, 1:, :] - obs[:, :-1, :]


def future_displacements(observed, future) -> np.ndarray:
    """Decoder target (N, F, 2) from observed (N, H, 2) and future (N, F, 2) positions, F >= 1.

    The first displacement is future[:, 0] - observed[:, -1]; the rest are between consecutive futures.
    """
    obs = _points("observed", observed, 1)
    fut = _points("future", future, 1)
    if obs.shape[0] != fut.shape[0]:
        raise ValueError(f"observed has {obs.shape[0]} windows but future has {fut.shape[0]}")
    return np.diff(np.concatenate([obs[:, -1:, :], fut], axis=1), axis=1)


def reconstruct(last_observed, displacements) -> np.ndarray:
    """Absolute positions (N, F, 2) from the final observed positions (N, 2) and displacements (N, F, 2).

    p_hat_{H+k} = p_H + cumulative sum of the first k displacements.
    """
    last = np.array(last_observed, dtype=np.float64)
    if last.ndim != 2 or last.shape[-1] != 2:
        raise ValueError(f"last_observed must have shape (N, 2), got {last.shape}")
    if not np.all(np.isfinite(last)):
        raise ValueError("last_observed must be finite")
    steps = _points("displacements", displacements, 1)
    if last.shape[0] != steps.shape[0]:
        raise ValueError(f"last_observed has {last.shape[0]} windows but displacements has {steps.shape[0]}")
    return last[:, None, :] + np.cumsum(steps, axis=1)
