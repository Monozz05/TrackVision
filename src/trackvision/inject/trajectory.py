"""Controlled injection on clean grid-sampled trajectories: A (localization noise), B (observation
dropout, with the B1 history reconstruction) and C (fragmentation). Spec v4.15 sections 3.5 and 3.6.

FEASIBILITY IMPLEMENTATION (U7): controlled proxies applied downstream to the trajectory
representation, not detector or tracker errors. D and E are not implemented here.

Every function returns new arrays and never modifies its input. A trajectory is corrupted once, so
overlapping windows share the same corrupted observations. Timesteps are grid samples (0.4 s).

Random numbers. Each (seed, family, ped_id) has its own stream, `SeedSequence([seed, family, ped_id])`,
so one pedestrian's draws do not depend on the others. Draws are shared across severity levels:
  A  one standard normal per coordinate, scaled by sigma (common Gaussian draws);
  B  one uniform u per observation, removed if u < p (masks are nested in p; B1 and B2 share them);
  C  one uniform u per eligible track frame, split if u < p (nested in p).
Within a level, observations stay independent, as the specification requires.

Achieved severity, over the whole clean base passed in:
  A  RMSE_pos = sqrt(mean over observations of ||p_corrupt - p_clean||^2), metres;
  B  removed_observations / eligible_observations, every clean observation being eligible;
  C  fragmentation_rate = split_events / eligible_track_frames, where a track of n samples has n - 2
     eligible frames (its first and last timesteps are excluded); fragments_per_original_track =
     (original_tracks + split_events) / original_tracks.
C convention: a split at timestep k (1 <= k <= n - 2) ends the identity at sample k; the next segment,
from k + 1, receives a new identity never used before. Fragments are never reconnected.

B1 reconstruction (`reconstruct_history`) is OFFLINE and NON-CAUSAL: linear interpolation uses later
observations. It is applied to one window's observed history at a time and sees nothing else, so it
never uses observations from the target period. Only interior gaps (a surviving observation on both
sides within that history) are filled; a missing first or last history point is an unresolved
boundary gap and is never extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.synthetic.tracks import GridTrack

FAMILY_STREAM = {"A": 1, "B": 2, "C": 3}


def _rng(seed: int, family: str, ped_id: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), FAMILY_STREAM[family], int(ped_id)]))


def _check(tracks) -> tuple[GridTrack, ...]:
    tracks = tuple(tracks)
    ids = [int(t.ped_id) for t in tracks]
    if len(set(ids)) != len(ids) or any(i < 1 for i in ids):
        raise ValueError("track ids must be unique positive integers")
    for t in tracks:
        if t.positions.shape != (len(t.frames), 2) or not np.all(np.isfinite(t.positions)):
            raise ValueError(f"track {t.ped_id}: positions must be finite with shape (n, 2)")
        frames = np.asarray(t.frames)
        if frames.ndim != 1 or np.any(frames[1:] <= frames[:-1]):
            raise ValueError(f"track {t.ped_id}: frames must be unique and strictly increasing")
    return tracks


def _probability(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must lie in [0, 1], got {p}")
    return float(p)


@dataclass(frozen=True)
class LocalizationNoise:
    tracks: tuple            # GridTrack with noisy positions (frames and ids unchanged)
    offsets: tuple           # (n, 2) per track: corrupted minus clean
    achieved: dict


def localization_noise(tracks, sigma: float, seed: int) -> LocalizationNoise:
    tracks = _check(tracks)
    if not (np.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"sigma must be finite and non-negative, got {sigma}")
    out, offsets = [], []
    for t in tracks:
        offset = sigma * _rng(seed, "A", t.ped_id).standard_normal((len(t.frames), 2))
        out.append(GridTrack(t.ped_id, t.frames.copy(), t.positions + offset))
        offsets.append(offset)
    sq = np.concatenate([np.sum(o ** 2, axis=1) for o in offsets]) if offsets else np.empty(0)
    achieved = {"rmse_pos_m": float(np.sqrt(sq.mean())) if len(sq) else None, "n_observations": int(len(sq))}
    return LocalizationNoise(tuple(out), tuple(offsets), achieved)


@dataclass(frozen=True)
class Dropout:
    present: tuple           # (n,) bool per track: True = observation survives
    achieved: dict


def dropout_masks(tracks, p: float, seed: int) -> Dropout:
    tracks, p = _check(tracks), _probability(p)
    present = tuple(~(_rng(seed, "B", t.ped_id).random(len(t.frames)) < p) for t in tracks)
    eligible = sum(len(t.frames) for t in tracks)
    removed = int(sum(np.sum(~m) for m in present))
    return Dropout(present, {"removed_observations": removed, "eligible_observations": int(eligible),
                             "rate": removed / eligible if eligible else None})


@dataclass(frozen=True)
class Reconstruction:
    positions: np.ndarray | None     # (H, 2) reconstructed history; None if a boundary gap is unresolved
    reconstructed: int               # interior observations filled by interpolation
    boundary_gap: bool               # the first or last history point is missing


def reconstruct_history(positions, present) -> Reconstruction:
    """B1 on ONE window's observed history: (H, 2) positions, (H,) survival mask. Missing entries are
    ignored (they may be NaN); nothing outside this history is available."""
    pos = np.asarray(positions, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if pos.ndim != 2 or pos.shape[1] != 2 or present.shape != (len(pos),) or len(pos) < 2:
        raise ValueError("history must be (H, 2) with an (H,) mask and H >= 2")
    if not np.all(np.isfinite(pos[present])):
        raise ValueError("surviving history positions must be finite")
    if not (present[0] and present[-1]):
        return Reconstruction(None, 0, True)
    known, missing = np.flatnonzero(present), np.flatnonzero(~present)
    out = np.where(present[:, None], pos, 0.0)
    for d in range(2):
        out[missing, d] = np.interp(missing, known, pos[known, d])
    return Reconstruction(out, int(len(missing)), False)


@dataclass(frozen=True)
class Fragmentation:
    tracks: tuple            # fragment GridTracks; the first fragment keeps the original id
    origin: dict             # fragment id -> (original ped_id, first sample, stop sample)
    achieved: dict


def fragment(tracks, p: float, seed: int) -> Fragmentation:
    tracks, p = _check(tracks), _probability(p)
    next_id = max((int(t.ped_id) for t in tracks), default=0) + 1
    out, origin = [], {}
    splits = eligible = 0
    for t in sorted(tracks, key=lambda t: t.ped_id):
        n = len(t.frames)
        k_eligible = max(0, n - 2)
        eligible += k_eligible
        after = 1 + np.flatnonzero(_rng(seed, "C", t.ped_id).random(k_eligible) < p)   # split after sample k
        splits += len(after)
        edges = [0] + [int(k) + 1 for k in after] + [n]
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            fid = int(t.ped_id) if j == 0 else next_id
            if j:
                next_id += 1
            out.append(GridTrack(fid, t.frames[a:b].copy(), t.positions[a:b].copy()))
            origin[fid] = (int(t.ped_id), a, b)
    n_tracks = len(tracks)
    achieved = {"split_events": int(splits), "eligible_track_frames": int(eligible),
                "fragmentation_rate": splits / eligible if eligible else None,
                "original_tracks": n_tracks,
                "fragments_per_original_track": (n_tracks + splits) / n_tracks if n_tracks else None}
    return Fragmentation(tuple(out), origin, achieved)
