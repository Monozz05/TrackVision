"""Controlled identity-switch injection (D) on clean grid-sampled trajectories. Spec v4.15 section 3.5 D.

FEASIBILITY IMPLEMENTATION (U8): a controlled proxy applied downstream to the trajectory representation,
not a tracker error. The achieved severity is `switch_event_rate`, an injection rate; it is never the
tracker identity-switch metric of TrackEval and is never stored under that name.

Exchange (approved convention). An event (frame f, initiator a, partner b) exchanges the two identity
labels from timestep f onwards, inclusively: label a carries a's samples before f and b's samples from f
on, and label b the reverse. Every (frame, position) pair is unchanged, as are the number of tracks and
of observations; no identity is created and no observation is removed. `occupant[label]` gives, per
output sample, the original pedestrian occupying that label.

Eligibility. Timesteps are grid samples `frame_step` frames apart. A track frame (i, f) is proximity-
eligible if track i has samples at f - frame_step and f, and at least one other track j does too with
||p_i(f) - p_j(f)||_2 <= d_max. The primary denominator is the number of proximity-eligible track
frames of the clean base, computed before injection, so it is the same at every level and seed.

Sampling. One uniform u per track sample from SeedSequence([seed, 4, ped_id]) (stream 4 continues U7's
A = 1, B = 2, C = 3); the draws do not depend on p, so they are shared across levels. Timesteps are
visited in ascending order and, within one, tracks in ascending ped_id. A proximity-eligible track that
is not locked is exposed (the lock-aware exposure counts these); it triggers if u < p. The partner is the
nearest unlocked track among its proximity partners at f (ties: lower ped_id); if none is unlocked the
trigger is suppressed. Cooldown: both participants stay locked until the end of the segment, which is
the whole scene passed in, so no track takes part in two events. Realized events are therefore not
nested across levels even though the draws are.

Achieved severity:
  switch_event_rate = 100 * switch_events / eligible_track_frames   (events per 100 eligible track
  frames; None when there are no eligible track frames)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.inject.trajectory import _check, _probability
from trackvision.synthetic.tracks import GridTrack

D_STREAM = 4


def _frame_step(frame_step) -> int:
    if isinstance(frame_step, (bool, np.bool_)) or not isinstance(frame_step, (int, np.integer)) or frame_step < 1:
        raise ValueError(f"frame_step must be a positive integer, got {frame_step!r}")
    return int(frame_step)


def switch_draws(tracks, seed: int) -> dict[int, np.ndarray]:
    """One uniform per sample of each track; independent of p, so shared across levels."""
    return {int(t.ped_id): np.random.default_rng(np.random.SeedSequence([int(seed), D_STREAM, int(t.ped_id)]))
            .random(len(t.frames)) for t in _check(tracks)}


@dataclass(frozen=True)
class SwitchEvent:
    frame: int               # first frame carried under the exchanged labels
    initiator: int
    partner: int
    distance_m: float        # ||p_initiator(frame) - p_partner(frame)||


@dataclass(frozen=True)
class Proximity:
    candidates: dict         # (ped_id, frame) -> ((distance, partner), ...) sorted by distance, then id
    eligible_track_frames: int


def proximity(tracks, d_max: float, frame_step: int) -> Proximity:
    """Pre-injection proximity eligibility of every track frame (module docstring)."""
    tracks, frame_step = _check(tracks), _frame_step(frame_step)
    if not d_max >= 0.0:
        raise ValueError(f"d_max must be non-negative, got {d_max}")
    present: dict[int, dict[int, np.ndarray]] = {}       # frame -> {ped_id: position}, with frame - step too
    for t in tracks:
        frames = set(int(f) for f in t.frames)
        for f, p in zip(t.frames, t.positions):
            if int(f) - frame_step in frames:
                present.setdefault(int(f), {})[int(t.ped_id)] = p
    candidates = {}
    for f, peds in present.items():
        for i, pi in peds.items():
            near = sorted((float(np.linalg.norm(pi - pj)), j) for j, pj in peds.items() if j != i)
            near = tuple((d, j) for d, j in near if d <= d_max)
            if near:
                candidates[(i, f)] = near
    return Proximity(candidates, len(candidates))


@dataclass(frozen=True)
class SwitchSample:
    events: tuple            # SwitchEvent, in processing order
    achieved: dict


def sample_switches(tracks, p: float, seed: int, d_max: float, frame_step: int) -> SwitchSample:
    tracks, p = _check(tracks), _probability(p)
    prox = proximity(tracks, d_max, frame_step)
    draws = switch_draws(tracks, seed)
    sample = {int(t.ped_id): {int(f): k for k, f in enumerate(t.frames)} for t in tracks}
    locked: set[int] = set()
    events = []
    triggers = suppressed = exposure = 0
    for i, f in sorted(prox.candidates, key=lambda key: (key[1], key[0])):
        if i in locked:
            continue
        exposure += 1
        if draws[i][sample[i][f]] >= p:
            continue
        triggers += 1
        free = [(d, j) for d, j in prox.candidates[(i, f)] if j not in locked]
        if not free:
            suppressed += 1
            continue
        d, j = free[0]
        events.append(SwitchEvent(f, i, j, d))
        locked |= {i, j}
    n = prox.eligible_track_frames
    achieved = {
        "switch_events": len(events),
        "eligible_track_frames": n,
        "switch_event_rate": 100.0 * len(events) / n if n else None,
        "unit": "events per 100 pre-injection proximity-eligible track frames",
        "triggers": triggers,
        "suppressed_triggers": suppressed,
        "participating_tracks": 2 * len(events),
        "lock_aware_exposure": exposure,
        "lock_aware_exposure_note": ("proximity-eligible track frames whose track was unlocked when reached "
                                     "(timesteps ascending, ped_id ascending); supplementary diagnostic"),
    }
    return SwitchSample(tuple(events), achieved)


@dataclass(frozen=True)
class Relabelled:
    tracks: tuple            # GridTrack per label, ordered by label (= the input ids)
    occupant: dict           # label -> (n,) original ped_id occupying the label at each sample


def apply_switches(tracks, events, frame_step: int) -> Relabelled:
    """Exchange labels for each event. Raises if a track would take part in two events or if either
    track lacks a sample at event.frame - frame_step or event.frame."""
    tracks, frame_step = _check(tracks), _frame_step(frame_step)
    by_id ={int(t.ped_id): t for t in tracks}
    pairs: dict[int, tuple[int, int]] = {}               # label -> (partner, frame)
    for e in events:
        a, b, f = int(e.initiator), int(e.partner), int(e.frame)
        if a == b or a not in by_id or b not in by_id:
            raise ValueError(f"event {e}: needs two distinct existing tracks")
        if a in pairs or b in pairs:
            raise ValueError(f"event {e}: a track may take part in one event only (locked until scene end)")
        for t in (by_id[a], by_id[b]):
            if not (np.any(t.frames == f) and np.any(t.frames == f - frame_step)):
                raise ValueError(f"event {e}: track {t.ped_id} needs samples at {f - frame_step} and {f}")
        pairs[a], pairs[b] = (b, f), (a, f)
    out, occupant = [], {}
    for label in sorted(by_id):
        own = by_id[label]
        if label not in pairs:
            out.append(GridTrack(label, own.frames.copy(), own.positions.copy()))
            occupant[label] = np.full(len(own.frames), label, dtype=np.int64)
            continue
        other, f = pairs[label]
        before, after = own.frames < f, by_id[other].frames >= f
        out.append(GridTrack(label, np.concatenate([own.frames[before], by_id[other].frames[after]]),
                             np.concatenate([own.positions[before], by_id[other].positions[after]])))
        occupant[label] = np.concatenate([np.full(int(before.sum()), label), np.full(int(after.sum()), other)])
    return Relabelled(tuple(out), occupant)


@dataclass(frozen=True)
class IdentitySwitch:
    tracks: tuple
    occupant: dict
    events: tuple
    achieved: dict


def identity_switch(tracks, p: float, seed: int, d_max: float, frame_step: int) -> IdentitySwitch:
    sample = sample_switches(tracks, p, seed, d_max, frame_step)
    relabelled = apply_switches(tracks, sample.events, frame_step)
    return IdentitySwitch(relabelled.tracks, relabelled.occupant, sample.events, sample.achieved)
