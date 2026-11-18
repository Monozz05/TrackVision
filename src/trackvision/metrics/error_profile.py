"""Project diagnostics of a tracker's error profile, from one-to-one tracker/ground-truth matches.

NOT official metrics and NOT the section 3.7 achieved severities of a real-tracker evaluation. Official IDSW and Frag
come from TrackEval only. `identity_exchanges` here is a different quantity from TrackEval's IDSW and from the
injector's `switch_event_rate`, and the three are never interchangeable (section 3.7).

The counting follows section 3.7's definitions over the given cadence. For each ground-truth identity, consecutive
matches under different tracker ids form a transition, classified once:
  identity exchange  two identities exchange the same two tracker ids, within `switch_gap_steps` missing timesteps
                     (only clean two-object exchanges count; each transition is used once)
  fragmentation      otherwise, the old id ends for this identity and a new id follows after 1 to `frag_gap_steps`
                     unmatched eligible timesteps
  other              anything else: a direct hand-over with no gap, a longer gap, or an old id that comes back

Every value states its unit and denominator. Position units are the caller's: pixels for MOT17, metres for synthetic
scenes. Nothing converts one to the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from trackvision.metrics.tracking import FrameTracking
from trackvision.tracking.assignment import gated_iou_assignment


@dataclass(frozen=True)
class MatchedFrame:
    index: int                  # consecutive position on the evaluated cadence
    eligible: tuple             # ground-truth ids that count at this timestep
    assignment: Mapping         # ground-truth id -> tracker id
    errors: Mapping             # ground-truth id -> position error, in the caller's unit


@dataclass(frozen=True)
class Transition:
    gt: int
    before: int                 # cadence index of the last match under `old`
    after: int                  # cadence index of the first match under `new`
    gap: int                    # eligible timesteps of this identity left unmatched in between
    old: int
    new: int


def match_frames(frames: Iterable[FrameTracking], iou_threshold: float) -> list[MatchedFrame]:
    """One-to-one matching per frame: evaluable ground truth first, then ignore regions, which are not counted."""
    out = []
    for k, fr in enumerate(frames):
        boxes = np.asarray(fr.track_boxes, dtype=np.float64).reshape(-1, 4)
        ev = np.flatnonzero(np.asarray(fr.gt_categories) == "evaluable")
        assignment, errors = {}, {}
        for i, g, _ in gated_iou_assignment(boxes, fr.gt_boxes[ev], iou_threshold):
            row = int(ev[g])
            gid = int(fr.gt_ids[row])
            assignment[gid] = int(fr.track_ids[i])
            errors[gid] = float(np.linalg.norm(fr.track_positions[i] - fr.gt_positions[row]))
        out.append(MatchedFrame(k, tuple(int(fr.gt_ids[g]) for g in ev), assignment, errors))
    return out


def transitions(frames: Iterable[MatchedFrame]) -> list[Transition]:
    seen: dict[int, tuple] = {}                 # gt -> (eligible count at last match, index, tracker id)
    count: dict[int, int] = {}
    out = []
    for fr in frames:
        for gid in fr.eligible:
            count[gid] = count.get(gid, 0) + 1
            if gid not in fr.assignment:
                continue
            tid = fr.assignment[gid]
            if gid in seen and seen[gid][2] != tid:
                out.append(Transition(gid, seen[gid][1], fr.index, count[gid] - seen[gid][0] - 1, seen[gid][2], tid))
            seen[gid] = (count[gid], fr.index, tid)
    return out


def identity_exchanges(moves: list[Transition], switch_gap_steps: int) -> list[tuple]:
    """Disjoint pairs of transitions in which two identities exchange the same two tracker ids."""
    ordered = sorted(moves, key=lambda m: (m.after, m.gt))
    used: set[int] = set()
    pairs = []
    for i, a in enumerate(ordered):
        if i in used:
            continue
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if j in used or a.gt == b.gt or (a.old, a.new) != (b.new, b.old):
                continue
            if max(a.after, b.after) - min(a.before, b.before) - 1 <= switch_gap_steps:
                used.update((i, j))
                pairs.append((a, b))
                break
    return pairs


def _rate(events: int, denominator: int | None, scale: float, reason: str | None = None) -> dict:
    if denominator is None:
        return {"value": None, "reason": reason or "denominator unavailable"}
    if denominator == 0:
        return {"value": None, "reason": "zero denominator"}
    return {"value": scale * events / denominator, "reason": None}


def track_frame_denominator(frames: Iterable[MatchedFrame]) -> int:
    """Section 3.5 C convention: an identity with n eligible samples has n - 2 eligible track frames."""
    n: dict[int, int] = {}
    for fr in frames:
        for gid in fr.eligible:
            n[gid] = n.get(gid, 0) + 1
    return sum(max(v - 2, 0) for v in n.values())


def error_profile(frames: list[MatchedFrame], frag_gap_steps: int, switch_gap_steps: int, unit: str,
                  fragmentation_denominator: int, exchange_denominator: int | None = None,
                  exchange_denominator_name: str | None = None, exchange_reason: str | None = None) -> dict:
    """Counts, rates and denominators for one sequence or scene. Never pooled by this function."""
    eligible = sum(len(fr.eligible) for fr in frames)
    matched = sum(len(fr.errors) for fr in frames)
    squares = [e ** 2 for fr in frames for e in fr.errors.values()]
    moves = transitions(frames)
    pairs = identity_exchanges(moves, switch_gap_steps)
    paired = {id(m) for pair in pairs for m in pair}
    later: dict[tuple, int] = {}
    for fr in frames:
        for gid, tid in fr.assignment.items():
            later[(gid, tid)] = fr.index
    frag = long_gap = direct = returned = 0
    for m in moves:
        if id(m) in paired:
            continue
        if later[(m.gt, m.old)] > m.after:
            returned += 1
        elif m.gap == 0:
            direct += 1
        elif m.gap > frag_gap_steps:
            long_gap += 1
        else:
            frag += 1
    return {
        "localization": {"rmse": math.sqrt(math.fsum(squares) / len(squares)) if squares else None, "unit": unit,
                         "denominator": "matched_observations", "matched_observations": matched},
        "dropout": {"unmatched_gt_observations": eligible - matched, "eligible_gt_observations": eligible,
                    "rate": _rate(eligible - matched, eligible, 1.0)},
        "fragmentation": {"events": frag, "gap_limit_steps": frag_gap_steps, "denominator": "eligible_track_frames",
                          "eligible_track_frames": fragmentation_denominator,
                          "rate": _rate(frag, fragmentation_denominator, 1.0)},
        "identity_exchange": {"events": len(pairs), "gap_limit_steps": switch_gap_steps,
                              "note": "two-object exchanges; not TrackEval IDSW and not switch_event_rate",
                              "denominator": exchange_denominator_name, "denominator_value": exchange_denominator,
                              "per_100": _rate(len(pairs), exchange_denominator, 100.0, exchange_reason),
                              "per_100_gt_observations": _rate(len(pairs), eligible, 100.0)},
        "other_identity_changes": {"direct_hand_over": direct, "after_long_gap": long_gap, "old_id_returned": returned,
                                   "transitions_total": len(moves)},
    }
