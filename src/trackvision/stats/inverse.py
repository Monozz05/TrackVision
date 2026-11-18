"""Inverse requirement mapping (spec v4.15 section 2.2.6).

For one predictor, error family and scene: s* is the largest achieved severity such that D(u) <= q for every
observed or interpolated u from the clean condition up to s*, the largest contiguous acceptable prefix. Later
recovery below q does not restore the limit. The first crossing is linear between adjacent measured points, and
nothing is extrapolated.

Points are sorted by raw achieved severity. Adjacent points within 1e-12 become one point at the plain mean of
their values (decision 0004's tie rule, reused; decision 0006). Outcomes are kept apart:
  crossed       s* is the interpolated first crossing
  not_reached   the curve stays at or below q: report s* > maximum observed severity, with no number for s*
  truncated     an undefined point interrupts the prefix before any crossing: nothing is bridged, s* has no value,
                and `verified_up_to` says how far the curve was acceptable
  exceeded_at_clean  the first measured point is already above q, so there is no acceptable prefix to interpolate
  uncomputable  no usable clean point, including a zero clean baseline (section 3.4)

Descriptive. Not a universal engineering safety limit.
"""

from __future__ import annotations

import math
from typing import Sequence

TIE_TOLERANCE = 1e-12
CROSSED, NOT_REACHED, TRUNCATED, EXCEEDED_AT_CLEAN, UNCOMPUTABLE = (
    "crossed", "not_reached", "truncated", "exceeded_at_clean", "uncomputable")


def _merged(points: Sequence) -> list:
    """(x, y) sorted by x; tied x values merge to the plain mean of y, undefined if any member is undefined."""
    groups: list[list] = []
    for x, y in sorted(points, key=lambda p: p[0]):
        if groups and x - groups[-1][-1][0] <= TIE_TOLERANCE:
            groups[-1].append((x, y))
        else:
            groups.append([(x, y)])
    return [(g[0][0], None if any(y is None for _, y in g) else math.fsum(y for _, y in g) / len(g)) for g in groups]


def inverse_requirement(points: Sequence, q: float) -> dict:
    """`points` holds (achieved severity, relative ADE degradation); either may be None if undefined."""
    if not q > 0:
        raise ValueError("the requirement q must be positive")
    usable = [(x, y) for x, y in points if x is not None]
    unplaced = len(points) - len(usable)
    base = {"q": q, "s_star": None, "verified_up_to": None, "maximum_observed_severity": None,
            "points": len(points), "points_without_achieved_severity": unplaced}
    if not usable:
        return {**base, "status": UNCOMPUTABLE, "reason": "no point has an achieved severity"}
    curve = _merged(usable)
    base["maximum_observed_severity"] = curve[-1][0]
    if unplaced or curve[0][1] is None:
        why = "a point has no achieved severity" if unplaced else "the clean point is undefined"
        return {**base, "status": UNCOMPUTABLE, "reason": why}
    if curve[0][1] > q:
        return {**base, "status": EXCEEDED_AT_CLEAN, "reason": "degradation exceeds q at the first measured point"}
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if y1 is None:
            return {**base, "status": TRUNCATED, "verified_up_to": x0,
                    "reason": f"undefined degradation at achieved severity {x1}; not bridged"}
        if y1 > q:
            s = x0 if y0 == q else x0 + (q - y0) / (y1 - y0) * (x1 - x0)
            return {**base, "status": CROSSED, "s_star": s, "verified_up_to": s, "reason": None}
    return {**base, "status": NOT_REACHED, "verified_up_to": curve[-1][0],
            "reason": "the curve never exceeds q: s* > maximum observed severity"}
