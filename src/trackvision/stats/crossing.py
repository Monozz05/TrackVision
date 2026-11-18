"""Threshold-crossing arithmetic from spec v4.15 section 9.3, and nothing else.

The first measured or linearly interpolated achieved severity at which one predictor's expected ADE equals
another's, for one error mode, at scene/fold-aggregated level.

THIS IS NOT ADAPTIVE MODEL SELECTION. Section 9.3 does not define how a system would estimate severity at run time,
which predictor it would then use, or what misclassification means, so none of that exists here: no function
chooses a predictor for any input, and the `runtime_selection` field is always "not_implemented" (decision 0006,
item 13). The interval section 9.3 asks for is not computed either: its constants are unspecified.

Outcomes are kept apart:
  crossed       the difference changes sign between two measured points, or is zero at a measured point
  no_crossing   one predictor stays below the other at every measured point; nothing is extrapolated
  truncated     an undefined value interrupts the curve before any crossing; nothing is bridged
  uncomputable  fewer than two usable points
"""

from __future__ import annotations

from typing import Sequence

CROSSED, NO_CROSSING, TRUNCATED, UNCOMPUTABLE = "crossed", "no_crossing", "truncated", "uncomputable"
NOT_IMPLEMENTED = "not_implemented"


def first_crossing(points: Sequence) -> dict:
    """`points` holds (achieved severity, ADE of a, ADE of b), with None where undefined."""
    base = {"runtime_selection": NOT_IMPLEMENTED, "s_crossing": None, "lower_before": None, "points": len(points),
            "reason": None}
    placed = sorted((p for p in points if p[0] is not None), key=lambda p: p[0])
    if len(placed) < 2 or len(placed) != len(points):
        return {**base, "status": UNCOMPUTABLE, "reason": "fewer than two usable points, or a point has no severity"}
    previous = None
    for x, a, b in placed:
        if a is None or b is None:
            if previous is None:
                return {**base, "status": UNCOMPUTABLE, "reason": "the clean point is undefined"}
            return {**base, "status": TRUNCATED, "reason": f"undefined ADE at achieved severity {x}; not bridged"}
        d = a - b
        if d == 0:
            return {**base, "status": CROSSED, "s_crossing": x, "lower_before": None, "measured_point": True}
        if previous is not None and previous[1] * d < 0:
            x0, d0 = previous
            return {**base, "status": CROSSED, "s_crossing": x0 + d0 / (d0 - d) * (x - x0),
                    "lower_before": "a" if d0 < 0 else "b", "measured_point": False}
        previous = (x, d)
    return {**base, "status": NO_CROSSING, "lower_before": "a" if previous[1] < 0 else "b",
            "reason": "no crossing within the measured severity range"}
