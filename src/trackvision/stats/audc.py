"""AUDC, family AUDC and the robustness score (spec v4.15 section 3.4, sections 3.3 and 5.4; decision 0004).

The input is one curve for one predictor, scene/fold and condition: `inject.severity.NormalizedSeverity`, whose
s_norm values are never clipped or rescaled here, and one seed-aggregated degradation value per requested level.
AUDC is computed once from those aggregated values, never per seed and then averaged.

Steps (decision 0004, section 1):
  - a point within 1e-12 of 1 counts as 1; points above 1 stay in the record, undefined ones included, but do not
    affect the integral;
  - an undefined value at or below 1 makes AUDC uncomputable; nothing is interpolated or extrapolated;
  - points are sorted by s_norm, and adjacent points within 1e-12 (absolute) become one point at the plain mean
    of their y values, the originals staying in the record;
  - the trapezoidal rule runs over the remaining points on [0, 1].

The highest requested level has s_norm exactly 1, so the integral always ends on a measured point; a defined
curve needs no boundary interpolation. s_norm below 0 needs s_0 above an achieved value, which cannot happen for
A to D (s_0 = 0). Decision 0004 does not cover it, so it is reported as uncomputable rather than resolved here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from trackvision.inject.severity import COMPUTABLE, UNCOMPUTABLE, NormalizedSeverity

TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AudcResult:
    status: str
    value: float | None
    reason: str | None
    points: tuple = field(default_factory=tuple)     # every observation, used in the integral or not
    points_above_one: int = 0
    tie_groups: tuple = field(default_factory=tuple)
    undefined_levels: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {"status": self.status, "value": self.value, "reason": self.reason, "points": list(self.points),
                "points_above_one": self.points_above_one, "tie_groups": [list(g) for g in self.tie_groups],
                "undefined_levels": list(self.undefined_levels)}


def audc(norm: NormalizedSeverity, y_by_level: Mapping, reasons: Mapping | None = None) -> AudcResult:
    """AUDC of one curve. `y_by_level` maps each requested level to its aggregated D, or None if undefined."""
    reasons = reasons or {}
    if sorted(y_by_level) != sorted(norm.levels):
        raise ValueError(f"levels {sorted(y_by_level)} do not match the severity levels {sorted(norm.levels)}")
    if norm.status != COMPUTABLE:
        points = tuple({"level": lv, "s_norm": None, "y": y_by_level[lv], "reason": reasons.get(lv), "used": False}
                       for lv in norm.levels)
        return AudcResult(UNCOMPUTABLE, None, norm.reason, points)

    points, inside = [], []
    for level in norm.levels:
        raw = norm.normalized[level]
        x = 1.0 if abs(raw - 1.0) <= TIE_TOLERANCE else raw
        y = y_by_level[level]
        used = x <= 1.0
        points.append({"level": level, "s_norm": raw, "y": y, "reason": reasons.get(level), "used": used})
        if used:
            inside.append((x, y, level))
    above = len(points) - len(inside)
    if any(x < -TIE_TOLERANCE for x, _, _ in inside):
        return AudcResult(UNCOMPUTABLE, None, "s_norm below 0 is not covered by decision 0004", tuple(points), above)
    undefined = tuple(level for _, y, level in inside if y is None)
    if undefined:
        why = sorted({str(reasons.get(lv, "undefined")) for lv in undefined})
        return AudcResult(UNCOMPUTABLE, None, f"undefined degradation at or below s_norm 1: {why}", tuple(points),
                          above, (), undefined)

    inside.sort(key=lambda p: (p[0], p[2]))
    groups: list[list] = [[inside[0]]]
    for point in inside[1:]:
        if point[0] - groups[-1][-1][0] <= TIE_TOLERANCE:
            groups[-1].append(point)
        else:
            groups.append([point])
    curve = [(g[0][0], math.fsum(p[1] for p in g) / len(g)) for g in groups]
    ties = tuple(tuple(p[2] for p in g) for g in groups if len(g) > 1)
    if curve[0][0] > TIE_TOLERANCE or curve[-1][0] != 1.0:
        return AudcResult(UNCOMPUTABLE, None, "the curve does not cover [0, 1]", tuple(points), above, ties)
    value = math.fsum((x2 - x1) * (y1 + y2) / 2.0 for (x1, y1), (x2, y2) in zip(curve, curve[1:]))
    return AudcResult(COMPUTABLE, value, None, tuple(points), above, ties)


def family_audc(by_predictor: Mapping[str, AudcResult]) -> dict:
    """Mean of the valid predictor AUDCs (section 5.4); uncomputable below two. No predictor is omitted."""
    valid = {p: r.value for p, r in by_predictor.items() if r.status == COMPUTABLE}
    invalid = {p: r.reason for p, r in by_predictor.items() if r.status != COMPUTABLE}
    base = {"n_valid": len(valid), "n_undefined": len(invalid), "valid": dict(sorted(valid.items())),
            "undefined": dict(sorted(invalid.items()))}
    if len(valid) < 2:
        return {**base, "status": UNCOMPUTABLE, "value": None, "reason": "fewer than two valid predictor AUDC values"}
    return {**base, "status": COMPUTABLE, "value": math.fsum(valid.values()) / len(valid), "reason": None}


def robustness(audc_value: float | None) -> dict:
    """R = 1 / (1 + AUDC). Retained above 1, never clipped; undefined at AUDC = -1 (section 3.4)."""
    if audc_value is None:
        return {"value": None, "reason": "AUDC is uncomputable"}
    if audc_value == -1.0:
        return {"value": None, "reason": "AUDC = -1"}
    return {"value": 1.0 / (1.0 + audc_value), "reason": None}
