"""Achieved-severity monotonicity, as qualified by decision 0004, section 3 (Implementation Plan Phase 3 gate).

One scene, one condition; each seed and the seed mean are evaluated.
  A          strict increase across the requested levels. A value that fails to rise is a failure, so a
             degenerate zero-noise realization cannot pass.
  B1, B2, C  non-decreasing.
  D          diagnostic only. The required cooldown can lower achieved severity as requested severity rises, so
             a decrease is a finding to investigate, never an automatic failure, and the absence of one is
             "no_violation_observed", never a strict pass.

A series with an undefined value, or with fewer than two levels, is "not_evaluated": no check passes on nothing.
"""

from __future__ import annotations

from typing import Mapping

from trackvision.stats.aggregate import aggregate

PASS, FAIL, NOT_EVALUATED = "pass", "fail", "not_evaluated"
FINDING, NO_VIOLATION = "diagnostic_finding", "no_violation_observed"
RULES = {"A": "strict", "B1": "non_decreasing", "B2": "non_decreasing", "C": "non_decreasing", "D": "diagnostic"}


def _series(levels: tuple, values: dict, rule: str) -> dict:
    if len(levels) < 2 or any(values[lv] is None for lv in levels):
        return {"status": NOT_EVALUATED, "violations": [],
                "reason": "fewer than two levels" if len(levels) < 2 else "undefined achieved severity"}
    violations = [{"from": a, "to": b, "achieved_from": values[a], "achieved_to": values[b]}
                  for a, b in zip(levels, levels[1:])
                  if (values[b] <= values[a] if rule == "strict" else values[b] < values[a])]
    if rule == "diagnostic":
        return {"status": FINDING if violations else NO_VIOLATION, "violations": violations, "reason": None}
    return {"status": FAIL if violations else PASS, "violations": violations, "reason": None}


def check_monotonicity(scene: str, condition: str, achieved: Mapping, diagnostics: Mapping | None = None) -> dict:
    """`achieved[level][seed]`. `diagnostics[level][seed]` (for D: triggers, exposure, events) is carried along."""
    if condition not in RULES:
        raise ValueError(f"unknown condition {condition!r}")
    if not achieved:
        return {"scene": scene, "condition": condition, "rule": RULES[condition], "status": NOT_EVALUATED,
                "reason": "no observations", "per_seed": {}, "seed_mean": None}
    rule = RULES[condition]
    levels = tuple(sorted(achieved))
    seeds = sorted({s for row in achieved.values() for s in row})
    per_seed = {s: _series(levels, {lv: achieved[lv].get(s) for lv in levels}, rule) for s in seeds}
    means = {lv: aggregate(achieved[lv]).mean for lv in levels}
    seed_mean = _series(levels, means, rule)
    statuses = {r["status"] for r in per_seed.values()} | {seed_mean["status"]}
    if NOT_EVALUATED in statuses:
        status = NOT_EVALUATED
    elif rule == "diagnostic":
        status = FINDING if FINDING in statuses else NO_VIOLATION
    else:
        status = FAIL if FAIL in statuses else PASS
    out = {"scene": scene, "condition": condition, "rule": rule, "status": status, "levels": list(levels),
           "per_seed": {str(s): r for s, r in per_seed.items()}, "seed_mean": seed_mean}
    if diagnostics:
        out["diagnostics"] = {str(lv): {str(s): v for s, v in row.items()} for lv, row in diagnostics.items()}
    return out
