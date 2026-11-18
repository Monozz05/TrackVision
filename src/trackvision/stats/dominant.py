"""Dominant failure-mode selection logic (spec v4.15 section 2.2.3), as an engineering dry-run only.

Steps 1 to 5 of the frozen rule: the candidates are A, B2, C and D; only finite, positive family AUDC values
qualify; the largest is the target; the two largest are co-dominant if they differ by less than the frozen gap; no
qualifying value means no target and no mitigation. E is rejected outright: it is never compared against A to D.

THIS MODULE CANNOT MAKE THE RESEARCH SELECTION. Section 2.2.3 applies the rule once, on valid P-4 to P-8 results, and
stores it in provenance. Those results do not exist. Every result here has status `engineering_dry_run`, and nothing
is written or remembered, so calling it spends nothing (decision 0006, item 9). There is deliberately no argument
that changes this.

A family whose feasibility gate failed is passed in `excluded` and the exclusion is recorded (step 4).

GAP_TOLERANCE is a project-level numerical convention (decision 0006), not part of section 2.2.3 and no amendment of
v4.15. A gap within 1e-12 of the frozen value counts as equal to it, so binary rounding cannot manufacture
co-dominance: 0.3 - 0.2 is slightly below 0.1 in floating point, but the rule says "less than".
"""

from __future__ import annotations

import math
from typing import Mapping

ENGINEERING_DRY_RUN = "engineering_dry_run"
CANDIDATES = ("A", "B2", "C", "D")
GAP_TOLERANCE = 1e-12
NOT_THE_RESEARCH_SELECTION = ("Engineering dry-run of the section 2.2.3 logic. It is not the research selection, which "
                              "is applied once on valid P-4 to P-8 results, and it does not consume that application.")


def _candidate(value, reason, excluded) -> dict:
    if excluded is not None:
        return {"value": value, "qualifies": False, "reason": f"excluded: {excluded}"}
    if value is None:
        return {"value": None, "qualifies": False, "reason": reason or "undefined"}
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"family AUDC must be a number or None, got {value!r}")
    if not math.isfinite(value):
        return {"value": None, "qualifies": False, "reason": f"not finite: {value!r}"}
    if value <= 0:
        return {"value": float(value), "qualifies": False, "reason": "not positive: did not degrade prediction"}
    return {"value": float(value), "qualifies": True, "reason": None}


def select_dominant_mode(family_audc: Mapping, gap: float, reasons: Mapping | None = None,
                         excluded: Mapping | None = None) -> dict:
    """`family_audc` maps condition to its value; a missing condition is undefined. `gap` is the frozen constant."""
    unknown = set(family_audc) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"only {CANDIDATES} are candidates; got {sorted(unknown)}. E is never compared with A to D")
    reasons, excluded = reasons or {}, excluded or {}
    candidates = {c: _candidate(family_audc.get(c), reasons.get(c) or ("family_missing" if c not in family_audc else None),
                                excluded.get(c)) for c in CANDIDATES}
    ranked = sorted((c for c in CANDIDATES if candidates[c]["qualifies"]), key=lambda c: (-candidates[c]["value"], c))
    top_gap = candidates[ranked[0]]["value"] - candidates[ranked[1]]["value"] if len(ranked) > 1 else None
    co_dominant = top_gap is not None and top_gap < gap - GAP_TOLERANCE
    targets = ranked[:2] if co_dominant else ranked[:1]
    return {"status": ENGINEERING_DRY_RUN, "research_selection_applied": False, "research_selection": "not_evaluable",
            "note": NOT_THE_RESEARCH_SELECTION, "target": targets[0] if len(targets) == 1 else None, "targets": targets,
            "co_dominant": co_dominant, "top_gap": top_gap, "codominance_gap": gap, "candidates": candidates,
            "no_target_reason": None if targets else "no candidate family has a finite, positive AUDC"}
