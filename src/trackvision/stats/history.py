"""Minimum sufficient history (spec v4.15 section 2.2.5; decision 0004, section 4), for one predictor, one
degradation condition and one population.

Clean ADE and FDE are shared by the five conditions of a population; AUDC is specific to the condition.

    clean ADE <= clean_ratio_max * best clean ADE
    clean FDE <= clean_ratio_max * best clean FDE
    delta_AUDC = |AUDC - AUDC_best| / max(|AUDC_best|, epsilon) <= delta_audc_max
    if AUDC_best = 0:  |AUDC - AUDC_best| <= epsilon

The minimum is the shortest history meeting all three. A history qualifies for the AUDC criterion only if every
fold of the population has a valid AUDC (decision 0005, section 1). A partial-fold history keeps its mean and counts
in the record, but it fails the criterion and stays out of AUDC_best. Outcomes are kept apart:
  found         a minimum history exists
  trade_off     qualifying histories exist but none meets all three (section 2.2.5); no winner is forced
  uncomputable  no history has a valid AUDC in every fold, or a clean metric is undefined; no substitute is assigned
"""

from __future__ import annotations

from typing import Mapping

FOUND, TRADE_OFF, UNCOMPUTABLE = "found", "trade_off", "uncomputable"


def minimum_history(clean_ade: Mapping, clean_fde: Mapping, audc: Mapping, clean_ratio_max: float,
                    delta_audc_max: float, epsilon_audc: float, valid_folds: Mapping, n_folds: int) -> dict:
    """Each mapping is history length -> value, or None if undefined. `audc` holds the mean over the valid folds,
    `valid_folds` their number, and `n_folds` the number of folds in the population."""
    histories = sorted(clean_ade)
    if not histories or any(sorted(m) != histories for m in (clean_fde, audc, valid_folds)):
        raise ValueError("clean ADE, clean FDE, AUDC and valid folds must cover the same history lengths")
    if any(not 0 <= valid_folds[h] <= n_folds for h in histories):
        raise ValueError(f"valid folds must lie between 0 and {n_folds}: {dict(valid_folds)}")
    ade_valid = [clean_ade[h] for h in histories if clean_ade[h] is not None]
    fde_valid = [clean_fde[h] for h in histories if clean_fde[h] is not None]
    qualifies = {h: audc[h] is not None and valid_folds[h] == n_folds for h in histories}
    ade_best, fde_best = min(ade_valid, default=None), min(fde_valid, default=None)
    audc_best = min((audc[h] for h in histories if qualifies[h]), default=None)

    per_history, minimum = {}, None
    for h in histories:
        ade_ok = None not in (clean_ade[h], ade_best) and clean_ade[h] <= clean_ratio_max * ade_best
        fde_ok = None not in (clean_fde[h], fde_best) and clean_fde[h] <= clean_ratio_max * fde_best
        if not qualifies[h]:
            distance, audc_ok = None, False
        elif audc_best == 0:
            distance, audc_ok = abs(audc[h]), abs(audc[h]) <= epsilon_audc
        else:
            distance = abs(audc[h] - audc_best) / max(abs(audc_best), epsilon_audc)
            audc_ok = distance <= delta_audc_max
        per_history[str(h)] = {"clean_ade_ok": ade_ok, "clean_fde_ok": fde_ok, "audc_ok": audc_ok,
                               "audc": audc[h], "audc_tolerance_distance": distance, "valid_folds": valid_folds[h],
                               "partial_folds": valid_folds[h] < n_folds, "audc_qualifies": qualifies[h]}
        if minimum is None and ade_ok and fde_ok and audc_ok:
            minimum = h
    if audc_best is None:
        return {"status": UNCOMPUTABLE, "minimum_history": None, "reason": "no history has a valid AUDC in every fold",
                "audc_best": None, "per_history": per_history}
    if ade_best is None or fde_best is None:
        return {"status": UNCOMPUTABLE, "minimum_history": None, "reason": "clean metrics are undefined",
                "audc_best": None, "per_history": per_history}
    return {"status": FOUND if minimum is not None else TRADE_OFF, "minimum_history": minimum,
            "reason": None if minimum is not None else "no history meets all three criteria",
            "audc_best": audc_best, "clean_ade_best": ade_best, "clean_fde_best": fde_best,
            "per_history": per_history}
