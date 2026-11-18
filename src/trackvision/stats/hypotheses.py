"""Hypothesis criteria of spec v4.15 section 2.3, kept apart from any research verdict.

No hypothesis can receive a research verdict today: the five-scene static protocol is unavailable, so every entry
has `research_verdict: not_evaluable`. Where a frozen criterion can be computed on synthetic engineering output, it
appears under `engineering_check`, which says only whether the arithmetic of the criterion was met on those
synthetic scenes. It is not support for or against the hypothesis.

Thresholds come from frozen configuration. Where the text does not settle a detail, `assumptions` says what was
done, so nothing passes as frozen. Criteria that need bootstrap confidence intervals report the point value only:
replicates and confidence level are not specified, and they are not invented here.

All values are scene-level (seeds aggregated first). The levels of a curve are compared as measured; B1 and B2 share
masks and predictors share corruption draws, so matched achieved severities coincide exactly (section 2.3.13) and
no interpolation arises.
"""

from __future__ import annotations

import math

import numpy as np

from trackvision.stats.aggregate import aggregate

NOT_EVALUABLE = "not_evaluable"
REASON = "the five-scene static research protocol is unavailable (decision 0001 F)"
NOT_RUNNABLE = {
    "H5": "moving-camera only: P-8 is blocked by moving_dataset",
    "H7": "needs P-13 real-tracker outputs, which are not in this phase",
    "H8": "P-10 is blocked (decision 0004, sections 5 and 6)",
    "H9": "needs P-14 collision experiments, which are not in this phase",
    "H10": "needs trajectory-class results, which this sweep does not record per class",
}
ENGINEERING = ("H1", "H2", "H3", "H4", "H6")
ZERO_LEVEL = ("Implementation assumption, not a frozen v4.15 requirement: the mean runs over the non-zero grid levels "
              "and leaves out the zero level. Section {} says '{}'; the exact zero-level interpretation needs "
              "reconciliation with the specification or v4.16 (decision 0005, section 5).")


def not_evaluated(scope: str) -> dict:
    out = {h: {"research_verdict": NOT_EVALUABLE, "reason": why, "engineering_check": None}
           for h, why in NOT_RUNNABLE.items()}
    for h in ENGINEERING:
        out[h] = {"research_verdict": NOT_EVALUABLE, "reason": REASON, "engineering_check": None}
    return dict(sorted(out.items()))


def _per_level(curves: dict, scenes: list, predictor: str, condition: str, field: str) -> list:
    """One Aggregate across scenes per requested level, in requested order."""
    rows = [curves[s][predictor][condition]["levels"] for s in scenes]
    return [aggregate({s: r[i][field]["mean"] for s, r in zip(scenes, rows)}) for i in range(len(rows[0]))]


def _se(a) -> float | None:
    return None if a.mean is None or a.sd is None else a.sd / math.sqrt(a.n_valid)


def _monotone_violations(means: list, errors: list) -> list:
    """Section 2.3.1: a violation is m[i+1] < m[i] - se[i]. The first pair touches the zero-severity endpoint."""
    return [i for i in range(1, len(means) - 1) if means[i + 1] < means[i] - errors[i]]


def _h1(curves, scenes, predictors) -> dict:
    out, assumptions = {}, ["Evaluated per predictor; section 2.3.1 does not name one."]
    for p in predictors:
        for metric in ("ade_m", "fde_m"):
            a = _per_level(curves, scenes, p, "A", metric)
            if any(x.mean is None or x.sd is None for x in a):
                out[f"{p}/{metric}"] = {"status": "uncomputable", "reason": "a level has no between-scene spread"}
                continue
            bad = _monotone_violations([x.mean for x in a], [_se(x) for x in a])
            out[f"{p}/{metric}"] = {"criterion_met": not bad, "violating_levels": bad, "means": [x.mean for x in a]}
    return {"values": out, "assumptions": assumptions}


def _h2(curves, scenes, predictors) -> dict:
    out = {}
    for p in predictors:
        b2, b1 = [curves[s][p]["B2"]["levels"] for s in scenes], [curves[s][p]["B1"]["levels"] for s in scenes]
        gaps = []
        for i in range(len(b2[0])):
            values = {}
            for s, x, y in zip(scenes, b2, b1):
                u, v = x[i]["d_ade"]["mean"], y[i]["d_ade"]["mean"]
                values[s] = None if u is None or v is None else u - v
            gaps.append(aggregate(values))
        nonzero = gaps[1:]
        if any(g.mean is None or g.sd is None for g in nonzero):
            out[p] = {"status": "uncomputable", "reason": "undefined degradation at a non-zero level"}
            continue
        part1 = [i for i, g in enumerate(nonzero, start=1) if g.mean < -_se(g)]
        part2 = _monotone_violations([g.mean for g in gaps], [(_se(g) or 0.0) for g in gaps])
        out[p] = {"part1_met": not part1, "part1_violations": part1, "part2_met": not part2,
                  "part2_violations": part2, "gaps": [g.mean for g in gaps]}
    return {"values": out, "assumptions": ["Evaluated per predictor; section 2.3.5 does not name one."]}


def _h3(curves, scenes, predictors, delta) -> dict:
    out = {}
    for p in predictors:
        d, c = _per_level(curves, scenes, p, "C", "d_ade"), _per_level(curves, scenes, p, "C", "prediction_coverage")
        if any(x.mean is None for x in d + c):
            out[p] = {"status": "uncomputable", "reason": "undefined at a level"}
            continue
        c0 = c[0].mean
        diffs = [(c0 - cv.mean) / c0 - dv.mean for cv, dv in zip(c[1:], d[1:])]
        value = float(np.mean(diffs))
        out[p] = {"mean_difference": value, "threshold": delta, "criterion_met": value >= delta}
    return {"values": out, "assumptions": [ZERO_LEVEL.format("2.3.7", "primary range"),
                                           "Confidence intervals not computed: bootstrap constants are unspecified."]}


def _fit_slope(x, y) -> float | None:
    return float(np.polyfit(x, y, 1)[0]) if len(set(x)) >= 2 else None


def _h4_curve(curves, scenes, p, metric, ratio_min, aic_min) -> dict:
    """The section 2.3.2 calculation for one predictor and one metric."""
    values = _per_level(curves, scenes, p, "D", metric)
    rows = [curves[s][p]["D"]["levels"] for s in scenes]
    achieved = [aggregate({s: r[i].get("achieved_seed_mean") for s, r in zip(scenes, rows)}).mean for i in range(len(rows[0]))]
    if any(a is None for a in achieved) or any(v.mean is None for v in values):
        return {"status": "uncomputable", "reason": "undefined at a level"}
    pairs = sorted(zip(achieved, [v.mean for v in values]))
    x, y = [a for a, _ in pairs], [b for _, b in pairs]
    mid = (x[0] + x[-1]) / 2.0
    lo = [(a, b) for a, b in pairs if a <= mid]
    hi = [(a, b) for a, b in pairs if a >= mid]
    b_lo, b_hi = _fit_slope(*zip(*lo)), _fit_slope(*zip(*hi))
    if b_lo is None or b_hi is None:
        return {"status": "uncomputable", "reason": "a half of the achieved range has under two distinct points"}
    result = {"b_lo": b_lo, "b_hi": b_hi}
    if b_lo <= 0:
        result.update(criterion_met=False, slope_ratio=None, note="b_lo <= 0: not supported as stated (section 2.3.2)")
    else:
        ratio = b_hi / b_lo
        result.update(slope_ratio=ratio, threshold=ratio_min, criterion_met=ratio >= ratio_min)
    delta = _delta_aic(np.array(x), np.array(y))
    result.update(delta_aic=delta, aic_threshold=aic_min, aic_preference_met=None if delta is None else delta >= aic_min)
    return result


def _h4_agreement(ade: dict, fde: dict) -> dict:
    """Whether ADE and FDE say the same thing. A disagreement is reported, never resolved (section 2.3.2)."""
    if "status" in ade or "status" in fde:
        return {"comparable": False, "reason": "ADE or FDE is uncomputable", "disagreements": []}
    pairs = {"slope_ratio": (ade["criterion_met"], fde["criterion_met"]),
             "delta_aic": (ade["aic_preference_met"], fde["aic_preference_met"])}
    compared = {k: v for k, v in pairs.items() if None not in v}
    return {"comparable": True, "compared": sorted(compared), "not_compared": sorted(set(pairs) - set(compared)),
            "disagreements": sorted(k for k, (a, f) in compared.items() if a != f)}


def _h4(curves, scenes, predictors, ratio_min, aic_min) -> dict:
    out = {}
    for p in predictors:
        ade = _h4_curve(curves, scenes, p, "ade_m", ratio_min, aic_min)
        fde = _h4_curve(curves, scenes, p, "fde_m", ratio_min, aic_min)
        out[p] = {"ade": ade, "fde": fde, "agreement": _h4_agreement(ade, fde)}
    return {"values": out, "assumptions": ["Evaluated per predictor, on ADE and FDE separately; section 2.3.2 names no "
                                           "predictor, and its text fits ADE while H4 names both metrics.",
                                           "Points at the midpoint belong to both halves.",
                                           "AIC = n ln(RSS / n) + 2k.",
                                           "The ratio's confidence interval is not computed: constants are unspecified."]}


def _delta_aic(x, y) -> float | None:
    n = len(x)
    rss = [float(np.sum((y - np.polyval(np.polyfit(x, y, d), x)) ** 2)) for d in (1, 2)]
    if n < 4 or len(set(x.tolist())) < 3 or min(rss) <= 0:
        return None
    return float((n * math.log(rss[0] / n) + 4) - (n * math.log(rss[1] / n) + 6))


def _h6(curves, scenes, delta) -> dict:
    out = {}
    cv = _per_level(curves, scenes, "CV", "D", "d_ade")
    for model in ("LSTM", "GRU"):
        d = _per_level(curves, scenes, model, "D", "d_ade")
        if any(v.mean is None for v in cv + d):
            out[model] = {"status": "uncomputable", "reason": "undefined at a level"}
            continue
        value = float(np.mean([a.mean - b.mean for a, b in zip(d[1:], cv[1:])]))
        out[model] = {"mean_gap_vs_cv": value, "threshold": delta, "criterion_met": value >= delta}
    return {"values": out, "assumptions": [ZERO_LEVEL.format("2.3.9", "range")]}


def engineering_checks(curves: dict, spec_doc: dict, cfg) -> dict:
    """Frozen criteria computed on synthetic engineering curves. Never a research verdict."""
    h = cfg.frozen["hypotheses"]
    tol, scenes, predictors = h["tolerances"], spec_doc["scenes"], spec_doc["predictors"]
    out = not_evaluated(spec_doc["scope"])
    checks = {"H1": _h1(curves, scenes, predictors), "H2": _h2(curves, scenes, predictors),
              "H3": _h3(curves, scenes, predictors, tol["delta_h3"]),
              "H4": _h4(curves, scenes, predictors, h["h4_slope_ratio_min"], h["h4_delta_aic_min"]),
              "H6": _h6(curves, scenes, tol["delta_h6"])}
    for name, check in checks.items():
        out[name]["engineering_check"] = {"scope": "synthetic engineering only; not evidence for the hypothesis", **check}
    return out
