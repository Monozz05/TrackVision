"""Descriptive across-scene family AUDC (decision 0006, item 11).

Order of work: `stats.audc.family_audc` gives one value per scene from that scene's valid predictor AUDCs, with the
frozen minimum of two (section 5.4); `across_scenes` then aggregates the valid scene values. Windows never appear
here. The order matters when predictors are missing: a scene with one valid predictor contributes nothing, rather
than letting that predictor's value into a per-predictor mean.

This is a descriptive summary, not research-grade five-scene evidence. Pooling ETH and HOTEL is refused by the
caller (`experiments.engineering_analysis`), because the scope is known there.
"""

from __future__ import annotations

from typing import Mapping

from trackvision.inject.severity import COMPUTABLE, UNCOMPUTABLE
from trackvision.stats.aggregate import aggregate


def across_scenes(by_scene: Mapping[str, Mapping]) -> dict:
    """`by_scene[scene]` is a `family_audc` result. Undefined scenes keep their reasons and contribute nothing."""
    if not by_scene:
        raise ValueError("no scenes were given")
    values = {s: e["value"] if e["status"] == COMPUTABLE else None for s, e in by_scene.items()}
    reasons = {s: e["reason"] for s, e in by_scene.items() if e["status"] != COMPUTABLE}
    a = aggregate(values, reasons)
    return {"status": UNCOMPUTABLE if a.mean is None else COMPUTABLE, "value": a.mean, "sd": a.sd,
            "n_valid_scenes": a.n_valid, "n_undefined_scenes": a.n_undefined,
            "undefined": {str(k): v for k, v in a.undefined.items()}, "scene_values": dict(values),
            "valid_predictors_by_scene": {s: e["n_valid"] for s, e in by_scene.items()},
            "reason": "no scene has a valid family AUDC" if a.mean is None else None}
