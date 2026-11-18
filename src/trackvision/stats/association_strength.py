"""Association arithmetic of spec v4.15 section 2.3.6, for the synthetic E5 engineering calculation only.

Spearman correlation is the Pearson correlation of average ranks. A metric with no variation has an undefined
correlation, which is recorded and never imputed. The group values follow the frozen formulas:

    |rho|_ASSOC = mean(|rho_IDF1|, |rho_HOTA|, |rho_IDSW|)
    |rho|_DET   = |rho_MOTA|
    delta_rho   = |rho|_ASSOC - |rho|_DET

The frozen text does not say what happens to the group mean when one correlation is undefined, so delta_rho is
uncomputable in that case rather than a mean over fewer metrics.

Observations are scene-level pairs and every pair comes from one scene. The scene-cluster bootstrap of section 2.3.6
has no specified constants, so no interval exists and `inference` says so. No H7 verdict is produced here.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

ASSOCIATION_METRICS = ("IDF1", "HOTA", "IDSW")
DETECTION_METRICS = ("MOTA",)
METRICS = (*ASSOCIATION_METRICS, *DETECTION_METRICS)
NO_INFERENCE = "uncomputable: the section 2.3.6 scene-cluster bootstrap has no specified constants"


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks; tied values share the mean of their positions."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> dict:
    """{value, n, reason}; the value is None where the correlation is mathematically undefined."""
    if len(x) != len(y):
        raise ValueError("paired observations must have the same length")
    n = len(x)
    if n < 2:
        return {"value": None, "n": n, "reason": "fewer than two paired observations"}
    rx, ry = average_ranks(x), average_ranks(y)
    dx, dy = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float(dx @ dx) * float(dy @ dy))
    if denominator == 0:
        return {"value": None, "n": n, "reason": "insufficient variation"}
    return {"value": float(dx @ dy) / denominator, "n": n, "reason": None}


def association_summary(observations: Mapping[str, Mapping], ade_key: str = "ade_m") -> dict:
    """`observations[scene]` holds METRICS and `ade_key`, for one tracker configuration and predictor."""
    scenes = sorted(observations)
    ade = [observations[s][ade_key] for s in scenes]
    defined = [i for i, v in enumerate(ade) if v is not None]
    rho = {m: spearman([observations[scenes[i]][m] for i in defined], [ade[i] for i in defined]) for m in METRICS}
    values = {m: r["value"] for m, r in rho.items()}
    assoc = (None if any(values[m] is None for m in ASSOCIATION_METRICS)
             else math.fsum(abs(values[m]) for m in ASSOCIATION_METRICS) / len(ASSOCIATION_METRICS))
    det = None if values["MOTA"] is None else abs(values["MOTA"])
    return {"scenes": scenes, "scene_count": len(scenes), "paired_observations": len(defined), "tracker_count": 1,
            "correlations": rho, "abs_rho_assoc": assoc, "abs_rho_det": det,
            "delta_rho": None if assoc is None or det is None else assoc - det, "inference": NO_INFERENCE,
            "h7_verdict": "not_evaluable"}
