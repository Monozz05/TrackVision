"""Aggregation in the fixed order of spec v4.15 section 7.4: seeds within a scene/fold, then across scenes.

The scene/fold is the statistical unit. Seeds describe within-fold variation and are not scene-level replicates.
Windows never appear here: cell values are already per-seed summaries. An undefined value stays undefined, with
its reason: it contributes nothing to a mean and is never replaced by zero, epsilon, interpolation or any other
surrogate (section 3.4). Every aggregate carries its valid and undefined counts.

These are descriptive means and standard deviations. Nothing here is an inferential statistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Aggregate:
    mean: float | None
    sd: float | None                 # n - 1; None below two valid values
    n_valid: int
    n_undefined: int
    undefined: dict = field(default_factory=dict)    # key -> reason

    def as_dict(self) -> dict:
        return {"mean": self.mean, "sd": self.sd, "n_valid": self.n_valid, "n_undefined": self.n_undefined,
                "undefined": {str(k): v for k, v in self.undefined.items()}}


def aggregate(values: Mapping, reasons: Mapping | None = None) -> Aggregate:
    """Mean and spread of the defined values; a None value is undefined, with its reason from `reasons`."""
    if not values:
        raise ValueError("nothing to aggregate")
    reasons = reasons or {}
    valid, undefined = [], {}
    for key, value in values.items():
        if value is None:
            undefined[key] = reasons.get(key, "undefined")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key}: expected a finite number or None, got {value!r}")
        valid.append(float(value))
    n = len(valid)
    if n == 0:
        return Aggregate(None, None, 0, len(undefined), undefined)
    mean = math.fsum(valid) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in valid) / (n - 1)) if n > 1 else None
    return Aggregate(mean, sd, n, len(undefined), undefined)


def seeds_then_scenes(table: Mapping[str, Mapping], reasons: Mapping | None = None) -> dict:
    """`table[scene][seed]` -> one Aggregate per scene (seeds first), then one across the scene means."""
    reasons = reasons or {}
    per_scene = {scene: aggregate(row, reasons.get(scene)) for scene, row in table.items()}
    across = aggregate({scene: a.mean for scene, a in per_scene.items()},
                       {scene: "no valid seed" for scene, a in per_scene.items() if a.mean is None})
    return {"per_scene": per_scene, "across_scenes": across}


@dataclass(frozen=True)
class Relative:
    value: float | None
    absolute_difference: float | None
    reason: str | None


def relative_degradation(x: float | None, x0: float | None) -> Relative:
    """(x - x0) / x0 (section 3.4). With x0 = 0 it is undefined and the absolute difference is reported."""
    if x0 is None:
        return Relative(None, None, "clean_baseline_undefined")
    if x is None:
        return Relative(None, None, "metric_undefined")
    if x0 == 0:
        return Relative(None, x - x0, "zero_clean_baseline")
    return Relative((x - x0) / x0, x - x0, None)
