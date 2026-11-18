"""Severity normalization for cross-mode comparison (spec v4.15 section 3.3), for ONE scene/fold and ONE
injection condition.

    s_0    = achieved severity at the zero-injection condition
    s_max  = achieved severity at the highest non-zero requested severity
    s_norm = (s_achieved - s_0) / (s_max - s_0)

Order of work (docs/decisions/0002-phase-3-completion.md):
  1. achieved severity is averaged across seeds at each requested level, within the scene/fold;
  2. s_0 and s_max are taken from those seed-aggregated values;
  3. the formula is applied to the seed-aggregated values only. Nothing is normalized per seed.

Raw achieved values and seed-level spread are returned unchanged and separately. If s_max = s_0, normalized
severity and AUDC are reported as uncomputable, never as zero. The same holds when achieved severity is
undefined (None) at every level, for example a scene with no eligible track frames.

Values are never clipped. s_max is the value at the highest requested level, so a non-monotone curve (mode D
saturates) can give s_norm above 1 at a lower level; `within_unit_interval` flags this for section 3.4, whose
AUDC integrates over [0, 1] without extrapolation. AUDC itself is not computed here.

Normalized severity is a protocol-level comparison coordinate, not a claim of physical equivalence between
error magnitudes (section 3.3). B1 and B2 share one family and one set of masks, so their achieved values,
and therefore their normalized values, are identical. Raw units remain the primary reporting units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

COMPUTABLE = "computable"
UNCOMPUTABLE = "uncomputable"


@dataclass(frozen=True)
class NormalizedSeverity:
    scene: str
    condition: str
    measure: str | None              # the achieved-severity quantity, e.g. rmse_pos_m
    levels: tuple                    # requested levels, ascending
    seeds: tuple                     # sorted
    raw: dict                        # level -> {seed: achieved}, unchanged
    seed_mean: dict | None           # level -> mean over seeds
    seed_spread: dict | None         # level -> {n, sd, min, max}; sd is None for one seed
    s_0: float | None
    s_max: float | None
    normalized: dict | None          # level -> s_norm; None when uncomputable
    status: str
    reason: str | None
    monotone: bool | None            # seed means never decrease with requested level
    within_unit_interval: bool | None

    @property
    def audc_status(self) -> str:
        """AUDC needs the normalized coordinate, so it is computable only when this result is."""
        return self.status

    def as_dict(self) -> dict:
        rows = [{"requested": level,
                 "achieved_by_seed": {str(s): v for s, v in self.raw[level].items()},
                 "achieved_seed_mean": self.seed_mean[level] if self.seed_mean else None,
                 "seed_spread": self.seed_spread[level] if self.seed_spread else None,
                 "s_norm": self.normalized[level] if self.normalized else None}
                for level in self.levels]
        return {"scene": self.scene, "condition": self.condition, "measure": self.measure, "seeds": list(self.seeds),
                "s_0": self.s_0, "s_max": self.s_max, "status": self.status, "audc_status": self.audc_status,
                "reason": self.reason, "monotone": self.monotone, "within_unit_interval": self.within_unit_interval,
                "levels": rows}


def _value(level: float, seed: int, value) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"level {level}, seed {seed}: achieved severity must be finite and non-negative, got {value!r}")
    return float(value)


def _table(achieved: Mapping) -> tuple[tuple, tuple, dict]:
    if not isinstance(achieved, Mapping) or not achieved:
        raise ValueError("achieved severity needs at least one requested level")
    levels = tuple(sorted(float(level) for level in achieved))
    if len(set(levels)) != len(achieved):
        raise ValueError("requested levels must be distinct")
    if any(not math.isfinite(level) or level < 0 for level in levels):
        raise ValueError(f"requested levels must be finite and non-negative, got {levels}")
    if levels[0] != 0.0:
        raise ValueError("the zero-injection level is required for s_0")
    if len(levels) < 2:
        raise ValueError("a non-zero requested level is required for s_max")
    rows = {float(level): row for level, row in achieved.items()}
    first = rows[levels[0]]
    if not isinstance(first, Mapping) or not first:
        raise ValueError("every requested level needs at least one seed")
    seeds = tuple(sorted(first))
    for level in levels:
        if not isinstance(rows[level], Mapping) or tuple(sorted(rows[level])) != seeds:
            raise ValueError(f"level {level} must hold exactly the seeds {list(seeds)}")
    return levels, seeds, {level: {s: _value(level, s, rows[level][s]) for s in seeds} for level in levels}


def _spread(values: list[float]) -> dict:
    n = len(values)
    mean = math.fsum(values) / n
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "sd": sd, "min": min(values), "max": max(values)}


def normalize_severity(scene: str, condition: str, achieved: Mapping, measure: str | None = None) -> NormalizedSeverity:
    """Normalize one scene/fold and condition. `achieved` maps requested level -> {seed: achieved severity}."""
    if not (isinstance(scene, str) and scene and isinstance(condition, str) and condition):
        raise ValueError("scene and condition must be non-empty strings")
    levels, seeds, raw = _table(achieved)
    cells = [raw[level][s] for level in levels for s in seeds]
    undefined = sum(v is None for v in cells)
    if undefined == len(cells):
        return NormalizedSeverity(scene, condition, measure, levels, seeds, raw, None, None, None, None, None,
                                  UNCOMPUTABLE, "achieved severity is undefined at every level", None, None)
    if undefined:
        raise ValueError("achieved severity is undefined for only some levels or seeds; denominators come from "
                         "the clean base, so this indicates inconsistent input")

    seed_mean = {level: math.fsum(raw[level].values()) / len(seeds) for level in levels}
    seed_spread = {level: _spread(list(raw[level].values())) for level in levels}
    s_0, s_max = seed_mean[levels[0]], seed_mean[levels[-1]]
    means = [seed_mean[level] for level in levels]
    monotone = all(b >= a for a, b in zip(means, means[1:]))
    if s_max < s_0:
        raise ValueError(f"{scene} {condition}: s_max {s_max} is below s_0 {s_0}; achieved severity at the highest "
                         "level cannot be below the zero-injection condition")
    if s_max == s_0:
        return NormalizedSeverity(scene, condition, measure, levels, seeds, raw, seed_mean, seed_spread, s_0, s_max,
                                  None, UNCOMPUTABLE, "s_max equals s_0: no non-zero achieved severity was realized",
                                  monotone, None)
    normalized = {level: (seed_mean[level] - s_0) / (s_max - s_0) for level in levels}
    inside = all(0.0 <= v <= 1.0 for v in normalized.values())
    return NormalizedSeverity(scene, condition, measure, levels, seeds, raw, seed_mean, seed_spread, s_0, s_max,
                              normalized, COMPUTABLE, None, monotone, inside)


def normalize_summaries(scene: str, summaries: Iterable[Mapping]) -> NormalizedSeverity:
    """Normalize the seed-level condition summaries of U7's and U8's sweeps, for one scene and one condition."""
    summaries = list(summaries)
    if not summaries:
        raise ValueError("no summaries were given")
    conditions = {s["condition"] for s in summaries}
    measures = {s["achieved_primary"]["name"] for s in summaries}
    if len(conditions) != 1 or len(measures) != 1:
        raise ValueError(f"summaries must share one condition and measure, got {sorted(conditions)}, {sorted(measures)}")
    achieved: dict = {}
    for s in summaries:
        row = achieved.setdefault(float(s["requested"]["value"]), {})
        if s["seed"] in row:
            raise ValueError(f"duplicate summary for level {s['requested']['value']}, seed {s['seed']}")
        row[s["seed"]] = s["achieved_primary"]["value"]
    return normalize_severity(scene, conditions.pop(), achieved, measures.pop())
