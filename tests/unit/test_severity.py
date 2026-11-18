"""Phase 3: severity normalization (spec v4.15 section 3.3; decision 0002).

Seeds are aggregated within each requested level first; s_0 and s_max come from the aggregated values; nothing
is normalized per seed. Hand-calculated tables give exact fractions. The sweep tests use the U7 and U8 feasibility
sweeps on a synthetic scene and check only properties that hold for any scene.
"""

import copy
import json
import math

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments import switch_feasibility as sf
from trackvision.inject import identity_switch as ids
from trackvision.inject.severity import COMPUTABLE, UNCOMPUTABLE, normalize_severity, normalize_summaries
from trackvision.synthetic.scene import SceneConfig
from trackvision.synthetic.tracks import GridTrack

CFG = config.validate().config
D_MAX = float(CFG.frozen["injection"]["d_max_m"])
SCENE = SceneConfig(seed=21, n_frames=750)

TWO_SEEDS = {0.0: {0: 0.0, 1: 0.0}, 0.1: {0: 1.0, 1: 3.0}, 0.2: {0: 2.0, 1: 4.0}}


def _norm(table, condition="A"):
    return normalize_severity("S1", condition, table)


# --- the formula ---------------------------------------------------------------------------------

def test_hand_calculated_example_with_a_non_zero_baseline():
    """s_0 = 0.5, s_max = 2.5: s = 1.5 -> (1.5 - 0.5) / (2.5 - 0.5) = 0.5; s = 1.0 -> 0.25."""
    r = _norm({0.0: {0: 0.5}, 0.1: {0: 1.0}, 0.2: {0: 1.5}, 0.4: {0: 2.5}})
    assert (r.s_0, r.s_max, r.status) == (0.5, 2.5, COMPUTABLE)
    assert r.normalized == {0.0: 0.0, 0.1: 0.25, 0.2: 0.5, 0.4: 1.0}


def test_zero_maps_to_zero_and_the_highest_level_maps_to_one():
    r = _norm({0.0: {0: 0.0, 1: 0.0}, 0.05: {0: 0.07, 1: 0.08}, 0.5: {0: 0.69, 1: 0.74}})
    assert r.normalized[0.0] == 0.0 and r.normalized[0.5] == 1.0


def test_seeds_are_aggregated_before_normalizing():
    """Means 0, 2, 3 give 2/3 at the middle level. Normalizing per seed would give 1/2 and 3/4, mean 0.625."""
    r = _norm(TWO_SEEDS)
    assert r.seed_mean == {0.0: 0.0, 0.1: 2.0, 0.2: 3.0}
    assert r.normalized[0.1] == pytest.approx(2 / 3, abs=1e-15)
    per_seed = [TWO_SEEDS[0.1][s] / TWO_SEEDS[0.2][s] for s in (0, 1)]
    assert sum(per_seed) / 2 == 0.625 and r.normalized[0.1] != 0.625


def test_endpoints_come_from_the_aggregated_values_not_from_any_seed():
    r = _norm({0.0: {0: 0.0, 1: 2.0}, 0.2: {0: 3.0, 1: 7.0}})
    assert (r.s_0, r.s_max) == (1.0, 5.0)


def test_result_does_not_depend_on_seed_or_level_order():
    shuffled = {0.2: {1: 4.0, 0: 2.0}, 0.0: {1: 0.0, 0: 0.0}, 0.1: {1: 3.0, 0: 1.0}}
    assert _norm(shuffled).normalized == _norm(TWO_SEEDS).normalized
    assert _norm(shuffled).levels == (0.0, 0.1, 0.2) and _norm(shuffled).seeds == (0, 1)


# --- uncomputable cases --------------------------------------------------------------------------

@pytest.mark.parametrize("table", [
    {0.0: {0: 0.0, 1: 0.0}, 0.1: {0: 0.0, 1: 0.0}, 0.2: {0: 0.0, 1: 0.0}},     # nothing was realized
    {0.0: {0: 0.4}, 0.2: {0: 0.4}},                                            # non-zero but equal
    {0.0: {0: 1.0, 1: 3.0}, 0.2: {0: 2.0, 1: 2.0}},                            # seed 0 alone would be computable
])
def test_s_max_equal_to_s_0_is_uncomputable_not_zero(table):
    r = _norm(table)
    assert r.status == r.audc_status == UNCOMPUTABLE and "s_max equals s_0" in r.reason
    assert r.normalized is None and r.s_0 == r.s_max
    assert r.seed_mean is not None and r.raw == table                          # raw values still reported
    assert all(row["s_norm"] is None for row in r.as_dict()["levels"])


def test_undefined_severity_at_every_level_is_uncomputable():
    r = _norm({0.0: {0: None, 1: None}, 0.2: {0: None, 1: None}}, "D")
    assert r.status == UNCOMPUTABLE and r.seed_mean is None and r.s_0 is None and r.normalized is None


def test_partly_undefined_severity_is_rejected():
    with pytest.raises(ValueError, match="only some"):
        _norm({0.0: {0: 0.0, 1: None}, 0.2: {0: 1.0, 1: 1.0}})


def test_real_injector_with_no_realized_events_is_uncomputable():
    """Eligible track frames exist, but the draws of this fixture realize no event at these low levels."""
    k = np.arange(4)
    tracks = (GridTrack(1, 10 * k, np.column_stack([0.3 * k, 0.0 * k])), GridTrack(2, 10 * k, np.column_stack([0.3 * k, 0.5 + 0.0 * k])))
    table = {p: {s: ids.sample_switches(tracks, p, s, D_MAX, 10).achieved["switch_event_rate"] for s in (0, 1, 2)}
             for p in (0.0, 0.001)}
    assert ids.proximity(tracks, D_MAX, 10).eligible_track_frames == 6
    assert all(v == 0.0 for row in table.values() for v in row.values())
    assert _norm(table, "D").status == UNCOMPUTABLE


# --- monotonicity, finiteness, no clipping -------------------------------------------------------

def test_normalization_preserves_order_and_stays_finite():
    rng = np.random.default_rng(0)
    for _ in range(50):
        means = np.sort(rng.uniform(0, 5, size=6))
        means[0] = 0.0
        table = {float(i): {0: float(m)} for i, m in enumerate(means)}
        if means[-1] == 0.0:
            continue
        r = _norm(table)
        values = [r.normalized[float(i)] for i in range(6)]
        assert all(math.isfinite(v) for v in values) and r.monotone and r.within_unit_interval
        assert all(b >= a for a, b in zip(values, values[1:]))


def test_non_monotone_curve_is_kept_unclipped_and_flagged():
    r = _norm({0.0: {0: 0.0}, 0.1: {0: 5.0}, 0.2: {0: 3.0}})
    assert r.normalized[0.1] == pytest.approx(5 / 3, abs=1e-15)
    assert r.monotone is False and r.within_unit_interval is False and r.status == COMPUTABLE


# --- raw values and seed spread ------------------------------------------------------------------

def test_raw_severity_is_preserved_and_the_input_is_not_modified():
    table = copy.deepcopy(TWO_SEEDS)
    r = _norm(table)
    assert table == TWO_SEEDS and r.raw == TWO_SEEDS
    rows = r.as_dict()["levels"]
    assert rows[1]["achieved_by_seed"] == {"0": 1.0, "1": 3.0} and rows[1]["achieved_seed_mean"] == 2.0
    json.dumps(r.as_dict())                                                    # serializable as it stands


def test_seed_spread_is_reported_separately():
    spread = _norm(TWO_SEEDS).seed_spread[0.1]
    assert spread == {"n": 2, "sd": math.sqrt(2.0), "min": 1.0, "max": 3.0}
    assert _norm({0.0: {0: 0.0}, 0.2: {0: 1.0}}).seed_spread[0.2]["sd"] is None


# --- invalid tables ------------------------------------------------------------------------------

@pytest.mark.parametrize("table", [
    {},                                                                       # empty
    {0.1: {0: 1.0}, 0.2: {0: 2.0}},                                           # no zero-injection level
    {0.0: {0: 0.0}},                                                          # no non-zero level
    {0.0: {}, 0.2: {}},                                                       # no seeds
    {0.0: {0: 0.0, 1: 0.0}, 0.2: {0: 1.0}},                                   # seeds differ between levels
    {0.0: {0: 0.0}, 0.2: {0: float("nan")}},
    {0.0: {0: 0.0}, 0.2: {0: float("inf")}},
    {0.0: {0: 0.0}, 0.2: {0: -1.0}},
    {0.0: {0: 0.0}, 0.2: {0: True}},
    {0.0: {0: 0.0}, -0.2: {0: 1.0}},
    {0.0: {0: 2.0}, 0.2: {0: 1.0}},                                           # s_max below s_0
])
def test_invalid_tables_raise(table):
    with pytest.raises(ValueError):
        _norm(table)


def test_scene_and_condition_are_required():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_severity("", "A", TWO_SEEDS)


# --- U7 and U8 sweep summaries -------------------------------------------------------------------

@pytest.fixture(scope="module")
def summaries():
    runs = list(inj.run_injection(CFG, SCENE).results) + list(sf.run_switch(CFG, SCENE).results)
    out: dict = {}
    for r in runs:
        out.setdefault(r.summary["condition"], []).append(r.summary)
    return out


def test_every_condition_normalizes_from_its_sweep_summaries(summaries):
    assert sorted(summaries) == ["A", "B1", "B2", "C", "D"]
    for condition, rows in summaries.items():
        before = copy.deepcopy(rows)
        r = normalize_summaries("SYN-21", rows)
        assert r.status == COMPUTABLE and r.condition == condition
        assert r.measure == rows[0]["achieved_primary"]["name"]
        assert r.normalized[0.0] == 0.0 and r.normalized[max(r.levels)] == 1.0
        assert list(r.levels) == inj.levels(CFG, condition) and r.seeds == (0, 1, 2)
        for s in rows:
            assert r.raw[s["requested"]["value"]][s["seed"]] == s["achieved_primary"]["value"]
        assert rows == before


def test_b1_and_b2_share_one_normalized_curve(summaries):
    assert normalize_summaries("SYN-21", summaries["B1"]).normalized == normalize_summaries("SYN-21", summaries["B2"]).normalized


def test_summaries_must_be_one_condition_without_duplicates(summaries):
    with pytest.raises(ValueError, match="one condition"):
        normalize_summaries("SYN-21", summaries["A"] + summaries["C"])
    with pytest.raises(ValueError, match="duplicate"):
        normalize_summaries("SYN-21", summaries["C"] + summaries["C"][:1])
    with pytest.raises(ValueError, match="no summaries"):
        normalize_summaries("SYN-21", [])
