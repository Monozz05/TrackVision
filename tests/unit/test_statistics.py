"""Phase 4: aggregation order, monotonicity diagnostics and the P-15 minimum-history criterion."""

import math

import numpy as np
import pytest

from trackvision.inject import identity_switch as ids
from trackvision.stats import hypotheses
from trackvision.stats.aggregate import aggregate, relative_degradation, seeds_then_scenes
from trackvision.stats.history import minimum_history
from trackvision.stats.monotonicity import check_monotonicity
from trackvision.synthetic.tracks import GridTrack

# --- aggregation ---------------------------------------------------------------------------------


def test_seeds_are_aggregated_before_scenes():
    """Scene means 2 and 5 give 3.5. Pooling the three values would wrongly give 3."""
    out = seeds_then_scenes({"S1": {0: 1.0, 1: 3.0}, "S2": {0: 5.0}})
    assert out["per_scene"]["S1"].mean == 2.0 and out["across_scenes"].mean == 3.5
    assert out["across_scenes"].sd == pytest.approx(math.sqrt(4.5), abs=1e-15) and out["across_scenes"].n_valid == 2


def test_undefined_values_stay_undefined_and_are_counted():
    a = aggregate({0: 1.0, 1: None, 2: 3.0}, {1: "no_predicted_windows"})
    assert a.mean == 2.0 and (a.n_valid, a.n_undefined) == (2, 1) and a.undefined == {1: "no_predicted_windows"}
    none = aggregate({0: None, 1: None})
    assert none.mean is None and none.sd is None and none.n_undefined == 2


def test_a_scene_with_no_valid_seed_contributes_nothing_across_scenes():
    out = seeds_then_scenes({"S1": {0: None, 1: None}, "S2": {0: 2.0, 1: 4.0}})
    assert out["across_scenes"].mean == 3.0 and out["across_scenes"].n_valid == 1
    assert out["across_scenes"].sd is None and "S1" in out["across_scenes"].undefined


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_aggregate_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        aggregate({0: bad})


def test_relative_degradation_never_imputes():
    assert relative_degradation(1.5, 1.0).value == 0.5
    assert relative_degradation(None, 1.0).reason == "metric_undefined"
    assert relative_degradation(1.0, None).reason == "clean_baseline_undefined"


# --- monotonicity --------------------------------------------------------------------------------

def _table(*rows):
    """rows[i] = per-seed achieved values at requested level i."""
    return {float(i): dict(enumerate(row)) for i, row in enumerate(rows)}


def test_a_requires_a_strict_increase():
    assert check_monotonicity("S", "A", _table((0.0, 0.0), (1.0, 1.1), (2.0, 2.1)))["status"] == "pass"
    assert check_monotonicity("S", "A", _table((0.0, 0.0), (1.0, 1.0), (1.0, 1.2)))["status"] == "fail"


def test_a_degenerate_zero_noise_realization_does_not_pass():
    r = check_monotonicity("S", "A", _table((0.0,), (0.0,), (0.0,)))
    assert r["status"] == "fail" and r["seed_mean"]["status"] == "fail"


@pytest.mark.parametrize("condition", ["B1", "B2", "C"])
def test_b_and_c_accept_ties_but_not_decreases(condition):
    assert check_monotonicity("S", condition, _table((0.0,), (0.1,), (0.1,)))["status"] == "pass"
    assert check_monotonicity("S", condition, _table((0.0,), (0.2,), (0.1,)))["status"] == "fail"


def test_seed_mean_is_checked_separately_from_seeds():
    r = check_monotonicity("S", "C", _table((0.0, 0.0), (0.3, 0.1), (0.2, 0.3)))
    assert r["per_seed"]["0"]["status"] == "fail" and r["per_seed"]["1"]["status"] == "pass"
    assert r["seed_mean"]["status"] == "pass" and r["status"] == "fail"


def test_d_decrease_is_a_diagnostic_finding_not_a_failure():
    diag = {1.0: {0: {"triggers": 2}}}
    r = check_monotonicity("S", "D", _table((0.0,), (2.0,), (1.0,)), diag)
    assert r["status"] == "diagnostic_finding" and r["rule"] == "diagnostic"
    assert r["per_seed"]["0"]["violations"][0]["from"] == 1.0 and r["diagnostics"]["1.0"]["0"]["triggers"] == 2


def test_d_without_a_decrease_is_never_called_a_pass():
    assert check_monotonicity("S", "D", _table((0.0,), (1.0,)))["status"] == "no_violation_observed"


@pytest.mark.parametrize("achieved", [{}, _table((0.0,)), _table((0.0,), (None,))])
@pytest.mark.parametrize("condition", ["A", "B2", "D"])
def test_no_check_passes_on_nothing(achieved, condition):
    assert check_monotonicity("S", condition, achieved)["status"] == "not_evaluated"


def _crossing_groups():
    """Decision 0004's geometry: (1, 2) close for 4 steps, then only (1, 3) and (2, 4) close for 8."""
    near, far = (0.0, 0.0), (100.0, 0.0)
    rows = {1: [near] * 13,
            2: [(1.0, 0.0)] * 5 + [far] * 8,
            3: [(0.0, 100.0)] * 5 + [(0.0, 1.0)] * 8,
            4: [(1.0, -100.0)] * 5 + [(100.0, 1.0)] * 8}
    return tuple(GridTrack(i, 10 * np.arange(13), np.array(p, dtype=float)) for i, p in rows.items())


def test_the_real_d_injector_can_lose_achieved_severity_as_requested_severity_rises():
    """The accepted injector, unchanged. An early (1, 2) event locks both tracks and removes two later events."""
    tracks = _crossing_groups()
    table = {p: {s: ids.sample_switches(tracks, p, s, 2.0, 10).achieved["switch_event_rate"] for s in range(200)}
             for p in (0.1, 0.2)}
    drops = [s for s in range(200) if table[0.2][s] < table[0.1][s]]
    assert drops, "the geometry should produce at least one decrease across 200 seeds"
    s = drops[0]
    r = check_monotonicity("S", "D", {0.0: {s: 0.0}, 0.1: {s: table[0.1][s]}, 0.2: {s: table[0.2][s]}})
    assert r["status"] == "diagnostic_finding"


# --- P-15 minimum history ------------------------------------------------------------------------

H = (4, 6, 8)
GOOD = {4: 1.0, 6: 1.0, 8: 1.0}


def _min(ade=GOOD, fde=GOOD, audc=GOOD, folds=None):
    """Every defined history has all five folds unless `folds` says otherwise."""
    valid = {h: 5 if audc[h] is not None else 0 for h in audc} if folds is None else folds
    return minimum_history(ade, fde, audc, 1.1, 0.1, 0.01, valid, 5)


def test_shortest_history_meeting_all_three_criteria():
    r = _min(ade={4: 1.2, 6: 1.05, 8: 1.0}, fde={4: 1.0, 6: 1.0, 8: 1.0}, audc={4: 0.5, 6: 0.52, 8: 0.5})
    assert r["status"] == "found" and r["minimum_history"] == 6


def test_clean_criteria_are_shared_while_audc_is_per_condition():
    ade, fde = {4: 1.0, 6: 1.0, 8: 1.0}, {4: 1.0, 6: 1.0, 8: 1.0}
    a = _min(ade, fde, {4: 0.5, 6: 0.5, 8: 0.5})
    b = _min(ade, fde, {4: 0.9, 6: 0.9, 8: 0.5})
    assert (a["minimum_history"], b["minimum_history"]) == (4, 8)


def test_audc_tolerance_is_sign_safe():
    """AUDC_best = -0.5: -0.46 is 0.08 away relatively, and -0.40 is 0.20 away."""
    r = _min(audc={4: -0.40, 6: -0.46, 8: -0.5})
    assert r["minimum_history"] == 6
    assert r["per_history"]["6"]["audc_tolerance_distance"] == pytest.approx(0.08, abs=1e-12)


def test_small_best_audc_uses_epsilon_in_the_denominator():
    """max(|0.001|, 0.01) = 0.01, so 0.0015 is 0.05 away and passes."""
    assert _min(audc={4: 0.0015, 6: 0.5, 8: 0.001})["minimum_history"] == 4


def test_audc_best_of_zero_uses_the_absolute_criterion():
    assert _min(audc={4: 0.02, 6: 0.009, 8: 0.0})["minimum_history"] == 6


def test_uncomputable_history_fails_and_is_left_out_of_audc_best():
    r = _min(audc={4: None, 6: 0.5, 8: 0.55})
    assert r["minimum_history"] == 6 and r["audc_best"] == 0.5
    assert r["per_history"]["4"]["audc_ok"] is False and r["per_history"]["4"]["audc"] is None


def test_no_computable_history_is_uncomputable_with_no_substitute():
    r = _min(audc={4: None, 6: None, 8: None})
    assert r["status"] == "uncomputable" and r["minimum_history"] is None and r["audc_best"] is None


def test_no_history_meeting_all_criteria_is_a_trade_off():
    r = _min(ade={4: 2.0, 6: 2.0, 8: 1.0}, audc={4: 0.5, 6: 0.5, 8: 0.9})
    assert r["status"] == "trade_off" and r["minimum_history"] is None and r["audc_best"] == 0.5


def test_mismatched_histories_are_rejected():
    with pytest.raises(ValueError, match="same history lengths"):
        _min(fde={4: 1.0})


# --- P-15 partial folds (decision 0005, section 1) ---

def test_a_history_with_five_valid_folds_qualifies():
    r = _min(audc={4: 0.5, 6: 0.5, 8: 0.5}, folds={4: 5, 6: 5, 8: 5})
    row = r["per_history"]["4"]
    assert r["minimum_history"] == 4 and row["audc_qualifies"] is True and row["partial_folds"] is False


@pytest.mark.parametrize("valid", [4, 1])
def test_a_partial_history_keeps_its_mean_and_counts_but_cannot_qualify(valid):
    r = _min(audc={4: 0.5, 6: 0.5, 8: 0.5}, folds={4: valid, 6: 5, 8: 5})
    row = r["per_history"]["4"]
    assert r["minimum_history"] == 6
    assert row["audc"] == 0.5 and row["valid_folds"] == valid and row["partial_folds"] is True
    assert row["audc_qualifies"] is False and row["audc_ok"] is False and row["audc_tolerance_distance"] is None


def test_no_fully_valid_history_is_uncomputable_and_partial_values_stay_reported():
    r = _min(audc={4: 0.5, 6: 0.4, 8: None}, folds={4: 4, 6: 1, 8: 0})
    assert r["status"] == "uncomputable" and r["minimum_history"] is None and r["audc_best"] is None
    assert "every fold" in r["reason"] and r["per_history"]["6"]["audc"] == 0.4
    assert r["per_history"]["8"]["partial_folds"] is True


def test_a_favourable_partial_history_does_not_set_audc_best():
    """H = 8 has the lowest mean but one fold. Counting it would put AUDC_best at -0.9 and fail H = 4 and H = 6."""
    r = _min(audc={4: 0.5, 6: 0.52, 8: -0.9}, folds={4: 5, 6: 5, 8: 1})
    assert r["audc_best"] == 0.5 and r["minimum_history"] == 4
    assert r["per_history"]["8"]["audc"] == -0.9 and r["per_history"]["8"]["audc_ok"] is False


def test_a_partial_history_can_never_be_the_minimum_even_when_only_it_meets_the_clean_criteria():
    r = _min(ade={4: 2.0, 6: 1.0, 8: 2.0}, audc={4: 0.5, 6: 0.5, 8: 0.5}, folds={4: 5, 6: 4, 8: 5})
    assert r["status"] == "trade_off" and r["minimum_history"] is None


def test_clean_criteria_do_not_depend_on_fold_validity():
    r = _min(audc={4: 0.5, 6: 0.5, 8: 0.5}, folds={4: 4, 6: 5, 8: 5})
    assert r["per_history"]["4"]["clean_ade_ok"] is True and r["per_history"]["4"]["clean_fde_ok"] is True


def test_impossible_fold_counts_are_rejected():
    with pytest.raises(ValueError, match="valid folds"):
        _min(folds={4: 6, 6: 5, 8: 5})


# --- H4 on ADE and FDE (decision 0005, section 4) ---

ACHIEVED = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CURVED = [1.0, 1.1, 1.21, 1.5, 2.0, 2.7]
STRAIGHT = [2.0, 2.21, 2.39, 2.62, 2.78, 3.01]
FALLING = [3.0, 2.5, 2.1, 2.2, 2.6, 3.3]


def _d_curves(ade, fde):
    """One predictor over two scenes with identical values, so the scene means are these values."""
    levels = [{"ade_m": {"mean": a}, "fde_m": {"mean": f}, "achieved_seed_mean": x} for a, f, x in zip(ade, fde, ACHIEVED)]
    return {s: {"CV": {"D": {"levels": levels}}} for s in ("S1", "S2")}


def _h4(ade, fde):
    return hypotheses._h4(_d_curves(ade, fde), ["S1", "S2"], ["CV"], 1.5, 2.0)


def test_h4_reports_ade_and_fde_separately_and_agreement():
    r = _h4(CURVED, CURVED)["values"]["CV"]
    assert r["ade"] == r["fde"] and r["ade"]["criterion_met"] is True
    assert r["agreement"]["comparable"] is True and r["agreement"]["disagreements"] == []
    assert "slope_ratio" in r["agreement"]["compared"]


def test_h4_ade_calculation_is_unchanged_by_the_fde_extension():
    """The halves are hand-checked: b_lo over x <= 0.5 and b_hi over x >= 0.5, from numpy's own least squares."""
    r = _h4(CURVED, STRAIGHT)["values"]["CV"]["ade"]
    b_lo, b_hi = np.polyfit(ACHIEVED[:3], CURVED[:3], 1)[0], np.polyfit(ACHIEVED[3:], CURVED[3:], 1)[0]
    assert r["b_lo"] == pytest.approx(b_lo, abs=1e-12) and r["b_hi"] == pytest.approx(b_hi, abs=1e-12)
    assert r["slope_ratio"] == pytest.approx(b_hi / b_lo, abs=1e-12) and r["threshold"] == 1.5


def test_h4_disagreement_is_explicit_and_not_resolved():
    r = _h4(CURVED, STRAIGHT)["values"]["CV"]
    assert r["ade"]["criterion_met"] is True and r["fde"]["criterion_met"] is False
    assert "slope_ratio" in r["agreement"]["disagreements"]


def test_h4_falling_lower_half_fails_that_metric_only():
    r = _h4(FALLING, CURVED)["values"]["CV"]
    assert r["ade"]["slope_ratio"] is None and r["ade"]["criterion_met"] is False and "b_lo <= 0" in r["ade"]["note"]
    assert r["fde"]["criterion_met"] is True and "slope_ratio" in r["agreement"]["disagreements"]


def test_h4_uncomputable_metric_is_not_called_an_agreement():
    r = _h4(CURVED, [*CURVED[:-1], None])["values"]["CV"]
    assert r["fde"]["status"] == "uncomputable" and "criterion_met" in r["ade"]
    assert r["agreement"]["comparable"] is False and r["agreement"]["disagreements"] == []


def test_h3_and_h6_zero_level_handling_is_a_labelled_assumption_and_verdicts_stay_not_evaluable():
    assert all(f"Section {x}" in hypotheses.ZERO_LEVEL.format(x, "range") for x in ("2.3.7", "2.3.9"))
    assert "not a frozen v4.15 requirement" in hypotheses.ZERO_LEVEL and "reconciliation" in hypotheses.ZERO_LEVEL
    assert all(v["research_verdict"] == "not_evaluable" for v in hypotheses.not_evaluated("synthetic_engineering").values())
