"""Phase 4: AUDC, family AUDC and robustness (spec v4.15 section 3.4; decision 0004, section 1).

Every value is hand-calculated. Each table gives achieved severity per requested level, so s_norm is known exactly.
"""

import pytest

from trackvision.inject.severity import normalize_severity
from trackvision.stats.aggregate import aggregate, relative_degradation
from trackvision.stats.audc import AudcResult, TIE_TOLERANCE, audc, family_audc, robustness


def _norm(achieved):
    return normalize_severity("S", "A", {level: {0: v} for level, v in achieved.items()})


def test_ordinary_curve():
    """x = 0, 0.5, 1 and y = 0, 0.2, 0.6: 0.5 * 0.1 + 0.5 * 0.4 = 0.25."""
    r = audc(_norm({0.0: 0.0, 0.1: 1.0, 0.2: 2.0}), {0.0: 0.0, 0.1: 0.2, 0.2: 0.6})
    assert r.status == "computable" and r.value == pytest.approx(0.25, abs=1e-15)
    assert r.points_above_one == 0 and r.tie_groups == ()


def test_points_are_sorted_by_s_norm_not_requested_order():
    """x = 0, 0.8, 0.5, 1 and y = 0, 0.2, 0.5, 0.7. Sorted: 0.125 + 0.105 + 0.09 = 0.32. Requested order gives 0.275."""
    r = audc(_norm({0.0: 0.0, 0.1: 0.8, 0.2: 0.5, 0.3: 1.0}), {0.0: 0.0, 0.1: 0.2, 0.2: 0.5, 0.3: 0.7})
    assert r.value == pytest.approx(0.32, abs=1e-15)


def test_points_above_one_are_retained_but_excluded():
    """Decision 0004's example: x = 0, 5/3, 1 and y = 0, 0.9, 0.6 gives 0.30, not 0.25 or 0.48."""
    r = audc(_norm({0.0: 0.0, 0.1: 5.0, 0.2: 3.0}), {0.0: 0.0, 0.1: 0.9, 0.2: 0.6})
    assert r.value == pytest.approx(0.30, abs=1e-15) and r.points_above_one == 1
    above = next(p for p in r.points if p["level"] == 0.1)
    assert above["s_norm"] == pytest.approx(5 / 3, abs=1e-15) and above["y"] == 0.9 and above["used"] is False


def test_an_undefined_point_above_one_is_retained_and_does_not_affect_the_integral():
    r = audc(_norm({0.0: 0.0, 0.1: 5.0, 0.2: 3.0}), {0.0: 0.0, 0.1: None, 0.2: 0.6}, {0.1: "no_predicted_windows"})
    assert r.status == "computable" and r.value == pytest.approx(0.30, abs=1e-15)
    assert next(p for p in r.points if p["level"] == 0.1)["reason"] == "no_predicted_windows"


def test_tied_x_values_use_the_plain_mean_and_keep_the_originals():
    """x = 0, 0.5, 0.5, 1 and y = 0, 0.2, 0.6, 0.8: the tie is 0.4, so 0.5 * 0.2 + 0.5 * 0.6 = 0.4."""
    r = audc(_norm({0.0: 0.0, 0.1: 1.0, 0.2: 1.0, 0.3: 2.0}), {0.0: 0.0, 0.1: 0.2, 0.2: 0.6, 0.3: 0.8})
    assert r.value == pytest.approx(0.4, abs=1e-15) and r.tie_groups == ((0.1, 0.2),)
    assert [p["y"] for p in r.points] == [0.0, 0.2, 0.6, 0.8]


def test_a_tie_at_zero_with_non_zero_y_is_averaged_without_any_extra_failure():
    """Decision 0004 adds no integrity rule here. x = 0, 0, 1 and y = 0, 0.4, 0.8: (0.2 + 0.8) / 2 = 0.5."""
    r = audc(_norm({0.0: 0.0, 0.1: 0.0, 0.2: 2.0}), {0.0: 0.0, 0.1: 0.4, 0.2: 0.8})
    assert r.status == "computable" and r.value == pytest.approx(0.5, abs=1e-15)


def test_tie_tolerance_is_absolute_on_s_norm():
    assert TIE_TOLERANCE == 1e-12
    near = {0.0: 0.0, 0.1: 0.5, 0.2: 0.5 + 5e-13, 0.3: 1.0}
    y = {0.0: 0.0, 0.1: 0.2, 0.2: 0.6, 0.3: 0.8}
    assert audc(_norm(near), y).tie_groups == ((0.1, 0.2),)
    apart = {**near, 0.2: 0.5 + 5e-12}
    assert audc(_norm(apart), y).tie_groups == ()


@pytest.mark.parametrize("offset", [5e-13, -5e-13])
def test_a_point_within_tolerance_of_one_is_treated_as_one(offset):
    r = audc(_norm({0.0: 0.0, 0.1: 1.0 + offset, 0.2: 1.0}), {0.0: 0.0, 0.1: 0.2, 0.2: 0.6})
    assert r.points_above_one == 0 and r.tie_groups == ((0.1, 0.2),)
    assert r.value == pytest.approx((0.0 + 0.4) / 2, abs=1e-12)


def test_undefined_interior_level_makes_audc_uncomputable_and_is_never_bridged():
    r = audc(_norm({0.0: 0.0, 0.1: 1.0, 0.2: 2.0}), {0.0: 0.0, 0.1: None, 0.2: 0.6}, {0.1: "no_predicted_windows"})
    assert r.status == "uncomputable" and r.value is None
    assert r.undefined_levels == (0.1,) and "no_predicted_windows" in r.reason


def test_undefined_top_level_makes_audc_uncomputable():
    r = audc(_norm({0.0: 0.0, 0.1: 1.0, 0.2: 2.0}), {0.0: 0.0, 0.1: 0.2, 0.2: None}, {0.2: "no_predicted_windows"})
    assert r.status == "uncomputable" and r.undefined_levels == (0.2,)


def test_top_level_undefined_is_not_rescued_by_a_point_above_one():
    """A segment from x = 0.5 to x = 5/3 crosses 1, but the top value is undefined, so nothing is interpolated."""
    r = audc(_norm({0.0: 0.0, 0.1: 1.5, 0.2: 5.0, 0.3: 3.0}), {0.0: 0.0, 0.1: 0.3, 0.2: 0.9, 0.3: None})
    assert r.status == "uncomputable" and r.undefined_levels == (0.3,)


def test_s_max_equal_to_s_0_is_uncomputable():
    r = audc(_norm({0.0: 0.0, 0.1: 0.0, 0.2: 0.0}), {0.0: 0.0, 0.1: 0.5, 0.2: 0.6})
    assert r.status == "uncomputable" and r.value is None and "s_max equals s_0" in r.reason
    assert [p["y"] for p in r.points] == [0.0, 0.5, 0.6]                      # observations are retained


def test_zero_clean_baseline_is_undefined_with_the_absolute_difference_kept():
    rel = relative_degradation(0.3, 0.0)
    assert rel.value is None and rel.absolute_difference == 0.3 and rel.reason == "zero_clean_baseline"
    norm = _norm({0.0: 0.0, 0.1: 1.0, 0.2: 2.0})
    r = audc(norm, {0.0: None, 0.1: None, 0.2: None}, {lv: rel.reason for lv in norm.levels})
    assert r.status == "uncomputable" and "zero_clean_baseline" in r.reason


def test_audc_comes_from_seed_aggregated_levels_not_from_per_seed_audcs():
    """Seed 1 is undefined at the middle level. Level means: 0, 0.2, 0.7, so AUDC = 0.05 + 0.225 = 0.275.
    Per-seed AUDCs would leave only seed 0: 0.05 + 0.2 = 0.25."""
    table = {0.0: {0: 0.0, 1: 0.0}, 0.1: {0: 0.2, 1: None}, 0.2: {0: 0.6, 1: 0.8}}
    levels = {lv: aggregate(row, {1: "no_predicted_windows"}) for lv, row in table.items()}
    assert (levels[0.1].n_valid, levels[0.1].n_undefined) == (1, 1)
    r = audc(_norm({0.0: 0.0, 0.1: 1.0, 0.2: 2.0}), {lv: a.mean for lv, a in levels.items()})
    assert r.value == pytest.approx(0.275, abs=1e-15)


def test_mismatched_levels_are_rejected():
    with pytest.raises(ValueError, match="do not match"):
        audc(_norm({0.0: 0.0, 0.1: 1.0}), {0.0: 0.0})


def _result(value):
    if value is None:
        return AudcResult("uncomputable", None, "undefined top level")
    return AudcResult("computable", value, None)


def test_family_audc_is_the_mean_of_valid_predictors():
    r = family_audc({"CV": _result(0.2), "CA": _result(0.4), "LSTM": _result(None), "GRU": _result(0.6)})
    assert r["value"] == pytest.approx(0.4, abs=1e-15) and (r["n_valid"], r["n_undefined"]) == (3, 1)
    assert r["undefined"] == {"LSTM": "undefined top level"}


def test_family_audc_needs_two_valid_predictors():
    r = family_audc({"CV": _result(0.2), "CA": _result(None), "LSTM": _result(None), "GRU": _result(None)})
    assert r["status"] == "uncomputable" and r["value"] is None and r["n_valid"] == 1


def test_robustness_is_undefined_at_minus_one_and_never_clipped():
    assert robustness(1.0)["value"] == 0.5 and robustness(-0.5)["value"] == 2.0
    assert robustness(-1.0)["value"] is None and robustness(None)["value"] is None
