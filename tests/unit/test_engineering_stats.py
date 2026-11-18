"""E5 to E8 arithmetic and across-scene family AUDC. Every value is hand-calculated."""

import inspect
import math

import pytest

from trackvision.stats import association_strength as a, crossing, dominant, inverse
from trackvision.stats.association_strength import association_summary, average_ranks, spearman
from trackvision.stats.audc import AudcResult, family_audc
from trackvision.stats.crossing import first_crossing
from trackvision.stats.dominant import ENGINEERING_DRY_RUN, select_dominant_mode
from trackvision.stats.family import across_scenes
from trackvision.stats.inverse import inverse_requirement

GAP = 0.1


def _dry(**values):
    return select_dominant_mode(values, GAP)


# --- E6: dominant mode ---------------------------------------------------------------------------

def test_the_largest_positive_family_is_the_single_target():
    r = _dry(A=0.9, B2=0.2, C=-0.1, D=0.3)
    assert r["target"] == "A" and r["targets"] == ["A"] and r["co_dominant"] is False
    assert r["top_gap"] == pytest.approx(0.6, abs=1e-12)


def test_two_families_within_the_gap_are_co_dominant():
    r = _dry(A=0.50, B2=0.1, C=0.45, D=0.2)
    assert r["co_dominant"] is True and r["targets"] == ["A", "C"] and r["target"] is None


def test_equal_top_values_are_co_dominant():
    assert _dry(A=0.5, B2=0.5, C=0.1, D=0.1)["targets"] == ["A", "B2"]


def test_a_gap_of_exactly_the_threshold_is_not_co_dominant_despite_binary_rounding():
    """0.3 - 0.2 is 0.09999999999999998 in floating point; the rule says 'less than 0.10'."""
    assert 0.3 - 0.2 < 0.1
    r = _dry(A=0.3, B2=0.2, C=0.0, D=0.0)
    assert r["co_dominant"] is False and r["targets"] == ["A"]


def test_the_gap_tolerance_is_labelled_a_project_convention():
    assert dominant.GAP_TOLERANCE == 1e-12 and "project-level numerical convention" in dominant.__doc__


def test_only_the_top_two_can_be_co_dominant():
    assert _dry(A=0.5, B2=0.45, C=0.42, D=0.41)["targets"] == ["A", "B2"]


def test_co_dominance_is_assessed_only_among_qualifying_families():
    assert _dry(A=0.05, B2=-0.01, C=0.0, D=-0.02)["targets"] == ["A"]


@pytest.mark.parametrize("values", [{"A": 0.0, "B2": 0.0, "C": 0.0, "D": 0.0}, {"A": -0.1, "B2": -2.0, "C": 0.0, "D": -0.5},
                                    {"A": None, "B2": None, "C": None, "D": None}, {}])
def test_no_qualifying_family_selects_nothing(values):
    r = select_dominant_mode(values, GAP)
    assert r["target"] is None and r["targets"] == [] and r["co_dominant"] is False and r["top_gap"] is None
    assert "no candidate family" in r["no_target_reason"]


@pytest.mark.parametrize("bad, word", [(math.inf, "not finite"), (-math.inf, "not finite"), (math.nan, "not finite"),
                                       (0.0, "not positive"), (-0.5, "not positive")])
def test_non_finite_zero_and_negative_values_never_qualify(bad, word):
    r = _dry(A=bad, B2=0.2, C=None, D=0.05)
    assert r["candidates"]["A"]["qualifies"] is False and word in r["candidates"]["A"]["reason"]
    assert r["targets"] == ["B2"]                      # D is 0.15 below B2, so it is not co-dominant


def test_undefined_values_keep_their_reason_and_a_missing_family_is_recorded():
    r = select_dominant_mode({"A": None, "B2": 0.2}, GAP, {"A": "no scene has a valid family AUDC"})
    assert r["candidates"]["A"]["reason"] == "no scene has a valid family AUDC"
    assert r["candidates"]["C"]["reason"] == "family_missing" and r["target"] == "B2"


def test_a_failed_feasibility_gate_excludes_the_family_and_records_why():
    r = select_dominant_mode({"A": 0.9, "B2": 0.2}, GAP, excluded={"A": "gate failed"})
    assert r["target"] == "B2" and r["candidates"]["A"] == {"value": 0.9, "qualifies": False, "reason": "excluded: gate failed"}


@pytest.mark.parametrize("name", ["E", "E3", "B1", "B"])
def test_nothing_outside_the_four_static_candidates_is_compared(name):
    with pytest.raises(ValueError, match="never compared"):
        select_dominant_mode({"A": 0.2, name: 0.9}, GAP)


def test_booleans_are_not_numbers():
    with pytest.raises(ValueError, match="number or None"):
        _dry(A=True)


def test_the_dry_run_cannot_consume_or_become_the_research_selection():
    first = _dry(A=0.9, B2=0.2)
    second = _dry(A=0.1, B2=0.9)                       # applied twice with different data: nothing was spent
    for r in (first, second):
        assert r["status"] == ENGINEERING_DRY_RUN == "engineering_dry_run"
        assert r["research_selection_applied"] is False and r["research_selection"] == "not_evaluable"
        assert "not the research selection" in r["note"]
    assert first["target"] == "A" and second["target"] == "B2"
    assert set(inspect.signature(select_dominant_mode).parameters) == {"family_audc", "gap", "reasons", "excluded"}


def test_the_frozen_gap_comes_from_the_caller():
    assert select_dominant_mode({"A": 0.5, "B2": 0.45}, 0.01)["co_dominant"] is False


# --- E7: inverse requirement mapping -------------------------------------------------------------

CURVE = [(0.0, 0.0), (1.0, 0.04), (2.0, 0.2), (3.0, 0.02)]


def test_first_crossing_is_linear_between_adjacent_points():
    """Between (1, 0.04) and (2, 0.20): 1 + (0.10 - 0.04) / 0.16 = 1.375."""
    r = inverse_requirement(CURVE, 0.10)
    assert r["status"] == "crossed" and r["s_star"] == pytest.approx(1.375, abs=1e-12)


def test_later_recovery_does_not_restore_the_limit():
    """The curve returns to 0.02 at severity 3, but the contiguous prefix ended at 1.0625."""
    assert inverse_requirement(CURVE, 0.05)["s_star"] == pytest.approx(1.0625, abs=1e-12)


def test_a_curve_that_never_exceeds_q_reports_no_number():
    r = inverse_requirement(CURVE, 0.20)
    assert r["status"] == "not_reached" and r["s_star"] is None and r["maximum_observed_severity"] == 3.0
    assert "s* > maximum observed severity" in r["reason"]


def test_there_is_no_extrapolation_past_the_last_point():
    r = inverse_requirement([(0.0, 0.0), (1.0, 0.09)], 0.10)
    assert r["status"] == "not_reached" and r["s_star"] is None and r["verified_up_to"] == 1.0


def test_an_undefined_point_is_never_bridged():
    r = inverse_requirement([(0.0, 0.0), (1.0, 0.05), (2.0, None), (3.0, 0.9)], 0.10)
    assert r["status"] == "truncated" and r["s_star"] is None and r["verified_up_to"] == 1.0 and "not bridged" in r["reason"]


def test_a_crossing_before_an_undefined_point_is_still_the_crossing():
    assert inverse_requirement([(0.0, 0.0), (1.0, 0.2), (2.0, None)], 0.10)["s_star"] == pytest.approx(0.5, abs=1e-12)


def test_points_are_ordered_by_achieved_severity_not_requested_order():
    """Sorted: (0, 0), (1, 0.1), (2, 0.4). For q = 0.2: 1 + 0.1 / 0.3."""
    r = inverse_requirement([(2.0, 0.4), (0.0, 0.0), (1.0, 0.1)], 0.20)
    assert r["s_star"] == pytest.approx(1 + 1 / 3, abs=1e-12)


def test_tied_severities_use_the_plain_mean():
    """(1, 0.1) and (1, 0.3) become (1, 0.2). For q = 0.15: 0.15 / 0.2 = 0.75."""
    r = inverse_requirement([(0.0, 0.0), (1.0, 0.1), (1.0, 0.3), (2.0, 0.4)], 0.15)
    assert r["s_star"] == pytest.approx(0.75, abs=1e-12)


def test_a_point_exactly_at_q_then_above_stays_at_that_point():
    assert inverse_requirement([(0.0, 0.0), (1.0, 0.1), (2.0, 0.3)], 0.1)["s_star"] == pytest.approx(1.0, abs=1e-12)


def test_degradation_above_q_at_the_first_point_has_no_prefix():
    r = inverse_requirement([(0.0, 0.2), (1.0, 0.3)], 0.1)
    assert r["status"] == "exceeded_at_clean" and r["s_star"] is None and r["verified_up_to"] is None


def test_a_zero_clean_baseline_is_not_mapped():
    r = inverse_requirement([(0.0, None), (1.0, None)], 0.1)
    assert r["status"] == "uncomputable" and r["s_star"] is None


def test_a_point_with_no_achieved_severity_is_not_placed_silently():
    r = inverse_requirement([(0.0, 0.0), (None, 0.5), (1.0, 0.2)], 0.1)
    assert r["status"] == "uncomputable" and r["points_without_achieved_severity"] == 1
    assert inverse_requirement([], 0.1)["status"] == "uncomputable"


def test_q_must_be_positive():
    with pytest.raises(ValueError):
        inverse_requirement(CURVE, 0.0)


# --- E8: threshold crossings ---------------------------------------------------------------------

def test_crossing_between_measured_points():
    """Differences a - b are -1, -1, +1; the sign changes between 1 and 2, half way."""
    r = first_crossing([(0.0, 1.0, 2.0), (1.0, 2.0, 3.0), (2.0, 4.0, 3.0)])
    assert r["status"] == "crossed" and r["s_crossing"] == pytest.approx(1.5, abs=1e-12) and r["lower_before"] == "a"


def test_crossing_exactly_at_a_measured_point():
    r = first_crossing([(0.0, 1.0, 2.0), (1.0, 2.0, 2.0), (2.0, 3.0, 2.0)])
    assert r["s_crossing"] == 1.0 and r["measured_point"] is True


def test_no_crossing_is_not_extrapolated():
    r = first_crossing([(0.0, 1.0, 2.0), (1.0, 2.0, 2.5)])
    assert r["status"] == "no_crossing" and r["s_crossing"] is None and r["lower_before"] == "a"


def test_an_undefined_value_stops_the_search():
    r = first_crossing([(0.0, 1.0, 2.0), (1.0, None, 2.0), (2.0, 4.0, 2.0)])
    assert r["status"] == "truncated" and r["s_crossing"] is None


@pytest.mark.parametrize("points", [[], [(0.0, 1.0, 2.0)], [(None, 1.0, 2.0), (1.0, 1.0, 2.0)], [(0.0, None, 2.0), (1.0, 1.0, 2.0)]])
def test_too_little_data_is_uncomputable(points):
    assert first_crossing(points)["status"] == "uncomputable"


def test_points_are_sorted_by_severity():
    assert first_crossing([(2.0, 4.0, 3.0), (0.0, 1.0, 2.0)])["s_crossing"] == pytest.approx(1.0, abs=1e-12)


def test_crossings_do_not_select_a_predictor_at_run_time():
    """Section 9.3's runtime meaning is unresolved, so nothing here takes an input and returns a predictor."""
    assert first_crossing([(0.0, 1.0, 2.0), (1.0, 2.0, 1.0)])["runtime_selection"] == "not_implemented"
    public = [n for n, v in vars(crossing).items() if inspect.isfunction(v) and not n.startswith("_")]
    assert public == ["first_crossing"]


# --- across-scene family AUDC --------------------------------------------------------------------

def _valid(value):
    return AudcResult("computable", value, None)


def _bad():
    return AudcResult("uncomputable", None, "undefined top level")


def _table():
    """S1: mean of 0.2 and 0.4 is 0.3. S2: one valid predictor, so uncomputable. S3: mean of 0.5, 0.7, 0.9 is 0.7."""
    return {"S1": family_audc({"CV": _valid(0.2), "CA": _valid(0.4), "LSTM": _bad()}),
            "S2": family_audc({"CV": _valid(0.9), "CA": _bad(), "LSTM": _bad()}),
            "S3": family_audc({"CV": _valid(0.5), "CA": _valid(0.7), "LSTM": _valid(0.9)})}


def test_predictors_are_aggregated_within_each_scene_first():
    """(0.3 + 0.7) / 2 = 0.5. Predictors first would be mean(0.2, 0.9, 0.5) etc., which lets S2's lone value in."""
    r = across_scenes(_table())
    assert r["status"] == "computable" and r["value"] == pytest.approx(0.5, abs=1e-12)
    predictor_first = sum([(0.2 + 0.9 + 0.5) / 3, (0.4 + 0.7) / 2, 0.9]) / 3
    assert abs(r["value"] - predictor_first) > 0.1


def test_counts_and_reasons_are_preserved():
    r = across_scenes(_table())
    assert (r["n_valid_scenes"], r["n_undefined_scenes"]) == (2, 1)
    assert r["undefined"] == {"S2": "fewer than two valid predictor AUDC values"}
    assert r["valid_predictors_by_scene"] == {"S1": 2, "S2": 1, "S3": 3} and r["scene_values"]["S2"] is None


def test_one_valid_scene_has_a_mean_and_no_spread():
    r = across_scenes({"S1": family_audc({"CV": _valid(0.2), "CA": _valid(0.4)})})
    assert r["value"] == pytest.approx(0.3, abs=1e-12) and r["sd"] is None and r["n_valid_scenes"] == 1


def test_no_valid_scene_is_uncomputable_with_no_substitute():
    r = across_scenes({"S1": family_audc({"CV": _bad(), "CA": _bad()})})
    assert r["status"] == "uncomputable" and r["value"] is None and "no scene" in r["reason"]


def test_no_scenes_is_an_error():
    with pytest.raises(ValueError):
        across_scenes({})


# --- E5: Spearman and the H7 arithmetic ----------------------------------------------------------

def test_average_ranks_share_ties():
    assert average_ranks([10.0, 20.0, 20.0, 30.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_without_ties():
    """1 - 6 * 2 / (4 * 15) = 0.8."""
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4])["value"] == pytest.approx(0.8, abs=1e-12)


def test_spearman_with_ties_is_pearson_on_ranks():
    """Rank deviations (-1.5, 0, 0, 1.5) and (-1.5, -0.5, 0.5, 1.5): 4.5 / sqrt(4.5 * 5)."""
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4])["value"] == pytest.approx(4.5 / math.sqrt(22.5), abs=1e-12)


def test_spearman_is_undefined_without_variation_or_pairs():
    assert spearman([1, 1, 1], [1, 2, 3]) == {"value": None, "n": 3, "reason": "insufficient variation"}
    assert spearman([1], [2])["value"] is None
    with pytest.raises(ValueError):
        spearman([1, 2], [1])


def _pairs(idsw=(1, 2, 3, 4)):
    ade = (1.0, 2.0, 3.0, 4.0)
    return {f"S{i}": {"ade_m": ade[i], "IDF1": (1, 2, 3, 4)[i], "HOTA": (4, 3, 2, 1)[i], "IDSW": idsw[i],
                      "MOTA": (1, 3, 2, 4)[i]} for i in range(4)}


def test_group_values_use_the_mean_of_three_absolute_correlations():
    """|rho| are 1, 1, 0.8 for the association group, mean 14/15; MOTA gives 0.8, so delta = 2/15."""
    r = association_summary(_pairs(idsw=(1, 3, 2, 4)))
    assert r["abs_rho_assoc"] == pytest.approx(14 / 15, abs=1e-12) and r["abs_rho_det"] == pytest.approx(0.8, abs=1e-12)
    assert r["delta_rho"] == pytest.approx(2 / 15, abs=1e-12)


def test_scene_pairing_is_preserved_whatever_the_input_order():
    pairs = _pairs()
    assert association_summary(dict(reversed(list(pairs.items()))))["correlations"] == association_summary(pairs)["correlations"]


def test_an_undefined_correlation_is_never_imputed_into_the_group():
    r = association_summary(_pairs(idsw=(0, 0, 0, 0)))
    assert r["correlations"]["IDSW"]["value"] is None and r["abs_rho_assoc"] is None and r["delta_rho"] is None


def test_a_scene_without_ade_is_left_out_and_counted():
    pairs = _pairs()
    pairs["S3"]["ade_m"] = None
    r = association_summary(pairs)
    assert r["scene_count"] == 4 and r["paired_observations"] == 3


def test_no_interval_or_verdict_is_invented():
    r = association_summary(_pairs())
    assert r["h7_verdict"] == "not_evaluable" and r["inference"].startswith("uncomputable")
    assert a.NO_INFERENCE in r["inference"] and not any("ci" in k or "bootstrap" in k for k in r)


def test_modules_are_importable_with_their_constants():
    assert dominant.CANDIDATES == ("A", "B2", "C", "D") and inverse.TIE_TOLERANCE == 1e-12
