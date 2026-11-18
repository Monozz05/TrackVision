"""Tracker error-profile diagnostics: categories, denominators, units. Every count is hand-calculated."""

import numpy as np
import pytest

from trackvision.metrics.error_profile import (MatchedFrame, error_profile, identity_exchanges, match_frames,
                                               track_frame_denominator, transitions)
from trackvision.metrics.tracking import FrameTracking


def _frames(*assignments, eligible=(1, 2)):
    """One MatchedFrame per assignment dict; every eligible identity counts at every timestep."""
    return [MatchedFrame(k, tuple(eligible), dict(a), {g: 2.0 for g in a}) for k, a in enumerate(assignments)]


def _profile(frames, frag=5, switch=1, **kw):
    return error_profile(frames, frag, switch, "pixels", track_frame_denominator(frames), **kw)


def test_transitions_count_unmatched_eligible_timesteps():
    moves = transitions(_frames({1: 7}, {}, {}, {1: 8}, eligible=(1,)))
    assert [(m.gt, m.before, m.after, m.gap, m.old, m.new) for m in moves] == [(1, 0, 3, 2, 7, 8)]


def test_a_new_id_after_a_short_gap_is_fragmentation():
    p = _profile(_frames({1: 7}, {}, {}, {1: 8}, eligible=(1,)))
    assert p["fragmentation"]["events"] == 1 and p["identity_exchange"]["events"] == 0


def test_a_gap_of_exactly_the_limit_counts_and_one_more_is_a_new_track():
    assert _profile(_frames({1: 7}, *[{}] * 5, {1: 8}, eligible=(1,)))["fragmentation"]["events"] == 1
    p = _profile(_frames({1: 7}, *[{}] * 6, {1: 8}, eligible=(1,)))
    assert p["fragmentation"]["events"] == 0 and p["other_identity_changes"]["after_long_gap"] == 1


def test_a_direct_hand_over_is_not_fragmentation():
    p = _profile(_frames({1: 7}, {1: 8}, eligible=(1,)))
    assert p["fragmentation"]["events"] == 0 and p["other_identity_changes"]["direct_hand_over"] == 1


def test_an_old_id_that_comes_back_never_terminated():
    p = _profile(_frames({1: 7}, {}, {1: 8}, {}, {1: 7}, eligible=(1,)))
    assert p["fragmentation"]["events"] == 1 and p["other_identity_changes"]["old_id_returned"] == 1
    assert p["other_identity_changes"]["transitions_total"] == 2


def test_a_two_object_exchange_is_one_event_and_not_fragmentation():
    p = _profile(_frames({1: 7, 2: 8}, {1: 8, 2: 7}))
    assert p["identity_exchange"]["events"] == 1 and p["fragmentation"]["events"] == 0


def test_a_single_missing_timestep_is_permitted_in_an_exchange():
    p = _profile(_frames({1: 7, 2: 8}, {}, {1: 8, 2: 7}))
    assert p["identity_exchange"]["events"] == 1


def test_a_longer_gap_breaks_the_exchange_and_is_handled_as_fragmentation():
    p = _profile(_frames({1: 7, 2: 8}, {}, {}, {1: 8, 2: 7}))
    assert p["identity_exchange"]["events"] == 0 and p["fragmentation"]["events"] == 2


def test_two_disjoint_exchanges_at_one_timestep_are_counted_separately():
    frames = _frames({1: 5, 2: 6, 3: 7, 4: 8}, {1: 6, 2: 5, 3: 8, 4: 7}, eligible=(1, 2, 3, 4))
    assert len(identity_exchanges(transitions(frames), 1)) == 2


def test_a_three_way_rotation_is_not_a_two_object_exchange():
    frames = _frames({1: 5, 2: 6, 3: 7}, {1: 6, 2: 7, 3: 5}, eligible=(1, 2, 3))
    assert identity_exchanges(transitions(frames), 1) == []


def test_a_transition_is_used_in_one_exchange_only():
    frames = _frames({1: 5, 2: 6}, {1: 6, 2: 5}, {1: 5, 2: 6})
    assert len(identity_exchanges(transitions(frames), 1)) == 2           # one each way, each pair disjoint


def test_track_frame_denominator_follows_the_section_3_5_convention():
    """Identity 1 has five samples, so 3; identity 2 has two, so 0."""
    frames = [MatchedFrame(k, (1, 2) if k < 2 else (1,), {}, {}) for k in range(5)]
    assert track_frame_denominator(frames) == 3


def test_every_rate_names_its_denominator():
    p = _profile(_frames({1: 7}, {}, {}, {1: 8}, eligible=(1,)))
    assert p["dropout"]["unmatched_gt_observations"] == 2 and p["dropout"]["eligible_gt_observations"] == 4
    assert p["dropout"]["rate"]["value"] == 0.5
    assert p["fragmentation"]["denominator"] == "eligible_track_frames" and p["fragmentation"]["eligible_track_frames"] == 2
    assert p["fragmentation"]["rate"]["value"] == 0.5
    assert p["localization"]["denominator"] == "matched_observations" and p["localization"]["matched_observations"] == 2


def test_zero_denominators_are_undefined_not_zero():
    p = _profile([])
    assert p["dropout"]["rate"] == {"value": None, "reason": "zero denominator"} and p["localization"]["rmse"] is None
    assert p["fragmentation"]["rate"]["value"] is None


def test_an_unavailable_denominator_stays_undefined_with_its_reason():
    p = _profile(_frames({1: 7, 2: 8}, {1: 8, 2: 7}), exchange_reason="needs D_MAX in metres")
    assert p["identity_exchange"]["per_100"] == {"value": None, "reason": "needs D_MAX in metres"}
    assert p["identity_exchange"]["per_100_gt_observations"]["value"] == pytest.approx(25.0)      # 1 of 4 observations


def test_the_exchange_rate_uses_the_denominator_it_was_given():
    p = _profile(_frames({1: 7, 2: 8}, {1: 8, 2: 7}), exchange_denominator=50, exchange_denominator_name="eligible")
    assert p["identity_exchange"]["per_100"]["value"] == 2.0 and p["identity_exchange"]["denominator"] == "eligible"


def test_localization_rmse_is_root_mean_square_in_the_given_unit():
    """Errors 3 and 4: sqrt((9 + 16) / 2)."""
    frames = [MatchedFrame(0, (1, 2), {1: 7, 2: 8}, {1: 3.0, 2: 4.0})]
    p = error_profile(frames, 5, 1, "pixels", 0)
    assert p["localization"]["rmse"] == pytest.approx(np.sqrt(12.5), abs=1e-12) and p["localization"]["unit"] == "pixels"


def test_exchange_counts_never_use_the_reserved_severity_names():
    keys = repr(_profile(_frames({1: 7, 2: 8}, {1: 8, 2: 7})))
    assert "'switch_event_rate'" not in keys and "real_switch_event_rate" not in keys and "'IDSW'" not in keys


def _frame(boxes, ids, gt_boxes, gt_ids, categories):
    t, g = np.array(boxes, dtype=float).reshape(-1, 4), np.array(gt_boxes, dtype=float).reshape(-1, 4)
    return FrameTracking(np.array(ids), t, t[:, :2], np.array(gt_ids), g, np.array(categories), g[:, :2])


def test_matching_is_one_to_one_at_the_threshold_and_ignores_ignore_regions():
    box, far, gone = [0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]
    fr = _frame([box, far], [5, 6], [box, far, gone], [1, 2, 3], ["evaluable", "ignore", "evaluable"])
    m = match_frames([fr], 0.5)[0]
    assert m.assignment == {1: 5} and m.eligible == (1, 3) and m.errors == {1: 0.0}


def test_a_box_below_the_threshold_is_not_matched():
    fr = _frame([[0, 0, 10, 10]], [5], [[6, 0, 16, 10]], [1], ["evaluable"])
    assert match_frames([fr], 0.5)[0].assignment == {}
