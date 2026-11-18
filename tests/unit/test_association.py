"""Phase 3: the association mechanism (spec v4.15 section 3.7; proposal 0003).

The value is produced and checked here. It is deliberately not yet a record field: section 11.3 has no such
field, so `test_not_yet_a_record_field` documents the pending state and must change when the amendment is adopted.
"""

import pytest

from conftest import SPEC_PATH
from trackvision.experiments import record as rec
from trackvision.inject.association import (ASSOCIATION_MECHANISMS, GROUND_TRUTH, REAL_TRACKER, REAL_TRACKER_IOU,
                                            SYNTHETIC_GT_LINEAGE, association_errors, association_fields,
                                            association_mechanism)


def test_values_are_spelled_as_in_the_specification():
    assert "association_mechanism = { real_tracker_iou | synthetic_gt_lineage }" in SPEC_PATH.read_text(encoding="utf-8")
    assert ASSOCIATION_MECHANISMS == (REAL_TRACKER_IOU, SYNTHETIC_GT_LINEAGE)
    assert (SYNTHETIC_GT_LINEAGE, REAL_TRACKER_IOU) == ("synthetic_gt_lineage", "real_tracker_iou")


def test_mechanism_follows_the_trajectory_base():
    assert association_mechanism(GROUND_TRUTH) == SYNTHETIC_GT_LINEAGE
    assert association_mechanism(REAL_TRACKER) == REAL_TRACKER_IOU
    assert association_fields(GROUND_TRUTH) == {"association_mechanism": SYNTHETIC_GT_LINEAGE}
    with pytest.raises(ValueError, match="trajectory_base"):
        association_mechanism("synthetic_box_iou")


@pytest.mark.parametrize("value, tracker, gt, association_type", [
    (SYNTHETIC_GT_LINEAGE, None, True, None),
    (REAL_TRACKER_IOU, "ByteTrack", True, None),
    (SYNTHETIC_GT_LINEAGE, None, True, SYNTHETIC_GT_LINEAGE),
    (REAL_TRACKER_IOU, "BoT-SORT", True, REAL_TRACKER_IOU),
    (None, "ByteTrack", False, None),                                         # no ground truth, nothing to associate
])
def test_valid_combinations(value, tracker, gt, association_type):
    assert association_errors(value, tracker, gt, association_type) == []


@pytest.mark.parametrize("value, tracker, gt, association_type, fragment", [
    (REAL_TRACKER_IOU, None, True, None, "category error"),                   # box association on synthetic corruption
    (SYNTHETIC_GT_LINEAGE, "ByteTrack", True, None, "category error"),        # lineage on a real tracker
    (None, None, True, None, "is not one of"),                                # required with ground truth
    ("box_iou", None, True, None, "is not one of"),
    (SYNTHETIC_GT_LINEAGE, None, False, None, "must be null"),
    (SYNTHETIC_GT_LINEAGE, None, True, REAL_TRACKER_IOU, "association_type"),
])
def test_invalid_combinations(value, tracker, gt, association_type, fragment):
    errors = association_errors(value, tracker, gt, association_type)
    assert errors and fragment in errors[0]


def test_not_yet_a_record_field():
    """Section 11.3 of the frozen v4.15 has no such field; proposal 0003 adds it to v4.16."""
    assert "association_mechanism" not in rec.TOP_LEVEL_FIELDS
    assert "association_type" in rec.RESULT_GROUPS["collision"]
