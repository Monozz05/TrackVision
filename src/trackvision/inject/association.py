"""The association mechanism of an experiment (spec v4.15 section 3.7).

    real_tracker_iou       tracker box -> GT box -> Hungarian IoU assignment -> GT identity
    synthetic_gt_lineage   a corrupted trajectory keeps its originating GT identity through the injector's
                           lineage map (`Fragmentation.origin`, `IdentitySwitch.occupant`; A and B never
                           change identity)

Applying box association to a synthetic corruption, or lineage to a real tracker, is a category error.

NOT YET A RECORD FIELD. Section 3.7 says every experiment records this value, but the section 11.3 schema
has no such field, only `results.collision.association_type`, and `experiments.record` mirrors section 11.3
exactly. This module produces and checks the value, ready for the schema change proposed in
docs/decisions/0003-proposed-v4-16-association-mechanism.md. Nothing writes it into a record until that
amendment is adopted.
"""

from __future__ import annotations

SYNTHETIC_GT_LINEAGE = "synthetic_gt_lineage"
REAL_TRACKER_IOU = "real_tracker_iou"
ASSOCIATION_MECHANISMS = (REAL_TRACKER_IOU, SYNTHETIC_GT_LINEAGE)

GROUND_TRUTH = "ground_truth"
REAL_TRACKER = "real_tracker"
_BY_TRAJECTORY_BASE = {GROUND_TRUTH: SYNTHETIC_GT_LINEAGE, REAL_TRACKER: REAL_TRACKER_IOU}


def association_mechanism(trajectory_base: str) -> str:
    """The mechanism for a trajectory base: ground truth (clean or synthetically corrupted) or a real tracker."""
    if trajectory_base not in _BY_TRAJECTORY_BASE:
        raise ValueError(f"trajectory_base must be one of {sorted(_BY_TRAJECTORY_BASE)}, got {trajectory_base!r}")
    return _BY_TRAJECTORY_BASE[trajectory_base]


def association_fields(trajectory_base: str) -> dict:
    """The field to merge into an experiment record once the schema carries it."""
    return {"association_mechanism": association_mechanism(trajectory_base)}


def association_errors(value, tracker, ground_truth_available, association_type=None) -> list[str]:
    """Violations of the validation rules proposed with the schema change (an empty list means valid)."""
    if ground_truth_available is False:
        return [] if value is None else ["association_mechanism: must be null without ground truth"]
    if value not in ASSOCIATION_MECHANISMS:
        return [f"association_mechanism: {value!r} is not one of {list(ASSOCIATION_MECHANISMS)}"]
    errors = []
    expected = REAL_TRACKER_IOU if tracker is not None else SYNTHETIC_GT_LINEAGE
    if value != expected:
        errors.append(f"association_mechanism: {value!r} is a category error here; expected {expected!r}")
    if association_type is not None and association_type != value:
        errors.append(f"results.collision.association_type: {association_type!r} differs from {value!r}")
    return errors
