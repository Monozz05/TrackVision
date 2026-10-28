"""Reserved: the six mandatory stabilization composition tests (spec v4.15 section 4.3).

Not implemented in Phase 0A. Each test skips and asserts nothing. They are built in
Implementation Plan 'Phase 1 - Data and scene geometry' (spec section 14.1, phase P2), and all
six must pass, strictly, before that phase closes. This gate is not downgradable.
"""

import pytest

PHASE = "Implementation Plan 'Phase 1 - Data and scene geometry' (spec §14.1 P2)"


def _reserved(test_name: str, row: str) -> None:
    pytest.skip(f"RESERVED {test_name} - spec v4.15 §4.3 'Mandatory composition tests', "
                f"row '{row}'. Built in {PHASE}. Not implemented; asserts nothing.")


@pytest.mark.phase1
def test_composition_order():
    _reserved("test_composition_order", "Composition order")


@pytest.mark.phase1
def test_identity_pose():
    _reserved("test_identity_pose", "Identity pose")


@pytest.mark.phase1
def test_frame_convention():
    _reserved("test_frame_convention", "Frame convention")


@pytest.mark.phase1
def test_dataset_conversion():
    _reserved("test_dataset_conversion", "Dataset conversion")


@pytest.mark.phase1
def test_static_object_invariance():
    _reserved("test_static_object_invariance", "Static object invariance")


@pytest.mark.phase1
def test_round_trip():
    _reserved("test_round_trip", "Round-trip")
