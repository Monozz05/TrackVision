"""Reserved: the six data-leakage checks (spec v4.15 section 11.7).

Not implemented in Phase 0A. Each test skips and asserts nothing. They are built in
Implementation Plan 'Phase 2 - Predictors and the clean baseline' (spec section 14.1, phase P3),
after which they run before every experiment and mark it invalid on failure.
"""

import pytest

PHASE = "Implementation Plan 'Phase 2 - Predictors and the clean baseline' (spec §14.1 P3)"


def _reserved(test_name: str, row: str) -> None:
    pytest.skip(f"RESERVED {test_name} - spec v4.15 §11.7 'Data leakage detection', "
                f"row '{row}'. Built in {PHASE}. Not implemented; asserts nothing.")


@pytest.mark.phase2
def test_fold_disjointness():
    _reserved("test_fold_disjointness", "Fold disjointness")


@pytest.mark.phase2
def test_scene_disjointness():
    _reserved("test_scene_disjointness", "Scene disjointness")


@pytest.mark.phase2
def test_target_purity():
    _reserved("test_target_purity", "Target purity")


@pytest.mark.phase2
def test_temporal_ordering():
    _reserved("test_temporal_ordering", "Temporal ordering")


@pytest.mark.phase2
def test_window_boundary():
    _reserved("test_window_boundary", "Window boundary")


@pytest.mark.phase2
def test_normalization_scope():
    _reserved("test_normalization_scope", "Normalization scope")
