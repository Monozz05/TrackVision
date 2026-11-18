"""The six data-leakage checks (spec v4.15 section 11.7), built in Implementation Plan 'Phase 2 -
Predictors and the clean baseline' (spec section 14.1, phase P3).

Every check is shown to pass on a valid split and to fail on deliberately constructed violations of
exactly what it guards. Violating splits are built directly as NestedSplit records (bypassing the
builder, which would refuse them), because the checks must not rely on the builder being right. Scenes
are synthetic (fixtures/synthetic_split.py); the rotation is a fixture mapping, not the project's
unresolved inner-validation rotation.
"""

import dataclasses

import numpy as np
import pytest

from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.predict.displacements import future_displacements, observed_displacements
from trackvision.predict.folds import nested_split, nested_splits
from trackvision.predict.leakage import (CHECKS, NORMALIZATION_NOT_USED, PARTITIONS, LeakageError, array_hash,
                                         fold_disjointness, normalization_scope, require_no_leakage,
                                         run_leakage_checks, scene_disjointness, target_purity, temporal_ordering,
                                         window_boundary)
from trackvision.predict.windowing import Windows, make_windows
from trackvision.synthetic.tracks import GridTrack

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, STRIDE = STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"]
SCENES = ss.make_scenes(4)
WINDOWS = {w.scene: w for w in ss.windows_of(SCENES, CFG)}
SPLIT = nested_split(WINDOWS.values(), "S1", ss.cyclic_rotation(WINDOWS))     # test S1, val S2, train S3 S4


def inputs_from_history(w: Windows) -> np.ndarray:
    return observed_displacements(w.observed)


def fit_displacement_stats(windows) -> dict:
    d = np.concatenate([observed_displacements(w.observed).reshape(-1, 2) for w in windows if len(w)])
    return {"mean": d.mean(axis=0), "std": d.std(axis=0)}


TRAIN_STATS = fit_displacement_stats(SPLIT.train)


def _valid_stats():
    return {p: TRAIN_STATS for p in PARTITIONS}


def _empty(scene):
    return make_windows(scene, [], ss.FRAME_STEP, H, F, STRIDE)


def _fails(result, fragment):
    assert not result.passed, result
    assert any(fragment in p for p in result.problems), result.problems


def test_valid_split_is_the_expected_one():
    assert (SPLIT.test_scene, SPLIT.validation_scene, SPLIT.train_scenes) == ("S1", "S2", ("S3", "S4"))
    assert all(len(w) > 0 for w in WINDOWS.values())


# --- 1. fold disjointness ----------------------------------------------------------------------

@pytest.mark.phase2
def test_fold_disjointness():
    assert fold_disjointness(SPLIT).passed
    _fails(fold_disjointness(dataclasses.replace(SPLIT, validation_scene="S3")), "in both train and validation")


@pytest.mark.parametrize("change, fragment", [
    ({"train_scenes": ("S1", "S4")}, "in both train and test"),
    ({"validation_scene": "S1"}, "in both validation and test"),
    ({"train_scenes": ()}, "train declares no scene"),
    ({"train_scenes": ("S3", "S3")}, "training scenes repeat"),
    ({"validation": WINDOWS["S1"]}, "trajectory ('S1', 1) occurs in both validation and test"),
])
def test_fold_disjointness_detects(change, fragment):
    _fails(fold_disjointness(dataclasses.replace(SPLIT, **change)), fragment)


# --- 2. scene disjointness ---------------------------------------------------------------------

@pytest.mark.phase2
def test_scene_disjointness():
    assert scene_disjointness(SPLIT).passed
    leaked = dataclasses.replace(SPLIT, train=(WINDOWS["S3"], WINDOWS["S1"]))      # test windows in training
    _fails(scene_disjointness(leaked), "test scene 'S1' is present in the training set")


@pytest.mark.parametrize("change, fragment", [
    ({"train": (WINDOWS["S3"], WINDOWS["S2"])}, "windows of scene 'S2' are in train"),
    ({"train": (WINDOWS["S3"], WINDOWS["S2"])}, "in both train and validation"),
    ({"validation": _empty("S2")}, "validation has no windows to test"),
])
def test_scene_disjointness_detects(change, fragment):
    _fails(scene_disjointness(dataclasses.replace(SPLIT, **change)), fragment)


def test_scene_disjointness_detects_a_window_assigned_twice():
    w = WINDOWS["S3"]
    doubled = dataclasses.replace(w, observed=np.concatenate([w.observed, w.observed[:1]]),
                                  target=np.concatenate([w.target, w.target[:1]]),
                                  track_ids=np.concatenate([w.track_ids, w.track_ids[:1]]),
                                  frames=np.concatenate([w.frames, w.frames[:1]]))
    _fails(scene_disjointness(dataclasses.replace(SPLIT, train=(doubled, WINDOWS["S4"]))), "occurs in train and again")


# --- 3. target purity --------------------------------------------------------------------------

@pytest.mark.phase2
def test_target_purity():
    assert target_purity(SPLIT, inputs_from_history).passed

    def peeks(w):          # appends the first target displacement to the encoder input
        return np.concatenate([observed_displacements(w.observed), future_displacements(w.observed, w.target)[:, :1]], 1)
    _fails(target_purity(SPLIT, peeks), "change when the targets change")


def test_target_purity_detects_target_dependent_preprocessing():
    def scaled_by_target_spread(w):          # a 'preprocessing' step fitted on the targets
        return observed_displacements(w.observed) / (np.std(w.target) + 1.0)
    _fails(target_purity(SPLIT, scaled_by_target_spread), "change when the targets change")


def test_target_purity_detects_inputs_that_modify_the_targets():
    originals = {name: array_hash(w.target) for name, w in WINDOWS.items()}

    def corrupts_in_place(win):
        win.target[:, 0] += 0.5             # a corruption step written into the target by mistake
        return observed_displacements(win.observed)
    copy = lambda w: dataclasses.replace(w, target=w.target.copy(), observed=w.observed.copy())
    own = dataclasses.replace(SPLIT, test=copy(SPLIT.test), validation=copy(SPLIT.validation),
                              train=tuple(copy(w) for w in SPLIT.train))
    _fails(target_purity(own, corrupts_in_place), "changed the target or observed arrays of test scene 'S1'")
    assert {name: array_hash(w.target) for name, w in WINDOWS.items()} == originals   # shared fixtures untouched


@pytest.mark.parametrize("make_inputs, split, fragment", [
    (None, SPLIT, "no input builder"),
    (inputs_from_history, dataclasses.replace(SPLIT, test=_empty("S1")), "test has no windows to test"),
])
def test_target_purity_never_passes_on_missing_inputs(make_inputs, split, fragment):
    _fails(target_purity(split, make_inputs), fragment)


# --- 4. temporal ordering ----------------------------------------------------------------------

def _with_frames(w, frames):
    return dataclasses.replace(w, frames=frames)


@pytest.mark.phase2
def test_temporal_ordering():
    assert temporal_ordering(SPLIT).passed
    w = WINDOWS["S1"]
    frames = w.frames.copy()
    frames[2] = frames[2][::-1]                                     # one window reversed
    _fails(temporal_ordering(dataclasses.replace(SPLIT, test=_with_frames(w, frames))), "window 2")


def test_temporal_ordering_detects_overlapping_input_and_target_times():
    w = WINDOWS["S3"]
    frames = w.frames.copy()
    frames[0, H] = frames[0, H - 1]                                 # first target at the last observed time
    frames[1, H:] = frames[1, 0] - np.arange(F, 0, -1)              # every target before the history
    result = temporal_ordering(dataclasses.replace(SPLIT, train=(_with_frames(w, frames), WINDOWS["S4"])))
    _fails(result, "window 0 (track 1) has a target at or before an observation")
    _fails(result, "window 1 (track 1) has a target at or before an observation")


def test_temporal_ordering_detects_non_increasing_history_and_malformed_frames():
    w = WINDOWS["S2"]
    frames = w.frames.copy()
    frames[0, 3] = frames[0, 2]                                     # a repeated timestamp inside the history
    _fails(temporal_ordering(dataclasses.replace(SPLIT, validation=_with_frames(w, frames))), "non-increasing")
    _fails(temporal_ordering(dataclasses.replace(SPLIT, validation=_with_frames(w, w.frames[:, :-1]))), "frames have shape")
    _fails(temporal_ordering(dataclasses.replace(SPLIT, validation=_empty("S2"))), "validation has no windows")


# --- 5. window boundary ------------------------------------------------------------------------

@pytest.mark.phase2
def test_window_boundary():
    assert window_boundary(SPLIT, SCENES).passed
    w = WINDOWS["S1"]
    spliced = w.target.copy()
    spliced[0] = WINDOWS["S2"].target[0]                           # future taken from another scene
    _fails(window_boundary(dataclasses.replace(SPLIT, test=dataclasses.replace(w, target=spliced)), SCENES),
           "window 0 (track 1): positions differ from the source track")


def test_window_boundary_detects_a_window_relabelled_into_another_scene():
    moved = dataclasses.replace(WINDOWS["S3"], scene="S1")         # S3's windows claimed by S1
    _fails(window_boundary(dataclasses.replace(SPLIT, test=moved), SCENES), "test scene 'S1'")


@pytest.mark.parametrize("edit, fragment", [
    (lambda w: dataclasses.replace(w, frames=w.frames + 10_000), "frames outside the track"),
    (lambda w: dataclasses.replace(w, track_ids=w.track_ids + 99), "no such track in the scene"),
])
def test_window_boundary_detects_indices_outside_the_scene(edit, fragment):
    _fails(window_boundary(dataclasses.replace(SPLIT, test=edit(WINDOWS["S1"])), SCENES), fragment)


def test_window_boundary_detects_a_window_across_a_gap():
    frames = ss.FRAME_STEP * np.concatenate([np.arange(10), np.arange(15, 30)])
    positions = np.column_stack([0.3 * np.arange(25.0), np.zeros(25)])
    gap_scene = ss.FakeScene("G", (GridTrack(1, frames, positions),))
    across = Windows("G", H, F, STRIDE, positions[None, :H], positions[None, H:H + F],
                     np.array([1]), frames[None, :H + F])
    split = dataclasses.replace(SPLIT, test_scene="G", test=across)
    _fails(window_boundary(split, [*SCENES, gap_scene]), "crosses a gap or a boundary")
    assert len(make_windows("G", gap_scene.tracks, ss.FRAME_STEP, H, F, STRIDE)) == 0   # windowing never cuts it


def test_window_boundary_never_passes_without_source_scenes():
    _fails(window_boundary(SPLIT, [s for s in SCENES if s.name != "S2"]), "no source scene")
    _fails(window_boundary(SPLIT, []), "no source scene")


# --- 6. normalization scope --------------------------------------------------------------------

@pytest.mark.phase2
def test_normalization_scope():
    assert normalization_scope(SPLIT, fit_displacement_stats, _valid_stats()).passed
    declared = normalization_scope(SPLIT, None, NORMALIZATION_NOT_USED)
    assert declared.passed and declared.note == "normalization declared not used"
    stats = _valid_stats()
    stats["validation"] = fit_displacement_stats((SPLIT.validation,))
    _fails(normalization_scope(SPLIT, fit_displacement_stats, stats),
           "applied to validation were not fitted on the training windows only (they match statistics fitted on validation)")


@pytest.mark.parametrize("part, source, hint", [
    ("test", lambda s: (s.test,), "fitted on test"),
    ("train", lambda s: (*s.train, s.validation, s.test), "fitted on train + validation + test"),
    ("train", lambda s: (*s.train, s.validation), "applied to train were not fitted"),
])
def test_normalization_scope_detects_statistics_from_held_out_data(part, source, hint):
    stats = _valid_stats()
    stats[part] = fit_displacement_stats(source(SPLIT))
    _fails(normalization_scope(SPLIT, fit_displacement_stats, stats), hint)


@pytest.mark.parametrize("fit, stats, fragment", [
    (fit_displacement_stats, None, "not declared"),
    (fit_displacement_stats, "none", "not declared"),
    (None, {p: TRAIN_STATS for p in PARTITIONS}, "no fitting function"),
    (fit_displacement_stats, {"train": TRAIN_STATS, "validation": TRAIN_STATS}, "no statistics recorded for test"),
    (fit_displacement_stats, {**{p: TRAIN_STATS for p in PARTITIONS}, "extra": TRAIN_STATS}, "unknown partition"),
    (fit_displacement_stats, {**{p: TRAIN_STATS for p in PARTITIONS}, "test": {"mean": TRAIN_STATS["mean"]}},
     "applied to test"),
])
def test_normalization_scope_never_passes_on_missing_or_partial_declarations(fit, stats, fragment):
    _fails(normalization_scope(SPLIT, fit, stats), fragment)


# --- all six together --------------------------------------------------------------------------

def test_every_builder_fold_passes_all_six_checks():
    windows = ss.windows_of(ss.make_scenes(5), CFG)
    scenes = ss.make_scenes(5)
    for split in nested_splits(windows, ss.cyclic_rotation(w.scene for w in windows)):
        stats = fit_displacement_stats(split.train)
        results = run_leakage_checks(split, scenes, inputs_from_history, fit_displacement_stats,
                                     {p: stats for p in PARTITIONS})
        assert [r.name for r in results] == list(CHECKS)
        require_no_leakage(results)


def test_any_failed_check_blocks_the_experiment():
    leaked = dataclasses.replace(SPLIT, train=(WINDOWS["S3"], WINDOWS["S1"]))
    results = run_leakage_checks(leaked, SCENES, inputs_from_history, None, NORMALIZATION_NOT_USED)
    with pytest.raises(LeakageError, match="scene_disjointness: test scene 'S1'"):
        require_no_leakage(results)
    good = run_leakage_checks(SPLIT, SCENES, inputs_from_history, None, NORMALIZATION_NOT_USED)
    require_no_leakage(good)
    with pytest.raises(LeakageError, match="expected the checks"):
        require_no_leakage(good[:-1])
