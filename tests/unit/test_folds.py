"""P2-c: nested leave-one-scene-out split builder (spec v4.15 section 5.4).

Scenes are small synthetic collections (fixtures/synthetic_split.py). The rotation passed to the builder
in these tests is a separate fixture mapping. The project's rotation, the research choice
`inner_validation_rotation` (docs/decisions/0001-p2d-protocol-decisions.md, C), covers the five real
static scenes only and is never accepted for synthetic scenes.
"""

import numpy as np
import pytest

from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.config import UnresolvedChoiceError
from trackvision.io import ewap
from trackvision.predict.folds import (MIN_SCENES, NestedSplit, SplitError, check_rotation, configured_rotation,
                                       nested_split, nested_splits, scene_table)
from trackvision.predict.windowing import make_windows

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]


def _collection(n):
    return ss.windows_of(ss.make_scenes(n), CFG)


@pytest.mark.parametrize("n", [3, 4, 5, 7])
def test_valid_nested_splits_for_three_or_more_scenes(n):
    windows = _collection(n)
    rotation = ss.cyclic_rotation(w.scene for w in windows)
    splits = nested_splits(windows, rotation)
    assert [s.test_scene for s in splits] == sorted(w.scene for w in windows)       # each scene held out once
    by_name = {w.scene: w for w in windows}
    for s in splits:
        assert s.validation_scene == rotation[s.test_scene]
        assert len(s.train_scenes) == n - 2 and list(s.train_scenes) == sorted(s.train_scenes)
        names = [s.test_scene, s.validation_scene, *s.train_scenes]
        assert sorted(names) == sorted(by_name)                                     # every scene, exactly once
        assert s.test is by_name[s.test_scene] and s.validation is by_name[s.validation_scene]
        assert all(w is by_name[name] for w, name in zip(s.train, s.train_scenes))  # whole objects, unchanged
        assert sum(len(w) for g in s.partitions().values() for w in g) == sum(len(w) for w in windows)
        sets = s.scene_sets()
        assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                    or sets["validation"] & sets["test"])


def test_split_preserves_scene_and_window_identity():
    windows = _collection(3)
    copies = [(w.scene, w.track_ids.copy(), w.frames.copy(), w.observed.copy(), w.target.copy()) for w in windows]
    s = nested_split(windows, "S1", ss.cyclic_rotation(["S1", "S2", "S3"]))
    assert (s.test_scene, s.validation_scene, s.train_scenes) == ("S1", "S2", ("S3",))
    for w, (scene, ids, frames, obs, tgt) in zip(windows, copies):
        assert w.scene == scene
        for a, b in ((w.track_ids, ids), (w.frames, frames), (w.observed, obs), (w.target, tgt)):
            np.testing.assert_array_equal(a, b)


def test_selection_is_deterministic_and_independent_of_input_order():
    windows = _collection(5)
    rotation = ss.cyclic_rotation(w.scene for w in windows)
    a = nested_splits(windows, rotation)
    b = nested_splits(windows, rotation)
    c = nested_splits(list(reversed(windows)), rotation)
    key = lambda splits: [(s.test_scene, s.validation_scene, s.train_scenes) for s in splits]
    assert key(a) == key(b) == key(c)
    assert all(x.test is y.test and all(p is q for p, q in zip(x.train, y.train)) for x, y in zip(a, c))


def test_validation_scene_comes_only_from_the_given_rotation():
    windows = _collection(4)
    names = [w.scene for w in windows]
    other = {"S1": "S4", "S2": "S1", "S3": "S1", "S4": "S2"}
    assert nested_split(windows, "S1", ss.cyclic_rotation(names)).validation_scene == "S2"
    assert nested_split(windows, "S1", other).validation_scene == "S4"
    assert nested_split(windows, "S3", other).train_scenes == ("S2", "S4")


def test_project_rotation_is_the_recorded_real_scene_mapping():
    """Decision 0001 C: two places later, cyclically, in the frozen scene order."""
    rotation = configured_rotation(CFG)
    assert rotation == {"ETH": "UNIV", "HOTEL": "ZARA1", "UNIV": "ZARA2", "ZARA1": "ETH", "ZARA2": "HOTEL"}
    order = list(STATIC["scenes"])
    assert rotation == {s: order[(i + 2) % len(order)] for i, s in enumerate(order)}
    assert sorted(rotation.values()) == sorted(order)                # each scene validates exactly one fold
    assert rotation["ZARA1"] != "ZARA2" and rotation["ZARA2"] != "ZARA1"
    assert check_rotation(rotation, order) == rotation


def test_project_rotation_is_never_accepted_for_synthetic_scenes():
    windows = _collection(5)
    with pytest.raises(SplitError, match="rotation must map exactly the scenes"):
        nested_splits(windows, configured_rotation(CFG))
    assert ss.cyclic_rotation(w.scene for w in windows) != configured_rotation(CFG)


def test_reading_the_rotation_still_raises_while_it_is_null(write_yaml, base_dict):
    base_dict["research_choices"]["inner_validation_rotation"]["value"] = None
    cfg = config.validate(base_path=write_yaml(base_dict), repo_root=config.REPO_ROOT).config
    with pytest.raises(UnresolvedChoiceError):
        configured_rotation(cfg)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_and_insufficient_collections_are_rejected(n):
    windows = _collection(n)
    assert MIN_SCENES == 3
    with pytest.raises(SplitError):
        nested_splits(windows, ss.cyclic_rotation(w.scene for w in windows))
    with pytest.raises(SplitError):
        scene_table(windows)


REAL_DATA = (ewap.DEFAULT_RAW_ROOT / ewap.DATASET_SUBDIR).is_dir() and ewap.DEFAULT_MANIFEST.is_file()


@pytest.mark.skipif(not REAL_DATA, reason="local EWAP data (Git-ignored) is not present")
def test_the_two_available_real_scenes_cannot_form_a_nested_split():
    """Only ETH and HOTEL are loadable in world metres; no scene is invented to make up the count."""
    from trackvision.predict.windowing import static_windows
    windows = [static_windows(ewap.load_scene(name, CFG), CFG) for name in ("ETH", "HOTEL")]
    with pytest.raises(SplitError, match="at least 3 scenes"):
        nested_splits(windows, {"ETH": "HOTEL", "HOTEL": "ETH"})


def test_duplicate_scene_names_are_rejected():
    scenes = ss.make_scenes(3) + [ss.make_scene("S2", 7)]
    with pytest.raises(SplitError, match="duplicate"):
        nested_splits(ss.windows_of(scenes, CFG), ss.cyclic_rotation(["S1", "S2", "S3"]))


@pytest.mark.parametrize("empty_role", ["test", "validation", "train"])
def test_empty_partitions_are_rejected(empty_role):
    rotation = {"S1": "S2", "S2": "S3", "S3": "S1"}
    empty = {"test": "S1", "validation": "S2", "train": "S3"}[empty_role]
    scenes = [ss.make_scene(name, i, lengths=(10,)) if name == empty else ss.make_scene(name, i)
              for i, name in enumerate(["S1", "S2", "S3"])]
    windows = ss.windows_of(scenes, CFG)
    assert len(next(w for w in windows if w.scene == empty)) == 0
    with pytest.raises(SplitError, match=f"the {empty_role} partition"):
        nested_split(windows, "S1", rotation)


def test_an_empty_training_scene_is_kept_not_discarded():
    scenes = ss.make_scenes(3) + [ss.make_scene("S4", 3, lengths=(10,))]
    windows = ss.windows_of(scenes, CFG)
    s = nested_split(windows, "S1", ss.cyclic_rotation(w.scene for w in windows))
    assert s.train_scenes == ("S3", "S4") and len(s.train[1]) == 0


@pytest.mark.parametrize("rotation", [
    {"S1": "S2", "S2": "S3"},                              # a scene is missing
    {"S1": "S2", "S2": "S3", "S3": "S1", "S9": "S1"},      # an unknown key
    {"S1": "S1", "S2": "S3", "S3": "S1"},                  # validates on its own test scene
    {"S1": "S9", "S2": "S3", "S3": "S1"},                  # an unknown validation scene
    [("S1", "S2"), ("S2", "S3"), ("S3", "S1")],            # not a mapping
])
def test_invalid_rotations_are_rejected(rotation):
    with pytest.raises(SplitError):
        nested_split(_collection(3), "S1", rotation)
    with pytest.raises(SplitError):
        check_rotation(rotation, ["S1", "S2", "S3"])


def test_unknown_test_scene_and_non_window_inputs_are_rejected():
    windows = _collection(3)
    rotation = ss.cyclic_rotation(["S1", "S2", "S3"])
    with pytest.raises(SplitError, match="test scene"):
        nested_split(windows, "S9", rotation)
    with pytest.raises(SplitError, match="expected Windows"):
        nested_split([*windows[:2], ss.make_scene("S3", 2)], "S1", rotation)


def test_mixed_protocols_are_rejected():
    scenes = ss.make_scenes(3)
    windows = ss.windows_of(scenes[:2], CFG)
    windows.append(make_windows("S3", scenes[2].tracks, ss.FRAME_STEP, STATIC["history_len"], 8, STATIC["window_stride"]))
    with pytest.raises(SplitError, match="different"):
        nested_splits(windows, ss.cyclic_rotation(["S1", "S2", "S3"]))


def test_nested_split_is_a_plain_record_of_whole_scenes():
    s = nested_split(_collection(3), "S2", ss.cyclic_rotation(["S1", "S2", "S3"]))
    assert isinstance(s, NestedSplit)
    assert s.partitions() == {"train": s.train, "validation": (s.validation,), "test": (s.test,)}
