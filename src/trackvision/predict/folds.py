"""Nested leave-one-scene-out splits (spec v4.15 section 5.4, 'Inner validation split').

Within each outer fold:

    OUTER   test        = the held-out scene            (never touched)
    INNER   validation  = one scene from the rest, given by a fixed rotation mapping
            training    = every remaining scene

Splits are made only at scene level: each scene's `Windows` object is placed whole in exactly one
partition and returned unchanged (the same object, so scene and window identity are preserved). No
window is split off, sampled or dropped, and a scene with no windows is kept, not discarded.

The rotation (test scene -> validation scene) is always passed in; there is no default rule. The
project's rotation is the research choice `inner_validation_rotation` (read with `configured_rotation`;
docs/decisions/0001-p2d-protocol-decisions.md, C). It maps the five real static scenes only, so
`check_rotation` rejects it for any other scene set; synthetic scenes need their own explicit mapping.
Seeds play no part in splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from trackvision.config import Config
from trackvision.predict.windowing import Windows

MIN_SCENES = 3              # one test, one validation and at least one training scene


class SplitError(ValueError):
    """Raised for scene collections or rotations that cannot form a valid nested split."""


@dataclass(frozen=True)
class NestedSplit:
    test_scene: str
    validation_scene: str
    train_scenes: tuple[str, ...]          # sorted
    test: Windows
    validation: Windows
    train: tuple[Windows, ...]             # in train_scenes order

    def partitions(self) -> dict[str, tuple[Windows, ...]]:
        return {"train": self.train, "validation": (self.validation,), "test": (self.test,)}

    def scene_sets(self) -> dict[str, frozenset[str]]:
        return {"train": frozenset(self.train_scenes), "validation": frozenset([self.validation_scene]),
                "test": frozenset([self.test_scene])}


def configured_rotation(cfg: Config) -> dict[str, str]:
    """The project's rotation; raises UnresolvedChoiceError while the research choice is null."""
    return dict(cfg.research_choice("inner_validation_rotation"))


def scene_table(scene_windows: Iterable[Windows]) -> dict[str, Windows]:
    """Scene name -> Windows, in input order, after checking names, count and protocol consistency."""
    items = list(scene_windows)
    if not items:
        raise SplitError("no scenes were given")
    for w in items:
        if not isinstance(w, Windows):
            raise SplitError(f"expected Windows, got {type(w).__name__}")
        if not isinstance(w.scene, str) or not w.scene:
            raise SplitError(f"every scene needs a non-empty name, got {w.scene!r}")
    names = [w.scene for w in items]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise SplitError(f"duplicate scene names {duplicates}")
    if len(items) < MIN_SCENES:
        raise SplitError(f"a nested split needs at least {MIN_SCENES} scenes (test, validation, training); "
                         f"got {len(items)}: {names}")
    protocols = {(w.history_len, w.horizon_len, w.stride) for w in items}
    if len(protocols) != 1:
        raise SplitError(f"scenes use different (history_len, horizon_len, stride): {sorted(protocols)}")
    return dict(zip(names, items))


def check_rotation(rotation: Mapping[str, str], scenes: Iterable[str]) -> dict[str, str]:
    """A rotation must map every scene to a different scene of the same collection."""
    scenes = list(scenes)
    if not isinstance(rotation, Mapping):
        raise SplitError(f"rotation must be a mapping, got {type(rotation).__name__}")
    if sorted(rotation) != sorted(scenes):
        raise SplitError(f"rotation must map exactly the scenes {sorted(scenes)}, got {sorted(rotation)}")
    for test_scene, val_scene in rotation.items():
        if val_scene not in scenes:
            raise SplitError(f"{test_scene} -> {val_scene!r} is not one of the scenes")
        if val_scene == test_scene:
            raise SplitError(f"{test_scene} cannot validate on its own held-out scene")
    return dict(rotation)


def nested_split(scene_windows: Iterable[Windows], test_scene: str, rotation: Mapping[str, str]) -> NestedSplit:
    """The nested split with `test_scene` held out and its validation scene taken from `rotation`."""
    table = scene_table(scene_windows)
    rot = check_rotation(rotation, table)
    if test_scene not in table:
        raise SplitError(f"test scene {test_scene!r} is not one of {list(table)}")
    val_scene = rot[test_scene]
    train_scenes = tuple(sorted(n for n in table if n not in (test_scene, val_scene)))
    for role, names in (("test", [test_scene]), ("validation", [val_scene]), ("train", train_scenes)):
        if sum(len(table[n]) for n in names) == 0:
            raise SplitError(f"the {role} partition {list(names)} has no windows")
    return NestedSplit(test_scene=test_scene, validation_scene=val_scene, train_scenes=train_scenes,
                       test=table[test_scene], validation=table[val_scene],
                       train=tuple(table[n] for n in train_scenes))


def nested_splits(scene_windows: Iterable[Windows], rotation: Mapping[str, str]) -> tuple[NestedSplit, ...]:
    """One nested split per scene, ordered by test-scene name."""
    table = scene_table(scene_windows)
    return tuple(nested_split(table.values(), name, rotation) for name in sorted(table))
