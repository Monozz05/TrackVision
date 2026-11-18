"""The six data-leakage checks (spec v4.15 section 11.7). They run before every experiment; any failure
marks the experiment invalid and blocks it (`require_no_leakage`).

Each check re-derives what it asserts from the split's windows and, where needed, the source scenes; it
does not trust the split builder. Each returns a CheckResult whose `problems` name every violation found.
Missing or empty inputs are reported as problems, never as a pass. The one declared pass without data is
normalization scope when normalization is explicitly declared unused (NORMALIZATION_NOT_USED): the
specification uses normalization only "if used at all".

  fold_disjointness     train, validation and test scene sets are non-empty and pairwise disjoint, and no
                        trajectory (scene, track id) occurs in two partitions
  scene_disjointness    every window sits in the partition that declares its scene, each scene's windows
                        are in one partition only, no window occurs twice, no partition is empty
  target_purity         predictor inputs are identical when every target is replaced, and building the
                        inputs leaves the target and observed arrays unchanged (hash before and after)
  temporal_ordering     window timestamps strictly increase, so every observed sample precedes every
                        target sample of its window
  window_boundary       every window is consecutive samples of one track of its own scene, one frame step
                        apart, with positions equal to that track's
  normalization_scope   statistics used for every partition equal those fitted on the training windows
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np

from trackvision.predict.folds import NestedSplit
from trackvision.predict.windowing import Windows

CHECKS = ("fold_disjointness", "scene_disjointness", "target_purity", "temporal_ordering",
          "window_boundary", "normalization_scope")
PARTITIONS = ("train", "validation", "test")
NORMALIZATION_NOT_USED = "not_used"


class LeakageError(RuntimeError):
    """Raised by `require_no_leakage` when any check fails."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    problems: tuple[str, ...]
    note: str | None = None


def _result(name: str, problems: list[str], note: str | None = None) -> CheckResult:
    return CheckResult(name, not problems, tuple(problems), note)


def array_hash(arr) -> str:
    """SHA-256 of an array's shape, dtype and bytes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256(f"{a.shape}|{a.dtype.str}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _windows_by_partition(split: NestedSplit) -> dict[str, tuple[Windows, ...]]:
    return {"train": tuple(split.train), "validation": (split.validation,), "test": (split.test,)}


def _all_windows(split: NestedSplit):
    for part, group in _windows_by_partition(split).items():
        for w in group:
            yield part, w


def _empty_partitions(split: NestedSplit) -> list[str]:
    """A partition with no windows cannot be checked; a single empty scene inside a non-empty partition
    has nothing to leak and is skipped."""
    return [f"{part} has no windows to test" for part, group in _windows_by_partition(split).items()
            if sum(len(w) for w in group) == 0]


# --- 1. fold disjointness ----------------------------------------------------------------------

def fold_disjointness(split: NestedSplit) -> CheckResult:
    problems = []
    sets = {"train": set(split.train_scenes), "validation": {split.validation_scene}, "test": {split.test_scene}}
    if len(split.train_scenes) != len(sets["train"]):
        problems.append(f"training scenes repeat: {list(split.train_scenes)}")
    for part, names in sets.items():
        if not names or any(not n for n in names):
            problems.append(f"{part} declares no scene")
    for i, a in enumerate(PARTITIONS):
        for b in PARTITIONS[i + 1:]:
            shared = sets[a] & sets[b]
            if shared:
                problems.append(f"scenes {sorted(shared)} are in both {a} and {b}")
    owner: dict[tuple[str, int], str] = {}
    for part, w in _all_windows(split):
        for tid in np.unique(np.asarray(w.track_ids)):
            key = (w.scene, int(tid))
            if key in owner and owner[key] != part:
                problems.append(f"trajectory {key} occurs in both {owner[key]} and {part}")
            owner.setdefault(key, part)
    return _result("fold_disjointness", problems)


# --- 2. scene disjointness ---------------------------------------------------------------------

def scene_disjointness(split: NestedSplit) -> CheckResult:
    problems = _empty_partitions(split)
    declared = {"train": set(split.train_scenes), "validation": {split.validation_scene}, "test": {split.test_scene}}
    scene_part: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    for part, group in _windows_by_partition(split).items():
        for w in group:
            if w.scene not in declared[part]:
                problems.append(f"windows of scene {w.scene!r} are in {part}, which declares {sorted(declared[part])}")
            if w.scene in scene_part and scene_part[w.scene] != part:
                problems.append(f"windows of scene {w.scene!r} are in both {scene_part[w.scene]} and {part}")
            scene_part.setdefault(w.scene, part)
            for tid, frames in zip(np.asarray(w.track_ids), np.asarray(w.frames)):
                key = (w.scene, int(tid), tuple(int(f) for f in frames))
                if key in seen:
                    problems.append(f"window {key[:2]} at frames {key[2][0]}-{key[2][-1]} occurs in {seen[key]} "
                                    f"and again in {part}")
                seen.setdefault(key, part)
    if split.test_scene in {w.scene for w in split.train}:
        problems.append(f"test scene {split.test_scene!r} is present in the training set")
    return _result("scene_disjointness", problems)


# --- 3. target purity --------------------------------------------------------------------------

def _replaced_targets(w: Windows) -> list[Windows]:
    """Two copies of `w` whose targets differ from the original (and each other) at every element."""
    ramp = np.arange(w.target.size, dtype=np.float64).reshape(w.target.shape)
    return [dataclasses.replace(w, target=w.target + 1.0e3 + ramp),
            dataclasses.replace(w, target=-7.0 * w.target - 3.0e3 - ramp)]


def target_purity(split: NestedSplit, make_inputs: Callable[[Windows], np.ndarray]) -> CheckResult:
    """`make_inputs` is everything that turns one partition's windows into predictor inputs, including any
    corruption or preprocessing."""
    if not callable(make_inputs):
        return _result("target_purity", ["no input builder was given"])
    problems = _empty_partitions(split)
    for part, w in _all_windows(split):
        if len(w) == 0:
            continue
        before = (array_hash(w.target), array_hash(w.observed))
        inputs = np.asarray(make_inputs(w))
        if (array_hash(w.target), array_hash(w.observed)) != before:
            problems.append(f"building inputs changed the target or observed arrays of {part} scene {w.scene!r}")
        for replaced in _replaced_targets(w):
            other = np.asarray(make_inputs(replaced))
            if other.shape != inputs.shape or not np.array_equal(other, inputs, equal_nan=True):
                problems.append(f"predictor inputs of {part} scene {w.scene!r} change when the targets change")
                break
    return _result("target_purity", problems)


# --- 4. temporal ordering ----------------------------------------------------------------------

def temporal_ordering(split: NestedSplit) -> CheckResult:
    problems = _empty_partitions(split)
    for part, w in _all_windows(split):
        frames = np.asarray(w.frames)
        length = w.history_len + w.horizon_len
        if frames.ndim != 2 or frames.shape[1] != length or len(frames) != len(w):
            problems.append(f"{part} scene {w.scene!r}: frames have shape {frames.shape}, expected ({len(w)}, {length})")
            continue
        bad = np.flatnonzero(np.any(np.diff(frames, axis=1) <= 0, axis=1))
        for i in bad:
            obs, fut = frames[i, :w.history_len], frames[i, w.history_len:]
            kind = "a target at or before an observation" if fut.min() <= obs.max() else "non-increasing timestamps"
            problems.append(f"{part} scene {w.scene!r} window {int(i)} (track {int(w.track_ids[i])}) has {kind}")
    return _result("temporal_ordering", problems)


# --- 5. window boundary ------------------------------------------------------------------------

def window_boundary(split: NestedSplit, scenes: Iterable) -> CheckResult:
    """`scenes` are the source scenes the windows were cut from, each with `name`, `tracks` (ped_id,
    frames, positions) and `frame_step`."""
    problems = _empty_partitions(split)
    source = {s.name: s for s in scenes}
    for part, w in _all_windows(split):
        scene = source.get(w.scene)
        if scene is None:
            problems.append(f"{part} scene {w.scene!r} has no source scene to check against")
            continue
        if len(w) == 0:
            continue
        tracks = {int(t.ped_id): t for t in scene.tracks}
        index = {pid: {int(f): i for i, f in enumerate(t.frames)} for pid, t in tracks.items()}
        full = np.concatenate([w.observed, w.target], axis=1)
        for i, (tid, frames) in enumerate(zip(np.asarray(w.track_ids), np.asarray(w.frames))):
            where = f"{part} scene {w.scene!r} window {i} (track {int(tid)})"
            track = tracks.get(int(tid))
            if track is None:
                problems.append(f"{where}: no such track in the scene")
                continue
            rows = [index[int(tid)].get(int(f)) for f in frames]
            if any(r is None for r in rows):
                problems.append(f"{where}: frames outside the track")
                continue
            rows = np.asarray(rows)
            if np.any(np.diff(rows) != 1) or np.any(np.diff(frames) != scene.frame_step):
                problems.append(f"{where}: not consecutive samples one frame step apart (crosses a gap or a boundary)")
                continue
            if not np.array_equal(np.asarray(track.positions)[rows], full[i]):
                problems.append(f"{where}: positions differ from the source track (spliced from elsewhere)")
    return _result("window_boundary", problems)


# --- 6. normalization scope --------------------------------------------------------------------

def _same_stats(a: Mapping, b: Mapping) -> bool:
    return (isinstance(a, Mapping) and isinstance(b, Mapping) and sorted(a) == sorted(b)
            and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a))


def normalization_scope(split: NestedSplit, fit: Callable[[tuple[Windows, ...]], Mapping] | None,
                        stats_used) -> CheckResult:
    """`stats_used` maps each partition to the statistics applied to it, or is NORMALIZATION_NOT_USED.
    `fit` maps a tuple of Windows to a mapping of named arrays."""
    if isinstance(stats_used, str) and stats_used == NORMALIZATION_NOT_USED:
        return _result("normalization_scope", [], note="normalization declared not used")
    if not isinstance(stats_used, Mapping):
        return _result("normalization_scope", ["normalization use is not declared (statistics or NOT_USED)"])
    if not callable(fit):
        return _result("normalization_scope", ["statistics are given but no fitting function to check them"])
    problems = [f"no statistics recorded for {p}" for p in PARTITIONS if p not in stats_used]
    problems += [f"unknown partition {p!r} in the statistics" for p in stats_used if p not in PARTITIONS]
    if sum(len(w) for w in split.train) == 0:
        problems.append("there are no training windows to fit statistics on")
        return _result("normalization_scope", problems)
    expected = fit(tuple(split.train))
    others = {"validation": fit((split.validation,)), "test": fit((split.test,)),
              "train + validation + test": fit((*split.train, split.validation, split.test))}
    for part in PARTITIONS:
        if part not in stats_used or _same_stats(stats_used[part], expected):
            continue
        source = next((name for name, s in others.items() if _same_stats(stats_used[part], s)), None)
        hint = f" (they match statistics fitted on {source})" if source else ""
        problems.append(f"statistics applied to {part} were not fitted on the training windows only{hint}")
    return _result("normalization_scope", problems)


# --- all six -----------------------------------------------------------------------------------

def run_leakage_checks(split: NestedSplit, scenes: Iterable, make_inputs, fit, stats_used) -> tuple[CheckResult, ...]:
    """All six checks, in section 11.7 order."""
    return (fold_disjointness(split), scene_disjointness(split), target_purity(split, make_inputs),
            temporal_ordering(split), window_boundary(split, scenes), normalization_scope(split, fit, stats_used))


def require_no_leakage(results: Iterable[CheckResult]) -> None:
    """Raise LeakageError unless all six checks ran and passed."""
    results = tuple(results)
    names = [r.name for r in results]
    if sorted(names) != sorted(CHECKS):
        raise LeakageError(f"expected the checks {list(CHECKS)}, got {names}")
    failed = [f"{r.name}: {p}" for r in results if not r.passed for p in r.problems]
    if failed:
        raise LeakageError("data leakage checks failed:\n" + "\n".join(failed))
