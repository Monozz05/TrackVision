"""Prediction windowing on tracks with known lengths and gaps (spec 5.3, 7.2)."""

from dataclasses import dataclass

import numpy as np
import pytest

from trackvision import config
from trackvision.predict.windowing import consecutive_runs, make_windows, static_windows

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, S = STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"]
L = H + F
STEP = 10


@dataclass(frozen=True)
class FakeTrack:
    ped_id: int
    frames: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class FakeScene:
    name: str
    tracks: tuple
    frame_step: int


def _track(ped_id, frames):
    frames = np.asarray(frames, dtype=np.int64)
    positions = np.column_stack([frames * 0.1, np.full(len(frames), float(ped_id))])
    return FakeTrack(ped_id, frames, positions)


def test_consecutive_runs_split_at_gaps():
    frames = np.array([0, 10, 20, 40, 50, 60, 70])
    assert consecutive_runs(frames, STEP) == [(0, 3), (3, 7)]
    assert consecutive_runs(np.array([], dtype=int), STEP) == []


def test_window_count_for_a_track_of_known_length():
    n = L + 5
    windows = make_windows("X", [_track(1, STEP * np.arange(n))], STEP, H, F, S)
    assert len(windows) == (n - L) // S + 1
    assert windows.observed.shape == (len(windows), H, 2)
    assert windows.target.shape == (len(windows), F, 2)


def test_short_track_gives_no_window():
    windows = make_windows("X", [_track(1, STEP * np.arange(L - 1))], STEP, H, F, S)
    assert len(windows) == 0
    assert windows.observed.shape == (0, H, 2)
    assert windows.target.shape == (0, F, 2)


def test_windows_never_span_a_gap():
    frames = np.concatenate([STEP * np.arange(L + 2), STEP * (np.arange(L) + L + 10)])
    windows = make_windows("X", [_track(1, frames)], STEP, H, F, S)
    assert len(windows) == 3 + 1
    assert np.all(np.diff(windows.frames, axis=1) == STEP)


def test_observed_and_target_are_consecutive_and_unchanged():
    track = _track(7, STEP * np.arange(L + 3))
    windows = make_windows("X", [track], STEP, H, F, S)
    for i in range(len(windows)):
        np.testing.assert_array_equal(np.concatenate([windows.observed[i], windows.target[i]]),
                                      track.positions[i * S:i * S + L])
        assert windows.frames[i, H - 1] < windows.frames[i, H]   # target strictly after history
    assert set(windows.track_ids.tolist()) == {7}


def test_stride_is_applied_within_each_run():
    windows = make_windows("X", [_track(1, STEP * np.arange(L + 6))], STEP, H, F, 3)
    np.testing.assert_array_equal(windows.frames[:, 0], STEP * np.array([0, 3, 6]))


@pytest.mark.parametrize("bad", [{"history_len": 0}, {"horizon_len": -1}, {"stride": 1.5}, {"frame_step": 0}])
def test_invalid_parameters_raise(bad):
    params = {"history_len": H, "horizon_len": F, "stride": S, "frame_step": STEP} | bad
    with pytest.raises(ValueError):
        make_windows("X", [], params["frame_step"], params["history_len"], params["horizon_len"], params["stride"])


def test_static_windows_use_the_frozen_protocol():
    scene = FakeScene("ETH", (_track(1, STEP * np.arange(L + 1)), _track(2, STEP * np.arange(L))), STEP)
    windows = static_windows(scene, CFG)
    assert (windows.history_len, windows.horizon_len, windows.stride) == (H, F, S)
    assert len(windows) == 2 + 1
    assert windows.scene == "ETH"
