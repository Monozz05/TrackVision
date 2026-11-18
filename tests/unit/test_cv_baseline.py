"""U2: constant-velocity prediction and ADE/FDE on synthetic ground truth.

Straight-line and turn cases use hand-built grid tracks with closed-form expected errors, so they do
not depend on what a random scene happens to contain. The U1 integration test checks only invariants
that hold for any scene. Every protocol value is read from configs/base.yaml.
"""

import math

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.synthetic_feasibility import (evaluate_cv_on_tracks, evaluate_ground_truth_cv,
                                                           protocol)
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
P = protocol(CFG)
H, F, STRIDE, DT = P["history_len"], P["horizon_len"], P["window_stride"], P["sampling_interval_s"]
MIN_OBS = P["cv_min_final_observations"]
FRAME_STEP = 10            # 0.4 s at 25 fps; any positive step works for hand-built tracks
L = H + F


def _expected_windows(n_grid: int) -> int:
    return 0 if n_grid < L else (n_grid - L) // STRIDE + 1


def _line(n, speed=1.3, heading=0.7, origin=(2.0, -1.0)):
    k = np.arange(n, dtype=float)[:, None]
    return np.asarray(origin) + k * speed * DT * np.array([math.cos(heading), math.sin(heading)])


# --- CV and metrics: analytic cases ------------------------------------------------------------

def test_straight_line_is_predicted_exactly():
    obs_future = _line(L)[None]
    pred = constant_velocity(obs_future[:, :H], F, MIN_OBS)
    assert pred.shape == (1, F, 2)
    ade, fde = ade_fde(displacement_errors(pred, obs_future[:, H:]))
    assert ade[0] <= 1e-12 and fde[0] <= 1e-12


def test_known_constant_offset_gives_unit_error():
    target = _line(L)[None, H:]
    perfect = target.copy()
    offset = target + np.array([0.0, 1.0])
    for pred, expected in ((perfect, 0.0), (offset, 1.0)):
        ade, fde = ade_fde(displacement_errors(pred, target))
        assert ade[0] == pytest.approx(expected, abs=1e-12)
        assert fde[0] == pytest.approx(expected, abs=1e-12)


def test_known_velocity_error_matches_closed_form():
    v, delta = 1.2, 0.5                         # observed at v; true future at v + delta, same direction
    k_obs = np.arange(H)[:, None]
    observed = (k_obs * v * DT * np.array([1.0, 0.0]))[None]
    last = observed[0, -1]
    k_fut = np.arange(1, F + 1)[:, None]
    target = (last + k_fut * (v + delta) * DT * np.array([1.0, 0.0]))[None]
    ade, fde = ade_fde(displacement_errors(constant_velocity(observed, F, MIN_OBS), target))
    assert ade[0] == pytest.approx(delta * DT * (F + 1) / 2, abs=1e-12)    # 1.3 m at F = 12
    assert fde[0] == pytest.approx(delta * DT * F, abs=1e-12)              # 2.4 m at F = 12


def test_cv_uses_only_the_last_two_observed_points():
    rng = np.random.default_rng(0)
    observed = rng.normal(size=(5, H, 2))
    base = constant_velocity(observed, F, MIN_OBS)
    earlier = observed.copy()
    earlier[:, :-2] += 100.0
    np.testing.assert_array_equal(constant_velocity(earlier, F, MIN_OBS), base)
    moved = observed.copy()
    moved[:, -1] += 1.0
    assert not np.allclose(constant_velocity(moved, F, MIN_OBS), base)


def test_naive_and_vectorized_metrics_agree():
    rng = np.random.default_rng(1)
    pred, target = rng.normal(size=(7, F, 2)), rng.normal(size=(7, F, 2))
    ade, fde = ade_fde(displacement_errors(pred, target))
    for n in range(7):
        errors = [math.hypot(*(pred[n, t] - target[n, t])) for t in range(F)]
        assert ade[n] == pytest.approx(sum(errors) / F, abs=1e-12)
        assert fde[n] == pytest.approx(errors[-1], abs=1e-12)


def test_horizon_length_is_not_hard_coded():
    observed = _line(H)[None]
    for horizon in (8, F):
        assert constant_velocity(observed, horizon, MIN_OBS).shape == (1, horizon, 2)


@pytest.mark.parametrize("bad", [
    lambda: constant_velocity(np.zeros((1, 1, 2)), F, MIN_OBS),            # history shorter than 2
    lambda: constant_velocity(np.zeros((1, H, 3)), F, MIN_OBS),
    lambda: constant_velocity(np.full((1, H, 2), np.nan), F, MIN_OBS),
    lambda: constant_velocity(np.zeros((1, H, 2)), 0, MIN_OBS),
    lambda: displacement_errors(np.zeros((1, F, 2)), np.zeros((1, F - 1, 2))),
    lambda: displacement_errors(np.zeros((1, F, 2)), np.full((1, F, 2), np.inf)),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bad()


# --- pipeline on hand-built tracks -------------------------------------------------------------

def test_straight_track_windows_all_have_zero_error():
    n = L + 6
    track = GridTrack(1, FRAME_STEP * np.arange(n), _line(n))
    result = evaluate_cv_on_tracks([track], FRAME_STEP, CFG)
    assert result.summary["n_windows"] == _expected_windows(n)
    assert np.max(result.ade) <= 1e-9 and np.max(result.fde) <= 1e-9


def _l_path(n: int, turn: int, speed: float) -> np.ndarray:
    """Along +x for steps 0..turn, then along +y; the corner is the grid sample at `turn`."""
    s = speed * DT
    k = np.arange(n)
    x = np.minimum(k, turn) * s
    y = np.maximum(k - turn, 0) * s
    return np.column_stack([x, y])


def _expected_turn_errors(last_observed_step: int, turn: int, speed: float) -> tuple[float, float]:
    """Closed form for CV on the L path.

    With j = turn - k_H: if 0 <= j < F, the last two observed points lie on the +x leg, CV continues
    along +x, and the truth leaves along +y after j steps, so DE_m = sqrt(2) * (m - j) * s for m > j.
    Hence ADE = sqrt(2) s (F - j)(F - j + 1) / (2F) and FDE = sqrt(2) s (F - j). Otherwise the last two
    observed points and every target lie on one leg (j >= F: all +x; j < 0: velocity already +y), so
    the error is zero.
    """
    s, j = speed * DT, turn - last_observed_step
    if 0 <= j < F:
        return math.sqrt(2) * s * (F - j) * (F - j + 1) / (2 * F), math.sqrt(2) * s * (F - j)
    return 0.0, 0.0


def test_turn_errors_match_closed_form_for_every_window():
    n, turn, speed = 2 * L + 5, L + 3, 1.4
    track = GridTrack(1, FRAME_STEP * np.arange(n), _l_path(n, turn, speed))
    result = evaluate_cv_on_tracks([track], FRAME_STEP, CFG)
    assert result.summary["n_windows"] == _expected_windows(n)
    seen_turn = seen_zero = 0
    for i in range(len(result.windows)):
        last_observed_step = result.windows.frames[i, H - 1] // FRAME_STEP
        exp_ade, exp_fde = _expected_turn_errors(int(last_observed_step), turn, speed)
        assert result.ade[i] == pytest.approx(exp_ade, abs=1e-9)
        assert result.fde[i] == pytest.approx(exp_fde, abs=1e-9)
        seen_turn += exp_ade > 0
        seen_zero += exp_ade == 0
    assert seen_turn == F and seen_zero > 0     # every turn offset j = 0..F-1 occurs exactly once


def test_predictions_come_only_from_observed_histories():
    n = L + 4
    track = GridTrack(1, FRAME_STEP * np.arange(n), _l_path(n, L, 1.1))
    result = evaluate_cv_on_tracks([track], FRAME_STEP, CFG)
    np.testing.assert_array_equal(result.predicted, constant_velocity(result.windows.observed, F, MIN_OBS))


def test_no_windows_gives_undefined_summaries():
    short = GridTrack(1, FRAME_STEP * np.arange(L - 1), _line(L - 1))
    summary = evaluate_cv_on_tracks([short], FRAME_STEP, CFG).summary
    assert summary["n_windows"] == 0
    for key in ("coverage", "mean_ade_m", "mean_fde_m", "per_track_mean_ade_m", "mean_error_by_step_m"):
        assert summary[key] is None


# --- integration with a U1 scene (invariants only) ---------------------------------------------

def _u1_scene(seed=7):
    return generate(SceneConfig(seed=seed, n_frames=1500), DT)


def test_u1_scene_windows_respect_identity_grid_and_ground_truth():
    scene = _u1_scene()
    result = evaluate_ground_truth_cv(scene, CFG)
    tracks = ground_truth_grid_tracks(scene)
    assert result.summary["n_windows"] > 0, "this fixed seed should yield windows; zero would make the test vacuous"

    for track in tracks:                          # grid anchored at frame 0, consecutive steps
        assert np.all(track.frames % scene.frames_per_step == 0)
        assert np.all(np.diff(track.frames) == scene.frames_per_step)
        assert np.sum(result.windows.track_ids == track.ped_id) == _expected_windows(len(track.frames))

    truth = {(int(p), int(f)): pos for p, f, pos in zip(scene.ped_id, scene.frame, scene.foot_world)}
    w = result.windows
    for i in range(len(w)):
        assert np.all(np.diff(w.frames[i]) == scene.frames_per_step)
        pid = int(w.track_ids[i])
        expected = np.array([truth[(pid, int(f))] for f in w.frames[i]])
        np.testing.assert_array_equal(np.concatenate([w.observed[i], w.target[i]]), expected)


def test_u1_scene_summary_is_deterministic():
    a = evaluate_ground_truth_cv(_u1_scene(seed=7), CFG)
    b = evaluate_ground_truth_cv(_u1_scene(seed=7), CFG)
    assert a.summary == b.summary
    np.testing.assert_array_equal(a.ade, b.ade)
    assert "not independent samples" in a.summary["note"]


def test_scene_with_a_different_interval_is_rejected():
    scene = generate(SceneConfig(seed=7, n_frames=200), DT / 2)
    with pytest.raises(ValueError, match="configuration requires"):
        evaluate_ground_truth_cv(scene, CFG)
