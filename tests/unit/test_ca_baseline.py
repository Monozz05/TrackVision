"""P2-a: constant-acceleration (CA) prediction (spec v4.15 section 5.4).

Analytic cases use hand-built trajectories with closed-form expected predictions and errors. CA as
specified is exact only for constant velocity: its velocity term is a backward difference, so on a true
quadratic it carries an exact, linear-in-k bias (tested in closed form). The U1 integration test checks
only properties that hold for any scene: windows straight from p_{H-2} onward are predicted exactly. The ETH/HOTEL test reads the local, Git-ignored EWAP data and
skips when it is absent; it checks finiteness, shapes and agreement with an independent loop, not
accuracy. Every protocol value is read from configs/base.yaml.
"""

import math

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.io import ewap
from trackvision.metrics.displacement import ade_fde, displacement_errors, rmse
from trackvision.predict.baselines import constant_acceleration, constant_velocity
from trackvision.predict.windowing import make_windows, static_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
P = protocol(CFG)
H, F, STRIDE, DT = P["history_len"], P["horizon_len"], P["window_stride"], P["sampling_interval_s"]
CV_MIN = P["cv_min_final_observations"]
CA_MIN = CFG.frozen["predictors"]["ca_min_final_observations"]
FRAME_STEP = 10            # 0.4 s at 25 fps; any positive step works for hand-built tracks
L = H + F
HORIZONS = (8, F)          # the moving fallback and the configured static horizon


def _quadratic(n, origin=(1.5, -2.0), velocity=(1.1, 0.4), accel=(0.3, -0.25)):
    """p_k = p_0 + v k dt + 0.5 a (k dt)^2, k = 0 ... n-1."""
    t = (np.arange(n, dtype=float) * DT)[:, None]
    return np.asarray(origin) + t * np.asarray(velocity) + 0.5 * t * t * np.asarray(accel)


def _naive_ca(observed, horizon):
    """Per-window, per-step loop over the step form p_H + k d1 + 0.5 k^2 d2."""
    out = np.empty((len(observed), horizon, 2))
    for n, obs in enumerate(observed):
        p2, p1, p0 = obs[-3], obs[-2], obs[-1]
        for k in range(1, horizon + 1):
            for c in range(2):
                d1 = p0[c] - p1[c]
                d2 = p0[c] - 2.0 * p1[c] + p2[c]
                out[n, k - 1, c] = p0[c] + k * d1 + 0.5 * k * k * d2
    return out


# --- analytic cases ----------------------------------------------------------------------------

def test_configured_minimum_is_three():
    assert CA_MIN == 3


@pytest.mark.parametrize("horizon", HORIZONS)
def test_quadratic_trajectory_has_the_exact_backward_difference_bias(horizon):
    """The section 5.4 velocity is the backward difference d1 = p_H - p_{H-1}, which is the velocity at
    t_H - dt/2, not at t_H. On p_n = c + b n + q n^2 with q = 0.5 a dt^2: d1 = b + q (2H - 1) and d2 = 2q, so

        p_hat_{H+k} - p_{H+k} = -k q        exactly, for every k,

    giving ADE = |q| (F + 1) / 2 and FDE = |q| F. CA as specified reproduces only constant-velocity motion
    exactly (test_ca_equals_cv_for_constant_velocity); a quadratic is not reproduced exactly."""
    accel = np.array([0.3, -0.25])
    path = _quadratic(H + horizon, accel=accel)[None]
    pred = constant_acceleration(path[:, :H], horizon, CA_MIN)
    assert pred.shape == (1, horizon, 2)
    q = 0.5 * accel * DT ** 2
    k = np.arange(1, horizon + 1, dtype=float)[:, None]
    np.testing.assert_allclose(pred[0] - path[0, H:], -k * q, rtol=0, atol=1e-12)
    ade, fde = ade_fde(displacement_errors(pred, path[:, H:]))
    assert ade[0] == pytest.approx(np.linalg.norm(q) * (horizon + 1) / 2, abs=1e-12)
    assert fde[0] == pytest.approx(np.linalg.norm(q) * horizon, abs=1e-12)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_ca_equals_cv_for_constant_velocity(horizon):
    rng = np.random.default_rng(3)
    origins, velocities = rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
    k = np.arange(H, dtype=float)[None, :, None]
    observed = origins[:, None, :] + k * DT * velocities[:, None, :]
    np.testing.assert_allclose(constant_acceleration(observed, horizon, CA_MIN),
                               constant_velocity(observed, horizon, CV_MIN), rtol=0, atol=1e-12)


@pytest.mark.parametrize("dt", [DT, 0.1, 1.0])
def test_delta_time_and_step_forms_agree(dt):
    """The section 5.4 form with v_H and a_H in physical units equals the step form for any dt."""
    rng = np.random.default_rng(4)
    observed = rng.normal(size=(9, H, 2))
    v = (observed[:, -1] - observed[:, -2]) / dt
    a = (observed[:, -1] - 2.0 * observed[:, -2] + observed[:, -3]) / dt ** 2
    t = (np.arange(1, F + 1) * dt)[None, :, None]
    physical = observed[:, -1][:, None] + t * v[:, None] + 0.5 * t * t * a[:, None]
    np.testing.assert_allclose(constant_acceleration(observed, F, CA_MIN), physical, rtol=0, atol=1e-12)


@pytest.mark.parametrize("horizon", HORIZONS)
def test_known_acceleration_error_matches_closed_form(horizon):
    """History at constant velocity, future accelerating at a along the heading.

    CA sees d2 = 0 and continues the line, so DE_k = 0.5 a dt^2 k^2:
      ADE  = 0.5 a dt^2 (F + 1)(2F + 1) / 6
      FDE  = 0.5 a dt^2 F^2
      RMSE = 0.5 a dt^2 sqrt((F + 1)(2F + 1)(3F^2 + 3F - 1) / 30)
    At F = 12, dt = 0.4, a = 0.25: ADE = 1.08333 m, FDE = 2.88 m.
    """
    v, a = 1.2, 0.25
    heading = np.array([math.cos(0.3), math.sin(0.3)])
    observed = (np.arange(H)[:, None] * v * DT * heading)[None]
    t = (np.arange(1, horizon + 1) * DT)[:, None]
    target = (observed[0, -1] + (v * t + 0.5 * a * t * t) * heading)[None]
    per_step = displacement_errors(constant_acceleration(observed, horizon, CA_MIN), target)
    ade, fde = ade_fde(per_step)
    c, f = 0.5 * a * DT ** 2, horizon
    np.testing.assert_allclose(per_step[0], c * np.arange(1, f + 1) ** 2, rtol=0, atol=1e-12)
    assert ade[0] == pytest.approx(c * (f + 1) * (2 * f + 1) / 6, abs=1e-12)
    assert fde[0] == pytest.approx(c * f * f, abs=1e-12)
    assert rmse(per_step)[0] == pytest.approx(c * math.sqrt((f + 1) * (2 * f + 1) * (3 * f * f + 3 * f - 1) / 30),
                                              abs=1e-12)
    if horizon == 12:
        assert ade[0] == pytest.approx(1.0833333333333333, abs=1e-12) and fde[0] == pytest.approx(2.88, abs=1e-12)


def test_ca_uses_only_the_last_three_observed_points():
    rng = np.random.default_rng(5)
    observed = rng.normal(size=(5, H, 2))
    base = constant_acceleration(observed, F, CA_MIN)
    earlier = observed.copy()
    earlier[:, :-3] += 100.0
    np.testing.assert_array_equal(constant_acceleration(earlier, F, CA_MIN), base)
    third = observed.copy()
    third[:, -3] += 1.0                                  # p_{H-2} is read
    assert not np.allclose(constant_acceleration(third, F, CA_MIN), base)


def test_target_values_do_not_affect_predictions():
    """Two tracks share the history and differ only in the future: identical predictions, different targets."""
    path = _quadratic(L)
    other = path.copy()
    other[H:] += np.array([3.0, -2.0])
    tracks = [GridTrack(1, FRAME_STEP * np.arange(L), path), GridTrack(2, FRAME_STEP * np.arange(L), other)]
    w = make_windows("synthetic", tracks, FRAME_STEP, H, F, STRIDE)
    assert len(w) == 2 and not np.allclose(w.target[0], w.target[1])
    np.testing.assert_array_equal(w.observed[0], w.observed[1])
    pred = constant_acceleration(w.observed, F, CA_MIN)
    np.testing.assert_array_equal(pred[0], pred[1])


def test_naive_and_vectorized_ca_agree():
    rng = np.random.default_rng(6)
    observed = rng.normal(scale=5.0, size=(11, H, 2))
    for horizon in HORIZONS:
        np.testing.assert_allclose(constant_acceleration(observed, horizon, CA_MIN), _naive_ca(observed, horizon),
                                   rtol=0, atol=1e-12)


def test_horizon_length_is_not_hard_coded():
    observed = _quadratic(H)[None]
    for horizon in (1, *HORIZONS, 20):
        assert constant_acceleration(observed, horizon, CA_MIN).shape == (1, horizon, 2)


def test_minimum_history_of_three_is_accepted():
    observed = _quadratic(CA_MIN)[None]
    np.testing.assert_allclose(constant_acceleration(observed, F, CA_MIN), _naive_ca(observed, F), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bad", [
    lambda: constant_acceleration(np.zeros((1, 2, 2)), F, CA_MIN),             # history shorter than 3
    lambda: constant_acceleration(np.zeros((1, 4, 2)), F, 5),                  # shorter than the given minimum
    lambda: constant_acceleration(np.zeros((1, H, 2)), F, 2),                  # CA cannot run on two points
    lambda: constant_acceleration(np.zeros((1, H, 2)), F, 3.0),
    lambda: constant_acceleration(np.zeros((1, H, 3)), F, CA_MIN),
    lambda: constant_acceleration(np.zeros((H, 2)), F, CA_MIN),
    lambda: constant_acceleration(np.full((1, H, 2), np.nan), F, CA_MIN),
    lambda: constant_acceleration(np.full((1, H, 2), np.inf), F, CA_MIN),
    lambda: constant_acceleration(np.zeros((1, H, 2)), 0, CA_MIN),
    lambda: constant_acceleration(np.zeros((1, H, 2)), 12.0, CA_MIN),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bad()


# --- synthetic scene integration ---------------------------------------------------------------

def test_hand_built_straight_track_has_zero_error():
    n = L + 6
    k = np.arange(n, dtype=float)[:, None]
    track = GridTrack(1, FRAME_STEP * np.arange(n), np.array([2.0, -1.0]) + k * 1.3 * DT * np.array([0.6, 0.8]))
    w = make_windows("synthetic", [track], FRAME_STEP, H, F, STRIDE)
    assert len(w) == n - L + 1
    ade, fde = ade_fde(displacement_errors(constant_acceleration(w.observed, F, CA_MIN), w.target))
    assert np.max(ade) <= 1e-9 and np.max(fde) <= 1e-9


def test_u1_straight_segments_are_predicted_exactly():
    """U1 walkers move at constant speed on straight segments. CA reads only p_{H-2} onward, so a window
    whose points from p_{H-2} to the end of the horizon have zero second differences lies on one segment
    and must be predicted exactly, whatever the earlier history does; a window that is straight at
    p_{H-2}..p_H but turns in the future must have non-zero error. (This scene's segments are shorter than
    a full 20-point window, so the condition starts at p_{H-2}, not p_1.)"""
    scene = generate(SceneConfig(seed=7, n_frames=1500), DT)
    w = make_windows("u1", ground_truth_grid_tracks(scene), scene.frames_per_step, H, F, STRIDE)
    ade, _ = ade_fde(displacement_errors(constant_acceleration(w.observed, F, CA_MIN), w.target))
    full = np.concatenate([w.observed, w.target], axis=1)
    second = np.abs(np.diff(full, n=2, axis=1)).max(axis=2)[:, H - 3:]    # centred on p_{H-1} ... p_{H+F-1}
    straight = second.max(axis=1) <= 1e-9
    turns_later = (second[:, 0] <= 1e-9) & (second.max(axis=1) > 1e-6)
    assert straight.sum() > 0 and turns_later.sum() > 0, "this fixed seed should give both kinds of window"
    assert np.max(ade[straight]) <= 1e-9
    assert np.min(ade[turns_later]) > 0


# --- ETH / HOTEL integration (local data only) -------------------------------------------------

REAL_DATA = (ewap.DEFAULT_RAW_ROOT / ewap.DATASET_SUBDIR).is_dir() and ewap.DEFAULT_MANIFEST.is_file()


@pytest.mark.skipif(not REAL_DATA, reason="local EWAP data (Git-ignored) is not present")
@pytest.mark.parametrize("name", ["ETH", "HOTEL"])
def test_real_scene_ca_is_finite_and_matches_the_naive_loop(name):
    scene = ewap.load_scene(name, CFG)
    assert scene.provenance["manifest_verified"]
    w = static_windows(scene, CFG)
    assert len(w) > 0
    pred = constant_acceleration(w.observed, F, CA_MIN)
    assert pred.shape == (len(w), F, 2) and np.all(np.isfinite(pred))
    np.testing.assert_allclose(pred, _naive_ca(w.observed, F), rtol=0, atol=1e-9)
    per_step = displacement_errors(pred, w.target)
    ade, fde = ade_fde(per_step)
    err = rmse(per_step)
    assert np.all(np.isfinite(ade)) and np.all(np.isfinite(fde)) and np.all(np.isfinite(err))
    assert np.all(err >= ade - 1e-12)
