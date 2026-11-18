"""P2-b: frame-to-frame displacement encoding and reconstruction (spec v4.15 section 5.4).

Hand-calculated cases use small integers or dyadic fractions, so encoding and reconstruction are exact in
float64 and are compared with assert_array_equal. Random cases use a 1e-12 tolerance. History and horizon
lengths are read from configs/base.yaml: the static primary protocol, the P-15 history lengths and the
moving-arm fallback horizon.
"""

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.predict.displacements import future_displacements, observed_displacements, reconstruct
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks

CFG = config.validate().config
P = protocol(CFG)
H, F, STRIDE, DT = P["history_len"], P["horizon_len"], P["window_stride"], P["sampling_interval_s"]
CV_MIN = P["cv_min_final_observations"]
HISTORY_LENS = tuple(CFG.frozen["analysis"]["history_ablation"]["history_lens"])
HORIZON_LENS = (CFG.frozen["protocol"]["moving"]["fallback"]["horizon_len"], F)


def _windows(n, history, horizon, seed=0, scale=10.0):
    rng = np.random.default_rng(seed)
    path = np.cumsum(rng.normal(scale=scale / 10, size=(n, history + horizon, 2)), axis=1) + rng.normal(scale=scale, size=(n, 1, 2))
    return path[:, :history], path[:, history:]


def _naive_observed(obs):
    out = np.empty((len(obs), obs.shape[1] - 1, 2))
    for n in range(len(obs)):
        for k in range(obs.shape[1] - 1):
            for c in range(2):
                out[n, k, c] = obs[n, k + 1, c] - obs[n, k, c]
    return out


def _naive_future(obs, fut):
    out = np.empty(fut.shape)
    for n in range(len(fut)):
        prev = obs[n, -1]
        for j in range(fut.shape[1]):
            for c in range(2):
                out[n, j, c] = fut[n, j, c] - prev[c]
            prev = fut[n, j]
    return out


def _naive_reconstruct(last, steps):
    out = np.empty(steps.shape)
    for n in range(len(steps)):
        pos = [last[n, 0], last[n, 1]]
        for k in range(steps.shape[1]):
            pos = [pos[0] + steps[n, k, 0], pos[1] + steps[n, k, 1]]
            out[n, k] = pos
    return out


# --- hand-calculated cases ---------------------------------------------------------------------

def test_observed_displacements_by_hand():
    observed = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [4.0, 6.0]]])
    np.testing.assert_array_equal(observed_displacements(observed), [[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]])


def test_first_future_displacement_bridges_the_observation_boundary():
    observed = np.array([[[0.0, 0.0], [1.0, 0.0], [4.0, 6.0]]])
    future = np.array([[[5.0, 6.0], [5.0, 8.0], [2.0, 4.0]]])
    target = future_displacements(observed, future)
    np.testing.assert_array_equal(target, [[[1.0, 0.0], [0.0, 2.0], [-3.0, -4.0]]])
    np.testing.assert_array_equal(target[:, 0], future[:, 0] - observed[:, -1])


def test_targets_are_not_offsets_from_the_last_observation():
    """The excluded formulation p_{H+k} - p_H agrees only at k = 1."""
    observed, future = _windows(4, H, F, seed=1)
    target = future_displacements(observed, future)
    offsets = future - observed[:, -1:, :]
    np.testing.assert_array_equal(target[:, 0], offsets[:, 0])
    assert np.all(np.abs(target[:, 1:] - offsets[:, 1:]).max(axis=2) > 1e-6)


def test_reconstruction_of_known_trajectories_is_exact():
    last = np.array([[2.0, 1.0], [-3.0, 0.5]])
    steps = np.zeros((2, F, 2))
    steps[0] = [0.5, -0.25]                                    # straight, constant step
    steps[1, : F // 2] = [0.25, 0.0]                           # an L-shaped turn half-way
    steps[1, F // 2:] = [0.0, 0.75]
    k = np.arange(1, F + 1)[:, None]
    expected0 = last[0] + k * np.array([0.5, -0.25])
    m = np.minimum(k, F // 2)
    expected1 = last[1] + m * np.array([0.25, 0.0]) + (k - m) * np.array([0.0, 0.75])
    got = reconstruct(last, steps)
    np.testing.assert_array_equal(got[0], expected0)
    np.testing.assert_array_equal(got[1], expected1)


# --- round trip, batches and protocol lengths --------------------------------------------------

def test_encode_then_reconstruct_round_trip():
    observed, future = _windows(25, H, F, seed=2)
    np.testing.assert_allclose(reconstruct(observed[:, -1], future_displacements(observed, future)), future,
                               rtol=0, atol=1e-12)
    integer_obs = np.arange(H * 2, dtype=float).reshape(1, H, 2) ** 2
    integer_fut = -np.arange(F * 2, dtype=float).reshape(1, F, 2) * 3
    np.testing.assert_array_equal(reconstruct(integer_obs[:, -1], future_displacements(integer_obs, integer_fut)),
                                  integer_fut)
    # observed history: p_1 plus the cumulative encoder input gives back p_2 ... p_H
    np.testing.assert_allclose(reconstruct(observed[:, 0], observed_displacements(observed)), observed[:, 1:],
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("history", HISTORY_LENS)
@pytest.mark.parametrize("horizon", HORIZON_LENS)
def test_shapes_follow_the_configured_lengths(history, horizon):
    observed, future = _windows(7, history, horizon, seed=history * 100 + horizon)
    enc = observed_displacements(observed)
    target = future_displacements(observed, future)
    assert enc.shape == (7, history - 1, 2) and target.shape == (7, horizon, 2)
    rec = reconstruct(observed[:, -1], target)
    assert rec.shape == (7, horizon, 2)
    np.testing.assert_allclose(rec, future, rtol=0, atol=1e-12)


def test_batch_rows_are_encoded_independently():
    observed, future = _windows(6, H, F, seed=3)
    enc, target = observed_displacements(observed), future_displacements(observed, future)
    for n in range(6):
        np.testing.assert_array_equal(enc[n], observed_displacements(observed[n:n + 1])[0])
        np.testing.assert_array_equal(target[n], future_displacements(observed[n:n + 1], future[n:n + 1])[0])


def test_single_step_horizon():
    observed, future = _windows(3, H, 1, seed=4)
    target = future_displacements(observed, future)
    assert target.shape == (3, 1, 2)
    np.testing.assert_array_equal(target[:, 0], future[:, 0] - observed[:, -1])
    np.testing.assert_array_equal(reconstruct(observed[:, -1], target)[:, 0], observed[:, -1] + target[:, 0])


def test_minimum_history_of_two_points():
    observed = np.array([[[1.0, 1.0], [1.5, 0.0]]])
    np.testing.assert_array_equal(observed_displacements(observed), [[[0.5, -1.0]]])


def test_empty_batch_keeps_its_shape():
    assert observed_displacements(np.zeros((0, H, 2))).shape == (0, H - 1, 2)
    assert future_displacements(np.zeros((0, H, 2)), np.zeros((0, F, 2))).shape == (0, F, 2)
    assert reconstruct(np.zeros((0, 2)), np.zeros((0, F, 2))).shape == (0, F, 2)


def test_displacements_are_translation_invariant():
    observed, future = _windows(5, H, F, seed=5)
    shift = np.array([1234.5, -678.25])
    np.testing.assert_allclose(observed_displacements(observed + shift), observed_displacements(observed),
                               rtol=0, atol=1e-9)
    target = future_displacements(observed, future)
    np.testing.assert_allclose(future_displacements(observed + shift, future + shift), target, rtol=0, atol=1e-9)
    np.testing.assert_allclose(reconstruct(observed[:, -1] + shift, target), future + shift, rtol=0, atol=1e-9)
    grid = np.arange(H * 2, dtype=float).reshape(1, H, 2)       # exact for a dyadic shift of small integers
    np.testing.assert_array_equal(observed_displacements(grid + 8.0), observed_displacements(grid))


def test_naive_and_vectorized_agree():
    for history, horizon in ((H, F), (HISTORY_LENS[0], HORIZON_LENS[0]), (2, 1)):
        observed, future = _windows(9, history, horizon, seed=history + horizon)
        np.testing.assert_allclose(observed_displacements(observed), _naive_observed(observed), rtol=0, atol=1e-12)
        target = future_displacements(observed, future)
        np.testing.assert_allclose(target, _naive_future(observed, future), rtol=0, atol=1e-12)
        np.testing.assert_allclose(reconstruct(observed[:, -1], target), _naive_reconstruct(observed[:, -1], target),
                                   rtol=0, atol=1e-12)


def test_inputs_are_not_modified_and_outputs_do_not_alias_them():
    observed, future = _windows(4, H, F, seed=6)
    last = observed[:, -1].copy()
    steps = future_displacements(observed, future)
    copies = [a.copy() for a in (observed, future, last, steps)]
    for a in (observed, future, last, steps):
        a.setflags(write=False)                              # any in-place write would raise
    outputs = [observed_displacements(observed), future_displacements(observed, future), reconstruct(last, steps)]
    for a, c in zip((observed, future, last, steps), copies):
        np.testing.assert_array_equal(a, c)
    for out in outputs:
        assert out.flags.writeable
        out += 1.0                                           # writing to an output leaves the inputs alone
    for a, c in zip((observed, future, last, steps), copies):
        np.testing.assert_array_equal(a, c)


# --- consistency with the existing predictors and windows --------------------------------------

def test_cv_is_reconstruction_of_the_repeated_last_displacement():
    observed, _ = _windows(8, H, F, seed=7)
    last_step = observed_displacements(observed)[:, -1:, :]
    np.testing.assert_allclose(reconstruct(observed[:, -1], np.repeat(last_step, F, axis=1)),
                               constant_velocity(observed, F, CV_MIN), rtol=0, atol=1e-12)


def test_true_displacements_of_u1_windows_reconstruct_to_zero_error():
    scene = generate(SceneConfig(seed=7, n_frames=1500), DT)
    w = make_windows("u1", ground_truth_grid_tracks(scene), scene.frames_per_step, H, F, STRIDE)
    assert len(w) > 0
    rec = reconstruct(w.observed[:, -1], future_displacements(w.observed, w.target))
    ade, fde = ade_fde(displacement_errors(rec, w.target))
    assert np.max(ade) <= 1e-12 and np.max(fde) <= 1e-12


# --- invalid inputs ----------------------------------------------------------------------------

GOOD_OBS, GOOD_FUT = np.zeros((2, H, 2)), np.zeros((2, F, 2))


@pytest.mark.parametrize("bad", [
    lambda: observed_displacements(np.zeros((2, H, 3))),
    lambda: observed_displacements(np.zeros((H, 2))),
    lambda: observed_displacements(np.zeros((2, 1, 2))),                 # one point: no displacement
    lambda: observed_displacements(np.zeros((2, 0, 2))),
    lambda: observed_displacements(np.full((2, H, 2), np.nan)),
    lambda: observed_displacements(np.full((2, H, 2), np.inf)),
    lambda: future_displacements(np.zeros((2, 0, 2)), GOOD_FUT),          # no last observation
    lambda: future_displacements(GOOD_OBS, np.zeros((2, 0, 2))),          # empty horizon
    lambda: future_displacements(GOOD_OBS, np.zeros((3, F, 2))),          # batch mismatch
    lambda: future_displacements(GOOD_OBS, np.zeros((2, F, 1))),
    lambda: future_displacements(np.where(np.eye(H, 2, dtype=bool)[None], np.nan, 0.0).repeat(2, 0), GOOD_FUT),
    lambda: future_displacements(GOOD_OBS, np.full((2, F, 2), -np.inf)),
    lambda: reconstruct(np.zeros((2, 1, 2)), GOOD_FUT),
    lambda: reconstruct(np.zeros((2, 3)), GOOD_FUT),
    lambda: reconstruct(np.zeros(2), GOOD_FUT),
    lambda: reconstruct(np.zeros((3, 2)), GOOD_FUT),                      # batch mismatch
    lambda: reconstruct(np.zeros((2, 2)), np.zeros((2, 0, 2))),           # empty horizon
    lambda: reconstruct(np.zeros((2, 2)), np.zeros((2, F))),
    lambda: reconstruct(np.array([[0.0, np.nan], [0.0, 0.0]]), GOOD_FUT),
    lambda: reconstruct(np.zeros((2, 2)), np.full((2, F, 2), np.inf)),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bad()
