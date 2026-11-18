"""P2-a: Euclidean RMSE (spec v4.15 section 3.1, property tests of section 11.2).

RMSE = sqrt((1/T) sum_t ||p_hat_t - p_t||^2), computed from the same per-step Euclidean errors as ADE and
FDE. The per-coordinate form sqrt((1/2T) sum (dx^2 + dy^2)) is never used; the (1, 1) offset case tells
the two apart (sqrt(2) versus 1).
"""

import math

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.metrics.displacement import ade_fde, displacement_errors, rmse

CFG = config.validate().config
F = protocol(CFG)["horizon_len"]


def _target(n=3, horizon=F, seed=0):
    return np.random.default_rng(seed).normal(scale=4.0, size=(n, horizon, 2))


@pytest.mark.parametrize("horizon", [8, F])
def test_perfect_prediction_gives_zero(horizon):
    target = _target(horizon=horizon)
    assert np.all(rmse(displacement_errors(target.copy(), target)) == 0.0)


@pytest.mark.parametrize("direction", [(1.0, 0.0), (0.0, -1.0), (0.6, 0.8), (-math.sqrt(0.5), math.sqrt(0.5))])
def test_constant_one_metre_offset_gives_one(direction):
    target = _target()
    per_step = displacement_errors(target + np.asarray(direction), target)
    np.testing.assert_allclose(rmse(per_step), 1.0, rtol=0, atol=1e-12)
    ade, fde = ade_fde(per_step)
    np.testing.assert_allclose(ade, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fde, 1.0, rtol=0, atol=1e-12)


def test_offset_one_one_gives_sqrt_two_not_the_per_coordinate_value():
    target = _target()
    pred = target + np.array([1.0, 1.0])
    value = rmse(displacement_errors(pred, target))
    np.testing.assert_allclose(value, math.sqrt(2.0), rtol=0, atol=1e-12)
    per_coordinate = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))    # the excluded form gives 1
    np.testing.assert_allclose(per_coordinate, 1.0, rtol=0, atol=1e-12)
    assert np.all(np.abs(value - per_coordinate) > 0.4)


def test_per_step_errors_three_and_four():
    assert rmse(np.array([[3.0, 4.0]]))[0] == pytest.approx(math.sqrt(12.5), abs=1e-12)
    pred = np.array([[[3.0, 0.0], [0.0, 4.0]]])                  # the same errors from positions
    assert rmse(displacement_errors(pred, np.zeros((1, 2, 2))))[0] == pytest.approx(math.sqrt(12.5), abs=1e-12)


def test_rmse_is_at_least_ade_with_equality_for_constant_errors():
    rng = np.random.default_rng(1)
    pred, target = rng.normal(size=(50, F, 2)), rng.normal(size=(50, F, 2))
    per_step = displacement_errors(pred, target)
    ade, _ = ade_fde(per_step)
    assert np.all(rmse(per_step) >= ade - 1e-12)
    assert np.all(rmse(per_step) > ade)                          # strict: these errors are never constant
    constant = np.full((4, F), 0.7)
    np.testing.assert_allclose(rmse(constant), ade_fde(constant)[0], rtol=0, atol=1e-12)


def test_naive_and_vectorized_rmse_agree():
    rng = np.random.default_rng(2)
    pred, target = rng.normal(size=(7, F, 2)), rng.normal(size=(7, F, 2))
    value = rmse(displacement_errors(pred, target))
    for n in range(7):
        squared = [(pred[n, t, 0] - target[n, t, 0]) ** 2 + (pred[n, t, 1] - target[n, t, 1]) ** 2 for t in range(F)]
        assert value[n] == pytest.approx(math.sqrt(sum(squared) / F), abs=1e-12)


def test_output_has_one_value_per_window():
    assert rmse(np.ones((5, F))).shape == (5,)
    assert rmse(np.ones((0, F))).shape == (0,)


@pytest.mark.parametrize("bad", [
    np.ones(F),                         # not (N, T)
    np.ones((2, F, 2)),                 # positions, not per-step errors
    np.ones((2, 0)),                    # empty horizon
    np.full((1, F), np.nan),
    np.full((1, F), np.inf),
    -np.ones((1, F)),                   # Euclidean errors are never negative
])
def test_invalid_per_step_errors_raise(bad):
    with pytest.raises(ValueError):
        rmse(bad)
