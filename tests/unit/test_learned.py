"""P2-e: LSTM and GRU displacement predictors (spec v4.15 section 5.4; decision 0001 G).

Parameter counts are checked against closed forms, not against the implementation. The controlled-fixture
overfitting test uses a threshold fixed from the fixture's numerical precision (decision 0001 G):

    fixture   4 straight constant-velocity windows; every position an exact dyadic multiple of q = 1/16 m
    tau       2**-10 m (about 0.98 mm) on every window's ADE
    anchors   above the float32 resolution at the largest coordinate (4.75 m -> about 4.8e-7 m), so the
              test never demands what float32 cannot represent; 1/64 of q, the smallest per-step difference
              between distinct fixture trajectories, so passing means each window is reproduced, not averaged
    budget    a fixed 4,000 full-batch steps with the approved optimiser, clipping and plateau schedule;
              training is deterministic, so one run decides and no run-to-run margin is needed
"""

import math

import numpy as np
import pytest
import torch

from trackvision import config
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict import learned
from trackvision.predict.displacements import future_displacements, reconstruct

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F = STATIC["history_len"], STATIC["horizon_len"]
LEARNED = CFG.frozen["predictors"]["learned"]
HIDDEN = LEARNED["hidden_size"]
HISTORY_LENS = tuple(CFG.frozen["analysis"]["history_ablation"]["history_lens"])
HORIZONS = (1, CFG.frozen["protocol"]["moving"]["fallback"]["horizon_len"], F)
TAU = 2.0 ** -10


def _expected_parameters(cell, horizon, hidden=HIDDEN, inputs=2):
    gates = 4 if cell == "LSTM" else 3
    recurrent = gates * (hidden * (inputs + hidden) + 2 * hidden)      # PyTorch keeps two bias vectors
    return recurrent + hidden * (2 * horizon) + 2 * horizon


def _model(cell, history=H, horizon=F, seed=0):
    torch.manual_seed(seed)
    return learned.build_model(cell, history, horizon, CFG)


def _observed(n, history=H, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.4, 0.1, size=(n, history, 2)), axis=1)


@pytest.mark.parametrize("cell", learned.CELLS)
def test_architecture_follows_the_configuration(cell):
    model = _model(cell)
    arch = model.architecture()
    assert (arch["hidden_size"], arch["num_layers"]) == (LEARNED["hidden_size"], LEARNED["num_layers"])
    assert arch["head"] == "linear_from_final_hidden_state"
    assert arch["attention"] is False and arch["autoregressive_feedback"] is False
    assert arch["n_parameters"] == _expected_parameters(cell, F)
    assert {name for name, _ in model.named_children()} == {"encoder", "head"}      # no decoder, no attention
    assert model.head.in_features == HIDDEN and model.head.out_features == 2 * F


def test_parameter_counts_at_the_static_protocol():
    assert _expected_parameters("LSTM", 12) == 70_680 and _expected_parameters("GRU", 12) == 53_784
    assert learned.parameter_count(_model("LSTM")) == 70_680
    assert learned.parameter_count(_model("GRU")) == 53_784


@pytest.mark.parametrize("cell", learned.CELLS)
@pytest.mark.parametrize("history", HISTORY_LENS)
@pytest.mark.parametrize("horizon", HORIZONS)
def test_output_dimensions_follow_history_and_horizon(cell, history, horizon):
    model = _model(cell, history, horizon)
    obs = _observed(5, history)
    disp = learned.predict_displacements(model, obs)
    pos = learned.predict_positions(model, obs)
    assert disp.shape == pos.shape == (5, horizon, 2)
    assert disp.dtype == pos.dtype == np.float64
    assert np.all(np.isfinite(pos))
    assert model.architecture()["n_parameters"] == _expected_parameters(cell, horizon)


@pytest.mark.parametrize("cell", learned.CELLS)
def test_positions_are_the_reconstruction_of_the_predicted_displacements(cell):
    model = _model(cell)
    obs = _observed(6)
    np.testing.assert_array_equal(learned.predict_positions(model, obs),
                                  reconstruct(obs[:, -1], learned.predict_displacements(model, obs)))


@pytest.mark.parametrize("cell", learned.CELLS)
def test_inputs_are_observed_displacements_only(cell):
    obs = _observed(4)
    np.testing.assert_array_equal(learned.encoder_inputs(obs), np.diff(obs, axis=1).astype(np.float32))
    model = _model(cell)
    shifted = obs + np.array([250.0, -40.0])                          # same displacements, new location
    np.testing.assert_allclose(learned.predict_displacements(model, shifted),
                               learned.predict_displacements(model, obs), rtol=0, atol=1e-5)


@pytest.mark.parametrize("cell", learned.CELLS)
def test_windows_are_predicted_independently_of_the_batch(cell):
    model = _model(cell)
    obs = _observed(7)
    together = learned.predict_displacements(model, obs)
    for i in range(7):
        np.testing.assert_allclose(learned.predict_displacements(model, obs[i:i + 1])[0], together[i],
                                   rtol=0, atol=1e-6)


@pytest.mark.parametrize("cell", learned.CELLS)
def test_forward_backward_gives_finite_loss_and_gradients(cell):
    model = _model(cell)
    obs, fut = _observed(16, H + F)[:, :H], _observed(16, H + F)[:, H:]
    x = torch.from_numpy(learned.encoder_inputs(obs))
    y = torch.from_numpy(future_displacements(obs, fut).astype(np.float32))
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


@pytest.mark.parametrize("bad", [
    lambda m: m(torch.zeros(2, H - 1, 2, dtype=torch.float64)),         # float64 input
    lambda m: m(torch.zeros(2, H, 2)),                                  # wrong history length
    lambda m: m(torch.zeros(2, H - 1, 3)),
    lambda m: m(torch.zeros(H - 1, 2)),
    lambda m: learned.predict_positions(m, _observed(2, H + 1)),        # model built for another history
    lambda m: learned.predict_positions(m, np.full((2, H, 2), np.nan)),
    lambda m: learned.predict_positions(m, np.zeros((2, H, 3))),
])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        bad(_model("LSTM"))


@pytest.mark.parametrize("args", [("RNN", H, F, HIDDEN, 1), ("LSTM", 1, F, HIDDEN, 1), ("GRU", H, 0, HIDDEN, 1),
                                  ("LSTM", H, F, 0, 1), ("GRU", H, F, HIDDEN, 0), ("LSTM", H, 12.0, HIDDEN, 1)])
def test_invalid_architectures_raise(args):
    with pytest.raises(ValueError):
        learned.DisplacementPredictor(*args)


def test_empty_batch_keeps_its_shape():
    assert learned.predict_positions(_model("GRU"), np.zeros((0, H, 2))).shape == (0, F, 2)


@pytest.mark.parametrize("cell", learned.CELLS)
def test_initialisation_is_seeded(cell):
    flat = lambda m: torch.cat([p.detach().flatten() for p in m.parameters()])
    assert torch.equal(flat(_model(cell, seed=0)), flat(_model(cell, seed=0)))
    inits = [flat(_model(cell, seed=s)) for s in (0, 1, 2)]
    assert not torch.equal(inits[0], inits[1]) and not torch.equal(inits[1], inits[2])


# --- controlled-fixture overfitting (P3 sanity) ------------------------------------------------

def _fixture():
    steps = np.array([[1 / 16, 1 / 16], [2 / 16, -1 / 16], [-1 / 16, 2 / 16], [3 / 16, 0.0]])
    pos = np.arange(H + F, dtype=float)[None, :, None] * steps[:, None, :]
    return pos[:, :H], pos[:, H:]


def test_fixture_is_exact_in_float32_and_tau_lies_between_its_anchors():
    obs, fut = _fixture()
    for arr in (obs, fut):
        np.testing.assert_array_equal(arr.astype(np.float32).astype(np.float64), arr)
        np.testing.assert_array_equal(arr * 16, np.round(arr * 16))
    floor = float(np.spacing(np.float32(np.abs(np.concatenate([obs, fut], axis=1)).max())))
    assert floor < TAU / 1000 and TAU == (1 / 16) / 64


@pytest.mark.parametrize("cell", learned.CELLS)
def test_controlled_fixture_is_overfit_below_tau(cell):
    obs, fut = _fixture()
    x = torch.from_numpy(learned.encoder_inputs(obs))
    y = torch.from_numpy(future_displacements(obs, fut).astype(np.float32))
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = _model(cell)
        opt = torch.optim.Adam(model.parameters(), lr=LEARNED["learning_rate"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, min_lr=1e-5)
        first_loss = None
        for _ in range(4000):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), y)
            first_loss = first_loss if first_loss is not None else loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), LEARNED["grad_clip_norm"])
            opt.step()
            sched.step(loss.item())
    finally:
        torch.use_deterministic_algorithms(previous)
    ade, fde = ade_fde(displacement_errors(learned.predict_positions(model, obs), fut))
    assert math.isfinite(loss.item()) and loss.item() < first_loss
    assert ade.max() < TAU, f"max ADE {ade.max():.3e} m is not below tau = {TAU:.3e} m"
