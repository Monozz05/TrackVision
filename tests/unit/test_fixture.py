import numpy as np

from fixtures import synthetic


def test_walker_has_one_point_per_window_step(smoke_config):
    data = synthetic.build(smoke_config)
    static = smoke_config.frozen["protocol"]["static"]
    assert data["n_points"] == static["history_len"] + static["horizon_len"]
    assert data["walker"].shape == (data["n_points"], 2)


def test_walker_recovers_constructed_speed_and_heading(smoke_config):
    data = synthetic.build(smoke_config)
    deltas = np.diff(data["walker"], axis=0)
    steps = np.linalg.norm(deltas, axis=1)
    np.testing.assert_allclose(steps, data["expected"]["step_m"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(steps / data["dt"], data["expected"]["speed_mps"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.arctan2(deltas[:, 1], deltas[:, 0]),
                               data["expected"]["heading_rad"], rtol=0, atol=1e-12)


def test_stationary_point_does_not_move(smoke_config):
    data = synthetic.build(smoke_config)
    assert np.all(np.diff(data["stationary"], axis=0) == 0.0)


def test_offset_copy_is_exactly_one_metre_away_everywhere(smoke_config):
    data = synthetic.build(smoke_config)
    distances = np.linalg.norm(data["walker_offset_1m"] - data["walker"], axis=1)
    np.testing.assert_allclose(distances, data["expected"]["offset_m"], rtol=0, atol=1e-12)
