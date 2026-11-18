"""Phase 3: validation of the accepted A, B1, B2, C and D injectors (U7, U8) against v4.15 sections 3.5, 3.6,
3.7 (synthetic lineage), 5.2 and 11.2, covering only what the U7/U8/R2 tests do not already cover:

  - association self-match (P4 gate): GT recovered through each injector's lineage map gives exactly zero error;
  - maximum dropout (p = 1): coverage saturates at zero without NaN inputs;
  - empty and degenerate inputs: achieved severity is undefined (None), never 0 or NaN;
  - identity-switch saturation: the cooldown caps events at floor(tracks / 2);
  - per-pedestrian, purpose-specific random streams for B, C and D (A is covered in test_injection);
  - D changed-input footprints for CV (2 points), CA (3) and the learned predictors (whole history);
  - integration: injected inputs feed CV, CA, LSTM and GRU and the displacement metrics, targets untouched.

The injectors are not modified here. Scenes are U1 synthetic ground truth.
"""

import numpy as np
import pytest
import torch

from trackvision import config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments import switch_feasibility as sf
from trackvision.inject import identity_switch as ids
from trackvision.inject.trajectory import FAMILY_STREAM, dropout_masks, fragment, localization_noise
from trackvision.metrics.displacement import ade_fde, displacement_errors, rmse
from trackvision.predict import learned, training
from trackvision.predict.baselines import constant_acceleration, constant_velocity
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, DT = STATIC["history_len"], STATIC["horizon_len"], STATIC["sampling_interval_s"]
PRED = CFG.frozen["predictors"]
D_MAX = float(CFG.frozen["injection"]["d_max_m"])
SCENE = SceneConfig(seed=4, n_frames=300)                  # as tests/unit/test_injection.py
SWITCH_SCENE = SceneConfig(seed=21, n_frames=300)          # as tests/unit/test_identity_switch.py


def _clean(scene_config):
    scene = generate(scene_config, DT)
    return ground_truth_grid_tracks(scene), scene.frames_per_step


@pytest.fixture(scope="module")
def run():
    return inj.run_injection(CFG, SCENE)


@pytest.fixture(scope="module")
def switch_run():
    return sf.run_switch(CFG, SWITCH_SCENE, seeds=(0,))


# --- association self-match through the synthetic lineage (P4 gate, spec 3.7, 11.2) ------------

def _position_of(clean, ped_id, frame):
    t = next(t for t in clean if int(t.ped_id) == ped_id)
    return t.positions[int(np.flatnonzero(t.frames == frame)[0])]


def test_lineage_recovers_ground_truth_exactly():
    """Every corrupted sample, mapped to its originating pedestrian through the injector's lineage, sits on
    that pedestrian's clean position at the same frame: zero error, not small error."""
    clean, step = _clean(SWITCH_SCENE)
    for p in (0.0, 1.0):
        frag = fragment(clean, p, 0)
        for t in frag.tracks:
            g, a, b = frag.origin[int(t.ped_id)]
            src = next(c for c in clean if int(c.ped_id) == g)
            assert np.array_equal(t.frames, src.frames[a:b]) and np.array_equal(t.positions, src.positions[a:b])
        sw = ids.identity_switch(clean, p, 0, D_MAX, step)
        assert (len(sw.events) > 0) == (p > 0)
        errors = [np.linalg.norm(pos - _position_of(clean, int(o), int(f)))
                  for t in sw.tracks for f, pos, o in zip(t.frames, t.positions, sw.occupant[int(t.ped_id)])]
        assert len(errors) == sum(len(t.frames) for t in clean) and max(errors) == 0.0
    for zero in (localization_noise(clean, 0.0, 0).tracks,):
        assert all(np.array_equal(a.positions, b.positions) for a, b in zip(clean, zero))


def test_level_zero_inputs_equal_the_clean_histories_exactly(run, switch_run):
    observed = run.population.windows.observed
    for c in inj.CONDITIONS:
        for r in (r for r in run.results if r.condition == c and r.level == 0.0):
            assert np.max(displacement_errors(r.inputs, observed)) == 0.0
    (d0,) = [r for r in switch_run.results if r.level == 0.0]
    assert np.max(displacement_errors(d0.windows.inputs, switch_run.population.windows.observed)) == 0.0


# --- maximum and degenerate severities ---------------------------------------------------------

def test_total_dropout_saturates_coverage_without_nan_inputs(run):
    clean, _ = _clean(SCENE)
    pop = run.population
    drop = dropout_masks(clean, 1.0, 0)
    assert drop.achieved["rate"] == 1.0 and not any(m.any() for m in drop.present)
    for reconstruct in (False, True):
        wi = inj.dropout_windows(clean, pop, drop.present, reconstruct)
        assert not wi.predicted.any() and wi.inputs.shape == (0, H, 2)
        assert wi.extra["prediction_coverage_pre"] == wi.extra["prediction_coverage_post"] == 0.0
        if reconstruct:
            rec = wi.extra["reconstruction"]
            assert rec["interior_reconstructed_window_observations"] == 0
            assert rec["windows_with_unresolved_boundary_gap"] == len(pop.windows)
    for condition in ("B1", "B2"):
        r = inj.evaluate_condition(clean, pop, condition, 1.0, 0, CFG)
        assert r.summary["coverage"]["prediction_coverage"] == 0.0 and r.per_step.shape == (0, F)
        assert r.summary["accuracy_on_predicted_windows"]["injected"]["mean_ade_m"] is None


def test_empty_and_degenerate_inputs_give_undefined_severity():
    assert localization_noise((), 0.1, 0).achieved["rmse_pos_m"] is None
    assert dropout_masks((), 0.1, 0).achieved["rate"] is None
    empty_c = fragment((), 0.1, 0).achieved
    assert empty_c["fragmentation_rate"] is None and empty_c["fragments_per_original_track"] is None
    assert ids.identity_switch((), 0.1, 0, D_MAX, 10).achieved["switch_event_rate"] is None
    single = (GridTrack(1, np.array([0]), np.array([[1.0, 2.0]])), GridTrack(2, np.array([0]), np.array([[1.5, 2.0]])))
    assert localization_noise(single, 0.0, 0).achieved == {"rmse_pos_m": 0.0, "n_observations": 2}
    assert dropout_masks(single, 1.0, 0).achieved["rate"] == 1.0
    assert fragment(single, 1.0, 0).achieved["eligible_track_frames"] == 0
    assert fragment(single, 1.0, 0).achieved["fragmentation_rate"] is None
    one_step = ids.identity_switch(single, 1.0, 0, D_MAX, 10)           # no sample at k - 1: nothing eligible
    assert one_step.achieved["eligible_track_frames"] == 0 and one_step.events == ()


def test_switch_saturation_is_capped_by_the_cooldown():
    clean, step = _clean(SWITCH_SCENE)
    sw = ids.identity_switch(clean, 1.0, 0, D_MAX, step)
    a = sw.achieved
    participants = [x for e in sw.events for x in (e.initiator, e.partner)]
    assert len(participants) == len(set(participants)) == a["participating_tracks"]
    assert a["switch_events"] <= len(clean) // 2
    assert a["switch_event_rate"] <= 100.0 * (len(clean) // 2) / a["eligible_track_frames"]
    assert a["triggers"] == a["switch_events"] + a["suppressed_triggers"] <= a["lock_aware_exposure"]
    assert a["lock_aware_exposure"] <= a["eligible_track_frames"]


# --- random streams ----------------------------------------------------------------------------

def test_random_streams_are_purpose_specific_and_per_pedestrian():
    codes = {**FAMILY_STREAM, "D": ids.D_STREAM}
    assert sorted(codes.values()) == [1, 2, 3, 4]
    assert not set(codes.values()) & {training.PURPOSE_INIT, training.PURPOSE_SHUFFLE}
    clean, step = _clean(SWITCH_SCENE)
    one = clean[:1]
    np.testing.assert_array_equal(dropout_masks(one, 0.3, 5).present[0], dropout_masks(clean, 0.3, 5).present[0])
    alone = fragment(one, 0.3, 5).origin
    with_all = {k: v for k, v in fragment(clean, 0.3, 5).origin.items() if v[0] == int(clean[0].ped_id)}
    assert sorted(v for v in alone.values()) == sorted(v for v in with_all.values())
    np.testing.assert_array_equal(ids.switch_draws(one, 5)[int(clean[0].ped_id)],
                                  ids.switch_draws(clean, 5)[int(clean[0].ped_id)])
    t = clean[0]
    n = len(t.frames)
    draws = {fam: np.random.default_rng(np.random.SeedSequence([5, code, int(t.ped_id)])).random(n)
             for fam, code in codes.items()}
    assert len({d.tobytes() for d in draws.values()}) == 4                  # families never share a stream
    assert not np.array_equal(dropout_masks(one, 0.3, 5).present[0], dropout_masks(one, 0.3, 6).present[0])


# --- invalid inputs and their valid boundary cases (decision 0002) -----------------------------

def _line(pid, y, n=10, step=10):
    return GridTrack(pid, step * np.arange(n), np.column_stack([0.3 * np.arange(n), np.full(n, y)]))


ALL_INJECTORS = [lambda t: localization_noise(t, 0.1, 0), lambda t: dropout_masks(t, 0.1, 0),
                 lambda t: fragment(t, 0.1, 0), lambda t: ids.identity_switch(t, 0.1, 0, D_MAX, 10)]


@pytest.mark.parametrize("sigma", [float("inf"), float("-inf"), float("nan"), np.inf, -0.1])
def test_non_finite_or_negative_sigma_is_rejected(sigma):
    with pytest.raises(ValueError, match="sigma must be finite and non-negative"):
        localization_noise((_line(1, 0.0),), sigma, 0)


@pytest.mark.parametrize("sigma", [0.0, 0, 1e-12, 0.5, 1e6, np.float64(0.2)])
def test_finite_sigma_boundary_cases_are_accepted(sigma):
    out = localization_noise((_line(1, 0.0),), sigma, 0)
    assert np.all(np.isfinite(out.tracks[0].positions))


@pytest.mark.parametrize("frames", [[0, 0, 10], [20, 0, 10], [0, 10, 10], [30, 20, 10]])
@pytest.mark.parametrize("inject", ALL_INJECTORS)
def test_repeated_or_non_increasing_frames_are_rejected(frames, inject):
    with pytest.raises(ValueError, match="unique and strictly increasing"):
        inject((GridTrack(1, np.array(frames), np.zeros((3, 2))),))


def test_two_dimensional_frames_are_rejected():
    with pytest.raises(ValueError):
        fragment((GridTrack(1, np.zeros((2, 1), dtype=np.int64), np.zeros((2, 2))),), 0.1, 0)


@pytest.mark.parametrize("frames", [[0], [0, 10, 20], [5, 6, 7], [0, 10, 50], [-20, -10, 0]])
@pytest.mark.parametrize("inject", ALL_INJECTORS)
def test_increasing_frames_are_accepted_including_gaps_and_a_single_sample(frames, inject):
    n = len(frames)
    inject((GridTrack(1, np.array(frames), np.zeros((n, 2))),))


def test_a_track_with_no_samples_is_still_accepted():
    assert fragment((GridTrack(1, np.empty(0, np.int64), np.empty((0, 2))),), 0.1, 0).achieved["split_events"] == 0


BAD_STEPS = [0, -10, 2.5, 10.0, True, np.True_, None, "10"]


@pytest.mark.parametrize("step", BAD_STEPS)
def test_invalid_frame_step_is_rejected_by_every_entry_point(step):
    tracks = (_line(1, 0.0), _line(2, 1.0))
    event = ids.SwitchEvent(10, 1, 2, 1.0)
    for call in (lambda: ids.proximity(tracks, D_MAX, step),
                 lambda: ids.sample_switches(tracks, 0.1, 0, D_MAX, step),
                 lambda: ids.identity_switch(tracks, 0.1, 0, D_MAX, step),
                 lambda: ids.apply_switches(tracks, [event], step)):
        with pytest.raises(ValueError, match="frame_step must be a positive integer"):
            call()


def test_valid_frame_steps_keep_the_exact_denominator():
    """Two tracks of 10 samples: 9 samples each have a predecessor one step back, so 18 eligible frames."""
    tracks = (_line(1, 0.0), _line(2, 1.0))
    assert ids.proximity(tracks, D_MAX, 10).eligible_track_frames == 18
    assert ids.proximity(tracks, D_MAX, np.int64(10)).eligible_track_frames == 18
    assert ids.proximity(tracks, D_MAX, 1).eligible_track_frames == 0           # valid: no sample one frame back
    assert ids.proximity((_line(1, 0.0, step=1), _line(2, 1.0, step=1)), D_MAX, 1).eligible_track_frames == 18


# --- D changed-input footprints per predictor --------------------------------------------------

DENSE_SWITCH_SCENE = SceneConfig(seed=21, n_frames=750)    # enough switches inside histories to separate footprints


def test_switch_changed_input_footprints_nest_by_predictor():
    clean, step = _clean(DENSE_SWITCH_SCENE)
    sw = ids.identity_switch(clean, max(inj.levels(CFG, "D")), 0, D_MAX, step)
    pop = sf.run_switch(CFG, DENSE_SWITCH_SCENE, seeds=(0,)).population
    cv = sf.switch_windows(pop, sw.tracks, sw.occupant, PRED["cv_min_final_observations"])
    ca = sf.switch_windows(pop, sw.tracks, sw.occupant, PRED["ca_min_final_observations"])
    full = sf.switch_windows(pop, sw.tracks, sw.occupant, H)                    # learned: the whole history
    assert np.array_equal(cv.predicted, full.predicted) and np.array_equal(cv.mixed, full.mixed)
    assert np.all(~cv.cv_changed | ca.cv_changed) and np.all(~ca.cv_changed | full.cv_changed)
    assert np.array_equal(full.cv_changed, full.mixed)
    assert full.mixed.sum() > cv.cv_changed.sum() > 0                          # the footprints really differ


# --- integration with the prediction and metric pipeline (Step 5) ------------------------------

def _predictors():
    models = {}
    for cell in learned.CELLS:
        torch.manual_seed(0)
        models[cell] = learned.build_model(cell, H, F, CFG)
    return {
        "CV": lambda x: constant_velocity(x, F, PRED["cv_min_final_observations"]),
        "CA": lambda x: constant_acceleration(x, F, PRED["ca_min_final_observations"]),
        **{cell: (lambda x, m=m: learned.predict_positions(m, x)) for cell, m in models.items()},
    }


def test_injected_inputs_feed_every_predictor_and_metric_without_touching_targets(run, switch_run):
    predictors = _predictors()
    target = run.population.windows.target
    before = inj._digest(target)
    assert not target.flags.writeable
    checked = 0
    for r in run.results:
        if r.level == 0.0 or r.seed != 0 or len(r.inputs) == 0:
            continue
        idx = np.flatnonzero(r.predicted)
        for name, predict in predictors.items():
            pred = predict(r.inputs)
            assert pred.shape == (len(idx), F, 2) and np.all(np.isfinite(pred)), (r.condition, name)
            per_step = displacement_errors(pred, target[idx])
            ade, fde = ade_fde(per_step)
            assert np.all(np.isfinite(ade)) and np.all(rmse(per_step) >= ade - 1e-12)
            checked += 1
        np.testing.assert_array_equal(predictors["CV"](r.inputs)[:, :1].shape, (len(idx), 1, 2))
    (d,) = [r for r in switch_run.results if r.level == max(inj.levels(CFG, "D"))]
    d_idx = np.flatnonzero(d.windows.predicted)
    for name, predict in predictors.items():
        pred = predict(d.windows.inputs)
        assert pred.shape == (len(d_idx), F, 2) and np.all(np.isfinite(pred))
        assert np.all(np.isfinite(ade_fde(displacement_errors(pred, switch_run.population.windows.target[d_idx]))[0]))
        checked += 1
    assert checked >= 4 * (len(inj.CONDITIONS) + 1)
    assert inj._digest(target) == before == run.targets_sha256
    np.testing.assert_array_equal(d.per_step, displacement_errors(
        predictors["CV"](d.windows.inputs), switch_run.population.windows.target[d_idx]))
