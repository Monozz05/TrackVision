"""U7: controlled injection A, B1, B2 and C on clean synthetic ground truth, with CV coverage and accuracy.

Hand-built trajectories give hand-derived window counts, interpolated values and errors. Injection seeds
(0, 1, 2) are provisional feasibility values. The generated scene's seed is declared here and is never
used to choose parameters. Saved runs go to pytest's temporary directories only.
"""

import json

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments.synthetic_feasibility import evaluate_ground_truth_cv
from trackvision.inject.trajectory import (Fragmentation, dropout_masks, fragment, localization_noise,
                                           reconstruct_history)
from trackvision.metrics.displacement import displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, STRIDE = STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"]
DT = STATIC["sampling_interval_s"]
S = 10                                    # frames per grid step at 25 fps


def gt_track(pid, positions, first_sample=0):
    positions = np.asarray(positions, dtype=np.float64)
    return GridTrack(pid, (first_sample + np.arange(len(positions), dtype=np.int64)) * S, positions)


def straight(n, step=0.5, y=0.0):
    return [(step * k, y) for k in range(n)]


def hand_pop(tracks):
    windows = make_windows("hand", tracks, S, H, F, STRIDE)
    per_step = displacement_errors(constant_velocity(windows.observed, F, 2), windows.target)
    return inj.population(tuple(tracks), windows, per_step)


def anchors(pop, mask):
    return (pop.windows.frames[mask, H - 1] // S).tolist()


def cv_errors(pop, wi):
    return displacement_errors(constant_velocity(wi.inputs, F, 2), pop.windows.target[wi.predicted])


def base(n_tracks, n_samples):
    return tuple(gt_track(i + 1, [(0.3 * k, 1.0 * i) for k in range(n_samples)]) for i in range(n_tracks))


# --- A: localization noise ---------------------------------------------------------------------

def test_a_zero_sigma_is_bitwise_clean_and_preserves_structure():
    tracks = (gt_track(1, straight(25)), gt_track(2, straight(22, y=3.0), first_sample=2))
    out = localization_noise(tracks, 0.0, 0)
    for a, b in zip(tracks, out.tracks):
        assert a.ped_id == b.ped_id
        np.testing.assert_array_equal(a.frames, b.frames)
        np.testing.assert_array_equal(a.positions, b.positions)
    assert out.achieved["rmse_pos_m"] == 0.0 and out.achieved["n_observations"] == 47
    noisy = localization_noise(tracks, 0.2, 0)
    for a, b in zip(tracks, noisy.tracks):
        np.testing.assert_array_equal(a.frames, b.frames)
        assert a.ped_id == b.ped_id and not np.array_equal(a.positions, b.positions)


def test_a_cv_error_is_exactly_the_propagated_offset_on_a_straight_line():
    tracks = (gt_track(1, straight(25)),)
    pop = hand_pop(tracks)
    res = inj.evaluate_condition(tracks, pop, "A", 0.1, 0, CFG)
    eps = localization_noise(tracks, 0.1, 0).offsets[0]
    assert res.predicted.all()
    for w in range(len(pop.windows)):
        a = int(pop.history_samples[w, -1])
        k = np.arange(1, F + 1)[:, None]
        expected = np.linalg.norm((1 + k) * eps[a] - k * eps[a - 1], axis=1)
        np.testing.assert_allclose(res.per_step[w], expected, atol=1e-12)


def test_a_achieved_rmse_definition_scaling_and_independent_streams():
    tracks = base(20, 200)                                  # N = 4000 observations
    a1, a2 = localization_noise(tracks, 0.1, 0), localization_noise(tracks, 0.2, 0)
    sq = np.concatenate([np.sum(o ** 2, axis=1) for o in a1.offsets])
    assert a1.achieved["rmse_pos_m"] == pytest.approx(np.sqrt(sq.mean()), rel=1e-15)
    # sigma is per coordinate and RMSE_pos is the 2-D Euclidean RMSE, so RMSE^2 * N / sigma^2 ~ chi2(2N):
    # E[RMSE^2] = 2 sigma^2, i.e. RMSE_pos ~ sigma * sqrt(2); relative sd ~ 1 / (2 sqrt(N)) ~ 0.008, so
    # 5 % is > 6 sd.
    assert abs(a1.achieved["rmse_pos_m"] / (0.1 * np.sqrt(2)) - 1) < 0.05
    for o1, o2 in zip(a1.offsets, a2.offsets):
        np.testing.assert_allclose(o2, 2 * o1, rtol=1e-12)    # common Gaussian draws across levels
    alone = localization_noise(tracks[:1], 0.1, 0)
    np.testing.assert_array_equal(alone.offsets[0], a1.offsets[0])   # per-pedestrian streams
    other = localization_noise(tracks, 0.1, 1)
    assert not np.array_equal(other.offsets[0], a1.offsets[0])


# --- B: dropout and reconstruction -------------------------------------------------------------

def test_b_masks_are_nested_and_achieved_rate_is_the_exact_count_ratio():
    tracks = base(10, 50)
    previous = None
    for p in inj.levels(CFG, "B2"):
        d = dropout_masks(tracks, p, 0)
        removed = sum(int(np.sum(~m)) for m in d.present)
        assert d.achieved == {"removed_observations": removed, "eligible_observations": 500, "rate": removed / 500}
        if p == 0:
            assert removed == 0
        removed_now = [~m for m in d.present]
        if previous is not None:                              # removed at lower p => removed at higher p
            assert all(np.all(~prev | cur) for prev, cur in zip(previous, removed_now))
        previous = removed_now


def test_b_hand_mask_b2_loses_eight_b1_loses_two_boundary_windows():
    tracks = (gt_track(1, straight(30)),)                   # eligible anchors 7 .. 17 (11 windows)
    pop = hand_pop(tracks)
    present = (np.arange(30) != 10,)
    b2 = inj.dropout_windows(tracks, pop, present, reconstruct=False)
    assert anchors(pop, ~b2.predicted) == list(range(10, 18))
    assert b2.extra["prediction_coverage_pre"] == b2.extra["prediction_coverage_post"] == pytest.approx(3 / 11)
    assert not np.isnan(b2.inputs).any() and b2.inputs.shape == (3, H, 2)    # never shortened

    b1 = inj.dropout_windows(tracks, pop, present, reconstruct=True)
    assert anchors(pop, ~b1.predicted) == [10, 17]           # sample 10 last / first in the history
    assert all(r == "boundary_gap" for r, ok in zip(b1.reason, b1.predicted) if not ok)
    assert b1.extra["prediction_coverage_pre"] == pytest.approx(3 / 11)
    assert b1.extra["prediction_coverage_post"] == pytest.approx(9 / 11)
    rec = b1.extra["reconstruction"]
    assert rec["windows_with_reconstruction"] == 6 and rec["interior_reconstructed_window_observations"] == 6
    assert rec["distinct_observations_reconstructed"] == 1 and rec["windows_with_unresolved_boundary_gap"] == 2
    assert "NON-CAUSAL" in rec["note"]
    # Constant speed: the midpoint of samples 9 and 11 is exact, so B1 equals the clean inputs.
    np.testing.assert_allclose(b1.inputs, pop.windows.observed[b1.predicted], atol=1e-12)


def test_b1_interpolation_inside_a_turn_differs_from_clean():
    pos = [(float(k), 0.0) for k in range(11)] + [(10.0, float(k - 10)) for k in range(11, 30)]
    tracks = (gt_track(1, pos),)
    pop = hand_pop(tracks)
    b1 = inj.dropout_windows(tracks, pop, (np.arange(30) != 10,), reconstruct=True)
    rows = np.flatnonzero(b1.predicted)
    for k, w in enumerate(rows):
        j = np.flatnonzero(pop.history_samples[w] == 10)
        if len(j):
            np.testing.assert_allclose(b1.inputs[k, j[0]], [9.5, 0.5])     # the straight-line midpoint
    errors = cv_errors(pop, b1)
    clean = pop.clean_per_step[b1.predicted]
    anchor = anchors(pop, b1.predicted)
    for k, a in enumerate(anchor):
        if a == 11:                       # sample 10 is the second-to-last point: CV is affected
            assert not np.allclose(errors[k], clean[k])
        else:                             # CV reads only the last two points, which survive
            np.testing.assert_allclose(errors[k], clean[k], atol=1e-12)


def test_b1_never_uses_the_target_period():
    # One eligible window: samples 0 .. 7 are the history, 8 .. 19 the target period.
    clean = (gt_track(1, [(0.1 * k * k, 0.4 * k) for k in range(20)]),)
    pop = hand_pop(clean)
    assert len(pop.windows) == 1
    present = np.ones(20, dtype=bool)
    present[[3, 4]] = False
    reference = inj.dropout_windows(clean, pop, (present,), reconstruct=True)
    assert reference.predicted.all()
    poisoned_pos = clean[0].positions.copy()
    poisoned_pos[8:] = np.nan                                 # target-period observations unusable
    poisoned = (GridTrack(1, clean[0].frames, poisoned_pos),)
    dropped = present.copy()
    dropped[8:] = False                                       # and all removed
    for tracks, mask in ((poisoned, present), (clean, dropped)):
        out = inj.dropout_windows(tracks, pop, (mask,), reconstruct=True)
        np.testing.assert_array_equal(out.inputs, reference.inputs)
    # A missing last history point is never filled from the surviving next sample.
    last = present.copy()
    last[7] = False
    out = inj.dropout_windows(clean, pop, (last,), reconstruct=True)
    assert not out.predicted.any() and out.reason == ("boundary_gap",)
    with pytest.raises(ValueError, match="finite"):
        reconstruct_history(np.full((H, 2), np.nan), np.ones(H, dtype=bool))


def test_b1_interior_gap_gets_the_same_values_in_every_window_containing_it():
    tracks = (gt_track(1, [(0.05 * k ** 2, np.sin(k / 3.0)) for k in range(30)]),)
    pop = hand_pop(tracks)
    present = np.ones(30, dtype=bool)
    present[[12, 13]] = False
    b1 = inj.dropout_windows(tracks, pop, (present,), reconstruct=True)
    values = {12: set(), 13: set()}
    for k, w in enumerate(np.flatnonzero(b1.predicted)):
        for sample in (12, 13):
            j = np.flatnonzero(pop.history_samples[w] == sample)
            if len(j):
                values[sample].add(tuple(b1.inputs[k, j[0]].round(12)))
    assert len(values[12]) == 1 and len(values[13]) == 1


# --- C: fragmentation --------------------------------------------------------------------------

def test_c_denominator_extremes_identities_and_positions():
    tracks = tuple(gt_track(i + 1, straight(n, y=float(i))) for i, n in enumerate((1, 2, 3, 10)))
    none = fragment(tracks, 0.0, 0)
    assert none.achieved["split_events"] == 0 and none.achieved["eligible_track_frames"] == 0 + 0 + 1 + 8
    assert none.achieved["fragmentation_rate"] == 0.0 and none.achieved["fragments_per_original_track"] == 1.0
    full = fragment(tracks, 1.0, 0)
    assert full.achieved["split_events"] == 9 and full.achieved["fragmentation_rate"] == 1.0
    assert full.achieved["fragments_per_original_track"] == (4 + 9) / 4
    ids = [t.ped_id for t in full.tracks]
    assert len(ids) == len(set(ids)) == 4 + 9
    assert all(fid > 4 for fid in ids if fid not in (1, 2, 3, 4))
    for t in tracks:                                          # positions and frames preserved, never reconnected
        parts = sorted((full.origin[f][1], f) for f in full.origin if full.origin[f][0] == t.ped_id)
        frames = np.concatenate([next(x for x in full.tracks if x.ped_id == f).frames for _, f in parts])
        pos = np.concatenate([next(x for x in full.tracks if x.ped_id == f).positions for _, f in parts])
        np.testing.assert_array_equal(frames, t.frames)
        np.testing.assert_array_equal(pos, t.positions)
    ten = [x for x in full.tracks if full.origin[x.ped_id][0] == 4]
    assert [len(x.frames) for x in sorted(ten, key=lambda x: x.frames[0])] == [2] + [1] * 8


def test_c_splits_are_nested_across_levels():
    tracks = base(10, 60)
    previous = set()
    for p in inj.levels(CFG, "C"):
        f = fragment(tracks, p, 0)
        starts = {(g, a) for g, a, _ in f.origin.values() if a > 0}
        assert previous <= starts
        assert f.achieved["split_events"] == len(starts)
        previous = starts


def test_c_single_split_loses_seven_windows_and_keeps_clean_accuracy():
    pos = np.array([(0.05 * k ** 2, 0.3 * k) for k in range(40)])
    tracks = (gt_track(1, pos),)                              # eligible anchors 7 .. 27 (21 windows)
    pop = hand_pop(tracks)
    t = tracks[0]
    frag = Fragmentation((GridTrack(1, t.frames[:15], t.positions[:15]), GridTrack(2, t.frames[15:], t.positions[15:])),
                         {1: (1, 0, 15), 2: (1, 15, 40)}, {})
    out = inj.fragment_windows(pop, frag)
    assert anchors(pop, ~out.predicted) == list(range(15, 22))
    assert all(r == "split_in_history" for r, ok in zip(out.reason, out.predicted) if not ok)
    np.testing.assert_array_equal(cv_errors(pop, out), pop.clean_per_step[out.predicted])


# --- the sweep on a generated scene ------------------------------------------------------------

SCENE = SceneConfig(seed=4, n_frames=300)


@pytest.fixture(scope="module")
def run():
    return inj.run_injection(CFG, SCENE)


def by(run, condition, level=None, seed=None):
    return [r for r in run.results if r.condition == condition and (level is None or r.level == level)
            and (seed is None or r.seed == seed)]


def test_population_targets_and_clean_tracks_are_protected(run):
    scene = generate(SCENE, DT)
    reference = evaluate_ground_truth_cv(scene, CFG)
    n = len(reference.windows)
    assert n > 0 and run.reference["n_windows"] == n
    assert run.targets_sha256 == inj._digest(reference.windows.target)
    assert run.clean_tracks_sha256 == inj.tracks_digest(ground_truth_grid_tracks(scene))
    assert len(run.results) == sum(len(inj.levels(CFG, c)) for c in inj.CONDITIONS) * 3
    for r in run.results:
        assert len(r.predicted) == n and r.summary["coverage"]["eligible_windows"] == n
    clean = ground_truth_grid_tracks(scene)
    pop = inj.population(clean, reference.windows, reference.per_step_error)
    first = inj.evaluate_condition(clean, pop, "A", 0.2, 1, CFG)
    for c in ("B1", "C", "B2"):
        inj.evaluate_condition(clean, pop, c, 0.3, 1, CFG)
    again = inj.evaluate_condition(clean, pop, "A", 0.2, 1, CFG)       # fresh clean copy each time
    np.testing.assert_array_equal(first.per_step, again.per_step)
    assert inj.tracks_digest(clean) == run.clean_tracks_sha256


def test_level_zero_reproduces_the_clean_baseline(run):
    for c in inj.CONDITIONS:
        for r in by(run, c, 0.0):
            assert r.predicted.all() and r.summary["achieved_primary"]["value"] == 0.0
            np.testing.assert_array_equal(r.per_step, run.population.clean_per_step)
            acc = r.summary["accuracy_on_predicted_windows"]
            assert acc["injected"]["mean_ade_m"] == pytest.approx(run.reference["mean_ade_m"], abs=0)


def test_condition_invariants_and_separate_severities(run):
    for r in run.results:
        s = r.summary
        assert s["requested"]["value"] == r.level and "value" in s["achieved_primary"]
        assert s["coverage"]["predicted_windows"] == len(r.per_step) == int(r.predicted.sum())
        if r.condition == "A":
            assert r.predicted.all()
            if r.level > 0:
                assert s["achieved_primary"]["value"] != r.level             # achieved is measured
        if r.condition == "C":
            np.testing.assert_array_equal(r.per_step, run.population.clean_per_step[r.predicted])
        if r.condition in ("B1", "B2"):
            a = s["achieved"]
            assert a["rate"] == a["removed_observations"] / a["eligible_observations"]
            assert s["achieved_predictor_visible"]["eligible_observations"] <= a["eligible_observations"]
    for level in inj.levels(CFG, "B1"):
        for seed in inj.PROVISIONAL_SEEDS:
            (b1,), (b2,) = by(run, "B1", level, seed), by(run, "B2", level, seed)
            assert b1.summary["achieved"] == b2.summary["achieved"]                # shared masks
            assert b1.summary["prediction_coverage_pre"] == b2.summary["prediction_coverage_post"]
            assert b1.summary["prediction_coverage_post"] >= b1.summary["prediction_coverage_pre"]
            assert np.all(b1.predicted | ~b2.predicted)


def test_save_run_is_reproducible_and_refuses_existing(run, tmp_path):
    out = inj.save_run(tmp_path / "run", run, CFG)
    assert sorted(p.name for p in out.iterdir()) == ["config.resolved.yaml", "manifest.json", "results.json",
                                                     "windows.csv"]
    with pytest.raises(FileExistsError):
        inj.save_run(out, run, CFG)
    reject = lambda token: (_ for _ in ()).throw(ValueError(token))
    results = json.loads((out / "results.json").read_text(encoding="utf-8"), parse_constant=reject)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"), parse_constant=reject)
    assert manifest["experiment_id"] == results["experiment_id"] == run.experiment_id
    assert manifest["injection_seeds"] == [0, 1, 2] and manifest["scene_seed"] == 4
    assert manifest["n_frames"] == 300
    for c in inj.CONDITIONS:
        assert manifest["conditions"][c]["levels"] == inj.levels(CFG, c)
        assert [lv["requested"]["value"] for lv in results["conditions"][c]] == inj.levels(CFG, c)
    assert "provisional" in results["settings"]["injection_seeds_status"]
    for name, digest in manifest["files"].items():
        assert inj.file_sha256(out / name) == digest
    lines = (out / "windows.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + len(run.results) * run.reference["n_windows"]

    second = inj.run_injection(CFG, SCENE)
    assert second.experiment_id == run.experiment_id
    assert inj._json(inj.results_document(second)) == inj._json(inj.results_document(run))
    assert inj.windows_csv(second) == inj.windows_csv(run)
    longer = inj.run_injection(CFG, SceneConfig(seed=4, n_frames=320), seeds=(0,))
    assert longer.experiment_id != run.experiment_id                       # scenes kept separate
