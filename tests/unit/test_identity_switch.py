"""U8: identity-switch injection (D) and window policy P.

Expected values are written out by hand or derived from the stated geometry, never computed with the
helper under test. Grid samples are STEP frames apart; d_max is read from the configuration.

Crossing pair: ped 1 at (k, 0), ped 2 at (9 - k, 1), k = 0..9. Their distance is sqrt((2k - 9)^2 + 1):
sqrt(2) at k = 4 and 5, sqrt(10) at k = 3 and 6, larger elsewhere, so with sqrt(2) <= d_max < sqrt(10)
exactly (1, 4), (1, 5), (2, 4), (2, 5) are eligible: denominator 4.
"""

import json

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments import switch_feasibility as sf
from trackvision.experiments.injection_feasibility import population
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.inject import identity_switch as ids
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
D_MAX = float(CFG.frozen["injection"]["d_max_m"])
LEVELS = [float(v) for v in CFG.frozen["injection"]["grids"]["D"]["levels"]]
P = protocol(CFG)
H, F = P["history_len"], P["horizon_len"]
STEP = 10


def track(pid, xy, start=0):
    xy = np.asarray(xy, dtype=np.float64)
    return GridTrack(pid, (start + np.arange(len(xy), dtype=np.int64)) * STEP, xy)


def line(n, y, start=0):
    """n samples at (k, y) for k = start .. start + n - 1: 1 m per step along x."""
    return np.column_stack([np.arange(start, start + n, dtype=np.float64), np.full(n, float(y))])


def crossing():
    k = np.arange(10.0)
    return (track(1, np.column_stack([k, np.zeros(10)])), track(2, np.column_stack([9 - k, np.ones(10)])))


def frame_positions(tracks):
    return sorted((int(f), float(x), float(y)) for t in tracks for f, (x, y) in zip(t.frames, t.positions))


# --- 1-3, 12: the exchange itself ---------------------------------------------------------------------

def test_genuine_exchange_inclusive_from_k():
    assert np.sqrt(2) <= D_MAX < np.sqrt(10)
    sw = ids.identity_switch(crossing(), 1.0, seed=0, d_max=D_MAX, frame_step=STEP)
    (e,) = sw.events
    assert (e.frame, e.initiator, e.partner) == (40, 1, 2) and e.distance_m == pytest.approx(np.sqrt(2))
    one, two = sw.tracks
    assert (one.ped_id, two.ped_id) == (1, 2)
    np.testing.assert_array_equal(one.positions, [[0, 0], [1, 0], [2, 0], [3, 0], [5, 1], [4, 1], [3, 1], [2, 1],
                                                  [1, 1], [0, 1]])
    np.testing.assert_array_equal(two.positions, [[9, 1], [8, 1], [7, 1], [6, 1], [4, 0], [5, 0], [6, 0], [7, 0],
                                                  [8, 0], [9, 0]])
    np.testing.assert_array_equal(one.frames, np.arange(0, 100, 10))
    np.testing.assert_array_equal(two.frames, np.arange(0, 100, 10))
    assert sw.occupant[1].tolist() == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    assert sw.occupant[2].tolist() == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
    a = sw.achieved
    assert (a["switch_events"], a["eligible_track_frames"], a["switch_event_rate"]) == (1, 4, 25.0)
    # k = 4: ped 1 exposed and triggers; ped 2 is then locked. k = 5: both locked.
    assert (a["triggers"], a["suppressed_triggers"], a["participating_tracks"], a["lock_aware_exposure"]) == \
           (1, 0, 2, 1)


def test_positions_timestamps_counts_and_labels_preserved():
    clean = crossing()
    for tracks in (clean, (track(1, line(12, 0.0)), track(2, line(9, 1.0, start=3), start=3),
                           track(3, line(6, 0.5, start=2), start=2))):
        for p in (0.5, 1.0):
            sw = ids.identity_switch(tracks, p, seed=3, d_max=D_MAX, frame_step=STEP)
            assert frame_positions(sw.tracks) == frame_positions(tracks)           # every (frame, x, y) kept
            assert [t.ped_id for t in sw.tracks] == [t.ped_id for t in tracks]     # no new ids, same count
            assert sum(len(t.frames) for t in sw.tracks) == sum(len(t.frames) for t in tracks)
            for t in sw.tracks:                                                     # no gaps, one row per frame
                assert np.all(np.diff(t.frames) == STEP)
                assert len(sw.occupant[t.ped_id]) == len(t.frames)


# --- 4-7: eligibility, cooldown, denominators ------------------------------------------------------------

def test_d_max_boundary():
    at = (track(1, line(10, 0.0)), track(2, line(10, D_MAX)))
    beyond = (track(1, line(10, 0.0)), track(2, line(10, np.nextafter(D_MAX, np.inf))))
    assert ids.proximity(at, D_MAX, STEP).eligible_track_frames == 2 * 9             # k = 1..9, both tracks
    assert ids.sample_switches(at, 1.0, 0, D_MAX, STEP).achieved["switch_events"] == 1
    far = ids.sample_switches(beyond, 1.0, 0, D_MAX, STEP).achieved
    assert far["eligible_track_frames"] == 0 and far["switch_event_rate"] is None and far["switch_events"] == 0


def test_cooldown_locks_both_participants():
    tracks = (track(1, line(10, 0.0)), track(2, line(10, 1.0)), track(3, line(10, 0.5)))
    s = ids.sample_switches(tracks, 1.0, 0, D_MAX, STEP)
    (e,) = s.events
    assert (e.frame, e.initiator, e.partner, e.distance_m) == (10, 1, 3, 0.5)       # 3 is nearer than 2
    a = s.achieved
    # ped 2 is exposed and triggers at k = 1..9, always with both partners locked: 9 suppressed.
    assert (a["switch_events"], a["eligible_track_frames"], a["triggers"], a["suppressed_triggers"],
            a["lock_aware_exposure"], a["participating_tracks"]) == (1, 27, 10, 9, 10, 2)
    assert a["switch_event_rate"] == pytest.approx(100 / 27)
    with pytest.raises(ValueError, match="one event only"):
        ids.apply_switches(tracks, [ids.SwitchEvent(10, 1, 3, 0.5), ids.SwitchEvent(50, 1, 2, 1.0)], STEP)
    with pytest.raises(ValueError, match="needs samples"):
        ids.apply_switches(tracks, [ids.SwitchEvent(0, 1, 2, 1.0)], STEP)            # no sample at k - 1


def test_disjoint_pairs_switch_at_the_same_timestep():
    tracks = (track(1, line(10, 0.0)), track(2, line(10, 1.0)), track(3, line(10, 50.0)), track(4, line(10, 51.0)))
    s = ids.sample_switches(tracks, 1.0, 0, D_MAX, STEP)
    assert [(e.frame, e.initiator, e.partner) for e in s.events] == [(10, 1, 2), (10, 3, 4)]
    assert s.achieved["eligible_track_frames"] == 36
    assert s.achieved["switch_event_rate"] == pytest.approx(200 / 36)


def test_no_eligible_track_frames():
    tracks = (track(1, line(10, 0.0)), track(2, line(10, 10.0)), track(3, line(10, 20.0)))
    sw = ids.identity_switch(tracks, 1.0, 0, D_MAX, STEP)
    a = sw.achieved
    assert (a["eligible_track_frames"], a["switch_events"], a["triggers"], a["lock_aware_exposure"]) == (0, 0, 0, 0)
    assert a["switch_event_rate"] is None
    for clean, out in zip(tracks, sw.tracks):
        assert np.array_equal(clean.positions, out.positions) and np.array_equal(clean.frames, out.frames)


def test_denominator_is_pre_injection_and_invariant_across_levels():
    for seed in (0, 1, 7):
        dens = {ids.sample_switches(crossing(), p, seed, D_MAX, STEP).achieved["eligible_track_frames"]
                for p in LEVELS + [1.0]}
        assert dens == {4}


# --- 8-11: window policy P -------------------------------------------------------------------------------

def windows_for(tracks):
    win = make_windows("hand", tracks, STEP, H, F, P["window_stride"])
    clean_cv = np.stack([win.observed[:, -1] + j * (win.observed[:, -1] - win.observed[:, -2])
                         for j in range(1, F + 1)], axis=1)                         # hand CV, independent
    return population(tracks, win, np.linalg.norm(clean_cv - win.target, axis=2))


def anchor(pop, w):
    return int(pop.windows.frames[w, 0]) // STEP


def hand_cv_errors(x, target):
    pred = np.array([x[-1] + j * (x[-1] - x[-2]) for j in range(1, F + 1)])
    return np.linalg.norm(pred - target, axis=1)


def test_policy_p_history_target_and_switch_positions():
    n, s = 45, 20                                       # switch at sample 20; anchors a = 0 .. n - H - F
    clean = (track(1, line(n, 0.0)), track(2, line(n, 1.0)))
    pop = windows_for(clean)
    rel = ids.apply_switches(clean, [ids.SwitchEvent(s * STEP, 1, 2, 1.0)], STEP)
    sw = sf.switch_windows(pop, rel.tracks, rel.occupant, P["cv_min_final_observations"])
    n_anchor = n - H - F + 1
    assert len(pop.windows) == 2 * n_anchor and sw.predicted.all()
    inside = set(range(s - H + 1, s))                    # s at t_2 .. t_H of the history a .. a + H - 1
    changed = s - H + 1                                  # s at t_H: CV's last two points straddle the switch
    k = 0
    for w in range(len(pop.windows)):
        g, a = int(pop.windows.track_ids[w]), anchor(pop, w)
        assert bool(sw.mixed[w]) == (a in inside), (g, a)
        assert bool(sw.cv_changed[w]) == (a == changed), (g, a)
        assert int(sw.label[w]) == (g if a + H - 1 < s else 3 - g)
        x = sw.inputs[k]
        if a not in inside:                              # before (a >= s), after (a + H - 1 < s): unchanged
            np.testing.assert_array_equal(x, pop.windows.observed[w])
        if g == 1 and a == changed:                      # hand-written policy-P history for ped 1
            expected = [[a + i, 1.0] for i in range(H - 1)] + [[s, 0.0]]
            np.testing.assert_array_equal(x, expected)
            np.testing.assert_array_equal(pop.windows.target[w], [[s + j, 0.0] for j in range(1, F + 1)])
            err = hand_cv_errors(x, pop.windows.target[w])
            np.testing.assert_allclose(err, np.arange(1, F + 1))                # velocity (1, -1) vs (1, 0)
            assert err.mean() == pytest.approx((F + 1) / 2) and err[-1] == pytest.approx(F)
        k += 1
    assert int(sw.mixed.sum()) == 2 * (H - 1) and int(sw.cv_changed.sum()) == 2


def test_policy_p_end_to_end_accuracy(monkeypatch):
    n, s = 45, 20
    clean = (track(1, line(n, 0.0)), track(2, line(n, 1.0)))
    pop = windows_for(clean)
    event = ids.SwitchEvent(s * STEP, 1, 2, 1.0)

    def hand_event(tracks, p, seed, d_max, frame_step):              # replaces sampling with one known event
        rel = ids.apply_switches(tracks, [event], frame_step)
        return ids.IdentitySwitch(rel.tracks, rel.occupant, (event,), {"switch_event_rate": None})
    monkeypatch.setattr(sf, "identity_switch", hand_event)
    r = sf.evaluate_switch(clean, pop, 0.1, 0, CFG, STEP)
    acc = r.summary["accuracy_on_predicted_windows"]
    n_win = 2 * (n - H - F + 1)
    assert r.summary["coverage"]["prediction_coverage"] == 1.0
    assert acc["injected"]["mean_ade_m"] == pytest.approx(2 * (F + 1) / 2 / n_win)
    assert acc["injected"]["mean_fde_m"] == pytest.approx(2 * F / n_win)
    assert acc["clean_same_windows"]["mean_ade_m"] == pytest.approx(0.0, abs=1e-12)
    assert r.summary["history"]["windows_with_mixed_history"] == 2 * (H - 1)
    assert r.summary["history"]["windows_with_changed_cv_input"] == 2


def test_policy_p_missing_label_history():
    n, s, start = 45, 20, 15                            # ped 2 exists from sample 15 only
    assert s - H + 1 < start <= s - 1
    clean = (track(1, line(n, 0.0)), track(2, line(n - start, 1.0, start=start), start=start))
    pop = windows_for(clean)
    rel = ids.apply_switches(clean, [ids.SwitchEvent(s * STEP, 1, 2, 1.0)], STEP)
    sw = sf.switch_windows(pop, rel.tracks, rel.occupant, P["cv_min_final_observations"])
    lost = {(int(pop.windows.track_ids[w]), anchor(pop, w)) for w in np.flatnonzero(~sw.predicted)}
    # ped 1's windows with t_H >= s use label 2, which has no sample before `start`.
    assert lost == {(1, a) for a in range(s - H + 1, start)}
    assert {sw.reason[w] for w in np.flatnonzero(~sw.predicted)} == {sf.LOST_REASON}
    assert {r for r in sw.reason if r} == {"label_history_incomplete"}              # never a C-style split reason


def test_no_target_leakage():
    n, s = 45, 30
    clean = (track(1, line(n, 0.0)), track(2, line(n, 1.0)))
    pop = windows_for(clean)
    targets = pop.windows.target.copy()
    rel = ids.apply_switches(clean, [ids.SwitchEvent(s * STEP, 1, 2, 1.0)], STEP)
    sw = sf.switch_windows(pop, rel.tracks, rel.occupant, P["cv_min_final_observations"])
    k = np.cumsum(sw.predicted) - 1
    for w in range(len(pop.windows)):
        if anchor(pop, w) + H - 1 < s:                  # the switch lies in this window's target period
            np.testing.assert_array_equal(sw.inputs[k[w]], pop.windows.observed[w])
            assert not sw.mixed[w] and not sw.cv_changed[w]
    np.testing.assert_array_equal(pop.windows.target, targets)


# --- 14-15: randomness ---------------------------------------------------------------------------------

def test_shared_draws_and_reproducible_events():
    tracks = crossing()
    for seed in range(25):
        draws = {pid: np.random.default_rng(np.random.SeedSequence([seed, 4, pid])).random(10) for pid in (1, 2)}
        got = ids.switch_draws(tracks, seed)
        assert all(np.array_equal(got[pid], draws[pid]) for pid in (1, 2))
        order = [(40, 1, 2), (40, 2, 1), (50, 1, 2), (50, 2, 1)]                   # timestep, then ped id
        for p in LEVELS:
            first = next(((f, i, j) for f, i, j in order if draws[i][f // STEP] < p), None)
            events = ids.sample_switches(tracks, p, seed, D_MAX, STEP).events
            assert [(e.frame, e.initiator, e.partner) for e in events] == ([first] if first else [])
            assert events == ids.sample_switches(tracks, p, seed, D_MAX, STEP).events
    far = tracks + (track(9, line(10, 100.0)),)
    assert np.array_equal(ids.switch_draws(far, 5)[1], ids.switch_draws(tracks, 5)[1])


def test_level_zero_is_bit_identical():
    tracks = (track(1, line(10, 0.0)), track(2, line(10, 1.0)), track(3, line(10, 0.5)))
    sw = ids.identity_switch(tracks, 0.0, 11, D_MAX, STEP)
    assert sw.events == () and sw.achieved["switch_event_rate"] == 0.0
    for clean, out in zip(tracks, sw.tracks):
        assert out.positions.dtype == clean.positions.dtype and np.array_equal(out.positions, clean.positions)
        assert np.array_equal(out.frames, clean.frames) and np.all(sw.occupant[out.ped_id] == out.ped_id)


# --- 16-17: naming, scene sweep, saved runs --------------------------------------------------------------

def keys(d):
    if isinstance(d, dict):
        for k, v in d.items():
            yield str(k)
            yield from keys(v)
    elif isinstance(d, list):
        for v in d:
            yield from keys(v)


@pytest.fixture(scope="module")
def scene_run():
    return sf.run_switch(CFG, SceneConfig(seed=21, n_frames=300), seeds=(0, 1))


def test_scene_sweep_integrity(scene_run):
    run = scene_run
    assert run.eligible_track_frames > 0
    assert {r.summary["achieved"]["eligible_track_frames"] for r in run.results} == {run.eligible_track_frames}
    assert [r.level for r in run.results] == [lv for lv in LEVELS for _ in (0, 1)]
    assert any(r.summary["achieved"]["switch_events"] for r in run.results)
    scene = generate(SceneConfig(seed=21, n_frames=300), P["sampling_interval_s"])
    clean = {t.ped_id: t for t in ground_truth_grid_tracks(scene)}
    for r in run.results:
        seen = set()
        for e in r.events:
            a, b = clean[e.initiator], clean[e.partner]
            for t in (a, b):
                assert e.frame in t.frames and e.frame - scene.frames_per_step in t.frames
            pa, pb = a.positions[a.frames == e.frame][0], b.positions[b.frames == e.frame][0]
            assert np.hypot(*(pa - pb)) <= D_MAX
            assert not {e.initiator, e.partner} & seen
            seen |= {e.initiator, e.partner}
        s = r.summary
        assert s["achieved"]["participating_tracks"] == 2 * len(r.events) == 2 * s["achieved"]["switch_events"]
        if r.level == 0.0:
            assert s["achieved"]["switch_events"] == 0 and s["coverage"]["prediction_coverage"] == 1.0
            assert s["history"]["windows_with_mixed_history"] == 0
            assert s["accuracy_on_predicted_windows"]["injected"]["mean_ade_m"] == run.reference["mean_ade_m"]
    ref = ids.identity_switch(tuple(clean.values()), 0.2, 1, D_MAX, scene.frames_per_step)
    assert frame_positions(ref.tracks) == frame_positions(tuple(clean.values()))


def test_naming_and_saved_run(scene_run, tmp_path):
    doc = sf.results_document(scene_run)
    assert not [k for k in keys(doc) if "idsw" in k.lower()]
    assert all(s["achieved_primary"]["name"] == "switch_event_rate" for lv in doc["levels"] for s in lv["per_seed"])
    out = sf.save_run(tmp_path / "a", scene_run, CFG)
    header = (out / "windows.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "idsw" not in header.lower() and "mixed_history" in header and "cv_input_changed" in header
    again = sf.save_run(tmp_path / "b", sf.run_switch(CFG, SceneConfig(seed=21, n_frames=300), seeds=(0, 1)), CFG)
    for name in ("results.json", "windows.csv", "config.resolved.yaml"):
        assert (out / name).read_bytes() == (again / name).read_bytes()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["d_max_m"] == D_MAX and manifest["d_cooldown"] == sf.PRIMARY_COOLDOWN
    with pytest.raises(FileExistsError):
        sf.save_run(out, scene_run, CFG)
    inside = config.REPO_ROOT / "u8-output-must-not-be-created"
    with pytest.raises(ValueError, match="inside the repository"):
        sf.save_run(inside, scene_run, CFG)
    assert not inside.exists()


def test_only_primary_cooldown_is_implemented():
    from types import SimpleNamespace
    data = CFG.canonical_dict()
    data["frozen"]["injection"]["d_cooldown"] = "some_other_rule"
    with pytest.raises(ValueError, match="only the primary cooldown"):
        sf.check_cooldown(SimpleNamespace(frozen=data["frozen"]))
    assert sf.check_cooldown(CFG) == "locked_until_end_of_segment"
