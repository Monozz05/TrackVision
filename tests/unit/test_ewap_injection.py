"""R2: A, B1, B2, C and D injection on EWAP-format scenes, on small synthetic EWAP files written to a
temporary directory. No test reads the real dataset files.

Walker scene (frame step 6, documented 0.4 s): ped 1 at (0.3 k, 1.0) and ped 2 at (0.3 k, 2.0) for
k = 0..29 (11 windows each, 1 m apart, so D-eligible), ped 3 a 5-sample track on another frame phase
(no window). Expected values are derived by hand from this geometry; CV is re-implemented by hand
where accuracy is checked.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fixtures import synthetic_ewap as se
from trackvision import config
from trackvision.experiments import ewap_baseline as eb
from trackvision.experiments import ewap_injection as ei
from trackvision.experiments import injection_feasibility as u7
from trackvision.experiments import switch_feasibility as u8
from trackvision.experiments.geometry_validation import LOCAL_NOTICE
from trackvision.inject.identity_switch import IdentitySwitch, SwitchEvent, apply_switches, identity_switch
from trackvision.inject.trajectory import Dropout, fragment

CFG = config.validate().config
H = CFG.frozen["protocol"]["static"]["history_len"]
F = CFG.frozen["protocol"]["static"]["horizon_len"]
D_MAX = float(CFG.frozen["injection"]["d_max_m"])
STEP = se.FRAME_STEP
N = 30
FIRST = 780


def walkers(shift_after=None):
    k = np.arange(float(N))
    y1 = np.full(N, 1.0)
    if shift_after is not None:
        y1[shift_after:] += 0.05                            # changes ped 1 only from sample `shift_after` on
    ped3 = np.column_stack([0.3 * np.arange(5.0), np.full(5, 3.5)])
    return (se.track_rows(1, FIRST, np.column_stack([0.3 * k, y1]))
            + se.track_rows(2, FIRST, np.column_stack([0.3 * k, np.full(N, 2.0)]))
            + se.track_rows(3, FIRST + 3, ped3))


def curved():
    k = np.arange(float(N))
    return (se.track_rows(1, FIRST, np.column_stack([0.3 * k, np.full(N, 1.0)]))
            + se.track_rows(2, FIRST, np.column_stack([0.3 * k, 1.5 + 0.004 * k ** 2])))


def write(tmp_path, name="ETH", rows=None):
    raw = tmp_path / "raw"
    se.write_scene(raw, name=name, rows=walkers() if rows is None else rows)
    manifest_path = tmp_path / "ewap.sha256"
    se.write_manifest(raw, manifest_path)
    return raw, manifest_path


def run(tmp_path, name="ETH", rows=None, cfg=CFG, **kwargs):
    raw, man = write(tmp_path, name, rows)
    return ei.run_ewap_injection(cfg, name, raw, man, **kwargs)


def tree_digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def predicted_inputs(result):
    """(predicted flags, {window index: CV input}) for a U7 or a U8 result."""
    flags, inputs = (result.windows.predicted, result.windows.inputs) if hasattr(result, "windows") else \
        (result.predicted, result.inputs)
    return flags, {int(w): inputs[k] for k, w in enumerate(np.flatnonzero(flags))}


def hand_cv_ade(observed, target):
    pred = np.stack([observed[:, -1] + j * (observed[:, -1] - observed[:, -2]) for j in range(1, F + 1)], axis=1)
    err = np.linalg.norm(pred - target, axis=2)
    return err.mean(axis=1), err[:, -1], err


def test_both_scenes_use_the_pipeline_and_stay_separate(tmp_path):
    eth = ei.results_document(run(tmp_path / "eth", "ETH", seeds=(0,)))
    hotel_run = run(tmp_path / "hotel", "HOTEL", seeds=(0,))
    hotel = ei.results_document(hotel_run)
    for doc, scene in ((eth, "ETH"), (hotel, "HOTEL")):
        assert doc["scene"] == scene and list(doc["conditions"]) == ["A", "B1", "B2", "C", "D"]
        d = doc["dataset"]
        assert (d["tracks"], d["observations"], d["eligible_windows"], d["pedestrians_contributing"]) == \
               (3, 2 * N + 5, 2 * (N - (H + F) + 1), 2)
        assert doc["timing"]["note"] == eb.TIMING_NOTES[scene] and doc["settings"]["timing_note"] == eb.TIMING_NOTES[scene]
        assert doc["evaluation_type"] == "trajectory_only" and doc["notice"] == LOCAL_NOTICE
        assert "not a real-tracker result" in doc["label"] and "never pooled" in doc["label"]
        for c, levels in doc["conditions"].items():
            assert [lv["requested"]["value"] for lv in levels] == u7.levels(CFG, c)
    assert eth["experiment_id"] != hotel["experiment_id"] and "UNRESOLVED" in eth["timing"]["note"]
    again = ei.results_document(run(tmp_path / "hotel_alone", "HOTEL", seeds=(0,)))
    assert again["conditions"] == hotel["conditions"]                       # unaffected by the ETH run


def test_source_files_unchanged_and_saved_run_reproducible(tmp_path):
    raw, man = write(tmp_path)
    before = tree_digest(raw)
    r = ei.run_ewap_injection(CFG, "ETH", raw, man, seeds=(0, 1))
    out = ei.save_run(tmp_path / "o1", r, CFG)
    again = ei.save_run(tmp_path / "o2", ei.run_ewap_injection(CFG, "ETH", raw, man, seeds=(0, 1)), CFG)
    for name in ("results.json", "windows_abc.csv", "windows_d.csv", "config.resolved.yaml"):
        assert (out / name).read_bytes() == (again / name).read_bytes()
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["timing_note"] == eb.TIMING_NOTES["ETH"] and m["notice"] == LOCAL_NOTICE and m["manifest_verified"]
    assert m["source_sha256"]["ewap_dataset/seq_eth/obsmat.txt"] == before["ewap_dataset/seq_eth/obsmat.txt"]
    for header in ((out / "windows_abc.csv"), (out / "windows_d.csv")):
        assert "idsw" not in header.read_text(encoding="utf-8").splitlines()[0].lower()
    with pytest.raises(FileExistsError):
        ei.save_run(out, r, CFG)
    inside = config.REPO_ROOT / "r2-output-must-not-be-created"
    with pytest.raises(ValueError, match="inside the repository"):
        ei.save_run(inside, r, CFG)
    assert not inside.exists() and tree_digest(raw) == before


def test_fresh_copy_for_every_condition(tmp_path):
    full = run(tmp_path / "full", seeds=(0, 1))
    for c in ("C", "D", "A"):
        alone = run(tmp_path / c, conditions=(c,), seeds=(0, 1))
        assert [r.summary for r in alone.results] == [r.summary for r in full.results if r.summary["condition"] == c]
    assert all(not t.positions.flags.writeable for t in full.baseline.scene.tracks)


def test_targets_are_clean_annotations(tmp_path):
    raw, man = write(tmp_path)
    r = ei.run_ewap_injection(CFG, "ETH", raw, man, seeds=(0,))
    obsmat = np.loadtxt(raw / "ewap_dataset" / "seq_eth" / "obsmat.txt")       # read independently
    lookup = {(int(row[1]), int(row[0])): (row[2], row[4]) for row in obsmat}
    win = r.population.windows
    assert not win.target.flags.writeable
    for w in range(len(win)):
        expected = [lookup[(int(win.track_ids[w]), int(f))] for f in win.frames[w, H:]]
        np.testing.assert_array_equal(win.target[w], expected)


def test_window_protocol_comes_from_config(tmp_path):
    data = CFG.canonical_dict()
    data["frozen"]["protocol"]["static"]["horizon_len"] = 8
    other = SimpleNamespace(frozen=data["frozen"], hash="0" * 64, meta=data["meta"])
    r = run(tmp_path, cfg=other, conditions=("A", "D"), seeds=(0,))
    assert r.settings["protocol"]["horizon_len"] == 8 and r.population.windows.target.shape[1] == 8
    assert len(r.population.windows) == 2 * (N - (H + 8) + 1)
    assert all(len(res.per_step[0]) == 8 for res in r.results if len(res.per_step))


def test_missing_observations_b1_and_b2(tmp_path, monkeypatch):
    def hand_mask(tracks, p, seed):                    # ped 1 loses samples 4 and 9; nothing else
        present = tuple(np.ones(len(t.frames), dtype=bool) for t in tracks)
        present[0][[4, 9]] = False
        return Dropout(present, {"removed_observations": 2, "eligible_observations": 2 * N + 5, "rate": 2 / 65})
    monkeypatch.setattr(u7, "dropout_masks", hand_mask)
    r = run(tmp_path, conditions=("B1", "B2"), seeds=(0,))
    n_win = N - (H + F) + 1
    # B2: every ped-1 history containing sample 4 or 9 (anchors 0..9) is lost.
    # B1: only a missing first or last history point is fatal: anchors 4 and 9 (first), 2 (last = 9).
    lost_b2, lost_b1 = set(range(0, 10)), {2, 4, 9}
    for res in r.results:
        s = res.summary
        lost = {int(r.population.windows.frames[w, 0] - FIRST) // STEP
                for w in np.flatnonzero(~res.predicted) if r.population.windows.track_ids[w] == 1}
        assert not any(not res.predicted[w] for w in range(len(res.predicted)) if r.population.windows.track_ids[w] == 2)
        if s["condition"] == "B2":
            assert lost == lost_b2 and s["coverage"]["lost_by_reason"]["missing_observation"] == 10
        else:
            assert lost == lost_b1 and s["coverage"]["lost_by_reason"]["boundary_gap"] == 3
            assert s["reconstruction"]["distinct_observations_reconstructed"] == 2
            _, inputs = predicted_inputs(res)                                   # straight line: interpolation exact
            for w, x in inputs.items():
                np.testing.assert_allclose(x, r.population.windows.observed[w], atol=1e-9)
        # U7 convention: _pre applies the no-reconstruction rule; _post the condition's own rule (B2: none).
        assert s["prediction_coverage_pre"] == pytest.approx((2 * n_win - 10) / (2 * n_win))
        expected_post = (2 * n_win - (10 if s["condition"] == "B2" else 3)) / (2 * n_win)
        assert s["prediction_coverage_post"] == pytest.approx(expected_post) == s["coverage"]["prediction_coverage"]


def test_fragmentation_and_switch_are_distinct(tmp_path):
    raw, man = write(tmp_path)
    clean = eb.run_baseline(CFG, "ETH", raw, man).scene.tracks
    frag = fragment(clean, 1.0, 0)
    assert len(frag.tracks) == 3 + (N - 2) * 2 + (5 - 2)                     # every eligible frame split
    assert {t.ped_id for t in frag.tracks} > {1, 2, 3}                       # new identities
    sw = identity_switch(clean, 1.0, 0, D_MAX, STEP)
    assert [(e.frame, e.initiator, e.partner) for e in sw.events] == [(FIRST + STEP, 1, 2)]   # first eligible k
    assert [t.ped_id for t in sw.tracks] == [1, 2, 3]
    assert sorted((int(f), *map(float, p)) for t in sw.tracks for f, p in zip(t.frames, t.positions)) == \
           sorted((int(f), *map(float, p)) for t in clean for f, p in zip(t.frames, t.positions))
    assert all(np.all(np.diff(t.frames) == STEP) for t in sw.tracks)          # no gaps
    r = run(tmp_path / "run", conditions=("C", "D"), seeds=(0, 1))
    reasons = {c: {k for res in r.results if res.summary["condition"] == c
                   for k, v in res.summary["coverage"]["lost_by_reason"].items() if v} for c in ("C", "D")}
    assert reasons["C"] <= {"split_in_history"} and reasons["D"] <= {u8.LOST_REASON}
    names = {res.summary["condition"]: res.summary["achieved_primary"]["name"] for res in r.results}
    assert names == {"C": "fragmentation_rate", "D": "switch_event_rate"}


def test_d_follows_policy_p(tmp_path, monkeypatch):
    s_k = 15
    event = SwitchEvent(FIRST + STEP * s_k, 1, 2, 1.0)

    def hand_event(tracks, p, seed, d_max, frame_step):
        rel = apply_switches(tracks, [event], frame_step)
        return IdentitySwitch(rel.tracks, rel.occupant, (event,),
                              {"switch_event_rate": None, "eligible_track_frames": 0, "switch_events": 1})
    monkeypatch.setattr(u8, "identity_switch", hand_event)
    r = run(tmp_path, conditions=("D",), seeds=(0,))
    win = r.population.windows
    n_win = 2 * (N - (H + F) + 1)
    for res in r.results:
        sw = res.windows
        anchors = (win.frames[:, 0] - FIRST) // STEP
        assert sw.predicted.all()
        mixed = set(range(s_k - H + 1, s_k)) & set(range(N - (H + F) + 1))    # switch at t_2..t_H, existing anchors
        assert set(anchors[sw.mixed]) == mixed and int(sw.mixed.sum()) == 2 * len(mixed)
        assert set(anchors[sw.cv_changed]) == {s_k - H + 1} and int(sw.cv_changed.sum()) == 2
        # ped 1's window with t_H = s_k: last two inputs are ped 2 at s_k - 1, then ped 1 at s_k
        w = int(np.flatnonzero((win.track_ids == 1) & (anchors == s_k - H + 1))[0])
        x = sw.inputs[w]
        np.testing.assert_allclose(x[-2:], [[0.3 * (s_k - 1), 2.0], [0.3 * s_k, 1.0]])
        ade, fde, err = hand_cv_ade(x[None], win.target[w][None])
        np.testing.assert_allclose(err[0], np.arange(1, F + 1), atol=1e-8)          # 1 m lateral error per step
        acc = res.summary["accuracy_on_predicted_windows"]
        assert acc["injected"]["mean_ade_m"] == pytest.approx(2 * (F + 1) / 2 / n_win, abs=1e-8)
        assert acc["injected"]["mean_fde_m"] == pytest.approx(2 * F / n_win, abs=1e-8)
        assert acc["clean_same_windows"]["mean_ade_m"] == pytest.approx(0.0, abs=1e-8)


def test_no_leakage_from_future_samples(tmp_path):
    s_k = 15                                           # the variant differs from sample 15 on (ped 1 only)
    base = run(tmp_path / "base", seeds=(0, 1))
    variant = run(tmp_path / "variant", rows=walkers(shift_after=s_k), seeds=(0, 1))
    win = base.population.windows
    early = np.flatnonzero(win.frames[:, H - 1] < FIRST + STEP * s_k)        # histories end before the change
    assert len(early) > 0
    for a, b in zip(base.results, variant.results):
        assert (a.summary["condition"], a.level, a.seed) == (b.summary["condition"], b.level, b.seed)
        fa, ia = predicted_inputs(a)
        fb, ib = predicted_inputs(b)
        for w in early:
            assert fa[w] == fb[w]
            if fa[w]:
                np.testing.assert_array_equal(ia[w], ib[w])


def test_coverage_and_paired_accuracy(tmp_path):
    r = run(tmp_path, rows=curved(), seeds=(0, 1, 2))
    win = r.population.windows
    clean_ade, clean_fde, _ = hand_cv_ade(win.observed, win.target)
    assert clean_ade.max() > 0.01                                              # the curve makes errors non-zero
    for res in r.results:
        flags, inputs = predicted_inputs(res)
        idx = np.flatnonzero(flags)
        s = res.summary
        assert s["coverage"]["eligible_windows"] == len(win)
        assert s["coverage"]["predicted_windows"] == len(idx)
        assert s["coverage"]["prediction_coverage"] == pytest.approx(len(idx) / len(win))
        acc = s["accuracy_on_predicted_windows"]
        if not len(idx):
            assert acc["injected"]["mean_ade_m"] is None
            continue
        assert acc["clean_same_windows"]["mean_ade_m"] == pytest.approx(clean_ade[idx].mean())
        assert acc["clean_same_windows"]["mean_fde_m"] == pytest.approx(clean_fde[idx].mean())
        inj_ade, inj_fde, inj_err = hand_cv_ade(np.stack([inputs[w] for w in idx]), win.target[idx])
        assert acc["injected"]["mean_ade_m"] == pytest.approx(inj_ade.mean())
        assert acc["injected"]["mean_fde_m"] == pytest.approx(inj_fde.mean())
        np.testing.assert_allclose(acc["injected"]["mean_error_by_step_m"], inj_err.mean(axis=0))
        assert acc["mean_ade_difference_injected_minus_clean_m"] == pytest.approx(inj_ade.mean() - clean_ade[idx].mean())
