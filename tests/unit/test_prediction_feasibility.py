"""U6: CV prediction on tracker trajectories, prediction coverage, paired L0/L1/L2 accuracy, saved runs.

Coverage and accuracy cases use hand-built labelled tracker samples with hand-derived expected
counts and errors. The end-to-end test runs a small rendered scene whose seed is declared here and
is never used to choose parameters; it asserts structural properties only. Saved runs go to
pytest's temporary directories only.
"""

import json

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments import feasibility_run as fr
from trackvision.experiments import prediction_feasibility as pf
from trackvision.experiments.detection_feasibility import ESTIMATED, ORACLE
from trackvision.experiments.tracking_feasibility import GT_BOXES, bottom_centre_world
from trackvision.metrics.detection import clip_boxes, iou_matrix
from trackvision.metrics.displacement import displacement_errors
from trackvision.predict.baselines import constant_velocity
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack
from trackvision.tracking.tracker import TrackerResult, TrackOutput

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, STRIDE = STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"]
DT = STATIC["sampling_interval_s"]
EVAL_IOU = CFG.frozen["association"]["tracker_gt_assoc_iou_threshold"]
S = 10                                    # frames per grid step at 25 fps


def gt_track(pid, positions):
    positions = np.asarray(positions, dtype=np.float64)
    return GridTrack(pid, np.arange(len(positions), dtype=np.int64) * S, positions)


def straight(n, y=0.0, step=0.5):
    return [(step * k, y) for k in range(n)]


def labelled(track_id, sample_idx, positions, labels):
    sample_idx = np.asarray(sample_idx)
    return pf.LabelledTrack(track_id, sample_idx.astype(np.int64) * S, np.asarray(positions, dtype=np.float64),
                            np.asarray(labels, dtype=np.int64))


def follow(track_id, gt, sample_idx, label=None, offset=(0.0, 0.0)):
    """A tracker track observing `gt` at the given sample indices."""
    sample_idx = list(sample_idx)
    return labelled(track_id, sample_idx, gt.positions[sample_idx] + np.asarray(offset),
                    [gt.ped_id if label is None else label] * len(sample_idx))


def evaluate(gts, tracks, causes=None, bias=None):
    assoc = pf.GridAssociation(tuple(tracks), causes or {})
    return pf.evaluate_prediction(gts, bias if bias is not None else gts, assoc, S, CFG, "hand-built")


def cov(ev):
    return ev.summary["coverage"]


# --- coverage definitions ----------------------------------------------------------------------

def test_perfect_tracking_full_coverage_and_available_histories_are_not_the_numerator():
    gt = gt_track(1, straight(25))                       # 25 samples -> 25 - 19 = 6 eligible windows
    ev = evaluate([gt], [follow(7, gt, range(25))])
    c = cov(ev)
    assert c["eligible_ground_truth_windows"] == 6
    assert c["available_tracker_histories"] == 25 - H + 1 == 18
    assert c["predicted_windows"] == 6 and c["prediction_coverage"] == 1.0
    assert c["scorable_tracker_histories"] == 6
    assert c["unscorable_tracker_histories"]["total"] == 12
    assert c["unscorable_tracker_histories"][pf.GT_FUTURE_INCOMPLETE] == 12
    assert set(ev.tracker_id.tolist()) == {7}
    acc = ev.summary["accuracy_on_predicted_windows"]
    for level in ("L0", "L1", "L2"):
        assert acc[level]["mean_ade_m"] == pytest.approx(0.0, abs=1e-12)


def test_missing_history_frame_loses_exactly_the_windows_containing_it():
    gt = gt_track(1, straight(30))                       # eligible anchors at samples 7 .. 17 (11 windows)
    ev = evaluate([gt], [follow(1, gt, [k for k in range(30) if k != 10])], {(1, 100): pf.NO_DETECTION})
    c = cov(ev)
    assert c["eligible_ground_truth_windows"] == 11
    assert c["lost_windows"] == 8                        # anchors 10 .. 17 contain sample 10 in their history
    assert c["predicted_windows"] == 3
    assert c["prediction_coverage"] == pytest.approx(3 / 11)
    assert c["lost_by_primary_reason"][pf.NO_DETECTION] == 8
    anchors = ev.windows.frames[~ev.predicted, H - 1] // S
    assert anchors.tolist() == list(range(10, 18))


def test_missing_future_frame_loses_nothing_unlike_u5_twenty_point_windows():
    gt = gt_track(1, straight(30))
    track = follow(1, gt, [k for k in range(30) if k != 20])      # sample 20 lies only in futures
    ev = evaluate([gt], [track], {(1, 200): pf.NO_DETECTION})
    assert cov(ev)["predicted_windows"] == 11 and cov(ev)["prediction_coverage"] == 1.0
    # U5 counted windows of history + horizon tracker observations: only one survives here.
    u5_style = make_windows("u5", [GridTrack(1, track.frames, track.positions)], S, H, F, STRIDE)
    assert len(u5_style) == 1


def test_identity_change_inside_history_loses_seven_windows():
    gt = gt_track(1, straight(40))                       # eligible anchors 7 .. 27 (21 windows)
    ev = evaluate([gt], [follow(1, gt, range(15)), follow(2, gt, range(15, 40))])
    c = cov(ev)
    assert c["eligible_ground_truth_windows"] == 21
    assert c["lost_windows"] == 7                        # anchors 15 .. 21 span both tracker IDs
    assert c["lost_by_primary_reason"][pf.IDENTITY_CHANGE] == 7
    assert c["predicted_windows"] == 14
    per_track = {t["track_id"]: t for t in ev.summary["per_tracker_track"]}
    assert per_track[1]["predicted_windows"] == 8 and per_track[2]["predicted_windows"] == 6
    assert per_track[2]["unscorable_histories"][pf.GT_FUTURE_INCOMPLETE] == 12


def test_identity_swap_scores_against_the_pedestrians_own_future_not_the_nearest():
    a_pos = [(float(k), 0.0) for k in range(15)] + [(14.0, float(k - 14)) for k in range(15, 30)]   # A turns
    b_pos = [(float(k), 10.0) for k in range(15)] + [(float(k), 0.0) for k in range(15, 30)]        # trap
    a, b = gt_track(1, a_pos), gt_track(2, b_pos)
    t5 = labelled(5, range(30), np.vstack([a.positions[:15], b.positions[15:]]), [1] * 15 + [2] * 15)
    t6 = labelled(6, range(30), np.vstack([b.positions[:15], a.positions[15:]]), [2] * 15 + [1] * 15)
    ev = evaluate([a, b], [t5, t6])
    c = cov(ev)
    assert c["unscorable_tracker_histories"][pf.IDENTITY_MIXED] == 14   # anchors 15 .. 21 on each track
    first = np.flatnonzero((ev.windows.track_ids == 1) & (ev.windows.frames[:, H - 1] == 70))[0]
    assert ev.predicted[first] and ev.tracker_id[first] == 5
    k = int(np.flatnonzero(ev.predicted_index == first)[0])
    # Prediction continues along y = 0 (exactly B's future); A turned, so errors grow after sample 14.
    expected = [0.0] * 7 + [np.sqrt(2) * d for d in range(1, 6)]
    np.testing.assert_allclose(ev.per_step["L2"][k], expected, atol=1e-12)
    np.testing.assert_array_equal(ev.windows.target[first], a.positions[8:20])
    lost_a = (ev.windows.track_ids == 1) & ~ev.predicted
    assert all(ev.primary_reason[w] == pf.IDENTITY_CHANGE for w in np.flatnonzero(lost_a))


def test_false_positive_histories_are_counted_but_never_scored():
    gt = gt_track(1, straight(25))
    ghost = labelled(9, range(3, 13), [(50.0, 50.0)] * 10, [0] * 10)
    ev = evaluate([gt], [follow(1, gt, range(25)), ghost])
    c = cov(ev)
    assert c["unscorable_tracker_histories"][pf.UNASSOCIATED_OBSERVATION] == 3
    assert c["eligible_ground_truth_windows"] == 6 and c["predicted_windows"] == 6
    assert c["available_tracker_histories"] == 18 + 3


def test_lost_reason_priority_and_non_exclusive_flags():
    gt = gt_track(1, straight(30))
    track = follow(1, gt, [k for k in range(30) if k not in (10, 11)])
    ev = evaluate([gt], [track], {(1, 100): pf.NO_DETECTION, (1, 110): pf.INVISIBLE})
    c = cov(ev)
    # Anchor 10 contains only sample 10; anchors 11 .. 17 contain both, and invisible has priority.
    assert c["lost_by_primary_reason"][pf.NO_DETECTION] == 1
    assert c["lost_by_primary_reason"][pf.INVISIBLE] == 7
    assert c["lost_windows_with_cause"][pf.NO_DETECTION] == 8
    assert c["lost_windows_with_cause"][pf.INVISIBLE] == 7
    assert sum(c["lost_by_primary_reason"].values()) == c["lost_windows"] == 8


def test_inputs_are_validated():
    gt = gt_track(1, straight(25))
    with pytest.raises(ValueError, match="no cause recorded"):
        evaluate([gt], [follow(1, gt, [k for k in range(25) if k != 3])])
    with pytest.raises(ValueError, match="claimed twice"):
        evaluate([gt], [follow(1, gt, range(25)), follow(2, gt, [4])])


def test_undefined_summaries():
    ev = evaluate([gt_track(1, straight(10))], [])                   # too short: nothing eligible
    assert cov(ev)["eligible_ground_truth_windows"] == 0 and cov(ev)["prediction_coverage"] is None
    assert ev.summary["accuracy_on_predicted_windows"]["L2"]["mean_ade_m"] is None
    gt = gt_track(1, straight(25))
    ev = evaluate([gt], [], {(1, f): pf.INVISIBLE for f in gt.frames})
    assert cov(ev)["prediction_coverage"] == 0.0
    assert ev.summary["accuracy_on_predicted_windows"]["L2"]["mean_ade_m"] is None
    assert ev.summary["survivorship_L0"]["all_eligible"]["n_windows"] == 6
    assert ev.summary["survivorship_L0"]["mean_ade_difference_predicted_minus_all_m"] is None


# --- paired accuracy levels --------------------------------------------------------------------

def test_constant_offsets_give_exact_level_errors():
    gt = gt_track(1, straight(25))
    bias = [gt_track(1, gt.positions + np.array([0.0, -0.2]))]
    ev = evaluate([gt], [follow(1, gt, range(25), offset=(0.3, 0.0))], bias=bias)
    acc = ev.summary["accuracy_on_predicted_windows"]
    assert acc["L0"]["mean_ade_m"] == pytest.approx(0.0, abs=1e-12)
    assert acc["L1"]["mean_ade_m"] == pytest.approx(0.2) and acc["L1"]["mean_fde_m"] == pytest.approx(0.2)
    assert acc["L2"]["mean_ade_m"] == pytest.approx(0.3) and acc["L2"]["mean_fde_m"] == pytest.approx(0.3)
    hist = acc["history_input_error_m"]
    assert hist["mean_over_history"]["mean"] == pytest.approx(0.3)
    assert hist["at_last_observation"]["max"] == pytest.approx(0.3)


def test_levels_use_the_unchanged_cv_predictor_and_the_same_targets():
    gt = gt_track(1, [(0.1 * k * k, 0.5 * k) for k in range(30)])   # curved: CV errors are non-zero
    ev = evaluate([gt], [follow(1, gt, range(30), offset=(0.05, -0.1))])
    idx = ev.predicted_index
    for level in ("L0", "L1", "L2"):
        np.testing.assert_array_equal(ev.predictions[level], constant_velocity(ev.inputs[level], F, 2))
        np.testing.assert_array_equal(ev.per_step[level],
                                      displacement_errors(ev.predictions[level], ev.windows.target[idx]))
    surv = ev.summary["survivorship_L0"]
    assert surv["all_eligible"]["mean_ade_m"] == pytest.approx(surv["predicted_only"]["mean_ade_m"])
    assert surv["all_eligible"]["mean_ade_m"] > 0


def test_survivorship_compares_all_eligible_with_predicted_only():
    gt = gt_track(1, [(0.5 * k, 0.0) for k in range(20)] + [(9.5, 0.5 * (k - 19)) for k in range(20, 40)])
    keep = [k for k in range(40) if k != 15]           # loses anchors 15 .. 22
    ev = evaluate([gt], [follow(1, gt, keep)], {(1, 150): pf.NO_DETECTION})
    surv = ev.summary["survivorship_L0"]
    l0_all = ev.l0_all_per_step.mean(axis=1)
    assert surv["all_eligible"]["mean_ade_m"] == pytest.approx(l0_all.mean())
    assert surv["predicted_only"]["mean_ade_m"] == pytest.approx(l0_all[ev.predicted].mean())
    assert surv["mean_ade_difference_predicted_minus_all_m"] == pytest.approx(
        l0_all[ev.predicted].mean() - l0_all.mean())


# --- association from boxes --------------------------------------------------------------------

def test_association_labels_low_visibility_and_causes():
    scene = generate(SceneConfig(seed=3, n_frames=200), DT)
    cam = scene.config.camera
    boxes = clip_boxes(scene.bbox_xyxy, cam.image_width, cam.image_height)
    def separated(f):          # rows whose box overlaps no other box at frame f, so causes cannot mix
        rows = np.flatnonzero(scene.frame == f)
        iou = iou_matrix(boxes[rows], boxes[rows])
        np.fill_diagonal(iou, 0.0)
        return rows, rows[iou.max(axis=1, initial=0.0) == 0.0]
    f = next(f for f in range(0, 200, S) if len(separated(f)[1]) >= 4)
    rows, chosen = separated(f)
    low, gone, undetected, untracked = chosen[:4]
    vf = np.ones(len(scene.frame))
    vf[low], vf[gone] = 0.3, 0.0
    dets = [boxes[scene.frame == k] for k in range(200)]
    dets[f] = boxes[[r for r in rows if r != undetected]]
    tracks = []
    for pid in range(1, scene.config.n_pedestrians + 1):
        r = [x for x in np.flatnonzero((scene.ped_id == pid) & (scene.frame % S == 0)) if x not in (undetected, untracked)]
        if r:
            tracks.append(TrackOutput(100 + pid, scene.frame[r].copy(), boxes[r]))
    result = TrackerResult(tuple(tracks), len(tracks), len(tracks), 0, 0)
    assoc = pf.associate_grid(scene, result, dets, vf, EVAL_IOU)
    for t in assoc.tracks:
        pid = t.track_id - 100
        expected = np.where(t.frames == f, 0 if pid == scene.ped_id[gone] else pid, pid)
        np.testing.assert_array_equal(t.labels, expected)            # the low-visibility stage still matches
        np.testing.assert_allclose(t.positions, bottom_centre_world(
            boxes[(scene.ped_id == pid) & np.isin(scene.frame, t.frames)], cam))
    assert assoc.causes[(int(scene.ped_id[gone]), f)] == pf.INVISIBLE
    assert assoc.causes[(int(scene.ped_id[undetected]), f)] == pf.NO_DETECTION
    assert assoc.causes[(int(scene.ped_id[untracked]), f)] == pf.NOT_IN_TRACKER_OUTPUT
    assert (int(scene.ped_id[low]), f) not in assoc.causes


# --- end to end and saving ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def run():
    return fr.run_feasibility(CFG, SceneConfig(seed=5, n_frames=240))


def test_end_to_end_invariants(run):
    scene = run.scene
    ref = run.reference
    eligible = None
    for source in fr.SOURCES:
        ev = run.prediction[source]
        c = ev.summary["coverage"]
        eligible = eligible if eligible is not None else c["eligible_ground_truth_windows"]
        assert c["eligible_ground_truth_windows"] == eligible == ref["n_windows"]
        assert c["predicted_windows"] <= c["eligible_ground_truth_windows"]
        assert c["predicted_windows"] <= c["available_tracker_histories"]
        assert c["scorable_tracker_histories"] == c["predicted_windows"]            # stride 1
        assert sum(c["lost_by_primary_reason"].values()) == c["lost_windows"]
        assert c["unscorable_tracker_histories"]["total"] == (c["available_tracker_histories"]
                                                              - c["scorable_tracker_histories"])
        surv = ev.summary["survivorship_L0"]["all_eligible"]
        assert surv["mean_ade_m"] == pytest.approx(ref["mean_ade_m"])
        for w in ev.predicted_index:
            pid, frames = int(ev.windows.track_ids[w]), ev.windows.frames[w]
            rows = [np.flatnonzero((scene.ped_id == pid) & (scene.frame == f))[0] for f in frames[H:]]
            np.testing.assert_array_equal(ev.windows.target[w], scene.foot_world[rows])
        np.testing.assert_array_equal(ev.predictions["L2"], constant_velocity(ev.inputs["L2"], F, 2))
    assert eligible > 0 and run.prediction[GT_BOXES].summary["coverage"]["predicted_windows"] > 0
    assert fr.ROLES[GT_BOXES] == fr.ROLES[ORACLE] == "diagnostic" and fr.ROLES[ESTIMATED] == "practical baseline"


def test_save_run_writes_provenance_and_refuses_existing(run, tmp_path):
    out = fr.save_run(tmp_path / "run", run, CFG)
    assert sorted(p.name for p in out.iterdir()) == ["config.resolved.yaml", "manifest.json", "results.json",
                                                     "windows.csv"]
    with pytest.raises(FileExistsError):
        fr.save_run(out, run, CFG)
    reject = lambda token: (_ for _ in ()).throw(ValueError(token))
    results = json.loads((out / "results.json").read_text(encoding="utf-8"), parse_constant=reject)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"), parse_constant=reject)
    assert manifest["experiment_id"] == results["experiment_id"] == run.experiment_id
    assert run.experiment_id.endswith("-s5")
    assert manifest["seed"] == 5 and results["settings"]["scene"]["seed"] == 5
    assert manifest["scene_content_hash"] == run.scene.content_hash() == results["scene_content_hash"]
    assert manifest["config_hash"] == CFG.hash
    assert manifest["spec"] == {"version": CFG.meta["spec"]["version"], "sha256": CFG.meta["spec"]["sha256"]}
    for name, digest in manifest["files"].items():
        assert fr.file_sha256(out / name) == digest
    s = results["settings"]
    assert s["tracker"]["match_iou"] == 0.3 and s["evaluation_iou_threshold"] == EVAL_IOU
    assert s["predictor"]["name"] == "CV" and s["protocol"]["horizon_len"] == F
    assert set(results["sources"]) == set(fr.SOURCES)
    for source, block in results["sources"].items():
        assert block["role"] == fr.ROLES[source]
        assert "20 consecutive tracker observations" in block["u5_grid_windows_note"]
    lines = (out / "windows.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 + 3 * results["ground_truth_reference"]["n_windows"]

    again = fr.save_run(tmp_path / "again", run, CFG)
    for name in ("results.json", "windows.csv"):
        assert (again / name).read_bytes() == (out / name).read_bytes()


def test_rerun_is_deterministic(run, tmp_path):
    second = fr.run_feasibility(CFG, SceneConfig(seed=5, n_frames=240))
    assert second.experiment_id == run.experiment_id and second.frames_sha256 == run.frames_sha256
    assert fr._json(fr.results_document(second)) == fr._json(fr.results_document(run))
    assert fr.windows_csv(second) == fr.windows_csv(run)
