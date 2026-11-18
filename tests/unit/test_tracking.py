"""U5: one-to-one assignment, online IoU tracker, ground-plane trajectories and tracking metrics.

Tracker and metric cases use hand-built detection sequences with hand-derived expected IDs and counts.
The metrics are project-specific feasibility metrics (not MOTChallenge/TrackEval). End-to-end tests on
rendered scenes assert structural properties only; their seeds are declared here and never used to
choose parameters.
"""

import itertools

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.detection_feasibility import ORACLE, rendered_visible_fraction
from trackvision.experiments.tracking_feasibility import (GT_BOXES, bottom_centre_world, detector_detections,
                                                          evaluate_tracking_on_scene, grid_tracks,
                                                          ground_truth_detections)
from trackvision.metrics.detection import clip_boxes, iou_matrix
from trackvision.metrics.tracking import FrameTracking, evaluate_tracking
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import DEFAULT_CAMERA, SceneConfig, generate
from trackvision.tracking.assignment import gated_assignment, gated_iou_assignment, linear_assignment
from trackvision.tracking.background_subtraction import BgSubConfig
from trackvision.tracking.tracker import TrackerConfig, run_tracker

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
EVAL_IOU = CFG.frozen["association"]["tracker_gt_assoc_iou_threshold"]
STATIC = CFG.frozen["protocol"]["static"]
L = STATIC["history_len"] + STATIC["horizon_len"]
TC = TrackerConfig()                 # match_iou 0.3, min_hits 3, max_age 25


def box(cx, cy, w=10.0, h=20.0):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def ids_of(result):
    return [t.track_id for t in result.tracks]


# --- assignment --------------------------------------------------------------------------------

def test_linear_assignment_known_cases():
    assert linear_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == [(0, 1), (1, 0), (2, 2)]
    assert linear_assignment([[1, 2, 3], [3, 2, 1]]) == [(0, 0), (1, 2)]
    assert linear_assignment([[1, 3], [2, 2], [3, 1]]) == [(0, 0), (2, 1)]
    assert linear_assignment(np.zeros((2, 2))) == [(0, 0), (1, 1)]          # deterministic ties
    assert linear_assignment(np.zeros((2, 3))) == [(0, 0), (1, 1)]
    assert linear_assignment(np.zeros((0, 3))) == []


def _brute_force_min(cost):
    n, m = cost.shape
    if n > m:
        return _brute_force_min(cost.T)
    return min(sum(cost[i, p[i]] for i in range(n)) for p in itertools.permutations(range(m), n))


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (3, 2), (4, 4), (3, 5), (5, 4)])
def test_linear_assignment_is_optimal_against_brute_force(shape):
    rng = np.random.default_rng(0)
    for _ in range(20):
        cost = rng.integers(0, 10, size=shape).astype(float)          # integer costs include ties
        pairs = linear_assignment(cost)
        assert len(pairs) == min(shape)
        assert len({r for r, _ in pairs}) == len({c for _, c in pairs}) == len(pairs)
        assert sum(cost[r, c] for r, c in pairs) == _brute_force_min(cost)


def test_gating_never_matches_below_threshold_and_prefers_one_strong_match():
    assert gated_iou_assignment([box(0, 0)], [box(6.5, 0)], 0.3) == []      # IoU < 0.3, the only candidate
    iou = np.array([[0.9, 0.35], [0.35, 0.0]])
    assert [(i, j) for i, j, _ in gated_assignment(iou, 0.3)] == [(0, 0)]   # 0.9 beats 0.35 + 0.35


# --- tracker lifecycle -------------------------------------------------------------------------

def test_single_object_constant_motion():
    dets = [[box(10 + 2 * k, 50)] for k in range(10)]
    result = run_tracker(dets, TC)
    assert ids_of(result) == [1]
    np.testing.assert_array_equal(result.tracks[0].frames, np.arange(10))
    np.testing.assert_array_equal(result.tracks[0].boxes, np.array(dets).reshape(-1, 4))


def test_short_lived_detection_is_never_output_and_ids_are_not_reused():
    dets = []
    for k in range(10):
        frame = [box(10 + 2 * k, 50)]                    # Q: whole sequence
        if k < 2:
            frame.append(box(100, 100))                  # P: two frames only
        if k >= 5:
            frame.append(box(200, 200))                  # R: from frame 5
        dets.append(frame)
    result = run_tracker(dets, TC)
    assert ids_of(result) == [1, 3]                      # P took ID 2 and was deleted
    assert result.n_created == 3 and result.n_deleted_tentative == 1
    np.testing.assert_array_equal(result.tracks[1].frames, np.arange(5, 10))   # tentative frames included


def test_ambiguous_assignment_tie_goes_to_lower_detection_index():
    dets = [[box(50, 50)] for _ in range(3)]
    dets.append([box(52, 50), box(48, 50)])              # equal IoU with the prediction box(50, 50)
    dets.append([box(54, 50), box(48, 50)])
    dets.append([box(56, 50), box(48, 50)])
    result = run_tracker(dets, TC)
    assert ids_of(result) == [1, 2]
    np.testing.assert_array_equal(result.tracks[0].boxes[3], box(52, 50))
    np.testing.assert_array_equal(result.tracks[1].frames, [3, 4, 5])


@pytest.mark.parametrize("gap, expected_ids", [(25, [1]), (26, [1, 2])])
def test_temporary_disappearance_up_to_max_age(gap, expected_ids):
    frames_present = list(range(5)) + list(range(5 + gap, 10 + gap))
    dets = [[box(10 + 2 * k, 50)] if k in frames_present else [] for k in range(10 + gap)]
    result = run_tracker(dets, TC)
    assert ids_of(result) == expected_ids
    observed = np.concatenate([t.frames for t in result.tracks])
    np.testing.assert_array_equal(np.sort(observed), frames_present)      # no coasting positions output


def test_empty_frames_cause_misses_and_no_births():
    dets = [[box(50, 50)], [], [], [box(50, 50)]]
    result = run_tracker(dets, TC)
    assert result.tracks == ()
    assert result.n_created == 2 and result.n_deleted_tentative == 1


def _crossing(k):
    return box(2 * k, 50), box(60 - 2 * k, 60)         # offset rows: IoU 1/3 when aligned


def test_crossing_tracks_keep_their_ids():
    dets = [list(_crossing(k)) for k in range(31)]
    result = run_tracker(dets, TC)
    assert ids_of(result) == [1, 2]
    for t, which in zip(result.tracks, (0, 1)):
        np.testing.assert_array_equal(t.boxes, [d[which] for d in dets])


def test_occlusion_during_crossing_recovers_the_same_id():
    dets = [[_crossing(k)[0]] if 13 <= k <= 17 else list(_crossing(k)) for k in range(31)]
    result = run_tracker(dets, TC)
    assert ids_of(result) == [1, 2]
    np.testing.assert_array_equal(result.tracks[1].frames, [k for k in range(31) if not 13 <= k <= 17])


def test_merged_boxes_keep_output_consistent():
    dets = []
    for k in range(31):
        a, b = _crossing(k)
        if abs(a[0] - b[0]) < 10:
            dets.append([[min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]])
        else:
            dets.append([a, b])
    result = run_tracker(dets, TC)
    claimed = {}
    for t in result.tracks:
        assert np.all(np.diff(t.frames) > 0)
        for f, b in zip(t.frames, t.boxes):                   # every output box is a real detection ...
            idx = [i for i, d in enumerate(np.asarray(dets[f]).reshape(-1, 4)) if np.array_equal(b, d)]
            assert len(idx) == 1
            key = (int(f), idx[0])
            assert key not in claimed                         # ... claimed by at most one track
            claimed[key] = t.track_id
    merged = [f for f in range(31) if len(dets[f]) == 1]
    assert all(sum((f, 0) == key for key in claimed) == 1 for f in merged)   # one observer per merged frame
    # Regression check of the documented failure mode (derived from the rules, observed once): the
    # merged box corrupts both tracks' velocity estimates, so neither prediction reaches the 0.3 gate
    # when the pedestrians separate at frame 18, and both are reborn as new tracks (fragmentation).
    assert ids_of(result) == [1, 2, 3, 4]
    assert [int(t.frames[0]) for t in result.tracks[2:]] == [18, 18]


# --- positions, grid and windows ---------------------------------------------------------------

def test_bottom_centre_maps_a_known_ground_point():
    u, v = DEFAULT_CAMERA.project([[10.0, 0.5, 0.0]])[0]
    np.testing.assert_allclose(bottom_centre_world([[u - 5, v - 30, u + 5, v]], DEFAULT_CAMERA), [[10.0, 0.5]],
                               atol=1e-9)


def test_grid_tracks_use_observations_only_and_gaps_split_windows():
    n, missing = 401, 100
    ground = np.column_stack([8.0 + 0.02 * np.arange(n), np.zeros(n)])
    pixels = DEFAULT_CAMERA.project(np.column_stack([ground, np.zeros(n)]))
    boxes = np.column_stack([pixels[:, 0] - 8, pixels[:, 1] - 40, pixels[:, 0] + 8, pixels[:, 1]])
    dets = [[] if k == missing else [boxes[k]] for k in range(n)]
    result = run_tracker(dets, TC)
    (track,) = grid_tracks(result, DEFAULT_CAMERA, 10)
    expected_frames = [f for f in range(0, n, 10) if f != missing]
    np.testing.assert_array_equal(track.frames, expected_frames)
    np.testing.assert_allclose(track.positions, ground[expected_frames], atol=1e-9)
    windows = make_windows("t", [track], 10, STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"])
    assert len(windows) == (30 - L) // STATIC["window_stride"] + 1          # runs of 10 and 30 samples
    assert not np.any(windows.frames == missing)
    assert np.all(np.diff(windows.frames, axis=1) == 10)


# --- metrics (project-specific) ----------------------------------------------------------------

def _frame(track_ids, track_boxes, gt_ids, gt_boxes, cats=None, track_pos=None, gt_pos=None):
    k, g = len(track_ids), len(gt_ids)
    return FrameTracking(np.array(track_ids), np.array(track_boxes, float).reshape(-1, 4),
                         np.zeros((k, 2)) if track_pos is None else np.array(track_pos, float),
                         np.array(gt_ids), np.array(gt_boxes, float).reshape(-1, 4),
                         np.array(cats if cats else ["evaluable"] * g, dtype=object),
                         np.zeros((g, 2)) if gt_pos is None else np.array(gt_pos, float))


def test_metrics_perfect_tracking():
    g1, g2 = box(20, 50), box(80, 50)
    m = evaluate_tracking([_frame([1, 2], [g1, g2], [1, 2], [g1, g2]) for _ in range(5)], EVAL_IOU)
    assert (m["tp"], m["fp"], m["fn"], m["idsw"], m["frag"]) == (10, 0, 0, 0, 0)
    assert m["mota"] == 1.0 and m["idf1"] == 1.0 and m["position_rmse_m"] == 0.0


def test_metrics_identity_swap():
    g1, g2 = box(20, 50), box(80, 50)
    frames = [_frame([1, 2], [g1, g2] if k < 5 else [g2, g1], [1, 2], [g1, g2]) for k in range(10)]
    m = evaluate_tracking(frames, EVAL_IOU)
    assert (m["tp"], m["idsw"]) == (20, 2)
    assert m["mota"] == pytest.approx(0.9)
    assert (m["idtp"], m["idfp"], m["idfn"]) == (10, 10, 10) and m["idf1"] == pytest.approx(0.5)


def test_metrics_fragmentation_and_misses():
    g = box(20, 50)
    frames = [_frame([1] if k in (0, 1, 2, 5, 6) else [], [g] if k in (0, 1, 2, 5, 6) else [], [1], [g])
              for k in range(7)]
    m = evaluate_tracking(frames, EVAL_IOU)
    assert (m["tp"], m["fn"], m["frag"], m["idsw"]) == (5, 2, 1, 0)
    assert m["mota"] == pytest.approx(1 - 2 / 7)


def test_metrics_ignore_regions_and_position_error():
    g1, g2 = box(20, 50), box(80, 50)
    frame = _frame([1, 2, 3], [g1, g2, box(300, 300)], [1, 2], [g1, g2], cats=["evaluable", "ignore"],
                   track_pos=[[3, 4], [0, 0], [0, 0]], gt_pos=[[0, 0], [9, 9]])
    m = evaluate_tracking([frame], EVAL_IOU)
    assert (m["tp"], m["ignored"], m["fp"], m["fn"]) == (1, 1, 1, 0)
    assert m["idfp"] == (3 - 1) - 1
    assert m["position_rmse_m"] == pytest.approx(5.0)


# --- end to end --------------------------------------------------------------------------------

def _scene(n_frames, seed=11):
    return generate(SceneConfig(seed=seed, n_frames=n_frames), DT)


def test_ground_truth_boxes_keep_one_id_for_isolated_pedestrians():
    scene = _scene(60)
    vf = rendered_visible_fraction(scene, RenderConfig())
    dets = ground_truth_detections(scene, vf)
    result = run_tracker(dets, TC)
    by_frame = result.frame_index()
    boxes = clip_boxes(scene.bbox_xyxy, 720, 576)
    checked = 0
    for pid in range(1, scene.config.n_pedestrians + 1):
        rows = np.flatnonzero(scene.ped_id == pid)
        if len(rows) < TC.min_hits or np.any(vf[rows] <= 0):
            continue
        isolated = True
        for r in rows:
            f = scene.frame[r]
            others = [o for o in np.flatnonzero(scene.frame == f) if o != r and vf[o] > 0]
            grown = boxes[r] + np.array([-2.0, -2.0, 2.0, 2.0])
            if others and np.any(iou_matrix([grown], boxes[others]) > 0):
                isolated = False
                break
        if not isolated:
            continue
        owners = set()
        for r in rows:
            match = [tid for tid, b in by_frame.get(int(scene.frame[r]), []) if np.array_equal(b, boxes[r])]
            assert len(match) == 1, f"pedestrian {pid} not observed at frame {scene.frame[r]}"
            owners.add(match[0])
        assert len(owners) == 1, f"pedestrian {pid} carried by tracks {owners}"
        checked += 1
    assert checked > 0
    summary = evaluate_tracking_on_scene(scene, dets, vf, CFG, GT_BOXES, TC).summary
    assert summary["mode"] == GT_BOXES and summary["parameters"]["match_iou"] == 0.3
    assert summary["parameters"]["evaluation_iou_threshold"] == EVAL_IOU


def test_oracle_detector_pipeline_identities_and_grid():
    scene = _scene(300)
    rc = RenderConfig()
    renderer = SceneRenderer(scene, rc)
    vf = rendered_visible_fraction(scene, rc)
    dets = detector_detections((renderer.frame(k).image for k in range(300)), renderer.background, BgSubConfig())
    ev = evaluate_tracking_on_scene(scene, dets, vf, CFG, ORACLE, TC)
    m = ev.summary["metrics"]
    assert m["tp"] + m["fn"] == m["evaluable_observations"]
    assert m["tp"] + m["fp"] + m["ignored"] == m["tracker_observations"]
    assert m["mota"] is not None and m["idf1"] is not None
    observed = {t.track_id: set(t.frames.tolist()) for t in ev.tracker.tracks}
    expected_windows = 0
    for g in ev.grid_tracks:
        assert np.all(g.frames % scene.frames_per_step == 0)
        assert set(g.frames.tolist()) <= observed[g.ped_id]                # observations only
        runs = np.split(g.frames, np.flatnonzero(np.diff(g.frames) != scene.frames_per_step) + 1)
        expected_windows += sum(max(0, (len(r) - L) // STATIC["window_stride"] + 1) for r in runs if len(r))
    assert ev.summary["grid"]["windows"] == expected_windows
