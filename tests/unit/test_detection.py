"""U4: background-subtraction detector and detection evaluation.

Analytic tests use hand-built masks, frames and boxes. End-to-end tests use a rendered scene with a seed
declared here (never used to choose parameters) and assert structural properties, not score targets.
The oracle-background path is a diagnostic; the estimated-background path is the practical baseline.
"""

import shutil

import numpy as np
import pytest

from trackvision import config
from trackvision.experiments.detection_feasibility import (ESTIMATED, MIN_VISIBLE_FRACTION, ORACLE, categorize,
                                                           evaluate_with_estimated_background,
                                                           evaluate_with_oracle_background,
                                                           rendered_visible_fraction)
from trackvision.metrics.detection import (clip_boxes, evaluate_frame, greedy_match, iou_matrix,
                                           precision_recall_f1)
from trackvision.synthetic.render import RenderConfig, SceneRenderer
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.tracking.background_subtraction import (BgSubConfig, connected_components, detect,
                                                         foreground_mask, median_background, sample_indices)

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
IOU = CFG.frozen["association"]["tracker_gt_assoc_iou_threshold"]
TEST_SEED, N_FRAMES = 11, 60                   # declared test scene
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="OPTIONAL: ffmpeg/ffprobe not on PATH")


# --- detector, analytic ------------------------------------------------------------------------

def _mask_with_shapes():
    m = np.zeros((10, 10), dtype=bool)
    m[1:4, 1:5] = True                     # rectangle: rows 1-3, cols 1-4 (12 px)
    m[6:9, 1] = True                       # L shape (5 px)
    m[8, 2:4] = True
    m[1, 7] = m[2, 8] = True               # diagonal pair
    return m


def test_connected_components_8_and_4():
    boxes, areas = connected_components(_mask_with_shapes(), 8)
    np.testing.assert_array_equal(boxes, [[1, 1, 5, 4], [7, 1, 9, 3], [1, 6, 4, 9]])
    np.testing.assert_array_equal(areas, [12, 2, 5])
    boxes4, areas4 = connected_components(_mask_with_shapes(), 4)
    assert len(boxes4) == 4 and sorted(areas4.tolist()) == [1, 1, 5, 12]


def test_runs_joined_through_a_lower_row_form_one_component():
    m = np.zeros((4, 5), dtype=bool)
    m[0:2, 0] = m[0:2, 2] = True           # two separate columns ...
    m[2, 0:3] = True                       # ... joined underneath (U shape)
    boxes, areas = connected_components(m, 8)
    np.testing.assert_array_equal(boxes, [[0, 0, 3, 3]])
    assert areas.tolist() == [7]


def test_foreground_threshold_is_strict_and_detects_exact_square():
    bg = np.full((40, 60), 100.0)
    pattern = (np.indices(bg.shape).sum(axis=0) % 11) - 5.0          # deterministic +-5 "noise"
    frame = np.clip(bg + pattern, 0, 255)
    frame[10:20, 20:30] = 140.0                                       # +40 square
    frame[30, 50] = 120.0                                             # exactly +20: not foreground
    mask = foreground_mask(frame.astype(np.uint8), bg, 20.0)
    expected = np.zeros_like(mask)
    expected[10:20, 20:30] = True
    np.testing.assert_array_equal(mask, expected)
    det = detect(frame.astype(np.uint8), bg, BgSubConfig())
    np.testing.assert_array_equal(det.boxes, [[20, 10, 30, 20]])
    assert det.areas.tolist() == [100]


def test_min_area_filter():
    bg = np.zeros((30, 30))
    frame = np.zeros((30, 30), dtype=np.uint8)
    frame[0:7, 0:7] = 200                 # 49 px
    frame[10:20, 10:15] = 200             # 50 px
    det = detect(frame, bg, BgSubConfig(min_area_px=50))
    np.testing.assert_array_equal(det.boxes, [[10, 10, 15, 20]])


def test_median_background_removes_a_moving_square():
    bg = np.full((8, 40), 90, dtype=np.uint8)
    frames = []
    for k in range(7):
        f = bg.copy()
        f[2:6, 4 * k:4 * k + 4] = 250     # each pixel covered in at most 1 of 7 frames
        frames.append(f)
    np.testing.assert_array_equal(median_background(iter(frames), 7, 50), bg.astype(float))
    np.testing.assert_array_equal(sample_indices(100, 5), [0, 25, 50, 74, 99])
    with pytest.raises(ValueError):
        median_background(iter(frames[:3]), 7, 7)


# --- evaluation, analytic ----------------------------------------------------------------------

def test_iou_values():
    a = [[0, 0, 10, 10]]
    assert iou_matrix(a, a)[0, 0] == 1.0
    assert iou_matrix(a, [[20, 20, 30, 30]])[0, 0] == 0.0
    assert iou_matrix(a, [[5, 0, 15, 10]])[0, 0] == pytest.approx(1 / 3)
    assert iou_matrix([[1, 1, 1, 1]], [[1, 1, 1, 1]])[0, 0] == 0.0      # empty union


def test_greedy_match_is_one_to_one_inclusive_and_deterministic():
    iou = np.array([[0.9, 0.6], [0.8, 0.5]])
    assert greedy_match(iou, 0.5) == [(0, 0), (1, 1)]                   # 0.5 is inclusive
    assert greedy_match(iou, 0.51) == [(0, 0)]
    tie = np.array([[0.7, 0.7], [0.7, 0.7]])
    assert greedy_match(tie, 0.5) == [(0, 0), (1, 1)]


def test_evaluate_frame_with_ignore_regions():
    gt = [[0, 0, 10, 20], [50, 0, 60, 20], [100, 0, 110, 20]]
    evaluable = [True, False, True]
    ignore = [False, True, False]
    det = [[0, 1, 10, 20],      # ~ gt0 -> TP
           [50, 2, 60, 20],     # ~ gt1 (ignore) -> ignored match
           [51, 3, 60, 19],     # ~ gt1 again -> already used: FP overlapping a visible GT
           [5, 10, 15, 30],     # overlaps gt0 weakly -> FP overlapping a visible GT
           [200, 0, 210, 10]]   # background -> FP
    c = evaluate_frame(det, gt, evaluable, ignore, IOU)
    assert (c.tp, c.fn, c.ignored_matches, c.fp_overlapping_visible, c.fp_background) == (1, 1, 1, 2, 1)
    assert c.matches == ((0, 0),)
    assert c.tp + c.fp + c.ignored_matches == c.n_detections
    assert precision_recall_f1(c.tp, c.fp, c.fn) == pytest.approx({"precision": 0.25, "recall": 0.5,
                                                                   "f1": 2 * 0.25 * 0.5 / 0.75})
    invisible_only = evaluate_frame([[0, 0, 10, 20]], gt, [False] * 3, [False] * 3, IOU)
    assert (invisible_only.tp, invisible_only.fn, invisible_only.fp_overlapping_visible,
            invisible_only.fp_background) == (0, 0, 0, 1)          # invisible GT excluded entirely


def test_undefined_rates_are_none():
    assert precision_recall_f1(0, 0, 0) == {"precision": None, "recall": None, "f1": None}
    assert precision_recall_f1(0, 3, 0)["recall"] is None
    assert precision_recall_f1(0, 2, 2)["f1"] == 0.0


def test_categories_and_clipping():
    cats = categorize(np.array([0.0, 0.2, 0.5, 1.0]), MIN_VISIBLE_FRACTION)
    assert cats.tolist() == ["invisible", "ignore", "evaluable", "evaluable"]
    np.testing.assert_array_equal(clip_boxes([[-5, -2, 30, 700]], 720, 576), [[0, 0, 30, 576]])


# --- end to end --------------------------------------------------------------------------------

def _scene():
    return generate(SceneConfig(seed=TEST_SEED, n_frames=N_FRAMES), DT)


def _check_identities(result):
    s = result.summary
    assert s["tp"] + s["fn"] == s["ground_truth"]["evaluable"]
    assert s["tp"] + s["fp"] + s["ignored_matches"] == s["n_detections"]
    assert s["parameters"]["min_visible_fraction"] == MIN_VISIBLE_FRACTION
    assert s["parameters"]["iou_threshold"] == IOU
    g = s["ground_truth"]
    assert g["evaluable"] + g["ignore_partial"] + g["invisible"] == g["rows"]


def test_oracle_background_detects_every_isolated_evaluable_pedestrian():
    scene = _scene()
    result = evaluate_with_oracle_background(scene, RenderConfig(), BgSubConfig(), CFG)
    assert result.mode == ORACLE
    _check_identities(result)
    gt = clip_boxes(scene.bbox_xyxy, 720, 576)
    # Rendered pixels lie within 0.5 px of their box and 8-connected pixels are <= 1 px apart, so two
    # silhouettes can merge into one blob only if their boxes come within 0.5 + 1 + 0.5 = 2 px.
    margin = 2.0
    isolated_checked = 0
    for k, counts in enumerate(result.frame_counts):
        start, stop = np.searchsorted(scene.frame, [k, k + 1])
        boxes, cats = gt[start:stop], result.categories[start:stop]
        grown = boxes + np.array([-margin, -margin, margin, margin])
        visible = cats != "invisible"
        overlap = (iou_matrix(grown, boxes) > 0) & visible[None, :]
        np.fill_diagonal(overlap, False)
        matched = {g for _, g in counts.matches}
        for g in range(len(boxes)):
            if cats[g] == "evaluable" and not overlap[g].any():
                assert g in matched, f"isolated evaluable pedestrian missed in frame {k}"
                isolated_checked += 1
    assert isolated_checked > 0


def test_estimated_background_on_rendered_frames():
    scene = _scene()
    renderer = SceneRenderer(scene, RenderConfig())
    vf = rendered_visible_fraction(scene, RenderConfig())
    result = evaluate_with_estimated_background(
        scene, lambda: (renderer.frame(k).image for k in range(N_FRAMES)), vf, BgSubConfig(), CFG)
    assert result.mode == ESTIMATED
    _check_identities(result)
    assert len(result.detections) == N_FRAMES


@needs_ffmpeg
def test_estimated_background_from_decoded_video_equals_rendered_path(tmp_path):
    from trackvision.synthetic.video import export_rendered_scene, iter_decoded_frames

    scene = _scene()
    out = export_rendered_scene(scene, RenderConfig(), tmp_path / "export")
    with np.load(out["dir"] / "render.npz") as render:
        vf_video = render["visible_fraction"].copy()
    from_video = evaluate_with_estimated_background(
        scene, lambda: iter_decoded_frames(out["dir"] / "video.mkv", 720, 576), vf_video, BgSubConfig(), CFG)
    renderer = SceneRenderer(scene, RenderConfig())
    from_render = evaluate_with_estimated_background(
        scene, lambda: (renderer.frame(k).image for k in range(N_FRAMES)),
        rendered_visible_fraction(scene, RenderConfig()), BgSubConfig(), CFG)
    assert from_video.summary == from_render.summary                     # lossless: identical results
    for a, b in zip(from_video.detections, from_render.detections):
        np.testing.assert_array_equal(a.boxes, b.boxes)
