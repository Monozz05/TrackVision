"""U3: grey-level renderer and lossless FFV1 video export of U1 synthetic scenes.

Artefacts go only to pytest's tmp_path. The ffmpeg-dependent tests are reported as OPTIONAL (skipped
with that prefix) on machines without ffmpeg/ffprobe; they are never counted as passed there.
"""

import hashlib
import json
import shutil

import numpy as np
import pytest

from trackvision import config
from trackvision.synthetic.render import (BACKGROUND_ID, RenderConfig, RenderError, SceneRenderer, convex_hull,
                                          paint_far_to_near, quality_report, rasterize_convex,
                                          validate_instance_ids)
from trackvision.synthetic.scene import SceneConfig, generate

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
N_FRAMES = 30
MIN_CONTRAST = 30.0          # grey levels; the bands are >= 38 levels from the background by construction
HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="OPTIONAL: ffmpeg/ffprobe not on PATH")


def _scene(seed=3):
    return generate(SceneConfig(seed=seed, n_frames=N_FRAMES), DT)


def _square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# --- geometry helpers --------------------------------------------------------------------------

def test_convex_hull_and_rasterization_of_a_square():
    pts = np.vstack([_square(10, 10, 20, 20), [[15.0, 15.0], [12.0, 18.0]]])     # interior points dropped
    hull = convex_hull(pts)
    assert len(hull) == 4
    mask, x0, y0 = rasterize_convex(hull)
    assert mask.sum() == 100                     # pixel centres 10.5 .. 19.5 in both axes
    assert (x0, y0) == (10, 10)


def test_nearer_layer_occludes_farther_regardless_of_order():
    near, far = (5.0, 1, _square(10, 10, 30, 30)), (9.0, 2, _square(20, 20, 40, 40))
    for layers in ([near, far], [far, near]):
        instances, totals = paint_far_to_near((50, 50), layers)
        assert totals == {1: 400, 2: 400}
        assert (instances == 1).sum() == 400                   # nearer is fully visible
        assert (instances == 2).sum() == 400 - 100              # the 10 x 10 overlap is hidden
        assert instances[25, 25] == 1


def test_silhouette_is_counted_before_clipping():
    instances, totals = paint_far_to_near((20, 20), [(1.0, 1, _square(10, 10, 30, 30))])
    assert totals[1] == 400                      # complete silhouette
    assert (instances == 1).sum() == 100         # only 10 x 10 lies inside the 20 x 20 image


@pytest.mark.parametrize("ids", [[0, 1], [-3], [np.iinfo(np.int32).max + 1]])
def test_instance_ids_must_be_positive_int32(ids):
    with pytest.raises(RenderError):
        validate_instance_ids(np.array(ids, dtype=np.int64))


def test_render_config_rejects_bands_inside_background_range():
    with pytest.raises(RenderError):
        RenderConfig(dark_band=(20, 105))


# --- rendering ---------------------------------------------------------------------------------

def test_frame_count_shape_dtype_and_indices():
    scene = _scene()
    frames = list(SceneRenderer(scene))
    assert [f.index for f in frames] == list(range(N_FRAMES))
    for f in frames:
        assert f.image.shape == (576, 720) and f.image.dtype == np.uint8
        assert f.instances.shape == (576, 720) and f.instances.dtype == np.int32


def test_rendered_pixels_align_with_ground_truth():
    scene = _scene()
    checked_unoccluded = 0
    for f in SceneRenderer(scene):
        np.testing.assert_array_equal(f.ped_ids, scene.ped_id[scene.frame == f.index])
        present = set(np.unique(f.instances)) - {BACKGROUND_ID}
        assert present <= set(f.ped_ids.tolist())                       # only pedestrians of this frame
        boxes = scene.bbox_xyxy[f.rows]
        for k, pid in enumerate(f.ped_ids):
            ys, xs = np.nonzero(f.instances == pid)
            assert len(xs) == f.visible_pixels[k]
            if len(xs):                                                  # every pixel centre inside the GT box
                x1, y1, x2, y2 = boxes[k]
                assert np.all((xs + 0.5 >= x1 - 1e-9) & (xs + 0.5 <= x2 + 1e-9))
                assert np.all((ys + 0.5 >= y1 - 1e-9) & (ys + 0.5 <= y2 + 1e-9))
            assert 0.0 <= f.visible_fraction[k] <= 1.0
            assert f.visible_fraction[k] == pytest.approx(f.visible_pixels[k] / f.silhouette_pixels[k])
            others = np.delete(boxes, k, axis=0)
            overlaps = np.any((others[:, 0] < boxes[k, 2]) & (others[:, 2] > boxes[k, 0])
                              & (others[:, 1] < boxes[k, 3]) & (others[:, 3] > boxes[k, 1]))
            if not scene.truncated[f.rows][k] and not overlaps:
                assert f.visible_fraction[k] == 1.0                     # nothing can hide or clip it
                checked_unoccluded += 1
    assert checked_unoccluded > 0


def test_rendering_is_deterministic_and_seed_sensitive():
    scene = _scene()
    a = [hashlib.sha256(f.image.tobytes()).hexdigest() for f in SceneRenderer(scene, RenderConfig(seed=0))]
    b = [hashlib.sha256(f.image.tobytes()).hexdigest() for f in SceneRenderer(scene, RenderConfig(seed=0))]
    assert a == b
    c = SceneRenderer(scene, RenderConfig(seed=1)).frame(0).image
    assert not np.array_equal(c, SceneRenderer(scene, RenderConfig(seed=0)).frame(0).image)


def test_noise_free_background_is_static():
    scene = _scene()
    renderer = SceneRenderer(scene, RenderConfig(noise_sigma=0.0))
    f0, f1 = renderer.frame(0), renderer.frame(N_FRAMES - 1)
    both_background = (f0.instances == BACKGROUND_ID) & (f1.instances == BACKGROUND_ID)
    np.testing.assert_array_equal(f0.image[both_background], f1.image[both_background])


def test_pedestrians_contrast_with_background():
    scene = _scene()
    renderer = SceneRenderer(scene)
    contrasts = np.concatenate([quality_report(renderer.frame(i), scene.bbox_xyxy[renderer.frame(i).rows])
                                for i in (0, N_FRAMES // 2, N_FRAMES - 1)])
    contrasts = contrasts[~np.isnan(contrasts)]
    assert len(contrasts) > 0 and contrasts.min() >= MIN_CONTRAST


# --- FFV1 round trip ---------------------------------------------------------------------------

@needs_ffmpeg
def test_ffv1_round_trip_preserves_frames_and_alignment(tmp_path):
    from trackvision.synthetic.video import export_rendered_scene, iter_decoded_frames, probe

    scene = _scene()
    out = export_rendered_scene(scene, RenderConfig(), tmp_path / "export")
    info = probe(out["dir"] / "video.mkv")
    assert info["codec_name"] == "ffv1" and info["pix_fmt"] == "gray"
    assert (info["width"], info["height"]) == (720, 576)
    assert info["avg_frame_rate"] == "25/1"
    assert info["nb_read_frames"] == N_FRAMES

    rendered = SceneRenderer(scene)
    with np.load(out["dir"] / "render.npz") as render:
        digests = render["frame_sha256"]
        decoded_count = 0
        for k, decoded in enumerate(iter_decoded_frames(out["dir"] / "video.mkv", 720, 576)):
            np.testing.assert_array_equal(decoded, rendered.frame(k).image)          # frame k = GT frame k
            assert hashlib.sha256(decoded.tobytes()).digest() == digests[k].tobytes()
            decoded_count += 1
        assert decoded_count == N_FRAMES
        np.testing.assert_array_equal(render["visible_fraction"][rendered.frame(5).rows],
                                      rendered.frame(5).visible_fraction)

    meta = json.loads((out["dir"] / "render_meta.json").read_text(encoding="utf-8"))
    assert meta["scene_content_hash"] == scene.content_hash() and meta["n_frames"] == N_FRAMES
    with np.load(out["dir"] / "ground_truth.npz") as gt:
        assert json.loads(str(gt["metadata"]))["content_hash"] == scene.content_hash()
    with pytest.raises(FileExistsError):
        export_rendered_scene(scene, RenderConfig(), tmp_path / "export")


@needs_ffmpeg
def test_preview_png_has_frame_dimensions(tmp_path):
    from trackvision.synthetic.video import probe, write_preview_png

    scene = _scene()
    frame = SceneRenderer(scene).frame(0)
    path = write_preview_png(tmp_path / "preview.png", frame.image, scene.bbox_xyxy[frame.rows])
    info = probe(path)
    assert info["codec_name"] == "png" and (info["width"], info["height"]) == (720, 576)


@needs_ffmpeg
def test_invalid_frame_leaves_no_partial_video(tmp_path):
    from trackvision.synthetic.video import write_ffv1

    good = np.zeros((576, 720), dtype=np.uint8)
    with pytest.raises(ValueError):
        write_ffv1(tmp_path / "bad.mkv", [good, np.zeros((10, 10), dtype=np.uint8)], 720, 576, 25)
    assert not (tmp_path / "bad.mkv").exists()
