"""U1: synthetic pinhole camera and waypoint-walker scene ground truth.

The camera is checked against the independent fixture tests/fixtures/synthetic_geometry.py (written
separately from production code) and against analytic properties, never only against itself.
Tolerances and the sampling interval come from configs/base.yaml.
"""

import json

import numpy as np
import pytest

from fixtures import synthetic_geometry as sg
from trackvision import config
from trackvision.synthetic.camera import CameraError, PinholeCamera
from trackvision.synthetic.scene import (SceneConfig, SceneError, frames_per_step, generate,
                                         ground_footprint)

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
TOL_PX = CFG.frozen["geometry"]["roundtrip_tolerance_px"]["synthetic_float64"]
FIXTURE_PARAMS = {"height_m": 12.0, "pitch_down_rad": float(np.deg2rad(50.0)), "yaw_rad": 0.3,
                  "position_xy": (-4.0, 2.0)}         # sg.static_camera()


def _fixture_like_camera(**overrides) -> PinholeCamera:
    params = {"image_width": sg.IMAGE_SIZE[0], "image_height": sg.IMAGE_SIZE[1], "focal_px": sg.FOCAL_PX,
              "principal_point": sg.PRINCIPAL_POINT, **FIXTURE_PARAMS, **overrides}
    return PinholeCamera(**params)


def _independent_projection(camera: PinholeCamera, xyz: np.ndarray) -> np.ndarray:
    """Full 3-D projection built from the fixture's rotation helper, not from production code."""
    k = np.array([[camera.focal_px, 0, camera.principal_point[0]],
                  [0, camera.focal_px, camera.principal_point[1]], [0, 0, 1.0]])
    r = sg.camera_rotation(camera.yaw_rad, camera.pitch_down_rad)
    centre = np.array([*camera.position_xy, camera.height_m])
    cam = (np.asarray(xyz, dtype=float) - centre) @ r.T
    p = cam @ k.T
    return p[..., :2] / p[..., 2:3]


# --- camera ----------------------------------------------------------------------------------

def test_ground_homography_matches_independent_fixture():
    camera = _fixture_like_camera()
    reference = sg.static_camera()
    np.testing.assert_allclose(camera.ground_homography() / camera.ground_homography()[2, 2],
                               reference["h_ground_to_img"] / reference["h_ground_to_img"][2, 2],
                               rtol=1e-12, atol=1e-12)
    world = sg.world_grid(reference)
    uv_production = camera.ground_transform().world_to_image(world)
    assert np.max(np.linalg.norm(uv_production - sg.project(reference["h_ground_to_img"], world), axis=1)) <= TOL_PX


def test_full_projection_agrees_with_ground_transform_and_independent_projection():
    camera = _fixture_like_camera()
    world = sg.world_grid(sg.static_camera())
    xyz = np.column_stack([world, np.zeros(len(world))])
    via_projection = camera.project(xyz)
    via_homography = camera.ground_transform().world_to_image(world)
    assert np.max(np.linalg.norm(via_projection - via_homography, axis=1)) <= TOL_PX
    # Points above the ground use the full projection only; compare with the independent one.
    lifted = xyz + np.array([0.0, 0.0, 1.75])
    assert np.max(np.linalg.norm(camera.project(lifted) - _independent_projection(camera, lifted), axis=1)) <= TOL_PX


def test_analytic_principal_point_horizon_and_left_right():
    camera = _fixture_like_camera()
    heading = np.array([np.cos(camera.yaw_rad), np.sin(camera.yaw_rad)])
    left = np.array([-heading[1], heading[0]])
    base = np.array(camera.position_xy)
    on_axis = base + heading * camera.height_m / np.tan(camera.pitch_down_rad)
    uv = camera.project([[*on_axis, 0.0]])[0]
    assert np.linalg.norm(uv - np.array(camera.principal_point)) <= TOL_PX
    assert camera.horizon_row == pytest.approx(camera.principal_point[1]
                                               - camera.focal_px * np.tan(camera.pitch_down_rad), abs=1e-12)
    far = camera.project([[*(base + heading * 1e8), 0.0]])[0]
    assert abs(far[1] - camera.horizon_row) <= 1e-3
    to_left = camera.project([[*(on_axis + left), 0.0]])[0]
    assert to_left[0] < camera.principal_point[0]              # world left of heading is image left


def test_round_trip_and_orientation():
    camera = _fixture_like_camera()
    transform = camera.ground_transform()
    world = sg.world_grid(sg.static_camera())
    assert transform.roundtrip_errors(world)["px"] <= TOL_PX
    # Right-handed world, z up, seen from above by a y-down image: orientation-reversing everywhere.
    assert np.all(transform.orientation_signs(transform.world_to_image(world)) < 0)


@pytest.mark.parametrize("override", [{"pitch_down_rad": float(np.deg2rad(10.0))}, {"focal_px": 0.0},
                                      {"height_m": -1.0}, {"pitch_down_rad": float(np.pi / 2)}])
def test_invalid_camera_is_rejected(override):
    with pytest.raises(CameraError):
        _fixture_like_camera(**override)


def test_point_behind_camera_is_rejected():
    camera = _fixture_like_camera()
    heading = np.array([np.cos(camera.yaw_rad), np.sin(camera.yaw_rad)])
    behind = np.array(camera.position_xy) - 50.0 * heading
    with pytest.raises(CameraError):
        camera.project([[*behind, 0.0]])


# --- scene -----------------------------------------------------------------------------------

MARKERS = ((10.0, 0.0), (8.0, 2.0))      # inside the default camera's footprint


def _scene(seed=7, **overrides):
    return generate(SceneConfig(seed=seed, markers_xy=MARKERS, **overrides), DT)


def test_frames_per_step_derived_from_config():
    assert frames_per_step(25, DT) == 10
    assert _scene().frames_per_step == 10
    with pytest.raises(SceneError):
        frames_per_step(24, DT)                     # 0.4 s is 9.6 frames at 24 fps


def test_same_seed_is_deterministic_and_seed_changes_scene():
    a, b = _scene(seed=7), _scene(seed=7)
    assert a.content_hash() == b.content_hash()
    for name, arr in a.arrays().items():
        np.testing.assert_array_equal(arr, b.arrays()[name], err_msg=name)
    c = _scene(seed=8)
    assert c.content_hash() != a.content_hash()
    assert not np.array_equal(c.speed_mps, a.speed_mps)


def test_ground_truth_arrays_are_read_only():
    scene = _scene()
    with pytest.raises(ValueError):
        scene.foot_world[0, 0] = 0.0


def test_identities_unique_and_contiguous():
    scene = _scene()
    pairs = set(zip(scene.frame.tolist(), scene.ped_id.tolist()))
    assert len(pairs) == len(scene.frame)                                  # one row per (frame, id)
    assert np.all(np.diff(scene.frame) >= 0)
    for i, pid in enumerate(range(1, scene.config.n_pedestrians + 1)):
        frames = scene.frame[scene.ped_id == pid]
        np.testing.assert_array_equal(frames, np.arange(scene.start_frame[i], scene.end_frame[i] + 1))
        assert scene.end_frame[i] <= scene.config.n_frames - 1
    assert np.all(scene.in_frame)                 # in-frame interval equals the active interval


def test_straight_segments_recover_constructed_speed():
    scene = _scene()
    fps, checked = scene.config.fps, 0
    for i, pid in enumerate(range(1, scene.config.n_pedestrians + 1)):
        pos = scene.foot_world[scene.ped_id == pid]
        frames = scene.frame[scene.ped_id == pid]
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(scene.waypoints[i], axis=0), axis=1))])
        s = scene.speed_mps[i] * (frames - scene.start_frame[i]) / fps
        segment = np.searchsorted(cum, s, side="right")
        same = segment[1:] == segment[:-1]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)[same]
        np.testing.assert_allclose(steps * fps, scene.speed_mps[i], rtol=0, atol=1e-9)
        checked += int(same.sum())
    assert checked > 100
    assert np.all((scene.speed_mps >= 1.0) & (scene.speed_mps <= 1.6))


def test_foot_point_consistent_with_box():
    scene = _scene()
    cfg, camera = scene.config, scene.config.camera
    foot_xyz = np.column_stack([scene.foot_world, np.zeros(len(scene.foot_world))])
    np.testing.assert_allclose(scene.foot_image, _independent_projection(camera, foot_xyz), atol=TOL_PX)
    x1, y1, x2, y2 = scene.bbox_xyxy.T
    u, v = scene.foot_image.T
    assert np.all((x1 <= u) & (u <= x2) & (y1 <= v) & (v <= y2))
    # Recompute every box independently from the cuboid corners.
    r, hgt = cfg.pedestrian_half_width_m, cfg.pedestrian_height_m
    corners = np.array([[sx * r, sy * r, z] for z in (0.0, hgt) for sx in (-1, 1) for sy in (-1, 1)])
    uv = _independent_projection(camera, foot_xyz[:, None, :] + corners[None, :, :])
    expected = np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)
    np.testing.assert_allclose(scene.bbox_xyxy, expected, atol=TOL_PX)
    w, h = camera.image_width, camera.image_height
    np.testing.assert_array_equal(scene.truncated, (x1 < 0) | (y1 < 0) | (x2 > w) | (y2 > h))


def test_foot_points_stay_inside_the_margin_footprint():
    scene = _scene()
    m, w, h = scene.config.footprint_margin_px, scene.config.camera.image_width, scene.config.camera.image_height
    u, v = scene.foot_image.T
    eps = 1e-6
    assert np.all((u >= m - eps) & (u <= w - m + eps) & (v >= m - eps) & (v <= h - m + eps))
    assert ground_footprint(scene.config).shape == (4, 2)


def test_stationary_markers_constant_in_world_and_image():
    scene = _scene()
    for marker_id, world in enumerate(MARKERS, start=1):
        rows = scene.marker_id == marker_id
        assert rows.sum() == scene.config.n_frames
        np.testing.assert_array_equal(scene.marker_world[rows], np.tile(world, (rows.sum(), 1)))
        images = scene.marker_image[rows]
        np.testing.assert_array_equal(images, np.tile(images[0], (rows.sum(), 1)))
        expected = _independent_projection(scene.config.camera, np.array([[*world, 0.0]]))[0]
        assert np.linalg.norm(images[0] - expected) <= TOL_PX


def test_save_npz_round_trip(tmp_path):
    scene = _scene()
    path = scene.save_npz(tmp_path / "scene.npz")
    with np.load(path) as data:
        meta = json.loads(str(data["metadata"]))
        assert meta["content_hash"] == scene.content_hash()
        assert meta["frames_per_step"] == 10
        for name, arr in scene.arrays().items():
            np.testing.assert_array_equal(data[name], arr, err_msg=name)
    with pytest.raises(FileExistsError):
        scene.save_npz(path)


@pytest.mark.parametrize("override", [{"fps": 0}, {"speed_range_mps": (1.6, 1.0)},
                                      {"waypoints_per_path": (1, 3)}, {"footprint_margin_px": 400.0},
                                      {"pedestrian_height_m": 0.0}])
def test_invalid_scene_config_is_rejected(override):
    with pytest.raises(SceneError):
        SceneConfig(seed=1, **override)
