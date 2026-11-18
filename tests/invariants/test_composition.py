"""The six mandatory stabilization composition tests (spec v4.15 section 4.3), all strict.

Ground truth comes from the synthetic pinhole harness in tests/fixtures/synthetic_geometry.py,
never from the code under test. Pixel tolerances are the precision-dependent values of spec
section 11.2, read from configs/base.yaml. A metre tolerance is the pixel tolerance times the
largest local ground distance per pixel over the points involved.

"Detectably different" (composition order, negative controls) means larger than DETECTABLE
times the tolerance; the wrong compositions here differ by metres.

The dataset-conversion row is tested on synthetic native conventions. Its instance for the
selected moving-camera dataset cannot be written until that research choice is made, and is
reported as BLOCKED rather than passed.
"""

import numpy as np
import pytest

from fixtures import synthetic_geometry as sg
from trackvision import config
from trackvision.geometry.homography import SWAP_UV, StaticTransform, apply_homography
from trackvision.geometry.moving import (ConventionError, MovingTransform, NativeConvention,
                                         native_to_project, platform_to_world, world_to_platform)

CFG = config.validate().config
TOL_PX = CFG.frozen["geometry"]["roundtrip_tolerance_px"]["synthetic_float64"]
DETECTABLE = 1e3


def _tol_m(h_img_to_ground, uv) -> float:
    return TOL_PX * sg.metres_per_pixel(h_img_to_ground, uv)


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


PLATFORM = sg.platform_camera()
POSE = (0.6, np.array([5.0, -3.0]))                     # non-trivial: rotation and translation
PLATFORM_POINTS = np.array([[5.0, 0.0], [8.0, 2.0], [12.0, -3.0], [20.0, 1.0]])


@pytest.mark.phase1
def test_composition_order():
    uv = sg.project(PLATFORM["h_ground_to_img"], PLATFORM_POINTS)
    moving = MovingTransform(PLATFORM["h_img_to_platform"])
    theta, t = POSE
    tol = _tol_m(PLATFORM["h_img_to_platform"], uv)

    correct = moving.image_to_world(uv, theta, t)
    expected = PLATFORM_POINTS @ _rot(theta).T + t
    assert np.max(np.linalg.norm(correct - expected, axis=1)) <= tol

    reversed_order = apply_homography(PLATFORM["h_img_to_platform"], platform_to_world(uv, theta, t))
    assert np.min(np.linalg.norm(reversed_order - correct, axis=1)) > DETECTABLE * tol


@pytest.mark.phase1
def test_identity_pose():
    u, v = np.meshgrid(np.linspace(20, 700, 12), np.linspace(300, 560, 8))
    uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    static = StaticTransform(PLATFORM["h_img_to_platform"])
    moving = MovingTransform(PLATFORM["h_img_to_platform"])

    world_static = static.image_to_world(uv)
    np.testing.assert_array_equal(moving.image_to_world(uv, 0.0, np.zeros(2)), world_static)
    np.testing.assert_array_equal(moving.world_to_image(world_static, 0.0, np.zeros(2)),
                                  static.world_to_image(world_static))


@pytest.mark.phase1
@pytest.mark.parametrize("theta, t", [(0.0, (0.0, 0.0)), (np.pi / 2, (2.0, 3.0)), (-2.3, (-4.0, 1.5))])
def test_frame_convention(theta, t):
    t = np.asarray(t)
    ahead, ahead_left = np.array([1.0, 0.0]), np.array([1.0, 1.0])
    uv = sg.project(PLATFORM["h_ground_to_img"], np.stack([ahead, ahead_left]))
    # The synthetic camera itself follows P = (x forward, y left): straight ahead is on the
    # centre column, and left of it is left in the image.
    assert abs(uv[0, 0] - sg.PRINCIPAL_POINT[0]) <= TOL_PX
    assert uv[1, 0] < sg.PRINCIPAL_POINT[0]

    world = MovingTransform(PLATFORM["h_img_to_platform"]).image_to_world(uv, theta, t)
    tol = _tol_m(PLATFORM["h_img_to_platform"], uv)
    expected_ahead = t + np.array([np.cos(theta), np.sin(theta)])
    expected_ahead_left = t + _rot(theta) @ ahead_left
    assert np.linalg.norm(world[0] - expected_ahead) <= tol
    assert np.linalg.norm(world[1] - expected_ahead_left) <= tol


def _rz3(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Synthetic native conventions, as a dataset might use them.
RIGHT_HANDED_MM = NativeConvention(
    name="synthetic: world (y, -x, z), body (x right, y forward, z up), millimetres",
    world_axes=((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    platform_axes=((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    metres_per_unit=0.001,
)
LEFT_HANDED_M = NativeConvention(
    name="synthetic: left-handed world and body (y axis flipped), metres",
    world_axes=((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    platform_axes=((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    metres_per_unit=1.0,
)


def _native_poses(motion: dict, convention: NativeConvention):
    """Express known project poses in a native convention, with a native world origin and
    heading that differ from the initial platform pose (which the conversion must remove)."""
    a_w = np.asarray(convention.world_axes, dtype=float)
    a_p = np.asarray(convention.platform_axes, dtype=float)
    g_rot, g_t = _rz3(1.1), np.array([100.0, -50.0, 2.0])
    rotations, translations = [], []
    for theta, (x, y) in zip(motion["theta"], motion["translation"]):
        r_p = g_rot @ _rz3(theta)
        t_p = g_rot @ np.array([x, y, 0.0]) + g_t
        rotations.append(a_w.T @ r_p @ a_p)
        translations.append(a_w.T @ t_p / convention.metres_per_unit)
    return np.array(rotations), np.array(translations)


@pytest.mark.phase1
@pytest.mark.parametrize("convention", [RIGHT_HANDED_MM, LEFT_HANDED_M], ids=["right_handed_mm", "left_handed_m"])
def test_dataset_conversion(convention):
    motion = sg.ego_motion()
    rotations, translations = _native_poses(motion, convention)

    converted = native_to_project(rotations, translations, convention)
    np.testing.assert_allclose(converted["theta"], motion["theta"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(converted["translation"], motion["translation"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(converted["tilt_rad"], 0.0, atol=1e-7)

    # Negative control: treating the native convention as if it were the project convention.
    as_is = NativeConvention("as-is", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1.0)
    wrong = native_to_project(rotations, translations, as_is)
    assert np.max(np.linalg.norm(wrong["translation"] - motion["translation"], axis=1)) > DETECTABLE * 1e-9


@pytest.mark.phase1
def test_dataset_conversion_rejects_handedness_mismatch():
    motion = sg.ego_motion()
    rotations, translations = _native_poses(motion, RIGHT_HANDED_MM)
    mismatched = NativeConvention("mismatch", RIGHT_HANDED_MM.world_axes, LEFT_HANDED_M.platform_axes, 0.001)
    with pytest.raises(ConventionError):
        native_to_project(rotations, translations, mismatched)


@pytest.mark.phase1
def test_dataset_conversion_static_row_col_homography():
    """Static instance: a native homography taking image (row, column) input, as measured for
    EWAP, is converted to the project (u, v) convention and reproduces the known mapping."""
    camera = sg.static_camera()
    h_native = camera["h_img_to_world"] @ SWAP_UV       # input [row, col, 1]
    converted = StaticTransform(h_native @ SWAP_UV)
    world = sg.world_grid(camera)
    uv = sg.project(camera["h_ground_to_img"], world)
    tol = _tol_m(camera["h_img_to_world"], uv)
    assert np.max(np.linalg.norm(converted.image_to_world(uv) - world, axis=1)) <= tol
    unconverted = StaticTransform(h_native)
    assert np.max(np.linalg.norm(unconverted.image_to_world(uv) - world, axis=1)) > DETECTABLE * tol


@pytest.mark.phase1
def test_dataset_conversion_selected_moving_dataset():
    """Spec 4.3: the conversion is implemented per dataset. Blocked until `moving_dataset` is set."""
    try:
        dataset = CFG.research_choice("moving_dataset")
    except config.UnresolvedChoiceError:
        pytest.skip("BLOCKED: research choice 'moving_dataset' is unresolved (spec 6.3, 6.4); the "
                    "dataset-specific native-pose conversion test cannot be written yet")
    pytest.fail(f"moving_dataset is {dataset!r} but no dataset-specific conversion test exists")


@pytest.mark.phase1
def test_static_object_invariance():
    motion = sg.ego_motion()
    stationary = np.array([25.0, 2.0])
    n = len(motion["theta"])
    platform_xy = sg.world_to_platform_reference(np.tile(stationary, (n, 1)), motion["theta"], motion["translation"])
    uv = sg.project(PLATFORM["h_ground_to_img"], platform_xy)

    # The ego-motion is non-trivial: the platform turns and travels, and the point moves in the image.
    assert np.ptp(motion["theta"]) > 0.4
    assert np.linalg.norm(motion["translation"][-1]) > 10.0
    assert np.linalg.norm(uv[-1] - uv[0]) > 100.0
    assert np.all((uv >= 0) & (uv < np.asarray(sg.IMAGE_SIZE))), "the point should stay in view"

    moving = MovingTransform(PLATFORM["h_img_to_platform"])
    world = moving.image_to_world(uv, motion["theta"], motion["translation"])
    tol = _tol_m(PLATFORM["h_img_to_platform"], uv)
    speed = np.linalg.norm(np.diff(world, axis=0), axis=1) / motion["dt"]
    assert np.max(np.linalg.norm(world - stationary, axis=1)) <= tol
    assert np.max(speed) <= 2 * tol / motion["dt"]

    # Negative control: the inverse ego-pose in place of the forward one makes the point move.
    wrong = world_to_platform(moving.stage1.image_to_world(uv), motion["theta"], motion["translation"])
    wrong_speed = np.linalg.norm(np.diff(wrong, axis=0), axis=1) / motion["dt"]
    assert np.max(wrong_speed) > DETECTABLE * 2 * tol / motion["dt"]


@pytest.mark.phase1
def test_round_trip():
    camera = sg.static_camera()
    static = StaticTransform(camera["h_img_to_world"])
    world = sg.world_grid(camera)
    uv_reference = sg.project(camera["h_ground_to_img"], world)
    tol_m = _tol_m(camera["h_img_to_world"], uv_reference)

    errors = static.roundtrip_errors(world)
    assert errors["px"] <= TOL_PX
    assert errors["m"] <= tol_m
    # The inverse is the reference projection, not merely self-consistent.
    assert np.max(np.linalg.norm(static.world_to_image(world) - uv_reference, axis=1)) <= TOL_PX

    motion = sg.ego_motion()
    n = len(motion["theta"])
    platform_xy = np.column_stack([np.linspace(4.0, 25.0, n), np.linspace(-3.0, 3.0, n)])
    world_moving = platform_to_world(platform_xy, motion["theta"], motion["translation"])
    moving = MovingTransform(PLATFORM["h_img_to_platform"])
    uv_moving = sg.project(PLATFORM["h_ground_to_img"], platform_xy)
    errors = moving.roundtrip_errors(world_moving, motion["theta"], motion["translation"])
    assert errors["px"] <= TOL_PX
    assert errors["m"] <= _tol_m(PLATFORM["h_img_to_platform"], uv_moving)
    assert np.max(np.linalg.norm(moving.world_to_image(world_moving, motion["theta"], motion["translation"])
                                 - uv_moving, axis=1)) <= TOL_PX
