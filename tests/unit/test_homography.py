"""Static homography: load-time singularity check (spec 4.3.1), transform guards and the
synthetic walker-speed property test (spec 11.2)."""

import numpy as np
import pytest

from fixtures import synthetic_geometry as sg
from trackvision import config
from trackvision.experiments import smoke
from trackvision.geometry.homography import (SWAP_UV, SingularHomographyError, StaticTransform,
                                             TransformError, apply_homography, check_homography)

CFG = config.validate().config
TOL_PX = CFG.frozen["geometry"]["roundtrip_tolerance_px"]["synthetic_float64"]


@pytest.mark.parametrize("matrix", [
    np.zeros((3, 3)),
    np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]]),     # rank 2
    np.diag([1.0, 1.0, 1e-17]),                                         # numerically singular
    np.eye(2),
    np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
])
def test_singular_or_malformed_homography_raises_at_load(matrix):
    with pytest.raises(SingularHomographyError):
        StaticTransform(matrix)


def test_loaded_homography_is_read_only():
    h = check_homography(np.eye(3))
    with pytest.raises(ValueError):
        h[0, 0] = 2.0


def test_point_at_infinity_raises():
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, -1.0]])   # w = x - 1
    with pytest.raises(TransformError):
        apply_homography(h, [[1.0, 5.0]])


def test_swap_matrix_turns_uv_into_row_col():
    uv = np.array([3.0, 7.0, 1.0])
    np.testing.assert_array_equal(SWAP_UV @ uv, [7.0, 3.0, 1.0])
    np.testing.assert_array_equal(SWAP_UV @ SWAP_UV, np.eye(3))


def test_downward_camera_reverses_orientation():
    camera = sg.static_camera()
    uv = sg.project(camera["h_ground_to_img"], sg.world_grid(camera))
    # The fixture world is right-handed with z up, seen from above by a y-down image.
    assert np.all(StaticTransform(camera["h_img_to_world"]).orientation_signs(uv) < 0)


def test_synthetic_walker_speed_is_recovered_through_the_camera():
    """Spec 11.2 'Synthetic walker speed': recovered speed equals constructed speed."""
    n_points, dt = smoke.static_window_shape(CFG)
    walker = smoke.constant_velocity_walker(n_points, dt) + np.array([2.0, -1.0])
    camera = sg.static_camera()
    uv = sg.project(camera["h_ground_to_img"], walker)
    world = StaticTransform(camera["h_img_to_world"]).image_to_world(uv)
    speeds = np.linalg.norm(np.diff(world, axis=0), axis=1) / dt
    tol_mps = 2 * TOL_PX * sg.metres_per_pixel(camera["h_img_to_world"], uv) / dt
    assert np.max(np.abs(speeds - smoke.WALKER_SPEED_MPS)) <= tol_mps
