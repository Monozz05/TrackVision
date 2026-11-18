"""Moving-platform stage 2: the spec 11.2 runtime assertion 'no NaNs after any transform'."""

import numpy as np
import pytest

from fixtures import synthetic_geometry as sg
from trackvision.geometry.homography import TransformError
from trackvision.geometry.moving import MovingTransform, platform_to_world, world_to_platform

POINT = np.array([[5.0, 1.0]])


@pytest.mark.parametrize("theta, t", [
    (np.nan, (0.0, 0.0)),
    (0.3, (np.nan, 0.0)),
    (0.3, (0.0, np.inf)),
    (np.inf, (1.0, 2.0)),
])
@pytest.mark.parametrize("stage2", [platform_to_world, world_to_platform])
def test_non_finite_pose_raises(stage2, theta, t):
    with pytest.raises(TransformError):
        stage2(POINT, theta, np.asarray(t))


def test_non_finite_pose_raises_through_the_composition():
    camera = sg.platform_camera()
    moving = MovingTransform(camera["h_img_to_platform"])
    uv = sg.project(camera["h_ground_to_img"], POINT)
    with pytest.raises(TransformError):
        moving.image_to_world(uv, np.nan, np.zeros(2))
    with pytest.raises(TransformError):
        moving.world_to_image(POINT, 0.0, np.array([np.nan, 0.0]))


def test_finite_pose_is_unchanged_by_the_guard():
    out = platform_to_world(POINT, np.pi / 2, np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [[0.0, 7.0]], atol=1e-15)
