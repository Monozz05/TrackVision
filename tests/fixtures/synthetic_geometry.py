"""Synthetic camera, homography and ego-pose harness with analytically known answers
(spec v4.15 section 11.2).

Ground truth is built from a pinhole camera model directly, never through the code under test:
a camera at a known position, height, heading and downward pitch over the plane z = 0 gives
H_ground->img = K [r1 r2 -R C]. Camera, pose and motion values are properties of this fixture,
not protocol constants.
"""

import numpy as np

# Fixture camera: 720x576 image, focal length in pixels, mounted above the ground.
IMAGE_SIZE = (720, 576)
FOCAL_PX = 800.0
PRINCIPAL_POINT = (360.0, 288.0)


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# Ground frame heading axes (x forward, y left, z up) to camera axes (x right, y down, z forward).
_HEADING_TO_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def camera_rotation(yaw: float, pitch_down: float) -> np.ndarray:
    """Rotation taking ground-frame vectors to camera-frame vectors."""
    return _rot_x(pitch_down) @ _HEADING_TO_CAMERA @ _rot_z(-yaw)


def ground_to_image_homography(height: float, pitch_down: float, yaw: float = 0.0,
                               position=(0.0, 0.0)) -> np.ndarray:
    k = np.array([[FOCAL_PX, 0.0, PRINCIPAL_POINT[0]], [0.0, FOCAL_PX, PRINCIPAL_POINT[1]], [0.0, 0.0, 1.0]])
    r = camera_rotation(yaw, pitch_down)
    centre = np.array([position[0], position[1], height])
    return k @ np.column_stack([r[:, 0], r[:, 1], -r @ centre])


def project(h_ground_to_img: np.ndarray, xy) -> np.ndarray:
    """Reference projection of ground points; also asserts they are in front of the camera."""
    xy = np.asarray(xy, dtype=np.float64)
    p = xy @ h_ground_to_img[:, :2].T + h_ground_to_img[:, 2]
    assert np.all(p[..., 2] > 0), "fixture point is behind the camera"
    return p[..., :2] / p[..., 2:3]


def static_camera() -> dict:
    """A static scene camera: 12 m up, pitched 50 degrees down, looking along world +x."""
    h_g2i = ground_to_image_homography(height=12.0, pitch_down=np.deg2rad(50.0), yaw=0.3, position=(-4.0, 2.0))
    return {"h_ground_to_img": h_g2i, "h_img_to_world": np.linalg.inv(h_g2i)}


def platform_camera() -> dict:
    """A forward-looking camera on the platform: 1.6 m up, 15 degrees down, on the x axis."""
    h_g2i = ground_to_image_homography(height=1.6, pitch_down=np.deg2rad(15.0))
    return {"h_ground_to_img": h_g2i, "h_img_to_platform": np.linalg.inv(h_g2i)}


def world_grid(camera: dict, n: int = 9) -> np.ndarray:
    """Ground points whose images cover the static camera's frame (from the reference model)."""
    u = np.linspace(20.0, IMAGE_SIZE[0] - 20.0, n)
    v = np.linspace(20.0, IMAGE_SIZE[1] - 20.0, n)
    uv = np.stack(np.meshgrid(u, v), axis=-1).reshape(-1, 2)
    h = camera["h_img_to_world"]
    p = uv @ h[:, :2].T + h[:, 2]
    return p[:, :2] / p[:, 2:3]


def ego_motion(n_frames: int = 30, dt: float = 0.4) -> dict:
    """A non-trivial platform trajectory starting at the identity pose: forward motion at
    1.2 m/s with a sinusoidal yaw of 0.25 rad amplitude, so rotation and translation both vary."""
    t = np.arange(n_frames) * dt
    theta = 0.25 * np.sin(2.0 * np.pi * t / (n_frames * dt))
    speed = 1.2
    x = np.concatenate([[0.0], np.cumsum(speed * dt * np.cos(theta[:-1]))])
    y = np.concatenate([[0.0], np.cumsum(speed * dt * np.sin(theta[:-1]))])
    return {"theta": theta, "translation": np.stack([x, y], axis=-1), "dt": dt}


def world_to_platform_reference(xy_w, theta, translation) -> np.ndarray:
    """Reference inverse ego-pose, written independently with explicit rotation matrices."""
    out = []
    for p, th, t in zip(np.atleast_2d(xy_w), np.atleast_1d(theta), np.atleast_2d(translation)):
        r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        out.append(r.T @ (p - t))
    return np.array(out)


def metres_per_pixel(h_img_to_ground: np.ndarray, uv) -> float:
    """Largest local ground distance per image pixel over the given points (Jacobian norm)."""
    worst = 0.0
    for u, v in np.atleast_2d(uv):
        p = h_img_to_ground @ np.array([u, v, 1.0])
        jac = (h_img_to_ground[:2, :2] * p[2] - np.outer(p[:2], h_img_to_ground[2, :2])) / p[2] ** 2
        worst = max(worst, float(np.linalg.norm(jac, 2)))
    return worst
