"""Moving-platform stabilization: T_P->W(t) composed with H_img->P, its inverse, and the
conversion of native dataset ego-poses into the project conventions (spec v4.15 section 4.3).

Project frames: platform P is x forward, y left, z up; world W is fixed, right-handed, z up,
with its origin at the sequence's initial platform pose. Stage 2 is planar (yaw about z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.geometry.homography import StaticTransform, TransformError


def _pose(theta, translation) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float64)
    t = np.asarray(translation, dtype=np.float64)
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(t))):
        raise TransformError("ego-pose contains non-finite values")
    return theta, t


def _finite(out: np.ndarray) -> np.ndarray:
    # Spec 11.2 runtime assertion: no NaNs after any transform.
    if not np.all(np.isfinite(out)):
        raise TransformError("ego-pose transform produced non-finite coordinates")
    return out


def platform_to_world(xy_p, theta, translation) -> np.ndarray:
    """Stage 2 of spec 4.3: [x_W, y_W] = R(theta) [x_P, y_P] + t. Broadcasts over leading axes."""
    xy = np.asarray(xy_p, dtype=np.float64)
    theta, t = _pose(theta, translation)
    c, s = np.cos(theta), np.sin(theta)
    x, y = xy[..., 0], xy[..., 1]
    return _finite(np.stack([c * x - s * y + t[..., 0], s * x + c * y + t[..., 1]], axis=-1))


def world_to_platform(xy_w, theta, translation) -> np.ndarray:
    """Inverse of stage 2 (spec 4.3.1): [x_P, y_P] = R(-theta) ([x_W, y_W] - t)."""
    theta, t = _pose(theta, translation)
    xy = np.asarray(xy_w, dtype=np.float64) - t
    c, s = np.cos(theta), np.sin(theta)
    x, y = xy[..., 0], xy[..., 1]
    return _finite(np.stack([c * x + s * y, -s * x + c * y], axis=-1))


class MovingTransform:
    """Image -> world for a rigidly mounted camera: T_P->W(t) o H_img->P, applied right to left.

    `theta` and `translation` give the platform pose T_P->W per point (broadcastable).
    """

    def __init__(self, h_img_to_platform) -> None:
        self.stage1 = StaticTransform(h_img_to_platform)

    def image_to_world(self, uv, theta, translation) -> np.ndarray:
        return platform_to_world(self.stage1.image_to_world(uv), theta, translation)

    def world_to_image(self, xy, theta, translation) -> np.ndarray:
        # The inverse ego-pose is applied before the inverse homography (spec 4.3.1).
        return self.stage1.world_to_image(world_to_platform(xy, theta, translation))

    def roundtrip_errors(self, world_points, theta, translation) -> dict:
        """World -> image -> world, as StaticTransform.roundtrip_errors, under the given poses."""
        xy = np.asarray(world_points, dtype=np.float64)
        uv = self.world_to_image(xy, theta, translation)
        xy_back = self.image_to_world(uv, theta, translation)
        uv_back = self.world_to_image(xy_back, theta, translation)
        return {
            "px": float(np.max(np.linalg.norm(uv_back - uv, axis=-1))),
            "m": float(np.max(np.linalg.norm(xy_back - xy, axis=-1))),
        }


# --------------------------------------------------------------------------------------------
# Dataset conversion (spec 4.3, "Dataset conversion requirement")
# --------------------------------------------------------------------------------------------

class ConventionError(ValueError):
    """Raised when a native pose cannot be expressed in the project conventions."""


def _check_axes(name: str, axes) -> np.ndarray:
    a = np.asarray(axes, dtype=np.float64)
    is_signed_permutation = (
        a.shape == (3, 3)
        and np.all(np.isin(a, (-1.0, 0.0, 1.0)))
        and np.all(np.count_nonzero(a, axis=0) == 1)
        and np.all(np.count_nonzero(a, axis=1) == 1)
    )
    if not is_signed_permutation:
        raise ConventionError(f"{name} must be a 3x3 signed permutation matrix")
    return a


@dataclass(frozen=True)
class NativeConvention:
    """How one dataset expresses ego-pose, relative to the project conventions.

    world_axes:     project world vector = world_axes @ native world vector
    platform_axes:  project platform vector = platform_axes @ native body vector
    metres_per_unit: native length unit in metres (1.0 for metres, 0.001 for millimetres)

    Each dataset gets its own instance, unit-tested against a known synthetic pose and recorded
    in transform_config.
    """

    name: str
    world_axes: tuple
    platform_axes: tuple
    metres_per_unit: float

    def as_record(self) -> dict:
        return {"name": self.name, "world_axes": [list(r) for r in self.world_axes],
                "platform_axes": [list(r) for r in self.platform_axes],
                "metres_per_unit": self.metres_per_unit}


def native_to_project(rotations, translations, convention: NativeConvention,
                      orthonormal_atol: float = 1e-6) -> dict:
    """Convert native platform-to-world poses (T, 3, 3) and (T, 3) into planar project poses.

    Returns `theta` (T,) and `translation` (T, 2) of T_P->W in the project world frame, plus
    `tilt_rad` (T,): the angle between the platform z axis and world z. The world frame takes its
    origin and heading from the initial platform pose and keeps z up from the converted native
    world, so an initial pitch or roll does not tilt it. Stage 2 is planar, so the caller must
    record the tilt; whether a dataset's tilt is negligible is a per-dataset decision and is not
    made here.
    """
    a_w = _check_axes("world_axes", convention.world_axes)
    a_p = _check_axes("platform_axes", convention.platform_axes)
    if not convention.metres_per_unit > 0:
        raise ConventionError("metres_per_unit must be positive")
    r_n = np.asarray(rotations, dtype=np.float64)
    t_n = np.asarray(translations, dtype=np.float64)
    if r_n.ndim != 3 or r_n.shape[1:] != (3, 3) or t_n.shape != (r_n.shape[0], 3) or len(r_n) == 0:
        raise ConventionError("expected rotations (T, 3, 3) and translations (T, 3) with T >= 1")
    if not (np.all(np.isfinite(r_n)) and np.all(np.isfinite(t_n))):
        raise ConventionError("native poses contain non-finite values")

    r_p = a_w @ r_n @ a_p.T
    t_p = (t_n * convention.metres_per_unit) @ a_w.T
    identity = np.broadcast_to(np.eye(3), r_p.shape)
    if not np.allclose(r_p @ np.swapaxes(r_p, 1, 2), identity, atol=orthonormal_atol):
        raise ConventionError("converted rotations are not orthonormal")
    if np.any(np.linalg.det(r_p) <= 0):
        raise ConventionError("converted rotations are improper (det <= 0): a handedness mismatch "
                              "between the native world and body axes")

    yaw = np.arctan2(r_p[:, 1, 0], r_p[:, 0, 0])
    # Re-express in the world frame anchored at the initial position and heading.
    theta = np.angle(np.exp(1j * (yaw - yaw[0])))
    translation = world_to_platform(t_p[:, :2], yaw[0], t_p[0, :2])
    return {
        "theta": theta,
        "translation": translation,
        "tilt_rad": np.arccos(np.clip(r_p[:, 2, 2], -1.0, 1.0)),
    }
