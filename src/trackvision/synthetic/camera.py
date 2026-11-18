"""Static pinhole camera over the ground plane z = 0.

Conventions (project-wide, spec v4.15 section 4.3):
  - World W: right-handed, z up, metres. The ground plane is z = 0.
  - Camera axes: x right, y down, z forward (optical axis).
  - Image: continuous pixel coordinates (u, v) = (column, row), origin at the top-left corner of the
    image, v downward. The image covers [0, width] x [0, height].
  - The camera stands at (position_xy, height_m), looks along heading `yaw_rad` (measured from world
    +x towards +y) and is pitched down by `pitch_down_rad` below the horizontal. No roll.

Ground-plane mapping is delegated to geometry.homography.StaticTransform; this module only derives
H_ground->img = K [r1 r2 -R C] from the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.geometry.homography import StaticTransform, check_homography


class CameraError(ValueError):
    """Raised for invalid camera parameters or points that cannot be projected."""


# Heading frame (x forward, y left, z up) to camera axes (x right, y down, z forward).
_HEADING_TO_CAMERA = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class PinholeCamera:
    image_width: int
    image_height: int
    focal_px: float
    principal_point: tuple[float, float]
    height_m: float
    pitch_down_rad: float
    yaw_rad: float = 0.0
    position_xy: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if not (isinstance(self.image_width, int) and isinstance(self.image_height, int)
                and self.image_width > 0 and self.image_height > 0):
            raise CameraError("image size must be positive integers")
        values = (self.focal_px, *self.principal_point, self.height_m, self.pitch_down_rad,
                  self.yaw_rad, *self.position_xy)
        if not all(np.isfinite(values)):
            raise CameraError("camera parameters must be finite")
        if self.focal_px <= 0 or self.height_m <= 0:
            raise CameraError("focal length and camera height must be positive")
        if not 0.0 < self.pitch_down_rad < np.pi / 2:
            raise CameraError("pitch must lie strictly between 0 and 90 degrees below horizontal")
        # Every image row must see the ground, so the ground footprint of the image is a convex
        # quadrilateral and no pixel looks at the sky.
        if self.horizon_row >= 0.0:
            raise CameraError(f"the horizon (row {self.horizon_row:.1f}) lies inside the image; "
                              "increase the pitch or reduce the vertical field of view")

    @property
    def intrinsics(self) -> np.ndarray:
        cx, cy = self.principal_point
        return np.array([[self.focal_px, 0.0, cx], [0.0, self.focal_px, cy], [0.0, 0.0, 1.0]])

    @property
    def rotation(self) -> np.ndarray:
        """R: world vectors to camera vectors."""
        return _rot_x(self.pitch_down_rad) @ _HEADING_TO_CAMERA @ _rot_z(-self.yaw_rad)

    @property
    def centre(self) -> np.ndarray:
        return np.array([self.position_xy[0], self.position_xy[1], self.height_m])

    @property
    def horizon_row(self) -> float:
        """Image row of the horizon (points at infinity on the ground plane)."""
        return float(self.principal_point[1] - self.focal_px * np.tan(self.pitch_down_rad))

    def project(self, points_xyz) -> np.ndarray:
        """Project world points (..., 3) to image points (..., 2); raises if any lies behind the camera."""
        pts = np.asarray(points_xyz, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise CameraError(f"points must have shape (..., 3), got {pts.shape}")
        cam = (pts - self.centre) @ self.rotation.T
        depth = cam[..., 2]
        if np.any(~(depth > 0.0)):
            raise CameraError("a point lies on or behind the camera plane")
        uv = (cam @ self.intrinsics.T)[..., :2] / depth[..., None]
        if not np.all(np.isfinite(uv)):
            raise CameraError("projection produced non-finite coordinates")
        return uv

    def ground_homography(self) -> np.ndarray:
        """H_ground->img for ground points (x, y, 0): K [r1 r2 -R C]."""
        r = self.rotation
        return check_homography(self.intrinsics @ np.column_stack([r[:, 0], r[:, 1], -r @ self.centre]))

    def ground_transform(self) -> StaticTransform:
        """Image <-> ground mapping as the project's StaticTransform (H_img->W)."""
        return StaticTransform(np.linalg.inv(self.ground_homography()))
