"""Static image-to-world homography and its inverse (spec v4.15 sections 4.3 and 4.3.1).

Image points use the project convention: pixels, origin top-left, y downward, given as (u, v) =
(column, row). A transform maps from the named source frame to the named target frame.
"""

from __future__ import annotations

import numpy as np

# [row, col, 1]^T = SWAP_UV @ [u, v, 1]^T. A native homography whose input is (row, column)
# becomes a project homography by right-multiplying with this matrix.
SWAP_UV = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

_EPS = np.finfo(np.float64).eps


class SingularHomographyError(ValueError):
    """Raised at load time for a homography that is not a finite, invertible 3x3 matrix."""


class TransformError(ValueError):
    """Raised when a transform would produce a point at infinity or a non-finite value."""


def check_homography(matrix) -> np.ndarray:
    """Return `matrix` as a read-only float64 3x3 array, or raise if it is singular (spec 4.3.1)."""
    h = np.array(matrix, dtype=np.float64)
    if h.shape != (3, 3):
        raise SingularHomographyError(f"homography must be 3x3, got shape {h.shape}")
    if not np.all(np.isfinite(h)):
        raise SingularHomographyError("homography contains non-finite entries")
    # Numerically singular when the condition number reaches 1/eps: the inverse is then
    # dominated by rounding and would give silently wrong coordinates.
    if not np.linalg.cond(h) < 1.0 / _EPS:
        raise SingularHomographyError(f"homography is singular (condition number {np.linalg.cond(h):.3g})")
    h.setflags(write=False)
    return h


def apply_homography(matrix: np.ndarray, points) -> np.ndarray:
    """Map points (..., 2) through a 3x3 homography and divide by the third component."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[-1] != 2:
        raise ValueError(f"points must have shape (..., 2), got {pts.shape}")
    homog = pts @ matrix[:, :2].T + matrix[:, 2]
    w = homog[..., 2]
    scale = np.max(np.abs(homog), axis=-1)
    if np.any(np.abs(w) <= _EPS * scale):
        raise TransformError("a point maps to infinity (third homogeneous component is zero)")
    out = homog[..., :2] / w[..., None]
    if not np.all(np.isfinite(out)):
        raise TransformError("transform produced non-finite coordinates")
    return out


class StaticTransform:
    """A fixed image-to-ground homography, H_img->W for a static scene (or H_img->P, stage 1 of
    the moving composition), with its inverse."""

    def __init__(self, h_img_to_ground) -> None:
        self.h_img_to_ground = check_homography(h_img_to_ground)
        self.h_ground_to_img = check_homography(np.linalg.inv(self.h_img_to_ground))

    def image_to_world(self, uv) -> np.ndarray:
        return apply_homography(self.h_img_to_ground, uv)

    def world_to_image(self, xy) -> np.ndarray:
        return apply_homography(self.h_ground_to_img, xy)

    def roundtrip_errors(self, world_points) -> dict:
        """World -> image -> world round trip (spec 4.3, 11.2).

        `px` is the largest pixel discrepancy, |image(world(p)) - p| for p = image(x), which is
        compared with the pixel tolerances of spec 11.2; `m` is the largest |world(image(x)) - x|.
        """
        xy = np.asarray(world_points, dtype=np.float64)
        uv = self.world_to_image(xy)
        xy_back = self.image_to_world(uv)
        uv_back = self.world_to_image(xy_back)
        return {
            "px": float(np.max(np.linalg.norm(uv_back - uv, axis=-1))),
            "m": float(np.max(np.linalg.norm(xy_back - xy, axis=-1))),
        }

    def orientation_signs(self, uv) -> np.ndarray:
        """Sign of the Jacobian determinant of image -> ground at each image point.

        For a homography the determinant equals det(H) / w^3. A negative sign means the map
        reverses orientation: with image y downward and a camera above the ground looking down,
        that corresponds to a right-handed ground frame with z up.
        """
        pts = np.asarray(uv, dtype=np.float64)
        w = pts @ self.h_img_to_ground[2, :2] + self.h_img_to_ground[2, 2]
        return np.sign(np.linalg.det(self.h_img_to_ground) * w)
