"""Grey-level renderer for U1 synthetic scenes (feasibility fixture; no research claim).

Each frame is a 2-D image (height, width) uint8 in the camera's pixel convention (camera.py): pixel
(x, y) covers [x, x+1) x [y, y+1), and its centre (x + 0.5, y + 0.5) decides coverage.

Pedestrians use U1's geometry unchanged: the upright cuboid of half-width
`config.pedestrian_half_width_m` and height `config.pedestrian_height_m` centred on the
ground-contact point, projected with the public `PinholeCamera.project`. Its silhouette is the convex
hull of the 8 projected corners. Silhouettes are painted farthest first (camera depth of the foot point,
ties broken by ped_id), so nearer pedestrians occlude farther ones. Rendered pixels are clipped to the
image, not to the ground-truth boxes.

Additional ground truth per rendered frame:
  - instance map (height, width) int32: 0 = background, otherwise the U1 ped_id (1 .. n_pedestrians)
    of the pedestrian visible at that pixel;
  - silhouette_pixels: pixel centres inside the complete projected silhouette, counted before
    clipping to the image;
  - visible_pixels: instance-map pixels with that ped_id (after clipping and occlusion);
  - visible_fraction = visible_pixels / silhouette_pixels (0.0 if the silhouette covers no pixel
    centre), reflecting both truncation and occlusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.synthetic.scene import SceneGroundTruth

INSTANCE_DTYPE = np.int32
BACKGROUND_ID = 0


class RenderError(ValueError):
    """Raised for an invalid render configuration or pedestrian identifiers."""


@dataclass(frozen=True)
class RenderConfig:
    seed: int = 0
    background_level: float = 110.0      # mean grey level of the ground
    texture_amplitude: float = 12.0      # peak deviation of the static ground texture
    noise_sigma: float = 2.0             # per-frame Gaussian sensor noise, grey levels
    dark_band: tuple[int, int] = (20, 60)
    bright_band: tuple[int, int] = (170, 230)

    def __post_init__(self) -> None:
        if self.noise_sigma < 0 or self.texture_amplitude < 0:
            raise RenderError("noise_sigma and texture_amplitude must be non-negative")
        lo_bg = self.background_level - self.texture_amplitude
        hi_bg = self.background_level + self.texture_amplitude
        (d0, d1), (b0, b1) = self.dark_band, self.bright_band
        if not (0 <= d0 <= d1 < lo_bg and hi_bg < b0 <= b1 <= 255):
            raise RenderError("pedestrian grey-level bands must lie outside the background range "
                              f"[{lo_bg}, {hi_bg}] and inside [0, 255]")


@dataclass(frozen=True)
class RenderedFrame:
    index: int                   # = U1 frame number
    image: np.ndarray            # (H, W) uint8
    instances: np.ndarray        # (H, W) int32, 0 = background, else ped_id
    rows: slice                  # U1 rows of this frame (rows are sorted by frame)
    ped_ids: np.ndarray          # (k,) U1 ped_id per row, in U1 row order
    silhouette_pixels: np.ndarray    # (k,) int64
    visible_pixels: np.ndarray       # (k,) int64
    visible_fraction: np.ndarray     # (k,) float64


def validate_instance_ids(ped_ids) -> np.ndarray:
    """ped_ids must be positive (0 is background) and fit in int32."""
    ids = np.asarray(ped_ids)
    if ids.size and (ids.min() < 1 or ids.max() > np.iinfo(INSTANCE_DTYPE).max):
        raise RenderError(f"pedestrian ids must lie in [1, {np.iinfo(INSTANCE_DTYPE).max}]")
    return ids.astype(INSTANCE_DTYPE)


def convex_hull(points) -> np.ndarray:
    """Counter-clockwise convex hull (monotone chain) of 2-D points (n, 2)."""
    pts = sorted(map(tuple, np.asarray(points, dtype=np.float64)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def rasterize_convex(polygon: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Pixels whose centres lie inside a counter-clockwise convex polygon (boundary included).

    Returns (mask, x0, y0): mask[r, c] covers pixel (x0 + c, y0 + r). Not clipped to any image.
    """
    poly = np.asarray(polygon, dtype=np.float64)
    x0, y0 = int(np.floor(poly[:, 0].min())), int(np.floor(poly[:, 1].min()))
    x1, y1 = int(np.ceil(poly[:, 0].max())), int(np.ceil(poly[:, 1].max()))
    cx = np.arange(x0, x1) + 0.5
    cy = np.arange(y0, y1) + 0.5
    px, py = np.meshgrid(cx, cy)
    mask = np.ones(px.shape, dtype=bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        mask &= (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]) >= -1e-9
    return mask, x0, y0


def paint_far_to_near(shape: tuple[int, int], layers) -> tuple[np.ndarray, dict]:
    """Instance map for layers (depth, ped_id, polygon), painted farthest first, clipped to `shape`.

    Returns the instance map and each ped_id's complete silhouette pixel count (before clipping).
    """
    height, width = shape
    instances = np.zeros(shape, dtype=INSTANCE_DTYPE)
    totals = {}
    for depth, pid, polygon in sorted(layers, key=lambda layer: (-layer[0], layer[1])):
        if pid in totals:
            raise RenderError(f"ped_id {pid} appears twice in one frame")
        mask, x0, y0 = rasterize_convex(polygon)
        totals[pid] = int(mask.sum())
        rows, cols = np.nonzero(mask)
        xs, ys = cols + x0, rows + y0
        keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        instances[ys[keep], xs[keep]] = pid
    return instances, totals


class SceneRenderer:
    """Renders any frame of a U1 scene; deterministic for a fixed scene and RenderConfig."""

    def __init__(self, scene: SceneGroundTruth, config: RenderConfig = RenderConfig()):
        self.scene, self.config = scene, config
        cam = scene.config.camera
        self.shape = (cam.image_height, cam.image_width)
        validate_instance_ids(scene.ped_id)
        base = [config.seed, scene.config.seed]

        rng = np.random.default_rng(np.random.SeedSequence(base + [0]))
        v, u = np.mgrid[0:self.shape[0], 0:self.shape[1]] + 0.5
        lam_u, lam_v = rng.uniform(80.0, 200.0, size=2)
        phi_u, phi_v = rng.uniform(0.0, 2 * np.pi, size=2)
        texture = 0.4 * np.sin(2 * np.pi * u / lam_u + phi_u) + 0.4 * np.sin(2 * np.pi * v / lam_v + phi_v)
        texture += rng.uniform(-0.2, 0.2, size=self.shape)
        self.background = config.background_level + config.texture_amplitude * texture

        rng = np.random.default_rng(np.random.SeedSequence(base + [1]))
        self.levels = {}
        for pid in range(1, scene.config.n_pedestrians + 1):
            band = config.dark_band if rng.random() < 0.5 else config.bright_band
            self.levels[pid] = float(rng.uniform(*band))

        r = scene.config.pedestrian_half_width_m
        hgt = scene.config.pedestrian_height_m
        self._corners = np.array([[sx * r, sy * r, z] for z in (0.0, hgt) for sx in (-1, 1) for sy in (-1, 1)])

    def frame(self, index: int) -> RenderedFrame:
        scene, cam = self.scene, self.scene.config.camera
        if not 0 <= index < scene.config.n_frames:
            raise IndexError(f"frame {index} outside 0 .. {scene.config.n_frames - 1}")
        start, stop = np.searchsorted(scene.frame, [index, index + 1])
        rows = slice(int(start), int(stop))
        ped_ids = scene.ped_id[rows]
        feet = scene.foot_world[rows]

        layers = []
        for pid, foot in zip(ped_ids, feet):
            base = np.array([foot[0], foot[1], 0.0])
            hull = convex_hull(cam.project(base + self._corners))
            depth = float(((base - cam.centre) @ cam.rotation.T)[2])
            layers.append((depth, int(pid), hull))
        instances, totals = paint_far_to_near(self.shape, layers)

        image = self.background.copy()
        for pid in ped_ids:
            image[instances == pid] = self.levels[int(pid)]
        if self.config.noise_sigma > 0:
            rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, scene.config.seed, 2, index]))
            image = image + rng.normal(0.0, self.config.noise_sigma, size=self.shape)
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)

        counts = np.bincount(instances.ravel(), minlength=int(ped_ids.max()) + 1 if len(ped_ids) else 1)
        silhouette = np.array([totals[int(p)] for p in ped_ids], dtype=np.int64)
        visible = np.array([counts[int(p)] for p in ped_ids], dtype=np.int64)
        fraction = np.divide(visible, silhouette, out=np.zeros(len(ped_ids)), where=silhouette > 0)
        return RenderedFrame(index, image, instances, rows, ped_ids.copy(), silhouette, visible, fraction)

    def __iter__(self):
        for index in range(self.scene.config.n_frames):
            yield self.frame(index)


def quality_report(frame: RenderedFrame, bbox_xyxy: np.ndarray, ring_px: int = 4) -> np.ndarray:
    """Contrast per row: |mean pedestrian grey level - mean background grey level in a ring around its
    ground-truth box|. NaN where the pedestrian is not visible or no background pixel is in the ring."""
    height, width = frame.image.shape
    img = frame.image.astype(np.float64)
    out = np.full(len(frame.ped_ids), np.nan)
    for k, (pid, box) in enumerate(zip(frame.ped_ids, bbox_xyxy)):
        body = frame.instances == pid
        if not body.any():
            continue
        x0 = max(int(np.floor(box[0])) - ring_px, 0)
        y0 = max(int(np.floor(box[1])) - ring_px, 0)
        x1 = min(int(np.ceil(box[2])) + ring_px, width)
        y1 = min(int(np.ceil(box[3])) + ring_px, height)
        region = frame.instances[y0:y1, x0:x1] == BACKGROUND_ID
        if not region.any():
            continue
        out[k] = abs(img[body].mean() - img[y0:y1, x0:x1][region].mean())
    return out
