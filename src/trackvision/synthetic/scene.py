"""Deterministic waypoint-walker scenes with analytically defined ground truth.

Feasibility fixture only: the motion model, speeds and sizes below are properties of this fixture,
not research choices.

Motion. Each pedestrian has one speed drawn uniformly from `speed_range_mps` and walks a
piecewise-linear path through its waypoints, turning instantly at each waypoint. Its position at frame
f is computed analytically from arc length s = speed * (f - start_frame) / fps. A pedestrian is active
from its start frame until it reaches its last waypoint or the scene ends; it then disappears.

Visibility. Waypoints are sampled inside the ground footprint of the image shrunk by
`footprint_margin_px`. That footprint is convex (the camera rejects a horizon inside the image), so
every straight segment, and therefore every foot point, stays inside it: each pedestrian is in frame
over one contiguous run of frames, equal to its active interval. Bounding boxes may still extend past
the image edge; they are kept unclipped and flagged `truncated`.

Sampling grid. `frames_per_step` = sampling_interval_s * fps must be an integer (10 at 25 fps and
0.4 s). Grid step n is frame n * frames_per_step, one global grid anchored at frame 0.

Arrays. Rows are sorted by (frame, ped_id). Image boxes are xyxy in the camera's pixel convention
(camera.py). All arrays are read-only.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trackvision.config import canonical_json
from trackvision.synthetic.camera import PinholeCamera

# Fixture defaults: the camera mirrors tests/fixtures/synthetic_geometry.py (720x576, f = 800 px,
# principal point at the image centre, 12 m high, pitched 50 degrees down).
DEFAULT_CAMERA = PinholeCamera(image_width=720, image_height=576, focal_px=800.0,
                               principal_point=(360.0, 288.0), height_m=12.0,
                               pitch_down_rad=float(np.deg2rad(50.0)))

ARRAY_DTYPES = {
    "frame": "<i8", "ped_id": "<i8", "foot_world": "<f8", "foot_image": "<f8", "bbox_xyxy": "<f8",
    "in_frame": "|b1", "truncated": "|b1", "speed_mps": "<f8", "start_frame": "<i8",
    "end_frame": "<i8", "waypoint_counts": "<i8", "waypoints_flat": "<f8",
    "marker_frame": "<i8", "marker_id": "<i8", "marker_world": "<f8", "marker_image": "<f8",
}


class SceneError(ValueError):
    """Raised for an invalid scene configuration."""


@dataclass(frozen=True)
class SceneConfig:
    seed: int
    fps: int = 25
    n_frames: int = 750
    n_pedestrians: int = 12
    waypoints_per_path: tuple[int, int] = (3, 5)          # inclusive range, start point included
    speed_range_mps: tuple[float, float] = (1.0, 1.6)
    pedestrian_height_m: float = 1.75
    pedestrian_half_width_m: float = 0.25
    footprint_margin_px: float = 40.0
    markers_xy: tuple[tuple[float, float], ...] = ()
    camera: PinholeCamera = field(default=DEFAULT_CAMERA)

    def __post_init__(self) -> None:
        def pos_int(v):
            return isinstance(v, int) and not isinstance(v, bool) and v >= 1
        if not (pos_int(self.fps) and pos_int(self.n_frames)):
            raise SceneError("fps and n_frames must be positive integers")
        if not (isinstance(self.n_pedestrians, int) and self.n_pedestrians >= 0):
            raise SceneError("n_pedestrians must be a non-negative integer")
        lo, hi = self.waypoints_per_path
        if not (isinstance(lo, int) and isinstance(hi, int) and 2 <= lo <= hi):
            raise SceneError("waypoints_per_path must be integers with 2 <= low <= high")
        s_lo, s_hi = self.speed_range_mps
        if not 0.0 < s_lo <= s_hi:
            raise SceneError("speed_range_mps must satisfy 0 < low <= high")
        if not (self.pedestrian_height_m > 0 and self.pedestrian_half_width_m > 0):
            raise SceneError("pedestrian size must be positive")
        m = self.footprint_margin_px
        if not 0.0 <= m < min(self.camera.image_width, self.camera.image_height) / 2:
            raise SceneError("footprint_margin_px must be non-negative and less than half the image size")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class SceneGroundTruth:
    config: SceneConfig
    sampling_interval_s: float
    frames_per_step: int
    # pedestrian rows (N), sorted by (frame, ped_id)
    frame: np.ndarray            # (N,) int64
    ped_id: np.ndarray           # (N,) int64, 1 .. n_pedestrians
    foot_world: np.ndarray       # (N, 2) metres, ground-contact point (centre of the body's footprint)
    foot_image: np.ndarray       # (N, 2) pixels
    bbox_xyxy: np.ndarray        # (N, 4) pixels, unclipped
    in_frame: np.ndarray         # (N,) bool: foot point inside [0, W] x [0, H]
    truncated: np.ndarray        # (N,) bool: box extends past the image
    # per pedestrian (P), index i is ped_id i + 1
    speed_mps: np.ndarray        # (P,)
    start_frame: np.ndarray      # (P,)
    end_frame: np.ndarray        # (P,) last active frame
    waypoints: tuple             # P arrays of shape (k_i, 2), metres
    # stationary markers (n_frames * M rows)
    marker_frame: np.ndarray
    marker_id: np.ndarray
    marker_world: np.ndarray     # (n_frames * M, 2)
    marker_image: np.ndarray     # (n_frames * M, 2)

    def metadata(self) -> dict:
        return {"config": self.config.as_dict(), "sampling_interval_s": self.sampling_interval_s,
                "frames_per_step": self.frames_per_step}

    def arrays(self) -> dict[str, np.ndarray]:
        counts = np.array([len(w) for w in self.waypoints], dtype=np.int64)
        flat = np.concatenate(self.waypoints) if self.waypoints else np.empty((0, 2))
        out = {name: getattr(self, name) for name in ARRAY_DTYPES
               if name not in ("waypoint_counts", "waypoints_flat")}
        out["waypoint_counts"], out["waypoints_flat"] = counts, flat
        return {name: np.ascontiguousarray(out[name], dtype=ARRAY_DTYPES[name]) for name in ARRAY_DTYPES}

    def content_hash(self) -> str:
        """SHA-256 over the canonical metadata and every array (fixed order, dtype, byte order, shape)."""
        digest = hashlib.sha256(canonical_json(self.metadata()).encode("utf-8"))
        for name, arr in self.arrays().items():
            digest.update(f"{name}:{arr.dtype.str}:{arr.shape}".encode("utf-8"))
            digest.update(arr.tobytes())
        return digest.hexdigest()

    def save_npz(self, path: Path, overwrite: bool = False) -> Path:
        """Write all arrays plus metadata and hash to `path` (a .npz chosen by the caller)."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("path must end in .npz")
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        meta = {**self.metadata(), "content_hash": self.content_hash()}
        np.savez(path, metadata=np.array(json.dumps(meta, sort_keys=True)), **self.arrays())
        return path


def frames_per_step(fps: int, sampling_interval_s: float) -> int:
    """Frames per sampling step; raises unless the interval is a whole number of frames."""
    step = int(round(sampling_interval_s * fps))
    if step < 1 or abs(step / fps - sampling_interval_s) > 1e-9:
        raise SceneError(f"{sampling_interval_s} s is not a whole number of frames at {fps} fps")
    return step


def ground_footprint(config: SceneConfig) -> np.ndarray:
    """World corners (4, 2) of the image rectangle shrunk by the margin, in image order."""
    m, w, h = config.footprint_margin_px, config.camera.image_width, config.camera.image_height
    corners = np.array([[m, m], [w - m, m], [w - m, h - m], [m, h - m]], dtype=np.float64)
    return config.camera.ground_transform().image_to_world(corners)


def _inside_convex(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    edges = np.roll(polygon, -1, axis=0) - polygon
    rel = points[:, None, :] - polygon[None, :, :]
    cross = edges[None, :, 0] * rel[..., 1] - edges[None, :, 1] * rel[..., 0]
    return np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)


def _sample_in_polygon(rng: np.random.Generator, polygon: np.ndarray, n: int) -> np.ndarray:
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    out = []
    while len(out) < n:
        p = rng.uniform(lo, hi)
        if _inside_convex(polygon, p[None, :])[0]:
            out.append(p)
    return np.array(out)


def _walk(waypoints: np.ndarray, speed: float, fps: int, start: int, n_frames: int):
    seg = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    n_active = int(np.floor(cum[-1] / speed * fps + 1e-9)) + 1
    frames = start + np.arange(n_active, dtype=np.int64)
    frames = frames[frames < n_frames]
    s = np.minimum(speed * (frames - start) / fps, cum[-1])
    j = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_len) - 1)
    pos = waypoints[j] + ((s - cum[j]) / seg_len[j])[:, None] * seg[j]
    return frames, pos


def _boxes(camera: PinholeCamera, foot: np.ndarray, half_width: float, height: float) -> np.ndarray:
    """xyxy of the projected upright, world-axis-aligned cuboid centred on each foot point."""
    dx = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * half_width
    dy = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * half_width
    dz = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * height
    corners = np.stack([foot[:, None, 0] + dx, foot[:, None, 1] + dy,
                        np.broadcast_to(dz, (len(foot), 8))], axis=-1)
    uv = camera.project(corners)
    return np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)


def _readonly(*arrays):
    for a in arrays:
        a.setflags(write=False)


def generate(config: SceneConfig, sampling_interval_s: float) -> SceneGroundTruth:
    """Generate the scene. `sampling_interval_s` comes from the protocol configuration."""
    step = frames_per_step(config.fps, sampling_interval_s)
    camera = config.camera
    rng = np.random.default_rng(config.seed)
    footprint = ground_footprint(config)

    speeds, starts, ends, paths, rows = [], [], [], [], []
    for index in range(config.n_pedestrians):
        speed = float(rng.uniform(*config.speed_range_mps))
        k = int(rng.integers(config.waypoints_per_path[0], config.waypoints_per_path[1] + 1))
        start = int(rng.integers(0, config.n_frames))
        waypoints = _sample_in_polygon(rng, footprint, k)
        while np.any(np.linalg.norm(np.diff(waypoints, axis=0), axis=1) <= 1e-9):
            waypoints = _sample_in_polygon(rng, footprint, k)
        frames, pos = _walk(waypoints, speed, config.fps, start, config.n_frames)
        speeds.append(speed)
        starts.append(start)
        ends.append(int(frames[-1]))
        paths.append(waypoints)
        rows.append((frames, np.full(len(frames), index + 1, dtype=np.int64), pos))

    if rows:
        frame = np.concatenate([r[0] for r in rows])
        ped_id = np.concatenate([r[1] for r in rows])
        foot_world = np.concatenate([r[2] for r in rows])
    else:
        frame, ped_id, foot_world = np.empty(0, np.int64), np.empty(0, np.int64), np.empty((0, 2))
    order = np.lexsort((ped_id, frame))
    frame, ped_id, foot_world = frame[order], ped_id[order], foot_world[order]

    foot_xyz = np.column_stack([foot_world, np.zeros(len(foot_world))])
    foot_image = camera.project(foot_xyz) if len(foot_world) else np.empty((0, 2))
    bbox = (_boxes(camera, foot_world, config.pedestrian_half_width_m, config.pedestrian_height_m)
            if len(foot_world) else np.empty((0, 4)))
    w, h = camera.image_width, camera.image_height
    in_frame = ((foot_image[:, 0] >= 0) & (foot_image[:, 0] <= w)
                & (foot_image[:, 1] >= 0) & (foot_image[:, 1] <= h))
    truncated = (bbox[:, 0] < 0) | (bbox[:, 1] < 0) | (bbox[:, 2] > w) | (bbox[:, 3] > h)

    markers = np.asarray(config.markers_xy, dtype=np.float64).reshape(-1, 2)
    m = len(markers)
    marker_frame = np.repeat(np.arange(config.n_frames, dtype=np.int64), m)
    marker_id = np.tile(np.arange(1, m + 1, dtype=np.int64), config.n_frames)
    marker_world = np.tile(markers, (config.n_frames, 1))
    marker_image = (camera.project(np.column_stack([marker_world, np.zeros(len(marker_world))]))
                    if m else np.empty((0, 2)))

    speed_arr = np.array(speeds, dtype=np.float64)
    start_arr = np.array(starts, dtype=np.int64)
    end_arr = np.array(ends, dtype=np.int64)
    _readonly(frame, ped_id, foot_world, foot_image, bbox, in_frame, truncated, speed_arr, start_arr,
              end_arr, marker_frame, marker_id, marker_world, marker_image, *paths)
    return SceneGroundTruth(
        config=config, sampling_interval_s=float(sampling_interval_s), frames_per_step=step,
        frame=frame, ped_id=ped_id, foot_world=foot_world, foot_image=foot_image, bbox_xyxy=bbox,
        in_frame=in_frame, truncated=truncated, speed_mps=speed_arr, start_frame=start_arr,
        end_frame=end_arr, waypoints=tuple(paths), marker_frame=marker_frame, marker_id=marker_id,
        marker_world=marker_world, marker_image=marker_image,
    )
