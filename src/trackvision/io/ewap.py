"""Loader for the ETH and HOTEL scenes of the ETH Walking Pedestrians (EWAP) dataset.

Trajectories are the dataset's own world positions in metres (obsmat.txt columns pos_x, pos_y).
The scene homography H.txt is converted to the project image convention and kept for mapping
image-space tracker output to world (spec v4.15 section 4.3).

Native convention, measured rather than documented: H.txt maps image (row, column) to world.
The conversion is H_project = H_native @ SWAP_UV. Reproducible evidence, recomputed by
trackvision.experiments.geometry_validation from the dataset's own files:
  1. direction: every obsmat position maps back through inv(H) to a whole pixel, which only
     the image -> world direction gives;
  2. order: map.png is the dataset's image-space obstacle mask at the video frame size. Read as
     (row, column), trajectories stay inside the frame and do not cross the obstacles (the ETH
     walkway walls); read as (column, row) they cross them (ETH) or leave the frame (HOTEL).
Supplementary, not reproducible from the repository: visual overlays of annotations on video
frames, inspected outside the repository on 2026-10-29 during the read-only G3 investigation.

Unverified assumptions are returned with every scene in `Scene.assumptions`; they are not
resolved here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trackvision.config import REPO_ROOT, Config
from trackvision.geometry.homography import SWAP_UV, StaticTransform
from trackvision.io import manifest, png

DATASET = "EWAP"
SCENES = {"ETH": "seq_eth", "HOTEL": "seq_hotel"}
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "ewap"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "ewap.sha256"
DATASET_SUBDIR = "ewap_dataset"
# obsmat.txt: [frame_number pedestrian_ID pos_x pos_z pos_y v_x v_z v_y] (dataset README.txt)
OBSMAT_COLUMNS = ("frame", "ped_id", "pos_x", "pos_z", "pos_y", "v_x", "v_z", "v_y")
_TIMESTEP_PATTERN = re.compile(r"timestep of ([0-9]*\.?[0-9]+) seconds")

ASSUMPTIONS = {
    "trajectory_point": (
        "UNVERIFIED. Positions are the dataset authors' world coordinates. Whether the annotated "
        "image point was the ground-contact point required by spec 4.2 is not documented; G3 "
        "overlays show many points on heads or upper bodies, which top-down views cannot resolve."
    ),
    "sampling_interval": (
        "The annotation timestep is taken from info.txt and checked against the frozen sampling "
        "interval. The obsmat velocity columns are centred differences divided by that same "
        "timestep, so they confirm the authors' timestep but are not independent evidence of the "
        "video capture rate (ETH: 6 frames per step in a container declaring 25 fps)."
    ),
    "frame_index_base": (
        "UNVERIFIED. Whether obsmat frame_number equals the decoded video frame index (0- or "
        "1-based) is not documented; the README records a frame-offset correction on 17/09/2009. "
        "Matters only for video synchronisation (spec 6.1.1 check 2), not for trajectories."
    ),
    "world_frame": (
        "Units are metres (README). Origin and axis directions are not documented; the orientation "
        "of the image-to-world map is measured and recorded in transform_config."
    ),
    "h_input_order": (
        "MEASURED, not documented: H.txt input is image (row, column); see the module docstring."
    ),
    "pixel_index_base": (
        "UNVERIFIED. The annotated pixels may be 1-based: HOTEL positions map back to u = 720 in a "
        "720-pixel-wide frame. A one-pixel offset is irrelevant to world trajectories but matters "
        "when tracker output in 0-based pixels is mapped through H (spec 6.1.1)."
    ),
}


class EwapFormatError(ValueError):
    """Raised when an EWAP file does not have the documented structure."""


@dataclass(frozen=True)
class Track:
    ped_id: int
    frames: np.ndarray        # (n,) int64 video frame numbers, strictly increasing
    positions: np.ndarray     # (n, 2) float64 world metres (pos_x, pos_y)
    velocities: np.ndarray    # (n, 2) float64 dataset-provided (v_x, v_y), m/s


@dataclass(frozen=True)
class Scene:
    name: str
    tracks: tuple[Track, ...]
    frame_step: int                       # video frames per annotation timestep
    sampling_interval_s: float
    frame_size: tuple[int, int]           # (width, height) in pixels, from map.png
    obstacle_map: np.ndarray              # (height, width) bool, image-space obstacles (map.png)
    transform: StaticTransform            # H_img->W in the project image convention
    transform_config: dict
    provenance: dict
    diagnostics: dict
    assumptions: dict = field(default_factory=lambda: dict(ASSUMPTIONS))


def _scene_files(name: str) -> dict[str, str]:
    seq = SCENES[name]
    return {kind: f"{DATASET_SUBDIR}/{seq}/{fname}"
            for kind, fname in (("obsmat", "obsmat.txt"), ("homography", "H.txt"), ("info", "info.txt"),
                                ("obstacles", "map.png"))}


def parse_obsmat(path: Path) -> list[Track]:
    data = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if data.shape[1] != len(OBSMAT_COLUMNS) or len(data) == 0:
        raise EwapFormatError(f"{path}: expected {len(OBSMAT_COLUMNS)} columns, got shape {data.shape}")
    if not np.all(np.isfinite(data)):
        raise EwapFormatError(f"{path}: non-finite values")
    if np.any(data[:, 3] != 0) or np.any(data[:, 6] != 0):
        raise EwapFormatError(f"{path}: pos_z / v_z are documented as unused but are non-zero")
    if np.any(data[:, :2] != np.round(data[:, :2])):
        raise EwapFormatError(f"{path}: frame numbers and pedestrian IDs must be integers")
    tracks = []
    for ped_id in np.unique(data[:, 1]).astype(np.int64):
        rows = data[data[:, 1] == ped_id]
        rows = rows[np.argsort(rows[:, 0], kind="stable")]
        frames = rows[:, 0].astype(np.int64)
        if np.any(np.diff(frames) <= 0):
            raise EwapFormatError(f"{path}: pedestrian {ped_id} has duplicate frame numbers")
        tracks.append(Track(int(ped_id), frames, rows[:, [2, 4]].copy(), rows[:, [5, 7]].copy()))
    return tracks


def frame_step_of(tracks: list[Track]) -> int:
    """The annotation step in video frames: the smallest within-track frame difference, which
    every within-track difference must be a multiple of."""
    diffs = np.concatenate([np.diff(t.frames) for t in tracks if len(t.frames) > 1])
    if len(diffs) == 0:
        raise EwapFormatError("no track has two or more annotations")
    step = int(diffs.min())
    if np.any(diffs % step):
        raise EwapFormatError(f"frame differences are not all multiples of the step {step}")
    return step


def documented_timestep(info_path: Path) -> float:
    match = _TIMESTEP_PATTERN.search(Path(info_path).read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise EwapFormatError(f"{info_path}: no 'timestep of <x> seconds' statement")
    return float(match.group(1))


def velocity_timestep(tracks: list[Track], frame_step: int) -> dict:
    """Least-squares timestep from (p[i+1] - p[i-1]) / 2 = v[i] * dt over evenly spaced triples.

    Implementation Plan, Phase 1: derive the timestep from the data, not the file header. See
    ASSUMPTIONS['sampling_interval'] for why this is not independent of the authors' timestep.
    """
    disp, vel = [], []
    for t in tracks:
        for i in range(1, len(t.frames) - 1):
            if t.frames[i + 1] - t.frames[i] == frame_step == t.frames[i] - t.frames[i - 1]:
                disp.append((t.positions[i + 1] - t.positions[i - 1]) / 2.0)
                vel.append(t.velocities[i])
    if not vel:
        return {"timestep_s": None, "triples": 0, "residual_rms_m": None}
    d, v = np.asarray(disp), np.asarray(vel)
    dt = float(np.sum(d * v) / np.sum(v * v))
    residual = d - v * dt
    return {"timestep_s": dt, "triples": int(len(v)),
            "residual_rms_m": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))}


def load_scene(name: str, config: Config, raw_root: Path = DEFAULT_RAW_ROOT,
               manifest_path: Path | None = DEFAULT_MANIFEST) -> Scene:
    """Load one EWAP scene. With `manifest_path`, every file read is first checked against it."""
    if name not in SCENES:
        raise KeyError(f"unknown EWAP scene {name!r}; expected one of {sorted(SCENES)}")
    raw_root = Path(raw_root)
    files = _scene_files(name)
    verified = {}
    manifest_digest = None
    if manifest_path is not None:
        verified = manifest.verify(manifest_path, raw_root, list(files.values()))
        manifest_digest = manifest.manifest_hash(manifest_path)

    sampling_interval = float(config.frozen["protocol"]["static"]["sampling_interval_s"])
    timestep = documented_timestep(raw_root / files["info"])
    if timestep != sampling_interval:
        raise EwapFormatError(f"{name}: documented timestep {timestep} s differs from the frozen "
                              f"sampling interval {sampling_interval} s")

    tracks = parse_obsmat(raw_root / files["obsmat"])
    step = frame_step_of(tracks)
    h_native = np.loadtxt(raw_root / files["homography"], dtype=np.float64)
    transform = StaticTransform(h_native @ SWAP_UV)
    # The README: obstacles are in image coordinates, brought to world with H.txt.
    obstacles = png.read_gray8(raw_root / files["obstacles"]) > 127

    first_frames = np.array([t.frames[0] for t in tracks])
    phases, counts = np.unique(first_frames % step, return_counts=True)
    transform_config = {
        "stabilizer": "static",
        "direction": "image_to_world",
        "source_file": files["homography"],
        "source_sha256": verified.get(files["homography"]),
        "native_input_order": "row_col",
        "project_input_order": "u_v",
        "conversion": "H_project = H_native @ [[0,1,0],[1,0,0],[0,0,1]]",
        "h_native": h_native.tolist(),
        "h_project": transform.h_img_to_ground.tolist(),
        "native_convention_evidence": (
            "measured, not documented by the dataset: whole-pixel inverse (direction) and map.png "
            "obstacle/frame-bounds check (order), recomputed by geometry_validation; supplementary "
            "visual overlays inspected outside the repository on 2026-10-29"),
    }
    return Scene(
        name=name,
        tracks=tuple(tracks),
        frame_step=step,
        sampling_interval_s=sampling_interval,
        frame_size=(int(obstacles.shape[1]), int(obstacles.shape[0])),
        obstacle_map=obstacles,
        transform=transform,
        transform_config=transform_config,
        provenance={
            "dataset": DATASET,
            "scene": name,
            "source_files": files,
            "source_sha256": verified,
            "dataset_manifest_hash": manifest_digest,
            "manifest_verified": manifest_path is not None,
        },
        diagnostics={
            "documented_timestep_s": timestep,
            "velocity_timestep": velocity_timestep(tracks, step),
            "frames_per_step": step,
            "implied_video_fps": step / timestep,
            "track_count": len(tracks),
            "annotation_count": int(sum(len(t.frames) for t in tracks)),
            "frame_grid_phases": {int(p): int(c) for p, c in zip(phases, counts)},
        },
    )
