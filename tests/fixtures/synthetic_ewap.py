"""Tiny synthetic scenes in the EWAP file format (obsmat.txt, H.txt, info.txt, map.png), written
to a temporary directory. The homography is the synthetic static camera of synthetic_geometry.py,
stored with (row, column) input as measured for the real EWAP files. The default obstacle map
is a corridor around the true image path, like the ETH walkway walls: the correct reading never
crosses an obstacle."""

import hashlib

import numpy as np

from fixtures import synthetic_geometry as sg
from fixtures import synthetic_png
from trackvision.geometry.homography import SWAP_UV
from trackvision.io import ewap

DT = 0.4                 # the documented EWAP annotation timestep written to info.txt
FRAME_STEP = 6
CORRIDOR_PX = 15         # half-width of the free corridor around the true image path
INFO = "INFO:\nThe annotation was done at 2.5 fps, that is with a timestep of 0.4 seconds.\n"


def track_rows(ped_id, first_frame, positions):
    """obsmat rows for one pedestrian; velocities are centred differences, as in EWAP."""
    positions = np.asarray(positions, dtype=float)
    vel = np.zeros_like(positions)
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * DT)
    frames = first_frame + FRAME_STEP * np.arange(len(positions))
    return [[f, ped_id, x, 0.0, y, vx, 0.0, vy] for f, (x, y), (vx, vy) in zip(frames, positions, vel)]


def corridor_map(camera: dict, rows) -> np.ndarray:
    """Obstacles everywhere except within CORRIDOR_PX of each pedestrian's true image path."""
    width, height = sg.IMAGE_SIZE
    free = np.zeros((height, width), dtype=bool)
    rows = np.asarray(rows)
    for ped in np.unique(rows[:, 1]):
        uv = sg.project(camera["h_ground_to_img"], rows[rows[:, 1] == ped][:, [2, 4]])
        for a, b in zip(uv[:-1], uv[1:]):
            for p in a + (b - a) * np.linspace(0.0, 1.0, 50)[:, None]:
                u, v = int(round(p[0])), int(round(p[1]))
                free[max(v - CORRIDOR_PX, 0):v + CORRIDOR_PX + 1, max(u - CORRIDOR_PX, 0):u + CORRIDOR_PX + 1] = True
    return np.where(free, 0, 255).astype(np.uint8)


def write_scene(root, name="ETH", rows=None, info=INFO, h_native=None, obstacles=None) -> dict:
    """Write one scene under root/ewap_dataset/<seq>/ and return the synthetic camera."""
    seq = root / ewap.DATASET_SUBDIR / ewap.SCENES[name]
    seq.mkdir(parents=True)
    if rows is None:
        walk = np.column_stack([np.linspace(0.0, 5.7, 20), np.full(20, 1.0)])
        rows = track_rows(1, 780, walk) + track_rows(2, 783, walk[:5] + [0.0, 2.0])
    np.savetxt(seq / "obsmat.txt", np.asarray(rows), fmt="%.10e")
    camera = sg.static_camera()
    np.savetxt(seq / "H.txt", camera["h_img_to_world"] @ SWAP_UV if h_native is None else h_native)
    (seq / "info.txt").write_text(info, encoding="utf-8")
    image = corridor_map(camera, rows) if obstacles is None else obstacles
    (seq / "map.png").write_bytes(synthetic_png.encode_gray8(image))
    return camera


def write_manifest(root, path) -> None:
    """A sha256sum-format manifest of every file under root."""
    lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(root).as_posix()}"
             for f in sorted(p for p in root.rglob("*") if p.is_file())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
