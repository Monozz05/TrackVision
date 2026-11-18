"""Who is where on a scene's video, under which ID, and the box around them (decisions 0010 and 0011).

Drawn tracks are for display. The ground truth of UCY and MOT17 also feeds the Workspace's live errors, in video pixels,
which are exploratory and never saved. This lives in the platform so that it cannot touch `source_sha256`.

Identities come from the best real source each dataset has:
  ETH, HOTEL, UCY   the existing IoU tracker on stored YOLO11 boxes where those exist (decision 0015), so everyone a
                    detector finds gets a box and an ID. With errors on, or with no stored boxes, human annotations,
                    in pixels, whose IDs are the true identities.
  Synthetic         the scene generator.
  MOT17             the existing IoU tracker on detections, so its IDs are real tracker output, switches included.
                    With errors on, MOT17's own ground truth replaces it, because ADE needs true positions.

Tracker output is for display only. It never reaches ADE, FDE or any other measurement.

Boxes. MOT17 boxes are detector output, or annotated boxes for its ground truth. The annotated scenes record one point
per person and no box, so a box there is a display estimate of the person's size around that point, never a ground-truth
box and never used in any measurement. Its median height only sets how close two people must be for an identity switch.
The annotators marked different body parts per scene (feet in UNIV and ZARA1, heads in ZARA2, ETH and HOTEL), which is
why each scene has its own `above`, read from the real frames. Synthetic scenes use their true boxes.

UCY gives positions in pixels with (0, 0) at the image centre and y pointing up, so nothing here needs a calibration
and no world coordinate is inferred. Its control points lie about 1.2 s apart and positions between them are linear.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackvision.io import mot17
from trackvision.synthetic.tracks import GridTrack
from trackvision.tracking.tracker import run_tracker

MOT17_VARIANT = "FRCNN"              # the public detection set the stored tracking results treat as primary
MIN_DETECTION_SCORE = 0.5            # display cut-off: removes short false tracks while keeping most detections
VIDEO_HEIGHT = 720                   # height of the browser copy of a MOT17 sequence
UCY_FRAME_STEP = 10                  # 0.4 s at 25 fps, the same grid as the other scenes
PERSON_CLASS = 0                     # COCO class id in YOLO detection files
EVALUABLE = "evaluable"              # the MOT17 ground-truth rows counted as pedestrians


@dataclass(frozen=True)
class BoxModel:
    """A person-sized box around a point. Size grows linearly from the top of the image to the bottom."""
    width: tuple[float, float]
    height: tuple[float, float]
    above: float                     # share of the box above the point: 1 means the point is the feet, 0 the head
    image_height: float

    def boxes(self, uv) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        t = np.clip(uv[:, 1] / self.image_height, 0.0, 1.0)
        w, h = np.interp(t, (0, 1), self.width), np.interp(t, (0, 1), self.height)
        return np.c_[uv[:, 0] - w / 2, uv[:, 1] - h * self.above, uv[:, 0] + w / 2, uv[:, 1] + h * (1 - self.above)]


# Read from real frames of each video. See the module docstring.
SCENE_BOXES = {
    "ETH": BoxModel((20, 30), (28, 42), 0.25, 480),
    "HOTEL": BoxModel((46, 52), (80, 64), 0.12, 576),      # the camera looks almost straight down near the bottom
    "UNIV": BoxModel((24, 42), (50, 92), 0.92, 576),
    "ZARA1": BoxModel((28, 38), (52, 78), 0.92, 576),
    "ZARA2": BoxModel((28, 38), (52, 78), 0.06, 576),
}


def track_row(track_id: int, frames, uv, boxes, **extra) -> dict:
    return {"id": int(track_id), "frames": np.asarray(frames).astype(int).tolist(), "uv": np.round(uv, 1).tolist(),
            "boxes": np.round(boxes, 1).tolist(), **extra}


def parse_vsp(text: str) -> list[np.ndarray]:
    """UCY splines: one (n, 3) array of x, y, frame per person, in file order. Comments follow ' - '."""
    lines = [line.split(" - ")[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError("empty annotation file")
    people, cursor = [], 1
    for _ in range(int(float(lines[0].split()[0]))):
        count = int(float(lines[cursor].split()[0]))
        rows = [[float(v) for v in line.split()[:3]] for line in lines[cursor + 1:cursor + 1 + count]]
        if len(rows) != count:
            raise ValueError("annotation file ends inside a spline")
        cursor += count + 1
        people.append(np.asarray(rows, dtype=np.float64))
    return people


def ucy_tracks(path: Path, width: int, height: int) -> tuple[GridTrack, ...]:
    """Annotated UCY people in image pixels, on one frame grid shared by everybody.

    The grid has to be shared: each spline starts on its own frame, and an identity switch needs two people sampled on
    the same frame.
    """
    tracks = []
    for person_id, points in enumerate(parse_vsp(path.read_text(encoding="latin-1")), start=1):
        order = np.argsort(points[:, 2], kind="stable")
        points = points[order]
        keep = np.r_[True, np.diff(points[:, 2]) > 0]
        points = points[keep]
        first = -(-int(points[0, 2]) // UCY_FRAME_STEP) * UCY_FRAME_STEP
        frames = np.arange(first, int(points[-1, 2]) + 1, UCY_FRAME_STEP, dtype=np.int64)
        if len(points) < 2 or len(frames) == 0:
            continue
        uv = np.c_[np.interp(frames, points[:, 2], points[:, 0]) + width / 2,
                   height / 2 - np.interp(frames, points[:, 2], points[:, 1])]
        tracks.append(GridTrack(person_id, frames, uv))
    return tuple(tracks)


def median_height(boxes) -> float:
    """The scene's person height in pixels. It scales the identity-switch distance and is never converted to metres."""
    boxes = np.asarray(boxes, dtype=np.float64)
    return float(np.median(boxes[:, 3] - boxes[:, 1]))


@dataclass(frozen=True)
class Part:
    """A numbered stretch of a long recording (decision 0012).

    `start` and `stop` are the recording's video frames. A part's own video and tracks count from `start`, so that
    everything downstream treats it like any other scene.
    """
    recording: str
    number: int
    start: int
    stop: int                        # the frame after its last


def part_ranges(first: int, stop: int, step: int, count: int) -> list[tuple[int, int]]:
    """Equal stretches of frames, cut on the sampling grid. The last one takes what the grid leaves over."""
    length = (stop - first) // count // step * step
    starts = [first + k * length for k in range(count)]
    return list(zip(starts, [*starts[1:], stop]))


def cut_tracks(tracks, part: Part) -> tuple[GridTrack, ...]:
    """A part's people, with the IDs of the full recording and frames counted from the part's start."""
    cut = []
    for track in tracks:
        keep = (track.frames >= part.start) & (track.frames < part.stop)
        if keep.any():
            cut.append(GridTrack(track.ped_id, track.frames[keep] - part.start, track.positions[keep]))
    return tuple(cut)


def cut_rows(rows: list[dict], part: Part) -> list[dict]:
    """`cut_tracks` for tracks already in the shape sent to the browser."""
    cut = []
    for row in rows:
        frames = np.asarray(row["frames"])
        keep = np.flatnonzero((frames >= part.start) & (frames < part.stop))
        if len(keep):
            cut.append({"id": row["id"], "frames": (frames[keep] - part.start).tolist(), "uv": [row["uv"][k] for k in keep],
                        "boxes": [row["boxes"][k] for k in keep]})
    return cut


def public_detections(archive: zipfile.ZipFile, sequence: str, info: mot17.SequenceInfo) -> list[np.ndarray]:
    """Per-frame boxes from the public detections shipped with MOT17, as listed for frame k = image k + 1."""
    det = mot17.parse_detections(archive.read(mot17.member(archive, sequence, MOT17_VARIANT, "det/det.txt")), info)
    keep = det.score >= MIN_DETECTION_SCORE
    frames, boxes = det.frame[keep], det.boxes[keep]
    return [boxes[np.searchsorted(frames, k, "left"):np.searchsorted(frames, k, "right")]
            for k in range(1, info.seq_length + 1)]


def yolo_boxes(document: dict, width: float, height: float) -> list[np.ndarray]:
    """Per-frame person boxes from a stored YOLO file, in pixels of an image of this size."""
    scale = np.array([width, height, width, height], dtype=np.float64)
    return [np.array([box[:4] for box in frame if box[5] == PERSON_CLASS], dtype=np.float64).reshape(-1, 4) * scale
            for frame in document["by_frame"]]


def _tracker_rows(per_frame: list[np.ndarray], scale: float = 1.0) -> list[dict]:
    tracks = []
    for track in run_tracker(per_frame, first_frame=0).tracks:
        boxes = np.asarray(track.boxes, dtype=np.float64) * scale
        tracks.append(track_row(track.track_id, track.frames, _feet(boxes), boxes))
    return tracks


def mot17_tracks(archive_path: Path, sequence: str, yolo: dict | None = None) -> list[dict]:
    """Run the existing tracker over a whole sequence. The default settings are the ones the stored results used."""
    with zipfile.ZipFile(archive_path) as archive:
        info = mot17.read_seqinfo(archive, sequence, MOT17_VARIANT)
        per_frame = yolo_boxes(yolo, info.im_width, info.im_height) if yolo else public_detections(archive, sequence, info)
    return _tracker_rows(per_frame, VIDEO_HEIGHT / info.im_height)


def video_tracks(document: dict) -> list[dict]:
    """The tracker on boxes made on a browser copy, so they are in the pixels of the video that is shown."""
    return _tracker_rows(yolo_boxes(document, document["image_width"], document["image_height"]))


def _feet(boxes: np.ndarray) -> np.ndarray:
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]              # trails and paths follow the feet


@dataclass(frozen=True)
class Mot17Truth:
    """MOT17's annotated people on one frame grid, in video pixels."""
    tracks: tuple
    frame_step: int
    seconds_per_step: float
    person_height: float
    relative_boxes: dict             # (pedestrian, frame) -> box relative to their foot point


def mot17_truth(archive_path: Path, sequence: str, interval_s: float) -> Mot17Truth:
    """The sequence's ground truth, sampled every `interval_s` as closely as its frame rate allows."""
    with zipfile.ZipFile(archive_path) as archive:
        info = mot17.read_seqinfo(archive, sequence, MOT17_VARIANT)
        gt = mot17.parse_gt(archive.read(mot17.member(archive, sequence, MOT17_VARIANT, "gt/gt.txt")), info)
    step = max(1, round(info.frame_rate * interval_s))
    frames = gt.frame - 1                                                   # MOT17 counts from 1, the video from 0
    keep = (mot17.categorise(gt) == EVALUABLE) & (frames % step == 0)
    boxes = gt.boxes[keep] * (VIDEO_HEIGHT / info.im_height)
    feet, people, frames = _feet(boxes), gt.ped_id[keep], frames[keep]
    tracks, relative = [], {}
    for person in np.unique(people):
        mine = np.flatnonzero(people == person)
        mine = mine[np.argsort(frames[mine], kind="stable")]
        tracks.append(GridTrack(int(person), frames[mine].astype(np.int64), feet[mine]))
        for k in mine:
            relative[(int(person), int(frames[k]))] = boxes[k] - feet[k][[0, 1, 0, 1]]
    return Mot17Truth(tuple(tracks), step, step / info.frame_rate, median_height(boxes), relative)
