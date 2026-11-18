"""Online IoU tracker with constant-velocity box prediction (feasibility tracker; not ByteTrack/BoT-SORT).

Per frame (frames must be consecutive integers, one `update` call per frame, including empty ones):
  1. Predict each active track's box at this frame: its last matched box centre moved at constant
     velocity (centre change between its last two matched detections / their frame gap; zero for a
     track with one observation). Box size stays at the last matched size.
  2. Associate predictions (tracks in ID order) with detections (in the given order) by
     `gated_iou_assignment` with `match_iou`. This internal gate (default 0.3) is a tracker parameter,
     distinct from the configured 0.5 threshold used only to evaluate tracks against ground truth.
  3. Matched track: record the detection box as an observation; consecutive-hit count + 1; misses = 0;
     confirmed once hits >= min_hits.
  4. Unmatched track: misses + 1. A tentative (unconfirmed) track is deleted; a confirmed track coasts
     (keeps predicting, records nothing) and is terminated after more than max_age consecutive misses.
     Terminated tracks are never revived.
  5. Unmatched detection: a new tentative track with the next ID (counter from 1, never reused).
Output contains confirmed tracks only, with their observed frames and boxes (including the frames
before confirmation). Coasting predictions are never part of the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trackvision.tracking.assignment import gated_iou_assignment


@dataclass(frozen=True)
class TrackerConfig:
    match_iou: float = 0.3          # internal association gate (provisional feasibility default)
    min_hits: int = 3
    max_age: int = 25               # frames (1.0 s at 25 fps)

    def __post_init__(self) -> None:
        if not 0.0 < self.match_iou <= 1.0:
            raise ValueError("match_iou must lie in (0, 1]")
        if not (isinstance(self.min_hits, int) and self.min_hits >= 1):
            raise ValueError("min_hits must be a positive integer")
        if not (isinstance(self.max_age, int) and self.max_age >= 0):
            raise ValueError("max_age must be a non-negative integer")


@dataclass(frozen=True)
class TrackOutput:
    track_id: int
    frames: np.ndarray       # (n,) int64 observed frames only
    boxes: np.ndarray        # (n, 4) float64 detection boxes xyxy


@dataclass(frozen=True)
class TrackerResult:
    tracks: tuple            # TrackOutput for confirmed tracks, sorted by track_id
    n_created: int
    n_confirmed: int
    n_deleted_tentative: int
    n_terminated: int

    def frame_index(self) -> dict[int, list[tuple[int, np.ndarray]]]:
        """frame -> [(track_id, box), ...] over all confirmed observations."""
        out: dict[int, list] = {}
        for t in self.tracks:
            for f, b in zip(t.frames, t.boxes):
                out.setdefault(int(f), []).append((t.track_id, b))
        return out


@dataclass
class _Track:
    track_id: int
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def predict(self, frame: int) -> np.ndarray:
        box = self.boxes[-1]
        shift = self.velocity * (frame - self.frames[-1])
        return box + np.array([shift[0], shift[1], shift[0], shift[1]])

    def observe(self, frame: int, box: np.ndarray) -> None:
        if self.frames:
            centre_old = (self.boxes[-1][:2] + self.boxes[-1][2:]) / 2
            centre_new = (box[:2] + box[2:]) / 2
            self.velocity = (centre_new - centre_old) / (frame - self.frames[-1])
        self.frames.append(frame)
        self.boxes.append(np.asarray(box, dtype=np.float64))
        self.hits += 1
        self.misses = 0


class IouTracker:
    def __init__(self, config: TrackerConfig = TrackerConfig()):
        self.config = config
        self._next_id = 1
        self._active: list[_Track] = []
        self._done: list[_Track] = []
        self._last_frame: int | None = None
        self.n_created = self.n_deleted_tentative = self.n_terminated = 0

    def update(self, frame: int, boxes) -> None:
        if self._last_frame is not None and frame != self._last_frame + 1:
            raise ValueError(f"frames must be consecutive: got {frame} after {self._last_frame}")
        self._last_frame = frame
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        tracks = sorted(self._active, key=lambda t: t.track_id)
        predicted = np.array([t.predict(frame) for t in tracks]).reshape(-1, 4)
        pairs = gated_iou_assignment(predicted, boxes, self.config.match_iou)
        matched_t = {i for i, _, _ in pairs}
        matched_d = {j for _, j, _ in pairs}

        for i, j, _ in pairs:
            t = tracks[i]
            t.observe(frame, boxes[j])
            if t.hits >= self.config.min_hits:
                t.confirmed = True
        survivors = []
        for i, t in enumerate(tracks):
            if i in matched_t:
                survivors.append(t)
                continue
            t.misses += 1
            t.hits = 0
            if not t.confirmed:
                self.n_deleted_tentative += 1
            elif t.misses > self.config.max_age:
                self.n_terminated += 1
                self._done.append(t)
            else:
                survivors.append(t)
        for j in range(len(boxes)):
            if j not in matched_d:
                t = _Track(self._next_id)
                self._next_id += 1
                self.n_created += 1
                t.observe(frame, boxes[j])
                t.confirmed = t.hits >= self.config.min_hits
                survivors.append(t)
        self._active = survivors

    def result(self) -> TrackerResult:
        confirmed = sorted((t for t in self._done + self._active if t.confirmed), key=lambda t: t.track_id)
        tracks = tuple(TrackOutput(t.track_id, np.array(t.frames, dtype=np.int64),
                                   np.array(t.boxes, dtype=np.float64).reshape(-1, 4)) for t in confirmed)
        return TrackerResult(tracks, self.n_created, len(tracks), self.n_deleted_tentative, self.n_terminated)


def run_tracker(detections_per_frame, config: TrackerConfig = TrackerConfig(), first_frame: int = 0) -> TrackerResult:
    """Run the tracker over a sequence of per-frame box arrays (frame = first_frame + index)."""
    tracker = IouTracker(config)
    for k, boxes in enumerate(detections_per_frame):
        tracker.update(first_frame + k, boxes)
    return tracker.result()
