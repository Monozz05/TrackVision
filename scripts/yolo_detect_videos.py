"""Run YOLO11 over the browser copies of ETH, HOTEL and UCY and store its boxes for the dashboard (decision 0015).

Like yolo_detect_mot17.py, this runs in its own environment, imports nothing from trackvision and writes only to the
Git-ignored `data/cache/detections/`, one file per recording, atomically. It reads the browser copies rather than the
source files, so frame k here is the frame k the browser shows, and a part of a recording uses its recording's boxes. A
finished recording is skipped on a rerun, so an interrupted run can simply be started again.

These boxes are for display. They are not ground truth, not experiment P-1 or P-9, and never enter ADE, FDE or any other
measurement. Nothing here enters `runs/`.

    python scripts/yolo_detect_videos.py --weights <folder>/yolo11m.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

from yolo_detect_mot17 import REPO_ROOT, detect_sequence, write_atomic, yolo_detector

SCENES = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")
PROGRESS_FRAMES = 2000


def video_frames(path: Path) -> Iterator:
    """Every frame of a video in order, so that position k is video frame k."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        while True:
            found, frame = capture.read()
            if not found:
                return
            yield frame
    finally:
        capture.release()


def with_progress(frames: Iterable, scene: str, total: int) -> Iterator:
    for count, frame in enumerate(frames, start=1):
        yield frame
        if count % PROGRESS_FRAMES == 0:
            print(f"{scene}: {count} of {total} frames", flush=True)


def detect_scene(scene: str, video_dir: Path, detect, metadata: dict, frames=video_frames) -> dict:
    """One recording's detections, refused if the frames read are not the frames the browser plays."""
    expected = json.loads((video_dir / f"{scene}.json").read_text(encoding="utf-8"))["frames"]
    document = detect_sequence(with_progress(frames(video_dir / f"{scene}.mp4"), scene, expected), detect,
                               {"scene": scene, **metadata})
    if document["frames"] != expected:
        raise ValueError(f"{scene}: read {document['frames']} frames, but its video has {expected}")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="path to the YOLO11 weights, for example yolo11m.pt")
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "data" / "cache" / "video")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "detections")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="for example cpu or 0; by default Ultralytics chooses")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES), choices=SCENES)
    args = parser.parse_args(argv)

    detect, metadata = yolo_detector(args.weights, args.imgsz, args.confidence, args.device)
    for scene in args.scenes:
        target = args.out / f"{scene}.json"
        if target.exists():
            print(f"{scene}: already done, skipped")
            continue
        started = time.time()
        document = detect_scene(scene, args.video, detect, metadata)
        write_atomic(target, document)
        boxes = sum(len(frame) for frame in document["by_frame"])
        print(f"{scene}: {document['frames']} frames, {boxes} boxes, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
