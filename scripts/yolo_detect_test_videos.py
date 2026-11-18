"""Run YOLO11 over the browser copies of the Test Videos and store its boxes for the dashboard (decision 0017).

Like yolo_detect_videos.py, this runs in its own environment with the same detector settings, imports nothing from
trackvision and reads the browser copies, so frame k here is the frame k the browser shows. Boxes go to the Git-ignored
`data/cache/test_video_detections/`, apart from every dataset's, one file per video, atomically. A finished video is
skipped on a rerun.

These boxes are for display. They are not ground truth, not experiment P-1 or P-9, and nothing here enters `runs/`.

    python scripts/yolo_detect_test_videos.py --weights <folder>/yolo11m.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from yolo_detect_mot17 import REPO_ROOT, write_atomic, yolo_detector
from yolo_detect_videos import detect_scene


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="path to the YOLO11 weights, for example yolo11m.pt")
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "data" / "cache" / "test_videos")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "test_video_detections")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="for example cpu or 0; by default Ultralytics chooses")
    args = parser.parse_args(argv)

    detect, metadata = yolo_detector(args.weights, args.imgsz, args.confidence, args.device)
    for sidecar in sorted(args.video.glob("*.json")):
        name, target = sidecar.stem, args.out / sidecar.name
        if target.exists():
            print(f"{name}: already done, skipped")
            continue
        started = time.time()
        document = detect_scene(name, args.video, detect, metadata)
        write_atomic(target, document)
        boxes = sum(len(frame) for frame in document["by_frame"])
        print(f"{name}: {document['frames']} frames, {boxes} boxes, {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
