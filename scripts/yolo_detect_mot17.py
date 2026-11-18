"""Run YOLO11 over the MOT17 training sequences and store its boxes for the dashboard (decision 0009).

Runs in its own environment (see scripts/setup_yolo.ps1), never the project `.venv`, and imports nothing from
trackvision. Images are read from MOT17.zip in memory: the archive is opened read-only and nothing is extracted. Output
goes to the Git-ignored `data/cache/detections/`, one file per sequence, written atomically. A finished sequence is
skipped on a rerun, so an interrupted run can simply be started again.

These boxes are for display. They are not experiment P-1: nothing here enters `runs/`, and no detector choice is made
for the research protocol.

    python scripts/yolo_detect_mot17.py --weights <folder>/yolo11m.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable

SCHEMA = "trackvision-detections/1"
SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13")
VARIANT = "FRCNN"            # the three detection sets share the same image folders
PERSON_CLASS = 0             # COCO
LABEL = "YOLO11 detections for display. Not experiment P-1 and not a tracking evaluation."
REPO_ROOT = Path(__file__).resolve().parents[1]


def image_members(names: Iterable[str], sequence: str) -> list[str]:
    """The sequence's frames in order, so that list position k is video frame k."""
    prefix = f"MOT17/train/{sequence}-{VARIANT}/img1/"
    return sorted(name for name in names if name.startswith(prefix) and name.endswith(".jpg"))


def normalise(box: Iterable[float], confidence: float, class_id: int, width: int, height: int) -> list:
    """A box as fractions of the image, so it fits the video at any display size."""
    x1, y1, x2, y2 = box
    return [round(x1 / width, 4), round(y1 / height, 4), round(x2 / width, 4), round(y2 / height, 4),
            round(float(confidence), 3), int(class_id)]


def write_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def detect_sequence(frames: Iterable, detect: Callable[..., tuple], metadata: dict) -> dict:
    """`detect` maps one image to (width, height, [(box, confidence, class id), ...])."""
    by_frame, width, height = [], None, None
    for image in frames:
        width, height, boxes = detect(image)
        by_frame.append([normalise(box, confidence, class_id, width, height) for box, confidence, class_id in boxes])
    return {"schema": SCHEMA, "label": LABEL, **metadata, "image_width": width, "image_height": height,
            "frames": len(by_frame), "by_frame": by_frame}


def yolo_detector(weights: Path, imgsz: int, confidence: float, device: str | None) -> tuple[Callable, dict]:
    """Load the model. Ultralytics is imported here so that the rest of this file can be tested without it."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(weights.parent / "ultralytics-config"))
    import cv2
    import numpy as np
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS

    SETTINGS.update({"sync": False})      # no usage reporting: some of this data is marked local only
    model = YOLO(str(weights))

    def detect(image) -> tuple:
        """`image` is an encoded file, or a frame OpenCV has already decoded."""
        if isinstance(image, bytes):
            image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        result = model.predict(image, imgsz=imgsz, conf=confidence, classes=[PERSON_CLASS], device=device, verbose=False)[0]
        boxes = result.boxes
        rows = zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.int().tolist())
        return image.shape[1], image.shape[0], [(box, score, class_id) for box, score, class_id in rows]

    metadata = {"model": weights.stem, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                "ultralytics_version": ultralytics.__version__, "imgsz": imgsz, "confidence_threshold": confidence,
                "device": str(model.device), "classes": {str(k): v for k, v in model.names.items()}}
    return detect, metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="path to the YOLO11 weights, for example yolo11m.pt")
    parser.add_argument("--archive", type=Path, default=REPO_ROOT / "data" / "raw" / "mot17" / "MOT17.zip")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cache" / "detections")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="for example cpu or 0; by default Ultralytics chooses")
    parser.add_argument("--sequences", nargs="*", default=list(SEQUENCES), choices=SEQUENCES)
    args = parser.parse_args(argv)

    detect, metadata = yolo_detector(args.weights, args.imgsz, args.confidence, args.device)
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        for sequence in args.sequences:
            target = args.out / f"{sequence}.json"
            if target.exists():
                print(f"{sequence}: already done, skipped")
                continue
            members = image_members(names, sequence)
            started = time.time()
            document = detect_sequence((archive.read(m) for m in members), detect, {"scene": sequence, **metadata})
            write_atomic(target, document)
            boxes = sum(len(frame) for frame in document["by_frame"])
            print(f"{sequence}: {len(members)} frames, {boxes} boxes, {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
