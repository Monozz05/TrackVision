"""Tiny synthetic MOT17-format archives (seqinfo.ini, gt/gt.txt, det/det.txt in a ZIP) and a compact
manifest in the tracked MOT17 manifest format, written to a temporary directory. No images.

`walker_sequence` builds a scene with known answers. Every frame 1..n holds:
  ped 1, ped 2  class 1 pedestrians moving right 5 px per frame (20 x 40 px boxes)
  ped 3         class 7 static person (a distractor: ignore region)
  ped 4         class 3 car (excluded from ground truth)
and one public-style detection exactly on each of the four boxes, in that order.
"""

import hashlib
import zipfile

FIELDS = ("frame", "id", "left", "top", "w", "h")
BOX = (20, 40)


def seqinfo(name, length, frame_rate=30, width=1920, height=1080):
    return (f"[Sequence]\nname={name}\nimDir=img1\nframeRate={frame_rate}\nseqLength={length}\n"
            f"imWidth={width}\nimHeight={height}\nimExt=.jpg\n")


def gt_line(frame, ped_id, left, top, w, h, consider=1, cls=1, vis=1.0):
    return f"{frame},{ped_id},{left},{top},{w},{h},{consider},{cls},{vis}\n"


def det_line(frame, left, top, w, h, score=0.9, ten_columns=False):
    return f"{frame},-1,{left},{top},{w},{h},{score}" + (",-1,-1,-1\n" if ten_columns else "\n")


def walker_boxes(frame):
    """MOT (left, top, w, h) of the four objects at a frame, in detection order."""
    return [(101 + 5 * (frame - 1), 201, *BOX), (501 + 5 * (frame - 1), 201, *BOX), (301, 101, *BOX),
            (701, 301, 60, 30)]


def walker_sequence(name="MOT17-02-FRCNN", length=10, det_frames=None):
    """{'seqinfo', 'gt', 'det'} texts; detections only in `det_frames` (default: every frame)."""
    gt, det = [], []
    labels = [(1, 1, 1, 1.0), (2, 1, 1, 0.6), (3, 0, 7, 1.0), (4, 0, 3, 1.0)]   # id, consider, class, vis
    for f in range(1, length + 1):
        for (pid, consider, cls, vis), box in zip(labels, walker_boxes(f)):
            gt.append(gt_line(f, pid, *box, consider=consider, cls=cls, vis=vis))
            if det_frames is None or f in det_frames:
                det.append(det_line(f, *box))
    return {"seqinfo": seqinfo(name, length), "gt": "".join(gt), "det": "".join(det)}


def write_zip(path, sequences, split="train", prefix="MOT17/", extra=None):
    """sequences: {folder name (MOT17-NN-VARIANT): {'seqinfo': text, 'gt': text | None, 'det': text}}."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, files in sequences.items():
            base = f"{prefix}{split}/{folder}/"
            zf.writestr(base + "seqinfo.ini", files["seqinfo"])
            if files.get("gt") is not None:
                zf.writestr(base + "gt/gt.txt", files["gt"])
            zf.writestr(base + "det/det.txt", files["det"])
        for name, text in (extra or {}).items():
            zf.writestr(name, text)
    return path


def write_manifest(zip_path, manifest_path, drop=()):
    """The compact manifest format: '#' comments, the archive hash, then 'MOT17.zip!/<member>' lines."""
    lines = ["# synthetic MOT17 manifest", f"{hashlib.sha256(zip_path.read_bytes()).hexdigest()}  {zip_path.name}"]
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith("/") and name not in drop:
                lines.append(f"{hashlib.sha256(zf.read(name)).hexdigest()}  {zip_path.name}!/{name}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest_path
