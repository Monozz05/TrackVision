"""Read-only access to MOT17 sequence metadata, ground truth and public detections inside MOT17.zip.

The archive is opened read-only and members are read in memory; nothing is extracted or modified.
Only the annotation text members are read (seqinfo.ini, gt/gt.txt, det/det.txt); images are never
decoded. Member checksums are verified against the tracked compact manifest.

MOT17 IS A TRACKING BENCHMARK ONLY in this project. Spec v4.15 section 6.1.1: MOT17 trajectories are
never used to train or evaluate the trajectory predictor. This module has no world coordinates and
produces no trajectories in metres.

Column conventions (MOTChallenge devkit, MOT16 paper; the numeric conventions below are documented,
not verified against images):
  gt.txt   frame, id, bb_left, bb_top, bb_width, bb_height, consider, class, visibility   (9 columns)
  det.txt  frame, -1, bb_left, bb_top, bb_width, bb_height, score [, -1, -1, -1]          (7 or 10)
  frames are 1-based; bb_left/bb_top are 1-based pixel coordinates (top-left pixel is (1, 1)).
Project boxes are xyxy in the 0-based continuous convention used by the U3/U4 code (pixel x covers
[x, x+1)): x1 = bb_left - 1, y1 = bb_top - 1, x2 = x1 + bb_width, y2 = y1 + bb_height. Ground truth
and detections share the convention, so IoU does not depend on the 1-pixel offset; `to_mot_box`
inverts it exactly for output files. Boxes may extend beyond the image and are not clipped.

Ground-truth categories (`categorise`), following the MOTChallenge evaluation classes:
  evaluable  class 1 (pedestrian) with consider == 1 and visibility >= min_visibility
  invisible  class 1 with consider == 1 and visibility < min_visibility (only if min_visibility > 0;
             the default 0.0 applies no visibility filter, as the MOT17 benchmark does)
  ignore     class 1 with consider == 0, and the distractor classes 2 (person on vehicle),
             7 (static person), 8 (distractor), 12 (reflection): a tracker box matched to one is
             neither TP nor FP
  excluded   every other class (vehicles 3-6, occluders 9-11, anything unknown): not ground truth;
             a tracker box on one counts as an ordinary false positive
"""

from __future__ import annotations

import configparser
import hashlib
import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackvision.config import REPO_ROOT

DATASET = "MOT17"
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "raw" / "mot17" / "MOT17.zip"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "mot17.sha256"
LICENCE = "CC BY-NC-SA 3.0 (MOTChallenge MOT17 page, Phase 0B): non-commercial research use, share-alike"
VARIANTS = ("DPM", "FRCNN", "SDP")
GT_COLUMNS = ("frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "consider", "class", "visibility")
DET_COLUMNS = ("frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "score")
CLASS_NAMES = {1: "pedestrian", 2: "person_on_vehicle", 3: "car", 4: "bicycle", 5: "motorbike",
               6: "non_motorized_vehicle", 7: "static_person", 8: "distractor", 9: "occluder",
               10: "occluder_on_the_ground", 11: "occluder_full", 12: "reflection"}
PEDESTRIAN = 1
DISTRACTOR_CLASSES = (2, 7, 8, 12)
CATEGORIES = ("evaluable", "invisible", "ignore", "excluded")
_SEQINFO_KEYS = ("name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight", "imExt")
_SEQ_DIR = re.compile(r"^(?:[^/]+/)?(train|test)/(MOT17-\d{2})-([A-Z]+)/seqinfo\.ini$")


class Mot17FormatError(ValueError):
    """Raised when a MOT17 member does not have the documented structure."""


class Mot17ManifestError(ValueError):
    """Raised when a member is missing from the manifest or does not match its recorded hash."""


@dataclass(frozen=True)
class SequenceInfo:
    name: str                # e.g. MOT17-02-FRCNN
    im_dir: str
    frame_rate: int
    seq_length: int          # frames 1 .. seq_length
    im_width: int
    im_height: int
    im_ext: str


@dataclass(frozen=True)
class GroundTruth:
    frame: np.ndarray        # (n,) int64, 1-based
    ped_id: np.ndarray       # (n,) int64
    boxes: np.ndarray        # (n, 4) xyxy, project convention
    consider: np.ndarray     # (n,) int64 (0 or 1)
    cls: np.ndarray          # (n,) int64
    visibility: np.ndarray   # (n,) float64 in [0, 1]


@dataclass(frozen=True)
class Detections:
    frame: np.ndarray        # (n,) int64, 1-based
    boxes: np.ndarray        # (n, 4) xyxy, project convention
    score: np.ndarray        # (n,) float64, as supplied


def member_prefix(zf: zipfile.ZipFile) -> str:
    """The archive's top-level folder ('MOT17/' in the official archive), or '' if members are at the root."""
    for name in zf.namelist():
        m = _SEQ_DIR.match(name)
        if m:
            return name[: m.start(1)]
    raise Mot17FormatError("no <split>/MOT17-NN-<variant>/seqinfo.ini member found")


def list_sequences(zf: zipfile.ZipFile, split: str = "train") -> dict[str, tuple[str, ...]]:
    """{base sequence (MOT17-NN): detection variants present} for one split, from seqinfo.ini members."""
    found: dict[str, set] = {}
    for name in zf.namelist():
        m = _SEQ_DIR.match(name)
        if m and m.group(1) == split:
            found.setdefault(m.group(2), set()).add(m.group(3))
    return {seq: tuple(sorted(v)) for seq, v in sorted(found.items())}


def member(zf: zipfile.ZipFile, seq: str, variant: str, rel: str, split: str = "train") -> str:
    return f"{member_prefix(zf)}{split}/{seq}-{variant}/{rel}"


def parse_seqinfo(data: bytes, folder: str) -> SequenceInfo:
    """Parse seqinfo.ini; `folder` is the sequence folder name (MOT17-NN-<variant>) it must match."""
    parser = configparser.ConfigParser()
    parser.optionxform = str                         # keys are case-sensitive (frameRate, seqLength)
    parser.read_string(data.decode("utf-8"))
    if "Sequence" not in parser or any(k not in parser["Sequence"] for k in _SEQINFO_KEYS):
        raise Mot17FormatError(f"{folder}: seqinfo.ini needs [Sequence] with {_SEQINFO_KEYS}")
    s = parser["Sequence"]
    try:
        info = SequenceInfo(s["name"], s["imDir"], int(s["frameRate"]), int(s["seqLength"]), int(s["imWidth"]),
                            int(s["imHeight"]), s["imExt"])
    except ValueError as exc:
        raise Mot17FormatError(f"{folder}: seqinfo.ini {exc}") from exc
    if info.name != folder:
        raise Mot17FormatError(f"seqinfo name {info.name!r} does not match folder {folder}")
    if min(info.frame_rate, info.seq_length, info.im_width, info.im_height) < 1:
        raise Mot17FormatError(f"{info.name}: frame rate, length and image size must be positive")
    return info


def read_seqinfo(zf: zipfile.ZipFile, seq: str, variant: str, split: str = "train") -> SequenceInfo:
    return parse_seqinfo(zf.read(member(zf, seq, variant, "seqinfo.ini", split)), f"{seq}-{variant}")


def _table(data: bytes, what: str) -> np.ndarray:
    text = data.decode("utf-8")
    if not text.strip():
        return np.empty((0, 0))
    try:
        return np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2, dtype=np.float64)
    except ValueError as exc:
        raise Mot17FormatError(f"{what}: {exc}") from exc


def _integers(values: np.ndarray, what: str) -> np.ndarray:
    if not np.all(values == np.round(values)):
        raise Mot17FormatError(f"{what} must be integers")
    return values.astype(np.int64)


def mot_to_xyxy(left, top, width, height) -> np.ndarray:
    """MOT 1-based (bb_left, bb_top, w, h) to project 0-based continuous xyxy."""
    x1, y1 = np.asarray(left, dtype=np.float64) - 1.0, np.asarray(top, dtype=np.float64) - 1.0
    return np.stack([x1, y1, x1 + width, y1 + height], axis=-1)


def to_mot_box(boxes) -> np.ndarray:
    """Project xyxy to MOT (bb_left, bb_top, bb_width, bb_height); the inverse of `mot_to_xyxy`."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([b[:, 0] + 1.0, b[:, 1] + 1.0, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]])


def _boxes(t: np.ndarray, what: str) -> np.ndarray:
    if np.any(t[:, 4] < 0) or np.any(t[:, 5] < 0):
        raise Mot17FormatError(f"{what}: box width and height must be non-negative")
    return mot_to_xyxy(t[:, 2], t[:, 3], t[:, 4], t[:, 5])


def _frames(t: np.ndarray, info: SequenceInfo, what: str) -> np.ndarray:
    frames = _integers(t[:, 0], f"{what} frame numbers")
    if len(frames) and (frames.min() < 1 or frames.max() > info.seq_length):
        raise Mot17FormatError(f"{what}: frames must lie in 1..{info.seq_length}")
    return frames


def parse_gt(data: bytes, info: SequenceInfo) -> GroundTruth:
    what = f"{info.name} gt.txt"
    t = _table(data, what)
    if t.size == 0:
        e = np.empty(0, dtype=np.int64)
        return GroundTruth(e, e, np.empty((0, 4)), e, e, np.empty(0))
    if t.shape[1] != len(GT_COLUMNS):
        raise Mot17FormatError(f"{what}: expected {len(GT_COLUMNS)} columns {GT_COLUMNS}, got {t.shape[1]}")
    frames = _frames(t, info, what)
    ids = _integers(t[:, 1], f"{what} ids")
    consider = _integers(t[:, 6], f"{what} consider flags")
    cls = _integers(t[:, 7], f"{what} classes")
    vis = t[:, 8]
    if not np.all(np.isin(consider, (0, 1))):
        raise Mot17FormatError(f"{what}: consider flag must be 0 or 1")
    if np.any(vis < 0) or np.any(vis > 1):
        raise Mot17FormatError(f"{what}: visibility must lie in [0, 1]")
    keys = frames * (int(ids.max()) + 1 if len(ids) else 1) + ids
    if len(np.unique(keys)) != len(keys):
        raise Mot17FormatError(f"{what}: duplicate (frame, id) rows")
    order = np.lexsort((ids, frames))
    return GroundTruth(frames[order], ids[order], _boxes(t, what)[order], consider[order], cls[order], vis[order])


def parse_detections(data: bytes, info: SequenceInfo) -> Detections:
    what = f"{info.name} det.txt"
    t = _table(data, what)
    if t.size == 0:
        return Detections(np.empty(0, dtype=np.int64), np.empty((0, 4)), np.empty(0))
    if t.shape[1] not in (7, 10):
        raise Mot17FormatError(f"{what}: expected 7 or 10 columns {DET_COLUMNS}, got {t.shape[1]}")
    if np.any(t[:, 1] != -1):
        raise Mot17FormatError(f"{what}: detection id column must be -1")
    frames = _frames(t, info, what)
    order = np.argsort(frames, kind="stable")       # file order is kept within a frame
    return Detections(frames[order], _boxes(t, what)[order], t[order, 6].copy())


def categorise(gt: GroundTruth, min_visibility: float = 0.0) -> np.ndarray:
    """One of CATEGORIES per ground-truth row (rules in the module docstring)."""
    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must lie in [0, 1]")
    out = np.full(len(gt.frame), "excluded", dtype=object)
    pedestrian = gt.cls == PEDESTRIAN
    out[np.isin(gt.cls, DISTRACTOR_CLASSES) | (pedestrian & (gt.consider == 0))] = "ignore"
    counted = pedestrian & (gt.consider == 1)
    out[counted] = "evaluable"
    out[counted & (gt.visibility < min_visibility)] = "invisible"
    return out


def half_split(seq_length: int) -> dict[str, tuple[int, int]]:
    """Inclusive 1-based frame ranges: first = 1 .. N // 2, second = N // 2 + 1 .. N.

    ASSUMPTION: the ByteTrack ablation convention (half = int(N / 2) of the 0-based image indices) as
    read from its conversion script; not verified against ByteTrack's published half-split files.
    A range (a, b) with b < a is empty."""
    if seq_length < 1:
        raise ValueError("seq_length must be positive")
    half = seq_length // 2
    return {"first": (1, half), "second": (half + 1, seq_length)}


def in_range(frames: np.ndarray, frame_range: tuple[int, int]) -> np.ndarray:
    return (frames >= frame_range[0]) & (frames <= frame_range[1])


def read_manifest(path: Path) -> dict[str, str]:
    """The compact MOT17 manifest: '#' comment lines, then '<sha256>  <path>' lines, where archive
    members are written 'MOT17.zip!/<member>'."""
    entries = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            raise Mot17ManifestError(f"{path}:{number}: malformed manifest line")
        entries[rel] = digest.lower()
    return entries


def read_verified(zf: zipfile.ZipFile, name: str, archive_name: str, manifest: dict | None,
                  record: dict) -> bytes:
    """Read one member in memory; check it against the manifest (if given); record its SHA-256."""
    data = zf.read(name)
    digest = hashlib.sha256(data).hexdigest()
    if manifest is not None:
        key = f"{archive_name}!/{name}"
        if key not in manifest:
            raise Mot17ManifestError(f"{key!r} is not listed in the manifest")
        if manifest[key] != digest:
            raise Mot17ManifestError(f"{key!r} does not match the manifest (expected {manifest[key]}, found {digest})")
    record[name] = digest
    return data
