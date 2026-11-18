"""MOT17 tracking feasibility (M1): the project's IoU feasibility tracker on the public FRCNN detections
of the static-camera MOT17 training sequences, scored with the project-specific tracking metrics.

PLUMBING AND FEASIBILITY ONLY, NOT A MOT17 BENCHMARK:
  - detections are the public FRCNN detections distributed with MOT17, not produced by a TrackVision
    detector, and all of them are used (no score threshold);
  - the tracker is `tracking.tracker.IouTracker` with its existing defaults, not ByteTrack or BoT-SORT,
    and nothing is tuned;
  - the metrics are `metrics.tracking.evaluate_tracking`: project-specific CLEAR-MOT-style counts and a
    simplified IDF1, not TrackEval and not MOTChallenge-compliant; HOTA is not computed;
  - it does not meet the spec v4.15 real-tracker requirements (section 6.2 private-detector protocol,
    TrackEval, ByteTrack/BoT-SORT; section 3.7 real-tracker severities).
Results are kept per sequence and never pooled. Outputs stay local (outside the repository or under
the Git-ignored runs/); no ground-truth file is written.

MOT17 IS NEVER USED FOR PREDICTION (spec section 6.1.1): this module imports nothing from
trackvision.predict or the displacement metrics and computes no windows, CV or ADE/FDE.

Split. Each sequence is cut by `io.mot17.half_split`; the tracker runs on the second half only, from an
empty state at its first frame, over every frame of that half (frames without detections included).
First-half rows are counted but never reach the tracker or the metrics. The first half is reserved for
tuning (section 6.2), and no parameter is tuned here. Output files keep the original 1-based MOT17
frame numbers; whether a later TrackEval half-split set-up needs renumbered frames is unresolved.

Position errors are not applicable (no world coordinates): the metric function's position fields are
removed from the results rather than reported.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.io import mot17
from trackvision.io.manifest import file_sha256
from trackvision.metrics.tracking import FrameTracking, evaluate_tracking
from trackvision.tracking.tracker import TrackerConfig, TrackerResult, run_tracker

NAME = "mot17-iou-tracker-feasibility"
DETECTION_VARIANT = "FRCNN"
EVALUATED_HALF = "second"
LABEL = ("MOT17 plumbing and tracking feasibility (M1). Public FRCNN detections supplied with MOT17 (not a "
         "TrackVision detector); IoU feasibility tracker (not ByteTrack/BoT-SORT); project-specific feasibility "
         "metrics (not TrackEval, not MOTChallenge-compliant, no HOTA). Not a MOT17 benchmark result and not "
         "the spec's real-tracker evaluation. Per sequence, never pooled. MOT17 is never used for prediction. "
         "v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
DETECTIONS_NOTE = ("public FRCNN detections distributed with MOT17 (det/det.txt of the -FRCNN folders); not produced "
                   "by a TrackVision detector; every row used, no score threshold")
METRICS_NOTE = ("project-specific feasibility metrics from trackvision.metrics.tracking (CLEAR-MOT-style counts, "
                "simplified IDF1, project ignore handling); not TrackEval, not MOTChallenge-compliant, not "
                "comparable to published MOT17 results")
PREDICTION_NOTE = ("MOT17 is never used for trajectory prediction (spec v4.15 section 6.1.1): no prediction "
                   "windows, CV or ADE/FDE are computed from it")
NOT_SATISFIED = (
    "section 6.2: own YOLO detections (private-detector protocol) - public FRCNN detections used instead",
    "section 6.2: metrics via TrackEval - project-specific metrics used instead; no HOTA",
    "section 11.5: ByteTrack and BoT-SORT - IoU feasibility tracker used instead",
    "section 3.7: real-tracker achieved severities on the 0.4 s grid - not computed",
    "section 6.2: first half for tuning - no tuning performed",
)
POSITION_FIELDS = ("position_rmse_m", "position_median_error_m")
VISIBILITY_BINS = (0.0, 0.25, 0.5, 0.75)


@dataclass(frozen=True)
class SequenceRun:
    sequence: str                # MOT17-NN
    info: mot17.SequenceInfo
    split: dict                  # half -> (first frame, last frame), inclusive
    counts: dict                 # half -> counts
    tracker: TrackerResult
    metrics: dict
    mot_text: str


@dataclass(frozen=True)
class Mot17Run:
    experiment_id: str
    settings: dict
    settings_hash: str
    config_hash: str
    sequences: tuple             # SequenceRun, in configured order


def static_sequences(cfg: Config) -> list[str]:
    return [str(s) for s in cfg.frozen["datasets"]["mot17"]["static_sequences"]]


def camera_group(cfg: Config, sequences) -> str:
    if set(sequences) <= set(static_sequences(cfg)):
        return "static (frozen.datasets.mot17.static_sequences)"
    return "static and moving (frozen.datasets.mot17)"


def _rows_at(frames: np.ndarray, f: int) -> slice:
    return slice(int(np.searchsorted(frames, f, "left")), int(np.searchsorted(frames, f, "right")))


def tracker_inputs(det: mot17.Detections, frame_range: tuple[int, int]) -> list[np.ndarray]:
    """Per-frame detection boxes for every frame of the range (empty arrays where there are none)."""
    return [det.boxes[_rows_at(det.frame, f)] for f in range(frame_range[0], frame_range[1] + 1)]


def evaluation_frames(gt: mot17.GroundTruth, categories: np.ndarray, result: TrackerResult,
                      frame_range: tuple[int, int]) -> list[FrameTracking]:
    """FrameTracking per frame of the range; excluded ground-truth rows are dropped; no positions exist."""
    index = result.frame_index()
    frames = []
    for f in range(frame_range[0], frame_range[1] + 1):
        obs = index.get(f, [])
        rows = _rows_at(gt.frame, f)
        keep = categories[rows] != "excluded"
        n_gt = int(np.sum(keep))
        frames.append(FrameTracking(
            track_ids=np.array([tid for tid, _ in obs], dtype=np.int64),
            track_boxes=np.array([b for _, b in obs], dtype=np.float64).reshape(-1, 4),
            track_positions=np.full((len(obs), 2), np.nan),
            gt_ids=gt.ped_id[rows][keep], gt_boxes=gt.boxes[rows][keep],
            gt_categories=categories[rows][keep], gt_positions=np.full((n_gt, 2), np.nan)))
    return frames


def mot_text(result: TrackerResult) -> str:
    """MOTChallenge tracker format, one line per confirmed observation, sorted by frame then id:
    frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z with conf = 1 (the tracker has no scores),
    x = y = z = -1, original 1-based frame numbers and 1-based MOT box coordinates."""
    rows = sorted((int(f), t.track_id, b) for t in result.tracks for f, b in zip(t.frames, t.boxes))
    lines = []
    for f, tid, b in rows:
        left, top, w, h = mot17.to_mot_box(b)[0]
        lines.append(f"{f},{tid},{left:.3f},{top:.3f},{w:.3f},{h:.3f},1,-1,-1,-1")
    return "".join(line + "\n" for line in lines)


def _stats(values) -> dict | None:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return None
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def half_counts(gt: mot17.GroundTruth, categories: np.ndarray, det: mot17.Detections,
                frame_range: tuple[int, int]) -> dict:
    g, d = mot17.in_range(gt.frame, frame_range), mot17.in_range(det.frame, frame_range)
    frames = np.arange(frame_range[0], frame_range[1] + 1)
    ev = g & (categories == "evaluable")
    bins = np.digitize(gt.visibility[ev], VISIBILITY_BINS[1:])
    labels = [f"[{a},{b})" for a, b in zip(VISIBILITY_BINS[:-1], VISIBILITY_BINS[1:])] + [f"[{VISIBILITY_BINS[-1]},1]"]
    return {
        "frames": int(len(frames)),
        "frame_range": list(frame_range),
        "gt_rows_by_category": {c: int(np.sum(g & (categories == c))) for c in mot17.CATEGORIES},
        "gt_rows_by_class": {mot17.CLASS_NAMES.get(int(c), f"class_{int(c)}"): int(np.sum(g & (gt.cls == c)))
                             for c in np.unique(gt.cls[g])},
        "evaluable_identities": int(len(np.unique(gt.ped_id[ev]))),
        "evaluable_rows_by_visibility": {lab: int(np.sum(bins == i)) for i, lab in enumerate(labels)},
        "frames_without_evaluable_gt": int(len(frames) - len(np.unique(gt.frame[ev]))),
        "detections": int(np.sum(d)),
        "frames_without_detections": int(len(frames) - len(np.unique(det.frame[d]))),
        "detection_score": _stats(det.score[d]),
    }


def run_sequence(sequence: str, info: mot17.SequenceInfo, gt: mot17.GroundTruth, det: mot17.Detections,
                 cfg: Config, tracker_config: TrackerConfig, min_visibility: float) -> SequenceRun:
    split = mot17.half_split(info.seq_length)
    categories = mot17.categorise(gt, min_visibility)
    counts = {half: half_counts(gt, categories, det, rng) for half, rng in split.items()}
    rng = split[EVALUATED_HALF]
    result = run_tracker(tracker_inputs(det, rng), tracker_config, first_frame=rng[0])
    metrics = evaluate_tracking(evaluation_frames(gt, categories, result, rng),
                                float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"]))
    for key in POSITION_FIELDS:
        metrics.pop(key)
    return SequenceRun(sequence, info, split, counts, result, metrics, mot_text(result))


def run_settings(cfg: Config, sequences, tracker_config: TrackerConfig, min_visibility: float,
                 manifest_hash: str | None, member_sha256: dict, archive: dict,
                 variant: str = DETECTION_VARIANT) -> dict:
    return {
        "dataset": mot17.DATASET,
        "licence": mot17.LICENCE,
        "archive": archive,
        "dataset_manifest_hash": manifest_hash,
        "member_sha256": member_sha256,
        "split": "train",
        "sequences": list(sequences),
        "camera_group": camera_group(cfg, sequences),
        "detection_variant": variant,
        "detections": DETECTIONS_NOTE.replace(DETECTION_VARIANT, variant),
        "evaluated_half": EVALUATED_HALF,
        "half_rule": "first = 1 .. N // 2, second = N // 2 + 1 .. N (1-based frames; ByteTrack convention, assumed)",
        "frame_numbering": "original 1-based MOT17 frame numbers, not renumbered",
        "box_convention": "xyxy = (bb_left - 1, bb_top - 1, bb_left - 1 + w, bb_top - 1 + h); output inverts it",
        "tracker": {"name": "IouTracker (feasibility)", **asdict(tracker_config),
                    "input": "every frame of the evaluated half in order, empty state at its first frame"},
        "evaluation_iou_threshold": float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"]),
        "category_rules": {"evaluable": "class 1, consider 1, visibility >= min_visibility",
                           "invisible": "class 1, consider 1, visibility < min_visibility",
                           "ignore": f"class 1 with consider 0, or class in {list(mot17.DISTRACTOR_CLASSES)}",
                           "excluded": "any other class; not ground truth"},
        "min_visibility": float(min_visibility),
        "metrics": METRICS_NOTE,
        "prediction": PREDICTION_NOTE,
    }


def run_feasibility(cfg: Config, archive_path: Path = mot17.DEFAULT_ARCHIVE,
                    manifest_path: Path | None = mot17.DEFAULT_MANIFEST, sequences=None,
                    tracker_config: TrackerConfig = TrackerConfig(), min_visibility: float = 0.0,
                    hash_archive: bool = False, variant: str = DETECTION_VARIANT) -> Mot17Run:
    """Run each sequence separately. Members are verified against the manifest when one is given;
    `hash_archive` also checks the whole archive's SHA-256 (slow for the real 5.9 GB file)."""
    archive_path = Path(archive_path)
    sequences = static_sequences(cfg) if sequences is None else list(sequences)
    manifest = mot17.read_manifest(manifest_path) if manifest_path is not None else None
    archive = {"file": archive_path.name, "size_bytes": archive_path.stat().st_size, "sha256": None,
               "sha256_in_manifest": manifest.get(archive_path.name) if manifest else None}
    if hash_archive:
        archive["sha256"] = file_sha256(archive_path)
        if manifest is not None and archive["sha256"] != archive["sha256_in_manifest"]:
            raise mot17.Mot17ManifestError(f"{archive_path.name} does not match the manifest")
    members: dict[str, str] = {}
    runs = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        available = mot17.list_sequences(zf, "train")
        for seq in sequences:
            if variant not in available.get(seq, ()):
                raise mot17.Mot17FormatError(f"{seq}-{variant} is not in the archive's train split")
            read = lambda rel: mot17.read_verified(zf, mot17.member(zf, seq, variant, rel),
                                                   archive_path.name, manifest, members)
            info = mot17.parse_seqinfo(read("seqinfo.ini"), f"{seq}-{variant}")
            gt = mot17.parse_gt(read("gt/gt.txt"), info)
            det = mot17.parse_detections(read("det/det.txt"), info)
            runs.append(run_sequence(seq, info, gt, det, cfg, tracker_config, min_visibility))
    settings = run_settings(cfg, sequences, tracker_config, min_visibility,
                            file_sha256(Path(manifest_path)) if manifest_path is not None else None,
                            dict(sorted(members.items())), archive, variant)
    settings_hash = config.config_hash({"config_hash": cfg.hash, "settings": settings})
    return Mot17Run(repro.make_experiment_id(NAME, settings_hash, None), settings, settings_hash, cfg.hash,
                    tuple(runs))


def output_name(run: SequenceRun) -> str:
    return f"{run.info.name}.txt"


def results_document(run: Mot17Run, label: str = LABEL) -> dict:
    sequences = {}
    for s in run.sequences:
        t = s.tracker
        sequences[s.sequence] = {
            "sequence_info": asdict(s.info),
            "split": {half: list(r) for half, r in s.split.items()},
            "counts": s.counts,
            "evaluated_half": EVALUATED_HALF,
            "tracker": {"created": t.n_created, "confirmed": t.n_confirmed,
                        "deleted_tentative": t.n_deleted_tentative, "terminated": t.n_terminated,
                        "output_observations": int(sum(len(x.frames) for x in t.tracks))},
            "metrics_project_specific": s.metrics,
            "position_error": "not applicable: MOT17 has no world coordinates",
            "mot_output": {"file": f"mot_tracker_output/{output_name(s)}", "lines": s.mot_text.count("\n"),
                           "sha256": hashlib.sha256(s.mot_text.encode("utf-8")).hexdigest()},
        }
    return {
        "label": label,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "metrics_note": METRICS_NOTE,
        "not_satisfied": [n.replace(DETECTION_VARIANT, run.settings["detection_variant"]) for n in NOT_SATISFIED],
        "pooled_result": None,
        "sequences": sequences,
    }


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _json(data) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=_native) + "\n"


def sequences_csv(run: Mot17Run) -> str:
    fields = ("evaluable_observations", "tracker_observations", "tp", "fp", "fn", "ignored", "idsw", "frag",
              "mota", "idf1")
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["sequence", "half", "first_frame", "last_frame", "detections"] + list(fields))
    for s in run.sequences:
        c = s.counts[EVALUATED_HALF]
        writer.writerow([s.sequence, EVALUATED_HALF, *c["frame_range"], c["detections"]]
                        + ["" if s.metrics[k] is None else s.metrics[k] for k in fields])
    return buffer.getvalue()


def check_output_location(out_dir: Path) -> None:
    """Local-only outputs: refuse any location inside the repository except the Git-ignored runs/."""
    out, repo = Path(out_dir).resolve(), config.REPO_ROOT.resolve()
    if out.is_relative_to(repo) and not out.is_relative_to(repo / "runs"):
        raise ValueError(f"{out} is inside the repository; local-only outputs go outside it or under runs/")


def save_run(out_dir: Path, run: Mot17Run, cfg: Config, label: str = LABEL) -> Path:
    """Write the run to a new directory `out_dir`; raise FileExistsError if it exists."""
    out = Path(out_dir)
    check_output_location(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    files = {
        "results.json": _json(results_document(run, label)),
        "sequences.csv": sequences_csv(run),
        "config.resolved.yaml": yaml.safe_dump(cfg.canonical_dict(), sort_keys=True, allow_unicode=True),
    }
    files.update({f"mot_tracker_output/{output_name(s)}": s.mot_text for s in run.sequences})
    (out / "mot_tracker_output").mkdir(parents=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "experiment_id": run.experiment_id,
        "experiment_name": NAME,
        "label": label,
        "dataset": mot17.DATASET,
        "licence": mot17.LICENCE,
        "sequences": run.settings["sequences"],
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "dataset_manifest_hash": run.settings["dataset_manifest_hash"],
        "archive": run.settings["archive"],
        "member_sha256": run.settings["member_sha256"],
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in files},
    }
    (out / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    return out
