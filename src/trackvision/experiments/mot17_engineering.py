"""E1, E2 and E3: limited MOT17 feasibility on public detections (decision 0006).

  E1  the existing IoU tracker on one public detection set, the second half of the seven training sequences with
      ground truth, scored by official TrackEval in its isolated environment, per sequence
  E2  precision, recall, false positives and false negatives of the public detections (project diagnostic)
  E3  image-space error profile of the tracker output (project diagnostic)

NOT P-1, NOT P-9 AND NOT A MOT17 BENCHMARK. P-1 needs private YOLO detections and ByteTrack and BoT-SORT, none of
which is available. P-9 needs ETH/UCY video and the world-coordinate protocol. Frames stay at each sequence's native
cadence; nothing is resampled to 2.5 Hz, so the section 3.7 severities are not computed. Localization is in pixels
and is never converted to metres. MOT17 is never used for prediction: this module computes no windows and no ADE.

Results are per sequence and never pooled. Official TrackEval values and project diagnostics sit in separate
sections, and nothing here is called IDSW except TrackEval's own value.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments import association_provenance, mot17_feasibility as mf
from trackvision.experiments import mot17_trackeval as mt
from trackvision.experiments.mot17_trackeval import check_output_location
from trackvision.io import mot17
from trackvision.metrics.detection import evaluate_frame, precision_recall_f1
from trackvision.metrics.error_profile import error_profile, match_frames, track_frame_denominator

NAME = "mot17-engineering"
PRIMARY_VARIANT = "FRCNN"
OPTIONAL_VARIANTS = ("DPM", "SDP")
DIAGNOSTIC = "project diagnostic; not an official metric, not a frozen P-1 output"
GAP_NOTE = ("engineering diagnostic, not a section 3.7 implementation: 0.4 s gap limits converted to whole frames at the "
            "native cadence; not comparable across frame rates and never projected onto a degradation curve")
LABEL = ("LIMITED MOT17 FEASIBILITY (decision 0006). Public {variant} detections supplied with MOT17, the existing IoU "
         "feasibility tracker, second half of seven training sequences, per sequence, native frame cadence. Not P-1, "
         "not P-9, not a full MOT17 benchmark, not ByteTrack or BoT-SORT, no private detector. MOT17 is never used for "
         "prediction. v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
NOT_SATISFIED = ("section 6.2: private YOLO detections - public detections used instead",
                 "section 11.5: ByteTrack and BoT-SORT - the existing IoU feasibility tracker used instead",
                 "section 3.7: achieved severities on the 0.4 s grid - native cadence diagnostics used instead",
                 "section 6.2: first half for tuning - no tuning performed")


def e1_sequences(cfg: Config) -> list[str]:
    """The seven training sequences with ground truth, static and moving, in sequence order."""
    m = cfg.frozen["datasets"]["mot17"]
    return sorted(str(s) for s in (*m["static_sequences"], *m["moving_sequences"]))


def gap_steps_at(frame_rate: int, steps: int, sampling_interval_s: float) -> int:
    """An engineering diagnostic, not a section 3.7 implementation (decision 0006): the limit's duration on the 0.4 s
    grid, rounded down to whole frames at the native cadence. It is not a resampling rule."""
    return int(steps * sampling_interval_s * frame_rate + 1e-9)


def bottom_centre(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]])


def detection_diagnostics(gt: mot17.GroundTruth, categories: np.ndarray, det: mot17.Detections,
                          frame_range: tuple[int, int], iou_threshold: float) -> dict:
    """E2 over one frame range. Excluded classes are dropped; ignore regions are neither true nor false positives."""
    totals = dict.fromkeys(("tp", "fp", "fn", "ignored_matches", "fp_overlapping_visible", "fp_background"), 0)
    evaluable = detections = 0
    for f in range(frame_range[0], frame_range[1] + 1):
        d = det.boxes[mf._rows_at(det.frame, f)]
        rows = mf._rows_at(gt.frame, f)
        cats = categories[rows]
        keep = cats != "excluded"
        counts = evaluate_frame(d, gt.boxes[rows][keep], cats[keep] == "evaluable", cats[keep] == "ignore", iou_threshold)
        for key in totals:
            totals[key] += getattr(counts, key)
        evaluable += counts.n_evaluable
        detections += counts.n_detections
    return {"note": DIAGNOSTIC, "iou_threshold": iou_threshold, "frames": frame_range[1] - frame_range[0] + 1,
            "detections": detections, "evaluable_gt_observations": evaluable, **totals,
            "denominators": {"precision": "tp + fp", "recall": "tp + fn = evaluable_gt_observations"},
            **precision_recall_f1(totals["tp"], totals["fp"], totals["fn"])}


def image_profile(run: mf.SequenceRun, gt: mot17.GroundTruth, categories: np.ndarray, cfg: Config) -> dict:
    """E3 for one sequence: matches on the evaluated half, at native cadence, with pixel positions."""
    rng = run.split[mf.EVALUATED_HALF]
    frames = [dataclasses.replace(fr, track_positions=bottom_centre(fr.track_boxes), gt_positions=bottom_centre(fr.gt_boxes))
              for fr in mf.evaluation_frames(gt, categories, run.tracker, rng)]
    assoc = cfg.frozen["association"]
    dt = cfg.frozen["protocol"]["static"]["sampling_interval_s"]
    matched = match_frames(frames, float(assoc["tracker_gt_assoc_iou_threshold"]))
    frag = gap_steps_at(run.info.frame_rate, assoc["frag_gap_max_steps"], dt)
    switch = gap_steps_at(run.info.frame_rate, assoc["switch_gap_max_steps"], dt)
    profile = error_profile(matched, frag, switch, "pixels", track_frame_denominator(matched),
                            exchange_reason="section 3.7's eligible-frame denominator needs D_MAX in metres; MOT17 has "
                                            "no world coordinates")
    return {"note": DIAGNOSTIC, "cadence": "native", "frame_rate": run.info.frame_rate, "gap_conversion": GAP_NOTE,
            "position_point": "box bottom-centre, pixels", **profile,
            "not_computed": "section 3.7 severities: they are defined on the 0.4 s grid and, for A and D, in metres"}


def run_variant(out_dir: Path, cfg: Config, variant: str, python: Path, source: Path, archive: Path, manifest: Path,
                sequences) -> dict:
    """One detection set: tracker, official TrackEval, then E2 and E3. Writes under a new `out_dir`."""
    label = LABEL.format(variant=variant)
    run = mf.run_feasibility(cfg, archive, manifest, sequences=sequences, variant=variant)
    tracker_dir, eval_dir = out_dir / "tracker", out_dir / "trackeval"
    mf.save_run(tracker_dir, run, cfg, label)
    mt.export_inputs(eval_dir, tracker_dir, archive, manifest)
    mt.validate_export(eval_dir, tracker_dir, archive)
    code = mt.run_trackeval(eval_dir, python, source, cfg)
    if code != 0:
        raise RuntimeError(f"TrackEval exited with code {code}; see {eval_dir / 'logs'}")
    official = mt.write_record(eval_dir, tracker_dir, source, cfg, label)["per_sequence_official"]
    threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    out = {}
    with zipfile.ZipFile(archive, "r") as zf:
        for s in run.sequences:
            folder = f"{s.sequence}-{variant}"
            gt = mot17.parse_gt(zf.read(mot17.member(zf, s.sequence, variant, "gt/gt.txt")), s.info)
            det = mot17.parse_detections(zf.read(mot17.member(zf, s.sequence, variant, "det/det.txt")), s.info)
            categories = mot17.categorise(gt)
            rng = s.split[mf.EVALUATED_HALF]
            out[s.sequence] = {
                "frames_evaluated": list(rng), "frame_rate": s.info.frame_rate,
                "official_trackeval": official[folder],
                "project_diagnostics": {"detection": detection_diagnostics(gt, categories, det, rng, threshold),
                                        "error_profile": image_profile(s, gt, categories, cfg)}}
    return out


def run(out_dir: Path, cfg: Config, python: Path, source: Path, variants=(PRIMARY_VARIANT,),
        archive: Path = mot17.DEFAULT_ARCHIVE, manifest: Path = mot17.DEFAULT_MANIFEST, sequences=None) -> dict:
    """Run E1 to E3 for each detection set in a new directory, and write results.json and manifest.json."""
    out_dir = Path(out_dir)
    check_output_location(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    unknown = set(variants) - {PRIMARY_VARIANT, *OPTIONAL_VARIANTS}
    if unknown or PRIMARY_VARIANT not in variants:
        raise ValueError(f"variants must include {PRIMARY_VARIANT} and only use {mot17.VARIANTS}, got {list(variants)}")
    sequences = e1_sequences(cfg) if sequences is None else list(sequences)
    results = {v: run_variant(out_dir / v, cfg, v, Path(python), Path(source), Path(archive), Path(manifest), sequences)
               for v in variants}
    doc = {"label": LABEL.format(variant=" / ".join(variants)), "experiment": "E1, E2, E3", "research_results": False,
           "not_p1": True, "not_p9": True, "primary_detection_set": PRIMARY_VARIANT,
           "optional_detection_sets": [v for v in variants if v != PRIMARY_VARIANT], "cadence": "native",
           "sequences": sequences, "pooled_result": None, "not_satisfied": list(NOT_SATISFIED),
           **association_provenance.mot17_field(),
           "prediction": mf.PREDICTION_NOTE, "by_detection_set": results}
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    (out_dir / "results.json").write_text(text, encoding="utf-8", newline="\n")
    manifest_doc = {"name": NAME, "label": doc["label"], "config_hash": cfg.hash, "trackeval_commit": mt.TRACKEVAL_COMMIT,
                    "dataset": mot17.DATASET, "licence": mot17.LICENCE,
                    "dataset_manifest_hash": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                    "results_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
                    "trackvision_version": __version__, "git": repro.git_state(config.REPO_ROOT),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "environment": repro.environment_info()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                           newline="\n")
    return doc
