"""Official TrackEval evaluation of the M1 MOT17 feasibility tracker outputs (M2).

Runs in the project .venv and never imports TrackEval. TrackEval (pinned commit, unmodified, Python 3.11
environment) is run as a separate process by `scripts/trackeval_mot17.py`; this module prepares its
MOTChallenge input layout, launches it, validates what it returns and compares it with M1.

NOT A MOT17 BENCHMARK RESULT. The tracker output being scored is M1's: public FRCNN detections and the
IoU feasibility tracker, second half of three static training sequences only. The metrics are TrackEval's
official MOTChallenge implementation (HOTA, CLEAR, Identity, Count); nothing else is official. Per
sequence only: TrackEval's COMBINED row is never reported. Outputs stay outside the repository (or under
the Git-ignored runs/); the exported ground truth is MOT17 content (CC BY-NC-SA) and is never committed.

Export (pinned TrackEval source, datasets/mot_challenge_2d_box.py):
  <input>/gt/mot_challenge/seqmaps/MOT17-half.txt                   header "name", then one sequence per row
  <input>/gt/mot_challenge/MOT17-half/<seq>/seqinfo.ini             [Sequence], seqLength = second-half length
  <input>/gt/mot_challenge/MOT17-half/<seq>/gt/gt.txt               second-half rows of the original gt.txt
  <input>/trackers/mot_challenge/MOT17-half/<tracker>/data/<seq>.txt  M1's tracker rows
Frames are renumbered f -> f - offset with offset = N // 2 (io.mot17.half_split), so the second half starts
at frame 1; TrackEval rejects any frame outside 1..seqLength. Only the frame field is rewritten: every other
character of every row is copied unchanged (annotations, classes, flags, coordinates, ids, confidences),
and restoring the offset reproduces the source rows exactly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.io import mot17

NAME = "mot17-trackeval"
BENCHMARK = "MOT17"
SPLIT = "half"
GT_SET = f"{BENCHMARK}-{SPLIT}"
TRACKER_NAME = "trackvision-iou-feasibility"
DETECTION_VARIANT = "FRCNN"
SCRIPT = config.REPO_ROOT / "scripts" / "trackeval_mot17.py"
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)                   # trackeval/metrics/hota.py:17
HOTA_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA")
LABEL = ("Official TrackEval (MOTChallenge 2D box) metrics for M1's tracker output: public FRCNN detections and "
         "the IoU feasibility tracker on the second half of MOT17-02, -04 and -09 only. Not a MOT17 benchmark "
         "result, not the private-detector protocol, not ByteTrack/BoT-SORT; per sequence, never pooled.")


class ExportError(ValueError):
    """Raised when inputs or the exported layout fail a validation check."""


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame(line: str) -> tuple[int, str]:
    field, sep, rest = line.partition(",")
    if not sep or not field.strip().lstrip("-").isdigit():
        raise ExportError(f"row has no integer frame field: {line!r}")
    return int(field), rest


def renumber(text: str, frame_range: tuple[int, int], keep_outside: bool) -> str:
    """Rows of `text` with frames in the inclusive range, frame field shifted so the range starts at 1.
    Rows outside the range are dropped if `keep_outside` is True, otherwise they raise."""
    first, last = frame_range
    out = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f, rest = _frame(line)
        if not first <= f <= last:
            if keep_outside:
                continue
            raise ExportError(f"frame {f} outside {first}..{last}: {line!r}")
        out.append(f"{f - first + 1},{rest}\n")
    return "".join(out)


def restore(text: str, offset: int) -> str:
    """Inverse of `renumber` for kept rows: shift the frame field back by `offset`."""
    out = []
    for line in text.splitlines():
        f, rest = _frame(line)
        out.append(f"{f + offset},{rest}\n")
    return "".join(out)


def seqinfo_text(info: mot17.SequenceInfo, seq_length: int) -> str:
    return (f"[Sequence]\nname={info.name}\nimDir={info.im_dir}\nframeRate={info.frame_rate}\n"
            f"seqLength={seq_length}\nimWidth={info.im_width}\nimHeight={info.im_height}\nimExt={info.im_ext}\n")


def layout(input_root: Path, folder: str, gt_set: str = GT_SET, tracker: str = TRACKER_NAME) -> dict[str, Path]:
    gt = input_root / "gt" / "mot_challenge"
    return {"seqmap": gt / "seqmaps" / f"{gt_set}.txt",
            "seqinfo": gt / gt_set / folder / "seqinfo.ini",
            "gt": gt / gt_set / folder / "gt" / "gt.txt",
            "tracker": input_root / "trackers" / "mot_challenge" / gt_set / tracker / "data" / f"{folder}.txt"}


def write_sequence(input_root: Path, info: mot17.SequenceInfo, seq_length: int, gt_text: str,
                   tracker_text: str, gt_set: str = GT_SET, tracker: str = TRACKER_NAME) -> dict[str, Path]:
    paths = layout(input_root, info.name, gt_set, tracker)
    for key, text in (("seqinfo", seqinfo_text(info, seq_length)), ("gt", gt_text), ("tracker", tracker_text)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_text(text, encoding="utf-8", newline="\n")
    return paths


def write_seqmap(input_root: Path, folders, gt_set: str = GT_SET) -> Path:
    path = layout(input_root, "", gt_set)["seqmap"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name\n" + "".join(f"{f}\n" for f in folders), encoding="utf-8", newline="\n")
    return path


@dataclass(frozen=True)
class SequenceExport:
    sequence: str                    # MOT17-NN
    folder: str                      # MOT17-NN-FRCNN
    original_frames: tuple           # (first, last) of the second half, original numbering
    frame_offset: int                # original = exported + offset
    seq_length: int
    gt_rows: int
    tracker_rows: int
    frames_without_tracker_rows: int
    gt_rows_by_class: dict


def check_output_location(out_dir: Path) -> None:
    out, repo = Path(out_dir).resolve(), config.REPO_ROOT.resolve()
    if out.is_relative_to(repo) and not out.is_relative_to(repo / "runs"):
        raise ValueError(f"{out} is inside the repository; outputs go outside it or under runs/")


def verified_m1(m1_dir: Path) -> tuple[dict, dict]:
    """M1's results.json and manifest.json, after checking every M1 file against its recorded hash."""
    m1_dir = Path(m1_dir)
    manifest = json.loads((m1_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        if sha256(m1_dir / name) != digest:
            raise ExportError(f"M1 file {name} does not match its recorded hash")
    return json.loads((m1_dir / "results.json").read_text(encoding="utf-8")), manifest


def tracker_variant(m1: dict) -> str:
    """The detection set of a stored tracker run."""
    return m1["settings"]["detection_variant"]


def export_inputs(out_dir: Path, m1_dir: Path, archive_path: Path = mot17.DEFAULT_ARCHIVE,
                  manifest_path: Path | None = mot17.DEFAULT_MANIFEST) -> tuple[SequenceExport, ...]:
    """Write the TrackEval input layout under out_dir/trackeval_input (a new directory)."""
    out_dir, archive_path = Path(out_dir), Path(archive_path)
    check_output_location(out_dir)
    input_root = out_dir / "trackeval_input"
    if input_root.exists():
        raise FileExistsError(f"{input_root} exists; inputs are never overwritten")
    m1, m1_manifest = verified_m1(m1_dir)
    variant = tracker_variant(m1)
    entries = mot17.read_manifest(manifest_path) if manifest_path is not None else None
    exports, record = [], {}
    with zipfile.ZipFile(archive_path, "r") as zf:
        for seq, s in m1["sequences"].items():
            folder = f"{seq}-{variant}"
            read = lambda rel: mot17.read_verified(zf, mot17.member(zf, seq, variant, rel),
                                                   archive_path.name, entries, record)
            info = mot17.parse_seqinfo(read("seqinfo.ini"), folder)
            first, last = mot17.half_split(info.seq_length)["second"]
            if [first, last] != list(s["split"]["second"]):
                raise ExportError(f"{seq}: M1 second half {s['split']['second']} differs from {first}..{last}")
            gt = renumber(read("gt/gt.txt").decode("utf-8"), (first, last), keep_outside=True)
            m1_text = (Path(m1_dir) / s["mot_output"]["file"]).read_text(encoding="utf-8")
            tracker = renumber(m1_text, (first, last), keep_outside=False)
            write_sequence(input_root, info, last - first + 1, gt, tracker)
            classes = [int(line.split(",")[7]) for line in gt.splitlines()]
            frames = {int(line.split(",", 1)[0]) for line in tracker.splitlines()}
            exports.append(SequenceExport(seq, folder, (first, last), first - 1, last - first + 1,
                                          len(classes), tracker.count("\n"),
                                          last - first + 1 - len(frames),
                                          {str(c): classes.count(c) for c in sorted(set(classes))}))
    write_seqmap(input_root, [e.folder for e in exports])
    files = {p.relative_to(out_dir).as_posix(): sha256(p) for p in sorted(input_root.rglob("*")) if p.is_file()}
    export_manifest = {
        "m1_run": {"dir": str(Path(m1_dir)), "experiment_id": m1_manifest["experiment_id"],
                   "files": m1_manifest["files"]},
        "archive": {"file": archive_path.name, "size_bytes": archive_path.stat().st_size,
                    "dataset_manifest_hash": sha256(manifest_path) if manifest_path is not None else None,
                    "members_sha256": dict(sorted(record.items()))},
        "sequences": [asdict(e) for e in exports],
        "files": files,
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2, sort_keys=True) + "\n",
                                                  encoding="utf-8", newline="\n")
    return tuple(exports)


def validate_export(out_dir: Path, m1_dir: Path, archive_path: Path = mot17.DEFAULT_ARCHIVE) -> list[dict]:
    """Independent re-check of the exported layout against M1's files and the archive."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    m1, _ = verified_m1(m1_dir)
    variant = tracker_variant(m1)
    input_root = out_dir / "trackeval_input"
    seqmap = layout(input_root, "")["seqmap"].read_text(encoding="utf-8").splitlines()
    folders = [e["folder"] for e in manifest["sequences"]]
    if seqmap != ["name"] + folders:
        raise ExportError(f"seqmap {seqmap} does not list {folders}")
    checks = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for e in manifest["sequences"]:
            paths = layout(input_root, e["folder"])
            first, last = e["original_frames"]
            ini = dict(line.split("=", 1) for line in paths["seqinfo"].read_text(encoding="utf-8").splitlines()[1:])
            gt, tracker = (paths[k].read_text(encoding="utf-8") for k in ("gt", "tracker"))
            source_gt = zf.read(mot17.member(zf, e["sequence"], variant, "gt/gt.txt")).decode("utf-8")
            source_gt_half = "".join(f"{line}\n" for line in source_gt.splitlines()
                                     if line.strip() and first <= int(line.split(",", 1)[0]) <= last)
            m1_text = (Path(m1_dir) / m1["sequences"][e["sequence"]]["mot_output"]["file"]).read_text(encoding="utf-8")
            gt_keys = [tuple(line.split(",")[:2]) for line in gt.splitlines()]
            tr_keys = [tuple(line.split(",")[:2]) for line in tracker.splitlines()]
            gt_frames = {int(f) for f, _ in gt_keys}
            tr_frames = {int(f) for f, _ in tr_keys}
            check = {
                "sequence": e["folder"],
                "seq_length_matches": int(ini["seqLength"]) == e["seq_length"] == last - first + 1,
                "gt_rows_preserved": restore(gt, e["frame_offset"]) == source_gt_half,
                "tracker_rows_preserved": restore(tracker, e["frame_offset"]) == m1_text,
                "frames_within_1_to_seqLength": min(gt_frames | tr_frames) >= 1 and max(gt_frames | tr_frames) <= e["seq_length"],
                "no_duplicate_gt_frame_id": len(set(gt_keys)) == len(gt_keys),
                "no_duplicate_tracker_frame_id": len(set(tr_keys)) == len(tr_keys),
                "frames_without_gt_rows": e["seq_length"] - len(gt_frames),
                "frames_without_tracker_rows": e["seq_length"] - len(tr_frames),
                "files_match_export_manifest": all(sha256(out_dir / f) == h for f, h in manifest["files"].items()),
            }
            checks.append(check)
            failed = [k for k, v in check.items() if v is False]
            if failed:
                raise ExportError(f"{e['folder']}: export validation failed: {failed}")
    return checks


def trackeval_config(out_dir: Path, trackeval_source: Path, cfg: Config, benchmark: str = BENCHMARK,
                     split: str = SPLIT, tracker: str = TRACKER_NAME) -> dict:
    """TrackEval configuration; every key name is taken from the pinned source (eval.py, metrics, dataset)."""
    out_dir = Path(out_dir)
    threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    return {
        "trackeval_source": str(trackeval_source),
        "results_file": str(out_dir / "trackeval_results.json"),
        "dataset_config": {
            "GT_FOLDER": str(out_dir / "trackeval_input" / "gt" / "mot_challenge"),
            "TRACKERS_FOLDER": str(out_dir / "trackeval_input" / "trackers" / "mot_challenge"),
            "OUTPUT_FOLDER": str(out_dir / "trackeval_output"),
            "TRACKERS_TO_EVAL": [tracker],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": split,
            "DO_PREPROC": True,
            "PRINT_CONFIG": True,
        },
        "eval_config": {
            "USE_PARALLEL": False,
            "BREAK_ON_ERROR": True,
            "LOG_ON_ERROR": str(out_dir / "logs" / "error_log.txt"),
            "PRINT_RESULTS": True,
            "PRINT_CONFIG": True,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
        },
        # Count is not listed: Evaluator.evaluate always appends it (eval.py:58) and rejects duplicates.
        "metrics": [["HOTA", None], ["CLEAR", {"THRESHOLD": threshold, "PRINT_CONFIG": True}],
                    ["Identity", {"THRESHOLD": threshold, "PRINT_CONFIG": True}]],
    }


def run_trackeval(out_dir: Path, python: Path, trackeval_source: Path, cfg: Config, **dataset) -> int:
    """Write the configuration and run the TrackEval script in its own interpreter; logs go to out_dir/logs."""
    out_dir = Path(out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "trackeval_config.json"
    config_path.write_text(json.dumps(trackeval_config(out_dir, trackeval_source, cfg, **dataset), indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    done = subprocess.run([str(python), "-I", str(SCRIPT), str(config_path)], capture_output=True, text=True)
    (out_dir / "logs" / "stdout.txt").write_text(done.stdout, encoding="utf-8", newline="\n")
    (out_dir / "logs" / "stderr.txt").write_text(done.stderr, encoding="utf-8", newline="\n")
    (out_dir / "logs" / "exit_code.txt").write_text(f"{done.returncode}\n", encoding="utf-8", newline="\n")
    return done.returncode


def evaluable_gt(gt_text: str) -> int:
    """Rows TrackEval keeps as ground truth: class 1 (pedestrian) with a non-zero consider flag."""
    rows = [line.split(",") for line in gt_text.splitlines()]
    return sum(1 for r in rows if int(r[7]) == mot17.PEDESTRIAN and int(r[6]) != 0)


def validate_results(out_dir: Path, gt_set: str = GT_SET, tracker: str = TRACKER_NAME) -> dict:
    """Per-sequence results after consistency checks; raises ExportError on any failure."""
    out_dir = Path(out_dir)
    res = json.loads((out_dir / "trackeval_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    expected = [e["folder"] for e in manifest["sequences"]]
    problems = []
    if set(res["messages"].values()) != {"Success"}:
        problems.append(f"TrackEval messages {res['messages']}")
    if sorted(res["sequences"]) != sorted(expected):
        problems.append(f"sequences {sorted(res['sequences'])} != {sorted(expected)}")
    if not np.allclose(res["hota_alphas"], HOTA_ALPHAS) or len(res["hota_alphas"]) != 19:
        problems.append(f"HOTA alphas {res['hota_alphas']}")
    detailed = (out_dir / "trackeval_output" / tracker / "pedestrian_detailed.csv").read_text(encoding="utf-8")
    rows = [line.split(",", 1)[0] for line in detailed.splitlines()[1:]]
    if rows != sorted(expected) + ["COMBINED"]:
        problems.append(f"detailed CSV rows {rows}")
    for e in manifest["sequences"]:
        r = res["sequences"].get(e["folder"])
        if r is None:
            continue
        gt_text = layout(out_dir / "trackeval_input", e["folder"], gt_set, tracker)["gt"].read_text(encoding="utf-8")
        c, i, n = r["CLEAR"], r["Identity"], r["Count"]
        consistency = {
            "GT_Dets == class-1 consider-1 rows": n["GT_Dets"] == evaluable_gt(gt_text),
            "Dets <= tracker rows": n["Dets"] <= e["tracker_rows"],
            "CLR_TP + CLR_FN == GT_Dets": c["CLR_TP"] + c["CLR_FN"] == n["GT_Dets"],
            "CLR_TP + CLR_FP == Dets": c["CLR_TP"] + c["CLR_FP"] == n["Dets"],
            "IDTP + IDFN == GT_Dets": i["IDTP"] + i["IDFN"] == n["GT_Dets"],
            "IDTP + IDFP == Dets": i["IDTP"] + i["IDFP"] == n["Dets"],
            "CLR_Frames == seqLength": c["CLR_Frames"] == e["seq_length"],
            "19 values per HOTA field": all(len(r["HOTA"][f]) == 19 for f in HOTA_FIELDS),
        }
        problems += [f"{e['folder']}: {k}" for k, ok in consistency.items() if not ok]
    if problems:
        raise ExportError("TrackEval result validation failed: " + "; ".join(problems))
    return res


def headline(r: dict) -> dict:
    """TrackEval's per-sequence headline values; HOTA-family values are means over the 19 alphas
    (as TrackEval's summary and detailed outputs report them, metrics/_base_metric.py:83, 132)."""
    c, i, n = r["CLEAR"], r["Identity"], r["Count"]
    out = {f: float(np.mean(r["HOTA"][f])) for f in HOTA_FIELDS}
    out.update({k: c[k] for k in ("MOTA", "MOTP", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "MT", "PT", "ML",
                                  "CLR_Frames")})
    out.update({k: i[k] for k in ("IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP")})
    out.update({k: n[k] for k in ("Dets", "GT_Dets", "IDs", "GT_IDs")})
    return out


def compare_with_m1(m1_results: dict, trackeval_results: dict) -> dict:
    """Side by side, per sequence: M1 project-specific metrics and TrackEval's official ones."""
    out = {}
    for seq, s in m1_results["sequences"].items():
        m, t = s["metrics_project_specific"], headline(trackeval_results["sequences"][f"{seq}-{tracker_variant(m1_results)}"])
        out[seq] = {
            "evaluable_gt": {"m1": m["evaluable_observations"], "trackeval": t["GT_Dets"]},
            "tracker_detections_counted": {"m1": m["tracker_observations"] - m["ignored"], "trackeval": t["Dets"]},
            "distractor_or_ignored_tracker_boxes": {"m1": m["ignored"],
                                                    "trackeval": m["tracker_observations"] - t["Dets"]},
            "tp": {"m1": m["tp"], "trackeval": t["CLR_TP"]},
            "fp": {"m1": m["fp"], "trackeval": t["CLR_FP"]},
            "fn": {"m1": m["fn"], "trackeval": t["CLR_FN"]},
            "idsw": {"m1": m["idsw"], "trackeval": t["IDSW"]},
            "frag": {"m1": m["frag"], "trackeval": t["Frag"]},
            "mota": {"m1": m["mota"], "trackeval": t["MOTA"]},
            "idf1": {"m1": m["idf1"], "trackeval": t["IDF1"]},
            "idtp": {"m1": m["idtp"], "trackeval": t["IDTP"]},
        }
    return out


def write_record(out_dir: Path, m1_dir: Path, trackeval_source: Path, cfg: Config, label: str = LABEL) -> dict:
    """results.json and manifest.json for a completed, validated run (no COMBINED values)."""
    out_dir = Path(out_dir)
    res = validate_results(out_dir)
    m1, _ = verified_m1(m1_dir)
    export_manifest = json.loads((out_dir / "export_manifest.json").read_text(encoding="utf-8"))
    results = {
        "label": label,
        "trackeval_commit": TRACKEVAL_COMMIT,
        "per_sequence_official": {seq: headline(r) for seq, r in sorted(res["sequences"].items())},
        "per_sequence_full": res["sequences"],
        "hota_alphas": res["hota_alphas"],
        "comparison_with_m1": compare_with_m1(m1, res),
        "combined_row": "not reported (TrackEval's pedestrian_detailed.csv contains one; it is not used)",
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8", newline="\n")
    produced = ["trackeval_config.json", "trackeval_results.json", "results.json", "report.md", "logs/stdout.txt",
                "logs/stderr.txt", "logs/exit_code.txt", f"trackeval_output/{TRACKER_NAME}/pedestrian_detailed.csv"]
    manifest = {
        "experiment_name": NAME,
        "label": label,
        "trackeval": {"commit": TRACKEVAL_COMMIT, "source": str(trackeval_source),
                      "source_head": _git_head(Path(trackeval_source)), "environment": res["environment"]},
        "runner_script": {"path": "scripts/trackeval_mot17.py", "sha256": sha256(SCRIPT)},
        "inputs": export_manifest,
        "outputs": {f: sha256(out_dir / f) for f in produced if (out_dir / f).exists()},
        "config_hash": cfg.hash,
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment_trackvision": repro.environment_info(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8", newline="\n")
    return results


def _git_head(source: Path) -> str | None:
    """The commit the source checkout points at, read from .git files (no Git command is run)."""
    head = source / ".git" / "HEAD"
    if not head.exists():
        return None
    text = head.read_text().strip()
    if not text.startswith("ref:"):
        return text
    ref = text.split(" ", 1)[1]
    loose = source / ".git" / ref
    if loose.exists():
        return loose.read_text().strip()
    packed = source / ".git" / "packed-refs"
    for line in packed.read_text().splitlines() if packed.exists() else []:
        if line.endswith(" " + ref):
            return line.split()[0]
    return None
