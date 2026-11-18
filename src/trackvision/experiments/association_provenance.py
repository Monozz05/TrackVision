"""Run-level `association_mechanism` provenance for the Phase 5 and 6 records (spec v4.15 section 3.7; decision 0006).

Section 3.7 is frozen: every experiment records which mechanism it used, from exactly two values. This module meets
that requirement in run-level metadata. It does not touch the section 11.3 schema or `experiments.record`, so proposal
0003, which would add the field to that schema, stays unadopted.

A value is written only where the implementation and the stored run evidence it directly:
  E1 and E3       tracker box, ground-truth box, one-to-one assignment at the frozen threshold, on real MOT17 footage
                  with public detections. MOT17 has no prediction step, so the mechanism ends at the ground-truth identity
  E6, E7 and E8   their input curves are ground-truth trajectories corrupted by the injectors, whose lineage maps
                  (`Fragmentation.origin`, `IdentitySwitch.occupant`) carry identity. No tracker or box is involved

E4 and E5 HAVE NO ESTABLISHED VALUE, and the field is recorded as null with the reason. They are never left out, and
no third value exists. See `SYNTHETIC_TRACKER_BLOCKER`.

Stored records are never edited: results keep their bytes and hashes, and each run gains one new sidecar file that
binds itself to the record by SHA-256. No numerical result is read into or changed by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trackvision.config import Config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments.store import RunStore
from trackvision.inject.association import REAL_TRACKER_IOU, SYNTHETIC_GT_LINEAGE, association_errors
from trackvision.io.atomic import atomic_write_text

SCHEMA = "trackvision-association-provenance/1"
SIDECAR = "association_provenance.json"
ESTABLISHED, UNRESOLVED = "established", "unresolved"
MOT17_TRACKER = "existing IoU tracker on public MOT17 detections"
CHAIN = ("metrics/tracking.py evaluate_tracking and metrics/error_profile.py match_frames call "
         "tracking/assignment.py gated_iou_assignment: tracker box, ground-truth box, one-to-one Hungarian assignment at "
         "frozen.association.tracker_gt_assoc_iou_threshold, ground-truth identity")
MOT17_NOTE = ("The value names the association mechanism only. It is no claim of P-1 or P-9. The chain ends at the "
              "ground-truth identity, because MOT17 is never used for prediction (section 6.1.1).")
MOT17_DOES_NOT_APPLY = {"E2": "public detections matched to ground truth with no tracker and no identity",
                        "official_trackeval": ("TrackEval matches internally: CLEAR and Identity at THRESHOLD 0.5 as "
                                               "configured, HOTA over 19 alphas; it is not the project's association")}
SYNTHETIC_TRACKER_BLOCKER = (
    "No permitted value can be recorded. The implementation matches tracker boxes to generated boxes by Hungarian IoU at "
    "the frozen threshold, then carries the identity into world-coordinate prediction, with no injector and no lineage "
    "map. So synthetic_gt_lineage is false for this record: section 3.7 defines it through the injector's lineage map and "
    "permits no box association. The only other value of section 3.7 names a real tracker, and decision 0006 item 8 "
    "forbids it for synthetic output. Resolving this needs a project-owner decision.")
LINEAGE_EVIDENCE = ("Input curves come from ground-truth trajectories and the A, B, C and D injectors. "
                    "Fragmentation.origin and IdentitySwitch.occupant carry identity; A and B never change it.")


def established(experiment: str, mechanism: str, tracker: str | None, evidence: list, applies_to: list,
                does_not_apply_to: dict | None = None, note: str | None = None) -> dict:
    """An entry whose value passes the section 3.7 category check; ground truth exists in all of these records."""
    errors = association_errors(mechanism, tracker, True)
    if errors:
        raise ValueError(f"{experiment}: {errors}")
    return {"experiment": experiment, "association_mechanism": mechanism, "status": ESTABLISHED, "evidence": list(evidence),
            "applies_to": list(applies_to), "does_not_apply_to": dict(does_not_apply_to or {}), "note": note}


def unresolved(experiment: str, evidence: list) -> dict:
    return {"experiment": experiment, "association_mechanism": None, "status": UNRESOLVED,
            "blocker": SYNTHETIC_TRACKER_BLOCKER, "evidence": list(evidence)}


def mot17_field() -> dict:
    """The field carried by a new E1 to E3 results document."""
    return {"association_mechanism": REAL_TRACKER_IOU, "association_status": ESTABLISHED, "association_note": MOT17_NOTE,
            "association_does_not_apply_to": MOT17_DOES_NOT_APPLY}


def synthetic_tracker_field() -> dict:
    """The field carried by a new E4 results document: present, null, and explained."""
    return {"association_mechanism": None, "association_status": UNRESOLVED, "association_blocker": SYNTHETIC_TRACKER_BLOCKER}


def analysis_field() -> dict:
    """The field carried by a new E5 to E8 results document, one entry per experiment."""
    lineage = {"association_mechanism": SYNTHETIC_GT_LINEAGE, "association_status": ESTABLISHED}
    return {"E5": synthetic_tracker_field(), "E6": lineage, "E7": lineage, "E8": lineage, "family_across_scenes": lineage}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mot17_entry(run_dir: Path, cfg: Config) -> dict:
    """Evidence read from one stored E1 to E3 run. Raises if the run does not show the mechanism."""
    run_dir = Path(run_dir)
    threshold = float(cfg.frozen["association"]["tracker_gt_assoc_iou_threshold"])
    doc = _json(run_dir / "results.json")
    if doc.get("not_p1") is not True or doc.get("cadence") != "native":
        raise ValueError(f"{run_dir} is not an E1 to E3 record")
    variants = sorted(doc["by_detection_set"])
    for variant in variants:
        settings = _json(run_dir / variant / "tracker" / "results.json")["settings"]
        if (settings["evaluation_iou_threshold"], settings["detection_variant"]) != (threshold, variant):
            raise ValueError(f"{run_dir / variant}: stored settings do not show the frozen threshold")
        if not settings["tracker"]["name"].startswith("IouTracker"):
            raise ValueError(f"{run_dir / variant}: stored settings do not show the IoU tracker")
    evidence = [CHAIN, f"stored settings of {variants}: tracker IouTracker, evaluation_iou_threshold {threshold}",
                "real MOT17 footage annotations and public detections; no injector is called on this path"]
    return established("E1, E3", REAL_TRACKER_IOU, MOT17_TRACKER, evidence,
                       ["E1 project tracking metrics", "E3 error profiles"], MOT17_DOES_NOT_APPLY, MOT17_NOTE)


def synthetic_tracker_entry(run_dir: Path) -> dict:
    """E4 stays unresolved, and the stored record must show why."""
    run_dir = Path(run_dir)
    doc = _json(run_dir / "results.json")
    spec = RunStore.open(run_dir).manifest()["spec"]
    if doc.get("scope") != "synthetic_tracker_engineering" or spec.get("scope") != doc["scope"]:
        raise ValueError(f"{run_dir} is not an E4 record")
    tracker = doc["provenance"]["tracker"]
    if not association_errors(SYNTHETIC_GT_LINEAGE, tracker, True):
        raise AssertionError("the lineage value must be a category error for a record with a tracker")
    return unresolved("E4", [
        "experiments/prediction_feasibility.py associate_grid calls gated_iou_assignment at the frozen threshold",
        f"stored tracker: {tracker}", f"stored detector: {doc['provenance']['detector']}",
        "stored ground_truth_lineage names where identities come from, not a section 3.7 mechanism"])


def ground_truth_base(run_dir: Path, scenes) -> dict:
    """Proof that a Phase 4 run used ground-truth tracks and no tracker: its scene digests and its specification."""
    spec = RunStore.open(run_dir).manifest()["spec"]
    digests = {s.name: inj.tracks_digest(s.tracks) for s in scenes}
    if digests != spec["scenes"]:
        raise ValueError(f"{run_dir}: scene digests differ from the ground-truth tracks")
    if any("track" in key or "detect" in key for key in spec):
        raise ValueError(f"{run_dir}: the specification names a tracker or detector")
    return {"scope": spec["scope"], "scenes_matching_ground_truth": sorted(digests)}


def analysis_entries(run_dir: Path, e4_dir: Path, synthetic_analysis: Path, ethhotel_analysis: Path,
                     synthetic_base: dict, ethhotel_base: dict) -> list:
    """Evidence for one stored E5 to E8 record, tied to its inputs by the hashes that record holds."""
    inputs = _json(Path(run_dir) / "results.json")["inputs"]
    pairs = (("synthetic_analysis_sha256", synthetic_analysis), ("ethhotel_analysis_sha256", ethhotel_analysis),
             ("e4_results_sha256", Path(e4_dir) / "results.json"))
    for key, path in pairs:
        if inputs[key] != _sha(path):
            raise ValueError(f"{path} is not the input recorded as {key}")
    base = f"{LINEAGE_EVIDENCE} Scene digests equal ground-truth tracks: "
    out = [unresolved("E5", ["derived from the E4 record, which has no established value",
                             f"input sha256 {inputs['e4_results_sha256']}"])]
    syn = [base + ", ".join(synthetic_base["scenes_matching_ground_truth"]),
           f"input sha256 {inputs['synthetic_analysis_sha256']}"]
    eth = [base + ", ".join(ethhotel_base["scenes_matching_ground_truth"]),
           f"input sha256 {inputs['ethhotel_analysis_sha256']}"]
    out.append(established("E6", SYNTHETIC_GT_LINEAGE, None, syn, ["dominant-mode dry-run", "across-scene family AUDC"]))
    out.append(established("E7 and E8, synthetic", SYNTHETIC_GT_LINEAGE, None, syn, ["inverse mapping", "threshold crossings"]))
    out.append(established("E7 and E8, ETH and HOTEL", SYNTHETIC_GT_LINEAGE, None, eth, ["inverse mapping", "threshold crossings"]))
    return out


def write_sidecar(run_dir: Path, entries: list) -> Path:
    """One new file beside the record. An existing sidecar is never overwritten, and the record is not touched."""
    path = Path(run_dir) / SIDECAR
    if path.exists():
        raise FileExistsError(f"{path} exists; provenance is never overwritten")
    atomic_write_text(path, json.dumps({"schema": SCHEMA, "record": "results.json",
                                        "results_sha256": _sha(Path(run_dir) / "results.json"),
                                        "spec_section": "v4.15 section 3.7", "section_11_3_schema_changed": False,
                                        "proposal_0003": "unadopted", "entries": entries}, indent=2, sort_keys=True) + "\n")
    return path


def write_all(root: Path, cfg: Config, synthetic_run: Path, ethhotel_run: Path, synthetic_analysis: Path,
              ethhotel_analysis: Path, synthetic_scenes, ethhotel_scenes) -> dict:
    """Sidecars for the five stored Phase 5 and 6 records under `root`."""
    root = Path(root)
    entries = {"e1_e3_mot17": [mot17_entry(root / "e1_e3_mot17", cfg)], "repro_e1_frcnn": [mot17_entry(root / "repro_e1_frcnn", cfg)],
               "e4_pilot": [synthetic_tracker_entry(root / "e4_pilot")], "e4_synthetic": [synthetic_tracker_entry(root / "e4_synthetic")],
               "e5_e8_analysis": analysis_entries(root / "e5_e8_analysis", root / "e4_synthetic", synthetic_analysis,
                                                  ethhotel_analysis, ground_truth_base(synthetic_run, synthetic_scenes),
                                                  ground_truth_base(ethhotel_run, ethhotel_scenes))}
    for name, items in entries.items():
        write_sidecar(root / name, items)
    return entries
