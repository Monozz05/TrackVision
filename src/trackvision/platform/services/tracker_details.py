"""What produced a scene's tracks: its detector, its tracker, their settings and their stored results (decision 0013).

Values come from stored runs and are passed on as stored, and one a run did not record stays None. Three rows are not
recorded fields but facts about this project: it trained no detector or tracker, and `tracking/tracker.py` has no camera
motion compensation and no appearance model.

ETH, HOTEL and UCY have no tracker at all, because their IDs are human annotations. E1 to E3 describe MOT17 and E4
describes the synthetic scenes. None of this is P-1 or P-9, and the IoU tracker is not ByteTrack or BoT-SORT.
"""

from __future__ import annotations

from pathlib import Path

from trackvision.platform.services import result_views
from trackvision.platform.services.result_reader import ArtifactError
from trackvision.platform.services.run_registry import Run, RunRegistry

ANNOTATED_NOTE = ("IDs on this scene are human annotations, so no detector or tracker runs here. "
                  "A real tracker on this video is P-9, which stays blocked.")
GENERATED_NOTE = "IDs on this scene come from the scene generator, and no stored tracker run covers it."
NO_RUN_NOTE = "No stored tracker run covers this sequence."
PROFILE_NOTE = ("Project diagnostics, not official metrics. They are not comparable across frame rates and are never "
                "projected onto a degradation curve.")
NOT_RUN = "None"


def _row(label: str, value) -> dict:
    return {"label": label, "value": value}


def _sentence(text: str | None) -> str | None:
    return text[:1].upper() + text[1:] if text else text


def _settings_rows(tracker: dict) -> list:
    max_age = tracker.get("max_age")
    return [_row("Match IoU", tracker.get("match_iou")), _row("Min hits", tracker.get("min_hits")),
            _row("Max age", None if max_age is None else f"{max_age} frames"),
            _row("Camera motion compensation", NOT_RUN), _row("ReID", NOT_RUN)]


def _scores(name: str, primary: bool, official: dict | None) -> dict:
    official = official or {}
    return {"name": name, "primary": primary, "mota": official.get("MOTA"), "idf1": official.get("IDF1"),
            "hota": official.get("HOTA"), "id_switches": official.get("IDSW"), "fragmentations": official.get("Frag"),
            "false_positives": official.get("CLR_FP"), "false_negatives": official.get("CLR_FN")}


def _profile(name: str, primary: bool, profile: dict | None, localization, unit: str) -> dict:
    profile = profile or {}
    return {"name": name, "primary": primary, "missed": ((profile.get("dropout") or {}).get("rate") or {}).get("value"),
            "localization": localization, "unit": unit, "fragmentations": (profile.get("fragmentation") or {}).get("events"),
            "identity_exchanges": (profile.get("identity_exchange") or {}).get("events")}


def _tracker_run(dataset: str, recording: str, registry: RunRegistry) -> Run | None:
    kinds = {"MOT17": "mot17_tracking", "Synthetic": "synthetic_tracker"}
    for run in registry.runs():
        if run.kind == kinds.get(dataset) and run.usable and recording in run.summary["scenes"]:
            return run
    return None


def _mot17(entry: dict, recording: str, run: Run, view: dict, overlay: str | None) -> dict:
    groups = view["groups"]
    primary = next((group for group in groups if group["primary"]), groups[0])
    settings = primary["tracker_settings"] or {}
    tracker, detections = settings.get("tracker") or {}, settings.get("detections")
    sequences = {group["name"]: next((s for s in group["sequences"] if s["sequence"] == recording), {}) for group in groups}
    row = sequences[primary["name"]]
    frames = row.get("frames_evaluated")
    frames_text = f"{frames[0]} to {frames[1]}" if frames else None
    results, detection, profiles = [], [], []
    for group in groups:
        sequence, is_primary = sequences[group["name"]], group["primary"]
        diagnostics = sequence.get("project_diagnostics") or {}
        found = diagnostics.get("detection") or {}
        profile = diagnostics.get("error_profile") or {}
        results.append(_scores(group["name"], is_primary, sequence.get("official_trackeval")))
        detection.append({"name": group["name"], "primary": is_primary, "precision": found.get("precision"),
                          "recall": found.get("recall"), "f1": found.get("f1")})
        profiles.append(_profile(group["name"], is_primary, profile, (profile.get("localization") or {}).get("rmse"), "px"))
    threshold = "None in stored results" if detections and "no score threshold" in detections else None
    rate = row.get("frame_rate")
    config = [_row("Detector", _sentence(detections)),
              _row("Detection sets", ", ".join(f"{g['name']} (primary)" if g["primary"] else g["name"] for g in groups)),
              _row("Training performed", "No"),
              _row("Score threshold", threshold),
              _row("Tracker", f"{tracker.get('name')}, not ByteTrack or BoT-SORT" if tracker.get("name") else None),
              *_settings_rows(tracker),
              _row("Frames evaluated", frames_text),
              _row("Frame cadence", _sentence(f"{view.get('cadence')}, {rate} fps" if rate else view.get("cadence"))),
              _row("Evaluation", f"TrackEval, IoU {settings.get('evaluation_iou_threshold')}"),
              _row("TrackEval commit", run.summary["provenance"].get("trackeval_commit")),
              *([_row("Workspace overlay", overlay)] if overlay else [])]
    return {"title": "MOT17 · IoU tracker", "config": config, "results": results, "detection": detection,
            "error_profile": profiles, "profile_note": PROFILE_NOTE,
            "results_note": f"Stored results are for the full sequence {recording}, frames {frames_text}." if entry["parent"] else None}


def _feeds_e5(run: Run, registry: RunRegistry) -> bool:
    """Whether the newest E5 to E8 run was built from exactly this run's results."""
    analysis = registry.newest(("engineering_analysis",))
    if analysis is None:
        return False
    inputs = registry.reader.read_json(analysis.path / "results.json").get("inputs") or {}
    return inputs.get("e4_results_sha256") == registry.reader.sha256(run.path / "results.json")


def _synthetic(recording: str, run: Run, view: dict, registry: RunRegistry) -> dict:
    group = view["groups"][0]
    stored = group["tracker_settings"] or {}
    try:
        tracker = ((registry.reader.read_json(run.path / "manifest.json").get("spec") or {}).get("tracker")) or {}
    except ArtifactError:
        tracker = {}
    row = next((s for s in group["sequences"] if s["sequence"] == recording), {})
    diagnostics = row.get("project_diagnostics") or {}
    config = [_row("Detector", _sentence(stored.get("detector"))), _row("Training performed", "No"),
              _row("Tracker", _sentence(stored.get("tracker"))),
              *_settings_rows(tracker), _row("Evaluation", "TrackEval"),
              _row("TrackEval commit", run.summary["provenance"].get("trackeval_commit")),
              _row("Used by", "E5" if _feeds_e5(run, registry) else None)]
    return {"title": "Synthetic · E4", "config": config, "results": [_scores("E4", True, row.get("official_trackeval"))],
            "detection": [], "profile_note": PROFILE_NOTE,
            "error_profile": [_profile("E4", True, diagnostics.get("error_profile"), diagnostics.get("position_rmse_m"), "m")],
            "results_note": view.get("official_note")}


def details(entry: dict, registry: RunRegistry, base_config: Path, overlay: str | None = None) -> dict:
    recording = entry["parent"] or entry["id"]
    out = {"scene": entry["id"], "recording": recording, "run": None, "evidence_level": None, "label": None, "title": None,
           "note": None, "results_note": None, "profile_note": None, "config": [], "results": [], "detection": [], "error_profile": []}
    run = _tracker_run(entry["dataset"], recording, registry)
    if run is None:
        return {**out, "note": {"MOT17": NO_RUN_NOTE, "Synthetic": GENERATED_NOTE}.get(entry["dataset"], ANNOTATED_NOTE)}
    view = result_views.tracking(run, registry.reader, base_config)
    body = (_mot17(entry, recording, run, view, overlay) if run.kind == "mot17_tracking"
            else _synthetic(recording, run, view, registry))
    return {**out, "run": run.id, "evidence_level": run.summary["evidence_level"], "label": run.summary["label"], **body}
