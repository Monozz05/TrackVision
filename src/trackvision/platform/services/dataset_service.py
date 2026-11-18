"""Describe the datasets the project actually holds, from manifests and stored run records.

Only metadata is returned: scene names, counts and hashes of manifests. Dataset files are never opened. A dataset
appears only if its manifest exists or a stored run used it, and a status with no evidence behind it stays `unknown`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from trackvision.platform.config import Settings
from trackvision.platform.services.result_reader import ArtifactError
from trackvision.platform.services.result_views import frozen_protocol
from trackvision.platform.services.run_registry import Run, RunRegistry

NO_BOXES = "No ground-truth boxes exist for this scene, so tracking metrics cannot be computed (decision 0006)."
_DATASETS = (("EWAP", "ewap.sha256", "ETH Walking Pedestrians (ETH, HOTEL)"),
             ("UCY", "ucy.sha256", "UCY crowds (UNIV, ZARA1, ZARA2)"),
             ("MOT17", "mot17.sha256", "MOT17 pedestrian tracking benchmark"))


def _manifest_info(path: Path) -> dict | None:
    """How many entries a dataset manifest lists, and its hash. The listed files are not touched."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    entries = [line for line in data.decode("utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return {"listed_entries": len(entries), "manifest_sha256": hashlib.sha256(data).hexdigest()}


def _experiments(registry: RunRegistry, dataset: str) -> list[str]:
    return [run.id for run in registry.runs() if dataset in run.summary["datasets"]]


def _read(run: Run | None, registry: RunRegistry, name: str) -> dict:
    if run is None:
        return {}
    try:
        return registry.reader.read_json(run.path / name)
    except ArtifactError:
        return {}


def _static_regime(frozen: dict) -> dict:
    return {scene: "static" for scene in ((frozen.get("protocol") or {}).get("static") or {}).get("scenes", [])}


def _ewap(registry: RunRegistry, entry: dict, regime: dict) -> dict:
    run = registry.newest(("geometry_validation",))
    report = _read(run, registry, "report.json")
    scenes = []
    for name, row in (report.get("scenes") or {}).items():
        diagnostics, roundtrip = row.get("diagnostics") or {}, row.get("roundtrip") or {}
        scenes.append({"name": name, "regime": regime.get(name), "status": "usable", "reason": None, "metadata": {
            "annotations": diagnostics.get("annotation_count"), "tracks": diagnostics.get("track_count"),
            "sampling_interval_s": diagnostics.get("documented_timestep_s"),
            "implied_video_fps": diagnostics.get("implied_video_fps"), "windows": (row.get("windows") or {}).get("count"),
            "median_speed_mps": (row.get("speed_diagnostic") or {}).get("median_mps"),
            "homography_roundtrip_passed": roundtrip.get("passed"),
            "homography_roundtrip_max_px": roundtrip.get("max_px")}})
    return {**entry, "scenes": scenes, "regime": "static", "notice": report.get("notice"),
            "ground_truth": {"trajectories": "available" if scenes else "unknown", "boxes": "unavailable",
                             "calibration": "available" if scenes else "unknown"},
            "ground_truth_note": NO_BOXES, "source_run": run.id if run else None}


def _ucy(registry: RunRegistry, entry: dict, regime: dict) -> dict:
    run = registry.newest(("geometry_validation",))
    report = _read(run, registry, "report.json")
    scenes = [{"name": name, "regime": regime.get(name), "status": "blocked", "reason": reason, "metadata": {}}
              for name, reason in (report.get("blocked_scenes") or {}).items()]
    return {**entry, "scenes": scenes, "regime": "static", "notice": report.get("notice"),
            "ground_truth": {"trajectories": "blocked" if scenes else "unknown", "boxes": "unavailable",
                             "calibration": "blocked" if scenes else "unknown"},
            "ground_truth_note": NO_BOXES, "source_run": run.id if run else None}


def _mot17(registry: RunRegistry, entry: dict, frozen: dict) -> dict:
    run = registry.newest(("mot17_tracking",))
    results = _read(run, registry, "results.json")
    manifest = _read(run, registry, "manifest.json")
    partition = (frozen.get("datasets") or {}).get("mot17") or {}
    regimes = {**{s: "static" for s in partition.get("static_sequences", [])},
               **{s: "moving" for s in partition.get("moving_sequences", [])}}
    primary = (results.get("by_detection_set") or {}).get(results.get("primary_detection_set")) or {}
    scenes = [{"name": name, "regime": regimes.get(name), "status": "usable", "reason": None, "metadata": {
        "frame_rate": row.get("frame_rate"), "frames_evaluated": row.get("frames_evaluated"),
        "detection_sets": [results.get("primary_detection_set"), *(results.get("optional_detection_sets") or [])]}}
        for name, row in primary.items()]
    return {**entry, "scenes": scenes, "regime": "static and moving", "notice": None, "licence": manifest.get("licence"),
            "ground_truth": {"trajectories": "not_used", "boxes": "available" if scenes else "unknown",
                             "calibration": "unavailable"},
            "ground_truth_note": results.get("prediction"), "source_run": run.id if run else None}


def _synthetic(registry: RunRegistry) -> dict | None:
    run = registry.newest(("sweep",), scope="synthetic_engineering")
    if run is None:
        return None
    return {"id": "synthetic", "name": "Synthetic scenes", "description": "Generated scenes with known ground truth",
            "regime": "static", "manifest": None, "notice": None, "source_run": run.id,
            "scenes": [{"name": s, "regime": "static", "status": "usable", "reason": None, "metadata": {}}
                       for s in run.summary["scenes"]],
            "ground_truth": {"trajectories": "available", "boxes": "available", "calibration": "available"},
            "ground_truth_note": "Ground truth comes from the generator, so these scenes are engineering evidence only.",
            "experiments": _experiments(registry, "Synthetic")}


def list_datasets(settings: Settings, registry: RunRegistry) -> list[dict]:
    frozen = frozen_protocol(settings.base_config)
    regime = _static_regime(frozen)
    out = []
    for name, manifest_file, description in _DATASETS:
        manifest = _manifest_info(settings.manifests_dir / manifest_file)
        used_by = _experiments(registry, name)
        if manifest is None and not used_by:
            continue
        entry = {"id": name.lower(), "name": name, "description": description, "manifest": manifest,
                 "experiments": used_by}
        out.append(_ewap(registry, entry, regime) if name == "EWAP" else _ucy(registry, entry, regime)
                   if name == "UCY" else _mot17(registry, entry, frozen))
    synthetic = _synthetic(registry)
    return out + ([synthetic] if synthetic else [])
