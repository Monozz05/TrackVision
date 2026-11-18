"""Discover the runs stored under `runs/` and describe each one from its own records.

A run is a folder holding at least one known record file. Discovery stops at a run, so a sweep's cells or a MOT17
run's per-detector sub-runs are never listed separately. `analysis.json` can be tens of megabytes and is not read
here; only views that need it open it.

An experiment ID is the folder's path below `runs/`, with `~` between its parts. It is only ever looked up in the
discovered set and is never turned into a path, so a request cannot name a folder that discovery did not find.

Status comes from what is recorded: `complete`, `incomplete` (a required file is absent or the run did not finish)
or `invalid` (a record is malformed, unrecognised, or differs from its recorded hash). No record marks a run as
superseded, so the API never reports that; `latest_in_series` only says which run is the newest of its series.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from trackvision.platform.services.result_reader import ArtifactError, ArtifactMissing, ResultReader

MARKERS = ("manifest.json", "run_manifest.json", "results.json", "report.json", "run_status.json")
RUN_STORE_SCHEMA = "trackvision-run-store/1"
E4_SCOPE = "synthetic_tracker_engineering"
ETHHOTEL_SCOPE = "ethhotel_descriptive"
MAX_DEPTH = 4
_NAME = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}(~[A-Za-z0-9._-]{1,100}){0,5}$")

KIND_LABELS = {
    "sweep": "Phase 4 injection sweep",
    "mot17_tracking": "MOT17 tracking (E1 to E3)",
    "synthetic_tracker": "Synthetic tracker pipeline (E4)",
    "engineering_analysis": "Engineering analysis (E5 to E8)",
    "ewap_baseline": "EWAP CV baseline (R1)",
    "ewap_injection": "EWAP injection (R2)",
    "synthetic_feasibility": "Synthetic feasibility (U1 to U8)",
    "geometry_validation": "Geometry validation",
    "smoke": "Scaffold smoke run",
    "unrecognised": "Unrecognised record",
}
# What a run can support, by the protocol that produced it. No runner writes research results yet, so no kind maps
# to a research level and the frontend has nothing to show under research measurements.
EVIDENCE = {
    "mot17_tracking": "engineering_only",
    "synthetic_tracker": "engineering_only",
    "engineering_analysis": "engineering_and_descriptive",
    "ewap_baseline": "descriptive_local_only",
    "ewap_injection": "descriptive_local_only",
    "synthetic_feasibility": "feasibility_only",
    "geometry_validation": "diagnostic_only",
    "smoke": "scaffold_check",
    "unrecognised": None,
}
_NAMED_KINDS = {"ewap-cv-baseline": "ewap_baseline", "ewap-injection": "ewap_injection",
                "mot17-engineering": "mot17_tracking", "synthetic-cv-feasibility": "synthetic_feasibility"}
_REQUIRED = {"sweep": ("run_status.json", "analysis.json"), "mot17_tracking": ("results.json",),
             "synthetic_tracker": ("results.json",), "engineering_analysis": ("results.json",),
             "ewap_baseline": ("results.json",), "ewap_injection": ("results.json",),
             "synthetic_feasibility": ("results.json",), "geometry_validation": ("report.json",)}
_PROVENANCE_KEYS = ("config_hash", "settings_hash", "source_sha256", "spec_sha256", "dataset_manifest_hash",
                    "trackeval_commit", "licence", "results_sha256", "version", "trackvision_version",
                    "clean_tracks_sha256", "inputs")
_ASSOCIATION_KEYS = ("association_mechanism", "association_status", "association_note", "association_blocker",
                     "association_does_not_apply_to", "association_by_experiment")


class InvalidExperimentId(ValueError):
    """The ID is not in the format the API issues."""


class UnknownExperiment(LookupError):
    """No discovered run has this ID."""


@dataclass(frozen=True)
class Run:
    id: str
    path: Path
    kind: str
    summary: dict

    @property
    def usable(self) -> bool:
        return self.summary["status"] == "complete"


def validate_id(run_id: str) -> None:
    if not _ID.match(run_id) or any(set(part) == {"."} for part in run_id.split("~")):
        raise InvalidExperimentId("experiment IDs hold letters, digits, dots, dashes, underscores and tildes only")


def inside(directory: Path, relative: str) -> Path | None:
    """`relative` under `directory`, or None if a recorded name would lead outside it."""
    target = (directory / relative).resolve()
    return target if target.is_relative_to(directory.resolve()) else None


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _names(value) -> list:
    return list(value) if isinstance(value, (dict, list, tuple)) else []


class RunRegistry:
    def __init__(self, runs_dir: Path, reader: ResultReader, rescan_seconds: float = 5.0):
        self.runs_dir = Path(runs_dir)
        self.reader = reader
        self.rescan_seconds = rescan_seconds
        self.skipped_folders = 0
        self._runs: dict[str, Run] = {}
        self._scanned_at: float | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.runs_dir.is_dir()

    def runs(self) -> list[Run]:
        """Every run, newest first. Rescans at most every `rescan_seconds`, so a new run appears without a restart."""
        with self._lock:
            now = time.monotonic()
            if self._scanned_at is None or now - self._scanned_at >= self.rescan_seconds:
                self._runs = self._scan()
                self._scanned_at = now
            return list(self._runs.values())

    def get(self, run_id: str) -> Run:
        validate_id(run_id)
        for run in self.runs():
            if run.id == run_id:
                return run
        raise UnknownExperiment(f"no experiment has the ID {run_id}")

    def newest(self, kinds: tuple[str, ...], **matching) -> Run | None:
        """The newest complete run of these kinds whose summary matches `matching`, or None."""
        for run in self.runs():
            if run.kind in kinds and run.usable and all(run.summary.get(k) == v for k, v in matching.items()):
                return run
        return None

    def _scan(self) -> dict[str, Run]:
        self.skipped_folders = 0
        found: list[Run] = []
        if self.available:
            self._walk(self.runs_dir, (), found)
        found.sort(key=lambda r: (r.summary["timestamp"] or "", r.id), reverse=True)
        seen: set = set()
        for run in found:
            series = run.summary.pop("_series")
            run.summary["latest_in_series"] = run.usable and series not in seen
            if run.usable:
                seen.add(series)
        return {run.id: run for run in found}

    def _walk(self, directory: Path, parts: tuple[str, ...], found: list[Run]) -> None:
        for child in sorted(directory.iterdir()):
            if child.is_symlink() or not child.is_dir():
                continue
            if not _NAME.match(child.name):
                self.skipped_folders += 1
                continue
            child_parts = (*parts, child.name)
            if any((child / marker).is_file() for marker in MARKERS):
                found.append(self._describe(child, "~".join(child_parts)))
            elif len(child_parts) < MAX_DEPTH:
                self._walk(child, child_parts, found)

    def _describe(self, directory: Path, run_id: str) -> Run:
        docs, problems = {}, {}
        for name in MARKERS:
            try:
                docs[name] = self.reader.read_json(directory / name, cache=False)
            except ArtifactMissing:
                docs[name] = {}
            except ArtifactError as exc:
                docs[name], problems[name] = {}, str(exc)
        manifest, results = docs["manifest.json"], docs["results.json"]
        run_manifest, report, run_status = docs["run_manifest.json"], docs["report.json"], docs["run_status.json"]
        kind = _kind(manifest, results, report)
        integrity = self._integrity(directory, manifest, run_manifest)
        status, reason = _status(kind, directory, problems, integrity, run_status)
        spec = _dict(manifest.get("spec"))
        summary = {
            "experiment_id": run_id,
            "record_id": manifest.get("experiment_id") or results.get("experiment_id"),
            "kind": kind,
            "kind_label": KIND_LABELS[kind],
            "scope": spec.get("scope") or results.get("scope") or run_status.get("scope"),
            "timestamp": (run_manifest.get("timestamp_utc") or manifest.get("timestamp_utc")
                          or manifest.get("created_utc") or report.get("timestamp_utc")),
            "status": status,
            "status_reason": reason,
            "evidence_level": _evidence(kind, spec),
            "research_results": results.get("research_results", run_status.get("research_complete")),
            "label": spec.get("label") or results.get("label") or manifest.get("label") or run_manifest.get("label"),
            "notice": results.get("notice") or manifest.get("notice") or report.get("notice"),
            "counts": _counts(run_status),
            "integrity": integrity,
            "provenance": self._provenance(directory, manifest, run_manifest, results, report),
            **_facts(kind, manifest, results, report, run_status),
        }
        summary["_series"] = (kind, summary["scope"], spec.get("common_training"), tuple(summary["scenes"])
                              if kind in ("ewap_baseline", "ewap_injection") else None)
        return Run(run_id, directory, kind, summary)

    def _integrity(self, directory: Path, manifest: dict, run_manifest: dict) -> dict:
        """Compare stored files with the hashes the run recorded."""
        recorded = {k: v for k, v in _dict(manifest.get("files")).items() if isinstance(v, str)}
        for doc in (manifest, run_manifest):
            if isinstance(doc.get("results_sha256"), str):
                recorded["results.json"] = doc["results_sha256"]
        if not recorded:
            return {"status": "not_recorded", "verified": 0, "failed": []}
        failed = []
        for name, expected in recorded.items():
            target = inside(directory, name)
            try:
                if target is None or self.reader.sha256(target) != expected:
                    failed.append(name)
            except ArtifactError:
                failed.append(name)
        return {"status": "mismatch" if failed else "verified", "verified": len(recorded) - len(failed),
                "failed": failed}

    def _provenance(self, directory: Path, manifest: dict, run_manifest: dict, results: dict, report: dict) -> dict:
        spec = _dict(manifest.get("spec"))
        out: dict = {"git": manifest.get("git") or run_manifest.get("git") or report.get("git"),
                     "environment": manifest.get("environment") or run_manifest.get("environment")
                     or report.get("environment"),
                     "specification": spec if "version" in spec else report.get("spec")}
        for key in _PROVENANCE_KEYS:
            for doc in (manifest, run_manifest, spec, results):
                if key in doc:
                    out[key] = doc[key]
                    break
            else:
                out[key] = None
        for key in ("torch", "torch_num_threads", "adopts_models_from", "preflight_sha256"):
            if key in spec:
                out[key] = spec[key]
        for key in _ASSOCIATION_KEYS:
            out[key] = results.get(key)
        out["association_sidecar"] = self._sidecar(directory)
        return out

    def _sidecar(self, directory: Path) -> dict | None:
        """The association provenance sidecar, and whether it still binds to `results.json`."""
        path = directory / "association_provenance.json"
        try:
            doc = self.reader.read_json(path, cache=False)
        except ArtifactError:
            return None
        target = inside(directory, str(doc.get("record", "results.json")))
        try:
            bound = target is not None and self.reader.sha256(target) == doc.get("results_sha256")
        except ArtifactError:
            bound = False
        return {"bound_to_results": bound, "entries": doc.get("entries"), "proposal_0003": doc.get("proposal_0003"),
                "spec_section": doc.get("spec_section"), "schema": doc.get("schema")}


def _kind(manifest: dict, results: dict, report: dict) -> str:
    if manifest.get("schema") == RUN_STORE_SCHEMA:
        return "synthetic_tracker" if _dict(manifest.get("spec")).get("scope") == E4_SCOPE else "sweep"
    if manifest.get("kind") == "smoke":
        return "smoke"
    name = manifest.get("experiment_name") or manifest.get("name")
    if name in _NAMED_KINDS:
        return _NAMED_KINDS[name]
    if "e6_dominant_dry_run" in results or "e4_results_sha256" in _dict(manifest.get("inputs")):
        return "engineering_analysis"
    if "blocked_scenes" in report:
        return "geometry_validation"
    return "unrecognised"


def _evidence(kind: str, spec: dict) -> str | None:
    if kind == "sweep":
        return "descriptive_local_only" if spec.get("scope") == ETHHOTEL_SCOPE else "engineering_only"
    return EVIDENCE[kind]


def _status(kind: str, directory: Path, problems: dict, integrity: dict, run_status: dict) -> tuple[str, str | None]:
    if problems:
        return "invalid", "; ".join(problems.values())
    if kind == "unrecognised":
        return "invalid", "the folder holds no record structure the platform recognises"
    if integrity["status"] == "mismatch":
        return "invalid", f"stored file differs from its recorded SHA-256: {', '.join(integrity['failed'])}"
    missing = [name for name in _REQUIRED.get(kind, ()) if not (directory / name).is_file()]
    if missing:
        return "incomplete", f"missing {', '.join(missing)}"
    if kind == "sweep":
        by_status = _dict(run_status.get("by_status"))
        if by_status.get("failed"):
            return "incomplete", f"{by_status['failed']} cells failed"
        if run_status.get("scope_complete") is not True:
            return "incomplete", (f"{run_status.get('finished_cells')} of {run_status.get('expected_cells')} cells "
                                  "finished")
    return "complete", None


def _counts(run_status: dict) -> dict | None:
    if not run_status:
        return None
    return {"expected_cells": run_status.get("expected_cells"), "finished_cells": run_status.get("finished_cells"),
            "by_status": run_status.get("by_status"), "by_experiment": run_status.get("by_experiment"),
            "scope_complete": run_status.get("scope_complete"), "research_complete": run_status.get("research_complete")}


def _facts(kind: str, manifest: dict, results: dict, report: dict, run_status: dict) -> dict:
    """What the run covers, read from its own records. Anything not recorded stays empty or None."""
    spec = _dict(manifest.get("spec"))
    facts = {"experiments": [], "datasets": [], "scenes": [], "predictors": [], "conditions": [], "histories": [],
             "seeds": [], "horizon_len": None, "detector": None, "tracker": None}
    if kind == "sweep":
        by_experiment = _dict(run_status.get("by_experiment"))
        synthetic = spec.get("scope") != ETHHOTEL_SCOPE
        facts.update(experiments=[k for k, v in by_experiment.items() if _dict(v).get("complete")],
                     datasets=["Synthetic" if synthetic else "EWAP"], scenes=_names(spec.get("scenes")),
                     predictors=_names(spec.get("predictors")), conditions=_names(spec.get("conditions")),
                     histories=_names(spec.get("histories")), seeds=_names(spec.get("seeds")),
                     horizon_len=spec.get("horizon_len"))
    elif kind == "mot17_tracking":
        sets = [results.get("primary_detection_set"), *_names(results.get("optional_detection_sets"))]
        facts.update(experiments=[e.strip() for e in str(results.get("experiment", "")).split(",") if e.strip()],
                     datasets=["MOT17"], scenes=_names(results.get("sequences")),
                     detector="MOT17 public detections: " + ", ".join(s for s in sets if s),
                     tracker="Existing IoU feasibility tracker")
    elif kind == "synthetic_tracker":
        scenes = _dict(results.get("by_scene"))
        first = _dict(next(iter(scenes.values()), None))
        provenance = _dict(results.get("provenance"))
        facts.update(experiments=[results.get("experiment", "E4")], datasets=["Synthetic"],
                     scenes=_names(results.get("scenes")),
                     predictors=_names(_dict(first.get("prediction")).get("predictors")),
                     detector=provenance.get("detector"), tracker=provenance.get("tracker"))
    elif kind == "engineering_analysis":
        facts.update(experiments=["E5", "E6", "E7", "E8"], datasets=["Synthetic", "EWAP"])
    elif kind == "ewap_baseline":
        facts.update(experiments=["R1"], datasets=[manifest.get("dataset")], scenes=[manifest.get("scene")],
                     predictors=["CV"])
    elif kind == "ewap_injection":
        facts.update(experiments=["R2"], datasets=[manifest.get("dataset")], scenes=[manifest.get("scene")],
                     predictors=["CV"], conditions=_names(manifest.get("conditions")),
                     seeds=_names(manifest.get("injection_seeds")))
    elif kind == "synthetic_feasibility":
        settings = _dict(results.get("settings"))
        facts.update(experiments=["U1 to U8"], datasets=["Synthetic"], predictors=["CV"],
                     seeds=[settings["seed"]] if "seed" in settings else [],
                     detector=_dict(settings.get("detector")).get("name"),
                     tracker=_dict(settings.get("tracker")).get("name"))
    elif kind == "geometry_validation":
        facts.update(datasets=["EWAP", "UCY"], scenes=[*_names(report.get("scenes")),
                                                       *_names(report.get("blocked_scenes"))])
    return facts
