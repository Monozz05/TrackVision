"""Run store for sweeps: one atomic, self-verifying record per experiment cell.

A cell is identified by its full identity: experiment, scene and fold, predictor, condition, requested level,
seed, history length, population, and the hashes of the configuration, the code that determines results, the
input data and any checkpoint. The record's location is the SHA-256 of that identity, so changing any input
gives a different cell, and a result is reused only when every input matches.

Reading a record verifies it: the stored identity must equal the requested one and the content digest must
match, otherwise StaleRecordError is raised. A filename is never trusted on its own.

Statuses, kept apart in `summary()`:
  complete    evaluated; metrics may still be undefined, with a reason
  unrealized  the requested condition cannot exist in this scene (for example, no eligible track frames)
  failed      an exception occurred; retried on resume
  blocked     the experiment cannot run (reason recorded)

The run itself has a manifest holding its specification. Resuming with a different specification raises
IncompatibleRunError. Timestamps and environment details are provenance: they are stored but never enter an
identity or a content digest.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from trackvision import __version__, config, repro
from trackvision.config import canonical_json
from trackvision.io.atomic import atomic_write_text

SCHEMA = "trackvision-run-store/1"
COMPLETE, UNREALIZED, FAILED, BLOCKED = "complete", "unrealized", "failed", "blocked"
STATUSES = (COMPLETE, UNREALIZED, FAILED, BLOCKED)
# Sources that determine results. Analysis, figures and the CLI read records and are left out on purpose, so
# changing them does not invalidate an expensive sweep.
RESULT_SOURCES = ("predict", "inject", "metrics", "synthetic", "io")
RESULT_MODULES = ("experiments/evaluation.py", "experiments/sweep.py", "experiments/injection_feasibility.py",
                  "experiments/switch_feasibility.py", "experiments/synthetic_feasibility.py", "config.py")


class StoreError(RuntimeError):
    """Base class for run-store problems."""


class StaleRecordError(StoreError):
    """A stored record does not verify against the identity it was requested for."""


class IncompatibleRunError(StoreError):
    """An existing run directory holds a different run specification."""


def source_sha256() -> str:
    """SHA-256 over the source files that determine results."""
    root = config.REPO_ROOT / "src" / "trackvision"
    files = sorted(p for d in RESULT_SOURCES for p in (root / d).rglob("*.py")) + [root / m for m in RESULT_MODULES]
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def identity_key(identity: Mapping) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def _content_digest(identity: Mapping, status: str, result, failure) -> str:
    return identity_key({"identity": identity, "status": status, "result": result, "failure": failure})


@dataclass(frozen=True)
class CellRecord:
    identity: dict
    status: str
    result: dict | None
    failure: dict | None
    provenance: dict
    digest: str

    @property
    def key(self) -> str:
        return identity_key(self.identity)


def failure_info(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"

    # --- the run -------------------------------------------------------------------------------

    @classmethod
    def create(cls, root: Path, spec: Mapping, resume: bool = False) -> "RunStore":
        """Start a run in a new directory, or with `resume`, reopen one with an identical specification."""
        store = cls(root)
        if store.manifest_path.exists():
            if not resume:
                raise FileExistsError(f"{root} already holds a run; pass resume to continue it")
            manifest = store.manifest()
            if manifest["spec_sha256"] != identity_key(spec):
                differing = sorted(k for k in set(spec) | set(manifest["spec"])
                                   if spec.get(k) != manifest["spec"].get(k))
                raise IncompatibleRunError(f"{root} holds a different run; differing fields: {differing}")
            return store
        if resume:
            raise FileNotFoundError(f"{root} holds no run to resume")
        if store.root.exists() and any(store.root.iterdir()):
            raise FileExistsError(f"{root} is not empty and holds no run")
        atomic_write_text(store.manifest_path, json.dumps(
            {"schema": SCHEMA, "spec": spec, "spec_sha256": identity_key(spec), "version": __version__,
             "created_utc": _now(), "git": repro.git_state(config.REPO_ROOT),
             "environment": repro.environment_info()}, indent=2, sort_keys=True) + "\n")
        return store

    @classmethod
    def open(cls, root: Path) -> "RunStore":
        store = cls(root)
        if not store.manifest_path.exists():
            raise FileNotFoundError(f"{root} holds no run")
        return store

    def manifest(self) -> dict:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise StoreError(f"{self.manifest_path}: unknown schema {manifest.get('schema')!r}")
        return manifest

    # --- cells ---------------------------------------------------------------------------------

    def path(self, identity: Mapping) -> Path:
        key = identity_key(identity)
        return self.root / "cells" / key[:2] / f"{key}.json"

    def put(self, identity: Mapping, status: str, result: Mapping | None = None, failure: Mapping | None = None,
            provenance: Mapping | None = None) -> CellRecord:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        identity, result = _plain(identity), _plain(result)
        failure = _plain(failure)
        digest = _content_digest(identity, status, result, failure)
        doc = {"schema": SCHEMA, "identity": identity, "status": status, "result": result, "failure": failure,
               "digest": digest, "provenance": {"written_utc": _now(), **(provenance or {})}}
        atomic_write_text(self.path(identity), json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
        return _record(doc)

    def get(self, identity: Mapping) -> CellRecord | None:
        """The verified record for this identity, or None if there is none."""
        path = self.path(identity)
        if not path.exists():
            return None
        record = _load(path)
        if record.identity != _plain(identity):
            raise StaleRecordError(f"{path.name}: stored identity differs from the requested one")
        return record

    def records(self) -> Iterator[CellRecord]:
        """Every record, verified, in a stable order. Stray temporary files are ignored."""
        for path in sorted((self.root / "cells").glob("*/*.json")):
            record = _load(path)
            if path.stem != record.key:
                raise StaleRecordError(f"{path.name}: filename does not match the record's identity")
            yield record

    def summary(self) -> dict:
        counts = {s: 0 for s in STATUSES}
        by_experiment: dict = {}
        for record in self.records():
            counts[record.status] += 1
            row = by_experiment.setdefault(record.identity.get("experiment", "?"), {s: 0 for s in STATUSES})
            row[record.status] += 1
        return {"cells": sum(counts.values()), "by_status": counts, "by_experiment": by_experiment}


def _record(doc: Mapping) -> CellRecord:
    return CellRecord(doc["identity"], doc["status"], doc["result"], doc["failure"], doc["provenance"], doc["digest"])


def _load(path: Path) -> CellRecord:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StaleRecordError(f"{path.name}: not valid JSON ({exc})") from exc
    if doc.get("schema") != SCHEMA or doc.get("status") not in STATUSES:
        raise StaleRecordError(f"{path.name}: unknown schema or status")
    if _content_digest(doc["identity"], doc["status"], doc["result"], doc["failure"]) != doc["digest"]:
        raise StaleRecordError(f"{path.name}: content does not match its digest")
    return _record(doc)


def _plain(value):
    """JSON-round-trip, so a record reads back exactly as it was written (tuples become lists)."""
    return None if value is None else json.loads(canonical_json(value))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
