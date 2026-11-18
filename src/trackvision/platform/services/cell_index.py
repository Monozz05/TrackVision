"""Seed-level records of a sweep, read through the run store so that every digest is verified.

A sweep holds up to tens of thousands of cell files, so they are read only when someone asks, and only compact rows
are kept. One failed digest fails the whole read: serving the rest of a run that no longer verifies would hide it.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from trackvision.experiments.store import RunStore, StoreError
from trackvision.platform.services.result_reader import ArtifactMalformed
from trackvision.platform.services.result_views import ViewUnavailable, _require_usable
from trackvision.platform.services.run_registry import Run

_METRICS = ("ade_m", "fde_m", "rmse_m", "prediction_coverage", "eligible_windows", "predicted_windows",
            "undefined_reason")
_IDENTITY = ("experiment", "scene", "fold", "predictor", "condition", "requested_level", "seed", "history_len",
             "horizon_len", "population", "validation_scene", "checkpoint_sha256")
FILTERS = ("experiment", "scene", "predictor", "condition", "seed", "history_len", "population", "status")


def _row(record) -> dict:
    result = record.result or {}
    metrics = result.get("metrics") or {}
    return {**{key: record.identity.get(key) for key in _IDENTITY}, "status": record.status,
            "achieved": result.get("achieved"), **{key: metrics.get(key) for key in _METRICS},
            "reason": result.get("reason"), "written_utc": record.provenance.get("written_utc"),
            "digest": record.digest}


class CellIndex:
    def __init__(self, max_runs: int = 2):
        self._rows: OrderedDict[str, list[dict]] = OrderedDict()
        self._max_runs = max_runs
        self._lock = threading.Lock()

    def _load(self, run: Run) -> list[dict]:
        with self._lock:
            if run.id in self._rows:
                self._rows.move_to_end(run.id)
                return self._rows[run.id]
            try:
                rows = [_row(record) for record in RunStore.open(run.path).records()]
            except (StoreError, ValueError, KeyError) as exc:
                raise ArtifactMalformed("a cell record failed verification, so this run's cells are not served") from exc
            self._rows[run.id] = rows
            while len(self._rows) > self._max_runs:
                self._rows.popitem(last=False)
            return rows

    def page(self, run: Run, filters: dict, limit: int, offset: int) -> dict:
        if run.kind != "sweep":
            raise ViewUnavailable(f"{run.id} holds no cell records")
        _require_usable(run)
        rows = self._load(run)
        wanted = {key: value for key, value in filters.items() if value is not None}
        matching = [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]
        return {"experiment_id": run.id, "total": len(matching), "limit": limit, "offset": offset,
                "digests_verified": True, "items": matching[offset:offset + limit]}
