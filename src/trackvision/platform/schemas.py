"""Response contracts for the API. Metric payloads stay as stored, so only the envelopes are modelled here."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    version: str
    read_only: bool
    results_directory_available: bool
    experiment_records: int
    invalid_records: int
    skipped_folders: int
    timestamp: str


class ExperimentSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    record_id: str | None
    kind: str
    kind_label: str
    scope: str | None
    timestamp: str | None
    status: str
    status_reason: str | None
    evidence_level: str | None
    research_results: bool | None
    label: str | None
    notice: str | None
    latest_in_series: bool
    experiments: list[str | None]
    datasets: list[str | None]
    scenes: list[str | None]
    predictors: list[str]
    conditions: list[str]
    histories: list[int]
    seeds: list[int]
    horizon_len: int | None
    detector: str | None
    tracker: str | None
    counts: dict | None
    integrity: dict
    provenance: dict


class ExperimentPage(BaseModel):
    items: list[ExperimentSummary]
    total: int
    limit: int
    offset: int
    counts: dict


class ViewIndex(BaseModel):
    items: list[ExperimentSummary]
    default: str | None
