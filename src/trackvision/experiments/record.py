"""The experiment record schema (spec v4.15 section 11.3): field names, permitted values,
validation and serialization. No runner writes records yet.

Structure, read from the section 11.3 block:
  - The upper-case headings (SOURCE PROVENANCE, DATASET PROVENANCE, ...) only label groups;
    their fields are top-level keys.
  - RESULTS is a nested `results` object. It must be: `mitigation` and the three `error_*`
    fields occur both under EXPERIMENT and inside RESULTS. Its `name:` lines are nested groups;
    its last two lines are fields of `results` itself.
  - `{a | b}` lists permitted values; `detector_params {imgsz, conf, iou}` lists sub-keys.
  - `X + version` is read as the two fields `X` and `X_version`.

Every record is complete: all keys are present and a value is null where it does not apply.
This is an implementation-level interpretation of "a full experiment record" (spec 13.4); the
specification does not state that every key must be present. tests/unit/test_record.py checks
these lists against the specification text.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from trackvision.config import Config, canonical_json

TOP_LEVEL_FIELDS = (
    "experiment_id", "timestamp", "git_sha", "config_hash", "origin",
    # SOURCE PROVENANCE
    "source_type", "source_hash", "source_metadata", "ground_truth_available",
    # DATASET PROVENANCE
    "dataset", "dataset_version", "dataset_manifest_hash", "annotation_version",
    "calibration_version", "scene", "regime", "split", "fold", "seed",
    # PREPROCESSING PROVENANCE
    "preprocessing_version", "preprocessing_config_hash", "resampling_config", "window_stride",
    # PIPELINE
    "detector", "detector_version", "detector_params", "tracker", "tracker_version",
    "tracker_params", "stabilizer", "transform_config",
    # PREDICTOR
    "predictor", "predictor_version", "predictor_params", "checkpoint_hash",
    "training_data_hash", "history_len", "horizon_len",
    # EXPERIMENT
    "error_mode", "error_requested", "error_achieved", "error_achieved_normalized",
    "training_protocol", "mitigation", "anchoring_type",
    # ENVIRONMENT
    "hardware", "software_versions",
    # lifecycle
    "artifacts", "status", "supersedes_experiment_id", "superseded_by_experiment_id",
    "invalid_reason",
)

PERMITTED_VALUES = {
    "origin": ("sweep", "guided"),
    "source_type": ("dataset", "upload"),
    "ground_truth_available": (True, False),
    "stabilizer": ("static", "moving"),
    "error_mode": ("none", "A", "B1", "B2", "C", "D", "E1", "E2", "E3", "combined"),
    "training_protocol": ("clean", "augmented"),
    "mitigation": ("none", "rejection", "adaptive"),
    "anchoring_type": ("none", "within_domain", "cross_domain"),
    "status": ("valid", "invalid", "superseded"),
}

MAPPING_KEYS = {"detector_params": ("imgsz", "conf", "iou")}

RESULT_GROUPS = {
    "prediction": ("ADE", "FDE", "RMSE", "per_step_errors"),
    "severity": (
        "error_requested", "error_achieved", "error_achieved_normalized",
        "fragmentation_rate", "fragments_per_original_track",
        "switch_event_rate", "real_switch_event_rate",
        "interior_reconstructed_observation_count", "unresolved_boundary_gap_count",
        "induced_position_RMSE", "final_drift", "achieved_drift_rate",
        "e3_eligible_window_count", "travel_distance_L", "window_duration_T", "sigma_step",
    ),
    "summary": ("AUDC", "robustness_score"),
    "collision": (
        "d_conflict", "d_warn", "tau_event", "association_type",
        "N_GT", "N_pred", "TP", "FP", "FN",
        "missed_warning_rate", "false_warning_rate",
        "valid_miss_denominator_count", "valid_false_warning_denominator_count",
        "unassociable_pair_count", "unassociable_event_count", "duplicate_warning_count",
    ),
    "interaction": ("interaction_pair", "severity_grid_id", "interaction_residual",
                    "simultaneous_ci_low", "simultaneous_ci_high"),
    "association": ("spearman_rho", "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high",
                    "scene_count", "tracker_count"),
    "history": ("audc_best", "audc_tolerance_distance", "minimum_sufficient_history"),
    "inverse_requirement": ("requirement_value", "acceptable_severity_limit", "threshold_crossing_method"),
    "mitigation": ("mitigation", "recovery_ADE", "recovery_FDE", "recovery_defined_flag"),
}

RESULT_FIELDS = (
    "tracking_coverage", "prediction_coverage_pre", "prediction_coverage_post",
    "MOTA", "IDF1", "HOTA", "IDSW", "fragmentations", "FPS", "latency",
)

# Spec 11.3: when ground_truth_available is false, "no ground-truth-dependent evaluation metric
# may be populated - ADE, FDE, RMSE, coverage, MOTA, IDF1, HOTA, IDSW, fragmentation counts".
# Each listed term and the result fields it names; (group, field), group None = results itself.
# "fragmentation counts" is read as `fragmentations` (approved interpretation). "coverage" is read
# as tracking_coverage, prediction_coverage_pre and prediction_coverage_post: an interpretation
# inferred from spec 3.2, which defines both coverages against ground truth (expected GT
# observations; windows eligible only if GT holds a complete trajectory). Section 11.3 itself
# lists only the word "coverage", not these field names.
GROUND_TRUTH_DEPENDENT = {
    "ADE": (("prediction", "ADE"),),
    "FDE": (("prediction", "FDE"),),
    "RMSE": (("prediction", "RMSE"),),
    "coverage": ((None, "tracking_coverage"), (None, "prediction_coverage_pre"),
                 (None, "prediction_coverage_post")),
    "MOTA": ((None, "MOTA"),),
    "IDF1": ((None, "IDF1"),),
    "HOTA": ((None, "HOTA"),),
    "IDSW": ((None, "IDSW"),),
    "fragmentation counts": ((None, "fragmentations"),),
}

# Fields a record cannot be identified or interpreted without (an implementation choice; the
# specification lists fields but does not mark any as required).
REQUIRED = ("experiment_id", "timestamp", "config_hash", "status")


class RecordError(ValueError):
    """Raised when writing a record that does not satisfy the schema."""


def new_record(experiment_id: str, timestamp: str, config_hash: str, status: str) -> dict:
    """A complete record with every other field null."""
    record: dict[str, Any] = {name: None for name in TOP_LEVEL_FIELDS}
    record["results"] = {group: {name: None for name in fields} for group, fields in RESULT_GROUPS.items()}
    record["results"].update({name: None for name in RESULT_FIELDS})
    record.update(experiment_id=experiment_id, timestamp=timestamp, config_hash=config_hash, status=status)
    return record


def _keys(where: str, value: Any, expected, errors: list[str]) -> bool:
    if not isinstance(value, Mapping):
        errors.append(f"{where}: must be a mapping")
        return False
    missing, unknown = set(expected) - set(value), set(value) - set(expected)
    if missing:
        errors.append(f"{where}: missing keys {sorted(missing)}")
    if unknown:
        errors.append(f"{where}: unknown keys {sorted(unknown)}")
    return not missing and not unknown


def _is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def validate_record(record: Any, config: Config) -> list[str]:
    """Return the schema violations of `record` (an empty list means valid)."""
    errors: list[str] = []
    if not _keys("record", record, TOP_LEVEL_FIELDS + ("results",), errors):
        return errors

    for name in REQUIRED:
        if record[name] is None:
            errors.append(f"{name}: required")
    for name, permitted in PERMITTED_VALUES.items():
        value = record[name]
        if value is not None and not any(value is p if isinstance(p, bool) else value == p for p in permitted):
            errors.append(f"{name}: {value!r} is not one of {list(permitted)}")
    for name, keys in MAPPING_KEYS.items():
        if record[name] is not None:
            _keys(name, record[name], keys, errors)
    if record["tracker_params"] is not None:
        # Spec 11.4: canonical names; parameters a tracker lacks are recorded as null.
        _keys("tracker_params", record["tracker_params"], config.frozen["trackers"]["canonical_param_names"], errors)
    # Implementation-level choice: the specification does not define the types of these fields.
    for name in ("history_len", "horizon_len"):
        if record[name] is not None and not _is_pos_int(record[name]):
            errors.append(f"{name}: must be a positive integer")

    # Lifecycle: an approved implementation decision, consistent with spec 11.3 (the lifecycle
    # fields keep corrected or invalidated results traceable) but not explicitly stated there.
    if record["status"] == "invalid" and not record["invalid_reason"]:
        errors.append("invalid_reason: required when status is 'invalid'")
    if record["status"] == "superseded" and not record["superseded_by_experiment_id"]:
        errors.append("superseded_by_experiment_id: required when status is 'superseded'")

    results = record["results"]
    if not _keys("results", results, tuple(RESULT_GROUPS) + RESULT_FIELDS, errors):
        return errors
    groups_ok = all(_keys(f"results.{g}", results[g], fields, errors) for g, fields in RESULT_GROUPS.items())
    if not groups_ok:
        return errors

    steps = results["prediction"]["per_step_errors"]
    if steps is not None:
        if not isinstance(steps, (list, tuple)) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in steps):
            errors.append("results.prediction.per_step_errors: must be a list of finite numbers")
        elif record["horizon_len"] is None or len(steps) != record["horizon_len"]:
            errors.append(f"results.prediction.per_step_errors: length {len(steps)} must equal "
                          f"horizon_len ({record['horizon_len']})")

    if record["ground_truth_available"] is False:
        for term, fields in GROUND_TRUTH_DEPENDENT.items():
            for group, name in fields:
                value = results[name] if group is None else results[group][name]
                if value is not None:
                    where = f"results.{name}" if group is None else f"results.{group}.{name}"
                    errors.append(f"{where}: must be null when ground_truth_available is false "
                                  f"(ground-truth-dependent: {term})")
    return errors


def write_record(path: Path, record: dict, config: Config) -> None:
    """Validate and write the record as canonical JSON; raise RecordError if it is invalid."""
    errors = validate_record(record, config)
    if errors:
        raise RecordError("; ".join(errors))
    try:
        text = canonical_json(record)
    except ValueError as exc:                     # NaN or infinity anywhere in the record
        raise RecordError(f"record is not serialisable as canonical JSON: {exc}") from exc
    Path(path).write_text(text + "\n", encoding="utf-8")
