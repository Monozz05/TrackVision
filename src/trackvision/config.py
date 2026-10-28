"""Loading, validation and hashing of the TrackVision configuration.

`configs/base.yaml` is the single source of frozen protocol constants. An experiment file may
contain only an `experiment` block that *selects* from what base.yaml permits; it can never
set or override anything in `meta`, `frozen`, `domains` or `research_choices`.

Validation has three outcomes for an experiment:

    INVALID     structurally wrong (unknown key, wrong type, override attempt, bad selection,
                spec hash mismatch, ...). Nothing may run.
    UNRESOLVED  structurally valid, but a research choice the experiment declares it uses is
                still null in base.yaml. Nothing may run until that choice is fixed.
    RUNNABLE    valid, and every research choice the experiment uses is resolved.

Validating base.yaml on its own gives VALID or INVALID.

This module defines key names and types only. Values live in base.yaml.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_PATH = REPO_ROOT / "configs" / "base.yaml"

INVALID = "INVALID"
UNRESOLVED = "UNRESOLVED"
RUNNABLE = "RUNNABLE"
VALID = "VALID"

EXPERIMENT_KINDS = ("smoke", "research")
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class ConfigError(Exception):
    """Raised for configuration files that cannot be parsed."""


class UnresolvedChoiceError(LookupError):
    """Raised when code reads a research choice that is still null."""


class UndeclaredChoiceError(LookupError):
    """Raised when an experiment reads a research choice it did not declare it uses."""


# --------------------------------------------------------------------------------------------
# YAML loading
# --------------------------------------------------------------------------------------------

class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(
                    f"duplicate key {key!r} at line {key_node.start_mark.line + 1}"
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: Path) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML error: {exc}") from exc


# --------------------------------------------------------------------------------------------
# Type schema (names and types only)
# --------------------------------------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_str(value: Any) -> str | None:
    return None if isinstance(value, str) and value else "must be a non-empty string"


def _check_bool(value: Any) -> str | None:
    return None if isinstance(value, bool) else "must be true or false"


def _check_pos_int(value: Any) -> str | None:
    return None if _is_int(value) and value > 0 else "must be a positive integer"


def _check_pos_num(value: Any) -> str | None:
    return None if _is_number(value) and value > 0 else "must be a positive finite number"


def _check_fraction(value: Any) -> str | None:
    return None if _is_number(value) and 0 < value <= 1 else "must be a number in (0, 1]"


def _check_str_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list"
    if not all(isinstance(v, str) and v for v in value):
        return "must contain only non-empty strings"
    if len(set(value)) != len(value):
        return "must not contain duplicates"
    return None


def _check_pos_num_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list"
    if not all(_is_number(v) and v > 0 for v in value):
        return "must contain only positive finite numbers"
    if any(b <= a for a, b in zip(value, value[1:])):
        return "must be strictly increasing"
    return None


def _check_pos_int_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value or not all(_is_int(v) and v > 0 for v in value):
        return "must be a non-empty list of positive integers"
    if any(b <= a for a, b in zip(value, value[1:])):
        return "must be strictly increasing"
    return None


def _check_range(value: Any) -> str | None:
    if (isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value)
            and value[0] < value[1]):
        return None
    return "must be [low, high] with low < high"


def _check_grid(value: Any) -> str | None:
    if not isinstance(value, list) or len(value) < 2:
        return "must be a list of at least two severity levels"
    if not all(_is_number(v) for v in value):
        return "must contain only finite numbers"
    if value[0] != 0:
        return "must start at 0 (every severity grid includes the zero-injection level)"
    if any(b <= a for a, b in zip(value, value[1:])):
        return "must be strictly increasing"
    return None


def _check_pairs(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return "must be a non-empty list of pairs"
    for pair in value:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(p, str) for p in pair)
                and pair[0] != pair[1]):
            return "each entry must be a pair of two different family names"
    return None


def _check_str_map(value: Any) -> str | None:
    if not isinstance(value, dict) or not value:
        return "must be a non-empty mapping"
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        return "must map names to names"
    return None


@dataclass(frozen=True)
class MapOf:
    """A mapping with arbitrary string keys whose values all follow `schema`."""

    schema: Any


STR, BOOL = _check_str, _check_bool
POS_INT, POS_NUM, FRACTION = _check_pos_int, _check_pos_num, _check_fraction
STR_LIST, POS_NUM_LIST, POS_INT_LIST = _check_str_list, _check_pos_num_list, _check_pos_int_list
RANGE, GRID, PAIRS, STR_MAP = _check_range, _check_grid, _check_pairs, _check_str_map

# Keys ending in "?" are optional.
_WINDOW_RULE = {"history_len": POS_INT, "horizon_len": POS_INT, "min_valid_windows": POS_INT}

FROZEN_SCHEMA = {
    "protocol": {
        "static": {
            "history_len": POS_INT, "horizon_len": POS_INT, "sampling_interval_s": POS_NUM,
            "window_stride": POS_INT, "split": STR, "folds": POS_INT, "scenes": STR_LIST,
        },
        "moving": {
            "split": STR, "sampling_interval_s": POS_NUM, "window_stride": POS_INT,
            "resampling_tolerance_s": POS_NUM, "primary": _WINDOW_RULE, "fallback": _WINDOW_RULE,
            "ego_pose_coverage_min": FRACTION, "timestamp_mismatch_max_s": POS_NUM,
        },
        "min_seeds": POS_INT,
    },
    "geometry": {
        "conventions": {
            "image": STR, "ground_plane": STR, "platform_frame": STR, "world_frame": STR,
            "rotation": STR, "transform_direction": STR,
        },
        "trajectory_point": {"bbox_2d": STR, "annotation_3d": STR},
        "roundtrip_tolerance_px": {
            "synthetic_float64": POS_NUM, "real_float64": POS_NUM, "float32": POS_NUM,
        },
        "median_speed_diagnostic_range_mps": RANGE,
    },
    "injection": {
        "grids": MapOf({"parameter": STR, "unit": STR, "levels": GRID}),
        "conditions": MapOf({
            "family": STR, "grid": STR, "regimes": STR_LIST, "reconstruction?": STR,
        }),
        "d_max_m": POS_NUM,
        "d_cooldown": STR,
    },
    "association": {
        "tracker_gt_assoc_iou_threshold": FRACTION,
        "frag_gap_max_steps": POS_INT,
        "switch_gap_max_steps": POS_INT,
    },
    "collision": {"d_conflict_m": POS_NUM, "d_warn_m": POS_NUM_LIST, "tau_event_s": POS_NUM},
    "predictors": {
        "primary_set": STR_LIST,
        "cv_min_final_observations": POS_INT,
        "ca_min_final_observations": POS_INT,
        "learned": {
            "encoder_input": STR, "decoder_target": STR, "reconstruction": STR,
            "hidden_size": POS_INT, "num_layers": POS_INT, "attention": BOOL,
            "autoregressive_feedback": BOOL, "loss": STR, "optimizer": STR,
            "learning_rate": POS_NUM, "lr_scheduler": STR, "batch_size": POS_INT,
            "max_epochs": POS_INT, "early_stopping_metric": STR,
            "early_stopping_patience": POS_INT, "grad_clip_norm": POS_NUM,
        },
    },
    "analysis": {
        "representative_conditions": STR_MAP,
        "dominant_mode_codominance_gap": POS_NUM,
        "recovery_practical_threshold": POS_NUM,
        "inverse_requirement_q": POS_NUM_LIST,
        "history_ablation": {
            "history_lens": POS_INT_LIST, "horizon_len": POS_INT, "clean_ratio_max": POS_NUM,
            "delta_audc_max": POS_NUM, "epsilon_audc": POS_NUM,
        },
    },
    "hypotheses": {
        "tolerances": {
            "delta_h3": POS_NUM, "delta_h5": POS_NUM, "delta_h6": POS_NUM,
            "delta_h9": POS_NUM, "delta_h10": POS_NUM,
        },
        "h4_slope_ratio_min": POS_NUM,
        "h4_delta_aic_min": POS_NUM,
        "h7_delta_rho_min": POS_NUM,
        "h8": {
            "mode_pairs": PAIRS, "family_b_representative": STR,
            "identity_mode_order": STR_LIST, "cell_tolerance_fraction": POS_NUM,
        },
        "h10_classes": {
            "straight_heading_change_lt_deg": POS_NUM, "straight_min_speed_gt_mps": POS_NUM,
            "turning_heading_change_ge_deg": POS_NUM, "stop_go_low_speed_lt_mps": POS_NUM,
            "stop_go_low_speed_min_consecutive_steps": POS_INT,
            "stop_go_max_speed_gt_mps": POS_NUM,
        },
    },
    "datasets": {
        "mot17": {"split": STR, "static_sequences": STR_LIST, "moving_sequences": STR_LIST},
        "moving_candidates": STR_LIST,
    },
    "trackers": {"primary": STR_LIST, "canonical_param_names": STR_LIST},
}

META_SCHEMA = {
    "schema_version": POS_INT,
    "spec": {"version": STR, "path": STR, "sha256": STR},
}

DOMAINS_SCHEMA = {"regime": STR_LIST, "error_mode": STR_LIST}

# `value` must be present (null while unresolved); its content is checked by the constraint.
RESEARCH_CHOICE_ENTRY_SCHEMA = {"value": None, "required_by": STR, "spec": STR}

SELECTION_KEYS = ("regime", "error_mode", "severity_levels", "predictors", "scenes")


def _check_shape(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    """Check `value` against `schema`, appending one message per problem to `errors`."""
    if callable(schema):
        problem = schema(value)
        if problem:
            errors.append(f"{path}: {problem}")
        return
    if isinstance(schema, MapOf):
        if not isinstance(value, dict) or not value:
            errors.append(f"{path}: must be a non-empty mapping")
            return
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path}: key {key!r} must be a string")
                continue
            _check_shape(item, schema.schema, f"{path}.{key}", errors)
        return
    if not isinstance(value, dict):
        errors.append(f"{path}: must be a mapping")
        return
    expected = {k.rstrip("?"): k.endswith("?") for k in schema}
    for key in value:
        if key not in expected:
            errors.append(f"{path}.{key}: unknown key")
    for key, optional in expected.items():
        if key not in value:
            if not optional:
                errors.append(f"{path}.{key}: required key missing")
            continue
        sub = schema[key + "?" if optional else key]
        if sub is not None:
            _check_shape(value[key], sub, f"{path}.{key}", errors)


# --------------------------------------------------------------------------------------------
# Research-choice constraints (checked only when a value is set)
# --------------------------------------------------------------------------------------------

def _rc_seeds(value: Any, frozen: Mapping) -> str | None:
    minimum = frozen["protocol"]["min_seeds"]
    if not isinstance(value, list) or not all(_is_int(v) and v >= 0 for v in value):
        return "must be a list of non-negative integers"
    if len(set(value)) != len(value):
        return "must not contain duplicate seeds"
    if len(value) < minimum:
        return f"needs at least {minimum} seeds (frozen.protocol.min_seeds)"
    return None


def _rc_rotation(value: Any, frozen: Mapping) -> str | None:
    scenes = list(frozen["protocol"]["static"]["scenes"])
    if not isinstance(value, dict) or sorted(value) != sorted(scenes):
        return f"must map every static scene {scenes} to a validation scene"
    for test_scene, val_scene in value.items():
        if val_scene not in scenes:
            return f"{test_scene} -> {val_scene!r} is not a static scene"
        if val_scene == test_scene:
            return f"{test_scene} cannot validate on its own held-out scene"
    return None


def _rc_moving_dataset(value: Any, frozen: Mapping) -> str | None:
    candidates = frozen["datasets"]["moving_candidates"]
    return None if value in candidates else f"must be one of {candidates}"


def _rc_trackers(value: Any, frozen: Mapping) -> str | None:
    problem = _check_str_list(value)
    if problem:
        return problem
    missing = [t for t in frozen["trackers"]["primary"] if t not in value]
    return f"must include the primary trackers {missing}" if missing else None


def _rc_tracker_params(value: Any, frozen: Mapping) -> str | None:
    names = set(frozen["trackers"]["canonical_param_names"])
    if not isinstance(value, dict) or not value:
        return "must map tracker names to parameter mappings"
    for tracker, params in value.items():
        if not isinstance(params, dict):
            return f"{tracker}: parameters must be a mapping"
        unknown = sorted(set(params) - names)
        if unknown:
            return f"{tracker}: non-canonical parameter names {unknown}"
    return None


def _rc_pos_int(value: Any, frozen: Mapping) -> str | None:
    return _check_pos_int(value)


def _rc_pos_num(value: Any, frozen: Mapping) -> str | None:
    return _check_pos_num(value)


RESEARCH_CHOICE_CONSTRAINTS: dict[str, Callable[[Any, Mapping], str | None]] = {
    "seeds": _rc_seeds,
    "inner_validation_rotation": _rc_rotation,
    "moving_dataset": _rc_moving_dataset,
    "trackers_evaluated_set": _rc_trackers,
    "tracker_params": _rc_tracker_params,
    "h8_generation_attempts": _rc_pos_int,
    "discontinuity_rejection_v_max_mps": _rc_pos_num,
}


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------

def _validate_base(base: Any, repo_root: Path, errors: list[str]) -> None:
    if not isinstance(base, dict):
        errors.append("base: must be a mapping")
        return
    allowed = {"meta", "frozen", "domains", "research_choices"}
    for key in base:
        if key not in allowed:
            errors.append(f"base.{key}: unknown top-level key")
    for key in sorted(allowed):
        if key not in base:
            errors.append(f"base.{key}: required block missing")
    if errors:
        return

    _check_shape(base["meta"], META_SCHEMA, "meta", errors)
    _check_shape(base["frozen"], FROZEN_SCHEMA, "frozen", errors)
    _check_shape(base["domains"], DOMAINS_SCHEMA, "domains", errors)
    choices = base["research_choices"]
    if not isinstance(choices, dict):
        errors.append("research_choices: must be a mapping")
    else:
        for name in choices:
            if name not in RESEARCH_CHOICE_CONSTRAINTS:
                errors.append(f"research_choices.{name}: unknown research choice")
        for name in RESEARCH_CHOICE_CONSTRAINTS:
            if name not in choices:
                errors.append(f"research_choices.{name}: required entry missing")
            else:
                _check_shape(choices[name], RESEARCH_CHOICE_ENTRY_SCHEMA,
                             f"research_choices.{name}", errors)
    if errors:
        return

    _check_spec_hash(base["meta"]["spec"], repo_root, errors)
    _check_cross_references(base, errors)
    if errors:
        return
    for name, entry in choices.items():
        value = entry["value"]
        if value is not None:
            problem = RESEARCH_CHOICE_CONSTRAINTS[name](value, base["frozen"])
            if problem:
                errors.append(f"research_choices.{name}.value: {problem}")


def _check_spec_hash(spec: Mapping, repo_root: Path, errors: list[str]) -> None:
    rel = Path(spec["path"])
    if rel.is_absolute():
        errors.append("meta.spec.path: must be relative to the repository root")
        return
    spec_file = repo_root / rel
    try:
        digest = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    except OSError as exc:
        errors.append(
            f"meta.spec.path: cannot read {rel} ({exc.__class__.__name__}). The governing "
            "specification is not redistributed in this repository; place your copy there and "
            "see docs/spec/README.md"
        )
        return
    if digest != spec["sha256"]:
        errors.append(
            f"meta.spec.sha256: does not match {rel} (file hash {digest}); the specification "
            "copy changed or the recorded hash is wrong (see docs/spec/README.md)"
        )


def _check_cross_references(base: Mapping, errors: list[str]) -> None:
    frozen, domains = base["frozen"], base["domains"]
    grids = frozen["injection"]["grids"]
    conditions = frozen["injection"]["conditions"]
    for name, cond in conditions.items():
        if cond["grid"] not in grids:
            errors.append(f"frozen.injection.conditions.{name}.grid: unknown grid {cond['grid']!r}")
        for regime in cond["regimes"]:
            if regime not in domains["regime"]:
                errors.append(f"frozen.injection.conditions.{name}.regimes: unknown regime {regime!r}")
    expected_modes = {"none", *conditions}
    if set(domains["error_mode"]) != expected_modes:
        errors.append(
            "domains.error_mode: must be 'none' plus exactly the conditions in "
            f"frozen.injection.conditions ({sorted(conditions)})"
        )
    for family, condition in frozen["analysis"]["representative_conditions"].items():
        if condition not in conditions:
            errors.append(f"frozen.analysis.representative_conditions.{family}: unknown condition")
    h8 = frozen["hypotheses"]["h8"]
    if h8["family_b_representative"] not in conditions:
        errors.append("frozen.hypotheses.h8.family_b_representative: unknown condition")
    families = {c["family"] for c in conditions.values()}
    for pair in h8["mode_pairs"]:
        if not set(pair) <= families:
            errors.append(f"frozen.hypotheses.h8.mode_pairs: unknown family in {pair}")


def _validate_experiment(exp_doc: Any, base: Mapping, errors: list[str]) -> None:
    if not isinstance(exp_doc, dict):
        errors.append("experiment file: must be a mapping")
        return
    for key in exp_doc:
        if key != "experiment":
            errors.append(
                f"experiment file: top-level key {key!r} is not overridable; an experiment "
                "file may contain only an 'experiment' block"
            )
    if "experiment" not in exp_doc:
        errors.append("experiment file: 'experiment' block missing")
        return
    exp = exp_doc["experiment"]
    _check_shape(exp, {"name": STR, "kind": STR, "uses_research_choices": None,
                       "selection": None}, "experiment", errors)
    if errors:
        return

    if not _NAME_PATTERN.match(exp["name"]):
        errors.append("experiment.name: use lowercase letters, digits, '_' and '-' only")
    if exp["kind"] not in EXPERIMENT_KINDS:
        errors.append(f"experiment.kind: must be one of {list(EXPERIMENT_KINDS)}")

    uses = exp["uses_research_choices"]
    if not isinstance(uses, list) or not all(isinstance(u, str) for u in uses):
        errors.append("experiment.uses_research_choices: must be a list of names")
        uses = []
    elif len(set(uses)) != len(uses):
        errors.append("experiment.uses_research_choices: must not contain duplicates")
    for name in uses:
        if name not in RESEARCH_CHOICE_CONSTRAINTS:
            errors.append(f"experiment.uses_research_choices: unknown research choice {name!r}")
    if exp["kind"] == "smoke" and uses:
        errors.append("experiment.uses_research_choices: a smoke experiment must use none")
    if exp["kind"] == "research" and "seeds" not in uses:
        errors.append("experiment.uses_research_choices: a research experiment must declare 'seeds'")

    _validate_selection(exp["selection"], base, errors)


def _validate_selection(selection: Any, base: Mapping, errors: list[str]) -> None:
    if not isinstance(selection, dict):
        errors.append("experiment.selection: must be a mapping")
        return
    for key in selection:
        if key not in SELECTION_KEYS:
            errors.append(f"experiment.selection.{key}: unknown key")
    frozen, domains = base["frozen"], base["domains"]
    regime = selection.get("regime")
    if "regime" not in selection:
        errors.append("experiment.selection.regime: required key missing")
    elif regime not in domains["regime"]:
        errors.append(f"experiment.selection.regime: must be one of {domains['regime']}")

    mode = selection.get("error_mode", "none")
    if mode not in domains["error_mode"]:
        errors.append(f"experiment.selection.error_mode: must be one of {domains['error_mode']}")
        mode = None
    elif mode != "none" and regime in domains["regime"]:
        allowed = frozen["injection"]["conditions"][mode]["regimes"]
        if regime not in allowed:
            errors.append(f"experiment.selection.error_mode: {mode} is not defined for the {regime} regime")

    if "severity_levels" in selection:
        levels = selection["severity_levels"]
        if mode in (None, "none"):
            errors.append("experiment.selection.severity_levels: requires an error_mode other than 'none'")
        else:
            grid = frozen["injection"]["grids"][frozen["injection"]["conditions"][mode]["grid"]]["levels"]
            problem = _check_grid(levels)
            if problem:
                errors.append(f"experiment.selection.severity_levels: {problem}")
            elif not set(levels) <= set(grid):
                errors.append(
                    f"experiment.selection.severity_levels: must be a subset of the frozen {mode} grid {grid}"
                )

    if "predictors" in selection:
        allowed = frozen["predictors"]["primary_set"]
        chosen = selection["predictors"]
        problem = _check_str_list(chosen)
        if problem:
            errors.append(f"experiment.selection.predictors: {problem}")
        elif not set(chosen) <= set(allowed):
            errors.append(f"experiment.selection.predictors: must be a subset of {allowed}")

    if "scenes" in selection:
        allowed = frozen["protocol"]["static"]["scenes"]
        chosen = selection["scenes"]
        problem = _check_str_list(chosen)
        if regime != "static":
            errors.append("experiment.selection.scenes: applies only to the static regime")
        elif problem:
            errors.append(f"experiment.selection.scenes: {problem}")
        elif not set(chosen) <= set(allowed):
            errors.append(f"experiment.selection.scenes: must be a subset of {allowed}")


# --------------------------------------------------------------------------------------------
# Hashing and the read-only view
# --------------------------------------------------------------------------------------------

def canonical_json(data: Any) -> str:
    """Serialize to canonical JSON: sorted keys, compact separators, floats via repr."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def config_hash(data: Mapping) -> str:
    """SHA-256 of the canonical merged configuration. Depends on nothing else."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class Config:
    """Read-only, validated configuration.

    Research choices are reachable only through `research_choice`, which refuses to return a
    value that is unresolved or that the experiment did not declare it uses.
    """

    meta: Mapping
    frozen: Mapping
    domains: Mapping
    experiment: Mapping | None
    hash: str
    _choices: Mapping = field(repr=False)
    _canonical: str = field(repr=False)

    def research_choice(self, name: str) -> Any:
        if name not in self._choices:
            raise KeyError(f"unknown research choice {name!r}")
        if self.experiment is not None and name not in self.experiment["uses_research_choices"]:
            raise UndeclaredChoiceError(
                f"experiment {self.experiment['name']!r} did not declare research choice {name!r}"
            )
        value = self._choices[name]["value"]
        if value is None:
            raise UnresolvedChoiceError(
                f"research choice {name!r} is unresolved (required by "
                f"{self._choices[name]['required_by']})"
            )
        return value

    def canonical_dict(self) -> dict:
        """A fresh, mutable copy of the merged configuration, for serialization only."""
        return json.loads(self._canonical)

    def protocol_complete(self) -> bool | None:
        """Whether the selected severity grid is the full frozen grid; None if no mode selected."""
        if self.experiment is None:
            return None
        selection = self.experiment["selection"]
        mode = selection.get("error_mode", "none")
        if mode == "none":
            return None
        grid_name = self.frozen["injection"]["conditions"][mode]["grid"]
        grid = list(self.frozen["injection"]["grids"][grid_name]["levels"])
        return "severity_levels" not in selection or list(selection["severity_levels"]) == grid


@dataclass(frozen=True)
class ValidationResult:
    status: str
    errors: tuple[str, ...]
    unresolved: tuple[str, ...]          # every research choice that is still null
    unresolved_used: tuple[str, ...]     # those the experiment declares it uses
    config: Config | None


def validate(experiment_path: Path | None = None, base_path: Path | None = None,
             repo_root: Path | None = None) -> ValidationResult:
    """Load and validate base.yaml, optionally merged with one experiment file."""
    base_path = Path(base_path) if base_path else DEFAULT_BASE_PATH
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    errors: list[str] = []
    try:
        base = load_yaml(base_path)
    except ConfigError as exc:
        return ValidationResult(INVALID, (str(exc),), (), (), None)
    _validate_base(base, repo_root, errors)

    exp_doc = None
    if experiment_path is not None and not errors:
        try:
            exp_doc = load_yaml(Path(experiment_path))
        except ConfigError as exc:
            errors.append(str(exc))
        else:
            _validate_experiment(exp_doc, base, errors)
    if errors:
        return ValidationResult(INVALID, tuple(errors), (), (), None)

    unresolved = tuple(name for name, entry in base["research_choices"].items()
                       if entry["value"] is None)
    merged = dict(base)
    experiment = None
    if exp_doc is not None:
        experiment = exp_doc["experiment"]
        merged["experiment"] = experiment
    canonical = canonical_json(merged)
    config = Config(
        meta=_deep_freeze(base["meta"]),
        frozen=_deep_freeze(base["frozen"]),
        domains=_deep_freeze(base["domains"]),
        experiment=_deep_freeze(experiment) if experiment is not None else None,
        hash=config_hash(merged),
        _choices=_deep_freeze(base["research_choices"]),
        _canonical=canonical,
    )
    if experiment is None:
        return ValidationResult(VALID, (), unresolved, (), config)
    used = tuple(n for n in experiment["uses_research_choices"] if n in unresolved)
    status = UNRESOLVED if used else RUNNABLE
    return ValidationResult(status, (), unresolved, used, config)
