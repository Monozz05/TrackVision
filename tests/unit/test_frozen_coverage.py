"""Every leaf under `frozen` in configs/base.yaml is verified against specification v4.15.

Two kinds of check, reported separately:

  text       The expected value is extracted from the specification text by a pattern anchored to
             its own wording, then compared. Where noted, the extracted text is normalised first
             (lower-case with '-' and spaces as '_', CamelCase to snake_case, a number word to an
             integer, or a Greek symbol / degree sign to its ASCII config name).
  assertion  The config value is a project label for specification prose that cannot be extracted
             mechanically. The expected value is written here, tied to the cited section, and the
             quoted anchor phrase must appear verbatim in the specification. This is an assertion,
             not a text comparison.

A completeness test fails if any frozen leaf has no check, and a mutation test shows that changing
any frozen value makes its check fail.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

import pytest
import yaml

from conftest import BASE_PATH, SPEC_PATH

# --- spec text extraction (moved verbatim from the former test_spec_transcription.py) -----------

SPEC = SPEC_PATH.read_text(encoding="utf-8")


def _unique(pattern: str) -> re.Match:
    matches = list(re.finditer(pattern, SPEC))
    assert len(matches) == 1, f"pattern {pattern!r} matched {len(matches)} times in the spec"
    return matches[0]


def _number(pattern: str) -> float:
    """A value the spec may state more than once; every statement must agree."""
    values = {m.group(1).replace(",", "") for m in re.finditer(pattern, SPEC)}
    assert values, f"pattern {pattern!r} matched nothing in the spec"
    assert len(values) == 1, f"the spec states conflicting values for {pattern!r}: {sorted(values)}"
    return float(values.pop())


def _number_set(pattern: str) -> list[float]:
    """Parse the `{a, b, c}` set that the anchored pattern captures."""
    items = _unique(pattern).group(1).split(",")
    return [float(re.sub(r"[^\d.]", "", item)) for item in items]


SCALARS = {
    "static history_len": (r"\*\*(\d+) / \d+ steps\*\* \(unconditional\)", "protocol.static.history_len"),
    "static horizon_len": (r"\*\*\d+ / (\d+) steps\*\* \(unconditional\)", "protocol.static.horizon_len"),
    "static sampling interval": (r"\| Sampling interval \| ([\d.]+) s \(2\.5 Hz\)", "protocol.static.sampling_interval_s"),
    "folds": (r"leave-one-scene-out, (\d+) folds", "protocol.static.folds"),
    "min seeds": (r"\| Seeds \| ≥ (\d+) \|", "protocol.min_seeds"),
    "resampling tolerance": (r"\| t_source − t_target \| ≤ ([\d.]+) s", "protocol.moving.resampling_tolerance_s"),
    "moving primary windows": (r"valid 8/12 windows \*\*≥ ([\d,]+)\*\*", "protocol.moving.primary.min_valid_windows"),
    "moving fallback windows": (r"valid 8/8 windows are \*\*([\d,]+) – [\d,]+\*\*", "protocol.moving.fallback.min_valid_windows"),
    "moving unavailable below": (r"valid 8/8 windows \*\*< ([\d,]+)\*\*", "protocol.moving.fallback.min_valid_windows"),
    "fallback horizon": (r"horizon_len\s+= (\d+)\n> sampling_interval = 0\.4 s", "protocol.moving.fallback.horizon_len"),
    "ego-pose coverage": (r"Continuous ego-pose available for \*\*≥ (\d+)%\*\*", "protocol.moving.ego_pose_coverage_min", 0.01),
    "timestamp mismatch": (r"timestamp mismatch \*\*≤ ([\d.]+) s\*\*", "protocol.moving.timestamp_mismatch_max_s"),
    "round-trip synthetic": (r"\| Synthetic tests, float64 \| ≤ ([\de.-]+) px", "geometry.roundtrip_tolerance_px.synthetic_float64"),
    "round-trip real": (r"\| Real dataset homographies, float64 \| ≤ ([\de.-]+) px", "geometry.roundtrip_tolerance_px.real_float64"),
    "round-trip float32": (r"\| Float32 paths \| ≤ ([\de.-]+) px", "geometry.roundtrip_tolerance_px.float32"),
    "D_MAX": (r"D_MAX = ([\d.]+) metres", "injection.d_max_m"),
    "IoU threshold": (r"TRACKER_GT_ASSOC_IOU_THRESHOLD = ([\d.]+)\n", "association.tracker_gt_assoc_iou_threshold"),
    "FRAG_GAP_MAX": (r"FRAG_GAP_MAX = (\d+) timesteps", "association.frag_gap_max_steps"),
    "SWITCH_GAP_MAX": (r"SWITCH_GAP_MAX = (\d+) timestep\n", "association.switch_gap_max_steps"),
    "d_conflict": (r"`d_conflict` is held fixed at ([\d.]+) m", "collision.d_conflict_m"),
    "tau_event": (r"τ_event = ([\d.]+) s is frozen", "collision.tau_event_s"),
    "hidden size": (r"Hidden (\d+), \d+ layer\.", "predictors.learned.hidden_size"),
    "layers": (r"Hidden \d+, (\d+) layer\.", "predictors.learned.num_layers"),
    "learning rate": (r"Adam, lr ([\de.-]+),", "predictors.learned.learning_rate"),
    "batch size": (r"ReduceLROnPlateau, batch (\d+),", "predictors.learned.batch_size"),
    "max epochs": (r"≤(\d+) epochs", "predictors.learned.max_epochs"),
    "patience": (r"patience (\d+), gradient", "predictors.learned.early_stopping_patience"),
    "grad clip": (r"clipping at norm ([\d.]+)\.", "predictors.learned.grad_clip_norm"),
    "co-dominance gap": (r"differ by \*\*less than ([\d.]+)\*\*", "analysis.dominant_mode_codominance_gap"),
    "practical recovery": (r"recovery fraction is \*\*≥ ([\d.]+)\*\*", "analysis.recovery_practical_threshold"),
    "P-15 clean ratio": (r"clean ADE\s+≤ ([\d.]+) × ADE_best", "analysis.history_ablation.clean_ratio_max"),
    "P-15 delta AUDC": (r"within tolerance  ⟺  δ_AUDC ≤ ([\d.]+)", "analysis.history_ablation.delta_audc_max"),
    "epsilon AUDC": (r"ε_AUDC = ([\d.]+)\n", "analysis.history_ablation.epsilon_audc"),
    "P-15 horizon": (r"remains fixed at (\d+) steps for every static P-15", "analysis.history_ablation.horizon_len"),
    "delta H3": (r"\| `δ_H3` \| \*\*([\d.]+)\*\*", "hypotheses.tolerances.delta_h3"),
    "delta H5": (r"\| `δ_H5` \| \*\*([\d.]+)\*\*", "hypotheses.tolerances.delta_h5"),
    "delta H6": (r"\| `δ_H6` \| \*\*([\d.]+)\*\*", "hypotheses.tolerances.delta_h6"),
    "delta H9": (r"\| `δ_H9` \| \*\*([\d.]+)\*\*", "hypotheses.tolerances.delta_h9"),
    "delta H10": (r"\| `δ_H10` \| \*\*([\d.]+)\*\*", "hypotheses.tolerances.delta_h10"),
    "H4 slope ratio": (r"if `b_hi / b_lo ≥ ([\d.]+)`", "hypotheses.h4_slope_ratio_min"),
    "H4 delta AIC": (r"A quadratic preference of `ΔAIC ≥ (\d+)`", "hypotheses.h4_delta_aic_min"),
    "H7 delta rho": (r"\*\*H7 is supported if `Δρ ≥ ([\d.]+)`", "hypotheses.h7_delta_rho_min"),
    "H8 cell tolerance": (r"≤ ([\d.]+) × target_cell_severity", "hypotheses.h8.cell_tolerance_fraction"),
    "H10 straight heading": (r"\| \*\*Straight\*\* \| Total absolute heading change over the window < (\d+)°", "hypotheses.h10_classes.straight_heading_change_lt_deg"),
    "H10 straight speed": (r"and minimum instantaneous speed > ([\d.]+) m/s \|", "hypotheses.h10_classes.straight_min_speed_gt_mps"),
    "H10 turning": (r"\| \*\*Turning\*\* \| Total absolute heading change over the window ≥ (\d+)°", "hypotheses.h10_classes.turning_heading_change_ge_deg"),
    "H10 stop-go low": (r"Minimum instantaneous speed < ([\d.]+) m/s for ≥ \d+ consecutive", "hypotheses.h10_classes.stop_go_low_speed_lt_mps"),
    "H10 stop-go steps": (r"< [\d.]+ m/s for ≥ (\d+) consecutive steps", "hypotheses.h10_classes.stop_go_low_speed_min_consecutive_steps"),
    "H10 stop-go high": (r"and maximum instantaneous speed > ([\d.]+) m/s \|", "hypotheses.h10_classes.stop_go_max_speed_gt_mps"),
}

SETS = {
    "grid A": (r"Levels: σ ∈ \{([^}]*)\} m", "injection.grids.A.levels"),
    "grid B": (r"Levels: p ∈ \{([^}]*)\}\n", "injection.grids.B.levels"),
    "grid C": (r"split probability p ∈ \{([^}]*)\}", "injection.grids.C.levels"),
    "grid D": (r"switch probability p ∈ \{([^}]*)\}", "injection.grids.D.levels"),
    "grid E1": (r"σ_t ∈ \{([^}]*)\} m", "injection.grids.E1.levels"),
    "grid E2": (r"σ_r ∈ \{([^}]*)\}°", "injection.grids.E2.levels"),
    "grid E3": (r"drift rate r ∈ \{([^}]*)\}", "injection.grids.E3.levels"),
    "d_warn": (r"d_warn\s+∈ \{([^}]*)\}\s+with", "collision.d_warn_m"),
    "inverse requirement q": (r"\*\*Primary requirement levels:\*\* `q ∈ \{([^}]*)\}`", "analysis.inverse_requirement_q"),
}


def spec_static_scenes() -> list[str]:
    line = _unique(r"\nETH · HOTEL · UNIV \(students003\) · ZARA1 · ZARA2\n").group(0)
    return [re.sub(r"\s*\(.*\)", "", s).strip() for s in line.strip().split("·")]


def spec_history_lens() -> list[int]:
    line = _unique(r"4 steps = 1\.6 s[^\n]*").group(0)
    return [int(n) for n in re.findall(r"(?:^|·)\s*(\d+)(?: steps)? =", line)]


def spec_mot17(kind: str) -> list[str]:
    seqs = _unique(rf"{kind}:\s+(MOT17-[\d, MOT17-]+)\n").group(1)
    return [s.strip() for s in seqs.split(",")]


def spec_h8_pairs() -> list[list[str]]:
    line = _unique(r"\nA × B    A × C    A × D    B × C    B × D    C × D\n").group(0)
    return [[p.strip() for p in pair.split("×")] for pair in re.split(r"\s{2,}", line.strip())]


def spec_tracker_param_names() -> list[str]:
    block = _unique(r"tracker_params:\n  ([^\n]+)\n  ([^\n]+)\n").groups()
    return [n.strip() for part in block for n in part.split("·")]


@dataclass(frozen=True)
class Check:
    section: str
    kind: str                       # "text" or "assertion"
    expected: Callable[[], Any]
    anchor: str | None = None       # assertion only: phrase that must appear verbatim in the spec
    note: str = ""


REGISTRY: dict[str, list[Check]] = {}


def _add(path: str, check: Check) -> None:
    REGISTRY.setdefault(path, []).append(check)


def text(path: str, section: str, extract: Callable[[], Any], note: str = "") -> None:
    _add(path, Check(section, "text", extract, None, note))


def asserted(path: str, section: str, value: Any, anchor: str, note: str = "") -> None:
    _add(path, Check(section, "assertion", lambda: value, anchor, note))


# --- normalisation helpers (each documented in the module docstring) --------------------------

def snake(s: str) -> str:
    return re.sub(r"[\s-]+", "_", s.strip()).lower()


def camel_to_snake(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", s).lower()


NUMBER_WORDS = {"two": 2, "three": 3}
SYMBOLS = {"σ": "sigma", "σ_t": "sigma_t", "σ_r": "sigma_r", "p": "p", "r": "r"}
UNIT_SIGNS = {"m": "m", "°": "deg"}


def group(pattern: str, index: int = 1) -> str:
    """A captured group that may occur several times in the spec; every occurrence must agree."""
    values = {m.group(index) for m in re.finditer(pattern, SPEC)}
    assert values, f"pattern {pattern!r} matched nothing in the spec"
    assert len(values) == 1, f"the spec states conflicting values for {pattern!r}: {sorted(values)}"
    return values.pop()


def cell(row_label: str) -> str:
    return _unique(rf"\n\| {row_label} \| ([^|\n]+) \|").group(1).strip()


# --- 1. existing text comparisons, now with their sections ------------------------------------

SCALAR_SECTIONS = {
    "static history_len": "§5.3", "static horizon_len": "§5.3", "static sampling interval": "§5.3",
    "folds": "§5.3", "min seeds": "§5.3",
    "resampling tolerance": "§5.3 (moving-camera temporal resampling)",
    "moving primary windows": "§6.4", "moving fallback windows": "§6.4", "moving unavailable below": "§6.4",
    "fallback horizon": "§5.3.3", "ego-pose coverage": "§6.4", "timestamp mismatch": "§6.4",
    "round-trip synthetic": "§11.2", "round-trip real": "§11.2", "round-trip float32": "§11.2",
    "D_MAX": "§3.5 D", "IoU threshold": "§3.7", "FRAG_GAP_MAX": "§3.7", "SWITCH_GAP_MAX": "§3.7",
    "d_conflict": "§3.8", "tau_event": "§3.8",
    "hidden size": "§5.4", "layers": "§5.4", "learning rate": "§5.4", "batch size": "§5.4",
    "max epochs": "§5.4", "patience": "§5.4", "grad clip": "§5.4",
    "co-dominance gap": "§2.2.3", "practical recovery": "§2.2.4",
    "P-15 clean ratio": "§2.2.5", "P-15 delta AUDC": "§2.2.5", "epsilon AUDC": "§2.2.5", "P-15 horizon": "§2.2.5",
    "delta H3": "§2.3.12", "delta H5": "§2.3.12", "delta H6": "§2.3.12", "delta H9": "§2.3.12", "delta H10": "§2.3.12",
    "H4 slope ratio": "§2.3.2", "H4 delta AIC": "§2.3.2", "H7 delta rho": "§2.3.6", "H8 cell tolerance": "§2.3.4",
    "H10 straight heading": "§2.3.3", "H10 straight speed": "§2.3.3", "H10 turning": "§2.3.3",
    "H10 stop-go low": "§2.3.3", "H10 stop-go steps": "§2.3.3", "H10 stop-go high": "§2.3.3",
}
SET_SECTIONS = {
    "grid A": "§3.5 A", "grid B": "§3.5 B", "grid C": "§3.5 C", "grid D": "§3.5 D",
    "grid E1": "§3.5 E", "grid E2": "§3.5 E", "grid E3": "§3.5 E",
    "d_warn": "§2.3.10", "inverse requirement q": "§2.2.6",
}
assert set(SCALAR_SECTIONS) == set(SCALARS) and set(SET_SECTIONS) == set(SETS)

for _label, (_pattern, _path, *_scale) in SCALARS.items():
    text(_path, SCALAR_SECTIONS[_label],
         lambda p=_pattern, s=(_scale[0] if _scale else 1): _number(p) * s)
for _label, (_pattern, _path) in SETS.items():
    text(_path, SET_SECTIONS[_label], lambda p=_pattern: _number_set(p))

text("protocol.static.scenes", "§5.3.1", spec_static_scenes)
text("analysis.history_ablation.history_lens", "§2.2.5", spec_history_lens)
text("datasets.mot17.static_sequences", "§6.2", lambda: spec_mot17("Static"))
text("datasets.mot17.moving_sequences", "§6.2", lambda: spec_mot17("Moving"))
text("hypotheses.h8.mode_pairs", "§2.3.4", spec_h8_pairs)
text("trackers.canonical_param_names", "§11.4", spec_tracker_param_names)

# --- 2. protocol -----------------------------------------------------------------------------

_STRIDE = r"\| Window stride \| (\d+), documented \| (\d+), documented \|"
_SPLIT = r"\| Split \| ([a-z-]+), 5 folds \(§5\.3\.1\) \| ([a-z-]+) \|"
text("protocol.static.window_stride", "§5.3", lambda: int(group(_STRIDE, 1)))
text("protocol.moving.window_stride", "§5.3", lambda: int(group(_STRIDE, 2)))
text("protocol.static.split", "§5.3", lambda: snake(group(_SPLIT, 1)), "normalised")
text("protocol.moving.split", "§5.3", lambda: snake(group(_SPLIT, 2)), "normalised")
text("protocol.moving.sampling_interval_s", "§5.3",
     lambda: float(group(r"\| Sampling interval \| [\d.]+ s \(2\.5 Hz\) \| ([\d.]+) s \(2\.5 Hz\), after deterministic resampling")))
_MOVING_PRIMARY = r"\*\*(\d+) / (\d+)\*\* if ≥ 2,000 valid windows"
text("protocol.moving.primary.history_len", "§5.3", lambda: int(group(_MOVING_PRIMARY, 1)))
text("protocol.moving.primary.horizon_len", "§5.3", lambda: int(group(_MOVING_PRIMARY, 2)))
text("protocol.moving.fallback.history_len", "§5.3.3",
     lambda: int(group(r"history_len\s+= (\d+)\n> horizon_len\s+= \d+\n> sampling_interval = 0\.4 s")))

# --- 3. geometry -----------------------------------------------------------------------------

text("geometry.conventions.image", "§4.3 project-wide conventions", lambda: cell("Image coordinates"))
text("geometry.conventions.ground_plane", "§4.3 project-wide conventions", lambda: cell("Ground plane"))
text("geometry.conventions.platform_frame", "§4.3 project-wide conventions", lambda: cell(r"Platform frame \*\*P\*\*"))
text("geometry.conventions.world_frame", "§4.3 project-wide conventions", lambda: cell(r"World frame \*\*W\*\*"))
text("geometry.conventions.rotation", "§4.3 project-wide conventions", lambda: cell("Rotation"))
asserted("geometry.conventions.transform_direction", "§4.3 project-wide conventions",
         "source frame to target frame",
         "all transforms map **from** the named source frame **to** the named target frame",
         "config label paraphrases the spec cell")
text("geometry.trajectory_point.bbox_2d", "§4.2",
     lambda: snake(_unique(r"\| 2D bounding box \| ([A-Za-z-]+):").group(1)), "normalised")
asserted("geometry.trajectory_point.annotation_3d", "§4.2", "bottom_face_centre_projected_to_ground",
         "**bottom-face centre of the 3D bounding box projected onto the ground plane**",
         "config label for the spec phrase")
text("geometry.median_speed_diagnostic_range_mps", "§14.1 (P2 gate)",
     lambda: [float(x) for x in _unique(r"≈([\d.]+)–([\d.]+) m/s is a diagnostic reference range").groups()])

# --- 4. injection: grid labels, conditions, cooldown -----------------------------------------

_PARAMETERS = {
    "A": r"Levels: (σ) ∈ \{", "B": r"Levels: (p) ∈ \{[^}]*\}\n", "C": r"split probability (p) ∈ \{",
    "D": r"switch probability (p) ∈ \{", "E1": r"(σ_t) ∈ \{", "E2": r"(σ_r) ∈ \{", "E3": r"drift rate (r) ∈ \{",
}
for _grid, _pattern in _PARAMETERS.items():
    text(f"injection.grids.{_grid}.parameter", "§3.5", lambda p=_pattern: SYMBOLS[group(p)], "Greek symbol to ASCII name")
text("injection.grids.A.unit", "§3.5 A", lambda: UNIT_SIGNS[group(r"Levels: σ ∈ \{[^}]*\} (m)\n")])
text("injection.grids.E1.unit", "§3.5 E", lambda: UNIT_SIGNS[group(r"σ_t ∈ \{[^}]*\} (m) \|")])
text("injection.grids.E2.unit", "§3.5 E", lambda: UNIT_SIGNS[group(r"σ_r ∈ \{[^}]*\}(°) \|")], "degree sign to 'deg'")
asserted("injection.grids.B.unit", "§3.5 B", "probability",
         "Remove each observation independently with probability p", "config label")
asserted("injection.grids.C.unit", "§3.5 C", "probability_per_timestep", "per-timestep split probability p", "config label")
asserted("injection.grids.D.unit", "§3.5 D", "probability_per_timestep", "per-timestep switch probability p", "config label")
asserted("injection.grids.E3.unit", "§3.5 E", "fraction_of_distance_travelled",
         "drift rate r ∈ {0, 0.005, 0.01, 0.02, 0.05} of distance travelled", "config label")


def spec_conditions() -> list[str]:
    return [c.strip() for c in _unique(r"8 injection conditions\s+([^\n]+)").group(1).split("·")]


def spec_families() -> list[str]:
    return [f.strip() for f in _unique(r"5 error families\s+([^\n]+)").group(1).split("·")]


def family_of(condition: str) -> str:
    """The family whose letter the condition name starts with (§3.5 terminology)."""
    (family,) = [f for f in spec_families() if condition.startswith(f)]
    return family


def regimes_of(condition: str) -> list[str]:
    both = _unique(r"\*\*Tracking errors \(both regimes\):\*\* ([^\n]+)").group(1)
    moving_only = _unique(r"\*\*Stabilization error \(moving regime only\):\*\* ([^\n]+)").group(1)
    family = family_of(condition)
    if re.search(rf"(^|· ){family} ", both):
        return ["static", "moving"]
    if re.search(rf"(^|· ){family} ", moving_only):
        return ["moving"]
    raise AssertionError(f"family {family} not found in §4.1")


_GRID_ANCHORS = {
    "A": ("A", "Levels: σ ∈ {0, 0.05"),
    "B1": ("B", "Remove each observation independently with probability p; two reconstruction conditions (§3.6)."),
    "B2": ("B", "Remove each observation independently with probability p; two reconstruction conditions (§3.6)."),
    "C": ("C", "Levels: per-timestep split probability p"),
    "D": ("D", "Levels: per-timestep switch probability p"),
    "E1": ("E1", "| **E1** Translation noise |"),
    "E2": ("E2", "| **E2** Rotation noise |"),
    "E3": ("E3", "| **E3** Pose drift |"),
}
for _condition in ("A", "B1", "B2", "C", "D", "E1", "E2", "E3"):
    text(f"injection.conditions.{_condition}.family", "§3.5 terminology",
         lambda c=_condition: (c in spec_conditions()) and family_of(c), "derived: family letter that starts the condition name")
    text(f"injection.conditions.{_condition}.regimes", "§4.1",
         lambda c=_condition: regimes_of(c), "derived from the §4.1 regime lines")
    _grid, _anchor = _GRID_ANCHORS[_condition]
    asserted(f"injection.conditions.{_condition}.grid", "§3.5", _grid, _anchor,
             "which level set the condition uses")
asserted("injection.conditions.B1.reconstruction", "§3.6", "linear_interior_gaps_only",
         "Linear reconstruction is applied **only to interior gaps**", "config label")
asserted("injection.conditions.B2.reconstruction", "§3.6", "none",
         "Missing observations remain missing.", "config label")
asserted("injection.d_cooldown", "§3.5 D", "locked_until_end_of_segment",
         "both participating tracks remain locked until the end of the current evaluation segment", "config label")

# --- 5. predictors ---------------------------------------------------------------------------

text("predictors.primary_set", "§5.4 primary predictor set",
     lambda: [p.strip() for p in _unique(r"### Primary predictor set\n[^{]*\n\{ ([A-Z, ]+) \}\n").group(1).split(",")])
_MIN_OBS = r"\*\*CV requires (\w+) valid final observations; CA requires (\w+)\.\*\*"
text("predictors.cv_min_final_observations", "§5.4", lambda: NUMBER_WORDS[group(_MIN_OBS, 1)], "number word")
text("predictors.ca_min_final_observations", "§5.4", lambda: NUMBER_WORDS[group(_MIN_OBS, 2)], "number word")
text("predictors.learned.encoder_input", "§5.4 learning target",
     lambda: snake(_unique(r"\*\*Encoder input\*\* — `history_len − 1` ([a-z -]+):").group(1)), "normalised")
text("predictors.learned.decoder_target", "§5.4 learning target",
     lambda: snake(_unique(r"\*\*Decoder targets\*\* — `horizon_len` ([a-z -]+),").group(1)), "normalised")
asserted("predictors.learned.reconstruction", "§5.4 learning target", "cumulative_sum_from_last_observation",
         "by cumulative summation from the final observed position", "config label")
asserted("predictors.learned.attention", "§5.4 decoder conditioning", False,
         "no attention over encoder states", "boolean for the spec's prohibition")
asserted("predictors.learned.autoregressive_feedback", "§5.4 decoder conditioning", False,
         "no autoregressive feedback from predicted positions", "boolean for the spec's prohibition")
_TRAINING = r"Loss (MSE on displacements)\. (Adam), lr [\de.-]+, (ReduceLROnPlateau),"
text("predictors.learned.loss", "§5.4", lambda: snake(group(_TRAINING, 1)), "normalised")
text("predictors.learned.optimizer", "§5.4", lambda: snake(group(_TRAINING, 2)), "normalised")
text("predictors.learned.lr_scheduler", "§5.4", lambda: camel_to_snake(group(_TRAINING, 3)), "CamelCase to snake_case")
text("predictors.learned.early_stopping_metric", "§5.4",
     lambda: snake(group(r"early stopping on (validation ADE) with patience")), "normalised")

# --- 6. analysis, hypotheses, datasets, trackers ---------------------------------------------

_REPRESENTATIVES = r"\nA → (\w+)\s+B → (\w+)\s+C → (\w+)\s+D → (\w+)\n"
for _index, _family in enumerate("ABCD", start=1):
    text(f"analysis.representative_conditions.{_family}", "§2.2.0, §2.2.1, §2.2.3",
         lambda i=_index: group(_REPRESENTATIVES, i))
text("analysis.representative_conditions.E", "§2.2.0", lambda: group(r"mode E represented by \*\*(E\d)\*\*"))
text("hypotheses.h8.family_b_representative", "§2.3.4", lambda: group(r"family B is represented by (B\d)"))
text("hypotheses.h8.identity_mode_order", "§2.3.4",
     lambda: list(_unique(r"frozen application order is ([A-E]) before ([A-E])").groups()))
text("datasets.mot17.split", "§6.2", lambda: snake(_unique(r"\*\*(Half)-split\.\*\*").group(1)), "normalised")
text("datasets.moving_candidates", "§6.3",
     lambda: [c.strip() for c in _unique(r"Candidates:\s+([^\n]+)").group(1).split("·")])
text("trackers.primary", "§11.5",
     lambda: list(_unique(r"\*\*Primary trackers: (\S+) and (\S+?)\.\*\*").groups()))


# --- verification ----------------------------------------------------------------------------

def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(actual - expected) <= 1e-12
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(_same(a, e) for a, e in zip(actual, expected))
    return actual == expected


def failures(path: str, value: Any) -> list[str]:
    problems = []
    for check in REGISTRY[path]:
        if check.anchor is not None and check.anchor not in SPEC:
            problems.append(f"{path}: anchor phrase not found in spec {check.section}: {check.anchor!r}")
            continue
        expected = check.expected()
        if not _same(value, expected):
            problems.append(f"{path} ({check.kind}, {check.section}): config {value!r} != expected {expected!r}")
    return problems


def _load_frozen() -> dict:
    return yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))["frozen"]


def _leaves(node: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{prefix}.{key}" if prefix else key))
        return out
    return {prefix: node}


FROZEN_LEAVES = _leaves(_load_frozen())


def test_every_frozen_leaf_has_a_check():
    missing = sorted(set(FROZEN_LEAVES) - set(REGISTRY))
    stale = sorted(set(REGISTRY) - set(FROZEN_LEAVES))
    assert not missing, f"frozen leaves without a check: {missing}"
    assert not stale, f"checks for paths that are not frozen leaves: {stale}"


@pytest.mark.parametrize("path", sorted(FROZEN_LEAVES))
def test_frozen_leaf_matches_spec(path):
    assert failures(path, FROZEN_LEAVES[path]) == []


def _mutate(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value * 1.5 + 0.1
    if isinstance(value, str):
        return value + "_mutated"
    if isinstance(value, list):
        return value + [copy.deepcopy(value[-1])]
    raise TypeError(f"no mutation for {type(value).__name__}")


@pytest.mark.parametrize("path", sorted(FROZEN_LEAVES))
def test_changing_a_frozen_value_fails_verification(path):
    mutated = _mutate(FROZEN_LEAVES[path])
    assert mutated != FROZEN_LEAVES[path]
    assert failures(path, mutated), f"mutating {path} to {mutated!r} was not detected"


LIST_LEAVES = sorted(p for p, v in FROZEN_LEAVES.items() if isinstance(v, list))


def _mutate_element(value: Any) -> Any:
    """Change one list element without changing its type's shape (nested lists keep their length)."""
    if isinstance(value, list):
        return [_mutate(value[0])] + value[1:]
    return _mutate(value)


def test_list_leaf_count_is_as_documented():
    assert len(LIST_LEAVES) == 28


@pytest.mark.parametrize("path", LIST_LEAVES)
def test_changing_one_list_element_keeps_length_and_fails_verification(path):
    """Mutations above lengthen a list; these keep the length and alter each element in turn."""
    original = FROZEN_LEAVES[path]
    for index in range(len(original)):
        mutated = copy.deepcopy(original)
        mutated[index] = _mutate_element(original[index])
        assert len(mutated) == len(original) and mutated != original
        assert failures(path, mutated), f"changing element {index} of {path} to {mutated[index]!r} was not detected"


@pytest.mark.parametrize("path, value", [
    ("injection.d_max_m", 2.5),                                  # a numeric constant
    ("injection.grids.D.levels", [0.0, 0.05, 0.10, 0.15, 0.20]),  # a grid with one level removed
    ("predictors.learned.attention", True),                      # an assertion-kind boolean
    ("protocol.static.split", "random_windows"),                 # a normalised text label
])
def test_realistic_edits_fail_verification(path, value):
    assert failures(path, value)


def coverage_summary() -> dict[str, int]:
    """Leaf counts by the strongest kind of check applied (text beats assertion)."""
    summary = {"leaves": len(FROZEN_LEAVES), "text": 0, "assertion_only": 0}
    for path in FROZEN_LEAVES:
        kinds = {c.kind for c in REGISTRY.get(path, [])}
        summary["text" if "text" in kinds else "assertion_only"] += 1
    return summary
