import copy

import pytest

from trackvision import config

from conftest import BASE_PATH, REPO_ROOT, SMOKE_PATH

ALL_RESEARCH_CHOICES = {
    "seeds", "inner_validation_rotation", "moving_dataset", "trackers_evaluated_set",
    "tracker_params", "h8_generation_attempts", "discontinuity_rejection_v_max_mps",
}


def _exp(name="probe", kind="smoke", uses=(), **selection):
    selection.setdefault("regime", "static")
    return {"experiment": {"name": name, "kind": kind,
                           "uses_research_choices": list(uses), "selection": selection}}


def _validate(write_yaml, base=None, exp=None):
    base_path = write_yaml(base) if base is not None else BASE_PATH
    exp_path = write_yaml(exp) if exp is not None else None
    return config.validate(exp_path, base_path=base_path, repo_root=REPO_ROOT)


# --- outcomes --------------------------------------------------------------------------------

def test_base_alone_is_valid_and_lists_every_unresolved_choice():
    result = config.validate()
    assert result.status == config.VALID, result.errors
    assert set(result.unresolved) == ALL_RESEARCH_CHOICES


def test_base_hash_is_stable_across_loads():
    assert config.validate().config.hash == config.validate().config.hash


def test_smoke_experiment_is_runnable():
    result = config.validate(SMOKE_PATH)
    assert result.status == config.RUNNABLE, result.errors
    assert result.unresolved_used == ()


def test_research_experiment_with_null_seeds_is_unresolved(write_yaml):
    result = _validate(write_yaml, exp=_exp(kind="research", uses=["seeds"]))
    assert result.status == config.UNRESOLVED
    assert result.unresolved_used == ("seeds",)


def test_research_experiment_becomes_runnable_once_seeds_are_set(write_yaml, base_dict):
    base_dict["research_choices"]["seeds"]["value"] = [0, 1, 2]
    result = _validate(write_yaml, base=base_dict, exp=_exp(kind="research", uses=["seeds"]))
    assert result.status == config.RUNNABLE, result.errors
    assert result.config.research_choice("seeds") == (0, 1, 2)


# --- INVALID rules ---------------------------------------------------------------------------

def _set(d, dotted, value):
    keys = dotted.split(".")
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


BASE_MUTATIONS = {
    "unknown key in frozen": ("frozen.protocol.static.extra_constant", 1),
    "unknown top-level block": ("experiment_defaults", {}),
    "wrong type": ("frozen.protocol.static.history_len", "8"),
    "non-positive horizon": ("frozen.protocol.static.horizon_len", 0),
    "boolean is not an integer": ("frozen.protocol.static.folds", True),
    "NaN in a grid": ("frozen.injection.grids.A.levels", [0.0, float("nan")]),
    "grid not starting at zero": ("frozen.injection.grids.B.levels", [0.05, 0.10]),
    "grid not strictly increasing": ("frozen.injection.grids.C.levels", [0.0, 0.10, 0.10]),
    "infinite tolerance": ("frozen.geometry.roundtrip_tolerance_px.float32", float("inf")),
    "spec hash mismatch": ("meta.spec.sha256", "0" * 64),
    "absolute spec path": ("meta.spec.path", str(REPO_ROOT / "docs" / "spec" / "x.md")),
    "condition with unknown grid": ("frozen.injection.conditions.A.grid", "Z"),
    "domain out of sync with conditions": ("domains.error_mode", ["none", "A"]),
    "unknown research choice": ("research_choices.learning_rate", {"value": None, "required_by": "x", "spec": "x"}),
    "too few seeds": ("research_choices.seeds.value", [0, 1]),
    "duplicate seeds": ("research_choices.seeds.value", [0, 0, 1]),
    "scene validates on itself": ("research_choices.inner_validation_rotation.value",
                                  {"ETH": "ETH", "HOTEL": "ETH", "UNIV": "ETH", "ZARA1": "ETH", "ZARA2": "ETH"}),
    "unknown moving dataset": ("research_choices.moving_dataset.value", "Waymo"),
    "evaluated trackers omit a primary tracker": ("research_choices.trackers_evaluated_set.value", ["ByteTrack"]),
    "non-canonical tracker parameter": ("research_choices.tracker_params.value", {"ByteTrack": {"thresh": 0.5}}),
}


@pytest.mark.parametrize("label", list(BASE_MUTATIONS))
def test_invalid_base_mutation(label, write_yaml, base_dict):
    dotted, value = BASE_MUTATIONS[label]
    _set(base_dict, dotted, value)
    result = _validate(write_yaml, base=base_dict)
    assert result.status == config.INVALID, label
    assert result.errors


def test_missing_research_choice_value_key_is_invalid(write_yaml, base_dict):
    del base_dict["research_choices"]["seeds"]["value"]
    assert _validate(write_yaml, base=base_dict).status == config.INVALID


def test_duplicate_key_in_base_is_invalid(write_yaml):
    text = BASE_PATH.read_text(encoding="utf-8")
    assert text.count("    min_seeds: 3") == 1
    result = _validate(write_yaml, base=text.replace("    min_seeds: 3", "    min_seeds: 3\n    min_seeds: 1"))
    assert result.status == config.INVALID
    assert "duplicate key 'min_seeds'" in result.errors[0]


def test_duplicate_key_in_experiment_is_invalid(write_yaml):
    text = "experiment:\n  name: a\n  name: b\n  kind: smoke\n  uses_research_choices: []\n  selection: {regime: static}\n"
    result = config.validate(write_yaml(text))
    assert result.status == config.INVALID
    assert "duplicate key 'name'" in result.errors[0]


EXPERIMENT_CASES = {
    "override of a frozen value": {**_exp(), "frozen": {"protocol": {"min_seeds": 1}}},
    "override of a research choice": {**_exp(), "research_choices": {"seeds": {"value": [0, 1, 2]}}},
    "unknown experiment key": {"experiment": {**_exp()["experiment"], "notes": "x"}},
    "unknown selection key": _exp(history_len=4),
    "missing regime": {"experiment": {**_exp()["experiment"], "selection": {}}},
    "unknown regime": _exp(regime="aerial"),
    "unknown error mode": _exp(error_mode="F"),
    "moving-only mode in static regime": _exp(error_mode="E1"),
    "levels outside the frozen grid": _exp(error_mode="A", severity_levels=[0.0, 0.15]),
    "levels without zero": _exp(error_mode="A", severity_levels=[0.05, 0.10]),
    "levels without error mode": _exp(severity_levels=[0.0, 0.05]),
    "unknown predictor": _exp(predictors=["Transformer"]),
    "unknown scene": _exp(scenes=["ETH", "Times Square"]),
    "scenes in moving regime": _exp(regime="moving", scenes=["ETH"]),
    "bad experiment name": _exp(name="Smoke Test"),
    "unknown kind": _exp(kind="pilot"),
    "smoke declaring a research choice": _exp(uses=["seeds"]),
    "research without seeds": _exp(kind="research", uses=["moving_dataset"]),
    "unknown research choice used": _exp(kind="research", uses=["seeds", "learning_rate"]),
}


@pytest.mark.parametrize("label", list(EXPERIMENT_CASES))
def test_invalid_experiment(label, write_yaml):
    result = _validate(write_yaml, exp=EXPERIMENT_CASES[label])
    assert result.status == config.INVALID, label
    assert result.errors


# --- permitted selections --------------------------------------------------------------------

def test_subset_of_frozen_grid_is_accepted_and_marked_incomplete(write_yaml, smoke_config):
    grid = list(smoke_config.frozen["injection"]["grids"]["A"]["levels"])
    result = _validate(write_yaml, exp=_exp(error_mode="A", severity_levels=[grid[0], grid[2]]))
    assert result.status == config.RUNNABLE, result.errors
    assert result.config.protocol_complete() is False


def test_full_frozen_grid_is_protocol_complete(write_yaml, smoke_config):
    grid = list(smoke_config.frozen["injection"]["grids"]["B"]["levels"])
    for mode in ("B1", "B2"):
        result = _validate(write_yaml, exp=_exp(error_mode=mode, severity_levels=grid))
        assert result.status == config.RUNNABLE, result.errors
        assert result.config.protocol_complete() is True


def test_no_error_mode_has_no_protocol_completeness(smoke_config):
    assert smoke_config.protocol_complete() is None


# --- research-choice guards and immutability -------------------------------------------------

def test_reading_an_unresolved_choice_raises():
    with pytest.raises(config.UnresolvedChoiceError):
        config.validate().config.research_choice("seeds")


def test_reading_an_undeclared_choice_raises_even_if_unresolved(smoke_config):
    with pytest.raises(config.UndeclaredChoiceError):
        smoke_config.research_choice("seeds")


def test_research_experiment_cannot_read_its_unresolved_choice(write_yaml):
    cfg = _validate(write_yaml, exp=_exp(kind="research", uses=["seeds"])).config
    with pytest.raises(config.UnresolvedChoiceError):
        cfg.research_choice("seeds")


def test_config_view_is_read_only(smoke_config):
    with pytest.raises(TypeError):
        smoke_config.frozen["protocol"]["static"]["history_len"] = 4
    with pytest.raises(AttributeError):
        smoke_config.frozen["protocol"]["static"]["scenes"].append("X")


def test_canonical_dict_is_a_copy(smoke_config):
    original = copy.deepcopy(smoke_config.canonical_dict())
    returned = smoke_config.canonical_dict()
    returned["frozen"]["protocol"]["min_seeds"] += 1
    assert smoke_config.canonical_dict() == original
    assert smoke_config.frozen["protocol"]["min_seeds"] == original["frozen"]["protocol"]["min_seeds"]
