"""The experiment record schema (spec v4.15 section 11.3).

The field lists in trackvision.experiments.record are checked against the section 11.3 block
parsed from the specification itself, so a transcription error or a later edit on either side
fails here.
"""

import json
import math
import re

import pytest

from conftest import SPEC_PATH
from trackvision import config
from trackvision.experiments import record as rec

CFG = config.validate().config
SPEC = SPEC_PATH.read_text(encoding="utf-8")


# --- parsing the section 11.3 block ------------------------------------------------------------

def _spec_block() -> list[str]:
    match = re.search(r"## 11\.3 Experiment record schema\s*```\n(.*?)\n```", SPEC, re.S)
    assert match, "section 11.3 schema block not found"
    return match.group(1).splitlines()


def _field_tokens(line: str, separator: str) -> list[str]:
    line = re.split(r"\s{2,}", line.strip())[0]            # drop a trailing description column
    return [t.strip() for t in line.split(separator) if t.strip()]


def parse_spec_schema() -> dict:
    top, permitted, mapping_keys, groups, result_fields = [], {}, {}, {}, []
    section, group = None, None

    def add_top(token: str) -> None:
        token = re.sub(r"\s*\(§[\d.]+\)$", "", token)           # "tracker_params (§11.4)"
        braces = re.fullmatch(r"(\w+) \{(.*)\}", token)
        if braces:
            name, inner = braces.groups()
            if "|" in inner:
                permitted[name] = tuple(v.strip() for v in inner.split("|"))
            else:
                mapping_keys[name] = tuple(v.strip() for v in inner.split(","))
            top.append(name)
        elif token.endswith(" + version"):
            name = token[: -len(" + version")]
            top.extend([name, f"{name}_version"])
        else:
            assert re.fullmatch(r"\w+", token), f"unparsed token {token!r}"
            top.append(token)

    for line in _spec_block():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0 and re.fullmatch(r"[A-Z ]+", line):
            section, group = line, None
        elif indent == 0:
            section = None
            for token in _field_tokens(line, "·"):
                add_top(token)
        elif section == "RESULTS" and indent == 2 and line.rstrip().endswith(":"):
            group = re.match(r"\s*(\w+)", line).group(1)            # "interaction (H8):" -> interaction
            groups[group] = []
        elif section == "RESULTS" and indent == 2:
            group = None
            result_fields.extend(_field_tokens(line, ","))
        elif section == "RESULTS" and indent == 4:
            groups[group].extend(_field_tokens(line, ","))
        else:
            for token in _field_tokens(line, "·"):
                add_top(token)
    return {"top": top, "permitted": permitted, "mapping_keys": mapping_keys,
            "groups": {g: tuple(f) for g, f in groups.items()}, "result_fields": tuple(result_fields)}


SPEC_SCHEMA = parse_spec_schema()


def test_top_level_fields_match_the_specification():
    assert len(SPEC_SCHEMA["top"]) == len(set(SPEC_SCHEMA["top"])), "the spec repeats a top-level field"
    assert rec.TOP_LEVEL_FIELDS == tuple(SPEC_SCHEMA["top"])


def test_permitted_values_match_the_specification():
    expected = {name: tuple({"true": True, "false": False}.get(v, v) for v in values)
                for name, values in SPEC_SCHEMA["permitted"].items()}
    assert rec.PERMITTED_VALUES == expected


def test_mapping_keys_match_the_specification():
    assert rec.MAPPING_KEYS == SPEC_SCHEMA["mapping_keys"]


def test_result_groups_and_fields_match_the_specification():
    assert rec.RESULT_GROUPS == SPEC_SCHEMA["groups"]
    assert rec.RESULT_FIELDS == SPEC_SCHEMA["result_fields"]


def test_results_must_be_nested_because_names_repeat():
    """The reason `results` is an object rather than flattened into the top level."""
    nested_names = {f for fields in rec.RESULT_GROUPS.values() for f in fields} | set(rec.RESULT_GROUPS)
    assert {"mitigation", "error_requested", "error_achieved"} <= nested_names & set(rec.TOP_LEVEL_FIELDS)


def test_ground_truth_dependent_terms_match_the_specification():
    sentence = re.search(r"no ground-truth-dependent evaluation metric may be populated\*\* — ([^.]+)\.", SPEC)
    assert sentence, "ground-truth sentence of section 11.3 not found"
    terms = [t.strip() for t in sentence.group(1).split(",")]
    assert list(rec.GROUND_TRUTH_DEPENDENT) == terms
    for fields in rec.GROUND_TRUTH_DEPENDENT.values():
        for group, name in fields:
            assert name in (rec.RESULT_FIELDS if group is None else rec.RESULT_GROUPS[group])


# --- validation --------------------------------------------------------------------------------

def _record(**overrides) -> dict:
    record = rec.new_record("exp-0123456789-s1", "2026-10-29T00:00:00+00:00", "a" * 64, "valid")
    record.update(overrides)
    return record


def test_new_record_is_complete_and_valid():
    record = _record()
    assert rec.validate_record(record, CFG) == []
    assert set(record) == set(rec.TOP_LEVEL_FIELDS) | {"results"}


@pytest.mark.parametrize("name", rec.REQUIRED)
def test_required_fields(name):
    assert any(e.startswith(f"{name}:") for e in rec.validate_record(_record(**{name: None}), CFG))


def test_missing_and_unknown_keys_are_rejected():
    record = _record()
    del record["scene"]
    record["extra"] = 1
    errors = rec.validate_record(record, CFG)
    assert any("missing keys ['scene']" in e for e in errors)
    assert any("unknown keys ['extra']" in e for e in errors)
    nested = _record()
    nested["results"]["summary"]["extra"] = 1
    assert any(e.startswith("results.summary") for e in rec.validate_record(nested, CFG))


@pytest.mark.parametrize("name, value", [("origin", "manual"), ("status", "pending"), ("error_mode", "F"),
                                         ("ground_truth_available", 1), ("ground_truth_available", "true"),
                                         ("stabilizer", "STATIC")])
def test_values_outside_the_permitted_set_are_rejected(name, value):
    assert any(e.startswith(f"{name}:") for e in rec.validate_record(_record(**{name: value}), CFG))


def test_detector_and_tracker_params_have_the_specified_keys():
    good = {"imgsz": 640, "conf": None, "iou": None}
    tracker = {name: None for name in CFG.frozen["trackers"]["canonical_param_names"]}
    assert rec.validate_record(_record(detector_params=good, tracker_params=tracker), CFG) == []
    assert rec.validate_record(_record(detector_params={"imgsz": 640}), CFG)
    assert rec.validate_record(_record(tracker_params={**tracker, "unknown": 1}), CFG)


def test_per_step_errors_length_equals_horizon_len():
    record = _record(horizon_len=12)
    record["results"]["prediction"]["per_step_errors"] = [0.1] * 12
    assert rec.validate_record(record, CFG) == []
    record["results"]["prediction"]["per_step_errors"] = [0.1] * 8
    assert any("per_step_errors" in e for e in rec.validate_record(record, CFG))
    record = _record(horizon_len=8)                       # the moving fallback horizon is allowed
    record["results"]["prediction"]["per_step_errors"] = [0.1] * 8
    assert rec.validate_record(record, CFG) == []
    record = _record()                                    # no recorded horizon to check against
    record["results"]["prediction"]["per_step_errors"] = [0.1]
    assert rec.validate_record(record, CFG)


@pytest.mark.parametrize("bad", [[math.nan], [True], "0.1"])
def test_per_step_errors_must_be_finite_numbers(bad):
    record = _record(horizon_len=1)
    record["results"]["prediction"]["per_step_errors"] = bad
    assert rec.validate_record(record, CFG)


@pytest.mark.parametrize("term", list(rec.GROUND_TRUTH_DEPENDENT))
def test_ground_truth_dependent_metrics_must_be_null_without_ground_truth(term):
    for group, name in rec.GROUND_TRUTH_DEPENDENT[term]:
        record = _record(ground_truth_available=False)
        (record["results"] if group is None else record["results"][group])[name] = 0.5
        assert any("ground_truth_available is false" in e for e in rec.validate_record(record, CFG))
        record["ground_truth_available"] = True
        assert rec.validate_record(record, CFG) == []


def test_system_measurements_remain_allowed_without_ground_truth():
    """Spec 11.3: FPS and latency are system measurements, valid without ground truth; fields
    outside the listed terms are not restricted."""
    record = _record(ground_truth_available=False)
    record["results"].update(FPS=25.0, latency=0.04)
    record["results"]["prediction"]["per_step_errors"] = None
    assert rec.validate_record(record, CFG) == []


def test_invalid_records_need_a_reason_and_superseded_records_a_replacement():
    assert rec.validate_record(_record(status="invalid"), CFG)
    assert rec.validate_record(_record(status="invalid", invalid_reason="leakage check failed"), CFG) == []
    assert rec.validate_record(_record(status="superseded"), CFG)
    assert rec.validate_record(_record(status="superseded", superseded_by_experiment_id="exp-2"), CFG) == []


@pytest.mark.parametrize("name", ["history_len", "horizon_len"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_protocol_lengths_must_be_positive_integers(name, value):
    assert rec.validate_record(_record(**{name: value}), CFG)


def test_write_record_writes_canonical_json_and_refuses_invalid_records(tmp_path):
    path = tmp_path / "record.json"
    record = _record(horizon_len=12, history_len=8)
    rec.write_record(path, record, CFG)
    assert json.loads(path.read_text(encoding="utf-8")) == record
    assert path.read_text(encoding="utf-8") == config.canonical_json(record) + "\n"
    with pytest.raises(rec.RecordError):
        rec.write_record(tmp_path / "bad.json", _record(status="invalid"), CFG)
    nan = _record(fold=math.nan)
    with pytest.raises(rec.RecordError):
        rec.write_record(tmp_path / "nan.json", nan, CFG)
    assert not (tmp_path / "bad.json").exists() and not (tmp_path / "nan.json").exists()
