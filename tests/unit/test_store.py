"""Phase 4: atomic writes, the run store and checkpoint safety."""

import dataclasses
import json
import os

import pytest

from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.experiments import store as st
from trackvision.experiments.store import (BLOCKED, COMPLETE, FAILED, IncompatibleRunError, RunStore, StaleRecordError,
                                           UNREALIZED, identity_key)
from trackvision.io.atomic import atomic_write_bytes, atomic_write_text
from trackvision.predict import training
from trackvision.predict.folds import nested_split
from trackvision.predict.training import CheckpointError, TrainingSettings, load_or_train, train_predictor

CFG = config.validate().config
IDENTITY = {"experiment": "P-4", "scene": "S1", "predictor": "CV", "condition": "A", "requested_level": 0.1, "seed": 0,
            "config_hash": "c" * 64}
SPEC = {"scope": "synthetic_engineering", "config_hash": "c" * 64, "seeds": [0, 1, 2]}


def _store(tmp_path):
    return RunStore.create(tmp_path / "run", SPEC)


# --- atomic writes -------------------------------------------------------------------------------

def test_atomic_write_replaces_the_whole_file(tmp_path):
    path = tmp_path / "a" / "b.txt"
    atomic_write_text(path, "first")
    atomic_write_text(path, "second")
    assert path.read_text(encoding="utf-8") == "second" and [p.name for p in path.parent.iterdir()] == ["b.txt"]


def test_interrupted_write_keeps_the_old_file_and_leaves_nothing_behind(tmp_path, monkeypatch):
    path = tmp_path / "b.txt"
    atomic_write_bytes(path, b"old")

    def crash(src, dst):
        raise KeyboardInterrupt
    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(path, b"new")
    assert path.read_bytes() == b"old" and [p.name for p in tmp_path.iterdir()] == ["b.txt"]


# --- creation and retrieval ----------------------------------------------------------------------

def test_a_record_reads_back_exactly(tmp_path):
    store = _store(tmp_path)
    result = {"metrics": {"ade_m": 0.5, "lost": (1, 2)}, "undefined": None}
    store.put(IDENTITY, COMPLETE, result, provenance={"note": "x"})
    record = store.get(IDENTITY)
    assert record.status == COMPLETE and record.result == {"metrics": {"ade_m": 0.5, "lost": [1, 2]}, "undefined": None}
    assert record.identity == IDENTITY and record.provenance["note"] == "x" and "written_utc" in record.provenance


def test_missing_record_is_none_and_unknown_status_is_refused(tmp_path):
    store = _store(tmp_path)
    assert store.get(IDENTITY) is None
    with pytest.raises(ValueError, match="status"):
        store.put(IDENTITY, "done")


def test_identity_is_independent_of_key_order_and_sensitive_to_every_field():
    assert identity_key(IDENTITY) == identity_key(dict(reversed(list(IDENTITY.items()))))
    for name, value in (("seed", 1), ("requested_level", 0.2), ("predictor", "CA"), ("config_hash", "d" * 64)):
        assert identity_key({**IDENTITY, name: value}) != identity_key(IDENTITY)


def test_statuses_are_kept_apart(tmp_path):
    store = _store(tmp_path)
    for i, status in enumerate((COMPLETE, UNREALIZED, FAILED, BLOCKED)):
        store.put({**IDENTITY, "seed": i}, status)
    summary = store.summary()
    assert summary["cells"] == 4 and set(summary["by_status"].values()) == {1}


def test_failure_details_are_preserved(tmp_path):
    store = _store(tmp_path)
    try:
        raise ValueError("boom")
    except ValueError as exc:
        store.put(IDENTITY, FAILED, failure=st.failure_info(exc))
    failure = store.get(IDENTITY).failure
    assert failure["type"] == "ValueError" and failure["message"] == "boom" and "raise ValueError" in failure["traceback"]


# --- rejection of stale or damaged records -------------------------------------------------------

def test_edited_content_is_rejected(tmp_path):
    store = _store(tmp_path)
    store.put(IDENTITY, COMPLETE, {"ade_m": 0.5})
    path = store.path(IDENTITY)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["result"]["ade_m"] = 0.1
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(StaleRecordError, match="digest"):
        store.get(IDENTITY)


def test_a_record_moved_under_another_identity_is_rejected(tmp_path):
    """A matching filename is not enough: the stored identity must equal the requested one."""
    store = _store(tmp_path)
    other = {**IDENTITY, "seed": 1}
    store.put(IDENTITY, COMPLETE, {"ade_m": 0.5})
    target = store.path(other)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(store.path(IDENTITY).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(StaleRecordError, match="identity"):
        store.get(other)
    with pytest.raises(StaleRecordError, match="filename"):
        list(store.records())


def test_truncated_record_is_rejected(tmp_path):
    store = _store(tmp_path)
    store.put(IDENTITY, COMPLETE, {"ade_m": 0.5})
    path = store.path(IDENTITY)
    path.write_text(path.read_text(encoding="utf-8")[:20], encoding="utf-8")
    with pytest.raises(StaleRecordError, match="JSON"):
        store.get(IDENTITY)


def test_nan_results_cannot_be_stored(tmp_path):
    with pytest.raises(ValueError):
        _store(tmp_path).put(IDENTITY, COMPLETE, {"ade_m": float("nan")})


# --- the run -------------------------------------------------------------------------------------

def test_a_run_is_never_overwritten_and_resume_needs_the_same_specification(tmp_path):
    root = tmp_path / "run"
    RunStore.create(root, SPEC)
    with pytest.raises(FileExistsError):
        RunStore.create(root, SPEC)
    assert RunStore.create(root, SPEC, resume=True).manifest()["spec"] == SPEC
    with pytest.raises(IncompatibleRunError, match="seeds"):
        RunStore.create(root, {**SPEC, "seeds": [0]}, resume=True)


def test_resume_needs_a_run_and_a_non_empty_folder_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunStore.create(tmp_path / "none", SPEC, resume=True)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RunStore.create(tmp_path / "busy", SPEC)


def test_manifest_records_provenance(tmp_path):
    manifest = _store(tmp_path).manifest()
    assert manifest["spec_sha256"] == identity_key(SPEC) and {"created_utc", "git", "environment"} <= set(manifest)


def test_source_digest_is_stable():
    assert st.source_sha256() == st.source_sha256() and len(st.source_sha256()) == 64


# --- checkpoint safety ---------------------------------------------------------------------------

SCENES = ss.make_scenes(4)
WINDOWS = ss.windows_of(SCENES, CFG)
SPLIT = nested_split(WINDOWS, "S1", ss.cyclic_rotation(w.scene for w in WINDOWS))
FAST = dataclasses.replace(TrainingSettings.from_config(CFG), max_epochs=2)


def test_load_or_train_reuses_an_identical_checkpoint(tmp_path):
    first, reused = load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    assert reused is False
    second, reused = load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    assert reused is True and second.checkpoint_sha256 == first.checkpoint_sha256


@pytest.mark.parametrize("request_args, field", [
    ({"settings": dataclasses.replace(FAST, max_epochs=3)}, "settings"),
    ({"split": nested_split(WINDOWS, "S2", ss.cyclic_rotation(w.scene for w in WINDOWS))}, "split"),
])
def test_a_checkpoint_from_a_different_setup_is_refused(tmp_path, request_args, field):
    load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    args = {"split": SPLIT, "settings": FAST, **request_args}
    with pytest.raises(CheckpointError, match=field):
        load_or_train(args["split"], SCENES, "LSTM", 0, CFG, tmp_path, "test", args["settings"])


def test_a_checkpoint_from_a_different_seed_or_config_is_detected(tmp_path):
    r = train_predictor(SPLIT, SCENES, "GRU", 0, CFG, tmp_path, "t", FAST)
    assert training.checkpoint_mismatches(r.metadata, SPLIT, "GRU", 0, CFG, FAST) == []
    assert training.checkpoint_mismatches(r.metadata, SPLIT, "GRU", 1, CFG, FAST) == ["seed"]
    assert "cell" in training.checkpoint_mismatches(r.metadata, SPLIT, "LSTM", 0, CFG, FAST)
    other = dataclasses.replace(CFG, hash="0" * 64)
    assert training.checkpoint_mismatches(r.metadata, SPLIT, "GRU", 0, other, FAST) == ["config_hash"]


def test_a_history_mismatch_is_detected(tmp_path):
    r = train_predictor(SPLIT, SCENES, "GRU", 0, CFG, tmp_path, "t", FAST)
    meta = json.loads(json.dumps(r.metadata))
    meta["model"]["architecture"]["history_len"] = 4
    assert "history_len" in training.checkpoint_mismatches(meta, SPLIT, "GRU", 0, CFG, FAST)


def test_an_interrupted_checkpoint_is_never_treated_as_complete(tmp_path):
    """A checkpoint with no sidecar did not finish: it is replaced, not loaded."""
    first, _ = load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    first.checkpoint_path.with_suffix(".pt.json").unlink()
    first.checkpoint_path.write_bytes(b"partial")
    again, reused = load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    assert reused is False and again.checkpoint_sha256 == first.checkpoint_sha256


def test_a_damaged_checkpoint_or_missing_file_is_refused(tmp_path):
    first, _ = load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    first.checkpoint_path.write_bytes(b"damaged")
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
    first.checkpoint_path.unlink()
    with pytest.raises(CheckpointError, match="missing"):
        load_or_train(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path, "test", FAST)
