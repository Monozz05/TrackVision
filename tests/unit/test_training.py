"""P2-e: deterministic training, checkpoints and leakage enforcement for the learned predictors.

Scenes are small synthetic collections (fixtures/synthetic_split.py) with a separate fixture rotation; the
real-scene rotation is never used for them. Training runs here are cut to a few epochs for speed; the approved
settings themselves are checked separately. Nothing here is a research result.
"""

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.experiments import learned_feasibility as lf
from trackvision.metrics.displacement import ade_fde, displacement_errors
from trackvision.predict import learned, training
from trackvision.predict.displacements import future_displacements
from trackvision.predict.folds import SplitError, configured_rotation, nested_split
from trackvision.predict.leakage import CHECKS, NORMALIZATION_NOT_USED, LeakageError
from trackvision.predict.training import (CheckpointError, EarlyStopping, TrainingError, TrainingSettings,
                                          load_checkpoint, state_sha256, train_predictor)

CFG = config.validate().config
TESTS_DIR = Path(__file__).resolve().parents[1]
SCENES = ss.make_scenes(4)
WINDOWS = ss.windows_of(SCENES, CFG)
SPLIT = nested_split(WINDOWS, "S1", ss.cyclic_rotation(w.scene for w in WINDOWS))     # test S1, val S2
APPROVED = TrainingSettings.from_config(CFG)
FAST = dataclasses.replace(APPROVED, max_epochs=12)
SCOPE = "unit test (synthetic fixture)"


def _train(tmp_path, cell="LSTM", seed=0, split=SPLIT, scenes=SCENES, settings=FAST, sub="ckpt"):
    return train_predictor(split, scenes, cell, seed, CFG, tmp_path / sub, SCOPE, settings)


# --- settings and early stopping ---------------------------------------------------------------

def test_approved_settings():
    assert APPROVED == TrainingSettings(learning_rate=1e-3, batch_size=64, max_epochs=200,
                                        early_stopping_patience=20, grad_clip_norm=1.0,
                                        lr_factor=0.5, lr_patience=10, min_lr=1e-5)
    assert training.PURPOSE_INIT not in (1, 2, 3, 4) and training.PURPOSE_SHUFFLE not in (1, 2, 3, 4)


@pytest.mark.parametrize("change", [{"batch_size": 0}, {"max_epochs": 2.5}, {"learning_rate": 0.0},
                                    {"grad_clip_norm": float("nan")}, {"lr_factor": 1.0}, {"min_lr": -1.0}])
def test_invalid_settings_raise(change):
    with pytest.raises(TrainingError):
        dataclasses.replace(APPROVED, **change)


def test_early_stopping_keeps_the_earlier_epoch_on_ties_and_stops_after_patience():
    stop = EarlyStopping(patience=3)
    improved = [stop.update(e, v) for e, v in enumerate([5.0, 4.0, 4.0, 3.0, 3.0, 3.5, 3.0], start=1)]
    assert improved == [True, True, False, True, False, False, False]
    assert (stop.best_epoch, stop.best_value) == (4, 3.0) and stop.should_stop


# --- a training run ----------------------------------------------------------------------------

@pytest.mark.parametrize("cell", learned.CELLS)
def test_training_run_is_finite_selects_the_best_epoch_and_records_everything(tmp_path, cell):
    r = _train(tmp_path, cell)
    h, m = r.history, r.metadata
    assert len(h) == m["epochs_run"] == FAST.max_epochs and m["stop_reason"] == "max_epochs"
    assert all(np.isfinite([e["train_mse"], e["validation_ade_m"], e["lr"]]).all() for e in h)
    assert h[-1]["train_mse"] < h[0]["train_mse"]
    vals = [e["validation_ade_m"] for e in h]
    assert m["best_epoch"] == 1 + int(np.argmin(vals)) and m["best_validation_ade_m"] == min(vals)
    # the returned model is the best epoch: recomputing validation ADE reproduces the recorded value
    ade, _ = ade_fde(displacement_errors(learned.predict_positions(r.model, SPLIT.validation.observed),
                                         SPLIT.validation.target))
    assert float(ade.mean()) == m["best_validation_ade_m"]
    assert state_sha256(r.model.state_dict()) == m["state_sha256"] != m["init_state_sha256"]
    assert m["model"]["type"] == cell and m["model"]["architecture"]["n_parameters"] == learned.parameter_count(r.model)
    assert m["seed"] == 0 and m["config_hash"] == CFG.hash and m["scope"] == SCOPE
    assert (m["split"]["test_scene"], m["split"]["validation_scene"], m["split"]["train_scenes"]) == ("S1", "S2", ["S3", "S4"])
    assert m["environment"]["torch"] == str(torch.__version__) and m["environment"]["torch_cuda"] is None
    assert [c["name"] for c in m["leakage_checks"]] == list(CHECKS) and all(c["passed"] for c in m["leakage_checks"])
    assert m["normalization"] == NORMALIZATION_NOT_USED and m["dtype"] == {"training": "float32", "metrics": "float64"}
    assert r.checkpoint_sha256 == training.file_sha256(r.checkpoint_path)
    sidecar = json.loads(r.checkpoint_path.with_suffix(".pt.json").read_text(encoding="utf-8"))
    assert sidecar["checkpoint_sha256"] == r.checkpoint_sha256 and sidecar["state_sha256"] == m["state_sha256"]


def test_early_stopping_in_a_real_run_respects_patience(tmp_path):
    settings = dataclasses.replace(APPROVED, max_epochs=60, early_stopping_patience=2)
    m = _train(tmp_path, "GRU", settings=settings).metadata
    if m["stop_reason"] == "early_stopping":
        assert m["epochs_run"] - m["best_epoch"] == 2
    else:
        assert m["epochs_run"] == 60


# --- checkpoints -------------------------------------------------------------------------------

@pytest.mark.parametrize("cell", learned.CELLS)
def test_checkpoint_reload_reproduces_predictions(tmp_path, cell):
    r = _train(tmp_path, cell)
    model, meta = load_checkpoint(r.checkpoint_path, r.checkpoint_sha256)
    for w in (SPLIT.test, SPLIT.validation):
        np.testing.assert_array_equal(learned.predict_positions(model, w.observed),
                                      learned.predict_positions(r.model, w.observed))
    assert meta == r.metadata


def test_checkpoint_hash_and_state_are_verified(tmp_path):
    r = _train(tmp_path)
    with pytest.raises(CheckpointError, match="does not match the expected"):
        load_checkpoint(r.checkpoint_path, "0" * 64)
    payload = torch.load(r.checkpoint_path, weights_only=True)
    first = sorted(payload["state_dict"])[0]
    payload["state_dict"][first] = payload["state_dict"][first] + 1.0          # altered weights
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    with pytest.raises(CheckpointError, match="recorded state hash"):
        load_checkpoint(forged, training.file_sha256(forged))


def test_an_existing_checkpoint_is_never_overwritten(tmp_path):
    _train(tmp_path)
    with pytest.raises(TrainingError, match="overwrite"):
        _train(tmp_path)


# --- determinism and seeds ---------------------------------------------------------------------

CHILD = """
import json, sys
sys.path.insert(0, {tests!r})
from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.predict import training
from trackvision.predict.folds import nested_split
cfg = config.validate().config
scenes = ss.make_scenes(4); wins = ss.windows_of(scenes, cfg)
split = nested_split(wins, "S1", ss.cyclic_rotation(w.scene for w in wins))
settings = training.TrainingSettings(**{{**training.TrainingSettings.from_config(cfg).__dict__, "max_epochs": 12}})
out = {{}}
for cell in ("LSTM", "GRU"):
    r = training.train_predictor(split, scenes, cell, 0, cfg, sys.argv[1], {scope!r}, settings)
    out[cell] = {{"state": r.metadata["state_sha256"], "file": r.checkpoint_sha256, "history": r.history}}
print(json.dumps(out))
"""


def test_training_is_identical_across_fresh_processes(tmp_path):
    code = CHILD.format(tests=str(TESTS_DIR), scope=SCOPE)
    runs = []
    for name in ("first", "second"):
        done = subprocess.run([sys.executable, "-c", code, str(tmp_path / name)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert runs[0] == runs[1]                               # weights, checkpoint bytes and full curves
    for cell in learned.CELLS:
        here = _train(tmp_path, cell, sub=f"here-{cell}")
        assert here.metadata["state_sha256"] == runs[0][cell]["state"]
        assert here.checkpoint_sha256 == runs[0][cell]["file"] and here.history == runs[0][cell]["history"]


@pytest.mark.parametrize("cell", learned.CELLS)
def test_different_seeds_give_different_initialisations_and_streams(tmp_path, cell):
    metas = [_train(tmp_path, cell, seed=s, sub=f"s{s}").metadata for s in (0, 1, 2)]
    for key in ("init_state_sha256", "state_sha256"):
        assert len({m[key] for m in metas}) == 3
    streams = [(m["purpose_seeds"]["init"], m["purpose_seeds"]["shuffle"]) for m in metas]
    assert len({s for pair in streams for s in pair}) == 6                     # independent per seed and purpose


# --- the test partition never influences training ----------------------------------------------

@pytest.mark.parametrize("cell", learned.CELLS)
def test_test_scene_content_does_not_affect_training_or_selection(tmp_path, cell):
    other_s1 = ss.make_scene("S1", 9, lengths=(28, 21))                       # entirely different test scene
    scenes = [other_s1, *SCENES[1:]]
    windows = ss.windows_of(scenes, CFG)
    swapped = nested_split(windows, "S1", ss.cyclic_rotation(w.scene for w in windows))
    assert not np.array_equal(swapped.test.target[:3], SPLIT.test.target[:3])
    a = _train(tmp_path, cell, sub="original").metadata
    b = _train(tmp_path, cell, split=swapped, scenes=scenes, sub="swapped").metadata
    for key in ("state_sha256", "init_state_sha256", "best_epoch", "best_validation_ade_m", "history"):
        assert a[key] == b[key], key
    assert a["split"]["scenes"]["S1"] != b["split"]["scenes"]["S1"]


# --- leakage enforcement -----------------------------------------------------------------------

def test_a_leaky_split_aborts_before_anything_is_written(tmp_path):
    leaky = dataclasses.replace(SPLIT, train=(SPLIT.train[0], SPLIT.test))
    with pytest.raises(LeakageError, match="scene_disjointness"):
        _train(tmp_path, split=leaky)
    assert not (tmp_path / "ckpt").exists()


def test_missing_source_scenes_abort(tmp_path):
    with pytest.raises(LeakageError, match="window_boundary"):
        _train(tmp_path, scenes=SCENES[1:])
    assert not (tmp_path / "ckpt").exists()


def test_all_six_checks_run_on_the_actual_training_inputs(tmp_path, monkeypatch):
    seen = {}
    real_run, real_require = training.run_leakage_checks, training.require_no_leakage

    def spy_run(split, scenes, make_inputs, fit, stats):
        seen.update(split=split, make_inputs=make_inputs, fit=fit, stats=stats)
        return real_run(split, scenes, make_inputs, fit, stats)

    def spy_require(results):
        seen["names"] = [r.name for r in results]
        return real_require(results)
    monkeypatch.setattr(training, "run_leakage_checks", spy_run)
    monkeypatch.setattr(training, "require_no_leakage", spy_require)
    _train(tmp_path)
    assert seen["split"] is SPLIT and seen["make_inputs"] is learned.window_inputs
    assert seen["fit"] is None and seen["stats"] == NORMALIZATION_NOT_USED
    assert seen["names"] == list(CHECKS)


def test_an_input_function_that_reads_targets_is_caught(tmp_path, monkeypatch):
    def peeking(w):
        x = learned.encoder_inputs(w.observed)
        x[:, -1] = future_displacements(w.observed, w.target)[:, 0]          # leaks the first target step
        return x
    monkeypatch.setattr(learned, "window_inputs", peeking)
    with pytest.raises(LeakageError, match="target_purity"):
        _train(tmp_path)
    assert not (tmp_path / "ckpt").exists()


def test_a_dropped_check_blocks_training(tmp_path, monkeypatch):
    real_run = training.run_leakage_checks
    monkeypatch.setattr(training, "run_leakage_checks", lambda *a: real_run(*a)[:-1])
    with pytest.raises(LeakageError, match="expected the checks"):
        _train(tmp_path)


# --- invalid or incomplete training data -------------------------------------------------------

def _empty(scene):
    return dataclasses.replace(SPLIT.validation, scene=scene, observed=np.zeros((0, 8, 2)), target=np.zeros((0, 12, 2)),
                               track_ids=np.zeros(0, np.int64), frames=np.zeros((0, 20), np.int64))


@pytest.mark.parametrize("make, fragment", [
    (lambda: dataclasses.replace(SPLIT, validation=_empty("S2")), "validation partition has no windows"),
    (lambda: dataclasses.replace(SPLIT, train=(_empty("S3"), _empty("S4"))), "training partition has no windows"),
    (lambda: dataclasses.replace(SPLIT, validation=dataclasses.replace(
        SPLIT.validation, observed=np.where(np.ones_like(SPLIT.validation.observed, bool), np.nan, 0.0))), "non-finite"),
    (lambda: dataclasses.replace(SPLIT, validation=dataclasses.replace(SPLIT.validation, horizon_len=8)),
     "different (history_len, horizon_len)"),
    (lambda: "not a split", "expected a NestedSplit"),
])
def test_invalid_training_data_raises(tmp_path, make, fragment):
    with pytest.raises(TrainingError, match=fragment.replace("(", r"\(").replace(")", r"\)")):
        _train(tmp_path, split=make())
    assert not (tmp_path / "ckpt").exists()


@pytest.mark.parametrize("kwargs", [{"cell": "RNN"}, {"seed": -1}, {"seed": 1.0}])
def test_invalid_arguments_raise(tmp_path, kwargs):
    with pytest.raises(TrainingError):
        _train(tmp_path, **kwargs)
    with pytest.raises(TrainingError, match="scope"):
        train_predictor(SPLIT, SCENES, "LSTM", 0, CFG, tmp_path / "x", " ", FAST)


# --- synthetic engineering runner --------------------------------------------------------------

def test_engineering_runner_uses_synthetic_scenes_and_a_separate_fixture_rotation(tmp_path):
    names = [s.name for s in lf.synthetic_scenes(CFG)]
    assert names == [f"SYN-{s}" for s in lf.SCENE_SEEDS]
    assert not set(names) & set(CFG.frozen["protocol"]["static"]["scenes"])
    assert lf.fixture_rotation(names) != configured_rotation(CFG)
    with pytest.raises(SplitError):
        nested_split([w for w in (ss.windows_of(lf.synthetic_scenes(CFG), CFG))], lf.TEST_SCENE, configured_rotation(CFG))
    out = tmp_path / "run"
    res = lf.run(out, CFG, dataclasses.replace(APPROVED, max_epochs=2))
    assert res["evaluation_type"] == "synthetic_engineering_feasibility" and "SYNTHETIC ENGINEERING" in res["label"]
    assert len(res["runs"]) == 2 * len(CFG.research_choice("seeds"))
    assert res["split"]["test_scene"] == lf.TEST_SCENE
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert all(training.file_sha256(out / f) == h for f, h in manifest["files"].items())
    with pytest.raises(FileExistsError):
        lf.run(out, CFG)
    with pytest.raises(ValueError, match="inside the repository"):
        lf.run(config.REPO_ROOT / "src" / "p2e-run", CFG)
