"""Phase 4: the evaluator, the sweep runner, analysis, the CLI and figures, on tiny synthetic fixtures.

Training is cut to two epochs for speed; nothing here is a result. Scenes named ETH and HOTEL below are synthetic
fixtures used only to exercise that scope's rules.
"""

import copy
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import REPO_ROOT
from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.experiments import analysis, evaluation, injection_feasibility as inj, store as st
from trackvision.experiments import switch_feasibility as sf, sweep
from trackvision.experiments.store import COMPLETE, FAILED, IncompatibleRunError, RunStore, StaleRecordError
from trackvision import cli
from trackvision.experiments.synthetic_feasibility import evaluate_ground_truth_cv
from trackvision.inject.trajectory import dropout_masks
from trackvision.predict import leakage, training
from trackvision.predict.training import TrainingSettings
from trackvision.predict.windowing import make_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks
from trackvision.viz import curves

CFG = config.validate().config
STATIC = CFG.frozen["protocol"]["static"]
H, F, STRIDE, DT = STATIC["history_len"], STATIC["horizon_len"], STATIC["window_stride"], STATIC["sampling_interval_s"]
SCENES = ss.make_scenes(4)
FAST = dataclasses.replace(TrainingSettings.from_config(CFG), max_epochs=2)
SPEC = sweep.synthetic_spec(CFG, histories=(8, 12), settings=FAST)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep") / "run"
    status = sweep.run_sweep(out, SPEC, SCENES, CFG)
    return out, status, analysis.write_analysis(RunStore.open(out), CFG)


@pytest.fixture(scope="module")
def companion(tiny, tmp_path_factory):
    """The revised companion, adopting the H = 12 models of `tiny`."""
    source, _, _ = tiny
    out = tmp_path_factory.mktemp("companion") / "run"
    spec = dataclasses.replace(SPEC, common_training=True)
    status = sweep.run_sweep(out, spec, SCENES, CFG, reuse_from=source)
    return out, status, analysis.write_analysis(RunStore.open(out), CFG), spec


def _short_spec(**changes):
    base = sweep.synthetic_spec(CFG, histories=(8,), settings=FAST)
    return dataclasses.replace(base, predictors=("CV", "LSTM"), conditions=("A", "B2"), seeds=(0,), **changes)


# --- evaluator -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def u7():
    scene = generate(SceneConfig(seed=4, n_frames=300), DT)
    clean = evaluate_ground_truth_cv(scene, CFG)
    tracks = ground_truth_grid_tracks(scene)
    return tracks, clean, inj.population(tracks, clean.windows, clean.per_step_error), scene


@pytest.mark.parametrize("condition", ["A", "B1", "B2", "C"])
def test_evaluator_reproduces_the_accepted_u7_results_exactly(u7, condition):
    tracks, clean, pop, scene = u7
    cv = evaluation.cv_predictor(CFG, F)
    clean_err = evaluation.clean_errors(cv, clean.windows)
    for level in inj.levels(CFG, condition):
        ref = inj.evaluate_condition(tracks, pop, condition, level, 1, CFG)
        c = evaluation.corrupt(tracks, pop, condition, level, 1, CFG, scene.frames_per_step)
        np.testing.assert_array_equal(c.inputs, ref.inputs)
        np.testing.assert_array_equal(c.predicted, ref.predicted)
        assert c.value == ref.summary["achieved_primary"]["value"]
        out = evaluation.score(cv, c, clean.windows, clean_err)
        assert out["predicted_windows"] == ref.summary["coverage"]["predicted_windows"]
        assert out["ade_m"] == ref.summary["accuracy_on_predicted_windows"]["injected"]["mean_ade_m"]


def test_evaluator_reproduces_the_accepted_u8_results_exactly(u7):
    tracks, clean, pop, scene = u7
    cv = evaluation.cv_predictor(CFG, F)
    for level in inj.levels(CFG, "D"):
        ref = sf.evaluate_switch(tracks, pop, level, 1, CFG, scene.frames_per_step)
        c = evaluation.corrupt(tracks, pop, "D", level, 1, CFG, scene.frames_per_step)
        np.testing.assert_array_equal(c.inputs, ref.windows.inputs)
        assert c.value == ref.summary["achieved_primary"]["value"]
        assert evaluation.score(cv, c, clean.windows, evaluation.clean_errors(cv, clean.windows))["ade_m"] == (
            ref.summary["accuracy_on_predicted_windows"]["injected"]["mean_ade_m"])
        assert c.extra["windows_with_mixed_history"] == ref.summary["history"]["windows_with_mixed_history"]


def test_coverage_counts_and_undefined_metrics(u7):
    tracks, clean, pop, scene = u7
    cv = evaluation.cv_predictor(CFG, F)
    clean_err = evaluation.clean_errors(cv, clean.windows)
    c = evaluation.corrupt(tracks, pop, "B2", 1.0, 0, CFG, scene.frames_per_step)
    out = evaluation.score(cv, c, clean.windows, clean_err)
    n = len(clean.windows)
    assert (out["eligible_windows"], out["predicted_windows"], out["prediction_coverage"]) == (n, 0, 0.0)
    assert out["ade_m"] is None and out["de_by_step_m"] is None and out["undefined_reason"] == "no_predicted_windows"
    assert out["lost_by_reason"] == {"missing_observation": n}
    partial = evaluation.corrupt(tracks, pop, "B2", 0.2, 0, CFG, scene.frames_per_step)
    p = evaluation.score(cv, partial, clean.windows, clean_err)
    assert p["predicted_windows"] + sum(p["lost_by_reason"].values()) == n
    assert p["prediction_coverage"] == p["predicted_windows"] / n and len(p["de_by_step_m"]) == F


def test_paired_reference_uses_the_same_windows(u7):
    tracks, clean, pop, scene = u7
    cv = evaluation.cv_predictor(CFG, F)
    err = evaluation.clean_errors(cv, clean.windows)
    c = evaluation.corrupt(tracks, pop, "C", 0.05, 0, CFG, scene.frames_per_step)
    out = evaluation.score(cv, c, clean.windows, err)
    assert 0 < out["predicted_windows"] and out["ade_m"] == out["clean_same_windows"]["ade_m"]                 # C leaves surviving inputs unchanged
    assert out["ade_m"] == pytest.approx(err[c.predicted].mean(axis=1).mean(), abs=1e-12)


def test_predictor_interface_is_shared_and_rejects_unknown_conditions(u7):
    tracks, clean, pop, scene = u7
    for predictor in (evaluation.cv_predictor(CFG, F), evaluation.ca_predictor(CFG, F)):
        assert predictor.predict(clean.windows.observed).shape == (len(clean.windows), F, 2)
    with pytest.raises(ValueError, match="unknown condition"):
        evaluation.corrupt(tracks, pop, "E1", 0.1, 0, CFG, scene.frames_per_step)


# --- populations ---------------------------------------------------------------------------------

def test_common_population_is_nested_in_each_history():
    tracks, step = SCENES[0].tracks, SCENES[0].frame_step
    anchors = sweep.common_anchors(tracks, step, 12, F, STRIDE)
    for h in (4, 8, 12):
        w = make_windows("S1", tracks, step, h, F, STRIDE)
        common = sweep.restrict_windows(w, anchors)
        assert len(common) == len(anchors) and len(common) <= len(w)
        assert {(int(g), int(f[h - 1])) for g, f in zip(common.track_ids, common.frames)} == anchors
    assert len(make_windows("S1", tracks, step, 4, F, STRIDE)) > len(anchors)   # eligibility differs by history


def test_per_history_and_common_populations_are_reported_separately(tiny):
    _, _, doc = tiny
    primary = doc["clean"]["per_history"]["8"]["S1"]["CV"]["ade_m"]["mean"]
    common = doc["clean"]["common_h12"]["8"]["S1"]["CV"]["ade_m"]["mean"]
    assert set(doc["curves"]) == {"per_history", "common_h12"} and primary != common


def test_cv_cannot_depend_on_history_on_the_common_population(tiny):
    """CV reads the last two points, and the anchors are the same, so clean ADE is identical across histories."""
    _, _, doc = tiny
    c = doc["clean"]["common_h12"]
    assert c["8"]["S1"]["CV"]["ade_m"]["mean"] == c["12"]["S1"]["CV"]["ade_m"]["mean"]


def test_achieved_severity_does_not_depend_on_history(tiny):
    _, _, doc = tiny
    a = doc["curves"]["per_history"]["8"]["S1"]["CV"]["A"]["achieved"]
    assert a == doc["curves"]["per_history"]["12"]["S1"]["CV"]["A"]["achieved"]


def test_p15_reports_each_condition_for_each_population(tiny):
    _, _, doc = tiny
    assert set(doc["p15"]) == {"per_history", "common_h12"}
    for by_predictor in doc["p15"].values():
        assert set(by_predictor) == {"CV", "CA", "LSTM", "GRU"}
        for row in by_predictor.values():
            assert set(row) == {"A", "B1", "B2", "C", "D"}
            assert all(r["status"] in ("found", "trade_off", "uncomputable") for r in row.values())
            assert all(set(r["valid_scenes_by_history"]) == {"8", "12"} for r in row.values())


def _p15_across(valid_common):
    """Two populations, two histories, five folds. `valid_common` is how many folds of B2 are valid at H = 12."""
    def audc(n):
        return {"mean": 0.2, "sd": None, "n_valid": n, "n_undefined": 5 - n, "undefined": {"S1": "no_predicted_windows"} if n < 5 else {}}

    def history(n):
        clean = {"ade_m": {"mean": 1.0}, "fde_m": {"mean": 2.0}, "eligible_windows": {}}
        level = {"requested": 0.0, "prediction_coverage": {"mean": 1.0}, "windows_by_scene": {
            "S1": {"0": {"eligible": 10, "predicted": 7}, "1": {"eligible": 10, "predicted": None}}}}
        return {"CV": {"clean": clean, "B2": {"audc": audc(n), "audc_paired": audc(n), "levels": [level]}}}
    return {"per_history": {"8": history(5), "12": history(5)}, "common_h12": {"8": history(5), "12": history(valid_common)}}


@pytest.mark.parametrize("valid", [5, 4, 1])
def test_p15_rule_is_applied_to_each_population_independently(valid):
    spec_doc = {"histories": [8, 12], "scenes": ["S1", "S2", "S3", "S4", "S5"], "predictors": ["CV"], "conditions": ["B2"]}
    doc = analysis._p15(_p15_across(valid), spec_doc, CFG, ["per_history", "common_h12"])
    primary, common = doc["per_history"]["CV"]["B2"], doc["common_h12"]["CV"]["B2"]
    assert primary["partial_folds"] is False and primary["per_history"]["12"]["audc_qualifies"] is True
    assert common["partial_folds"] is (valid < 5) and common["per_history"]["12"]["audc_qualifies"] is (valid == 5)
    assert (common["status"], common["minimum_history"]) == ("found", 8)
    assert common["valid_scenes_by_history"]["12"] == valid and common["undefined_scenes_by_history"]["12"] == 5 - valid


def test_p15_keeps_the_undefined_reason_of_each_fold():
    spec_doc = {"histories": [8, 12], "scenes": ["S1", "S2", "S3", "S4", "S5"], "predictors": ["CV"], "conditions": ["B2"]}
    entry = analysis._p15(_p15_across(4), spec_doc, CFG, ["per_history", "common_h12"])["common_h12"]["CV"]["B2"]
    assert entry["undefined_reasons_by_history"]["12"] == {"S1": "no_predicted_windows"}


def test_p15_window_counts_are_never_invented_for_a_missing_cell():
    spec_doc = {"histories": [8, 12], "scenes": ["S1", "S2", "S3", "S4", "S5"], "predictors": ["CV"], "conditions": ["B2"]}
    entry = analysis._p15(_p15_across(5), spec_doc, CFG, ["per_history", "common_h12"])["per_history"]["CV"]["B2"]
    windows = entry["conditional_error"]["by_history"]["8"]["levels"][0]["windows"]
    assert windows["predicted_total"] == 7 and windows["eligible_total"] == 20
    assert windows["by_scene"]["S1"]["1"]["predicted"] is None


def test_the_evaluation_only_population_carries_its_label(tiny):
    _, _, doc = tiny
    assert doc["population_labels"]["common_h12"].startswith("Common evaluation population \u2014 training and validation "
                                                             "remain per-history.")
    assert "does not isolate history length for LSTM or GRU" in doc["population_labels"]["common_h12"]


def test_b2_and_c_report_conditional_error_evidence_and_others_do_not(tiny):
    _, _, doc = tiny
    for population, by_predictor in doc["p15"].items():
        for predictor, by_condition in by_predictor.items():
            assert all("conditional_error" not in by_condition[c] for c in ("A", "B1", "D"))
            for condition in ("B2", "C"):
                block = by_condition[condition]["conditional_error"]
                assert "does not establish history sufficiency" in block["note"]
                assert "not evidence of improved robustness" in block["note"]
                row = block["by_history"]["8"]
                assert {"audc", "audc_paired", "levels"} == set(row)
                assert {"n_valid", "n_undefined", "undefined"} <= set(row["audc"]) and len(row["levels"]) == 6
                assert {"prediction_coverage", "windows"} <= set(row["levels"][0])
                assert row["levels"][0]["windows"]["eligible_total"] >= row["levels"][-1]["windows"]["predicted_total"]


def test_no_coverage_threshold_decides_a_minimum_history(tiny):
    """Decision 0005 adds no window-count floor: coverage is reported beside the criteria, not used by them."""
    _, _, doc = tiny
    row = doc["p15"]["per_history"]["CV"]["B2"]["per_history"]["8"]
    assert set(row) == {"clean_ade_ok", "clean_fde_ok", "audc_ok", "audc", "audc_tolerance_distance", "valid_folds",
                        "partial_folds", "audc_qualifies"}


def test_family_audc_records_a_normalization_failure_instead_of_omitting_it(tiny, monkeypatch):
    out, _, _ = tiny
    real = analysis.normalize_severity

    def broken(scene, condition, achieved, measure):
        if scene == "S1" and condition == "A":
            raise ValueError("inconsistent achieved severity")
        return real(scene, condition, achieved, measure)
    monkeypatch.setattr(analysis, "normalize_severity", broken)
    doc = analysis.analyze(RunStore.open(out), CFG)
    entry = doc["family_audc"]["per_history"]["8"]["S1"]["A"]
    assert entry["status"] == "uncomputable" and entry["value"] is None and "fewer than two" in entry["reason"]
    assert (entry["n_valid"], entry["n_undefined"]) == (0, 4)
    assert set(entry["undefined"].values()) == {"inconsistent achieved severity"}
    assert doc["family_audc"]["per_history"]["8"]["S2"]["A"]["status"] == "computable"
    assert doc["p15"]["per_history"]["CV"]["A"]["undefined_reasons_by_history"]["8"] == {"S1": "inconsistent achieved severity"}


# --- the revised common-population companion (decision 0005, section 2) --------------------------

def test_anchors_come_from_frames_and_never_from_positions_or_targets():
    tracks, step = SCENES[0].tracks, SCENES[0].frame_step
    moved = tuple(dataclasses.replace(t, positions=t.positions + 500.0) for t in tracks)
    assert sweep.common_anchors(tracks, step, 12, F, STRIDE) == sweep.common_anchors(moved, step, 12, F, STRIDE)


def test_preflight_shows_only_the_input_length_differs_and_every_fold_passes_six_checks():
    report = sweep.common_preflight(dataclasses.replace(SPEC, common_training=True), SCENES, CFG)
    assert len(report["folds"]) == 2 * 4 and report["common_history"] == 12
    assert report["input_length_is_the_only_difference"] is True
    for fold in report["folds"]:
        assert sorted(fold["leakage_checks"]) == sorted(leakage.CHECKS) and all(fold["leakage_checks"].values())
        scene = {f["test_scene"]: f for f in report["folds"] if f["history_len"] == 12}[fold["test_scene"]]
        assert [fold[k] for k in ("train_windows", "validation_windows", "test_windows")] == (
            [scene[k] for k in ("train_windows", "validation_windows", "test_windows")])
        assert fold["test_windows"] == report["anchors_by_scene"][fold["test_scene"]]
        assert fold["test_scene"] != fold["validation_scene"] and fold["test_scene"] not in fold["train_scenes"]


def test_common_windows_of_one_scene_differ_only_in_the_input_length():
    by_history = {h: sweep.common_windows(SCENES, h, 12, F, STRIDE) for h in (4, 8, 12)}
    for short, full in zip(by_history[4], by_history[12]):
        sweep._require_only_input_length_differs(short, full)
        np.testing.assert_array_equal(short.target, full.target)
        np.testing.assert_array_equal(short.observed, full.observed[:, -4:])


@pytest.mark.parametrize("field", ["target", "observed", "track_ids"])
def test_a_window_that_changes_in_anything_else_is_refused(field):
    short, full = sweep.common_windows(SCENES, 8, 12, F, STRIDE)[0], sweep.common_windows(SCENES, 12, 12, F, STRIDE)[0]
    changed = np.array(getattr(short, field))
    changed[0] = changed[0] + 1
    with pytest.raises(sweep.IntegrityError, match="more than the input length"):
        sweep._require_only_input_length_differs(dataclasses.replace(short, **{field: changed}), full)


def test_a_failed_preflight_trains_nothing_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "run_leakage_checks", lambda *args: ())
    with pytest.raises(leakage.LeakageError):
        sweep.run_sweep(tmp_path / "run", dataclasses.replace(SPEC, common_training=True), SCENES, CFG)
    assert not (tmp_path / "run").exists()


def test_companion_trains_validates_and_evaluates_on_the_common_anchors_at_every_history(companion):
    out, _, _, _ = companion
    report = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    for fold in report["folds"]:
        path = out / "checkpoints" / f"h{fold['history_len']}" / fold["test_scene"] / "lstm-seed0.pt.json"
        split = json.loads(path.read_text(encoding="utf-8"))["split"]
        assert sum(split["scenes"][s]["windows"] for s in split["train_scenes"]) == fold["train_windows"]
        assert split["scenes"][split["validation_scene"]]["windows"] == fold["validation_windows"]
        assert split["scenes"][split["test_scene"]]["windows"] == fold["test_windows"]


def test_companion_changes_only_the_input_length_of_training(companion):
    """Same seed, same anchors, so the same start weights and batch order: history is the one difference."""
    out, _, _, _ = companion
    for scene in ("S1", "S2", "S3", "S4"):
        a = json.loads((out / "checkpoints" / "h8" / scene / "gru-seed1.pt.json").read_text(encoding="utf-8"))
        b = json.loads((out / "checkpoints" / "h12" / scene / "gru-seed1.pt.json").read_text(encoding="utf-8"))
        assert a["init_state_sha256"] == b["init_state_sha256"] and a["purpose_seeds"] == b["purpose_seeds"]
        assert a["settings"] == b["settings"] and a["config_hash"] == b["config_hash"]
        assert a["model"]["architecture"]["history_len"] == 8 and b["model"]["architecture"]["history_len"] == 12


def test_identical_h12_models_are_adopted_and_only_shorter_histories_train(companion, tiny):
    source, _, _ = tiny
    out, status, _, _ = companion
    assert status["counts"]["trainings_run"] == 1 * 4 * 2 * 3 and status["counts"]["trainings_adopted"] == 1 * 4 * 2 * 3
    assert status["scope_complete"] is True and status["research_complete"] is False
    for name in ("S1/lstm-seed0.pt", "S4/gru-seed2.pt"):
        assert (out / "checkpoints" / "h12" / name).read_bytes() == (source / "checkpoints" / "h12" / name).read_bytes()


def test_a_model_is_not_adopted_unless_it_is_proved_identical(tiny, tmp_path):
    """A fifth scene changes each fold's training set, so the stored split differs and nothing may be reused."""
    source, _, _ = tiny
    spec = dataclasses.replace(SPEC, common_training=True, histories=(12,), predictors=("CV", "LSTM"), conditions=("A",), seeds=(0,))
    status = sweep.run_sweep(tmp_path / "run", spec, ss.make_scenes(5), CFG, reuse_from=source)
    records = [r for r in RunStore.open(tmp_path / "run").records() if r.identity.get("experiment") == "training"]
    assert status["counts"].get("trainings_run", 0) == 0 and status["counts"]["trainings_failed"] == 5
    assert len(records) == 5 and all(r.status == FAILED for r in records)
    messages = sorted(r.failure["message"] for r in records)
    assert sum("cannot be adopted" in m and "split" in m for m in messages) == 4
    assert sum("no checkpoint to adopt" in m for m in messages) == 1


def test_companion_leaves_the_earlier_run_untouched(tmp_path):
    earlier = dataclasses.replace(_short_spec(), histories=(12,), predictors=("CV", "GRU"), conditions=("A",))
    source = tmp_path / "earlier"
    sweep.run_sweep(source, earlier, SCENES, CFG)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    status = sweep.run_sweep(tmp_path / "companion", dataclasses.replace(earlier, common_training=True), SCENES, CFG, reuse_from=source)
    assert status["counts"]["trainings_adopted"] == 4 and "trainings_run" not in status["counts"]
    assert {p: p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


def test_companion_analysis_is_kept_apart_from_the_evaluation_only_population(companion, tiny):
    _, _, doc, _ = companion
    _, _, original = tiny
    assert set(doc["curves"]) == set(doc["p15"]) == {"common_h12_trained"}
    assert list(doc["population_labels"]) == ["common_h12_trained"] and "training, validation and evaluation" in (
        doc["population_labels"]["common_h12_trained"])
    assert "hypotheses" not in doc and "p12" not in doc and doc["monotonicity"] == []
    assert doc["research_results"] is False and "ENGINEERING FEASIBILITY ONLY" in doc["label"]
    assert "common_h12_trained" not in original["curves"]


def test_companion_cv_and_ca_match_the_evaluation_only_population_exactly(companion, tiny):
    """They are not trained, and the windows and corruption draws are the same, so every number must agree."""
    _, _, doc, _ = companion
    _, _, original = tiny
    for predictor in ("CV", "CA"):
        ours = doc["curves"]["common_h12_trained"]["8"]["S2"][predictor]["D"]["levels"]
        theirs = original["curves"]["common_h12"]["8"]["S2"][predictor]["D"]["levels"]
        assert [r["ade_m"]["mean"] for r in ours] == [r["ade_m"]["mean"] for r in theirs]
        assert doc["clean"]["common_h12_trained"]["8"]["S2"][predictor] == original["clean"]["common_h12"]["8"]["S2"][predictor]


def test_companion_has_its_own_provenance(companion, tiny):
    source, _, _ = tiny
    out, _, _, _ = companion
    ours, theirs = RunStore.open(out).manifest()["spec"], RunStore.open(source).manifest()["spec"]
    assert ours["common_training"] is True and theirs["common_training"] is False
    assert ours["adopts_models_from"] == str(source) and len(ours["preflight_sha256"]) == 64
    assert RunStore.open(out).manifest()["spec_sha256"] != RunStore.open(source).manifest()["spec_sha256"]


def test_companion_is_refused_outside_the_synthetic_scope_and_adoption_outside_the_companion(tiny, tmp_path):
    source, _, _ = tiny
    spec = dataclasses.replace(sweep.ethhotel_spec(CFG), common_training=True)
    with pytest.raises(sweep.ScopeError, match="synthetic engineering only"):
        sweep.run_sweep(tmp_path / "a", spec, [ss.make_scene("ETH", 0)], CFG)
    with pytest.raises(sweep.ScopeError, match="only the common-population companion"):
        sweep.run_sweep(tmp_path / "b", _short_spec(), SCENES, CFG, reuse_from=source)


def test_companion_resume_repeats_nothing(tmp_path):
    spec = dataclasses.replace(SPEC, common_training=True, histories=(4, 12), predictors=("CV",), conditions=("A",), seeds=(0,))
    first = sweep.run_sweep(tmp_path / "run", spec, SCENES, CFG)
    second = sweep.run_sweep(tmp_path / "run", spec, SCENES, CFG, resume=True)
    assert second["counts"] == {"cells_reused": first["counts"]["cells_computed"]}


# --- the run -------------------------------------------------------------------------------------

def test_every_expected_cell_finishes_and_a_scope_run_is_never_research_complete(tiny):
    _, status, _ = tiny
    assert status["scope_complete"] is True and status["research_complete"] is False
    assert status["finished_cells"] == status["expected_cells"] == 4 * 2 * 2 * 3 * 4 * 31
    assert status["by_status"]["failed"] == 0 and status["counts"]["trainings_run"] == 4 * 2 * 2 * 3


def test_blocked_experiments_are_recorded_not_skipped(tiny):
    out, status, doc = tiny
    assert set(status["blocked"]) == {"P-8", "P-10", "P-11", "five-scene-static-research"} == set(doc["blocked"])
    assert RunStore.open(out).summary()["by_status"]["blocked"] == 4
    assert "moving_dataset" in status["blocked"]["P-8"] and "h8_generation_attempts" in status["blocked"]["P-10"]


def test_cell_identity_carries_full_provenance(tiny):
    out, _, _ = tiny
    record = next(r for r in RunStore.open(out).records()
                  if r.identity.get("predictor") == "LSTM" and r.identity.get("condition") == "D")
    i = record.identity
    assert {"scene", "fold", "validation_scene", "seed", "history_len", "population", "requested_level", "config_hash",
            "source_sha256", "scene_sha256", "windows_sha256", "checkpoint_sha256", "settings"} <= set(i)
    assert i["checkpoint_sha256"] and i["config_hash"] == CFG.hash and record.result["achieved"]["measure"] == "switch_event_rate"


def test_resume_repeats_nothing(tiny, tmp_path):
    out = tmp_path / "run"
    spec = _short_spec()
    first = sweep.run_sweep(out, spec, SCENES, CFG)
    second = sweep.run_sweep(out, spec, SCENES, CFG, resume=True)
    assert second["counts"] == {"cells_reused": first["counts"]["cells_computed"],
                                "trainings_reused": first["counts"]["trainings_run"]}


def _digest(out) -> dict:
    return {r.key: r.digest for r in RunStore.open(out).records()}


def test_interrupted_run_resumes_without_repeating_work_and_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    spec = _short_spec()
    reference = sweep.run_sweep(tmp_path / "whole", spec, SCENES, CFG)
    real, calls = evaluation.score, {"n": 0}

    def interrupted(*args):
        calls["n"] += 1
        if calls["n"] == 30:
            raise KeyboardInterrupt
        return real(*args)
    monkeypatch.setattr(evaluation, "score", interrupted)
    with pytest.raises(KeyboardInterrupt):
        sweep.run_sweep(tmp_path / "cut", spec, SCENES, CFG)
    monkeypatch.setattr(evaluation, "score", real)
    done = sum(1 for r in RunStore.open(tmp_path / "cut").records() if r.status == COMPLETE and "predictor" in r.identity
               and "condition" in r.identity)
    assert done == 29
    resumed = sweep.run_sweep(tmp_path / "cut", spec, SCENES, CFG, resume=True)
    assert resumed["counts"]["cells_reused"] == 29
    assert resumed["counts"]["cells_computed"] == reference["counts"]["cells_computed"] - 29
    assert resumed["scope_complete"] is True and _digest(tmp_path / "cut") == _digest(tmp_path / "whole")


def test_incompatible_cached_outputs_are_rejected(tmp_path):
    out = tmp_path / "run"
    sweep.run_sweep(out, _short_spec(), SCENES, CFG)
    with pytest.raises(FileExistsError):
        sweep.run_sweep(out, _short_spec(), SCENES, CFG)
    with pytest.raises(IncompatibleRunError, match="settings"):
        sweep.run_sweep(out, _short_spec(settings=dataclasses.replace(FAST, max_epochs=3)), SCENES, CFG, resume=True)
    changed = [ss.make_scene("S1", 9), *SCENES[1:]]
    with pytest.raises(IncompatibleRunError, match="scenes"):
        sweep.run_sweep(out, _short_spec(), changed, CFG, resume=True)


def test_a_tampered_cell_is_rejected_on_resume(tmp_path):
    out = tmp_path / "run"
    spec = _short_spec()
    sweep.run_sweep(out, spec, SCENES, CFG)
    store = RunStore.open(out)
    record = next(r for r in store.records() if r.identity.get("condition") == "A")
    path = store.path(record.identity)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["result"]["metrics"]["ade_m"] = 0.0
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(StaleRecordError):
        sweep.run_sweep(out, spec, SCENES, CFG, resume=True)


def test_a_failed_cell_is_recorded_kept_apart_and_retried(tmp_path, monkeypatch):
    spec = dataclasses.replace(_short_spec(), predictors=("CV",))
    real = evaluation.corrupt

    def broken(tracks, pop, condition, level, seed, cfg, step):
        if condition == "A" and level == 0.1:
            raise RuntimeError("injected failure")
        return real(tracks, pop, condition, level, seed, cfg, step)
    monkeypatch.setattr(evaluation, "corrupt", broken)
    status = sweep.run_sweep(tmp_path / "run", spec, SCENES, CFG)
    assert status["by_status"][FAILED] == 4 and status["scope_complete"] is False
    doc = analysis.write_analysis(RunStore.open(tmp_path / "run"), CFG)
    curve = doc["curves"]["per_history"]["8"]["S1"]["CV"]["A"]
    assert curve["audc"]["status"] == "uncomputable" and curve["audc"]["value"] is None
    monkeypatch.setattr(evaluation, "corrupt", real)
    resumed = sweep.run_sweep(tmp_path / "run", spec, SCENES, CFG, resume=True)
    assert resumed["by_status"][FAILED] == 0 and resumed["scope_complete"] is True and resumed["counts"]["cells_computed"] == 4


def test_a_leakage_failure_blocks_that_fold_and_writes_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "run_leakage_checks", lambda *args: ())      # every check missing
    status = sweep.run_sweep(tmp_path / "run", _short_spec(), SCENES, CFG)
    assert status["counts"] == {"units_failed": 4} and status["scope_complete"] is False


# --- scope ---------------------------------------------------------------------------------------

def test_learned_models_are_refused_on_real_scenes(tmp_path):
    spec = dataclasses.replace(sweep.ethhotel_spec(CFG), predictors=("CV", "LSTM"))
    with pytest.raises(sweep.ScopeError, match="never run on ETH or HOTEL"):
        sweep.run_sweep(tmp_path / "run", spec, [ss.make_scene("ETH", 0)], CFG)


def test_real_scene_names_are_refused_in_the_synthetic_scope(tmp_path):
    with pytest.raises(sweep.ScopeError, match="refuses real scenes"):
        sweep.run_sweep(tmp_path / "run", _short_spec(), [ss.make_scene("ETH", 0), *SCENES[:2]], CFG)


def test_outputs_inside_the_repository_are_refused():
    with pytest.raises(ValueError, match="inside the repository"):
        sweep.run_sweep(REPO_ROOT / "src" / "p4-run", _short_spec(), SCENES, CFG)


@pytest.fixture(scope="module")
def eth_like(tmp_path_factory):
    out = tmp_path_factory.mktemp("ethhotel") / "run"
    scenes = [ss.make_scene("ETH", 0), ss.make_scene("HOTEL", 1)]
    status = sweep.run_sweep(out, sweep.ethhotel_spec(CFG), scenes, CFG)
    return status, analysis.write_analysis(RunStore.open(out), CFG)


def test_descriptive_scope_is_never_pooled_or_a_family_mean(eth_like):
    status, doc = eth_like
    assert status["scope_complete"] is True and status["counts"].get("trainings_run", 0) == 0
    assert doc["across_scenes"] == {} and doc["family_audc"] == {}
    assert "LOCAL ONLY" in doc["label"] and doc["research_results"] is False
    assert all(h["engineering_check"] is None for h in doc["hypotheses"].values())
    assert "p12" not in doc and "p15" not in doc


# --- analysis ------------------------------------------------------------------------------------

def test_analysis_keeps_raw_seed_values_and_counts(tiny):
    _, _, doc = tiny
    curve = doc["curves"]["per_history"]["8"]["S1"]["GRU"]["D"]
    assert set(curve["achieved"]["0.2"]) == {"0", "1", "2"}
    row = curve["levels"][-1]
    assert row["d_ade"]["n_valid"] + row["d_ade"]["n_undefined"] == 3 and set(row["windows_by_seed"]) == {"0", "1", "2"}


def test_across_scene_values_are_means_of_scene_values(tiny):
    _, _, doc = tiny
    node = doc["across_scenes"]["per_history"]["8"]["CV"]["A"]
    values = [doc["curves"]["per_history"]["8"][s]["CV"]["A"]["audc"]["value"] for s in ("S1", "S2", "S3", "S4")]
    assert node["audc"]["mean"] == pytest.approx(sum(values) / 4, rel=1e-12) and node["audc"]["n_valid"] == 4


def test_family_audc_covers_all_four_predictors(tiny):
    _, _, doc = tiny
    family = doc["family_audc"]["per_history"]["8"]["S1"]["A"]
    assert family["n_valid"] + family["n_undefined"] == 4


def test_paired_audc_is_kept_alongside_the_primary(tiny):
    _, _, doc = tiny
    curve = doc["curves"]["per_history"]["8"]["S1"]["CV"]["C"]
    assert {"audc", "audc_paired"} <= set(curve) and curve["audc_paired"]["status"] in ("computable", "uncomputable")


def test_monotonicity_covers_every_scene_and_condition(tiny):
    _, _, doc = tiny
    assert len(doc["monotonicity"]) == 4 * 5
    assert {m["rule"] for m in doc["monotonicity"]} == {"strict", "non_decreasing", "diagnostic"}
    d = next(m for m in doc["monotonicity"] if m["condition"] == "D")
    assert d["status"] in ("diagnostic_finding", "no_violation_observed", "not_evaluated") and "diagnostics" in d


def test_p12_gaps_exist_for_each_learned_model(tiny):
    _, _, doc = tiny
    assert set(doc["p12"]) == {"LSTM", "GRU"} and len(doc["p12"]["GRU"]["D"]) == 6


def test_no_hypothesis_has_a_research_verdict(tiny):
    _, _, doc = tiny
    h = doc["hypotheses"]
    assert set(h) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"}
    assert all(v["research_verdict"] == "not_evaluable" for v in h.values())
    assert h["H8"]["engineering_check"] is None and "P-10" in h["H8"]["reason"] and "moving" in h["H5"]["reason"]
    assert "not evidence" in h["H1"]["engineering_check"]["scope"]


# --- reproducibility -----------------------------------------------------------------------------

CHILD = """
import dataclasses, json, sys
sys.path.insert(0, {tests!r})
from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.experiments import sweep
from trackvision.experiments.store import RunStore
from trackvision.predict.training import TrainingSettings
cfg = config.validate().config
fast = dataclasses.replace(TrainingSettings.from_config(cfg), max_epochs=2)
base = sweep.synthetic_spec(cfg, histories=(8,), settings=fast)
spec = dataclasses.replace(base, predictors=("CV", "GRU"), conditions=("A", "D"), seeds=(0,))
sweep.run_sweep(sys.argv[1], spec, ss.make_scenes(4), cfg)
print(json.dumps(sorted((r.key, r.digest) for r in RunStore.open(sys.argv[1]).records())))
"""


def test_a_sweep_is_identical_across_fresh_processes(tmp_path):
    code = CHILD.format(tests=str(REPO_ROOT / "tests"))
    runs = []
    for name in ("first", "second"):
        done = subprocess.run([sys.executable, "-c", code, str(tmp_path / name)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        runs.append(json.loads(done.stdout.strip().splitlines()[-1]))
    assert runs[0] == runs[1] and len(runs[0]) > 100


# --- CLI -----------------------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, "-m", "trackvision", "sweep", *map(str, args)], capture_output=True, text=True,
                          cwd=REPO_ROOT, check=False)


@pytest.mark.parametrize("experiment", ["P-8", "P-10", "P-11"])
def test_cli_refuses_blocked_experiments(experiment):
    from trackvision import cli
    proc = _cli("run", "--experiment", experiment)
    assert proc.returncode == cli.EXIT_BLOCKED and f"BLOCKED {experiment}" in proc.stdout


def test_cli_plan_separates_eligible_from_blocked():
    doc = json.loads(_cli("plan").stdout)
    assert "P-15" in doc["eligible"]["synthetic_engineering"] and "P-15" not in doc["eligible"]["ethhotel_descriptive"]
    assert sweep.COMPANION in doc["eligible"]["synthetic_engineering"] and "P-10" in doc["blocked"]


@pytest.mark.parametrize("experiment, conditions, histories", [
    ("P-3", (), (8,)), ("P-4", ("A",), (8,)), ("P-5", ("B1", "B2"), (8,)), ("P-6", ("C",), (8,)), ("P-7", ("D",), (8,)),
    ("P-12", evaluation.CONDITIONS, (8,)), ("P-15", evaluation.CONDITIONS, (4, 6, 8, 10, 12)),
])
def test_an_experiment_selects_only_its_own_cells(experiment, conditions, histories):
    spec = sweep.experiment_spec(sweep.synthetic_spec(CFG), experiment, CFG)
    assert spec.conditions == conditions and spec.histories == histories and spec.common_training is False


def test_the_companion_experiment_selects_common_training():
    spec = sweep.experiment_spec(sweep.synthetic_spec(CFG), sweep.COMPANION, CFG)
    assert spec.common_training is True and spec.histories == (4, 6, 8, 10, 12)


def test_an_experiment_outside_the_scope_is_refused():
    with pytest.raises(sweep.ScopeError, match="not eligible"):
        sweep.experiment_spec(sweep.ethhotel_spec(CFG), "P-15", CFG)
    with pytest.raises(sweep.ScopeError, match="not eligible"):
        sweep.experiment_spec(sweep.synthetic_spec(CFG), "P-99", CFG)


def test_cli_passes_the_selected_experiment_to_the_runner(monkeypatch, tmp_path):
    seen = {}

    def run_sweep(out, spec, scenes, cfg, resume=False, reuse_from=None):
        seen.update(out=out, spec=spec, reuse_from=reuse_from)
        return {"scope": spec.scope, "scope_complete": True, "research_complete": False, "expected_cells": 0,
                "finished_cells": 0, "counts": {}, "by_status": {}, "blocked": {}}
    monkeypatch.setattr(sweep, "run_sweep", run_sweep)
    monkeypatch.setattr(cli, "_scenes", lambda scope, cfg: [])
    assert cli.main(["sweep", "run", "--experiment", "P-4", "--out", str(tmp_path)]) == cli.EXIT_OK
    assert seen["spec"].conditions == ("A",) and seen["spec"].histories == (8,)
    assert cli.main(["sweep", "run", "--experiment", sweep.COMPANION, "--out", str(tmp_path), "--reuse-from", str(tmp_path)]) == cli.EXIT_OK
    assert seen["spec"].common_training is True and seen["reuse_from"] == tmp_path
    assert cli.main(["sweep", "run", "--out", str(tmp_path)]) == cli.EXIT_OK
    assert seen["spec"].conditions == evaluation.CONDITIONS and len(seen["spec"].histories) == 5


def test_a_narrowed_run_can_still_be_analyzed(tmp_path):
    spec = dataclasses.replace(sweep.experiment_spec(SPEC, "P-4", CFG), seeds=(0,), predictors=("CV", "LSTM", "GRU"))
    assert spec.conditions == ("A",)
    status = sweep.run_sweep(tmp_path / "run", spec, SCENES, CFG)
    assert status["scope_complete"] is True
    doc = analysis.write_analysis(RunStore.open(tmp_path / "run"), CFG)
    assert "hypotheses" not in doc and "A" in doc["p12"]["LSTM"] and curves.export_figures(doc, tmp_path / "figures")


@pytest.mark.parametrize("arguments", [("run",), ("status",), ("export",), ("run", "--experiment", "P-4")])
def test_cli_missing_out_is_an_argument_error_not_a_traceback(arguments):
    proc = _cli(*arguments)
    assert proc.returncode == 2 and "requires --out" in proc.stderr and "Traceback" not in proc.stderr


def test_cli_unknown_experiment_and_misplaced_reuse_are_argument_errors(tmp_path):
    proc = _cli("run", "--experiment", "P-99", "--out", tmp_path / "run")
    assert proc.returncode == 2 and "not eligible" in proc.stderr and not (tmp_path / "run").exists()
    proc = _cli("run", "--experiment", "P-4", "--out", tmp_path / "run", "--reuse-from", tmp_path)
    assert proc.returncode == 2 and "--reuse-from applies only" in proc.stderr


def test_cli_status_and_export_read_a_stored_run(tiny):
    out, _, _ = tiny
    status = json.loads(_cli("status", "--out", out).stdout)
    assert status["research_complete"] is False and status["by_status"]["blocked"] == 4 and "git" in status
    proc = _cli("export", "--out", out)
    assert proc.returncode == 0, proc.stderr
    assert len(list((out / "figures").glob("degradation_*.svg"))) == 4 * 5
    assert len(list((out / "figures").glob("p15_*.svg"))) == 2 * 4
    assert len(list((out / "figures").glob("conditional_error_*.svg"))) == 2 * 4 * 2


# --- figures -------------------------------------------------------------------------------------

def test_figures_show_coverage_with_error_and_name_their_scope(tiny):
    _, _, doc = tiny
    svg = curves.degradation_figure(doc, "per_history", "8", "S1", "B2")
    assert "Prediction coverage" in svg and "Relative ADE degradation" in svg and "Synthetic engineering only" in svg
    assert all(p in svg for p in ("CV", "CA", "LSTM", "GRU"))


def test_undefined_and_above_one_points_stay_visible(tiny):
    _, _, doc = tiny
    edited = copy.deepcopy(doc)
    row = edited["curves"]["per_history"]["8"]["S1"]["CV"]["A"]["levels"]
    row[2]["d_ade"]["mean"] = None
    row[3]["s_norm"] = 1.4
    svg = curves.degradation_figure(edited, "per_history", "8", "S1", "A")
    assert "× undefined" in svg and "above 1, not in AUDC" in svg
    untouched = curves.degradation_figure(doc, "per_history", "8", "S1", "A")
    assert "× undefined" not in untouched and "above 1, not in AUDC" not in untouched


def test_a_line_never_bridges_an_undefined_point():
    points = [(0.0, 0.0, "used", "a"), (0.5, None, "undefined", "b"), (1.0, 0.8, "used", "c")]
    body = "".join(curves._panel(80, "t", "x", "y", (0.0, 1.0), [("CV", 1, points)]))
    assert "<polyline" not in body and body.count("×") >= 2


def test_a_lone_series_does_not_borrow_a_categorical_colour(tiny):
    """In the P-15 figure GRU would otherwise share condition C's colour."""
    _, _, doc = tiny
    svg = curves.p15_figure(doc, "per_history", "GRU")
    assert f"var(--s{curves.NEUTRAL})" in svg


def test_conditional_error_figure_shows_every_required_quantity(tiny):
    _, _, doc = tiny
    svg = curves.conditional_error_figure(doc, "common_h12", "LSTM", "B2")
    for text in ("ordinary", "paired", "Prediction coverage", "Predicted windows", "Ordinary AUDC folds", "does not",
                 "history sufficiency", "Common evaluation population", "Synthetic engineering only"):
        assert text in svg, text


def test_a_partial_fold_mean_is_hollow_and_unconnected():
    points = [(4.0, 0.2, "used", "a"), (8.0, 0.3, "used", "b"), (12.0, -0.3, "partial", "c")]
    body = "".join(curves._panel(80, "t", "x", "y", (4.0, 12.0), [("B2", 3, points)]))
    assert body.count("<polyline") == 1 and 'fill="var(--surface)"' in body


def test_the_p15_figure_names_its_population(tiny):
    _, _, doc = tiny
    assert "training and validation" in curves.p15_figure(doc, "common_h12", "CV")
    assert "Per-history population" in curves.p15_figure(doc, "per_history", "CV")


def test_analysis_can_be_written_outside_the_run(tiny, tmp_path):
    out, _, _ = tiny
    before = (out / "analysis.json").stat().st_mtime_ns if (out / "analysis.json").exists() else None
    analysis.write_analysis(RunStore.open(out), CFG, tmp_path)
    assert (tmp_path / "analysis.json").exists()
    assert ((out / "analysis.json").stat().st_mtime_ns if (out / "analysis.json").exists() else None) == before


def test_figures_are_written_atomically_from_the_stored_document(tiny, tmp_path):
    _, _, doc = tiny
    names = curves.export_figures(doc, tmp_path / "figures")
    assert names and all((tmp_path / "figures" / n).read_text(encoding="utf-8").startswith("<svg") for n in names)
    assert not list((tmp_path / "figures").glob(".*tmp"))
