"""E4 on a small generated scene. Generated scenes are fixtures, never research data, and the checkpoint fixtures are
trained for two epochs, so no number here is a result."""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fixtures import synthetic_split as ss
from trackvision import config
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments import mot17_trackeval as mt
from trackvision.experiments import synthetic_tracker_engineering as ste
from trackvision.experiments.store import RunStore
from trackvision.predict import learned
from trackvision.predict.folds import nested_split
from trackvision.predict.training import TrainingSettings, train_predictor
from trackvision.synthetic.render import RenderConfig
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks
from trackvision.tracking.background_subtraction import BgSubConfig
from trackvision.tracking.tracker import TrackerConfig

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
F = CFG.frozen["protocol"]["static"]["horizon_len"]
H = CFG.frozen["protocol"]["static"]["history_len"]
TE_PYTHON = os.environ.get("TRACKVISION_TRACKEVAL_PYTHON")
TE_SOURCE = os.environ.get("TRACKVISION_TRACKEVAL_SOURCE")
needs_trackeval = pytest.mark.skipif(not (TE_PYTHON and TE_SOURCE and Path(TE_PYTHON).exists()),
                                     reason="isolated TrackEval environment not configured")
SCENE = "SYN-501"


@pytest.fixture(scope="module")
def scene():
    return generate(SceneConfig(seed=7, n_frames=400, n_pedestrians=6), DT)


@pytest.fixture(scope="module")
def models():
    return {(c, 0): (learned.build_model(c, H, F, CFG), "fixture-checkpoint") for c in learned.CELLS}


def stub(patch, scene, models):
    """Small scene, fixture models and a stubbed TrackEval, for tests that exercise the run's own logic."""
    patch.setattr(ste, "scene_ground_truth", lambda seed, cfg: scene)
    patch.setattr(ste, "load_fold_models", lambda run, name, seeds, cfg: models)
    patch.setattr(ste, "run_official", lambda *a: {SCENE: {"HOTA": "stub", "IDSW": "stub"}})


@pytest.fixture(scope="module")
def e4(scene, models, tmp_path_factory):
    """A whole E4 run on the small scene. TrackEval is stubbed here; the real one is exercised below when configured."""
    base = tmp_path_factory.mktemp("e4")
    digest = inj.tracks_digest(ground_truth_grid_tracks(scene))
    models_run = base / "phase4"
    RunStore.create(models_run, {"scope": "synthetic_engineering", "config_hash": CFG.hash, "source_sha256": "s",
                                 "scenes": {SCENE: digest}, "seeds": [0]})
    before = {p: p.read_bytes() for p in models_run.rglob("*") if p.is_file()}
    out = base / "run"
    with pytest.MonkeyPatch.context() as patch:
        stub(patch, scene, models)
        doc = ste.run(out, CFG, models_run, Path("p"), Path("s"), [SCENE])
    return out, doc, models_run, before


def test_the_detector_is_never_fed_ground_truth_boxes(scene):
    boxes, visible, digest = ste.blob_detections(scene, RenderConfig(), BgSubConfig())
    assert len(boxes) == scene.config.n_frames and len(visible) == len(scene.frame) and len(digest) == 64
    source = Path(ste.__file__).read_text(encoding="utf-8")
    assert "ground_truth_detections" not in source and ste.SOURCE == "synthetic_blob_detector_estimated_background"


def test_rendering_and_detection_are_deterministic(scene):
    a, b = ste.blob_detections(scene, RenderConfig(), BgSubConfig()), ste.blob_detections(scene, RenderConfig(), BgSubConfig())
    assert a[2] == b[2] and all(np.array_equal(x, y) for x, y in zip(a[0], b[0]))


def test_ground_truth_export_marks_partial_visibility_and_omits_invisible(scene):
    visible = np.zeros(len(scene.frame))
    visible[0], visible[1] = 1.0, 0.2
    rows = ste.ground_truth_mot_text(scene, visible).splitlines()
    assert len(rows) == 2
    first, second = rows[0].split(","), rows[1].split(",")
    assert int(first[0]) == int(scene.frame[0]) + 1 and (first[6], first[7]) == ("1", "1")
    assert (second[6], second[7]) == ("0", str(ste.DISTRACTOR_CLASS))


def test_tracker_export_is_one_based_with_mot_boxes():
    from trackvision.tracking.tracker import TrackerResult, TrackOutput
    track = TrackOutput(3, np.array([0, 1]), np.array([[10.0, 20.0, 30.0, 60.0], [11.0, 20.0, 31.0, 60.0]]))
    result = TrackerResult((track,), 1, 1, 0, 1)
    assert ste.tracker_mot_text(result).splitlines()[0] == "1,3,11.000,21.000,20.000,40.000,1,-1,-1,-1"


def test_record_carries_synthetic_lineage_and_provenance(e4):
    out, doc, _, _ = e4
    assert doc["provenance"]["ground_truth_lineage"] == "synthetic_generator_ped_id"
    assert "not YOLO" in doc["provenance"]["detector"] and "not ByteTrack" in doc["provenance"]["tracker"]
    assert doc["research_results"] is False and doc["not_p9"] is True and "SYNTHETIC ENGINEERING ONLY" in doc["label"]
    spec = RunStore.open(out).manifest()["spec"]
    assert spec["scope"] == ste.SCOPE and spec["trackeval_commit"] == mt.TRACKEVAL_COMMIT and spec["detector"]["min_area_px"] == 50


def test_no_real_tracker_label_appears_in_any_synthetic_output(e4):
    out, _, _, _ = e4
    for path in out.rglob("*"):
        if path.is_file() and path.suffix == ".json":
            text = path.read_text(encoding="utf-8")
            assert "real_tracker_iou" not in text and "real_switch_event_rate" not in text


def test_the_section_3_7_field_is_present_null_and_explained(e4):
    _, doc, _, _ = e4
    assert "association_mechanism" in doc and doc["association_mechanism"] is None
    assert doc["association_status"] == "unresolved" and "project-owner decision" in doc["association_blocker"]


def test_official_values_and_project_diagnostics_are_kept_apart(e4):
    _, doc, _, _ = e4
    assert set(doc["official_trackeval"]) == {"note", SCENE} and "not a benchmark" in doc["official_trackeval"]["note"]
    record = doc["by_scene"][SCENE]
    assert "official_trackeval" not in record and "not TrackEval" in record["project_diagnostics"]["note"]
    assert "HOTA" not in record["project_diagnostics"] and "error_profile" in record["project_diagnostics"]


def test_the_grid_profile_uses_clean_ground_truth_denominators(e4, scene):
    _, doc, _, _ = e4
    tracks = ground_truth_grid_tracks(scene)
    p = doc["by_scene"][SCENE]["project_diagnostics"]["error_profile"]
    assert p["dropout"]["eligible_gt_observations"] == sum(len(t.frames) for t in tracks)
    assert p["fragmentation"]["eligible_track_frames"] == sum(max(len(t.frames) - 2, 0) for t in tracks)
    assert p["localization"]["unit"] == "metres" and p["fragmentation"]["gap_limit_steps"] == 5
    assert p["identity_exchange"]["gap_limit_steps"] == 1 and "D_MAX" in p["identity_exchange"]["denominator"]


def test_coverage_is_reported_beside_conditional_accuracy(e4, scene):
    _, doc, _, _ = e4
    prediction = doc["by_scene"][SCENE]["prediction"]
    c = prediction["coverage"]
    assert 0 <= c["predicted_windows"] <= c["eligible_ground_truth_windows"]
    assert c["prediction_coverage"] == pytest.approx(c["predicted_windows"] / c["eligible_ground_truth_windows"])
    assert set(prediction["predictors"]) == {"CV", "CA", "LSTM", "GRU"}


def test_seeds_are_aggregated_within_the_scene_and_the_model_is_recorded(e4):
    _, doc, _, _ = e4
    row = doc["by_scene"][SCENE]["prediction"]["predictors"]["GRU"]
    assert set(row["by_seed"]) == {"0"} and row["by_seed"]["0"]["checkpoint_sha256"] == "fixture-checkpoint"
    assert row["seed_mean"]["tracker_ade_m"]["n_valid"] + row["seed_mean"]["tracker_ade_m"]["n_undefined"] == 1


def test_a_run_never_overwrites_and_earlier_records_stay_as_they_were(e4):
    out, _, models_run, before = e4
    assert {p: p.read_bytes() for p in models_run.rglob("*") if p.is_file()} == before
    with pytest.raises(FileExistsError):
        ste.run(out, CFG, models_run, Path("p"), Path("s"), [SCENE])


def test_resume_repeats_no_scene(e4, scene, models, monkeypatch):
    out, doc, models_run, _ = e4
    stub(monkeypatch, scene, models)
    monkeypatch.setattr(ste, "scene_record", lambda *a, **k: pytest.fail("a stored scene was recomputed"))
    assert ste.run(out, CFG, models_run, Path("p"), Path("s"), [SCENE], resume=True)["by_scene"] == doc["by_scene"]


def test_a_scene_that_differs_from_the_phase_4_scene_is_refused(tmp_path, scene, models, monkeypatch):
    stub(monkeypatch, scene, models)
    run = tmp_path / "phase4"
    RunStore.create(run, {"scope": "x", "config_hash": CFG.hash, "source_sha256": "s", "scenes": {SCENE: "0" * 64}, "seeds": [0]})
    with pytest.raises(ValueError, match="is not the scene"):
        ste.run(tmp_path / "r", CFG, run, Path("p"), Path("s"), [SCENE])


def test_unknown_scenes_are_refused(tmp_path):
    with pytest.raises(ValueError, match="scenes must come from"):
        ste.run(tmp_path / "r", CFG, tmp_path, Path("p"), Path("s"), ["ETH"])


def _fold_models(tmp_path):
    """A real two-epoch checkpoint for the fold that holds out S1, stored as the Phase 4 layout does."""
    scenes = ss.make_scenes(4)
    windows = ss.windows_of(scenes, CFG)
    split = nested_split(windows, "S1", ss.cyclic_rotation(w.scene for w in windows))
    fast = dataclasses.replace(TrainingSettings.from_config(CFG), max_epochs=2)
    folder = tmp_path / "checkpoints" / f"h{H}" / "S1"
    for cell in learned.CELLS:
        train_predictor(split, scenes, cell, 0, CFG, folder, "test", fast)
    return tmp_path


def test_models_are_used_only_on_the_scene_they_never_saw(tmp_path):
    run = _fold_models(tmp_path)
    assert set(ste.load_fold_models(run, "S1", [0], CFG)) == {("LSTM", 0), ("GRU", 0)}
    folder = run / "checkpoints" / f"h{H}"
    (folder / "S2").mkdir()
    for path in (folder / "S1").iterdir():
        (folder / "S2" / path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="was not held out"):
        ste.load_fold_models(run, "S2", [0], CFG)


@needs_trackeval
def test_the_real_trackeval_scores_ground_truth_against_itself(scene, tmp_path):
    visible = np.ones(len(scene.frame))
    gt = ste.ground_truth_mot_text(scene, visible)
    tracks = "".join(",".join(r.split(",")[:6]) + ",1,-1,-1,-1\n" for r in gt.splitlines())
    for folder, text in (("ground_truth", gt), ("tracker_output", tracks)):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"{SCENE}.txt").write_text(text, encoding="utf-8", newline="\n")
    h = ste.run_official(tmp_path, {SCENE: scene}, CFG, Path(TE_PYTHON), Path(TE_SOURCE))[SCENE]
    assert (h["MOTA"], h["IDF1"], h["HOTA"], h["CLR_FP"], h["CLR_FN"], h["IDSW"]) == (1.0, 1.0, 1.0, 0, 0, 0)


def test_the_benchmark_name_is_not_mot17():
    c = mt.trackeval_config(Path("o"), Path("s"), CFG, benchmark=ste.BENCHMARK, split=ste.SPLIT, tracker=ste.TRACKER)
    assert c["dataset_config"]["BENCHMARK"] == "SYNTHETIC" and c["dataset_config"]["TRACKERS_TO_EVAL"] == [ste.TRACKER]
    assert json.dumps(c).count("MOT17") == 0
