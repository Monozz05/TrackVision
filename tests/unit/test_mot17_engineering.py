"""E1, E2 and E3 on small MOT-format fixtures (hand-built boxes, not MOT17 content). No fabricated result is used:
official values come from the real TrackEval when it is configured, and a labelled stub otherwise."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fixtures import synthetic_mot17 as sm
from trackvision import config
from trackvision.experiments import mot17_engineering as me
from trackvision.experiments import mot17_feasibility as mf
from trackvision.experiments import mot17_trackeval as mt
from trackvision.io import mot17

CFG = config.validate().config
TE_PYTHON = os.environ.get("TRACKVISION_TRACKEVAL_PYTHON")
TE_SOURCE = os.environ.get("TRACKVISION_TRACKEVAL_SOURCE")
needs_trackeval = pytest.mark.skipif(not (TE_PYTHON and TE_SOURCE and Path(TE_PYTHON).exists()),
                                     reason="isolated TrackEval environment not configured")
SEQ = "MOT17-02"


def archive(tmp_path, variant="FRCNN", length=10, det_frames=None):
    folder = f"{SEQ}-{variant}"
    zip_path = sm.write_zip(tmp_path / "MOT17.zip", {folder: sm.walker_sequence(folder, length, det_frames)})
    return zip_path, sm.write_manifest(zip_path, tmp_path / "mot17.sha256")


def parsed(variant="FRCNN", length=10, det_frames=None):
    folder = f"{SEQ}-{variant}"
    seq = sm.walker_sequence(folder, length, det_frames)
    info = mot17.parse_seqinfo(seq["seqinfo"].encode(), folder)
    gt, det = mot17.parse_gt(seq["gt"].encode(), info), mot17.parse_detections(seq["det"].encode(), info)
    return info, gt, det, mot17.categorise(gt)


def stub_official(monkeypatch, sequences=(SEQ,)):
    """TrackEval replaced by a labelled stub, so the plumbing is tested without its environment."""
    monkeypatch.setattr(mt, "run_trackeval", lambda *a, **k: 0)
    monkeypatch.setattr(mt, "write_record", lambda *a, **k: {"per_sequence_official": {
        f"{s}-{v}": {"HOTA": "stub", "IDSW": "stub"} for s in sequences for v in mot17.VARIANTS}})


# --- E1 ------------------------------------------------------------------------------------------

def test_e1_uses_the_seven_training_sequences_with_ground_truth():
    assert me.e1_sequences(CFG) == ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]


@pytest.mark.parametrize("length, first, last", [(600, 301, 600), (1050, 526, 1050), (837, 419, 837), (525, 263, 525)])
def test_second_half_boundaries(length, first, last):
    assert mot17.half_split(length)["second"] == (first, last)


def test_only_the_second_half_reaches_the_tracker(tmp_path):
    zip_path, manifest = archive(tmp_path)
    run = mf.run_feasibility(CFG, zip_path, manifest, sequences=[SEQ])
    frames = {int(f) for t in run.sequences[0].tracker.tracks for f in t.frames}
    assert min(frames) >= 6 and max(frames) == 10


@pytest.mark.parametrize("variant", ["FRCNN", "DPM", "SDP"])
def test_a_detection_set_is_a_parameter_of_the_run(tmp_path, variant):
    zip_path, manifest = archive(tmp_path, variant)
    run = mf.run_feasibility(CFG, zip_path, manifest, sequences=[SEQ], variant=variant)
    assert run.settings["detection_variant"] == variant and variant in run.settings["detections"]


def test_a_missing_detection_set_is_refused(tmp_path):
    zip_path, manifest = archive(tmp_path, "FRCNN")
    with pytest.raises(mot17.Mot17FormatError, match="DPM"):
        mf.run_feasibility(CFG, zip_path, manifest, sequences=[SEQ], variant="DPM")


def test_the_default_keeps_the_m1_settings():
    settings = mf.run_settings(CFG, mf.static_sequences(CFG), mf.TrackerConfig(), 0.0, None, {}, {})
    assert settings["detection_variant"] == "FRCNN" and settings["camera_group"].startswith("static (frozen")
    assert settings["detections"] == mf.DETECTIONS_NOTE


def test_a_run_over_both_camera_groups_says_so():
    assert mf.camera_group(CFG, me.e1_sequences(CFG)) == "static and moving (frozen.datasets.mot17)"


def test_trackeval_runs_in_its_own_interpreter_with_the_pinned_script(tmp_path, monkeypatch):
    seen = {}

    def fake(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0, "out", "err")
    monkeypatch.setattr(mt.subprocess, "run", fake)
    assert mt.run_trackeval(tmp_path, Path("py311.exe"), Path("source"), CFG) == 0
    assert seen["command"][:3] == ["py311.exe", "-I", str(mt.SCRIPT)]
    assert json.loads((tmp_path / "trackeval_config.json").read_text(encoding="utf-8"))["trackeval_source"] == "source"


def test_official_values_are_parsed_as_means_over_the_nineteen_alphas():
    n = 19
    series = {f: list(np.linspace(0.0, 1.0, n)) for f in mt.HOTA_FIELDS}
    clear = {k: 1 for k in ("MOTA", "MOTP", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "MT", "PT", "ML", "CLR_Frames")}
    identity = {k: 1 for k in ("IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP")}
    h = mt.headline({"HOTA": series, "CLEAR": clear, "Identity": identity, "Count": dict.fromkeys(("Dets", "GT_Dets", "IDs", "GT_IDs"), 2)})
    assert h["HOTA"] == pytest.approx(0.5) and h["IDSW"] == 1 and h["GT_Dets"] == 2


def test_a_failed_trackeval_stops_the_run(tmp_path, monkeypatch):
    zip_path, manifest = archive(tmp_path)
    monkeypatch.setattr(mt, "run_trackeval", lambda *a, **k: 3)
    with pytest.raises(RuntimeError, match="exited with code 3"):
        me.run(tmp_path / "e1", CFG, Path("p"), Path("s"), ("FRCNN",), zip_path, manifest, [SEQ])


def test_results_separate_official_from_project_and_are_never_pooled(tmp_path, monkeypatch):
    zip_path, manifest = archive(tmp_path)
    stub_official(monkeypatch)
    doc = me.run(tmp_path / "e1", CFG, Path("p"), Path("s"), ("FRCNN",), zip_path, manifest, [SEQ])
    row = doc["by_detection_set"]["FRCNN"][SEQ]
    assert set(row) == {"frames_evaluated", "frame_rate", "official_trackeval", "project_diagnostics"}
    assert set(row["project_diagnostics"]) == {"detection", "error_profile"} and "IDSW" not in row["project_diagnostics"]
    assert doc["pooled_result"] is None and doc["research_results"] is False and doc["cadence"] == "native"
    assert doc["not_p1"] is True and doc["not_p9"] is True and "Not P-1" in doc["label"] and "not a full MOT17" in doc["label"]
    assert doc["association_mechanism"] == "real_tracker_iou" and "no claim of P-1 or P-9" in doc["association_note"]


def test_a_run_never_overwrites_and_needs_the_primary_set(tmp_path, monkeypatch):
    zip_path, manifest = archive(tmp_path)
    with pytest.raises(ValueError, match="FRCNN"):
        me.run(tmp_path / "x", CFG, Path("p"), Path("s"), ("DPM",), zip_path, manifest, [SEQ])
    (tmp_path / "e1").mkdir()
    with pytest.raises(FileExistsError):
        me.run(tmp_path / "e1", CFG, Path("p"), Path("s"), ("FRCNN",), zip_path, manifest, [SEQ])


@needs_trackeval
def test_the_real_trackeval_scores_the_fixture_end_to_end(tmp_path):
    zip_path, manifest = archive(tmp_path)
    doc = me.run(tmp_path / "e1", CFG, Path(TE_PYTHON), Path(TE_SOURCE), ("FRCNN",), zip_path, manifest, [SEQ])
    official = doc["by_detection_set"]["FRCNN"][SEQ]["official_trackeval"]
    # Ten pedestrian observations, none missed; the car track adds five false positives: 1 - 5 / 10 = 0.5.
    assert official["MOTA"] == pytest.approx(0.5) and official["GT_Dets"] == 10 and official["IDSW"] == 0


# --- E2 ------------------------------------------------------------------------------------------

def test_detection_denominators_on_the_fixture():
    """Five frames of the second half. Two pedestrians, one static person (ignore) and one car (excluded), one
    detection on each: TP 10, ignored 5, and the 5 car boxes are false positives."""
    _, gt, det, cats = parsed()
    r = me.detection_diagnostics(gt, cats, det, (6, 10), 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["ignored_matches"]) == (10, 5, 0, 5) and r["detections"] == 20
    assert r["evaluable_gt_observations"] == 10 and r["precision"] == pytest.approx(10 / 15) and r["recall"] == 1.0
    assert r["denominators"]["recall"] == "tp + fn = evaluable_gt_observations"


def test_frames_without_detections_become_false_negatives():
    _, gt, det, cats = parsed(det_frames={6, 7})
    r = me.detection_diagnostics(gt, cats, det, (6, 10), 0.5)
    assert (r["tp"], r["fn"]) == (4, 6) and r["recall"] == pytest.approx(0.4)


def test_no_evaluable_ground_truth_leaves_recall_undefined():
    _, gt, det, cats = parsed()
    r = me.detection_diagnostics(gt, np.full(len(gt.frame), "excluded", dtype=object), det, (6, 10), 0.5)
    assert r["recall"] is None and r["precision"] == 0.0 and r["fp"] == 20


def test_detection_diagnostics_say_what_they_are():
    _, gt, det, cats = parsed()
    assert "not a frozen P-1 output" in me.detection_diagnostics(gt, cats, det, (6, 10), 0.5)["note"]


# --- E3 ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rate, steps, frames", [(30, 5, 60), (25, 5, 50), (14, 5, 28), (30, 1, 12), (25, 1, 10), (14, 1, 5)])
def test_gap_limits_keep_their_duration_in_whole_frames(rate, steps, frames):
    assert me.gap_steps_at(rate, steps, 0.4) == frames


def test_the_position_point_is_the_box_bottom_centre_in_pixels():
    assert me.bottom_centre([[10, 20, 30, 60]]).tolist() == [[20.0, 60.0]]


def test_e3_reports_pixels_never_metres(tmp_path):
    zip_path, manifest = archive(tmp_path)
    run = mf.run_feasibility(CFG, zip_path, manifest, sequences=[SEQ]).sequences[0]
    _, gt, _, cats = parsed()
    p = me.image_profile(run, gt, cats, CFG)
    assert p["localization"]["unit"] == "pixels" and p["cadence"] == "native" and p["position_point"].endswith("pixels")
    assert "not a section 3.7 implementation" in p["gap_conversion"]
    assert not any(k.endswith("_m") for k in p["localization"])
    assert p["identity_exchange"]["per_100"]["value"] is None and "metres" in p["identity_exchange"]["per_100"]["reason"]
    assert p["identity_exchange"]["denominator"] is None and "section 3.7 severities" in p["not_computed"]


def test_e3_counts_match_the_fixture(tmp_path):
    """Detections on both pedestrians in every frame, so ten eligible observations after confirmation; zero error."""
    zip_path, manifest = archive(tmp_path)
    run = mf.run_feasibility(CFG, zip_path, manifest, sequences=[SEQ]).sequences[0]
    _, gt, _, cats = parsed()
    p = me.image_profile(run, gt, cats, CFG)
    assert p["dropout"]["eligible_gt_observations"] == 10 and p["localization"]["rmse"] == pytest.approx(0.0, abs=1e-3)
    assert p["fragmentation"]["events"] == 0 and p["identity_exchange"]["events"] == 0


def test_no_prediction_module_is_reachable_from_e1():
    source = Path(me.__file__).read_text(encoding="utf-8")
    assert "trackvision.predict" not in source and "make_windows" not in source and "ade_fde" not in source
