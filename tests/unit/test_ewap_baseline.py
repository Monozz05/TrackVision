"""R1: trajectory-only CV baseline on EWAP scenes, on small synthetic EWAP files written to a temporary
directory. No test reads the real dataset files.

Hand scene (frame step 6, documented 0.4 s), with expected values derived by hand:
  ped 1  25 samples on a straight line                      -> 6 windows, ADE = FDE = 0
  ped 2  8 samples at 0.4 m/step, then 12 at 0.6 m/step      -> 1 window, error 0.2 k at step k:
                                                              ADE = 0.2 * 6.5 = 1.3, FDE = 0.2 * 12 = 2.4
  ped 3  15 samples, a 5-step gap, 22 samples, straight      -> 0 + 3 windows (none crosses the gap)
  ped 4  5 samples                                           -> no window
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fixtures import synthetic_ewap as se
from trackvision import config
from trackvision.experiments import ewap_baseline as eb
from trackvision.experiments.geometry_validation import LOCAL_NOTICE
from trackvision.experiments.synthetic_feasibility import protocol
from trackvision.io import ewap

CFG = config.validate().config
P = protocol(CFG)
H, F = P["history_len"], P["horizon_len"]
STEP = se.FRAME_STEP


def hand_rows():
    k = np.arange(25.0)
    ped1 = np.column_stack([0.3 * k, np.full(25, 1.0)])
    x2 = np.concatenate([0.4 * np.arange(8.0), 2.8 + 0.6 * np.arange(1.0, 13.0)])
    ped2 = np.column_stack([x2, np.full(20, 2.0)])
    run_a = np.column_stack([0.2 * np.arange(15.0), np.full(15, 3.0)])
    run_b = np.column_stack([4.0 + 0.2 * np.arange(22.0), np.full(22, 3.0)])
    ped4 = np.column_stack([0.3 * np.arange(5.0), np.full(5, 0.5)])
    return (se.track_rows(1, 780, ped1) + se.track_rows(2, 780, ped2) + se.track_rows(3, 780, run_a)
            + se.track_rows(3, 780 + STEP * 20, run_b) + se.track_rows(4, 783, ped4))


def write(tmp_path, name="ETH", rows=None):
    raw = tmp_path / "raw"
    se.write_scene(raw, name=name, rows=hand_rows() if rows is None else rows)
    manifest_path = tmp_path / "ewap.sha256"
    se.write_manifest(raw, manifest_path)
    return raw, manifest_path


def tree_digest(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_hand_scene_counts_and_errors(tmp_path):
    raw, man = write(tmp_path)
    doc = eb.results_document(eb.run_baseline(CFG, "ETH", raw, man))
    s = doc["summary"]
    assert (s["n_tracks"], s["n_windows"], s["n_pedestrians_contributing"]) == (4, 6 + 1 + 3, 3)
    assert s["coverage"]["prediction_coverage"] == 1.0
    assert s["coverage"]["eligible_windows"] == s["coverage"]["predicted_windows"] == 10
    assert s["mean_ade_m"] == pytest.approx(1.3 / 10, abs=1e-8)
    assert s["mean_fde_m"] == pytest.approx(2.4 / 10, abs=1e-8)
    np.testing.assert_allclose(s["mean_error_by_step_m"], 0.2 * np.arange(1, F + 1) / 10, atol=1e-8)
    np.testing.assert_allclose(s["median_error_by_step_m"], 0.0, atol=1e-8)
    assert len(s["mean_error_by_step_m"]) == F
    per = {r["ped_id"]: r for r in doc["per_pedestrian"]}
    assert [(per[i]["samples"], per[i]["runs"], per[i]["windows"]) for i in (1, 2, 3, 4)] == \
           [(25, 1, 6), (20, 1, 1), (37, 2, 3), (5, 1, 0)]
    assert per[2]["mean_ade_m"] == pytest.approx(1.3, abs=1e-8)
    assert per[2]["mean_fde_m"] == pytest.approx(2.4, abs=1e-8)
    assert per[1]["mean_ade_m"] == pytest.approx(0.0, abs=1e-8) and per[4]["mean_ade_m"] is None

    t = doc["track_lengths"]
    assert t["window_length_samples"] == H + F
    assert t["histogram_samples"] == {f"[1,{H})": 1, f"[{H},{H + F})": 0, f"[{H + F},{2 * (H + F)})": 3,
                                      f"[{2 * (H + F)},{4 * (H + F)})": 0, f">={4 * (H + F)}": 0}
    assert (t["samples"]["min"], t["samples"]["max"], t["samples"]["median"]) == (5, 37, 22.5)
    assert t["longest_consecutive_run"]["max"] == 25
    assert t["tracks_with_internal_gaps"] == 1 and t["tracks_with_a_window_length_run"] == 3
    assert t["span_s"]["max"] == pytest.approx((20 + 21) * P["sampling_interval_s"])


def test_windows_are_the_original_annotations(tmp_path):
    raw, man = write(tmp_path)
    run = eb.run_baseline(CFG, "ETH", raw, man)
    obsmat = np.loadtxt(raw / "ewap_dataset" / "seq_eth" / "obsmat.txt")   # read independently
    lookup = {(int(r[1]), int(r[0])): (r[2], r[4]) for r in obsmat}
    w = run.evaluation.windows
    assert w.observed.shape[1:] == (H, 2) and w.target.shape[1:] == (F, 2)
    for i in range(len(w)):
        assert np.all(np.diff(w.frames[i]) == STEP)
        expected = [lookup[(int(w.track_ids[i]), int(f))] for f in w.frames[i]]
        np.testing.assert_array_equal(np.concatenate([w.observed[i], w.target[i]]), expected)
    # ped 3's windows all lie in its second run; none spans the gap
    assert all(f[0] >= 780 + STEP * 20 for f, g in zip(w.frames, w.track_ids) if g == 3)
    assert all(not t.positions.flags.writeable for t in run.scene.tracks)


def test_protocol_values_come_from_config(tmp_path):
    raw, man = write(tmp_path)
    run = eb.run_baseline(CFG, "ETH", raw, man)
    assert run.settings["protocol"] == P
    data = CFG.canonical_dict()
    data["frozen"]["protocol"]["static"]["horizon_len"] = 8           # a different configured horizon
    other = SimpleNamespace(frozen=data["frozen"], hash="0" * 64, meta=data["meta"])
    alt = eb.run_baseline(other, "ETH", raw, man)
    assert alt.evaluation.windows.target.shape[1] == 8
    # 16-sample windows: ped 1 25 - 15, ped 2 20 - 15, ped 3 run b 22 - 15
    assert len(alt.evaluation.windows) == 10 + 5 + 7
    assert alt.experiment_id != run.experiment_id


def test_eth_timing_discrepancy_is_recorded_not_resolved(tmp_path):
    raw, man = write(tmp_path / "eth")
    eth = eb.results_document(eb.run_baseline(CFG, "ETH", raw, man))
    timing = eth["timing"]
    assert "UNRESOLVED" in timing["note"] and "25 fps" in timing["note"] and timing["note"] == eb.TIMING_NOTES["ETH"]
    assert timing["documented_timestep_s"] == timing["configured_sampling_interval_s"] == P["sampling_interval_s"]
    assert timing["frames_per_step"] == STEP
    assert timing["implied_video_fps"] == pytest.approx(STEP / P["sampling_interval_s"])
    assert eth["not_used"]["video"] == "not read"

    raw_h, man_h = write(tmp_path / "hotel", name="HOTEL")
    hotel = eb.results_document(eb.run_baseline(CFG, "HOTEL", raw_h, man_h))
    assert hotel["timing"]["note"] == eb.TIMING_NOTES["HOTEL"] != eb.TIMING_NOTES["ETH"]
    assert hotel["scene"] == "HOTEL" and hotel["settings"]["source_files"]["obsmat"].startswith("ewap_dataset/seq_hotel/")


def test_timestep_differing_from_config_is_refused(tmp_path):
    se.write_scene(tmp_path, rows=hand_rows(), info="The annotation was done with a timestep of 0.24 seconds.")
    with pytest.raises(ewap.EwapFormatError, match="differs from the frozen sampling interval"):
        eb.run_baseline(CFG, "ETH", tmp_path, manifest_path=None)


def test_labelled_trajectory_only_and_local_only(tmp_path):
    raw, man = write(tmp_path)
    doc = eb.results_document(eb.run_baseline(CFG, "ETH", raw, man))
    assert doc["evaluation_type"] == doc["settings"]["evaluation_type"] == "trajectory_only"
    assert "Trajectory-only" in doc["label"] and "No detector or tracker" in doc["label"]
    assert "LOCAL ONLY" in doc["label"] and doc["notice"] == LOCAL_NOTICE
    assert not {"detector", "tracker", "render"} & set(doc["settings"])
    assert set(doc["assumptions"]) == set(ewap.ASSUMPTIONS)


def test_save_is_reproducible_and_source_files_unchanged(tmp_path):
    raw, man = write(tmp_path)
    before = tree_digest(raw)
    run = eb.run_baseline(CFG, "ETH", raw, man)
    out = eb.save_run(tmp_path / "run1", run, CFG)
    assert sorted(p.name for p in out.iterdir()) == ["config.resolved.yaml", "manifest.json", "results.json",
                                                     "windows.csv"]
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["experiment_id"] == run.experiment_id and m["notice"] == LOCAL_NOTICE
    assert m["dataset_manifest_hash"] == hashlib.sha256(man.read_bytes()).hexdigest() and m["manifest_verified"]
    assert m["source_sha256"] == run.scene.provenance["source_sha256"]
    assert m["source_sha256"]["ewap_dataset/seq_eth/obsmat.txt"] == before["ewap_dataset/seq_eth/obsmat.txt"]
    assert m["timing_note"] == eb.TIMING_NOTES["ETH"]
    for name, digest in m["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest

    header = (out / "windows.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[:6] == ["ped_id", "first_frame", "anchor_frame", "last_frame", "ade_m", "fde_m"]
    assert len(header) == 6 + F and not any("pos" in c or c in ("x", "y") for c in header)

    with pytest.raises(FileExistsError):
        eb.save_run(out, run, CFG)
    again = eb.save_run(tmp_path / "run2", eb.run_baseline(CFG, "ETH", raw, man), CFG)
    for name in ("results.json", "windows.csv", "config.resolved.yaml"):
        assert (again / name).read_bytes() == (out / name).read_bytes()
    assert tree_digest(raw) == before


def test_output_inside_repository_is_refused(tmp_path):
    raw, man = write(tmp_path)
    run = eb.run_baseline(CFG, "ETH", raw, man)
    target = config.REPO_ROOT / "r1-output-must-not-be-created"
    with pytest.raises(ValueError, match="inside the repository"):
        eb.save_run(target, run, CFG)
    assert not target.exists()
    eb.check_output_location(config.REPO_ROOT / "runs" / "r1-local")      # the Git-ignored runs/ is allowed
    eb.check_output_location(tmp_path / "elsewhere")


def test_experiment_id_tracks_the_data(tmp_path):
    raw, man = write(tmp_path / "a")
    same_raw, same_man = write(tmp_path / "b")
    assert eb.run_baseline(CFG, "ETH", raw, man).experiment_id == \
           eb.run_baseline(CFG, "ETH", same_raw, same_man).experiment_id
    rows = hand_rows()
    rows[0][2] += 0.01                                                    # one annotation differs
    other_raw, other_man = write(tmp_path / "c", rows=rows)
    assert eb.run_baseline(CFG, "ETH", other_raw, other_man).experiment_id != \
           eb.run_baseline(CFG, "ETH", raw, man).experiment_id


def test_scene_without_windows(tmp_path):
    walk = np.column_stack([0.3 * np.arange(6.0), np.full(6, 1.0)])
    raw, man = write(tmp_path, rows=se.track_rows(1, 780, walk) + se.track_rows(2, 786, walk + [0.0, 2.0]))
    run = eb.run_baseline(CFG, "ETH", raw, man)
    s = eb.results_document(run)["summary"]
    assert (s["n_tracks"], s["n_windows"], s["n_pedestrians_contributing"]) == (2, 0, 0)
    assert s["coverage"]["prediction_coverage"] is None
    assert s["mean_ade_m"] is None and s["ade_m"] is None and s["mean_error_by_step_m"] is None
    out = eb.save_run(tmp_path / "run", run, CFG)
    assert (out / "windows.csv").read_text(encoding="utf-8").count("\n") == 1
