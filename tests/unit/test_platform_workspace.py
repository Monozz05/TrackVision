"""Exploratory injection and the scene workspace. Scenes here are small fixtures, and no value is a result."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from trackvision import config
from trackvision.experiments import evaluation, injection_feasibility as inj
from trackvision.platform.app import create_app
from trackvision.platform.config import Settings
from trackvision.platform.services import exploratory as ex, video_cache as vc, video_tracks as vt, workspace as ws
from trackvision.predict.windowing import static_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import GridTrack, ground_truth_grid_tracks

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
LEVELS = {"A": 0.2, "B1": 0.2, "B2": 0.2, "C": 0.1, "D": 0.1}


class Scene:
    name = "FIXTURE"

    def __init__(self):
        truth = generate(SceneConfig(seed=7, n_frames=1500, n_pedestrians=14), DT)
        self.tracks, self.frame_step = ground_truth_grid_tracks(truth), truth.frames_per_step


@pytest.fixture(scope="module")
def scene():
    return Scene()


@pytest.fixture(scope="module")
def windows(scene):
    return static_windows(scene, CFG)


@pytest.fixture(scope="module")
def pop(scene, windows):
    return evaluation.scene_population(scene.tracks, windows)


def request_for(condition, level):
    return {"A": ex.ErrorRequest(a_sigma=level), "B1": ex.ErrorRequest(b_p=level, reconstruct=True),
            "B2": ex.ErrorRequest(b_p=level), "C": ex.ErrorRequest(c_p=level), "D": ex.ErrorRequest(d_p=level)}[condition]


@pytest.mark.parametrize("condition", list(LEVELS))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_error_gives_exactly_the_frozen_inputs(scene, pop, condition, seed):
    """The strongest check here: with one error active, the composed path equals `evaluation.corrupt` bit for bit."""
    frozen = evaluation.corrupt(scene.tracks, pop, condition, LEVELS[condition], seed, CFG, scene.frame_step)
    ours = ex.combine(scene.tracks, pop, request_for(condition, LEVELS[condition]), seed, CFG, scene.frame_step).corrupted
    assert np.array_equal(ours.predicted, frozen.predicted) and ours.reason == frozen.reason
    assert ours.inputs.shape == frozen.inputs.shape and np.array_equal(ours.inputs, frozen.inputs)


@pytest.mark.parametrize("condition", list(LEVELS))
def test_one_error_reports_the_frozen_achieved_severity(scene, pop, condition):
    frozen = evaluation.corrupt(scene.tracks, pop, condition, LEVELS[condition], 0, CFG, scene.frame_step)
    ours = ex.combine(scene.tracks, pop, request_for(condition, LEVELS[condition]), 0, CFG, scene.frame_step)
    assert ours.achieved[condition[0]]["value"] == frozen.value


def test_no_error_is_the_clean_population(scene, windows, pop):
    ours = ex.combine(scene.tracks, pop, ex.ErrorRequest(), 0, CFG, scene.frame_step).corrupted
    assert ours.predicted.all() and np.array_equal(ours.inputs, windows.observed)


def test_a_combination_leaves_clean_tracks_and_targets_untouched(scene, windows, pop):
    before = inj.tracks_digest(scene.tracks), hashlib.sha256(windows.target.tobytes()).hexdigest()
    ex.combine(scene.tracks, pop, ex.ErrorRequest(0.3, 0.2, 0.1, 0.1), 0, CFG, scene.frame_step)
    assert (inj.tracks_digest(scene.tracks), hashlib.sha256(windows.target.tobytes()).hexdigest()) == before


def test_each_error_in_a_combination_reports_its_own_achieved_severity(scene, pop):
    """A, B and C read separate random streams, so combining them does not change what each one does."""
    combined = ex.combine(scene.tracks, pop, ex.ErrorRequest(0.3, 0.2, 0.1), 0, CFG, scene.frame_step).achieved
    for code, condition, level in (("A", "A", 0.3), ("B", "B2", 0.2), ("C", "C", 0.1)):
        alone = ex.combine(scene.tracks, pop, request_for(condition, level), 0, CFG, scene.frame_step).achieved
        assert combined[code]["value"] == alone[code]["value"]


def test_combining_errors_never_raises_coverage(scene, pop):
    alone = [ex.combine(scene.tracks, pop, r, 0, CFG, scene.frame_step).corrupted.predicted.sum()
             for r in (ex.ErrorRequest(b_p=0.2), ex.ErrorRequest(c_p=0.1))]
    both = ex.combine(scene.tracks, pop, ex.ErrorRequest(b_p=0.2, c_p=0.1), 0, CFG, scene.frame_step).corrupted
    assert both.predicted.sum() <= min(alone)


def test_a_combined_window_is_lost_for_a_recorded_reason(scene, pop):
    corrupted = ex.combine(scene.tracks, pop, ex.ErrorRequest(0.2, 0.2, 0.1, 0.2), 0, CFG, scene.frame_step).corrupted
    reasons = {r for r in corrupted.reason if r is not None}
    assert reasons and reasons <= {"label_history_incomplete", "split_in_history", "missing_observation"}
    assert all((r is None) == p for r, p in zip(corrupted.reason, corrupted.predicted))


def test_noise_in_a_combination_lands_on_the_labelled_track(scene, pop):
    with_d = ex.combine(scene.tracks, pop, ex.ErrorRequest(d_p=0.2), 0, CFG, scene.frame_step)
    with_both = ex.combine(scene.tracks, pop, ex.ErrorRequest(a_sigma=0.3, d_p=0.2), 0, CFG, scene.frame_step)
    assert with_both.offsets and np.array_equal(with_both.corrupted.predicted, with_d.corrupted.predicted)
    assert not np.array_equal(with_both.corrupted.inputs, with_d.corrupted.inputs)


def test_at_total_dropout_metrics_are_undefined_rather_than_zero(scene, windows, pop):
    predictor = evaluation.cv_predictor(CFG, windows.horizon_len)
    combined = {0: ex.combine(scene.tracks, pop, ex.ErrorRequest(b_p=1.0), 0, CFG, scene.frame_step)}
    row = ex.summarise(windows, (predictor,), {"CV": evaluation.clean_errors(predictor, windows)}, combined)["CV"]
    assert row["ade"] is None and row["de_by_step"] is None and row["ade_change"] is None
    assert row["prediction_coverage"] == 0.0 and row["undefined_reason"] == "no_predicted_windows"
    assert row["clean_ade"] > 0


def test_summary_uses_the_frozen_clean_reference(scene, windows, pop):
    predictor = evaluation.cv_predictor(CFG, windows.horizon_len)
    errors = evaluation.clean_errors(predictor, windows)
    combined = {0: ex.combine(scene.tracks, pop, ex.ErrorRequest(a_sigma=0.2), 0, CFG, scene.frame_step)}
    row = ex.summarise(windows, (predictor,), {"CV": errors}, combined)["CV"]
    assert row["clean_ade"] == pytest.approx(float(errors.mean()))
    assert row["ade_change"] == pytest.approx((row["ade"] - row["clean_ade"]) / row["clean_ade"])


@pytest.mark.parametrize("condition", list(LEVELS))
def test_a_unit_changes_no_arithmetic(scene, pop, condition):
    """Pixels run through the same injectors: only the name of the measure differs."""
    request = request_for(condition, LEVELS[condition])
    metres = ex.combine(scene.tracks, pop, request, 0, CFG, scene.frame_step)
    pixels = ex.combine(scene.tracks, pop, request, 0, CFG, scene.frame_step, unit="px",
                        d_max=float(CFG.frozen["injection"]["d_max_m"]))
    assert np.array_equal(metres.corrupted.inputs, pixels.corrupted.inputs)
    assert metres.corrupted.reason == pixels.corrupted.reason
    assert pixels.achieved[condition[0]]["value"] == metres.achieved[condition[0]]["value"]


def test_only_active_errors_are_listed_with_the_identity_error_first():
    assert ex.ErrorRequest().active == ()
    assert ex.ErrorRequest(0.1, 0.1, 0.1, 0.1).active == ("D", "A", "B2", "C")
    assert ex.ErrorRequest(b_p=0.1, reconstruct=True).active == ("B1",)


# --- the API ---------------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    runs, raw, manifests = tmp_path / "runs", tmp_path / "raw", tmp_path / "manifests"
    runs.mkdir(), raw.mkdir(), manifests.mkdir()
    settings = Settings(runs_dir=runs, manifests_dir=manifests, base_config=config.DEFAULT_BASE_PATH, raw_dir=raw,
                        cache_dir=tmp_path / "cache")
    return TestClient(create_app(settings), raise_server_exceptions=False)


def test_the_catalogue_lists_every_dataset_and_what_each_scene_can_do(client):
    body = client.get("/api/workspace/scenes").json()
    scenes = {s["id"]: s for s in body["items"]}
    assert {s["dataset"] for s in scenes.values()} == {"EWAP", "UCY", "MOT17", "Synthetic"}
    assert scenes["ETH"]["errors_available"] and scenes["ETH"]["predictors"] == ["CV", "CA"]
    assert scenes["SYN-501"]["errors_available"]
    assert not scenes["UNIV"]["errors_available"] and "not on this machine" in scenes["UNIV"]["errors_reason"]
    assert not scenes["MOT17-02"]["errors_available"] and "not on this machine" in scenes["MOT17-02"]["errors_reason"]
    assert not scenes["ETH"]["has_video"], "the fixture raw folder is empty"


def test_the_sliders_know_the_frozen_protocol_range(client):
    controls = client.get("/api/workspace/scenes").json()["controls"]
    grids = CFG.frozen["injection"]["grids"]
    assert controls["A"]["protocol_max"] == max(grids["A"]["levels"]) and controls["D"]["protocol_max"] == max(grids["D"]["levels"])
    assert controls["B"]["max"] == 1.0 and controls["A"]["unit"] == "m"


def test_no_slider_claims_a_tested_range_in_pixels(client):
    controls = client.get("/api/workspace/scenes").json()["pixel_controls"]
    assert controls["A"] == {"max": ws.MAX_SIGMA_PX, "unit": "px", "protocol_max": None}
    assert all(controls[code]["protocol_max"] is None and controls[code]["max"] == 1.0 for code in "BCD")


def test_regimes_come_from_the_frozen_partition(client):
    scenes = {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}
    assert scenes["MOT17-02"]["regime"] == "static" and scenes["MOT17-05"]["regime"] == "moving"


def test_injection_is_labelled_exploratory_and_returns_the_overlay(client):
    body = client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.2, "b": 0.1}).json()
    assert body["exploratory"] is True and "not a research result" in body["label"] and "not P-10" in body["label"]
    assert body["active"] == ["A", "B2"] and set(body["achieved"]) == {"A", "B"} and set(body["predictors"]) == {"CV", "CA"}
    assert len(body["achieved"]["A"]["by_seed"]) == len(CFG.research_choice("seeds"))
    track = body["overlay"]["tracks"][0]
    assert len(track["frames"]) == len(track["uv"]) and "dropped" in track
    assert body["overlay"]["predicted"]["CV"] and body["overlay"]["predicted"]["CA"]


def test_one_error_is_labelled_exploratory_without_the_combination_warning(client):
    body = client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.2}).json()
    assert body["exploratory"] is True and "one error" in body["label"]


def test_clean_tracks_are_served_in_pixels(client):
    body = client.get("/api/workspace/scenes/SYN-501/tracks").json()
    assert body["history_len"] == 8 and body["source"] == "Scene generator"
    track = body["tracks"][0]
    assert len(track["uv"]) == len(track["boxes"]) == len(track["frames"]) and len(track["boxes"][0]) == 4


@pytest.mark.parametrize("scene", ["UNIV", "ZARA1", "MOT17-02"])
def test_scenes_without_their_annotations_refuse_injection_and_say_why(client, scene):
    response = client.get(f"/api/workspace/scenes/{scene}/inject")
    assert response.status_code == 409 and "not on this machine" in response.json()["detail"]


@pytest.mark.parametrize("scene", ["nothing", "..", "ETH%2F..%2F..", "SYN-999", "SYN-511", "ETH-0", "ETH-5", "MOT17-09-1"])
def test_unknown_scenes_are_404(client, scene):
    assert client.get(f"/api/workspace/scenes/{scene}/inject").status_code == 404


@pytest.mark.parametrize("params", [{"a": -0.1}, {"a": 1.5}, {"a": 61}, {"b": 1.1}, {"c": -1}, {"d": 2}, {"b": "x"}])
def test_out_of_range_controls_are_refused(client, params):
    assert client.get("/api/workspace/scenes/SYN-501/inject", params=params).status_code == 422


def test_every_new_route_is_still_read_only(client):
    assert client.post("/api/workspace/scenes/SYN-501/inject").status_code == 405
    assert client.delete("/api/workspace/scenes/SYN-501/video").status_code == 405


def test_injection_writes_nothing_under_runs(client, tmp_path):
    client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.2, "b": 0.1, "c": 0.1, "d": 0.1})
    assert list((tmp_path / "runs").iterdir()) == []


def test_a_scene_with_no_source_video_says_so(client):
    body = client.get("/api/workspace/scenes/ETH/video/status").json()
    assert body["state"] == "missing" and "not on this machine" in body["reason"]
    assert client.get("/api/workspace/scenes/ETH/video").status_code == 404


def test_the_stored_run_of_a_scene_comes_from_the_registry(client):
    assert {s["stored_run"] for s in client.get("/api/workspace/scenes").json()["items"]} == {None}


def test_the_platform_never_writes_into_raw_data():
    source = (REPO_ROOT / "src/trackvision/platform/services/video_cache.py").read_text(encoding="utf-8")
    assert "extract" not in source.replace("never extracted", "").replace("nothing is extracted", "")
    assert 'zipfile.ZipFile(archive)' in source and 'ZipFile(archive, "w"' not in source


# --- the video cache ---------------------------------------------------------------------------


def _clip(path, seconds=2):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=160x120:rate=25:duration={seconds}",
                    "-c:v", "mpeg4", str(path)], check=True)
    return path


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
def test_a_video_is_built_once_keeps_every_frame_and_leaves_the_source_alone(tmp_path):
    source = _clip(tmp_path / "clip.avi")
    before = source.read_bytes()
    cache = vc.VideoCache(tmp_path / "cache")
    calls = []

    def build(target):
        calls.append(target)
        vc.from_file(source)(target)

    assert cache.request("X", build)["state"] == "preparing"
    cache._pool.shutdown(wait=True)
    state = cache.status("X")
    assert state == {"state": "ready", "width": 160, "height": 120, "fps": 25.0, "frames": 50}
    assert cache.request("X", build)["state"] == "ready" and len(calls) == 1
    assert source.read_bytes() == before and not list((tmp_path / "cache").glob(".*"))


def test_a_failed_build_is_reported_and_leaves_nothing_behind(tmp_path):
    cache = vc.VideoCache(tmp_path / "cache")

    def build(target):
        target.write_bytes(b"partial")
        raise RuntimeError("no")

    cache.request("X", build)
    cache._pool.shutdown(wait=True)
    assert cache.status("X") == {"state": "failed"} and not cache.path("X").exists()
    assert not list((tmp_path / "cache").glob(".*"))


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
def test_video_is_served_with_range_support_for_seeking(tmp_path):
    raw = tmp_path / "raw"
    _clip(_parent(raw / ws.EWAP_VIDEO["ETH"]))
    (tmp_path / "runs").mkdir(), (tmp_path / "manifests").mkdir()
    client = TestClient(create_app(Settings(runs_dir=tmp_path / "runs", manifests_dir=tmp_path / "manifests", raw_dir=raw,
                                            cache_dir=tmp_path / "cache")))
    assert client.get("/api/workspace/scenes/ETH/video/status").json()["state"] == "preparing"
    client.app.state.workspace.videos._pool.shutdown(wait=True)
    assert client.get("/api/workspace/scenes/ETH/video/status").json()["state"] == "ready"
    whole = client.get("/api/workspace/scenes/ETH/video")
    assert whole.status_code == 200 and whole.headers["content-type"] == "video/mp4"
    part = client.get("/api/workspace/scenes/ETH/video", headers={"Range": "bytes=0-99", "Accept-Encoding": "gzip"})
    assert part.status_code == 206 and len(part.content) == 100 and part.content == whole.content[:100]
    assert "content-encoding" not in part.headers


def _parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_the_exploratory_label_is_not_stored_as_json_results_anywhere():
    """A guard on the module boundary: nothing in the workspace services opens a path under runs for writing."""
    for name in ("workspace.py", "exploratory.py"):
        source = (REPO_ROOT / "src/trackvision/platform/services" / name).read_text(encoding="utf-8")
        assert "write_text" not in source and "atomic_write" not in source


# --- stored detections -------------------------------------------------------------------------


def _detect_module():
    spec = importlib.util.spec_from_file_location("yolo_detect_mot17", REPO_ROOT / "scripts" / "yolo_detect_mot17.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def detect():
    return _detect_module()


def _boxes(frames=2, scene="MOT17-02"):
    return {"schema": "trackvision-detections/1", "scene": scene, "label": "fixture", "model": "yolo11m", "frames": frames,
            "imgsz": 1280, "confidence_threshold": 0.25, "device": "cpu", "classes": {"0": "person"},
            "by_frame": [[[0.1, 0.2, 0.3, 0.6, 0.9, 0]] for _ in range(frames)]}


def _write_boxes(tmp_path, doc, scene="MOT17-02"):
    folder = tmp_path / "cache-boxes"
    folder.mkdir(exist_ok=True)
    (folder / f"{scene}.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder


def _client_with_boxes(tmp_path, folder):
    for name in ("runs", "raw", "manifests"):
        (tmp_path / name).mkdir(exist_ok=True)
    return TestClient(create_app(Settings(runs_dir=tmp_path / "runs", manifests_dir=tmp_path / "manifests", raw_dir=tmp_path / "raw",
                                          cache_dir=tmp_path / "cache", detections_dir=folder)), raise_server_exceptions=False)


def _mot17_archive(path, length=6, rate=30, truth=None):
    """A tiny sequence: two people walking apart, one low-score false detection, and a gap in person 1."""
    folder = "MOT17/train/MOT17-02-FRCNN/"
    info = (f"[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate={rate}\nseqLength={length}\nimWidth=1920\n"
            "imHeight=1080\nimExt=.jpg\n")
    rows = []
    for frame in range(1, 7):
        rows.append(f"{frame},-1,{100 + 4 * frame},100,100,300,0.9")
        rows.append(f"{frame},-1,{900 - 4 * frame},200,120,360,0.8")
    rows.append("3,-1,1500,500,50,50,0.2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(folder + "seqinfo.ini", info)
        archive.writestr(folder + "det/det.txt", "\n".join(rows))
        if truth is not None:
            archive.writestr(folder + "gt/gt.txt", "\n".join(truth))
    return path


def _client_with_raw(tmp_path, detections=None):
    for name in ("runs", "raw", "manifests"):
        (tmp_path / name).mkdir(exist_ok=True)
    return TestClient(create_app(Settings(runs_dir=tmp_path / "runs", manifests_dir=tmp_path / "manifests", raw_dir=tmp_path / "raw",
                                          cache_dir=tmp_path / "cache", detections_dir=detections or tmp_path / "none")),
                      raise_server_exceptions=False)


def test_a_box_sits_around_the_annotated_point_by_the_scene_rule():
    feet = vt.BoxModel((20, 20), (100, 100), 1.0, 100)
    head = vt.BoxModel((20, 20), (100, 100), 0.0, 100)
    assert feet.boxes([[50, 60]]).tolist() == [[40, -40, 60, 60]]
    assert head.boxes([[50, 60]]).tolist() == [[40, 60, 60, 160]]


def test_a_box_grows_towards_the_bottom_of_the_picture():
    model = vt.BoxModel((10, 30), (100, 200), 0.5, 100)
    top, bottom = model.boxes([[50, 0], [50, 100]])
    assert top[2] - top[0] == 10 and bottom[2] - bottom[0] == 30 and top[3] - top[1] == 100 and bottom[3] - bottom[1] == 200


def test_every_annotated_scene_has_a_box_rule():
    assert set(vt.SCENE_BOXES) == set(ws.EWAP_VIDEO) | set(ws.UCY_VIDEO)


VSP = """2 - the number of splines
2 - Num of control points
10.000000 -20.000000 0 90.000000 - (2D point, m_id)
30.000000 20.000000 20 90.000000 - (2D point, m_id)
3 - Num of control points
0.000000 0.000000 5 0.000000
0.000000 0.000000 5 0.000000
100.000000 100.000000 15 0.000000
"""


def test_ucy_splines_are_parsed_in_order():
    people = vt.parse_vsp(VSP)
    assert len(people) == 2 and people[0].tolist() == [[10, -20, 0], [30, 20, 20]] and len(people[1]) == 3


@pytest.mark.parametrize("text", ["", "1\n3\n0 0 0 0\n"])
def test_a_broken_annotation_file_is_refused(text):
    with pytest.raises(ValueError):
        vt.parse_vsp(text)


def test_ucy_positions_become_image_pixels_with_y_pointing_down(tmp_path):
    path = tmp_path / "a.vsp"
    path.write_text(VSP, encoding="latin-1")
    first, second = vt.ucy_tracks(path, 720, 576)
    assert first.ped_id == 1 and first.frames.tolist() == [0, 10, 20]
    assert first.positions.tolist() == [[370.0, 308.0], [380.0, 288.0], [390.0, 268.0]]
    assert second.ped_id == 2 and second.frames.tolist() == [10] and second.positions.tolist() == [[410.0, 238.0]]


def test_ucy_people_share_one_frame_grid_however_their_splines_start(tmp_path):
    """Each spline starts on its own frame, and an identity switch needs two people sampled on the same frame."""
    path = tmp_path / "a.vsp"
    path.write_text(_vsp(_walker(3, 45, 10), _walker(7, 45, -20)), encoding="latin-1")
    tracks = vt.ucy_tracks(path, 720, 576)
    assert all(np.all(t.frames % vt.UCY_FRAME_STEP == 0) for t in tracks)
    assert tracks[0].frames.tolist() == tracks[1].frames.tolist() == [10, 20, 30, 40]


def test_a_spline_too_short_to_reach_the_grid_is_left_out(tmp_path):
    path = tmp_path / "a.vsp"
    path.write_text(_vsp([(0, 0, 11), (5, 5, 19)], _walker(0, 30, 10)), encoding="latin-1")
    assert [t.ped_id for t in vt.ucy_tracks(path, 720, 576)] == [2]


def test_ucy_tracks_are_served_with_their_true_identities(tmp_path):
    path = tmp_path / "raw" / ws.UCY_ANNOTATIONS["UNIV"]
    path.parent.mkdir(parents=True)
    path.write_text(VSP, encoding="latin-1")
    body = _client_with_raw(tmp_path).get("/api/workspace/scenes/UNIV/tracks").json()
    assert body["source"] == "Human annotations" and [t["id"] for t in body["tracks"]] == [1, 2] and "estimate" in body["note"]


def test_tracks_say_so_when_their_source_is_missing(client):
    assert client.get("/api/workspace/scenes/UNIV/tracks").status_code == 404
    assert client.get("/api/workspace/scenes/MOT17-02/tracks").status_code == 404


def test_mot17_is_tracked_with_stable_separate_ids(tmp_path):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE)
    body = _client_with_raw(tmp_path).get("/api/workspace/scenes/MOT17-02/tracks").json()
    assert body["source"] == "IoU tracker on MOT17 public FRCNN detections" and body["frame_step"] == 1
    assert sorted(t["id"] for t in body["tracks"]) == [1, 2]
    assert all(t["frames"] == [0, 1, 2, 3, 4, 5] for t in body["tracks"])


def test_mot17_boxes_are_detections_scaled_to_the_video(tmp_path):
    archive = _mot17_archive(tmp_path / "mot.zip")
    track = vt.mot17_tracks(archive, "MOT17-02")[0]
    scale = vt.VIDEO_HEIGHT / 1080
    # MOT17 pixels count from 1 and project pixels from 0, so the source box 104, 100 starts at 103, 99.
    assert track["boxes"][0] == [round(103 * scale, 1), round(99 * scale, 1), round(203 * scale, 1), round(399 * scale, 1)]
    assert track["uv"][0] == [round(153 * scale, 1), round(399 * scale, 1)]


def test_a_low_score_detection_never_becomes_a_person(tmp_path):
    tracks = vt.mot17_tracks(_mot17_archive(tmp_path / "mot.zip"), "MOT17-02")
    assert len(tracks) == 2 and not any(box[0] > 700 for t in tracks for box in t["boxes"])


def _yolo(frames=6, scene="MOT17-02"):
    person = [0.1, 0.1, 0.2, 0.5, 0.9, 0]
    bag = [0.7, 0.7, 0.8, 0.8, 0.9, 26]
    return {"schema": "trackvision-detections/1", "scene": scene, "model": "yolo11m", "weights_sha256": "abc", "frames": frames,
            "by_frame": [[person, bag] for _ in range(frames)]}


def _write_yolo(tmp_path, doc):
    folder = tmp_path / "yolo"
    folder.mkdir(exist_ok=True)
    (folder / "MOT17-02.json").write_text(json.dumps(doc), encoding="utf-8")
    return folder


def test_no_detector_is_reported_until_yolo_has_run(tmp_path):
    client = _client_with_raw(tmp_path)      # an empty detection folder, whatever this machine has generated
    assert {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}["MOT17-02"]["detector"] is None


def test_yolo_boxes_feed_the_tracker_when_they_exist_and_only_people_are_tracked(tmp_path):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE)
    client = _client_with_raw(tmp_path, _write_yolo(tmp_path, _yolo()))
    body = client.get("/api/workspace/scenes/MOT17-02/tracks").json()
    assert body["source"] == "IoU tracker on yolo11m detections" and len(body["tracks"]) == 1
    assert {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}["MOT17-02"]["detector"] == "yolo11m"


@pytest.mark.parametrize("change", [{"schema": "other"}, {"scene": "MOT17-04"}, {"frames": 3}, {"by_frame": "x"}])
def test_a_wrong_detection_file_falls_back_to_the_public_detections(tmp_path, change):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE)
    client = _client_with_raw(tmp_path, _write_yolo(tmp_path, {**_yolo(), **change}))
    assert "public FRCNN" in client.get("/api/workspace/scenes/MOT17-02/tracks").json()["source"]
    assert {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}["MOT17-02"]["detector"] is None


def _synthetic_truth():
    return generate(SceneConfig(seed=501, n_frames=ws.SYNTHETIC_FRAMES, n_pedestrians=ws.SYNTHETIC_PEOPLE), DT)


def test_synthetic_scene_sizes_match_the_stored_experiments():
    from trackvision.experiments.learned_feasibility import N_FRAMES, N_PEDESTRIANS, SCENE_SEEDS, scene_name
    assert (ws.SYNTHETIC_FRAMES, ws.SYNTHETIC_PEOPLE) == (N_FRAMES, N_PEDESTRIANS)
    stored = {scene_name(seed): seed for seed in SCENE_SEEDS}
    assert list(ws.SYNTHETIC_SEEDS.items())[:len(stored)] == list(stored.items())
    assert list(ws.SYNTHETIC_SEEDS.values())[len(stored):] == list(range(506, 511)), "new scenes never reuse a stored seed"


def test_synthetic_boxes_are_the_true_boxes(client):
    truth = _synthetic_truth()
    index = {(int(p), int(f)): box for p, f, box in zip(truth.ped_id, truth.frame, truth.bbox_xyxy)}
    tracks = client.get("/api/workspace/scenes/SYN-501/tracks").json()["tracks"]
    for track in tracks[:6]:
        for frame, box in zip(track["frames"], track["boxes"]):
            assert box == pytest.approx(index[(track["id"], frame)].tolist(), abs=0.06)


def test_an_identity_switch_moves_the_id_to_the_other_persons_box(client):
    """The core of the feature: after a switch, a label's box is exactly where the other person's clean box is."""
    clean = {t["id"]: t for t in client.get("/api/workspace/scenes/SYN-501/tracks").json()["tracks"]}
    overlay = client.get("/api/workspace/scenes/SYN-501/inject", params={"d": 0.1}).json()["overlay"]
    switched = {t["id"]: t for t in overlay["tracks"]}
    events = [e for e in overlay["events"] if e["kind"] == "switch"]
    assert events
    for event in events:
        a, b, frame = event["a"], event["b"], event["frame"]
        before = dict(zip(clean[a]["frames"], clean[a]["boxes"]))
        other = dict(zip(clean[b]["frames"], clean[b]["boxes"]))
        for at, box in zip(switched[a]["frames"], switched[a]["boxes"]):
            assert box == pytest.approx((other[at] if at >= frame else before[at]), abs=1e-9)


def test_identity_switches_are_never_pushed_off_the_timeline_by_breaks(client, monkeypatch):
    monkeypatch.setattr(ws, "MAX_EVENTS", 2)
    overlay = client.get("/api/workspace/scenes/SYN-501/inject", params={"c": 0.3, "d": 0.2}).json()["overlay"]
    assert sum(e["kind"] == "switch" for e in overlay["events"]) > 2 and overlay["events_total"] > len(overlay["events"])


def test_tracks_open_without_loading_torch():
    code = ("import sys; from trackvision.platform.app import create_app; from fastapi.testclient import TestClient;"
            "r = TestClient(create_app()).get('/api/workspace/scenes/SYN-501/tracks'); assert r.status_code == 200, r.text;"
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True, cwd=REPO_ROOT)


def test_frames_are_taken_in_order_from_the_right_sequence_only(detect):
    names = ["MOT17/train/MOT17-02-FRCNN/img1/000002.jpg", "MOT17/train/MOT17-02-FRCNN/img1/000001.jpg",
             "MOT17/train/MOT17-02-DPM/img1/000001.jpg", "MOT17/train/MOT17-04-FRCNN/img1/000001.jpg",
             "MOT17/train/MOT17-02-FRCNN/seqinfo.ini"]
    assert detect.image_members(names, "MOT17-02") == names[1::-1]


def test_a_box_becomes_fractions_of_the_image(detect):
    assert detect.normalise((192.0, 108.0, 960.0, 540.0), 0.87654, 3, 1920, 1080) == [0.1, 0.1, 0.5, 0.5, 0.877, 3]


def test_a_sequence_records_one_entry_per_frame_with_its_metadata(detect):
    answers = iter([(100, 50, [((10, 5, 20, 25), 0.9, 0)]), (100, 50, [])])
    doc = detect.detect_sequence([b"a", b"b"], lambda _: next(answers), {"scene": "MOT17-02", "model": "yolo11m"})
    assert doc["schema"] == detect.SCHEMA and doc["frames"] == 2 and doc["scene"] == "MOT17-02" and doc["image_width"] == 100
    assert doc["by_frame"] == [[[0.1, 0.1, 0.2, 0.5, 0.9, 0]], []] and "Not experiment P-1" in doc["label"]


def test_the_detection_script_output_is_accepted_by_the_api(tmp_path, detect):
    doc = detect.detect_sequence([b"a"], lambda _: (100, 50, [((10, 5, 20, 25), 0.9, 0)]), {"scene": "MOT17-02", "model": "yolo11m"})
    folder = tmp_path / "boxes"
    detect.write_atomic(folder / "MOT17-02.json", doc)
    assert [p.name for p in folder.iterdir()] == ["MOT17-02.json"]
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE)
    assert "yolo11m" in _client_with_raw(tmp_path, folder).get("/api/workspace/scenes/MOT17-02/tracks").json()["source"]


def test_the_detection_script_stays_outside_the_research_code():
    source = (REPO_ROOT / "scripts" / "yolo_detect_mot17.py").read_text(encoding="utf-8")
    assert "trackvision" not in source.replace("trackvision-detections", "").replace("imports nothing from\ntrackvision", "")
    assert "zipfile.ZipFile(args.archive)" in source and "extractall" not in source and '"sync": False' in source
    for path in (REPO_ROOT / "src").rglob("*.py"):
        assert "ultralytics" not in path.read_text(encoding="utf-8"), path


# --- scenes in pixels (decision 0011) ----------------------------------------------------------


def _vsp(*people):
    lines = [f"{len(people)} - the number of splines"]
    for points in people:
        lines.append(f"{len(points)} - Num of control points")
        lines.extend(f"{x} {y} {frame} 0.0 - (2D point, m_id)" for x, y, frame in points)
    return "\n".join(lines) + "\n"


def _walker(start, end, height):
    """A straight walk at one pixel per frame. Control points are 30 frames apart, as in the real files."""
    return [(-300 + f, height, f) for f in [*range(start, end, 30), end]]


def _ucy_client(tmp_path):
    """Two people walking side by side, close enough to switch, and one far away. Splines start on different frames."""
    path = tmp_path / "raw" / ws.UCY_ANNOTATIONS["UNIV"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_vsp(_walker(3, 303, 10), _walker(7, 303, -20), _walker(1, 301, -280)), encoding="latin-1")
    return _client_with_raw(tmp_path)


TOP = {1: 100, 2: 150, 3: 700}


def _gt_rows(length=300):
    """People 1 and 2 walk side by side, person 3 walks far away, and two rows must never become people."""
    rows = []
    for frame in range(1, length + 1):
        for person, top in TOP.items():
            rows.append(f"{frame},{person},{100 + 2 * frame},{top},100,300,1,1,1")
        rows.append(f"{frame},8,500,500,100,300,0,1,1")          # a person flagged not to be counted
        rows.append(f"{frame},9,800,500,200,100,1,3,1")          # a car
    return rows


def _feet(person, frame):
    """The foot point of a video frame in video pixels. MOT17 counts frames and pixels from 1."""
    left, top = 100 + 2 * (frame + 1) - 1, TOP[person] - 1
    return np.array([left + 50, top + 300]) * (vt.VIDEO_HEIGHT / 1080)


def _mot17_client(tmp_path, rate=30):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE, length=300, rate=rate, truth=_gt_rows())
    return _client_with_raw(tmp_path)


def test_ucy_and_mot17_offer_errors_in_pixels_with_their_caution(tmp_path):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE, length=300, truth=_gt_rows())
    scenes = {s["id"]: s for s in _ucy_client(tmp_path).get("/api/workspace/scenes").json()["items"]}
    for name in ("UNIV", "MOT17-02", "MOT17-05"):
        assert scenes[name]["errors_available"] and scenes[name]["unit"] == "px" and scenes[name]["predictors"] == ["CV", "CA"]
        assert "video pixels" in scenes[name]["caution"] and scenes[name]["errors_reason"] is None
    assert "6.1.1" not in scenes["UNIV"]["caution"] and "6.1.1" in scenes["MOT17-02"]["caution"]
    assert "camera moves" not in scenes["MOT17-02"]["caution"] and "camera moves" in scenes["MOT17-05"]["caution"]


def test_scenes_in_metres_keep_their_unit_and_the_frozen_switch_distance(client):
    scenes = {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}
    assert scenes["ETH"]["unit"] == "m" and scenes["SYN-501"]["unit"] == "m" and scenes["ETH"]["caution"] is None
    body = client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.2}).json()
    assert body["unit"] == "m" and body["achieved"]["A"]["measure"] == "rmse_pos_m"
    assert client.app.state.workspace._base("SYN-501").d_max == CFG.frozen["injection"]["d_max_m"]


def _keys(value):
    if isinstance(value, dict):
        return {k for k in value if isinstance(k, str)} | {key for v in value.values() for key in _keys(v)}
    return {key for v in value for key in _keys(v)} if isinstance(value, list) else set()


@pytest.mark.parametrize("make, scene", [(_ucy_client, "UNIV"), (_mot17_client, "MOT17-02")])
def test_a_pixel_result_never_carries_a_metre_field(tmp_path, make, scene):
    body = make(tmp_path).get(f"/api/workspace/scenes/{scene}/inject", params={"a": 10, "b": 0.1, "c": 0.1, "d": 0.5}).json()
    assert body["unit"] == "px" and not [key for key in _keys(body) if key.endswith("_m")]
    assert body["achieved"]["A"]["measure"] == "rmse_pos_px" and "video pixels" in body["label"]
    assert "not a research result" in body["label"] and body["exploratory"] is True


@pytest.mark.parametrize("make, scene", [(_ucy_client, "UNIV"), (_mot17_client, "MOT17-02")])
def test_a_pixel_scene_accepts_noise_in_pixels(tmp_path, make, scene):
    assert make(tmp_path).get(f"/api/workspace/scenes/{scene}/inject", params={"a": 60}).status_code == 200


@pytest.mark.parametrize("make, scene", [(_ucy_client, "UNIV"), (_mot17_client, "MOT17-02")])
def test_straight_walkers_are_predicted_exactly(tmp_path, make, scene):
    """CV and CA are exact on a straight constant walk, so any bias in the pixel path would show up here."""
    predictors = make(tmp_path).get(f"/api/workspace/scenes/{scene}/inject", params={"b": 0.01}).json()["predictors"]
    for row in predictors.values():
        assert row["clean_ade"] == pytest.approx(0, abs=1e-6) and row["clean_fde"] == pytest.approx(0, abs=1e-6)


def test_position_noise_is_achieved_in_pixels(tmp_path):
    achieved = _ucy_client(tmp_path).get("/api/workspace/scenes/UNIV/inject", params={"a": 10}).json()["achieved"]["A"]
    assert achieved["mean"] == pytest.approx(10 * np.sqrt(2), rel=0.1)


@pytest.mark.parametrize("make, scene, source", [(_ucy_client, "UNIV", {}), (_mot17_client, "MOT17-02", {"source": "truth"})])
def test_a_switch_moves_the_id_onto_the_other_persons_box_in_pixels(tmp_path, make, scene, source):
    client = make(tmp_path)
    clean = {t["id"]: t for t in client.get(f"/api/workspace/scenes/{scene}/tracks", params=source).json()["tracks"]}
    overlay = client.get(f"/api/workspace/scenes/{scene}/inject", params={"d": 1.0}).json()["overlay"]
    events = [e for e in overlay["events"] if e["kind"] == "switch"]
    assert [sorted((e["a"], e["b"])) for e in events] == [[1, 2]], "only the two people walking together are close enough"
    switched = {t["id"]: t for t in overlay["tracks"]}
    frame = events[0]["frame"]
    for label, partner in ((1, 2), (2, 1)):
        mine, theirs = (dict(zip(clean[p]["frames"], clean[p]["boxes"])) for p in (label, partner))
        for at, box in zip(switched[label]["frames"], switched[label]["boxes"]):
            assert box == pytest.approx(theirs[at] if at >= frame else mine[at], abs=1e-9)


def test_the_switch_distance_is_one_person_height_in_pixels(tmp_path):
    client = _mot17_client(tmp_path)
    assert client.app.state.workspace._base("MOT17-02").d_max == pytest.approx(300 * vt.VIDEO_HEIGHT / 1080)
    client = _ucy_client(tmp_path)
    base = client.app.state.workspace._base("UNIV")
    assert base.unit == "px" and vt.SCENE_BOXES["UNIV"].height[0] < base.d_max < vt.SCENE_BOXES["UNIV"].height[1]


def test_mot17_ground_truth_is_only_its_counted_pedestrians_on_a_shared_grid(tmp_path):
    body = _mot17_client(tmp_path).get("/api/workspace/scenes/MOT17-02/tracks", params={"source": "truth"}).json()
    assert body["source"] == "MOT17 ground truth" and body["frame_step"] == 12
    assert sorted(t["id"] for t in body["tracks"]) == [1, 2, 3]
    for track in body["tracks"]:
        assert track["frames"] == list(range(0, 300, 12))
        for frame, uv, box in zip(track["frames"], track["uv"], track["boxes"]):
            assert uv == pytest.approx(_feet(track["id"], frame), abs=0.06)
            assert box[3] - box[1] == pytest.approx(300 * vt.VIDEO_HEIGHT / 1080, abs=0.11)


def test_mot17_targets_are_the_ground_truth_files(tmp_path):
    client = _mot17_client(tmp_path)
    client.get("/api/workspace/scenes/MOT17-02/inject", params={"b": 0.01})
    windows = client.app.state.workspace._load("MOT17-02").windows
    assert len(windows) == 3 * (25 - 19), "25 samples give 6 windows of 20 steps each"
    for person, frames, target in zip(windows.track_ids, windows.frames, windows.target):
        expected = np.array([_feet(int(person), int(f)) for f in frames[windows.history_len:]])
        assert target == pytest.approx(expected, abs=1e-9)


def test_mot17_still_shows_the_tracker_unless_asked_for_its_ground_truth(tmp_path):
    client = _mot17_client(tmp_path)
    tracker = client.get("/api/workspace/scenes/MOT17-02/tracks").json()
    truth = client.get("/api/workspace/scenes/MOT17-02/tracks", params={"source": "truth"}).json()
    assert tracker["source"].startswith("IoU tracker") and tracker["frame_step"] == 1
    assert truth["source"] == "MOT17 ground truth" and "annotated boxes" in truth["note"]


def test_mot17_without_ground_truth_says_so(tmp_path):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE)
    client = _client_with_raw(tmp_path)
    for path, params in (("tracks", {"source": "truth"}), ("inject", {"a": 10})):
        response = client.get(f"/api/workspace/scenes/MOT17-02/{path}", params=params)
        assert response.status_code == 404 and "not on this machine" in response.json()["detail"]


def test_a_sequence_that_cannot_keep_the_step_says_so(tmp_path):
    client = _mot17_client(tmp_path, rate=14)
    truth = client.get("/api/workspace/scenes/MOT17-02/tracks", params={"source": "truth"}).json()
    assert truth["frame_step"] == 6 and "0.43 s" in truth["note"]
    assert "0.43 s" in client.get("/api/workspace/scenes/MOT17-02/inject", params={"a": 10}).json()["label"]


def test_a_sequence_on_the_step_says_nothing_about_it(tmp_path):
    client = _mot17_client(tmp_path)
    assert "One step" not in client.get("/api/workspace/scenes/MOT17-02/tracks", params={"source": "truth"}).json()["note"]


def test_an_unknown_track_source_is_refused(client):
    assert client.get("/api/workspace/scenes/SYN-501/tracks", params={"source": "elsewhere"}).status_code == 422


@pytest.mark.parametrize("make, scene", [(_ucy_client, "UNIV"), (_mot17_client, "MOT17-02")])
def test_pixel_errors_write_nothing_under_runs(tmp_path, make, scene):
    make(tmp_path).get(f"/api/workspace/scenes/{scene}/inject", params={"a": 10, "b": 0.1, "c": 0.1, "d": 0.1})
    assert list((tmp_path / "runs").iterdir()) == []


# --- moving a slider several times -------------------------------------------------------------


def _inject_in_thread(workspace, request, outcome):
    def run():
        try:
            outcome.append(workspace.inject("SYN-501", request)["request"]["a_sigma"])
        except ws.SceneUnavailable as error:
            outcome.append(str(error))
    thread = threading.Thread(target=run)
    thread.start()
    return thread


def _wait_until(condition):
    end = time.monotonic() + 10
    while not condition() and time.monotonic() < end:
        time.sleep(0.005)
    assert condition()


def test_a_request_replaced_while_it_waits_is_dropped_and_the_newest_runs(client):
    workspace = client.app.state.workspace
    older, newer = ex.ErrorRequest(a_sigma=0.1), ex.ErrorRequest(a_sigma=0.2)
    first, second = [], []
    with workspace._running["SYN-501"]:                  # another calculation is in progress
        waiting = _inject_in_thread(workspace, older, first)
        _wait_until(lambda: workspace._latest.get("SYN-501") == older)
        latest = _inject_in_thread(workspace, newer, second)
        _wait_until(lambda: workspace._latest.get("SYN-501") == newer)
    waiting.join(), latest.join()
    assert first == ["a newer request replaced this one"] and second == [0.2]


def test_a_request_replaced_between_seeds_stops_there(client, monkeypatch):
    workspace = client.app.state.workspace
    older, newer = ex.ErrorRequest(a_sigma=0.1), ex.ErrorRequest(a_sigma=0.2)
    calls, combine = [], ex.combine

    def move_on_after_the_first_seed(*args, **kwargs):
        calls.append(args[3])
        workspace._latest["SYN-501"] = newer
        return combine(*args, **kwargs)

    monkeypatch.setattr(ex, "combine", move_on_after_the_first_seed)
    with pytest.raises(ws.SceneUnavailable, match="newer request"):
        workspace.inject("SYN-501", older)
    assert len(calls) == 1 and ("SYN-501", older) not in workspace._results


def test_the_same_request_twice_is_calculated_once(client, monkeypatch):
    workspace = client.app.state.workspace
    request = ex.ErrorRequest(a_sigma=0.1)
    calls, compute = [], workspace._compute

    def counted(*args):
        calls.append(args)
        return compute(*args)

    monkeypatch.setattr(workspace, "_compute", counted)
    first, second = [], []
    with workspace._running["SYN-501"]:
        threads = [_inject_in_thread(workspace, request, first), _inject_in_thread(workspace, request, second)]
        _wait_until(lambda: workspace._latest.get("SYN-501") == request)
    for thread in threads:
        thread.join()
    assert first == second == [0.1] and len(calls) == 1


def test_a_replaced_request_is_reported_as_a_conflict(client, monkeypatch):
    def replaced(*_):
        raise ws.SceneUnavailable("a newer request replaced this one")

    monkeypatch.setattr(client.app.state.workspace, "_compute", replaced)
    response = client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.1})
    assert response.status_code == 409 and "newer request" in response.json()["detail"]


# --- numbered parts of long recordings, and more synthetic scenes (decision 0012) --------------


def _long_ucy_client(tmp_path):
    """Two people walking together across the cut between UNIV parts 1 and 2, and one far away inside part 2."""
    path = tmp_path / "raw" / ws.UCY_ANNOTATIONS["UNIV"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_vsp(_walker(1250, 1650, 10), _walker(1250, 1650, -20), _walker(1400, 1700, -280)), encoding="latin-1")
    return _client_with_raw(tmp_path)


def _long_mot17_client(tmp_path):
    _mot17_archive(tmp_path / "raw" / ws.MOT17_ARCHIVE, length=600, truth=_gt_rows(600))
    return _client_with_raw(tmp_path)


LONG_SCENES = [(_long_ucy_client, "UNIV-2", {}), (_long_mot17_client, "MOT17-02-2", {"source": "truth"})]


def _video_frame(path, k):
    run = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{k})", "-vframes", "1",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], check=True, capture_output=True)
    return np.frombuffer(run.stdout, dtype=np.uint8).astype(np.float64)


def test_a_range_is_cut_on_the_sampling_grid_and_the_last_part_takes_what_is_left():
    assert vt.part_ranges(780, 12382, 6, 4) == [(780, 3678), (3678, 6576), (6576, 9474), (9474, 12382)]
    assert vt.part_ranges(1, 18062, 10, 4) == [(1, 4511), (4511, 9021), (9021, 13531), (13531, 18062)]
    assert vt.part_ranges(0, 837, 6, 2) == [(0, 414), (414, 837)]


@pytest.mark.parametrize("name", ws.SPLIT_RECORDINGS)
def test_parts_cover_their_recording_on_its_grid_with_no_gap_or_overlap(name):
    first, stop, step, count = ws.SPLIT_RECORDINGS[name]
    parts = [part for part in ws.PARTS.values() if part.recording == name]
    assert [part.number for part in parts] == list(range(1, count + 1))
    assert parts[0].start == first and parts[-1].stop == stop
    assert all(one.stop == next_one.start for one, next_one in zip(parts, parts[1:]))
    assert all((part.start - first) % step == 0 for part in parts)
    lengths = [part.stop - part.start for part in parts]
    assert max(lengths) - min(lengths) <= count * (step + 1), "parts are equal apart from what the grid leaves over"


def test_the_catalogue_lists_each_part_right_after_its_recording(client):
    items = client.get("/api/workspace/scenes").json()["items"]
    ids = [scene["id"] for scene in items]
    assert len(ids) == len(set(ids)) == 52
    assert ids[:6] == ["ETH", "ETH-1", "ETH-2", "ETH-3", "ETH-4", "HOTEL"]
    assert [scene["name"] for scene in items if scene["parent"] == "MOT17-02"] == ["02 part 1", "02 part 2"]
    assert {dataset: sum(scene["dataset"] == dataset for scene in items) for dataset in ("EWAP", "UCY", "MOT17", "Synthetic")} \
        == {"EWAP": 10, "UCY": 15, "MOT17": 17, "Synthetic": 10}


@pytest.mark.parametrize("make, recording", [(_long_ucy_client, "UNIV"), (_long_mot17_client, "MOT17-02")])
def test_a_part_offers_what_its_recording_offers_and_says_what_it_is(tmp_path, make, recording):
    scenes = {scene["id"]: scene for scene in make(tmp_path).get("/api/workspace/scenes").json()["items"]}
    whole, part = scenes[recording], scenes[f"{recording}-2"]
    for field in ("dataset", "regime", "has_video", "errors_available", "errors_reason", "evidence_level", "predictors", "unit",
                  "detector"):
        assert part[field] == whole[field], field
    assert whole["parent"] is None and part["parent"] == recording
    assert "not a separate scene" in part["caution"] and whole["caution"] in part["caution"]


def test_a_part_of_a_metre_scene_is_cautioned_though_its_recording_is_not(client):
    scenes = {scene["id"]: scene for scene in client.get("/api/workspace/scenes").json()["items"]}
    assert scenes["ETH"]["caution"] is None and scenes["ETH-1"]["caution"] == ws.PART_CAUTION


def test_a_part_has_no_stored_run_even_when_its_recording_has_one(client, monkeypatch):
    monkeypatch.setattr(client.app.state.workspace.registry, "newest", lambda kinds: SimpleNamespace(id="stored"))
    scenes = {scene["id"]: scene for scene in client.get("/api/workspace/scenes").json()["items"]}
    assert scenes["MOT17-02"]["stored_run"] == "stored" and scenes["MOT17-02-1"]["stored_run"] is None


def test_cutting_keeps_ids_and_counts_frames_from_the_start_of_the_part():
    tracks = (GridTrack(7, np.arange(0, 100, 10), np.arange(20.0).reshape(10, 2)), GridTrack(8, np.arange(200, 300, 10), np.zeros((10, 2))))
    cut = vt.cut_tracks(tracks, vt.Part("X", 1, 30, 60))
    assert [track.ped_id for track in cut] == [7]
    assert cut[0].frames.tolist() == [0, 10, 20] and cut[0].positions.tolist() == [[6, 7], [8, 9], [10, 11]]


def test_cutting_rows_matches_cutting_tracks():
    rows = [vt.track_row(7, np.arange(0, 50, 10), np.arange(10.0).reshape(5, 2), np.arange(20.0).reshape(5, 4)),
            vt.track_row(8, np.arange(200, 250, 10), np.zeros((5, 2)), np.zeros((5, 4)))]
    assert vt.cut_rows(rows, vt.Part("X", 1, 10, 30)) == [{"id": 7, "frames": [0, 10], "uv": [[2.0, 3.0], [4.0, 5.0]],
                                                           "boxes": [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]}]


@pytest.mark.parametrize("make, scene, source", LONG_SCENES)
def test_every_window_of_a_part_is_a_window_of_its_recording(tmp_path, make, scene, source):
    """Cutting changes no target: a part's windows are exactly the recording's windows that lie inside it."""
    workspace, part = make(tmp_path).app.state.workspace, ws.PARTS[scene]
    whole, cut = workspace._load(part.recording).windows, workspace._load(scene).windows
    inside = ((whole.frames >= part.start) & (whole.frames < part.stop)).all(axis=1)
    assert len(cut) == inside.sum() > 0
    windows = {(int(person), *frames.tolist()): (observed, target)
               for person, frames, observed, target in zip(whole.track_ids, whole.frames, whole.observed, whole.target)}
    for person, frames, observed, target in zip(cut.track_ids, cut.frames, cut.observed, cut.target):
        same = windows[(int(person), *(frames + part.start).tolist())]
        assert np.array_equal(same[0], observed) and np.array_equal(same[1], target)


@pytest.mark.parametrize("make, scene, source", LONG_SCENES)
def test_a_parts_tracks_are_its_recordings_tracks_from_its_start(tmp_path, make, scene, source):
    client, part = make(tmp_path), ws.PARTS[scene]
    whole = {track["id"]: track for track in client.get(f"/api/workspace/scenes/{part.recording}/tracks", params=source).json()["tracks"]}
    body = client.get(f"/api/workspace/scenes/{scene}/tracks", params=source).json()
    assert body["scene"] == scene and f"Part {part.number} of {part.recording}" in body["note"]
    assert body["tracks"] and {track["id"] for track in body["tracks"]} <= set(whole)
    for track in body["tracks"]:
        mine = whole[track["id"]]
        for frame, uv, box in zip(track["frames"], track["uv"], track["boxes"]):
            k = mine["frames"].index(frame + part.start)
            assert 0 <= frame < part.stop - part.start and uv == mine["uv"][k] and box == mine["boxes"][k]


def test_a_person_crossing_a_cut_appears_in_both_parts_under_one_id(tmp_path):
    client = _long_ucy_client(tmp_path)
    first, second = ({track["id"] for track in client.get(f"/api/workspace/scenes/UNIV-{n}/tracks").json()["tracks"]} for n in (1, 2))
    assert first == {1, 2} and second == {1, 2, 3}


def test_a_mot17_part_cuts_the_tracker_output_of_its_recording(tmp_path):
    client = _long_mot17_client(tmp_path)
    whole = client.get("/api/workspace/scenes/MOT17-02/tracks").json()
    first, second = (client.get(f"/api/workspace/scenes/MOT17-02-{n}/tracks").json() for n in (1, 2))
    assert first["tracks"] == whole["tracks"] and first["source"] == whole["source"] and whole["tracks"], "all detections are in part 1"
    assert first["scene"] == "MOT17-02-1" and "Part 1 of MOT17-02" in first["note"] and whole["note"] in first["note"]
    assert second["tracks"] == [] and second["frame_step"] == 1


@pytest.mark.parametrize("make, scene, source", LONG_SCENES)
def test_a_part_takes_errors_like_any_scene(tmp_path, make, scene, source):
    client, part = make(tmp_path), ws.PARTS[scene]
    noise = client.get(f"/api/workspace/scenes/{scene}/inject", params={"a": 12}).json()
    assert set(noise["predictors"]) == {"CV", "CA"} and noise["predictors"]["CV"]["ade"] is not None
    response = client.get(f"/api/workspace/scenes/{scene}/inject", params={"a": 12, "b": 0.1, "c": 0.1, "d": 0.2})
    body = response.json()
    assert response.status_code == 200 and body["scene"] == scene and body["unit"] == "px" and body["exploratory"] is True
    assert "not a separate scene" in body["label"] and "not a research result" in body["label"]
    row = body["predictors"]["CV"]
    assert row["ade"] is not None or row["undefined_reason"] == "no_predicted_windows", "an ADE is only ever missing for a reason"
    assert all(0 <= frame < part.stop - part.start for track in body["overlay"]["tracks"] for frame in track["frames"])
    assert not [key for key in _keys(body) if key.endswith("_m")]
    assert list((tmp_path / "runs").iterdir()) == []


@pytest.mark.parametrize("make, scene, source", LONG_SCENES)
def test_a_parts_noise_limit_and_switch_distance_are_its_recordings(tmp_path, make, scene, source):
    client, part = make(tmp_path), ws.PARTS[scene]
    workspace = client.app.state.workspace
    assert workspace.noise_limit(scene) == ws.MAX_SIGMA_PX
    assert client.get(f"/api/workspace/scenes/{scene}/inject", params={"a": 60}).status_code == 200
    assert client.get(f"/api/workspace/scenes/{scene}/inject", params={"a": 61}).status_code == 422
    assert workspace._base(scene).d_max == workspace._base(part.recording).d_max
    assert workspace._base(scene).frame_step == workspace._base(part.recording).frame_step


@pytest.mark.parametrize("make, scene, source", LONG_SCENES)
def test_a_switch_in_a_part_moves_the_id_onto_the_other_persons_box(tmp_path, make, scene, source):
    client = make(tmp_path)
    clean = {track["id"]: track for track in client.get(f"/api/workspace/scenes/{scene}/tracks", params=source).json()["tracks"]}
    overlay = client.get(f"/api/workspace/scenes/{scene}/inject", params={"d": 1.0}).json()["overlay"]
    events = [event for event in overlay["events"] if event["kind"] == "switch"]
    assert [sorted((event["a"], event["b"])) for event in events] == [[1, 2]]
    switched = {track["id"]: track for track in overlay["tracks"]}
    for label, partner in ((1, 2), (2, 1)):
        mine, theirs = (dict(zip(clean[person]["frames"], clean[person]["boxes"])) for person in (label, partner))
        for at, box in zip(switched[label]["frames"], switched[label]["boxes"]):
            assert box == pytest.approx(theirs[at] if at >= events[0]["frame"] else mine[at], abs=1e-9)


def test_a_part_with_nothing_on_this_machine_says_so(client):
    assert client.get("/api/workspace/scenes/UNIV-1/inject").status_code == 409
    assert "not on this machine" in client.get("/api/workspace/scenes/UNIV-1/tracks").json()["detail"]
    assert client.get("/api/workspace/scenes/ETH-1/video/status").json()["state"] == "missing"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
def test_a_part_of_a_video_keeps_exactly_its_frames(tmp_path):
    """Each source frame is one flat shade, four levels brighter than the last, so its brightness says which frame it is."""
    source = tmp_path / "numbered.avi"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=size=160x120:rate=25:duration=2,format=yuv420p,"
                    "geq=lum='16+4*N':cb=128:cr=128", "-c:v", "mpeg4", "-q:v", "2", str(source)], check=True)
    whole, part = tmp_path / "whole.mp4", tmp_path / "part.mp4"
    vc.from_file(source)(whole), vc.from_file(source, (10, 30))(part)
    assert vc._probe(whole)["frames"] == 50 and vc._probe(part)["frames"] == 20
    shades = [_video_frame(whole, at).mean() for at in range(50)]
    assert all(later - earlier > 2 for earlier, later in zip(shades, shades[1:])), "every frame is recognisable"
    for k in range(20):
        assert int(np.argmin(np.abs(np.array(shades) - _video_frame(part, k).mean()))) == 10 + k


def test_a_part_is_built_from_the_right_stretch_of_its_recordings_video(client, monkeypatch):
    monkeypatch.setattr(vc, "from_file", lambda source, frames=None, fps=None: (source, frames))
    workspace = client.app.state.workspace
    assert workspace._builder(workspace.entry("ETH-2"))[1] == (3678, 6576)
    assert workspace._builder(workspace.entry("ETH-2"))[0].name == "seq_eth.avi"
    assert workspace._builder(workspace.entry("UNIV-4"))[0].name == "students003.avi"
    assert workspace._builder(workspace.entry("ETH")) [1] is None


@pytest.mark.parametrize("scene, fps", [("ETH", 15), ("ETH-3", 15), ("HOTEL", None), ("HOTEL-2", None)])
def test_eth_is_built_at_the_rate_it_was_filmed_and_hotel_as_its_file_says(client, monkeypatch, scene, fps):
    monkeypatch.setattr(vc, "from_file", lambda source, frames=None, fps=None: fps)
    assert client.app.state.workspace._builder(client.app.state.workspace.entry(scene)) == fps


def test_the_filmed_rate_of_eth_is_the_one_its_annotation_step_gives():
    assert ws.EWAP_FILMED_FPS["ETH"] == ws.SPLIT_RECORDINGS["ETH"][2] / DT


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not on PATH")
def test_correcting_a_files_rate_keeps_every_frame_in_order(tmp_path):
    source = _clip(tmp_path / "clip.avi")
    normal, slower = tmp_path / "normal.mp4", tmp_path / "slower.mp4"
    vc.from_file(source)(normal), vc.from_file(source, None, 15)(slower)
    assert vc._probe(normal)["fps"] == 25.0 and vc._probe(normal)["frames"] == 50
    assert vc._probe(slower)["fps"] == 15.0 and vc._probe(slower)["frames"] == 50
    part = tmp_path / "part.mp4"
    vc.from_file(source, (10, 30), 15)(part)
    assert vc._probe(part)["fps"] == 15.0 and vc._probe(part)["frames"] == 20


def test_a_mot17_part_takes_exactly_its_images(tmp_path, monkeypatch):
    workspace = _long_mot17_client(tmp_path).app.state.workspace
    seen = {}
    monkeypatch.setattr(vc, "zip_images", lambda archive, members: seen.update(members=members))
    monkeypatch.setattr(vc, "from_frames", lambda *args: lambda target: None)
    workspace._builder(workspace.entry("MOT17-02-2"))(tmp_path / "part.mp4")
    assert len(seen["members"]) == 300, "video frame 0 of this part is MOT17 image 301, because MOT17 counts from 1"
    assert seen["members"][0].endswith("/img1/000301.jpg") and seen["members"][-1].endswith("/img1/000600.jpg")


def test_the_added_synthetic_scenes_work_in_metres_and_store_nothing(client, tmp_path):
    scenes = {scene["id"]: scene for scene in client.get("/api/workspace/scenes").json()["items"]}
    for seed in range(506, 511):
        scene = scenes[f"SYN-{seed}"]
        assert scene["unit"] == "m" and scene["errors_available"] and scene["has_video"] and scene["stored_run"] is None
    body = client.get("/api/workspace/scenes/SYN-510/inject", params={"a": 0.2, "d": 0.1}).json()
    assert body["unit"] == "m" and body["predictors"]["CV"]["ade"] is not None and body["overlay"]["predicted"]["CV"]
    assert list((tmp_path / "runs").iterdir()) == []


# --- YOLO boxes on every real video (decision 0015) --------------------------------------------


def _video_boxes(scene="UNIV", frames=1700):
    """Two people found by a detector, walking apart, as fractions of a 720 by 576 video."""
    rows = [[[0.1 + k / 40000, 0.1, 0.15 + k / 40000, 0.3, 0.9, 0], [0.7 - k / 40000, 0.4, 0.75 - k / 40000, 0.6, 0.8, 0]]
            for k in range(frames)]
    return {"schema": ws.DETECTIONS_SCHEMA, "scene": scene, "model": "yolo11m", "weights_sha256": "abc", "imgsz": 1280,
            "image_width": 720, "image_height": 576, "frames": frames, "by_frame": rows}


def _ucy_with_boxes(tmp_path, **boxes):
    _long_ucy_client(tmp_path)
    return _client_with_raw(tmp_path, _write_boxes(tmp_path, _video_boxes(**boxes), boxes.get("scene", "UNIV")))


def _videos_module():
    _detect_module()
    spec = importlib.util.spec_from_file_location("yolo_detect_videos", REPO_ROOT / "scripts" / "yolo_detect_videos.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_boxes_made_on_a_video_are_in_the_pixels_of_that_video():
    tracks = vt.video_tracks(_video_boxes(frames=6))
    assert len(tracks) == 2 and tracks[0]["frames"] == [0, 1, 2, 3, 4, 5]
    assert tracks[0]["boxes"][0] == pytest.approx([72, 57.6, 108, 172.8], abs=0.06)
    assert tracks[0]["uv"][0] == pytest.approx([90, 172.8], abs=0.06), "paths follow the feet"


def test_only_people_are_tracked_on_a_video():
    boxes = _video_boxes(frames=6)
    boxes["by_frame"] = [[[*frame[0][:5], 2], frame[1]] for frame in boxes["by_frame"]]
    assert len(vt.video_tracks(boxes)) == 1


def test_a_video_scene_shows_the_tracker_once_yolo_has_run(tmp_path):
    client = _ucy_with_boxes(tmp_path)
    body = client.get("/api/workspace/scenes/UNIV/tracks").json()
    assert body["source"] == "IoU tracker on yolo11m detections" and body["frame_step"] == 1 and len(body["tracks"]) == 2
    assert "YOLO11" in body["note"] and "can still be missed" in body["note"]
    scenes = {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}
    assert scenes["UNIV"]["detector"] == "yolo11m" and scenes["UNIV-2"]["detector"] == "yolo11m"
    assert scenes["ZARA1"]["detector"] is None and scenes["SYN-501"]["detector"] is None


def test_a_video_scene_keeps_its_annotations_until_yolo_has_run(tmp_path):
    client = _long_ucy_client(tmp_path)
    assert client.get("/api/workspace/scenes/UNIV/tracks").json()["source"] == "Human annotations"
    assert {s["id"]: s for s in client.get("/api/workspace/scenes").json()["items"]}["UNIV"]["detector"] is None


def test_the_annotated_people_stay_available_for_errors(tmp_path):
    client = _ucy_with_boxes(tmp_path)
    assert client.get("/api/workspace/scenes/UNIV/tracks", params={"source": "truth"}).json()["source"] == "Human annotations"


def test_a_part_cuts_the_tracker_output_of_its_recording(tmp_path):
    client = _ucy_with_boxes(tmp_path)
    whole = client.get("/api/workspace/scenes/UNIV/tracks").json()["tracks"][0]
    part = client.get("/api/workspace/scenes/UNIV-2/tracks").json()
    start = ws.PARTS["UNIV-2"].start
    assert part["note"].startswith("Part 2 of UNIV") and part["tracks"][0]["id"] == whole["id"]
    assert part["tracks"][0]["frames"][0] == 0 and part["tracks"][0]["boxes"][0] == whole["boxes"][start]


def test_stored_boxes_never_reach_a_measurement(tmp_path):
    """The same annotations with and without a detection file must give the same numbers, to the last digit."""
    plain, boxed = _long_ucy_client(tmp_path), _ucy_with_boxes(tmp_path)
    for query in ({"a": 20}, {"b": 0.2}, {"d": 0.1}):
        assert plain.get("/api/workspace/scenes/UNIV/inject", params=query).json() == boxed.get("/api/workspace/scenes/UNIV/inject", params=query).json()

    def swept(client):
        for _ in range(4000):
            body = client.get("/api/workspace/scenes/UNIV/sweep").json()
            if body["state"] != "calculating":
                return body
            time.sleep(0.01)
        raise AssertionError("the sweep did not finish")

    assert swept(plain)["errors"] == swept(boxed)["errors"] and swept(plain)["clean"] == swept(boxed)["clean"]


def test_a_detection_file_is_read_once_until_it_changes(tmp_path, monkeypatch):
    client = _ucy_with_boxes(tmp_path)
    workspace, reads = client.app.state.workspace, []
    original = workspace._detections
    monkeypatch.setattr(workspace, "_detections", lambda scene: reads.append(scene) or original(scene))
    client.get("/api/workspace/scenes"), client.get("/api/workspace/scenes")
    assert reads == ["UNIV"]
    path = tmp_path / "cache-boxes" / "UNIV.json"
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10 ** 9))
    client.get("/api/workspace/scenes")
    assert reads == ["UNIV", "UNIV"]


def test_a_synthetic_scene_never_takes_detections(tmp_path):
    client = _client_with_boxes(tmp_path, _write_boxes(tmp_path, _video_boxes("SYN-501"), "SYN-501"))
    assert client.get("/api/workspace/scenes/SYN-501/tracks").json()["source"] == "Scene generator"


def test_a_wrong_detection_file_leaves_the_annotations_in_place(tmp_path):
    _long_ucy_client(tmp_path)
    client = _client_with_raw(tmp_path, _write_boxes(tmp_path, {**_video_boxes(), "frames": 3}, "UNIV"))
    assert client.get("/api/workspace/scenes/UNIV/tracks").json()["source"] == "Human annotations"


def test_a_recording_is_refused_unless_every_browser_frame_was_read(tmp_path):
    videos = _videos_module()
    (tmp_path / "ETH.json").write_text(json.dumps({"frames": 2}), encoding="utf-8")
    answer = lambda _: (640, 480, [((64, 48, 128, 240), 0.9, 0)])
    doc = videos.detect_scene("ETH", tmp_path, answer, {"model": "yolo11m"}, frames=lambda path: iter(["a", "b"]))
    assert doc["scene"] == "ETH" and doc["frames"] == 2 and doc["image_width"] == 640
    with pytest.raises(ValueError, match="read 1 frames"):
        videos.detect_scene("ETH", tmp_path, answer, {"model": "yolo11m"}, frames=lambda path: iter(["a"]))


def test_the_video_script_output_is_accepted_by_the_api(tmp_path):
    videos = _videos_module()
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "UNIV.json").write_text(json.dumps({"frames": 6}), encoding="utf-8")
    boxes = iter([(720, 576, [((72, 57, 108, 172), 0.9, 0)])] * 6)
    doc = videos.detect_scene("UNIV", folder, lambda _: next(boxes), {"model": "yolo11m"}, frames=lambda path: iter(range(6)))
    _detect_module().write_atomic(tmp_path / "cache-boxes" / "UNIV.json", doc)
    body = _client_with_raw(tmp_path, tmp_path / "cache-boxes").get("/api/workspace/scenes/UNIV/tracks").json()
    assert body["source"] == "IoU tracker on yolo11m detections" and len(body["tracks"]) == 1


def test_the_video_script_stays_outside_the_research_code():
    source = (REPO_ROOT / "scripts" / "yolo_detect_videos.py").read_text(encoding="utf-8")
    assert "trackvision" not in source.replace("imports nothing from trackvision", "")
    assert set(_videos_module().SCENES) == set(ws.EWAP_VIDEO) | set(ws.UCY_VIDEO), "MOT17 has its own script, and synthetic scenes take none"


def test_the_setup_script_never_touches_the_project_environment():
    source = (REPO_ROOT / "scripts" / "setup_yolo.ps1").read_text(encoding="utf-8")
    assert "tv-envs\\yolo-py312" in source and "-InstallOnly" in source and ".venv\\Scripts" not in source


# --- the noise example (decision 0013) ---------------------------------------------------------


def test_the_noise_example_uses_the_clean_paths_and_levels_from_the_frozen_grid(client):
    body = client.get("/api/workspace/scenes/SYN-501/noise-example").json()
    grid = CFG.frozen["injection"]["grids"]["A"]["levels"]
    assert [(level["name"], level["sigma"]) for level in body["levels"]] == [("Low", 0.05), ("Medium", 0.2), ("High", 0.5)]
    assert all(level["sigma"] in grid for level in body["levels"])
    assert body["unit"] == "m" and body["image_coordinates"] is False and body["exploratory"] is True and "not a saved figure" in body["label"]
    windows = client.app.state.workspace._load("SYN-501").windows
    index = ws._example_window(windows)
    assert body["observed"] == windows.observed[index].tolist() and body["future"] == windows.target[index].tolist()
    assert body["person"] == int(windows.track_ids[index]) and body["frames"] == windows.frames[index].tolist()


def test_the_noise_example_is_exactly_what_the_frozen_injector_gives(client):
    body = client.get("/api/workspace/scenes/SYN-501/noise-example").json()
    loaded = client.app.state.workspace._load("SYN-501")
    index = ws._example_window(loaded.windows)
    for level in body["levels"]:
        frozen = evaluation.corrupt(loaded.tracks, loaded.population, "A", level["sigma"], body["seed"], CFG, loaded.frame_step)
        assert level["noisy"] == frozen.inputs[index].tolist()


def test_more_noise_moves_the_points_further(client):
    body = client.get("/api/workspace/scenes/SYN-501/noise-example").json()
    clean = np.array(body["observed"])
    moved = [np.linalg.norm(np.array(level["noisy"]) - clean, axis=1).mean() for level in body["levels"]]
    assert 0 < moved[0] < moved[1] < moved[2]


def test_the_example_person_is_chosen_the_same_way_every_time(client):
    first = client.get("/api/workspace/scenes/SYN-501/noise-example").json()
    assert client.get("/api/workspace/scenes/SYN-501/noise-example").json() == first


def test_the_example_person_really_walks():
    points = np.zeros((5, 20, 2))
    points[:, :, 0] = np.arange(20)[None, :] * np.array([0.0, 1.0, 2.0, 3.0, 4.0])[:, None]
    windows = SimpleNamespace(observed=points[:, :8], target=points[:, 8:])
    assert ws._example_window(windows) == 3, "the 75th percentile of distance walked, not the extreme and not someone standing"


@pytest.mark.parametrize("make, scene", [(_ucy_client, "UNIV"), (_mot17_client, "MOT17-02"), (_long_ucy_client, "UNIV-2")])
def test_a_pixel_scene_uses_pixel_levels_and_no_metre_field(tmp_path, make, scene):
    body = make(tmp_path).get(f"/api/workspace/scenes/{scene}/noise-example").json()
    assert [level["sigma"] for level in body["levels"]] == [10.0, 30.0, 60.0] and body["unit"] == "px" and body["image_coordinates"] is True
    assert not [key for key in _keys(body) if key.endswith("_m")]


def test_the_noise_example_writes_nothing_under_runs(client, tmp_path):
    client.get("/api/workspace/scenes/SYN-501/noise-example")
    assert list((tmp_path / "runs").iterdir()) == []


def test_the_noise_example_of_a_scene_without_its_annotations_says_why(client):
    response = client.get("/api/workspace/scenes/UNIV/noise-example")
    assert response.status_code == 409 and "not on this machine" in response.json()["detail"]
    assert client.get("/api/workspace/scenes/nothing/noise-example").status_code == 404
    assert client.post("/api/workspace/scenes/SYN-501/noise-example").status_code == 405
