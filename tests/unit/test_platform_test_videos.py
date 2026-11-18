"""The Test Videos section (decision 0017). Videos here are small fixtures, and no value is a result."""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from trackvision import config
from trackvision.experiments import evaluation
from trackvision.platform.app import create_app
from trackvision.platform.config import Settings
from trackvision.platform.services import exploratory as ex, test_videos as tv, workspace as ws
from trackvision.predict.baselines import constant_acceleration, constant_velocity
from trackvision.synthetic.tracks import GridTrack

CFG = config.validate().config
PROTOCOL = CFG.frozen["protocol"]["static"]
HISTORY, HORIZON = PROTOCOL["history_len"], PROTOCOL["horizon_len"]
VIDEO = "caracas-boulevard"
META = {"width": 1280, "height": 720, "fps": 25.0, "frames": 600}
ACCURACY = re.compile(r"^(ade|fde|rmse|mota|motp|idf1|hota|idsw|coverage|real_tracker_iou|precision|recall|.*_(ade|fde|rmse))$", re.I)
HAS_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def _boxes(video=VIDEO, frames=600):
    """Two people walking side by side, close enough for an ID switch, as fractions of a 1280 by 720 video."""
    rows = [[[0.10 + k / 4000, 0.3, 0.15 + k / 4000, 0.6, 0.9, 0], [0.26 + k / 4000, 0.3, 0.31 + k / 4000, 0.6, 0.8, 0]]
            for k in range(frames)]
    return {"schema": ws.DETECTIONS_SCHEMA, "scene": video, "model": "yolo11m", "weights_sha256": "abc", "imgsz": 1280,
            "image_width": 1280, "image_height": 720, "frames": frames, "by_frame": rows}


def _app(tmp_path, *, ready=True, document=None):
    for name in ("runs", "manifests"):
        (tmp_path / name).mkdir(exist_ok=True)
    cache = tmp_path / "cache"
    copies, boxes = cache / "test_videos", cache / "test_video_detections"
    copies.mkdir(parents=True, exist_ok=True)
    boxes.mkdir(parents=True, exist_ok=True)
    if ready:
        (copies / f"{VIDEO}.mp4").write_bytes(b"a video")
        (copies / f"{VIDEO}.json").write_text(json.dumps(META), encoding="utf-8")
        (boxes / f"{VIDEO}.json").write_text(json.dumps(document or _boxes()), encoding="utf-8")
    return create_app(Settings(runs_dir=tmp_path / "runs", manifests_dir=tmp_path / "manifests", raw_dir=tmp_path / "raw",
                               cache_dir=cache / "video", detections_dir=cache / "detections"))


@pytest.fixture
def client(tmp_path):
    return TestClient(_app(tmp_path), raise_server_exceptions=False)


def _forecast(client, **levels):
    response = client.get(f"/api/test-videos/{VIDEO}/forecast", params=levels)
    assert response.status_code == 200, response.text
    return response.json()


def _kept(result):
    return list(result["forecasts"]["kept_by_seed"].values())


def _keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


# --- reading a position from tracker boxes ----------------------------------------------------


def test_a_point_is_the_mean_of_the_step_before_it():
    frames = np.arange(1, 21)
    uv = np.c_[frames, 2 * frames].astype(float)
    grid, points, boxes = tv.sampled(frames, uv, np.tile(frames[:, None], (1, 4)), 10)
    assert grid.tolist() == [10, 20]
    assert points.tolist() == [[5.5, 11.0], [15.5, 31.0]]
    assert boxes[:, 0].tolist() == [10, 20], "the box is the last one in the step"


def test_a_point_takes_nothing_from_the_future():
    frames = np.arange(1, 31)
    uv = np.c_[frames, frames].astype(float)
    boxes = np.zeros((30, 4))
    before = tv.sampled(frames, uv, boxes, 10)[1][0]
    later = uv.copy()
    later[10:] += 1000
    assert tv.sampled(frames, later, boxes, 10)[1][0].tolist() == before.tolist()


def test_a_person_last_seen_part_way_through_a_step_has_no_point_at_its_end():
    frames = np.arange(1, 16)
    assert tv.sampled(frames, np.zeros((15, 2)), np.zeros((15, 4)), 10)[0].tolist() == [10]


def test_a_short_gap_keeps_the_point_and_a_whole_empty_step_does_not():
    short = np.r_[1:11, 15:21]
    assert tv.sampled(short, np.zeros((16, 2)), np.zeros((16, 4)), 10)[0].tolist() == [10, 20]
    long = np.r_[1:11, 21:31]
    assert tv.sampled(long, np.zeros((20, 2)), np.zeros((20, 4)), 10)[0].tolist() == [10, 30]


def test_a_forecast_needs_its_history_in_a_row():
    whole = GridTrack(1, np.arange(0, 100, 10), np.zeros((10, 2)))
    assert len(tv.history_windows([whole], 8, 10)) == 3
    broken = GridTrack(2, np.r_[np.arange(0, 70, 10), np.arange(80, 160, 10)], np.zeros((15, 2)))
    windows = tv.history_windows([broken], 8, 10)
    assert windows.frames[0].tolist() == list(range(80, 160, 10)) and len(windows) == 1


def test_nobody_on_a_video_gives_no_forecast_and_no_failure():
    windows = tv.history_windows([], 8, 10)
    assert len(windows) == 0 and windows.observed.shape == (0, 8, 2)


# --- forecasts and errors ---------------------------------------------------------------------


def test_forecasts_are_the_frozen_predictors(tmp_path):
    app = _app(tmp_path)
    result = _forecast(TestClient(app))
    loaded = app.state.test_videos._prepare(VIDEO).loaded
    observed = loaded.windows.observed
    last = loaded.windows.frames[:, -1].tolist()
    for name, frozen in (("CV", constant_velocity(observed, HORIZON, 2)), ("CA", constant_acceleration(observed, HORIZON, 3))):
        shown = result["overlay"]["predicted"][name]
        for k in (0, 7, len(observed) - 1):
            people = {person: path for person, path in shown[str(last[k])]}
            assert people[int(loaded.windows.track_ids[k])] == np.round(frozen[k], 1).tolist()


def test_the_sampling_step_follows_the_frame_rate(tmp_path):
    result = _forecast(TestClient(_app(tmp_path)))
    assert result["grid"]["frame_step"] == 10 and result["seconds_per_step"] == pytest.approx(0.4)
    assert result["horizon_len"] == HORIZON
    assert all(frame % 10 == 0 for track in result["grid"]["tracks"] for frame in track["frames"])


def test_with_no_errors_every_forecast_stays_and_none_moves(client):
    result = _forecast(client)
    assert result["active"] == [] and result["achieved"] == {}
    assert result["forecasts"]["total"] > 0 and set(_kept(result)) == {result["forecasts"]["total"]}
    assert result["moved"] == {"CV": pytest.approx(0, abs=1e-9), "CA": pytest.approx(0, abs=1e-9)}
    assert result["overlay"]["events"] == []


def test_position_noise_moves_forecasts_and_ca_more_than_cv(client):
    result = _forecast(client, a=20)
    assert result["active"] == ["A"] and set(_kept(result)) == {result["forecasts"]["total"]}
    assert result["achieved"]["A"]["measure"] == "rmse_pos_px", "the injector's own report of the noise it added"
    assert result["achieved"]["A"]["mean"] == pytest.approx(20 * 2 ** 0.5, rel=0.2), "20 px on each axis"
    assert 0 < result["moved"]["CV"] < result["moved"]["CA"]


def test_missing_detections_remove_forecasts_and_leave_the_rest_alone(client):
    result = _forecast(client, b=0.2)
    assert max(_kept(result)) < result["forecasts"]["total"]
    assert result["moved"]["CV"] == pytest.approx(0, abs=1e-9)
    assert any(track.get("dropped") for track in result["overlay"]["tracks"])


def test_filling_gaps_keeps_more_forecasts_than_not(client):
    assert min(_kept(_forecast(client, b=0.2, reconstruct=True))) > max(_kept(_forecast(client, b=0.2)))


def test_broken_tracks_remove_forecasts_and_mark_where(client):
    result = _forecast(client, c=0.2)
    assert max(_kept(result)) < result["forecasts"]["total"]
    assert any(event["kind"] == "split" for event in result["overlay"]["events"])


def test_id_switches_swap_two_people_and_move_their_forecasts(client):
    result = _forecast(client, d=0.2)
    assert any(event["kind"] == "switch" for event in result["overlay"]["events"])
    assert result["moved"]["CV"] > 0


def test_every_error_at_once_runs(client):
    result = _forecast(client, a=10, b=0.1, c=0.05, d=0.05)
    assert set(result["active"]) == {"A", "B2", "C", "D"}


def test_too_much_error_leaves_no_forecast_and_says_so_with_nothing(client):
    result = _forecast(client, b=1)
    assert set(_kept(result)) == {0} and result["moved"] == {"CV": None, "CA": None}


def test_results_repeat(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = (_forecast(TestClient(_app(tmp_path / name)), a=20, c=0.1) for name in "ab")
    assert first["moved"] == second["moved"] and first["forecasts"] == second["forecasts"]


def test_a_tracks_row_is_the_tracker_on_the_stored_boxes_frame_by_frame(client):
    data = client.get(f"/api/test-videos/{VIDEO}/tracks").json()
    assert data["frame_step"] == 1 and data["source"] == "IoU tracker on yolo11m detections"
    assert len(data["tracks"]) == 2 and data["tracks"][0]["frames"] == list(range(600))
    assert (data["history_len"], data["horizon_len"]) == (HISTORY, HORIZON)


def test_boxes_keep_their_size_around_the_sampled_point(client):
    track = _forecast(client)["grid"]["tracks"][0]
    box = track["boxes"][5]
    assert box[2] - box[0] == pytest.approx(64, abs=0.2) and box[3] - box[1] == pytest.approx(216, abs=0.2)


# --- nothing that needs ground truth, and nothing that reaches other data ---------------------


def test_no_response_holds_an_accuracy_value(client):
    documents = [client.get("/api/test-videos").json(), client.get(f"/api/test-videos/{VIDEO}/tracks").json(),
                 _forecast(client), _forecast(client, a=20, b=0.1, c=0.1, d=0.1)]
    for document in documents:
        found = [key for key in _keys(document) if ACCURACY.match(str(key))]
        assert found == []


def test_the_page_says_why_there_is_no_accuracy(client):
    data = client.get("/api/test-videos").json()
    assert "No true positions" in data["no_accuracy"] and "not accuracy" in data["moved_note"]
    assert "not a dataset" in data["label"] and "not a research result" in data["label"]


def test_it_uses_its_own_folders_and_writes_nothing_else(tmp_path):
    app = _app(tmp_path)
    client = TestClient(app)
    service = app.state.test_videos
    settings = app.state.settings
    assert service.copies_dir != settings.cache_dir and service.detections_dir != settings.detections_dir
    _forecast(client, a=20)
    assert list((tmp_path / "runs").iterdir()) == []
    assert not settings.detections_dir.exists() and not settings.cache_dir.exists()
    assert not any(item["id"] in tv.VIDEOS for item in client.get("/api/workspace/scenes").json()["items"])


def test_boxes_left_for_a_dataset_never_reach_this_section(tmp_path):
    app = _app(tmp_path, ready=False)
    settings = app.state.settings
    settings.detections_dir.mkdir(parents=True)
    (settings.detections_dir / f"{VIDEO}.json").write_text(json.dumps(_boxes()), encoding="utf-8")
    assert TestClient(app).get(f"/api/test-videos/{VIDEO}/tracks").status_code == 409


# --- the list and the API ---------------------------------------------------------------------


def test_every_video_carries_its_credit():
    assert len(tv.STREET_VIDEOS) == len(tv.VIDEOS) == 5
    for video in tv.STREET_VIDEOS:
        assert re.fullmatch(r"[a-z0-9-]+", video.id) and re.fullmatch(r"[0-9a-f]{40}", video.sha1)
        assert video.author and video.licence and video.camera
        assert video.source_url.startswith("https://commons.wikimedia.org/") and "creativecommons.org" in video.licence_url
        assert "/" not in video.file and "\\" not in video.file


def test_the_two_moving_cameras_carry_the_caution():
    moving = {video.id for video in tv.STREET_VIDEOS if video.caution}
    assert moving == {"buchanan-street", "singapore-scramble"}
    assert all(tv.VIDEOS[name].caution == ws.MOVING_CAUTION for name in moving)


def test_the_list_says_which_videos_are_ready(client):
    data = client.get("/api/test-videos").json()
    by_id = {item["id"]: item for item in data["items"]}
    assert list(by_id) == [video.id for video in tv.STREET_VIDEOS]
    assert by_id[VIDEO]["ready"] is True and by_id[VIDEO]["reason"] is None and by_id[VIDEO]["frames"] == 600
    other = by_id["toronto-diagonal"]
    assert other["ready"] is False and "prepare_test_videos" in other["reason"] and other["fps"] is None
    assert data["predictors"] == ["CV", "CA"] and data["controls"]["A"] == {"max": 60.0, "unit": "px", "protocol_max": None}


@pytest.mark.parametrize("path", ["video", "tracks", "forecast"])
def test_a_video_that_is_not_prepared_gets_a_clean_refusal(tmp_path, path):
    client = TestClient(_app(tmp_path, ready=False), raise_server_exceptions=False)
    response = client.get(f"/api/test-videos/{VIDEO}/{path}")
    assert response.status_code == 409 and "prepare_test_videos" in response.json()["detail"]
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("name", ["nobody", "ETH", "..%2F..%2Fsecret", "%2e%2e", "caracas-boulevard.mp4", "CARACAS-BOULEVARD"])
@pytest.mark.parametrize("path", ["video", "tracks", "forecast"])
def test_only_the_listed_videos_can_be_reached(client, name, path):
    response = client.get(f"/api/test-videos/{name}/{path}")
    assert response.status_code == 404 and "Internship" not in response.text


def test_the_video_is_served(client):
    response = client.get(f"/api/test-videos/{VIDEO}/video")
    assert response.status_code == 200 and response.content == b"a video" and response.headers["content-type"] == "video/mp4"


@pytest.mark.parametrize("query", ["a=-1", "a=61", "b=1.5", "c=-0.1", "d=2", "a=text"])
def test_levels_outside_the_sliders_are_refused(client, query):
    assert client.get(f"/api/test-videos/{VIDEO}/forecast?{query}").status_code == 422


@pytest.mark.parametrize("path", ["", f"/{VIDEO}/video", f"/{VIDEO}/tracks", f"/{VIDEO}/forecast"])
@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
def test_nothing_can_be_written(client, method, path):
    assert getattr(client, method)(f"/api/test-videos{path}").status_code == 405


def test_a_damaged_box_file_is_refused_without_a_path(tmp_path):
    client = TestClient(_app(tmp_path, document={"schema": "something else"}), raise_server_exceptions=False)
    response = client.get(f"/api/test-videos/{VIDEO}/tracks")
    assert response.status_code == 409 and str(tmp_path) not in response.text


def test_box_file_for_another_video_is_refused(tmp_path):
    client = TestClient(_app(tmp_path, document=_boxes("toronto-diagonal")), raise_server_exceptions=False)
    assert client.get(f"/api/test-videos/{VIDEO}/tracks").status_code == 409


def test_new_boxes_replace_old_ones_with_no_restart(tmp_path):
    app = _app(tmp_path)
    client = TestClient(app)
    assert len(client.get(f"/api/test-videos/{VIDEO}/tracks").json()["tracks"]) == 2
    document = _boxes()
    document["by_frame"] = [row[:1] for row in document["by_frame"]]
    path = app.state.test_videos.detections_dir / f"{VIDEO}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    import os
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 5_000_000_000))
    assert len(client.get(f"/api/test-videos/{VIDEO}/tracks").json()["tracks"]) == 1


# --- the scripts ------------------------------------------------------------------------------


def _script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_copy_sets_an_even_rate_before_reading_and_cuts_nothing(monkeypatch, tmp_path):
    prepare = _script("prepare_test_videos", monkeypatch)
    command = prepare.copy_command(tmp_path / "in.webm", "30000/1001", tmp_path / "out.mp4")
    assert command.index("-r") < command.index("-i") and command[command.index("-r") + 1] == "30000/1001"
    assert "-t" not in command and "-ss" not in command and command[command.index("-fps_mode") + 1] == "passthrough"
    assert "min(720,ih)" in command[command.index("-vf") + 1]


def test_a_file_that_is_not_the_published_one_is_refused(monkeypatch, tmp_path):
    prepare = _script("prepare_test_videos", monkeypatch)
    (tmp_path / "in.webm").write_bytes(b"something else")
    video = tv.StreetVideo("x", "X", "in.webm", "0" * 40, "A", "CC0", "u", "s", "Fixed")
    with pytest.raises(ValueError, match="not the file"):
        prepare.prepare(video, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_a_copy_keeps_every_frame_at_an_even_rate(monkeypatch, tmp_path):
    prepare = _script("prepare_test_videos", monkeypatch)
    source = tmp_path / "in.webm"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30:duration=1", "-c:v", "libvpx", str(source)],
                   check=True)
    video = tv.StreetVideo("x", "X", "in.webm", hashlib.sha1(source.read_bytes()).hexdigest(), "A", "CC0", "u", "s", "Fixed")
    meta = prepare.prepare(video, tmp_path, tmp_path / "out")
    assert meta == {"width": 1280, "height": 720, "fps": 30.0, "frames": 30}
    assert json.loads((tmp_path / "out" / "x.json").read_text(encoding="utf-8")) == meta
    assert (tmp_path / "out" / "x.mp4").is_file() and [p.name for p in (tmp_path / "out").iterdir() if p.name.startswith(".")] == []
    assert source.read_bytes()[:4] == b"\x1aE\xdf\xa3", "the original is only read"


def test_detection_writes_to_its_own_folder_and_skips_finished_videos(monkeypatch, tmp_path):
    module = _script("yolo_detect_test_videos", monkeypatch)
    copies, out = tmp_path / "copies", tmp_path / "out"
    copies.mkdir()
    (copies / "one.json").write_text("{}", encoding="utf-8")
    (copies / "two.json").write_text("{}", encoding="utf-8")
    out.mkdir()
    (out / "two.json").write_text("done", encoding="utf-8")
    seen = []

    def detect_scene(scene, video_dir, detect, metadata):
        seen.append((scene, video_dir, metadata))
        return {"frames": 1, "by_frame": [[]]}

    monkeypatch.setattr(module, "yolo_detector", lambda *args: (None, {"model": "stand-in"}))
    monkeypatch.setattr(module, "detect_scene", detect_scene)
    assert module.main(["--weights", "w.pt", "--video", str(copies), "--out", str(out)]) == 0
    assert seen == [("one", copies, {"model": "stand-in"})]
    assert json.loads((out / "one.json").read_text(encoding="utf-8"))["frames"] == 1
    assert (out / "two.json").read_text(encoding="utf-8") == "done"


def test_the_default_folders_are_apart_from_every_dataset(monkeypatch):
    module = _script("yolo_detect_test_videos", monkeypatch)
    source = (REPO_ROOT / "scripts" / "yolo_detect_test_videos.py").read_text(encoding="utf-8")
    assert '"test_videos"' in source and '"test_video_detections"' in source and '"detections"' not in source
    assert module.REPO_ROOT == REPO_ROOT
    prepare = (REPO_ROOT / "scripts" / "prepare_test_videos.py").read_text(encoding="utf-8")
    assert 'REPO_ROOT / "videos"' in prepare and '"cache" / "test_videos"' in prepare


def test_videos_stay_out_of_git():
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "videos/" in ignored


def test_the_frozen_predictors_are_what_the_section_uses():
    cv, ca = evaluation.cv_predictor(CFG, HORIZON), evaluation.ca_predictor(CFG, HORIZON)
    x = np.arange(16, dtype=float).reshape(1, 8, 2)
    assert cv.predict(x).tolist() == constant_velocity(x, HORIZON, CFG.frozen["predictors"]["cv_min_final_observations"]).tolist()
    assert ca.predict(x).tolist() == constant_acceleration(x, HORIZON, CFG.frozen["predictors"]["ca_min_final_observations"]).tolist()
    assert ex.ErrorRequest().active == ()
