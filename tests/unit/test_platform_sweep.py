"""The per-scene sweep behind Analysis and Applications. Small scenes here are fixtures, and no value is a result."""

import json
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from trackvision import config
from trackvision.platform.app import create_app
from trackvision.platform.config import Settings, load_settings
from trackvision.platform.services import scene_sweep as ss, workspace as ws
from trackvision.platform.services.result_reader import ResultReader
from trackvision.platform.services.run_registry import RunRegistry
from trackvision.stats.crossing import first_crossing
from trackvision.stats.inverse import inverse_requirement

CFG = config.validate().config
QS = CFG.frozen["analysis"]["inverse_requirement_q"]
STORED_REAL = REPO_ROOT / "runs" / "run-2026-11-01" / "ethhotel" / "analysis.json"
STORED_SYNTHETIC = REPO_ROOT / "runs" / "full-verify-2026-11-01" / "p4_synthetic" / "analysis.json"
STORED_ANALYSIS = REPO_ROOT / "runs" / "run-2026-11-01" / "analysis" / "results.json"
RAW_ETH = REPO_ROOT / "data" / "raw" / "ewap" / "ewap_dataset" / "seq_eth" / "seq_eth.avi"
CONDITIONS = {"A": "A", "B": "B2", "C": "C", "D": "D"}


def finished(request, timeout=120.0):
    """Ask as a page does until the sweep stops calculating."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = request()
        if body["state"] != "calculating":
            return body
        time.sleep(0.05)
    raise AssertionError("the sweep did not finish")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    for name in ("runs", "raw", "manifests"):
        (root / name).mkdir()
    return TestClient(create_app(Settings(runs_dir=root / "runs", manifests_dir=root / "manifests", raw_dir=root / "raw",
                                          cache_dir=root / "cache", detections_dir=root / "none")), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def synthetic(client):
    return finished(lambda: client.get("/api/workspace/scenes/SYN-501/sweep").json())


def test_the_levels_are_the_frozen_grids_without_their_zero():
    levels = ss.tested_levels(CFG, "m", ws.MAX_SIGMA_PX)
    for code in "ABCD":
        assert [0.0, *levels[code]] == list(CFG.frozen["injection"]["grids"][code]["levels"])


def test_pixel_noise_is_the_metre_grid_scaled_to_the_top_of_the_slider():
    levels = ss.tested_levels(CFG, "px", ws.MAX_SIGMA_PX)
    assert levels["A"] == [6.0, 12.0, 24.0, 42.0, 60.0] and levels["A"][-1] == ws.MAX_SIGMA_PX
    for code in "BCD":
        assert levels[code] == ss.tested_levels(CFG, "m", ws.MAX_SIGMA_PX)[code], "only position noise has a unit"


def test_a_sweep_covers_every_error_at_every_level(synthetic):
    assert synthetic["state"] == "ready" and synthetic["done"] == synthetic["total"] == 20 and synthetic["exploratory"] is True
    assert [e["code"] for e in synthetic["errors"]] == list("ABCD")
    assert synthetic["unit"] == "m" and synthetic["seeds"] == [0, 1, 2] and synthetic["seconds_per_step"] == 0.4
    for error in synthetic["errors"]:
        assert len(error["levels"]) == error["tested"] == 5
        assert error["better"] is not None and set(error["limits"]) == {str(q) for q in QS}
        for level in error["levels"]:
            assert set(level["predictors"]) == {"CV", "CA"} and level["achieved"] is not None and level["coverage"] is not None
            assert len(level["predictors"]["CV"]["de_by_step"]) == CFG.frozen["protocol"]["static"]["horizon_len"]


def test_clean_values_are_the_workspaces_clean_values(client, synthetic):
    injected = client.get("/api/workspace/scenes/SYN-501/inject", params={"a": 0.2}).json()["predictors"]
    for name in ("CV", "CA"):
        assert synthetic["clean"][name]["ade"] == pytest.approx(injected[name]["clean_ade"], abs=1e-12)
        assert synthetic["clean"][name]["fde"] == pytest.approx(injected[name]["clean_fde"], abs=1e-12)


def test_a_level_is_the_workspaces_result_at_that_setting(client, synthetic):
    for code, parameter in (("A", "a"), ("B", "b"), ("C", "c"), ("D", "d")):
        row = next(e for e in synthetic["errors"] if e["code"] == code)["levels"][2]
        body = client.get("/api/workspace/scenes/SYN-501/inject", params={parameter: row["requested"]}).json()
        for name in ("CV", "CA"):
            assert row["predictors"][name]["ade"] == pytest.approx(body["predictors"][name]["ade"], abs=1e-12)
            assert row["predictors"][name]["fde"] == pytest.approx(body["predictors"][name]["fde"], abs=1e-12)
        assert row["coverage"] == pytest.approx(body["predictors"]["CV"]["prediction_coverage"], abs=1e-12)
        assert row["achieved"] == pytest.approx(body["achieved"][code]["mean"], abs=1e-12)


def test_windows_that_only_lose_others_keep_their_forecast_exactly(synthetic):
    """Missing detections and broken tracks remove windows and never change the ones left, so their rise is exactly zero."""
    for error in synthetic["errors"]:
        if error["code"] in "BC":
            for level in error["levels"]:
                for row in level["predictors"].values():
                    if row["ade"] is not None:
                        assert row["ade_rise"] == 0.0 and row["fde_rise"] == 0.0
            assert error["levels"][-1]["coverage"] < 0.5


def test_position_noise_rises_are_measured_on_every_window(synthetic):
    top = next(e for e in synthetic["errors"] if e["code"] == "A")["levels"][-1]
    assert top["coverage"] == 1.0
    assert top["predictors"]["CV"]["ade_rise"] == pytest.approx(top["predictors"]["CV"]["ade_degradation"], abs=1e-12)
    assert top["predictors"]["CV"]["ade_rise"] > 0


def test_a_limit_on_noise_uses_ade_and_a_limit_on_missing_detections_uses_windows_lost():
    def row(achieved, cv, coverage):
        return {"achieved": achieved, "coverage": coverage, "predictors": {"CV": {"ade_degradation": cv}}}
    rows = [row(1.0, 0.05, 0.9), row(2.0, 0.3, 0.7)]
    on_ade = ss.limits("A", rows, ["CV"], [0.1])["0.1"]["CV"]
    on_lost = ss.limits("B", rows, ["CV"], [0.2])["0.2"]["CV"]
    assert on_ade["status"] == "crossed" and on_ade["value"] == pytest.approx(1.2)
    assert on_lost["status"] == "crossed" and on_lost["value"] == pytest.approx(1.5)


def test_a_limit_never_reached_says_how_far_it_was_tested():
    rows = [{"achieved": 3.0, "coverage": 1.0, "predictors": {"CV": {"ade_degradation": 0.01}}}]
    limit = ss.limits("D", rows, ["CV"], [0.05])["0.05"]["CV"]
    assert limit == {"status": "not_reached", "value": None, "verified_up_to": 3.0, "tested_up_to": 3.0}


def test_a_level_nobody_could_be_predicted_at_stays_undefined():
    clean = {"CV": {"ade": 1.0}, "CA": {"ade": 2.0}}
    rows = [{"achieved": 1.0, "coverage": 0.2, "predictors": {"CV": {"ade": 1.5, "ade_degradation": 0.5}, "CA": {"ade": 2.5, "ade_degradation": 0.25}}},
            {"achieved": 2.0, "coverage": 0.0, "predictors": {"CV": {"ade": None, "ade_degradation": None}, "CA": {"ade": None, "ade_degradation": None}}}]
    assert ss.better(rows, clean) == {"status": "truncated", "at": None, "better_when_clean": "CV"}
    assert ss.limits("A", rows[1:], ["CV"], [0.1])["0.1"]["CV"]["status"] == "truncated"


def test_the_predictor_better_on_clean_tracks_is_named(synthetic):
    assert all(error["better"]["better_when_clean"] == "CV" for error in synthetic["errors"])


def test_a_scene_that_does_not_exist_is_refused(client):
    assert client.get("/api/workspace/scenes/NOPE/sweep").status_code == 404


def test_a_scene_with_no_annotations_here_says_so(client):
    response = client.get("/api/workspace/scenes/UNIV-1/sweep")
    assert response.status_code == 409 and "not on this machine" in response.json()["detail"]


def test_a_sweep_writes_nothing(client, synthetic, tmp_path):
    root = client.app.state.settings.runs_dir
    assert list(root.iterdir()) == []
    source = (REPO_ROOT / "src/trackvision/platform/services/scene_sweep.py").read_text(encoding="utf-8")
    assert "write_text" not in source and "open(" not in source and "atomic_write" not in source


# --- the queue ---------------------------------------------------------------------------------


class Queue:
    """A queue whose levels take no time and record the order they ran in."""

    def __init__(self, monkeypatch, levels=None, fail_on=None):
        self.order, self.gate, self.entered, self.fail_on = [], threading.Event(), threading.Event(), fail_on
        self.gate.set()
        self.levels = levels or {"A": [1.0, 2.0]}
        monkeypatch.setattr(ss, "clean_reference", lambda loaded: {"CV": {"ade": 1.0, "fde": 1.0}, "CA": {"ade": 2.0, "fde": 2.0}})
        monkeypatch.setattr(ss, "measure_level", self._measure)
        self.queue = ss.SweepQueue(lambda scene: ss.Job(scene, self.levels, [0], CFG, [0.1]), _Loaded)
        self.entry = {"unit": "m", "caution": None}

    def _measure(self, loaded, code, level, seeds, cfg, clean):
        self.entered.set()
        self.gate.wait(5)
        self.order.append((loaded.scene, level))
        if self.fail_on == level:
            self.fail_on = None
            raise RuntimeError("stopped")
        return {"requested": level, "achieved": level, "coverage": 1.0,
                "predictors": {name: {"ade": 1.0, "fde": 1.0, "ade_degradation": 0.0} for name in ("CV", "CA")}}

    def ask(self, scene):
        return self.queue.request(scene, self.entry)


class _Loaded:
    windows = ()
    base = type("Base", (), {"caution": None, "seconds_per_step": None})()

    def __init__(self, scene):
        self.scene = scene


def test_finished_levels_are_kept_and_not_measured_again(monkeypatch):
    fake = Queue(monkeypatch)
    assert finished(lambda: fake.ask("S"))["done"] == 2
    assert fake.ask("S")["state"] == "ready" and fake.order == [("S", 1.0), ("S", 2.0)]


def test_two_requests_never_measure_a_level_twice(monkeypatch):
    fake = Queue(monkeypatch)
    fake.gate.clear()
    fake.ask("S"), fake.ask("S")
    fake.gate.set()
    finished(lambda: fake.ask("S"))
    assert fake.order == [("S", 1.0), ("S", 2.0)]


def test_the_scene_opened_last_goes_first(monkeypatch):
    fake = Queue(monkeypatch, {"A": [1.0, 2.0]})
    fake.gate.clear()
    fake.ask("FIRST")
    assert fake.entered.wait(5)
    fake.ask("SECOND")
    fake.gate.set()
    finished(lambda: (fake.ask("FIRST"), fake.ask("SECOND"))[1])
    finished(lambda: fake.ask("FIRST"))
    assert fake.order == [("FIRST", 1.0), ("SECOND", 1.0), ("SECOND", 2.0), ("FIRST", 2.0)]


def test_a_scene_nobody_is_looking_at_stops_calculating(monkeypatch):
    fake = Queue(monkeypatch, {"A": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(ss, "IDLE_SECONDS", 0.05)
    fake.gate.clear()
    fake.ask("S")
    assert fake.entered.wait(5)
    time.sleep(0.1)
    fake.gate.set()
    time.sleep(0.2)
    assert len(fake.order) == 1, "only the level already running finishes"
    assert finished(lambda: fake.ask("S"))["done"] == 3, "and asking again carries on from there"


def test_a_stopped_sweep_says_so_once_and_asking_again_retries(monkeypatch):
    fake = Queue(monkeypatch, fail_on=2.0)
    body = finished(lambda: fake.ask("S"))
    assert body["state"] == "failed" and body["reason"] == ss.FAILED_REASON and ":\\" not in body["reason"]
    assert finished(lambda: fake.ask("S"))["state"] == "ready"


# --- the stored results ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    settings = load_settings()
    return ws.Workspace(settings, RunRegistry(settings.runs_dir, ResultReader()))


def _stored(path):
    return json.loads(path.read_text(encoding="utf-8"))["curves"]["per_history"]["8"]


@pytest.mark.parametrize("scene", ["ETH", "HOTEL", "SYN-501", "SYN-502", "SYN-503", "SYN-504", "SYN-505"])
def test_the_live_sweep_equals_the_stored_sweep(real, scene):
    """The strongest check here: on every scene of the stored sweeps, every value of every level is the stored value."""
    path = STORED_SYNTHETIC if scene.startswith("SYN") else STORED_REAL
    if not path.is_file() or not STORED_ANALYSIS.is_file() or (scene in ("ETH", "HOTEL") and not RAW_ETH.is_file()):
        pytest.skip("the stored runs or the dataset are not on this machine")
    live = finished(lambda: real.sweep(scene))
    stored, analysis = _stored(path)[scene], json.loads(STORED_ANALYSIS.read_text(encoding="utf-8"))
    for error in live["errors"]:
        condition = CONDITIONS[error["code"]]
        for name in ("CV", "CA"):
            levels = stored[name][condition]["levels"]
            assert levels[0]["requested"] == 0.0
            assert live["clean"][name]["ade"] == pytest.approx(levels[0]["ade_m"]["mean"], abs=1e-12)
            assert live["clean"][name]["fde"] == pytest.approx(levels[0]["fde_m"]["mean"], abs=1e-12)
            for ours, theirs in zip(error["levels"], levels[1:], strict=True):
                row = ours["predictors"][name]
                assert ours["requested"] == theirs["requested"]
                assert ours["achieved"] == pytest.approx(theirs["achieved_seed_mean"], abs=1e-12)
                assert ours["coverage"] == pytest.approx(theirs["prediction_coverage"]["mean"], abs=1e-12)
                assert row["ade"] == pytest.approx(theirs["ade_m"]["mean"], abs=1e-12)
                assert row["fde"] == pytest.approx(theirs["fde_m"]["mean"], abs=1e-12)
                assert row["ade_degradation"] == pytest.approx(theirs["d_ade"]["mean"], abs=1e-12)
                assert row["ade_rise"] == pytest.approx(theirs["d_ade_paired"]["mean"], abs=1e-12)
                assert row["de_by_step"] == pytest.approx(theirs["de_by_step_m"], abs=1e-9)
            if error["limit_on"] == ss.ON_ADE:
                group = next(g for g in analysis["e7_inverse_mapping"].values() if f"{scene}/{name}/{condition}/q0.1" in g["entries"])
                for q in QS:
                    kept, ours = group["entries"][f"{scene}/{name}/{condition}/q{q}"], error["limits"][str(q)][name]
                    assert ours["status"] == kept["status"]
                    assert ours["value"] == (None if kept["s_star"] is None else pytest.approx(kept["s_star"], abs=1e-12))
        group = next(g for g in analysis["e8_threshold_crossings"].values() if f"{scene}/CV-CA/{condition}" in g["per_scene"])
        kept = group["per_scene"][f"{scene}/CV-CA/{condition}"]
        assert error["better"]["status"] == kept["status"]
        assert error["better"]["at"] == (None if kept["s_crossing"] is None else pytest.approx(kept["s_crossing"], abs=1e-12))


def test_the_limits_and_crossings_are_the_frozen_arithmetic_unchanged():
    points = [(0.0, 0.0), (1.0, 0.05), (2.0, 0.3)]
    assert ss.inverse_requirement is inverse_requirement and ss.first_crossing is first_crossing
    assert inverse_requirement(points, 0.1)["s_star"] == pytest.approx(1.2)
