"""The read-only API on small hand-built run folders. Scene names and values are fixtures, and no value is a result."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from conftest import REPO_ROOT
from trackvision.experiments import store
from trackvision.experiments.store import RunStore
from trackvision.platform.app import WITHHELD, create_app, withhold_paths
from trackvision.platform.config import Settings, load_settings
from trackvision.platform.services import result_views
from trackvision.platform.services.result_reader import ArtifactMalformed, ResultReader
from trackvision.platform.services.run_registry import RunRegistry, validate_id, InvalidExperimentId

BASE = REPO_ROOT / "configs" / "base.yaml"
SYNTHETIC_LABEL = "SYNTHETIC ENGINEERING FEASIBILITY ONLY. Fixture label."
LOCAL_LABEL = "DESCRIPTIVE AND LOCAL ONLY. Fixture label."
NO_WINDOWS = "no_predicted_windows"


def write(path: Path, doc) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def stat(mean, n_valid=3):
    return {"mean": mean, "sd": None if mean is None else 0.1, "n_valid": n_valid, "n_undefined": 3 - n_valid, "undefined": {}}


def level(requested, achieved, ade):
    undefined = ade is None
    return {"requested": requested, "achieved_seed_mean": achieved, "s_norm": achieved, "ade_m": stat(ade, 0 if undefined else 3),
            "fde_m": stat(None if undefined else ade * 2, 0 if undefined else 3), "rmse_m": stat(None if undefined else ade * 1.5, 0 if undefined else 3),
            "prediction_coverage": stat(0.0 if undefined else 1.0), "d_ade": stat(None if undefined else ade - 1.0, 0 if undefined else 3),
            "d_fde": stat(None), "de_by_step_m": [] if undefined else [ade, ade * 2],
            "windows_by_seed": {"0": {"eligible": 10, "predicted": 0 if undefined else 10}},
            "undefined_reason_by_seed": {"0": NO_WINDOWS if undefined else None}}


def curve(ade_top=None):
    levels = [level(0.0, 0.0, 1.0), level(0.1, 0.13, 1.4), level(0.2, 0.21, ade_top)]
    return {"levels": levels, "audc": {"status": "computable", "value": 0.5}, "robustness": {"value": 1.5, "reason": None},
            "normalization": {"measure": "rmse_pos_m", "levels": [
                {"requested": l["requested"], "achieved_seed_mean": l["achieved_seed_mean"], "achieved_by_seed": {"0": l["achieved_seed_mean"]},
                 "seed_spread": {"sd": 0.0}} for l in levels]}}


def analysis_doc(scenes, across=True, label=SYNTHETIC_LABEL):
    predictors = ("CV", "GRU")
    curves = {scene: {p: {"A": curve(), "B2": curve(1.9)} for p in predictors} for scene in scenes}
    clean = {scene: {p: {k: stat(1.0) for k in ("ade_m", "fde_m", "rmse_m", "prediction_coverage")} for p in predictors} for scene in scenes}
    pooled = {p: {"A": {"levels": [{"requested": 0.0, "achieved": stat(0.0), "ade_m": stat(1.25), "fde_m": stat(2.5), "rmse_m": stat(1.75), "prediction_coverage": stat(1.0)}]}}
              for p in predictors}
    doc = {"label": label, "scope": "fixture", "research_results": False, "aggregation": "seeds within scene/fold, then across scenes",
           "population_labels": {"per_history": "Primary."}, "blocked": {"P-8": "moving_dataset is unresolved"}, "failures": [],
           "curves": {"per_history": {"8": curves, "4": curves}}, "clean": {"per_history": {"8": clean, "4": clean}},
           "monotonicity": [{"scene": scenes[0], "condition": "A", "rule": "strict", "status": "pass"}],
           "hypotheses": {"H1": {"research_verdict": "not_evaluable", "reason": "fixture"}}, "across_scenes": {}, "family_audc": {}}
    if across:
        doc["across_scenes"] = {"per_history": {"8": pooled}}
        doc["family_audc"] = {"per_history": {"8": {scenes[0]: {"A": {"value": 3.0, "status": "computable"}}}}}
        doc["p15"] = {"per_history": {"CV": {"A": {"audc_best": 2.0, "minimum_history": 4, "partial_folds": False, "status": "ok", "reason": None, "eligible_windows": {}}}}}
        doc["p12"] = {"GRU": {"A": []}}
    return doc


def sweep(root: Path, scope="synthetic_engineering", scenes=("SYN-1", "SYN-2"), across=True, label=SYNTHETIC_LABEL, status=None, analysis=True):
    spec = {"scope": scope, "label": label, "scenes": list(scenes), "predictors": ["CV", "GRU"], "conditions": ["A", "B2"], "histories": [4, 8],
            "seeds": [0, 1, 2], "horizon_len": 12, "config_hash": "c" * 64, "source_sha256": "d" * 64, "common_training": False}
    run = RunStore.create(root, spec)
    identity = {"experiment": "P-4", "scene": scenes[0], "predictor": "CV", "condition": "A", "requested_level": 0.1, "seed": 0, "history_len": 8,
                "horizon_len": 12, "population": "per_history", "fold": scenes[0], "scope": scope}
    run.put(identity, store.COMPLETE, {"achieved": {"measure": "rmse_pos_m", "value": 0.13, "detail": {"removed_observations": 7}},
                                       "metrics": {"ade_m": 1.4, "fde_m": 2.8, "rmse_m": 2.1, "prediction_coverage": 1.0, "undefined_reason": None}})
    run.put({"experiment": "P-8", "scope": scope}, store.BLOCKED, {"reason": "moving_dataset is unresolved"})
    write(root / "run_status.json", status or {
        "scope": scope, "scope_complete": True, "research_complete": False, "expected_cells": 1, "finished_cells": 1,
        "by_status": {"complete": 1, "failed": 0, "blocked": 1, "unrealized": 0}, "by_experiment": {"P-4": {"complete": 1}, "P-8": {"complete": 0, "blocked": 1}}})
    if analysis:
        write(root / "analysis.json", analysis_doc(list(scenes), across, label))
    (root / "figures").mkdir()
    (root / "figures" / f"degradation_{scenes[0]}_A_h8.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    return root


def mot17(root: Path, mota=0.5):
    official = {"MOTA": mota, "IDF1": 0.6, "HOTA": 0.55, "IDSW": 12, "Frag": 3.0}
    results = {"experiment": "E1, E2, E3", "label": "LIMITED MOT17 FEASIBILITY (decision 0006). Fixture.", "research_results": False, "pooled_result": None,
               "primary_detection_set": "FRCNN", "optional_detection_sets": [], "sequences": ["MOT17-02"], "association_mechanism": "real_tracker_iou",
               "association_status": "established", "prediction": "MOT17 is never used for trajectory prediction", "not_p1": True, "not_p9": True,
               "by_detection_set": {"FRCNN": {"MOT17-02": {"frame_rate": 30, "frames_evaluated": [301, 600], "official_trackeval": official, "project_diagnostics": {"detection": {"recall": 0.33}}}}}}
    path = write(root / "results.json", results)
    write(root / "manifest.json", {"name": "mot17-engineering", "dataset": "MOT17", "timestamp_utc": "2026-11-01T09:00:00+00:00", "config_hash": "c" * 64,
                                   "git": {"sha": "abc", "dirty": True}, "licence": "CC BY-NC-SA 3.0",
                                   "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trackeval_commit": "12c8791b"})
    write(root / "FRCNN" / "tracker" / "results.json", {"settings": {"tracker": {"name": "IouTracker", "match_iou": 0.3}}})
    return root


def e4(root: Path, scene="SYN-1"):
    predictor = {"seed_mean": {"tracker_ade_m": stat(1.9, 1), "tracker_fde_m": stat(4.2, 1), "clean_all_eligible_ade_m": stat(1.6, 1), "d_ade_paired": stat(0.1, 1)}, "by_seed": {}}
    results = {"experiment": "E4", "scope": "synthetic_tracker_engineering", "label": "SYNTHETIC ENGINEERING ONLY (E4, decision 0006). Fixture.", "research_results": False,
               "association_mechanism": None, "association_status": "unresolved", "association_blocker": "No permitted value can be recorded.", "scenes": [scene], "not_p9": True,
               "official_trackeval": {"note": "not a benchmark", scene: {"MOTA": 0.9, "IDF1": 0.7, "HOTA": 0.8, "IDSW": 38, "Frag": 61.0}},
               "provenance": {"detector": "synthetic blob detector", "tracker": "existing IoU tracker; not ByteTrack or BoT-SORT"},
               "by_scene": {scene: {"project_diagnostics": {"position_rmse_m": 0.28, "error_profile": {"dropout": {"rate": {"value": 0.07}}, "fragmentation": {"events": 17}, "identity_exchange": {"events": 0}}}, "prediction": {"coverage": {"prediction_coverage": 0.7}, "predictors": {"CV": predictor}}}}}
    path = write(root / "results.json", results)
    write(root / "manifest.json", {"schema": store.SCHEMA, "spec": {"scope": "synthetic_tracker_engineering", "tracker": {"match_iou": 0.3, "max_age": 25, "min_hits": 3}, "trackeval_commit": "12c8791b"},
                                   "created_utc": "2026-11-01T06:00:00+00:00"})
    write(root / "run_manifest.json", {"timestamp_utc": "2026-11-01T06:10:00+00:00", "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return root


def engineering(root: Path):
    results = {"label": "ENGINEERING AND DESCRIPTIVE ONLY (E5 to E8, decision 0006). Fixture.", "research_results": False, "blocked": {"P-1": "needs YOLO"},
               "deferred": {"E10/section 3.8": "collision population unresolved"}, "family_across_scenes": {"by_condition": {"A": {"value": 3.0}}},
               "e5_p13_engineering": {"level": "engineering_only", "h7_verdict": "not_evaluable"},
               "e6_dominant_dry_run": {"status": "engineering_dry_run", "target": "A", "research_selection_applied": False},
               "e7_inverse_mapping": {"synthetic": {"level": "engineering_only", "entries": {}}}, "e8_threshold_crossings": {"synthetic": {"level": "engineering_only"}},
               "association_by_experiment": {"E5": {"association_mechanism": None, "association_status": "unresolved"}}}
    path = write(root / "results.json", results)
    write(root / "manifest.json", {"timestamp_utc": "2026-11-01T09:10:00+00:00", "inputs": {"e4_results_sha256": "e" * 64},
                                   "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return root


def smoke(root: Path):
    write(root / "manifest.json", {"kind": "smoke", "experiment_id": "smoke-fixture-sna", "scientific_validity": "none - scaffold check only", "timestamp_utc": "2026-10-01T00:00:00+00:00"})
    write(root / "fixture_summary.json", {"checks": {"constant_heading": True}})
    return root


def geometry(root: Path):
    write(root / "report.json", {"timestamp_utc": "2026-11-01T06:00:00+00:00", "passed": True, "notice": "LOCAL ONLY - dataset permissions unconfirmed",
                                 "blocked_scenes": {"UNIV": "BLOCKED: no image-to-world homography"},
                                 "scenes": {"ETH": {"diagnostics": {"annotation_count": 8908, "track_count": 360}, "roundtrip": {"passed": True, "max_px": 2e-13},
                                                    "windows": {"count": 2614}, "speed_diagnostic": {"median_mps": 1.47}}}})
    return root


@pytest.fixture
def runs(tmp_path):
    root = tmp_path / "runs"
    sweep(root / "syn")
    sweep(root / "local", scope="ethhotel_descriptive", scenes=("ETH", "HOTEL"), across=False, label=LOCAL_LABEL)
    mot17(root / "group" / "mot17")
    e4(root / "group" / "e4")
    engineering(root / "group" / "analysis")
    smoke(root / "smoke-fixture-sna")
    geometry(root / "geometry")
    (root / "logs").mkdir()
    (root / "logs" / "pipeline.log").write_text("not a run", encoding="utf-8")
    return root


def client_for(tmp_path, runs_dir):
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    settings = Settings(runs_dir=runs_dir, manifests_dir=manifests, base_config=BASE)
    return TestClient(create_app(settings), raise_server_exceptions=False)


@pytest.fixture
def client(tmp_path, runs):
    return client_for(tmp_path, runs)


def ids(client):
    return {row["experiment_id"]: row for row in client.get("/api/experiments").json()["items"]}


def test_health_reports_status_version_records_and_time(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["read_only"] is True and body["version"]
    assert body["results_directory_available"] is True and body["experiment_records"] == 7 and body["timestamp"]


def test_health_is_degraded_when_the_results_folder_is_absent(tmp_path):
    body = client_for(tmp_path, tmp_path / "absent").get("/api/health").json()
    assert body["status"] == "degraded" and body["results_directory_available"] is False and body["experiment_records"] == 0


def test_an_empty_results_folder_gives_empty_lists_not_errors(tmp_path):
    (tmp_path / "runs").mkdir()
    client = client_for(tmp_path, tmp_path / "runs")
    assert client.get("/api/experiments").json()["items"] == [] and client.get("/api/datasets").json()["items"] == []
    for view in ("tracking", "error-injection", "prediction", "analysis"):
        assert client.get(f"/api/{view}").json() == {"items": [], "default": None}


def test_discovery_finds_each_run_once_and_stops_at_a_run(client):
    found = ids(client)
    assert set(found) == {"syn", "local", "group~mot17", "group~e4", "group~analysis", "smoke-fixture-sna", "geometry"}
    assert not any("FRCNN" in key or "logs" in key for key in found)


def test_each_run_is_classified_from_its_own_records(client):
    assert {k: v["kind"] for k, v in ids(client).items()} == {
        "syn": "sweep", "local": "sweep", "group~mot17": "mot17_tracking", "group~e4": "synthetic_tracker",
        "group~analysis": "engineering_analysis", "smoke-fixture-sna": "smoke", "geometry": "geometry_validation"}


def test_runs_are_listed_newest_first_and_can_be_filtered(client):
    stamps = [r["timestamp"] for r in client.get("/api/experiments").json()["items"]]
    assert stamps == sorted(stamps, reverse=True)
    assert [r["kind"] for r in client.get("/api/experiments", params={"kind": "smoke"}).json()["items"]] == ["smoke"]


def test_a_new_run_appears_without_a_restart(tmp_path, runs):
    registry = RunRegistry(runs, ResultReader(), rescan_seconds=0)
    before = len(registry.runs())
    smoke(runs / "smoke-later")
    assert len(registry.runs()) == before + 1


def test_evidence_levels_are_never_research(client):
    levels = {k: v["evidence_level"] for k, v in ids(client).items()}
    assert levels["syn"] == levels["group~mot17"] == levels["group~e4"] == "engineering_only"
    assert levels["local"] == "descriptive_local_only" and levels["group~analysis"] == "engineering_and_descriptive"
    assert levels["smoke-fixture-sna"] == "scaffold_check" and not any("research" in (v or "") for v in levels.values())


def test_stored_labels_and_research_flags_pass_through(client):
    found = ids(client)
    assert found["syn"]["label"] == SYNTHETIC_LABEL and found["local"]["label"] == LOCAL_LABEL
    assert found["group~e4"]["label"].startswith("SYNTHETIC ENGINEERING ONLY (E4")
    assert found["syn"]["research_results"] is False and found["group~mot17"]["research_results"] is False


def test_provenance_keeps_hashes_git_and_association(client):
    p = ids(client)["group~mot17"]["provenance"]
    assert p["config_hash"] == "c" * 64 and p["git"] == {"sha": "abc", "dirty": True} and p["trackeval_commit"] == "12c8791b"
    assert p["association_mechanism"] == "real_tracker_iou" and p["licence"] == "CC BY-NC-SA 3.0"


def test_e4_keeps_its_unresolved_association_as_null_with_the_blocker(client):
    p = ids(client)["group~e4"]["provenance"]
    assert p["association_mechanism"] is None and p["association_status"] == "unresolved" and "No permitted value" in p["association_blocker"]


def test_a_bound_sidecar_is_exposed_and_a_stale_one_is_flagged(tmp_path, runs):
    results = runs / "group" / "mot17" / "results.json"
    sidecar = {"record": "results.json", "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest(), "entries": [{"experiment": "E1"}]}
    write(runs / "group" / "mot17" / "association_provenance.json", sidecar)
    client = client_for(tmp_path, runs)
    assert ids(client)["group~mot17"]["provenance"]["association_sidecar"]["bound_to_results"] is True
    write(runs / "group" / "mot17" / "association_provenance.json", {**sidecar, "results_sha256": "0" * 64})
    assert ids(client_for(tmp_path, runs))["group~mot17"]["provenance"]["association_sidecar"]["bound_to_results"] is False


def test_individual_lookup_returns_views_and_artifacts(client):
    body = client.get("/api/experiments/syn").json()
    assert body["experiment"]["experiment_id"] == "syn" and body["views"] == ["error-injection", "prediction", "analysis"]
    served = {a["name"]: a["served"] for a in body["artifacts"]}
    assert served["degradation_SYN-1_A_h8.svg"] is True and served["analysis.json"] is False and served["manifest.json"] is False


def test_kinds_without_a_view_still_show_their_stored_values(client):
    assert client.get("/api/experiments/smoke-fixture-sna").json()["stored_summary"]["scientific_validity"].startswith("none")
    assert client.get("/api/experiments/geometry").json()["stored_summary"]["passed"] is True


def test_a_dataset_with_no_manifest_and_no_run_is_not_listed(tmp_path):
    root = tmp_path / "runs"
    mot17(root / "only")
    assert [d["id"] for d in client_for(tmp_path, root).get("/api/datasets").json()["items"]] == ["mot17"]


def test_datasets_list_what_stored_runs_used(client):
    body = {d["id"]: d for d in client.get("/api/datasets").json()["items"]}
    assert set(body) == {"ewap", "ucy", "mot17", "synthetic"}
    mot = body["mot17"]
    assert [s["name"] for s in mot["scenes"]] == ["MOT17-02"] and mot["scenes"][0]["regime"] == "static"
    assert mot["ground_truth"]["boxes"] == "available" and mot["ground_truth"]["trajectories"] == "not_used" and "group~mot17" in mot["experiments"]


def test_dataset_manifests_add_datasets_and_blocked_scenes_keep_their_reason(tmp_path, runs):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "ewap.sha256").write_text("# comment\nabc  a.txt\n", encoding="utf-8")
    (tmp_path / "manifests" / "ucy.sha256").write_text("abc  b.txt\ndef  c.txt\n", encoding="utf-8")
    body = {d["id"]: d for d in client_for(tmp_path, runs).get("/api/datasets").json()["items"]}
    assert body["ewap"]["manifest"]["listed_entries"] == 1 and body["ewap"]["ground_truth"]["boxes"] == "unavailable"
    assert body["ewap"]["scenes"][0]["metadata"]["annotations"] == 8908 and "LOCAL ONLY" in body["ewap"]["notice"]
    assert body["ucy"]["scenes"] == [{"name": "UNIV", "regime": "static", "status": "blocked", "reason": "BLOCKED: no image-to-world homography", "metadata": {}}]


def test_tracking_returns_stored_sequence_metrics_and_no_pooled_value(client):
    index = client.get("/api/tracking").json()
    assert index["default"] == "group~mot17" and {r["experiment_id"] for r in index["items"]} == {"group~mot17", "group~e4"}
    body = client.get("/api/tracking/group~mot17").json()
    row = body["groups"][0]["sequences"][0]
    assert row["metrics"] == {"mota": 0.5, "idf1": 0.6, "hota": 0.55, "idSwitches": 12, "fragmentations": 3.0}
    assert row["frames_evaluated"] == [301, 600] and row["regime"] == "static" and body["pooled_result"] is None
    assert body["groups"][0]["tracker_settings"]["tracker"]["name"] == "IouTracker" and body["not_p9"] is True


def test_a_tracking_metric_that_was_not_stored_stays_null(tmp_path, runs):
    path = runs / "group" / "mot17" / "results.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    del doc["by_detection_set"]["FRCNN"]["MOT17-02"]["official_trackeval"]["HOTA"]
    write(path, doc)
    manifest = json.loads((runs / "group" / "mot17" / "manifest.json").read_text(encoding="utf-8"))
    write(runs / "group" / "mot17" / "manifest.json", {**manifest, "results_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    row = client_for(tmp_path, runs).get("/api/tracking/group~mot17").json()["groups"][0]["sequences"][0]
    assert row["metrics"]["hota"] is None and row["metrics"]["mota"] == 0.5


def test_e4_tracking_is_served_with_its_own_note(client):
    body = client.get("/api/tracking/group~e4").json()
    assert body["groups"][0]["sequences"][0]["metrics"]["idSwitches"] == 38 and body["official_note"] == "not a benchmark"


def test_error_injection_keeps_requested_and_achieved_apart(client):
    body = client.get("/api/error-injection/syn").json()
    row = next(r for r in body["rows"] if r["scene"] == "SYN-1" and r["predictor"] == "CV" and r["condition"] == "A" and r["requested"] == 0.1)
    assert row["achieved"] == 0.13 and row["achieved_by_seed"] == {"0": 0.13} and row["severity_measure"] == "rmse_pos_m"
    assert body["figures"][0]["figure"] == "degradation" and body["figures"][0]["scene"] == "SYN-1" and body["figures"][0]["condition"] == "A"
    assert body["blocked"]["P-8"] and body["selection"]["history"] == 8


def test_prediction_returns_stored_rows_and_the_stored_across_scene_mean(client):
    body = client.get("/api/prediction/syn").json()
    assert body["rows"][0]["ade_m"] == stat(1.0) and body["rows"][0]["de_by_step_m"] == [1.0, 2.0]
    assert [r["ade_m"]["mean"] for r in body["across_scenes"]] == [1.25, 1.25]


def test_eth_and_hotel_are_never_given_an_across_scene_value(client):
    assert client.get("/api/prediction/local").json()["across_scenes"] == []
    body = client.get("/api/analysis/local").json()
    assert body["across_scenes"] is None and body["family_audc"] is None and body["p15"] is None
    assert "local only" in body["selection"]["scope_label"].lower()


def test_e4_prediction_has_no_invented_rmse_or_horizon(client):
    row = client.get("/api/prediction/group~e4").json()["rows"][0]
    assert row["ade_m"]["mean"] == 1.9 and row["rmse_m"] is None and row["de_by_step_m"] is None and row["prediction_coverage"] == 0.7


def test_an_undefined_level_stays_null_with_its_reason_and_is_not_dropped(client):
    levels = client.get("/api/analysis/syn").json()["curves"]["SYN-1"]["CV"]["A"]["levels"]
    assert len(levels) == 3 and levels[2]["achieved_seed_mean"] == 0.21
    assert levels[2]["ade_m"]["mean"] is None and levels[2]["undefined_reason_by_seed"]["0"] == NO_WINDOWS and levels[2]["de_by_step_m"] == []


def test_values_are_the_stored_values_unchanged(tmp_path, runs):
    stored = json.loads((runs / "syn" / "analysis.json").read_text(encoding="utf-8"))
    served = client_for(tmp_path, runs).get("/api/analysis/syn").json()
    level_on_disk = stored["curves"]["per_history"]["8"]["SYN-2"]["GRU"]["B2"]["levels"][1]
    assert served["curves"]["SYN-2"]["GRU"]["B2"]["levels"][1] == {k: level_on_disk[k] for k in result_views.LEVEL_FIELDS}
    assert served["across_scenes"] == stored["across_scenes"]["per_history"]["8"] and served["hypotheses"] == stored["hypotheses"]


def test_a_selection_the_run_did_not_store_is_refused(client):
    assert client.get("/api/analysis/syn", params={"history": 6}).status_code == 404
    assert client.get("/api/analysis/syn", params={"population": "common_h12"}).status_code == 404
    assert client.get("/api/analysis/syn", params={"population": "../x"}).status_code == 422
    assert client.get("/api/analysis/syn", params={"history": 4}).json()["selection"]["history"] == 4


def test_engineering_analysis_keeps_its_labels_and_verdicts(client):
    body = client.get("/api/analysis/group~analysis").json()
    assert body["e6_dominant_dry_run"]["status"] == "engineering_dry_run" and body["e6_dominant_dry_run"]["research_selection_applied"] is False
    assert body["e5_p13_engineering"]["h7_verdict"] == "not_evaluable" and body["e7_inverse_mapping"]["synthetic"]["level"] == "engineering_only"
    assert body["e8_threshold_crossings"]["synthetic"]["level"] == "engineering_only" and "E10/section 3.8" in body["deferred"]


def test_cells_are_filtered_paged_and_digest_verified(client):
    body = client.get("/api/experiments/syn/cells", params={"predictor": "CV", "history": 8}).json()
    assert body["total"] == 1 and body["digests_verified"] is True
    row = body["items"][0]
    assert row["ade_m"] == 1.4 and row["achieved"]["detail"]["removed_observations"] == 7 and row["requested_level"] == 0.1 and row["seed"] == 0
    blocked = client.get("/api/experiments/syn/cells", params={"status": "blocked"}).json()["items"][0]
    assert blocked["reason"] == "moving_dataset is unresolved" and blocked["ade_m"] is None


def test_a_tampered_cell_fails_the_whole_read(tmp_path, runs):
    path = next((runs / "syn" / "cells").glob("*/*.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))
    (doc.get("result") or {}).update({"reason": "edited", "metrics": {"ade_m": 0.001}})
    write(path, doc)
    response = client_for(tmp_path, runs).get("/api/experiments/syn/cells")
    assert response.status_code == 409 and "failed verification" in response.json()["detail"]


def test_views_of_the_wrong_kind_are_404(client):
    for path in ("/api/tracking/syn", "/api/error-injection/group~mot17", "/api/prediction/group~mot17", "/api/analysis/group~mot17", "/api/experiments/group~mot17/cells"):
        assert client.get(path).status_code == 404, path


def test_a_figure_is_served_with_a_locked_down_content_policy(client):
    response = client.get("/api/experiments/syn/artifacts/degradation_SYN-1_A_h8.svg")
    assert response.status_code == 200 and response.headers["content-type"].startswith("image/svg+xml")
    assert "default-src 'none'" in response.headers["content-security-policy"] and response.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("name", ["manifest.json", "analysis.json", "..%2Fmanifest.json", "..%5Cmanifest.json", "%2e%2e%2fmanifest.json", "missing.svg"])
def test_only_listed_figures_can_be_fetched(client, name):
    assert client.get(f"/api/experiments/syn/artifacts/{name}").status_code in (400, 404)


@pytest.mark.parametrize("value", ["..", "...", "a b", "a/b", "a\\b", "~", "a~", "~a", "a~~b", "a~..~b", "x" * 300, "a;b", "a%00b"])
def test_invalid_ids_are_refused_by_format(value):
    with pytest.raises(InvalidExperimentId):
        validate_id(value)


@pytest.mark.parametrize("path", ["../secret", "..%2F..%2Fsecret", "..%5Csecret", "%2e%2e", "group~..~..~secret", "a b", "a%00b", "C:%5CWindows"])
def test_traversal_attempts_reach_nothing(client, path):
    response = client.get(f"/api/experiments/{path}")
    assert response.status_code in (400, 404) and "Users" not in response.text and "tmp" not in response.text.lower()


def test_an_unknown_id_is_404_and_a_badly_formed_one_is_400(client):
    assert client.get("/api/experiments/nothing-here").status_code == 404
    assert client.get("/api/experiments/a b").status_code == 400


def test_a_folder_outside_the_root_cannot_be_named_even_if_it_holds_a_run(tmp_path, runs):
    smoke(tmp_path / "outside")
    assert client_for(tmp_path, runs).get("/api/experiments/outside").status_code == 404


def test_a_malformed_record_marks_the_run_invalid_without_hiding_it_or_others(tmp_path, runs):
    (runs / "group" / "mot17" / "results.json").write_text("{not json", encoding="utf-8")
    client = client_for(tmp_path, runs)
    found = ids(client)
    assert found["group~mot17"]["status"] == "invalid" and "results.json is not valid JSON" in found["group~mot17"]["status_reason"]
    assert found["syn"]["status"] == "complete" and client.get("/api/health").json()["invalid_records"] == 1
    response = client.get("/api/tracking/group~mot17")
    assert response.status_code == 409 and "invalid" in response.json()["detail"]
    assert client.get("/api/experiments/group~mot17").json()["views"] == []


@pytest.mark.parametrize("content", ['{"mean": NaN}', '{"mean": Infinity}', "[1, 2]", "\"text\""])
def test_records_that_are_not_finite_objects_are_malformed(tmp_path, content):
    path = tmp_path / "x.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ArtifactMalformed):
        ResultReader().read_json(path)


def test_a_run_that_differs_from_its_recorded_hash_is_invalid(tmp_path, runs):
    path = runs / "group" / "analysis" / "results.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["family_across_scenes"]["by_condition"]["A"]["value"] = 99.0
    write(path, doc)
    client = client_for(tmp_path, runs)
    row = ids(client)["group~analysis"]
    assert row["status"] == "invalid" and row["integrity"] == {"status": "mismatch", "verified": 0, "failed": ["results.json"]}
    assert client.get("/api/analysis/group~analysis").status_code == 409


def test_a_recorded_filename_cannot_lead_outside_the_run(tmp_path, runs):
    write(tmp_path / "secret.txt", {})
    manifest = json.loads((runs / "smoke-fixture-sna" / "manifest.json").read_text(encoding="utf-8"))
    write(runs / "smoke-fixture-sna" / "manifest.json", {**manifest, "files": {"../../secret.txt": "0" * 64}})
    assert ids(client_for(tmp_path, runs))["smoke-fixture-sna"]["status"] == "invalid"


def test_missing_analysis_makes_a_sweep_incomplete(tmp_path):
    root = tmp_path / "runs"
    sweep(root / "unfinished", analysis=False)
    client = client_for(tmp_path, root)
    row = ids(client)["unfinished"]
    assert row["status"] == "incomplete" and "analysis.json" in row["status_reason"] and row["latest_in_series"] is False
    assert client.get("/api/analysis/unfinished").status_code == 409 and client.get("/api/analysis").json()["default"] is None


def test_unfinished_cells_make_a_sweep_incomplete(tmp_path):
    root = tmp_path / "runs"
    status = {"scope_complete": False, "expected_cells": 10, "finished_cells": 3, "by_status": {"failed": 0}}
    sweep(root / "partial", status=status)
    assert ids(client_for(tmp_path, root))["partial"]["status_reason"] == "3 of 10 cells finished"


def test_failed_cells_make_a_sweep_incomplete(tmp_path):
    root = tmp_path / "runs"
    sweep(root / "failed", status={"scope_complete": True, "by_status": {"failed": 2}})
    assert ids(client_for(tmp_path, root))["failed"]["status"] == "incomplete"


def test_an_unrecognised_folder_is_invalid_rather_than_treated_as_a_result(tmp_path):
    root = tmp_path / "runs"
    write(root / "mystery" / "results.json", {"value": 1})
    client = client_for(tmp_path, root)
    assert ids(client)["mystery"]["status"] == "invalid" and ids(client)["mystery"]["evidence_level"] is None


def test_only_the_newest_complete_run_of_a_series_is_marked_latest(tmp_path):
    root = tmp_path / "runs"
    mot17(root / "older")
    mot17(root / "newer")
    manifest = json.loads((root / "older" / "manifest.json").read_text(encoding="utf-8"))
    write(root / "older" / "manifest.json", {**manifest, "timestamp_utc": "2026-01-01T00:00:00+00:00"})
    client = client_for(tmp_path, root)
    found = ids(client)
    assert found["newer"]["latest_in_series"] is True and found["older"]["latest_in_series"] is False
    assert client.get("/api/tracking").json()["default"] == "newer"


def test_no_run_is_ever_called_superseded_because_no_record_says_so(client):
    assert not {"superseded"} & {row["status"] for row in ids(client).values()}


def test_absolute_paths_in_stored_records_are_withheld():
    doc = {"a": "C:\\Users\\someone\\data", "b": ["/home/someone/x", "see /Users/x", "D:/other"], "c": {"d": 1.5, "e": None},
           "keep": "runs\\relative\\p4", "url": "https://example.org/a", "text": "ratio: 1/2 at 0.4 s"}
    assert withhold_paths(doc) == {"a": WITHHELD, "b": [WITHHELD, WITHHELD, WITHHELD], "c": {"d": 1.5, "e": None},
                                   "keep": "runs\\relative\\p4", "url": "https://example.org/a", "text": "ratio: 1/2 at 0.4 s"}


def test_a_path_inside_a_record_never_reaches_the_client(tmp_path, runs):
    manifest = json.loads((runs / "smoke-fixture-sna" / "manifest.json").read_text(encoding="utf-8"))
    write(runs / "smoke-fixture-sna" / "manifest.json", {**manifest, "scientific_validity": "C:\\Users\\someone\\secret"})
    assert "someone" not in client_for(tmp_path, runs).get("/api/experiments/smoke-fixture-sna").text


def test_errors_do_not_leak_filesystem_paths(tmp_path, runs):
    (runs / "syn" / "analysis.json").write_text("{bad", encoding="utf-8")
    client = client_for(tmp_path, runs)
    response = client.get("/api/analysis/syn")
    assert response.status_code == 409 and str(tmp_path) not in response.text and response.json()["detail"] == "analysis.json is not valid JSON"


def test_an_unexpected_error_is_a_generic_500_that_the_browser_can_read(client, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("secret detail C:\\Users\\someone")
    monkeypatch.setattr(result_views, "analysis", explode)
    response = client.get("/api/analysis/syn", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 500 and "someone" not in response.text and "secret" not in response.text
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_allows_the_local_frontend_only(client):
    good = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    bad = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert good.headers["access-control-allow-origin"] == "http://localhost:3000" and "access-control-allow-origin" not in bad.headers


def test_cors_origins_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("TRACKVISION_CORS_ORIGINS", "http://localhost:3100, http://127.0.0.1:3100")
    assert load_settings().cors_origins == ("http://localhost:3100", "http://127.0.0.1:3100")


def test_every_route_is_read_only(client):
    methods = {m for route in client.app.routes if getattr(route, "path", "").startswith("/api") for m in route.methods}
    assert methods <= {"GET", "HEAD"}


@pytest.mark.parametrize("method", ["post", "put", "patch", "delete"])
@pytest.mark.parametrize("path", ["/api/experiments", "/api/experiments/syn", "/api/tracking/group~mot17", "/api/analysis/syn"])
def test_write_methods_are_refused(client, method, path):
    assert getattr(client, method)(path).status_code == 405


def _fingerprint(root: Path) -> dict:
    return {str(p.relative_to(root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_exercising_every_endpoint_changes_nothing_on_disk(client, runs):
    before = _fingerprint(runs)
    paths = ["health", "datasets", "experiments", "tracking", "error-injection", "prediction", "analysis"]
    for run in ids(client):
        paths += [f"experiments/{run}", f"experiments/{run}/cells"] + [f"{view}/{run}" for view in ("tracking", "error-injection", "prediction", "analysis")]
    assert all(client.get(f"/api/{path}").status_code < 500 for path in paths)
    assert _fingerprint(runs) == before


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_research_code_never_depends_on_the_platform():
    """Spec v4.15 section 11.1: computational modules never depend on the UI."""
    root = REPO_ROOT / "src" / "trackvision"
    research = [p for p in root.rglob("*.py") if "platform" not in p.relative_to(root).parts]
    assert research and not [str(p) for p in research if any(n.startswith("trackvision.platform") for n in _imports(p))]


def test_the_platform_cannot_change_the_results_digest():
    assert "platform" not in store.RESULT_SOURCES and not any("platform" in m for m in store.RESULT_MODULES)


def test_the_platform_imports_no_write_path_of_the_store():
    source = (REPO_ROOT / "src" / "trackvision" / "platform" / "services" / "cell_index.py").read_text(encoding="utf-8")
    assert ".put(" not in source and ".create(" not in source and "RunStore.open" in source


def test_the_default_results_folder_is_runs_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("TRACKVISION_RUNS_DIR", raising=False)
    assert load_settings().runs_dir == REPO_ROOT / "runs"
    monkeypatch.setenv("TRACKVISION_RUNS_DIR", "elsewhere")
    assert load_settings().runs_dir == Path("elsewhere")


# --- tracker details and the sources of E5 to E8 (decision 0013) -----------------------------------


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_client(tmp_path, runs_dir):
    manifests = tmp_path / "manifests"
    manifests.mkdir(exist_ok=True)
    settings = Settings(runs_dir=runs_dir, manifests_dir=manifests, base_config=BASE, raw_dir=tmp_path / "raw", cache_dir=tmp_path / "cache")
    return TestClient(create_app(settings), raise_server_exceptions=False)


def rows(body):
    return {row["label"]: row["value"] for row in body["config"]}


@pytest.fixture
def tracked(tmp_path):
    root = tmp_path / "runs"
    mot17(root / "group" / "mot17")
    e4(root / "group" / "e4", "SYN-501")
    return scene_client(tmp_path, root), root


def test_tracker_details_on_mot17_are_the_stored_values(tracked):
    body = tracked[0].get("/api/workspace/scenes/MOT17-02/tracker").json()
    assert body["run"] == "group~mot17" and body["recording"] == "MOT17-02" and body["evidence_level"] == "engineering_only"
    assert "LIMITED MOT17 FEASIBILITY" in body["label"]
    config = rows(body)
    assert config["Match IoU"] == 0.3 and config["Tracker"] == "IouTracker, not ByteTrack or BoT-SORT"
    assert config["Frames evaluated"] == "301 to 600" and config["TrackEval commit"] == "12c8791b" and config["Training performed"] == "No"
    assert config["Camera motion compensation"] == "None" and config["ReID"] == "None"
    assert "score 0.5 and above" in config["Workspace overlay"]
    row = body["results"][0]
    assert (row["name"], row["primary"], row["mota"], row["idf1"], row["hota"], row["id_switches"], row["fragmentations"]) == ("FRCNN", True, 0.5, 0.6, 0.55, 12, 3.0)
    assert body["detection"][0]["recall"] == 0.33


def test_a_setting_or_result_a_run_did_not_record_stays_missing(tracked):
    body = tracked[0].get("/api/workspace/scenes/MOT17-02/tracker").json()
    config = rows(body)
    assert config["Min hits"] is None and config["Max age"] is None and config["Detector"] is None and config["Score threshold"] is None
    assert body["results"][0]["false_positives"] is None and body["detection"][0]["precision"] is None
    assert body["error_profile"][0]["missed"] is None and body["error_profile"][0]["localization"] is None


def test_a_mot17_part_shows_its_recordings_run_and_says_the_results_are_the_full_sequences(tracked):
    body = tracked[0].get("/api/workspace/scenes/MOT17-02-1/tracker").json()
    assert body["scene"] == "MOT17-02-1" and body["recording"] == "MOT17-02" and body["run"] == "group~mot17"
    assert body["results_note"] == "Stored results are for the full sequence MOT17-02, frames 301 to 600."
    assert tracked[0].get("/api/workspace/scenes/MOT17-02/tracker").json()["results_note"] is None


def test_a_mot17_sequence_no_run_covers_says_so(tracked):
    body = tracked[0].get("/api/workspace/scenes/MOT17-04/tracker").json()
    assert body["run"] is None and body["note"] == "No stored tracker run covers this sequence." and body["results"] == []


def test_a_synthetic_scene_shows_its_e4_run(tracked):
    body = tracked[0].get("/api/workspace/scenes/SYN-501/tracker").json()
    config = rows(body)
    assert body["run"] == "group~e4" and config["Detector"] == "Synthetic blob detector" and config["Tracker"] == "Existing IoU tracker; not ByteTrack or BoT-SORT"
    assert (config["Match IoU"], config["Min hits"], config["Max age"]) == (0.3, 3, "25 frames")
    row = body["results"][0]
    assert row["mota"] == 0.9 and row["id_switches"] == 38 and body["detection"] == []
    assert (body["error_profile"][0]["missed"], body["error_profile"][0]["localization"], body["error_profile"][0]["unit"]) == (0.07, 0.28, "m")


def test_synthetic_tracker_details_never_carry_the_real_tracker_label(tracked):
    assert "real_tracker_iou" not in tracked[0].get("/api/workspace/scenes/SYN-501/tracker").text


def test_e5_is_named_only_when_it_really_used_that_e4_run(tmp_path):
    root = tmp_path / "runs"
    e4(root / "group" / "e4", "SYN-501")
    engineering(root / "group" / "analysis")
    client = scene_client(tmp_path, root)
    assert rows(client.get("/api/workspace/scenes/SYN-501/tracker").json())["Used by"] is None
    results = root / "group" / "analysis" / "results.json"
    doc = {**json.loads(results.read_text(encoding="utf-8")), "inputs": {"e4_results_sha256": sha(root / "group" / "e4" / "results.json")}}
    path = write(results, doc)
    manifest = root / "group" / "analysis" / "manifest.json"
    write(manifest, {**json.loads(manifest.read_text(encoding="utf-8")), "results_sha256": sha(path)})
    assert rows(scene_client(tmp_path, root).get("/api/workspace/scenes/SYN-501/tracker").json())["Used by"] == "E5"


@pytest.mark.parametrize("scene", ["ETH", "HOTEL", "UNIV", "ZARA2", "ETH-1", "UNIV-3"])
def test_scenes_with_human_annotations_say_no_tracker_runs_there(tracked, scene):
    body = tracked[0].get(f"/api/workspace/scenes/{scene}/tracker").json()
    assert body["run"] is None and "human annotations" in body["note"] and "P-9" in body["note"]
    assert body["config"] == [] and body["results"] == []


def test_a_generated_scene_with_no_e4_run_says_so(tracked):
    body = tracked[0].get("/api/workspace/scenes/SYN-507/tracker").json()
    assert body["run"] is None and "scene generator" in body["note"]


def test_tracker_details_of_an_unknown_scene_are_404(tracked):
    assert tracked[0].get("/api/workspace/scenes/nothing/tracker").status_code == 404
    assert tracked[0].post("/api/workspace/scenes/MOT17-02/tracker").status_code == 405


def _chain(tmp_path):
    root = tmp_path / "runs"
    sweep(root / "syn")
    sweep(root / "local", scope="ethhotel_descriptive", scenes=("ETH", "HOTEL"), across=False, label=LOCAL_LABEL)
    e4(root / "group" / "e4")
    engineering(root / "group" / "analysis")
    results = root / "group" / "analysis" / "results.json"
    doc = {**json.loads(results.read_text(encoding="utf-8")),
           "inputs": {"e4_results_sha256": sha(root / "group" / "e4" / "results.json"), "synthetic_analysis_sha256": sha(root / "syn" / "analysis.json"),
                      "ethhotel_analysis_sha256": sha(root / "local" / "analysis.json")}}
    path = write(results, doc)
    manifest = root / "group" / "analysis" / "manifest.json"
    write(manifest, {**json.loads(manifest.read_text(encoding="utf-8")), "results_sha256": sha(path)})
    return root


def test_each_input_of_e5_to_e8_names_the_run_it_came_from(tmp_path):
    body = scene_client(tmp_path, _chain(tmp_path)).get("/api/analysis/group~analysis").json()
    assert body["input_runs"] == [{"input": "E4 results", "used_by": ["E5"], "run": "group~e4", "unchanged": True},
                                  {"input": "Synthetic curves", "used_by": ["E6", "E7", "E8"], "run": "syn", "unchanged": True},
                                  {"input": "ETH and HOTEL curves", "used_by": ["E7", "E8"], "run": "local", "unchanged": True}]


def test_a_changed_input_of_e5_to_e8_is_reported_not_hidden(tmp_path):
    root = _chain(tmp_path)
    analysis = root / "syn" / "analysis.json"
    analysis.write_text(analysis.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    body = scene_client(tmp_path, root).get("/api/analysis/group~analysis").json()
    assert body["input_runs"][1] == {"input": "Synthetic curves", "used_by": ["E6", "E7", "E8"], "run": None, "unchanged": False}
    assert body["input_runs"][0]["unchanged"] is True


def test_an_e5_to_e8_run_that_records_no_inputs_shows_none_as_unchanged(client):
    assert not any(row["unchanged"] for row in client.get("/api/analysis/group~analysis").json()["input_runs"])
