"""E5 to E8 wiring on small hand-built documents. Scene names here are fixtures, and no value is a result."""

import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT
from trackvision import config
from trackvision.experiments import engineering_analysis as ea
from trackvision.experiments.synthetic_tracker_engineering import SCOPE

CFG = config.validate().config
CONDITIONS = ("A", "B2", "C", "D")
Y = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _curve(scale=1.0, base=1.0):
    return {"normalization": {"measure": "rmse_pos_m"}, "levels": [
        {"achieved_seed_mean": x, "d_ade": {"mean": x * scale}, "ade_m": {"mean": base + x * scale, "n_valid": 3}} for x in Y]}


def _family(value):
    if value is None:
        return {"status": "uncomputable", "value": None, "n_valid": 1, "reason": "fewer than two valid predictor AUDC values"}
    return {"status": "computable", "value": value, "n_valid": 4, "reason": None}


def synthetic_doc(a=3.0, d=0.07):
    scenes = ["SYN-1", "SYN-2"]
    predictors = ["CV", "GRU"]
    curves = {s: {p: {c: _curve(1.0 if p == "CV" else 0.5, 1.0 if p == "CV" else 1.2) for c in CONDITIONS} for p in predictors}
              for s in scenes}
    across = {p: {c: {"levels": [{"ade_m": {"mean": (1.0 if p == "CV" else 1.2) + x * (1.0 if p == "CV" else 0.5), "n_valid": 2},
                                  "achieved": {"mean": x}} for x in Y]} for c in CONDITIONS} for p in predictors}
    family = {s: {"A": _family(a), "B2": _family(-0.04), "C": _family(None if s == "SYN-2" else -0.01), "D": _family(d)}
              for s in scenes}
    return {"scope": ea.SYNTHETIC, "spec": {"scenes": scenes, "predictors": predictors},
            "curves": {"per_history": {"8": curves}}, "across_scenes": {"per_history": {"8": across}},
            "family_audc": {"per_history": {"8": family}}}


def ethhotel_doc():
    scenes, predictors = ["ETH", "HOTEL"], ["CV", "CA"]
    curves = {s: {p: {c: _curve(1.0 if p == "CV" else 2.0) for c in CONDITIONS} for p in predictors} for s in scenes}
    return {"scope": ea.ETHHOTEL, "spec": {"scenes": scenes, "predictors": predictors},
            "curves": {"per_history": {"8": curves}}, "family_audc": {}}


def e4_doc():
    def scene(i):
        def predictor(ade):
            return {"seed_mean": {"tracker_ade_m": {"mean": ade}}}
        return {"prediction": {"predictors": {"CV": predictor(1.0 + i), "GRU": predictor(4.0 - i)}}}
    official = {f"S{i}": {"MOTA": 0.1 * i, "IDF1": 0.2 * i, "HOTA": 0.3 * i, "IDSW": 3 - i} for i in range(3)}
    return {"scope": SCOPE, "by_scene": {f"S{i}": scene(i) for i in range(3)}, "official_trackeval": {"note": "n", **official}}


def test_family_is_aggregated_within_scenes_then_across():
    r = ea.family_across_scenes(synthetic_doc(), CFG)["by_condition"]
    assert r["A"]["value"] == 3.0 and (r["C"]["n_valid_scenes"], r["C"]["n_undefined_scenes"]) == (1, 1)
    assert "fewer than two" in r["C"]["undefined"]["SYN-2"]


def test_only_the_static_representatives_are_summarised():
    assert list(ea.family_across_scenes(synthetic_doc(), CFG)["by_condition"]) == ["A", "B2", "C", "D"]


def test_family_says_it_is_descriptive_and_not_research_evidence():
    r = ea.family_across_scenes(synthetic_doc(), CFG)
    assert "not real-world research evidence" in r["note"] and r["level"] == "engineering_only"


def test_eth_and_hotel_are_never_pooled():
    with pytest.raises(ValueError, match="never pooled"):
        ea.family_across_scenes(ethhotel_doc(), CFG)


def test_the_dry_run_records_population_source_and_counts():
    r = ea.dominant_dry_run(ea.family_across_scenes(synthetic_doc(), CFG), "abc123", CFG)
    assert r["status"] == "engineering_dry_run" and r["target"] == "A" and r["source_run"] == "abc123"
    assert r["contributing_scenes"]["C"] == 1 and r["undefined_scenes"]["C"] == 1 and "SYN-501" in r["population"]
    assert r["research_selection_applied"] is False and r["ethhotel"]["status"] == "not_evaluated"


def test_co_dominance_reaches_the_dry_run():
    r = ea.dominant_dry_run(ea.family_across_scenes(synthetic_doc(a=0.1, d=0.07), CFG), "x", CFG)
    assert r["targets"] == ["A", "D"] and r["co_dominant"] is True


def test_the_frozen_gap_is_read_from_configuration():
    r = ea.dominant_dry_run(ea.family_across_scenes(synthetic_doc(), CFG), "x", CFG)
    assert r["codominance_gap"] == CFG.frozen["analysis"]["dominant_mode_codominance_gap"] == 0.1


def test_inverse_mapping_has_one_entry_per_scene_predictor_condition_and_q():
    r = ea.inverse_mapping(synthetic_doc(), CFG)
    assert len(r["entries"]) == 2 * 2 * 4 * 3 and r["level"] == "engineering_only" and "not a universal" in r["not_a_requirement"]


def test_inverse_mapping_reads_the_stored_curve():
    """D = x for CV, so s* = q; for GRU D = x / 2, so s* = 2 q."""
    e = ea.inverse_mapping(synthetic_doc(), CFG)["entries"]
    assert e["SYN-1/CV/A/q0.1"]["s_star"] == pytest.approx(0.1, abs=1e-12)
    assert e["SYN-1/GRU/A/q0.2"]["s_star"] == pytest.approx(0.4, abs=1e-12)
    assert e["SYN-1/CV/A/q0.05"]["severity_measure"] == "rmse_pos_m"


def test_eth_and_hotel_inverse_mapping_stays_per_scene_and_descriptive():
    r = ea.inverse_mapping(ethhotel_doc(), CFG)
    assert r["level"] == "descriptive_local_only" and {k.split("/")[0] for k in r["entries"]} == {"ETH", "HOTEL"}


def test_crossings_cover_each_pair_and_the_across_scene_curve():
    """ADE: CV = 1 + x, GRU = 1.2 + x / 2, equal at x = 0.4."""
    r = ea.threshold_crossings(synthetic_doc(), CFG)
    assert r["across_synthetic_scenes"]["CV-GRU/A"]["s_crossing"] == pytest.approx(0.4, abs=1e-12)
    assert r["per_scene"]["SYN-2/CV-GRU/D"]["s_crossing"] == pytest.approx(0.4, abs=1e-12)
    assert "runtime selection" in r["not_implemented"] and r["level"] == "engineering_only"


def test_a_level_missing_a_scene_is_not_compared_across_scenes():
    doc = synthetic_doc()
    doc["across_scenes"]["per_history"]["8"]["CV"]["A"]["levels"][3]["ade_m"]["n_valid"] = 1
    assert ea.threshold_crossings(doc, CFG)["across_synthetic_scenes"]["CV-GRU/A"]["status"] == "crossed"
    doc["across_scenes"]["per_history"]["8"]["CV"]["A"]["levels"][1]["ade_m"]["n_valid"] = 1
    assert ea.threshold_crossings(doc, CFG)["across_synthetic_scenes"]["CV-GRU/A"]["status"] == "truncated"


def test_eth_and_hotel_have_no_across_scene_crossings():
    r = ea.threshold_crossings(ethhotel_doc(), CFG)
    assert "across_synthetic_scenes" not in r and r["level"] == "descriptive_local_only"


def test_e5_pairs_stay_scene_level_and_per_predictor():
    """MOTA rises with CV's ADE (rho = 1) and falls with GRU's (rho = -1); IDSW is the reverse."""
    r = ea.p13_engineering(e4_doc())
    cv, gru = r["by_predictor"]["CV"], r["by_predictor"]["GRU"]
    assert r["statistical_unit"] == "scene" and r["not_p13"] is True and r["h7_verdict"] == "not_evaluable"
    assert cv["correlations"]["MOTA"]["value"] == pytest.approx(1.0) and gru["correlations"]["MOTA"]["value"] == pytest.approx(-1.0)
    assert cv["correlations"]["IDSW"]["value"] == pytest.approx(-1.0) and cv["paired_observations"] == 3
    assert cv["pairs"]["S1"]["ade_m"] == 2.0 and cv["tracker_count"] == 1


def _write(tmp_path):
    paths = []
    for name, doc in (("syn", synthetic_doc()), ("eth", ethhotel_doc()), ("e4", e4_doc())):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        paths.append(path)
    return paths


def test_a_run_records_its_inputs_and_blocked_gates_and_leaves_inputs_alone(tmp_path):
    paths = _write(tmp_path)
    before = [p.read_bytes() for p in paths]
    doc = ea.run(tmp_path / "out", CFG, *paths)
    assert [p.read_bytes() for p in paths] == before
    assert doc["research_results"] is False and len(doc["inputs"]["synthetic_analysis_sha256"]) == 64
    assert set(doc["blocked"]) >= {"P-1", "P-9", "P-13/H7", "P-14/RQ8", "H9", "RQ3", "P-8", "P-10/H8", "P-11"}
    assert set(doc["deferred"]) >= {"E9/section 9.1", "E10/section 3.8", "section 9.2", "full section 9.3", "recovery"}
    assert "No research gate is passed" in doc["label"] and doc["ethhotel_family"].startswith("family AUDC is not computed")
    assert (tmp_path / "out" / "results.json").exists() and (tmp_path / "out" / "manifest.json").exists()
    assert doc["association_by_experiment"]["E5"]["association_mechanism"] is None
    assert doc["association_by_experiment"]["E6"]["association_mechanism"] == "synthetic_gt_lineage"


def test_blocked_items_have_reasons_and_no_placeholder_results(tmp_path):
    doc = ea.run(tmp_path / "out", CFG, *_write(tmp_path))
    assert all(isinstance(v, str) and v for v in {**doc["blocked"], **doc["deferred"]}.values())
    assert not {"recovery", "mitigation", "h7", "h9", "collision"} & set(doc)


def test_a_run_never_overwrites_and_checks_its_inputs(tmp_path):
    paths = _write(tmp_path)
    (tmp_path / "busy").mkdir()
    with pytest.raises(FileExistsError):
        ea.run(tmp_path / "busy", CFG, *paths)
    with pytest.raises(ValueError, match="in that order"):
        ea.run(tmp_path / "out", CFG, paths[1], paths[0], paths[2])


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "trackvision", "engineering", *map(str, args)], capture_output=True, text=True,
                          cwd=REPO_ROOT, check=False)


def test_cli_status_lists_blocked_and_deferred_work():
    doc = json.loads(_cli("status").stdout)
    assert "P-9" in doc["blocked"] and "section 9.2" in doc["deferred"]


@pytest.mark.parametrize("action, words", [("mot17", "--out"), ("synthetic", "--out"), ("analysis", "--out")])
def test_cli_missing_out_is_an_argument_error(action, words):
    proc = _cli(action)
    assert proc.returncode == 2 and words in proc.stderr and "Traceback" not in proc.stderr


def test_cli_names_missing_inputs(tmp_path):
    proc = _cli("synthetic", "--out", tmp_path / "x")
    assert proc.returncode == 2 and "--models" in proc.stderr and "--trackeval-python" in proc.stderr
