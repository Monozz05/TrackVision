"""Section 3.7 association provenance of the Phase 5 and 6 records. Stored records stay byte-identical; a value is
written only where the record evidences it; E4 and E5 stay explicitly unresolved."""

import json
import subprocess
import sys

import pytest

from conftest import REPO_ROOT, SPEC_PATH
from trackvision import config
from trackvision.experiments import association_provenance as ap
from trackvision.experiments import injection_feasibility as inj
from trackvision.experiments import record as rec
from trackvision.experiments.store import RunStore
from trackvision.inject.association import ASSOCIATION_MECHANISMS, REAL_TRACKER_IOU, SYNTHETIC_GT_LINEAGE
from trackvision.synthetic.tracks import GridTrack

import numpy as np

CFG = config.validate().config
DECISION = REPO_ROOT / "docs" / "decisions" / "0006-phase-5-6-engineering-scope.md"
THRESHOLD = CFG.frozen["association"]["tracker_gt_assoc_iou_threshold"]


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def mot17_run(path, threshold=THRESHOLD, tracker="IouTracker (feasibility)"):
    _write(path / "results.json", {"not_p1": True, "cadence": "native", "by_detection_set": {"FRCNN": {}}})
    _write(path / "FRCNN" / "tracker" / "results.json", {"settings": {
        "evaluation_iou_threshold": threshold, "detection_variant": "FRCNN", "tracker": {"name": tracker}}})
    return path


def e4_run(path):
    RunStore.create(path, {"scope": "synthetic_tracker_engineering"})
    _write(path / "results.json", {"scope": "synthetic_tracker_engineering", "provenance": {
        "tracker": "existing IoU tracker; not ByteTrack or BoT-SORT", "detector": "synthetic blob detector; not YOLO"}})
    return path


def _scene(name):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Scene:
        name: str
        tracks: tuple
    frames = 10 * np.arange(5)
    return Scene(name, (GridTrack(1, frames, np.zeros((5, 2))),))


def base_run(path, scenes, **extra):
    RunStore.create(path, {"scope": "x", "scenes": {s.name: inj.tracks_digest(s.tracks) for s in scenes}, **extra})
    return path


# --- the frozen requirement ----------------------------------------------------------------------

def test_the_frozen_requirement_is_every_experiment_and_exactly_two_values():
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "Every experiment records which mechanism was used:" in text
    assert "association_mechanism = { real_tracker_iou | synthetic_gt_lineage }" in text
    assert set(ASSOCIATION_MECHANISMS) == {REAL_TRACKER_IOU, SYNTHETIC_GT_LINEAGE}


def test_the_section_11_3_schema_is_unchanged_so_proposal_0003_stays_unadopted():
    assert "association_mechanism" not in rec.TOP_LEVEL_FIELDS


# --- E1 and E3 -----------------------------------------------------------------------------------

def test_e1_value_rests_on_the_stored_settings(tmp_path):
    e = ap.mot17_entry(mot17_run(tmp_path / "r"), CFG)
    assert e["association_mechanism"] == REAL_TRACKER_IOU and e["status"] == ap.ESTABLISHED
    assert any("evaluation_iou_threshold" in x for x in e["evidence"]) and "gated_iou_assignment" in e["evidence"][0]


def test_e1_scope_is_stated_so_the_value_is_not_read_as_p1_or_p9(tmp_path):
    e = ap.mot17_entry(mot17_run(tmp_path / "r"), CFG)
    assert "no claim of P-1 or P-9" in e["note"] and "never used for prediction" in e["note"]
    assert set(e["does_not_apply_to"]) == {"E2", "official_trackeval"} and "no tracker" in e["does_not_apply_to"]["E2"]
    assert e["applies_to"] == ["E1 project tracking metrics", "E3 error profiles"]


@pytest.mark.parametrize("kwargs, word", [({"threshold": 0.3}, "frozen threshold"), ({"tracker": "ByteTrack"}, "IoU tracker")])
def test_no_value_is_written_if_the_stored_settings_do_not_show_it(tmp_path, kwargs, word):
    with pytest.raises(ValueError, match=word):
        ap.mot17_entry(mot17_run(tmp_path / "r", **kwargs), CFG)


def test_an_unrelated_record_is_refused(tmp_path):
    _write(tmp_path / "r" / "results.json", {})
    with pytest.raises(ValueError, match="not an E1"):
        ap.mot17_entry(tmp_path / "r", CFG)


# --- E4 and E5 -----------------------------------------------------------------------------------

def test_e4_is_recorded_as_null_with_its_blocker_and_is_not_omitted(tmp_path):
    e = ap.synthetic_tracker_entry(e4_run(tmp_path / "r"))
    assert "association_mechanism" in e and e["association_mechanism"] is None and e["status"] == ap.UNRESOLVED
    assert "decision 0006 item 8" in e["blocker"] and "synthetic_gt_lineage is false" in e["blocker"]
    assert "project-owner decision" in e["blocker"]


def test_e4_never_receives_the_forbidden_label_or_a_third_value(tmp_path):
    e = ap.synthetic_tracker_entry(e4_run(tmp_path / "r"))
    assert "real_tracker_iou" not in json.dumps(e) and "real_tracker_iou" not in json.dumps(ap.synthetic_tracker_field())
    assert ap.synthetic_tracker_field()["association_mechanism"] is None


def test_the_lineage_value_cannot_be_written_for_a_record_with_a_tracker():
    with pytest.raises(ValueError, match="category error"):
        ap.established("E4", SYNTHETIC_GT_LINEAGE, "existing IoU tracker", [], [])


def test_a_value_outside_the_two_frozen_values_cannot_be_written():
    with pytest.raises(ValueError, match="is not one of"):
        ap.established("E4", "synthetic_box_iou", "existing IoU tracker", [], [])


def test_a_new_e1_e4_and_analysis_record_carries_the_field():
    assert ap.mot17_field()["association_mechanism"] == REAL_TRACKER_IOU
    field = ap.analysis_field()
    assert field["E5"]["association_mechanism"] is None and field["E5"]["association_status"] == ap.UNRESOLVED
    assert {field[k]["association_mechanism"] for k in ("E6", "E7", "E8")} == {SYNTHETIC_GT_LINEAGE}


# --- E6 to E8 ------------------------------------------------------------------------------------

def test_a_phase_4_run_must_match_ground_truth_and_name_no_tracker(tmp_path):
    scenes = [_scene("SYN-1")]
    assert ap.ground_truth_base(base_run(tmp_path / "a", scenes), scenes)["scenes_matching_ground_truth"] == ["SYN-1"]
    with pytest.raises(ValueError, match="differ"):
        ap.ground_truth_base(base_run(tmp_path / "b", [_scene("SYN-2")]), scenes)
    with pytest.raises(ValueError, match="tracker or detector"):
        ap.ground_truth_base(base_run(tmp_path / "c", scenes, tracker_params={}), scenes)


def _analysis(tmp_path, e4_dir, wrong=False):
    syn, eth = tmp_path / "syn.json", tmp_path / "eth.json"
    _write(syn, {"a": 1})
    _write(eth, {"b": 2})
    sha = ap._sha
    run = tmp_path / "analysis"
    _write(run / "results.json", {"inputs": {"synthetic_analysis_sha256": "0" * 64 if wrong else sha(syn),
                                             "ethhotel_analysis_sha256": sha(eth),
                                             "e4_results_sha256": sha(e4_dir / "results.json")}})
    return run, syn, eth


def test_analysis_entries_are_per_experiment_and_tied_to_input_hashes(tmp_path):
    e4_dir = e4_run(tmp_path / "e4")
    run, syn, eth = _analysis(tmp_path, e4_dir)
    bases = ({"scenes_matching_ground_truth": ["SYN-1"]}, {"scenes_matching_ground_truth": ["ETH", "HOTEL"]})
    entries = ap.analysis_entries(run, e4_dir, syn, eth, *bases)
    assert [e["experiment"] for e in entries] == ["E5", "E6", "E7 and E8, synthetic", "E7 and E8, ETH and HOTEL"]
    assert entries[0]["association_mechanism"] is None
    assert {e["association_mechanism"] for e in entries[1:]} == {SYNTHETIC_GT_LINEAGE}
    assert "ETH, HOTEL" in entries[3]["evidence"][0] and "Fragmentation.origin" in entries[1]["evidence"][0]


def test_analysis_entries_refuse_inputs_that_are_not_the_recorded_ones(tmp_path):
    e4_dir = e4_run(tmp_path / "e4")
    run, syn, eth = _analysis(tmp_path, e4_dir, wrong=True)
    with pytest.raises(ValueError, match="is not the input"):
        ap.analysis_entries(run, e4_dir, syn, eth, {}, {})


# --- stored records ------------------------------------------------------------------------------

def test_a_sidecar_binds_to_the_record_and_leaves_it_unchanged(tmp_path):
    run = mot17_run(tmp_path / "r")
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    path = ap.write_sidecar(run, [ap.mot17_entry(run, CFG)])
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["results_sha256"] == ap._sha(run / "results.json") and doc["proposal_0003"] == "unadopted"
    assert doc["section_11_3_schema_changed"] is False
    assert {p: p.read_bytes() for p in before} == before


def test_a_sidecar_is_never_overwritten(tmp_path):
    run = mot17_run(tmp_path / "r")
    ap.write_sidecar(run, [])
    with pytest.raises(FileExistsError):
        ap.write_sidecar(run, [])


# --- the decision record -------------------------------------------------------------------------

def test_decision_0006_no_longer_contradicts_section_3_7():
    text = " ".join(DECISION.read_text(encoding="utf-8").split())
    assert "is not added as a protocol or record field" not in text
    assert "Proposal 0003 stays unadopted" in text and "section 3.7 requirement stays in force" in text.replace("§", "section ")


def test_decision_0006_separates_identity_source_from_the_section_3_7_mechanism():
    assert "not §3.7's `synthetic_gt_lineage`" in " ".join(DECISION.read_text(encoding="utf-8").split())


def test_decision_0006_labels_the_e3_conversion_and_e6_tolerance():
    text = " ".join(DECISION.read_text(encoding="utf-8").split())
    assert "engineering diagnostic, not a §3.7 implementation" in text
    assert "project-level numerical convention" in text and "not part of §2.2.3" in text


def test_cli_requires_every_input():
    proc = subprocess.run([sys.executable, "-m", "trackvision", "engineering", "provenance", "--out", "x"], capture_output=True,
                          text=True, cwd=REPO_ROOT, check=False)
    assert proc.returncode == 2 and "--models" in proc.stderr and "--ethhotel-run" in proc.stderr
