"""M2: TrackEval input export (frame renumbering, GT and tracker preservation, sequence lengths) on small
synthetic MOT17-format archives and M1 runs written to a temporary directory. No test reads the real archive.

The known-answer TrackEval runs need the isolated TrackEval environment and are skipped unless
TRACKVISION_TRACKEVAL_PYTHON (its python.exe) and TRACKVISION_TRACKEVAL_SOURCE (the pinned source tree) are
set. Expected values are derived by hand from the walker scene (fixtures/synthetic_mot17.py): on the second
half (5 frames) the two pedestrians give 10 TP; the static-person box is removed as a distractor; the car
box is FP (5): MOTA = (10 - 5) / 10 = 0.5, IDF1 = 10 / (10 + 2.5) = 0.8, DetA = 10 / 15, AssA = 1.
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fixtures import synthetic_mot17 as sm
from trackvision import config
from trackvision.experiments import mot17_feasibility as mf
from trackvision.experiments import mot17_trackeval as mt
from trackvision.io import mot17

CFG = config.validate().config
TE_PYTHON = os.environ.get("TRACKVISION_TRACKEVAL_PYTHON")
TE_SOURCE = os.environ.get("TRACKVISION_TRACKEVAL_SOURCE")
needs_trackeval = pytest.mark.skipif(not (TE_PYTHON and TE_SOURCE and Path(TE_PYTHON).exists()),
                                     reason="isolated TrackEval environment not configured")


def m1_run(tmp_path, sequences):
    """A synthetic archive, its manifest and a saved M1 run over `sequences` ({folder: texts})."""
    zip_path = sm.write_zip(tmp_path / "MOT17.zip", sequences)
    manifest = sm.write_manifest(zip_path, tmp_path / "mot17.sha256")
    names = [f.rsplit("-", 1)[0] for f in sequences]
    run = mf.run_feasibility(CFG, zip_path, manifest, sequences=names)
    return zip_path, manifest, mf.save_run(tmp_path / "m1", run, CFG)


def export(tmp_path, sequences):
    zip_path, manifest, m1_dir = m1_run(tmp_path, sequences)
    out = tmp_path / "m2"
    exports = mt.export_inputs(out, m1_dir, zip_path, manifest)
    return out, m1_dir, zip_path, exports


def test_renumber_and_restore():
    text = "".join(f"{f},{f % 3 + 1},1.5,2,3,4,1,1,0.5\n" for f in range(1, 11)) + "\n"
    out = mt.renumber(text, (6, 10), keep_outside=True)
    lines = out.splitlines()
    assert [int(line.split(",")[0]) for line in lines] == [1, 2, 3, 4, 5]
    assert [line.split(",", 1)[1] for line in lines] == [f"{f % 3 + 1},1.5,2,3,4,1,1,0.5" for f in range(6, 11)]
    assert mt.restore(out, 5) == "".join(f"{f},{f % 3 + 1},1.5,2,3,4,1,1,0.5\n" for f in range(6, 11))
    with pytest.raises(mt.ExportError, match="outside 6..10"):
        mt.renumber(text, (6, 10), keep_outside=False)
    with pytest.raises(mt.ExportError, match="integer frame"):
        mt.renumber("1.5,1,2,3,4,5\n", (1, 2), keep_outside=True)


def test_export_layout_and_sequence_lengths(tmp_path):
    out, _, _, exports = export(tmp_path, {"MOT17-02-FRCNN": sm.walker_sequence("MOT17-02-FRCNN", length=10),
                                           "MOT17-09-FRCNN": sm.walker_sequence("MOT17-09-FRCNN", length=7)})
    by = {e.sequence: e for e in exports}
    assert (by["MOT17-02"].original_frames, by["MOT17-02"].frame_offset, by["MOT17-02"].seq_length) == ((6, 10), 5, 5)
    assert (by["MOT17-09"].original_frames, by["MOT17-09"].frame_offset, by["MOT17-09"].seq_length) == ((4, 7), 3, 4)
    root = out / "trackeval_input"
    seqmap = root / "gt" / "mot_challenge" / "seqmaps" / "MOT17-half.txt"
    assert seqmap.read_text(encoding="utf-8") == "name\nMOT17-02-FRCNN\nMOT17-09-FRCNN\n"
    ini = root / "gt" / "mot_challenge" / "MOT17-half" / "MOT17-02-FRCNN" / "seqinfo.ini"
    assert ini.read_text(encoding="utf-8") == ("[Sequence]\nname=MOT17-02-FRCNN\nimDir=img1\nframeRate=30\n"
                                               "seqLength=5\nimWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    assert (root / "gt" / "mot_challenge" / "MOT17-half" / "MOT17-09-FRCNN" / "gt" / "gt.txt").exists()
    assert (root / "trackers" / "mot_challenge" / "MOT17-half" / mt.TRACKER_NAME / "data" / "MOT17-09-FRCNN.txt").exists()
    checks = mt.validate_export(out, tmp_path / "m1", tmp_path / "MOT17.zip")
    assert all(c["seq_length_matches"] and c["frames_within_1_to_seqLength"] for c in checks)


def test_gt_rows_preserved(tmp_path):
    seq = sm.walker_sequence("MOT17-02-FRCNN", length=10)
    out, _, _, (e,) = export(tmp_path, {"MOT17-02-FRCNN": seq})
    gt = mt.layout(out / "trackeval_input", "MOT17-02-FRCNN")["gt"].read_text(encoding="utf-8")
    source_half = [line for line in seq["gt"].splitlines() if int(line.split(",")[0]) >= 6]
    assert len(gt.splitlines()) == len(source_half) == 4 * 5 == e.gt_rows
    for exported, original in zip(gt.splitlines(), source_half):            # only the frame field differs
        assert int(exported.split(",")[0]) == int(original.split(",")[0]) - 5
        assert exported.split(",", 1)[1] == original.split(",", 1)[1]
    assert e.gt_rows_by_class == {"1": 10, "3": 5, "7": 5}
    assert {int(line.split(",")[0]) for line in gt.splitlines()} == {1, 2, 3, 4, 5}


def test_tracker_rows_preserved(tmp_path):
    out, m1_dir, _, (e,) = export(tmp_path, {"MOT17-02-FRCNN": sm.walker_sequence("MOT17-02-FRCNN", length=10)})
    tracker = mt.layout(out / "trackeval_input", "MOT17-02-FRCNN")["tracker"].read_text(encoding="utf-8")
    m1_text = (m1_dir / "mot_tracker_output" / "MOT17-02-FRCNN.txt").read_text(encoding="utf-8")
    assert mt.restore(tracker, e.frame_offset) == m1_text                     # ids, boxes, confidence unchanged
    assert tracker.splitlines()[0] == "1,1,126.000,201.000,20.000,40.000,1,-1,-1,-1"
    assert e.tracker_rows == 20 and e.frames_without_tracker_rows == 0


def test_frames_without_tracker_rows(tmp_path):
    seq = sm.walker_sequence("MOT17-02-FRCNN", length=20, det_frames=set(range(1, 21)) - {15})
    out, m1_dir, zip_path, (e,) = export(tmp_path, {"MOT17-02-FRCNN": seq})
    tracker = mt.layout(out / "trackeval_input", "MOT17-02-FRCNN")["tracker"].read_text(encoding="utf-8")
    assert e.frames_without_tracker_rows == 1
    assert 15 - e.frame_offset not in {int(line.split(",")[0]) for line in tracker.splitlines()}
    (check,) = mt.validate_export(out, m1_dir, zip_path)
    assert check["frames_without_tracker_rows"] == 1 and check["tracker_rows_preserved"]


def test_validation_and_refusals(tmp_path):
    out, m1_dir, zip_path, _ = export(tmp_path, {"MOT17-02-FRCNN": sm.walker_sequence("MOT17-02-FRCNN")})
    with pytest.raises(FileExistsError):
        mt.export_inputs(out, m1_dir, zip_path, tmp_path / "mot17.sha256")
    with pytest.raises(ValueError, match="inside the repository"):
        mt.export_inputs(config.REPO_ROOT / "m2-must-not-be-created", m1_dir, zip_path, tmp_path / "mot17.sha256")
    assert not (config.REPO_ROOT / "m2-must-not-be-created").exists()
    gt = mt.layout(out / "trackeval_input", "MOT17-02-FRCNN")["gt"]
    gt.write_text(gt.read_text(encoding="utf-8").replace(",201,", ",202,", 1), encoding="utf-8", newline="\n")
    with pytest.raises(mt.ExportError, match="gt_rows_preserved"):
        mt.validate_export(out, m1_dir, zip_path)
    m1_file = m1_dir / "mot_tracker_output" / "MOT17-02-FRCNN.txt"
    m1_file.write_bytes(m1_file.read_bytes() + b"\n")                        # any change to an M1 file
    with pytest.raises(mt.ExportError, match="does not match its recorded hash"):
        mt.verified_m1(m1_dir)


def test_config_keys_exist_in_pinned_source(tmp_path):
    cfg = mt.trackeval_config(tmp_path, Path("trackeval-src"), CFG)
    assert cfg["dataset_config"]["BENCHMARK"] == "MOT17" and cfg["dataset_config"]["SPLIT_TO_EVAL"] == "half"
    assert cfg["dataset_config"]["CLASSES_TO_EVAL"] == ["pedestrian"] and cfg["dataset_config"]["DO_PREPROC"] is True
    e = cfg["eval_config"]
    assert (e["PLOT_CURVES"], e["OUTPUT_SUMMARY"], e["OUTPUT_DETAILED"]) == (False, False, True)
    assert Path(e["LOG_ON_ERROR"]).parent == tmp_path / "logs"
    assert [m for m, _ in cfg["metrics"]] == ["HOTA", "CLEAR", "Identity"]          # Count: added by TrackEval
    if not TE_SOURCE:
        pytest.skip("pinned TrackEval source not configured")
    src = Path(TE_SOURCE) / "trackeval"
    texts = {"dataset_config": (src / "datasets" / "mot_challenge_2d_box.py").read_text(encoding="utf-8"),
             "eval_config": (src / "eval.py").read_text(encoding="utf-8")}
    for section, text in texts.items():
        for key in cfg[section]:
            assert f"'{key}':" in text, (section, key)
    for name, options in cfg["metrics"]:
        for key in options or {}:
            assert f"'{key}':" in (src / "metrics" / f"{name.lower()}.py").read_text(encoding="utf-8"), (name, key)


def test_trackvision_never_imports_trackeval():
    tree = ast.parse(Path(mt.__file__).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
              {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in modules if m and m.startswith("trackeval")}
    script = ast.parse(mt.SCRIPT.read_text(encoding="utf-8"))
    script_modules = {a.name for n in ast.walk(script) if isinstance(n, ast.Import) for a in n.names} | \
                     {n.module for n in ast.walk(script) if isinstance(n, ast.ImportFrom)}
    assert not {m for m in script_modules if m and m.startswith("trackvision")}


# --- known-answer runs of the real, pinned TrackEval (isolated environment) ---------------------------------

def run_layout(tmp_path, gt_rows, tracker_rows, length=5):
    """A one-sequence layout written directly, run through TrackEval; returns its per-sequence results."""
    info = mot17.parse_seqinfo(sm.seqinfo("MOT17-02-FRCNN", length).encode(), "MOT17-02-FRCNN")
    root = tmp_path / "trackeval_input"
    mt.write_sequence(root, info, length, "".join(gt_rows), "".join(tracker_rows))
    mt.write_seqmap(root, ["MOT17-02-FRCNN"])
    assert mt.run_trackeval(tmp_path, Path(TE_PYTHON), Path(TE_SOURCE), CFG) == 0, \
        (tmp_path / "logs" / "stderr.txt").read_text(encoding="utf-8")
    return json.loads((tmp_path / "trackeval_results.json").read_text(encoding="utf-8"))


def walker_gt(length=5):
    seq = sm.walker_sequence(length=length)
    return [line + "\n" for line in seq["gt"].splitlines()]


@needs_trackeval
def test_trackeval_ground_truth_as_tracker_is_perfect(tmp_path):
    gt = walker_gt()
    tracker = [f"{r.split(',')[0]},{r.split(',')[1]},{','.join(r.split(',')[2:6])},1,-1,-1,-1\n"
               for r in gt if r.split(",")[7] == "1"]
    res = run_layout(tmp_path, gt, tracker)
    h = mt.headline(res["sequences"]["MOT17-02-FRCNN"])
    assert (h["MOTA"], h["IDF1"], h["HOTA"], h["CLR_FP"], h["CLR_FN"], h["IDSW"]) == (1.0, 1.0, 1.0, 0, 0, 0)
    assert len(res["hota_alphas"]) == 19 and np.allclose(res["hota_alphas"], mt.HOTA_ALPHAS)


@needs_trackeval
def test_trackeval_identity_swap(tmp_path):
    gt = walker_gt()
    tracker = []
    for r in gt:
        f, pid, *rest = r.split(",")
        if rest[5] == "1":                                           # class 1 only; swap ids from frame 3 on
            tid = int(pid) if int(f) < 3 else 3 - int(pid)
            tracker.append(f"{f},{tid},{','.join(rest[:4])},1,-1,-1,-1\n")
    h = mt.headline(run_layout(tmp_path, gt, tracker)["sequences"]["MOT17-02-FRCNN"])
    # each pedestrian switches once (frame 3); best identity matching keeps 3 of 5 frames per pedestrian
    assert (h["IDSW"], h["CLR_TP"], h["CLR_FP"], h["CLR_FN"]) == (2, 10, 0, 0)
    assert h["MOTA"] == pytest.approx(0.8) and h["IDF1"] == pytest.approx(0.6)


@needs_trackeval
def test_trackeval_on_exported_m1_run(tmp_path):
    out, m1_dir, zip_path, _ = export(tmp_path, {"MOT17-02-FRCNN": sm.walker_sequence("MOT17-02-FRCNN", length=10)})
    mt.validate_export(out, m1_dir, zip_path)
    assert mt.run_trackeval(out, Path(TE_PYTHON), Path(TE_SOURCE), CFG) == 0
    res = mt.validate_results(out)
    h = mt.headline(res["sequences"]["MOT17-02-FRCNN"])
    assert (h["CLR_TP"], h["CLR_FP"], h["CLR_FN"], h["IDSW"], h["GT_Dets"], h["Dets"]) == (10, 5, 0, 0, 10, 15)
    assert h["MOTA"] == pytest.approx(0.5) and h["IDF1"] == pytest.approx(0.8)
    assert h["DetA"] == pytest.approx(10 / 15) and h["AssA"] == pytest.approx(1.0)
    assert h["HOTA"] == pytest.approx(np.sqrt(10 / 15))
    m1 = json.loads((m1_dir / "results.json").read_text(encoding="utf-8"))
    cmp = mt.compare_with_m1(m1, res)["MOT17-02"]
    assert cmp["mota"]["m1"] == pytest.approx(cmp["mota"]["trackeval"]) == 0.5
    assert cmp["idf1"]["m1"] == pytest.approx(cmp["idf1"]["trackeval"]) == pytest.approx(0.8)
    detailed = (out / "trackeval_output" / mt.TRACKER_NAME / "pedestrian_detailed.csv").read_text(encoding="utf-8")
    assert [line.split(",")[0] for line in detailed.splitlines()[1:]] == ["MOT17-02-FRCNN", "COMBINED"]
