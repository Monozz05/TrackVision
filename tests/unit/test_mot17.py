"""M1: MOT17 reading, filtering, half split, IoU feasibility tracker and MOT-format output, on small
synthetic MOT17-format ZIP archives written to a temporary directory. No test reads the real archive.

Walker scene (fixtures/synthetic_mot17.py), 10 frames, second half = frames 6..10, detections exactly on
all four objects. Expected by hand on the second half: 2 pedestrians x 5 frames = 10 evaluable rows, all
TP; the static-person detection is ignored (5); the car detection is FP (5); MOTA = 1 - 5/10 = 0.5;
IDTP = 10, IDFP = (20 - 5) - 10 = 5, IDFN = 0, IDF1 = 20 / 25 = 0.8.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fixtures import synthetic_mot17 as sm
from trackvision import config
from trackvision.experiments import mot17_feasibility as mf
from trackvision.io import mot17
from trackvision.predict.windowing import make_windows

CFG = config.validate().config
SEQ = "MOT17-02"
GT_MEMBER = "MOT17/train/MOT17-02-FRCNN/gt/gt.txt"


def archive(tmp_path, sequences=None, name="MOT17.zip", **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    zip_path = sm.write_zip(tmp_path / name, sequences or {f"{SEQ}-FRCNN": sm.walker_sequence()}, **kwargs)
    return zip_path, sm.write_manifest(zip_path, tmp_path / "mot17.sha256")


def run(tmp_path, sequences=None, **kwargs):
    zip_path, man = archive(tmp_path, sequences)
    return mf.run_feasibility(CFG, zip_path, man, sequences=kwargs.pop("seqs", [SEQ]), **kwargs)


def info(length=10, name=f"{SEQ}-FRCNN"):
    return mot17.parse_seqinfo(sm.seqinfo(name, length).encode(), name)


# --- archive discovery and metadata ---------------------------------------------------------------

def test_zip_member_discovery_and_seqinfo(tmp_path):
    seqs = {f"MOT17-02-{v}": sm.walker_sequence(f"MOT17-02-{v}") for v in mot17.VARIANTS}
    seqs["MOT17-05-FRCNN"] = sm.walker_sequence("MOT17-05-FRCNN")
    zip_path = sm.write_zip(tmp_path / "MOT17.zip", seqs,
                            extra={"MOT17/test/MOT17-01-FRCNN/seqinfo.ini": sm.seqinfo("MOT17-01-FRCNN", 5)})
    with zipfile.ZipFile(zip_path) as zf:
        assert mot17.member_prefix(zf) == "MOT17/"
        assert mot17.list_sequences(zf, "train") == {"MOT17-02": ("DPM", "FRCNN", "SDP"), "MOT17-05": ("FRCNN",)}
        assert mot17.list_sequences(zf, "test") == {"MOT17-01": ("FRCNN",)}
        i = mot17.read_seqinfo(zf, "MOT17-02", "FRCNN")
    assert (i.name, i.frame_rate, i.seq_length, i.im_width, i.im_height, i.im_ext) == \
           ("MOT17-02-FRCNN", 30, 10, 1920, 1080, ".jpg")

    flat = sm.write_zip(tmp_path / "flat.zip", {"MOT17-09-FRCNN": sm.walker_sequence("MOT17-09-FRCNN")}, prefix="")
    with zipfile.ZipFile(flat) as zf:
        assert mot17.member_prefix(zf) == "" and mot17.list_sequences(zf) == {"MOT17-09": ("FRCNN",)}

    with pytest.raises(mot17.Mot17FormatError, match="does not match folder"):
        mot17.parse_seqinfo(sm.seqinfo("MOT17-04-FRCNN", 10).encode(), "MOT17-02-FRCNN")
    with pytest.raises(mot17.Mot17FormatError, match="needs"):
        mot17.parse_seqinfo(b"[Sequence]\nname=MOT17-02-FRCNN\n", "MOT17-02-FRCNN")


# --- columns, frames and coordinates -------------------------------------------------------------

def test_gt_columns_and_box_convention():
    gt = mot17.parse_gt((sm.gt_line(2, 7, 101, 51, 20, 40, consider=1, cls=1, vis=0.8)
                         + sm.gt_line(1, 7, 1, 1, 10, 10, consider=0, cls=7, vis=0.0)).encode(), info())
    assert gt.frame.tolist() == [1, 2] and gt.ped_id.tolist() == [7, 7]            # sorted by frame
    np.testing.assert_array_equal(gt.boxes[1], [100.0, 50.0, 120.0, 90.0])       # 1-based -> 0-based xyxy
    np.testing.assert_array_equal(gt.boxes[0], [0.0, 0.0, 10.0, 10.0])           # pixel (1, 1) -> origin
    assert gt.consider.tolist() == [0, 1] and gt.cls.tolist() == [7, 1]
    np.testing.assert_array_equal(gt.visibility, [0.0, 0.8])
    np.testing.assert_array_equal(mot17.to_mot_box(gt.boxes), [[1, 1, 10, 10], [101, 51, 20, 40]])

    bad = {
        "expected 9 columns": "1,1,1,1,10,10,1,1\n",
        "consider flag": sm.gt_line(1, 1, 1, 1, 10, 10, consider=2),
        "visibility": sm.gt_line(1, 1, 1, 1, 10, 10, vis=1.5),
        "frames must lie": sm.gt_line(0, 1, 1, 1, 10, 10),
        "frames must lie in 1..10": sm.gt_line(11, 1, 1, 1, 10, 10),
        "duplicate": sm.gt_line(1, 1, 1, 1, 10, 10) * 2,
        "frame numbers must be integers": "1.5,1,1,1,10,10,1,1,1\n",
        "non-negative": sm.gt_line(1, 1, 1, 1, -3, 10),
    }
    for message, text in bad.items():
        with pytest.raises(mot17.Mot17FormatError, match=message):
            mot17.parse_gt(text.encode(), info())


def test_detection_columns():
    det = mot17.parse_detections((sm.det_line(3, 913.8, 483.9, 97.6, 112.6, 0.99)
                                  + sm.det_line(1, 11, 21, 30, 60, -0.5)).encode(), info())
    assert det.frame.tolist() == [1, 3]                                          # sorted by frame
    np.testing.assert_allclose(det.boxes, [[10, 20, 40, 80], [912.8, 482.9, 1010.4, 595.5]])
    np.testing.assert_array_equal(det.score, [-0.5, 0.99])
    np.testing.assert_allclose(mot17.to_mot_box(det.boxes)[1], [913.8, 483.9, 97.6, 112.6])
    ten = mot17.parse_detections(sm.det_line(2, 11, 21, 30, 60, 0.5, ten_columns=True).encode(), info())
    np.testing.assert_array_equal(ten.boxes, [[10, 20, 40, 80]])
    assert ten.frame.tolist() == [2] and ten.score.tolist() == [0.5]
    with pytest.raises(mot17.Mot17FormatError, match="id column must be -1"):
        mot17.parse_detections(b"1,4,1,1,10,10,0.9\n", info())
    with pytest.raises(mot17.Mot17FormatError, match="7 or 10 columns"):
        mot17.parse_detections(b"1,-1,1,1,10,10,0.9,-1\n", info())
    empty = mot17.parse_detections(b"\n", info())
    assert empty.boxes.shape == (0, 4) and len(empty.frame) == 0


# --- category and visibility filtering -------------------------------------------------------------

def test_category_rules():
    rows = [(1, 1, 0.0), (1, 0, 1.0), (2, 0, 1.0), (7, 0, 1.0), (8, 0, 1.0), (12, 0, 1.0), (3, 0, 1.0),
            (4, 0, 1.0), (5, 0, 1.0), (6, 0, 1.0), (9, 0, 1.0), (10, 0, 1.0), (11, 0, 1.0), (13, 1, 1.0),
            (1, 1, 0.1), (1, 1, 0.25)]
    text = "".join(sm.gt_line(1, k + 1, 1, 1, 10, 10, consider=c, cls=cls, vis=v) for k, (cls, c, v) in enumerate(rows))
    gt = mot17.parse_gt(text.encode(), info())
    assert mot17.categorise(gt).tolist() == (
        ["evaluable", "ignore"] + ["ignore"] * 4 + ["excluded"] * 8 + ["evaluable", "evaluable"])
    filtered = mot17.categorise(gt, min_visibility=0.25).tolist()
    assert filtered[0] == filtered[14] == "invisible" and filtered[15] == "evaluable"
    with pytest.raises(ValueError):
        mot17.categorise(gt, min_visibility=1.5)


def test_walker_metrics_ignore_and_false_positive(tmp_path):
    r = run(tmp_path)
    s = r.sequences[0]
    c = s.counts["second"]
    assert c["frame_range"] == [6, 10] and c["frames"] == 5
    assert c["gt_rows_by_category"] == {"evaluable": 10, "invisible": 0, "ignore": 5, "excluded": 5}
    assert c["gt_rows_by_class"] == {"pedestrian": 10, "static_person": 5, "car": 5}
    assert c["evaluable_identities"] == 2 and c["detections"] == 20
    assert c["evaluable_rows_by_visibility"] == {"[0.0,0.25)": 0, "[0.25,0.5)": 0, "[0.5,0.75)": 5, "[0.75,1]": 5}
    m = s.metrics
    assert {k: m[k] for k in ("evaluable_observations", "tracker_observations", "tp", "fp", "fn", "ignored",
                              "idsw", "frag", "idtp", "idfp", "idfn")} == \
           {"evaluable_observations": 10, "tracker_observations": 20, "tp": 10, "fp": 5, "fn": 0, "ignored": 5,
            "idsw": 0, "frag": 0, "idtp": 10, "idfp": 5, "idfn": 0}
    assert m["mota"] == pytest.approx(0.5) and m["idf1"] == pytest.approx(0.8)
    assert not set(mf.POSITION_FIELDS) & set(m)
    assert (s.tracker.n_created, s.tracker.n_confirmed, s.tracker.n_deleted_tentative) == (4, 4, 0)


# --- half split ------------------------------------------------------------------------------------

def test_half_split_rule():
    assert mot17.half_split(600) == {"first": (1, 300), "second": (301, 600)}
    assert mot17.half_split(525) == {"first": (1, 262), "second": (263, 525)}
    assert mot17.half_split(1) == {"first": (1, 0), "second": (1, 1)}
    for n in (1, 2, 7, 600, 1050):
        a, b = mot17.half_split(n)["first"], mot17.half_split(n)["second"]
        assert a[0] == 1 and b[0] == a[1] + 1 and b[1] == n                    # disjoint, complete
    with pytest.raises(ValueError):
        mot17.half_split(0)


def test_first_half_never_reaches_tracker_or_metrics(tmp_path):
    base = sm.walker_sequence()
    gt_lines = base["gt"].splitlines(keepends=True)
    changed_gt = [re.sub(r"^(\d+),1,", r"\1,99,", ln) if int(ln.split(",")[0]) <= 5 else ln for ln in gt_lines]
    junk = "".join(sm.det_line(f, 150 + f, 220, 20, 40) for f in range(1, 6))
    bridge = sm.det_line(5, 126, 201, 20, 40)                                    # where ped 1 is at frame 6
    variant = {"seqinfo": base["seqinfo"], "gt": "".join(changed_gt), "det": base["det"] + junk + bridge}
    a = run(tmp_path / "a")
    b = run(tmp_path / "b", {f"{SEQ}-FRCNN": variant})
    sa, sb = a.sequences[0], b.sequences[0]
    assert sa.counts["first"] != sb.counts["first"]                            # the first halves differ
    assert sa.mot_text == sb.mot_text and sa.metrics == sb.metrics and sa.counts["second"] == sb.counts["second"]
    frames = [int(line.split(",")[0]) for line in sa.mot_text.splitlines()]
    assert min(frames) == 6 and max(frames) == 10


# --- MOT-format output -------------------------------------------------------------------------------

def test_mot_format_output(tmp_path):
    s = run(tmp_path).sequences[0]
    lines = s.mot_text.splitlines()
    assert len(lines) == 20 and s.mot_text.endswith("\n")
    assert lines[:4] == ["6,1,126.000,201.000,20.000,40.000,1,-1,-1,-1",
                         "6,2,526.000,201.000,20.000,40.000,1,-1,-1,-1",
                         "6,3,301.000,101.000,20.000,40.000,1,-1,-1,-1",
                         "6,4,701.000,301.000,60.000,30.000,1,-1,-1,-1"]
    table = np.loadtxt(lines, delimiter=",")
    assert table.shape == (20, 10)
    order = np.lexsort((table[:, 1], table[:, 0]))
    np.testing.assert_array_equal(order, np.arange(20))                          # sorted by frame, then id
    np.testing.assert_array_equal(table[:, 6:], np.tile([1, -1, -1, -1], (20, 1)))
    for f, tid, left, top, w, h in table[:, :6]:                                 # every box is an input detection
        assert tuple(np.round([left, top, w, h], 3)) in {tuple(map(float, b)) for b in sm.walker_boxes(int(f))}


def test_frames_without_detections(tmp_path):
    seqs = {f"{SEQ}-FRCNN": sm.walker_sequence(length=20, det_frames=set(range(1, 21)) - {15})}
    s = run(tmp_path, seqs).sequences[0]
    assert s.counts["second"]["frames_without_detections"] == 1
    assert 15 not in {int(line.split(",")[0]) for line in s.mot_text.splitlines()}
    assert len(s.mot_text.splitlines()) == 4 * 9
    m = s.metrics
    assert (m["evaluable_observations"], m["tp"], m["fn"], m["fp"], m["ignored"], m["idsw"], m["frag"]) == \
           (20, 18, 2, 9, 9, 0, 2)
    assert m["mota"] == pytest.approx(1 - 11 / 20) and m["idf1"] == pytest.approx(36 / 47)


def test_sequence_without_detections_or_ground_truth(tmp_path):
    no_det = sm.walker_sequence()
    no_det["det"] = ""
    s = run(tmp_path / "nodet", {f"{SEQ}-FRCNN": no_det}).sequences[0]
    assert s.mot_text == "" and s.tracker.n_created == 0
    assert s.counts["second"]["detections"] == 0 and s.counts["second"]["detection_score"] is None
    assert (s.metrics["tp"], s.metrics["fn"], s.metrics["mota"], s.metrics["idf1"]) == (0, 10, 0.0, 0.0)

    no_gt = sm.walker_sequence()
    no_gt["gt"] = ""
    s = run(tmp_path / "nogt", {f"{SEQ}-FRCNN": no_gt}).sequences[0]
    assert s.metrics["evaluable_observations"] == 0 and s.metrics["mota"] is None
    assert s.metrics["fp"] == 20 and len(s.mot_text.splitlines()) == 20


# --- provenance, reproducibility, saving ----------------------------------------------------------

def test_sequences_come_from_config_and_are_not_pooled(tmp_path):
    names = mf.static_sequences(CFG)
    assert names == list(CFG.frozen["datasets"]["mot17"]["static_sequences"])
    zip_path, man = archive(tmp_path, {f"{n}-FRCNN": sm.walker_sequence(f"{n}-FRCNN") for n in names})
    r = mf.run_feasibility(CFG, zip_path, man)
    doc = mf.results_document(r)
    assert [s.sequence for s in r.sequences] == names and list(doc["sequences"]) == names
    assert doc["pooled_result"] is None
    assert doc["settings"]["detection_variant"] == "FRCNN" and "not produced by a TrackVision detector" in \
        doc["settings"]["detections"]
    assert "not TrackEval" in doc["metrics_note"] and "not MOTChallenge-compliant" in doc["label"]


def test_manifest_verification(tmp_path):
    zip_path, man = archive(tmp_path / "ok")
    r = mf.run_feasibility(CFG, zip_path, man, sequences=[SEQ], hash_archive=True)
    assert r.settings["archive"]["sha256"] == r.settings["archive"]["sha256_in_manifest"] == \
           hashlib.sha256(zip_path.read_bytes()).hexdigest()
    with zipfile.ZipFile(zip_path) as zf:
        assert r.settings["member_sha256"][GT_MEMBER] == hashlib.sha256(zf.read(GT_MEMBER)).hexdigest()
    assert r.settings["dataset_manifest_hash"] == hashlib.sha256(man.read_bytes()).hexdigest()

    other = sm.walker_sequence()
    other["gt"] = other["gt"].replace(",201,", ",202,", 1)
    (tmp_path / "bad").mkdir()
    tampered = sm.write_zip(tmp_path / "bad" / "MOT17.zip", {f"{SEQ}-FRCNN": other})
    with pytest.raises(mot17.Mot17ManifestError, match="does not match the manifest"):
        mf.run_feasibility(CFG, tampered, man, sequences=[SEQ])
    with pytest.raises(mot17.Mot17ManifestError, match="does not match the manifest"):
        mf.run_feasibility(CFG, tampered, man, sequences=[SEQ], hash_archive=True)
    partial = sm.write_manifest(zip_path, tmp_path / "partial.sha256", drop={GT_MEMBER})
    with pytest.raises(mot17.Mot17ManifestError, match="not listed"):
        mf.run_feasibility(CFG, zip_path, partial, sequences=[SEQ])


def test_reproducible_saved_output(tmp_path):
    zip_path, man = archive(tmp_path / "data")
    before = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    r1 = mf.run_feasibility(CFG, zip_path, man, sequences=[SEQ])
    r2 = mf.run_feasibility(CFG, zip_path, man, sequences=[SEQ])
    assert r1.experiment_id == r2.experiment_id
    out1, out2 = mf.save_run(tmp_path / "o1", r1, CFG), mf.save_run(tmp_path / "o2", r2, CFG)
    names = ["results.json", "sequences.csv", "config.resolved.yaml", f"mot_tracker_output/{SEQ}-FRCNN.txt"]
    for name in names:
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    assert (out1 / f"mot_tracker_output/{SEQ}-FRCNN.txt").read_text(encoding="utf-8") == r1.sequences[0].mot_text
    m = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
    assert set(m["files"]) == set(names)
    assert not any("gt" in p.name for p in out1.rglob("*"))                      # no ground-truth file written
    with pytest.raises(FileExistsError):
        mf.save_run(out1, r1, CFG)
    inside = config.REPO_ROOT / "m1-output-must-not-be-created"
    with pytest.raises(ValueError, match="inside the repository"):
        mf.save_run(inside, r1, CFG)
    assert not inside.exists()
    assert hashlib.sha256(zip_path.read_bytes()).hexdigest() == before          # archive unchanged


# --- MOT17 never enters prediction -------------------------------------------------------------------

PREDICTION_MODULES = ("trackvision.predict", "trackvision.metrics.displacement", "trackvision.synthetic",
                      "trackvision.inject", "trackvision.experiments.synthetic_feasibility",
                      "trackvision.experiments.prediction_feasibility", "trackvision.experiments.injection_feasibility",
                      "trackvision.experiments.feasibility_run", "trackvision.experiments.ewap_baseline")


def test_mot17_code_never_loads_the_prediction_pipeline():
    for module in (mot17, mf):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
                   {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not {m for m in imported if m and m.startswith(PREDICTION_MODULES)}, module.__name__
    code = ("import sys; import trackvision.experiments.mot17_feasibility, trackvision.io.mot17; "
            f"bad = [m for m in sys.modules if m.startswith({PREDICTION_MODULES!r})]; print(bad); "
            "sys.exit(1 if bad else 0)")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_mot17_data_never_reaches_windows_cv_or_ade(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("MOT17 data reached the prediction pipeline")
    import trackvision.metrics.displacement as disp
    import trackvision.predict.baselines as base
    import trackvision.predict.windowing as win
    for module, name in ((win, "make_windows"), (win, "static_windows"), (base, "constant_velocity"),
                         (disp, "displacement_errors"), (disp, "ade_fde")):
        monkeypatch.setattr(module, name, forbidden)
    r = run(tmp_path, {f"{SEQ}-FRCNN": sm.walker_sequence(length=50)})
    doc = mf.results_document(r)

    def keys(d):
        if isinstance(d, dict):
            for k, v in d.items():
                yield k
                yield from keys(v)
        elif isinstance(d, list):
            for v in d:
                yield from keys(v)
    assert not [k for k in keys(doc) if re.search(r"(^|_)(ade|fde|window|windows)(_|$)", str(k))]
    assert "never used for trajectory prediction" in doc["settings"]["prediction"]
    # Structurally unusable as prediction input: tracker output has image boxes, no world positions.
    tracks = r.sequences[0].tracker.tracks
    assert max(len(t.frames) for t in tracks) >= 20 and not hasattr(tracks[0], "positions")
    with pytest.raises(AttributeError):
        make_windows("MOT17", tracks, 1, 8, 12, 1)
