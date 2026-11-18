"""The local Phase 1 geometry validation runner, on synthetic EWAP-format scenes."""

import json

import numpy as np

from fixtures import synthetic_ewap as se
from fixtures import synthetic_geometry as sg
from trackvision import config
from trackvision.experiments import geometry_validation as gv
from trackvision.io import ewap

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
STATIC = CFG.frozen["protocol"]["static"]
WINDOW = STATIC["history_len"] + STATIC["horizon_len"]
SPEED = 1.3
N = WINDOW + 5


def _write_both_scenes(root):
    walk = np.column_stack([3.0 + SPEED * DT * np.arange(N), np.full(N, 1.0)])
    for name in ewap.SCENES:
        se.write_scene(root, name, rows=se.track_rows(1, 780, walk))


def test_step_speeds_use_only_one_step_pairs():
    track = ewap.Track(1, np.array([0, 6, 12, 30]), np.array([[0.0, 0], [0.5, 0], [1.0, 0], [5.0, 0]]),
                       np.zeros((4, 2)))
    np.testing.assert_allclose(gv.step_speeds([track], 6, DT), [0.5 / DT, 0.5 / DT])


def test_run_reports_roundtrip_speed_windows_and_blocked_scenes(tmp_path):
    _write_both_scenes(tmp_path)
    report = gv.run(CFG, tmp_path, manifest_path=None)
    assert report["passed"]
    assert report["notice"].startswith("LOCAL ONLY")
    assert set(report["blocked_scenes"]) == {"UNIV", "ZARA1", "ZARA2"}
    for scene in report["scenes"].values():
        assert scene["roundtrip"]["passed"]
        assert abs(scene["speed_diagnostic"]["median_mps"] - SPEED) < 1e-9
        assert scene["speed_diagnostic"]["in_range"]
        assert scene["windows"]["count"] == (N - WINDOW) // STATIC["window_stride"] + 1
        assert scene["orientation"]["positive_jacobian_points"] == 0
        check = scene["convention_check"]
        assert check["project_reading_preferred"]
        assert check["readings"]["project_u_v"]["segments_touching_obstacles"] == 0
        assert check["plausible_bounds_points_outside_frame"] == 0
    json.dumps(report)                      # the report must be serialisable as written


def test_convention_check_rejects_a_scene_stored_in_the_wrong_order(tmp_path):
    """If H.txt actually took (column, row) input, loading it as (row, column) must be flagged."""
    walk = np.column_stack([3.0 + SPEED * DT * np.arange(N), np.full(N, 1.0)])
    rows = se.track_rows(1, 780, walk)
    camera = sg.static_camera()
    se.write_scene(tmp_path, "ETH", rows=rows, h_native=camera["h_img_to_world"])   # no swap
    scene = ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)
    check = gv.convention_check(scene, slack_px=1e-3)
    assert not check["project_reading_preferred"]
    assert check["readings"]["swapped"]["segments_touching_obstacles"] == 0


def test_points_outside_the_frame_are_counted(tmp_path):
    walk = np.column_stack([3.0 + SPEED * DT * np.arange(N), np.full(N, 1.0)])
    se.write_scene(tmp_path, "ETH", rows=se.track_rows(1, 780, walk))
    scene = ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)
    check = gv.convention_check(scene, slack_px=1e-3)
    narrow = ewap.Scene(**{**scene.__dict__, "frame_size": (10, 10)})
    assert gv.convention_check(narrow, slack_px=1e-3)["plausible_bounds_points_outside_frame"] == N
    assert check["plausible_bounds_points_outside_frame"] == 0


def test_main_writes_once_and_refuses_to_overwrite(tmp_path):
    raw, out, manifest_path = tmp_path / "raw", tmp_path / "out", tmp_path / "ewap.sha256"
    _write_both_scenes(raw)
    se.write_manifest(raw, manifest_path)
    args = ["--raw-root", str(raw), "--manifest", str(manifest_path), "--out", str(out)]
    assert gv.main(args) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert all(s["provenance"]["manifest_verified"] for s in report["scenes"].values())
    assert gv.main(args) == 4
    assert gv.main(args + ["--force"]) == 0
