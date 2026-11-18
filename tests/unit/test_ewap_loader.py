"""EWAP loader and dataset manifests, on small synthetic files written to a temporary directory.

The checks against the real local ETH and HOTEL files run only where those Git-ignored files
exist; they assert structure and conventions, not research results.
"""

import hashlib

import numpy as np
import pytest

from fixtures import synthetic_ewap as se
from fixtures import synthetic_geometry as sg
from trackvision import config
from trackvision.geometry.homography import SingularHomographyError
from trackvision.io import ewap, manifest

CFG = config.validate().config
DT = CFG.frozen["protocol"]["static"]["sampling_interval_s"]
TOL_PX = CFG.frozen["geometry"]["roundtrip_tolerance_px"]["synthetic_float64"]


def test_loads_tracks_in_world_metres(tmp_path):
    se.write_scene(tmp_path)
    scene = ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)
    assert [t.ped_id for t in scene.tracks] == [1, 2]
    assert scene.frame_step == se.FRAME_STEP
    assert scene.sampling_interval_s == DT
    np.testing.assert_allclose(scene.tracks[0].positions[:, 1], 1.0)
    assert scene.diagnostics["frame_grid_phases"] == {0: 1, 3: 1}
    assert scene.diagnostics["implied_video_fps"] == pytest.approx(se.FRAME_STEP / DT)
    assert scene.diagnostics["velocity_timestep"]["timestep_s"] == pytest.approx(DT, abs=1e-12)
    assert set(scene.assumptions) == set(ewap.ASSUMPTIONS)
    assert "UNVERIFIED" in scene.assumptions["trajectory_point"]
    assert scene.frame_size == sg.IMAGE_SIZE
    assert scene.obstacle_map.shape == (sg.IMAGE_SIZE[1], sg.IMAGE_SIZE[0])


def test_row_col_homography_is_converted_to_project_convention(tmp_path):
    camera = se.write_scene(tmp_path)
    scene = ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)
    world = sg.world_grid(camera)
    uv = sg.project(camera["h_ground_to_img"], world)
    tol_m = TOL_PX * sg.metres_per_pixel(camera["h_img_to_world"], uv)
    assert np.max(np.linalg.norm(scene.transform.image_to_world(uv) - world, axis=1)) <= tol_m
    assert scene.transform_config["native_input_order"] == "row_col"
    assert scene.transform_config["project_input_order"] == "u_v"


def test_manifest_is_verified_and_hash_recorded(tmp_path):
    raw = tmp_path / "raw"
    se.write_scene(raw)
    manifest_path = tmp_path / "ewap.sha256"
    se.write_manifest(raw, manifest_path)
    scene = ewap.load_scene("ETH", CFG, raw, manifest_path)
    assert scene.provenance["dataset_manifest_hash"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert scene.transform_config["source_sha256"] == scene.provenance["source_sha256"]["ewap_dataset/seq_eth/H.txt"]


def test_modified_file_fails_manifest_check(tmp_path):
    raw = tmp_path / "raw"
    se.write_scene(raw)
    manifest_path = tmp_path / "ewap.sha256"
    se.write_manifest(raw, manifest_path)
    with open(raw / "ewap_dataset" / "seq_eth" / "obsmat.txt", "a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(manifest.ManifestError):
        ewap.load_scene("ETH", CFG, raw, manifest_path)


def test_file_missing_from_manifest_fails(tmp_path):
    raw = tmp_path / "raw"
    se.write_scene(raw)
    manifest_path = tmp_path / "ewap.sha256"
    se.write_manifest(raw, manifest_path)
    kept = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if "H.txt" not in line]
    manifest_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="not listed"):
        ewap.load_scene("ETH", CFG, raw, manifest_path)


def test_documented_timestep_must_match_frozen_interval(tmp_path):
    se.write_scene(tmp_path, info="The annotation was done with a timestep of 0.04 seconds.")
    with pytest.raises(ewap.EwapFormatError, match="frozen sampling interval"):
        ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)


def test_missing_timestep_statement_raises(tmp_path):
    se.write_scene(tmp_path, info="no timing information")
    with pytest.raises(ewap.EwapFormatError):
        ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)


@pytest.mark.parametrize("bad", ["duplicate_frame", "nonzero_z", "irregular_step"])
def test_malformed_obsmat_raises(tmp_path, bad):
    rows = se.track_rows(1, 780, np.column_stack([np.arange(6.0), np.zeros(6)]))
    if bad == "duplicate_frame":
        rows.append(list(rows[-1]))
    elif bad == "nonzero_z":
        rows[0][3] = 0.5
    else:
        rows[-1][0] += 4                      # 10 frames after its predecessor: not a multiple of 6
    se.write_scene(tmp_path, rows=rows)
    with pytest.raises(ewap.EwapFormatError):
        ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)


def test_singular_scene_homography_raises(tmp_path):
    se.write_scene(tmp_path, h_native=np.zeros((3, 3)))
    with pytest.raises(SingularHomographyError):
        ewap.load_scene("ETH", CFG, tmp_path, manifest_path=None)


def test_unknown_scene_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        ewap.load_scene("ZARA1", CFG, tmp_path, manifest_path=None)


REAL_DATA = (ewap.DEFAULT_RAW_ROOT / ewap.DATASET_SUBDIR).is_dir() and ewap.DEFAULT_MANIFEST.is_file()


@pytest.mark.skipif(not REAL_DATA, reason="local EWAP data (Git-ignored) is not present")
@pytest.mark.parametrize("name, frame_step", [("ETH", 6), ("HOTEL", 10)])
def test_real_scene_loads_with_verified_manifest(name, frame_step):
    scene = ewap.load_scene(name, CFG)
    assert scene.provenance["manifest_verified"]
    assert f"ewap_dataset/{ewap.SCENES[name]}/map.png" in scene.provenance["source_sha256"]
    assert scene.frame_step == frame_step
    assert scene.diagnostics["velocity_timestep"]["timestep_s"] == pytest.approx(DT, abs=1e-6)
    tol = CFG.frozen["geometry"]["roundtrip_tolerance_px"]["real_float64"]
    world = np.concatenate([t.positions for t in scene.tracks])
    assert scene.transform.roundtrip_errors(world)["px"] <= tol
