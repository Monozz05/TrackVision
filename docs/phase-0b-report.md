# Phase 0B report: GPU and dataset feasibility

**Date:** 2026-10-29. **Commit at start:** `a7fa27f` (Phase 0A).
**Governing sources:** spec v4.15 §14.1 P0, §6.1.1, §6.4, §5.3, §5.5; Implementation Plan, Phase 0 gate and failure table.
**Scope kept:** no experiments, no change to frozen methodology, `configs/base.yaml`, `src/`, `tests/` or
dependencies; the seven research choices stay unresolved (including `moving_dataset`). Temporary environments
and measurement scripts were kept outside the repository; raw data is in the Git-ignored `data/raw/`.

## Summary

| Gate | Criterion | Status |
|---|---|---|
| G1 | CUDA names the RTX 3060 | **BLOCKED**: PyTorch download timed out twice; CUDA untested |
| G2 | Datasets present and checksummed | **PASS** for the five static scenes and MOT17 (moving-camera data: see G4) |
| G3 | Static-arm raw-video gate (§6.1.1) | **BLOCKED**: licence terms unconfirmed for all 5 scenes; no homography for the 3 UCY scenes |
| G4 | Moving-arm feasibility gate (§6.4) | **BLOCKED / unresolved**: all three candidates require an account |
| G5 | AGPL / Ultralytics decision | **BLOCKED**: needs the supervisor's decision |
| G6 | Literature note with verified CV-baseline citation | **PASS** |
| G7 | Synthetic harness | **DEFERRED** to the start of Phase 1 |
| G8 | Test suite green on the empty implementation | **PASS** |

**Phase 0B is not closed.** Its gates can't conclude until the user actions listed at the end are done.
No arm of the study is confirmed or descoped yet, and no RQ3 consequence is recorded yet, because the
evidence those decisions require is missing. Nothing has been inferred.

## G1: CUDA names the RTX 3060 (BLOCKED)

**Evidence gathered**
- `nvidia-smi`: NVIDIA GeForce RTX 3060 Laptop GPU, 6,144 MiB, compute capability 8.6, driver 555.99,
  driver-supported CUDA 12.5. The machine also has an AMD Radeon integrated GPU (hybrid graphics); CUDA
  enumerates NVIDIA devices only.
- A temporary venv (python.org CPython 3.12.10) was created in the session scratchpad, outside the repository.
  Installing `torch==2.6.0+cu124` (the newest build on the cu124 index; the driver supports CUDA up to 12.5)
  failed twice with the same error while downloading the wheel:
  `ReadTimeoutError: HTTPSConnectionPool(host='download-r2.pytorch.org', port=443): Read timed out.`
  (second attempt used pip `--timeout 180 --retries 10`; exit code 2). No further retry, as instructed.
- **Not a Smart App Control block.** Smart App Control is still enforcing (state 1). Of 936 Code Integrity
  events logged since the G1 start (10:32:30), none mention the temporary venv or torch; torch was never
  downloaded, so nothing could be blocked. The project `.venv` was not touched.

**Untested:** `torch.cuda.is_available()`, `get_device_name()`, and kernel execution on the GPU. The probe
(`g1_probe.py`, a 4096×4096 matrix product checked against the CPU) is ready but could not run.

**Classification:** a setup problem, not a finding (Implementation Plan failure table). No research
consequence follows.

**Next action (user):** provide a stable connection and allow one install of `torch==2.6.0+cu124` from
`https://download.pytorch.org/whl/cu124` into a temporary venv, or download the wheel yourself and give its
path. Then run the probe and recheck the Code Integrity log; if Smart App Control then blocks torch's DLLs,
that is reported, not worked around.

## G2: Datasets present and checksummed

Manifests are in `data/manifests/` (one SHA-256 line per file; the manifest hash is the SHA-256 of the
manifest file itself). Raw data is in `data/raw/`, which Git ignores.

| Dataset | Source | Access and licence | Files checked | Archive SHA-256 | Manifest hash |
|---|---|---|---|---|---|
| ETH and HOTEL (EWAP) | `https://data.vision.ee.ethz.ch/cvl/aem/ewap_dataset_full.tgz` (79,753,734 bytes) | Public download, no login. **No licence stated** in the package's README (contact address only). | 16 (archive plus 15 extracted files) | `403f0afc64f2635a28dd7960ede97fde947c38335f3c0fb131a260b4d2ffd203` | `ac53d498e127cd6c821553264974f48f9f014fd4d204308acea8870618034dfd` |
| UNIV (students003), ZARA1, ZARA2 (UCY) | `https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data` (redirects to a zip, 158,883,825 bytes, CRC check OK) | Public download, no login. **No licence stated** in the package. | 21 (archive, extracted files and the 6 extracted annotation files) | listed in `ucy.sha256` | `4b2561b93a596605628e421c27ff5a5ac77276314446b6f2ad2286237d59d1f4` |
| MOT17 | `https://motchallenge.net/data/MOT17.zip` (5,860,214,001 bytes) | Public download, no login. **CC BY-NC-SA 3.0** (MOTChallenge MOT17 page): non-commercial research use; only checksums are tracked here | 33,810 members (42 sequence folders, 33,705 images, 21 gt, 42 det, 42 seqinfo), every CRC-32 verified | `7b486bb2af5cfedbfabc020511ff933a3edbf12182ce4985eec1a210f20f9e06` | Full: `9f50b6e82e7c8ccc9c849bb64b345890dff1605c23979304f27309457bb14651`; tracked compact: `c3804c10d3c9149aa4618eb4eb63a81b8a5d4b18252baed3ca4d811f952747db` |

**MOT17 notes.** The download was interrupted once and resumed from byte 405,069,824 with an HTTP range
request (server answered `206 Partial Content`, `Last-Modified` 2020-12-18, so the file had not changed); the
final size equals the server's `Content-Length` exactly, and reading every member verified all CRC-32 values.
A per-file manifest of 33,811 entries is about 4 MB, over the repository's 1 MB file policy, so it stays
beside the data (`data/raw/mot17/MOT17.full.sha256`, Git-ignored). The tracked `data/manifests/mot17.sha256`
holds the archive hash, every non-image member, and the full manifest's own hash, which pins every image.

The UCY package also contains scenes the study does not use (`crowds_zara03`, `students001`,
`uni_examples`, `arxiepiskopi`). Spec §5.3.1 uses only the five named scenes.

## G3: Static-arm raw-video gate, per scene (BLOCKED)

The four §6.1.1 checks, from the downloaded files themselves:

| Scene | 1. Raw video obtainable and legally usable | 2. Frame-to-annotation mapping | 3. Pedestrian ground truth | 4. Image-to-world mapping |
|---|---|---|---|---|
| ETH | Video `seq_eth.avi` (640×480, 12,974 frames). **Licence not stated → unconfirmed** | Documented: `obsmat.txt` frame numbers index video frames; `info.txt` gives a 0.4 s annotation step. **Caveat:** the container declares 25 fps but annotations are 6 frames apart, which equals 0.4 s only at 15 fps. The Implementation Plan anticipates this; the timestep must come from the data in Phase 1 | `obsmat.txt`: 8,908 rows, 360 pedestrians, frames 780 to 12,381 (inside the video) | `H.txt` provided. **Available; not validated** until the Phase 1 composition tests |
| HOTEL | Video `seq_hotel.avi` (720×576, 19,350 frames). **Licence not stated → unconfirmed** | Documented and consistent: 10-frame steps at 25 fps = 0.4 s, as `info.txt` states | `obsmat.txt`: 6,544 rows, 390 pedestrians, frames 1 to 18,061 (inside the video) | `H.txt` provided. **Available; not validated** |
| UNIV (students003) | Video `students003.avi` (720×576, 5,405 frames). **Licence not stated → unconfirmed** | Documented: `crowd_file_format.txt` says each control point's frame number is the video frame. Annotation frames 0 to 5,404 fall inside the video | `students003.vsp`: 434 tracks as spline control points in **pixel** coordinates | **None shipped** (no homography or calibration file; the format file mentions neither). Not established |
| ZARA1 | Video `crowds_zara01.avi` (720×576, 9,031 frames). **Licence not stated → unconfirmed** | As UNIV; frames 0 to 9,014 inside the video | `crowds_zara01.vsp`: 148 tracks, pixel splines | **None shipped.** Not established |
| ZARA2 | Video `crowds_zara02.avi` (720×576, 10,519 frames). **Licence not stated → unconfirmed** | As UNIV; frames 6 to 10,518 inside the video | `crowds_zara02.vsp`: 204 tracks, pixel splines | **None shipped.** Not established |

**Outcome: no scene yet passes all four checks**, because check 1 (legal usability) is unconfirmed for all
five and check 4 is not established for the three UCY scenes. This is missing evidence, not a failed check,
so the RQ3 consequence is **not recorded yet**. What each resolution would mean under the predeclared table:

| If... | Scenes passing | Consequence for RQ3 (§6.1.1) |
|---|---|---|
| licences confirmed and UCY homographies obtained (and later validated) | 5 | RQ3 answered as specified |
| licences confirmed, no UCY homography | 2 (ETH, HOTEL) | RQ3 answered on the named subset ETH, HOTEL, with the reduced coverage stated |
| licences not confirmed | 0 | RQ3 cannot be answered within-domain; downgraded to cross-domain descriptive |

**Next actions (user):** (1) confirm with your supervisor, or the dataset authors (EWAP README contact
`stefpell@vision.ee.ethz.ch`; UCY Graphics Lab), that research use of these videos is permitted; (2) choose a
source for the UCY image-to-world homographies, or approve deriving them from calibration points in Phase 1,
where they must pass the §4.3 tests.

## G4: Moving-arm feasibility gate (BLOCKED / unresolved)

No candidate's data could be obtained, so none of the eight §6.4 checks was measured, and no window count
exists. **The moving arm is neither confirmed nor descoped**, and `moving_dataset` stays unresolved.

| Candidate | Evidence from the host | Status | Exact user action |
|---|---|---|---|
| KITTI Tracking | The tracking page's downloads (calibration 1 MB, GPS/IMU 8 MB, training labels 9 MB, images 15 GB) all link to `user_login.php` | BLOCKED (account) | Register and log in at `https://www.cvlibs.net/datasets/kitti/user_login.php`; download the tracking calibration, GPS/IMU and training-label archives (images are not needed for the gate); place them in `data/raw/kitti/` |
| JRDB | `https://jrdb.erc.monash.edu/` states: "you are required to log in before downloading JRDB" | BLOCKED (account) | Sign up and log in there; download the annotation, calibration and odometry files; place them in `data/raw/jrdb/` |
| nuScenes | `https://www.nuscenes.org/nuscenes` is a JavaScript application; its access and licence terms could not be read without a browser | BLOCKED (terms unverified) | Sign in at nuscenes.org, accept the terms, download the metadata archive (not the sensor blobs); place it in `data/raw/nuscenes/` |

Once data is supplied, each candidate is measured against all eight checks, with the window count taken after
resampling to the 0.4 s grid (nearest sample within 0.20 s, strictly increasing) by the staged rule
(at least 2,000 8/12 windows; otherwise 500 to 1,999 8/8 windows; otherwise unavailable). The results are then
presented for the user to choose the primary dataset; nothing is chosen automatically.

## G5: AGPL / Ultralytics (BLOCKED)

The Implementation Plan requires the AGPL question settled with the supervisor before Ultralytics enters the
repository. Nothing was installed. **Next action (user):** obtain and record the supervisor's decision.

## G6: Literature positioning note (PASS)

`docs/literature-positioning.md`. The Schöller et al. citation (IEEE RA-L 5(2):1696-1703, 2020,
DOI 10.1109/LRA.2020.2969925) was verified against Crossref's record and the arXiv v3 abstract and text: the
headline claim and the ETH/UCY evaluation (5 scenes, 8 observed / 12 predicted steps) are confirmed. The IEEE
version text was not read, and no survey of the wider prior work was done; both are stated in the note.

## G7: Synthetic harness (DEFERRED)

Phase 0A provides the known-answer trajectories and the passing harness. The synthetic camera, homography
and ego-pose parts require the geometry code this phase excludes; they are built at the start of Phase 1,
together with the six composition tests.

## G8: Test suite (PASS)

Run once after the feasibility work (2026-10-29 15:41): `pytest` **600 passed, 12 skipped** (the reserved
composition and leakage tests, which assert nothing until Phases 1 and 2); `pip check` reports no broken
requirements.

## Measurements and tools

- Measurement scripts (`make_manifest.py`, `mot17_manifest.py`, `g1_probe.py`) and the temporary venv lived in the session
  scratchpad, not the repository.
- Machine at the time: Ryzen 7 5800H, 13.9 GB RAM visible, Windows 11 power plan Balanced; free disk 176 to
  181 GB during the phase.

## User actions needed to close Phase 0B

1. **G1:** allow one PyTorch CUDA install on a stable connection (or supply the wheel file).
2. **G3:** confirm research-use permission for the ETH/UCY videos; choose a UCY homography source or approve
   deriving it in Phase 1.
3. **G4:** register for KITTI Tracking, JRDB and nuScenes, download the gate files listed above, and later
   choose the primary moving dataset from the measured results.
4. **G5:** record the supervisor's AGPL decision.
