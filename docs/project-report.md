# TrackVision: Project Report

**Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments**
Governing specification v4.15 (frozen research core). Report date 2026-11-02.

> **Status in one line.** The engineering pipeline and the dashboard work. **No research gate has been passed**, so nothing here is a finished research result. Where this report says "measured", it means measured live on a named scene by the project's own code, as exploratory output.

---

## 1. Introduction

Pedestrian-aware systems (self-driving, robots, CCTV, crowd management) run a chain: **detect people → track them → convert to ground positions → predict where they go next**. Each stage makes mistakes, and the mistakes travel downstream. TrackVision studies that travel in a controlled way.

**Research question:** *How does tracking quality propagate into pedestrian trajectory prediction?*

The approach: take clean pedestrian trajectories, corrupt them with four precisely defined tracking-error types (one at a time, at graded severities), run trajectory predictors on the corrupted input, and measure how much the forecast error (ADE/FDE) grows. A Next.js dashboard over a FastAPI backend lets a user watch this happen on real and synthetic video.

## 2. Research gap

- Tracking papers report tracking scores (MOTA, IDF1, HOTA). Prediction papers assume clean input trajectories. **Almost nobody measures the link between them.**
- Known prior work on perception-to-prediction error propagation is mostly automotive. The project positions itself as a **controlled, multi-mode decomposition for pedestrians** (spec Part 15; [docs/literature-positioning.md](literature-positioning.md)).
- Real trackers mix all error types at once, so their effects cannot be separated. Controlled injection separates them.
- It is unclear whether a better tracking score means better prediction, which error matters most, and whether learned predictors keep their edge over constant velocity when input degrades.
- **Caveat:** a full literature survey has *not* been done. This is an open item for the owner and supervisor.

## 3. Objectives

From spec §2.1 (O1–O8). Status is as of today.

| ID | Objective | Status |
|---|---|---|
| O1 | Detection + tracking with YOLO and ByteTrack/BoT-SORT, benchmarked on MOT17 | **Partial.** YOLO11 runs (display only). The tracker is a project IoU tracker, *not* ByteTrack/BoT-SORT. Tracker set is still an unresolved choice (`trackers_evaluated_set: null`) |
| O2 | Stabilization: image → world via homography (static), plus ego-pose (moving) | Static done for ETH, HOTEL, synthetic. Moving blocked (no dataset chosen) |
| O3 | Metric trajectories, validated for plausibility and invariance | Done for ETH, HOTEL, synthetic |
| O4 | Predictors: CV, CA, LSTM, GRU | All four exist. LSTM/GRU are synthetic-engineering only |
| O5 | Five-mode error injection with measured severity | **A, B (B1/B2), C, D implemented. E (ego-motion) specified only** |
| O6 | Degradation curves, horizon profiles, inverse requirements | Engineering versions computed (E5–E8) |
| O7 | Error-aware mitigation | **Not started** (deferred, decision 0006) |
| O8 | Optional reliability-aware application | Replaced by the engineering-judgement Applications page |

**Research questions** RQ1–RQ9 (spec §2.2): effect of each error on ADE/FDE; which family matters most; real-tracker vs injected curves; additivity; static vs moving ranking; learned vs baseline; which tracking metric associates with prediction; mitigation recovery; required history length.

## 4. Methodology

**4.1 Pipeline.** Video → detections (boxes) → tracks (IDs) → foot point (bottom-centre of box) → ground position (homography) → trajectory sampled every **0.4 s** → windows of **8 observed + 12 predicted steps** → predictor → ADE/FDE.

**4.2 Controlled injection.** Errors are applied **downstream, to the trajectory**, not to detector output. They are *proxies*, not simulations of tracker internals (spec §3.5).

**4.3 Severity.** Every error has a fixed grid that includes zero. Curves are plotted against **achieved** severity, never the requested probability.

**4.4 Evaluation.**
- ADE = mean Euclidean error over the 12 predicted steps; FDE = error at the last step.
- Relative degradation = (ADE_corrupt − ADE_clean) / ADE_clean.
- 3 seeds (0, 1, 2), averaged within a scene.
- Leave-one-scene-out folds for learned models, with an inner validation rotation to prevent leakage.
- Tracking metrics come from the official **TrackEval** (never hand-rolled).

**4.5 Governance.** All protocol constants live only in [configs/base.yaml](../configs/base.yaml) (143 frozen leaves, each tested against the spec). Claims are limited to measurement and association levels (L1/L2, spec §1.2). Seventeen written decisions in [docs/decisions/](decisions/) record every scope choice.

## 5. Directory structure

```
Internship COE/
├── configs/        base.yaml (the ONLY place protocol constants live), experiments/smoke.yaml
├── src/trackvision/
│   ├── config.py, cli.py, repro.py   config validation + hashing, CLI, reproducibility
│   ├── io/           dataset loaders (EWAP/ETH-HOTEL, MOT17), PNG reader, SHA-256 manifest check, atomic writes
│   ├── geometry/     homography.py (static), moving.py (ego-pose composition)
│   ├── tracking/     tracker.py (IoU tracker), assignment.py (Hungarian), background_subtraction.py
│   ├── inject/       trajectory.py (A, B, C), identity_switch.py (D), severity.py, association.py
│   ├── predict/      baselines.py (CV, CA), learned.py (LSTM/GRU), training.py, windowing.py, folds.py, leakage.py
│   ├── metrics/      displacement.py (ADE/FDE/RMSE), detection.py, tracking.py (project CLEAR-MOT-style), error_profile.py
│   ├── stats/        AUDC, aggregation, inverse requirement, crossing, monotonicity, hypotheses
│   ├── synthetic/    camera, scene generator, renderer, video (known-answer test scenes)
│   ├── experiments/  one module per experiment (feasibility, sweep, TrackEval, analysis) + run record/store
│   ├── viz/          degradation-curve drawing
│   └── platform/     FastAPI app: app.py, routes.py, test_video_routes.py, schemas.py, services/
├── web/            Next.js dashboard (app/ pages, components/, lib/api/, config/, stores/, types/)
├── scripts/        dev.ps1, setup_yolo.ps1, yolo_detect_*.py, prepare_test_videos.py, trackeval_mot17.py, check_staged.py
├── tests/          unit/ (about 60 files), invariants/ (composition, leakage), fixtures/ (synthetic known-answer data)
├── data/           raw/ (datasets), manifests/ (SHA-256 lists), cache/ (video copies + YOLO boxes; Git-ignored)
├── videos/         11 downloaded CC-licensed street videos + SOURCES.md (Git-ignored)
├── runs/           stored experiment results, one folder per run ID (source of truth for the API; Git-ignored)
├── docs/           spec/, decisions/0001–0017, roadmap, literature note, phase-0b report
├── env/            pip lock + platform package list (Windows, Python 3.12)
└── .venv/          project Python environment
```

Outside the repo: `~/tv-envs/yolo-py312` (PyTorch + Ultralytics), `~/tv-envs/trackeval-py311` (TrackEval).

## 6. Datasets and how they differ

| Dataset | What it is | Camera | Ground truth | Units | Role here |
|---|---|---|---|---|---|
| **ETH / HOTEL** (EWAP) | Overhead pedestrian video, 2 scenes | Static, nearly straight down | Annotated world positions (`obsmat.txt`) + homography `H.txt` | **Metres** | Core static scenes; true metric trajectories |
| **UCY** (UNIV/students003, ZARA1, ZARA2) | Pedestrian videos with spline annotations | Static, elevated | Annotated image points (`.vsp`), no calibration | **Pixels** | Static scenes; no metres possible |
| **MOT17** | 7 training sequences (02, 04, 05, 09, 10, 11, 13), crowded street scenes | Static: 02, 04, 09. Moving: 05, 10, 11, 13 | Full boxes + IDs (`gt.txt`) | **Pixels** | Tracking benchmark. Trajectories are *never* used to train/evaluate predictors (spec §6.1.1) |
| **Synthetic** (SYN-501…510) | Generated waypoint walkers on a known pinhole camera | Static, 12 m high | Exact boxes, IDs and positions | Metres | Known-answer testing; learned models; 10 scenes |
| **Test Videos** | 5 of 11 downloaded CC-licensed street videos | Various | **None** | Pixels | Live demo only; no ADE/FDE possible |

Key differences:
- **Metres vs pixels.** Only ETH, HOTEL and synthetic have a homography, so only they give real distances. UCY and MOT17 work in pixels and cannot be compared with metres or between scenes.
- **Boxes.** ETH/HOTEL/UCY store *one point per person*. Boxes drawn there are display estimates. MOT17 and synthetic have real boxes.
- **Dashboard scene count: 52** = 5 ETH + 5 HOTEL + 15 UCY + 17 MOT17 + 10 synthetic. Long recordings are also offered as numbered *parts* (4 for ETH/HOTEL/UCY, 2 for MOT17). **A part is a stretch of one recording, never pooled with others** (decision 0012).
- **Moving camera.** The moving-camera research dataset (JRDB / KITTI / nuScenes) is *not yet selected*. MOT17-05/10/11/13 are moving, so their paths include camera motion.
- **Licences.** MOT17 is CC BY-NC-SA; raw data is Git-ignored and never redistributed.

## 7. Object detection

**What:** YOLO11 (`yolo11m`, image size 1280, confidence ≥ 0.25, COCO class 0 = person) finds people in video frames. It is **for display only**: it never enters ADE, FDE or any stored result, and it is not experiment P-1 or P-9 (decisions 0009, 0015).

| Step | File | Code |
|---|---|---|
| Install YOLO11 + PyTorch (own env) | [scripts/setup_yolo.ps1](../scripts/setup_yolo.ps1) | lines 27–37 |
| Load model, run `model.predict(... classes=[0])` | [scripts/yolo_detect_mot17.py](../scripts/yolo_detect_mot17.py) | `yolo_detector()` 63–87, predict at 79 |
| Per-sequence loop; boxes stored as fractions of image size | same | `detect_sequence()` 53–60, `normalise()` 39–43 |
| Run over MOT17 zip (7 sequences) | same | `main()` 90+ |
| Run over browser copies of ETH/HOTEL/UCY | [scripts/yolo_detect_videos.py](../scripts/yolo_detect_videos.py) | `detect_scene()` 50–57 |
| Run over Test Videos | [scripts/yolo_detect_test_videos.py](../scripts/yolo_detect_test_videos.py) | |
| Outputs | `data/cache/detections/`, `data/cache/test_video_detections/` | |
| Boxes → pixels for tracker | [platform/services/video_tracks.py](../src/trackvision/platform/services/video_tracks.py) | `yolo_boxes()` 172–176 |
| MOT17 public (FRCNN) detections, score ≥ 0.5 | same | `public_detections()` 163–169 |
| Synthetic scenes: background-subtraction blob detector | [tracking/background_subtraction.py](../src/trackvision/tracking/background_subtraction.py) | |
| Detection metrics (IoU, matching) | [metrics/detection.py](../src/trackvision/metrics/detection.py) | |

**Where YOLO is actually used** (decision 0015): MOT17 only. On ETH/HOTEL (overhead) it found 1–3 of ~6 people, and on UCY far fewer than the annotations, so annotated boxes are used there instead. UCY's YOLO files sit unused in `data/cache/detections_unused/`.

## 8. Object tracking

**What:** an online **IoU tracker with constant-velocity box prediction**. It is *not* ByteTrack or BoT-SORT (those are not implemented here; ByteTrack/BoT-SORT were tried informally on Test Videos and were no better, decision 0017).

Per frame ([tracking/tracker.py](../src/trackvision/tracking/tracker.py)):
1. **Predict** each track's box: last centre moved at constant velocity (`_Track.predict`, 78–81; velocity updated in `observe`, 83–91).
2. **Associate** predictions with detections by Hungarian assignment on gated IoU (`update`, 103–113 → `gated_iou_assignment`).
3. **Update:** matched track gains a hit; confirmed after `min_hits` = 3 (114–118).
4. **Unmatched track:** tentative tracks are deleted; confirmed tracks coast, then end after `max_age` = 25 missed frames (119–132).
5. **Unmatched detection** starts a new track with a never-reused ID (133–140).

Defaults: `match_iou` 0.3, `min_hits` 3, `max_age` 25 (`TrackerConfig`, 29–41). Entry point `run_tracker()` at 150–155.

| Part | File | Lines |
|---|---|---|
| Tracker | [tracking/tracker.py](../src/trackvision/tracking/tracker.py) | 94–147 |
| Hungarian algorithm (numpy only) | [tracking/assignment.py](../src/trackvision/tracking/assignment.py) | `linear_assignment` 20–66 |
| Gated IoU assignment | same | `gated_assignment` 69–75, `gated_iou_assignment` 78–79 |
| Apply tracker to a video's boxes, foot points as paths | [video_tracks.py](../src/trackvision/platform/services/video_tracks.py) | `_tracker_rows` 179–184, `mot17_tracks` 187–192 |
| Tracker on synthetic scenes | [experiments/tracking_feasibility.py](../src/trackvision/experiments/tracking_feasibility.py) | |
| Project tracking metrics (CLEAR-MOT-style, simplified IDF1) | [metrics/tracking.py](../src/trackvision/metrics/tracking.py) | `evaluate_tracking` 44 |
| **Official** HOTA / CLEAR / Identity via TrackEval | [experiments/mot17_trackeval.py](../src/trackvision/experiments/mot17_trackeval.py), [scripts/trackeval_mot17.py](../scripts/trackeval_mot17.py) | runs as a separate process |

**ID sources by dataset** (decision 0010): MOT17 = this IoU tracker on YOLO11 boxes; ETH/HOTEL/UCY = human annotations; synthetic = the scene generator. The tracker breaks IDs often (missed detections, overlapping people), and the dashboard says so.

## 9. Pixels → real distance (homography)

A **homography** is a 3×3 matrix H that maps image pixels to ground-plane coordinates in metres, assuming people stand on a flat ground plane.

**Step 1: choose the point.** The foot point is the **bottom-centre of the box**: ((x1+x2)/2, y2) (config [base.yaml:59-61](../configs/base.yaml#L59-L61); `_feet()` in [video_tracks.py:200-201](../src/trackvision/platform/services/video_tracks.py#L200-L201); synthetic: `bottom_centre_world()` in [tracking_feasibility.py:52-55](../src/trackvision/experiments/tracking_feasibility.py#L52-L55)).

**Step 2: apply H.** Homogeneous multiply, then divide by the third component. Rejects points at infinity.

| Part | File | Lines |
|---|---|---|
| Safety check (finite, 3×3, not singular via condition number) | [geometry/homography.py](../src/trackvision/geometry/homography.py) | `check_homography` 26–38 |
| Apply H, divide by w | same | `apply_homography` 41–54 |
| `StaticTransform`: image→world, world→image, round-trip test, orientation sign | same | 57–95 |
| Load ETH/HOTEL `H.txt` and convert to the project convention | [io/ewap.py](../src/trackvision/io/ewap.py) | 191–192: `StaticTransform(h_native @ SWAP_UV)` |
| Why the swap is needed (dataset's H takes (row, column)) | same | docstring 1–20; `SWAP_UV` in homography.py:13 |
| Evidence the convention is right (whole-pixel inverse, obstacle map) | [experiments/geometry_validation.py](../src/trackvision/experiments/geometry_validation.py) | |
| Synthetic camera ground homography | [synthetic/camera.py](../src/trackvision/synthetic/camera.py) | `ground_transform` 109–111 |
| Moving camera: `T_P→W(t) ∘ H_img→P` (ego-pose after homography) | [geometry/moving.py](../src/trackvision/geometry/moving.py) | `platform_to_world` 32–38, `MovingTransform` 50–64 |
| Round-trip tolerances | [base.yaml:62-65](../configs/base.yaml#L62-L65) | 1e-6 px synthetic, 1e-3 px real |

**Where it applies:** ETH, HOTEL, synthetic (metres). **UCY and MOT17 have no calibration**, so nothing is converted: errors, ADE and FDE are in **video pixels** (decision 0011). Moving-camera stabilization exists as code but is unexercised on real data.

**Stated unknowns** (`ASSUMPTIONS` in ewap.py:43–72): whether ETH/HOTEL annotated points are true ground-contact points, and whether frame/pixel indices are 0- or 1-based.

## 10. Error injection

Four error types are implemented, plus one specified. Each uses its own random stream per (seed, family, pedestrian), so results are reproducible and independent of other pedestrians.

| Code | Error | Mechanism | Levels (base.yaml:70-76) | Achieved severity |
|---|---|---|---|---|
| **A** | Localization noise | Add N(0, σ²) to x and y of each point | σ = 0, 0.05, 0.10, 0.20, 0.35, 0.50 m | RMSE of displacement (m) |
| **B** | Observation dropout | Remove each point with probability p. **B2** = no repair; **B1** = linear interpolation of interior gaps inside the history only | p = 0 … 0.40 | removed / eligible |
| **C** | Fragmentation | At a sampled step, end the track and start a new, unused ID. Positions unchanged | p = 0 … 0.30 | splits / eligible track frames |
| **D** | Identity switch | At a sampled step, swap IDs with a neighbour within **2.0 m** from then on. Both tracks locked afterwards (no cascades) | p = 0 … 0.20 | switch events per 100 eligible frames |
| **E** | Ego-motion error (moving only) | Perturb the ego-pose before transform composition | — | **Specified in spec §3.5 and base.yaml; not implemented** |

| Part | File | Lines |
|---|---|---|
| A | [inject/trajectory.py](../src/trackvision/inject/trajectory.py) | `localization_noise` 75–86 |
| B (masks) | same | `dropout_masks` 95–101 |
| B1 reconstruction | same | `reconstruct_history` 111–126 |
| C | same | `fragment` 136–159 |
| D: pick partners, lock, count | [inject/identity_switch.py](../src/trackvision/inject/identity_switch.py) | `sample_switches` 97–132 |
| D: swap labels from the event onward | same | `apply_switches` 141–169; `identity_switch` 180–183 |
| Severity normalization across modes | [inject/severity.py](../src/trackvision/inject/severity.py) | `normalize_severity` 110 |
| Combine several errors on one window (D first, then A/B/C) | [platform/services/exploratory.py](../src/trackvision/platform/services/exploratory.py) | `combine` 83–150 |
| Run every error at every level on a scene | [platform/services/scene_sweep.py](../src/trackvision/platform/services/scene_sweep.py) | `measure_level` 79–99 |
| Stored sweep | [experiments/sweep.py](../src/trackvision/experiments/sweep.py), [evaluation.py](../src/trackvision/experiments/evaluation.py) | `corrupt` 83 |
| Pixel scenes | decision 0011 | noise 0–60 px; ID-switch reach = one median person height |

**What the dashboard sliders do:** A, B, C, D sliders call `/inject`, which runs these same frozen injectors **live**; results are labelled exploratory and never saved. A window's forecast is lost if its 8-point history contains a dropped point (B2) or a split (C).

## 11. Prediction

**Window:** 8 observed positions (3.2 s) → 12 future positions (4.8 s), 0.4 s steps ([windowing.py](../src/trackvision/predict/windowing.py) `make_windows` 45).

| Predictor | Idea | File / lines |
|---|---|---|
| **CV** | Last-step velocity extrapolated: p̂(H+k) = p(H) + k·(p(H) − p(H−1)). Uses 2 points | [predict/baselines.py](../src/trackvision/predict/baselines.py) 22–41 |
| **CA** | Adds acceleration from 3 points: + ½k²·(p(H) − 2p(H−1) + p(H−2)). Magnifies noise with k² | same, 44–64 |
| **LSTM / GRU** | Encoder over 7 frame-to-frame displacements, 1 layer, 128 hidden units, linear head to 12×2 future displacements; no attention, no feedback | [predict/learned.py](../src/trackvision/predict/learned.py) 30–54 |
| Training (Adam, lr 1e-3, early stopping on validation ADE) | | [predict/training.py](../src/trackvision/predict/training.py) `train_predictor` 192 |
| Leave-one-scene-out folds + inner validation | | [folds.py](../src/trackvision/predict/folds.py), [leakage.py](../src/trackvision/predict/leakage.py) |
| Metrics | ADE = mean step error, FDE = last step | [metrics/displacement.py](../src/trackvision/metrics/displacement.py) 14–32 |
| Scoring | | [experiments/evaluation.py](../src/trackvision/experiments/evaluation.py) `score` 136 |

Learned models run only in stored synthetic-engineering experiments (decision 0001); the dashboard uses **CV and CA only**. Ground truth for the target is always the *clean* future, never the corrupted one.

**Measured so far** (decision 0016, live on all 52 scene choices, exploratory):
- Position noise (A) raised ADE and FDE the most on every scene.
- CV beat CA on clean tracks on every scene.
- B and C do not change forecasts that survive. They *remove* forecasts, so their damage shows as lost coverage. D raises error, much less than A.
- MOT17 pixel ADE (CV): 24–108 px on static sequences vs 113–190 px on moving ones, because paths include camera motion.

## 12. Applications: how errors are prioritized

**Where:** [web/config/applications.ts](../web/config/applications.ts) (data), [web/components/applications.tsx](../web/components/applications.tsx) (UI), page `/applications`.

**Ten applications:** self-driving car, driver-assistance, delivery robot, assistive navigation, drone monitoring, CCTV, crowd management, smart junction, retail analytics, workplace safety.

**Each application defines:** camera type, look-ahead time, what it needs, the metric that matters (ADE, FDE or neither), the best predictor, 2–3 representative scenes, and a **ranked list of the four errors (A, B, C, D), each with a reason and a real-world consequence.**

**Basis for the ranking: two layers.**
1. **Measured facts** from the live sweep (the "benchmark"): A caused the largest ADE/FDE rise; CV beat CA; B and C cost forecasts, not accuracy; D raises error modestly. Numbers come from the named scene via `scene_sweep.py` and are never pooled across scenes.
2. **Engineering judgement** (hand-written): what the application needs decides the order.
   - *Accuracy-driven* (self-driving, junction → FDE; driver-assistance, robot, drone, workplace → ADE): **A** ranks high, because it hurts accuracy most.
   - *Coverage-driven* (assistive navigation, drone, crowd, workplace): **B** ranks first, because a person with no forecast is unprotected.
   - *Identity-driven* (CCTV, retail): **D** or **C** rank first, because following the right person matters more than ADE. For these the page says ADE is the wrong guide.

**Benchmark used:** the frozen protocol itself (8 observed / 12 predicted steps, frozen severity grids, 3 seeds, CV vs CA on ADE/FDE) run on the application's named scenes. **No collision or application performance was measured.** The page states on every view that priorities are engineering judgement and that no research gate is passed. This is a deliberate, recorded exception to the spec's "no recommending language" rule (decision 0016).

## 13. Tech stack

| Layer | Technology |
|---|---|
| Language / runtime | Python 3.12 (python.org CPython), Windows |
| Core numerics | NumPy ≥2, PyYAML |
| Deep learning | PyTorch (LSTM/GRU, in `.venv`); Ultralytics **YOLO11** (own env, `yolo11m`) |
| Tracking evaluation | TrackEval (pinned commit, Python 3.11 env) |
| Video | OpenCV (detection scripts), **ffmpeg/ffprobe** (browser-copy generation) |
| Backend | **FastAPI**, Uvicorn, httpx; GZip + CORS middleware; all routes GET |
| Frontend | **Next.js 16.3.3**, React 19, TypeScript 5.7, Tailwind CSS 4, Recharts 3, shadcn / Base UI, lucide-react |
| Package managers | pip (lock file in `env/`), pnpm |
| Testing | pytest (about 60 unit files + invariant tests), Vitest + Testing Library |
| Reproducibility | Config hash → experiment ID, SHA-256 manifests, atomic writes, pre-commit staged-file check |
| Storage | Files only (`runs/`, `data/`); **no database** |

## 14. How everything connects (pipeline)

```
RAW DATA                    OFFLINE PREPARATION                         STORED / CACHED
ETH, HOTEL  ──load_scene──► tracks in metres + H ───────────────────────┐
UCY .vsp    ──ucy_tracks──► tracks in pixels                            │
MOT17 zip   ──YOLO11──► boxes ──IoU tracker──► foot points (px)         │  data/cache/
Test videos ──YOLO11──► boxes ──IoU tracker──► foot points (px)         │  (video, boxes)
Synthetic   ──generator──► exact tracks (m)                             │
                                                                         ▼
                    GRID TRACKS: positions every 0.4 s ──► windows (8 observed → 12 future)
                                                                         │
        ┌────────────────── LIVE (per request) ─────────────────────────┤
        │  ErrorRequest(A,B,C,D) ► exploratory.combine ► corrupted inputs│
        │  CV / CA ► forecast ► evaluation.score ► ADE, FDE, coverage    │
        │  scene_sweep: every error × every level × 3 seeds ► curves     │
        └────────────────────────────────────────────────────────────────┘
                                   │ JSON over HTTP (GET only)
                                   ▼
FastAPI  (platform/app.py + routes.py)   ◄── reads runs/ (stored results, hash-checked)
                                   │
                                   ▼
Next.js  Workspace · Analysis · Applications · Test Videos · Experiments · Settings
```

Rules that hold the pieces together:
- `runs/` is the single source of truth; the API has **no write route and no database**. Slider results are exploratory and never saved.
- Frozen constants come only from `base.yaml`; code never hard-codes 8/12.
- Detection files are display-only; a test shows ADE/FDE are identical with and without them.
- Scenes and parts are never pooled; Test Videos are isolated from every dataset.

## 15. APIs

Backend: FastAPI at `http://127.0.0.1:8000` (dev script can use 8010). Prefix `/api`. Interactive docs at `/docs`. Frontend base URL: `NEXT_PUBLIC_API_BASE_URL` ([web/lib/api/services.ts](../web/lib/api/services.ts)). **All routes are GET.** Errors return JSON (400 bad ID, 404 unavailable, 409 unusable run or scene, 422 bad parameter, 500 generic). Absolute file paths are scrubbed from every response ([app.py](../src/trackvision/platform/app.py) `withhold_paths` 39–47).

| Route | Purpose | Source |
|---|---|---|
| `/health` | Status, record count | [routes.py:39](../src/trackvision/platform/routes.py#L39) |
| `/datasets` | Dataset catalogue | :50 |
| `/experiments` (filters: kind, status, dataset) | Stored run list | :55 |
| `/experiments/{id}` · `/cells` · `/artifacts/{name}` | One run, its result cells, its figures | :67, :75, :86 |
| `/workspace/scenes` | The 52-scene catalogue | :92 |
| `/workspace/scenes/{scene}/video/status` · `/video` | Video build state; the MP4 | :97, :102 |
| `/workspace/scenes/{scene}/tracks?source=default\|truth` | Boxes, IDs, trails | :107 |
| `/workspace/scenes/{scene}/inject?a&b&c&d&reconstruct` | **Live** injection + forecasts + ADE/FDE | :112 |
| `/workspace/scenes/{scene}/noise-example` | One history at three noise levels | :125 |
| `/workspace/scenes/{scene}/sweep` | Background sweep progress/results (Analysis, Applications) | :131 |
| `/workspace/scenes/{scene}/tracker` | Tracker details (kept, unused) | :137 |
| `/tracking`, `/error-injection`, `/prediction`, `/analysis` (+ `/{experiment_id}`) | Views over stored runs | :148–189 |
| `/test-videos` | Catalogue of the 5 videos | [test_video_routes.py:16](../src/trackvision/platform/test_video_routes.py#L16) |
| `/test-videos/{video}/video` · `/tracks` | MP4; boxes, IDs, trails | :21, :26 |
| `/test-videos/{video}/forecast?a&b&c&d&reconstruct` | Live forecast with errors, no ADE/FDE | :31 |

Parameter limits: `a` (noise) 0 to `MAX_SIGMA_PX` (60 px), `b/c/d` 0 to 1; metre scenes enforce their own noise limit (422 if exceeded). CLI: `python -m trackvision config validate | run --config ...` (exit codes 0–5, [cli.py](../src/trackvision/cli.py)).

## 16. Project work (chronology)

| Phase | Work | State |
|---|---|---|
| 0A | Scaffold, config system (143 frozen leaves each tested), test harness, smoke run | Done, committed |
| 0B | Feasibility gates (G1–G8) | G2, G6, G8 passed; G1, G3, G4, G5 blocked on user actions |
| 1–3 | Data loaders, geometry validation, CV/CA/LSTM/GRU, leakage checks, proposed association mechanism (decisions 0001–0003) | Engineering scope |
| 4 | Injection, severity normalization, ETH/HOTEL sweeps, P-15 history corrections (0004–0005) | Engineering scope |
| 5–6 | MOT17 feasibility + TrackEval, synthetic tracker, E4–E8 analyses (0006) | Engineering scope, awaiting review |
| 7 | Platform: Workspace (0008), YOLO11 display (0009, 0015), overlays (0010), pixel scenes (0011), dropdowns/parts (0012), Analysis rebuild (0013, 0016), Applications (0016), Test Videos (0017) | Working; uncommitted |

Workspace, Analysis and Applications were **frozen on 2026-11-02**; new work goes in its own section.

## 17. Implementation (how a run works end to end)

1. **Setup:** `py -3.12 -m venv .venv`; install `env/requirements-lock-win_amd64-py312.txt` and `requirements-platform-…txt`; `pnpm install` in `web/`. Optional: `scripts/setup_yolo.ps1`.
2. **Data:** place datasets in `data/raw/`; loaders verify every file against its SHA-256 manifest.
3. **Detect (optional):** run the `yolo_detect_*` scripts in the YOLO env to fill `data/cache/`.
4. **Start:** `.\scripts\dev.ps1` starts the backend and the dashboard (it stops and names the process if a port is taken).
5. **Use:** pick a scene → the first play builds a browser copy with ffmpeg (seconds to minutes, once) → the tracker overlay appears → move sliders → `/inject` returns forecasts and ADE/FDE → Analysis fills in curves from the background sweep → Applications ranks errors for a chosen use.
6. **Verify:** the header shows `Backend connected · N records`, and N must match `/api/health`. Compare any value with its run's files under `runs/`.

## 18. Limitations (must be stated with any result)

- **No research gate is passed.** Phases 1–6 are engineering scope. Nothing is research-grade five-scene evidence.
- Injection is a **proxy** applied to trajectories, not to detector output.
- The tracker is a simple IoU tracker with frequent ID breaks; ByteTrack/BoT-SORT are not implemented.
- **Mode E and all mitigation (O7) are not implemented.** The moving-camera dataset is unselected.
- UCY annotations are interpolated splines; standing people lower clean ADE. MOT17 forecasts are viewing-only and an approved exception to spec §6.1.1.
- Pixel values are not comparable with metres or across scenes.
- Learned models are not in the dashboard and have only synthetic-engineering results.
- Applications priorities are engineering judgement, not measurements.
- Literature survey not done. Five research choices remain null (`moving_dataset`, `trackers_evaluated_set`, `tracker_params`, `h8_generation_attempts`, `discontinuity_rejection_v_max_mps`).
- Test Videos have no ground truth, so they show no ADE or FDE.
