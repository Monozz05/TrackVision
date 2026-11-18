# Decision 0009: YOLO11 detections for display on MOT17

- **Date:** 2026-11-01
- **Decided by:** the project owner, who asked for YOLO11 object identification and bounding boxes on the MOT17 videos.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **State:** the code is in place and tested. **YOLO11 has not been run yet**, because installing it needs the owner's
  approval. Until it runs, MOT17 uses the public detections (decision 0010).

## Approved scope

1. **Display only.** Its detections feed the tracker drawn on the MOT17 scenes (decision 0010). This is **not experiment
   P-1**: it is not a tracking evaluation, it adds no research evidence, and it writes nothing under `runs/`.
2. **Separate environment.** Ultralytics runs in its own environment outside the repository. The research `.venv`, its
   lock file and `pyproject.toml` do not gain it, and no module under `src/` imports it (a test enforces this).
3. **Stored, not live.** `scripts/yolo_detect_mot17.py` runs once per sequence and writes to the Git-ignored
   `data/cache/detections/`. The backend only reads those files.
4. **Source data untouched.** Images are read from `MOT17.zip` in memory, and the archive is never extracted or changed.
5. **MOT17 stays out of prediction.** Boxes do not change specification section 6.1.1: no trajectories, forecasts,
   errors or ADE are produced on MOT17.
6. **No invented boxes.** If a sequence has no detection file, the tracker runs on the public detections and the scene
   names that source. The stored tracking metrics on Analysis come from MOT17's public detections, and the screen says
   so.

## Implementation-level choices, not specification requirements

- **Model and settings.** `yolo11m` with the standard COCO weights, image size 1280 and minimum confidence 0.25, because
  people in MOT17 are often small. These are display settings, not the research detector.
- **Usage reporting.** Ultralytics' analytics are turned off, because some of this project's data is marked local only.

## Licence

Ultralytics and its YOLO11 weights are published under AGPL-3.0. That is workable for this local use. It would need
reviewing before any public release or use in a closed product.

## Not in this scope

Tracking IDs from YOLO, tuning or training, YOLO on other datasets, precision or recall against ground truth, and
choosing the detector for P-1. Specification section 6.2 still needs that choice before P-1 can run.

**Later.** [Decision 0015](0015-yolo11-on-every-real-video.md) extends these display-only boxes to ETH, HOTEL and UCY.
