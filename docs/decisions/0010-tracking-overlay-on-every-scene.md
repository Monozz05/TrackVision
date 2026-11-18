# Decision 0010: Tracking overlay on every scene

- **Date:** 2026-11-01
- **Decided by:** the project owner, who asked for boxes with separate IDs on every video, because the project studies
  identity switching, and for clearer trajectory lines.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to decision 0008.** This narrowly changes item 6 of decision 0008: UCY and MOT17 now show tracks as well as
  video. They still offer no error controls, and everything else in 0008 stands.
- **Relation to decision 0009.** YOLO11 has still not been run. If its detection files exist, they replace the public
  detections as the tracker's input on MOT17. Nothing else changes.

## Approved scope

1. **Display only.** Boxes, IDs and trails are for viewing in the Workspace. They feed no metric, are not experiment P-1
   or P-9, and nothing is written under `runs/`.
2. **Every ID has a real source.**
   - **ETH, HOTEL, UCY:** human annotations, so these are the true identities.
   - **Synthetic:** the scene generator, with its true boxes.
   - **MOT17:** the existing IoU tracker over whole sequences, with its default settings, on the public FRCNN detections.
     Tracker mistakes show as ID changes on purpose. It is not ByteTrack or BoT-SORT.
3. **Boxes on annotated scenes are display estimates.** ETH, HOTEL and UCY record one point per person and no box. The
   box there is a person-sized estimate around that point, and the interface says so. This does not lift the rule
   "do not derive ETH/UCY ground-truth boxes": these are not ground truth, are not stored as such, and enter no
   measurement, so P-9 stays blocked for the same reason as before.
4. **UCY stays in pixels.** Annotations are read in image pixels, so no calibration is inferred and no world coordinate
   exists. UCY still offers no error controls.
5. **Identity switches stay the frozen injector's.** After a switch, a label's box is the other person's box. A test
   enforces this exactly on synthetic scenes.

## Implementation-level choices, not specification requirements

- **Background subtraction was rejected.** On real footage the existing blob detector boxed railings, shadows and
  branches, missed dark clothing and merged crowds. It remains a synthetic-scene tool.
- **Box rules by scene.** The annotators marked different body parts (feet in UNIV and ZARA1, heads in ZARA2, ETH and
  HOTEL), so each scene has its own rule, read from real frames (`SCENE_BOXES` in `video_tracks.py`).
- **MOT17 score cut-off.** Display uses detections scoring 0.5 or more, which reduces short false tracks.
- **Trails.** Each ID has one colour. Trails fade with age, default to 2 seconds, and break at gaps, dropped points and
  fragments. Per-frame tracker trails are drawn at 5 Hz so that box jitter does not read as motion.
- **The code stays outside `io/`.** `io/` feeds `source_sha256`, so these loaders live in the platform.

## Not in this scope

Real detector boxes on ETH, HOTEL or UCY (these need a person detector such as YOLO11, on the owner's approval), real
tracker IDs on those scenes, stored or evaluated tracker results for them, and error controls on UCY or MOT17.

**Later.** The owner approved YOLO11 boxes with tracker IDs on ETH, HOTEL and UCY, for display only, in
[decision 0015](0015-yolo11-on-every-real-video.md).
