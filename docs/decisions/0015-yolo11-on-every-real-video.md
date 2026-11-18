# Decision 0015: YOLO11 boxes on every real video, for display

- **Date:** 2026-11-02
- **Decided by:** the project owner, who asked for a box on every person visible on screen, using YOLO11, and approved this plan.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This narrows two limits. Decision 0009 kept YOLO to MOT17. Decision 0010 left real detector
  boxes on ETH, HOTEL and UCY to the owner. Everything else in decision 0009 still holds.
- **State, updated 2026-11-02:** YOLO11 is installed and **in use on MOT17 only.** The owner approved the install in chat, and it went
  into its own environment at `tv-envs\yolo-py312`. After seeing real results the owner chose to keep annotated boxes on ETH, HOTEL and
  UCY. See "What happened on real video" below. Items 2 to 4 of the scope still describe what the code does if detection files exist.

## Why

ETH, HOTEL and UCY annotations leave people out. ETH has nobody annotated between frames 2394 and 2862 although at least 8 people
are visible, and nobody on 25% of its 0.4 s steps (decision 0014). Boxing everyone needs a real person detector.

## Approved scope

1. **Display only.** YOLO boxes and tracker IDs are drawn and counted. They are not ground truth, they are not experiment P-1 or
   P-9, and they never enter ADE, FDE or any other measurement. A test shows that a scene gives exactly the same injection and
   sweep results with and without a detection file.
2. **With no errors on,** ETH, HOTEL, UCY and all their parts show YOLO11 boxes with IDs from the existing IoU tracker, as MOT17
   does. Synthetic scenes keep their true boxes.
3. **With errors on,** errors still run on the annotated people, because ADE needs true positions.
4. **Anyone else keeps a thin grey box,** with no ID, on every video scene including MOT17, so nobody loses their box when errors
   go on. The "people tracked" count includes them. A tracker box counts as the same person as an annotated one when the two cover
   at least half of the smaller box.
5. **Separate environment, stored not live.** As in decision 0009. `scripts/yolo_detect_videos.py` reads the browser copies, so
   frame k of a detection file is the frame k the browser shows, and it refuses a recording unless every browser frame was read.
   A part uses its recording's boxes. Output goes to the Git-ignored `data/cache/detections/`.

## What this cannot promise

YOLO will not catch literally everyone. ETH looks almost straight down and its people are tiny, so very small or hidden people can
be missed. The Workspace says so beside the boxes.

## What happened on real video

- **ETH and HOTEL.** YOLO11 fails from straight overhead. On ETH frame 2473 it found 1 to 3 of about 6 people, even with `yolo11x`,
  larger image sizes, enlarged frames and lower confidence, and it found nobody on a HOTEL frame with about 9 people visible. Showing it
  would replace the annotated boxes with fewer boxes, so **no detection file exists for them**, and ETH's empty stretches remain.
- **UCY.** It ran on UNIV, ZARA1 and ZARA2, but drew far fewer people than the annotations and broke IDs far more often. The owner chose
  annotated boxes here too, so those three files sit unused in `data/cache/detections_unused/`, which means the choice can be reversed.
  They were in use for a short time, and ZARA2's arrived after the owner froze the Workspace, because the run was already going.
- **MOT17.** It runs here, as the owner first asked, and every one of its 17 scene choices shows YOLO11 boxes.

Running `scripts\setup_yolo.ps1` without `-InstallOnly` runs detection on all of these again, which would put YOLO back on ETH, HOTEL
and UCY. Use `scripts/yolo_detect_mot17.py` alone to refresh MOT17.

## Implementation-level choices, not specification requirements

- **Model and settings.** `yolo11m`, image size 1280, minimum confidence 0.25, as decision 0009. A larger image size or model may
  be used on a video where the first test is poor, and the file records what was used.
- **Tracker.** The existing `IouTracker` with its default settings, on every video.
- **GPU.** PyTorch is installed from its own index, because PyPI's Windows build has no GPU support.

## Not in this scope

Tuning or training YOLO, precision or recall as a shown metric, choosing the detector for P-1, and any change to how errors or
forecasts are calculated. A check of how many annotated people fall inside a YOLO box is run once, for the owner only, and is
neither stored nor shown.
