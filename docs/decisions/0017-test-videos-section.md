# Decision 0017: A separate Test Videos section

- **Date:** 2026-11-02
- **Decided by:** the project owner, who froze Workspace, Analysis and Applications, asked for a few downloaded pedestrian videos kept
  separately in a `videos` folder with the working shown on them in a section of their own, and approved this plan.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This adds a section and changes none. The three frozen sections keep their files byte for byte.
  The top menu gains one item, and `app.py` connects the new routes. Boxes, tracker and predictors are the ones already in use.

## What the section shows

Five freely licensed street videos from Wikimedia Commons, picked from a dropdown. Each shows YOLO11 boxes, IDs from the existing IoU
tracker, trails, CV or CA forecasts, and the four error sliders. Every video credits its author, licence and source, because CC BY
and CC BY-SA require it. `videos/SOURCES.md` lists all 11 downloaded files and says which are on the page.

## The rule that shapes it

**These videos have no true positions, so no accuracy value exists for them.** The backend sends no ADE, FDE, RMSE, coverage or tracking
metric, and a test confirms that no such field appears in any response. The page shows `—` for ADE and FDE and says why. Uploaded
or downloaded video is never treated as a ground-truth dataset.

What can run without true positions does run:

- **Errors act on the tracker's own tracks,** through the frozen injectors and `exploratory.combine`, unchanged. This corrects what
  was first said to the owner, that errors could not carry over to such a video. Only accuracy cannot.
- **"Forecasts kept"** is a count.
- **"Forecast moved"** is new: the mean distance between a forecast with and without errors. It needs no true position and it is
  change, not accuracy. The page says so beside it.
- **The dashed line** is where the tracker saw a person next. It is tracker output, labelled as such, and never called truth.

## What was tested before choosing

Scratch checks, for this decision only. None is stored, shown or evidence.

- **The videos.** 11 candidates ran through YOLO11 and the tracker. Five were kept. Left out: Shibuya (six camera cuts and too dense),
  Fremantle, Knäppingsborg and Quebec (almost nobody usable), the Hong Kong crossing and the close-up pedestrian area (about one
  forecast at a time). Detection is good on clear street-level HD footage, unlike the overhead videos of ETH and HOTEL (decision 0015).
- **The tracker stays as it is.** IDs break often, because the detector misses someone for a moment or two overlapping people swap
  boxes. ByteTrack and BoT-SORT were no better. Looser settings made tracks less clean. Rejoining broken tracks afterwards joined the
  wrong people too often: on MOT17, where the truth is known, only 18 to 72% of joins were right. So the page says plainly that one
  person can get several IDs.
- **CV is the default.** CA reaches four to five times further than CV on tracker boxes and often leaves the picture. CA stays one
  click away, and one predictor is drawn at a time.

## Where things live

| What | Where |
|---|---|
| Originals, all 11, and `SOURCES.md` | `videos/`, Git-ignored |
| Browser copies of the 5 | `data/cache/test_videos/` |
| YOLO11 boxes of the 5 | `data/cache/test_video_detections/` |

Nothing goes into `data/raw`, `data/cache/video`, `data/cache/detections` or `runs/`, and no dataset's boxes can reach this section or
the other way round. The five videos are a fixed list in `services/test_videos.py`, which is also the allowlist: no other name
reaches the disk. Results stay in memory.

`scripts/prepare_test_videos.py` checks each original against the SHA-1 Commons publishes, then builds a copy at most 720 pixels high
with every frame kept and even timing, so frame k is at exactly k / fps seconds. `scripts/yolo_detect_test_videos.py` then runs YOLO11
in its own environment.

## Implementation-level choices, not specification requirements

- **Position for a forecast.** The mean of the foot point over the step before it, using nothing from the future, because a box edge
  jumps from frame to frame and CV and CA read that as speed. This cut a forecast's reach by 7 to 40% in testing. CV and CA themselves
  are unchanged, and this applies to this section only.
- **Short gaps.** A person has a point on a step if any box falls in it, so a gap shorter than a step needs no bridging line, and one
  missed frame does not remove a forecast. This replaces the straight-line bridge first planned.
- **Forecast settings** follow the protocol: 8 points of history, 12 ahead, about 0.4 s a step, seeds 0, 1 and 2. A step is a whole
  number of frames, so at Toronto's 23.98 fps it is 0.42 s, and the page says so.
- **Detector settings** are those of MOT17: `yolo11m`, image size 1280, minimum confidence 0.25. The tracker keeps its defaults.
- **Noise slider** up to 60 px, as on the Workspace's scenes in pixels.
- **ID-switch reach** is the median box height of the video, as on MOT17.
- **The video list and the name "Test Videos"** are easy to change.

## Not in this scope

An upload feature, any accuracy measure on these videos, any new tracker or detector, and any change to the frozen sections. These
videos are not a dataset, not experiment P-1 or P-9, and not evidence about video in general. No research gate is passed.
