# Decision 0012: A dropdown per dataset, numbered parts of long recordings, and more synthetic scenes

- **Date:** 2026-11-01
- **Decided by:** the project owner, who asked for more scenes and a dropdown per dataset, rejected any added scene without
  the same video and errors as the existing scenes, and approved this plan.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This adds to decisions 0008, 0010 and 0011 and narrows none of them. The 17 existing
  scenes return exactly what they returned before.

## Approved scope

1. **One dropdown per dataset** replaces the scene buttons on Workspace and Analysis. The workspace goes from 17 scenes to 52.
2. **Every added scene is a full scene:** video, boxes and IDs, all four sliders, forecasts, ADE and FDE, in the unit of its
   recording. The owner ruled out anything less.
3. **Numbered parts.** The datasets hold no other usable recording with both video and ground truth, so real-data scenes are
   added by cutting long recordings: 4 parts each of ETH, HOTEL, UNIV, ZARA1 and ZARA2, and 2 parts each of MOT17-02, 04, 05,
   10 and 11. That is 30 parts.
4. **Five new synthetic scenes,** SYN-506 to SYN-510, from the same generator and settings as SYN-501 to SYN-505.
5. **Existing limits stay.** Every value is exploratory, nothing is written under `runs/`, scenes are shown one at a time and
   never pooled, and only CV and CA run. MOT17 parts stay under the viewing-only exception in decision 0011.

## A part is not a scene in any statistic

The specification's statistical unit is the scene. A part shares its camera and its recording with the other parts of that
recording, so a value from a part must never be pooled with another part, counted as an extra scene, or treated as independent
evidence. Each part says so beside its sliders, and its result label repeats it. Stored results cover full recordings only, so
a part has no stored run, and Analysis points to the full recording where one exists.

## Implementation-level choices, not specification requirements

- **Number of parts.** 4 for ETH, HOTEL and UCY, and 2 for MOT17.
- **Cut points.** Equal lengths over the annotated stretch of each recording, on its sampling grid. The last part takes what the
  grid leaves over.
- **When to split.** Only if every part keeps at least 40 windows and 4 people with a forecast. MOT17-09 and MOT17-13 stay whole,
  because one of their halves would keep only 5 windows.
- **Part video.** Each part is its own short video, cut frame-exactly from the source, so frame k of a part is frame start + k
  of its recording. Its tracks are shifted to match.
- **IDs.** Those of the full recording. A person crossing a cut appears in both parts under one ID.
- **Windows.** The frozen 8 observed and 12 predicted steps. A window crossing a cut belongs to neither part, so forecasts thin
  out near a cut. A part's windows are exactly the recording's windows that lie inside it, with the same targets.
- **Settings.** Unit, position noise range, box sizes and identity-switch distance all stay those of the recording.
- **MOT17 at zero errors.** A part shows the full sequence's tracker output cut to the part, not a new tracker run, so its IDs
  match the full sequence.
- **Seeds.** 506 to 510 are the next numbers, and nothing else uses them.

## Measured

Windows and people with a forecast per part, recomputed independently from the raw files.

| Recording | Parts | Windows per part | People with a forecast per part |
|---|---|---|---|
| ETH | 4 | 285 to 1,217 | 37 to 111 |
| HOTEL | 4 | 180 to 402 | 16 to 58 |
| UNIV | 4 | 1,084 to 2,708 | 70 to 117 |
| ZARA1 | 4 | 329 to 673 | 27 to 47 |
| ZARA2 | 4 | 785 to 1,852 | 38 to 55 |
| MOT17-02 | 2 | 108 to 130 | 22 to 27 |
| MOT17-04 | 2 | 917 to 958 | 44 to 45 |
| MOT17-05 | 2 | 110 to 145 | 6 to 7 |
| MOT17-10 | 2 | 58 to 67 | 8 to 12 |
| MOT17-11 | 2 | 54 to 66 | 5 to 6 |

## Considered and left out

- **MOT17's 7 test sequences.** Real video, but their ground truth is not public (specification section 6.2), so ADE would have
  nothing true to compare with.
- **UCY's `students001`, `crowds_zara03` and `uni_examples`.** Real annotations in pixels, but the package has no video for them,
  and no trustworthy source for the videos was found.
- **UCY's Arxiepiskopi.** Real video, but its annotations are not in pixels: read as pixels, all 24 people would fit in a
  33 × 7 px patch. Using it would mean inferring a calibration.
