# Decision 0013: A rebuilt Analysis tab, with live results, tracker details and E5 to E8 together

- **Date:** 2026-11-02
- **Decided by:** the project owner, who asked for a cleaner Analysis tab, a fix for "No errors applied" showing while errors
  were set, tracker details like two pictures they supplied, and a check of E5 to E8, and approved this plan.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This changes the page described in decision 0008 and adds to decisions 0010 to 0012. It
  changes no scene, no stored result and no calculation, and the Workspace behaves as it did before this work.

## Approved scope

1. **Analysis calculates for itself.** It used to show only a finished result left behind by the Workspace, so leaving early,
   changing scene or refreshing lost it, and 6 of 7 routes to the page showed "No errors applied". It now calculates from the
   current sliders and scene. The result is the same exploratory calculation as the Workspace, and it is never saved.
2. **Four tabs:** Your errors, Stored results, Tracker, and E5 to E8, under one line naming the errors in use.
3. **Tracker tab.** Stored detector and tracker settings beside stored results, including E2, E3 and E4 values that no screen
   showed before. A value that was not stored shows as a dash.
4. **Noise example.** One real person at three noise levels, from the frozen injector. It is a live example, not a saved figure.
5. **E7 and E8 follow the page's scene,** and each E5 to E8 section names its source run and whether that file still has its
   recorded hash.
6. **Saved figures.** Stored results shows the SVG figures saved with a run, for the scene and error mode chosen.

## The two pictures come from another project

Both pictures come from `traj_pred-main`, a separate project with its own runs. This project has never run YOLOv8s or BoT-SORT,
and P-1 stays blocked on exactly those. So the same panels here show this project's real setup: MOT17's public detections and
the simple IoU tracker. No value from that project is shown, and nothing is read from it. The panel says the tracker is not
ByteTrack or BoT-SORT, and stored labels such as "not P-1, not P-9" stay as stored.

The approved plan said `runs/` holds no saved figures. That was wrong: it holds 126 SVG figures, and the first search looked only
for PNG files. None is a noise example, so that panel is built live as approved, and item 6 was added for the figures that exist.

## What was checked in E5 to E8

The run the dashboard shows, `run-2026-11-01~analysis`, was audited:

- all three inputs still exist unchanged, and rerunning E5 to E8 from them gave byte-identical results;
- the API serves all five sections as stored;
- by hand: 20 of 20 E5 pairs against E4, 16 of 16 Spearman values, 4 of 4 family AUDC values, 141 of 141 E7 values with the tie
  rule of decisions 0004 and 0006 applied, and 128 of 128 E8 entries.

**E1 to E3 do not feed E5 to E8, by design.** Decision 0006 keeps E3 off the curves. E5 is built from the synthetic E4 run, and
E6 to E8 from stored curves. The older run `phase5-6~e5_e8_analysis` has lost two input files. Its results match the current run
in every result section, apart from the input hash E6 records, and the dashboard does not use it.

## Implementation-level choices, not specification requirements

- **Noise levels.** 0.05, 0.20 and 0.50 m from the frozen grid. No grid exists in pixels, so those are 10, 30 and 60 px.
- **Example person.** The window at the 75th percentile of distance walked, so the person visibly moves. Seed 0.
- **Tracker run for a synthetic scene.** The newest E4 run covering it. Its settings come from that run's manifest.
- **Fixed rows.** "No training", "no camera motion compensation" and "no ReID" are not stored fields. They are facts about the
  detection sets and `tracking/tracker.py`.
- **Parts.** A MOT17 part shows its recording's settings and says the results are for the full sequence. E7 and E8 use the full
  recording.

## Reverted: remembering scene and sliders

This work first kept the scene, sliders and open tab in the browser, so that a refresh kept them. The owner had it removed on
2026-11-02, the day it was added, because it changed the Workspace after they had accepted it: sliders left on came back after a
refresh, so the video opened with errors on. Boxes went missing or jumped with no slider moved. State again resets on a refresh,
as it did before. On Analysis, a refresh or a new tab therefore shows "No errors applied", which is then true, because the sliders
are back at zero.

## Not in this scope

Showing anything from `traj_pred-main`, running a new detector or tracker, and any change to stored results.

**Later.** [Decision 0016](0016-simple-analysis-and-applications.md) replaces these four tabs with one simple page.
