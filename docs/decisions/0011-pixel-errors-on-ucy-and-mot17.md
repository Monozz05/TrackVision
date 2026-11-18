# Decision 0011: Live errors and forecasts in pixels on UCY and MOT17

- **Date:** 2026-11-01
- **Decided by:** the project owner, who asked for error sliders, trajectories and prediction on every dataset, and
  approved this plan after being told it needs an exception to specification section 6.1.1.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This narrowly changes item 6 of decision 0008, item 5 of decision 0009 and item 4 of
  decision 0010: UCY and MOT17 now take live errors and show forecasts. Everything else in those decisions stands.

## The exception

Specification section 6.1.1 says "MOT17 trajectories are never used to train or evaluate the trajectory predictor."
The owner allowed one exception, **for live viewing in the Workspace only**. The rule still governs every research
result: no MOT17 forecast or ADE is saved, counts as evidence, trains anything, or passes any gate.

## Approved scope

1. **Pixels, not metres.** UCY has no calibration on this machine and MOT17 has no world coordinates. Noise, distances,
   ADE, FDE and RMSE on these 10 scenes are in video pixels. No calibration is inferred or downloaded, and no pixel value
   is converted to metres.
2. **Exploratory, never a result.** Every value is labelled exploratory and in pixels, and is never written under
   `runs/`. Values cannot be compared with metres or between scenes. As in decision 0008, this is not P-10.
3. **Frozen code, called unchanged.** The injectors, predictors and scorer work on plain numbers, so nothing under
   `io/`, `predict/`, `inject/`, `metrics/` or `synthetic/` changed and `source_sha256` is unchanged. CV and CA only.
4. **Metre scenes are unchanged.** ETH, HOTEL and the synthetic scenes return the same values as before, overlays
   included. Only the field names changed, because no pixel value may sit in a field ending `_m`.
5. **Ground truth, not tracker output.** ADE needs true positions, so MOT17 errors run on `gt.txt` (rows the reader
   classes as evaluable pedestrians). At zero errors MOT17 still shows the tracker. With errors on, the overlay uses
   ground-truth tracks, whose ID numbers differ, and the scene says so. This is not P-9.
6. **Existing limits stay.** LSTM and GRU stay out of the Workspace, and scenes are shown one at a time and never pooled.

## Implementation-level choices, not specification requirements

- **Position noise range.** 0 to 60 px on every pixel scene. No sweep ran in pixels, so these sliders show no tested range.
- **Identity-switch distance.** The frozen 2.0 m has no pixel meaning, so people must be within one median person height,
  measured per scene (57 to 284 px). On UCY that height comes from the display box estimates, and this is the only use of
  those estimates outside drawing.
- **Windows.** The frozen 8 observed and 12 predicted steps on every scene, including moving-camera sequences.
- **Tracked point.** The bottom centre of the box on MOT17, and the annotated point on UCY.
- **UCY grid.** Each spline starts on its own frame, which would stop identity switches from ever finding a partner, so
  all UCY people now share one grid of multiples of 10 frames. This also moves their drawn samples by up to 9 frames.
- **MOT17 grid.** Multiples of `round(0.4 s × frame rate)`. MOT17-05 runs at 14 fps, so its step is 6 frames, or 0.43 s,
  and the scene says so.
- **Slider moves.** One request at a time per scene, and a request the user has moved on from is dropped. Moving a slider
  repeatedly on UNIV took 65 s to settle without this.

## Known limits of these values

- **UCY positions are interpolated.** Control points lie about 1.2 s apart (median 29 to 35 frames), so 0.4 s samples are
  linear between them. CV and CA are helped on those straight stretches, so clean ADE may read lower than on denser
  annotations.
- **Standing people.** Long gaps between control points are people standing still, up to 27% of annotated time in ZARA2.
  They also lower clean ADE.
- **Moving cameras.** On MOT17-05, 10, 11 and 13 paths include the camera's motion, so forecasts there partly predict the
  camera. Their clean CV ADE is 113 to 190 px, against 24 to 108 px on the static sequences. The scene says so.
- **Short tracks.** A forecast needs 8 s of unbroken track, so only 9 to 62 people per MOT17 sequence get one.

## Not in this scope

Any calibration for UCY, metres on MOT17, stored results or curves for these scenes, tracker output as a clean input,
camera-motion compensation, and any change to the Analysis page beyond correct units.

## Open

- Whether the owner formally amends specification section 6.1.1. This decision does not, and it does not need to for
  live viewing.
