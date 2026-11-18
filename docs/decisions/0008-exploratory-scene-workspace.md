# Decision 0008: Exploratory scene workspace

- **Date:** 2026-11-01
- **Decided by:** the project owner, who chose exploratory combinations, all datasets with UCY and MOT17 as video only,
  and a two-page Workspace and Analysis layout.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to decision 0007.** This narrowly changes item 3 of decision 0007: stored results are still served
  unchanged, but the workspace routes now compute live values. Everything else in 0007 stands, including that no
  research gate is passed and that platform work started ahead of the P6 checkpoint.

## Approved scope

1. **Live injection on a selected scene.** `/api/workspace` runs the frozen injectors and the frozen scorer, unchanged,
   on ETH, HOTEL and the synthetic scenes, with CV and CA.
2. **Exploratory, never a result.** Live values are labelled exploratory on screen and in the API, and are never written
   under `runs/`. They are not experiment records, and they are not evidence for any hypothesis.
3. **Combinations are not P-10.** Combining errors here does not resolve P-10 or H8. Decision 0004 keeps P-10 blocked,
   and `h8_generation_attempts` stays null.
4. **One error equals the frozen path.** With one error active, the inputs equal `evaluation.corrupt` exactly. Tests
   enforce this, and a live value at a stored level reproduces the stored sweep.
5. **Relaxed for local viewing only.** The owner relaxed two earlier rules for video playback on this machine:
   "no UCY conversions" and "MOT17 images are never decoded". Browser copies are kept in the Git-ignored `data/cache/`.
   `data/raw` is only read, the MOT17 ZIP is never extracted, and the copies are never committed.
6. **Video-only scenes.** UCY has no image-to-world calibration, and MOT17 has no world coordinates and is never used
   for prediction, so neither offers error controls. No calibration is inferred.

## Implementation-level choices, not specification requirements

- **Composition order.** D acts first, because it decides which pedestrian each label follows. A, B and C then act on
  the relabelled tracks. They read separate random streams over the same samples, so their order among themselves has
  no effect.
- **Window validity.** A combined window needs a complete label history (D), one fragment (C), and either surviving
  observations (B2) or a resolved interpolation (B1). A lost window lowers coverage and is never filled in.
- **Slider ranges.** B, C and D run from 0 to 100 percent, and A from 0 to 1 m. The frozen grids stop lower, so the
  interface marks values above the tested range.
- **Seeds.** Metrics are means over the frozen seeds within one scene. Forecasts drawn on the video use the first seed.
- **Overlay.** Tracks are drawn on the matching video frame number. The exact annotation frame offset is undocumented
  (the EWAP assumptions in `io/ewap.py`), and positions between samples are interpolated for display only.

## Not in this scope

Saving live results, LSTM or GRU live (stored curves are shown instead), any calibration for UCY, injection on MOT17,
injecting errors into pixels, and E1 to E3. Mitigation and collision work stay deferred.

## Open

- Whether the owner formally amends the order in specification Part 13 (carried over from decision 0007).
- P-10 stays blocked until decision 0004's open items are resolved.
