# Decision 0016: A simple Analysis page for every scene, and an Applications page

- **Date:** 2026-11-02
- **Decided by:** the project owner, who asked for simple graphs available on every dataset and every divided scene, no tracker
  part, no labels such as E5 and E6, and a new section on applications with a dropdown, and approved this plan.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Relation to earlier decisions.** This replaces the four tabs of decision 0013. The noise example and Analysis calculating for
  itself stay. **The Applications page is an exception to the language rule of specification section 1.2**, described below.

## Analysis

One page with no tabs. Pick any of the 52 scenes and every graph fills in. Removed: the tabs, the Tracker part, every E-label and the
Stored results tab. Saved figures stay in Records, and the read-only `/workspace/scenes/{scene}/tracker` route stays in place, unused.

**A new live calculation.** `services/scene_sweep.py` runs each of the four errors at every tested level on one scene, with the
frozen injectors, three seeds and the frozen scorer. It reuses `exploratory.combine`, `evaluation.score`, `aggregate`,
`inverse_requirement` and `first_crossing` unchanged.

**It equals the stored results.** On ETH, HOTEL and SYN-501 to 505, all 1,120 stored values of ADE, FDE, coverage and achieved
severity are equal, as are all 168 limits of E7 and all 28 CV against CA results of E8. A test enforces this.

**It runs in the background, one level at a time.** A full sweep took 45 s on UNIV, 10 to 15 s on four other scenes, and under
10 s on the other 47. The page polls and shows progress. The scene opened last goes first, and a scene nobody is viewing stops.

**Nothing is saved.** Decision 0011 keeps MOT17 forecasts out of every saved file, so results stay in memory for the life of the
backend process.

**Scenes and parts are never combined.** Each scene is shown alone. Two saved project results stay at the bottom under plain
names, "Which error matters most overall" (from E6) and "Does a better tracking score mean better prediction?" (from E5), each
saying that it covers synthetic scenes only.

## Implementation-level choices, not specification requirements

- **Rises use the same windows, clean against with errors.** Missing detections and broken tracks only remove windows, and the ones
  left keep exactly their clean forecast, so their rise is exactly 0%. Against all windows they would show changes caused only by
  which windows are left, such as −38%.
- **Limits.** Position noise and identity switches use the stored definition, relative ADE degradation against all clean windows
  (decision 0004). Missing detections and broken tracks cannot raise ADE, so their limit is on the share of forecasts lost. Those two
  rows therefore differ from stored E7.
- **Noise levels in pixels.** 6, 12, 24, 42 and 60 px, which is the metre grid scaled to the top of the slider. No grid exists in
  pixels.
- **Missing detections** means condition B2, the representative condition, with no gap filling.

## Applications, and the exception to section 1.2

Section 1.2 limits platform output to descriptive language: no causal, generalising or recommending statements. A ranked "reduce
this error first, because..." is a recommendation. The owner asked for exactly that, so this page is an exception, and only this page.

**What keeps it honest.** The page says on every view that the priorities and reasons are engineering judgement, that none was
measured on that application, that numbers come from the named scene and not from a real system, that collision performance has
not been measured, and that no research gate is passed. The ranks, reasons, look-ahead times and scene choices are hand-written in
`web/config/applications.ts`, so they are easy to change. No measured number lives in that file.

**What the reasons must respect.** At the strongest tested level, position noise raised ADE and FDE the most on all 52 scene choices,
CV beat CA on clean tracks on all 52, and missing detections and broken tracks remove forecasts without changing those left. So a
rank depends on what the application needs: accurate forecasts, a forecast for everyone, or keeping one identity. Where following a
person matters more than ADE, as in CCTV security, the page says so and does not imply that ADE supports the order.

**Numbers.** They come from the live sweep of a scene chosen from those named for the application, and are never pooled. MOT17 keeps
its viewing-only label, and moving-camera scenes keep the caution that values include the camera's motion. Where no matching
footage exists, for moving drones, signal-controlled junctions and factories, the page says the scene is only the nearest available.

## Not in this scope

Saved runs from the interface, mitigation, collision operating curves, adaptive model selection, and any change to a stored result.
The research stage "Application impact and mitigation" stays blocked: this page does not measure it.
