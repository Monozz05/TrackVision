# TrackVision research roadmap

The build order follows the *TrackVision Implementation Plan* (`docs/spec/TrackVision Implementation Plan.pdf`).
The research content of every phase, and every phase's gate, is defined by specification v4.15
(`docs/spec/trackvision_specification_v4_15.md`). Both are local copies, not redistributed in this
repository (see `docs/spec/README.md`). This roadmap points to the gates; it does not
restate their values. The frontend's own roadmap is kept unchanged at `web/docs/implementation-roadmap.md`.

## Phases

The Implementation Plan reorders the specification's phases: it moves the detector and tracker
later and stabilization earlier. It changes no experiment. Both numberings are given.

| Implementation Plan phase | Spec §14.1 | Gate defined in |
|---|---|---|
| Phase 0A - Repository audit and scaffold | part of P0 | The approved Phase 0A plan (see the root `README.md`) |
| Phase 0B - Feasibility gates | rest of P0 | Spec §14.1 P0, §6.1.1, §6.4; Implementation Plan p.4 |
| Phase 1 - Data and scene geometry | P2 (+ windowing from P3) | Spec §14.1 P2, §4.3 (six strict composition tests) |
| Phase 2 - Predictors and the clean baseline | P3 | Spec §14.1 P3, §5.4, §11.7 (leakage checks) |
| Phase 3 - Error injection | P4 | Spec §14.1 P4, §3.5, §3.6, §11.2 |
| Phase 4 - The core sweep | P5 | Spec §14.1 P5, §2.3, §7 |
| Phase 5 - The real tracker | P1 | Spec §14.1 P1, §3.7, §6.2 |
| Phase 6 - Mitigation and the application layer | P6 | Spec §14.1 P6 and the Tier-0 checkpoint, §14.2 |
| Phase 7 - The platform | P7-P13 | Spec §14.1 P7-P13 |

The research contribution is complete at the end of Phase 6. Phase 7 is individually droppable.

## Phase-completion gate

A phase is complete only when all of the following hold:

1. Its specified tests and its gate (as defined above) pass **on a real run**, not just in code.
2. Its end-to-end command runs with no manual steps, and its artefacts are on disk under `runs/`
   or `reports/`.
3. The full existing test suite still passes.
4. The phase commit has been reviewed and explicitly approved (see the commit policy below).

A gate value outside its stated range triggers investigation; it does not automatically invalidate
the phase. A failed gate downgrades scope only under a rule that was declared in advance. The
geometry composition tests and the injector tests cannot be downgraded.

Phase 4 protocol decisions, including how the Plan's Phase 3 monotonicity gate is evaluated for mode D,
are recorded in [decision 0004](decisions/0004-phase-4-protocol-decisions.md).

The limited Phase 5 and Phase 6 engineering scope, which closes no research gate, is recorded in
[decision 0006](decisions/0006-phase-5-6-engineering-scope.md).

The read-only results platform, which adds no research evidence and started ahead of the P6 checkpoint at the project
owner's instruction, is recorded in [decision 0007](decisions/0007-read-only-results-platform.md). Its exploratory scene
workspace, which runs the frozen injectors live and never saves results, is recorded in
[decision 0008](decisions/0008-exploratory-scene-workspace.md). YOLO11 boxes on MOT17, for display only, are recorded in
[decision 0009](decisions/0009-yolo11-display-detections.md). The tracking overlay on every scene is recorded in
[decision 0010](decisions/0010-tracking-overlay-on-every-scene.md). Live errors in pixels on UCY and MOT17, and the
exception to specification section 6.1.1 that allows them for viewing only, are recorded in
[decision 0011](decisions/0011-pixel-errors-on-ucy-and-mot17.md). Dataset dropdowns, numbered parts of long recordings and
more synthetic scenes are recorded in [decision 0012](decisions/0012-dataset-dropdowns-and-numbered-parts.md). The rebuilt
Analysis tab, and the check of E5 to E8 that went with it, are recorded in
[decision 0013](decisions/0013-analysis-tab-rebuild.md). ETH's video rate is recorded in
[decision 0014](decisions/0014-eth-video-frame-rate.md), and YOLO11 boxes on every real video, for display only, in
[decision 0015](decisions/0015-yolo11-on-every-real-video.md). The simple Analysis page for every scene, and the Applications
page with its exception to the language rule of specification section 1.2, are recorded in
[decision 0016](decisions/0016-simple-analysis-and-applications.md). The Test Videos section, on downloaded street videos with no
ground truth and so no accuracy values, is recorded in [decision 0017](decisions/0017-test-videos-section.md). None of this adds
research evidence.

## Commit policy

The repository is private, has no remote, and nothing is pushed without explicit approval.

1. **Before any change:** run `git status` and `git branch --show-current`, and report both. If
   there is uncommitted work from someone else, stop and ask. Never stash, reset, check out over,
   clean or overwrite it.
2. **Scope:** implement only the approved scope of the phase.
3. **Verify:** run the phase's specified tests and its gate, plus the full existing suite. Any
   failure blocks the phase-completion commit.
4. **Review:** check the diff against the phase scope. Stage files explicitly by path, then run
   `.venv\Scripts\python.exe scripts\check_staged.py`.
5. **Show for approval:** the test results; the gate outcome with artefact paths;
   `git diff --cached --stat`; the exact staged list (`git diff --cached --name-status`); the
   `check_staged.py` output; and the proposed commit message.
6. **Commit only after explicit approval**, with a descriptive conventional commit message (for
   example `feat(geometry): Phase 1 data loaders and scene geometry`). The body summarises the
   scope, the gate outcome and what is still unverified.
7. **No push** without explicit approval. **No phase-completion commit** if tests or the gate
   fail: fix the problem or report the blocker.
8. **No `--amend`, no `--no-verify`, no history rewrites.** Commits go on `main` unless a branch is
   requested for a phase.

Never committed: the governing documents in `docs/spec/` (not redistributed), datasets, run
outputs, credentials, environment folders, model weights, generated videos, caches and build
outputs. `scripts/check_staged.py` enforces this, plus the
1 MB file-size policy.
