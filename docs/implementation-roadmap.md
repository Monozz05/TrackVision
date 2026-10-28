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
