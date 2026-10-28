# TrackVision

**Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments**

Research question: *How does tracking quality propagate into pedestrian trajectory prediction?* A
controlled study of error propagation in a detection-tracking-prediction pipeline.

The governing documents are specification v4.15 (research core frozen) and the Implementation Plan.
They are **not redistributed** in this public repository. Each collaborator keeps a local copy in
`docs/spec/`, as described in [`docs/spec/README.md`](docs/spec/README.md). When the code and the
specification disagree, the code is wrong by default.

## Status

| Phase | State |
|---|---|
| Phase 0A - Repository audit and scaffold | Scaffold, configuration system, test harness and synthetic smoke run in place, pending review |
| Phase 0B - Feasibility gates | **Not started** |
| Phases 1-7 | **Not started** |

**Not yet verified:**
- CUDA and the RTX 3060 GPU
- The datasets (none are downloaded)
- The static-arm raw-video gate and the moving-arm feasibility gate
- Homographies and stabilization
- The six composition tests and the six leakage checks (their places are reserved; they skip)
- Every research algorithm

The smoke experiment checks the scaffold only and has **no scientific validity**.

## Setup (Windows, PowerShell)

The project uses a `.venv` on the python.org CPython 3.12 interpreter. See
[`env/README.md`](env/README.md) for why (Smart App Control blocks the conda-forge interpreter on
this machine) and for what the lock file does and doesn't guarantee.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r env\requirements-lock-win_amd64-py312.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python.exe -m pip check
```

Then get the governing specification from the project owner and place it in `docs/spec/`,
checking it against its recorded SHA-256 (see [`docs/spec/README.md`](docs/spec/README.md)).
**Without it, validation reports INVALID and the test suite stops at startup**; nothing is
skipped or bypassed.

## Commands

```powershell
.venv\Scripts\python.exe -m pytest                                                     # full test suite
.venv\Scripts\python.exe -m trackvision config validate                                # base.yaml + unresolved choices
.venv\Scripts\python.exe -m trackvision config validate --config configs\experiments\smoke.yaml
.venv\Scripts\python.exe -m trackvision run --config configs\experiments\smoke.yaml    # writes runs\<experiment-id>\
```

**Smoke run.** It writes `config.resolved.yaml`, `fixture_summary.json` and `manifest.json` to
`runs\<experiment-id>\`.
- **Experiment ID:** `<name>-<first 10 hex of the config hash>-s<seed|na>`. It depends only on the
  configuration and the seed. Timestamps, Git state and environment details are recorded in the
  manifest but never affect the ID or the hash.
- **Existing runs:** an existing run folder is never overwritten unless you pass `--force`.

**`run` exit codes:**

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Smoke check failed |
| 2 | Invalid configuration |
| 3 | Unresolved research choice |
| 4 | Run folder already exists |
| 5 | No runner exists yet for this experiment kind |

## Configuration

[`configs/base.yaml`](configs/base.yaml) is the **only** place protocol constants are defined. Code
reads them and never hard-codes them. The file has four blocks:

- `frozen`: immutable protocol constants, each with its spec section. Examples:
  - the static protocol: `history_len`, `horizon_len`, `sampling_interval_s` and the
    leave-one-scene-out `folds` (§5.3);
  - the injection grids for A, B1/B2, C, D and E1-E3 (§3.5), and `d_max_m` (§3.5);
  - the association gaps and IoU threshold (§3.7);
  - the collision thresholds (§3.8);
  - the predictor training constants (§5.4);
  - the hypothesis tolerances (§2.3).
- `domains`: permitted selections that aren't already listed under `frozen`.
- `research_choices`: values the specification requires but doesn't give. All seven are **null**,
  and any experiment that uses one is blocked:
  - `seeds`
  - `inner_validation_rotation`
  - `moving_dataset`
  - `trackers_evaluated_set`
  - `tracker_params`
  - `h8_generation_attempts`
  - `discontinuity_rejection_v_max_mps`

  They must be fixed by a recorded decision before the phase that needs them, and **before**
  looking at any results.
- `meta`: the specification version, its path and its SHA-256. Validation fails if the local copy
  in `docs/spec/` is missing or doesn't match.

**How the frozen values are verified.** `tests/unit/test_frozen_coverage.py` checks all **143**
leaves under `frozen` automatically, each against a cited spec section. The checks are of four
strengths, and only the first is a direct comparison with the specification text:

| Leaves | Kind | What the test does |
|---|---|---|
| 87 | **Direct extraction** | The value is extracted from the spec by a pattern anchored to its wording and compared unchanged. |
| 20 | **Normalised or transliterated extraction** | The value is extracted, then put through one stated conversion before comparison: lower-case with `-`/spaces as `_` (`leave-one-scene-out` becomes `leave_one_scene_out`), CamelCase to snake_case (`ReduceLROnPlateau` becomes `reduce_lr_on_plateau`), a number word to an integer ("three" becomes 3), or a symbol to its config name (`σ_t` becomes `sigma_t`, `°` becomes `deg`). |
| 16 | **Derived** | The spec text supplies the inputs and the test applies a stated rule. The 8 `injection.conditions.*.family` values come from the condition and family lists of §3.5, using the rule that a condition's family is the letter its name starts with. The 8 `injection.conditions.*.regimes` values come from the §4.1 lines saying which families occur in both regimes and which only in the moving regime. |
| 20 | **Explicit assertion** | These are project labels or flags for spec prose that can't be extracted mechanically: the 8 condition-to-grid assignments, 4 unit labels (B, C, D, E3), the B1/B2 reconstruction labels, the mode-D cooldown label, the decoder's no-attention and no-autoregressive-feedback flags, the learned-model reconstruction label, the transform-direction label and the 3D trajectory-point label. The expected value is written in the test with its section, and the quoted spec phrase it stands for must appear verbatim in the specification. These are assertions, not comparisons with the spec text. |

What the guards prove:

- **Completeness.** A test fails if a frozen leaf has no check, or if a check names a path that is
  no longer a frozen leaf.
- **Mutation, one per leaf (143 tests).** Each leaf's value is changed once and must then fail its
  check. Numbers are incremented, strings get a suffix and booleans are inverted. List values are
  *lengthened*, so these tests on their own only show that a change of length is caught.
- **Same-length list changes (28 tests).** For every one of the 28 list leaves, each element is
  changed in turn while the length stays the same, 102 variants in all, and every variant must be
  rejected.
- **Realistic edits (4 tests).** A changed `d_max_m`, a grid with one level removed, attention
  switched on, and a relabelled split must each be rejected.
- **The original comparisons.** The 63 text comparisons from the former
  `test_spec_transcription.py` now live in `test_frozen_coverage.py`, moved verbatim. They run as
  part of the 143-leaf checks above, with the same patterns, sections and expected values, so
  they're no longer repeated as separate tests.

These tests prove the checks reject the specific changes above. They don't prove every
conceivable edit is caught.

**Not automated:** whether each project label (the 20 assertions and the derivation rules) is a
*good* reading of its spec phrase is a matter of human review. So are the research-choice
descriptions and the comments in `base.yaml`.

An experiment file (`configs/experiments/*.yaml`) contains only an `experiment` block. It can
select from what base.yaml permits, for example a subset of a frozen severity grid, but it can
never change a frozen value.

## Repository map

| Path | Contents |
|---|---|
| `configs/` | `base.yaml` and `experiments/smoke.yaml`. Folders such as `scenes/`, `predictors/` and `injectors/` are added in the phases that need them, and may select from `base.yaml` but never redefine a frozen value |
| `src/trackvision/` | `config.py`, `repro.py`, `cli.py`, and the `io`, `geometry`, `tracking`, `predict`, `inject`, `metrics`, `stats`, `experiments`, `viz` and `platform` subpackages, each documented with the phase that fills it |
| `tests/unit/` | Tests that run now |
| `tests/invariants/` | Reserved composition and leakage checks |
| `tests/fixtures/` | Synthetic known-answer data |
| `docs/` | `spec/` (your local, untracked copies of the governing documents, plus the tracked README) and the research roadmap, [`implementation-roadmap.md`](docs/implementation-roadmap.md) |
| `env/` | The environment notes and the pip lock |
| `data/` | Datasets, Git-ignored apart from the README |
| `runs/` | One folder per experiment ID, Git-ignored |
| `scripts/` | `check_staged.py`, the pre-commit staged-file check |
| `web/` | The Next.js frontend, relocated unchanged; reserved for Implementation Plan Phase 7 |

## Git workflow

- **Layout:** one repository, intended for public collaboration. The Python research backbone is
  at the root, and the Next.js frontend is in `web/`.
- **Pushing:** nothing is pushed, and no remote is added or changed, without explicit approval.
- **Governing documents:** never committed, because they aren't redistributed. `.gitignore`
  excludes them and `scripts/check_staged.py` blocks them if they're force-staged.
- **Phase commits:** each phase ends in one reviewed commit. Before committing:
  1. Run the phase's tests, its gate and the full suite.
  2. Stage files explicitly by path.
  3. Run `.venv\Scripts\python.exe scripts\check_staged.py`.
  4. Review the staged list and the diff, then commit only after approval.

  The full policy is in [`docs/implementation-roadmap.md`](docs/implementation-roadmap.md#commit-policy).
- **Never committed:** datasets, run outputs, credentials, environment folders, model weights,
  generated videos, caches, and any file over 1 MB without an approved exception.
