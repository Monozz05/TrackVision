# Decision 0001: P2-d protocol decisions (predictors, seeds, rotation, UCY, fallback)

- **Date:** 2026-10-31
- **Decided by:** the project owner, on the P2-d audit (read-only predictor-protocol and feasibility audit).
- **Governing specification:** v4.15 (unchanged). v4.16 is not frozen; nothing here amends either.
- **Recorded before any learned-model result exists.** No result informed these choices.

Configuration hash (`trackvision config hash`, base only):
- before: `dc1fba9ae82288552d219a76aa9cbadabb2cad018a3fcd290cfcdaca9ee1b079`
- after:  `b9baf63dc1638efe967f2327eb740987f7f7297ace135c550e9d79b6748efb11`

Experiment IDs embed the configuration hash, so runs made after this change get new IDs. Earlier runs
and their manifests are unchanged and keep their recorded hash.

## A. Constant-acceleration (CA) estimator: retained

- The CA predictor stays exactly as v4.15 §5.4 specifies, and as implemented in
  `src/trackvision/predict/baselines.py` (P2-a):
  - p̂_{H+k} = p_H + k·d1 + ½k²·d2;
  - d1 = p_H − p_{H−1}, d2 = p_H − 2p_{H−1} + p_{H−2}.
- **Estimator label:** `backward_difference`. Record it in `predictor_params` once experiment records
  are written.
- **Known property, not a defect:**
  - The velocity term is a backward difference: it estimates the velocity at t_H − Δt/2.
  - On constant acceleration a, the prediction is off by exactly −k·½aΔt² at step k (tested in
    `tests/unit/test_ca_baseline.py`).
  - CA reproduces constant velocity exactly.
- **Rejected alternative:** a second-order backward velocity, (3p_H − 4p_{H−1} + p_{H−2}) / 2Δt.
  - It is a different predictor and would need a specification amendment.
  - Under white position noise it is about 7.7% noisier at k = 12.
- No configuration or test change.

## B. Seeds: `[0, 1, 2]`

- `research_choices.seeds.value: [0, 1, 2]`.
- **Count:** three, the frozen minimum (`frozen.protocol.min_seeds`).
  - The specification sizes the sweep as "5 folds × 3 seeds" (§13.1).
  - Seeds describe within-fold variation only: they are aggregated before scenes and are not
    scene-level replicates (§7.3, §7.4).
- **Values:** arbitrary but fixed in advance. The count may be revisited only before the first
  learned-model run, and only on measured compute, never on results.
- **Streams:**
  - Each seed gives independent deterministic streams, one per purpose, following the existing
    `numpy.random.SeedSequence` convention: `SeedSequence([seed, purpose, ...])`, as used by the
    injectors with family codes A = 1 … D = 4.
  - The purpose codes for model initialisation, batch order and any other training randomness are
    assigned when that code is implemented. They must not collide with the corruption family codes.
- The deterministic predictors (CV, CA) use no seed.
- **Not yet implemented:** no training or corruption code changes with this decision.

## C. Inner-validation rotation (real static scenes only)

```yaml
inner_validation_rotation:
  value: {ETH: UNIV, HOTEL: ZARA1, UNIV: ZARA2, ZARA1: ETH, ZARA2: HOTEL}
```

- **Rule:** the validation scene is the scene two places later, cyclically, in the frozen order
  ETH, HOTEL, UNIV, ZARA1, ZARA2 (§5.3.1).
  - Every scene validates exactly one fold.
  - No scene validates its own fold.
  - ZARA1 and ZARA2 (same location and camera) never validate each other.
  - The rule does not depend on any data.
- **Implementation-level reasoning, not a specification requirement:** avoiding a same-location
  validation scene. When ZARA1 is the test scene, ZARA2 remains in training, as in standard
  leave-one-scene-out practice.
- **Scope:** the five real static scenes only.
  - It is never applied to synthetic scenes. Synthetic engineering experiments use an explicitly
    separate fixture mapping (`tests/fixtures/synthetic_split.py`, `cyclic_rotation`).
  - The fold builder rejects this mapping for any other scene set.
- **Currently unusable in practice:** only ETH and HOTEL have world coordinates (see D).
- Under v4.16 the choice would be redefined over development units. This mapping does not decide
  that.

## D. UCY world coordinates: blocked

- UNIV (students003), ZARA1 and ZARA2 remain blocked **pending calibration provenance and licensing
  permission**.
- **Local package** (`data/raw/ucy`, manifest `data/manifests/ucy.sha256`):
  - spline control points in centre-origin pixel coordinates;
  - videos;
  - no homography, no calibration and no licence text. The `.rar` archives contain only the same
    annotations and software.
- **Not adopted:** community homographies or world-coordinate conversions (OpenTraj `H.txt`, RAPID,
  Social GAN preprocessed files). Their provenance, conventions and data licences are unestablished
  (Phase 0B G3 findings).
- **Not done:** no UCY data was downloaded, modified or converted.
- **Unblocking needs:** calibration from an authoritative source with known provenance, plus
  research-use permission. Any other source would be a dataset-provenance change to the
  specification and needs an explicit decision.

## E. Learned-model training environment

- Authorised: an isolated CPU-only PyTorch probe in a disposable environment outside the
  repository and the project `.venv`.
- **No project environment change at this stage:** `pyproject.toml`, the lock file and
  `env/README.md` were unchanged by the probe.
- Adding PyTorch to the project is a separate decision.
- **Update (P2-e, 2026-10-31):** CPU-only `torch==2.13.0+cpu` was installed into the project
  `.venv` with approval, from the same hash-verified wheels.
  - The lock file now carries verified SHA-256 hashes for every package.
  - `env/README.md` documents the installation and its verification.
  - `pyproject.toml` was not part of that approval and does not declare PyTorch.
  - No CUDA component is installed.

## F. Evaluation fallback

- **No two-scene ETH/HOTEL learned-model research experiment.**
  - v4.15 fixes five leave-one-scene-out folds with nested validation (§5.3, §5.4) and has no
    predeclared downgrade for fewer scenes.
  - Two-scene alternatives break §5.4: they drop validation early stopping or validate within the
    training scene.
- **LSTM/GRU** implementation and training may proceed **only as synthetic multi-scene engineering
  feasibility**: at least 3 synthetic scenes, real nested splits and the six §11.7 leakage checks.
- **Synthetic learned-model results** are never presented as real-world pedestrian generalisation,
  nor as five-fold real-scene research results (P-3, P-12, P-15, RQ6, RQ9, H6).
- **CV/CA results on ETH and HOTEL** remain descriptive and local-only:
  - trajectory-only;
  - one scene at a time, never pooled;
  - dataset permissions unconfirmed.

## G. P2-e learned-predictor implementation parameters (approved 2026-10-31)

These complete what v4.15 §5.4 leaves open. The §5.4 values themselves are read from
`frozen.predictors.learned` and are restated here only for completeness.

| Item | Value | Source |
|---|---|---|
| Cells | LSTM and GRU | §5.4 |
| Recurrent layers / hidden size | 1 / 128 | §5.4 (config) |
| Encoder input | `history_len − 1` frame-to-frame observed displacements, shape (N, H−1, 2) | §5.4 |
| Prediction head | one linear layer from the **final encoder hidden state** to `horizon_len × 2` values; no attention, no autoregressive feedback | G |
| Output | `horizon_len` frame-to-frame displacement vectors, shape (N, F, 2); positions by cumulative sum from p_H | §5.4, G |
| Optimizer, learning rate | Adam, 0.001 | §5.4 (config) |
| Loss | MSE on displacements | §5.4 (config) |
| Learning-rate schedule | ReduceLROnPlateau on validation ADE: factor 0.5, patience 10, minimum 1e-5 | G |
| Batch size | 64 | §5.4 (config) |
| Batch order | reshuffled every epoch by a seeded generator | G |
| Epochs | at most 200 | §5.4 (config) |
| Early stopping | on validation ADE, patience 20 | §5.4 (config) |
| Checkpoint selection | lowest validation ADE; a tie keeps the earlier epoch | G |
| Precision | float32 model training; float64 reconstruction and final metrics | G |
| Gradient clipping | maximum norm 1.0 | §5.4 (config) |
| Seeds | `[0, 1, 2]` (B), each with separate purpose-specific streams | B, G |

### Implementation-level details, fixed in code and recorded with every checkpoint

These are not research parameters. They are recorded so that nothing is silent.

- **Random streams:**
  - `SeedSequence([seed, 101])` for weight initialisation;
  - `SeedSequence([seed, 102])` for batch order.
  - Codes 101 and 102 avoid the corruption family codes 1 to 4.
- **Last batch:** the last, partial batch of an epoch is kept, not dropped.
- **Scheduler arguments not set above** use PyTorch's defaults and are recorded: mode `min`,
  threshold `1e-4`, threshold mode `rel`, cooldown 0.
  - With the relative threshold, the scheduler counts very small validation improvements as no
    improvement.
  - Early stopping and checkpoint selection count any strict decrease.
- **Determinism:**
  - `torch.use_deterministic_algorithms(True)` during training.
  - The training thread count is recorded, since repeatability is established on the same thread
    count.
- **Normalization:** none (v4.15 §5.4 normalization note). It is declared `not_used` to the
  normalization-scope leakage check.
- **Leakage checks:** run before every training run on the actual input function and arrays
  (`predict.training`). Any failure aborts before anything is written.
- **Controlled-fixture overfitting criterion** (P3 sanity test,
  `tests/unit/test_training.py`):
  - **Fixture:** 4 straight constant-velocity windows. Every position is an exact dyadic
    multiple of q = 1/16 m, so it is exact in float32 and float64.
  - **Threshold:** τ = 2⁻¹⁰ m ≈ 0.98 mm on every window's ADE.
  - **Why τ lies where it does:** between the fixture's two numerical anchors.
    - Lower anchor: float32 resolution at the fixture's largest coordinate, about 4.8e-7 m.
      Accuracy below this cannot be demanded.
    - Upper anchor: the fixture quantum q = 1/16 m, the smallest per-step difference between
      distinct fixture trajectories.
    - τ is about three orders of magnitude above the float32 floor and 1/64 of q, and is exactly
      representable.
  - **Fixed before any result:** τ was fixed before any result was seen, not set from any
    dataset's performance.
  - **Reproducibility:** training is deterministic, so a single run decides; no margin for
    run-to-run variation is needed.
  - **Fixture size was iterated:**
    - With a 16-window fixture, GRU reached τ only after 8,000 steps and LSTM plateaued near
      2.6 mm.
    - With 8 windows, LSTM reached 1.18 mm.
    - With the 4-window fixture, both reach τ within a fixed 4,000-step budget: LSTM first at
      about step 3,300, GRU at about step 600.
    - The LSTM margin is narrow.

## Unchanged

- v4.15 Phase 1 remains BLOCKED; v4.16 remains unfrozen; no gate is claimed passed.
- These research choices remain unresolved: `moving_dataset`, `trackers_evaluated_set`,
  `tracker_params`, `h8_generation_attempts`, `discontinuity_rejection_v_max_mps`.
