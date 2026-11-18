# Decision 0004: Phase 4 protocol decisions (AUDC, ADE₀, monotonicity, P-15, P-10)

- **Date:** 2026-10-31
- **Decided by:** the project owner, on the Phase 4 audit and protocol reconciliation.
- **Governing specification:** v4.15 (unchanged). v4.16 is not frozen; nothing here amends either.
- **Recorded before any research sweep exists.** No result informed these choices.
- **Scope:** this record changes no code, configuration, specification or plan. AUDC, P-15 and P-10 are
  not implemented by it.

**Status labels.** *Frozen* restates v4.15. *Project decision* is not stated by v4.15 and is decided here.
*Plan qualification* qualifies the Implementation Plan without editing it. *Open* is unresolved.

## 1. AUDC

### Normalization (frozen)

The exact v4.15 §3.3 formula is kept, as implemented in `inject/severity.py` under decision 0002. s_norm
is never clipped or rescaled. In particular, s_norm is not rescaled by its maximum for AUDC.

### Procedure (project decision)

For one predictor, scene/fold and condition:

1. **Seed-level values.** At each requested level, compute relative degradation D per seed, using that
   seed's ADE₀ (section 2).
2. **Seed aggregation.** At each severity level, average the valid seed-level D values. Record the number
   of valid seeds and of undefined seeds, with the reasons. If no seed is valid, the level is undefined.
3. **One curve.** Calculate AUDC from these aggregated level values. Do **not** calculate a separate AUDC
   for each seed and average them.
4. **Sort.** Sort the points by s_norm. Requested order is not used for integration.
5. **Ties.** Points whose adjacent sorted s_norm values differ by at most 1e-12 form a tie group. The
   tolerance is **absolute on s_norm**, not relative. The group becomes one point at the unweighted mean of
   its y values. The original points are retained in the record.
6. **The upper boundary.** A point within 1e-12 of 1 is treated as 1.
7. **Points above 1.** Points above 1 are retained in records and figures, including undefined observations
   and their reasons. They **must not affect the integral**: they are excluded from it, and an undefined
   point above 1 does not make AUDC uncomputable. Their count is reported.
8. **Undefined points.** If any level at or below 1 is undefined, AUDC is **uncomputable** and the reason
   is recorded. An undefined value is never interpolated, extrapolated or replaced.
9. **Integrate.** Apply the §3.4 trapezoidal rule to the remaining points over [0, 1].

### Consequences

- **No boundary interpolation.** The highest requested level has s_norm exactly 1 by the §3.3 formula. If
  its value is defined, the integral ends on a measured point. If it is undefined, AUDC is uncomputable. So
  no interpolation at x = 1 is ever needed.
- **Worked example.** For requested-order points x = (0, 5/3, 1), y = (0, 0.9, 0.6), the sorted points at or
  below 1 are (0, 0) and (1, 0.6), so AUDC = 0.30. The point at x = 5/3 stays in the record and figures.

### Uncomputable and undefined cases (frozen)

- s_max = s_0: the normalized curve and AUDC are uncomputable for that scene/fold (§3.3).
- ADE₀ = 0: D is undefined and not imputed. The absolute difference is reported, and the scene/fold stays
  in the record but contributes no value to the relative aggregate, with valid and undefined counts
  reported (§3.4).
- Family-level AUDC follows §5.4: the mean of valid predictor-level values, uncomputable below two.

## 2. ADE₀ population (project decision)

- **Primary:** ADE₀ is computed on **all eligible windows**.
- **Sensitivity check:** a paired baseline on the same predicted windows is retained alongside it.
- This is a methodological decision, **not a v4.15 requirement**. §3.4 says only "clean-input ADE". The
  choice changes how degradation reads when prediction coverage changes, because a window subset can move
  ADE with inputs unchanged. Coverage and ADE are always reported together.
- Under P-15, "eligible" means eligible in the population being analysed (section 4).

**B2 with no predictions (project decision).** The record is retained. If that leaves a level at or below 1
undefined, AUDC is uncomputable. This can leave family B with few valid values, and that is reported rather
than repaired.

## 3. Monotonicity (plan qualification)

The Plan's Phase 3 gate says that for every condition the achieved-severity curve is monotone in requested
severity. v4.15 §11.2 lists injector monotonicity as "expected", and the §14.1 P4 gate does not include it.
The Plan PDF is not edited; this section qualifies how its gate is evaluated and reported.

| Condition | Check | Evaluated |
|---|---|---|
| A | Strict increase | Per scene and seed, and on the seed mean |
| B1, B2, C | Non-decreasing | Per scene and seed, and on the seed mean |
| D | Diagnostic, not a pass/fail check | Per scene and seed, and on the seed mean |

- **A.** The check needs at least one observation, and must verify that achieved severity **actually
  increases across the non-zero requested levels**. A degenerate zero-noise realization is never silently
  counted as a pass.
- **D.** Violations are **diagnostic findings requiring investigation, not automatic failures**. Each is
  reported with its triggers, suppressed triggers and exposure. The injector is never silently modified to
  remove one.
- **Reporting.** D's result must not be described as a strict monotonicity pass.

**Why D differs.** The cooldown required by §3.5 means an early event can lock tracks that would otherwise
have produced more events later. Take four tracks where only (1, 2) are close for k timesteps, and later
only (1, 3) and (2, 4) are close for m timesteps. With q = 1 − (1 − p)^(2k) and r = 1 − (1 − p)^(2m), the
expected number of events is q + (1 − q)·2r. For k = 4 and m = 8 this is 1.271, 1.232 and 1.158 at
p = 0.10, 0.15 and 0.20. So achieved severity can fall with requested severity, even in expectation, inside
the configured grid. These figures are derived by hand from the injector's rules.

## 4. P-15 (project decision)

History lengths, horizon, predictors, conditions, retraining and folds are frozen in §2.2.5. The following
settles what §2.2.5 leaves open.

**Populations.**
- **Primary:** per-history eligible populations, as set by the active history length under §3.2.
- **Mandatory companion:** the common population of anchors eligible at H = 12.
- Coverage and window counts are reported for every history.
- If the two populations give different minimum histories, both are reported.

**Clean metrics.** Clean ADE and FDE are calculated on the eligible population of the analysis: the
per-history population for the primary analysis, and the H = 12 common population for the companion. Within
a population, the clean criteria are shared across the five degradation conditions.

**AUDC criterion.** The criterion is applied separately to each of A, B1, B2, C and D. This can give up to
five minimum histories per predictor, reported as a table with no single summary.

**Uncomputable AUDC.**
- An uncomputable history **fails the AUDC criterion** for that condition, with the reason reported.
- AUDC_best is calculated **only from computable histories**.
- If no history has a computable AUDC, that condition's minimum history is **uncomputable**.
- No substitute value is assigned.

This is distinct from the frozen §2.2.5 trade-off outcome, where computable histories exist but none meets
all three conditions.

## 5. P-10 / H8

**Not implemented.** P-10 stays unimplemented until the open items below are resolved and documented.
`h8_generation_attempts` stays null in `configs/base.yaml`, with no value chosen.

**Retained (frozen):** four predictors (the §5.4 default set), six pairs (A×B, A×C, A×D, B×C, B×D, C×D),
family B represented by B2, and 36 cells per pair (the Cartesian product of the two six-level grids).

**Maximum-statistic scope (project decision).** One maximum statistic spans all four predictors, all six
pairs and all 25 non-zero cells per pair: **600 components** (4 × 6 × 25). This is the conservative choice
for multiplicity control.

**Zero-component cells (project decision, needs amendment).** The 11 cells per pair with a zero component are
checked separately as integrity checks and are outside the maximum statistic. The §2.3.4 steps refer to every
cell of the complete grid, so this exclusion must be written into the protocol before P-10 can be
research-grade.

**Limited resampling diversity.** With five scenes, resampling whole scenes with replacement gives only
**126 distinct resamples**. This is a genuine limitation, and H8 may have little chance of detecting an
effect. It is **not a reason to weaken multiplicity control after seeing results**.

**Open (not decided here):**
- target severity definition across seeds;
- attempt reseeding and the value of `h8_generation_attempts`;
- C×D generation and merging mechanics;
- bootstrap constants and degenerate-resample handling;
- handling of achieved values outside the single-mode curve range;
- storage of two achieved severities and of unrealized cells.

## 6. v4.16 conflict (open)

v4.15 Part 8 and §2.3.4 fix all six pairs. The v4.16 draft lists P-10′ as "selected-pair interactions". This
conflict is **unresolved**. Neither interpretation is adopted here, and neither may be silently written into
a frozen specification.

The draft on disk is revision 2, and later revisions exist only in session chat, so this must be rechecked
against the consolidated text.

## 7. For v4.16 consideration (not adopted)

- The AUDC procedure in section 1, including §3.3's conflict between its formula (highest *requested* level)
  and its prose (highest *achieved* point) for non-monotone curves.
- The ADE₀ population in section 2.
- Monotonicity strictness: the draft's S4 gate repeats "achieved-severity monotonicity" without it.
- The P-10 maximum-statistic scope, zero-component cells and the open items in section 5.
- Proposal 0003 (`association_mechanism`), which remains pending.

## Unchanged

- v4.15 Phase 1 remains BLOCKED; v4.16 remains unfrozen; no gate is claimed passed.
- P-8 and P-11 remain blocked. No five-fold real-scene result exists, ETH and HOTEL remain descriptive and
  local-only, and synthetic learned-model results remain engineering-only (decision 0001, F).
- These research choices remain unresolved: `moving_dataset`, `trackers_evaluated_set`, `tracker_params`,
  `h8_generation_attempts`, `discontinuity_rejection_v_max_mps`.
