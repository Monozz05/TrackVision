# Decision 0005: Phase 4 P-15 corrections and minimal fixes

- **Date:** 2026-10-31
- **Decided by:** the project owner, on the Phase 4 targeted P-15 methodology audit.
- **Governing specification:** v4.15 (unchanged). v4.16 is not frozen; nothing here amends either.
- **Refines decision 0004, section 4,** which did not define what makes a history computable, or how the
  companion population is trained.
- **Timing, stated plainly.** These choices were made after synthetic engineering output had been inspected. No
  research result exists, so none was steered, but this record must not be read as predeclared.

**Status labels.** *Project decision* is not stated by v4.15 and is decided here. *Implementation assumption* is
a working choice that still needs reconciliation with the specification or v4.16. *Open* is unresolved. Nothing
below is a frozen v4.15 requirement.

## 1. P-15 partial-fold AUDC (project decision)

For P-15, a history qualifies for the AUDC criterion only if **all folds of the analysis population have a valid
AUDC**. With five scenes, that is all five.

1. Every valid scene/fold AUDC is calculated and retained. The scene AUDC procedure of decision 0004 does not
   change.
2. Partial-fold means, valid counts, undefined counts and reasons continue to be reported.
3. A history with fewer valid folds than the population has is marked `partial_folds`.
4. A partial-fold history does not qualify for the AUDC criterion.
5. Partial-fold values are excluded from AUDC_best.
6. If no history has every fold valid, that condition's minimum history is **uncomputable**.
7. The clean ADE and FDE criteria and all other P-15 rules are unchanged.
8. The rule applies independently to each analysis population.

**Why.** §7.4 allows aggregating over valid scene/fold results, so a partial mean is a legitimate aggregate. It
is not a legitimate comparison across histories: §2.2.5 says "the same five leave-one-scene-out folds" and §7.3
pairs fold-level results by held-out scene. Comparing a one-scene mean with a five-scene mean compares scenes as
well as histories.

## 2. Common H = 12 population (project decision)

Two distinct companions exist. They must never be merged or described as the same thing.

| Name | Key | Training and validation | Evaluation | Isolates history length? |
|---|---|---|---|---|
| Common evaluation population — training and validation remain per-history | `common_h12` | Per-history | H = 12-eligible anchors | CV and CA only. **Not LSTM or GRU** |
| Revised common companion | `common_h12_trained` | H = 12-eligible anchors | H = 12-eligible anchors | Yes, within this design |

- `common_h12` is kept as a **supplementary descriptive** result. Its key is unchanged because stored records
  carry it; its label says what it is.
- The revised companion uses the H = 12-eligible anchor IDs for training, validation and evaluation at every
  history length, on the same five folds, three seeds, histories 4, 6, 8, 10 and 12, and horizon 12. Each history
  takes its last H observations from the same anchors. Architecture, training settings, early stopping and
  checkpoint selection are unchanged.
- The primary per-history analysis is unchanged.
- The revised companion is a separate run with its own directory, manifest and digests.
- Before any training, `common_preflight` must show: consistent anchors across histories, disjoint folds and
  scenes, anchors independent of positions, the six leakage checks on the restricted split, and that only the
  input length differs. A failure stops the run with nothing written.
- H = 12 models may be reused only after their checkpoint hash, state hash, configuration, settings and the hash
  of every training, validation and test scene's windows are shown to equal the revised population's.

**Consequence.** The revised companion trains on fewer windows than the per-history analysis at shorter histories
(7–29% fewer at the training level). It answers a different question and does not replace the primary result.

## 3. B2 and C interpretation (project decision)

P-15 tables and figures report, for B2 and C: ordinary AUDC, paired AUDC, prediction coverage at each severity
level, predicted-window counts, and valid and undefined fold counts.

**Conditional-error AUDC for B2 and C does not establish history sufficiency when the surviving predicted inputs
are unchanged.** A negative AUDC is not evidence of improved robustness without examining the population and
coverage.

No window-count threshold or minimum-coverage criterion is added.

## 4. H4 engineering check (project decision)

H4 names ADE and FDE. The engineering check evaluates both, separately, with the existing calculation and the
frozen thresholds, and records whether they agree. A disagreement is reported and not resolved. No new threshold
exists, and the H4 research verdict stays `not_evaluable`.

## 5. H3 and H6 zero-level handling (implementation assumption, open)

The existing calculation over non-zero grid levels is retained as a documented **implementation assumption**.
§2.3.7 says "primary range" and §2.3.9 says "range"; neither settles the zero level. The zero-level cell is not
silently included, and the non-zero reading is not a frozen requirement. The exact interpretation **needs
reconciliation with the specification or v4.16**. Thresholds are unchanged, verdicts stay `not_evaluable`, and the
engineering checks do not establish either hypothesis.

## 6. Minimal correctness fixes

1. If severity normalization fails, family AUDC emits an explicit uncomputable result with its reason and counts.
2. `sweep run --experiment` selects that experiment instead of running the whole eligible scope.
3. A missing `--out` is an argument-validation error instead of a raw `TypeError`.

## Deferred

- Across-scene family AUDC, until RQ2 is in scope.
- Bootstrap confidence intervals, until constants are approved.
- All P-10 protocol questions (decision 0004, sections 5 and 6).

## For v4.16 consideration (not adopted)

- The all-folds rule in section 1 and the two companions in section 2.
- The zero-level interpretation in section 5.
- Whether §3.4's undefined-value wording, written for zero baselines, should cover a fold with no predicted
  windows.

## Unchanged

- v4.15 Phase 1 remains BLOCKED; v4.16 remains unfrozen; no gate is claimed passed.
- Synthetic results remain engineering feasibility only. ETH and HOTEL remain descriptive and local-only.
- P-8, P-10 and P-11 remain blocked. `h8_generation_attempts` remains null.
