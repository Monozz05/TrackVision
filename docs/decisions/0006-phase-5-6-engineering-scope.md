# Decision 0006: Phase 5 and Phase 6 limited engineering scope

- **Date:** 2026-11-01
- **Decided by:** the project owner, on the Phase 5 and Phase 6 read-only audit.
- **Governing specification:** v4.15 (unchanged). v4.16 is not frozen; nothing here amends either.
- **Timing, stated plainly.** These choices were made after Phase 4 synthetic and descriptive output had been
  inspected, including stored family AUDC values. No research result exists, so none was steered, but this record
  must not be read as predeclared.

**Status labels.** Everything under *Approved scope* is a *project decision*, not a v4.15 requirement or amendment.
*Open* is unresolved. Nothing here changes `configs/base.yaml`.

## Approved scope

1. **No research gate closes.** Phase 5 and Phase 6 work in this scope closes no research gate, and none is passed.
2. **Engineering labels.** Synthetic E4, E5 and E6 results are engineering-only.
3. **E1** is limited MOT17 feasibility using public detections and the existing IoU tracker on seven training
   sequences, second halves. It is not P-1 and not a MOT17 benchmark.
4. **Detection sets.** FRCNN is primary. DPM and SDP are optional additional runs, because they need no new code
   path, and are never presented as primary.
5. **Cadence.** MOT17 is evaluated at native frame cadence. No 14 fps resampling rule is adopted.
6. **E2** detection metrics are project diagnostics, not frozen P-1 outputs.
7. **E3** localization severity is in pixels only and is never converted to metres. E3 is not P-9.
8. **E4 and E5** use synthetic ground-truth lineage (U1 `ped_id`) and are labelled synthetic engineering. They are
   not P-9 or P-13, and `real_tracker_iou` is never used for them. Here "lineage" names where the ground-truth
   identities come from. It is not §3.7's `synthetic_gt_lineage`, which is defined through an injector's lineage map:
   E4 has no injector and matches tracker boxes to generated boxes by IoU. See the association item under *Open*.
9. **E6** is an `engineering_dry_run` and does not consume the one-time research selection of §2.2.3.
10. **Mitigation target.** The dominant mode is the primary mitigation target. Additional modes, if any, are
    exploratory and separately labelled.
11. **Across-scene family AUDC** may be computed descriptively: valid predictor AUDCs within each scene, the frozen
    minimum of two valid predictors within that scene, then aggregation across eligible scenes, with counts and
    undefined reasons preserved. This is not research-grade five-scene evidence. This lifts the deferral in
    decision 0005 for this narrow, descriptive use only. ETH and HOTEL are never pooled.
12. **E7** inverse mapping may use existing stored curves.
13. **E8** implements threshold-crossing calculations only. It does not implement runtime adaptive model selection.
14. **Deferred:** E9, E10, §9.1, §9.2, full §9.3 and recovery calculations that depend on unresolved protocol
    decisions.
15. **Proposal 0003 stays unadopted.** The §11.3 schema and `experiments/record.py` do not gain a field. The §3.7
    requirement stays in force independently of that proposal: every experiment records its `association_mechanism`,
    from the two frozen values only. In this scope it is recorded as run-level metadata, outside the §11.3 schema,
    and only where the run directly evidences a value. No third value exists, and a value that cannot be established
    is recorded as null with its blocker, never left out.
16. **No ETH/UCY ground-truth boxes are derived.**

## Implementation choices within this scope (not protocol rules)

These were needed to compute diagnostics. Each is recorded with the results and none is a §3.7 definition.

- **E3 gap limits: an engineering diagnostic, not a §3.7 implementation.** §3.7 defines `FRAG_GAP_MAX` and
  `SWITCH_GAP_MAX` in timesteps of the 0.4 s protocol grid, and E3 does not run on that grid. For diagnostics only,
  E3 multiplies each limit by 0.4 s and the sequence's frame rate, then rounds down to whole frames: 60 and 12 frames
  at 30 fps, 50 and 10 at 25 fps, 28 and 5 at 14 fps. This is not a resampling rule. E3 counts are not §3.7 achieved
  severities, are not comparable across frame rates, and must not be projected onto a degradation curve.
- **Identity exchanges.** A `SWITCH_GAP_MAX` of 1 is read as at most one missing timestep between the two
  observations, following §3.7's "single-timestep association gap".
- **E3 identity-exchange denominator.** §3.7's denominator needs `D_MAX` in metres, which MOT17 lacks, so that rate is
  recorded as uncomputable. Exchanges are given per 100 evaluable ground-truth observations, under a separate name.
- **E7 axis.** Raw achieved severity, sorted ascending, using the same 1e-12 tie rule as decision 0004.
- **E6 boundary: a project-level numerical convention.** §2.2.3 says "less than 0.10" and sets no tolerance. So that
  binary rounding cannot decide co-dominance, E6 treats a top-two gap within 1e-12 of 0.10 as equal to 0.10, which is
  not co-dominant. The tolerance is not part of §2.2.3 and amends nothing in v4.15. It applies to the engineering
  dry-run only; the research selection would need it approved separately. It did not affect the recorded dry-run,
  whose gap is 3.004.
- **Association provenance.** Stored Phase 5 and 6 records are not edited. Each run gains an
  `association_provenance.json` sidecar bound to its `results.json` by SHA-256, and new runs carry the same field.
  E1 and E3 record `real_tracker_iou`, as a statement about the mechanism only: no claim of P-1 or P-9 follows. E6, E7
  and E8 record `synthetic_gt_lineage` for their ground-truth input curves. E4 and E5 record null, unresolved.

## Open (unresolved, not decided here)

- 14 fps resampling of MOT17.
- Recovery clean-baseline population.
- Primary severity range (§2.2.4, §2.3.7, §2.3.10, §2.3.11).
- Adoption of proposal 0003 into the §11.3 schema.
- The `association_mechanism` of E4 and E5, which stays null. E4's mechanism is box IoU, so `synthetic_gt_lineage` would
  be false, and item 8 forbids the only other frozen value for synthetic output. Only the project owner can resolve
  this: no third value may be added.
- §9.1 training levels and validation corruption.
- §9.2 `v_max`, fixed-length learned-model input handling and the false-positive denominator.
- §9.3 runtime selection and the misclassification definition.
- Collision population and window rules (§3.8).
- Bootstrap constants for P-13.
- Video permissions and calibration.
- Five-scene static research availability.
- ETH/UCY ground-truth boxes, which §3.7 assumes and the annotations do not supply.
- §2.2.3 timing: selection follows "completion of P-4–P-8", while P-8 is blocked.

## Unchanged

- v4.15 Phase 1 remains BLOCKED; v4.16 remains unfrozen; no gate is claimed passed.
- P-1, P-9, P-13 and P-14 remain research-blocked, as do H7, RQ3 and RQ8.
- P-8, P-10, P-11 and H8 remain blocked. `h8_generation_attempts` remains null.
- MOT17 trajectories are never used for prediction (§6.1.1).
