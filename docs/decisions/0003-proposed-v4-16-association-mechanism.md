# Proposal 0003: add `association_mechanism` to the experiment record (for v4.16)

- **Status: PROPOSED.** Not approved for freezing, not adopted and not governing.
- **Date:** 2026-10-31
- **Target:** v4.16, as an amendment to the §11.3 schema. Provisional label: AMEND-2.
- **v4.15 is unchanged and stays frozen.** This proposal does not modify it, does not freeze v4.16, and
  changes no record schema or configuration.
- **Caveat:** the v4.16 draft on disk is revision 2. Later revisions exist only in session chat, so the
  label and section numbers below must be checked against the consolidated text.

## The problem

- v4.15 §3.7 says **every experiment** records:
  `association_mechanism = { real_tracker_iou | synthetic_gt_lineage }`.
- The §11.3 record schema has no such field. The nearest is `results.collision.association_type`, which
  applies only to collision experiments (§3.8).
- `src/trackvision/experiments/record.py` mirrors §11.3 exactly and rejects unknown keys, and
  `tests/unit/test_record.py` checks it against the specification text. So the value §3.7 requires
  cannot be recorded today.
- The v4.16 draft (revision 2, §5.2 rule 4) already says every record carries this field, but does not
  add it to the schema block.

## Proposed change

### Schema (§11.3, EXPERIMENT block)

Add one line after `anchoring_type`:

```
  association_mechanism {synthetic_gt_lineage | real_tracker_iou}
```

| Value | Meaning | Applies to |
|---|---|---|
| `synthetic_gt_lineage` | A trajectory keeps its originating GT identity through the injector's lineage map. No box association is performed or permitted | Clean ground-truth baselines and synthetic corruption of ground truth |
| `real_tracker_iou` | Tracker box to GT box, one-to-one Hungarian assignment at `TRACKER_GT_ASSOC_IOU_THRESHOLD`, then GT identity carried forward | Any real tracker output, including corruption of tracker output declared on that base (§5.2) |

### Validation rules

1. **Required** when `ground_truth_available` is true.
2. **Null** when `ground_truth_available` is false, since no association against ground truth exists
   (§11.3).
3. **Category check:** a record with a `tracker` must use `real_tracker_iou`, and a record without one
   must use `synthetic_gt_lineage` (§3.7 calls the opposite a category error).
4. **Consistency:** where `results.collision.association_type` is set, it must equal
   `association_mechanism`.

Rule 3's use of the `tracker` field is an implementation-level reading and needs review. In particular,
v4.16's optional O3 arm feeds GT boxes into a real tracker and is intended to use `real_tracker_iou`,
which rule 3 handles, but the arms have not been checked one by one.

### Relationship to `association_type`

`results.collision.association_type` stays as it is. It describes the collision result, and rule 4 keeps
the two fields consistent. Removing it would be a separate change.

## Implementation readiness

`src/trackvision/inject/association.py` already provides:

- the two values, spelled as in §3.7;
- `association_mechanism(trajectory_base)` and `association_fields(trajectory_base)`, which produce the
  field for a record;
- `association_errors(...)`, which implements rules 1 to 4.

The lineage maps themselves already exist: `Fragmentation.origin` and `IdentitySwitch.occupant`, while
A and B never change identity.

**Nothing writes the field into a record today.** On adoption, the remaining work is:

1. Add the field to `TOP_LEVEL_FIELDS` and `PERMITTED_VALUES` in `experiments/record.py`.
2. Call `association_errors` from `validate_record`.
3. Update `tests/unit/test_record.py`, which checks the schema against the governing specification
   text, and point the configuration at the new specification hash.

No existing record needs back-filling: no runner writes §11.3 records yet.

## Decisions needed

1. Approve or change the field, its placement and rules 1 to 4.
2. Confirm rule 3 against each v4.16 arm once the text is consolidated.
3. Include this amendment in the v4.16 freeze, as a separate decision from approving the proposal.
