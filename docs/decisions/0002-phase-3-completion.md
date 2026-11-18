# Decision 0002: Phase 3 completion (input validation, severity normalization)

- **Date:** 2026-10-31
- **Decided by:** the project owner, on the Phase 3 audit.
- **Governing specification:** v4.15 (unchanged). v4.16 is not frozen; nothing here amends either.
- **Recorded before any research sweep exists.** No result informed these choices.

## Input validation in the injectors

Three guards were added to the accepted U7 and U8 injectors. They reject invalid input only.

| Guard | Location | Rejects |
|---|---|---|
| Finite sigma | `inject/trajectory.py`, `localization_noise` | infinite or NaN sigma |
| Frame order | `inject/trajectory.py`, `_check` (all injectors) | duplicate or non-increasing frames |
| Frame step | `inject/identity_switch.py`, `proximity`, `apply_switches` | a `frame_step` that is not a positive integer |

Valid-input behaviour is unchanged. Digests of the synthetic A, B1, B2, C and D sweeps, and of the
injector outputs on the ETH and HOTEL tracks, are identical before and after the guards.

## Severity normalization (§3.3)

§3.3 fixes the formula and its per-scene/fold, per-family scope, but does not say how seeds are handled.
This decision settles that:

1. Achieved severity is averaged across seeds, within each scene/fold and requested level.
2. s_0 and s_max are taken from those seed-aggregated values.
3. s_norm = (s_achieved − s_0) / (s_max − s_0) is applied to the seed-aggregated values only.
4. **Normalization is never done independently per seed.**
5. Raw achieved values are preserved, and seed-level spread is reported separately.
6. If s_max = s_0, normalized severity and AUDC are reported as **uncomputable**, not zero (§3.3).

This follows the aggregation order of §7.4: seeds first, then scenes.

### Implementation-level details (`inject/severity.py`)

These are not research parameters. They are recorded so that nothing is silent.

- **Not clipped.** §3.3 takes s_max at the highest *requested* level. If the seed-mean curve is not
  monotone, an intermediate level can give s_norm above 1. The value is kept and
  `within_unit_interval` is false, because §3.4 integrates AUDC over [0, 1] without extrapolation. How
  Phase 4 treats such a curve is still open.
- **Undefined severity.** If achieved severity is undefined (`None`) at every level and seed, for
  example a scene with no eligible track frames, the result is uncomputable. A table that is only
  partly undefined raises an error, because every denominator comes from the clean base.
- **s_max below s_0** raises an error: the highest level cannot fall below the zero-injection condition.
- **Seed spread:** standard deviation with n − 1, plus minimum and maximum. The standard deviation is
  undefined for one seed.
- **AUDC is not computed here.** Only its computable or uncomputable status is reported.
