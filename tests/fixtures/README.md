# tests/fixtures/

Small synthetic data with analytically known answers, so code is tested against something other
than itself (spec v4.15 §11.2).

**What Git tracks here:** ordinary source and data files, plus small deliberate data-like fixtures
(`*.log`, model-weight formats such as `*.pt`, and video formats such as `*.mp4`). Those types are
ignored everywhere else in the repository and are re-included only here. Every other ignore rule
still applies inside this folder, so caches, credentials (`.env*`), environments, editor and OS
metadata and build output are never tracked here either.

`synthetic.py` builds its geometry from the frozen sampling protocol in `configs/base.yaml`. It
provides:

- **A constant-velocity walker**, one point per window step. Its known step length, speed and
  heading come from the fixture's own walker speed, which is a property of the fixture, not a
  protocol constant.
- **A stationary point.**
- **A copy of the walker offset by exactly 1 m.** It's reserved for the Phase 2 metric tests:
  against it, ADE = FDE = RMSE = 1.0.

## Size policy (repository rule, not a research parameter)

- Each file here must be **at most 1 MB**. `tests/unit/test_fixture_policy.py` and
  `scripts/check_staged.py` enforce this.
- A larger fixture needs explicit approval, recorded as one line in `LARGE_FIXTURES_APPROVED.txt`:
  `path | size_bytes | sha256 | approver | date | reason`
- The approval covers exactly that content. Changing the file needs a new approval line.
