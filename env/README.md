# Python environment

TrackVision uses a project virtual environment, `.venv/`, built on the python.org CPython 3.12
interpreter. The `.venv/` folder itself is never committed.

## Why not conda (approved deviation, 2026-10-29)

The approved Phase 0A plan originally used a Miniforge conda environment. On this machine,
Windows **Smart App Control** is in enforce mode and blocked the conda-forge interpreter: Code
Integrity events 3033/3077 report that `python312.dll` did not meet the signing requirements,
and every process in that environment failed to start. A reversible feasibility test then
showed that a `.venv` on the python.org interpreter runs Python, NumPy (core, OpenBLAS linear
algebra, FFT, random) and PyYAML (including the libyaml C loader) with no Code Integrity
events. The environment strategy was switched to `.venv` with approval, and Smart App Control
remains enabled.

This changes only how the environment is built. No research parameter changed.

## Files

| File | Role |
|---|---|
| `../pyproject.toml` | The **intended constraints**: Python `>=3.12,<3.13`, runtime `numpy>=2,<3` and `pyyaml>=6,<7`, the `[test]` extra `pytest>=8`, and the build backend `setuptools>=64`. The only place version ranges are written. **It does not yet declare PyTorch** (see below); declaring it there is a separate, pending decision. |
| `requirements-platform-win_amd64-py312.txt` | The packages the read-only results API adds to `.venv` (FastAPI, uvicorn, httpx and their dependencies). Kept apart so the research lock stays unchanged. The `[platform]` extra in `pyproject.toml` holds the ranges. |
| `requirements-lock-win_amd64-py312.txt` | The **exact record** of the tested environment: `pip freeze --all --exclude-editable` of `.venv`, under a generated header, with the **SHA-256 of the wheel** each package was installed from. |

## Set up (PowerShell, from the repository root)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes --index-url https://pypi.org/simple `
    --extra-index-url https://download.pytorch.org/whl/cpu -r env\requirements-lock-win_amd64-py312.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "import torch; assert torch.version.cuda is None and not torch.backends.cuda.is_built(); print(torch.__version__)"
.venv\Scripts\python.exe -m pytest
```

The commands call the venv's interpreter directly, so they need neither activation scripts nor
any change to PowerShell's execution policy. `--no-deps` makes the lock file the only source of
dependencies, and `--no-build-isolation` builds the editable install with the setuptools version
recorded in the lock. `--require-hashes` makes pip refuse any file whose SHA-256 differs from the
lock, which is what makes the second index safe to add.

## PyTorch: CPU-only (added in P2-e, 2026-10-31)

- **Build:** `torch==2.13.0+cpu`, from the official PyTorch CPU index
  (`https://download.pytorch.org/whl/cpu`). The `+cpu` local version exists only there, so the
  setup command adds that index. No CUDA package, runtime or DLL is installed, and none is needed:
  the learned predictors are small recurrent networks trained on the CPU.
- **Why CPU:** the GPU (CUDA) remains the separate G1 gate.
- **Approval:** docs/decisions/0001-p2d-protocol-decisions.md (E, G).
- **Hash verification:**
  - The wheel's SHA-256 (`a8b450c1…1e6f`) and those of its dependencies were checked against the
    official indexes: the PyTorch CPU index for torch, PyPI's JSON API for everything else.
  - The wheels were first tested in a disposable environment.
  - Every file installed in `.venv` was compared with its wheel's RECORD.
- **What the install changed:** it only added packages: torch, filelock, fsspec, Jinja2,
  MarkupSafe, mpmath, networkx, sympy and typing_extensions. numpy 2.5.4 and setuptools 84.2.1 were
  already at the required versions, and no existing package was upgraded.
- **Verify a CPU-only build:**
  - `torch.version.cuda` is `None`;
  - `torch.backends.cuda.is_built()` and `torch.cuda.is_available()` are `False`;
  - `torch.__config__.show()` reports `USE_CUDA=0`.
- **Smart App Control (enforcing on this machine):** no Code Integrity event was logged for
  installing, importing or training with this build.
- **Determinism:** training is deterministic only with `torch.use_deterministic_algorithms(True)`,
  which the training code sets, and on the same thread count. The thread count is recorded with
  every checkpoint.

## What the lock file does and does not guarantee

It was generated from the tested environment. A throwaway venv built from it alone, offline from the
verified wheels with `--require-hashes`, reproduced exactly the same pins and passed `pip check`
(P2-e). Its limits:

- **Hashes cover one build per package.** Each pin carries the hash of the single wheel actually
  installed. pip can therefore verify that file, but cannot use another platform's build of the
  same version.
- **Windows x86-64 and CPython 3.12 only.** NumPy and PyYAML ship platform-specific wheels. On
  another operating system, architecture or Python minor version, pip may choose different builds
  or fail.
- **The interpreter is not locked.** The header records CPython 3.12.10. A rebuild uses whatever
  `py -3.12` finds on that machine.
- **It needs PyPI and the PyTorch CPU index** to stay reachable and to keep serving the hashed
  files.
- **Smart App Control.** A rebuild on a machine with Smart App Control enforced can still be
  blocked if Windows does not trust a particular file. Check the Code Integrity log
  (`Microsoft-Windows-CodeIntegrity/Operational`) if a process exits with `0xC0E90002`.

## Updating the lock

Add or change constraints in `pyproject.toml` only. Install into `.venv`, run `pip check` and the
full test suite, and only then regenerate the lock from that environment, joining each pin to its
verified wheel hash. Never edit the lock by hand. Later dependencies (Ultralytics and others) are
added in the phase that needs them, and each must be checked against Smart App Control on this
machine first.

The P2-e regeneration is the one exception to "constraints first": PyTorch was installed and locked
with approval, but `pyproject.toml` was not part of that approval and does not declare it yet.
TrackEval is not part of this environment: it runs in its own isolated Python 3.11 environment
outside the repository (M2).
