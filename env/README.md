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
| `../pyproject.toml` | The **intended constraints**: Python `>=3.12,<3.13`, runtime `numpy>=2,<3` and `pyyaml>=6,<7`, the `[test]` extra `pytest>=8`, and the build backend `setuptools>=64`. The only place version ranges are written. |
| `requirements-lock-win_amd64-py312.txt` | The **exact record** of the tested environment: `pip freeze --all --exclude-editable` of `.venv`, under a generated header. |

## Set up (PowerShell, from the repository root)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r env\requirements-lock-win_amd64-py312.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest
```

The commands call the venv's interpreter directly, so they need neither activation scripts nor
any change to PowerShell's execution policy. `--no-deps` makes the lock file the only source of
dependencies, and `--no-build-isolation` builds the editable install with the setuptools version
recorded in the lock.

## What the lock file does and does not guarantee

It was generated from the environment the Phase 0A tests passed in, and a throwaway venv built
from it alone reproduced the same pins and passed `pip check`. Its limits:

- **No artifact hashes.** It pins versions, not the content of the downloaded files. pip cannot
  check hashes against it (`--require-hashes`), and it relies on PyPI never changing a published
  file for a given version. Adding hashes would need a tool that resolves the wheel files again,
  which would no longer be a record of the tested environment, so this is left out on purpose.
- **Windows x86-64 and CPython 3.12 only.** NumPy and PyYAML ship platform-specific wheels. On
  another operating system, architecture or Python minor version, pip may choose different builds
  or fail.
- **The interpreter is not locked.** The header records CPython 3.12.10. A rebuild uses whatever
  `py -3.12` finds on that machine.
- **It needs PyPI.** It assumes PyPI stays reachable and that no newer wheel build is added for
  the same version; that is rare, but possible.
- **Smart App Control.** A rebuild on a machine with Smart App Control enforced can still be
  blocked if Windows does not trust a particular file. Check the Code Integrity log
  (`Microsoft-Windows-CodeIntegrity/Operational`) if a process exits with `0xC0E90002`.

## Updating the lock

Add or change constraints in `pyproject.toml` only. Install into `.venv`, run `pip check` and the
full test suite, and only then regenerate the lock from that environment. Never edit the lock by
hand. Phase 0B and later dependencies (PyTorch, Ultralytics, TrackEval and others) are added in the
phase that needs them, and each must be checked against Smart App Control on this machine first.
