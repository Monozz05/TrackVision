"""Reproducibility helpers: seeding, experiment IDs and runtime provenance.

Identity and provenance are kept apart on purpose. `make_experiment_id` depends only on the
experiment name, the configuration hash and the seed. `git_state` and `environment_info`
describe the run for the manifest and never feed into an ID or a hash.
"""

from __future__ import annotations

import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's `random` and return a numpy Generator for the given seed.

    A torch hook is added once torch enters the environment (Phase 0B).
    """
    random.seed(seed)
    return np.random.default_rng(seed)


def make_experiment_id(name: str, config_hash: str, seed: int | None) -> str:
    """Deterministic experiment ID: `<name>-<first 10 hex of config hash>-s<seed|na>`."""
    seed_part = "na" if seed is None else str(seed)
    return f"{name}-{config_hash[:10]}-s{seed_part}"


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(repo_root), *args],
                              capture_output=True, text=True, check=False)
    except OSError:
        return None


def git_state(repo_root: Path) -> dict:
    """Current commit and dirty flag; `sha` is None before the first commit."""
    head = _git(repo_root, "rev-parse", "--verify", "--quiet", "HEAD")
    status = _git(repo_root, "status", "--porcelain")
    return {
        "sha": head.stdout.strip() if head is not None and head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status is not None and status.returncode == 0 else None,
    }


def environment_info() -> dict:
    """Software and platform versions for the run manifest."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "gpu": "not checked (Phase 0B)",
    }
