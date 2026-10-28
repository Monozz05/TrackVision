from pathlib import Path

import hashlib

import pytest
import yaml

from trackvision import config

REPO_ROOT = config.REPO_ROOT
BASE_PATH = REPO_ROOT / "configs" / "base.yaml"
SMOKE_PATH = REPO_ROOT / "configs" / "experiments" / "smoke.yaml"
_SPEC_META = yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))["meta"]["spec"]
SPEC_PATH = REPO_ROOT / _SPEC_META["path"]

MISSING_SPEC_MESSAGE = (
    "TrackVision tests need the governing specification, which is not redistributed in this "
    "repository: {problem}. Place the file at {path} and see docs/spec/README.md."
)


def pytest_configure(config):  # noqa: ARG001 - pytest hook signature
    """Stop the whole run, before collection, if the governing specification is unavailable.

    Validation is strict: nearly every test depends on the verified specification, so a clone
    without it fails at startup with one clear message instead of skipping tests.
    """
    if not SPEC_PATH.is_file():
        problem = "the file is missing"
    elif hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest() != _SPEC_META["sha256"]:
        problem = "the file does not match the SHA-256 recorded in configs/base.yaml"
    else:
        return
    rel = SPEC_PATH.relative_to(REPO_ROOT).as_posix()
    pytest.exit(MISSING_SPEC_MESSAGE.format(problem=problem, path=rel),
                returncode=pytest.ExitCode.USAGE_ERROR)


@pytest.fixture
def base_dict() -> dict:
    """A fresh, mutable copy of base.yaml for building modified configurations."""
    return yaml.safe_load(BASE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Write a mapping (or raw text) to a temporary YAML file and return its path."""
    counter = iter(range(10_000))

    def _write(content, name: str | None = None) -> Path:
        path = tmp_path / (name or f"cfg_{next(counter)}.yaml")
        text = content if isinstance(content, str) else yaml.safe_dump(content, sort_keys=False)
        path.write_text(text, encoding="utf-8")
        return path

    return _write


@pytest.fixture
def smoke_config() -> config.Config:
    result = config.validate(SMOKE_PATH)
    assert result.status == config.RUNNABLE, result.errors
    return result.config
