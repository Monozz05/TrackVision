"""Probe .gitignore with paths that must and must not be ignored (no files are created)."""

import shutil
import subprocess

import pytest

from conftest import REPO_ROOT

MUST_IGNORE = [
    "data/raw/eth/video.mp4", "data/interim/windows.npz", "data/cache/detections.parquet",
    "data/external/x.csv", "runs/smoke-abc-sna/manifest.json", "yolov8n.pt", "src/model.safetensors",
    "reports/figures/demo.mp4", ".env", "web/.env.local", "node_modules/x/index.js",
    "web/node_modules/x/index.js", ".next/cache/x", "web/.next/x",
    "src/trackvision/__pycache__/config.cpython-312.pyc", ".pytest_cache/v/x", ".venv/Scripts/python.exe",
    ".venv/pyvenv.cfg", ".venv/Lib/site-packages/numpy/__init__.py", "src/trackvision.egg-info/PKG-INFO",
    "env/dev.local.yml", "some/dir/.cache/blob",
    "tests/fixtures/__pycache__/synthetic.cpython-312.pyc", "tests/fixtures/sub/__pycache__/x.pyc",
    # audit gaps (2026-10-29)
    ".mypy_cache/3.12/x.json", "src/.mypy_cache/x", ".ruff_cache/x", "htmlcov/index.html", ".tox/py312/x",
    "web/out/index.html", "web/tsconfig.tsbuildinfo", "tsconfig.tsbuildinfo", "run.log", "runs-notes/train.log",
    "wandb/run-1/files/x", "mlruns/0/meta.yaml", "lightning_logs/version_0/x", "Thumbs.db", "web/public/Thumbs.db",
    ".vscode/settings.json", ".idea/workspace.xml",
    # generated files inside tests/fixtures, which the fixtures exception must not re-include
    "tests/fixtures/.DS_Store", "tests/fixtures/sub/Thumbs.db", "tests/fixtures/.cache/x",
    "tests/fixtures/.pytest_cache/v/x", "tests/fixtures/.mypy_cache/x", "tests/fixtures/.ruff_cache/x",
    "tests/fixtures/.ipynb_checkpoints/x", "tests/fixtures/node_modules/x/index.js",
    # checker/ignore consistency pass (2026-10-29)
    ".coverage", ".coverage.host.1234", "src/trackvision.egg-info/SOURCES.txt", "web/.vercel/project.json",
    ".vercel/project.json", "__v0_runtime_loader.js", "web/__v0_devtools.tsx", "web/__v0_jsx-dev-runtime.ts",
    ".snowflake/x", ".v0-trash/x", "env/dev.local.yaml", "env/sub/x.local.yml", ".env.production", "web/.env.staging",
    # every ignored category stays ignored inside tests/fixtures/ (only data-like types are re-included)
    "tests/fixtures/.env", "tests/fixtures/sub/.env.local", "tests/fixtures/.coverage",
    "tests/fixtures/pkg.egg-info/PKG-INFO", "tests/fixtures/.vercel/project.json", "tests/fixtures/__v0_devtools.tsx",
    "tests/fixtures/.venv/pyvenv.cfg", "tests/fixtures/htmlcov/index.html", "tests/fixtures/.tox/x",
    "tests/fixtures/wandb/x", "tests/fixtures/.vscode/settings.json", "tests/fixtures/x.tsbuildinfo",
    "tests/fixtures/.next/x",
    # governing documents are kept locally but never tracked in the public repository
    "docs/spec/trackvision_specification_v4_15.md", "docs/spec/TrackVision Implementation Plan.pdf",
    "docs/spec/any_other_document.md",
]

# Matched only where Git compares case-insensitively (core.ignorecase=true, the Windows default).
CASE_VARIANTS_MUST_IGNORE = ["THUMBS.DB", ".VSCode/settings.json", "Run.LOG", "model.PT", ".IDEA/x", "Clip.MP4"]

MUST_TRACK = [
    "README.md", "pyproject.toml", "configs/base.yaml", "configs/experiments/smoke.yaml",
    "src/trackvision/config.py", "tests/fixtures/tiny_trajectory.npz", "tests/fixtures/clip.mp4",
    "reports/figures/degradation.png", "runs/README.md", "data/README.md",
    "env/requirements-lock-win_amd64-py312.txt", "env/README.md", "docs/spec/README.md",
    "web/package.json", "web/pnpm-lock.yaml", "scripts/check_staged.py",
    # look-alikes that must stay tracked
    "web/app/page.tsx", "web/tsconfig.json", "web/next-env.d.ts", "src/trackvision/io/output.py",
    "docs/logbook.md", "tests/fixtures/sample.log", "tests/fixtures/synthetic.py",
    "tests/fixtures/LARGE_FIXTURES_APPROVED.txt",
    # deliberate data-like fixtures and legitimate look-alikes (consistency pass)
    "tests/fixtures/sub/tiny.pt", "tests/fixtures/model.safetensors", "tests/fixtures/clip.mkv",
    "tests/fixtures/README.md", ".env.example", "web/.env.example", "docs/coverage.md",
    "src/trackvision/metrics/coverage.py", "env/README.md", "env/local-notes.md", "configs/local.yaml",
    "web/components/vercel-badge.tsx", "docs/environment.md",
]


def _ignored(path: str) -> bool:
    proc = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO_ROOT,
                          capture_output=True, check=False)
    if proc.returncode not in (0, 1):
        pytest.fail(f"git check-ignore failed for {path}: {proc.stderr!r}")
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="git or the repository is not available",
)


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_path_is_ignored(path):
    assert _ignored(path)


@pytest.mark.parametrize("path", MUST_TRACK)
def test_path_is_not_ignored(path):
    assert not _ignored(path)


def _ignorecase() -> bool:
    proc = subprocess.run(["git", "config", "--get", "core.ignorecase"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=False)
    return proc.stdout.strip().lower() == "true"


@pytest.mark.parametrize("path", CASE_VARIANTS_MUST_IGNORE)
def test_case_variant_is_ignored_where_git_ignores_case(path):
    if not _ignorecase():
        pytest.skip("core.ignorecase is not true here; .gitignore matching is case-sensitive "
                    "(scripts/check_staged.py still blocks these case-insensitively)")
    assert _ignored(path)
