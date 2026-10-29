"""scripts/check_staged.py against a throwaway repository with good and bad files staged."""

import shutil
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "check_staged.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _stage(repo, files: dict[str, bytes]):
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        # -f: stage even if a global ignore would hide it; we are testing the checker itself
        subprocess.run(["git", "-C", str(repo), "add", "-f", rel], check=True)


def _check(repo):
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo)],
                          capture_output=True, text=True, check=False)


def test_clean_staged_set_passes(repo):
    _stage(repo, {
        "src/pkg/mod.py": b"x = 1\n", "configs/base.yaml": b"a: 1\n", "data/README.md": b"#\n",
        "runs/README.md": b"#\n", "env/requirements-lock-win_amd64-py312.txt": b"numpy==2.0.0\n",
        "tests/fixtures/tiny.mp4": b"\0" * 10,
        "web/pnpm-lock.yaml": b"x" * 1_500_000,
        # look-alikes that must not be blocked
        "tests/fixtures/sample.log": b"line\n", "src/pkg/output.py": b"\n", "docs/logbook.md": b"#\n",
        "web/app/page.tsx": b"x\n", "web/next-env.d.ts": b"x\n",
        # consistency pass: deliberate fixtures (any case) and legitimate look-alikes
        "tests/fixtures/model.PT": b"\0", "tests/fixtures/Clip.MP4": b"\0", ".env.example": b"KEY=\n",
        "web/.env.example": b"KEY=\n", "docs/coverage.md": b"#\n", "src/trackvision/metrics/coverage.py": b"\n",
        "env/local-notes.md": b"#\n", "configs/local.yaml": b"a: 1\n", "web/components/vercel-badge.tsx": b"x\n",
        # checksum manifests directly under data/manifests/ are the one tracked kind of data file
        "data/manifests/ewap.sha256": b"0" * 64 + b"  x\n", "data/manifests/MOT17.SHA256": b"0" * 64 + b"  y\n",
    })
    proc = _check(repo)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("path", [
    "data/raw/eth.avi.txt", "runs/smoke-x/manifest.json",
    # only *.sha256 directly under data/manifests/ is allowed; look-alikes stay blocked
    "data/manifests/notes.txt", "data/manifests/sub/x.sha256", "data/manifests/x.sha256.bak",
    "data/raw/manifests/x.sha256", "data/x.sha256", ".env", "web/.env.production",
    "deploy/server.pem", "keys/id_rsa", "credentials-gcp.json", ".venv/pyvenv.cfg",
    "venv2/Lib/site-packages/numpy/__init__.py",
    "env/dev.local.yml", "src/__pycache__/m.cpython-312.pyc", "web/node_modules/a/i.js",
    "web/.next/build.js", "models/yolov8n.pt", "weights/m.safetensors", "out/demo.mp4",
    # audit gaps (2026-10-29)
    ".mypy_cache/3.12/x.json", ".ruff_cache/x", "htmlcov/index.html", ".tox/py312/x", "web/out/index.html",
    "web/tsconfig.tsbuildinfo", "build/lib/trackvision/x.py", "dist/trackvision-0.1.0.whl", "run.log",
    "wandb/run-1/files/x", "mlruns/0/meta.yaml", "lightning_logs/version_0/x", "Thumbs.db", ".DS_Store",
    ".vscode/settings.json", ".idea/workspace.xml",
    # generated files inside tests/fixtures are blocked too (the __pycache__ regression and its kin)
    "tests/fixtures/__pycache__/synthetic.cpython-312.pyc", "tests/fixtures/.mypy_cache/x",
    "tests/fixtures/.DS_Store", "tests/fixtures/node_modules/x/index.js",
    # consistency pass (2026-10-29): categories .gitignore protects
    ".coverage", ".coverage.host.1234", "src/trackvision.egg-info/PKG-INFO", "web/.vercel/project.json",
    "__v0_runtime_loader.js", "web/__v0_devtools.tsx", "web/__v0_jsx-dev-runtime.ts", ".snowflake/x",
    ".v0-trash/x", "env/dev.local.yaml", "env/sub/x.local.yml", "tests/fixtures/.env",
    "tests/fixtures/.coverage", "tests/fixtures/pkg.egg-info/PKG-INFO", "tests/fixtures/.vercel/x",
    # case variants are blocked regardless of core.ignorecase
    "THUMBS.DB", ".VSCode/settings.json", "Run.LOG", "models/yolo.PT", "out/Demo.MP4", "DATA/raw/x.csv",
    "RUNS/x/manifest.json", ".ENV", "web/.Env.Local", "src/__PYCACHE__/m.pyc", "keys/ID_RSA",
])
def test_forbidden_path_is_blocked(repo, path):
    _stage(repo, {path: b"content\n"})
    proc = _check(repo)
    assert proc.returncode == 1
    assert f"BLOCKED  {path}" in proc.stdout


def test_oversized_file_is_blocked_unless_approved(repo):
    big = b"y" * 1_200_000
    _stage(repo, {"tests/fixtures/big.npz": big})
    assert _check(repo).returncode == 1

    import hashlib
    line = f"tests/fixtures/big.npz | {len(big)} | {hashlib.sha256(big).hexdigest()} | Reviewer | 2026-10-29 | test\n"
    _stage(repo, {"tests/fixtures/LARGE_FIXTURES_APPROVED.txt": line.encode()})
    proc = _check(repo)
    assert proc.returncode == 0, proc.stdout

    _stage(repo, {"tests/fixtures/big.npz": big + b"changed"})
    assert _check(repo).returncode == 1  # approval covers exactly the approved content


GOVERNING_DOCUMENTS = [
    "docs/spec/trackvision_specification_v4_15.md",
    "docs/spec/TrackVision Implementation Plan.pdf",
    "docs/spec/any_other_document.md",
    "DOCS/Spec/trackvision_specification_v4_15.md",   # case variant
]


@pytest.mark.parametrize("path", GOVERNING_DOCUMENTS)
def test_governing_documents_are_blocked_even_when_small(repo, path):
    """The spec and the plan are not redistributed: blocked whatever their size or hash."""
    _stage(repo, {path: b"small copy of a governing document\n"})
    proc = _check(repo)
    assert proc.returncode == 1
    assert f"BLOCKED  {path}: governing document" in proc.stdout


def test_spec_readme_is_allowed_and_other_oversized_files_stay_blocked(repo):
    import hashlib
    big = b"z" * 1_100_000
    readme = f"| `listed.md` | governing document | `{hashlib.sha256(big).hexdigest()}` |\n".encode()
    _stage(repo, {
        "docs/spec/README.md": readme,        # the only tracked file under docs/spec/
        "docs/other/listed.md": big,          # oversized; a recorded hash grants no exception
    })
    proc = _check(repo)
    assert proc.returncode == 1
    assert "BLOCKED  docs/other/listed.md" in proc.stdout
    assert "docs/spec/README.md" not in proc.stdout


def test_checker_does_not_change_the_index(repo):
    _stage(repo, {"runs/x/manifest.json": b"{}", "src/a.py": b"\n"})
    before = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--name-only"],
                            capture_output=True, text=True, check=True).stdout
    _check(repo)
    after = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--name-only"],
                           capture_output=True, text=True, check=True).stdout
    assert before == after
