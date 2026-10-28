"""Pre-commit check of the staged file set. Read-only: it never unstages, deletes or edits.

Usage: python scripts/check_staged.py [--repo PATH]

Exits 1 and lists every offending path if anything staged (added, copied, modified, renamed or
type-changed) is: data or run output, a governing document under docs/spec/ (only its README is
tracked), a credential, an environment directory, a cache or build
output, test-coverage output, a v0/Vercel sandbox artefact, editor or operating-system metadata,
experiment-tracker output, a log file, model weights, a generated video, or larger than 1 MB
without an approved exception. Matching is case-insensitive. Under tests/fixtures/ only
deliberate data-like files (logs, weights, videos) are allowed, mirroring .gitignore.
Standard library only; this is repository tooling, not research code.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import PurePosixPath

MAX_BYTES = 1_000_000  # repository policy, shared with tests/fixtures/README.md
FIXTURES_PREFIX = "tests/fixtures/"
ALWAYS_ALLOWED_LARGE = {"web/pnpm-lock.yaml"}

WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".safetensors"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".token"}
# All names below are lower-case: matching is case-insensitive, like .gitignore on Windows
# (core.ignorecase=true), so a case variant such as THUMBS.DB or model.PT cannot slip through.
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".cache", "node_modules", ".next", ".ipynb_checkpoints",
               ".mypy_cache", ".ruff_cache", "htmlcov", ".tox"}
TRACKER_OUTPUT_PARTS = {"wandb", "mlruns", "lightning_logs"}
EDITOR_OS_PARTS = {".vscode", ".idea"}
EDITOR_OS_NAMES = {".ds_store", "thumbs.db"}
SANDBOX_PARTS = {".vercel", ".snowflake", ".v0-trash"}
SANDBOX_NAMES = {"__v0_runtime_loader.js", "__v0_devtools.tsx", "__v0_jsx-dev-runtime.ts"}
BUILD_PREFIXES = ("build/", "dist/", "web/out/")
ENV_PARTS = {".venv", "conda-meta", "site-packages"}
ALLOWED_IN_DATA_RUNS = {"data/readme.md", "runs/readme.md"}


def _git(repo: str, *args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(["git", "-C", repo, *args], input=input_bytes, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def staged_paths(repo: str) -> list[str]:
    out = _git(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMRT", "-z")
    return [p for p in out.decode("utf-8").split("\0") if p]


def staged_blob(repo: str, path: str) -> bytes:
    return _git(repo, "cat-file", "blob", f":{path}")


def _staged_text(repo: str, path: str) -> str | None:
    """Text of a staged file, or None if that path is not in the index."""
    proc = subprocess.run(["git", "-C", repo, "cat-file", "blob", f":{path}"],
                          capture_output=True, check=False)
    return proc.stdout.decode("utf-8") if proc.returncode == 0 else None


def _fixture_approvals(repo: str) -> dict[str, tuple[int, str]]:
    text = _staged_text(repo, f"{FIXTURES_PREFIX}LARGE_FIXTURES_APPROVED.txt") or ""
    approved = {}
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            fields = [f.strip() for f in line.split(" | ")]
            if len(fields) == 6:
                approved[fields[0]] = (int(fields[1]), fields[2])
    return approved


def rule_violation(path: str) -> str | None:
    """The content-independent rule a path breaks, or None. Matching is case-insensitive."""
    lower = path.lower()
    p = PurePosixPath(lower)
    parts, name, suffix = set(p.parts), p.name, p.suffix
    in_fixtures = lower.startswith(FIXTURES_PREFIX)
    if lower.startswith("data/") and lower not in ALLOWED_IN_DATA_RUNS:
        return "dataset content (data/)"
    if lower.startswith("runs/") and lower not in ALLOWED_IN_DATA_RUNS:
        return "experiment output (runs/)"
    if lower.startswith("docs/spec/") and lower != "docs/spec/readme.md":
        return "governing document, not redistributed (see docs/spec/README.md)"
    if (name == ".env" or name.startswith(".env.")) and name != ".env.example":
        return "environment variables file (possible credentials)"
    if suffix in SECRET_SUFFIXES or name.startswith("id_rsa") or re.match(r"credentials.*\.json$", name):
        return "credential or key file"
    if parts & ENV_PARTS or re.match(r"env/(.+/)?[^/]+\.local\.ya?ml$", lower):
        return "environment directory or local environment file"
    if parts & CACHE_PARTS or suffix in {".pyc", ".pyo"}:
        return "cache or build output"
    if lower.startswith(BUILD_PREFIXES) or suffix == ".tsbuildinfo" or any(x.endswith(".egg-info") for x in parts):
        return "cache or build output"
    if name == ".coverage" or name.startswith(".coverage."):
        return "test-coverage output"
    if parts & SANDBOX_PARTS or name in SANDBOX_NAMES:
        return "v0 or Vercel sandbox artefact"
    if parts & EDITOR_OS_PARTS or name in EDITOR_OS_NAMES:
        return "editor or operating-system metadata"
    if parts & TRACKER_OUTPUT_PARTS:
        return "experiment-tracker output (results belong under runs/)"
    if suffix == ".log" and not in_fixtures:
        return "log file"
    if suffix in WEIGHT_SUFFIXES and not in_fixtures:
        return "model weights"
    if suffix in VIDEO_SUFFIXES and not in_fixtures:
        return "generated video"
    return None


def check(repo: str) -> list[str]:
    problems = []
    paths = staged_paths(repo)
    approvals = _fixture_approvals(repo)
    for path in paths:
        reason = rule_violation(path)
        if reason:
            problems.append(f"{path}: {reason}")
            continue
        size = int(_git(repo, "cat-file", "-s", f":{path}").decode().strip())
        if size <= MAX_BYTES or path.lower() in ALWAYS_ALLOWED_LARGE:
            continue
        digest = hashlib.sha256(staged_blob(repo, path)).hexdigest()
        if approvals.get(path) == (size, digest):
            continue
        problems.append(f"{path}: {size} bytes exceeds the 1 MB policy without an approved exception")
    print(f"checked {len(paths)} staged file(s)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=".", help="repository to check (default: current directory)")
    args = parser.parse_args(argv)
    problems = check(args.repo)
    for problem in problems:
        print(f"BLOCKED  {problem}")
    if problems:
        print(f"{len(problems)} problem(s); nothing was changed. Unstage or fix these before committing.")
        return 1
    print("OK: no forbidden or oversized files staged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
