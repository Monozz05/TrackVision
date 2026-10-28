"""Repository policy: files under tests/fixtures/ are at most 1 MB unless explicitly approved."""

import hashlib

from conftest import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures"
APPROVALS = FIXTURES / "LARGE_FIXTURES_APPROVED.txt"
MAX_BYTES = 1_000_000  # repository policy, not a research parameter


def _approvals() -> dict[str, tuple[int, str]]:
    approved = {}
    for line in APPROVALS.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = [f.strip() for f in line.split(" | ")]
        assert len(fields) == 6 and all(fields), f"malformed approval line: {line!r}"
        path, size, digest, _approver, _date, _reason = fields
        approved[path] = (int(size), digest)
    return approved


def test_fixtures_respect_the_size_policy():
    approved = _approvals()
    violations = []
    for path in sorted(p for p in FIXTURES.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts:
            continue
        size = path.stat().st_size
        if size <= MAX_BYTES:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if approved.get(rel) != (size, digest):
            violations.append(f"{rel} ({size} bytes) has no matching approval")
    assert not violations, violations
