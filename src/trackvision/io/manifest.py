"""Dataset checksum manifests (`data/manifests/<dataset>.sha256`, sha256sum format).

Each line is `<sha256>  <path relative to the dataset's raw directory>`. The manifest hash, the
SHA-256 of the manifest file itself, is the `dataset_manifest_hash` of spec v4.15 section 11.3.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class ManifestError(ValueError):
    """Raised when a file is missing from the manifest or does not match its recorded hash."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or not rel:
            raise ManifestError(f"{path}:{number}: malformed manifest line")
        if rel in entries:
            raise ManifestError(f"{path}:{number}: duplicate entry {rel!r}")
        entries[rel] = digest.lower()
    return entries


def manifest_hash(path: Path) -> str:
    return file_sha256(Path(path))


def verify(manifest_path: Path, raw_root: Path, relpaths: list[str]) -> dict[str, str]:
    """Check each listed file under `raw_root` against the manifest; return {relpath: sha256}."""
    entries = read_manifest(manifest_path)
    verified = {}
    for rel in relpaths:
        if rel not in entries:
            raise ManifestError(f"{rel!r} is not listed in {manifest_path}")
        actual = file_sha256(Path(raw_root) / rel)
        if actual != entries[rel]:
            raise ManifestError(f"{rel!r} does not match {manifest_path} "
                                f"(expected {entries[rel]}, found {actual})")
        verified[rel] = actual
    return verified
