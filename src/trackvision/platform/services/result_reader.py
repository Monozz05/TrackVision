"""Safe reads of stored artifacts.

Files are opened for reading only. Errors name the file, never its full path, because messages reach the client.
Parsed documents are cached by modification time and size, so a large `analysis.json` is read once and a changed
file is read again. Callers must treat a returned document as read-only.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path


class ArtifactError(RuntimeError):
    """A stored artifact cannot be served."""


class ArtifactMissing(ArtifactError):
    """The file does not exist."""


class ArtifactMalformed(ArtifactError):
    """The file exists but is not a usable record."""


def _reject_constant(name: str):
    raise ValueError(f"non-finite value {name}")


class ResultReader:
    def __init__(self, max_documents: int = 6):
        self._documents: OrderedDict[tuple, dict] = OrderedDict()
        self._hashes: dict[tuple, str] = {}
        self._max_documents = max_documents
        self._lock = threading.Lock()

    @staticmethod
    def _signature(path: Path) -> tuple:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ArtifactMissing(f"{path.name} is missing") from exc
        return str(path), stat.st_mtime_ns, stat.st_size

    def read_json(self, path: Path, cache: bool = True) -> dict:
        """The parsed object in `path`. Non-finite numbers count as malformed: stored records never contain them.

        Discovery passes `cache=False`: it reads many small files on every rescan, which would otherwise push the
        large analysis documents out of the cache.
        """
        key = self._signature(path)
        with self._lock:
            if key in self._documents:
                self._documents.move_to_end(key)
                return self._documents[key]
        try:
            doc = json.loads(path.read_bytes().decode("utf-8"), parse_constant=_reject_constant)
        except OSError as exc:
            raise ArtifactMissing(f"{path.name} could not be read") from exc
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are both ValueError
            raise ArtifactMalformed(f"{path.name} is not valid JSON") from exc
        if not isinstance(doc, dict):
            raise ArtifactMalformed(f"{path.name} does not hold a JSON object")
        if not cache:
            return doc
        with self._lock:
            self._documents[key] = doc
            while len(self._documents) > self._max_documents:
                self._documents.popitem(last=False)
        return doc

    def sha256(self, path: Path) -> str:
        """SHA-256 of the file's bytes, which is what manifests record."""
        key = self._signature(path)
        with self._lock:
            if key in self._hashes:
                return self._hashes[key]
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        with self._lock:
            self._hashes[key] = digest.hexdigest()
        return digest.hexdigest()
