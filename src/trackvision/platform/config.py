"""Settings for the read-only API: where results live and which local origins may read them."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from trackvision import config

DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


@dataclass(frozen=True)
class Settings:
    runs_dir: Path
    manifests_dir: Path
    base_config: Path = config.DEFAULT_BASE_PATH
    cors_origins: tuple[str, ...] = DEFAULT_ORIGINS
    raw_dir: Path = config.REPO_ROOT / "data" / "raw"
    cache_dir: Path = config.REPO_ROOT / "data" / "cache" / "video"
    detections_dir: Path = config.REPO_ROOT / "data" / "cache" / "detections"


def load_settings() -> Settings:
    """Read the settings. `TRACKVISION_RUNS_DIR`, `TRACKVISION_DETECTIONS_DIR` and `TRACKVISION_CORS_ORIGINS` override."""
    origins = os.environ.get("TRACKVISION_CORS_ORIGINS")
    return Settings(
        runs_dir=Path(os.environ.get("TRACKVISION_RUNS_DIR", config.REPO_ROOT / "runs")),
        manifests_dir=config.REPO_ROOT / "data" / "manifests",
        detections_dir=Path(os.environ.get("TRACKVISION_DETECTIONS_DIR", config.REPO_ROOT / "data" / "cache" / "detections")),
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()) if origins else DEFAULT_ORIGINS)
