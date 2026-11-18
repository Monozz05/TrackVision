"""The read-only FastAPI application over stored TrackVision results.

Run from the repository root:
    .venv\\Scripts\\python.exe -m uvicorn trackvision.platform.app:app --host 127.0.0.1 --port 8000

The pipeline's stored records stay the single source of truth: the API has no write route and no database, and it
computes no research metric (spec v4.15 section 11.1, the UI is a read-only consumer). The workspace routes also run
the frozen injectors live, as an exploratory view that is never saved (decision 0008). The API binds to the local
machine because some stored results are labelled local only.
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from trackvision import __version__
from trackvision.platform.config import Settings, load_settings
from trackvision.platform.routes import router
from trackvision.platform.services.cell_index import CellIndex
from trackvision.platform.services.result_reader import ArtifactError, ArtifactMissing, ResultReader
from trackvision.platform.services.result_views import RunNotUsable, ViewUnavailable
from trackvision.platform.services.run_registry import InvalidExperimentId, RunRegistry, UnknownExperiment
from trackvision.platform.services.test_videos import StreetVideos
from trackvision.platform.services.workspace import SceneUnavailable, Workspace
from trackvision.platform.test_video_routes import router as test_video_router

log = logging.getLogger("trackvision.platform")
_ABSOLUTE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|/(?:home|Users|mnt|tmp|var)/)")
WITHHELD = "[path withheld]"


def withhold_paths(value):
    """Replace any absolute path a stored record happens to hold, so filesystem layout never reaches a client."""
    if isinstance(value, str):
        return WITHHELD if _ABSOLUTE_PATH.search(value) else value
    if isinstance(value, dict):
        return {key: withhold_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [withhold_paths(item) for item in value]
    return value


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(withhold_paths(content), ensure_ascii=False, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    reader = ResultReader()
    app = FastAPI(title="TrackVision results API", version=__version__, default_response_class=SafeJSONResponse,
                  description="Read-only access to stored TrackVision results.")
    app.state.settings = settings
    app.state.registry = RunRegistry(settings.runs_dir, reader)
    app.state.cells = CellIndex()
    app.state.workspace = Workspace(settings, app.state.registry)
    app.state.test_videos = StreetVideos(settings, reader)

    def error(status: int, detail: str) -> SafeJSONResponse:
        return SafeJSONResponse(status_code=status, content={"detail": detail})

    # Added before CORS so that it sits inside it: an unexpected error must still reach the browser as a readable
    # 500, not as a blocked response that looks like the backend being down.
    @app.middleware("http")
    async def _unexpected(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            log.exception("unhandled error on %s", request.url.path)
            return error(500, "the server could not read this result; see the backend log")

    app.add_middleware(GZipMiddleware, minimum_size=4096)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET"],
                       allow_headers=["*"])
    app.include_router(router)
    app.include_router(test_video_router)

    @app.exception_handler(InvalidExperimentId)
    def _invalid_id(_: Request, exc: InvalidExperimentId):
        return error(400, str(exc))

    @app.exception_handler(UnknownExperiment)
    def _unknown(_: Request, exc: UnknownExperiment):
        return error(404, str(exc))

    @app.exception_handler(ViewUnavailable)
    def _unavailable(_: Request, exc: ViewUnavailable):
        return error(404, str(exc))

    @app.exception_handler(ArtifactMissing)
    def _missing(_: Request, exc: ArtifactMissing):
        return error(404, str(exc))

    @app.exception_handler(RunNotUsable)
    def _not_usable(_: Request, exc: RunNotUsable):
        return error(409, str(exc))

    @app.exception_handler(SceneUnavailable)
    def _scene_unavailable(_: Request, exc: SceneUnavailable):
        return error(409, str(exc))

    @app.exception_handler(ArtifactError)
    def _malformed(_: Request, exc: ArtifactError):
        return error(409, str(exc))

    return app


app = create_app()
