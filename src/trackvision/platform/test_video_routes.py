"""The read-only endpoints of the Test Videos section (decision 0017). Every route is a GET."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import FileResponse

from trackvision.platform.routes import ERRORS
from trackvision.platform.services.workspace import MAX_SIGMA_PX

router = APIRouter(prefix="/api/test-videos")


@router.get("")
def test_videos(request: Request):
    return request.app.state.test_videos.catalogue()


@router.get("/{video}/video", responses=ERRORS)
def test_video_file(request: Request, video: str):
    return FileResponse(request.app.state.test_videos.video_file(video), media_type="video/mp4")


@router.get("/{video}/tracks", responses=ERRORS)
def test_video_tracks(request: Request, video: str):
    return request.app.state.test_videos.tracks(video)


@router.get("/{video}/forecast", responses=ERRORS)
def test_video_forecast(request: Request, video: str, a: Annotated[float, Query(ge=0, le=MAX_SIGMA_PX)] = 0.0,
                        b: Annotated[float, Query(ge=0, le=1)] = 0.0, c: Annotated[float, Query(ge=0, le=1)] = 0.0,
                        d: Annotated[float, Query(ge=0, le=1)] = 0.0, reconstruct: bool = False):
    """Computed live. It is a GET because it changes nothing: results are never stored."""
    from trackvision.platform.services.exploratory import ErrorRequest
    return request.app.state.test_videos.forecast(video, ErrorRequest(a, b, c, d, reconstruct))
