"""The read-only endpoints. Every route is a GET; none of them can reach a research result for writing."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from trackvision import __version__
from trackvision.platform import schemas
from trackvision.platform.services import dataset_service, result_views
from trackvision.platform.services.cell_index import CellIndex
from trackvision.platform.services.run_registry import RunRegistry
from trackvision.platform.services.workspace import MAX_SIGMA_PX

router = APIRouter(prefix="/api")
ERRORS = {code: {"model": schemas.ErrorResponse} for code in (400, 404, 409)}
# Stored figures are SVG files. These headers stop one from running anything if it is opened directly.
SVG_HEADERS = {"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
               "X-Content-Type-Options": "nosniff"}
Population = Annotated[str | None, Query(pattern=r"^[a-z0-9_]{1,40}$")]
History = Annotated[int | None, Query(ge=1, le=100)]
Name = Annotated[str | None, Query(pattern=r"^[A-Za-z0-9._-]{1,60}$")]


def _registry(request: Request) -> RunRegistry:
    return request.app.state.registry


def _index(registry: RunRegistry, kinds: tuple[str, ...]) -> dict:
    items = [run.summary for run in registry.runs() if run.kind in kinds]
    default = registry.newest(kinds)
    return {"items": items, "default": default.id if default else None}


@router.get("/health", response_model=schemas.HealthResponse)
def health(request: Request):
    registry = _registry(request)
    runs = registry.runs()
    return {"status": "ok" if registry.available else "degraded", "version": __version__, "read_only": True,
            "results_directory_available": registry.available, "experiment_records": len(runs),
            "invalid_records": sum(run.summary["status"] == "invalid" for run in runs),
            "skipped_folders": registry.skipped_folders,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


@router.get("/datasets")
def datasets(request: Request):
    return {"items": dataset_service.list_datasets(request.app.state.settings, _registry(request))}


@router.get("/experiments", response_model=schemas.ExperimentPage)
def experiments(request: Request, kind: Name = None, status: Name = None, dataset: Name = None,
                limit: Annotated[int, Query(ge=1, le=500)] = 100, offset: Annotated[int, Query(ge=0)] = 0):
    runs = [run.summary for run in _registry(request).runs()]
    matching = [s for s in runs if (kind is None or s["kind"] == kind) and (status is None or s["status"] == status)
                and (dataset is None or dataset in s["datasets"])]
    return {"items": matching[offset:offset + limit], "total": len(matching), "limit": limit, "offset": offset,
            "counts": {"by_kind": Counter(s["kind"] for s in runs), "by_status": Counter(s["status"] for s in runs),
                       "by_evidence": Counter(s["evidence_level"] or "unknown" for s in runs),
                       "all": len(runs)}}


@router.get("/experiments/{experiment_id}", responses=ERRORS)
def experiment(request: Request, experiment_id: str):
    registry = _registry(request)
    run = registry.get(experiment_id)
    return {"experiment": run.summary, "views": result_views.views_of(run), "artifacts": result_views.artifacts(run),
            "stored_summary": result_views.stored_summary(run, registry.reader)}


@router.get("/experiments/{experiment_id}/cells", responses=ERRORS)
def cells(request: Request, experiment_id: str, experiment: Name = None, scene: Name = None, predictor: Name = None,
          condition: Name = None, population: Population = None, status: Name = None, seed: int | None = None,
          history: History = None, limit: Annotated[int, Query(ge=1, le=200)] = 50,
          offset: Annotated[int, Query(ge=0)] = 0):
    index: CellIndex = request.app.state.cells
    filters = {"experiment": experiment, "scene": scene, "predictor": predictor, "condition": condition,
               "population": population, "status": status, "seed": seed, "history_len": history}
    return index.page(_registry(request).get(experiment_id), filters, limit, offset)


@router.get("/experiments/{experiment_id}/artifacts/{name}", responses=ERRORS)
def figure(request: Request, experiment_id: str, name: str):
    content = result_views.figure_bytes(_registry(request).get(experiment_id), name)
    return Response(content=content, media_type="image/svg+xml", headers=SVG_HEADERS)


@router.get("/workspace/scenes")
def workspace_scenes(request: Request):
    return request.app.state.workspace.scenes()


@router.get("/workspace/scenes/{scene}/video/status", responses=ERRORS)
def workspace_video_status(request: Request, scene: str):
    return request.app.state.workspace.video_status(scene)


@router.get("/workspace/scenes/{scene}/video", responses=ERRORS)
def workspace_video(request: Request, scene: str):
    return FileResponse(request.app.state.workspace.video_file(scene), media_type="video/mp4")


@router.get("/workspace/scenes/{scene}/tracks", responses=ERRORS)
def workspace_tracks(request: Request, scene: str, source: Literal["default", "truth"] = "default"):
    return request.app.state.workspace.clean_tracks(scene, source)


@router.get("/workspace/scenes/{scene}/inject", responses=ERRORS)
def workspace_inject(request: Request, scene: str, a: Annotated[float, Query(ge=0, le=MAX_SIGMA_PX)] = 0.0,
                     b: Annotated[float, Query(ge=0, le=1)] = 0.0, c: Annotated[float, Query(ge=0, le=1)] = 0.0,
                     d: Annotated[float, Query(ge=0, le=1)] = 0.0, reconstruct: bool = False):
    """Exploratory and computed live. It is a GET because it changes nothing: results are never stored."""
    from trackvision.platform.services.exploratory import ErrorRequest
    workspace = request.app.state.workspace
    limit = workspace.noise_limit(scene)
    if a > limit:
        raise HTTPException(status_code=422, detail=f"position noise on {scene} cannot be above {limit:g}")
    return workspace.inject(scene, ErrorRequest(a, b, c, d, reconstruct))


@router.get("/workspace/scenes/{scene}/noise-example", responses=ERRORS)
def workspace_noise_example(request: Request, scene: str):
    """One person's history at three noise levels. Exploratory and computed live, like `inject`."""
    return request.app.state.workspace.noise_example(scene)


@router.get("/workspace/scenes/{scene}/sweep", responses=ERRORS)
def workspace_sweep(request: Request, scene: str):
    """Every error at every tested level, as far as it has got. Exploratory, calculated in the background, never stored."""
    return request.app.state.workspace.sweep(scene)


@router.get("/workspace/scenes/{scene}/tracker", responses=ERRORS)
def workspace_tracker(request: Request, scene: str):
    return request.app.state.workspace.tracker(scene)


TRACKING = ("mot17_tracking", "synthetic_tracker")
INJECTION = ("sweep", "ewap_injection")
PREDICTION = ("sweep", "synthetic_tracker", "ewap_baseline")
ANALYSIS = ("sweep", "engineering_analysis")


@router.get("/tracking", response_model=schemas.ViewIndex)
def tracking_index(request: Request):
    return _index(_registry(request), TRACKING)


@router.get("/tracking/{experiment_id}", responses=ERRORS)
def tracking(request: Request, experiment_id: str):
    registry = _registry(request)
    return result_views.tracking(registry.get(experiment_id), registry.reader, request.app.state.settings.base_config)


@router.get("/error-injection", response_model=schemas.ViewIndex)
def error_injection_index(request: Request):
    return _index(_registry(request), INJECTION)


@router.get("/error-injection/{experiment_id}", responses=ERRORS)
def error_injection(request: Request, experiment_id: str, population: Population = None, history: History = None):
    registry = _registry(request)
    return result_views.error_injection(registry.get(experiment_id), registry.reader,
                                        request.app.state.settings.base_config, population, history)


@router.get("/prediction", response_model=schemas.ViewIndex)
def prediction_index(request: Request):
    return _index(_registry(request), PREDICTION)


@router.get("/prediction/{experiment_id}", responses=ERRORS)
def prediction(request: Request, experiment_id: str, population: Population = None, history: History = None):
    registry = _registry(request)
    return result_views.prediction(registry.get(experiment_id), registry.reader,
                                   request.app.state.settings.base_config, population, history)


@router.get("/analysis", response_model=schemas.ViewIndex)
def analysis_index(request: Request):
    return _index(_registry(request), ANALYSIS)


@router.get("/analysis/{experiment_id}", responses=ERRORS)
def analysis(request: Request, experiment_id: str, population: Population = None, history: History = None):
    registry = _registry(request)
    return result_views.analysis(registry.get(experiment_id), registry.reader, request.app.state.settings.base_config,
                                 population, history, registry.runs())
