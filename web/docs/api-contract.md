# API contract

The frontend talks to the FastAPI backend in `src/trackvision/platform/`. It has no write route. Components call the typed functions in `lib/api/services.ts`; only `lib/api/client.ts` touches the network.

Base URL: `NEXT_PUBLIC_API_BASE_URL`, default `http://127.0.0.1:8000/api`. Interactive documentation is at `http://127.0.0.1:8000/docs`.

## Endpoints

Every route is a GET. Any other method returns 405.

| Route | Returns |
|---|---|
| `GET /health` | Status, version, read-only flag, results-folder availability, record counts, timestamp |
| `GET /datasets` | `{items}`: datasets with a manifest or a stored run, their scenes, ground-truth status and covering runs |
| `GET /experiments` | Page of run summaries. Filters: `kind`, `status`, `dataset`, `limit` (up to 500), `offset` |
| `GET /experiments/{id}` | `{experiment, views, artifacts, stored_summary}` |
| `GET /experiments/{id}/cells` | Page of digest-verified seed-level records of a sweep. Filters: `experiment`, `scene`, `predictor`, `condition`, `population`, `status`, `seed`, `history`, `limit` (up to 200), `offset` |
| `GET /experiments/{id}/artifacts/{name}` | One stored SVG figure |
| `GET /tracking`, `/error-injection`, `/prediction`, `/analysis` | `{items, default}`: the runs that have this view, and the newest complete one |
| `GET /tracking/{id}` | MOT17 or E4 results by sequence, tracker settings, recorded limits |
| `GET /error-injection/{id}` | Requested and achieved severity by level, monotonicity, figures. Query: `population`, `history` |
| `GET /prediction/{id}` | ADE, FDE, RMSE, coverage and horizon profile by scene and predictor, plus any stored across-scene means. Query: `population`, `history` |
| `GET /analysis/{id}` | Sweep: curves, AUDC, family AUDC, P-15, hypotheses. E5 to E8 run: association, dry run, inverse mapping, crossings, blocked and deferred work. Query: `population`, `history` |
| `GET /workspace/scenes` | Every scene, with whether it has video and error controls, the reason if not, its stored run, and slider limits. `parent` names the recording a numbered part comes from, and is `null` on a full recording. A part has its own video and tracks, with frames counted from its start, and never a stored run |
| `GET /workspace/scenes/{scene}/video/status` | `ready`, `preparing`, `missing` or `failed`. The first call starts building the browser copy |
| `GET /workspace/scenes/{scene}/video` | The H.264 copy, with Range support for seeking |
| `GET /workspace/scenes/{scene}/noise-example` | One real person's clean history, that history at three noise levels from the frozen injector, and their true future. Exploratory and computed live. 409 if the annotations are missing |
| `GET /workspace/scenes/{scene}/sweep` | Every error at every tested level on this scene, as far as it has got. `state` is `calculating`, `ready` or `failed`, with `done` and `total`. Each level holds requested and achieved amounts, coverage, and ADE, FDE, rises on the same windows and error by step for CV and CA. Each finished error adds `limits` and `better`. It runs in the background: ask again until `state` is no longer `calculating`. Exploratory, never saved. 409 if the annotations are missing |
| `GET /workspace/scenes/{scene}/tracker` | The stored detector and tracker settings and results behind a scene, as label and value rows and result tables. A value that was not stored is `null`. Scenes with no tracker return `run: null` and a `note`. No page uses this now |
| `GET /workspace/scenes/{scene}/tracks` | Every person with an ID, positions and boxes in video pixels, plus the source of those IDs. Works on all scenes. Query: `source=truth` gives the annotated people instead of the tracker, on any scene that has one. 404 if the source is not on this machine |
| `GET /workspace/scenes/{scene}/inject` | Live exploratory errors. Query: `a` (0 to 1 m, or 0 to 60 px on scenes in pixels), `b`, `c`, `d` (0 to 1), `reconstruct`. Returns its `unit`, achieved severity, CV and CA metrics under unit-neutral names (`ade`, `fde`, `clean_ade`), and the corrupted tracks with their boxes and events. 409 if the annotations are missing, or if a newer request for the scene replaced this one |
| `GET /test-videos` | The fixed list of test videos, with each one's credit, licence, source, camera note, whether it is ready and the reason if not, slider limits, and the notes the page shows. Decision 0017 |
| `GET /test-videos/{video}/video` | The H.264 copy of a test video. 404 for any name not on the list, 409 if it has not been prepared |
| `GET /test-videos/{video}/tracks` | Everyone the tracker follows on that video, frame by frame, in video pixels. Same shape as a scene's tracks |
| `GET /test-videos/{video}/forecast` | Forecasts, computed live and never saved. Query as for `inject`, with `a` from 0 to 60 px. Returns `forecasts` (the total, and how many each seed kept), `moved` (how far errors moved the forecast for CV and CA, in pixels, `null` if none is left), achieved severity, the tracks on the sampling grid, and the overlay. **It has no ADE, FDE or other accuracy field**, because these videos have no true positions |

## Experiment IDs

An ID is the run folder's path below `runs/`, joined with `~`, for example `run-2026-11-01~mot17`. IDs are looked up among discovered runs and are never turned into a path. The ID recorded by the run itself, if any, is `record_id`.

## Run status

| `status` | Meaning |
|---|---|
| `complete` | All required records are present and verify |
| `incomplete` | A required file is absent, or the run did not finish; `status_reason` says which |
| `invalid` | A record is malformed or unrecognised, or differs from its recorded hash; `status_reason` says which |

No record marks a run as superseded, so that status does not exist. `latest_in_series` only says which run is the newest complete one of its kind and scope.

## Errors

Errors are JSON, `{"detail": "..."}`, and never contain a filesystem path.

| Code | Meaning |
|---|---|
| 400 | Badly formed experiment ID |
| 404 | No such experiment, or no such view, population, history or figure in that run |
| 405 | Method other than GET |
| 409 | The run is not usable, or a stored record failed verification |
| 422 | Invalid query parameter |
| 500 | Unexpected error; details are in the backend log only |

## Rules

- **Values are stored values,** except under `/workspace`. There the frozen injectors and scorer run live, every response is labelled exploratory, and nothing is saved (decisions 0008, 0011, 0012 and 0013). A value that does not exist is `null`, and undefined levels keep their reason.
- **Achieved severity.** Levels carry `requested` and achieved values separately, and degradation plots use the achieved one.
- **Pooling.** Across-scene values are returned only where the analysis stored them, and never for ETH and HOTEL.
- **Files.** Only figures under a run's `figures/` folder and the browser copies of scene video can be fetched. Other files are listed by name and size, and source dataset files are never served.
- **Paths.** Any absolute path found in a stored record is replaced with `[path withheld]`.
