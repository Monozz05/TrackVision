# Future API contract

The frontend expects typed service functions rather than fetch calls in components.

- `GET /datasets` → `Dataset[]`
- `POST /videos` → `VideoMetadata`
- `POST /experiments/{id}/tracking` → queued job / `TrackingResult`
- `POST /experiments/{id}/errors` → queued job / `ErrorResult`
- `POST /experiments/{id}/prediction` → queued job / `PredictionResult`
- `GET /experiments/{id}/analysis` → `AnalysisResult`
- `GET /experiments` and `GET /experiments/{id}` → experiment metadata

All result metrics are nullable. Backend progress should supply status, progress, current stage, and elapsed time; the frontend must not fabricate them.
