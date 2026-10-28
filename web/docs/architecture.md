# TrackVision frontend architecture

TrackVision is a Next.js App Router research workspace for studying error propagation from multi-object tracking into future trajectory prediction. This foundation deliberately contains no ML algorithms, inference, database, storage, or backend execution.

## Layers
- `app/`: thin route entry points for the eight-stage guided workflow plus experiments and settings.
- `components/trackvision-app.tsx`: composable shell, page layouts, viewers, forms, charts, and scientific-integrity notices.
- `types/`: domain contracts with nullable results so absent backend data is explicit.
- `stores/`: one React context for navigation-safe experiment state.
- `lib/api/`: typed service boundary. Replace these functions with REST/WebSocket adapters later.
- `mocks/`: keep mock fixtures separate from services when expanding the demo.

State is intentionally frontend-only and resets on a full reload. Persistence can be added later without changing page contracts.
