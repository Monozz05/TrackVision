# TrackVision

TrackVision is a dark research-lab frontend for **Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments**. It studies how controlled tracking errors may propagate into trajectory prediction and downstream measured application effects.

## Run

```bash
pnpm install
pnpm dev
```

Build with `pnpm build`. The project uses Next.js App Router, React, TypeScript, Tailwind CSS, Lucide, and Recharts.

## Workflow

The guided workflow is exactly eight stages: Overview, Data & Ground Truth, Tracking, Error Injection, Error Visualization, Prediction, Analysis, and Application Impact. Experiment history and settings are available outside the workflow.

## Architecture

Routes are thin and compose `components/trackvision-app.tsx`. Domain contracts live in `types/index.ts`; navigation-safe frontend state lives in `stores/experiment-store.tsx`; typed mock service boundaries live in `lib/api/services.ts`. No research algorithms, real video processing, database, MLflow, or backend are included.

## Mock data and integrity

Mock fixtures are labeled `MOCK DATA · UI DEVELOPMENT ONLY`. Result metrics are nullable and shown as em dashes until valid backend data exists. Uploaded videos are explicitly qualitative-only unless annotations are supplied. The clean future target is represented separately from corrupted observed history.

## Future integration

Replace the functions in `lib/api/services.ts` with REST and WebSocket adapters for a future FastAPI backend. Keep components unaware of transport details. See `docs/architecture.md`, `docs/api-contract.md`, and `docs/implementation-roadmap.md`.
