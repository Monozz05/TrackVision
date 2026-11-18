# TrackVision

TrackVision is the frontend for **Deep Learning-Based Multi-Object Tracking and Future Trajectory Prediction in Dynamic Environments**. Pick a scene, watch its video with a box, an ID and a trail on each person, set errors with sliders, and see the effect on prediction beside the stored results. It talks to the FastAPI backend in `src/trackvision/platform/`.

## Run

Start the backend first (see the root `README.md`), then:

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`. If that port is taken, Next.js picks another one and prints it, and the backend must then allow that origin (see the root `README.md`).

`NEXT_PUBLIC_API_BASE_URL` sets where the backend is. It defaults to `http://127.0.0.1:8000/api`, and `.env.example` shows the format. It is read when the app starts or builds, so restart after changing it.

| Command | Purpose |
|---|---|
| `pnpm build` | Production build, including the TypeScript check |
| `pnpm test` | Frontend tests |

The project uses Next.js App Router, React, TypeScript, Tailwind CSS, Lucide and Recharts.

## Pages

**Workspace** holds one scene dropdown per dataset, the video and the error sliders. **Analysis** is one page of simple graphs for the chosen scene, measured live on every error at every tested level, plus what your sliders did. **Applications** has a dropdown of real-world uses, each with which error to reduce first, why, and measured effects on a named scene. Its priorities are engineering judgement. **Test Videos** shows boxes, IDs, forecasts and error sliders on downloaded street videos, with no ADE or FDE because they have no true positions. **Records** lists every stored run, and **Settings** shows the backend connection.

Long recordings are also listed as numbered parts. A part has its own video and all the controls of its recording, and works on Analysis like any scene, but it is not a separate scene: its values are never pooled with other parts, and it has no stored results.

## Architecture

Routes are thin and compose `components/trackvision-app.tsx`. API contracts live in `types/index.ts`, navigation-safe state lives in `stores/experiment-store.tsx`, and all network access goes through `lib/api/`. See `docs/architecture.md` and `docs/api-contract.md`.

## Integrity

Numbers on screen come from stored runs, or from the frozen injectors run live and labelled exploratory. There is no mock data and no fallback: if the backend is unreachable, the screen says so. A missing value is an em dash in a table and a gap in a chart, never zero. Each scene names where its IDs come from. Scenes with no calibration work in video pixels and say so.
