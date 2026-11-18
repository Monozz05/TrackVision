# TrackVision frontend architecture

TrackVision is a Next.js App Router workspace for studying how tracking errors propagate into trajectory prediction. It has four main pages: **Workspace** (pick a scene, watch its video, set errors), **Analysis** (simple graphs of every error on the chosen scene), **Applications** (which error matters most for a chosen use) and **Test Videos** (the same working on downloaded street videos). Records and Settings sit behind small links in the top bar.

```
Python pipeline → runs/ ──────────────────┐
datasets → data/cache (browser video) ────┤→ FastAPI (src/trackvision/platform) → lib/api → screens
frozen injectors, called live ────────────┘
```

The frontend contains no ML algorithm, no calculation of a research metric, no database and no mock data.

## Layers
- `app/`: thin routes for `/workspace`, `/analysis`, `/applications`, `/test-videos`, `/experiments` and `/settings`.
- `components/workspace.tsx`: the scene's video, error sliders and the live result.
- `components/video-stage.tsx`: the video, its controls, and the canvas overlay kept in step with the frame on screen.
- `components/analysis.tsx`: one page of graphs for the chosen scene, the result of the current sliders, and two saved project results.
- `components/applications.tsx` and `config/applications.ts`: the Applications page. Its ranks and reasons are hand-written engineering judgement in that one file, and its numbers come from the live sweep.
- `components/test-videos.tsx`, `components/test-video-stage.tsx` and `lib/test-videos.ts`: the Test Videos page, kept apart from the Workspace. It has its own state, player and API calls, and draws with `lib/overlay.ts` as it is, adding only its forecast lines.
- `components/sweep.tsx` and `lib/api/use-sweep.ts`: follow a scene's sweep until it finishes, with progress, failure and retry in one place.
- `components/trackvision-app.tsx`: the shell, the scene dropdowns, shared pieces, Records and Settings.
- `components/charts.tsx`: Recharts wrappers. Series colours belong to the entity.
- `lib/overlay.ts`: drawing boxes, IDs and trails on the canvas. Each ID has one colour. Positions between samples are interpolated for display only.
- `lib/api/`: `client.ts` is the only network code, `services.ts` holds the typed calls, and `use-resource.ts` tracks loading, empty and error states and discards stale responses.
- `stores/`: the selected scene, slider values, last live result and backend connection state.
- `tests/`: Vitest and Testing Library, run with `pnpm test`. Fixtures there are test-only.

## Rules the code keeps
- **No fallback data.** A failed request becomes an error state with a retry. Nothing substitutes for a result.
- **Missing stays missing.** `null` renders as an em dash in tables and as a gap in charts, never as zero.
- **Live is not stored.** Live values carry an exploratory label and are never presented as research results.
- **Labels travel with results.** An evidence line sits beside every result. The full stored label is in Records.
- **Units are honest.** Every live value is shown in its scene's unit, metres or pixels. Position noise resets when the unit changes, and scenes in pixels carry a caution.
- **Analysis never waits on the Workspace.** It calculates from the sliders in the store, so leaving the Workspace early or changing scene cannot lose a result.
- **A sweep runs in the background.** The backend answers at once with the levels finished so far, and the page asks again until it is done, so a long scene never holds a request open.
- **Judgement is labelled.** Every view of Applications says its priorities are engineering judgement, and no measured number lives in its text.
- **No truth, no accuracy.** Test videos have no true positions, so the backend sends no ADE or FDE and the page shows a dash with the reason. Lines drawn from later tracker output are named as that, never as truth.
- **A part is an ordinary scene to the frontend.** A numbered part of a recording has its own video, and frames that count from its start, so the player and overlay need no special case. Its caution says it is not a separate scene.

State resets on a full reload.
