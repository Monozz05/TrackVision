# Frontend mock values that differ from specification v4.15

The files in `web/` were moved here byte-identical from the repository root in Phase 0A.
They were not edited. This note records the places where the frontend's **mock** values
disagree with the frozen research specification (`docs/spec/trackvision_specification_v4_15.md`),
so nobody mistakes them for protocol values. Fixing them belongs to the platform phase
(Implementation Plan "Phase 7 — The platform"); the research code never reads them.

| Location | Frontend mock value | Specification v4.15 |
|---|---|---|
| `types/index.ts`, `createInitialState` → `predictionConfiguration.historyLength` | `12` | Static primary protocol is `history_len = 8`, `horizon_len = 12` (§5.3). |
| `types/index.ts`, `mockDatasets` entries `eth` and `hotel` | `fps: 25` | Sampling interval is 0.4 s (2.5 Hz) (§5.3). The Implementation Plan (Phase 1) notes the ETH video header declares 25 fps but the timestep must be derived from the data. |
| `types/index.ts`, `mockDatasets` entry `eth` | `scene: 'Hotel'` | ETH and HOTEL are separate static scenes (§5.3.1: ETH · HOTEL · UNIV · ZARA1 · ZARA2). |
| `types/index.ts`, `mockDatasets` entry `mot17` | `MOT17-04` with `regime: 'Moving camera'` | MOT17-04 is in the project's **static** partition {02, 04, 09} (§6.2). |
| `types/index.ts`, `ErrorType` / `errorFamilies` | Five families A–E | Five families but **eight injection conditions**: A, B1, B2, C, D, E1, E2, E3 (§3.5). Experiment records store the condition. |
| `types/index.ts`, `experimentId` values such as `TV-2026-001` | Free-form IDs | Research runs use deterministic IDs derived from the experiment name, configuration hash and seed (see the root `README.md`). |

The frozen values themselves live only in `configs/base.yaml`.
