# Implementation roadmap

| Stage | State |
|---|---|
| 0. Frontend foundation | Done |
| 1. Read-only connection to stored results (decision 0007) | Done |
| 2. Scene workspace: video, overlays, live exploratory errors, in pixels on UCY and MOT17, with a dropdown per dataset and numbered parts of long recordings (decisions 0008, 0010, 0011, 0012) | Done |
| 2a. Analysis tab with live results, tracker details and E5 to E8 together (decision 0013) | Done |
| 2b. One simple Analysis page for every scene, and YOLO11 boxes for display (decisions 0015, 0016) | Done. YOLO11 runs on MOT17, and ETH, HOTEL and UCY keep annotated boxes |
| 2c. Applications page: priorities as engineering judgement, with measured effects on named scenes (decision 0016) | Done |
| 2d. Test Videos section on downloaded street videos, with no accuracy values (decision 0017) | Done |
| 3. Application impact and mitigation, as research | Blocked on research work; no measurement of any application exists |
| 4. Saved runs started from the interface | Not started; needs its own decision |

The research order is set by `docs/implementation-roadmap.md` at the repository root and by specification v4.15. Platform work adds no research evidence, and no research gate is passed.

Future work should keep the distinction between clean future targets and corrupted observed history, and never expose ADE or FDE without valid ground truth.
