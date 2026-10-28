# Implementation roadmap

0. Frontend foundation: routes, reusable UI, types, state, mock services.
1. Data and tracking backend.
2. Trajectory extraction and coordinate handling.
3. Clean prediction.
4. Controlled error injection.
5. Error visualization.
6. Analysis and comparison.
7. Application impact and mitigation.
8. Final integration and testing.

Future coding agents should implement one phase at a time behind the existing API contracts. Preserve the distinction between clean future targets and corrupted observed history, and never expose ADE/FDE without valid ground truth.
