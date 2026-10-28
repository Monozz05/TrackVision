# data/

Datasets are never committed. Everything here except this README is Git-ignored.

| Folder | Contents |
|---|---|
| `raw/` | Downloaded datasets, exactly as obtained (ETH/UCY, MOT17, the selected moving-camera dataset) |
| `interim/` | Derived trajectories and windows |
| `cache/` | Detection cache and other regenerable intermediates |
| `external/` | Third-party files such as published homographies |

**Checksum policy.** Each dataset gets a manifest that lists every source file with its SHA-256.
The manifest hash is recorded in each experiment record as `dataset_manifest_hash` (spec v4.15 §11.3).
Downloads, the dataset manifests and the static raw-video gate (§6.1.1) belong to Phase 0B.
Nothing has been downloaded in Phase 0A.
