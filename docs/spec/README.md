# Governing documents

> **Not redistributed.** This repository is public, so the two governing documents below are
> deliberately **not** tracked. Each collaborator keeps a local copy in this folder, where
> `.gitignore` keeps it out of Git and `scripts/check_staged.py` blocks it if it is ever staged.
> Do not commit or publish them without confirmed redistribution permission.

| Document | Role | SHA-256 |
|---|---|---|
| `trackvision_specification_v4_15.md` | Research specification v4.15, research core **frozen**. Governs objectives, protocols, definitions, datasets, experiments and acceptance criteria. Where any earlier version conflicts, v4.15 governs. | `19216cdae477bf9b5ae50644e1db802a2983fd4b23bf4991ca9d186a7c5495ce` |
| `TrackVision Implementation Plan.pdf` | Implementation Plan, Oct 2, 2026. Governs the build order and repository layout. Its phase order differs from spec §14.1 by moving the detector and tracker later; it changes no experiment. | `b9cf20fcdd68d35613cd7b31754992aeb2357ba40a923b8b969e77aa5d865b4d` |

## Setting up your local copy

1. Get both documents from the project owner.
2. Place them in this folder (`docs/spec/`) under exactly the file names above.
3. Check that each file's SHA-256 matches the table. In PowerShell:

   ```powershell
   Get-FileHash -Algorithm SHA256 docs\spec\trackvision_specification_v4_15.md
   ```

   The specification's hash is also recorded in `configs/base.yaml` (`meta.spec.sha256`).

**Without a matching specification, nothing runs.** Configuration validation reports
`INVALID` and names this file, so no experiment can run. The test suite also stops at startup
with one message, rather than skipping tests. This is deliberate: almost every check depends on
the verified specification.

## Rules

- These files are never edited in place. A specification error found during implementation is
  resolved by a new specification version with a change-log entry, never by a silent edit.
- When the code and the specification disagree, the code is wrong by default.
- Earlier specification versions (v4.0–v4.14 and the pre-v4 drafts) are superseded and are not
  kept here.
