# Decision 0007: Read-only results platform

- **Date:** 2026-11-01
- **Decided by:** the project owner, who asked for the existing frontend to be connected to stored pipeline outputs.
- **Governing specification:** v4.15 (unchanged). Nothing here amends it or `configs/base.yaml`.
- **Timing, stated plainly.** Specification Part 13 says the interface "does not begin before the P6 checkpoint", and
  section 14.2 defines that checkpoint as Tier 0 complete. **Tier 0 is not complete and no research gate is passed.**
  This work starts ahead of that order at the project owner's instruction. It therefore adds no research evidence, and
  it must not be read as the checkpoint having passed.

## Approved scope

1. **A view over stored outputs.** `src/trackvision/platform/` serves stored run records through FastAPI, and `web/`
   displays them. The stored records under `runs/` stay the single source of truth.
2. **Read-only.** Every route is a GET. There is no write route, no job execution, no upload and no database.
3. **No research computation.** The API selects and reshapes recorded values. It computes no metric, mean or
   statistic. A missing value stays null, with its recorded reason, and the frontend shows an em dash.
4. **One-way dependency.** Research code never imports the platform (specification section 11.1), and the platform is
   outside the code that feeds `source_sha256`, so it cannot invalidate a stored run. Tests enforce both.
5. **Claim levels.** Interface text stays at L1 and L2 (specification section 1.2). It makes no causal, generalising
   or recommending statement.
6. **Evidence labels.** Each run shows the label it recorded, plus an evidence level taken from the protocol that
   produced it. No current kind of run maps to a research level, and the research count comes only from stored
   `research_results` values.
7. **Local only.** The API binds to `127.0.0.1`, because stored ETH and HOTEL results are labelled local only while
   dataset permissions are unconfirmed.
8. **Integrity.** A run that fails its recorded hashes or holds malformed records is listed as invalid, and its
   results are not served. One failed cell digest fails the whole read of that run's cells.

## Project choices that are not specification requirements

- **JSON instead of the experiment database.** Specification section 11.1 shows a database layer. The owner asked for
  stored JSON to remain the source of truth, so the registry reads run folders directly.
- **Experiment IDs.** An API ID is the run folder's path below `runs/`, joined with `~`. The recorded experiment ID
  stays available as `record_id`.
- **Latest in series.** `latest_in_series` means the newest complete run of its kind and scope. No record marks a run
  as superseded, so the API never reports that status.

## Not in this scope

Write APIs, live experiment execution, video upload and processing, mitigation, collision operating curves, adaptive
model selection, user accounts, and report generation. E9, E10, sections 9.1, 9.2 and full 9.3 stay deferred, and the
Application Impact screen says that no measurement exists.

## Open

- Whether the owner formally amends the order in Part 13 for later platform work.
- Whether E4 and E5 may record a mechanism-only `real_tracker_iou` value (decision 0006). They stay null until then,
  and the platform shows null with the stored blocker.
