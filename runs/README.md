# runs/

One folder per experiment ID, written by `python -m trackvision run`. Everything here except this
README is Git-ignored.

**Experiment ID:** `<name>-<first 10 hex of the config hash>-s<seed|na>`. It depends only on the
configuration and the seed, so re-running an experiment ID with the same configuration
reproduces its identity.

**What each run folder holds in Phase 0A:**
- `config.resolved.yaml`: the full merged configuration.
- `fixture_summary.json`: the smoke experiment's synthetic checks.
- `manifest.json`: provenance, namely timestamp, Git state, environment and spec hash.

The experiment record schema (spec v4.15 §11.3) is defined in `src/trackvision/experiments/record.py`; no runner writes it yet.

An existing run folder is never overwritten unless `--force` is passed, and `--force` rewrites
only the run's own files.
