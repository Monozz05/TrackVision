"""Trajectory-error injection (A, B1, B2, C, D) on the real ETH and HOTEL ground-truth trajectories
(EWAP), one scene per run, with CV prediction (R2).

TRAJECTORY-ONLY and LOCAL ONLY. Real-annotation trajectory-injection experiments: the controlled
injectors corrupt the dataset's own annotated world trajectories; no video is read and no detector or
tracker is evaluated, so nothing here is a real-tracker result. The dataset licences are unresolved, so
runs are saved outside the repository (or under the Git-ignored runs/). Not a v4.15 section 11.3 record,
not gate evidence, no inference: stride-1 windows overlap and are correlated; ETH and HOTEL are never
pooled. Injection seeds (0, 1, 2 by default) are provisional feasibility values.

Reuse, unchanged. Scene loading, source hashing, read-only clean tracks and the clean CV baseline come
from R1 (`ewap_baseline.run_baseline`); A, B1, B2 and C are evaluated by U7's `evaluate_condition` and D
by U8's `evaluate_switch` (window policy P), with their approved levels, seeds and semantics. Every
injection starts from a fresh copy of the clean trajectories; targets are always the clean windows'
targets. The clean-trajectory and target hashes are checked unchanged after the sweep, and D's
pre-injection denominator is checked identical across levels and seeds.

Real-data structure checked before R2 (read-only, recorded in the R2 report): EWAP ids are unique
positive integers, every within-track frame difference equals the annotation step, and in ETH, whose
tracks start on three different frame phases, no pedestrian pair that comes within D_MAX at co-temporal
annotations lies on different phases, so U8's same-timestep proximity rule loses no pair.

Timing. The ETH annotation-step / video-container discrepancy is carried, unresolved, in every result and
manifest (ewap_baseline.TIMING_NOTES). Frame numbers are never changed and nothing is resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments import ewap_baseline as eb
from trackvision.experiments import injection_feasibility as u7
from trackvision.experiments import switch_feasibility as u8
from trackvision.experiments.geometry_validation import LOCAL_NOTICE
from trackvision.io import ewap
from trackvision.io.manifest import file_sha256

NAME = "ewap-injection"
CONDITIONS = u7.CONDITIONS + (u8.CONDITION,)                # A, B1, B2, C, D
LABEL = ("Real-annotation trajectory-injection experiment on EWAP ground truth (R2): controlled injectors A, B1, "
         "B2, C and D applied to the dataset's annotated world trajectories, CV prediction, clean targets. Not a "
         "detector or tracker evaluation and not a real-tracker result. LOCAL ONLY: dataset licences are "
         "unresolved. Not a v4.15 section 11.3 record; not gate evidence; descriptive only (overlapping, "
         "correlated windows); never pooled across scenes. v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
C_VS_D_NOTE = ("C ends an identity at a sampled timestep and gives the rest of the track a new, unused identity "
               "(new ids, broken continuity); D exchanges two existing labels from a timestep on (no new ids, no "
               "gaps). Their effects are reported separately and are not assumed equivalent.")


@dataclass(frozen=True)
class _Rows:
    """The two fields the U7 and U8 window CSV writers read."""
    population: u7.Population
    results: tuple


@dataclass(frozen=True)
class EwapInjectionRun:
    experiment_id: str
    scene_name: str
    settings: dict
    settings_hash: str
    config_hash: str
    baseline: eb.BaselineRun         # R1 clean baseline on the same scene (all eligible windows)
    clean_tracks_sha256: str
    targets_sha256: str
    population: u7.Population
    results: tuple                   # U7 ConditionResult (A-C) and U8 SwitchResult (D), by condition, level, seed
    d_eligible_track_frames: int | None


def run_settings(cfg: Config, baseline: eb.BaselineRun, seeds, conditions) -> dict:
    injection = cfg.frozen["injection"]
    return {
        **baseline.settings,
        "evaluation_type": eb.EVALUATION_TYPE,
        "injection_seeds": [int(s) for s in seeds],
        "injection_seeds_status": "provisional feasibility values; not the research choice 'seeds'",
        "conditions": {c: {"grid": injection["conditions"][c]["grid"], "levels": u7.levels(cfg, c)} for c in conditions},
        "base": "clean EWAP ground-truth world trajectories (obsmat pos_x, pos_y), unchanged; fresh copy per injection",
        "random_numbers": "U7 / U8 conventions: SeedSequence([seed, family stream, ped_id]); A=1, B=2, C=3, D=4; "
                          "shared across levels; B1 and B2 share dropout masks",
        "fragmentation_convention": "U7: split after sample k (1 <= k <= n - 2); new identity from k + 1",
        "b1_reconstruction": u7.B1_NOTE,
        "d_max_m": float(injection["d_max_m"]),
        "d_cooldown": u8.check_cooldown(cfg) if u8.CONDITION in conditions else injection["d_cooldown"],
        "d_segment": "the whole scene (approved U8-1a)",
        "d_window_policy": u8.POLICY_NOTE,
        "d_naming": u8.NAMING_NOTE,
        "c_vs_d": C_VS_D_NOTE,
        "timing_note": eb.TIMING_NOTES[baseline.scene_name],
    }


def run_ewap_injection(cfg: Config, scene_name: str, raw_root: Path = ewap.DEFAULT_RAW_ROOT,
                       manifest_path: Path | None = ewap.DEFAULT_MANIFEST, seeds=u7.PROVISIONAL_SEEDS,
                       conditions=CONDITIONS) -> EwapInjectionRun:
    unknown = set(conditions) - set(CONDITIONS)
    if unknown:
        raise ValueError(f"unknown conditions {sorted(unknown)}; R2 runs {CONDITIONS}")
    baseline = eb.run_baseline(cfg, scene_name, raw_root, manifest_path)      # clean tracks set read-only here
    clean, frame_step = baseline.scene.tracks, baseline.scene.frame_step
    win = baseline.evaluation.windows
    win.target.setflags(write=False)
    clean_digest, target_digest = u7.tracks_digest(clean), u7._digest(win.target)
    pop = u7.population(clean, win, baseline.evaluation.per_step_error)

    results = []
    for c in (c for c in CONDITIONS if c in conditions):
        for level in u7.levels(cfg, c):
            for s in seeds:
                if c == u8.CONDITION:
                    results.append(u8.evaluate_switch(clean, pop, level, s, cfg, frame_step))
                else:
                    results.append(u7.evaluate_condition(clean, pop, c, level, s, cfg))
    if u7.tracks_digest(clean) != clean_digest:
        raise u7.InjectionIntegrityError("clean trajectories changed during the sweep")
    if u7._digest(win.target) != target_digest:
        raise u7.InjectionIntegrityError("clean targets changed during the sweep")
    d_dens = {r.summary["achieved"]["eligible_track_frames"] for r in results if r.summary["condition"] == u8.CONDITION}
    if len(d_dens) > 1:
        raise u7.InjectionIntegrityError(f"D pre-injection denominator varies across levels/seeds: {d_dens}")

    settings = run_settings(cfg, baseline, seeds, conditions)
    settings_hash = config.config_hash({"config_hash": cfg.hash, "settings": settings})
    return EwapInjectionRun(repro.make_experiment_id(f"{NAME}-{scene_name.lower()}", settings_hash, None), scene_name,
                            settings, settings_hash, cfg.hash, baseline, clean_digest, target_digest, pop,
                            tuple(results), d_dens.pop() if d_dens else None)


def _mean_by_step(results, key: str) -> list | None:
    """Mean over seeds of each seed's per-step mean error (seeds without predicted windows are skipped)."""
    rows = [r.summary["accuracy_on_predicted_windows"][key]["mean_error_by_step_m"] for r in results]
    rows = [v for v in rows if v is not None]
    return np.mean(np.asarray(rows), axis=0).tolist() if rows else None


def _condition_block(results, condition: str) -> list[dict]:
    aggregate = u8._switch_aggregate if condition == u8.CONDITION else u7._seed_aggregate
    block = []
    for level in sorted({r.level for r in results}):
        rows = [r for r in results if r.level == level]
        block.append({
            "requested": rows[0].summary["requested"],
            "per_seed": [r.summary for r in rows],
            "across_seeds": {
                **aggregate(rows),
                "injected_mean_error_by_step_m": _mean_by_step(rows, "injected"),
                "clean_same_windows_mean_error_by_step_m": _mean_by_step(rows, "clean_same_windows"),
                "by_step_note": "mean over seeds of each seed's per-step mean over its predicted windows",
            },
        })
    return block


def results_document(run: EwapInjectionRun) -> dict:
    base = eb.results_document(run.baseline)
    conditions = {}
    for c in CONDITIONS:
        rows = [r for r in run.results if r.summary["condition"] == c]
        if rows:
            conditions[c] = _condition_block(rows, c)
    return {
        "label": LABEL,
        "notice": LOCAL_NOTICE,
        "evaluation_type": eb.EVALUATION_TYPE,
        "experiment_id": run.experiment_id,
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "settings": run.settings,
        "scene": run.scene_name,
        "clean_tracks_sha256": run.clean_tracks_sha256,
        "targets_sha256": run.targets_sha256,
        "dataset": {"tracks": base["summary"]["n_tracks"],
                    "observations": int(sum(len(t.frames) for t in run.baseline.scene.tracks)),
                    "eligible_windows": base["summary"]["n_windows"],
                    "pedestrians_contributing": base["summary"]["n_pedestrians_contributing"],
                    "track_lengths": base["track_lengths"]},
        "clean_reference_all_eligible": base["summary"],
        "d_eligible_track_frames": run.d_eligible_track_frames,
        "conditions": conditions,
        "c_vs_d": C_VS_D_NOTE,
        "timing": base["timing"],
        "assumptions": base["assumptions"],
        "not_used": base["not_used"],
    }


def save_run(out_dir: Path, run: EwapInjectionRun, cfg: Config) -> Path:
    """Write the run to a new directory `out_dir`; raise FileExistsError if it exists."""
    out = Path(out_dir)
    eb.check_output_location(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; results are never overwritten")
    d = tuple(r for r in run.results if r.summary["condition"] == u8.CONDITION)
    abc = tuple(r for r in run.results if r.summary["condition"] != u8.CONDITION)
    files = {
        "results.json": u7._json(results_document(run)),
        "windows_abc.csv": u7.windows_csv(_Rows(run.population, abc)),
        "windows_d.csv": u8.windows_csv(_Rows(run.population, d)),
        "config.resolved.yaml": yaml.safe_dump(cfg.canonical_dict(), sort_keys=True, allow_unicode=True),
    }
    out.mkdir(parents=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    manifest = {
        "experiment_id": run.experiment_id,
        "experiment_name": NAME,
        "label": LABEL,
        "notice": LOCAL_NOTICE,
        "evaluation_type": eb.EVALUATION_TYPE,
        "dataset": ewap.DATASET,
        "scene": run.scene_name,
        "injection_seeds": run.settings["injection_seeds"],
        "conditions": run.settings["conditions"],
        "config_hash": run.config_hash,
        "settings_hash": run.settings_hash,
        "dataset_manifest_hash": run.settings["dataset_manifest_hash"],
        "manifest_verified": run.settings["manifest_verified"],
        "source_sha256": run.settings["source_sha256"],
        "clean_tracks_sha256": run.clean_tracks_sha256,
        "targets_sha256": run.targets_sha256,
        "timing_note": eb.TIMING_NOTES[run.scene_name],
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
        "files": {name: file_sha256(out / name) for name in files},
    }
    (out / "manifest.json").write_text(u7._json(manifest), encoding="utf-8", newline="\n")
    return out
