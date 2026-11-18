"""P2-e: synthetic multi-scene ENGINEERING FEASIBILITY of the learned predictors (LSTM, GRU).

SYNTHETIC ENGINEERING ONLY (docs/decisions/0001-p2d-protocol-decisions.md, F). It shows that the training
pipeline runs end to end on a genuine scene-level nested split: leakage checks, deterministic training, early
stopping, hashed checkpoints and evaluation. It is not a research result:
  - no real-world pedestrian generalisation;
  - not the five-fold ETH/UCY protocol, and not P-3, P-12, P-15, RQ6, RQ9 or H6;
  - no ETH, HOTEL or UCY data is read.

Scenes. Five independent U1 waypoint-walker scenes, one per generator seed (SCENE_SEEDS), named
`SYN-<seed>`. They share one generator family (same motion model, speed range and camera), so they are
independent instances, not different scene types. The generator settings are fixture values, not research
choices. Windows come from each scene's own ground-truth tracks under the frozen static protocol (8/12,
stride 1); a window never spans scenes.

Split. One nested split: SYN-501 held out for testing; the validation scene comes from FIXTURE_ROTATION, a
synthetic fixture mapping that is separate from the real-scene research choice `inner_validation_rotation`
(which the fold builder rejects for these scenes). Every configured seed trains LSTM and GRU. CV and CA are
evaluated on the same test windows for reference. Overlapping windows are correlated: descriptive values
for one synthetic test scene, no inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trackvision import __version__, repro
from trackvision.config import Config
from trackvision.experiments.mot17_trackeval import check_output_location
from trackvision.io.manifest import file_sha256
from trackvision.metrics.displacement import ade_fde, displacement_errors, rmse
from trackvision.predict import learned
from trackvision.predict.baselines import constant_acceleration, constant_velocity
from trackvision.predict.folds import nested_split
from trackvision.predict.training import TrainingSettings, split_identity, train_predictor
from trackvision.predict.windowing import static_windows
from trackvision.synthetic.scene import SceneConfig, generate
from trackvision.synthetic.tracks import ground_truth_grid_tracks

NAME = "synthetic-learned-engineering"
LABEL = ("SYNTHETIC ENGINEERING FEASIBILITY ONLY (P2-e). Five U1 synthetic waypoint-walker scenes; not ETH/UCY, "
         "not real-world generalisation, not a five-fold real-scene research result (P-3, P-12, P-15, RQ6, RQ9, "
         "H6). Descriptive values for one synthetic test scene; overlapping windows, no inference. Not a v4.15 "
         "section 11.3 record. v4.15 Phase 1 remains BLOCKED and v4.16 is unfrozen.")
SCENE_SEEDS = (501, 502, 503, 504, 505)          # synthetic fixture identities, not the research seeds
N_FRAMES, N_PEDESTRIANS = 4500, 30
TEST_SCENE = "SYN-501"


@dataclass(frozen=True)
class SyntheticScene:
    """A synthetic scene in the form the windowing and leakage code expects."""
    name: str
    tracks: tuple
    frame_step: int
    generator: dict


def scene_name(seed: int) -> str:
    return f"SYN-{seed}"


def synthetic_scenes(cfg: Config) -> list[SyntheticScene]:
    dt = cfg.frozen["protocol"]["static"]["sampling_interval_s"]
    real = set(cfg.frozen["protocol"]["static"]["scenes"])
    scenes = []
    for seed in SCENE_SEEDS:
        sc_cfg = SceneConfig(seed=seed, n_frames=N_FRAMES, n_pedestrians=N_PEDESTRIANS)
        sc = generate(sc_cfg, dt)
        name = scene_name(seed)
        if name in real:
            raise ValueError(f"synthetic scene name {name!r} collides with a real scene")
        scenes.append(SyntheticScene(name, ground_truth_grid_tracks(sc), sc.frames_per_step,
                                     {"seed": seed, "n_frames": N_FRAMES, "n_pedestrians": N_PEDESTRIANS,
                                      "sampling_interval_s": dt}))
    return scenes


def fixture_rotation(names) -> dict[str, str]:
    """SYNTHETIC FIXTURE mapping: each scene validates on the next name in sorted order."""
    order = sorted(names)
    return {name: order[(i + 1) % len(order)] for i, name in enumerate(order)}


def _scores(pred: np.ndarray, target: np.ndarray) -> dict:
    per_step = displacement_errors(pred, target)
    ade, fde = ade_fde(per_step)
    return {"n_windows": int(len(ade)), "mean_ade_m": float(ade.mean()), "mean_fde_m": float(fde.mean()),
            "mean_rmse_m": float(rmse(per_step).mean()), "mean_error_by_step_m": per_step.mean(axis=0).tolist()}


def run(out_dir: Path, cfg: Config, settings: TrainingSettings | None = None) -> dict:
    """Train LSTM and GRU for every configured seed on the synthetic split and write results to `out_dir`."""
    out_dir = Path(out_dir)
    check_output_location(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; engineering runs are never overwritten")
    scenes = synthetic_scenes(cfg)
    windows = [static_windows(s, cfg) for s in scenes]
    rotation = fixture_rotation(w.scene for w in windows)
    split = nested_split(windows, TEST_SCENE, rotation)
    seeds = list(cfg.research_choice("seeds"))
    p = cfg.frozen["predictors"]
    horizon = split.test.horizon_len

    runs = []
    for cell in learned.CELLS:
        for seed in seeds:
            r = train_predictor(split, scenes, cell, seed, cfg, out_dir / "checkpoints", LABEL, settings)
            m = r.metadata
            runs.append({"cell": cell, "seed": seed, "checkpoint": r.checkpoint_path.name,
                         "checkpoint_sha256": r.checkpoint_sha256, "state_sha256": m["state_sha256"],
                         "init_state_sha256": m["init_state_sha256"],
                         "n_parameters": m["model"]["architecture"]["n_parameters"], "best_epoch": m["best_epoch"],
                         "best_validation_ade_m": m["best_validation_ade_m"], "epochs_run": m["epochs_run"],
                         "stop_reason": m["stop_reason"], "final_lr": m["history"][-1]["lr"],
                         "test": _scores(learned.predict_positions(r.model, split.test.observed), split.test.target)})
    baselines = {
        "CV": _scores(constant_velocity(split.test.observed, horizon, p["cv_min_final_observations"]), split.test.target),
        "CA": _scores(constant_acceleration(split.test.observed, horizon, p["ca_min_final_observations"]),
                      split.test.target)}
    summary = {}
    for cell in learned.CELLS:
        vals = np.array([r["test"]["mean_ade_m"] for r in runs if r["cell"] == cell])
        fdes = np.array([r["test"]["mean_fde_m"] for r in runs if r["cell"] == cell])
        summary[cell] = {"seeds": seeds, "test_ade_mean_over_seeds_m": float(vals.mean()),
                         "test_ade_sd_over_seeds_m": float(vals.std(ddof=1)) if len(vals) > 1 else None,
                         "test_fde_mean_over_seeds_m": float(fdes.mean())}

    results = {
        "name": NAME, "label": LABEL, "evaluation_type": "synthetic_engineering_feasibility",
        "trackvision_version": __version__, "config_hash": cfg.hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenes": [{"name": s.name, "generator": s.generator, "n_tracks": len(s.tracks),
                    "n_windows": len(w)} for s, w in zip(scenes, windows)],
        "fixture_rotation": rotation, "split": split_identity(split),
        "settings": (settings or TrainingSettings.from_config(cfg)).__dict__, "seeds": seeds,
        "runs": runs, "baselines_on_test": baselines, "seed_summary_descriptive": summary,
        "notes": ["Learned models are not required to beat CV; no ranking is claimed.",
                  "Seed spread describes training stability on one synthetic test scene only."],
        "environment": repro.environment_info(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in out_dir.rglob("*") if p.is_file())
    manifest = {"name": NAME, "label": LABEL, "config_hash": cfg.hash,
                "files": {p.relative_to(out_dir).as_posix(): file_sha256(p) for p in files}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results
