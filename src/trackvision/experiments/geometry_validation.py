"""Local Phase 1 geometry validation of the static scenes that have an image-to-world mapping.

LOCAL ONLY. Dataset permissions are not yet confirmed, so the output is written under the
Git-ignored runs/ directory and is not reported or published.

For ETH and HOTEL it records: the manifest check, the non-singularity check at load (spec 4.3.1),
the H.txt convention check against the dataset's obstacle map and frame bounds,
the world -> image -> world round trip against the real-homography tolerance (spec 11.2), the
orientation of the image-to-world map, the documented and velocity-derived timesteps, the median
step speed against the diagnostic range (spec 14.1 P2; diagnostic, not acceptance) and the
window count (Implementation Plan Phase 1 gate). UNIV, ZARA1 and ZARA2 are listed as BLOCKED:
no image-to-world mapping is available for them, and no pixel-space substitute is produced.

Usage: python -m trackvision.experiments.geometry_validation [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from trackvision import __version__, config, repro
from trackvision.io import ewap
from trackvision.predict.windowing import static_windows

LOCAL_NOTICE = ("LOCAL ONLY - dataset permissions unconfirmed; do not report, publish or commit "
                "these results")
DEFAULT_OUT = config.REPO_ROOT / "runs" / "phase1-geometry-validation-local"
BLOCKED_SCENES = {
    scene: ("BLOCKED: no image-to-world homography in the UCY distribution, and no reliable metric "
            "reference points found in the UCY package or in the published sources searched to "
            "derive one; third-party matrices are not adopted")
    for scene in ("UNIV", "ZARA1", "ZARA2")
}


def step_speeds(tracks, frame_step: int, dt: float) -> np.ndarray:
    """Speeds (m/s) between consecutive samples exactly one annotation step apart."""
    speeds = []
    for t in tracks:
        one_step = np.diff(t.frames) == frame_step
        steps = np.linalg.norm(np.diff(t.positions, axis=0), axis=1)
        speeds.append(steps[one_step] / dt)
    return np.concatenate(speeds) if speeds else np.empty(0)


def _segment_touches(mask: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """Whether the image segment a-b passes over a mask pixel (sampled at <= 1 px spacing)."""
    n = max(2, int(np.ceil(np.linalg.norm(b - a))) + 1)
    p = np.rint(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]).astype(np.int64)
    ok = (p[:, 0] >= 0) & (p[:, 0] < mask.shape[1]) & (p[:, 1] >= 0) & (p[:, 1] < mask.shape[0])
    return bool(np.any(mask[p[ok, 1], p[ok, 0]]))


def convention_check(scene: ewap.Scene, slack_px: float) -> dict:
    """Recompute the evidence for the H.txt convention from the dataset's own files.

    For the project reading (u = column) and the swapped reading it counts annotated points
    outside the closed frame [0, W] x [0, H] (closed, so both 0- and 1-based pixel indices are
    accepted, with `slack_px` for rounding) and one-step path segments crossing obstacle pixels
    of map.png. The project reading is preferred only if it has no more of either count and
    strictly fewer of at least one. The outside-frame count of the project reading is the
    static-scene 'plausible bounds' diagnostic (spec 11.2 runtime assertions).
    """
    width, height = scene.frame_size
    readings, whole_pixel = {}, 0.0
    for label, swap in (("project_u_v", False), ("swapped", True)):
        outside = touching = segments = 0
        for t in scene.tracks:
            uv = scene.transform.world_to_image(t.positions)
            if swap:
                uv = uv[:, ::-1]
            whole_pixel = max(whole_pixel, float(np.max(np.abs(uv - np.rint(uv)))))
            inside = ((uv[:, 0] >= -slack_px) & (uv[:, 0] <= width + slack_px)
                      & (uv[:, 1] >= -slack_px) & (uv[:, 1] <= height + slack_px))
            outside += int(np.sum(~inside))
            for i in np.flatnonzero(np.diff(t.frames) == scene.frame_step):
                segments += 1
                touching += _segment_touches(scene.obstacle_map, uv[i], uv[i + 1])
        readings[label] = {"points_outside_frame": outside, "segments_touching_obstacles": touching,
                           "segments": segments}
    p, s = readings["project_u_v"], readings["swapped"]
    keys = ("points_outside_frame", "segments_touching_obstacles")
    preferred = all(p[k] <= s[k] for k in keys) and any(p[k] < s[k] for k in keys)
    return {
        "frame_size": [width, height],
        "slack_px": slack_px,
        "whole_pixel_max_deviation_px": whole_pixel,
        "readings": readings,
        "project_reading_preferred": preferred,
        "plausible_bounds_points_outside_frame": p["points_outside_frame"],
    }


def validate_scene(scene: ewap.Scene, cfg: config.Config) -> dict:
    geometry = cfg.frozen["geometry"]
    tolerance = float(geometry["roundtrip_tolerance_px"]["real_float64"])
    low, high = (float(v) for v in geometry["median_speed_diagnostic_range_mps"])

    world = np.concatenate([t.positions for t in scene.tracks])
    roundtrip = scene.transform.roundtrip_errors(world)
    image = scene.transform.world_to_image(world)
    signs = scene.transform.orientation_signs(image)
    speeds = step_speeds(scene.tracks, scene.frame_step, scene.sampling_interval_s)
    median = float(np.median(speeds))
    windows = static_windows(scene, cfg)

    return {
        "provenance": scene.provenance,
        "transform_config": scene.transform_config,
        "diagnostics": scene.diagnostics,
        "assumptions": scene.assumptions,
        "nonsingular_at_load": True,     # load_scene raises otherwise
        "convention_check": convention_check(scene, tolerance),
        "condition_number": float(np.linalg.cond(scene.transform.h_img_to_ground)),
        "roundtrip": {"points": int(len(world)), "max_px": roundtrip["px"], "max_m": roundtrip["m"],
                      "tolerance_px": tolerance, "passed": roundtrip["px"] <= tolerance},
        "annotated_points_in_image_px": {"u_min": float(image[:, 0].min()), "u_max": float(image[:, 0].max()),
                                         "v_min": float(image[:, 1].min()), "v_max": float(image[:, 1].max())},
        "orientation": {"negative_jacobian_points": int(np.sum(signs < 0)),
                        "positive_jacobian_points": int(np.sum(signs > 0)),
                        "reading": ("orientation-reversing everywhere: right-handed ground frame, z up"
                                    if np.all(signs < 0) else
                                    "NOT uniformly orientation-reversing: investigate world-frame handedness")},
        "speed_diagnostic": {
            "definition": "all one-step displacements / sampling interval; no moving-pedestrian "
                          "filter applied (the specification does not define one)",
            "steps": int(len(speeds)), "median_mps": median,
            "iqr_mps": [float(np.percentile(speeds, 25)), float(np.percentile(speeds, 75))],
            "diagnostic_range_mps": [low, high], "in_range": low <= median <= high,
            "independence": "the positions were produced by the dataset authors with this H, so this "
                            "checks plausibility of the data, not of our transform handling",
        },
        "windows": {"history_len": windows.history_len, "horizon_len": windows.horizon_len,
                    "stride": windows.stride, "count": len(windows),
                    "tracks_contributing": int(len(np.unique(windows.track_ids)))},
    }


def run(cfg: config.Config, raw_root: Path = ewap.DEFAULT_RAW_ROOT,
        manifest_path: Path | None = ewap.DEFAULT_MANIFEST) -> dict:
    scenes = {name: validate_scene(ewap.load_scene(name, cfg, raw_root, manifest_path), cfg)
              for name in ewap.SCENES}
    return {
        "notice": LOCAL_NOTICE,
        "scenes": scenes,
        "blocked_scenes": BLOCKED_SCENES,
        "passed": all(s["roundtrip"]["passed"] and s["convention_check"]["project_reading_preferred"]
                      for s in scenes.values()),
        "config_hash": cfg.hash,
        "spec": {"version": cfg.meta["spec"]["version"], "sha256": cfg.meta["spec"]["sha256"]},
        "trackvision_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": repro.git_state(config.REPO_ROOT),
        "environment": repro.environment_info(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw-root", type=Path, default=ewap.DEFAULT_RAW_ROOT)
    parser.add_argument("--manifest", type=Path, default=ewap.DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="rewrite an existing report")
    args = parser.parse_args(argv)

    result = config.validate()
    if result.status != config.VALID:
        print(f"configuration is {result.status}: {list(result.errors)}")
        return 2
    report_path = args.out / "report.json"
    if report_path.exists() and not args.force:
        print(f"report exists: {report_path}; use --force to rewrite it")
        return 4
    report = run(result.config, args.raw_root, args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(LOCAL_NOTICE)
    for name, s in report["scenes"].items():
        print(f"{name}: round trip {s['roundtrip']['max_px']:.3e} px (tol {s['roundtrip']['tolerance_px']}) "
              f"{'PASS' if s['roundtrip']['passed'] else 'FAIL'}; median step speed "
              f"{s['speed_diagnostic']['median_mps']:.3f} m/s (in range: {s['speed_diagnostic']['in_range']}); "
              f"windows {s['windows']['count']}; convention check: project reading "
              f"{'preferred' if s['convention_check']['project_reading_preferred'] else 'NOT preferred'} "
              f"{s['convention_check']['readings']}")
    for name, reason in report["blocked_scenes"].items():
        print(f"{name}: {reason}")
    print(f"wrote: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
