"""E5 to E8 and across-scene family AUDC, from stored Phase 4 and E4 results (decision 0006).

Reads stored documents and trains nothing. Every section names its population, source and level:
  synthetic curves   engineering only, SYN-501 to SYN-505, the primary history
  ETH and HOTEL      descriptive and local only, CV and CA, one scene at a time, never pooled
  E4 outputs         synthetic engineering only

E5 is a calculation on synthetic pairs, not P-13. E6 is a dry-run, not the research selection. E7 is descriptive and
is not a requirement. E8 is threshold arithmetic, not model selection. Each research gate stays blocked, with its
reason, in `BLOCKED` and `DEFERRED`; nothing there has a placeholder result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from trackvision import __version__, config, repro
from trackvision.config import Config
from trackvision.experiments import association_provenance
from trackvision.experiments.mot17_trackeval import check_output_location
from trackvision.stats.association_strength import association_summary
from trackvision.stats.crossing import first_crossing
from trackvision.stats.dominant import select_dominant_mode
from trackvision.stats.family import across_scenes
from trackvision.stats.inverse import inverse_requirement

SYNTHETIC, ETHHOTEL = "synthetic_engineering", "ethhotel_descriptive"
LEVELS = {SYNTHETIC: "engineering_only", ETHHOTEL: "descriptive_local_only"}
LABEL = ("ENGINEERING AND DESCRIPTIVE ONLY (E5 to E8, decision 0006). Calculated from stored Phase 4 and E4 results. "
         "No research gate is passed; P-1, P-9, P-13, P-14, H7, RQ3 and RQ8 remain research-blocked. v4.15 Phase 1 "
         "remains BLOCKED and v4.16 is unfrozen.")
FAMILY_NOT_COMPUTED = ("family AUDC is not computed for ETH and HOTEL: LSTM and GRU are not run on real scenes, and a "
                       "family mean over CV and CA alone would omit them; the scenes are never pooled")
BLOCKED = {
    "P-1": "needs private YOLO detections, ByteTrack and BoT-SORT; the AGPL decision is open and none is available",
    "P-9": "needs ETH/UCY video permission, UCY calibration, ground-truth boxes and a real detector and tracker",
    "P-13/H7": "depends on P-9; the scene-cluster bootstrap has no constants",
    "P-14/RQ8": "needs a research dominant-mode selection, which needs valid five-scene P-4 to P-8 results",
    "H9": "static five-scene collision protocol unavailable; collision window rules unresolved",
    "RQ3": "within-domain anchoring unavailable; the section 6.1.1 outcome is not recorded",
    "P-8": "moving_dataset is unresolved",
    "P-10/H8": "protocol decisions open (decision 0004); h8_generation_attempts is null",
    "P-11": "moving_dataset is unresolved",
}
DEFERRED = {
    "E9/section 9.1": "training levels and validation corruption are unresolved",
    "E10/section 3.8": "collision population and window rules are unresolved",
    "section 9.2": "v_max, fixed-length input handling and the false-positive denominator are unresolved",
    "full section 9.3": "runtime selection and misclassification are undefined; only E8 arithmetic exists",
    "recovery": "the clean-baseline population is unresolved",
    "equivalent-severity projection": "belongs to P-9; E4 records diagnostics only",
}


def primary_history(doc: dict, cfg: Config) -> str:
    history = str(cfg.frozen["protocol"]["static"]["history_len"])
    if history not in doc["curves"]["per_history"]:
        raise ValueError(f"the stored analysis has no H = {history} curves")
    return history


def representative_conditions(cfg: Config) -> list[str]:
    """A, B2, C and D: the static representatives. E is left out because it is never compared with them."""
    return [v for k, v in cfg.frozen["analysis"]["representative_conditions"].items() if k != "E"]


def family_across_scenes(doc: dict, cfg: Config) -> dict:
    """Decision 0006, item 11. Refuses a scope whose scenes must not be pooled."""
    if doc["scope"] != SYNTHETIC:
        raise ValueError(f"{doc['scope']} scenes are never pooled")
    table = doc["family_audc"]["per_history"][primary_history(doc, cfg)]
    return {"level": LEVELS[SYNTHETIC], "history_len": int(primary_history(doc, cfg)), "scenes": doc["spec"]["scenes"],
            "order": "valid predictor AUDCs within each scene, minimum two, then across valid scenes",
            "note": "descriptive; synthetic scenes from one generator; not real-world research evidence",
            "by_condition": {c: across_scenes({s: table[s][c] for s in doc["spec"]["scenes"]})
                             for c in representative_conditions(cfg)}}


def dominant_dry_run(family: dict, source: str, cfg: Config) -> dict:
    """E6 on the synthetic family values. ETH and HOTEL have none, and no research target is selected from either."""
    rows = family["by_condition"]
    values = {c: r["value"] for c, r in rows.items()}
    reasons = {c: r["reason"] for c, r in rows.items()}
    result = select_dominant_mode(values, cfg.frozen["analysis"]["dominant_mode_codominance_gap"], reasons)
    return {**result, "population": "SYN-501 to SYN-505, per-history, H = 8", "source_run": source,
            "contributing_scenes": {c: r["n_valid_scenes"] for c, r in rows.items()},
            "undefined_scenes": {c: r["n_undefined_scenes"] for c, r in rows.items()},
            "ethhotel": {"status": "not_evaluated", "reason": FAMILY_NOT_COMPUTED}}


def _inverse_points(curve: dict) -> list:
    return [(r.get("achieved_seed_mean"), r["d_ade"]["mean"]) for r in curve["levels"]]


def inverse_mapping(doc: dict, cfg: Config) -> dict:
    """E7: one entry per scene, predictor, family and q. Scenes are never combined here."""
    curves = doc["curves"]["per_history"][primary_history(doc, cfg)]
    out = {}
    for scene in doc["spec"]["scenes"]:
        for predictor in doc["spec"]["predictors"]:
            for condition in representative_conditions(cfg):
                curve = curves[scene][predictor][condition]
                measure = curve["normalization"].get("measure")
                for q in cfg.frozen["analysis"]["inverse_requirement_q"]:
                    key = f"{scene}/{predictor}/{condition}/q{q}"
                    out[key] = {**inverse_requirement(_inverse_points(curve), q), "severity_measure": measure}
    return {"level": LEVELS[doc["scope"]], "metric": "relative ADE degradation, primary baseline (decision 0004)",
            "not_a_requirement": "descriptive; not a universal engineering safety limit", "entries": out}


def _pairs(predictors) -> list:
    return list(combinations(predictors, 2))


def _scene_points(curves, scene, a, b, condition) -> list:
    rows = zip(curves[scene][a][condition]["levels"], curves[scene][b][condition]["levels"])
    return [(x.get("achieved_seed_mean"), x["ade_m"]["mean"], y["ade_m"]["mean"]) for x, y in rows]


def _across_points(across, a, b, condition, n_scenes) -> list:
    """Across-scene means, usable only where every scene contributes, so the means cover one population."""
    points = []
    for x, y in zip(across[a][condition]["levels"], across[b][condition]["levels"]):
        whole = x["ade_m"]["n_valid"] == n_scenes == y["ade_m"]["n_valid"]
        points.append((x["achieved"]["mean"], x["ade_m"]["mean"] if whole else None, y["ade_m"]["mean"] if whole else None))
    return points


def threshold_crossings(doc: dict, cfg: Config) -> dict:
    """E8: where one predictor's ADE meets another's. No predictor is chosen for anything."""
    history = primary_history(doc, cfg)
    curves = doc["curves"]["per_history"][history]
    scenes, predictors = doc["spec"]["scenes"], doc["spec"]["predictors"]
    per_scene, across = {}, {}
    for a, b in _pairs(predictors):
        for condition in representative_conditions(cfg):
            for scene in scenes:
                per_scene[f"{scene}/{a}-{b}/{condition}"] = first_crossing(_scene_points(curves, scene, a, b, condition))
            if doc["scope"] == SYNTHETIC:
                table = doc["across_scenes"]["per_history"][history]
                across[f"{a}-{b}/{condition}"] = first_crossing(_across_points(table, a, b, condition, len(scenes)))
    out = {"level": LEVELS[doc["scope"]], "metric": "ADE", "interpolation": "linear between adjacent measured points",
           "conditional_on": "predicted windows; read B2 and C with their coverage",
           "not_implemented": "runtime selection, misclassification cost and the section 9.3 interval", "per_scene": per_scene}
    if doc["scope"] == SYNTHETIC:
        out["across_synthetic_scenes"] = across
    return out


def p13_engineering(e4: dict) -> dict:
    """E5: scene-level pairs from one tracker configuration, per predictor. Windows are never the observation."""
    by_scene, official = e4["by_scene"], e4["official_trackeval"]
    scenes = sorted(by_scene)
    out = {}
    for predictor in next(iter(by_scene.values()))["prediction"]["predictors"]:
        pairs = {}
        for s in scenes:
            t = official[s]
            pairs[s] = {"MOTA": t["MOTA"], "IDF1": t["IDF1"], "HOTA": t["HOTA"], "IDSW": t["IDSW"],
                        "ade_m": by_scene[s]["prediction"]["predictors"][predictor]["seed_mean"]["tracker_ade_m"]["mean"]}
        out[predictor] = {**association_summary(pairs), "pairs": pairs}
    return {"level": "engineering_only", "not_p13": True, "population": "synthetic E4 outputs, one tracker configuration",
            "statistical_unit": "scene", "h7_verdict": "not_evaluable", "by_predictor": out}


def _load(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def run(out_dir: Path, cfg: Config, synthetic_analysis: Path, ethhotel_analysis: Path, e4_results: Path) -> dict:
    out_dir = Path(out_dir)
    check_output_location(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; results are never overwritten")
    synthetic, synthetic_sha = _load(synthetic_analysis)
    ethhotel, ethhotel_sha = _load(ethhotel_analysis)
    e4, e4_sha = _load(e4_results)
    if synthetic["scope"] != SYNTHETIC or ethhotel["scope"] != ETHHOTEL or e4.get("scope") != "synthetic_tracker_engineering":
        raise ValueError("inputs must be the synthetic analysis, the ETH/HOTEL analysis and the E4 results, in that order")
    family = family_across_scenes(synthetic, cfg)
    doc = {"label": LABEL, "research_results": False, "association_by_experiment": association_provenance.analysis_field(),
           "inputs": {"synthetic_analysis_sha256": synthetic_sha, "ethhotel_analysis_sha256": ethhotel_sha,
                      "e4_results_sha256": e4_sha},
           "family_across_scenes": family, "ethhotel_family": FAMILY_NOT_COMPUTED,
           "e5_p13_engineering": p13_engineering(e4),
           "e6_dominant_dry_run": dominant_dry_run(family, synthetic_sha, cfg),
           "e7_inverse_mapping": {"synthetic": inverse_mapping(synthetic, cfg), "ethhotel": inverse_mapping(ethhotel, cfg)},
           "e8_threshold_crossings": {"synthetic": threshold_crossings(synthetic, cfg),
                                      "ethhotel": threshold_crossings(ethhotel, cfg)},
           "blocked": BLOCKED, "deferred": DEFERRED}
    out_dir.mkdir(parents=True)
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    (out_dir / "results.json").write_text(text, encoding="utf-8", newline="\n")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"label": LABEL, "results_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "inputs": doc["inputs"],
         "config_hash": cfg.hash, "trackvision_version": __version__, "git": repro.git_state(config.REPO_ROOT),
         "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    return doc


def status() -> dict:
    return {"blocked": BLOCKED, "deferred": DEFERRED}
