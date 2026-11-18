"""Analysis of a stored sweep: everything here is computed from run-store records and written to analysis.json.

For each population, history, scene and predictor, one curve per condition holds raw per-seed values, seed
aggregates, relative degradation, normalized severity and AUDC. The order is the fixed one: seeds within a
scene/fold, then across scenes. Primary AUDC uses ADE0 on all eligible windows; the paired-window AUDC is the
sensitivity check (decision 0004, section 2). A failed or missing cell stays undefined, with its reason.

Scopes differ. Synthetic scenes are also summarized across scenes, descriptively. ETH and HOTEL are never pooled,
and family AUDC is not computed for them because the predictor set would omit LSTM and GRU.

P-15 follows decision 0005: a history needs a valid AUDC in every fold of its population to qualify, and B2 and C
carry their ordinary and paired AUDC, coverage, window counts and fold counts together. Conditional error on
unchanged surviving inputs says nothing about history sufficiency. A run of the revised common-population companion
holds that population alone, so only the P-15 sections exist for it.
"""

from __future__ import annotations

import json
from pathlib import Path

from trackvision.config import Config
from trackvision.experiments.evaluation import CONDITIONS
from trackvision.experiments.injection_feasibility import levels
from trackvision.experiments.store import COMPLETE, FAILED, UNREALIZED, RunStore
from trackvision.experiments.sweep import (COMMON, COMMON_TRAINED, ETHHOTEL, LABELS, PER_HISTORY, POPULATION_LABELS,
                                           blocked_experiments)
from trackvision.inject.severity import UNCOMPUTABLE, normalize_severity
from trackvision.io.atomic import atomic_write_text
from trackvision.stats import hypotheses
from trackvision.stats.aggregate import aggregate, relative_degradation
from trackvision.stats.audc import AudcResult, audc, family_audc, robustness
from trackvision.stats.history import minimum_history
from trackvision.stats.monotonicity import check_monotonicity

CONDITIONAL_ERROR = ("B2", "C")
CONDITIONAL_NOTE = ("Conditional-error AUDC for B2 and C does not establish history sufficiency when the surviving "
                    "predicted inputs are unchanged. A negative AUDC is not evidence of improved robustness without "
                    "examining the population and coverage.")


def _index(store: RunStore) -> tuple[dict, dict]:
    cells, failures = {}, {}
    for r in store.records():
        i = r.identity
        if "condition" not in i:
            if r.status == FAILED:
                failures[r.key] = {"identity": i, "failure": r.failure}
            continue
        key = (i["population"], i["history_len"], i["scene"], i["predictor"], i["condition"], i["requested_level"], i["seed"])
        cells[key] = r
    return cells, failures


def _cell(cells, pop, history, scene, predictor, condition, level, seed) -> tuple[dict | None, str | None]:
    record = cells.get((pop, history, scene, predictor, condition, level, seed))
    if record is None:
        return None, "cell_missing"
    if record.status not in (COMPLETE, UNREALIZED):
        return None, f"cell_{record.status}"
    return record.result, None


def _metric(result, name):
    return None if result is None else result["metrics"][name]


def _level_row(cells, key, condition, level, seeds, clean) -> dict:
    pop, history, scene, predictor = key
    ade, fde, rms, coverage, d_ade, d_fde, d_paired, diff = {}, {}, {}, {}, {}, {}, {}, {}
    why, steps, counts = {}, {}, {}
    for seed in seeds:
        result, problem = _cell(cells, pop, history, scene, predictor, condition, level, seed)
        ade[seed], fde[seed], rms[seed] = _metric(result, "ade_m"), _metric(result, "fde_m"), _metric(result, "rmse_m")
        coverage[seed] = _metric(result, "prediction_coverage")
        ade0, fde0 = clean[seed]
        rel, rel_fde = relative_degradation(ade[seed], ade0), relative_degradation(fde[seed], fde0)
        d_ade[seed], d_fde[seed], diff[seed] = rel.value, rel_fde.value, rel.absolute_difference
        same = None if result is None else result["metrics"]["clean_same_windows"]
        d_paired[seed] = relative_degradation(ade[seed], None if same is None else same["ade_m"]).value
        why[seed] = problem or (None if result is None else result["metrics"]["undefined_reason"]) or rel.reason
        steps[seed] = _metric(result, "de_by_step_m")
        counts[seed] = {"eligible": _metric(result, "eligible_windows"), "predicted": _metric(result, "predicted_windows")}
    usable = [s for s in steps.values() if s is not None]
    return {"requested": level, "ade_m": aggregate(ade, why).as_dict(), "fde_m": aggregate(fde, why).as_dict(),
            "rmse_m": aggregate(rms, why).as_dict(), "prediction_coverage": aggregate(coverage, why).as_dict(),
            "d_ade": aggregate(d_ade, why).as_dict(), "d_fde": aggregate(d_fde, why).as_dict(),
            "d_ade_paired": aggregate(d_paired, why).as_dict(), "ade_absolute_difference": aggregate(diff, why).as_dict(),
            "de_by_step_m": [sum(c) / len(usable) for c in zip(*usable)] if usable else None,
            "windows_by_seed": {str(s): c for s, c in counts.items()}, "undefined_reason_by_seed": {str(s): r for s, r in why.items()}}


def _reason(row: dict, name: str) -> str | None:
    return None if row[name]["mean"] is not None else (sorted(set(row[name]["undefined"].values())) or ["undefined"])[0]


def _curve(cells, key, condition, grid, seeds, clean) -> dict:
    pop, history, scene, predictor = key
    achieved, measure = {}, None
    for level in grid:
        achieved[level] = {}
        for seed in seeds:
            result, _ = _cell(cells, pop, history, scene, predictor, condition, level, seed)
            block = None if result is None else result["achieved"]
            achieved[level][seed] = None if block is None else block["value"]
            measure = measure or (None if block is None else block["measure"])
    rows = {level: _level_row(cells, key, condition, level, seeds, clean) for level in grid}
    try:
        norm = normalize_severity(scene, condition, achieved, measure)
    except ValueError as exc:
        keys = ("status", "reason")
        return {"normalization": dict(zip(keys, ("uncomputable", str(exc)))), "levels": list(rows.values()),
                "audc": {"status": "uncomputable", "value": None, "reason": str(exc)},
                "audc_paired": {"status": "uncomputable", "value": None, "reason": str(exc)},
                "robustness": robustness(None), "achieved": _keys(achieved),
                "_audc": AudcResult(UNCOMPUTABLE, None, str(exc))}
    primary = audc(norm, {lv: rows[lv]["d_ade"]["mean"] for lv in grid}, {lv: _reason(rows[lv], "d_ade") for lv in grid})
    paired = audc(norm, {lv: rows[lv]["d_ade_paired"]["mean"] for lv in grid},
                  {lv: _reason(rows[lv], "d_ade_paired") for lv in grid})
    for level in grid:
        rows[level]["s_norm"] = norm.normalized[level] if norm.normalized else None
        rows[level]["achieved_seed_mean"] = norm.seed_mean[level] if norm.seed_mean else None
    return {"normalization": norm.as_dict(), "levels": [rows[lv] for lv in grid], "audc": primary.as_dict(),
            "audc_paired": paired.as_dict(), "robustness": robustness(primary.value), "achieved": _keys(achieved),
            "_audc": primary}


def _keys(table: dict) -> dict:
    return {str(lv): {str(s): v for s, v in row.items()} for lv, row in table.items()}


def _clean(cells, key, seeds) -> tuple[dict, dict]:
    pop, history, scene, predictor = key
    out, by_seed, why = {}, {}, {}
    table = {n: {} for n in ("ade_m", "fde_m", "rmse_m", "prediction_coverage")}
    for seed in seeds:
        result, problem = _cell(cells, pop, history, scene, predictor, "clean", None, seed)
        for name in table:
            table[name][seed] = _metric(result, name)
        by_seed[seed] = (table["ade_m"][seed], table["fde_m"][seed])
        why[seed] = problem or (None if result is None else result["metrics"]["undefined_reason"])
    for name, row in table.items():
        out[name] = aggregate(row, why).as_dict()
    return out, by_seed


def _across(values: dict, reasons: dict | None = None) -> dict:
    reasons = reasons or {}
    return aggregate(values, {k: reasons.get(k) or "uncomputable or undefined" for k, v in values.items() if v is None}).as_dict()


def _monotonicity(cells, spec_doc, cfg) -> list:
    primary = cfg.frozen["protocol"]["static"]["history_len"]
    history = primary if primary in spec_doc["histories"] else spec_doc["histories"][0]
    predictor = spec_doc["predictors"][0]
    out = []
    for scene in spec_doc["scenes"]:
        for condition in spec_doc["conditions"]:
            achieved, diagnostics = {}, {}
            for level in levels(cfg, condition):
                achieved[level], diagnostics[level] = {}, {}
                for seed in spec_doc["seeds"]:
                    result, _ = _cell(cells, PER_HISTORY, history, scene, predictor, condition, level, seed)
                    block = None if result is None else result["achieved"]
                    achieved[level][seed] = None if block is None else block["value"]
                    if block is not None and condition == "D":
                        d = block["detail"]
                        diagnostics[level][seed] = {k: d[k] for k in ("switch_events", "triggers", "suppressed_triggers",
                                                                      "lock_aware_exposure", "eligible_track_frames")}
            out.append(check_monotonicity(scene, condition, achieved, diagnostics if condition == "D" else None))
    return out


def _p12(curves: dict, scenes: list, conditions: list) -> dict:
    """G_M(s) and the degradation gap against CV, per scene and paired by scene across scenes."""
    out: dict = {}
    for model in ("LSTM", "GRU"):
        for condition in conditions:
            per_level: dict = {}
            for scene in scenes:
                cv, other = curves[scene]["CV"][condition]["levels"], curves[scene][model][condition]["levels"]
                for a, b in zip(cv, other):
                    row = per_level.setdefault(a["requested"], {"G": {}, "delta": {}})
                    ade_a, ade_b = a["ade_m"]["mean"], b["ade_m"]["mean"]
                    d_a, d_b = a["d_ade"]["mean"], b["d_ade"]["mean"]
                    row["G"][scene] = None if ade_a is None or ade_b is None else ade_b - ade_a
                    row["delta"][scene] = None if d_a is None or d_b is None else d_b - d_a
            out.setdefault(model, {})[condition] = [
                {"requested": lv, "absolute_accuracy_gap": _across(r["G"]), "degradation_gap": _across(r["delta"])}
                for lv, r in sorted(per_level.items())]
    return out


def _window_totals(by_scene: dict) -> dict:
    """Counts over every scene and seed of one level, with the per-scene, per-seed table they come from."""
    counts = [c for row in by_scene.values() for c in row.values()]
    return {"predicted_total": sum(c["predicted"] or 0 for c in counts),
            "eligible_total": sum(c["eligible"] or 0 for c in counts), "by_scene": by_scene}


def _conditional_error(node: dict) -> dict:
    """Decision 0005, section 3: everything needed to read B2 or C at one history, reported side by side."""
    return {"audc": node["audc"], "audc_paired": node["audc_paired"],
            "levels": [{"requested": row["requested"], "prediction_coverage": row["prediction_coverage"],
                        "windows": _window_totals(row["windows_by_scene"])} for row in node["levels"]]}


def _p15(across: dict, spec_doc: dict, cfg: Config, populations: list) -> dict:
    h = cfg.frozen["analysis"]["history_ablation"]
    histories, n_folds = spec_doc["histories"], len(spec_doc["scenes"])
    out: dict = {}
    for population in populations:
        for predictor in spec_doc["predictors"]:
            nodes = {x: across[population][str(x)][predictor] for x in histories}
            ade = {x: n["clean"]["ade_m"]["mean"] for x, n in nodes.items()}
            fde = {x: n["clean"]["fde_m"]["mean"] for x, n in nodes.items()}
            coverage = {str(x): n["clean"]["eligible_windows"] for x, n in nodes.items()}
            for condition in spec_doc["conditions"]:
                cells = {x: n[condition]["audc"] for x, n in nodes.items()}
                result = minimum_history(ade, fde, {x: c["mean"] for x, c in cells.items()}, h["clean_ratio_max"],
                                         h["delta_audc_max"], h["epsilon_audc"],
                                         {x: c["n_valid"] for x, c in cells.items()}, n_folds)
                entry = {**result, "eligible_windows": coverage,
                         "valid_scenes_by_history": {str(x): c["n_valid"] for x, c in cells.items()},
                         "undefined_scenes_by_history": {str(x): c["n_undefined"] for x, c in cells.items()},
                         "undefined_reasons_by_history": {str(x): c["undefined"] for x, c in cells.items()},
                         "partial_folds": any(c["n_valid"] < n_folds for c in cells.values())}
                if condition in CONDITIONAL_ERROR:
                    entry["conditional_error"] = {"note": CONDITIONAL_NOTE, "by_history": {
                        str(x): _conditional_error(n[condition]) for x, n in nodes.items()}}
                out.setdefault(population, {}).setdefault(predictor, {})[condition] = entry
    return out


def analyze(store: RunStore, cfg: Config) -> dict:
    manifest = store.manifest()
    spec_doc = dict(manifest["spec"])
    spec_doc["scenes"] = sorted(spec_doc["scenes"])
    cells, failures = _index(store)
    seeds, scope = spec_doc["seeds"], spec_doc["scope"]
    if spec_doc.get("common_training"):
        populations = [COMMON_TRAINED]
    else:
        populations = [PER_HISTORY] + ([COMMON] if len(spec_doc["histories"]) > 1 else [])
    curves: dict = {}
    clean_section: dict = {}
    family: dict = {}
    across: dict = {}
    for population in populations:
        for history in spec_doc["histories"]:
            h_key = str(history)
            scene_curves: dict = {}
            for scene in spec_doc["scenes"]:
                for predictor in spec_doc["predictors"]:
                    key = (population, history, scene, predictor)
                    summary, clean = _clean(cells, key, seeds)
                    clean_section.setdefault(population, {}).setdefault(h_key, {}).setdefault(scene, {})[predictor] = summary
                    for condition in spec_doc["conditions"]:
                        curve = _curve(cells, key, condition, levels(cfg, condition), seeds, clean)
                        scene_curves.setdefault(scene, {}).setdefault(predictor, {})[condition] = curve
                for condition in spec_doc["conditions"]:
                    if scope != ETHHOTEL:
                        family.setdefault(population, {}).setdefault(h_key, {}).setdefault(scene, {})[condition] = family_audc(
                            {p: scene_curves[scene][p][condition]["_audc"] for p in spec_doc["predictors"]})
            for row in scene_curves.values():
                for by_condition in row.values():
                    for curve in by_condition.values():
                        curve.pop("_audc", None)
            curves.setdefault(population, {})[h_key] = scene_curves
            if scope == ETHHOTEL:
                continue
            for predictor in spec_doc["predictors"]:
                node = across.setdefault(population, {}).setdefault(h_key, {}).setdefault(predictor, {})
                sc = spec_doc["scenes"]
                node["clean"] = {
                    **{m: _across({s: clean_section[population][h_key][s][predictor][m]["mean"] for s in sc})
                       for m in ("ade_m", "fde_m", "rmse_m", "prediction_coverage")},
                    "eligible_windows": {s: next((v.result["metrics"]["eligible_windows"] for k, v in cells.items()
                                                  if k[:4] == (population, history, s, predictor) and k[4] == "clean"
                                                  and v.result), None) for s in sc}}
                for condition in spec_doc["conditions"]:
                    rows = [scene_curves[s][predictor][condition] for s in sc]
                    node[condition] = {
                        "audc": _across({s: r["audc"]["value"] for s, r in zip(sc, rows)},
                                        {s: r["audc"]["reason"] for s, r in zip(sc, rows)}),
                        "audc_paired": _across({s: r["audc_paired"]["value"] for s, r in zip(sc, rows)},
                                               {s: r["audc_paired"]["reason"] for s, r in zip(sc, rows)}),
                        "levels": [{"requested": rows[0]["levels"][i]["requested"],
                                    **{m: _across({s: r["levels"][i][m]["mean"] for s, r in zip(sc, rows)})
                                       for m in ("ade_m", "fde_m", "rmse_m", "prediction_coverage", "d_ade")},
                                    "achieved": _across({s: r["levels"][i].get("achieved_seed_mean") for s, r in zip(sc, rows)}),
                                    "windows_by_scene": {s: r["levels"][i]["windows_by_seed"] for s, r in zip(sc, rows)}}
                                   for i in range(len(rows[0]["levels"]))]}

    primary = cfg.frozen["protocol"]["static"]["history_len"]
    doc = {"label": LABELS[scope], "scope": scope, "research_results": False, "spec": spec_doc,
           "aggregation": "seeds within scene/fold, then across scenes; descriptive means and standard deviations",
           "blocked": blocked_experiments(cfg), "failures": list(failures.values()),
           "population_labels": {p: POPULATION_LABELS[p] for p in populations},
           "curves": curves, "clean": clean_section, "family_audc": family, "across_scenes": across,
           "monotonicity": _monotonicity(cells, spec_doc, cfg) if PER_HISTORY in populations else []}
    if scope == ETHHOTEL:
        doc["notes"] = ["ETH and HOTEL are never pooled, so nothing is aggregated across scenes.",
                        "Family AUDC is not computed: LSTM and GRU are not run on real scenes, and a family mean over "
                        "CV and CA alone would omit them.",
                        "Hypothesis criteria are not evaluated: two scenes cannot support them."]
        doc["hypotheses"] = hypotheses.not_evaluated(ETHHOTEL)
        return doc
    if (PER_HISTORY in populations and primary in spec_doc["histories"]
            and all(p in spec_doc["predictors"] for p in ("CV", "LSTM", "GRU"))):
        doc["p12"] = _p12(curves[PER_HISTORY][str(primary)], spec_doc["scenes"], spec_doc["conditions"])
        if set(CONDITIONS) <= set(spec_doc["conditions"]):
            doc["hypotheses"] = hypotheses.engineering_checks(curves[PER_HISTORY][str(primary)], spec_doc, cfg)
    if len(spec_doc["histories"]) > 1:
        doc["p15"] = _p15(across, spec_doc, cfg, populations)
    return doc


def write_analysis(store: RunStore, cfg: Config, out_dir: Path | None = None) -> dict:
    """Write analysis.json to the run, or to `out_dir` so that a stored run's own files stay as they are."""
    doc = analyze(store, cfg)
    path = Path(out_dir or store.root) / "analysis.json"
    atomic_write_text(path, json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n")
    return doc
