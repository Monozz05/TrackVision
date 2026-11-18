"""Views over stored results: each one selects and reshapes recorded values for a screen.

Nothing here computes a research metric. Every number returned is a number on disk, so a value that is absent stays
None and carries its recorded reason. Scene-level means stay scene-level, across-scene means are returned only where
the analysis stored them (never for ETH and HOTEL), and degradation levels carry both requested and achieved
severity so plots can use the achieved one.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from trackvision import config
from trackvision.platform.services.result_reader import ArtifactError, ArtifactMissing, ResultReader
from trackvision.platform.services.run_registry import Run

FIGURE = re.compile(r"^[A-Za-z0-9._-]{1,160}\.svg$")
LEVEL_FIELDS = ("requested", "achieved_seed_mean", "s_norm", "ade_m", "fde_m", "rmse_m", "prediction_coverage",
                "d_ade", "d_fde", "de_by_step_m", "windows_by_seed", "undefined_reason_by_seed")
P15_FIELDS = ("audc_best", "clean_ade_best", "clean_fde_best", "minimum_history", "partial_folds", "status", "reason")
_FIGURE_PATTERNS = (
    ("degradation", re.compile(r"^degradation_(?P<scene>.+)_(?P<condition>[A-Z]\d?)_h(?P<history>\d+)$")),
    ("horizon", re.compile(r"^horizon_(?P<scene>.+)_(?P<condition>[A-Z]\d?)_h(?P<history>\d+)$")),
    ("conditional_error", re.compile(r"^conditional_error_(?P<population>.+)_(?P<predictor>[A-Z]+)_(?P<condition>[A-Z]\d?)$")),
)
# The files an E5 to E8 run records by hash: what each is, which kind of run holds it, its name there, and what it feeds.
E5_E8_INPUTS = {"e4_results_sha256": ("E4 results", "synthetic_tracker", "results.json", ["E5"]),
                "synthetic_analysis_sha256": ("Synthetic curves", "sweep", "analysis.json", ["E6", "E7", "E8"]),
                "ethhotel_analysis_sha256": ("ETH and HOTEL curves", "sweep", "analysis.json", ["E7", "E8"])}


class ViewUnavailable(LookupError):
    """The run exists but has no such view, or lacks the selection that was asked for."""


class RunNotUsable(RuntimeError):
    """The run's records failed validation, so its results are not served."""


@lru_cache(maxsize=4)
def frozen_protocol(base_config: Path) -> dict:
    """The frozen block of `base.yaml`, read without validation so that a missing spec cannot stop the API."""
    try:
        return config.load_yaml(base_config)["frozen"]
    except (config.ConfigError, OSError, KeyError, TypeError):
        return {}


def _require_usable(run: Run) -> None:
    if not run.usable:
        raise RunNotUsable(f"{run.id} is {run.summary['status']}: {run.summary['status_reason']}")


def _results(run: Run, reader: ResultReader) -> dict:
    _require_usable(run)
    return reader.read_json(run.path / "results.json")


def _analysis(run: Run, reader: ResultReader) -> dict:
    _require_usable(run)
    return reader.read_json(run.path / "analysis.json")


def _selection(doc: dict, population: str | None, history: int | None, default_history: int | None) -> tuple[str, str]:
    """Resolve the population and history against what the analysis stored; refuse anything it did not store."""
    curves = doc.get("curves") or {}
    population = population or ("per_history" if "per_history" in curves else next(iter(curves), None))
    if population not in curves:
        raise ViewUnavailable(f"this run stores no population {population}; it has {sorted(curves)}")
    stored = sorted(curves[population], key=int)
    if history is not None and str(history) not in stored:
        raise ViewUnavailable(f"this run stores no history length {history}; it has {stored}")
    if history is not None:
        return population, str(history)
    return population, str(default_history) if str(default_history) in stored else stored[0]


def _selection_info(doc: dict, population: str, history: str) -> dict:
    return {"population": population, "history": int(history), "populations": sorted(doc["curves"]),
            "histories": sorted(int(h) for h in doc["curves"][population]),
            "population_label": (doc.get("population_labels") or {}).get(population),
            "aggregation": doc.get("aggregation"), "scope_label": doc.get("label")}


def _level(level: dict) -> dict:
    return {key: level.get(key) for key in LEVEL_FIELDS}


# --- tracking ----------------------------------------------------------------------------------------------------


def _tracking_metrics(official: dict) -> dict:
    return {"mota": official.get("MOTA"), "idf1": official.get("IDF1"), "hota": official.get("HOTA"),
            "idSwitches": official.get("IDSW"), "fragmentations": official.get("Frag")}


def _tracker_settings(run: Run, reader: ResultReader, detection_set: str) -> dict | None:
    try:
        return reader.read_json(run.path / detection_set / "tracker" / "results.json").get("settings")
    except ArtifactError:
        return None


def tracking(run: Run, reader: ResultReader, base_config: Path) -> dict:
    if run.kind == "mot17_tracking":
        doc = _results(run, reader)
        partition = (frozen_protocol(base_config).get("datasets") or {}).get("mot17") or {}
        regime = {**{s: "static" for s in partition.get("static_sequences", [])},
                  **{s: "moving" for s in partition.get("moving_sequences", [])}}
        groups = [{
            "name": name,
            "primary": name == doc.get("primary_detection_set"),
            "tracker_settings": _tracker_settings(run, reader, name),
            "sequences": [{"sequence": seq, "regime": regime.get(seq), "frame_rate": row.get("frame_rate"),
                           "frames_evaluated": row.get("frames_evaluated"),
                           "metrics": _tracking_metrics(row.get("official_trackeval") or {}),
                           "official_trackeval": row.get("official_trackeval"),
                           "project_diagnostics": row.get("project_diagnostics")} for seq, row in sequences.items()],
        } for name, sequences in (doc.get("by_detection_set") or {}).items()]
        return {"experiment": run.summary, "source": run.kind, "groups": groups, "pooled_result": doc.get("pooled_result"),
                "not_satisfied": doc.get("not_satisfied"), "prediction_note": doc.get("prediction"),
                "cadence": doc.get("cadence"), "not_p1": doc.get("not_p1"), "not_p9": doc.get("not_p9")}
    if run.kind == "synthetic_tracker":
        doc = _results(run, reader)
        official = doc.get("official_trackeval") or {}
        scenes = doc.get("by_scene") or {}
        return {"experiment": run.summary, "source": run.kind, "groups": [{
            "name": "Synthetic scenes", "primary": True, "tracker_settings": doc.get("provenance"),
            "sequences": [{"sequence": scene, "regime": None, "frame_rate": None, "frames_evaluated": None,
                           "metrics": _tracking_metrics(official.get(scene) or {}),
                           "official_trackeval": official.get(scene),
                           "project_diagnostics": row.get("project_diagnostics")} for scene, row in scenes.items()],
        }], "pooled_result": None, "official_note": official.get("note"), "not_p9": doc.get("not_p9")}
    raise ViewUnavailable(f"{run.id} holds no tracking results")


# --- error injection ---------------------------------------------------------------------------------------------


def _figures(run: Run) -> list[dict]:
    figures = []
    directory = run.path / "figures"
    if not directory.is_dir():
        return figures
    for path in sorted(directory.glob("*.svg")):
        if not FIGURE.match(path.name) or path.is_symlink():
            continue
        info = {"name": path.name, "figure": "other", "scene": None, "condition": None, "history": None,
                "predictor": None, "population": None}
        for figure, pattern in _FIGURE_PATTERNS:
            found = pattern.match(path.stem)
            if found:
                info.update(figure=figure, **found.groupdict())
                break
        figures.append(info)
    return figures


def error_injection(run: Run, reader: ResultReader, base_config: Path, population: str | None = None,
                    history: int | None = None) -> dict:
    if run.kind == "ewap_injection":
        doc = _results(run, reader)
        return {"experiment": run.summary, "source": run.kind, "stored_conditions": doc.get("conditions"),
                "clean_reference": doc.get("clean_reference_all_eligible"), "c_vs_d": doc.get("c_vs_d")}
    if run.kind != "sweep":
        raise ViewUnavailable(f"{run.id} holds no error-injection results")
    doc = _analysis(run, reader)
    population, history = _selection(doc, population, history, _default_history(base_config))
    rows = []
    for scene, predictors in doc["curves"][population][history].items():
        for predictor, conditions in predictors.items():
            for condition, curve in conditions.items():
                normalization = curve.get("normalization") or {}
                by_level = {n.get("requested"): n for n in normalization.get("levels") or []}
                for level in curve.get("levels") or []:
                    measured = by_level.get(level.get("requested")) or {}
                    rows.append({"scene": scene, "predictor": predictor, "condition": condition,
                                 "severity_measure": normalization.get("measure"),
                                 "requested": level.get("requested"), "achieved": level.get("achieved_seed_mean"),
                                 "achieved_by_seed": measured.get("achieved_by_seed"),
                                 "seed_spread": measured.get("seed_spread"), "ade_m": level.get("ade_m"),
                                 "d_ade": level.get("d_ade"), "prediction_coverage": level.get("prediction_coverage"),
                                 "windows_by_seed": level.get("windows_by_seed"),
                                 "undefined_reason_by_seed": level.get("undefined_reason_by_seed")})
    return {"experiment": run.summary, "source": run.kind, "selection": _selection_info(doc, population, history),
            "rows": rows, "monotonicity": doc.get("monotonicity"), "blocked": doc.get("blocked"),
            "figures": _figures(run)}


def _default_history(base_config: Path) -> int | None:
    return ((frozen_protocol(base_config).get("protocol") or {}).get("static") or {}).get("history_len")


# --- prediction --------------------------------------------------------------------------------------------------


def prediction(run: Run, reader: ResultReader, base_config: Path, population: str | None = None,
               history: int | None = None) -> dict:
    if run.kind == "sweep":
        return _sweep_prediction(run, reader, base_config, population, history)
    if run.kind == "synthetic_tracker":
        doc = _results(run, reader)
        rows = []
        for scene, entry in (doc.get("by_scene") or {}).items():
            forecast = entry.get("prediction") or {}
            for predictor, row in (forecast.get("predictors") or {}).items():
                stats = row.get("seed_mean") or {}
                rows.append({"scene": scene, "predictor": predictor, "ade_m": stats.get("tracker_ade_m"),
                             "fde_m": stats.get("tracker_fde_m"), "rmse_m": None, "de_by_step_m": None,
                             "prediction_coverage": (forecast.get("coverage") or {}).get("prediction_coverage"),
                             "clean_all_eligible_ade_m": stats.get("clean_all_eligible_ade_m"),
                             "d_ade_paired": stats.get("d_ade_paired"), "by_seed": row.get("by_seed")})
        return {"experiment": run.summary, "source": run.kind, "rows": rows, "across_scenes": [],
                "coverage_by_scene": {s: (e.get("prediction") or {}).get("coverage")
                                      for s, e in (doc.get("by_scene") or {}).items()}}
    if run.kind == "ewap_baseline":
        doc = _results(run, reader)
        summary = doc.get("summary") or {}
        coverage = summary.get("coverage") or {}
        return {"experiment": run.summary, "source": run.kind, "across_scenes": [], "rows": [{
            "scene": doc.get("scene"), "predictor": "CV", "ade_m": summary.get("ade_m"),
            "fde_m": summary.get("fde_m"), "rmse_m": None, "prediction_coverage": coverage.get("prediction_coverage"),
            "de_by_step_m": summary.get("mean_error_by_step_m")}]}
    raise ViewUnavailable(f"{run.id} holds no prediction results")


def _sweep_prediction(run: Run, reader: ResultReader, base_config: Path, population: str | None,
                      history: int | None) -> dict:
    doc = _analysis(run, reader)
    population, history = _selection(doc, population, history, _default_history(base_config))
    curves = doc["curves"][population][history]
    rows = []
    for scene, predictors in ((doc.get("clean") or {}).get(population, {}).get(history) or {}).items():
        for predictor, row in predictors.items():
            rows.append({"scene": scene, "predictor": predictor, **{k: row.get(k) for k in (
                "ade_m", "fde_m", "rmse_m", "prediction_coverage")},
                "de_by_step_m": _clean_steps(curves.get(scene, {}).get(predictor, {}))})
    across = []
    for predictor, conditions in ((doc.get("across_scenes") or {}).get(population, {}).get(history) or {}).items():
        clean = _clean_level(conditions)
        if clean:
            across.append({"predictor": predictor, **{k: clean.get(k) for k in (
                "ade_m", "fde_m", "rmse_m", "prediction_coverage")}})
    return {"experiment": run.summary, "source": run.kind, "selection": _selection_info(doc, population, history),
            "rows": rows, "across_scenes": across}


def _clean_level(conditions: dict) -> dict | None:
    """The requested-zero level, which every condition shares; condition A is read because it always exists."""
    for level in (conditions.get("A") or {}).get("levels") or []:
        if level.get("requested") == 0:
            return level
    return None


def _clean_steps(conditions: dict) -> list | None:
    clean = _clean_level(conditions)
    return clean.get("de_by_step_m") if clean else None


# --- analysis ----------------------------------------------------------------------------------------------------


def input_runs(doc: dict, runs: list[Run], reader: ResultReader) -> list[dict]:
    """Where each recorded input of an E5 to E8 run lives now. An input is unchanged only if a file still has its hash."""
    out = []
    for key, (label, kind, name, used_by) in E5_E8_INPUTS.items():
        digest = (doc.get("inputs") or {}).get(key)
        source = next((run for run in runs if run.kind == kind and (run.path / name).is_file()
                       and reader.sha256(run.path / name) == digest), None)
        out.append({"input": label, "used_by": used_by, "run": source.id if source else None, "unchanged": source is not None})
    return out


def analysis(run: Run, reader: ResultReader, base_config: Path, population: str | None = None,
             history: int | None = None, runs: list[Run] | None = None) -> dict:
    if run.kind == "engineering_analysis":
        doc = _results(run, reader)
        return {"experiment": run.summary, "source": run.kind, "input_runs": input_runs(doc, runs or [], reader),
                **{key: doc.get(key) for key in (
                    "family_across_scenes", "e5_p13_engineering", "e6_dominant_dry_run", "e7_inverse_mapping",
                    "e8_threshold_crossings", "association_by_experiment", "ethhotel_family", "blocked", "deferred",
                    "inputs", "label")}}
    if run.kind != "sweep":
        raise ViewUnavailable(f"{run.id} holds no analysis results")
    doc = _analysis(run, reader)
    population, history = _selection(doc, population, history, _default_history(base_config))
    curves = {}
    for scene, predictors in doc["curves"][population][history].items():
        curves[scene] = {predictor: {condition: {
            "severity_measure": (curve.get("normalization") or {}).get("measure"), "audc": curve.get("audc"),
            "robustness": curve.get("robustness"), "levels": [_level(l) for l in curve.get("levels") or []],
        } for condition, curve in conditions.items()} for predictor, conditions in predictors.items()}
    stored_p15 = (doc.get("p15") or {}).get(population)
    p15 = None if stored_p15 is None else {predictor: {condition: {k: row.get(k) for k in P15_FIELDS} for condition, row
                                                       in conditions.items()} for predictor, conditions in stored_p15.items()}
    return {"experiment": run.summary, "source": run.kind, "selection": _selection_info(doc, population, history),
            "curves": curves, "family_audc": (doc.get("family_audc") or {}).get(population, {}).get(history),
            "across_scenes": (doc.get("across_scenes") or {}).get(population, {}).get(history),
            "p15": p15, "p12": doc.get("p12"), "monotonicity": doc.get("monotonicity"),
            "hypotheses": doc.get("hypotheses"), "blocked": doc.get("blocked"), "failures": doc.get("failures")}


# --- one experiment ----------------------------------------------------------------------------------------------


def views_of(run: Run) -> list[str]:
    """Which result views exist for this kind of run."""
    if not run.usable:
        return []
    return {"sweep": ["error-injection", "prediction", "analysis"], "mot17_tracking": ["tracking"],
            "synthetic_tracker": ["tracking", "prediction"], "engineering_analysis": ["analysis"],
            "ewap_baseline": ["prediction"], "ewap_injection": ["error-injection"]}.get(run.kind, [])


def artifacts(run: Run) -> list[dict]:
    """The run's top-level files and its figures. Only figures can be fetched; other files are listed, not served."""
    if not run.path.is_dir():
        return []
    listed = [{"name": p.name, "type": "record", "size_bytes": p.stat().st_size, "served": False}
              for p in sorted(run.path.iterdir()) if p.is_file() and not p.is_symlink()]
    listed += [{"name": f["name"], "type": "chart", "size_bytes": (run.path / "figures" / f["name"]).stat().st_size,
                "served": True} for f in _figures(run)]
    return listed


def stored_summary(run: Run, reader: ResultReader) -> dict | None:
    """Recorded values for kinds that have no dedicated view."""
    if not run.usable:
        return None
    try:
        if run.kind == "smoke":
            return {"scientific_validity": reader.read_json(run.path / "manifest.json").get("scientific_validity"),
                    "fixture": reader.read_json(run.path / "fixture_summary.json")}
        if run.kind == "synthetic_feasibility":
            doc = reader.read_json(run.path / "results.json")
            return {"ground_truth_reference": doc.get("ground_truth_reference"), "roles": doc.get("roles"),
                    "tracking_by_source": {name: (source.get("tracking_u5") or {}).get("metrics")
                                           for name, source in (doc.get("sources") or {}).items()}}
        if run.kind == "geometry_validation":
            doc = reader.read_json(run.path / "report.json")
            return {"passed": doc.get("passed"), "blocked_scenes": doc.get("blocked_scenes"), "scenes": {
                scene: {"roundtrip": row.get("roundtrip"), "speed_diagnostic": row.get("speed_diagnostic"),
                        "windows": row.get("windows"), "diagnostics": row.get("diagnostics")}
                for scene, row in (doc.get("scenes") or {}).items()}}
    except ArtifactMissing:
        return None
    return None


def figure_bytes(run: Run, name: str) -> bytes:
    """The bytes of one stored figure. `name` must be a bare SVG filename that this run lists."""
    if not FIGURE.match(name) or name not in {f["name"] for f in _figures(run)}:
        raise ViewUnavailable(f"{run.id} has no figure with that name")
    return (run.path / "figures" / name).read_bytes()
