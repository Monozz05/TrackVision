"""Every error at every tested level on one scene, for the Analysis and Applications pages (decision 0016).

Exploratory and computed live, like the Workspace's errors. Each level is the operation of a stored sweep cell: one error
from the frozen injectors, three seeds, scored by the frozen scorer, seeds averaged within the scene. So on the scenes of
the stored sweeps these values equal the stored ones (tests enforce this), and on every other scene they are the same
calculation in that scene's unit. Scenes and parts are never combined here. Nothing here is saved anywhere: decision 0011
keeps MOT17 forecasts out of every saved file, so results live in memory for the life of the process.

Two choices belong to these pages, not to the protocol:
  rise    measured on the same windows, clean against with errors. Missing detections and broken tracks only remove windows
          and never change the ones left, so their rise is exactly zero and their damage shows in coverage. Measured against
          all windows, they would show changes caused only by which windows are left.
  limits  position noise and ID switches use the stored definition, relative ADE degradation against all clean windows
          (decision 0004). Missing detections and broken tracks cannot raise ADE, so their limit is on windows lost instead.

The levels are the frozen grids. No grid exists in pixels, so on pixel scenes position noise is the metre grid scaled to the
top of the Workspace slider, which is an implementation choice.

`evaluation` imports torch, so it is imported on first use and the API starts quickly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from trackvision.stats.aggregate import aggregate, relative_degradation
from trackvision.stats.crossing import first_crossing
from trackvision.stats.inverse import inverse_requirement

log = logging.getLogger("trackvision.platform")
FIELDS = {"A": "a_sigma", "B": "b_p", "C": "c_p", "D": "d_p"}        # B without gap filling, the representative B2
ON_ADE, ON_LOST = "ade", "windows_lost"
LIMIT_ON = {"A": ON_ADE, "B": ON_LOST, "C": ON_LOST, "D": ON_ADE}
IDLE_SECONDS = 15.0                  # a scene nobody has asked about for this long waits until someone asks again
LABEL = ("Measured live on this scene with the frozen injectors, one error at a time, averaged over {seeds} seeds. "
         "Exploratory, not a research result, and never saved.")
FAILED_REASON = "The calculation for this scene stopped; see the backend log."


def tested_levels(cfg, unit: str, pixel_noise_top: float) -> dict[str, list[float]]:
    """The frozen grids without their zero. On a pixel scene, position noise is the metre grid scaled to the slider's top."""
    grids = cfg.frozen["injection"]["grids"]
    levels = {code: [float(level) for level in grids[code]["levels"] if level > 0] for code in FIELDS}
    if unit == "px":
        scale = pixel_noise_top / levels["A"][-1]
        levels["A"] = [round(level * scale, 6) for level in levels["A"]]
    return levels


def _mean(values: dict) -> float | None:
    return aggregate(values).mean


def clean_reference(loaded) -> dict:
    """Each predictor on the clean windows: the primary baseline, ADE0 (decision 0004)."""
    from trackvision.experiments import evaluation
    out = {}
    for predictor in loaded.predictors:
        clean = loaded.clean_errors[predictor.name]
        scored = evaluation.score(predictor, evaluation.clean_inputs(loaded.windows), loaded.windows, clean)
        out[predictor.name] = {"ade": scored["ade_m"], "fde": scored["fde_m"], "de_by_step": scored["de_by_step_m"]}
    return out


def _steps(scored: dict) -> list | None:
    rows = [row["de_by_step_m"] for row in scored.values() if row["de_by_step_m"] is not None]
    return [sum(column) / len(rows) for column in zip(*rows)] if rows else None


def _rise(scored: dict, metric: str) -> float | None:
    """The change on the windows still predicted, against those same windows clean."""
    return _mean({seed: relative_degradation(row[metric], (row["clean_same_windows"] or {}).get(metric)).value
                  for seed, row in scored.items()})


def measure_level(loaded, code: str, level: float, seeds: list, cfg, clean: dict) -> dict:
    """One error at one level. `clean` is `clean_reference` of the same scene."""
    from trackvision.experiments import evaluation
    from trackvision.platform.services import exploratory as ex
    base, request = loaded.base, ex.ErrorRequest(**{FIELDS[code]: level})
    by_seed = {seed: ex.combine(loaded.tracks, loaded.population, request, seed, cfg, loaded.frame_step, d_max=base.d_max,
                                unit=base.unit) for seed in seeds}
    row: dict = {"requested": level, "achieved": _mean({seed: c.achieved[code]["value"] for seed, c in by_seed.items()}),
                 "coverage": None, "predictors": {}}
    for predictor in loaded.predictors:
        scored = {seed: evaluation.score(predictor, c.corrupted, loaded.windows, loaded.clean_errors[predictor.name])
                  for seed, c in by_seed.items()}
        row["coverage"] = _mean({seed: result["prediction_coverage"] for seed, result in scored.items()})
        row["predictors"][predictor.name] = {
            "ade": _mean({seed: result["ade_m"] for seed, result in scored.items()}),
            "fde": _mean({seed: result["fde_m"] for seed, result in scored.items()}),
            "ade_rise": _rise(scored, "ade_m"), "fde_rise": _rise(scored, "fde_m"),
            "ade_degradation": _mean({seed: relative_degradation(result["ade_m"], clean[predictor.name]["ade"]).value
                                      for seed, result in scored.items()}),
            "de_by_step": _steps(scored)}
    return row


def _limit_points(code: str, rows: list, predictor: str) -> list:
    if LIMIT_ON[code] == ON_ADE:
        return [(0.0, 0.0), *((row["achieved"], row["predictors"][predictor]["ade_degradation"]) for row in rows)]
    return [(0.0, 0.0), *((row["achieved"], None if row["coverage"] is None else 1.0 - row["coverage"]) for row in rows)]


def limits(code: str, rows: list, predictors: list, qs: list) -> dict:
    """How much of this error each predictor stays within, for each allowed change. Descriptive, not a safety limit."""
    out: dict = {}
    for q in qs:
        for predictor in predictors:
            result = inverse_requirement(_limit_points(code, rows, predictor), float(q))
            out.setdefault(str(q), {})[predictor] = {"status": result["status"], "value": result["s_star"],
                                                     "verified_up_to": result["verified_up_to"],
                                                     "tested_up_to": result["maximum_observed_severity"]}
    return out


def better(rows: list, clean: dict) -> dict:
    """Where CV and CA meet on ADE as this error grows. Arithmetic only: no predictor is chosen for anything."""
    a, b = list(clean)[:2]
    points = [(0.0, clean[a]["ade"], clean[b]["ade"]),
              *((row["achieved"], row["predictors"][a]["ade"], row["predictors"][b]["ade"]) for row in rows)]
    result = first_crossing(points)
    usable = clean[a]["ade"] is not None and clean[b]["ade"] is not None
    return {"status": result["status"], "at": result["s_crossing"],
            "better_when_clean": None if not usable else a if clean[a]["ade"] < clean[b]["ade"] else b}


class Job:
    """One scene's sweep, filled in level by level."""

    def __init__(self, scene: str, levels: dict, seeds: list, cfg, qs: list):
        self.scene, self.levels, self.seeds, self.cfg, self.qs = scene, levels, seeds, cfg, qs
        self.todo = [(code, level) for code, grid in levels.items() for level in grid]
        self.total = len(self.todo)
        self.rows: dict[str, list] = {code: [] for code in levels}
        self.summary: dict[str, dict] = {}
        self.loaded = self.clean = None
        self.failure: str | None = None
        self.failure_seen = False
        self.opened = 0
        self.polled = -float("inf")

    def prepare(self, loaded) -> None:
        self.clean, self.loaded = clean_reference(loaded), loaded

    def step(self) -> tuple[str, dict, dict | None]:
        """Measure the next level. Returns what `add` stores, so the caller can store it under its lock."""
        code, level = self.todo[0]
        row = measure_level(self.loaded, code, level, self.seeds, self.cfg, self.clean)
        rows = [*self.rows[code], row]
        finished = None
        if len(rows) == len(self.levels[code]):
            finished = {"limits": limits(code, rows, list(self.clean), self.qs), "better": better(rows, self.clean)}
        return code, row, finished

    def add(self, code: str, row: dict, finished: dict | None) -> None:
        self.rows[code].append(row)
        self.todo.pop(0)
        if finished:
            self.summary[code] = finished

    def document(self, entry: dict) -> dict:
        base = self.loaded.base if self.loaded else None
        state = "failed" if self.failure else "calculating" if self.todo else "ready"
        return {"scene": self.scene, "exploratory": True, "state": state, "done": self.total - len(self.todo),
                "total": self.total, "unit": entry["unit"], "seeds": self.seeds, "reason": self.failure,
                "label": " ".join(filter(None, (LABEL.format(seeds=len(self.seeds)), entry["caution"], base and base.caution))),
                "windows": len(self.loaded.windows) if self.loaded else None,
                "seconds_per_step": self.cfg.frozen["protocol"]["static"]["sampling_interval_s"] if base is None
                else base.seconds_per_step or self.cfg.frozen["protocol"]["static"]["sampling_interval_s"],
                "clean": self.clean,
                "limits_allowed": [str(q) for q in self.qs],
                "errors": [{"code": code, "limit_on": LIMIT_ON[code], "tested": len(self.levels[code]),
                            "measure": _measure(entry["unit"], code), "levels": list(self.rows[code]),
                            **self.summary.get(code, {"limits": None, "better": None})} for code in self.levels]}


def _measure(unit: str, code: str) -> str:
    return {"A": f"rmse_pos_{unit}", "B": "rate", "C": "fragmentation_rate", "D": "switch_event_rate"}[code]


class SweepQueue:
    """Runs sweeps on one background thread, one level at a time, so a long scene never holds a request open.

    The scene opened most recently goes first, and a scene nobody has asked about for `IDLE_SECONDS` waits until someone
    asks again, so moving between scenes never leaves calculations running for a page already left. Finished levels stay in
    memory for the life of the process and are never written anywhere.
    """

    def __init__(self, make_job: Callable[[str], Job], load: Callable[[str], object]):
        self._make_job, self._load = make_job, load
        self._jobs: dict[str, Job] = {}
        self._opened = 0                 # counts openings, because the clock on Windows can give two of them the same time
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def request(self, scene_id: str, entry: dict) -> dict:
        """The sweep as far as it has got, starting or resuming it if it is not finished."""
        job = self._jobs.get(scene_id) or self._make_job(scene_id)
        now = time.monotonic()
        with self._lock:
            job = self._jobs.setdefault(scene_id, job)
            if job.failure_seen:             # asking again after seeing a failure is a retry
                job.failure, job.failure_seen = None, False
            if now - job.polled > IDLE_SECONDS:
                self._opened += 1
                job.opened = self._opened
            job.polled = now
            if job.todo:
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="scene-sweep", daemon=True)
                    self._thread.start()
                self._wake.set()
            job.failure_seen = job.failure is not None
            return job.document(entry)

    def _next(self) -> Job | None:
        now = time.monotonic()
        with self._lock:
            waiting = [job for job in self._jobs.values()
                       if job.todo and job.failure is None and now - job.polled <= IDLE_SECONDS]
            return max(waiting, key=lambda job: job.opened, default=None)

    def _step(self, job: Job) -> None:
        if job.loaded is None:
            job.prepare(self._load(job.scene))
        result = job.step()
        with self._lock:
            job.add(*result)

    def _run(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait()
                self._wake.clear()
                continue
            try:
                self._step(job)
            except Exception:
                log.exception("the sweep of %s stopped", job.scene)
                with self._lock:
                    job.failure = FAILED_REASON
