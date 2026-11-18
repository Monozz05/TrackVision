"""Exploratory error injection for the workspace (decisions 0008, 0011). NOT a research procedure and NOT P-10.

The injectors and the scorer are the frozen ones, called unchanged. What is added here is only how several of them
meet on one window, and that composition has no protocol behind it, so nothing computed here is a research result
and nothing here is ever written under `runs/`.

Units. The injectors and the scorer work on plain numbers, and the `_m` in their field names is their usual unit. On a
scene with no calibration the same arithmetic runs on video pixels, so everything leaving this module uses neutral
names and says which unit it is in. A pixel value never sits in a field ending `_m`.

Composition. D acts first because it decides which pedestrian each label follows. A, B and C then act on the
relabelled tracks. They read separate random streams over the same samples, so among themselves their order has no
effect. With one error active, the inputs equal `evaluation.corrupt` exactly (tests enforce this), so a single error
here is the same operation as a stored sweep cell at that level.

A window's input is built as the frozen window builders build it. The label holding the window's pedestrian at the
last history step supplies the history (policy P of D), the history must lie in one fragment (C), and a dropped
observation either loses the window (B2) or is interpolated within that history only (B1). Targets always come from
the clean windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackvision.config import Config
from trackvision.experiments import evaluation, injection_feasibility as inj, switch_feasibility as sf
from trackvision.inject.identity_switch import identity_switch
from trackvision.inject.trajectory import dropout_masks, fragment, localization_noise, reconstruct_history
from trackvision.predict.windowing import Windows

LABEL = ("Exploratory, not a research result, not P-10. Combined errors have no agreed protocol. "
         "Computed live and never saved to runs/.")
SINGLE_LABEL = "Exploratory view of one error at a chosen level, computed live and never saved to runs/."


@dataclass(frozen=True)
class ErrorRequest:
    a_sigma: float = 0.0             # in the scene's unit
    b_p: float = 0.0
    c_p: float = 0.0
    d_p: float = 0.0
    reconstruct: bool = False

    @property
    def active(self) -> tuple[str, ...]:
        b = ("B1" if self.reconstruct else "B2",) if self.b_p > 0 else ()
        return (*(("D",) if self.d_p > 0 else ()), *(("A",) if self.a_sigma > 0 else ()), *b,
                *(("C",) if self.c_p > 0 else ()))


@dataclass(frozen=True)
class Combined:
    corrupted: evaluation.Corrupted
    achieved: dict                   # primary achieved severity per active error, from each injector's report
    labelled: tuple                  # tracks after D (the clean tracks if D is off)
    occupant: dict | None            # label -> (n,) pedestrian at each sample, or None if D is off
    offsets: dict                    # label -> (n, 2) noise, or absent
    present: dict                    # label -> (n,) bool, or absent
    segment: dict                    # label -> (n,) fragment id, or absent
    switch_events: tuple
    split_events: tuple              # (label, sample index of the first sample after the split)


def _segments(frag) -> tuple[dict, list]:
    """Per-sample fragment ids, from the lineage map the injector returns."""
    pieces: dict[int, list] = {}
    for fid, (label, a, b) in frag.origin.items():
        pieces.setdefault(label, []).append((a, b, fid))
    segment, splits = {}, []
    for label, parts in pieces.items():
        ids = np.zeros(max(b for _, b, _ in parts), dtype=np.int64)
        for a, b, fid in parts:
            ids[a:b] = fid
            if a > 0:
                splits.append((label, a))
        segment[label] = ids
    return segment, splits


def combine(tracks, pop: inj.Population, request: ErrorRequest, seed: int, cfg: Config, frame_step: int, *,
            d_max: float | None = None, unit: str = "m") -> Combined:
    """Apply the requested errors to a fresh copy of the clean tracks. Targets are never touched.

    `d_max` is how close two people must be for an identity switch, in `unit`. It defaults to the frozen value in metres.
    """
    win, n, h = pop.windows, len(pop.windows), pop.windows.history_len
    work = inj.fresh_copy(tracks)
    labelled, occupant, events, achieved = work, None, (), {}
    if request.d_p > 0:
        sf.check_cooldown(cfg)
        reach = float(cfg.frozen["injection"]["d_max_m"]) if d_max is None else d_max
        switched = identity_switch(work, request.d_p, seed, reach, frame_step)
        labelled, occupant, events = switched.tracks, switched.occupant, switched.events
        achieved["D"] = {"measure": "switch_event_rate", "value": switched.achieved["switch_event_rate"],
                         "events": switched.achieved["switch_events"]}
    offsets, present, segment, splits = {}, {}, {}, []
    if request.a_sigma > 0:
        noise = localization_noise(labelled, request.a_sigma, seed)
        offsets = {int(t.ped_id): o for t, o in zip(labelled, noise.offsets)}
        achieved["A"] = {"measure": f"rmse_pos_{unit}", "value": noise.achieved["rmse_pos_m"]}
    if request.b_p > 0:
        drop = dropout_masks(labelled, request.b_p, seed)
        present = {int(t.ped_id): m for t, m in zip(labelled, drop.present)}
        achieved["B"] = {"measure": "rate", "value": drop.achieved["rate"],
                         "removed": drop.achieved["removed_observations"]}
    if request.c_p > 0:
        frag = fragment(labelled, request.c_p, seed)
        segment, splits = _segments(frag)
        achieved["C"] = {"measure": "fragmentation_rate", "value": frag.achieved["fragmentation_rate"],
                         "events": frag.achieved["split_events"]}

    by_label = {int(t.ped_id): t for t in labelled}
    index = {label: {int(f): k for k, f in enumerate(t.frames)} for label, t in by_label.items()}
    holder = (None if occupant is None else
              {(int(o), int(f)): label for label, t in by_label.items() for f, o in zip(t.frames, occupant[label])})
    predicted = np.zeros(n, dtype=bool)
    reason: list = [None] * n
    inputs = []
    for w, (g, frames) in enumerate(zip(win.track_ids, win.frames)):
        history = [int(f) for f in frames[:h]]
        label = int(g) if holder is None else holder[(int(g), history[-1])]
        if any(f not in index[label] for f in history):
            reason[w] = sf.LOST_REASON
            continue
        k = [index[label][f] for f in history]
        if segment and len(set(segment[label][k].tolist())) > 1:
            reason[w] = inj.LOST_REASON["C"]
            continue
        x = by_label[label].positions[k]
        if offsets:
            x = x + offsets[label][k]
        if present:
            alive = present[label][k]
            if request.reconstruct:
                rebuilt = reconstruct_history(np.where(alive[:, None], x, np.nan), alive)
                if rebuilt.positions is None:
                    reason[w] = inj.LOST_REASON["B1"]
                    continue
                x = rebuilt.positions
            elif not alive.all():
                reason[w] = inj.LOST_REASON["B2"]
                continue
        predicted[w] = True
        inputs.append(x)
    corrupted = evaluation.Corrupted("+".join(request.active) or "clean", 0.0, int(seed), predicted, tuple(reason),
                                     np.asarray(inputs, dtype=np.float64).reshape(-1, h, 2), achieved, "", None, {})
    return Combined(corrupted, achieved, labelled, occupant, offsets, present, segment, tuple(events), tuple(splits))


def _mean(values: list) -> float | None:
    defined = [v for v in values if v is not None]
    return float(np.mean(defined)) if defined else None


def _steps(rows: list) -> list | None:
    defined = [r for r in rows if r is not None]
    return np.mean(defined, axis=0).tolist() if defined else None


def summarise(windows: Windows, predictors, clean_errors: dict, combined_by_seed: dict) -> dict:
    """Per-predictor means over seeds, within this one scene. Windows overlap, so these are descriptive only."""
    out = {}
    for predictor in predictors:
        clean = clean_errors[predictor.name]
        scored = [evaluation.score(predictor, c.corrupted, windows, clean) for c in combined_by_seed.values()]
        paired = [s["clean_same_windows"] or {} for s in scored]
        reference = evaluation.score(predictor, evaluation.clean_inputs(windows), windows, clean)
        ade = _mean([s["ade_m"] for s in scored])
        clean_ade = reference["ade_m"]
        out[predictor.name] = {
            "ade": ade, "fde": _mean([s["fde_m"] for s in scored]), "rmse": _mean([s["rmse_m"] for s in scored]),
            "prediction_coverage": _mean([s["prediction_coverage"] for s in scored]),
            "de_by_step": _steps([s["de_by_step_m"] for s in scored]),
            "clean_ade": clean_ade, "clean_fde": reference["fde_m"], "clean_rmse": reference["rmse_m"],
            "clean_de_by_step": reference["de_by_step_m"],
            "clean_same_windows_ade": _mean([p.get("ade_m") for p in paired]),
            "ade_change": None if ade is None or not clean_ade else (ade - clean_ade) / clean_ade,
            "seeds_with_predictions": sum(s["ade_m"] is not None for s in scored),
            "undefined_reason": next((s["undefined_reason"] for s in scored if s["undefined_reason"]), None),
        }
    return out
