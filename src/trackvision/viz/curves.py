"""Static SVG figures drawn from a stored analysis.json. Standard library only (matplotlib is not a dependency).

Every number drawn is read from the analysis document; nothing here computes a research metric. Missing values stay
visible: an undefined value is a hollow cross on the rail under its panel, a point above normalized severity 1 is
hollow and sits in the shaded region, and a line never bridges an undefined point. Coverage has its own panel below
conditional error, so a low-coverage predictor cannot look good on error alone. There is one y-scale per panel. A
partial-fold P-15 mean is hollow and unconnected: it is reported, but it rests on fewer folds and cannot qualify.

Identity is never colour alone: each series has a fixed colour, its own marker shape and a direct label. Colours
follow the entity. The data table for any figure is the matching section of analysis.json.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap

from trackvision.io.atomic import atomic_write_text

WIDTH, PANEL, GAP, RAIL = 720, 230, 62, 22
NOTE_LINE, NOTE_CHARS = 16, 104
LEFT, RIGHT, TOP = 70, 110, 76
SLOTS = {"CV": 1, "CA": 2, "LSTM": 3, "GRU": 4, "A": 1, "B1": 2, "B2": 3, "C": 4, "D": 5}
SHAPES = {1: "circle", 2: "square", 3: "triangle", 4: "diamond", 5: "cross", 6: "circle"}
NEUTRAL = 6                 # a lone series names nothing in a legend, so it takes no categorical colour
STYLE = """
.root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e0;--band:#f1f0ec;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#52514e}
@media (prefers-color-scheme:dark){.root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#333331;--band:#262625;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#c3c2b7}}
text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;fill:var(--ink2);font-size:11px}
.title{fill:var(--ink);font-size:14px;font-weight:600}.note{font-size:11px}.label{fill:var(--ink);font-size:11px}
.grid{stroke:var(--grid);stroke-width:1}.axis{stroke:var(--ink2);stroke-width:1}
"""


def _ticks(low: float, high: float, n: int = 5) -> list[float]:
    if high <= low:
        high = low + 1.0
    return [low + (high - low) * i / (n - 1) for i in range(n)]


def _marker(shape: str, x: float, y: float, slot: int, hollow: bool, tip: str) -> str:
    colour = f"var(--s{slot})"
    fill = "var(--surface)" if hollow else colour
    style = f'fill="{fill}" stroke="{colour if hollow else "var(--surface)"}" stroke-width="2"'
    tip = f"<title>{escape(tip)}</title>"
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" {style}>{tip}</circle>'
    if shape == "square":
        return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" {style}>{tip}</rect>'
    points = {"triangle": [(0, -6), (5.5, 4), (-5.5, 4)], "diamond": [(0, -6), (6, 0), (0, 6), (-6, 0)],
              "cross": [(-2, -6), (2, -6), (2, -2), (6, -2), (6, 2), (2, 2), (2, 6), (-2, 6), (-2, 2), (-6, 2), (-6, -2), (-2, -2)]}[shape]
    return f'<polygon points="{" ".join(f"{x + a:.1f},{y + b:.1f}" for a, b in points)}" {style}>{tip}</polygon>'


def _panel(y0: float, title: str, x_label: str, y_label: str, x_range, series: list, shade_from=None,
           y_limits=None, x_ticks=None, shade_note=True) -> list[str]:
    """One panel. A series is (name, slot, [(x, y or None, kind, tip)]); kind is 'used', 'above', 'partial' or
    'undefined'."""
    x_low, x_high = x_range
    defined = [p[1] for s in series for p in s[2] if p[1] is not None]
    y_low, y_high = y_limits or (None, None)
    y_low = min(defined, default=0.0) if y_low is None else y_low
    y_high = max(defined, default=1.0) if y_high is None else y_high
    if y_high <= y_low:
        y_high = y_low + 1.0
    plot_w = WIDTH - LEFT - RIGHT
    sx = lambda v: LEFT + (v - x_low) / ((x_high - x_low) or 1.0) * plot_w
    sy = lambda v: y0 + PANEL - (v - y_low) / (y_high - y_low) * PANEL
    out = [f'<text class="label" x="{LEFT}" y="{y0 - 10}">{escape(title)}</text>']
    if shade_from is not None and shade_from < x_high:
        out.append(f'<rect x="{sx(shade_from):.1f}" y="{y0}" width="{sx(x_high) - sx(shade_from):.1f}" height="{PANEL}" fill="var(--band)"/>')
        if shade_note:
            out.append(f'<text class="note" x="{WIDTH - RIGHT}" y="{y0 - 10}" text-anchor="end">shaded, hollow: above 1, not in AUDC</text>')
    for t in _ticks(y_low, y_high):
        out.append(f'<line class="grid" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text x="{LEFT - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:.3g}</text>')
    for t in x_ticks or _ticks(x_low, x_high):
        out.append(f'<text x="{sx(t):.1f}" y="{y0 + PANEL + RAIL + 16}" text-anchor="middle">{t:.3g}</text>')
    out.append(f'<line class="axis" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{y0 + PANEL}" y2="{y0 + PANEL}"/>')
    out.append(f'<text x="{WIDTH - RIGHT}" y="{y0 + PANEL + RAIL + 32}" text-anchor="end">{escape(x_label)}</text>')
    out.append(f'<text transform="rotate(-90 16 {y0 + PANEL / 2})" x="16" y="{y0 + PANEL / 2}" text-anchor="middle">{escape(y_label)}</text>')
    rail_y, any_undefined, ends = y0 + PANEL + RAIL / 2 + 2, False, []
    for name, slot, points in series:
        run: list[str] = []
        for x, y, kind, _ in sorted(points, key=lambda p: p[0]):
            if y is None or kind == "partial":
                if len(run) > 1:
                    out.append(f'<polyline fill="none" stroke="var(--s{slot})" stroke-width="2" points="{" ".join(run)}"/>')
                run = []
            else:
                run.append(f"{sx(x):.1f},{sy(y):.1f}")
        if len(run) > 1:
            out.append(f'<polyline fill="none" stroke="var(--s{slot})" stroke-width="2" points="{" ".join(run)}"/>')
        for x, y, kind, tip in points:
            if y is None:
                any_undefined = True
                out.append(f'<text x="{sx(x):.1f}" y="{rail_y + 4:.1f}" text-anchor="middle" style="fill:var(--s{slot});font-weight:700">×<title>{escape(tip)}</title></text>')
            else:
                out.append(_marker(SHAPES[slot], sx(x), sy(y), slot, kind in ("above", "partial"), tip))
        last = [p for p in sorted(points, key=lambda p: p[0]) if p[2] == "used" and p[1] is not None]
        if last:
            ends.append([sy(last[-1][1]), sx(last[-1][0]), name])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, _, name in ends:
        out.append(f'<text class="label" x="{WIDTH - RIGHT + 12}" y="{y + 4:.1f}">{escape(name)}</text>')
    if any_undefined:
        out.append(f'<text class="note" x="{LEFT}" y="{rail_y + 4:.1f}">× undefined</text>')
    return out


def _notes(paragraphs: list[str]) -> list[str]:
    return [line for text in paragraphs for line in wrap(text, NOTE_CHARS)]


def _document(name: str, title: str, subtitle: str, panels: int, body: list[str], notes: tuple = ()) -> str:
    height = TOP + panels * (PANEL + GAP + RAIL) + 12
    lines = _notes(list(notes))
    body = body + [f'<text class="note" x="{LEFT}" y="{height + i * NOTE_LINE}">{escape(t)}</text>' for i, t in enumerate(lines)]
    height += len(lines) * NOTE_LINE
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" '
            f'role="img" aria-labelledby="t" class="root"><title id="t">{escape(title)}</title><style>{STYLE}</style>'
            f'<rect width="{WIDTH}" height="{height}" style="fill:var(--surface)"/>'
            f'<text class="title" x="{LEFT}" y="22">{escape(title)}</text>'
            f'<text class="note" x="{LEFT}" y="39">{escape(subtitle)}</text>' + "".join(body) + "</svg>\n")


def _legend(y: float, entries: list, x: float = LEFT) -> list[str]:
    out = []
    for name, slot in entries:
        out.append(_marker(SHAPES[slot], x + 5, y - 4, slot, False, name))
        out.append(f'<text x="{x + 15}" y="{y}">{escape(name)}</text>')
        x += 24 + 7 * len(name) + 14
    return out


def degradation_figure(doc: dict, population: str, history: str, scene: str, condition: str) -> str:
    curves = doc["curves"][population][history][scene]
    predictors = list(curves)
    measure, error_series, coverage_series, xs, above = None, [], [], [], False
    for p in predictors:
        c = curves[p][condition]
        norm_ok = c["normalization"].get("status") == "computable"
        slot, err, cov = SLOTS[p], [], []
        for row in c["levels"]:
            x = row.get("s_norm") if norm_ok else row["requested"]
            kind = "above" if norm_ok and x > 1.0 + 1e-12 else "used"
            above = above or kind == "above"
            d = row["d_ade"]["mean"]
            tip = f'{p}, requested {row["requested"]}: D = {d}, ADE = {row["ade_m"]["mean"]} m'
            err.append((x, d, "undefined" if d is None else kind, tip))
            cover = row["prediction_coverage"]["mean"]
            cov.append((x, cover, "undefined" if cover is None else kind, f'{p}, requested {row["requested"]}: coverage {cover}'))
            xs.append(x)
        error_series.append((p, slot, err))
        coverage_series.append((p, slot, cov))
        measure = measure or c["normalization"].get("measure")
    norm_ok = curves[predictors[0]][condition]["normalization"].get("status") == "computable"
    x_label = "normalized achieved severity s_norm" if norm_ok else "requested level (s_norm uncomputable)"
    x_range = (min(xs), max(xs))
    audc_text = "; ".join(f'{p} AUDC {curves[p][condition]["audc"]["value"]:.3g}' if curves[p][condition]["audc"]["value"] is not None
                          else f"{p} AUDC uncomputable" for p in predictors)
    ticks = None
    if norm_ok:
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0] + ([x_range[1]] if x_range[1] >= 1.15 else [])
    body = _legend(58, [(p, SLOTS[p]) for p in predictors])
    y1 = TOP + 16
    body += _panel(y1, "Relative ADE degradation D (conditional on predicted windows)", x_label, "D", x_range, error_series,
                   1.0 if above else None, x_ticks=ticks)
    body += _panel(y1 + PANEL + GAP + RAIL, "Prediction coverage", x_label, "coverage", x_range, coverage_series,
                   1.0 if above else None, (0.0, 1.0), x_ticks=ticks, shade_note=False)
    return _document(f"{scene}-{condition}", f"{scene}, condition {condition}: degradation and coverage, H = {history}",
                     f"{population}. {audc_text}. {_scope(doc)}", 2, body)


def horizon_figure(doc: dict, population: str, history: str, scene: str, condition: str) -> str:
    curves = doc["curves"][population][history][scene]
    series = []
    for p, by_condition in curves.items():
        rows = by_condition[condition]["levels"]
        top = rows[-1]
        steps = top["de_by_step_m"]
        points = ([(i + 1.0, v, "used", f"{p}, step {i + 1}: {v} m") for i, v in enumerate(steps)] if steps else
                  [(1.0, None, "undefined", f"{p}: no horizon errors at the top level")])
        series.append((p, SLOTS[p], points))
    level = next(iter(curves.values()))[condition]["levels"][-1]["requested"]
    horizon = max(len(s[2]) for s in series)
    body = _legend(58, [(p, SLOTS[p]) for p in curves])
    body += _panel(TOP + 16, f"Displacement error by predicted step, requested level {level}", "predicted step", "error (m)",
                   (1.0, float(max(horizon, 2))), series, x_ticks=[float(t) for t in range(1, horizon + 1)])
    return _document("horizon", f"{scene}, condition {condition}: horizon-resolved error, H = {history}", _scope(doc), 1, body)


def p15_figure(doc: dict, population: str, predictor: str) -> str:
    across, histories = doc["across_scenes"][population], [int(h) for h in doc["spec"]["histories"]]
    clean = [(float(h), across[str(h)][predictor]["clean"]["ade_m"]["mean"], "used", f"H = {h}") for h in histories]
    audc_series = []
    for condition in doc["spec"]["conditions"]:
        points = []
        for h in histories:
            cell = across[str(h)][predictor][condition]["audc"]
            v = cell["mean"]
            partial = doc["p15"][population][predictor][condition]["per_history"][str(h)]["partial_folds"]
            tip = f'{condition}, H = {h}: AUDC {v}, {cell["n_valid"]} valid of {cell["n_valid"] + cell["n_undefined"]} scenes'
            kind = "undefined" if v is None else "partial" if partial else "used"
            points.append((float(h), v, kind, tip))
        audc_series.append((condition, SLOTS[condition], points))
    minimum = "; ".join(
        f'{c}: {r["minimum_history"] if r["status"] == "found" else r["status"]}{" (partial folds)" if r["partial_folds"] else ""}'
        for c, r in doc["p15"][population][predictor].items())
    ticks = [float(h) for h in histories]
    x_range = (ticks[0], ticks[-1])
    body = _legend(58, [(c, SLOTS[c]) for c in doc["spec"]["conditions"]])
    y1 = TOP + 16
    body += _panel(y1, "Clean ADE (m), mean across scenes", "history length H", "clean ADE (m)", x_range,
                   [(predictor, NEUTRAL, clean)], x_ticks=ticks)
    body += _panel(y1 + PANEL + GAP + RAIL, "AUDC by condition, mean across scenes (hollow: partial folds, does not qualify)",
                   "history length H", "AUDC", x_range, audc_series, x_ticks=ticks)
    return _document("p15", f"P-15, {predictor}, {population}", f"Minimum history: {minimum}. {_scope(doc)}", 2, body,
                     (doc["population_labels"][population],))


def conditional_error_figure(doc: dict, population: str, predictor: str, condition: str) -> str:
    """Decision 0005, section 3: ordinary and paired AUDC beside coverage, window counts and fold counts."""
    entry = doc["p15"][population][predictor][condition]
    rows, histories = entry["conditional_error"]["by_history"], [str(h) for h in doc["spec"]["histories"]]
    ticks = [float(h) for h in histories]

    def audc_points(name: str, field: str) -> list:
        points = []
        for h in histories:
            cell = rows[h][field]
            kind = "undefined" if cell["mean"] is None else "partial" if cell["n_undefined"] else "used"
            tip = f'{name}, H = {h}: AUDC {cell["mean"]}, {cell["n_valid"]} valid, {cell["n_undefined"]} undefined folds'
            points.append((float(h), cell["mean"], kind, tip))
        return points

    audc_series = [("ordinary", 1, audc_points("ordinary", "audc")), ("paired", 2, audc_points("paired", "audc_paired"))]
    coverage, predicted = [], []
    for i, h in enumerate(histories, start=1):
        cover = [(r["requested"], r["prediction_coverage"]["mean"]) for r in rows[h]["levels"]]
        coverage.append((f"H{h}", i, [(x, y, "undefined" if y is None else "used", f"H = {h}, requested {x}: coverage {y}")
                                      for x, y in cover]))
        counts = [(r["requested"], r["windows"]["predicted_total"], r["windows"]["eligible_total"]) for r in rows[h]["levels"]]
        predicted.append((f"H{h}", i, [(float(x), float(n), "used", f"H = {h}, requested {x}: {n} of {e} windows predicted")
                                       for x, n, e in counts]))
    levels = [float(r["requested"]) for r in rows[histories[0]]["levels"]]
    folds = "; ".join(f'H{h}: {rows[h]["audc"]["n_valid"]} valid, {rows[h]["audc"]["n_undefined"]} undefined' for h in histories)
    top = "; ".join(f'H{h}: {rows[h]["levels"][-1]["windows"]["predicted_total"]}' for h in histories)
    step = PANEL + GAP + RAIL
    y1 = TOP + 16
    body = _legend(58, [("ordinary", 1), ("paired", 2)])
    body += _panel(y1, "AUDC by history (hollow: partial folds, does not qualify)", "history length H", "AUDC",
                   (ticks[0], ticks[-1]), audc_series, x_ticks=ticks)
    body += _legend(y1 + step - 10, [(f"H{h}", i) for i, h in enumerate(histories, start=1)], 330)
    body += _panel(y1 + step, "Prediction coverage by requested level", "requested level", "coverage",
                   (levels[0], levels[-1]), coverage, y_limits=(0.0, 1.0), x_ticks=levels)
    body += _panel(y1 + 2 * step, "Predicted windows by requested level, total over scenes and seeds", "requested level",
                   "predicted windows", (levels[0], levels[-1]), predicted, y_limits=(0.0, None), x_ticks=levels)
    return _document("conditional-error", f"P-15, {predictor}, condition {condition}: conditional error, {population}", _scope(doc),
                     3, body, (entry["conditional_error"]["note"], f"Ordinary AUDC folds. {folds}.",
                               f"Predicted windows at the top level, total over scenes and seeds. {top}.",
                               doc["population_labels"][population]))


def _scope(doc: dict) -> str:
    return "Local only, descriptive." if doc["scope"] == "ethhotel_descriptive" else "Synthetic engineering only."


def export_figures(doc: dict, folder: Path) -> list[str]:
    """Write every figure the analysis supports; returns the file names."""
    folder, names = Path(folder), []

    def write(name: str, svg: str) -> None:
        atomic_write_text(folder / name, svg)
        names.append(name)

    spec = doc["spec"]
    for population, by_history in doc["curves"].items():
        for history, by_scene in by_history.items():
            if population != "per_history" or (len(spec["histories"]) > 1 and history != "8"):
                continue
            for scene in by_scene:
                for condition in spec["conditions"]:
                    write(f"degradation_{scene}_{condition}_h{history}.svg",
                          degradation_figure(doc, population, history, scene, condition))
                    write(f"horizon_{scene}_{condition}_h{history}.svg",
                          horizon_figure(doc, population, history, scene, condition))
    if "p15" in doc:
        for population, by_predictor in doc["p15"].items():
            for predictor, by_condition in by_predictor.items():
                write(f"p15_{population}_{predictor}.svg", p15_figure(doc, population, predictor))
                for condition, entry in by_condition.items():
                    if "conditional_error" in entry:
                        write(f"conditional_error_{population}_{predictor}_{condition}.svg",
                              conditional_error_figure(doc, population, predictor, condition))
    return names
