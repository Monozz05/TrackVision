import { statOf } from '@/lib/format'
import type { AcrossLevel, Curve, CurveLevel, ErrorCode, FamilyEntry, SweepClean, SweepError, TrackingSequence } from '@/types'

// Pure shaping for the charts. Values are copied from the API record unchanged: a missing value stays null, so a line
// breaks at that point instead of dropping to zero or joining its neighbours. Nothing is averaged here.
export interface Point { x: number | null; y: number | null; requested: number | null; note: string | null }
export interface Series { name: string; points: Point[] }

function undefinedNote(level: CurveLevel) {
  const reasons = Object.values(level.undefined_reason_by_seed ?? {}).filter(Boolean)
  return reasons.length ? `Undefined on ${reasons.length} seed(s): ${reasons[0]}` : null
}

export function degradationSeries(curve: Curve | undefined, metric: 'ade_m' | 'fde_m', name: string): Series {
  return { name, points: (curve?.levels ?? []).map((level) => ({ x: level.achieved_seed_mean, y: statOf(level[metric]), requested: level.requested, note: undefinedNote(level) })) }
}

export function acrossSeries(levels: AcrossLevel[] | undefined, metric: 'ade_m' | 'fde_m', name: string): Series {
  return { name, points: (levels ?? []).map((level) => ({ x: statOf(level.achieved), y: statOf(level[metric]), requested: level.requested, note: null })) }
}

// A level with no achieved severity has no position on the axis, so it is left out and counted, not placed at zero.
export function plottable(series: Series): Series { return { ...series, points: series.points.filter((point) => point.x !== null) } }
export function unplacedLevels(series: Series) { return series.points.filter((point) => point.x === null).length }

export function horizonRows(profiles: Array<{ name: string; steps: number[] | null | undefined }>) {
  const length = Math.max(0, ...profiles.map((profile) => profile.steps?.length ?? 0))
  return Array.from({ length }, (_, index) => {
    const row: Record<string, number | null> = { step: index + 1 }
    for (const profile of profiles) row[profile.name] = profile.steps?.[index] ?? null
    return row
  })
}

export function trackingRows(sequences: TrackingSequence[]) {
  return sequences.map((row) => ({ sequence: row.sequence.replace('MOT17-', ''), MOTA: row.metrics.mota, IDF1: row.metrics.idf1, HOTA: row.metrics.hota }))
}

export function familyRows(family: Record<string, FamilyEntry> | undefined, conditions: string[]) {
  return conditions.map((condition) => ({ condition, value: family?.[condition]?.value ?? null, reason: family?.[condition]?.reason ?? null }))
}

const SHOWN_AS_PERCENT: Record<ErrorCode, number> = { A: 1, B: 100, C: 100, D: 1 }

// One predictor's ADE or FDE as an error grows, starting from the clean tracks. A level nobody could be predicted at is a gap.
export function sweepSeries(error: SweepError, clean: SweepClean | null, metric: 'ade' | 'fde', predictor: string): Series {
  const scale = SHOWN_AS_PERCENT[error.code]
  return { name: predictor, points: [{ x: 0, y: clean?.[predictor]?.[metric] ?? null, requested: 0, note: null }, ...error.levels.map((level) => {
    const y = level.predictors[predictor]?.[metric] ?? null
    return { x: level.achieved === null ? null : level.achieved * scale, y, requested: level.requested, note: y === null ? 'Nobody could be predicted at this level.' : null }
  })] }
}

export const finishedErrors = (errors: SweepError[]) => errors.filter((error) => error.levels.length === error.tested)
export const strongestLevel = (error: SweepError) => error.levels[error.levels.length - 1]

// The change at the strongest tested level, as a percentage for a chart axis.
export function riseRows(errors: SweepError[], metric: 'ade' | 'fde', predictors: string[], label: (code: ErrorCode) => string) {
  return finishedErrors(errors).map((error) => {
    const row: Record<string, string | number | null> = { error: label(error.code) }
    for (const predictor of predictors) {
      const rise = strongestLevel(error).predictors[predictor]?.[`${metric}_rise`] ?? null
      row[predictor] = rise === null ? null : rise * 100
    }
    return row
  })
}

// Who still gets a forecast at each level. Level 0 is the clean tracks, where everyone does.
export function coverageRows(errors: SweepError[], label: (code: ErrorCode) => string) {
  const length = Math.max(0, ...errors.map((error) => error.tested))
  return Array.from({ length: length + 1 }, (_, level) => {
    const row: Record<string, number | null> = { step: level }
    for (const error of errors) {
      const coverage = error.levels[level - 1]?.coverage ?? null
      row[label(error.code)] = level === 0 ? 100 : coverage === null ? null : coverage * 100
    }
    return row
  })
}

export function secondsAhead(rows: Array<Record<string, number | null>>, secondsPerStep: number) {
  return rows.map((row) => ({ ...row, step: Number(((row.step as number) * secondsPerStep).toFixed(1)) }))
}

export function uniqueValues<T>(values: T[]) { return [...new Set(values)] }

export function historyByDay(timestamps: Array<string | null>) {
  const counts = new Map<string, number>()
  for (const value of timestamps) if (value) counts.set(value.slice(0, 10), (counts.get(value.slice(0, 10)) ?? 0) + 1)
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([day, runs]) => ({ day, runs }))
}
