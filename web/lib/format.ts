import type { ErrorCode, EvidenceLevel, Stat, SweepBetter, SweepLimit } from '@/types'

export const MISSING = '—'

export function formatMetric(value: number | null | undefined, digits = 3) {
  return value === null || value === undefined || !Number.isFinite(value) ? MISSING : value.toFixed(digits)
}

export function formatStat(stat: Stat | number | null | undefined, digits = 3) {
  return formatMetric(typeof stat === 'object' && stat !== null ? stat.mean : stat, digits)
}

export function formatCount(value: number | null | undefined) {
  return value === null || value === undefined ? MISSING : value.toLocaleString('en-US')
}

export function formatPercent(value: number | null | undefined) {
  return value === null || value === undefined ? MISSING : `${(value * 100).toFixed(1)}%`
}

export function formatDate(value: string | null | undefined) {
  if (!value) return MISSING
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : `${date.toISOString().slice(0, 16).replace('T', ' ')} UTC`
}

export function formatHash(value: string | null | undefined) {
  return value ? value.slice(0, 12) : MISSING
}

export function statOf(stat: Stat | number | null | undefined) {
  if (stat === null || stat === undefined) return null
  return typeof stat === 'number' ? stat : stat.mean ?? null
}

export const evidenceLabels: Record<EvidenceLevel, string> = {
  engineering_only: 'Engineering only',
  engineering_and_descriptive: 'Engineering and descriptive',
  descriptive_local_only: 'Descriptive, local only',
  feasibility_only: 'Feasibility only',
  diagnostic_only: 'Diagnostic only',
  scaffold_check: 'Scaffold check',
}

export function severityLabel(measure: string | null | undefined) {
  if (measure === 'rmse_pos_m') return 'Achieved position RMSE (m)'
  if (measure === 'rmse_pos_px') return 'Achieved position RMSE (px)'
  if (measure === 'rate') return 'Achieved rate'
  if (!measure) return 'Achieved severity'
  return `Achieved ${measure.replaceAll('_', ' ')}`
}

// A change as a percentage, with its sign. Small changes keep a decimal, so +0.4% does not read as nothing.
export function formatRise(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING
  if (value === 0) return '0%'
  const percent = value * 100
  const size = Math.abs(percent) < 10 ? Math.abs(percent).toFixed(1) : Math.round(Math.abs(percent)).toLocaleString('en-US')
  return `${percent > 0 ? '+' : '\u2212'}${size}%`
}

// How much of an error, in words a reader can follow. Missing detections and broken tracks are shares of observations.
export function errorAmount(code: ErrorCode, value: number | null | undefined, unit: string) {
  if (value === null || value === undefined) return MISSING
  if (code === 'A') return `${formatMetric(value, 2)} ${unit}`
  if (code === 'D') return `${formatMetric(value, 2)} per 100`
  return `${formatMetric(value * 100, 1)}%`
}

export function errorAxis(code: ErrorCode, unit: string) {
  return { A: `Measured position error (${unit})`, B: 'Detections missing (%)', C: 'Tracks broken per step (%)', D: 'Switches per 100 close-by frames' }[code]
}

export function betterCaption(code: ErrorCode, better: SweepBetter | null, unit: string) {
  if (!better) return 'Calculating\u2026'
  const first = better.better_when_clean
  const other = first === 'CV' ? 'CA' : 'CV'
  if (better.status === 'no_crossing') return `${first} stays better at every tested level.`
  if (better.status === 'crossed') return `${first} is better at first, and ${other} catches up at about ${errorAmount(code, better.at, unit)}.`
  if (better.status === 'truncated') return `${first} stays better until too few people are left to measure.`
  return 'There is not enough data to compare CV and CA here.'
}

export function limitText(code: ErrorCode, limit: SweepLimit | undefined, unit: string) {
  if (!limit) return 'Calculating\u2026'
  if (limit.status === 'crossed') return `Up to ${errorAmount(code, limit.value, unit)}`
  if (limit.status === 'not_reached') return `Stays within the limit at every tested level, up to ${errorAmount(code, limit.verified_up_to, unit)}`
  if (limit.status === 'truncated') return `Within the limit up to ${errorAmount(code, limit.verified_up_to, unit)}, then too few people are left to measure`
  if (limit.status === 'exceeded_at_clean') return 'Over the limit with no error at all'
  return 'There is not enough data to say'
}

// Plain definitions for the metric tooltips. They say what a value measures, never what it implies (spec section 1.2).
export const glossary: Record<string, string> = {
  ADE: 'Average displacement error: mean distance between predicted and true positions over the horizon, in metres, or in video pixels on scenes with no calibration.',
  FDE: 'Final displacement error: distance between predicted and true positions at the last horizon step, in metres, or in video pixels on scenes with no calibration.',
  RMSE: 'Root mean square position error over the horizon, in metres, or in video pixels on scenes with no calibration.',
  Coverage: 'Share of eligible windows that received a prediction.',
  MOTA: 'Multiple Object Tracking Accuracy, from official TrackEval.',
  IDF1: 'Identity F1 score, from official TrackEval.',
  HOTA: 'Higher Order Tracking Accuracy, from official TrackEval, averaged over 19 alpha thresholds.',
  IDSW: 'Identity switches counted by official TrackEval.',
  Frag: 'Track fragmentations counted by official TrackEval.',
  AUDC: 'Area under the degradation curve, computed by the analysis stage and read here unchanged.',
  'Achieved severity': 'The severity measured after injection. Curves use this value, not the requested one.',
  'Requested severity': 'The severity level the experiment asked for.',
  's*': 'Lowest achieved severity at which relative ADE degradation reaches q. Descriptive, not a requirement.',
}
