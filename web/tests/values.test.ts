import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { applications, errorEffects } from '@/config/applications'
import { MISSING, betterCaption, errorAmount, formatCount, formatDate, formatHash, formatMetric, formatPercent, formatRise, formatStat, limitText, statOf } from '@/lib/format'
import { acrossSeries, coverageRows, degradationSeries, familyRows, finishedErrors, historyByDay, horizonRows, plottable, riseRows, secondsAhead, strongestLevel, sweepSeries, trackingRows, unplacedLevels } from '@/lib/series'
import type { Curve, CurveLevel } from '@/types'
import { sceneSweep, stat, tracking } from './fixtures'

const level = (requested: number, achieved: number | null, ade: number | null, reason: string | null = null): CurveLevel => ({ requested, achieved_seed_mean: achieved, s_norm: achieved, ade_m: stat(ade), fde_m: stat(ade === null ? null : ade * 2), rmse_m: stat(null), prediction_coverage: stat(1), d_ade: stat(null), d_fde: stat(null), de_by_step_m: [], windows_by_seed: null, undefined_reason_by_seed: { 0: reason } })
const curve = (levels: CurveLevel[]): Curve => ({ severity_measure: 'rate', audc: null, robustness: null, levels })

describe('formatting', () => {
  it('shows an em dash for missing values and never for zero', () => {
    for (const value of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) expect(formatMetric(value)).toBe(MISSING)
    expect(formatMetric(0)).toBe('0.000')
    expect(formatCount(0)).toBe('0')
    expect(formatPercent(0)).toBe('0.0%')
    expect(formatStat(stat(null))).toBe(MISSING)
    expect(formatStat(stat(0))).toBe('0.000')
    expect(formatStat(null)).toBe(MISSING)
    expect(formatCount(null)).toBe(MISSING)
    expect(formatPercent(null)).toBe(MISSING)
    expect(formatHash(null)).toBe(MISSING)
    expect(formatDate(null)).toBe(MISSING)
  })

  it('reads a stored mean or a plain number, and keeps null as null', () => {
    expect(statOf(stat(1.25))).toBe(1.25)
    expect(statOf(0.7)).toBe(0.7)
    expect(statOf(stat(null))).toBeNull()
    expect(statOf(null)).toBeNull()
  })

  it('reports timestamps in UTC', () => { expect(formatDate('2026-11-01T09:18:50+00:00')).toBe('2026-11-01 09:18 UTC') })
})

describe('chart series', () => {
  it('plots achieved severity, not requested', () => {
    const series = degradationSeries(curve([level(0.1, 0.13, 1.4)]), 'ade_m', 'CV')
    expect(series.points[0]).toEqual({ x: 0.13, y: 1.4, requested: 0.1, note: null })
  })

  it('keeps an undefined level as null with its reason, so the line breaks there', () => {
    const series = degradationSeries(curve([level(0.3, 0.3, 1.5), level(0.4, 0.4, null, 'no_predicted_windows')]), 'ade_m', 'CV')
    expect(series.points).toHaveLength(2)
    expect(series.points[1].y).toBeNull()
    expect(series.points[1].x).toBe(0.4)
    expect(series.points[1].note).toContain('no_predicted_windows')
  })

  it('does not place a level with no achieved severity at zero', () => {
    const series = degradationSeries(curve([level(0.1, null, 1.4), level(0.2, 0.2, 1.5)]), 'ade_m', 'CV')
    expect(plottable(series).points.map((p) => p.x)).toEqual([0.2])
    expect(unplacedLevels(series)).toBe(1)
  })

  it('handles a missing curve without inventing points', () => {
    expect(degradationSeries(undefined, 'ade_m', 'CV').points).toEqual([])
    expect(acrossSeries(undefined, 'ade_m', 'CV').points).toEqual([])
  })

  it('reads stored across-scene levels without recomputing them', () => {
    const series = acrossSeries([{ requested: 0.1, achieved: stat(0.14), ade_m: stat(2.3), fde_m: stat(4.9) }], 'fde_m', 'CV')
    expect(series.points[0]).toMatchObject({ x: 0.14, y: 4.9, requested: 0.1 })
  })

  it('leaves a gap where a predictor has no horizon value', () => {
    expect(horizonRows([{ name: 'CV', steps: [1, 2] }, { name: 'GRU', steps: null }])).toEqual([{ step: 1, CV: 1, GRU: null }, { step: 2, CV: 2, GRU: null }])
    expect(horizonRows([{ name: 'CV', steps: [] }])).toEqual([])
  })

  it('keeps a missing tracking metric null, not zero', () => {
    expect(trackingRows(tracking.groups[0].sequences)).toEqual([{ sequence: '02', MOTA: 0.305, IDF1: 0.369, HOTA: null }])
  })

  it('keeps an undefined family value null with its reason', () => {
    expect(familyRows({ A: { value: null, reason: 'fewer than two' } }, ['A', 'D'])).toEqual([{ condition: 'A', value: null, reason: 'fewer than two' }, { condition: 'D', value: null, reason: null }])
  })

  it('counts runs per day and ignores runs with no timestamp', () => {
    expect(historyByDay(['2026-11-01T09:00:00', null, '2026-10-31T01:00:00', '2026-11-01T10:00:00'])).toEqual([{ day: '2026-10-31', runs: 1 }, { day: '2026-11-01', runs: 2 }])
  })
})

describe('no mock data can reach a screen', () => {
  const read = (path: string) => readFileSync(join(process.cwd(), path), 'utf-8')

  it('the application has no mock fixtures left', () => {
    for (const path of ['types/index.ts', 'lib/api/services.ts', 'lib/api/client.ts', 'components/trackvision-app.tsx', 'components/workspace.tsx', 'components/analysis.tsx', 'stores/experiment-store.tsx']) {
      expect(read(path)).not.toMatch(/mockDatasets|mockExperiments|mockChartSeries|mockEvents|createMockTrajectory|demoIndicator|MOCK DATA/)
    }
  })

  it('offers no control for a tracker or detector the pipeline does not have', () => {
    for (const path of ['components/trackvision-app.tsx', 'components/workspace.tsx', 'components/analysis.tsx']) {
      expect(read(path)).not.toMatch(/ByteTrack|BoT-SORT/)
      expect(read(path)).not.toMatch(/Run tracking|Run injection|Run prediction|Select file/)
    }
  })

  it('stays reachable at 127.0.0.1 and stays on this machine', () => {
    expect(read('next.config.mjs')).toMatch(/allowedDevOrigins:\s*\['127\.0\.0\.1'\]/)
    const { scripts } = JSON.parse(read('package.json')) as { scripts: Record<string, string> }
    expect(scripts.dev).toContain('--hostname 127.0.0.1')
    expect(scripts.start).toContain('--hostname 127.0.0.1')
  })

  it('names the API address in one place only', () => {
    expect(read('config/app.ts')).toContain('NEXT_PUBLIC_API_BASE_URL')
    for (const path of ['components/trackvision-app.tsx', 'components/workspace.tsx', 'components/video-stage.tsx', 'lib/api/services.ts', 'lib/api/client.ts']) expect(read(path)).not.toMatch(/127\.0\.0\.1|localhost/)
  })
})

describe('the sweep on a chart', () => {
  const body = sceneSweep()
  const label = (code: string) => code

  it('starts every curve from the clean tracks and uses the achieved amount, not the requested one', () => {
    const series = sweepSeries(body.errors[0], body.clean, 'ade', 'CV')
    expect(series.points[0]).toEqual({ x: 0, y: 0.678, requested: 0, note: null })
    expect(series.points[1]).toMatchObject({ x: 0.07, y: 0.95, requested: 0.05 })
  })

  it('shows shares of observations as percentages', () => {
    expect(sweepSeries(body.errors[1], body.clean, 'fde', 'CA').points[2].x).toBeCloseTo(39.9)
  })

  it('leaves a gap, not a zero, where nobody could be predicted', () => {
    const error = structuredClone(body.errors[1])
    error.levels[1].predictors.CV.ade = null
    expect(sweepSeries(error, body.clean, 'ade', 'CV').points[2]).toMatchObject({ y: null, note: 'Nobody could be predicted at this level.' })
  })

  it('compares errors only once each is finished, at its strongest level', () => {
    const partial = [{ ...body.errors[0], levels: body.errors[0].levels.slice(0, 1) }, body.errors[3]]
    expect(finishedErrors(partial).map((error) => error.code)).toEqual(['D'])
    const row = riseRows(body.errors, 'ade', ['CV', 'CA'], label)[0]
    expect(row.error).toBe('A')
    expect(row.CV).toBeCloseTo(821)
    expect(row.CA).toBeCloseTo(1221)
    expect(strongestLevel(body.errors[3]).requested).toBe(0.2)
  })

  it('starts coverage at everyone and stays empty for levels not finished', () => {
    const partial = { ...body.errors[1], levels: body.errors[1].levels.slice(0, 1), tested: 2 }
    expect(coverageRows([partial], label)).toEqual([{ step: 0, B: 100 }, { step: 1, B: 67 }, { step: 2, B: null }])
  })

  it('turns forecast steps into seconds, on scenes where a step is not 0.4 s too', () => {
    expect(secondsAhead([{ step: 1 }, { step: 12 }], 0.4).map((row) => row.step)).toEqual([0.4, 4.8])
    expect(secondsAhead([{ step: 1 }, { step: 12 }], 6 / 14).map((row) => row.step)).toEqual([0.4, 5.1])
  })
})

describe('the wording of measured changes', () => {
  it('signs a change and keeps large numbers readable', () => {
    expect(formatRise(8.21)).toBe('+821%')
    expect(formatRise(217.31)).toBe('+21,731%')
    expect(formatRise(0.004)).toBe('+0.4%')
    expect(formatRise(-0.38)).toBe('\u221238%')
    expect(formatRise(0)).toBe('0%')
    expect(formatRise(null)).toBe(MISSING)
  })

  it('states each amount in the words of its error', () => {
    expect(errorAmount('A', 0.0176, 'm')).toBe('0.02 m')
    expect(errorAmount('A', 6.23, 'px')).toBe('6.23 px')
    expect(errorAmount('B', 0.0154, 'm')).toBe('1.5%')
    expect(errorAmount('D', 1.41, 'm')).toBe('1.41 per 100')
  })

  it('has a sentence for every outcome, so no dash is left unexplained', () => {
    const limit = (status: string) => ({ status, value: 0.1, verified_up_to: 0.3, tested_up_to: 0.4 })
    expect(limitText('B', limit('crossed'), 'm')).toBe('Up to 10.0%')
    expect(limitText('B', limit('not_reached'), 'm')).toBe('Stays within the limit at every tested level, up to 30.0%')
    expect(limitText('B', limit('truncated'), 'm')).toContain('too few people are left')
    expect(limitText('B', limit('uncomputable'), 'm')).toBe('There is not enough data to say')
    expect(limitText('B', undefined, 'm')).toContain('Calculating')
    expect(betterCaption('A', { status: 'uncomputable', at: null, better_when_clean: null }, 'm')).toContain('not enough data')
    expect(betterCaption('A', null, 'm')).toContain('Calculating')
  })
})

describe('the application details', () => {
  const available = new Set(['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2', 'MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13'])

  it('offers a wide range, each with a unique name', () => {
    expect(applications.length).toBeGreaterThanOrEqual(10)
    expect(new Set(applications.map((item) => item.id)).size).toBe(applications.length)
    expect(new Set(applications.map((item) => item.name)).size).toBe(applications.length)
  })

  it('ranks each of the four errors exactly once in every application, with reasons', () => {
    for (const item of applications) {
      expect(item.priorities.map((priority) => priority.code).sort()).toEqual(['A', 'B', 'C', 'D'])
      for (const priority of item.priorities) {
        expect(priority.why.length).toBeGreaterThan(40)
        expect(priority.inUse.length).toBeGreaterThan(10)
      }
    }
  })

  it('measures only on full scenes that exist, never on a part', () => {
    for (const item of applications) {
      expect(item.scenes.length).toBeGreaterThan(0)
      for (const scene of item.scenes) expect(available.has(scene.id)).toBe(true)
    }
  })

  it('says something about ADE, FDE, CV and CA for every application and every error', () => {
    for (const item of applications) for (const text of [item.metric.why, item.metric.ade, item.metric.fde, item.predictor, item.needs, item.summary]) expect(text.length).toBeGreaterThan(10)
    for (const effect of Object.values(errorEffects)) for (const text of [effect.cv, effect.ca, effect.ade, effect.fde]) expect(text.length).toBeGreaterThan(20)
  })

  it('never puts a measured number or an experiment label in the hand-written text', () => {
    const text = JSON.stringify(applications)
    expect(text).not.toMatch(/\bE[1-9]\b|\bP-\d+\b|\bH\d\b/)
    expect(text).not.toMatch(/\d+\.\d+|\+\d+%/)
  })
})
