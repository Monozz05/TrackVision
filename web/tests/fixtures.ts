import { vi } from 'vitest'
import type { ErrorCode, Experiment, Health, InjectionResult, SceneSweep, Stat, SweepError, SweepLevel, TrackingResult, WorkspaceCatalogue, WorkspaceScene } from '@/types'

// Test fixtures only. Nothing under web/tests is imported by the application.
export const stat = (mean: number | null): Stat => ({ mean, sd: mean === null ? null : 0.1, n_valid: mean === null ? 0 : 3, n_undefined: mean === null ? 3 : 0, undefined: {} })

export const health: Health = { status: 'ok', version: '0.1.0', read_only: true, results_directory_available: true, experiment_records: 2, invalid_records: 0, skipped_folders: 0, timestamp: '2026-11-01T09:00:00+00:00' }

export function run(overrides: Partial<Experiment> = {}): Experiment {
  return {
    experiment_id: 'fixture~sweep', record_id: null, kind: 'sweep', kind_label: 'Phase 4 injection sweep', scope: 'synthetic_engineering', timestamp: '2026-11-01T06:10:52+00:00',
    status: 'complete', status_reason: null, evidence_level: 'engineering_only', research_results: false, label: 'SYNTHETIC ENGINEERING FEASIBILITY ONLY. Fixture label.', notice: null, latest_in_series: true,
    experiments: ['P-4'], datasets: ['Synthetic'], scenes: ['SYN-1'], predictors: ['CV', 'GRU'], conditions: ['A', 'B2'], histories: [8], seeds: [0, 1, 2], horizon_len: 12, detector: null, tracker: null,
    counts: { expected_cells: 10, finished_cells: 10, by_status: { failed: 0, blocked: 4 }, by_experiment: {}, scope_complete: true, research_complete: false },
    integrity: { status: 'verified', verified: 1, failed: [] },
    provenance: { git: { sha: 'abcdef0123456789', dirty: true }, environment: null, specification: { version: '4.15' }, config_hash: 'c'.repeat(64), settings_hash: null, source_sha256: 'd'.repeat(64), spec_sha256: null, dataset_manifest_hash: null, trackeval_commit: null, licence: null, results_sha256: null, version: '0.1.0', trackvision_version: null, association_mechanism: null, association_status: 'unresolved', association_note: null, association_blocker: 'No permitted value can be recorded.', association_sidecar: null },
    ...overrides,
  }
}

export const tracking: TrackingResult = {
  experiment: run({ experiment_id: 'fixture~mot17', kind: 'mot17_tracking', kind_label: 'MOT17 tracking (E1 to E3)', label: 'LIMITED MOT17 FEASIBILITY (decision 0006). Fixture.' }),
  source: 'mot17_tracking', pooled_result: null, prediction_note: 'MOT17 is never used for trajectory prediction', not_satisfied: ['section 11.5: ByteTrack and BoT-SORT - the existing IoU feasibility tracker used instead'],
  groups: [{ name: 'FRCNN', primary: true, tracker_settings: { tracker: { name: 'IouTracker (feasibility)', match_iou: 0.3 } }, sequences: [
    { sequence: 'MOT17-02', regime: 'static', frame_rate: 30, frames_evaluated: [301, 600], metrics: { mota: 0.305, idf1: 0.369, hota: null, idSwitches: 55, fragmentations: 74 }, official_trackeval: null, project_diagnostics: null },
  ] }],
}

const scene = (id: string, dataset: WorkspaceScene['dataset'], name: string, full: boolean, stored: string | null): WorkspaceScene => ({ id, dataset, name, parent: null, regime: 'static', has_video: true, errors_available: full, errors_reason: full ? null : 'The annotations for this scene are not on this machine.', evidence_level: 'descriptive_local_only', predictors: full ? ['CV', 'CA'] : [], stored_run: stored, detector: null, video_state: 'ready', ...(dataset === 'UCY' || dataset === 'MOT17' ? { unit: 'px', caution: 'In video pixels, because this scene has no calibration.' } : { unit: 'm', caution: null }) })

const part = (whole: WorkspaceScene, number: number): WorkspaceScene => ({ ...whole, id: `${whole.id}-${number}`, name: `${whole.name} part ${number}`, parent: whole.id, stored_run: null, caution: [`Part of a longer recording, not a separate scene, so its values must not be pooled with other parts.`, whole.caution].filter(Boolean).join(' ') })
const eth = scene('ETH', 'EWAP', 'ETH', true, 'fixture~ethhotel')
const zara1 = scene('ZARA1', 'UCY', 'ZARA1', true, null)

export const catalogue: WorkspaceCatalogue = {
  items: [eth, part(eth, 1), scene('HOTEL', 'EWAP', 'HOTEL', true, 'fixture~ethhotel'), scene('UNIV', 'UCY', 'UNIV', false, null), zara1, part(zara1, 1), scene('MOT17-02', 'MOT17', '02', true, 'fixture~mot17'), scene('SYN-501', 'Synthetic', '501', true, 'fixture~sweep')],
  controls: { A: { max: 1, unit: 'm', protocol_max: 0.5 }, B: { max: 1, unit: 'fraction', protocol_max: 0.4 }, C: { max: 1, unit: 'fraction', protocol_max: 0.3 }, D: { max: 1, unit: 'fraction', protocol_max: 0.2 } },
  pixel_controls: { A: { max: 60, unit: 'px', protocol_max: null }, B: { max: 1, unit: 'fraction', protocol_max: null }, C: { max: 1, unit: 'fraction', protocol_max: null }, D: { max: 1, unit: 'fraction', protocol_max: null } },
  ffmpeg_available: true, exploratory_label: 'Exploratory, not a research result.',
}

export const injection = (): InjectionResult => ({
  scene: 'ETH', exploratory: true, active: ['A', 'B2'], unit: 'm', label: 'Exploratory, not a research result, not P-10.', seeds: [0, 1, 2],
  request: { a_sigma: 0.2, b_p: 0.1, c_p: 0, d_p: 0, reconstruct: false },
  achieved: { A: { measure: 'rmse_pos_m', by_seed: { 0: 0.28 }, mean: 0.28 }, B: { measure: 'rate', by_seed: { 0: 0.1 }, mean: 0.1 } },
  predictors: {
    CV: { ade: 2.709, fde: 4.1, rmse: 3, prediction_coverage: 0.302, de_by_step: [1, 2], clean_ade: 0.678, clean_fde: 1.3, clean_rmse: 0.8, clean_de_by_step: [0.1, 0.2], clean_same_windows_ade: 0.7, ade_change: 2.995, seeds_with_predictions: 3, undefined_reason: null },
    CA: { ade: 20, fde: 40, rmse: 25, prediction_coverage: 0.302, de_by_step: [5, 9], clean_ade: 3.557, clean_fde: 9, clean_rmse: 4.5, clean_de_by_step: [0.5, 1], clean_same_windows_ade: 3.6, ade_change: 4.6, seeds_with_predictions: 3, undefined_reason: null },
  },
  windows: { eligible: 100, predicted_by_seed: { 0: 30 }, lost_by_reason: {} },
  overlay: { seed: 0, tracks: [], events: [], events_total: 0, predicted: {} },
})

// One scene's finished sweep, with ETH's real shape. The values are fixtures and no value here is a result.
export function sceneSweep(scene = 'ETH', unit = 'm'): SceneSweep {
  const predictor = (ade: number, fde: number, adeRise: number, fdeRise: number) => ({ ade, fde, ade_rise: adeRise, fde_rise: fdeRise, de_by_step: [ade / 2, fde] })
  const level = (requested: number, achieved: number, coverage: number, cv: [number, number, number, number], ca: [number, number, number, number]): SweepLevel => ({ requested, achieved, coverage, predictors: { CV: predictor(...cv), CA: predictor(...ca) } })
  const limit = (status: string, value: number | null, up: number | null) => ({ status, value, verified_up_to: up, tested_up_to: 0.7 })
  const limits = (a: number, b: number, c: number, status = 'crossed') => ({ '0.05': { CV: limit(status, a, a), CA: limit(status, a, a) }, '0.1': { CV: limit(status, b, b), CA: limit(status, b, b) }, '0.2': { CV: limit(status, c, c), CA: limit(status, c, c) } })
  const better = { status: 'no_crossing', at: null, better_when_clean: 'CV' }
  const error = (code: ErrorCode, limitOn: SweepError['limit_on'], measure: string, levels: SweepLevel[], limitsOf: SweepError['limits']): SweepError => ({ code, limit_on: limitOn, tested: levels.length, measure, levels, limits: limitsOf, better })
  return {
    scene, exploratory: true, state: 'ready', done: 20, total: 20, unit, seeds: [0, 1, 2], reason: null, label: 'Measured live on this scene, exploratory, and never saved.', windows: 2614, seconds_per_step: 0.4,
    clean: { CV: { ade: 0.678, fde: 1.344, de_by_step: [0.1, 1.344] }, CA: { ade: 3.557, fde: 9.098, de_by_step: [0.2, 9.098] } }, limits_allowed: ['0.05', '0.1', '0.2'],
    errors: [
      error('A', 'ade', `rmse_pos_${unit}`, [level(0.05, 0.07, 1, [0.95, 1.79, 0.4, 0.33], [6.1, 15.7, 0.72, 0.72]), level(0.5, 0.71, 1, [6.25, 11.14, 8.21, 7.29], [46.98, 120.14, 12.21, 12.21])], limits(0.01, 0.02, 0.05)),
      error('B', 'windows_lost', 'rate', [level(0.05, 0.049, 0.67, [0.66, 1.32, 0, 0], [3.5, 8.9, 0, 0]), level(0.4, 0.399, 0.019, [0.67, 1.39, 0, 0], [3.6, 9.3, 0, 0])], limits(0.008, 0.015, 0.03)),
      error('C', 'windows_lost', 'fragmentation_rate', [level(0.02, 0.021, 0.87, [0.68, 1.35, 0, 0], [3.57, 9.14, 0, 0]), level(0.3, 0.298, 0.09, [0.68, 1.36, 0, 0], [3.53, 9.02, 0, 0])], limits(0.008, 0.016, 0.03)),
      error('D', 'ade', 'switch_event_rate', [level(0.025, 1.37, 0.99, [0.79, 1.55, 0.17, 0.15], [4.6, 11.8, 0.3, 0.3]), level(0.2, 2.21, 0.98, [0.79, 1.55, 0.16, 0.15], [4.48, 11.48, 0.27, 0.26])], limits(2.21, 2.21, 2.21, 'not_reached')),
    ],
  }
}

type Route = { status?: number; body: unknown }

// A stand-in for the backend. Any path that is not listed fails the test visibly as a 404.
export function stubBackend(routes: Record<string, Route | unknown>) {
  const calls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const path = String(input).replace('http://127.0.0.1:8000/api', '')
    calls.push(path)
    const route = (routes[path] ?? routes[path.split('?')[0]]) as Route | undefined
    if (route === undefined) return new Response(JSON.stringify({ detail: `no fixture for ${path}` }), { status: 404 })
    const wrapped = typeof route === 'object' && route !== null && 'body' in route
    return new Response(JSON.stringify(wrapped ? route.body : route), { status: wrapped ? route.status ?? 200 : 200 })
  }))
  return calls
}

export function stubDisconnected() {
  vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('Failed to fetch') }))
}
