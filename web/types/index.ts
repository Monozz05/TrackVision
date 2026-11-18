// Contracts for the TrackVision API (src/trackvision/platform). Every result value is nullable: a value the pipeline did
// not record is null here and an em dash on screen. There is no mock data in this file, so no screen can fall back to it.
export type ExperimentStatus = 'complete' | 'incomplete' | 'invalid'
export type EvidenceLevel = 'engineering_only' | 'engineering_and_descriptive' | 'descriptive_local_only' | 'feasibility_only' | 'diagnostic_only' | 'scaffold_check'
export type ExperimentKind = 'sweep' | 'mot17_tracking' | 'synthetic_tracker' | 'engineering_analysis' | 'ewap_baseline' | 'ewap_injection' | 'synthetic_feasibility' | 'geometry_validation' | 'smoke' | 'unrecognised'
export type ErrorKind = 'not_configured' | 'disconnected' | 'not_found' | 'rejected' | 'server'

export interface ApiResponse<T> { data: T | null; status: 'success' | 'empty' | 'error' | 'unavailable'; message?: string; errorKind?: ErrorKind }

export interface Health { status: string; version: string; read_only: boolean; results_directory_available: boolean; experiment_records: number; invalid_records: number; skipped_folders: number; timestamp: string }

export interface Stat { mean: number | null; sd?: number | null; n_valid?: number | null; n_undefined?: number | null; undefined?: Record<string, string> | null; median?: number | null; n?: number | null }

export interface Provenance {
  git: { sha?: string; dirty?: boolean } | null
  environment: Record<string, string> | null
  specification: { sha256?: string; version?: string } | null
  config_hash: string | null
  settings_hash: string | null
  source_sha256: string | null
  spec_sha256: string | null
  dataset_manifest_hash: string | null
  trackeval_commit: string | null
  licence: string | null
  results_sha256: string | null
  version: string | null
  trackvision_version: string | null
  association_mechanism: string | null
  association_status: string | null
  association_note: string | null
  association_blocker: string | null
  association_sidecar: { bound_to_results: boolean; proposal_0003?: string | null; spec_section?: string | null } | null
  [key: string]: unknown
}

export interface Integrity { status: 'verified' | 'mismatch' | 'not_recorded'; verified: number; failed: string[] }

export interface Experiment {
  experiment_id: string
  record_id: string | null
  kind: ExperimentKind
  kind_label: string
  scope: string | null
  timestamp: string | null
  status: ExperimentStatus
  status_reason: string | null
  evidence_level: EvidenceLevel | null
  research_results: boolean | null
  label: string | null
  notice: string | null
  latest_in_series: boolean
  experiments: (string | null)[]
  datasets: (string | null)[]
  scenes: (string | null)[]
  predictors: string[]
  conditions: string[]
  histories: number[]
  seeds: number[]
  horizon_len: number | null
  detector: string | null
  tracker: string | null
  counts: { expected_cells: number | null; finished_cells: number | null; by_status: Record<string, number> | null; by_experiment: Record<string, Record<string, number>> | null; scope_complete: boolean | null; research_complete: boolean | null } | null
  integrity: Integrity
  provenance: Provenance
}

export interface ExperimentPage { items: Experiment[]; total: number; limit: number; offset: number; counts: { by_kind: Record<string, number>; by_status: Record<string, number>; by_evidence: Record<string, number>; all: number } }
export interface ViewIndex { items: Experiment[]; default: string | null }
export type ExperimentView = 'tracking' | 'error-injection' | 'prediction' | 'analysis'
export interface Artifact { name: string; type: 'record' | 'chart'; size_bytes: number; served: boolean }
export interface ExperimentDetail { experiment: Experiment; views: ExperimentView[]; artifacts: Artifact[]; stored_summary: Record<string, unknown> | null }


export interface TrackingMetrics { mota: number | null; idf1: number | null; hota: number | null; idSwitches: number | null; fragmentations: number | null }
export interface TrackingSequence { sequence: string; regime: string | null; frame_rate: number | null; frames_evaluated: number[] | null; metrics: TrackingMetrics; official_trackeval: Record<string, number> | null; project_diagnostics: Record<string, unknown> | null }
export interface TrackingGroup { name: string; primary: boolean; tracker_settings: Record<string, unknown> | null; sequences: TrackingSequence[] }
export interface TrackingResult { experiment: Experiment; source: string; groups: TrackingGroup[]; pooled_result: unknown; not_satisfied?: string[] | null; prediction_note?: string | null; official_note?: string | null; cadence?: string | null; not_p1?: boolean | null; not_p9?: boolean | null }

export interface Selection { population: string; history: number; populations: string[]; histories: number[]; population_label: string | null; aggregation: string | null; scope_label: string | null }
export interface SelectionQuery { population?: string; history?: number }

export interface Figure { name: string; figure: string; scene: string | null; condition: string | null; history: string | null; predictor: string | null; population: string | null }
export interface MonotonicityRow { scene: string; condition: string; rule?: string; status: string }
export interface ErrorRow { scene: string; predictor: string; condition: string; severity_measure: string | null; requested: number | null; achieved: number | null; achieved_by_seed: Record<string, number> | null; seed_spread: { sd?: number | null; min?: number | null; max?: number | null } | null; ade_m: Stat | null; d_ade: Stat | null; prediction_coverage: Stat | null; windows_by_seed: Record<string, { eligible: number; predicted: number }> | null; undefined_reason_by_seed: Record<string, string | null> | null }
export interface ErrorResult { experiment: Experiment; source: string; selection?: Selection; rows?: ErrorRow[]; monotonicity?: MonotonicityRow[] | null; blocked?: Record<string, string> | null; figures?: Figure[]; stored_conditions?: unknown }

export interface CellRow { experiment: string | null; scene: string | null; fold: string | null; predictor: string | null; condition: string | null; requested_level: number | null; seed: number | null; history_len: number | null; population: string | null; status: string; achieved: { measure?: string; value?: number | null; detail?: Record<string, unknown> } | null; ade_m: number | null; fde_m: number | null; rmse_m: number | null; prediction_coverage: number | null; eligible_windows: number | null; predicted_windows: number | null; undefined_reason: string | null; reason: string | null; digest: string }
export interface CellPage { experiment_id: string; total: number; limit: number; offset: number; digests_verified: boolean; items: CellRow[] }

export interface PredictionRow { scene: string | null; predictor: string; ade_m: Stat | null; fde_m: Stat | null; rmse_m: Stat | null; prediction_coverage: Stat | number | null; de_by_step_m: number[] | null; clean_all_eligible_ade_m?: Stat | null; d_ade_paired?: Stat | null }
export interface AcrossSceneRow { predictor: string; ade_m: Stat | null; fde_m: Stat | null; rmse_m: Stat | null; prediction_coverage: Stat | null }
export interface PredictionResult { experiment: Experiment; source: string; selection?: Selection; rows: PredictionRow[]; across_scenes: AcrossSceneRow[]; coverage_by_scene?: Record<string, Record<string, unknown> | null> }

export interface CurveLevel { requested: number | null; achieved_seed_mean: number | null; s_norm: number | null; ade_m: Stat | null; fde_m: Stat | null; rmse_m: Stat | null; prediction_coverage: Stat | null; d_ade: Stat | null; d_fde: Stat | null; de_by_step_m: number[] | null; windows_by_seed: Record<string, { eligible: number; predicted: number }> | null; undefined_reason_by_seed: Record<string, string | null> | null }
export interface Curve { severity_measure: string | null; audc: { status?: string; value?: number | null; reason?: string | null } | null; robustness: { value?: number | null; reason?: string | null } | null; levels: CurveLevel[] }
export type Curves = Record<string, Record<string, Record<string, Curve>>>
export interface AcrossLevel { requested: number | null; achieved: Stat | null; ade_m: Stat | null; fde_m: Stat | null; rmse_m?: Stat | null; prediction_coverage?: Stat | null }
export interface FamilyEntry { value: number | null; status?: string; reason?: string | null; valid?: Record<string, number>; n_valid?: number }
export interface Hypothesis { research_verdict: string; reason: string | null }
export interface P15Row { audc_best: number | null; clean_ade_best: number | null; clean_fde_best: number | null; minimum_history: number | null; partial_folds: boolean | null; status: string | null; reason: string | null }
export interface SweepAnalysis { experiment: Experiment; source: 'sweep'; selection: Selection; curves: Curves; family_audc: Record<string, Record<string, FamilyEntry>> | null; across_scenes: Record<string, Record<string, { levels: AcrossLevel[] }>> | null; p15: Record<string, Record<string, P15Row>> | null; monotonicity: MonotonicityRow[] | null; hypotheses: Record<string, Hypothesis> | null; blocked: Record<string, string> | null; failures: unknown[] | null }

export interface InverseEntry { q: number | null; s_star: number | null; status: string; reason: string | null; severity_measure: string | null; maximum_observed_severity: number | null; verified_up_to: number | null }
export interface InverseGroup { level: string; metric: string; not_a_requirement: string; entries: Record<string, InverseEntry> }
export interface CrossingEntry { status: string; s_crossing: number | null; reason: string | null; lower_before?: string | null; points?: number | null }
export interface CrossingGroup { level: string; metric: string; conditional_on: string; not_implemented: string; per_scene: Record<string, CrossingEntry>; across_synthetic_scenes?: Record<string, CrossingEntry> }
export interface AssociationPair { HOTA: number | null; IDF1: number | null; IDSW: number | null; MOTA: number | null; ade_m: number | null }
export interface AssociationPredictor { correlations: Record<string, { n: number | null; value: number | null; reason: string | null }>; pairs: Record<string, AssociationPair>; paired_observations: number | null; scene_count: number | null; tracker_count: number | null; h7_verdict: string; inference: string | null }
export interface EngineeringAnalysis {
  experiment: Experiment
  source: 'engineering_analysis'
  label: string | null
  family_across_scenes: { level?: string; note?: string; scenes?: string[]; by_condition: Record<string, { value: number | null; sd?: number | null; status?: string; n_valid_scenes?: number; n_undefined_scenes?: number }> } | null
  e5_p13_engineering: { level: string; h7_verdict: string; population: string; statistical_unit: string; not_p13: boolean; by_predictor: Record<string, AssociationPredictor> } | null
  e6_dominant_dry_run: { status: string; target: string | null; targets: string[]; co_dominant: boolean; codominance_gap: number | null; note: string; population: string; research_selection: string; research_selection_applied: boolean; candidates: Record<string, { qualifies: boolean; reason: string | null; value: number | null }> } | null
  e7_inverse_mapping: { synthetic?: InverseGroup; ethhotel?: InverseGroup } | null
  e8_threshold_crossings: { synthetic?: CrossingGroup; ethhotel?: CrossingGroup } | null
  input_runs: InputRun[]
  ethhotel_family: string | null
  blocked: Record<string, string> | null
  deferred: Record<string, string> | null
}
export type AnalysisResult = SweepAnalysis | EngineeringAnalysis
// Where a recorded input of E5 to E8 lives now. `unchanged` means a stored file still has the recorded hash.
export interface InputRun { input: string; used_by: string[]; run: string | null; unchanged: boolean }

export type DatasetName = 'EWAP' | 'UCY' | 'MOT17' | 'Synthetic'
export type VideoState = 'ready' | 'preparing' | 'missing' | 'failed'
export interface WorkspaceScene { id: string; dataset: DatasetName; name: string; parent: string | null; regime: string; has_video: boolean; errors_available: boolean; errors_reason: string | null; evidence_level: EvidenceLevel; predictors: string[]; stored_run: string | null; detector: string | null; video_state: VideoState; unit: string; caution: string | null }
export interface ErrorControl { max: number; unit: string; protocol_max: number | null }
export type ErrorControls = Record<'A' | 'B' | 'C' | 'D', ErrorControl>
export interface WorkspaceCatalogue { items: WorkspaceScene[]; controls: ErrorControls; pixel_controls: ErrorControls; ffmpeg_available: boolean; exploratory_label: string }
export interface VideoStatus { state: VideoState; width?: number; height?: number; fps?: number; frames?: number; reason?: string }
export interface OverlayTrack { id: number; frames: number[]; uv: [number, number][]; boxes: number[][]; dropped?: number[]; segment?: number[] }
export interface CleanTracks { scene: string; source: string; note: string | null; frame_step: number; history_len: number | null; horizon_len: number | null; tracks: OverlayTrack[] }
export interface ErrorSettings { a: number; b: number; c: number; d: number; reconstruct: boolean }
export const noErrors: ErrorSettings = { a: 0, b: 0, c: 0, d: 0, reconstruct: false }
export const anyError = (errors: ErrorSettings) => errors.a > 0 || errors.b > 0 || errors.c > 0 || errors.d > 0
export interface OverlayEvent { frame: number; kind: 'switch' | 'split'; a: number; b: number | null }
// Values are in the result's `unit`: metres, or video pixels on a scene with no calibration.
export interface InjectionPredictor { ade: number | null; fde: number | null; rmse: number | null; prediction_coverage: number | null; de_by_step: number[] | null; clean_ade: number | null; clean_fde: number | null; clean_rmse: number | null; clean_de_by_step: number[] | null; clean_same_windows_ade: number | null; ade_change: number | null; seeds_with_predictions: number; undefined_reason: string | null }
export interface InjectionResult {
  scene: string
  exploratory: true
  active: string[]
  unit: string
  label: string
  seeds: number[]
  request: { a_sigma: number; b_p: number; c_p: number; d_p: number; reconstruct: boolean }
  achieved: Record<string, { measure: string; by_seed: Record<string, number | null>; mean: number | null }>
  predictors: Record<string, InjectionPredictor>
  windows: { eligible: number; predicted_by_seed: Record<string, number>; lost_by_reason: Record<string, number> }
  overlay: { seed: number; tracks: OverlayTrack[]; events: OverlayEvent[]; events_total: number; predicted: Record<string, Record<string, [number, [number, number][]][]>> }
}

export type NoisePoint = [number, number]
export interface NoiseExampleData { scene: string; exploratory: true; unit: string; label: string; seed: number; person: number; frames: number[]; image_coordinates: boolean; observed: NoisePoint[]; future: NoisePoint[]; levels: { name: string; sigma: number; noisy: NoisePoint[] }[] }

// Every error at every tested level on one scene, measured live and never saved. Values are in the sweep's `unit`.
export type ErrorCode = 'A' | 'B' | 'C' | 'D'
export interface SweepPredictor { ade: number | null; fde: number | null; ade_rise: number | null; fde_rise: number | null; de_by_step: number[] | null }
export interface SweepLevel { requested: number; achieved: number | null; coverage: number | null; predictors: Record<string, SweepPredictor> }
export interface SweepLimit { status: string; value: number | null; verified_up_to: number | null; tested_up_to: number | null }
export interface SweepBetter { status: string; at: number | null; better_when_clean: string | null }
export interface SweepError { code: ErrorCode; limit_on: 'ade' | 'windows_lost'; tested: number; measure: string; levels: SweepLevel[]; limits: Record<string, Record<string, SweepLimit>> | null; better: SweepBetter | null }
export type SweepClean = Record<string, { ade: number | null; fde: number | null; de_by_step: number[] | null }>
export interface SceneSweep { scene: string; exploratory: true; state: 'calculating' | 'ready' | 'failed'; done: number; total: number; unit: string; seeds: number[]; reason: string | null; label: string; windows: number | null; seconds_per_step: number; clean: SweepClean | null; limits_allowed: string[]; errors: SweepError[] }

// The error families the sliders control. Each description says what the injector does, nothing more.
export const errorControls: Array<{ key: 'a' | 'b' | 'c' | 'd'; code: 'A' | 'B' | 'C' | 'D'; label: string; help: string }> = [
  { key:'a', code:'A', label:'Position noise', help:'Random offset added to every observed position.' },
  { key:'b', code:'B', label:'Missing detections', help:'Observations removed at random.' },
  { key:'c', code:'C', label:'Broken tracks', help:'Chance per step that a track splits into a new identity.' },
  { key:'d', code:'D', label:'Identity switches', help:'Chance per step that two nearby people swap identities.' },
]
