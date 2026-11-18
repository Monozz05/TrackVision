import { appConfig } from '@/config/app'
import { request } from '@/lib/api/client'
import type { AnalysisResult, CellPage, CleanTracks, ErrorResult, ErrorSettings, ExperimentDetail, ExperimentPage, Health, InjectionResult, NoiseExampleData, PredictionResult, SelectionQuery, SceneSweep, TrackingResult, VideoStatus, ViewIndex, WorkspaceCatalogue } from '@/types'

type Options = { signal?: AbortSignal }
const id = encodeURIComponent

function query(values: Record<string, string | number | null | undefined>) {
  const entries = Object.entries(values).filter(([, value]) => value !== undefined && value !== null && value !== '')
  return entries.length ? `?${new URLSearchParams(entries.map(([key, value]) => [key, String(value)])).toString()}` : ''
}

export function getHealth(options?: Options) { return request<Health>('/health', { ...options, timeoutMs: 8_000 }) }

export function listExperiments(options?: Options) { return request<ExperimentPage>('/experiments?limit=500', options) }
export function getExperiment(experimentId: string, options?: Options) { return request<ExperimentDetail>(`/experiments/${id(experimentId)}`, options) }
export function getTrackingResult(experimentId: string, options?: Options) { return request<TrackingResult>(`/tracking/${id(experimentId)}`, options) }
export function getPredictionResult(experimentId: string, selection: SelectionQuery = {}, options?: Options) { return request<PredictionResult>(`/prediction/${id(experimentId)}${query({ ...selection })}`, options) }
export function listAnalysisRuns(options?: Options) { return request<ViewIndex>('/analysis', options) }
export function getAnalysis<T extends AnalysisResult = AnalysisResult>(experimentId: string, selection: SelectionQuery = {}, options?: Options) { return request<T>(`/analysis/${id(experimentId)}${query({ ...selection })}`, options) }

// Reading a large sweep's cells for the first time verifies every digest, so it needs longer than other calls.
export function getCells(experimentId: string, filters: Record<string, string | number | undefined>, options?: Options) {
  return request<CellPage>(`/experiments/${id(experimentId)}/cells${query(filters)}`, { ...options, timeoutMs: 120_000 })
}

export function figureUrl(experimentId: string, name: string) { return `${appConfig.apiBaseUrl}/experiments/${id(experimentId)}/artifacts/${id(name)}` }

export function listScenes(options?: Options) { return request<WorkspaceCatalogue>('/workspace/scenes', options) }
export function getVideoStatus(sceneId: string, options?: Options) { return request<VideoStatus>(`/workspace/scenes/${id(sceneId)}/video/status`, options) }
export function videoUrl(sceneId: string) { return `${appConfig.apiBaseUrl}/workspace/scenes/${id(sceneId)}/video` }
export function getCleanTracks(sceneId: string, options?: Options, source: 'default' | 'truth' = 'default') { return request<CleanTracks>(`/workspace/scenes/${id(sceneId)}/tracks${query({ source: source === 'truth' ? source : undefined })}`, options) }
export function injectErrors(sceneId: string, errors: ErrorSettings, options?: Options) {
  return request<InjectionResult>(`/workspace/scenes/${id(sceneId)}/inject${query({ a: errors.a, b: errors.b, c: errors.c, d: errors.d, reconstruct: errors.reconstruct ? 'true' : undefined })}`, { ...options, timeoutMs: 60_000 })
}
export function getNoiseExample(sceneId: string, options?: Options) { return request<NoiseExampleData>(`/workspace/scenes/${id(sceneId)}/noise-example`, { ...options, timeoutMs: 60_000 }) }
export function getSceneSweep(sceneId: string, options?: Options) { return request<SceneSweep>(`/workspace/scenes/${id(sceneId)}/sweep`, options) }
