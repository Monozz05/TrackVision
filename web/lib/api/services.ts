import type { ApiResponse, AnalysisResult, Dataset, ErrorConfiguration, ErrorResult, Experiment, PredictionConfiguration, PredictionResult, TrackingResult, VideoMetadata } from '@/types'
import { createInitialExperiment, emptyMetrics, emptyPredictionMetrics, mockDatasets, mockExperiments, mockChartSeries } from '@/types'

const pending = <T>(data: T): ApiResponse<T> => ({ data, status:'success' })
export async function getDatasets(): Promise<ApiResponse<Dataset[]>> { return pending(mockDatasets) }
export async function uploadVideo(file: File): Promise<ApiResponse<VideoMetadata>> { return pending({name:file.name,size:`${(file.size / 1024 / 1024).toFixed(1)} MB`}) }
export async function runTracking(): Promise<ApiResponse<TrackingResult>> { return pending({status:'queued', activeTracks:null, totalDetections:null, averageTrackLength:null, fps:null, latency:null, metrics:null, tracks:[]}) }
export async function getTrackingResult(): Promise<ApiResponse<TrackingResult>> { return pending({status:'empty', activeTracks:null, totalDetections:null, averageTrackLength:null, fps:null, latency:null, metrics:emptyMetrics, tracks:[]}) }
export async function runErrorInjection(_configuration: ErrorConfiguration): Promise<ApiResponse<ErrorResult>> { return pending({status:'queued', requestedSeverity:null, achievedSeverity:null, affectedTracks:null, affectedFrames:null, events:[]}) }
export async function getErrorVisualization(): Promise<ApiResponse<ErrorResult>> { return pending({status:'empty', requestedSeverity:null, achievedSeverity:null, affectedTracks:null, affectedFrames:null, events:[]}) }
export async function runPrediction(_configuration: PredictionConfiguration): Promise<ApiResponse<PredictionResult>> { return pending({status:'queued', modelVersion:null, metrics:null}) }
export async function getPredictionResult(): Promise<ApiResponse<PredictionResult>> { return pending({status:'empty', modelVersion:null, metrics:emptyPredictionMetrics}) }
export async function getAnalysis(): Promise<ApiResponse<AnalysisResult>> { return pending({hasData:false, isMock:true, series:mockChartSeries}) }
export async function getExperiment(id: string): Promise<ApiResponse<Experiment>> { return pending(mockExperiments.find((item) => item.experimentId === id) ?? createInitialExperiment()) }
export async function listExperiments(): Promise<ApiResponse<Experiment[]>> { return pending(mockExperiments) }
export const serviceStatus = { backend:'placeholder', persistence:'not connected', realtime:'not connected' } as const
