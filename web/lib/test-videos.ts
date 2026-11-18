import { appConfig } from '@/config/app'
import { request } from '@/lib/api/client'
import { colours, lastSample } from '@/lib/overlay'
import type { CleanTracks, ErrorControls, ErrorSettings, InjectionResult, OverlayTrack } from '@/types'

// The Test Videos section. These videos have no true positions, so nothing here is an accuracy value: the backend sends no
// ADE or FDE and the page shows a dash. The dashed line is where the tracker saw a person next, which is tracker output, not truth.
export interface StreetVideo { id: string; title: string; author: string; licence: string; licence_url: string; source_url: string; camera: string; caution: string | null; ready: boolean; reason: string | null; width: number | null; height: number | null; fps: number | null; frames: number | null }
export interface StreetCatalogue { items: StreetVideo[]; predictors: string[]; label: string; no_accuracy: string; moved_note: string; controls: ErrorControls }
export interface StreetForecast {
  scene: string
  exploratory: true
  active: string[]
  unit: string
  label: string
  seeds: number[]
  seconds_per_step: number
  horizon_len: number
  request: InjectionResult['request']
  achieved: InjectionResult['achieved']
  forecasts: { total: number; kept_by_seed: Record<string, number>; lost_by_reason: Record<string, number> }
  moved: Record<string, number | null>
  grid: { frame_step: number; tracks: OverlayTrack[] }
  overlay: InjectionResult['overlay']
}

type Options = { signal?: AbortSignal }
const id = encodeURIComponent

export function listStreetVideos(options?: Options) { return request<StreetCatalogue>('/test-videos', options) }
export function streetVideoUrl(videoId: string) { return `${appConfig.apiBaseUrl}/test-videos/${id(videoId)}/video` }
export function getStreetTracks(videoId: string, options?: Options) { return request<CleanTracks>(`/test-videos/${id(videoId)}/tracks`, { ...options, timeoutMs: 60_000 }) }
export function getStreetForecast(videoId: string, errors: ErrorSettings, options?: Options) {
  const params = new URLSearchParams({ a: String(errors.a), b: String(errors.b), c: String(errors.c), d: String(errors.d) })
  if (errors.reconstruct) params.set('reconstruct', 'true')
  return request<StreetForecast>(`/test-videos/${id(videoId)}/forecast?${params}`, { ...options, timeoutMs: 60_000 })
}

export function keptForecasts(forecast: StreetForecast) {
  const counts = Object.values(forecast.forecasts.kept_by_seed)
  return counts.length ? Math.round(counts.reduce((sum, count) => sum + count, 0) / counts.length) : 0
}

// The forecasts on screen at a frame: those made at the last sampled frame before it.
export function forecastsAt(forecast: StreetForecast, predictor: string, frame: number) {
  const step = forecast.grid.frame_step
  const at = Math.floor(frame / step) * step
  return { at, rows: forecast.overlay.predicted[predictor]?.[String(at)] ?? [] }
}

export interface ForecastView { forecast: StreetForecast | null; predictor: string; predicted: boolean; later: boolean; pixelsPerCssPixel: number }

export function drawForecasts(ctx: CanvasRenderingContext2D, frame: number, view: ForecastView) {
  const { forecast, predictor, predicted, later, pixelsPerCssPixel: scale } = view
  if (!forecast || (!predicted && !later)) return
  const byId = new Map(forecast.grid.tracks.map((track) => [track.id, track]))
  const { at, rows } = forecastsAt(forecast, predictor, frame)
  const line = (points: number[][], colour: string, dash: number[]) => {
    if (points.length < 2) return
    ctx.setLineDash(dash)
    ctx.strokeStyle = colour
    ctx.lineWidth = 2 * scale
    ctx.beginPath()
    ctx.moveTo(points[0][0], points[0][1])
    for (const [u, v] of points.slice(1)) ctx.lineTo(u, v)
    ctx.stroke()
    ctx.setLineDash([])
  }
  for (const [person, path] of rows) {
    const track = byId.get(person)
    if (!track) continue
    const i = lastSample(track.frames, at)
    if (later) line(track.uv.slice(i, i + forecast.horizon_len + 1), colours.future, [5 * scale, 4 * scale])
    if (predicted) line([track.uv[i], ...path], colours.predicted, [])
  }
}
