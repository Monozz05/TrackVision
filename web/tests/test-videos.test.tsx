import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TestVideosPage } from '@/components/test-videos'
import { navItems } from '@/config/app'
import { colours } from '@/lib/overlay'
import { drawForecasts, forecastsAt, getStreetForecast, keptForecasts, streetVideoUrl, type StreetCatalogue, type StreetForecast, type StreetVideo } from '@/lib/test-videos'
import { ExperimentProvider } from '@/stores/experiment-store'
import { noErrors } from '@/types'
import { health, stubBackend } from './fixtures'

const video = (id: string, title: string, overrides: Partial<StreetVideo> = {}): StreetVideo => ({ id, title, author: `Author of ${title}`, licence: 'CC BY 4.0', licence_url: 'https://creativecommons.org/licenses/by/4.0', source_url: `https://commons.wikimedia.org/wiki/File:${id}`, camera: 'Fixed', caution: null, ready: true, reason: null, width: 1280, height: 720, fps: 25, frames: 500, ...overrides })
const catalogue: StreetCatalogue = {
  items: [video('caracas', 'Caracas'), video('glasgow', 'Glasgow', { camera: 'Moves slightly', caution: 'The camera moves, so paths and forecasts include its motion.' }), video('toronto', 'Toronto', { ready: false, reason: 'This video has not been prepared.', width: null, height: null, fps: null, frames: null })],
  predictors: ['CV', 'CA'], label: 'Display only. These videos are not a dataset and not a research result.', no_accuracy: 'No true positions exist for this video, so ADE and FDE cannot be measured.', moved_note: 'Forecast moved is the mean distance between the forecast with and without errors: change, not accuracy.',
  controls: { A: { max: 60, unit: 'px', protocol_max: null }, B: { max: 1, unit: 'fraction', protocol_max: null }, C: { max: 1, unit: 'fraction', protocol_max: null }, D: { max: 1, unit: 'fraction', protocol_max: null } },
}
const person = (id: number) => ({ id, frames: [0, 10, 20, 30], uv: [[10, 10], [20, 10], [30, 10], [40, 10]] as [number, number][], boxes: [[0, 0, 20, 10], [10, 0, 30, 10], [20, 0, 40, 10], [30, 0, 50, 10]] })
const tracks = { scene: 'caracas', source: 'IoU tracker on yolo11m detections', note: 'Tracker output for display.', frame_step: 1, history_len: 8, horizon_len: 12, tracks: [person(4)] }
const result = (overrides: Partial<StreetForecast> = {}): StreetForecast => ({
  scene: 'caracas', exploratory: true, active: [], unit: 'px', label: 'x', seeds: [0, 1, 2], seconds_per_step: 0.4, horizon_len: 2,
  request: { a_sigma: 0, b_p: 0, c_p: 0, d_p: 0, reconstruct: false }, achieved: {},
  forecasts: { total: 1400, kept_by_seed: { 0: 1400, 1: 1400, 2: 1400 }, lost_by_reason: {} }, moved: { CV: 0, CA: 0 },
  grid: { frame_step: 10, tracks: [person(4)] },
  overlay: { seed: 0, tracks: [person(4)], events: [], events_total: 0, predicted: { CV: { 10: [[4, [[30, 10], [40, 10]]]] }, CA: { 10: [[4, [[35, 10], [60, 10]]]] } } },
  ...overrides,
})
const routes = { '/health': health, '/test-videos': catalogue, '/test-videos/caracas/tracks': tracks, '/test-videos/caracas/forecast': result(), '/test-videos/glasgow/tracks': { ...tracks, scene: 'glasgow' }, '/test-videos/glasgow/forecast': { ...result(), scene: 'glasgow' } }
const show = () => render(<ExperimentProvider><TestVideosPage/></ExperimentProvider>)
const tile = (label: string) => within(screen.getByTestId('result-panel')).getByText(label).parentElement as HTMLElement

describe('test videos', () => {
  it('has a place in the menu, after the three frozen sections', () => {
    expect(navItems.map((item) => item.label)).toEqual(['Workspace', 'Analysis', 'Applications', 'Test Videos'])
    expect(navItems[3].href).toBe('/test-videos')
  })

  it('lists every video in a dropdown and starts on the first one ready', async () => {
    stubBackend(routes)
    show()
    const list = await screen.findByRole('combobox', { name: 'Test video' }) as HTMLSelectElement
    expect([...list.options].map((option) => option.text)).toEqual(['Caracas', 'Glasgow', 'Toronto'])
    expect(list.value).toBe('caracas')
    expect(await screen.findByTestId('video-stage')).toBeTruthy()
    expect(document.querySelector('video')?.getAttribute('src')).toContain('/test-videos/caracas/video')
  })

  it('credits the author, licence and source of the video on screen', async () => {
    stubBackend(routes)
    show()
    const credit = await screen.findByTestId('video-credit')
    expect(credit.textContent).toContain('Author of Caracas')
    expect(within(credit).getByRole('link', { name: /CC BY 4.0/ }).getAttribute('href')).toBe('https://creativecommons.org/licenses/by/4.0')
    expect(within(credit).getByRole('link', { name: /Source/ }).getAttribute('href')).toBe('https://commons.wikimedia.org/wiki/File:caracas')
    fireEvent.change(screen.getByRole('combobox', { name: 'Test video' }), { target: { value: 'glasgow' } })
    await waitFor(() => expect(screen.getByTestId('video-credit').textContent).toContain('Author of Glasgow'))
  })

  it('shows a dash for ADE and FDE, with the reason, and never a number', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect(tile('ADE').textContent).toBe('ADE—')
    expect(tile('FDE').textContent).toBe('FDE—')
    expect(screen.getByTestId('no-accuracy').textContent).toContain('No true positions exist')
    expect(screen.getByTestId('no-accuracy').textContent).toContain('change, not accuracy')
  })

  it('says what this section is on every video', async () => {
    stubBackend(routes)
    show()
    expect((await screen.findByTestId('section-label')).textContent).toContain('not a research result')
  })

  it('shows forecasts kept with no errors applied', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect(tile('Forecasts kept').textContent).toContain('1,400 of 1,400')
    expect(tile('Forecast moved').textContent).toContain('0.0 px')
  })

  it('sends the sliders to the backend and shows what came back', async () => {
    const withNoise = result({ active: ['A'], request: { a_sigma: 24, b_p: 0, c_p: 0, d_p: 0, reconstruct: false }, achieved: { A: { measure: 'rmse_pos_px', by_seed: { 0: 24.2 }, mean: 24.2 } }, moved: { CV: 88.4, CA: 412.6 } })
    const calls = stubBackend({ ...routes, '/test-videos/caracas/forecast?a=24&b=0&c=0&d=0': withNoise })
    show()
    await screen.findByTestId('no-accuracy')
    fireEvent.change(screen.getByRole('slider', { name: 'Position noise' }), { target: { value: '24' } })
    await waitFor(() => expect(tile('Forecast moved').textContent).toContain('88.4 px'))
    expect(calls).toContain('/test-videos/caracas/forecast?a=24&b=0&c=0&d=0')
    expect(screen.getByText('Position noise 24.2 px')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'CA' }))
    expect(tile('Forecast moved').textContent).toContain('412.6 px')
  })

  it('asks for filled gaps only once missing detections are on', async () => {
    const calls = stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect(screen.queryByText('Fill gaps by interpolation')).toBeNull()
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '0.2' } })
    fireEvent.click(await screen.findByRole('checkbox', { name: 'Fill gaps by interpolation' }))
    await waitFor(() => expect(calls.some((call) => call.includes('b=0.2') && call.includes('reconstruct=true'))).toBe(true))
  })

  it('says so when no forecast is left', async () => {
    const empty = result({ active: ['B2'], forecasts: { total: 1400, kept_by_seed: { 0: 0, 1: 0, 2: 0 }, lost_by_reason: {} }, moved: { CV: null, CA: null } })
    stubBackend({ ...routes, '/test-videos/caracas/forecast?a=0&b=1&c=0&d=0': empty })
    show()
    await screen.findByTestId('no-accuracy')
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '1' } })
    await waitFor(() => expect(screen.getByText(/No forecast is left/)).toBeTruthy())
    expect(tile('Forecast moved').textContent).toBe('Forecast moved—')
  })

  it('resets the sliders', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    const reset = screen.getByRole('button', { name: /Reset/ }) as HTMLButtonElement
    expect(reset.disabled).toBe(true)
    fireEvent.change(screen.getByRole('slider', { name: 'Identity switches' }), { target: { value: '0.1' } })
    expect(reset.disabled).toBe(false)
    fireEvent.click(reset)
    expect((screen.getByRole('slider', { name: 'Identity switches' }) as HTMLInputElement).value).toBe('0')
  })

  it('warns about a moving camera only on those videos', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect(screen.queryByTestId('video-caution')).toBeNull()
    fireEvent.change(screen.getByRole('combobox', { name: 'Test video' }), { target: { value: 'glasgow' } })
    expect((await screen.findByTestId('video-caution')).textContent).toContain('camera moves')
  })

  it('explains a video that is not prepared instead of failing', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    fireEvent.change(screen.getByRole('combobox', { name: 'Test video' }), { target: { value: 'toronto' } })
    expect((await screen.findByTestId('error-state')).textContent).toContain('has not been prepared')
    expect(screen.queryByTestId('video-stage')).toBeNull()
  })

  it('names the dashed line as tracker output, never as truth', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect(screen.getByRole('checkbox', { name: 'Tracker saw next' })).toBeTruthy()
    expect(screen.queryByText(/true path/i)).toBeNull()
    expect(screen.queryByText(/true future/i)).toBeNull()
  })

  it('offers the path without errors only while errors are on', async () => {
    stubBackend(routes)
    show()
    await screen.findByTestId('no-accuracy')
    expect((screen.getByRole('checkbox', { name: 'Path without errors' }) as HTMLInputElement).disabled).toBe(true)
  })

  it('states the real step length where the frame rate cannot reach 0.4 s', async () => {
    stubBackend({ ...routes, '/test-videos/caracas/forecast': result({ seconds_per_step: 0.417 }) })
    show()
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('One step here is 0.42 s.'))
  })

  it('shows the backend failure with a retry', async () => {
    stubBackend({ ...routes, '/test-videos/caracas/forecast': { status: 409, body: { detail: 'This video has not been prepared.' } } })
    show()
    expect((await screen.findByTestId('error-state')).textContent).toContain('has not been prepared')
    expect(screen.getByRole('button', { name: /retry/i })).toBeTruthy()
  })
})

describe('test video helpers', () => {
  interface Stroke { strokeStyle: string; dash: number[]; points: number[][] }
  const context = () => {
    const calls: Stroke[] = []
    const state = { strokeStyle: '', lineWidth: 0 }
    let dash: number[] = []
    let points: number[][] = []
    const ctx = Object.assign(state, {
      setLineDash(value: number[]) { dash = value },
      beginPath() { points = [] },
      moveTo(x: number, y: number) { points.push([x, y]) },
      lineTo(x: number, y: number) { points.push([x, y]) },
      stroke() { calls.push({ strokeStyle: state.strokeStyle, dash, points }) },
    })
    return { calls, ctx: ctx as unknown as CanvasRenderingContext2D }
  }
  const view = { forecast: result(), predictor: 'CV', predicted: true, later: false, pixelsPerCssPixel: 1 }

  it('finds the forecasts made at the last sampled frame', () => {
    expect(forecastsAt(result(), 'CV', 10).rows).toHaveLength(1)
    expect(forecastsAt(result(), 'CV', 19).at).toBe(10)
    expect(forecastsAt(result(), 'CV', 19).rows).toHaveLength(1)
    expect(forecastsAt(result(), 'CV', 20).rows).toHaveLength(0)
    expect(forecastsAt(result(), 'CV', 9).rows).toHaveLength(0)
  })

  it('draws the forecast from the person to its end in the forecast colour', () => {
    const { calls, ctx } = context()
    drawForecasts(ctx, 12, view)
    expect(calls).toHaveLength(1)
    expect(calls[0].strokeStyle).toBe(colours.predicted)
    expect(calls[0].points).toEqual([[20, 10], [30, 10], [40, 10]])
  })

  it('draws one predictor at a time', () => {
    const { calls, ctx } = context()
    drawForecasts(ctx, 12, { ...view, predictor: 'CA' })
    expect(calls).toHaveLength(1)
    expect(calls[0].points).toEqual([[20, 10], [35, 10], [60, 10]])
  })

  it('draws where the tracker saw them next as a dashed line, up to the horizon', () => {
    const { calls, ctx } = context()
    drawForecasts(ctx, 12, { ...view, predicted: false, later: true })
    expect(calls).toHaveLength(1)
    expect(calls[0].strokeStyle).toBe(colours.future)
    expect(calls[0].dash).toEqual([5, 4])
    expect(calls[0].points).toEqual([[20, 10], [30, 10], [40, 10]])
  })

  it('draws nothing with both lines off, or with no result yet', () => {
    const { calls, ctx } = context()
    drawForecasts(ctx, 12, { ...view, predicted: false })
    drawForecasts(ctx, 12, { ...view, forecast: null })
    expect(calls).toHaveLength(0)
  })

  it('reports forecasts kept as the mean over seeds', () => {
    expect(keptForecasts(result({ forecasts: { total: 10, kept_by_seed: { 0: 7, 1: 8, 2: 8 }, lost_by_reason: {} } }))).toBe(8)
    expect(keptForecasts(result({ forecasts: { total: 10, kept_by_seed: {}, lost_by_reason: {} } }))).toBe(0)
  })

  it('addresses only this section on the backend', async () => {
    expect(streetVideoUrl('a b')).toContain('/test-videos/a%20b/video')
    const calls = stubBackend({ '/test-videos/x/forecast': result() })
    await getStreetForecast('x', { ...noErrors, a: 5, reconstruct: true })
    expect(calls).toEqual(['/test-videos/x/forecast?a=5&b=0&c=0&d=0&reconstruct=true'])
    vi.unstubAllGlobals()
  })
})
