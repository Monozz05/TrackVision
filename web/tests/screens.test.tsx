import { useEffect } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { AnalysisPage } from '@/components/analysis'
import { ApplicationsPage } from '@/components/applications'
import { ExperimentDetailPage, ExperimentsPage, SettingsPage } from '@/components/trackvision-app'
import { WorkspacePage } from '@/components/workspace'
import { applications } from '@/config/applications'
import { ExperimentProvider, useExperiment } from '@/stores/experiment-store'
import { noErrors, type ErrorSettings, type Experiment, type InjectionResult, type NoiseExampleData, type PredictionResult } from '@/types'
import { catalogue, health, injection, run, sceneSweep, stat, stubBackend, stubDisconnected } from './fixtures'

const show = (page: React.ReactNode) => render(<ExperimentProvider>{page}</ExperimentProvider>)
const page = (items: Experiment[]) => ({ items, total: items.length, limit: 500, offset: 0, counts: { all: items.length, by_kind: {}, by_status: {}, by_evidence: { engineering_only: items.length } } })
const sweep = run()
const ready = { state: 'ready', width: 640, height: 480, fps: 25, frames: 1000 }
const tracks = (source: string, note: string | null = null) => ({ scene: 'X', source, note, frame_step: 6, history_len: 8, horizon_len: 12, tracks: [{ id: 3, frames: [0, 6], uv: [[10, 10], [12, 10]], boxes: [[5, 5, 15, 25], [7, 5, 17, 25]] }, { id: 4, frames: [0, 6], uv: [[50, 10], [52, 10]], boxes: [[45, 5, 55, 25], [47, 5, 57, 25]] }] })
const scenes = { '/health': health, '/workspace/scenes': catalogue, '/workspace/scenes/ETH/video/status': ready, '/workspace/scenes/ETH/tracks': tracks('Human annotations', 'Box size is a display estimate.') }
const prediction: PredictionResult = { experiment: sweep, source: 'sweep', across_scenes: [], rows: [{ scene: 'SYN-1', predictor: 'CV', ade_m: stat(1.562), fde_m: stat(3.86), rmse_m: stat(null), prediction_coverage: stat(1), de_by_step_m: [0.1, 0.2] }] }

const list = (dataset: string) => screen.getByRole('combobox', { name: dataset }) as HTMLSelectElement
const pick = (dataset: string, id: string) => fireEvent.change(list(dataset), { target: { value: id } })
const pickOnceLoaded = async (dataset: string, id: string) => { await screen.findByRole('combobox', { name: dataset }); pick(dataset, id) }

const pixelResult = (): InjectionResult => ({ ...injection(), scene: 'ZARA1', unit: 'px', achieved: { A: { measure: 'rmse_pos_px', by_seed: { 0: 14.1 }, mean: 14.1 } } })

// Sliders set elsewhere: Analysis must work from these alone, with no result left behind by the Workspace.
function WithErrors({ scene, errors, children }: { scene: string; errors: ErrorSettings; children: React.ReactNode }) {
  const { selectScene, setErrors } = useExperiment()
  useEffect(() => { selectScene(scene); setErrors(errors) }, [scene, errors, selectScene, setErrors])
  return <>{children}</>
}

describe('backend connection', () => {
  it('says so when the backend is down, offers retry, and shows no results', async () => {
    stubDisconnected()
    show(<WorkspacePage />)
    expect((await screen.findAllByText('Backend disconnected', {}, { timeout: 3000 })).length).toBeGreaterThan(0)
    await waitFor(() => expect(screen.getByTestId('error-state')).toBeTruthy())
    expect(screen.getAllByRole('button', { name: /retry/i }).length).toBeGreaterThan(0)
    expect(screen.queryByTestId('video-stage')).toBeNull()
  })

  it('recovers when retry finds the backend', async () => {
    stubDisconnected()
    show(<ExperimentsPage />)
    const banner = await screen.findByTestId('error-state')
    stubBackend({ '/health': health, '/experiments': page([sweep]) })
    fireEvent.click(within(banner).getByRole('button', { name: /retry/i }))
    expect(await screen.findByText('fixture~sweep')).toBeTruthy()
  })

  it('names the likely cause when another application answers at the address', async () => {
    stubBackend({})
    show(<SettingsPage />)
    expect((await screen.findAllByText(/it is not the TrackVision results API/)).length).toBeGreaterThan(0)
  })

  it('shows live status on Settings and has no fake controls', async () => {
    stubBackend({ '/health': health })
    show(<SettingsPage />)
    expect(await screen.findByText('Connected')).toBeTruthy()
    expect(screen.queryByRole('combobox')).toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()
  })
})

describe('workspace', () => {
  it('shows a loading state before the catalogue arrives', async () => {
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
    show(<WorkspacePage />)
    expect(await screen.findByTestId('loading-state')).toBeTruthy()
  })

  it('offers scenes from every dataset and opens on one with full controls', async () => {
    stubBackend(scenes)
    show(<WorkspacePage />)
    const picker = await screen.findByTestId('scene-picker')
    for (const label of ['ETH', 'HOTEL', 'UNIV', '02', '501']) expect(within(picker).getByRole('option', { name: label })).toBeTruthy()
    expect(list('ETH / HOTEL').value).toBe('ETH')
    expect(await screen.findByTestId('video-stage')).toBeTruthy()
  })

  it('keeps one list per dataset and shows no scene in the lists not in use', async () => {
    stubBackend(scenes)
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    expect(list('ETH / HOTEL').value).toBe('ETH')
    for (const dataset of ['UCY', 'MOT17', 'Synthetic']) {
      expect(list(dataset).value).toBe('')
      expect(within(list(dataset)).getByText('Choose scene')).toBeTruthy()
    }
  })

  it('moves between datasets and leaves the list it came from showing no scene', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/MOT17-02/video/status': ready, '/workspace/scenes/MOT17-02/tracks': tracks('IoU tracker on MOT17 public FRCNN detections') })
    show(<WorkspacePage />)
    await pickOnceLoaded('MOT17', 'MOT17-02')
    await waitFor(() => expect(list('MOT17').value).toBe('MOT17-02'))
    expect(list('ETH / HOTEL').value).toBe('')
  })

  it('lists the parts of a recording beside it and opens one like any other scene', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH-1/video/status': ready, '/workspace/scenes/ETH-1/tracks': tracks('Human annotations', 'Part 1 of ETH.') })
    const { container } = show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    expect([...list('ETH / HOTEL').options].map((option) => option.text)).toEqual(['ETH', 'ETH part 1', 'HOTEL'])
    pick('ETH / HOTEL', 'ETH-1')
    await waitFor(() => expect(container.querySelector('video')?.getAttribute('src')).toContain('/workspace/scenes/ETH-1/video'))
    expect((await screen.findByTestId('scene-caution')).textContent).toContain('not a separate scene')
    expect((await screen.findByTestId('track-source')).textContent).toContain('Part 1 of ETH.')
    for (const label of ['Position noise', 'Missing detections', 'Broken tracks', 'Identity switches']) expect(screen.getByRole('slider', { name: label })).toBeTruthy()
  })

  it('does not let a scene with no source video be chosen', async () => {
    stubBackend({ ...scenes, '/workspace/scenes': { ...catalogue, items: catalogue.items.map((s) => (s.id === 'HOTEL' ? { ...s, has_video: false } : s)) } })
    show(<WorkspacePage />)
    const picker = await screen.findByTestId('scene-picker')
    expect((within(picker).getByRole('option', { name: 'HOTEL' }) as HTMLOptionElement).disabled).toBe(true)
  })

  it('shows the video with its overlay and every error control', async () => {
    stubBackend(scenes)
    const { container } = show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    expect(container.querySelector('video')?.getAttribute('src')).toBe('http://127.0.0.1:8000/api/workspace/scenes/ETH/video')
    expect(screen.getByTestId('overlay')).toBeTruthy()
    for (const label of ['Position noise', 'Missing detections', 'Broken tracks', 'Identity switches']) expect(screen.getByRole('slider', { name: label })).toBeTruthy()
  })

  it('says the video is being prepared rather than showing a broken player', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/video/status': { state: 'preparing' } })
    show(<WorkspacePage />)
    expect(await screen.findByText('Preparing video, first time only')).toBeTruthy()
    expect(screen.queryByTestId('video-stage')).toBeNull()
  })

  it('explains a video that cannot be shown', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/video/status': { state: 'missing', reason: 'The source video for this scene is not on this machine.' } })
    show(<WorkspacePage />)
    expect(await screen.findByText('The source video for this scene is not on this machine.')).toBeTruthy()
  })

  it('sends a combination of errors and shows the measured effect', async () => {
    const calls = stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': injection() })
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    fireEvent.change(screen.getByRole('slider', { name: 'Position noise' }), { target: { value: '0.2' } })
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '0.1' } })
    expect(await screen.findByText('2.709 m', {}, { timeout: 3000 })).toBeTruthy()
    expect(calls.some((c) => c.includes('/inject?') && c.includes('a=0.2') && c.includes('b=0.1'))).toBe(true)
    expect(screen.getByText('0.678 m')).toBeTruthy()
    expect(screen.getByText('30.2%')).toBeTruthy()
    expect(screen.getByText('Exploratory, not a research result.')).toBeTruthy()
  })

  it('does not call the backend while no error is set', async () => {
    const calls = stubBackend(scenes)
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    await act(async () => { await new Promise((r) => setTimeout(r, 450)) })
    expect(calls.some((c) => c.includes('/inject'))).toBe(false)
  })

  it('shows a dash, not zero, when no window could be predicted', async () => {
    const empty = injection()
    empty.predictors.CV = { ...empty.predictors.CV, ade: null, ade_change: null, prediction_coverage: 0 }
    stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': empty })
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '1' } })
    expect(await screen.findByText(/No window could be predicted/, {}, { timeout: 3000 })).toBeTruthy()
    expect(screen.getByText('ADE with errors').parentElement?.textContent).toContain('—')
  })

  it('offers no error controls when a scene has no annotations and says why', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/UNIV/video/status': ready })
    show(<WorkspacePage />)
    await pickOnceLoaded('UCY', 'UNIV')
    expect(await screen.findByText('Errors unavailable here')).toBeTruthy()
    expect(screen.getByText(/not on this machine/)).toBeTruthy()
    expect(screen.queryByRole('slider', { name: 'Position noise' })).toBeNull()
  })

  const pixelScene = (result = pixelResult()) => ({ ...scenes, '/workspace/scenes/ZARA1/video/status': ready, '/workspace/scenes/ZARA1/tracks': tracks('Human annotations'), '/workspace/scenes/ZARA1/inject': result })

  it('works in pixels on a scene with no calibration and says so', async () => {
    stubBackend(pixelScene())
    show(<WorkspacePage />)
    await pickOnceLoaded('UCY', 'ZARA1')
    const noise = await screen.findByRole('slider', { name: 'Position noise' })
    expect(noise.getAttribute('max')).toBe('60')
    expect(noise.getAttribute('step')).toBe('1')
    expect(screen.getByTestId('scene-caution').textContent).toContain('video pixels')
    fireEvent.change(noise, { target: { value: '12' } })
    expect(await screen.findByText('12 px')).toBeTruthy()
    const shown = (await screen.findByText('ADE with errors', {}, { timeout: 3000 })).parentElement as HTMLElement
    expect(shown.textContent).toContain('2.709 px')
    expect(screen.getByText('ADE clean').parentElement?.textContent).toContain('0.678 px')
    expect(screen.getByText('A 14.10 px')).toBeTruthy()
  })

  it('shows no caution and no pixels on a scene in metres', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': injection() })
    show(<WorkspacePage />)
    const noise = await screen.findByRole('slider', { name: 'Position noise' })
    expect(screen.queryByTestId('scene-caution')).toBeNull()
    fireEvent.change(noise, { target: { value: '0.2' } })
    expect(await screen.findByText('0.20 m')).toBeTruthy()
    expect((await screen.findByText('ADE with errors', {}, { timeout: 3000 })).parentElement?.textContent).toContain('2.709 m')
  })

  it('has no tested range to warn about in pixels', async () => {
    stubBackend(pixelScene())
    show(<WorkspacePage />)
    await pickOnceLoaded('UCY', 'ZARA1')
    fireEvent.change(await screen.findByRole('slider', { name: 'Missing detections' }), { target: { value: '0.8' } })
    expect(screen.queryByText('Above the tested range')).toBeNull()
  })

  it('never carries position noise across a change of unit', async () => {
    stubBackend(pixelScene())
    show(<WorkspacePage />)
    const metres = await screen.findByRole('slider', { name: 'Position noise' }) as HTMLInputElement
    fireEvent.change(metres, { target: { value: '0.2' } })
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '0.3' } })
    pick('UCY', 'ZARA1')
    await screen.findByTestId('scene-caution')
    expect((screen.getByRole('slider', { name: 'Position noise' }) as HTMLInputElement).value).toBe('0')
    expect((screen.getByRole('slider', { name: 'Missing detections' }) as HTMLInputElement).value).toBe('0.3')
  })

  it('keeps position noise when the unit stays the same', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/HOTEL/video/status': ready, '/workspace/scenes/HOTEL/tracks': tracks('Human annotations') })
    show(<WorkspacePage />)
    fireEvent.change(await screen.findByRole('slider', { name: 'Position noise' }), { target: { value: '0.2' } })
    pick('ETH / HOTEL', 'HOTEL')
    await waitFor(() => expect(list('ETH / HOTEL').value).toBe('HOTEL'))
    expect((screen.getByRole('slider', { name: 'Position noise' }) as HTMLInputElement).value).toBe('0.2')
  })

  it('shows the tracker on MOT17 until errors are on, then its ground truth', async () => {
    const result = { ...pixelResult(), scene: 'MOT17-02' }
    const calls = stubBackend({ ...scenes, '/workspace/scenes/MOT17-02/video/status': ready, '/workspace/scenes/MOT17-02/tracks': tracks('IoU tracker on MOT17 public FRCNN detections'),
      '/workspace/scenes/MOT17-02/tracks?source=truth': tracks('MOT17 ground truth'), '/workspace/scenes/MOT17-02/inject': result })
    show(<WorkspacePage />)
    await pickOnceLoaded('MOT17', 'MOT17-02')
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('IoU tracker'))
    expect(calls.some((c) => c.includes('source=truth'))).toBe(false)
    fireEvent.change(await screen.findByRole('slider', { name: 'Position noise' }), { target: { value: '12' } })
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('MOT17 ground truth'), { timeout: 3000 })
    fireEvent.change(screen.getByRole('slider', { name: 'Position noise' }), { target: { value: '0' } })
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('IoU tracker'), { timeout: 3000 })
  })

  it('shows YOLO boxes on a video scene once they exist, then the annotated people with errors on', async () => {
    const withBoxes = { ...catalogue, items: catalogue.items.map((item) => item.id === 'ETH' ? { ...item, detector: 'yolo11m' } : item) }
    const calls = stubBackend({ ...scenes, '/workspace/scenes': withBoxes, '/workspace/scenes/ETH/tracks': tracks('IoU tracker on yolo11m detections', 'Boxes are YOLO11 detections.'),
      '/workspace/scenes/ETH/tracks?source=truth': tracks('Human annotations'), '/workspace/scenes/ETH/inject': injection() })
    show(<WorkspacePage />)
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('IoU tracker on yolo11m detections'))
    expect(calls.some((c) => c.includes('source=truth'))).toBe(false)
    expect(screen.getByTestId('track-source').textContent).not.toContain('Thin grey boxes')
    fireEvent.change(await screen.findByRole('slider', { name: 'Position noise' }), { target: { value: '0.2' } })
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('Human annotations'), { timeout: 3000 })
    expect(screen.getByTestId('track-source').textContent).toContain('Thin grey boxes are people with no annotation')
    fireEvent.change(screen.getByRole('slider', { name: 'Position noise' }), { target: { value: '0' } })
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('IoU tracker on yolo11m detections'), { timeout: 3000 })
  })

  it('offers the YOLO step on a video scene with no boxes yet, and never on a synthetic one', async () => {
    const withBoxes = { ...catalogue, items: catalogue.items.map((item) => item.id === 'HOTEL' ? { ...item, detector: 'yolo11m' } : item) }
    stubBackend({ ...scenes, '/workspace/scenes': withBoxes, '/workspace/scenes/HOTEL/video/status': ready, '/workspace/scenes/HOTEL/tracks': tracks('IoU tracker on yolo11m detections'), '/workspace/scenes/SYN-501/video/status': ready, '/workspace/scenes/SYN-501/tracks': tracks('Scene generator') })
    show(<WorkspacePage />)
    expect(await screen.findByText(/to track on YOLO11 detections instead/)).toBeTruthy()
    pick('ETH / HOTEL', 'HOTEL')
    await waitFor(() => expect(screen.queryByText(/to track on YOLO11 detections instead/)).toBeNull())
    pick('Synthetic', 'SYN-501')
    await screen.findByTestId('track-source')
    expect(screen.queryByText(/to track on YOLO11 detections instead/)).toBeNull()
  })

  it('tracks people on every scene and names where the IDs come from', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/MOT17-02/video/status': ready, '/workspace/scenes/MOT17-02/tracks': tracks('IoU tracker on MOT17 public FRCNN detections', 'IDs are real tracker output.') })
    show(<WorkspacePage />)
    expect((await screen.findByTestId('track-source')).textContent).toContain('Human annotations')
    expect((await screen.findByTestId('people-count')).textContent).toBe('2 people tracked')
    pick('MOT17', 'MOT17-02')
    await waitFor(() => expect(screen.getByTestId('track-source').textContent).toContain('IoU tracker on MOT17 public FRCNN detections'))
    expect(screen.getByText(/scripts\\setup_yolo\.ps1/)).toBeTruthy()
  })

  it('offers no error-only layers where errors are unavailable', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/UNIV/video/status': ready, '/workspace/scenes/UNIV/tracks': tracks('Human annotations') })
    show(<WorkspacePage />)
    await pickOnceLoaded('UCY', 'UNIV')
    expect(await screen.findByRole('checkbox', { name: 'Boxes and IDs' })).toBeTruthy()
    expect(screen.getByRole('checkbox', { name: 'Trails' })).toBeTruthy()
    expect(screen.queryByRole('checkbox', { name: 'Predicted' })).toBeNull()
  })

  it('offers true paths and forecasts only once errors are applied', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': injection() })
    show(<WorkspacePage />)
    expect((await screen.findByRole('checkbox', { name: 'True paths' }) as HTMLInputElement).disabled).toBe(true)
    fireEvent.change(screen.getByRole('slider', { name: 'Position noise' }), { target: { value: '0.2' } })
    await waitFor(() => expect((screen.getByRole('checkbox', { name: 'True paths' }) as HTMLInputElement).disabled).toBe(false), { timeout: 3000 })
  })

  it('lets the trail length change', async () => {
    stubBackend(scenes)
    show(<WorkspacePage />)
    const longer = await screen.findByRole('button', { name: '8 s' })
    expect(screen.getByRole('button', { name: '2 s' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(longer)
    expect(longer.getAttribute('aria-pressed')).toBe('true')
  })

  it('counts identity switches and offers a jump to them', async () => {
    const swapped = injection()
    swapped.overlay.events = [{ frame: 6, kind: 'switch', a: 3, b: 4 }, { frame: 12, kind: 'split', a: 3, b: null }]
    stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': swapped })
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    fireEvent.change(screen.getByRole('slider', { name: 'Identity switches' }), { target: { value: '0.1' } })
    expect((await screen.findByTestId('switch-count', {}, { timeout: 3000 })).textContent).toBe('1 ID switch')
    expect((screen.getByRole('button', { name: 'Next ID switch' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Previous ID switch' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('reports it when a scene has no tracks instead of showing an empty overlay silently', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/tracks': { status: 404, body: { detail: 'the annotations for ETH are not on this machine' } } })
    show(<WorkspacePage />)
    expect(await screen.findByText('the annotations for ETH are not on this machine')).toBeTruthy()
    expect(screen.queryByTestId('people-count')).toBeNull()
  })

  it('never lets the default scene overwrite a click made as the page opens', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/UNIV/video/status': ready })
    show(<WorkspacePage />)
    await pickOnceLoaded('UCY', 'UNIV')
    await act(async () => { await new Promise((r) => setTimeout(r, 50)) })
    expect(list('UCY').value).toBe('UNIV')
    expect(list('ETH / HOTEL').value).toBe('')
  })

  it('starts with no errors and on the first scene, whatever a previous visit left behind', async () => {
    localStorage.setItem('trackvision.selection', JSON.stringify({ sceneId: 'HOTEL', errors: { ...noErrors, a: 0.5, b: 0.4 }, analysisTab: 'stored' }))
    stubBackend(scenes)
    show(<WorkspacePage />)
    expect(((await screen.findByRole('slider', { name: 'Position noise' })) as HTMLInputElement).value).toBe('0')
    expect((screen.getByRole('slider', { name: 'Missing detections' }) as HTMLInputElement).value).toBe('0')
    expect(list('ETH / HOTEL').value).toBe('ETH')
    localStorage.clear()
  })

  it('warns when a slider goes beyond the tested range', async () => {
    stubBackend({ ...scenes, '/workspace/scenes/ETH/inject': injection() })
    show(<WorkspacePage />)
    await screen.findByTestId('video-stage')
    fireEvent.change(screen.getByRole('slider', { name: 'Missing detections' }), { target: { value: '0.8' } })
    expect(await screen.findByText('Above the tested range')).toBeTruthy()
  })
})

describe('analysis', () => {
  const noise = (scene: string): NoiseExampleData => ({ scene, exploratory: true, unit: 'm', label: 'Live example from the frozen injector, not a saved figure.', seed: 0, person: 7, frames: [0, 6], image_coordinates: false, observed: [[0, 0], [1, 0]], future: [[2, 0], [3, 0]], levels: [{ name: 'Low', sigma: 0.05, noisy: [[0, 0.1], [1, 0.1]] }, { name: 'Medium', sigma: 0.2, noisy: [[0, 0.2], [1, 0.2]] }, { name: 'High', sigma: 0.5, noisy: [[0, 0.5], [1, 0.5]] }] })
  const engineering = { experiment: run({ experiment_id: 'fixture~e5e8', kind: 'engineering_analysis', label: 'ENGINEERING AND DESCRIPTIVE ONLY.' }), source: 'engineering_analysis', label: null, ethhotel_family: null, blocked: null, deferred: null, input_runs: [],
    family_across_scenes: { scenes: ['SYN-501', 'SYN-502', 'SYN-503', 'SYN-504', 'SYN-505'], by_condition: { A: { value: 3.071, sd: 0.096, n_valid_scenes: 5 } } },
    e5_p13_engineering: { level: 'engineering_only', h7_verdict: 'not_evaluable', population: 'synthetic E4 outputs', statistical_unit: 'scene', not_p13: true, by_predictor: { CV: { correlations: { MOTA: { n: 5, value: 0.3, reason: null } }, pairs: { 'SYN-501': { HOTA: 0.8, IDF1: 0.77, IDSW: 38, MOTA: 0.91, ade_m: 1.9 } }, paired_observations: 5, scene_count: 5, tracker_count: 1, h7_verdict: 'not_evaluable', inference: 'uncomputable' } } },
    e6_dominant_dry_run: { status: 'engineering_dry_run', target: 'A', targets: ['A'], co_dominant: false, codominance_gap: 0.1, note: 'Not the research selection.', population: 'SYN-501 to SYN-505', research_selection: 'not_evaluable', research_selection_applied: false, candidates: { A: { qualifies: true, reason: null, value: 3.071 }, B2: { qualifies: false, reason: null, value: -0.041 } } },
    e7_inverse_mapping: null, e8_threshold_crossings: null }
  const scenes = ['ETH', 'ETH-1', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA1-1', 'MOT17-02']
  const routes = { '/health': health, '/workspace/scenes': catalogue, '/analysis': { items: [engineering.experiment], default: null }, '/analysis/fixture~e5e8': engineering,
    ...Object.fromEntries(scenes.flatMap((scene) => [[`/workspace/scenes/${scene}/noise-example`, noise(scene)], [`/workspace/scenes/${scene}/sweep`, sceneSweep(scene, scene === 'ZARA1' ? 'px' : 'm')]])) }
  const tile = (label: string) => screen.getByText(label).parentElement?.textContent

  describe('every scene', () => {
    it('shows every graph for the chosen scene on one page, with no tabs', async () => {
      const calls = stubBackend(routes)
      show(<AnalysisPage />)
      await screen.findByTestId('clean-numbers')
      expect(calls).toContain('/workspace/scenes/ETH/sweep')
      expect(screen.queryByRole('tab')).toBeNull()
      expect(tile('CV average error (ADE)')).toContain('0.678 m')
      expect(tile('CA final error (FDE)')).toContain('9.098 m')
      for (const chart of ['horizon-chart', 'curve-A', 'curve-B', 'curve-C', 'curve-D', 'hurts-chart', 'coverage-chart', 'time-ahead-CV', 'time-ahead-CA']) expect(screen.getByTestId(chart).querySelector('svg')).toBeTruthy()
      expect(screen.getByTestId('sweep-label').textContent).toContain('never saved')
    })

    it('uses plain names, with no experiment labels and no tracker part', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      await screen.findByText('Which error matters most overall')
      expect(document.body.textContent).not.toMatch(/\bE[5-8]\b/)
      expect(document.body.textContent).not.toContain('Detector and tracker configuration')
      expect(document.body.textContent).not.toMatch(/Spearman|AUDC|Inverse requirement|Threshold crossing/)
    })

    it('works on a part of a recording the same way', async () => {
      const calls = stubBackend(routes)
      show(<AnalysisPage />)
      await pickOnceLoaded('ETH / HOTEL', 'ETH-1')
      await waitFor(() => expect(calls).toContain('/workspace/scenes/ETH-1/sweep'))
      expect(await screen.findByTestId('curve-A')).toBeTruthy()
    })

    it('labels a scene in pixels with pixels', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      await pickOnceLoaded('UCY', 'ZARA1')
      await waitFor(() => expect(tile('CV average error (ADE)')).toContain('0.678 px'))
      expect(screen.getAllByText('Up to 0.02 px').length).toBeGreaterThan(0)
    })

    it('shows how far a calculation has got and fills in what is finished', async () => {
      const partial = { ...sceneSweep('ETH'), state: 'calculating', done: 5 }
      partial.errors = partial.errors.map((error) => error.code === 'A' ? error : { ...error, levels: [], limits: null, better: null })
      stubBackend({ ...routes, '/workspace/scenes/ETH/sweep': partial })
      show(<AnalysisPage />)
      expect((await screen.findByTestId('sweep-progress')).textContent).toContain('5 of 20')
      expect(screen.getByTestId('better-A').textContent).toBe('CV stays better at every tested level.')
      expect(screen.getByTestId('better-B').textContent).toBe('Calculating\u2026')
    })

    it('keeps asking until the calculation finishes', async () => {
      const partial = { ...sceneSweep('ETH'), state: 'calculating', done: 5 }
      const finished = sceneSweep('ETH')
      let asked = 0
      const base = routes as Record<string, unknown>
      vi.stubGlobal('fetch', vi.fn(async (input: string) => {
        const path = String(input).replace('http://127.0.0.1:8000/api', '')
        if (path === '/workspace/scenes/ETH/sweep') { asked += 1; return new Response(JSON.stringify(asked > 1 ? finished : partial)) }
        const route = base[path]
        return new Response(JSON.stringify(route ?? { detail: 'no fixture' }), { status: route === undefined ? 404 : 200 })
      }))
      show(<AnalysisPage />)
      await screen.findByTestId('sweep-progress')
      await waitFor(() => expect(screen.queryByTestId('sweep-progress')).toBeNull(), { timeout: 3000 })
      expect(asked).toBe(2)
    })

    it('reports a stopped calculation with a retry, and shows no numbers', async () => {
      const calls = stubBackend({ ...routes, '/workspace/scenes/ETH/sweep': { ...sceneSweep('ETH'), state: 'failed', reason: 'The calculation for this scene stopped; see the backend log.' } })
      show(<AnalysisPage />)
      expect((await screen.findByTestId('error-state')).textContent).toContain('stopped')
      expect(screen.queryByTestId('clean-numbers')).toBeNull()
      const before = calls.filter((path) => path.endsWith('/sweep')).length
      fireEvent.click(screen.getByRole('button', { name: /Retry/ }))
      await waitFor(() => expect(calls.filter((path) => path.endsWith('/sweep')).length).toBe(before + 1))
    })

    it('says why when a scene cannot be calculated', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      await screen.findByTestId('clean-numbers')
      pick('UCY', 'UNIV')
      expect(await screen.findByText('The annotations for this scene are not on this machine.')).toBeTruthy()
    })

    it('flips the charts between ADE and FDE without calculating again', async () => {
      const calls = stubBackend(routes)
      show(<AnalysisPage />)
      await screen.findByTestId('curve-A')
      const hurts = screen.getByText('Which error hurts most').closest('section') as HTMLElement
      expect(within(hurts).getByText('+821%')).toBeTruthy()
      const before = calls.filter((path) => path.endsWith('/sweep')).length
      fireEvent.click(screen.getByRole('button', { name: 'Final error (FDE)' }))
      expect(within(hurts).getByText('+729%')).toBeTruthy()
      expect(calls.filter((path) => path.endsWith('/sweep')).length).toBe(before)
    })

    it('says in words how CV and CA compare, in place of a dash', async () => {
      const body = sceneSweep('ETH')
      body.errors[1] = { ...body.errors[1], better: { status: 'truncated', at: null, better_when_clean: 'CV' } }
      body.errors[2] = { ...body.errors[2], better: { status: 'crossed', at: 0.2, better_when_clean: 'CV' } }
      stubBackend({ ...routes, '/workspace/scenes/ETH/sweep': body })
      show(<AnalysisPage />)
      expect((await screen.findByTestId('better-A')).textContent).toBe('CV stays better at every tested level.')
      expect(screen.getByTestId('better-B').textContent).toBe('CV stays better until too few people are left to measure.')
      expect(screen.getByTestId('better-C').textContent).toBe('CV is better at first, and CA catches up at about 20.0%.')
    })

    it('says in words how much error a scene can take, and follows the limit chosen', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      await screen.findByText('How much error this scene can take')
      const limits = screen.getByText('How much error this scene can take').closest('section') as HTMLElement
      expect(within(limits).getAllByText('Up to 0.02 m').length).toBeGreaterThan(0)
      expect(within(limits).getAllByText('Stays within the limit at every tested level, up to 2.21 per 100').length).toBe(2)
      expect(within(limits).getAllByText('Average error rises by 10%').length).toBe(2)
      expect(within(limits).getAllByText('10% of forecasts lost').length).toBe(2)
      fireEvent.change(within(limits).getByRole('combobox', { name: 'Limit' }), { target: { value: '+20%' } })
      expect(within(limits).getAllByText('Up to 0.05 m').length).toBeGreaterThan(0)
      expect(within(limits).getAllByText('20% of forecasts lost').length).toBe(2)
    })

    it('leaves a gap and says why when nobody could be predicted', async () => {
      const body = sceneSweep('ETH')
      const missing = body.errors[1]
      const last = missing.levels[missing.levels.length - 1]
      missing.levels[missing.levels.length - 1] = { ...last, coverage: 0, predictors: { CV: { ...last.predictors.CV, ade: null, fde: null, ade_rise: null, fde_rise: null }, CA: { ...last.predictors.CA, ade: null, fde: null, ade_rise: null, fde_rise: null } } }
      stubBackend({ ...routes, '/workspace/scenes/ETH/sweep': body })
      show(<AnalysisPage />)
      expect(await screen.findByText('Nobody could be predicted at the last level, so the line stops.')).toBeTruthy()
    })

    it('shows the noise example at three levels and says it is not a saved figure', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      const example = await screen.findByTestId('noise-example')
      for (const label of ['Clean observed', 'Noisy observed', 'Future ground truth']) expect(within(example).getByText(label)).toBeTruthy()
      for (const name of ['Low noise, sigma 0.05 m', 'Medium noise, sigma 0.2 m', 'High noise, sigma 0.5 m']) expect(within(example).getByRole('img', { name })).toBeTruthy()
      expect(within(example).getByText(/not a saved figure/)).toBeTruthy()
    })
  })

  describe('your errors', () => {
    it('says so in one line when none are set', async () => {
      const calls = stubBackend(routes)
      show(<AnalysisPage />)
      await waitFor(() => expect(screen.getByTestId('errors-in-use').textContent).toContain('No errors set'))
      expect(calls.some((path) => path.includes('/inject'))).toBe(false)
    })

    it('calculates the result itself, so it never depends on the Workspace having finished', async () => {
      const calls = stubBackend({ ...routes, '/workspace/scenes/ETH/inject': injection() })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, a: 0.2 }}><AnalysisPage /></WithErrors>)
      await screen.findByTestId('live-numbers')
      expect(calls.some((path) => path.startsWith('/workspace/scenes/ETH/inject?') && path.includes('a=0.2'))).toBe(true)
      expect(tile('Average error, with your errors')).toContain('2.709 m')
      expect(tile('Average error, clean')).toContain('0.678 m')
      expect(screen.getByText(/not P-10/)).toBeTruthy()
      expect(screen.getByTestId('live-ade-chart').querySelector('svg')).toBeTruthy()
      expect(screen.getByTestId('live-horizon-chart').querySelector('svg')).toBeTruthy()
    })

    it('names the errors in use, in the unit of the scene', async () => {
      stubBackend({ ...routes, '/workspace/scenes/ETH/inject': injection() })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, a: 0.2, b: 0.1, reconstruct: true, d: 0.05 }}><AnalysisPage /></WithErrors>)
      await waitFor(() => expect(screen.getByTestId('errors-in-use').textContent).toContain('Position noise 0.20 m · Missing detections 10%, gaps filled · Identity switches 5%'))
    })

    it('recalculates when the scene changes on this page', async () => {
      const calls = stubBackend({ ...routes, '/workspace/scenes/ETH/inject': injection(), '/workspace/scenes/HOTEL/inject': { ...injection(), scene: 'HOTEL' } })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, b: 0.2 }}><AnalysisPage /></WithErrors>)
      await screen.findByTestId('live-numbers')
      pick('ETH / HOTEL', 'HOTEL')
      await waitFor(() => expect(calls.some((path) => path.startsWith('/workspace/scenes/HOTEL/inject?'))).toBe(true))
      expect(await screen.findByTestId('live-numbers')).toBeTruthy()
    })

    it('labels a result in pixels with pixels', async () => {
      stubBackend({ ...routes, '/workspace/scenes/ZARA1/inject': pixelResult() })
      show(<WithErrors scene="ZARA1" errors={{ ...noErrors, a: 12 }}><AnalysisPage /></WithErrors>)
      await screen.findByTestId('live-numbers')
      expect(tile('Average error, clean')).toContain('0.678 px')
      expect(screen.getByTestId('errors-in-use').textContent).toContain('Position noise 12 px')
    })

    it('reports a failed calculation as a failure, never as no errors', async () => {
      stubBackend({ ...routes, '/workspace/scenes/ETH/inject': { status: 500, body: { detail: 'the calculation failed' } } })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, a: 0.2 }}><AnalysisPage /></WithErrors>)
      expect(await screen.findByTestId('error-state')).toBeTruthy()
      expect(screen.getByTestId('errors-in-use').textContent).toContain('In use')
    })

    it('says why a value is missing instead of showing a number', async () => {
      const result = injection()
      result.predictors.CV = { ...result.predictors.CV, ade: null, fde: null, de_by_step: null }
      stubBackend({ ...routes, '/workspace/scenes/ETH/inject': result })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, b: 1 }}><AnalysisPage /></WithErrors>)
      expect(await screen.findByText(/Nobody could be predicted at these settings, so there is no average error/)).toBeTruthy()
      expect(tile('Average error, with your errors')).toContain('—')
    })

    it('switches predictor without calculating again', async () => {
      const calls = stubBackend({ ...routes, '/workspace/scenes/ETH/inject': injection() })
      show(<WithErrors scene="ETH" errors={{ ...noErrors, a: 0.2 }}><AnalysisPage /></WithErrors>)
      await screen.findByTestId('live-numbers')
      const before = calls.filter((path) => path.includes('/inject')).length
      fireEvent.click(within(screen.getByRole('group', { name: 'Predictor' })).getByRole('button', { name: 'CA' }))
      expect(tile('Average error, with your errors')).toContain('20.000 m')
      expect(calls.filter((path) => path.includes('/inject')).length).toBe(before)
    })
  })

  describe('saved results', () => {
    it('answers the two project questions in plain words, whichever scene is chosen', async () => {
      stubBackend(routes)
      show(<AnalysisPage />)
      expect(await screen.findByText('Which error matters most overall')).toBeTruthy()
      expect(screen.getByText('Does a better tracking score mean better prediction?')).toBeTruthy()
      expect(screen.getByTestId('overall-chart').querySelector('svg')).toBeTruthy()
      expect(screen.getByTestId('pair-scatter').querySelector('svg')).toBeTruthy()
      expect(screen.getByText(/Averaged over 5 synthetic scenes/)).toBeTruthy()
      expect(screen.getByTestId('tracking-link').textContent).toContain('Not answered')
      expect(screen.getByText(/Collision warning performance has not been measured/)).toBeTruthy()
    })

    it('says so when none are stored', async () => {
      stubBackend({ ...routes, '/analysis': { items: [], default: null } })
      show(<AnalysisPage />)
      expect(await screen.findByText('No saved project results are stored.')).toBeTruthy()
    })
  })
})

describe('applications', () => {
  const routes = { '/health': health, ...Object.fromEntries([...new Set(applications.flatMap((item) => item.scenes.map((scene) => scene.id)))].map((id) => [`/workspace/scenes/${id}/sweep`, sceneSweep(id, id.startsWith('MOT17') || id.startsWith('Z') || id === 'UNIV' ? 'px' : 'm')])) }
  const choose = (label: string, value: string) => fireEvent.change(screen.getByRole('combobox', { name: label }), { target: { value } })

  it('has a dropdown of applications and opens on the first with its full details', async () => {
    const calls = stubBackend(routes)
    show(<ApplicationsPage />)
    const options = within(screen.getByRole('combobox', { name: 'Application' })).getAllByRole('option').map((option) => option.textContent)
    expect(options).toEqual(applications.map((item) => item.name))
    expect(options).toContain('Self-driving car')
    expect(options).toContain('CCTV security')
    expect(within(screen.getByTestId('application-overview')).getByRole('heading', { name: 'Self-driving car' })).toBeTruthy()
    const ranked = within(screen.getByTestId('priorities')).getAllByRole('listitem').map((item) => item.textContent)
    ;['1Position noise', '2Missing detections', '3Identity switches', '4Broken tracks'].forEach((start, index) => expect(ranked[index]?.startsWith(start)).toBe(true))
    await screen.findByTestId('application-chart')
    expect(calls).toContain('/workspace/scenes/MOT17-13/sweep')
  })

  it('changes everything when another application is chosen', async () => {
    const calls = stubBackend(routes)
    show(<ApplicationsPage />)
    await screen.findByTestId('application-chart')
    choose('Application', 'cctv')
    expect(within(screen.getByTestId('application-overview')).getByRole('heading', { name: 'CCTV security' })).toBeTruthy()
    expect(within(screen.getByTestId('priorities')).getAllByRole('listitem')[0].textContent).toContain('Identity switches')
    await waitFor(() => expect(calls).toContain('/workspace/scenes/MOT17-04/sweep'))
    expect((screen.getByRole('combobox', { name: 'Measured on' }) as HTMLSelectElement).value).toBe('MOT17-04')
  })

  it('measures on the scene chosen, and offers only the scenes named for that application', async () => {
    const calls = stubBackend(routes)
    show(<ApplicationsPage />)
    await screen.findByTestId('application-chart')
    choose('Application', 'cctv')
    expect(within(screen.getByRole('combobox', { name: 'Measured on' })).getAllByRole('option').map((option) => option.textContent)).toEqual(['MOT17-04', 'ETH', 'HOTEL'])
    choose('Measured on', 'HOTEL')
    await waitFor(() => expect(calls).toContain('/workspace/scenes/HOTEL/sweep'))
  })

  it('always says this is engineering judgement and that nothing here was measured on the application', async () => {
    stubBackend(routes)
    show(<ApplicationsPage />)
    for (const item of applications) {
      choose('Application', item.id)
      const note = screen.getByTestId('judgement-note').textContent
      expect(note).toContain('engineering judgement')
      expect(note).toContain('Collision performance has not been measured')
      expect(note).toContain('no research gate is passed')
    }
  })

  it('says where the scene is only the nearest match', async () => {
    stubBackend(routes)
    show(<ApplicationsPage />)
    expect(screen.queryByTestId('nearest-note')).toBeNull()
    choose('Application', 'drone')
    expect(screen.getByTestId('nearest-note').textContent).toContain('no footage from a moving drone')
  })

  it('shows how each error changes CV ADE, CV FDE, CA ADE, CA FDE and who is still predicted', async () => {
    stubBackend(routes)
    show(<ApplicationsPage />)
    await screen.findByTestId('application-chart')
    const row = screen.getAllByText('Position noise').map((cell) => cell.closest('tr')).find(Boolean) as HTMLElement
    for (const value of ['+821%', '+729%', '+1,221%', '+1,221%', '100.0%']) expect(within(row).getAllByText(value).length).toBeGreaterThan(0)
    expect(screen.getByTestId('sweep-label').textContent).toContain('never saved')
  })

  it('explains a dash, and keeps the row', async () => {
    const body = sceneSweep('MOT17-13', 'px')
    const missing = body.errors[1]
    const last = missing.levels[missing.levels.length - 1]
    missing.levels[missing.levels.length - 1] = { ...last, coverage: 0, predictors: { CV: { ...last.predictors.CV, ade: null, fde: null, ade_rise: null, fde_rise: null }, CA: { ...last.predictors.CA, ade: null, fde: null, ade_rise: null, fde_rise: null } } }
    stubBackend({ ...routes, '/workspace/scenes/MOT17-13/sweep': body })
    show(<ApplicationsPage />)
    expect(await screen.findByText(/A dash means nobody could be predicted at the strongest tested level/)).toBeTruthy()
  })

  it('explains how each error acts on CV, CA, ADE and FDE', async () => {
    stubBackend(routes)
    show(<ApplicationsPage />)
    const mechanics = screen.getByTestId('mechanics')
    expect(within(mechanics).getAllByText(/^(CV|CA|ADE|FDE):/).length).toBe(16)
  })

  it('reports a failed calculation without inventing numbers', async () => {
    stubBackend({ ...routes, '/workspace/scenes/MOT17-13/sweep': { status: 409, body: { detail: 'The annotations for this scene are not on this machine.' } } })
    show(<ApplicationsPage />)
    expect((await screen.findByTestId('error-state')).textContent).toContain('not on this machine')
    expect(screen.queryByTestId('application-chart')).toBeNull()
  })
})

describe('records', () => {
  it('shows an empty state when no runs exist', async () => {
    stubBackend({ '/health': health, '/experiments': page([]) })
    show(<ExperimentsPage />)
    expect(await screen.findByTestId('empty-state')).toBeTruthy()
  })

  it('lists runs, marks invalid ones, filters, and links to each run', async () => {
    const invalid = run({ experiment_id: 'fixture~broken', status: 'invalid', status_reason: 'results.json is not valid JSON' })
    stubBackend({ '/health': health, '/experiments': page([sweep, invalid]) })
    show(<ExperimentsPage />)
    expect(await screen.findByText('fixture~sweep')).toBeTruthy()
    expect(screen.getByText('results.json is not valid JSON')).toBeTruthy()
    expect(screen.getAllByRole('link', { name: 'Open' })[0].getAttribute('href')).toBe('/experiments/fixture~sweep')
    fireEvent.change(screen.getByRole('combobox', { name: 'Status' }), { target: { value: 'invalid' } })
    expect(screen.queryByText('fixture~sweep')).toBeNull()
    expect(screen.getByText('fixture~broken')).toBeTruthy()
  })

  it('shows metadata, provenance, a null association and artifacts for one run', async () => {
    const detail = { experiment: sweep, views: ['prediction'], stored_summary: null, artifacts: [{ name: 'analysis.json', type: 'record', size_bytes: 2048, served: false }, { name: 'degradation_SYN-1_A_h8.svg', type: 'chart', size_bytes: 512, served: true }] }
    stubBackend({ '/health': health, [`/experiments/${sweep.experiment_id}`]: detail, [`/prediction/${sweep.experiment_id}`]: prediction })
    show(<ExperimentDetailPage id={sweep.experiment_id} />)
    expect(await screen.findByText('Provenance')).toBeTruthy()
    expect(screen.getByText('abcdef012345 (uncommitted changes)')).toBeTruthy()
    expect(screen.getByText('No permitted value can be recorded.')).toBeTruthy()
    expect(screen.getByText('Association mechanism').parentElement?.textContent).toContain('—')
    expect(await screen.findByText('1.562')).toBeTruthy()
  })

  it('reports an unknown ID as not found', async () => {
    stubBackend({ '/health': health, '/experiments/unknown': { status: 404, body: { detail: 'no experiment has the ID unknown' } } })
    show(<ExperimentDetailPage id="unknown" />)
    expect(await screen.findByText('no experiment has the ID unknown')).toBeTruthy()
  })
})
