import { describe, expect, it, vi } from 'vitest'
import { activeSwaps, colours, displayId, drawOverlay, firstAnnotatedFrame, gateFor, idColour, lastSample, othersAt, peopleAt, stateAt, trailRuns, type OverlayView } from '@/lib/overlay'
import type { CleanTracks, InjectionResult, OverlayEvent, OverlayTrack } from '@/types'

const track: OverlayTrack = { id: 7, frames: [10, 20, 30, 60], uv: [[0, 0], [10, 0], [20, 10], [50, 40]], boxes: [[0, 0, 4, 8], [10, 0, 14, 8], [20, 10, 24, 18], [50, 40, 54, 48]] }
const clean: CleanTracks = { scene: 'X', source: 'Test', note: null, frame_step: 10, history_len: 8, horizon_len: 12, tracks: [track] }
const layersOn = { boxes: true, trails: true, truth: false, predicted: false }
const layersOff = { boxes: false, trails: false, truth: false, predicted: false }
const swap: OverlayEvent = { frame: 20, kind: 'switch', a: 7, b: 9 }

function context() {
  const calls: Record<string, number> = {}
  const count = (name: string) => () => { calls[name] = (calls[name] ?? 0) + 1 }
  const ctx = { clearRect: count('clearRect'), beginPath: count('beginPath'), moveTo: count('moveTo'), lineTo: count('lineTo'), stroke: count('stroke'), strokeRect: count('strokeRect'), fillRect: count('fillRect'), fillText: vi.fn(), setLineDash: vi.fn(), measureText: () => ({ width: 10 }) }
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls }
}

const injection = (overrides: Partial<InjectionResult['overlay']> = {}) => ({ overlay: { seed: 0, tracks: [track], events: [], events_total: 0, predicted: {}, ...overrides } }) as unknown as InjectionResult
const view = (overrides: Partial<OverlayView> = {}): OverlayView => ({ clean, others: null, injection: null, layers: layersOn, predictor: 'CV', trailFrames: 50, fps: 25, pixelsPerCssPixel: 1, ...overrides })

describe('following a person', () => {
  it('finds the last sample at or before a frame', () => {
    expect(lastSample(track.frames, 5)).toBe(-1)
    expect(lastSample(track.frames, 10)).toBe(0)
    expect(lastSample(track.frames, 29)).toBe(1)
    expect(lastSample(track.frames, 100)).toBe(3)
  })

  it('returns the stored position and box on a sample', () => {
    expect(stateAt(track, 20, 10)).toEqual({ uv: [10, 0], box: [10, 0, 14, 8] })
  })

  it('moves the box smoothly between samples', () => {
    expect(stateAt(track, 15, 10)).toEqual({ uv: [5, 0], box: [5, 0, 9, 8] })
  })

  it('shows nobody before a track starts, or across a long gap', () => {
    expect(stateAt(track, 5, 10)).toBeNull()
    expect(stateAt(track, 45, 10)).toBeNull()
    expect(stateAt(track, 200, 10)).toBeNull()
  })

  it('bridges a short detector miss so a tracked person does not blink', () => {
    const tracked: OverlayTrack = { id: 1, frames: [0, 1, 3], uv: [[0, 0], [1, 0], [3, 0]], boxes: [[0, 0, 1, 1], [1, 0, 2, 1], [3, 0, 4, 1]] }
    const gate = gateFor({ ...clean, frame_step: 1 })
    expect(gate).toBe(3)
    expect(stateAt(tracked, 2, gate)?.uv).toEqual([2, 0])
  })

  it('counts the people on screen', () => {
    expect(peopleAt([track, { ...track, id: 8, frames: [100, 110] }], 15, 10)).toBe(1)
    expect(peopleAt([track], 500, 10)).toBe(0)
  })

  it('shows a new ID when a track breaks, as a tracker would', () => {
    const broken = { ...track, segment: [7, 7, 99, 99] }
    expect(displayId(broken, 25)).toBe(7)
    expect(displayId(broken, 35)).toBe(99)
    expect(displayId(track, 35)).toBe(7)
  })

  it('gives each ID its own stable colour', () => {
    expect(idColour(7)).toBe(idColour(7))
    expect(new Set([1, 2, 3, 4, 5, 6, 7, 8].map(idColour)).size).toBe(8)
  })

  it('opens a scene on its first frame with anyone on it', () => {
    expect(firstAnnotatedFrame(clean)).toBe(10)
    expect(firstAnnotatedFrame(null)).toBe(0)
    expect(firstAnnotatedFrame({ ...clean, tracks: [] })).toBe(0)
  })
})

describe('trails', () => {
  const long: OverlayTrack = { id: 1, frames: [0, 10, 20, 30, 40, 50], uv: Array.from({ length: 6 }, (_, i) => [i, 0]) as [number, number][], boxes: [] }

  it('reaches back only as far as the chosen length, oldest first', () => {
    expect(trailRuns(long, 50, 20, 10, 1)).toEqual([[3, 4, 5]])
  })

  it('never draws ahead of the current frame', () => {
    expect(trailRuns(long, 25, 100, 10, 1)).toEqual([[0, 1, 2]])
  })

  it('has no trail before a track starts', () => {
    expect(trailRuns({ ...long, frames: long.frames.map((f) => f + 100) }, 50, 100, 10, 1)).toEqual([])
  })

  it('breaks at a dropped observation instead of drawing through it', () => {
    expect(trailRuns({ ...long, dropped: [3] }, 50, 100, 10, 1)).toEqual([[0, 1, 2], [4, 5]])
  })

  it('breaks where a track fragments, so two IDs are never joined', () => {
    expect(trailRuns({ ...long, segment: [1, 1, 1, 9, 9, 9] }, 50, 100, 10, 1)).toEqual([[0, 1, 2], [3, 4, 5]])
  })

  it('breaks across a long gap', () => {
    expect(trailRuns({ ...long, frames: [0, 10, 20, 90, 100, 110] }, 110, 200, 10, 1)).toEqual([[0, 1, 2], [3, 4, 5]])
  })

  it('thins a per-frame trail so box jitter does not read as motion', () => {
    expect(trailRuns(long, 50, 100, 10, 2)).toEqual([[1, 3, 5]])
  })
})

describe('identity switches', () => {
  it('highlights a switch from its frame for a short time only', () => {
    expect(activeSwaps([swap], 19, 25)).toEqual([])
    expect(activeSwaps([swap], 20, 25)).toEqual([swap])
    expect(activeSwaps([swap], 57, 25)).toEqual([swap])
    expect(activeSwaps([swap], 58, 25)).toEqual([])
  })

  it('does not treat a broken track as a switch', () => {
    expect(activeSwaps([{ ...swap, kind: 'split', b: null }], 20, 25)).toEqual([])
  })

  it('outlines a switching person in white as well as in their ID colour', () => {
    const plain = context()
    drawOverlay(plain.ctx, 100, 100, 20, view({ injection: injection(), layers: { ...layersOff, boxes: true } }))
    const flashing = context()
    drawOverlay(flashing.ctx, 100, 100, 20, view({ injection: injection({ events: [swap] }), layers: { ...layersOff, boxes: true } }))
    expect(flashing.calls.strokeRect).toBe(plain.calls.strokeRect * 2)
  })
})

describe('drawing', () => {
  it('draws nothing but clears the canvas with no tracks', () => {
    const { ctx, calls } = context()
    drawOverlay(ctx, 100, 100, 20, view({ clean: null }))
    expect(calls).toEqual({ clearRect: 1 })
  })

  it('tags a box with its ID', () => {
    const { ctx } = context()
    drawOverlay(ctx, 100, 100, 20, view({ layers: { ...layersOff, boxes: true } }))
    expect(ctx.fillText).toHaveBeenCalledWith('7', expect.any(Number), expect.any(Number))
  })

  it('shows the changed ID when a track has broken', () => {
    const { ctx } = context()
    drawOverlay(ctx, 100, 100, 35, view({ injection: injection({ tracks: [{ ...track, segment: [7, 7, 99, 99] }] }), layers: { ...layersOff, boxes: true }, clean: { ...clean, frame_step: 40 } }))
    expect(ctx.fillText).toHaveBeenCalledWith('99', expect.any(Number), expect.any(Number))
  })

  it('respects the layer switches', () => {
    const { ctx, calls } = context()
    drawOverlay(ctx, 100, 100, 20, view({ layers: layersOff }))
    expect(calls).toEqual({ clearRect: 1 })
  })

  it('draws a trail without boxes when only trails are on', () => {
    const { ctx, calls } = context()
    drawOverlay(ctx, 100, 100, 25, view({ layers: { ...layersOff, trails: true } }))
    expect(calls.stroke).toBeGreaterThan(0)
    expect(calls.strokeRect).toBeUndefined()
  })

  it('draws true paths only when errors are applied', () => {
    const none = context()
    drawOverlay(none.ctx, 100, 100, 25, view({ layers: { ...layersOff, truth: true } }))
    expect(none.calls.stroke).toBeUndefined()
    const some = context()
    drawOverlay(some.ctx, 100, 100, 25, view({ injection: injection(), layers: { ...layersOff, truth: true } }))
    expect(some.calls.stroke).toBeGreaterThan(0)
  })

  it('draws only the chosen predictor, on the frame that has a forecast', () => {
    const predicted = { CV: { 20: [[7, [[11, 0], [12, 0]]]] }, CA: {} } as unknown as InjectionResult['overlay']['predicted']
    const layers = { ...layersOff, predicted: true }
    const chosen = context()
    drawOverlay(chosen.ctx, 100, 100, 25, view({ injection: injection({ predicted }), layers }))
    expect(chosen.calls.stroke).toBeGreaterThan(0)
    const other = context()
    drawOverlay(other.ctx, 100, 100, 25, view({ injection: injection({ predicted }), layers, predictor: 'CA' }))
    expect(other.calls.stroke).toBeUndefined()
    const elsewhere = context()
    drawOverlay(elsewhere.ctx, 100, 100, 55, view({ injection: injection({ predicted }), layers }))
    expect(elsewhere.calls.stroke).toBeUndefined()
  })
})

describe('people with no annotation', () => {
  const tracker = (...boxes: number[][]): CleanTracks => ({ ...clean, frame_step: 1, tracks: boxes.map((box, index) => ({ id: 100 + index, frames: [20], uv: [[box[0], box[3]]], boxes: [box] })) })
  const annotatedBox = [10, 0, 14, 8]         // where the annotated person stands at frame 20

  it('keeps a box for anyone the tracker finds who is not annotated', () => {
    expect(othersAt(tracker([50, 50, 60, 70]), clean, 20)).toEqual([[50, 50, 60, 70]])
  })

  it('leaves out a box around someone already annotated', () => {
    expect(othersAt(tracker(annotatedBox), clean, 20)).toEqual([])
    expect(othersAt(tracker([11, 1, 15, 9]), clean, 20)).toEqual([])
  })

  it('counts a box that only brushes an annotated person as someone else', () => {
    expect(othersAt(tracker([13, 0, 17, 8]), clean, 20)).toEqual([[13, 0, 17, 8]])
  })

  it('finds nobody when there is no tracker, or nobody is there at that frame', () => {
    expect(othersAt(null, clean, 20)).toEqual([])
    expect(othersAt(tracker([50, 50, 60, 70]), null, 20)).toEqual([])
    expect(othersAt(tracker([50, 50, 60, 70]), clean, 40)).toEqual([])
  })

  it('draws them in plain grey, with no tag, and only while boxes are on', () => {
    const others = tracker([50, 50, 60, 70])
    const shown = context()
    const colourUsed: string[] = []
    Object.defineProperty(shown.ctx, 'strokeStyle', { set: (value: string) => { colourUsed.push(value) } })
    drawOverlay(shown.ctx, 100, 100, 20, view({ others, layers: { ...layersOff, boxes: true } }))
    expect(colourUsed).toContain(colours.other)
    const plain = context()
    drawOverlay(plain.ctx, 100, 100, 20, view({ layers: { ...layersOff, boxes: true } }))
    expect(shown.calls.fillRect).toBe(plain.calls.fillRect)
    const hidden = context()
    drawOverlay(hidden.ctx, 100, 100, 20, view({ others, layers: layersOff }))
    expect(hidden.calls.strokeRect ?? 0).toBe(0)
  })
})
