import type { CleanTracks, InjectionResult, OverlayEvent, OverlayTrack } from '@/types'

// Drawing people over video. Each ID has one colour, shared by its box, tag and trail, so a person can be followed
// through a crowd and a changed ID is obvious. Tracks are sampled every few frames, so positions between two samples
// are interpolated for display only. Numbers shown anywhere else come from the backend, never from here.
export interface Layers { boxes: boolean; trails: boolean; truth: boolean; predicted: boolean }
// `pixelsPerCssPixel` keeps lines and tags the same size on screen, whatever the resolution of the video.
// `others` is the tracker on a scene whose errors run on annotated people. Anyone it finds who is not annotated keeps a plain box.
export interface OverlayView { clean: CleanTracks | null; others: CleanTracks | null; injection: InjectionResult | null; layers: Layers; predictor: string; trailFrames: number; fps: number; pixelsPerCssPixel: number }
export const colours = { truth: '#ffffff', dropped: '#f87171', predicted: '#f472b6', future: '#4ade80', swap: '#ffffff', other: '#94a3b8' } as const
export const TRAIL_SECONDS = [2, 4, 8] as const
const MIN_GATE_FRAMES = 3             // a detector can miss a frame or two; a person should not blink for that
const TRAIL_RATE_HZ = 5               // per-frame tracker trails are drawn at this rate, so box jitter does not read as motion
const SWAP_SECONDS = 1.5              // how long an identity switch stays highlighted
const SAME_PERSON_OVERLAP = 0.5       // share of the smaller box two boxes must cover to be one person
type Point = [number, number]

// Golden-angle steps keep neighbouring IDs far apart in hue. The tag carries the ID, so colour is never the only cue.
export function idColour(id: number) { return `hsl(${(id * 137.508) % 360} 85% 62%)` }

export function lastSample(frames: number[], frame: number) {
  let low = 0, high = frames.length - 1, found = -1
  while (low <= high) {
    const mid = (low + high) >> 1
    if (frames[mid] <= frame) { found = mid; low = mid + 1 } else high = mid - 1
  }
  return found
}

function blend(a: number[], b: number[], t: number) { return a.map((value, i) => value + (b[i] - value) * t) }

// The position and box at a frame, or null if this person is not there. `gate` is the longest gap to bridge.
export function stateAt(track: OverlayTrack, frame: number, gate: number): { uv: Point; box: number[] } | null {
  const i = lastSample(track.frames, frame)
  if (i < 0) return null
  const at = track.frames[i]
  if (at === frame) return { uv: track.uv[i], box: track.boxes[i] }
  const next = track.frames[i + 1]
  if (next === undefined || next - at > gate) return frame - at < gate ? { uv: track.uv[i], box: track.boxes[i] } : null
  const t = (frame - at) / (next - at)
  return { uv: blend(track.uv[i], track.uv[i + 1], t) as Point, box: blend(track.boxes[i], track.boxes[i + 1], t) }
}

function segmentOf(track: OverlayTrack, sample: number) { return track.segment ? track.segment[sample] : track.id }

// The fragment id is what a tracker would report, so a broken track shows up as a changed ID and colour.
export function displayId(track: OverlayTrack, frame: number) {
  const i = lastSample(track.frames, frame)
  return i < 0 ? track.id : segmentOf(track, i)
}

export function gateFor(clean: CleanTracks) { return Math.max(clean.frame_step, MIN_GATE_FRAMES) }

export function peopleAt(tracks: OverlayTrack[], frame: number, gate: number) {
  return tracks.reduce((count, track) => count + (stateAt(track, frame, gate) ? 1 : 0), 0)
}

function overlap(a: number[], b: number[]) {
  const width = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const height = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  if (width <= 0 || height <= 0) return 0
  const area = (box: number[]) => (box[2] - box[0]) * (box[3] - box[1])
  return width * height / Math.min(area(a), area(b))
}

// The tracker's boxes at this frame around people with no annotation. Errors cannot act on them, because there is no truth
// to measure a forecast against, but they are still there on screen.
export function othersAt(others: CleanTracks | null, clean: CleanTracks | null, frame: number) {
  if (!others || !clean) return []
  const gate = gateFor(clean)
  const annotated = clean.tracks.flatMap((track) => { const state = stateAt(track, frame, gate); return state ? [state.box] : [] })
  const otherGate = gateFor(others)
  return others.tracks.flatMap((track) => {
    const state = stateAt(track, frame, otherGate)
    return state && !annotated.some((box) => overlap(box, state.box) >= SAME_PERSON_OVERLAP) ? [state.box] : []
  })
}

// Consecutive runs of the trail, oldest first. A gap, a dropped point or a new fragment starts a new run.
export function trailRuns(track: OverlayTrack, frame: number, trailFrames: number, gate: number, stride: number) {
  const last = lastSample(track.frames, frame)
  if (last < 0) return []
  const dropped = new Set(track.dropped ?? [])
  const runs: number[][] = []
  let run: number[] = []
  for (let k = last; k >= 0 && frame - track.frames[k] <= trailFrames; k -= 1) {
    if ((last - k) % stride !== 0) continue
    const previous = run[run.length - 1]
    const broken = previous !== undefined && (track.frames[previous] - track.frames[k] > gate * stride || segmentOf(track, previous) !== segmentOf(track, k))
    if (dropped.has(k) || broken) { if (run.length) runs.push(run); run = [] }
    if (!dropped.has(k)) run.push(k)
  }
  if (run.length) runs.push(run)
  return runs.map((indices) => indices.reverse()).reverse()
}

// Identity switches still highlighted at this frame.
export function activeSwaps(events: OverlayEvent[], frame: number, fps: number) {
  const span = SWAP_SECONDS * fps
  return events.filter((event) => event.kind === 'switch' && frame >= event.frame && frame - event.frame <= span)
}

function trailStride(clean: CleanTracks, fps: number) { return Math.max(1, Math.round(fps / TRAIL_RATE_HZ / clean.frame_step)) }

function drawTrail(ctx: CanvasRenderingContext2D, track: OverlayTrack, here: Point, colour: string, indices: number[], frame: number, trailFrames: number, scale: number, width: number) {
  const points = indices.map((k) => track.uv[k] as Point)
  const atHead = indices[indices.length - 1] === lastSample(track.frames, frame)
  if (atHead) points.push(here)
  const freshness = (i: number) => 1 - Math.min(1, (frame - (i < indices.length ? track.frames[indices[i]] : frame)) / trailFrames)
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  for (const [stroke, lineWidth] of [['#000000', (width + 2) * scale], [colour, width * scale]] as const) {
    ctx.strokeStyle = stroke
    ctx.lineWidth = lineWidth
    for (let i = 1; i < points.length; i++) {
      ctx.globalAlpha = stroke === colour ? 0.12 + 0.88 * freshness(i) : 0.45 * freshness(i)
      ctx.beginPath()
      ctx.moveTo(points[i - 1][0], points[i - 1][1])
      ctx.lineTo(points[i][0], points[i][1])
      ctx.stroke()
    }
  }
  ctx.globalAlpha = 1
}

function drawBox(ctx: CanvasRenderingContext2D, box: number[], colour: string, lineWidth: number, dash: number[] = []) {
  const [left, top, right, bottom] = box
  ctx.setLineDash([])
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = lineWidth + 2
  ctx.globalAlpha = 0.55
  ctx.strokeRect(left, top, right - left, bottom - top)
  ctx.globalAlpha = 1
  ctx.setLineDash(dash)
  ctx.strokeStyle = colour
  ctx.lineWidth = lineWidth
  ctx.strokeRect(left, top, right - left, bottom - top)
  ctx.setLineDash([])
}

function drawTag(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, background: string, scale: number) {
  const size = Math.round(11 * scale)
  ctx.font = `700 ${size}px ui-monospace, monospace`
  const width = ctx.measureText(text).width + 8 * scale
  const height = size + 5 * scale
  const top = Math.max(0, y - height)
  ctx.fillStyle = background
  ctx.fillRect(x, top, width, height)
  ctx.fillStyle = '#0b1017'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, x + 4 * scale, top + height / 2 + 0.5)
}

function drawForecast(ctx: CanvasRenderingContext2D, clean: CleanTracks, injection: InjectionResult, predictor: string, frame: number, scale: number) {
  const byId = new Map(clean.tracks.map((track) => [track.id, track]))
  const horizon = clean.horizon_len ?? 0
  const line = (points: Point[], colour: string, dash: number[]) => {
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
  for (let f = frame; f > frame - clean.frame_step; f--) {
    for (const [ped, path] of injection.overlay.predicted[predictor]?.[String(f)] ?? []) {
      const truth = byId.get(ped)
      const i = truth ? lastSample(truth.frames, f) : -1
      if (truth) line(truth.uv.slice(i, i + horizon + 1) as Point[], colours.future, [5 * scale, 4 * scale])
      line([...(truth ? [truth.uv[i] as Point] : []), ...path], colours.predicted, [])
    }
  }
}

export function drawOverlay(ctx: CanvasRenderingContext2D, width: number, height: number, frame: number, view: OverlayView) {
  ctx.clearRect(0, 0, width, height)
  const { clean, others, injection, layers, predictor, trailFrames, fps, pixelsPerCssPixel: scale } = view
  if (!clean) return
  const gate = gateFor(clean)
  const stride = trailStride(clean, fps)
  const tracks = injection ? injection.overlay.tracks : clean.tracks
  const swapping = new Set((injection ? activeSwaps(injection.overlay.events, frame, fps) : []).flatMap((event) => [event.a, event.b]))
  const present = tracks.flatMap((track) => { const state = stateAt(track, frame, gate); return state ? [{ track, state }] : [] })

  if (injection && layers.predicted) drawForecast(ctx, clean, injection, predictor, frame, scale)
  if (injection && layers.truth) {
    for (const track of clean.tracks) {
      const state = stateAt(track, frame, gate)
      if (!state) continue
      for (const run of trailRuns(track, frame, trailFrames, gate, stride)) drawTrail(ctx, track, state.uv, colours.truth, run, frame, trailFrames, scale, 1.25)
    }
  }
  if (layers.trails) {
    for (const { track, state } of present) {
      const colour = idColour(displayId(track, frame))
      for (const run of trailRuns(track, frame, trailFrames, gate, stride)) drawTrail(ctx, track, state.uv, colour, run, frame, trailFrames, scale, 2.25)
    }
  }
  if (layers.boxes) {
    for (const box of othersAt(others, clean, frame)) drawBox(ctx, box, colours.other, 1.25 * scale)
    const isDropped = (track: OverlayTrack) => { const i = lastSample(track.frames, frame); return track.frames[i] === frame && (track.dropped ?? []).includes(i) }
    for (const { track, state } of present) {
      if (swapping.has(track.id)) drawBox(ctx, state.box, colours.swap, 4 * scale)
      if (isDropped(track)) drawBox(ctx, state.box, colours.dropped, 1.5 * scale, [4 * scale, 3 * scale])
      else drawBox(ctx, state.box, idColour(displayId(track, frame)), 2 * scale)
    }
    // Nearer people are lower in the picture, so their tags go on top.
    for (const { track, state } of [...present].sort((a, b) => a.state.box[3] - b.state.box[3])) {
      const id = displayId(track, frame)
      drawTag(ctx, String(id), state.box[0], state.box[1], swapping.has(track.id) ? colours.swap : idColour(id), scale)
    }
  }
}

export function firstAnnotatedFrame(clean: CleanTracks | null) {
  if (!clean || clean.tracks.length === 0) return 0
  return Math.min(...clean.tracks.map((track) => track.frames[0]))
}
