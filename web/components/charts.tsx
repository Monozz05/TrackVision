'use client'
import { Bar, BarChart, CartesianGrid, Label, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import { formatMetric } from '@/lib/format'
import { plottable, type Point, type Series } from '@/lib/series'
import type { NoiseExampleData } from '@/types'

// Series colours belong to the entity, so CV is the same blue on every chart and filtering never repaints a survivor.
// Validated for this surface (#0b1017) with the dataviz palette checker: all five checks pass, worst adjacent CVD
// delta E 8.4. Lines also carry distinct markers, so identity never rests on colour alone.
export const seriesColours = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181'] as const
export const predictorColours: Record<string, string> = { CV: seriesColours[0], CA: seriesColours[1], LSTM: seriesColours[2], GRU: seriesColours[3] }
export const metricColours: Record<string, string> = { MOTA: seriesColours[0], IDF1: seriesColours[1], HOTA: seriesColours[2] }
// The four errors, on charts that compare them. These hues are not CV's or CA's, and the checker passes them on this surface too.
export const errorColours: Record<string, string> = { 'Position noise': seriesColours[2], 'Missing detections': seriesColours[3], 'Broken tracks': seriesColours[4], 'Identity switches': '#8f7ee6', Clean: '#cbd5e1' }

const axisTick = { fill: '#94a3b8', fontSize: 10 }
const axisLine = { stroke: '#334155' }
const grid = <CartesianGrid stroke="#1e293b" strokeDasharray="2 4" />
const legend = <Legend verticalAlign="top" height={26} iconSize={9} itemSorter={null} wrapperStyle={{ fontSize: 11, color: '#cbd5e1' }} />
const tooltipStyle = { background: '#0b1017', border: '1px solid #334155', fontSize: 11, color: '#e2e8f0' }
const tooltipItem = { color: '#e2e8f0' }
const noAnimation = { isAnimationActive: false } as const

function axisLabel(value: string, position: 'insideBottom' | 'insideLeft') {
  return <Label value={value} position={position} offset={position === 'insideBottom' ? -2 : 8} angle={position === 'insideLeft' ? -90 : 0} style={{ fill: '#94a3b8', fontSize: 10, textAnchor: 'middle' }} />
}

function numeric(value: unknown) { return typeof value === 'number' ? formatMetric(value) : '—' }
function colourFor(palette: Record<string, string>, name: string, index: number) { return palette[name] ?? seriesColours[index % seriesColours.length] }

function PointTooltip({ active, payload, xLabel, yLabel }: { active?: boolean; payload?: Array<{ payload: Point; name: string; color?: string; stroke?: string }>; xLabel: string; yLabel: string }) {
  if (!active || !payload?.length) return null
  const first = payload[0].payload
  return <div style={tooltipStyle} className="px-3 py-2"><div className="mb-1 text-slate-400">{xLabel}: <span className="font-mono text-slate-200">{formatMetric(first.x)}</span></div><div className="mb-1 text-slate-400">Requested: <span className="font-mono text-slate-200">{formatMetric(first.requested)}</span></div>{payload.map((item) => <div key={item.name} className="text-slate-300">{item.name}: <span className="font-mono">{formatMetric(item.payload.y)}</span> <span className="text-slate-500">{yLabel}</span></div>)}{first.note && <div className="mt-1 max-w-56 text-amber-200/80">{first.note}</div>}</div>
}

// One line per predictor on a numeric achieved-severity axis. Each line has its own points because each predictor
// records its own achieved values. A null y leaves a gap: connectNulls stays off.
export function DegradationChart({ series, xLabel, yLabel, palette = predictorColours, testId = 'degradation-chart', xDigits = 2 }: { series: Series[]; xLabel: string; yLabel: string; palette?: Record<string, string>; testId?: string; xDigits?: number }) {
  return <div className="h-[260px]" data-testid={testId}><ResponsiveContainer width="100%" height="100%"><LineChart margin={{ top: 4, right: 12, bottom: 16, left: 8 }}>
    {grid}
    <XAxis type="number" dataKey="x" domain={['auto', 'auto']} tick={axisTick} axisLine={axisLine} tickLine={false} tickFormatter={(v) => formatMetric(v, xDigits)} allowDuplicatedCategory={false}>{axisLabel(xLabel, 'insideBottom')}</XAxis>
    <YAxis type="number" tick={axisTick} axisLine={axisLine} tickLine={false} width={50}>{axisLabel(yLabel, 'insideLeft')}</YAxis>
    <Tooltip content={<PointTooltip xLabel={xLabel} yLabel="" />} />
    {legend}
    {series.map(plottable).map((item, index) => <Line key={item.name} data={item.points} dataKey="y" name={item.name} stroke={colourFor(palette, item.name, index)} strokeWidth={2} connectNulls={false} dot={{ r: 3.5, fill: colourFor(palette, item.name, index), stroke: '#0b1017', strokeWidth: 2 }} activeDot={{ r: 5 }} {...noAnimation} />)}
  </LineChart></ResponsiveContainer></div>
}

export function HorizonChart({ rows, names, palette = predictorColours, unit = 'm', xLabel = 'Horizon step', yLabel = `Mean displacement error (${unit})`, stepName = 'Step', testId = 'horizon-chart' }: { rows: Array<Record<string, number | null>>; names: string[]; palette?: Record<string, string>; unit?: string; xLabel?: string; yLabel?: string; stepName?: string; testId?: string }) {
  return <div className="h-[260px]" data-testid={testId}><ResponsiveContainer width="100%" height="100%"><LineChart data={rows} margin={{ top: 4, right: 12, bottom: 16, left: 8 }}>
    {grid}
    <XAxis dataKey="step" tick={axisTick} axisLine={axisLine} tickLine={false}>{axisLabel(xLabel, 'insideBottom')}</XAxis>
    <YAxis tick={axisTick} axisLine={axisLine} tickLine={false} width={50}>{axisLabel(yLabel, 'insideLeft')}</YAxis>
    <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipItem} labelFormatter={(v) => `${stepName} ${v}`} formatter={(value) => numeric(value)} />
    {legend}
    {names.map((name, index) => <Line key={name} dataKey={name} name={name} stroke={colourFor(palette, name, index)} strokeWidth={2} connectNulls={false} dot={{ r: 2.5, fill: colourFor(palette, name, index), strokeWidth: 0 }} {...noAnimation} />)}
  </LineChart></ResponsiveContainer></div>
}

export function GroupedBars({ rows, category, bars, yLabel, palette = metricColours, testId = 'bar-chart', format = numeric }: { rows: Array<Record<string, string | number | null>>; category: string; bars: string[]; yLabel: string; palette?: Record<string, string>; testId?: string; format?: (value: unknown) => string }) {
  return <div className="h-[260px]" data-testid={testId}><ResponsiveContainer width="100%" height="100%"><BarChart data={rows} margin={{ top: 4, right: 12, bottom: 16, left: 8 }} barGap={2}>
    {grid}
    <XAxis dataKey={category} tick={axisTick} axisLine={axisLine} tickLine={false} />
    <YAxis tick={axisTick} axisLine={axisLine} tickLine={false} width={50}>{axisLabel(yLabel, 'insideLeft')}</YAxis>
    <Tooltip cursor={{ fill: 'rgb(148 163 184 / .08)' }} contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipItem} formatter={(value) => format(value)} />
    {bars.length > 1 && legend}
    {bars.map((bar, index) => <Bar key={bar} dataKey={bar} name={bar} fill={colourFor(palette, bar, index)} maxBarSize={26} radius={[3, 3, 0, 0]} {...noAnimation} />)}
  </BarChart></ResponsiveContainer></div>
}

const NOISE_WIDTH = 240
const NOISE_HEIGHT = 130
const NOISE_MARGIN = 1.15            // room around the path, so markers at the edge are not cut off
const noiseMarks = { clean: seriesColours[0], noisy: seriesColours[1], future: seriesColours[2] }
const line = (points: [number, number][]) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ')

function NoiseKey({ mark, colour, label }: { mark: 'clean' | 'noisy' | 'future'; colour: string; label: string }) {
  return <span className="flex items-center gap-2 text-[11px] text-slate-300"><svg width="22" height="10" aria-hidden="true"><line x1="0" y1="5" x2="22" y2="5" stroke={colour} strokeWidth="2" strokeDasharray={mark === 'noisy' ? '4 3' : undefined} /><NoiseMark mark={mark} x={11} y={5} colour={colour} /></svg>{label}</span>
}

// Shape as well as colour tells the three series apart, as on the other charts.
function NoiseMark({ mark, x, y, colour, title }: { mark: 'clean' | 'noisy' | 'future'; x: number; y: number; colour: string; title?: string }) {
  const hover = title ? <title>{title}</title> : null
  if (mark === 'clean') return <circle cx={x} cy={y} r={2.6} fill={colour} stroke="#0b1017" strokeWidth={0.75}>{hover}</circle>
  if (mark === 'future') return <rect x={x - 2.4} y={y - 2.4} width={4.8} height={4.8} rx={0.75} fill={colour} stroke="#0b1017" strokeWidth={0.75}>{hover}</rect>
  return <path d={`M${x - 2.6} ${y - 2.6}L${x + 2.6} ${y + 2.6}M${x - 2.6} ${y + 2.6}L${x + 2.6} ${y - 2.6}`} stroke={colour} strokeWidth={1.5} strokeLinecap="round" fill="none">{hover}</path>
}

// One person at three noise levels. Every panel shares one scale, the same on x and y, so sizes can be compared by eye.
export function NoiseExample({ example }: { example: NoiseExampleData }) {
  const all = [...example.observed, ...example.future, ...example.levels.flatMap((level) => level.noisy)]
  const [xs, ys] = [all.map((p) => p[0]), all.map((p) => p[1])]
  const unitsPerPixel = Math.max((Math.max(...xs) - Math.min(...xs)) / NOISE_WIDTH, (Math.max(...ys) - Math.min(...ys)) / NOISE_HEIGHT, 1e-9) * NOISE_MARGIN
  const [middleX, middleY] = [(Math.max(...xs) + Math.min(...xs)) / 2, (Math.max(...ys) + Math.min(...ys)) / 2]
  const place = ([x, y]: [number, number]): [number, number] => {
    const down = (y - middleY) / unitsPerPixel
    return [NOISE_WIDTH / 2 + (x - middleX) / unitsPerPixel, NOISE_HEIGHT / 2 + (example.image_coordinates ? down : -down)]
  }
  const across = unitsPerPixel * NOISE_WIDTH
  const reading = ([x, y]: [number, number], step: string) => `${step}: ${formatMetric(x, 2)}, ${formatMetric(y, 2)} ${example.unit}`
  const clean = example.observed.map(place), future = example.future.map(place)
  return <div data-testid="noise-example">
    <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1.5"><NoiseKey mark="clean" colour={noiseMarks.clean} label="Clean observed" /><NoiseKey mark="noisy" colour={noiseMarks.noisy} label="Noisy observed" /><NoiseKey mark="future" colour={noiseMarks.future} label="Future ground truth" /></div>
    <div className="grid gap-3 sm:grid-cols-3">{example.levels.map((level) => {
      const noisy = level.noisy.map(place)
      return <figure key={level.name} className="min-w-0">
        <figcaption className="mb-1.5 text-xs text-slate-300">{level.name} <span className="font-mono text-slate-500">sigma {level.sigma} {example.unit}</span></figcaption>
        <svg viewBox={`0 0 ${NOISE_WIDTH} ${NOISE_HEIGHT}`} role="img" aria-label={`${level.name} noise, sigma ${level.sigma} ${example.unit}`} className="w-full border border-white/[.08] bg-[#080c12]">
          <polyline points={line(future)} fill="none" stroke={noiseMarks.future} strokeWidth={1.25} />
          <polyline points={line(clean)} fill="none" stroke={noiseMarks.clean} strokeWidth={1.25} />
          <polyline points={line(noisy)} fill="none" stroke={noiseMarks.noisy} strokeWidth={1.25} strokeDasharray="3 2" />
          {future.map(([x, y], k) => <NoiseMark key={`f${k}`} mark="future" x={x} y={y} colour={noiseMarks.future} title={reading(example.future[k], `Future step ${k + 1}`)} />)}
          {clean.map(([x, y], k) => <NoiseMark key={`c${k}`} mark="clean" x={x} y={y} colour={noiseMarks.clean} title={reading(example.observed[k], `Observed step ${k + 1}`)} />)}
          {noisy.map(([x, y], k) => <NoiseMark key={`n${k}`} mark="noisy" x={x} y={y} colour={noiseMarks.noisy} title={reading(level.noisy[k], `Noisy step ${k + 1}`)} />)}
        </svg>
      </figure>
    })}</div>
    <p className="mt-2 text-[11px] leading-5 text-slate-500">Each panel is {formatMetric(across, across < 100 ? 1 : 0)} {example.unit} wide. {example.label}</p>
  </div>
}

export function PairScatter({ rows, xKey, yKey, xLabel, yLabel }: { rows: Array<Record<string, string | number | null>>; xKey: string; yKey: string; xLabel: string; yLabel: string }) {
  return <div className="h-[260px]" data-testid="pair-scatter"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 12, bottom: 16, left: 8 }}>
    {grid}
    <XAxis type="number" dataKey={xKey} name={xLabel} domain={['auto', 'auto']} tick={axisTick} axisLine={axisLine} tickLine={false} tickFormatter={(v) => formatMetric(v, 2)}>{axisLabel(xLabel, 'insideBottom')}</XAxis>
    <YAxis type="number" dataKey={yKey} name={yLabel} domain={['auto', 'auto']} tick={axisTick} axisLine={axisLine} tickLine={false} width={50} tickFormatter={(v) => formatMetric(v, 2)}>{axisLabel(yLabel, 'insideLeft')}</YAxis>
    <Tooltip contentStyle={tooltipStyle} itemStyle={tooltipItem} labelStyle={tooltipItem} formatter={(value) => numeric(value)} />
    <Scatter data={rows} fill={seriesColours[0]} {...noAnimation} />
  </ScatterChart></ResponsiveContainer></div>
}
