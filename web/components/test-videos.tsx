'use client'
import { useEffect, useState } from 'react'
import { ExternalLink, RotateCcw } from 'lucide-react'
import { TestVideoStage, type StreetLayers } from '@/components/test-video-stage'
import { AppShell, EmptyState, ErrorState, LabelledChoice, LoadingState, ResourceView, Status } from '@/components/trackvision-app'
import { useResource } from '@/lib/api/use-resource'
import { MISSING, formatMetric, formatPercent } from '@/lib/format'
import { TRAIL_SECONDS, colours } from '@/lib/overlay'
import { getStreetForecast, getStreetTracks, keptForecasts, listStreetVideos, type StreetCatalogue, type StreetForecast, type StreetVideo } from '@/lib/test-videos'
import { anyError, errorControls, noErrors, type ErrorSettings, type InjectionResult } from '@/types'

const DEBOUNCE_MS = 350

export function TestVideosPage(){
  const {resource,retry}=useResource(signal=>listStreetVideos({signal}),'street-videos')
  return <AppShell><ResourceView resource={resource} retry={retry} loading="Loading test videos">{data=><TestVideos catalogue={data}/>}</ResourceView></AppShell>
}

function TestVideos({catalogue}:{catalogue:StreetCatalogue}){
  const [choice,setChoice]=useState<string>()
  const video=catalogue.items.find(item=>item.id===choice) ?? catalogue.items.find(item=>item.ready) ?? catalogue.items[0]
  if(!video) return <EmptyState title="No test videos" message="No test video is listed."/>
  return <div>
    <div className="mb-4 grid items-end gap-x-6 gap-y-2 lg:grid-cols-[minmax(0,360px)_minmax(0,1fr)]">
      <LabelledChoice label="Test video" value={video.id} options={catalogue.items.map(item=>({value:item.id,label:item.title}))} onChange={setChoice}/>
      <p data-testid="section-label" className="text-[11px] leading-5 text-slate-500">{catalogue.label}</p>
    </div>
    {video.ready?<VideoWork key={video.id} video={video} catalogue={catalogue}/>:<ErrorState message={video.reason ?? 'This video is not ready.'}/>}
  </div>
}

function VideoWork({video,catalogue}:{video:StreetVideo;catalogue:StreetCatalogue}){
  const [errors,setErrors]=useState<ErrorSettings>(noErrors); const [applied,setApplied]=useState<ErrorSettings>(noErrors)
  const [predictor,setPredictor]=useState('CV'); const [trailSeconds,setTrailSeconds]=useState<number>(2)
  const [layers,setLayers]=useState<StreetLayers>({boxes:true,trails:true,truth:false,predicted:true,later:false})
  useEffect(()=>{ const timer=setTimeout(()=>setApplied(errors),DEBOUNCE_MS); return ()=>clearTimeout(timer) },[errors])
  const tracks=useResource(signal=>getStreetTracks(video.id,{signal}),`street-tracks-${video.id}`)
  const result=useResource(signal=>getStreetForecast(video.id,applied,{signal}),`${video.id}|${JSON.stringify(applied)}`)
  // The last result stays on screen while a new one is computed, marked as updating.
  const [forecast,setForecast]=useState<StreetForecast|null>(null)
  useEffect(()=>{ if(result.resource.phase==='ready') setForecast(result.resource.data) },[result.resource])
  const busy=result.resource.phase==='loading'
  const perFrame=tracks.resource.phase==='ready'?tracks.resource.data:null
  // With no errors the boxes follow the tracker frame by frame. Errors act on the tracks sampled every step.
  const withErrors=forecast!==null&&forecast.active.length>0
  const clean=withErrors&&perFrame?{...perFrame,frame_step:forecast.grid.frame_step,tracks:forecast.grid.tracks}:perFrame
  const injection=withErrors?{overlay:forecast.overlay} as InjectionResult:null
  const failure=tracks.resource.phase==='error'?tracks.resource.message:null
  return <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
    <section className="min-w-0 border border-white/[.08] bg-[#0b1017] p-3">
      <TestVideoStage video={video} clean={clean} injection={injection} forecast={forecast} layers={layers} predictor={predictor} trailSeconds={trailSeconds}/>
      <LayerToggles layers={layers} setLayers={setLayers} withErrors={withErrors} trailSeconds={trailSeconds} setTrailSeconds={setTrailSeconds}/>
      {clean&&<p data-testid="track-source" className="mt-2 text-[11px] leading-5 text-slate-500"><span className="text-slate-400">{clean.source}.</span> {clean.note}{forecast&&Math.abs(forecast.seconds_per_step-0.4)>0.005?` One step here is ${forecast.seconds_per_step.toFixed(2)} s.`:''}</p>}
      {failure&&<ErrorState message={failure} onRetry={tracks.retry}/>}
    </section>
    <aside className="flex min-w-0 flex-col gap-4">
      <ErrorSliders video={video} catalogue={catalogue} errors={errors} setErrors={setErrors} busy={busy} failure={result.resource.phase==='error'?result.resource.message:null} onRetry={result.retry}/>
      <ResultPanel forecast={forecast} catalogue={catalogue} predictor={predictor} setPredictor={setPredictor} busy={busy}/>
      <Credit video={video}/>
    </aside>
  </div>
}

function LayerToggles({layers,setLayers,withErrors,trailSeconds,setTrailSeconds}:{layers:StreetLayers;setLayers:(l:StreetLayers)=>void;withErrors:boolean;trailSeconds:number;setTrailSeconds:(s:number)=>void}){
  const items:[keyof StreetLayers,string,boolean][]=[['boxes','Boxes and IDs',true],['trails','Trails',true],['predicted','Forecast',true],['later','Tracker saw next',true],['truth','Path without errors',withErrors]]
  return <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-white/[.07] pt-3 text-xs text-slate-400">
    {items.map(([key,label,usable])=><label key={key} className={`flex items-center gap-2 ${usable?'cursor-pointer':'opacity-40'}`}><input type="checkbox" disabled={!usable} checked={layers[key]} onChange={e=>setLayers({...layers,[key]:e.target.checked})} className="accent-cyan-300"/>{label}{key==='truth'&&<span className="h-px w-4 bg-white"/>}{key==='predicted'&&<span className="h-0.5 w-4" style={{background:colours.predicted}}/>}{key==='later'&&<span className="h-px w-4 border-t border-dashed" style={{borderColor:colours.future}}/>}</label>)}
    <div className="ml-auto flex items-center gap-2"><span className="text-slate-500">Trail</span><div className="flex border border-white/[.12]">{TRAIL_SECONDS.map(value=><button key={value} aria-pressed={trailSeconds===value} onClick={()=>setTrailSeconds(value)} className={`px-2 py-1 text-[11px] ${trailSeconds===value?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{value} s</button>)}</div></div>
  </div>
}

function ErrorSliders({video,catalogue,errors,setErrors,busy,failure,onRetry}:{video:StreetVideo;catalogue:StreetCatalogue;errors:ErrorSettings;setErrors:(e:ErrorSettings)=>void;busy:boolean;failure:string|null;onRetry:()=>void}){
  return <div className="border border-white/[.08] bg-[#0b1017] p-4">
    <div className="mb-4 flex items-center justify-between"><div className="text-sm font-medium text-slate-200">Errors</div><button onClick={()=>setErrors(noErrors)} disabled={!anyError(errors)} className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-200 disabled:opacity-30"><RotateCcw className="size-3"/> Reset</button></div>
    {video.caution&&<p data-testid="video-caution" className="mb-4 text-[11px] leading-5 text-amber-200/80">{video.caution}</p>}
    <p className="mb-4 text-[11px] leading-5 text-slate-500">These act on the tracker&apos;s own tracks, because this video has no true positions.</p>
    <div className="flex flex-col gap-4">{errorControls.map(control=>{
      const limit=catalogue.controls[control.code]; const value=errors[control.key]
      const shown=limit.unit==='px'?`${Math.round(value)} px`:`${Math.round(value*100)}%`
      return <div key={control.key} title={control.help}><label className="block"><div className="mb-1.5 flex items-baseline justify-between gap-2"><span className="text-xs text-slate-300">{control.label}</span><span className={`font-mono text-xs ${value>0?'text-amber-300':'text-slate-600'}`}>{shown}</span></div><input aria-label={control.label} type="range" min={0} max={limit.max} step={limit.unit==='px'?1:0.01} value={value} onChange={e=>setErrors({...errors,[control.key]:Number(e.target.value)})} className="w-full accent-amber-400"/></label>{control.key==='b'&&value>0&&<label className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500"><input type="checkbox" checked={errors.reconstruct} onChange={e=>setErrors({...errors,reconstruct:e.target.checked})} className="accent-cyan-300"/>Fill gaps by interpolation</label>}</div>})}</div>
    {busy&&<div role="status" className="mt-3 text-[11px] text-slate-500">Applying errors…</div>}
    {failure&&<div className="mt-3"><ErrorState message={failure} onRetry={onRetry}/></div>}
  </div>
}

function ResultPanel({forecast,catalogue,predictor,setPredictor,busy}:{forecast:StreetForecast|null;catalogue:StreetCatalogue;predictor:string;setPredictor:(p:string)=>void;busy:boolean}){
  const moved=forecast?.moved[predictor]
  return <div data-testid="result-panel" className="border border-white/[.08] bg-[#0b1017] p-4">
    <div className="mb-3 flex items-center justify-between gap-2"><div className="text-sm font-medium text-slate-200">Effect on prediction</div><div className="flex border border-white/[.12]">{catalogue.predictors.map(name=><button key={name} aria-pressed={name===predictor} onClick={()=>setPredictor(name)} className={`px-2.5 py-1 text-[11px] ${name===predictor?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{name}</button>)}</div></div>
    {!forecast?<LoadingState label="Forecasting"/>:<div className={busy?'opacity-50':undefined}>
      <div className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08]"><Tile label="Forecasts kept" value={`${keptForecasts(forecast).toLocaleString('en-GB')} of ${forecast.forecasts.total.toLocaleString('en-GB')}`}/><Tile label="Forecast moved" value={moved===null||moved===undefined?MISSING:`${formatMetric(moved,1)} px`} strong/><Tile label="ADE" value={MISSING}/><Tile label="FDE" value={MISSING}/></div>
      <p data-testid="no-accuracy" className="mt-2 text-[11px] leading-5 text-slate-500">{catalogue.no_accuracy} {catalogue.moved_note}</p>
      {moved===null&&<p className="mt-2 text-[11px] leading-5 text-amber-200/80">No forecast is left at these settings, so nothing can be compared.</p>}
      {forecast.active.length>0&&<div className="mt-3 flex flex-wrap gap-1.5">{Object.entries(forecast.achieved).map(([code,item])=><Status key={code} label={`${errorControls.find(c=>c.code===code)?.label ?? code} ${item.measure.startsWith('rmse_pos')?`${formatMetric(item.mean,1)} px`:item.measure==='switch_event_rate'?`${formatMetric(item.mean,2)}/100`:formatPercent(item.mean)}`}/>)}</div>}
    </div>}
  </div>
}
function Tile({label,value,strong=false}:{label:string;value:string;strong?:boolean}){return <div className="bg-[#0b1017] p-2.5"><div className="text-[10px] text-slate-500">{label}</div><div className={`mt-0.5 font-mono text-sm ${strong?'text-amber-300':'text-slate-200'}`}>{value}</div></div>}

function Credit({video}:{video:StreetVideo}){
  return <div data-testid="video-credit" className="border border-white/[.08] bg-[#0b1017] p-4 text-[11px] leading-5 text-slate-400">
    <div className="mb-1 text-sm font-medium text-slate-200">{video.title}</div>
    <div>By {video.author}</div>
    <div>Camera: {video.camera.toLowerCase()}</div>
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1"><a href={video.licence_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-cyan-300 hover:underline">{video.licence} <ExternalLink className="size-3"/></a><a href={video.source_url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-cyan-300 hover:underline">Source <ExternalLink className="size-3"/></a></div>
  </div>
}
