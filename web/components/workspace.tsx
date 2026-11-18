'use client'
import Link from 'next/link'
import { useEffect, useState } from 'react'
import { ArrowRight, RotateCcw } from 'lucide-react'
import { AppShell, EmptyState, ErrorState, LoadingState, ResourceView, ScenePicker, EvidenceLine, Status } from '@/components/trackvision-app'
import { VideoStage } from '@/components/video-stage'
import { getCleanTracks, getVideoStatus, injectErrors, listScenes } from '@/lib/api/services'
import { useResource } from '@/lib/api/use-resource'
import { MISSING, formatMetric, formatPercent } from '@/lib/format'
import { TRAIL_SECONDS, colours, type Layers } from '@/lib/overlay'
import { errorControls, noErrors, type ErrorSettings, type InjectionResult, type VideoStatus, type WorkspaceCatalogue, type WorkspaceScene } from '@/types'
import { useExperiment } from '@/stores/experiment-store'

const DEBOUNCE_MS = 350
const anyError=(e:ErrorSettings)=>e.a>0||e.b>0||e.c>0||e.d>0

export function WorkspacePage(){
  const {resource,retry}=useResource(signal=>listScenes({signal}),'scenes')
  return <AppShell><ResourceView resource={resource} retry={retry} loading="Loading scenes">{data=><Workspace catalogue={data}/>}</ResourceView></AppShell>
}

function Workspace({catalogue}:{catalogue:WorkspaceCatalogue}){
  const {sceneId,selectScene}=useExperiment()
  const fallback=catalogue.items.find(s=>s.errors_available&&s.has_video) ?? catalogue.items[0]
  const scene=catalogue.items.find(s=>s.id===sceneId) ?? fallback
  // The default is derived, never written to the store: writing it from an effect could overwrite a click.
  if(!scene) return <EmptyState title="No scenes" message="No dataset is available on this machine."/>
  return <div>
    <ScenePicker scenes={catalogue.items} value={scene.id} onChange={next=>selectScene(next.id,next.unit!==scene.unit)}/>
    <SceneWork key={scene.id} scene={scene} catalogue={catalogue}/>
  </div>
}

function useVideoStatus(sceneId:string){
  const [status,setStatus]=useState<VideoStatus|null>(null); const [attempt,setAttempt]=useState(0)
  useEffect(()=>{
    const controller=new AbortController(); let timer:ReturnType<typeof setTimeout>
    const poll=()=>getVideoStatus(sceneId,{signal:controller.signal}).then(response=>{
      if(response.data) setStatus(response.data); else setStatus({state:'failed',reason:response.message})
      if(response.data?.state==='preparing') timer=setTimeout(poll,1200)
    }).catch(()=>{})
    void poll()
    return ()=>{ controller.abort(); clearTimeout(timer) }
  },[sceneId,attempt])
  return {status,retry:()=>{setStatus(null);setAttempt(n=>n+1)}}
}

function SceneWork({scene,catalogue}:{scene:WorkspaceScene;catalogue:WorkspaceCatalogue}){
  const {errors,injection,setInjection}=useExperiment()
  const {status,retry}=useVideoStatus(scene.id)
  const tracks=useResource(signal=>getCleanTracks(scene.id,{signal}),`tracks-${scene.id}`)
  const [applied,setApplied]=useState(errors); const [predictor,setPredictor]=useState('CV')
  const [layers,setLayers]=useState<Layers>({boxes:true,trails:true,truth:false,predicted:false})
  const [trailSeconds,setTrailSeconds]=useState(2)
  useEffect(()=>{ const timer=setTimeout(()=>setApplied(errors),DEBOUNCE_MS); return ()=>clearTimeout(timer) },[errors])
  const active=scene.errors_available&&anyError(applied)
  const result=useResource(signal=>injectErrors(scene.id,applied,{signal}),`${scene.id}|${JSON.stringify(applied)}`,active)
  // The last result stays on screen while a new one is computed, marked as updating. It only goes when errors are off.
  useEffect(()=>{ if(!active) setInjection(null); else if(result.resource.phase==='ready') setInjection(result.resource.data) },[active,result.resource,setInjection])
  const shown=injection?.scene===scene.id?injection:null
  const busy=active&&result.resource.phase==='loading'
  // A scene with a tracker shows it until errors are on. Errors run on the annotated people, because ADE needs true
  // positions, and anyone else the tracker finds keeps a plain box.
  const tracked=scene.dataset==='MOT17'||scene.detector!==null
  const onTruth=tracked&&active
  const truthTracks=useResource(signal=>getCleanTracks(scene.id,{signal},'truth'),`truth-${scene.id}`,onTruth)
  const truth=onTruth&&truthTracks.resource.phase==='ready'?truthTracks.resource.data:null
  const clean=truth??(tracks.resource.phase==='ready'?tracks.resource.data:null)
  const drawn=tracked&&!truth?null:shown
  const others=truth&&tracks.resource.phase==='ready'?tracks.resource.data:null
  const trackFailure=[onTruth?truthTracks:tracks].flatMap(({resource})=>resource.phase==='error'?[resource.message]:[])[0]
  return <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
    <section className="min-w-0 border border-white/[.08] bg-[#0b1017] p-3">
      {status===null?<LoadingState label="Checking video"/>:status.state==='ready'?<VideoStage sceneId={scene.id} status={status} clean={clean} others={others} injection={drawn} layers={layers} predictor={predictor} trailSeconds={trailSeconds}/>:status.state==='preparing'?<LoadingState label="Preparing video, first time only"/>:<ErrorState message={status.reason ?? 'The video could not be prepared.'} onRetry={status.state==='failed'?retry:undefined}/>}
      {status?.state==='ready'&&<LayerToggles layers={layers} setLayers={setLayers} injection={drawn} errorsAvailable={scene.errors_available} trailSeconds={trailSeconds} setTrailSeconds={setTrailSeconds}/>}
      {clean&&<p data-testid="track-source" className="mt-2 text-[11px] leading-5 text-slate-500"><span className="text-slate-400">{clean.source}.</span> {clean.note}{others&&' Thin grey boxes are people with no annotation, so errors do not act on them.'}</p>}
      {trackFailure&&<p className="mt-2 text-[11px] text-red-200/80">{trackFailure}</p>}
    </section>
    <aside className="flex min-w-0 flex-col gap-4">
      {scene.errors_available?<ErrorPanel scene={scene} catalogue={catalogue} result={result.resource.phase==='error'?result.resource.message:null} busy={busy} onRetry={result.retry}/>:<div className="border border-white/[.08] bg-[#0b1017] p-4"><div className="mb-2 text-sm text-slate-200">Errors unavailable here</div><p className="text-xs leading-5 text-slate-400">{scene.errors_reason}</p></div>}
      {scene.dataset!=='Synthetic'&&scene.detector===null&&<p className="px-1 text-[11px] leading-5 text-slate-500">Run <span className="font-mono text-slate-400">scripts\setup_yolo.ps1</span> to track on YOLO11 detections instead.</p>}
      {scene.errors_available&&<ResultPanel injection={shown} predictor={predictor} setPredictor={setPredictor} predictors={scene.predictors} busy={busy}/>}
      <EvidenceLine level={scene.evidence_level} exploratory={shown?.active.length?(shown.active.length>1?catalogue.exploratory_label:'Exploratory view.'):null}/>
    </aside>
  </div>
}

function LayerToggles({layers,setLayers,injection,errorsAvailable,trailSeconds,setTrailSeconds}:{layers:Layers;setLayers:(l:Layers)=>void;injection:InjectionResult|null;errorsAvailable:boolean;trailSeconds:number;setTrailSeconds:(s:number)=>void}){
  const items:[keyof Layers,string,boolean][]=[['boxes','Boxes and IDs',true],['trails','Trails',true],...(errorsAvailable?[['truth','True paths',injection!==null],['predicted','Predicted',injection!==null]] as [keyof Layers,string,boolean][]:[])]
  return <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-white/[.07] pt-3 text-xs text-slate-400">
    {items.map(([key,label,usable])=><label key={key} className={`flex items-center gap-2 ${usable?'cursor-pointer':'opacity-40'}`}><input type="checkbox" disabled={!usable} checked={layers[key]} onChange={e=>setLayers({...layers,[key]:e.target.checked})} className="accent-cyan-300"/>{label}{key==='truth'&&<span className="h-px w-4 bg-white"/>}{key==='predicted'&&<span className="h-0.5 w-4" style={{background:colours.predicted}}/>}</label>)}
    {injection&&layers.predicted&&<span className="flex items-center gap-2"><span className="h-px w-4 border-t border-dashed" style={{borderColor:colours.future}}/>True future</span>}
    <div className="ml-auto flex items-center gap-2"><span className="text-slate-500">Trail</span><div className="flex border border-white/[.12]">{TRAIL_SECONDS.map(value=><button key={value} aria-pressed={trailSeconds===value} onClick={()=>setTrailSeconds(value)} className={`px-2 py-1 text-[11px] ${trailSeconds===value?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{value} s</button>)}</div></div>
  </div>
}

function ErrorPanel({scene,catalogue,result,busy,onRetry}:{scene:WorkspaceScene;catalogue:WorkspaceCatalogue;result:string|null;busy:boolean;onRetry:()=>void}){
  const {errors,setErrors}=useExperiment()
  const controls=scene.unit==='px'?catalogue.pixel_controls:catalogue.controls
  return <div className="border border-white/[.08] bg-[#0b1017] p-4">
    <div className="mb-4 flex items-center justify-between"><div className="text-sm font-medium text-slate-200">Errors</div><button onClick={()=>setErrors(noErrors)} disabled={!anyError(errors)} className="flex items-center gap-1.5 text-[11px] text-slate-500 hover:text-slate-200 disabled:opacity-30"><RotateCcw className="size-3"/> Reset</button></div>
    {scene.caution&&<p data-testid="scene-caution" className="mb-4 text-[11px] leading-5 text-amber-200/80">{scene.caution}</p>}
    <div className="flex flex-col gap-4">{errorControls.map(control=>{
      const limit=controls[control.code]; const value=errors[control.key]
      const shown=limit.unit==='m'?`${value.toFixed(2)} m`:limit.unit==='px'?`${Math.round(value)} px`:`${Math.round(value*100)}%`
      const beyond=limit.protocol_max!==null&&value>limit.protocol_max
      return <label key={control.key} className="block" title={control.help}><div className="mb-1.5 flex items-baseline justify-between gap-2"><span className="text-xs text-slate-300">{control.label}</span><span className={`font-mono text-xs ${value>0?'text-amber-300':'text-slate-600'}`}>{shown}</span></div><input aria-label={control.label} type="range" min={0} max={limit.max} step={limit.unit==='px'?1:0.01} value={value} onChange={e=>setErrors({...errors,[control.key]:Number(e.target.value)})} className="w-full accent-amber-400"/>{beyond&&<div className="mt-1 text-[10px] text-amber-300/70">Above the tested range</div>}{control.key==='b'&&value>0&&<span className="mt-1.5 flex items-center gap-2 text-[11px] text-slate-500"><input type="checkbox" checked={errors.reconstruct} onChange={e=>setErrors({...errors,reconstruct:e.target.checked})} className="accent-cyan-300"/>Fill gaps by interpolation</span>}</label>})}</div>
    {busy&&<div role="status" className="mt-3 text-[11px] text-slate-500">Applying errors…</div>}
    {result&&<div className="mt-3"><ErrorState message={result} onRetry={onRetry}/></div>}
  </div>
}

function ResultPanel({injection,predictor,setPredictor,predictors,busy}:{injection:InjectionResult|null;predictor:string;setPredictor:(p:string)=>void;predictors:string[];busy:boolean}){
  const row=injection?.predictors[predictor]; const unit=injection?.unit
  return <div className="border border-white/[.08] bg-[#0b1017] p-4">
    <div className="mb-3 flex items-center justify-between gap-2"><div className="text-sm font-medium text-slate-200">Effect on prediction</div><div className="flex border border-white/[.12]">{predictors.map(name=><button key={name} onClick={()=>setPredictor(name)} className={`px-2.5 py-1 text-[11px] ${name===predictor?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{name}</button>)}</div></div>
    {!injection||!row?<p className="text-xs leading-5 text-slate-500">{busy?'Calculating…':'Move a slider to apply errors.'}</p>:<div className={busy?'opacity-50':undefined}>
      <div className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08]"><Tile label="ADE clean" value={`${formatMetric(row.clean_ade)} ${unit}`}/><Tile label="ADE with errors" value={row.ade===null?MISSING:`${formatMetric(row.ade)} ${unit}`} strong/><Tile label="Change" value={row.ade_change===null?MISSING:`${row.ade_change>=0?'+':''}${formatPercent(row.ade_change)}`}/><Tile label="Coverage" value={formatPercent(row.prediction_coverage)}/></div>
      {row.ade===null&&<p className="mt-2 text-[11px] leading-5 text-amber-200/80">No window could be predicted at these settings, so there is no ADE.</p>}
      <div className="mt-3 flex flex-wrap gap-1.5">{Object.entries(injection.achieved).map(([code,item])=><Status key={code} label={`${code} ${item.measure.startsWith('rmse_pos')?`${formatMetric(item.mean,2)} ${unit}`:item.measure==='switch_event_rate'?`${formatMetric(item.mean,2)}/100`:formatPercent(item.mean)}`}/>)}</div>
      <Link href="/analysis" className="mt-3 flex items-center gap-2 text-xs text-cyan-300 hover:underline">Open analysis <ArrowRight className="size-3"/></Link>
    </div>}
  </div>
}
function Tile({label,value,strong=false}:{label:string;value:string;strong?:boolean}){return <div className="bg-[#0b1017] p-2.5"><div className="text-[10px] text-slate-500">{label}</div><div className={`mt-0.5 font-mono text-sm ${strong?'text-amber-300':'text-slate-200'}`}>{value}</div></div>}
