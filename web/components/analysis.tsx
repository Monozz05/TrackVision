'use client'
import Link from 'next/link'
import { useState } from 'react'
import { ArrowRight } from 'lucide-react'
import { DegradationChart, GroupedBars, HorizonChart, NoiseExample, PairScatter, errorColours, predictorColours } from '@/components/charts'
import { errorLabel, SweepView } from '@/components/sweep'
import { AppShell, ChoiceField, EmptyState, LoadingState, Note, Panel, ResourceView, ScenePicker, Table, Td, Th } from '@/components/trackvision-app'
import { getAnalysis, getNoiseExample, injectErrors, listAnalysisRuns, listScenes } from '@/lib/api/services'
import { useResource } from '@/lib/api/use-resource'
import { MISSING, betterCaption, errorAxis, formatMetric, formatPercent, formatRise, limitText } from '@/lib/format'
import { coverageRows, finishedErrors, horizonRows, riseRows, secondsAhead, strongestLevel, sweepSeries } from '@/lib/series'
import { anyError, errorControls, type EngineeringAnalysis, type ErrorSettings, type InjectionResult, type SceneSweep, type SweepError, type WorkspaceCatalogue, type WorkspaceScene } from '@/types'
import { useExperiment } from '@/stores/experiment-store'

type Metric='ade'|'fde'
const metricNames:Record<Metric,string>={ade:'Average error (ADE)',fde:'Final error (FDE)'}
const comparison={Clean:'#3987e5','With errors':'#d95926'}
const predictorsOf=(sweep:SceneSweep)=>Object.keys(sweep.clean ?? {})
const percent=(value:unknown)=>typeof value==='number'?formatRise(value/100):MISSING
const CHANGED_FORECASTS=['A','D']      // the two errors that change a forecast; the others only remove forecasts

export function AnalysisPage(){
  const {resource,retry}=useResource(signal=>listScenes({signal}),'scenes')
  return <AppShell><ResourceView resource={resource} retry={retry} loading="Loading scenes">{data=><Analysis catalogue={data}/>}</ResourceView></AppShell>
}

function Analysis({catalogue}:{catalogue:WorkspaceCatalogue}){
  const {sceneId,selectScene}=useExperiment()
  const fallback=catalogue.items.find(s=>s.errors_available&&s.has_video) ?? catalogue.items[0]
  const scene=catalogue.items.find(s=>s.id===sceneId) ?? fallback
  // The default is derived, never written to the store: writing it from an effect could overwrite a click.
  if(!scene) return <EmptyState title="No scenes" message="No dataset is available on this machine."/>
  return <div>
    <ScenePicker scenes={catalogue.items} value={scene.id} onChange={next=>selectScene(next.id,next.unit!==scene.unit)}/>
    {scene.errors_available?<SweepView key={scene.id} sceneId={scene.id}>{sweep=><SceneAnalysis scene={scene} sweep={sweep}/>}</SweepView>:<Note>{scene.errors_reason}</Note>}
    <SavedResults/>
  </div>
}

function Section({title,note,action,children}:{title:string;note?:string;action?:React.ReactNode;children:React.ReactNode}){
  return <section className="flex min-w-0 flex-col gap-3"><div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-2"><div className="min-w-0"><h2 className="text-sm font-medium text-slate-100">{title}</h2>{note&&<p className="mt-0.5 max-w-4xl text-xs leading-5 text-slate-500">{note}</p>}</div>{action}</div>{children}</section>
}

function SceneAnalysis({scene,sweep}:{scene:WorkspaceScene;sweep:SceneSweep}){
  const [metric,setMetric]=useState<Metric>('ade')
  const switcher=<div role="group" aria-label="Measure" className="flex border border-white/[.12]">{(Object.keys(metricNames) as Metric[]).map(key=><button key={key} aria-pressed={key===metric} onClick={()=>setMetric(key)} className={`px-3 py-1.5 text-xs ${key===metric?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{metricNames[key]}</button>)}</div>
  return <div className="flex flex-col gap-7">
    <p data-testid="sweep-label" className="text-[11px] leading-5 text-amber-200/80">{sweep.label}</p>
    <CleanPrediction sweep={sweep}/>
    <YourErrors scene={scene}/>
    <Section title="What each error does" note="Each chart starts from clean tracks on the left and shows one error growing. Lower is better." action={switcher}><div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">{sweep.errors.map(error=><ErrorCurve key={error.code} error={error} sweep={sweep} metric={metric}/>)}</div></Section>
    <div className="grid items-start gap-4 xl:grid-cols-2">
      <Section title="Which error hurts most" note={`Rise in ${metric==='ade'?'average error (ADE)':'final error (FDE)'} at the strongest tested level, on the people still predicted.`}><HurtsMost sweep={sweep} metric={metric}/></Section>
      <Section title="People still predicted" note="Share of forecasts that can still be made. Position noise and identity switches keep nearly all of them. Missing detections and broken tracks remove them."><Panel><HorizonChart rows={coverageRows(sweep.errors,errorLabel)} names={sweep.errors.map(error=>errorLabel(error.code))} palette={errorColours} xLabel="Error level (0 is none, 5 is the strongest tested)" yLabel="Still predicted (%)" stepName="Level" testId="coverage-chart"/></Panel></Section>
    </div>
    <TimeAhead sweep={sweep}/>
    <Limits sweep={sweep}/>
    <NoisePanel scene={scene}/>
  </div>
}

function Tile({label,value,strong=false,large=false}:{label:string;value:string;strong?:boolean;large?:boolean}){return <div className="flex flex-col justify-center bg-[#0b1017] p-3"><div className="text-[10px] text-slate-500">{label}</div><div className={`mt-0.5 font-mono ${large?'text-xl':'text-sm'} ${strong?'text-amber-300':'text-slate-200'}`}>{value}</div></div>}

function CleanPrediction({sweep}:{sweep:SceneSweep}){
  const predictors=predictorsOf(sweep); const unit=sweep.unit
  const longest=Math.max(0,...predictors.map(name=>sweep.clean?.[name].de_by_step?.length ?? 0))
  const measured=(value:number|null|undefined)=>value===null||value===undefined?MISSING:`${formatMetric(value)} ${unit}`
  const profiles=predictors.map(name=>({name,steps:sweep.clean?.[name].de_by_step}))
  return <Section title="Prediction with clean tracks" note={`CV assumes each person keeps their current speed and direction. CA also assumes they keep changing speed in the same way. Average error (ADE) covers the whole forecast, and final error (FDE) is at its last point${longest?`, ${(longest*sweep.seconds_per_step).toFixed(1)} s ahead`:''}.`}>
    {!sweep.clean?<LoadingState label="Reading this scene"/>:<div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
      <div className="flex h-full flex-col gap-3"><div data-testid="clean-numbers" className="grid flex-1 auto-rows-fr grid-cols-2 gap-px border border-white/[.08] bg-white/[.08]">{predictors.flatMap(name=>[<Tile key={`${name}-ade`} label={`${name} average error (ADE)`} value={measured(sweep.clean?.[name].ade)} strong large/>,<Tile key={`${name}-fde`} label={`${name} final error (FDE)`} value={measured(sweep.clean?.[name].fde)} large/>])}</div><p className="text-[11px] leading-5 text-slate-500">Measured on {sweep.windows?.toLocaleString('en-US') ?? MISSING} forecasts in this scene. Each forecast uses 8 points of a person&apos;s past path.</p></div>
      <Panel title="Error over time ahead"><HorizonChart rows={secondsAhead(horizonRows(profiles),sweep.seconds_per_step)} names={predictors} unit={unit} xLabel="Seconds ahead" yLabel={`Error (${unit})`} stepName="Seconds ahead:"/></Panel>
    </div>}
  </Section>
}

function ErrorCurve({error,sweep,metric}:{error:SweepError;sweep:SceneSweep;metric:Metric}){
  const predictors=predictorsOf(sweep); const unit=sweep.unit
  const gaps=error.levels.some(level=>predictors.some(name=>level.predictors[name]?.[metric]===null))
  return <Panel title={errorLabel(error.code)}><DegradationChart series={predictors.map(name=>sweepSeries(error,sweep.clean,metric,name))} xLabel={errorAxis(error.code,unit)} yLabel={`${metric.toUpperCase()} (${unit})`} xDigits={error.code==='A'||error.code==='D'?2:0} testId={`curve-${error.code}`}/><p data-testid={`better-${error.code}`} className="mt-2 text-[11px] leading-5 text-slate-300">{betterCaption(error.code,error.better,unit)}</p>{!CHANGED_FORECASTS.includes(error.code)&&<p className="mt-1 text-[11px] leading-5 text-slate-500">This error removes forecasts and leaves the rest unchanged, so ups and downs here come from who is left.</p>}{gaps&&<p className="mt-1 text-[11px] leading-5 text-amber-200/80">Nobody could be predicted at the last level, so the line stops.</p>}</Panel>
}

function HurtsMost({sweep,metric}:{sweep:SceneSweep;metric:Metric}){
  const predictors=predictorsOf(sweep); const finished=finishedErrors(sweep.errors)
  if(!finished.length) return <LoadingState label="Calculating"/>
  return <Panel><GroupedBars rows={riseRows(sweep.errors,metric,predictors,errorLabel)} category="error" bars={predictors} yLabel="Rise (%)" palette={predictorColours} testId="hurts-chart" format={percent}/>
    <div className="mt-3"><Table min={300} headers={<><Th>Error</Th>{predictors.map(name=><Th key={name}>{name}</Th>)}</>}>{finished.map(error=><tr key={error.code} className="border-b border-white/[.05]"><Td strong>{errorLabel(error.code)}</Td>{predictors.map(name=><Td key={name} mono>{formatRise(strongestLevel(error).predictors[name]?.[`${metric}_rise`])}</Td>)}</tr>)}</Table></div>
  </Panel>
}

function TimeAhead({sweep}:{sweep:SceneSweep}){
  const shown=finishedErrors(sweep.errors).filter(error=>CHANGED_FORECASTS.includes(error.code))
  const names=['Clean',...shown.map(error=>errorLabel(error.code))]
  return <Section title="Time ahead at the strongest level" note="How far forecasts drift the further ahead they look. Missing detections and broken tracks are not drawn, because the forecasts left are the same as with clean tracks."><div className="grid gap-4 xl:grid-cols-2">{predictorsOf(sweep).map(name=><Panel key={name} title={name}><HorizonChart rows={secondsAhead(horizonRows([{name:'Clean',steps:sweep.clean?.[name].de_by_step},...shown.map(error=>({name:errorLabel(error.code),steps:strongestLevel(error).predictors[name]?.de_by_step}))]),sweep.seconds_per_step)} names={names} palette={errorColours} unit={sweep.unit} xLabel="Seconds ahead" yLabel={`Error (${sweep.unit})`} stepName="Seconds ahead:" testId={`time-ahead-${name}`}/></Panel>)}</div></Section>
}

function Limits({sweep}:{sweep:SceneSweep}){
  const allowed=sweep.limits_allowed; const options=allowed.map(q=>`+${Math.round(Number(q)*100)}%`)
  const [choice,setChoice]=useState(options[1] ?? options[0])
  const q=allowed[options.indexOf(choice)] ?? allowed[0]; const predictors=predictorsOf(sweep)
  return <Section title="How much error this scene can take" note="The most of each error before the change below passes the limit you choose. Describes this scene only, and is not a safety limit." action={<div className="w-32"><ChoiceField label="Limit" value={choice} options={options} onChange={setChoice}/></div>}>
    <Panel><Table min={620} headers={<><Th>Error</Th><Th>What the limit is on</Th>{predictors.map(name=><Th key={name}>{name}</Th>)}</>}>{sweep.errors.map(error=><tr key={error.code} className="border-b border-white/[.05]"><Td strong>{errorLabel(error.code)}</Td><Td>{error.limit_on==='ade'?`Average error rises by ${choice.replace('+','')}`:`${choice.replace('+','')} of forecasts lost`}</Td>{predictors.map(name=><Td key={name} mono>{limitText(error.code,error.limits?.[q]?.[name],sweep.unit)}</Td>)}</tr>)}</Table></Panel>
  </Section>
}

function describeErrors(errors:ErrorSettings,unit:string){
  return errorControls.filter(control=>errors[control.key]>0).map(control=>{
    const value=errors[control.key]
    if(control.key==='a') return `${control.label} ${unit==='px'?`${Math.round(value)} px`:`${value.toFixed(2)} m`}`
    return `${control.label} ${Math.round(value*100)}%${control.key==='b'&&errors.reconstruct?', gaps filled':''}`
  })
}

// Calculated here from the current sliders, so a result never depends on the Workspace having finished first.
function YourErrors({scene}:{scene:WorkspaceScene}){
  const {errors}=useExperiment()
  const active=anyError(errors); const used=describeErrors(errors,scene.unit)
  const {resource,retry}=useResource(signal=>injectErrors(scene.id,errors,{signal}),`${scene.id}|${JSON.stringify(errors)}`,active)
  return <Section title="Your Workspace errors" action={<Link href="/workspace" className="inline-flex items-center gap-1.5 text-xs text-cyan-300 hover:underline">Change in Workspace <ArrowRight className="size-3"/></Link>}>
    <p data-testid="errors-in-use" className="text-xs text-slate-400">{used.length?`In use: ${used.join(' · ')}`:'No errors set. Set them in the Workspace to see their combined effect here.'}</p>
    {active&&<ResourceView resource={resource} retry={retry} loading="Calculating your errors">{data=><LiveResult injection={data}/>}</ResourceView>}
  </Section>
}

function LiveResult({injection}:{injection:InjectionResult}){
  const predictors=Object.keys(injection.predictors)
  const [choice,setChoice]=useState(predictors[0]); const predictor=predictors.includes(choice)?choice:predictors[0]
  const row=injection.predictors[predictor]; const unit=injection.unit
  const bars=predictors.map(name=>({predictor:name,Clean:injection.predictors[name].clean_ade,'With errors':injection.predictors[name].ade}))
  const steps=[{name:'Clean',steps:row?.clean_de_by_step},{name:'With errors',steps:row?.de_by_step}]
  const measured=(value:number|null|undefined)=>value===null||value===undefined?MISSING:`${formatMetric(value)} ${unit}`
  return <div className="flex flex-col gap-3">
    <div className="flex flex-wrap items-center gap-3">
      <div role="group" aria-label="Predictor" className="flex border border-white/[.12]">{predictors.map(name=><button key={name} aria-pressed={name===predictor} onClick={()=>setChoice(name)} className={`px-3 py-1.5 text-xs ${name===predictor?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{name}</button>)}</div>
      <p className="text-[11px] leading-5 text-amber-200/80">{injection.label}</p>
    </div>
    <div data-testid="live-numbers" className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08] md:grid-cols-5"><Tile label="Average error, clean" value={measured(row?.clean_ade)}/><Tile label="Average error, with your errors" value={measured(row?.ade)} strong/><Tile label="Same people, clean" value={measured(row?.clean_same_windows_ade)}/><Tile label="Final error, with your errors" value={measured(row?.fde)}/><Tile label="People still predicted" value={formatPercent(row?.prediction_coverage)}/></div>
    {row?.ade===null&&<p className="text-[11px] leading-5 text-amber-200/80">Nobody could be predicted at these settings, so there is no average error.</p>}
    <div className="grid gap-4 xl:grid-cols-2">
      <Panel title="Average error, clean and with your errors"><GroupedBars rows={bars} category="predictor" bars={['Clean','With errors']} yLabel={`ADE (${unit})`} palette={comparison} testId="live-ade-chart"/></Panel>
      <Panel title={`Error over time ahead, ${predictor}`}>{row?.de_by_step?.length?<HorizonChart rows={horizonRows(steps)} names={['Clean','With errors']} palette={comparison} unit={unit} xLabel="Steps ahead" yLabel={`Error (${unit})`} testId="live-horizon-chart"/>:<EmptyState title="No predictions" message="Nobody could be predicted at these settings."/>}</Panel>
    </div>
  </div>
}

function NoisePanel({scene}:{scene:WorkspaceScene}){
  const {resource,retry}=useResource(signal=>getNoiseExample(scene.id,{signal}),`noise-${scene.id}`)
  return <Section title="Position noise example" note="One real person's path at three noise levels."><ResourceView resource={resource} retry={retry} loading="Preparing the noise example">{example=><Panel><NoiseExample example={example}/></Panel>}</ResourceView></Section>
}

// Saved project results. These two are whole-project answers from synthetic scenes, so they do not follow the chosen scene.
function SavedResults(){
  const {resource,retry}=useResource(signal=>listAnalysisRuns({signal}),'analysis-index')
  return <div className="mt-7"><ResourceView resource={resource} retry={retry} loading="Finding saved results">{index=>{const run=index.items.find(r=>r.kind==='engineering_analysis'&&r.status==='complete'); return run?<SavedRun runId={run.experiment_id}/>:<Note>No saved project results are stored.</Note>}}</ResourceView></div>
}

function SavedRun({runId}:{runId:string}){
  const {resource,retry}=useResource(signal=>getAnalysis<EngineeringAnalysis>(runId,{},{signal}),runId)
  return <ResourceView resource={resource} retry={retry} loading="Loading saved results">{data=><SavedDetail data={data}/>}</ResourceView>
}

const CONDITION_NAMES:Record<string,string>={A:'Position noise',B2:'Missing detections',C:'Broken tracks',D:'Identity switches'}

function SavedDetail({data}:{data:EngineeringAnalysis}){
  const association=data.e5_p13_engineering; const dry=data.e6_dominant_dry_run
  const stored=Object.keys(association?.by_predictor ?? {})
  const predictors=['CV','CA',...stored.filter(name=>name!=='CV'&&name!=='CA')].filter(name=>stored.includes(name))
  const [choice,setChoice]=useState(predictors[0]); const predictor=predictors.includes(choice)?choice:predictors[0]
  const row=association?.by_predictor[predictor]
  const pairs=Object.entries(row?.pairs ?? {}).map(([scene,pair])=>({scene,...pair}))
  const scenes=data.family_across_scenes?.scenes?.length
  return <Section title="Across the synthetic scenes" note="Saved results from the project's stored runs. They cover the synthetic scenes only, so they stay the same whichever scene is chosen above.">
    <div className="grid items-start gap-4 xl:grid-cols-2">
      {dry&&<Panel title="Which error matters most overall"><GroupedBars rows={Object.entries(dry.candidates).map(([mode,candidate])=>({error:CONDITION_NAMES[mode] ?? mode,'Damage score':candidate.value}))} category="error" bars={['Damage score']} yLabel="Damage score" testId="overall-chart"/><p className="mt-2 text-[11px] leading-5 text-slate-500">A higher score means the error raised forecast error more across its tested levels. Averaged over {scenes ?? MISSING} synthetic scenes. This is not real-world evidence.</p></Panel>}
      {association&&<Panel title="Does a better tracking score mean better prediction?"><div className="mb-3 flex items-center gap-3"><div role="group" aria-label="Predictor for tracking score" className="flex border border-white/[.12]">{predictors.map(name=><button key={name} aria-pressed={name===predictor} onClick={()=>setChoice(name)} className={`px-3 py-1.5 text-xs ${name===predictor?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{name}</button>)}</div></div><PairScatter rows={pairs} xKey="MOTA" yKey="ade_m" xLabel="Tracking score (MOTA)" yLabel="Average error, ADE (m)"/><p data-testid="tracking-link" className="mt-2 text-[11px] leading-5 text-slate-500">{association.h7_verdict==='not_evaluable'?`Not answered. Each dot is one of ${pairs.length} synthetic scenes with one tracker, which is too few to say.`:`Saved verdict: ${association.h7_verdict.replaceAll('_',' ')}.`}</p></Panel>}
    </div>
    <p className="text-[11px] leading-5 text-slate-500">Collision warning performance has not been measured or validated. Mitigation, collision operating curves and adaptive model selection are not implemented.</p>
  </Section>
}
