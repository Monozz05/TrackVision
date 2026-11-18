'use client'
import { useState } from 'react'
import { GroupedBars, seriesColours } from '@/components/charts'
import { errorLabel, SweepView } from '@/components/sweep'
import { AppShell, LabelledChoice, LoadingState, Panel, Table, Td, Th } from '@/components/trackvision-app'
import { JUDGEMENT_NOTE, applications, errorEffects, type Application } from '@/config/applications'
import { MISSING, formatPercent, formatRise } from '@/lib/format'
import { finishedErrors, strongestLevel } from '@/lib/series'
import type { SceneSweep } from '@/types'

const MEASURES=[['CV','ade','CV ADE'],['CV','fde','CV FDE'],['CA','ade','CA ADE'],['CA','fde','CA FDE']] as const
const measureColours=Object.fromEntries(MEASURES.map(([,,name],index)=>[name,seriesColours[index]]))
const percent=(value:unknown)=>typeof value==='number'?formatRise(value/100):MISSING

export function ApplicationsPage(){return <AppShell><Applications/></AppShell>}

function Applications(){
  const [applicationId,setApplicationId]=useState(applications[0].id)
  const [sceneChoice,setSceneChoice]=useState<string>()
  const application=applications.find(item=>item.id===applicationId) ?? applications[0]
  const scene=application.scenes.find(item=>item.id===sceneChoice) ?? application.scenes[0]
  return <div className="flex flex-col gap-6">
    <div data-testid="application-picker" className="grid max-w-3xl gap-3 sm:grid-cols-2">
      <LabelledChoice label="Application" value={application.id} options={applications.map(item=>({value:item.id,label:item.name}))} onChange={setApplicationId}/>
      <LabelledChoice label="Measured on" value={scene.id} options={application.scenes.map(item=>({value:item.id,label:item.id}))} onChange={setSceneChoice}/>
    </div>
    <p data-testid="judgement-note" className="border border-amber-500/20 bg-amber-500/[.05] px-4 py-3 text-xs leading-5 text-amber-100/80">{JUDGEMENT_NOTE}</p>
    <Overview application={application}/>
    <Priorities application={application}/>
    <section className="flex flex-col gap-3">
      <div><h2 className="text-sm font-medium text-slate-100">How each error affects it</h2><p className="mt-0.5 max-w-4xl text-xs leading-5 text-slate-500">Measured live on {scene.id}: {scene.shows.charAt(0).toLowerCase()}{scene.shows.slice(1)}. Each value is the rise at the strongest tested level, on the people still predicted.</p></div>
      {application.nearest&&<p data-testid="nearest-note" className="text-xs leading-5 text-amber-200/80">{application.nearest}</p>}
      <SweepView key={scene.id} sceneId={scene.id}>{sweep=><Measured sweep={sweep}/>}</SweepView>
    </section>
    <Mechanics/>
  </div>
}

function Fact({label,value}:{label:string;value:string}){return <div className="bg-[#0b1017] p-3"><div className="text-[10px] text-slate-500">{label}</div><div className="mt-0.5 text-xs leading-5 text-slate-200">{value}</div></div>}

function Overview({application}:{application:Application}){
  return <section data-testid="application-overview" className="flex flex-col gap-3">
    <div><h1 className="text-lg font-semibold tracking-tight text-white">{application.name}</h1><p className="mt-1 max-w-4xl text-sm leading-6 text-slate-400">{application.summary}</p></div>
    <div className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08] lg:grid-cols-4"><Fact label="Needs most" value={application.needs}/><Fact label="Camera" value={application.camera}/><Fact label="Time ahead" value={application.timeAhead}/><Fact label="Matters more" value={application.metric.choice}/></div>
    <div className="grid gap-4 lg:grid-cols-2">
      <Panel title={`ADE or FDE: ${application.metric.choice}`}><p className="text-xs leading-5 text-slate-300">{application.metric.why}</p><dl className="mt-3 flex flex-col gap-2 text-xs leading-5"><div><dt className="inline text-slate-500">ADE: </dt><dd className="inline text-slate-400">{application.metric.ade}</dd></div><div><dt className="inline text-slate-500">FDE: </dt><dd className="inline text-slate-400">{application.metric.fde}</dd></div></dl></Panel>
      <Panel title="CV or CA"><p className="text-xs leading-5 text-slate-300">{application.predictor}</p></Panel>
    </div>
  </section>
}

function Priorities({application}:{application:Application}){
  return <section className="flex flex-col gap-3">
    <h2 className="text-sm font-medium text-slate-100">Which error to reduce first</h2>
    <ol data-testid="priorities" className="grid gap-4 lg:grid-cols-2">{application.priorities.map((priority,index)=><li key={priority.code} className="min-w-0 border border-white/[.08] bg-[#0b1017] p-4"><div className="flex items-baseline gap-3"><span className="font-mono text-sm text-cyan-300">{index+1}</span><span className="text-sm font-medium text-slate-200">{errorLabel(priority.code)}</span></div><p className="mt-2 text-xs leading-5 text-slate-300">{priority.why}</p><p className="mt-2 text-xs leading-5 text-slate-500"><span className="text-slate-400">In use: </span>{priority.inUse}</p></li>)}</ol>
  </section>
}

function Measured({sweep}:{sweep:SceneSweep}){
  const finished=finishedErrors(sweep.errors)
  if(!finished.length) return <LoadingState label="Calculating"/>
  const nobodyLeft=finished.some(error=>MEASURES.some(([predictor,metric])=>strongestLevel(error).predictors[predictor]?.[metric]===null))
  const rows=finished.map(error=>Object.fromEntries([['error',errorLabel(error.code)],...MEASURES.map(([predictor,metric,name])=>{const rise=strongestLevel(error).predictors[predictor]?.[`${metric}_rise`] ?? null; return [name,rise===null?null:rise*100]})]))
  return <div className="flex flex-col gap-4">
    <div className="grid items-start gap-4 xl:grid-cols-2">
      <Panel><Table min={560} headers={<><Th>Error</Th>{MEASURES.map(([,,name])=><Th key={name}>{name}</Th>)}<Th>Still predicted</Th></>}>{finished.map(error=>{const top=strongestLevel(error); return <tr key={error.code} className="border-b border-white/[.05]"><Td strong>{errorLabel(error.code)}</Td>{MEASURES.map(([predictor,metric,name])=><Td key={name} mono>{formatRise(top.predictors[predictor]?.[`${metric}_rise`])}</Td>)}<Td mono>{formatPercent(top.coverage)}</Td></tr>})}</Table><p className="mt-3 text-[11px] leading-5 text-slate-500">0% means the forecasts left are exactly the clean ones, so that error does its damage through the last column, by removing forecasts.{nobodyLeft&&' A dash means nobody could be predicted at the strongest tested level.'}</p></Panel>
      <Panel><GroupedBars rows={rows} category="error" bars={MEASURES.map(([,,name])=>name)} yLabel="Rise (%)" palette={measureColours} testId="application-chart" format={percent}/></Panel>
    </div>
    <p data-testid="sweep-label" className="text-[11px] leading-5 text-amber-200/80">{sweep.label}</p>
  </div>
}

function Mechanics(){
  const entries=(['A','B','C','D'] as const)
  return <section className="flex flex-col gap-3">
    <div><h2 className="text-sm font-medium text-slate-100">Why CV, CA, ADE and FDE react this way</h2><p className="mt-0.5 max-w-4xl text-xs leading-5 text-slate-500">These follow from how the predictors and the errors work, so they are the same for every application.</p></div>
    <div data-testid="mechanics" className="grid gap-4 md:grid-cols-2 2xl:grid-cols-4">{entries.map(code=><Panel key={code} title={errorLabel(code)}><dl className="flex flex-col gap-2 text-xs leading-5">{([['CV',errorEffects[code].cv],['CA',errorEffects[code].ca],['ADE',errorEffects[code].ade],['FDE',errorEffects[code].fde]] as const).map(([name,text])=><div key={name}><dt className="inline font-mono text-slate-500">{name}: </dt><dd className="inline text-slate-400">{text}</dd></div>)}</dl></Panel>)}</div>
  </section>
}
