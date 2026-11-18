'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useState } from 'react'
import { ArrowLeft, CircleHelp, RotateCcw, Settings2 } from 'lucide-react'
import { appConfig, navItems } from '@/config/app'
import { GroupedBars, HorizonChart } from '@/components/charts'
import { figureUrl, getExperiment, getPredictionResult, getTrackingResult, listExperiments } from '@/lib/api/services'
import { useResource, type Resource } from '@/lib/api/use-resource'
import { MISSING, evidenceLabels, formatCount, formatDate, formatHash, formatMetric, formatPercent, formatStat, glossary, statOf } from '@/lib/format'
import { historyByDay, horizonRows, trackingRows, uniqueValues } from '@/lib/series'
import type { EvidenceLevel, Experiment, ErrorKind, ExperimentDetail, PredictionResult, SelectionQuery, TrackingResult, WorkspaceScene } from '@/types'
import { useExperiment } from '@/stores/experiment-store'

function classNames(...values:(string|false|undefined)[]) { return values.filter(Boolean).join(' ') }

function BackendIndicator(){
  const {backend}=useExperiment()
  const label=backend.phase==='connected'?`Backend connected · ${backend.health.experiment_records} records`:backend.phase==='checking'?'Checking backend':'Backend disconnected'
  return <span data-testid="backend-indicator" className="inline-flex items-center gap-2"><span className={classNames('size-1.5',backend.phase==='connected'?'bg-emerald-400':backend.phase==='checking'?'bg-slate-500':'bg-red-400')} />{label}</span>
}

function BackendBanner(){
  const {backend,recheck}=useExperiment()
  if(backend.phase!=='disconnected') return null
  return <div role="alert" className="mb-6 flex flex-wrap items-center justify-between gap-3 border border-red-400/30 bg-red-400/[.06] px-4 py-3"><div><div className="font-mono text-[10px] uppercase tracking-wider text-red-300">Backend disconnected</div><div className="mt-1 text-xs text-red-100/80">{backend.message} No results are shown while it is unreachable, and nothing is substituted for them.</div></div><button onClick={recheck} className="flex items-center gap-2 border border-red-300/30 px-3 py-2 text-xs text-red-200 hover:bg-red-400/10"><RotateCcw className="size-3.5" /> Retry</button></div>
}

export function AppShell({children}:{children:React.ReactNode}) {
  const pathname=usePathname()
  return <div className="min-h-screen bg-[#070a0f] text-slate-100">
    <header className="sticky top-0 z-30 flex h-14 items-center gap-2 border-b sm:gap-5 border-white/[.07] bg-[#070a0f]/95 px-4 backdrop-blur-xl lg:px-6">
      <Link href="/workspace" className="flex shrink-0 items-center gap-2.5"><span className="grid size-7 place-items-center border border-cyan-400/40 bg-cyan-400/10 font-mono text-[11px] font-bold text-cyan-300">TV</span><span className="hidden text-sm font-semibold tracking-tight sm:inline">TrackVision</span></Link>
      <nav className="flex min-w-0 items-center gap-0.5 overflow-x-auto sm:gap-1">{navItems.map(item=><Link key={item.href} href={item.href} className={classNames('shrink-0 px-2 py-1.5 text-xs sm:px-3 sm:text-sm',pathname.startsWith(item.href)?'bg-white/[.07] text-white':'text-slate-500 hover:text-slate-200')}>{item.label}</Link>)}</nav>
      <div className="ml-auto flex shrink-0 items-center gap-2 text-[11px] text-slate-500 sm:gap-4"><span className="hidden sm:inline"><BackendIndicator/></span><Link href="/experiments" className="hover:text-slate-200">Records</Link><Link href="/settings" aria-label="Settings" className="hover:text-slate-200"><Settings2 className="size-4"/></Link></div>
    </header>
    <main className="mx-auto max-w-[1600px] px-4 py-4 lg:px-6"><BackendBanner/>{children}</main>
  </div>
}

export function PageHeader({eyebrow,title,description,stage}:{eyebrow:string;title:string;description:string;stage?:string}){return <div className="mb-8 flex flex-col justify-between gap-5 border-b border-white/[.07] pb-7 xl:flex-row xl:items-end"><div><div className="mb-3 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.25em] text-cyan-400"><span className="size-1.5 bg-cyan-400" /> {eyebrow}</div><h1 className="max-w-4xl text-2xl font-semibold tracking-tight text-white sm:text-3xl">{title}</h1><p className="mt-3 max-w-3xl text-sm leading-6 text-slate-500">{description}</p></div>{stage&&<div className="shrink-0 font-mono text-[10px] uppercase tracking-[.2em] text-slate-600">Stage {stage} / 08</div>}</div>}
export function Panel({title,eyebrow,children,className=''}:{title?:string;eyebrow?:string;children:React.ReactNode;className?:string}){return <section className={classNames('min-w-0 border border-white/[.08] bg-[#0b1017]',className)}>{title&&<div className="border-b border-white/[.07] px-5 py-3.5"><div className="text-sm font-medium text-slate-200">{title}</div>{eyebrow&&<div className="mt-1 font-mono text-[9px] uppercase tracking-[.18em] text-slate-500">{eyebrow}</div>}</div>}<div className="p-5">{children}</div></section>}
export function MetricGrid({items}:{items:{label:string;value:string;detail?:string;term?:string}[]}){return <div className="grid grid-cols-2 gap-px border border-white/[.08] bg-white/[.08] md:grid-cols-4">{items.map(item=><div key={item.label} className="bg-[#0b1017] p-4"><div className="font-mono text-[10px] uppercase tracking-wider text-slate-500">{item.label}{item.term&&<Tip term={item.term}/>}</div><div className="mt-2 font-mono text-xl text-slate-200">{item.value}</div>{item.detail&&<div className="mt-1 text-[11px] text-slate-500">{item.detail}</div>}</div>)}</div>}
export function EmptyState({title='No stored result',message='No stored run provides this result.'}:{title?:string;message?:string}){return <div data-testid="empty-state" className="grid min-h-36 place-items-center border border-dashed border-white/[.12] p-5 text-center"><div><div className="font-mono text-xs uppercase tracking-wider text-slate-400">{title}</div><p className="mx-auto mt-2 max-w-xl text-xs leading-5 text-slate-500">{message}</p></div></div>}
export function Note({children}:{children:React.ReactNode}){return <p data-testid="note" className="border border-white/[.08] bg-[#0b1017] px-5 py-3.5 text-xs leading-5 text-slate-400">{children}</p>}
export function Status({label,kind='placeholder'}:{label:string;kind?:'success'|'warning'|'danger'|'placeholder'}){return <span className={classNames('inline-flex items-center gap-2 border px-2 py-1 font-mono text-[10px] uppercase tracking-wider',kind==='success'?'border-emerald-400/20 text-emerald-300':kind==='warning'?'border-amber-400/20 text-amber-300':kind==='danger'?'border-red-400/30 text-red-300':'border-white/[.1] text-slate-400')}><span className="size-1.5 bg-current" />{label}</span>}
function Field({label,children}:{label:string;children:React.ReactNode}){return <label className="block min-w-0"><span className="mb-2 block font-mono text-[10px] uppercase tracking-wider text-slate-500">{label}</span>{children}</label>}
function Select({value,children,onChange,label,active=false}:{value?:string;children:React.ReactNode;onChange?:React.ChangeEventHandler<HTMLSelectElement>;label?:string;active?:boolean}){return <select aria-label={label} value={value} onChange={onChange} className={classNames('w-full min-w-0 border bg-[#080c12] px-3 py-2.5 text-xs outline-none focus:border-cyan-400/50',active?'border-cyan-400/50 text-cyan-300':'border-white/[.1] text-slate-300')}>{children}</select>}

export function Tip({term}:{term:string}){return <span title={glossary[term]} aria-label={`${term}: ${glossary[term]}`} className="ml-1 inline-flex cursor-help align-middle text-slate-600 hover:text-cyan-300"><CircleHelp className="size-3" /></span>}
export function Th({children,term}:{children:React.ReactNode;term?:string}){return <th className="pb-3 pr-5 font-normal whitespace-nowrap">{children}{term&&<Tip term={term}/>}</th>}
export function Table({headers,children,min=600}:{headers:React.ReactNode;children:React.ReactNode;min?:number}){return <div className="overflow-x-auto"><table className="w-full text-left text-xs" style={{minWidth:min}}><thead className="border-b border-white/[.08] font-mono text-[10px] uppercase tracking-wider text-slate-500"><tr>{headers}</tr></thead><tbody>{children}</tbody></table></div>}
export function Td({children,mono=false,strong=false}:{children:React.ReactNode;mono?:boolean;strong?:boolean}){return <td className={classNames('py-3 pr-5',mono&&'font-mono',strong?'text-slate-200':'text-slate-400')}>{children}</td>}
export function KeyValues({rows}:{rows:[string,React.ReactNode][]}){return <div className="grid gap-0">{rows.map(([key,value])=><div key={key} className="flex justify-between gap-6 border-b border-white/[.06] py-2.5 text-xs"><span className="shrink-0 text-slate-500">{key}</span><span className="min-w-0 break-words text-right text-slate-300">{value}</span></div>)}</div>}

export function LoadingState({label='Loading stored results'}:{label?:string}){return <div data-testid="loading-state" role="status" className="grid min-h-36 place-items-center border border-dashed border-white/[.1] text-center"><div className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-wider text-slate-500"><span className="size-1.5 animate-pulse bg-cyan-300" />{label}</div></div>}
export function ErrorState({message,errorKind,onRetry}:{message:string;errorKind?:ErrorKind;onRetry?:()=>void}){return <div data-testid="error-state" role="alert" className="grid min-h-36 place-items-center border border-red-400/20 bg-red-400/[.04] p-5 text-center"><div><div className="font-mono text-xs uppercase tracking-wider text-red-300">{errorKind==='disconnected'?'Backend unreachable':errorKind==='not_found'?'Not found':errorKind==='rejected'?'Record not served':'Could not load this result'}</div><p className="mx-auto mt-2 max-w-xl text-xs leading-5 text-red-100/70">{message}</p>{onRetry&&<button onClick={onRetry} className="mt-4 inline-flex items-center gap-2 border border-red-300/30 px-3 py-2 text-xs text-red-200 hover:bg-red-400/10"><RotateCcw className="size-3.5" /> Retry</button>}</div></div>}
export function ResourceView<T>({resource,retry,empty,loading,children}:{resource:Resource<T>;retry:()=>void;empty?:{title:string;message:string};loading?:string;children:(data:T)=>React.ReactNode}){
  if(resource.phase==='loading') return <LoadingState label={loading}/>
  if(resource.phase==='error') return <ErrorState message={resource.message} errorKind={resource.errorKind} onRetry={retry}/>
  if(resource.phase==='empty') return <EmptyState title={empty?.title} message={resource.message ?? empty?.message}/>
  return <>{children(resource.data)}</>
}

const evidenceKind:Record<EvidenceLevel,'warning'|'placeholder'>={engineering_only:'warning',engineering_and_descriptive:'warning',descriptive_local_only:'warning',feasibility_only:'placeholder',diagnostic_only:'placeholder',scaffold_check:'placeholder'}
export function EvidenceBadge({level}:{level:EvidenceLevel|null}){return level?<Status label={evidenceLabels[level]} kind={evidenceKind[level]}/>:<Status label="Evidence level unknown"/>}
export function RunStatusBadge({experiment}:{experiment:Experiment}){return <Status label={experiment.status} kind={experiment.status==='complete'?'success':experiment.status==='invalid'?'danger':'warning'}/>}
export function RunNotice({experiment}:{experiment:Experiment}){return <div data-testid="run-notice" className="mb-5 border border-amber-500/20 bg-amber-500/[.05] p-4"><div className="flex flex-wrap items-center gap-2"><EvidenceBadge level={experiment.evidence_level}/><span className="font-mono text-[10px] text-amber-200/70">{experiment.experiment_id}</span><span className="font-mono text-[10px] text-slate-500">{formatDate(experiment.timestamp)}</span></div>{experiment.label&&<p className="mt-3 text-xs leading-5 text-amber-100/80">{experiment.label}</p>}{experiment.notice&&<p className="mt-2 text-xs leading-5 text-amber-200">{experiment.notice}</p>}</div>}

function SelectionFields({selection,query,onChange}:{selection:{population:string;history:number;populations:string[];histories:number[]};query:SelectionQuery;onChange:(next:SelectionQuery)=>void}){return <><Field label="Population"><Select label="Population" value={query.population ?? selection.population} onChange={e=>onChange({population:e.target.value})}>{selection.populations.map(p=><option key={p}>{p}</option>)}</Select></Field><Field label="History length"><Select label="History length" value={String(query.history ?? selection.history)} onChange={e=>onChange({...query,history:Number(e.target.value)})}>{selection.histories.map(h=><option key={h} value={h}>{h}</option>)}</Select></Field></>}
export function ChoiceField({label,value,options,onChange}:{label:string;value:string;options:string[];onChange:(v:string)=>void}){return <Field label={label}><Select label={label} value={value} onChange={e=>onChange(e.target.value)}>{options.map(o=><option key={o}>{o}</option>)}</Select></Field>}
export function LabelledChoice({label,value,options,onChange}:{label:string;value:string;options:{value:string;label:string}[];onChange:(v:string)=>void}){return <Field label={label}><Select label={label} value={value} onChange={e=>onChange(e.target.value)}>{options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</Select></Field>}
export function choose(value:string|undefined,options:string[]){return value&&options.includes(value)?value:options[0] ?? ''}
function PopulationNote({label,aggregation}:{label?:string|null;aggregation?:string|null}){if(!label&&!aggregation) return null; return <div className="mb-5 text-[11px] leading-5 text-slate-500">{label}{label&&aggregation?' ':''}{aggregation&&`Aggregation: ${aggregation}.`}</div>}

export function TrackingDetail({data,withNotice=true}:{data:TrackingResult;withNotice?:boolean}){
  const primary=data.groups.find(g=>g.primary)?.name ?? data.groups[0]?.name
  const [selected,setSelected]=useState(primary)
  const group=data.groups.find(g=>g.name===selected) ?? data.groups[0]
  if(!group) return <EmptyState title="No sequences" message="This run stores no sequence results."/>
  return <div>
    {withNotice&&<RunNotice experiment={data.experiment}/>}
    {data.groups.length>1&&<div className="mb-5 flex flex-wrap gap-2">{data.groups.map(g=><button key={g.name} onClick={()=>setSelected(g.name)} className={classNames('border px-3 py-2 text-xs',g.name===group.name?'border-cyan-400/40 bg-cyan-400/[.06] text-cyan-300':'border-white/[.1] text-slate-400 hover:bg-white/[.04]')}>{g.name}{g.primary?' · primary':''}</button>)}</div>}
    <div className="grid gap-5 xl:grid-cols-[1.4fr_.8fr]">
      <Panel title="Official tracking metrics by sequence" eyebrow={`${group.name} · ${data.experiment.evidence_level?evidenceLabels[data.experiment.evidence_level]:'evidence unknown'}`}><GroupedBars rows={trackingRows(group.sequences)} category="sequence" bars={['MOTA','IDF1','HOTA']} yLabel="Score (0 to 1)" testId="tracking-chart"/></Panel>
      <Panel title="Tracker configuration" eyebrow="As recorded by the run"><TrackerSettings settings={group.tracker_settings}/></Panel>
    </div>
    <Panel title="Sequence results" eyebrow="Official TrackEval · not pooled" className="mt-5"><Table min={720} headers={<><Th>Sequence</Th><Th>Regime</Th><Th>Frames</Th><Th term="MOTA">MOTA</Th><Th term="IDF1">IDF1</Th><Th term="HOTA">HOTA</Th><Th term="IDSW">IDSW</Th><Th term="Frag">Frag</Th></>}>{group.sequences.map(row=><tr key={row.sequence} className="border-b border-white/[.05]"><Td strong mono>{row.sequence}</Td><Td>{row.regime ?? MISSING}</Td><Td mono>{row.frames_evaluated?.join(' to ') ?? MISSING}</Td><Td mono>{formatMetric(row.metrics.mota)}</Td><Td mono>{formatMetric(row.metrics.idf1)}</Td><Td mono>{formatMetric(row.metrics.hota)}</Td><Td mono>{formatCount(row.metrics.idSwitches)}</Td><Td mono>{formatCount(row.metrics.fragmentations)}</Td></tr>)}</Table>{data.official_note&&<p className="mt-3 text-xs text-slate-500">{data.official_note}</p>}</Panel>
    {(data.prediction_note||data.not_satisfied?.length)&&<Panel title="What this run does not establish" eyebrow="Recorded limits" className="mt-5">{data.prediction_note&&<p className="mb-3 text-xs leading-5 text-slate-400">{data.prediction_note}</p>}{data.not_satisfied&&<ul className="grid gap-2">{data.not_satisfied.map(item=><li key={item} className="border-l border-amber-400/30 pl-3 text-xs leading-5 text-slate-500">{item}</li>)}</ul>}</Panel>}
  </div>
}
function TrackerSettings({settings}:{settings:Record<string,unknown>|null}){
  if(!settings) return <EmptyState title="Not recorded" message="This run stores no tracker settings."/>
  const tracker=(settings.tracker && typeof settings.tracker==='object'?settings.tracker:settings) as Record<string,unknown>
  const rows=Object.entries(tracker).filter(([,v])=>typeof v!=='object'||v===null).map(([k,v])=>[k.replaceAll('_',' '),v===null?MISSING:String(v)] as [string,string])
  return rows.length?<KeyValues rows={rows}/>:<EmptyState title="Not recorded" message="This run stores no tracker settings."/>
}

const adeFdePalette={ADE:'#3987e5',FDE:'#d95926'}
export function PredictionDetail({data,query,setQuery,withNotice=true}:{data:PredictionResult;query?:SelectionQuery;setQuery?:(q:SelectionQuery)=>void;withNotice?:boolean}){
  const scenes=uniqueValues(data.rows.map(r=>r.scene ?? MISSING)); const [sceneChoice,setScene]=useState<string>(); const scene=choose(sceneChoice,scenes)
  const rows=data.rows.filter(r=>(r.scene ?? MISSING)===scene)
  const bars=rows.map(r=>({predictor:r.predictor,ADE:statOf(r.ade_m),FDE:statOf(r.fde_m)}))
  const profiles=rows.filter(r=>r.de_by_step_m?.length).map(r=>({name:r.predictor,steps:r.de_by_step_m}))
  if(!data.rows.length) return <EmptyState title="No predictions" message="This run stores no clean prediction rows."/>
  return <div>
    {withNotice&&<RunNotice experiment={data.experiment}/>}
    <PopulationNote label={data.selection?.population_label} aggregation={data.selection?.aggregation}/>
    <div className="mb-5 grid gap-3 border border-white/[.08] bg-[#0b1017] p-4 sm:grid-cols-2 lg:grid-cols-3">{data.selection&&setQuery&&<SelectionFields selection={data.selection} query={query ?? {}} onChange={setQuery}/>}<ChoiceField label="Scene" value={scene} options={scenes} onChange={setScene}/></div>
    <div className="grid gap-5 xl:grid-cols-2">
      <Panel title="Predictor comparison" eyebrow={`${scene} · metres`}><GroupedBars rows={bars} category="predictor" bars={['ADE','FDE']} yLabel="Error (m)" palette={adeFdePalette} testId="prediction-chart"/></Panel>
      <Panel title="Error by horizon step" eyebrow={`${scene} · requested severity 0`}>{profiles.length?<HorizonChart rows={horizonRows(profiles)} names={profiles.map(p=>p.name)}/>:<EmptyState title="Not recorded" message="This run stores no horizon-wise error for this scene."/>}</Panel>
    </div>
    <Panel title="Scene results" eyebrow="Means over seeds within each scene" className="mt-5"><Table min={640} headers={<><Th>Scene</Th><Th>Predictor</Th><Th term="ADE">ADE (m)</Th><Th term="FDE">FDE (m)</Th><Th term="RMSE">RMSE (m)</Th><Th term="Coverage">Coverage</Th></>}>{data.rows.map(row=><tr key={`${row.scene}-${row.predictor}`} className="border-b border-white/[.05]"><Td strong>{row.scene ?? MISSING}</Td><Td>{row.predictor}</Td><Td mono>{formatStat(row.ade_m)}</Td><Td mono>{formatStat(row.fde_m)}</Td><Td mono>{formatStat(row.rmse_m)}</Td><Td mono>{formatPercent(statOf(row.prediction_coverage))}</Td></tr>)}</Table></Panel>
    {data.selection&&(data.across_scenes.length?<Panel title="Across scenes" eyebrow="Stored mean of scene means" className="mt-5"><Table headers={<><Th>Predictor</Th><Th term="ADE">ADE (m)</Th><Th term="FDE">FDE (m)</Th><Th term="RMSE">RMSE (m)</Th><Th term="Coverage">Coverage</Th></>}>{data.across_scenes.map(row=><tr key={row.predictor} className="border-b border-white/[.05]"><Td strong>{row.predictor}</Td><Td mono>{formatStat(row.ade_m)}</Td><Td mono>{formatStat(row.fde_m)}</Td><Td mono>{formatStat(row.rmse_m)}</Td><Td mono>{formatPercent(statOf(row.prediction_coverage))}</Td></tr>)}</Table></Panel>:<p className="mt-5 text-xs text-slate-500">No across-scene value is stored for this run, so scenes are shown one at a time.</p>)}
  </div>
}

// One short line. Results must not read as research findings, but the full stored label belongs in Records.
export function EvidenceLine({level,exploratory}:{level:EvidenceLevel|null;exploratory?:string|null}){return <div data-testid="evidence-line" className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500"><EvidenceBadge level={level}/>{exploratory&&<span className="text-amber-200/80">{exploratory}</span>}</div>}

const DATASET_ORDER=['EWAP','UCY','MOT17','Synthetic'] as const
const DATASET_LABEL:Record<string,string>={EWAP:'ETH / HOTEL',UCY:'UCY',MOT17:'MOT17',Synthetic:'Synthetic'}
const NO_CHOICE=''
// Each dataset has its own list. A list not in use shows no scene, so picking any of its scenes always registers.
export function ScenePicker({scenes,value,onChange}:{scenes:WorkspaceScene[];value:string;onChange:(scene:WorkspaceScene)=>void}){
  return <div data-testid="scene-picker" className="mb-4 grid grid-cols-2 gap-x-4 gap-y-2 xl:grid-cols-4">{DATASET_ORDER.map(dataset=>{const group=scenes.filter(s=>s.dataset===dataset); if(!group.length) return null; const active=group.some(s=>s.id===value); return <label key={dataset} className="flex min-w-0 flex-col gap-1 xl:flex-row xl:items-center xl:gap-3"><span className="shrink-0 font-mono text-[10px] uppercase tracking-wider text-slate-500">{DATASET_LABEL[dataset]}</span><Select label={DATASET_LABEL[dataset]} active={active} value={active?value:NO_CHOICE} onChange={e=>{const next=group.find(s=>s.id===e.target.value); if(next) onChange(next)}}>{!active&&<option value={NO_CHOICE} disabled hidden>Choose scene</option>}{group.map(scene=><option key={scene.id} value={scene.id} disabled={!scene.has_video}>{scene.name}</option>)}</Select></label>})}</div>
}

export function ExperimentsPage(){
  const {resource,retry}=useResource(signal=>listExperiments({signal}),'experiments')
  return <AppShell><PageHeader eyebrow="Experiment registry" title="Experiment history" description="Every run folder discovered under runs/, with its status and integrity as recorded."/><ResourceView resource={resource} retry={retry} loading="Discovering experiments" empty={{title:'No experiments',message:'No run folders were found.'}}>{page=>page.items.length===0?<EmptyState title="No experiments" message="The results folder holds no recognised runs yet."/>:<ExperimentBrowser items={page.items}/>}</ResourceView></AppShell>
}
const ALL='All'
function ExperimentBrowser({items}:{items:Experiment[]}){
  const [kind,setKind]=useState(ALL); const [status,setStatus]=useState(ALL); const [dataset,setDataset]=useState(ALL)
  const kinds=[ALL,...uniqueValues(items.map(r=>r.kind_label))]; const statuses=[ALL,...uniqueValues(items.map(r=>r.status))]; const datasets=[ALL,...uniqueValues(items.flatMap(r=>r.datasets).filter(Boolean) as string[])]
  const shown=items.filter(r=>(kind===ALL||r.kind_label===kind)&&(status===ALL||r.status===status)&&(dataset===ALL||r.datasets.includes(dataset)))
  return <div>
    <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]"><Panel title="Filters" eyebrow={`${shown.length} of ${items.length} runs`}><div className="grid gap-3 sm:grid-cols-3"><ChoiceField label="Kind" value={kind} options={kinds} onChange={setKind}/><ChoiceField label="Status" value={status} options={statuses} onChange={setStatus}/><ChoiceField label="Dataset" value={dataset} options={datasets} onChange={setDataset}/></div></Panel><Panel title="Experiment history" eyebrow="Runs recorded per day (UTC)"><GroupedBars rows={historyByDay(items.map(r=>r.timestamp))} category="day" bars={['runs']} yLabel="Runs" testId="history-chart"/></Panel></div>
    <Panel title="Stored experiments" eyebrow="Discovered from run folders" className="mt-5">{shown.length?<Table min={1020} headers={<><Th>Experiment ID</Th><Th>Recorded</Th><Th>Kind</Th><Th>Dataset</Th><Th>Predictors</Th><Th>Modes</Th><Th>Evidence</Th><Th>Integrity</Th><Th>Status</Th><Th> </Th></>}>{shown.map(row=><tr key={row.experiment_id} className="border-b border-white/[.05] align-top"><Td mono><span className="text-cyan-300">{row.experiment_id}</span></Td><Td>{formatDate(row.timestamp)}</Td><Td>{row.kind_label}</Td><Td>{row.datasets.join(', ')||MISSING}</Td><Td>{row.predictors.join(', ')||MISSING}</Td><Td>{row.conditions.join(', ')||MISSING}</Td><Td><EvidenceBadge level={row.evidence_level}/></Td><Td>{row.integrity.status.replaceAll('_',' ')}</Td><Td><RunStatusBadge experiment={row}/>{row.status_reason&&<div className="mt-1 max-w-52 text-[11px] text-red-200/70">{row.status_reason}</div>}</Td><td className="py-3"><Link href={`/experiments/${encodeURIComponent(row.experiment_id)}`} className="text-cyan-300 hover:underline">Open</Link></td></tr>)}</Table>:<EmptyState title="No matches" message="No run matches these filters."/>}</Panel>
  </div>
}

export function ExperimentDetailPage({id}:{id:string}){
  const {resource,retry}=useResource(signal=>getExperiment(id,{signal}),id)
  return <AppShell><PageHeader eyebrow="Experiment detail" title={id} description="Metadata, provenance, artifacts and recorded results for this stored run."/><div className="mb-5"><Link href="/experiments" className="inline-flex items-center gap-2 text-xs text-slate-500 hover:text-white"><ArrowLeft className="size-3.5" /> All experiments</Link></div><ResourceView resource={resource} retry={retry} loading="Loading this experiment">{detail=><ExperimentDetailBody detail={detail}/>}</ResourceView></AppShell>
}
function ExperimentDetailBody({detail}:{detail:ExperimentDetail}){
  const {experiment:run,views,artifacts,stored_summary:stored}=detail
    const p=run.provenance
  const charts=artifacts.filter(a=>a.type==='chart'); const records=artifacts.filter(a=>a.type==='record')
  return <div>
    <RunNotice experiment={run}/>
    {run.status!=='complete'&&<div role="alert" className="mb-5 border border-red-400/20 bg-red-400/[.04] p-4 text-xs text-red-100/80">This run is {run.status}: {run.status_reason}. Its results are not served.</div>}
    <div className="grid gap-5 xl:grid-cols-2">
      <Panel title="Configuration" eyebrow="As recorded"><KeyValues rows={[['Kind',run.kind_label],['Record ID',run.record_id ?? MISSING],['Scope',run.scope ?? MISSING],['Experiments',run.experiments.join(', ')||MISSING],['Datasets',run.datasets.join(', ')||MISSING],['Scenes',run.scenes.join(', ')||MISSING],['Predictors',run.predictors.join(', ')||MISSING],['Error modes',run.conditions.join(', ')||MISSING],['History lengths',run.histories.join(', ')||MISSING],['Horizon',run.horizon_len ?? MISSING],['Seeds',run.seeds.join(', ')||MISSING],['Detector',run.detector ?? MISSING],['Tracker',run.tracker ?? MISSING],['Recorded',formatDate(run.timestamp)],['Status',<RunStatusBadge key="s" experiment={run}/>],['Newest in its series',run.latest_in_series?'Yes':'No']]}/></Panel>
      <Panel title="Provenance" eyebrow="Hashes and association"><KeyValues rows={[['Git commit',<span key="g" className="font-mono">{formatHash(p.git?.sha)}{p.git?.dirty?' (uncommitted changes)':''}</span>],['Config hash',<span key="c" className="font-mono">{formatHash(p.config_hash)}</span>],['Source hash',<span key="u" className="font-mono">{formatHash(p.source_sha256)}</span>],['Settings hash',<span key="t" className="font-mono">{formatHash(p.settings_hash)}</span>],['Results hash',<span key="r" className="font-mono">{formatHash(p.results_sha256)}</span>],['Dataset manifest',<span key="d" className="font-mono">{formatHash(p.dataset_manifest_hash)}</span>],['Specification',p.specification?.version ?? MISSING],['Package version',p.trackvision_version ?? p.version ?? MISSING],['TrackEval commit',<span key="e" className="font-mono">{formatHash(p.trackeval_commit)}</span>],['Association mechanism',p.association_mechanism ?? MISSING],['Association status',p.association_status ?? MISSING],...(p.association_blocker?[['Association blocker',p.association_blocker] as [string,string]]:[]),...(p.association_sidecar?[['Sidecar bound to results',p.association_sidecar.bound_to_results?'Yes':'No'] as [string,string]]:[]),['Integrity',`${run.integrity.status.replaceAll('_',' ')} · ${run.integrity.verified} files`]]}/></Panel>
    </div>
    {run.counts&&<div className="mt-5"><MetricGrid items={[{label:'Finished cells',value:formatCount(run.counts.finished_cells),detail:`of ${formatCount(run.counts.expected_cells)} expected`},{label:'Failed',value:formatCount(run.counts.by_status?.failed)},{label:'Blocked records',value:formatCount(run.counts.by_status?.blocked)},{label:'Research complete',value:run.counts.research_complete===null?MISSING:run.counts.research_complete?'Yes':'No'}]}/></div>}
        {views.includes('tracking')&&<DetailTracking id={run.experiment_id}/>}
    {views.includes('prediction')&&<DetailPrediction id={run.experiment_id}/>}
    {stored&&<Panel title="Recorded values" eyebrow="Stored summary" className="mt-5"><pre data-testid="stored-summary" className="max-h-96 overflow-auto text-[11px] leading-5 text-slate-400">{JSON.stringify(stored,null,2)}</pre></Panel>}
    <Panel title="Artifacts" eyebrow={`${records.length} records · ${charts.length} figures`} className="mt-5">{artifacts.length?<div className="grid gap-5 lg:grid-cols-2"><div><div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-slate-500">Stored records (listed, not served)</div>{records.length?<KeyValues rows={records.map(a=>[a.name,`${formatCount(Math.ceil(a.size_bytes/1024))} KB`] as [string,string])}/>:<span className="text-xs text-slate-500">{MISSING}</span>}</div><div><div className="mb-2 font-mono text-[10px] uppercase tracking-wider text-slate-500">Figures</div>{charts.length?<div className="flex max-h-72 flex-col gap-1 overflow-auto">{charts.map(a=><a key={a.name} href={figureUrl(run.experiment_id,a.name)} target="_blank" rel="noreferrer" className="truncate font-mono text-[11px] text-cyan-300 hover:underline">{a.name}</a>)}</div>:<span className="text-xs text-slate-500">No figures</span>}</div></div>:<EmptyState title="No artifacts" message="This run holds no files."/>}</Panel>
  </div>
}
function DetailTracking({id}:{id:string}){const {resource,retry}=useResource(signal=>getTrackingResult(id,{signal}),`detail-tracking-${id}`); return <div className="mt-5"><ResourceView resource={resource} retry={retry} loading="Loading tracking results">{data=><TrackingDetail data={data} withNotice={false}/>}</ResourceView></div>}
function DetailPrediction({id}:{id:string}){const {resource,retry}=useResource(signal=>getPredictionResult(id,{},{signal}),`detail-prediction-${id}`); return <div className="mt-5"><ResourceView resource={resource} retry={retry} loading="Loading prediction results">{data=><PredictionDetail data={data} withNotice={false}/>}</ResourceView></div>}

export function SettingsPage(){
  const {backend,recheck}=useExperiment()
  const health=backend.phase==='connected'?backend.health:null
  return <AppShell><PageHeader eyebrow="Workspace settings" title="Settings" description="The live connection to the results backend. This release has no other settings."/><div className="grid gap-5 xl:grid-cols-2">
    <Panel title="Backend connection" eyebrow="Live status"><div className="mb-4"><Status label={backend.phase==='connected'?'Connected':backend.phase==='checking'?'Checking':'Disconnected'} kind={backend.phase==='connected'?'success':backend.phase==='checking'?'placeholder':'danger'}/></div>{backend.phase==='disconnected'&&<p className="mb-4 text-xs leading-5 text-red-100/80">{backend.message}</p>}<KeyValues rows={[['API base URL',<span key="u" className="font-mono">{appConfig.apiBaseUrl||'Not configured'}</span>],['Backend version',health?.version ?? MISSING],['Access',health?(health.read_only?'Read-only':'Writable'):MISSING],['Results directory',health?(health.results_directory_available?'Available':'Missing'):MISSING],['Experiment records',formatCount(health?.experiment_records)],['Invalid records',formatCount(health?.invalid_records)],['Backend time',formatDate(health?.timestamp)]]}/><button onClick={recheck} className="mt-4 flex items-center gap-2 border border-white/[.12] px-3 py-2 text-xs text-slate-300 hover:bg-white/[.05]"><RotateCcw className="size-3.5" /> Check again</button></Panel>
    <Panel title="Where these values come from" eyebrow="Configuration"><p className="text-xs leading-6 text-slate-400">The API base URL comes from <span className="font-mono text-slate-300">NEXT_PUBLIC_API_BASE_URL</span> when the frontend starts. The backend reads stored runs from <span className="font-mono text-slate-300">runs/</span>, or from <span className="font-mono text-slate-300">TRACKVISION_RUNS_DIR</span> if set. Models, trackers and experiment settings live in the research pipeline and its configuration, so they cannot be changed here.</p></Panel>
  </div></AppShell>
}

