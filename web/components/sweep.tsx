'use client'
import { ErrorState, LoadingState } from '@/components/trackvision-app'
import { useSweep } from '@/lib/api/use-sweep'
import { errorControls, type ErrorCode, type SceneSweep } from '@/types'

export const errorLabel = (code: ErrorCode) => errorControls.find((control) => control.code === code)?.label ?? code

function SweepProgress({sweep}:{sweep:SceneSweep}){
  if(sweep.state!=='calculating') return null
  const share=sweep.total?sweep.done/sweep.total:0
  return <div role="status" data-testid="sweep-progress" className="mb-3 border border-white/[.08] bg-[#0b1017] px-4 py-2.5"><div className="mb-1.5 text-[11px] text-slate-400">Calculating {sweep.done} of {sweep.total} error levels. The first view of a scene takes a moment, then it stays ready.</div><div className="h-1 bg-white/[.07]"><div className="h-1 bg-cyan-300 transition-[width]" style={{width:`${share*100}%`}}/></div></div>
}

// The sweep of one scene, with its progress, its failure and its retry in one place for both pages.
export function SweepView({sceneId,children}:{sceneId:string;children:(sweep:SceneSweep)=>React.ReactNode}){
  const {sweep,failure,retry}=useSweep(sceneId)
  if(failure) return <ErrorState message={failure} onRetry={retry}/>
  if(!sweep) return <LoadingState label="Starting the calculation"/>
  if(sweep.state==='failed') return <ErrorState message={sweep.reason ?? 'The calculation stopped.'} onRetry={retry}/>
  return <><SweepProgress sweep={sweep}/>{children(sweep)}</>
}
