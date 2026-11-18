'use client'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play, SkipBack, SkipForward } from 'lucide-react'
import { videoUrl } from '@/lib/api/services'
import { activeSwaps, drawOverlay, firstAnnotatedFrame, gateFor, othersAt, peopleAt, type Layers } from '@/lib/overlay'
import type { CleanTracks, InjectionResult, VideoStatus } from '@/types'

const SPEEDS = [0.25, 0.5, 1, 2]
const eventColour = { switch: '#f472b6', split: '#f59e0b' } as const

export function VideoStage({sceneId,status,clean,others,injection,layers,predictor,trailSeconds}:{sceneId:string;status:VideoStatus;clean:CleanTracks|null;others:CleanTracks|null;injection:InjectionResult|null;layers:Layers;predictor:string;trailSeconds:number}){
  const video=useRef<HTMLVideoElement>(null); const canvas=useRef<HTMLCanvasElement>(null)
  const [frame,setFrame]=useState(0); const [playing,setPlaying]=useState(false); const [speed,setSpeed]=useState(1)
  const fps=status.fps ?? 25; const total=status.frames ?? 0; const width=status.width ?? 720; const height=status.height ?? 576
  const view={clean,others,injection,layers,predictor,trailFrames:Math.round(trailSeconds*fps),fps}
  const latest=useRef(view); latest.current=view
  const pixelsPerCssPixel=()=>width/(canvas.current?.clientWidth||width)

  const paint=useCallback((at:number)=>{
    const context=canvas.current?.getContext('2d')
    if(context) drawOverlay(context,width,height,at,{...latest.current,pixelsPerCssPixel:pixelsPerCssPixel()})
  },[width,height])

  const sync=useCallback((mediaTime:number)=>{ const at=Math.round(mediaTime*fps); setFrame(at); paint(at) },[fps,paint])

  // requestVideoFrameCallback reports the frame actually on screen, so the overlay stays locked to the picture.
  useEffect(()=>{
    const element=video.current
    if(!element) return
    let handle=0; let stopped=false
    const onFrame=(_:number,meta:{mediaTime:number})=>{ if(stopped) return; sync(meta.mediaTime); handle=element.requestVideoFrameCallback(onFrame) }
    const onSeeked=()=>sync(element.currentTime)
    element.addEventListener('seeked',onSeeked)
    if('requestVideoFrameCallback' in element) handle=element.requestVideoFrameCallback(onFrame)
    return ()=>{ stopped=true; element.removeEventListener('seeked',onSeeked); if('cancelVideoFrameCallback' in element) element.cancelVideoFrameCallback(handle) }
  },[sync])

  useEffect(()=>{ paint(frame) },[paint,frame,clean,others,injection,layers,predictor,trailSeconds])
  useEffect(()=>{ if(video.current) video.current.playbackRate=speed },[speed])

  const seek=(to:number)=>{ const element=video.current; if(element) element.currentTime=Math.max(0,Math.min(total-1,to))/fps }
  // Open on the first frame with anyone on it, so a scene never starts on an empty picture.
  useEffect(()=>{ const first=firstAnnotatedFrame(clean); if(first>0&&video.current&&video.current.currentTime===0) seek(first) // eslint-disable-next-line react-hooks/exhaustive-deps
  },[clean])

  const toggle=()=>{ const element=video.current; if(element) { if(element.paused) void element.play(); else element.pause() } }
  const switches=(injection?.overlay.events ?? []).filter(event=>event.kind==='switch')
  const jump=(to:number|undefined)=>{ if(to===undefined) return; video.current?.pause(); seek(to) }
  const previous=[...switches].reverse().find(event=>event.frame<frame)
  const next=switches.find(event=>event.frame>frame)
  const swaps=injection?activeSwaps(injection.overlay.events,frame,fps):[]
  const step=clean?.frame_step ?? 1
  const people=clean?peopleAt(injection?injection.overlay.tracks:clean.tracks,frame,gateFor(clean))+othersAt(others,clean,frame).length:null
  const stepButton='grid size-9 place-items-center border border-white/[.12] text-slate-300 hover:bg-white/[.05]'
  return <div data-testid="video-stage">
    <div className="relative mx-auto w-full overflow-hidden bg-black" style={{aspectRatio:`${width} / ${height}`,maxWidth:`calc((100vh - 270px) * ${width / height})`}}>
      <video key={sceneId} ref={video} src={videoUrl(sceneId)} muted playsInline preload="auto" onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onLoadedMetadata={()=>{const first=firstAnnotatedFrame(clean); if(first>0) seek(first)}} className="absolute inset-0 size-full object-contain"/>
      <canvas ref={canvas} width={width} height={height} data-testid="overlay" className="pointer-events-none absolute inset-0 size-full"/>
      {swaps.length>0&&<div data-testid="swap-note" className="absolute left-2 top-2 bg-white px-2 py-1 font-mono text-[11px] font-semibold text-slate-950">{swaps.map(event=>`IDs ${event.a} and ${event.b} swapped`).join(' · ')}</div>}
    </div>
    <div className="relative mt-3">
      <input aria-label="Video position" type="range" min={0} max={Math.max(0,total-1)} value={frame} onChange={e=>seek(Number(e.target.value))} className="w-full accent-cyan-300"/>
      {injection&&total>0&&<div className="pointer-events-none absolute inset-x-1 top-[18px] h-1.5">{injection.overlay.events.map(event=><span key={`${event.kind}-${event.frame}-${event.a}`} title={event.kind} className="absolute h-1.5 w-px" style={{left:`${event.frame/total*100}%`,background:eventColour[event.kind]}}/>)}</div>}
    </div>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <button aria-label={playing?'Pause':'Play'} onClick={toggle} className="grid size-9 place-items-center bg-cyan-300 text-slate-950 hover:bg-cyan-200">{playing?<Pause className="size-4"/>:<Play className="size-4"/>}</button>
      <button aria-label="Back one step" onClick={()=>seek(frame-step)} className={stepButton}><ChevronLeft className="size-4"/></button>
      <button aria-label="Forward one step" onClick={()=>seek(frame+step)} className={stepButton}><ChevronRight className="size-4"/></button>
      <div className="ml-1 flex border border-white/[.12]">{SPEEDS.map(value=><button key={value} onClick={()=>setSpeed(value)} className={`px-2 py-1.5 text-[11px] ${speed===value?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{value}×</button>)}</div>
      {switches.length>0&&<div className="ml-1 flex items-center gap-1"><button aria-label="Previous ID switch" disabled={!previous} onClick={()=>jump(previous?.frame)} className={`${stepButton} disabled:opacity-30`}><SkipBack className="size-3.5"/></button><span data-testid="switch-count" className="px-1 text-[11px] text-slate-400">{switches.length} ID {switches.length===1?'switch':'switches'}</span><button aria-label="Next ID switch" disabled={!next} onClick={()=>jump(next?.frame)} className={`${stepButton} disabled:opacity-30`}><SkipForward className="size-3.5"/></button></div>}
      {people!==null&&<span data-testid="people-count" className="ml-auto text-[11px] text-slate-400">{people} {people===1?'person':'people'} tracked</span>}
      <span data-testid="frame-readout" className={`${people!==null?'ml-3':'ml-auto'} font-mono text-[11px] text-slate-500`}>frame {frame} / {total}</span>
    </div>
  </div>
}
