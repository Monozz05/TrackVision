'use client'
import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, Pause, Play } from 'lucide-react'
import { drawOverlay, firstAnnotatedFrame, gateFor, peopleAt, type Layers } from '@/lib/overlay'
import { drawForecasts, forecastsAt, streetVideoUrl, type StreetForecast, type StreetVideo } from '@/lib/test-videos'
import type { CleanTracks, InjectionResult } from '@/types'

const SPEEDS = [0.25, 0.5, 1, 2]
export interface StreetLayers extends Layers { later: boolean }

export function TestVideoStage({video,clean,injection,forecast,layers,predictor,trailSeconds}:{video:StreetVideo;clean:CleanTracks|null;injection:InjectionResult|null;forecast:StreetForecast|null;layers:StreetLayers;predictor:string;trailSeconds:number}){
  const element=useRef<HTMLVideoElement>(null); const canvas=useRef<HTMLCanvasElement>(null)
  const [frame,setFrame]=useState(0); const [playing,setPlaying]=useState(false); const [speed,setSpeed]=useState(1)
  const fps=video.fps ?? 25; const total=video.frames ?? 0; const width=video.width ?? 1280; const height=video.height ?? 720
  const view={clean,others:null,injection,layers,predictor,trailFrames:Math.round(trailSeconds*fps),fps}
  const latest=useRef({view,forecast}); latest.current={view,forecast}
  const pixelsPerCssPixel=()=>width/(canvas.current?.clientWidth||width)

  const paint=useCallback((at:number)=>{
    const context=canvas.current?.getContext('2d')
    if(!context) return
    const scale=pixelsPerCssPixel()
    const {view:now,forecast:lines}=latest.current
    drawOverlay(context,width,height,at,{...now,layers:{...now.layers,predicted:false},pixelsPerCssPixel:scale})
    drawForecasts(context,at,{forecast:lines,predictor:now.predictor,predicted:now.layers.predicted,later:now.layers.later,pixelsPerCssPixel:scale})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[width,height])

  const sync=useCallback((mediaTime:number)=>{ const at=Math.round(mediaTime*fps); setFrame(at); paint(at) },[fps,paint])

  // requestVideoFrameCallback reports the frame actually on screen, so the overlay stays locked to the picture.
  useEffect(()=>{
    const player=element.current
    if(!player) return
    let handle=0; let stopped=false
    const onFrame=(_:number,meta:{mediaTime:number})=>{ if(stopped) return; sync(meta.mediaTime); handle=player.requestVideoFrameCallback(onFrame) }
    const onSeeked=()=>sync(player.currentTime)
    player.addEventListener('seeked',onSeeked)
    if('requestVideoFrameCallback' in player) handle=player.requestVideoFrameCallback(onFrame)
    return ()=>{ stopped=true; player.removeEventListener('seeked',onSeeked); if('cancelVideoFrameCallback' in player) player.cancelVideoFrameCallback(handle) }
  },[sync])

  useEffect(()=>{ paint(frame) },[paint,frame,clean,injection,forecast,layers,predictor,trailSeconds])
  useEffect(()=>{ if(element.current) element.current.playbackRate=speed },[speed])

  const seek=(to:number)=>{ const player=element.current; if(player) player.currentTime=(Math.max(0,Math.min(total-1,to))+0.1)/fps }
  useEffect(()=>{ const first=firstAnnotatedFrame(clean); if(first>0&&element.current&&element.current.currentTime===0) seek(first) // eslint-disable-next-line react-hooks/exhaustive-deps
  },[clean])

  const toggle=()=>{ const player=element.current; if(player) { if(player.paused) void player.play(); else player.pause() } }
  const step=forecast?.grid.frame_step ?? 1
  const people=clean?peopleAt(injection?injection.overlay.tracks:clean.tracks,frame,gateFor(clean)):null
  const lines=forecast?forecastsAt(forecast,predictor,frame).rows.length:null
  const stepButton='grid size-9 place-items-center border border-white/[.12] text-slate-300 hover:bg-white/[.05]'
  return <div data-testid="video-stage">
    <div className="relative mx-auto w-full overflow-hidden bg-black" style={{aspectRatio:`${width} / ${height}`,maxWidth:`calc((100vh - 270px) * ${width / height})`}}>
      <video key={video.id} ref={element} src={streetVideoUrl(video.id)} muted playsInline preload="auto" onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} className="absolute inset-0 size-full object-contain"/>
      <canvas ref={canvas} width={width} height={height} data-testid="overlay" className="pointer-events-none absolute inset-0 size-full"/>
    </div>
    <input aria-label="Video position" type="range" min={0} max={Math.max(0,total-1)} value={frame} onChange={e=>seek(Number(e.target.value))} className="mt-3 w-full accent-cyan-300"/>
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <button aria-label={playing?'Pause':'Play'} onClick={toggle} className="grid size-9 place-items-center bg-cyan-300 text-slate-950 hover:bg-cyan-200">{playing?<Pause className="size-4"/>:<Play className="size-4"/>}</button>
      <button aria-label="Back one step" onClick={()=>seek(frame-step)} className={stepButton}><ChevronLeft className="size-4"/></button>
      <button aria-label="Forward one step" onClick={()=>seek(frame+step)} className={stepButton}><ChevronRight className="size-4"/></button>
      <div className="ml-1 flex border border-white/[.12]">{SPEEDS.map(value=><button key={value} onClick={()=>setSpeed(value)} className={`px-2 py-1.5 text-[11px] ${speed===value?'bg-cyan-400/10 text-cyan-300':'text-slate-500 hover:text-slate-300'}`}>{value}×</button>)}</div>
      {people!==null&&<span data-testid="people-count" className="ml-auto text-[11px] text-slate-400">{people} {people===1?'person':'people'} tracked{lines!==null&&layers.predicted?`, ${lines} with a forecast`:''}</span>}
      <span data-testid="frame-readout" className={`${people!==null?'ml-3':'ml-auto'} font-mono text-[11px] text-slate-500`}>frame {frame} / {total}</span>
    </div>
  </div>
}
