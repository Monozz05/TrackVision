'use client'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { appConfig } from '@/config/app'
import { getHealth } from '@/lib/api/services'
import { noErrors, type ErrorKind, type ErrorSettings, type Health, type InjectionResult } from '@/types'

const POLL_MS = 15_000
export type BackendState = { phase: 'checking' } | { phase: 'connected'; health: Health } | { phase: 'disconnected'; message: string; errorKind: ErrorKind }
type Store = {
  backend: BackendState
  recheck: () => void
  sceneId: string | null
  selectScene: (id: string, unitChanged?: boolean) => void
  errors: ErrorSettings
  setErrors: (next: ErrorSettings) => void
  injection: InjectionResult | null
  setInjection: (result: InjectionResult | null) => void
}
const ExperimentContext = createContext<Store | null>(null)

// What survives moving between Workspace and Analysis: the chosen scene, the sliders, the last injection's result and
// whether the backend answers. A result is dropped when the scene changes, so it can never appear under another scene.
export function ExperimentProvider({children}:{children:ReactNode}) {
  const [backend, setBackend] = useState<BackendState>({ phase: 'checking' })
  const [sceneId, setSceneId] = useState<string | null>(null)
  const [errors, setErrors] = useState<ErrorSettings>(noErrors)
  const [injection, setInjection] = useState<InjectionResult | null>(null)
  const latest = useRef(0)

  const recheck = useCallback(() => {
    const attempt = ++latest.current
    getHealth().then((response) => {
      if (attempt !== latest.current) return
      if (response.status === 'success' && response.data) setBackend({ phase: 'connected', health: response.data })
      else if (response.errorKind === 'not_found') setBackend({ phase: 'disconnected', message: `Something answered at ${appConfig.apiBaseUrl}, but it is not the TrackVision results API. Another application may be using that port.`, errorKind: 'not_found' })
      else setBackend({ phase: 'disconnected', message: response.message ?? 'The backend cannot be reached.', errorKind: response.errorKind ?? 'disconnected' })
    })
  }, [])

  useEffect(() => {
    recheck()
    const timer = setInterval(recheck, POLL_MS)
    return () => clearInterval(timer)
  }, [recheck])

  const selectScene = useCallback((id: string, unitChanged = false) => {
    setSceneId(id)
    setInjection((current) => (current?.scene === id ? current : null))
    // 0.2 m must never become 0.2 px, so position noise starts again in the new unit.
    if (unitChanged) setErrors((current) => ({ ...current, a: 0 }))
  }, [])

  const value = useMemo<Store>(() => ({ backend, recheck, sceneId, selectScene, errors, setErrors, injection, setInjection }), [backend, recheck, sceneId, selectScene, errors, injection])
  return <ExperimentContext.Provider value={value}>{children}</ExperimentContext.Provider>
}

export function useExperiment() {
  const context = useContext(ExperimentContext)
  if (!context) throw new Error('useExperiment must be used inside ExperimentProvider')
  return context
}
