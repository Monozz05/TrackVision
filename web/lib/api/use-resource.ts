'use client'
import { useCallback, useEffect, useState } from 'react'
import type { ApiResponse, ErrorKind } from '@/types'

export type Resource<T> = { phase: 'loading' } | { phase: 'ready'; data: T } | { phase: 'empty'; message?: string } | { phase: 'error'; message: string; errorKind: ErrorKind }

// Loads one API resource and tracks its state. A response for an older selection is discarded, so a screen can never
// show one run's numbers under another run's name. `key` must change whenever `load` would ask for something else.
export function useResource<T>(load: (signal: AbortSignal) => Promise<ApiResponse<T>>, key: string, enabled = true) {
  const [resource, setResource] = useState<Resource<T>>({ phase: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    if (!enabled) return
    const controller = new AbortController()
    setResource({ phase: 'loading' })
    load(controller.signal).then((response) => {
      if (controller.signal.aborted) return
      if (response.status === 'success' && response.data !== null) setResource({ phase: 'ready', data: response.data })
      else if (response.status === 'error') setResource({ phase: 'error', message: response.message ?? 'The request failed.', errorKind: response.errorKind ?? 'server' })
      else setResource({ phase: 'empty', message: response.message })
    }).catch(() => {
      if (!controller.signal.aborted) setResource({ phase: 'error', message: 'The request failed.', errorKind: 'server' })
    })
    return () => controller.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` stands for everything `load` depends on
  }, [key, attempt, enabled])

  return { resource, retry: useCallback(() => setAttempt((n) => n + 1), []) }
}
