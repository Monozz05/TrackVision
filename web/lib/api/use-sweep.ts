'use client'
import { useCallback, useEffect, useState } from 'react'
import { getSceneSweep } from '@/lib/api/services'
import type { SceneSweep } from '@/types'

const POLL_MS = 500

// One scene's sweep, followed until it finishes. The backend works in the background, so each answer holds the levels
// finished so far, and leaving the scene stops the asking, which lets the backend move on to the next one.
export function useSweep(sceneId: string) {
  const [sweep, setSweep] = useState<SceneSweep | null>(null)
  const [failure, setFailure] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    setSweep(null)
    setFailure(null)
    const poll = () => getSceneSweep(sceneId, { signal: controller.signal }).then((response) => {
      if (controller.signal.aborted) return
      if (response.data) {
        setSweep(response.data)
        if (response.data.state === 'calculating') timer = setTimeout(poll, POLL_MS)
      } else setFailure(response.message ?? 'The calculation could not start.')
    }).catch(() => { if (!controller.signal.aborted) setFailure('The request failed.') })
    void poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [sceneId, attempt])

  return { sweep, failure, retry: useCallback(() => setAttempt((n) => n + 1), []) }
}
