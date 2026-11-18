import { appConfig } from '@/config/app'
import type { ApiResponse, ErrorKind } from '@/types'

const DEFAULT_TIMEOUT_MS = 30_000

function failure<T>(errorKind: ErrorKind, message: string): ApiResponse<T> {
  return { data: null, status: 'error', errorKind, message }
}

// The one place the frontend talks to the network. A failed request becomes an error response, never data: there is
// nothing to fall back to.
export async function request<T>(path: string, options: { signal?: AbortSignal; timeoutMs?: number } = {}): Promise<ApiResponse<T>> {
  if (!appConfig.apiBaseUrl) return failure('not_configured', 'NEXT_PUBLIC_API_BASE_URL is not set.')
  const timeout = new AbortController()
  const timer = setTimeout(() => timeout.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS)
  const signal = options.signal ? AbortSignal.any([options.signal, timeout.signal]) : timeout.signal
  try {
    const response = await fetch(`${appConfig.apiBaseUrl}${path}`, { signal, cache: 'no-store' })
    if (response.ok) return { data: await response.json() as T, status: 'success' }
    const detail = await readDetail(response)
    if (response.status === 404) return failure('not_found', detail ?? 'The backend has no such record.')
    if (response.status >= 500) return failure('server', detail ?? 'The backend could not read this result.')
    return failure('rejected', detail ?? `The backend refused the request (${response.status}).`)
  } catch (error) {
    if (options.signal?.aborted) throw error
    const timedOut = timeout.signal.aborted
    return failure('disconnected', timedOut ? 'The backend did not respond in time.' : 'The backend cannot be reached. Start it, then retry.')
  } finally {
    clearTimeout(timer)
  }
}

async function readDetail(response: Response): Promise<string | null> {
  try {
    const body = await response.json() as { detail?: unknown }
    return typeof body.detail === 'string' ? body.detail : null
  } catch {
    return null
  }
}
