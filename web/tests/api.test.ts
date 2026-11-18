import { describe, expect, it, vi } from 'vitest'
import { request } from '@/lib/api/client'
import { figureUrl, getCells, getExperiment, injectErrors, videoUrl } from '@/lib/api/services'
import { noErrors } from '@/types'
import { stubBackend, stubDisconnected } from './fixtures'

describe('request', () => {
  it('returns the body of a successful response', async () => {
    stubBackend({ '/health': { status: 'ok' } })
    expect(await request('/health')).toEqual({ data: { status: 'ok' }, status: 'success' })
  })

  it('reports an unreachable backend as disconnected, with no data', async () => {
    stubDisconnected()
    const response = await request('/health')
    expect(response).toMatchObject({ data: null, status: 'error', errorKind: 'disconnected' })
    expect(response.message).toContain('cannot be reached')
  })

  it('separates missing, refused and failed responses and keeps the backend message', async () => {
    stubBackend({ '/refused': { status: 409, body: { detail: 'fixture is invalid: results.json is not valid JSON' } }, '/failed': { status: 500, body: { detail: 'the server could not read this result' } } })
    expect(await request('/missing')).toMatchObject({ data: null, errorKind: 'not_found' })
    expect(await request('/refused')).toMatchObject({ data: null, errorKind: 'rejected', message: 'fixture is invalid: results.json is not valid JSON' })
    expect(await request('/failed')).toMatchObject({ data: null, errorKind: 'server' })
  })

  it('survives an error body that is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>bad gateway</html>', { status: 502 })))
    expect(await request('/x')).toMatchObject({ data: null, errorKind: 'server' })
  })

  it('treats a timeout as disconnected', async () => {
    vi.stubGlobal('fetch', vi.fn((_url: string, init: RequestInit) => new Promise((_, reject) => init.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError'))))))
    expect(await request('/slow', { timeoutMs: 5 })).toMatchObject({ data: null, errorKind: 'disconnected', message: 'The backend did not respond in time.' })
  })

  it('rethrows when the caller cancels, so a stale response is never shown', async () => {
    const controller = new AbortController()
    vi.stubGlobal('fetch', vi.fn(async () => { controller.abort(); throw new DOMException('aborted', 'AbortError') }))
    await expect(request('/x', { signal: controller.signal })).rejects.toThrow()
  })
})

describe('services', () => {
  it('encodes experiment IDs into the path', async () => {
    const calls = stubBackend({ '/experiments/run~a%20b': { experiment: {} } })
    await getExperiment('run~a b')
    expect(calls).toEqual(['/experiments/run~a%20b'])
    expect(figureUrl('r~1', 'a b.svg')).toBe('http://127.0.0.1:8000/api/experiments/r~1/artifacts/a%20b.svg')
  })

  it('sends only the filter values that are set', async () => {
    const calls = stubBackend({ '/experiments/x/cells': { items: [] } })
    await getCells('x', { scene: 'ETH', seed: undefined })
    expect(calls).toEqual(['/experiments/x/cells?scene=ETH'])
  })

  it('sends the error settings', async () => {
    const calls = stubBackend({ '/workspace/scenes/ETH/inject': {} })
    await injectErrors('ETH', { ...noErrors, a: 0.2, d: 0.05, reconstruct: true })
    expect(calls).toEqual(['/workspace/scenes/ETH/inject?a=0.2&b=0&c=0&d=0.05&reconstruct=true'])
  })

  it('builds the video address from the configured base URL', () => {
    expect(videoUrl('MOT17-02')).toBe('http://127.0.0.1:8000/api/workspace/scenes/MOT17-02/video')
  })
})
