import { cloneElement, type ReactElement } from 'react'
import { cleanup, configure } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

// Analysis renders about a dozen charts at once, which can take longer than the default second on a busy machine.
configure({ asyncUtilTimeout: 8000 })

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

vi.mock('next/navigation', () => ({ usePathname: () => '/overview' }))

// jsdom has no layout, so give charts a fixed size. Everything else in Recharts runs for real.
vi.mock('recharts', async (original) => ({
  ...await original<typeof import('recharts')>(),
  ResponsiveContainer: ({ children }: { children: ReactElement<{ width?: number; height?: number }> }) => cloneElement(children, { width: 640, height: 260 }),
}))
