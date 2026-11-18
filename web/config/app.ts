export const appConfig = { name:'TrackVision', apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000/api' }

export const navItems = [
  { label:'Workspace', href:'/workspace' },
  { label:'Analysis', href:'/analysis' },
  { label:'Applications', href:'/applications' },
  { label:'Test Videos', href:'/test-videos' },
]
