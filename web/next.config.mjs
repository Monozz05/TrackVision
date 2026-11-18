/** @type {import('next').NextConfig} */
const nextConfig = {
  agentRules: false,
  // Without this, `next dev` answers 403 for the app's scripts when it is opened at 127.0.0.1 and the page never loads.
  allowedDevOrigins: ['127.0.0.1'],
  devIndicators: { position: 'bottom-right' },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
