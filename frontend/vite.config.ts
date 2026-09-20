import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.ROBOT_API_URL || 'http://127.0.0.1:8000'
  const routes = ['/direct', '/stop', '/resume', '/status', '/video', '/capabilities', '/voice', '/heartbeat', '/commands']
  return {
    plugins: [react()],
    server: { proxy: Object.fromEntries(routes.map(route => [route, { target, changeOrigin: true }])) },
  }
})
