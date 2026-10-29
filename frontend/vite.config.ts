import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  // Plotly's source modules (src/plotly.ts builds from them rather than
  // the prebuilt bundle) refer to Node's `global`; without this the first
  // chart threw "global is not defined" and unmounted the whole app.
  define: {
    global: 'globalThis',
  },
  build: {
    // Plotly is the largest chunk (~1.5 MB, see src/plotly.ts); a build over
    // this means a full Plotly bundle has crept back in.
    chunkSizeWarningLimit: 1600,
  },
})
