import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      // Forward API calls to a FastAPI backend if added later
      '/api': 'http://localhost:8000',
    },
  },
})
