import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      }
    }
  },
  build: {
    // TF.js + COCO-SSD are lazy-loaded via dynamic import() so they are
    // code-split into a separate chunk. The chunk is still large (~1.3 MB)
    // because TF is inherently big, so raise the warning limit to avoid noise.
    chunkSizeWarningLimit: 1600,
  },
})
