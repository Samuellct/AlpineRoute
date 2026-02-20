import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    // proxy dev: le front appelle /api/calculate, le backend ecoute sur /calculate, le rewrite enleve le prefix /api avant de transmettre a uvicorn
    // en prod (docker), c'est nginx qui fait la meme chose via le trailing slash dans proxy_pass http://backend:8000/
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
