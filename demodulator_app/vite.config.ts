import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig(({ command }) => ({
  // Dev server serves all of data/ (wav, json, mp3s for quick iteration).
  // Production build copies nothing from publicDir; the build script does
  // a targeted copy of only wav + json so MP3 source files stay out of dist.
  publicDir: command === 'serve' ? '../data' : false,
  base: command === 'build' ? '/cadillac-radio/' : '/',
  worker: { format: 'es' },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        lab:  resolve(__dirname, 'back-to-the-future.html'),
      },
    },
  },
}))
