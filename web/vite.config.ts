import { defineConfig } from 'vite';
import { copyFileSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

export default defineConfig({
  build: { rollupOptions: { output: { manualChunks: { three: ['three', 'three/addons/controls/OrbitControls.js'] } } } },
  server: { proxy: { '/api': 'http://127.0.0.1:18765' } },
  plugins: [{
    name: 'local-openapi-assets',
    closeBundle() {
      mkdirSync('dist/docs-assets', { recursive: true });
      for (const file of ['swagger-ui-bundle.js', 'swagger-ui.css']) {
        copyFileSync(resolve('node_modules/swagger-ui-dist', file), resolve('dist/docs-assets', file));
      }
    },
  }],
});
