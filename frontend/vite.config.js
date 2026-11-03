import react from '@vitejs/plugin-react'
import legacy from '@vitejs/plugin-legacy'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Vite's default build output is an ES module script
    // (<script type="module">). Chrome (and other browsers) refuse to load a
    // module script from a file:// page at all, as a security restriction -
    // unrelated to and not fixed by the base:'./' relative-path setting below.
    // This plugin additionally builds a plain, non-module ("nomodule") script,
    // which Chrome does allow to run from a local file, and wires up index.html
    // to use it. This is the actual fix for double-clicking dist/index.html;
    // base:'./' alone was necessary but not sufficient.
    // renderModernChunks: false skips the modern/legacy auto-detection entirely
    // and emits ONLY the plain classic script. The auto-detection (default
    // behaviour) tests JS *feature* support, which Chrome has - it would
    // conclude "modern browser, use the ES module bundle" and then that bundle
    // would still fail to load from file://, since the real problem isn't a
    // missing feature, it's Chrome's CORS-style restriction on loading module
    // scripts specifically from the file:// protocol.
    legacy({ targets: ['defaults', 'not IE 11'], renderModernChunks: false }),
  ],
  // base: './' makes the production build (npm run build -> dist/index.html)
  // reference its JS/CSS with relative paths ("./assets/...") instead of
  // server-absolute ones ("/assets/..."). Relative paths resolve correctly when
  // dist/index.html is opened directly by double-clicking it (file://), which is
  // the whole point - the default absolute paths only resolve when a dev/prod
  // server is serving the folder, which is why double-clicking index.html during
  // `npm run dev` never worked. The backend API (api.py) still needs to be
  // running separately for the page to load real data.
  base: './',
})
