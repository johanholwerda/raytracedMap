// The GPU time of the passes with and without the trees of the far tiles (the treeCullPixels param of MapWorld, see
// WorldTileStreamer.cullTrees): serves the app, opens ?bench (see src/mapWorld/debug/MapWorldBench.ts) at a view per
// variant, the variants in turns (other Chrome runs load the GPU too: compare the medians), and prints the mean GPU ms
// per frame of the drags per pass, and how many city meshes had their trees left out.
//
//   node video/treeCullBench.mjs [--view "lat=..&lng=..&dist=..&pitch=..&yaw=.."] [--variants 0,1,1e9] [--repeats 3]
//     [--rounds 4] [--dragFrames 60] [--dragSpeed .01] [--size 1280x800]
//
// A variant is a treeCullPixels value: 0 draws every tree, 1e9 none (the upper bound of what the cut can save).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createServer } from 'vite';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
// the Veluwe (wood), from the farthest the orbit camera goes
const view = option('view', 'lat=52.08&lng=5.83&dist=1800&pitch=-20&yaw=0');
const variants = option('variants', '0,1,1e9').split(',');
const repeats = parseInt(option('repeats', '3'));
const rounds = option('rounds', '4');
const dragFrames = option('dragFrames', '60');
const dragSpeed = option('dragSpeed', '.01');
const [width, height] = option('size', '1280x800').split('x').map(Number);
const port = parseInt(option('port', '5199'));
fs.mkdirSync(path.join(root, 'video', 'out'), { recursive: true });

const server = await createServer({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  cacheDir: 'node_modules/.vite-video',
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  server: { host: 'localhost', port },
});
await server.listen();

const browser = await puppeteer.launch({
  channel: 'chrome',
  headless: false,
  defaultViewport: null,
  protocolTimeout: 3600_000,
  args: [
    `--window-size=${width},${height}`,
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  ],
});

const median = (values) => {
  const s = [...values].sort((a, b) => a - b);
  return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

// per variant the runs: { gpuMs, passes, culled, meshes, canvas }
const runs = Object.fromEntries(variants.map(v => [v, []]));
try {
  for (let r = 0; r < repeats; r++) {
    for (const variant of variants) {
      const page = await browser.newPage();
      page.on('pageerror', e => console.error('[page]', e.message));
      page.on('console', m => {
        const text = m.text();
        if (m.type() === 'error' && !text.includes('404')) console.error('[page]', text);
      });
      // the meshes whose trees were left out, sampled while the drags and settles run (after the load). On every
      // document: vite reloads the page when it optimized new dependencies
      await page.evaluateOnNewDocument(() => {
        const s = window.__treeCull = { culled: 0, meshes: 0, samples: 0 };
        window.__treeCullTimer = setInterval(() => {
          const bench = window.__mapWorldBench;
          if (!(bench?.results?.length >= 1) || bench.status === 'done') return;
          const st = window.__mapWorld.tileStreamer;
          s.culled += st.culledTreeMeshes;
          s.meshes += st.cityMeshes.length;
          s.samples++;
        }, 100);
      });
      const url = `http://localhost:${port}/?bench=${rounds}&uncapped&dragFrames=${dragFrames}&dragSpeed=${dragSpeed}&${view}&set=treeCullPixels:${variant}`;
      await page.goto(url);
      await page.waitForFunction(() => window.__mapWorldBench?.status === 'done', { timeout: 1200_000, polling: 200 });
      const out = await page.evaluate(() => {
        clearInterval(window.__treeCullTimer);
        const s = window.__treeCull;
        const summary = window.__mapWorldBench.summary;
        return { canvas: summary.canvas, gpuMs: summary.drag.gpuMs, passes: summary.drag.passes, culled: s.culled / s.samples, meshes: s.meshes / s.samples };
      });
      runs[variant].push(out);
      // the settled image of the first run, to compare the look
      if (r === 0) await page.screenshot({ path: path.join(root, 'video', 'out', `treeCull-${variant}.png`) });
      console.log(`treeCullPixels ${variant} (run ${r + 1}): ${out.canvas}, drag GPU ${out.gpuMs.toFixed(2)} ms/frame (gBuffer ${(out.passes.mapGBuffer ?? 0).toFixed(3)}, Render Pass ${(out.passes['Render Pass'] ?? 0).toFixed(3)}), ` +
        `trees left out of ${(out.culled ?? NaN).toFixed(1)} of ${(out.meshes ?? NaN).toFixed(1)} city meshes`);
      await page.close();
    }
  }
} finally {
  await browser.close();
  await server.close();
}

// the medians per variant: the total and the passes that took more than .05 ms in some variant
console.log(`\n${view}, medians of ${repeats} runs, GPU ms per drag frame:`);
const labels = new Set();
for (const v of variants) for (const run of runs[v]) for (const [k, ms] of Object.entries(run.passes)) if (ms > .05) labels.add(k);
const rows = [['total', v => median(runs[v].map(r => r.gpuMs))],
  ...[...labels].map(k => [k, v => median(runs[v].map(r => r.passes[k] ?? 0))])];
console.log(['pass'.padEnd(28), ...variants.map(v => `px ${v}`.padStart(10))].join(''));
for (const [label, value] of rows) console.log([label.padEnd(28), ...variants.map(v => value(v).toFixed(3).padStart(10))].join(''));
