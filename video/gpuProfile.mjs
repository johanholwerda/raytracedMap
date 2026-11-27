// The GPU time of the passes of a view: serves the app, opens a camera of the URL (?lat=&lng=&dist=&pitch=&yaw=&time=),
// waits until its tiles are in and the image converged (a screenshot of it), then animates the water, which draws the
// map every frame, and reads the timestamps of the passes (the Profiler of the renderer) for a number of frames. Per
// quality: the median of each pass and of the frame.
//
//   node video/gpuProfile.mjs [--root dir] [--label name] [--view "lat=..,lng=..,dist=..,pitch=..,yaw=..,time=morning"]
//     [--qualities "0,1"] [--frames 600] [--size 1920x1080] [--set "waterReflections:true,reflectionRays:0"]
//     [--rounds 1] [--out file.json] [--port 5199]
//
// --root: the source tree to serve (default this repo), for a baseline in a git worktree: put it in a folder of the repo
//   (video/out/...), the imports resolve upwards to its node_modules
// --qualities: the indices of the quality param, 0 very high, 1 high, 2 medium
// --set: params by their path from MapWorld, after the quality (which sets most of the others)
// --rounds: the qualities this many times over, in turns; the median is over all of their frames
// The screenshots of the converged view go to video/out/gpuProfile-<label>-q<quality>.png
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { createServer } from 'vite';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const root = path.resolve(option('root', repo));
const label = option('label', path.basename(root));
// the params of the view may be separated by commas too: an & on the command line is a separator of commands in cmd
const view = option('view', 'lat=22.369051&lng=114.100074&dist=233&pitch=-25.46&yaw=53.74&time=morning').replaceAll(',', '&');
const qualities = option('qualities', '0,1').split(',').map(Number);
const frames = parseInt(option('frames', '600'));
const [width, height] = option('size', '1920x1080').split('x').map(Number);
const set = option('set', '');
const rounds = parseInt(option('rounds', '1'));
const port = parseInt(option('port', '5199'));
const outDir = path.join(repo, 'video', 'out');
fs.mkdirSync(outDir, { recursive: true });
const outFile = option('out', path.join(outDir, `gpuProfile-${label}.json`));

const server = await createServer({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  cacheDir: path.join(root, 'node_modules/.vite-gpuProfile'),
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  server: { host: 'localhost', port, strictPort: true },
});
await server.listen();

const browser = await puppeteer.launch({
  channel: 'chrome',
  headless: false,
  defaultViewport: { width, height, deviceScaleFactor: 1 },
  protocolTimeout: 3600_000,
  args: [
    `--window-size=${width + 40},${height + 200}`,
    // timestamps at full resolution: Chrome quantizes them to 100 us otherwise
    '--enable-webgpu-developer-features', '--enable-dawn-features=allow_unsafe_apis', '--disable-dawn-features=timestamp_quantization',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  ],
});

const median = (values) => {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// the params of a list "path:value,..." on MapWorld, in the page
function setParams(list) {
  const w = window.__mapWorld;
  for (const entry of list.split(',').filter(Boolean)) {
    const [p, value] = entry.split(':');
    let target = w;
    for (const key of p.split('.')) target = target?.[key];
    if (!target?.setValue) {
      console.error(`[gpuProfile] no param ${p}`);
      continue;
    }
    target.setValue(value === 'true' ? true : value === 'false' ? false : parseFloat(value));
  }
}

// the 10th percentile: a pass that shares the GPU with another app (time slicing) times longer, its fastest frames least
const p10 = (values) => {
  if (!values.length) return NaN;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor((s.length - 1) * .1)];
};
const mean = (values) => values.reduce((s, v) => s + v, 0) / Math.max(values.length, 1);

const results = {};
try {
  const page = (await browser.pages())[0] ?? await browser.newPage();
  page.on('pageerror', (e) => console.error('[page]', e.message));
  page.on('console', (m) => {
    const text = m.text();
    if (m.type() === 'error' && !text.includes('404')) console.error('[page]', text);
    // --verbose: the warnings too (a shader that doesn't compile)
    else if (args.includes('--verbose') && m.type() === 'warn') console.warn('[page]', text.slice(0, 2000));
    if (text.startsWith('[gpuProfile]')) console.log(text);
  });
  // the async pipelines that are being compiled: a material draws with the one it had before until its new one is ready
  await page.evaluateOnNewDocument(() => {
    window.__pendingPipelines = 0;
    for (const name of ['createRenderPipelineAsync', 'createComputePipelineAsync']) {
      const create = GPUDevice.prototype[name];
      GPUDevice.prototype[name] = function () {
        window.__pendingPipelines++;
        const done = () => window.__pendingPipelines--;
        const promise = create.apply(this, arguments);
        promise.then(done, done);
        return promise;
      };
    }
  });
  // tileTimings: the app puts itself on window.__mapWorld. uncapped: the frames aren't tied to vsync, the GPU stays busy
  // and at its clocks (idle vsyncs let it clock down, the passes then time 2-6x longer)
  await page.goto(`http://localhost:${port}/?${view}&tileTimings&uncapped`);
  await page.waitForFunction(() => window.__mapWorld?.tileStreamer && window.__mapWorld.quality, { timeout: 120_000, polling: 200 });

  for (let round = 0; round < rounds; round++) {
    for (const quality of qualities) {
      await page.evaluate((q, list, setParamsSource) => {
        const w = window.__mapWorld;
        w.renderer.profiler.enabled = false;
        w.quality.setValue(q);
        // eslint-disable-next-line no-new-func
        new Function('list', `(${setParamsSource})(list)`)(list);
        w.animateWater.setValue(false);
      }, quality, set, setParams.toString());
      // the tiles of the view in, the lightmap and the TAA converged: the map stops being drawn
      await page.waitForFunction(() => {
        const w = window.__mapWorld;
        const s = (window.__converged ??= { count: 0 });
        s.count = w.tileStreamer.isIdle() && !w.skyLightmap.converging && !w.frameState.renderMap ? s.count + 1 : 0;
        return s.count > 30;
      }, { timeout: 300_000, polling: 'raf' });
      if (round === 0) {
        await page.screenshot({ path: path.join(outDir, `gpuProfile-${label}-q${quality}.png`) });
      }
      const sample = await page.evaluate(async (count) => {
        const w = window.__mapWorld;
        const frames = [];
        const callback = (total, labels, passMs) => frames.push({ total, labels: [...labels], passMs: [...passMs] });
        w.animateWater.setValue(true);
        w.renderer.profiler.enabled = true;
        // the pipelines of the quality compile and the clocks go up first: until no async pipeline is pending (a material
        // draws with its pipeline before until the new one is compiled, seconds for a cold shader cache) and 3 s
        const t0 = performance.now();
        await new Promise((resolve) => {
          let n = 0;
          const tick = () => (++n > 120 && performance.now() - t0 > 3000 && window.__pendingPipelines === 0 ? resolve() : requestAnimationFrame(tick));
          requestAnimationFrame(tick);
        });
        w.renderer.profiler.addFrameCallback(callback);
        await new Promise((resolve) => {
          const check = () => (frames.length >= count ? resolve() : requestAnimationFrame(check));
          requestAnimationFrame(check);
        });
        w.renderer.profiler.removeFrameCallback(callback);
        w.renderer.profiler.enabled = false;
        // its overlay goes away at once, not a second later: it would be in the next screenshot
        w.renderer.profiler.lastActiveTime = 0;
        w.animateWater.setValue(false);
        const canvas = w.renderer.canvas;
        return { frames, canvas: [canvas.width, canvas.height] };
      }, frames);
      const r = (results[`q${quality}`] ??= { canvas: sample.canvas, total: [], passes: {} });
      for (const f of sample.frames) {
        r.total.push(f.total);
        // a label that repeats in a frame: summed
        const sums = {};
        f.labels.forEach((l, i) => sums[l] = (sums[l] ?? 0) + f.passMs[i]);
        for (const [l, ms] of Object.entries(sums)) (r.passes[l] ??= []).push(ms);
      }
      console.log(`${label} q${quality} round ${round + 1}: frame median ${median(sample.frames.map((f) => f.total)).toFixed(3)} ms`);
    }
  }

  const summary = {};
  for (const [q, r] of Object.entries(results)) {
    const passes = Object.fromEntries(Object.entries(r.passes)
      .map(([l, ms]) => [l, { median: +median(ms).toFixed(4), p10: +p10(ms).toFixed(4), mean: +mean(ms).toFixed(4), frames: ms.length }])
      .sort((a, b) => b[1].median - a[1].median));
    summary[q] = { canvas: r.canvas, frames: r.total.length, totalMedian: +median(r.total).toFixed(4), totalMean: +mean(r.total).toFixed(4), totalP10: +p10(r.total).toFixed(4), passes };
    console.log(`\n${label} ${q} (${r.canvas.join('x')}, ${r.total.length} frames): frame median ${median(r.total).toFixed(3)} ms, mean ${mean(r.total).toFixed(3)}`);
    for (const [l, p] of Object.entries(passes)) {
      if (p.median >= .005) {
        console.log(`  ${l.padEnd(40)} ${p.median.toFixed(3)} ms (mean ${p.mean.toFixed(3)})${p.frames < r.total.length ? ` in ${p.frames} frames` : ''}`);
      }
    }
  }
  fs.writeFileSync(outFile, JSON.stringify({ label, root, view, set, summary }, null, 2));
  console.log(`\nwrote ${outFile}`);
} finally {
  await browser.close();
  await server.close();
}
