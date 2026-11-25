// Profiles the main thread while the map is dragged onto new tiles: serves the app, opens ?bench (see
// src/mapWorld/MapWorldBench.ts) with every drag the same way, and records a V8 CPU profile of the drags and the
// settles, the long animation frames and the intervals between the frames (at vsync, not uncapped).
//
//   node video/dragProfile.mjs [--city Amsterdam] [--rounds 4] [--dragFrames 90] [--dragSpeed .03] [--out file.json]
//
// Prints the frame intervals, the functions with the most self time and the long frames, and writes the profile
// (.cpuprofile, opens in the Performance panel of Chrome's devtools) to video/out/.
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
const city = option('city', 'Amsterdam');
const rounds = option('rounds', '4');
const dragFrames = option('dragFrames', '90');
const dragSpeed = option('dragSpeed', '.03');
const extra = option('params', '');
// also the load of the city at the start
const fromStart = args.includes('--fromStart');
const outDir = path.join(root, 'video', 'out');
fs.mkdirSync(outDir, { recursive: true });
const out = option('out', path.join(outDir, 'dragProfile.cpuprofile'));

const server = await createServer({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  cacheDir: 'node_modules/.vite-video',
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  server: { host: 'localhost', port: 5198 },
});
await server.listen();

const browser = await puppeteer.launch({
  channel: 'chrome',
  headless: false,
  defaultViewport: null,
  protocolTimeout: 3600_000,
  args: [
    '--window-size=1280,800',
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
  ],
});
try {
  const page = (await browser.pages())[0] ?? await browser.newPage();
  page.on('pageerror', e => console.error('[page]', e.message));
  page.on('console', m => {
    const text = m.text();
    if (m.type() === 'error' && !text.includes('404')) console.error('[page]', text);
    if (text.startsWith('[perf]')) console.log(text);
  });
  // frame intervals and long animation frames, from before the app starts
  await page.evaluateOnNewDocument(() => {
    const perf = window.__perf = { frames: [], loaf: [], recording: window.location.search.includes('fromStart') };
    let last = 0;
    const tick = (t) => {
      if (perf.recording && last) perf.frames.push(t - last);
      last = t;
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    // the queue writes that block (> 5 ms) with the bytes written in the 200 ms before, and when pipelines are made
    perf.slowWrites = [];
    perf.pipelines = [];
    const writes = [];
    const writeBuffer = GPUQueue.prototype.writeBuffer;
    GPUQueue.prototype.writeBuffer = function (buffer, offset, data, dataOffset, size) {
      const t0 = performance.now();
      const result = writeBuffer.apply(this, arguments);
      const ms = performance.now() - t0;
      const bytes = size !== undefined ? size * (ArrayBuffer.isView(data) ? data.BYTES_PER_ELEMENT : 1) : (data.byteLength - (dataOffset ?? 0));
      writes.push([t0, bytes]);
      while (writes.length && writes[0][0] < t0 - 200) writes.shift();
      if (ms > 5) perf.slowWrites.push({ t: t0, ms, bytes, before: writes.reduce((s, w) => s + w[1], 0), label: buffer.label });
      return result;
    };
    // --pipelineCosts: the shader code of each module and the descriptor of each pipeline, compiled again one by one
    // after the load (see pipelineCosts below)
    perf.modules = new Map();
    perf.descriptors = [];
    const createShaderModule = GPUDevice.prototype.createShaderModule;
    GPUDevice.prototype.createShaderModule = function (descriptor) {
      const module = createShaderModule.apply(this, arguments);
      perf.modules.set(module, descriptor.code);
      return module;
    };
    for (const name of ['createRenderPipeline', 'createComputePipeline', 'createRenderPipelineAsync', 'createComputePipelineAsync']) {
      const create = GPUDevice.prototype[name];
      GPUDevice.prototype[name] = function (descriptor) {
        perf.descriptors.push({ compute: name.startsWith('createCompute'), descriptor });
        return create.apply(this, arguments);
      };
    }
    // --timePipelines: each sync pipeline is also made async, to see how long it takes to compile (+ms)
    const timePipelines = window.location.search.includes('timePipelines');
    for (const name of ['createRenderPipeline', 'createComputePipeline']) {
      const create = GPUDevice.prototype[name];
      const createAsync = GPUDevice.prototype[name + 'Async'];
      GPUDevice.prototype[name] = function () {
        perf.pipelines.push(performance.now());
        const entry = `${(performance.now() / 1000).toFixed(2)} ${arguments[0]?.label ?? arguments[0]?.vertex?.module?.label ?? arguments[0]?.compute?.module?.label ?? '?'}`;
        (perf.pipelineLabels ??= []).push(entry);
        if (timePipelines) {
          const index = perf.pipelineLabels.length - 1, t0 = performance.now();
          createAsync.apply(this, arguments).then(() => perf.pipelineLabels[index] = entry + ` (+${(performance.now() - t0).toFixed(0)} ms)`, () => {});
        }
        return create.apply(this, arguments);
      };
    }
    // async ones: when they are asked for, and when they are ready (+ms)
    const createAsync = GPUDevice.prototype.createRenderPipelineAsync;
    GPUDevice.prototype.createRenderPipelineAsync = function (descriptor) {
      const t0 = performance.now();
      const entry = `${(t0 / 1000).toFixed(2)} async ${descriptor?.label ?? '?'}`;
      (perf.pipelineLabels ??= []).push(entry);
      const index = perf.pipelineLabels.length - 1;
      return createAsync.apply(this, arguments).then((p) => {
        perf.pipelineLabels[index] = entry + ` (+${(performance.now() - t0).toFixed(0)} ms)`;
        return p;
      });
    };
    new PerformanceObserver((list) => {
      if (!perf.recording) return;
      for (const e of list.getEntries()) {
        perf.loaf.push({
          t: e.startTime, duration: e.duration, blocking: e.blockingDuration, render: e.renderStart ? e.startTime + e.duration - e.renderStart : 0,
          scripts: e.scripts.map(s => ({ d: s.duration, invoker: s.invoker, fn: s.sourceFunctionName, url: s.sourceURL.split('/').pop(), pos: s.sourceCharPosition })),
        });
      }
    }).observe({ type: 'long-animation-frame', buffered: false });
  });

  // --workerProfile: a CPU profile of each worker too, their self time summed
  const workerSessions = [];
  if (args.includes('--workerProfile')) {
    page.on('workercreated', async (worker) => {
      try {
        await worker.client.send('Profiler.enable');
        await worker.client.send('Profiler.setSamplingInterval', { interval: 500 });
        await worker.client.send('Profiler.start');
        workerSessions.push(worker.client);
      } catch (e) {
        console.error('worker profile:', e.message);
      }
    });
  }
  const client = await page.createCDPSession();
  await client.send('Profiler.enable');
  await client.send('Profiler.setSamplingInterval', { interval: 200 });

  const url = `http://localhost:5198/?bench=${rounds}&dragDir=same&dragFrames=${dragFrames}&dragSpeed=${dragSpeed}&city=${encodeURIComponent(city)}${extra ? '&' + extra : ''}`;
  console.log(url);
  if (fromStart) await client.send('Profiler.start');
  await page.goto(url + (fromStart ? '&fromStart' : '') + (args.includes('--timePipelines') ? '&timePipelines' : ''));
  // the city loaded and settled: the bench starts its first drag. From the start: as soon as the streamer is there
  await page.waitForFunction(() => window.__mapWorld?.tileStreamer &&
    window.__mapWorldBench.results.length >= (window.location.search.includes('fromStart') ? 0 : 1), { timeout: 600_000, polling: 20 });
  // the workers and the tile requests, sampled every 20 ms
  const startStats = await page.evaluate(() => {
    const w = window.__mapWorld;
    const s = window.__perf.streamer = { samples: 0, inFlight: 0, queued: 0, workersBusy: 0, poolQueued: 0, series: [] };
    window.__perf.timer = setInterval(() => {
      const st = w.tileStreamer, pool = w.workerPool;
      // every 500 ms: in flight / fetch queue / workers busy / results waiting to be applied
      if (s.samples % 25 === 0) s.series.push(`${st.inFlight.size}/${st.fetchQueue.length}/${pool.workers.length - pool.idle.length}/${st.resultQueue?.size ?? 0}`);
      s.samples++;
      s.inFlight += st.inFlight.size;
      s.queued += st.fetchQueue.length;
      s.workersBusy += pool.workers.length - pool.idle.length;
      s.poolQueued += pool.queue.length;
    }, 20);
    return { ...w.workerPool.stats, workers: w.workerPool.size, t: performance.now() };
  });
  await page.evaluate(() => window.__perf.recording = true);
  if (!fromStart) await client.send('Profiler.start');
  const startTime = Date.now();
  await page.waitForFunction(() => window.__mapWorldBench?.status === 'done', { timeout: 1200_000, polling: 200 });
  const { profile } = await client.send('Profiler.stop');
  // --screenshot file.png: the page after the last settle
  if (option('screenshot', '')) await page.screenshot({ path: option('screenshot', '') });
  // --pipelineCosts: every pipeline the app made, compiled again one at a time on a device of its own, with a comment
  // added to its code so no shader cache has it. Layout 'auto'. The time until each is ready, the slowest first
  if (args.includes('--pipelineCosts')) {
    const costs = await page.evaluate(async () => {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      const requiredLimits = {};
      for (const key in adapter.limits) requiredLimits[key] = adapter.limits[key];
      const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features], requiredLimits });
      const modules = new Map();
      const bust = String(Math.random());
      const remake = (stage) => {
        if (!stage) return stage;
        let m = modules.get(stage.module);
        if (!m) {
          m = device.createShaderModule({ code: window.__perf.modules.get(stage.module) + `\n// ${bust}\n` });
          modules.set(stage.module, m);
        }
        return { ...stage, module: m };
      };
      const results = [];
      // a copy: the pipelines made here are recorded too
      for (const { compute, descriptor: d } of window.__perf.descriptors.slice()) {
        const t0 = performance.now();
        try {
          if (compute) await device.createComputePipelineAsync({ ...d, layout: 'auto', compute: remake(d.compute) });
          else await device.createRenderPipelineAsync({ ...d, layout: 'auto', vertex: remake(d.vertex), fragment: remake(d.fragment) });
          results.push([performance.now() - t0, d.label]);
        } catch (e) {
          results.push([performance.now() - t0, d.label + ' (failed: ' + String(e.message).slice(0, 80) + ')']);
        }
      }
      device.destroy();
      return results.sort((a, b) => b[0] - a[0]);
    });
    console.log(`\npipeline compile times, one at a time (${costs.length}, ${costs.reduce((s, c) => s + c[0], 0).toFixed(0)} ms together):`);
    for (const [ms, label] of costs) console.log(`  ${ms.toFixed(0).padStart(6)} ms  ${label}`);
  }
  if (workerSessions.length) {
    const byFn = new Map();
    let total = 0;
    for (const session of workerSessions) {
      const { profile: wp } = await session.send('Profiler.stop');
      const wn = new Map(wp.nodes.map(n => [n.id, n]));
      for (let i = 0; i < wp.samples.length; i++) {
        const cf = wn.get(wp.samples[i]).callFrame;
        if (cf.functionName === '(idle)') continue;
        const key = `${cf.functionName || '(anonymous)'} ${cf.url.split('/').pop().split('?')[0]}:${cf.lineNumber + 1}`;
        byFn.set(key, (byFn.get(key) || 0) + (wp.timeDeltas[i] || 0));
        total += wp.timeDeltas[i] || 0;
      }
    }
    console.log(`workers busy ${(total / 1e6).toFixed(1)} s, self time:`);
    for (const [k, us] of [...byFn].sort((a, b) => b[1] - a[1]).slice(0, 30)) console.log(`  ${(us / 1000).toFixed(0).padStart(7)} ms  ${(us / total * 100).toFixed(1).padStart(5)}%  ${k}`);
  }
  const seconds = (Date.now() - startTime) / 1000;
  const endStats = await page.evaluate(() => {
    clearInterval(window.__perf.timer);
    const w = window.__mapWorld, g = w.geometry;
    const buffers = [...g.layers.map(l => [l.def.key, l.buffer]), ['ground', g.groundBuffer]]
      .map(([k, b]) => `${k} ${(b.usedBytes / 1048576).toFixed(1)}/${(b.capacityBytes / 1048576).toFixed(1)}`);
    return { ...w.workerPool.stats, t: performance.now(), streamer: window.__perf.streamer, buffers };
  });
  {
    const ms = endStats.t - startStats.t;
    const busy = endStats.busyMs - startStats.busyMs, jobs = endStats.jobs - startStats.jobs;
    const s = endStats.streamer, n = Math.max(s.samples, 1);
    console.log(`workers ${startStats.workers}: ${jobs} jobs, ${(busy / jobs).toFixed(0)} ms each, busy ${(busy / (ms * startStats.workers) * 100).toFixed(0)}% ` +
      `of the time, queue wait ${((endStats.queueMs - startStats.queueMs) / Math.max(jobs, 1)).toFixed(0)} ms/job; ` +
      `mean in flight ${(s.inFlight / n).toFixed(1)}, fetch queue ${(s.queued / n).toFixed(1)}, workers busy ${(s.workersBusy / n).toFixed(1)}, pool queue ${(s.poolQueued / n).toFixed(1)}`);
    console.log(`buffers MB used/capacity: ${endStats.buffers.join(', ')}`);
    console.log(`every 500 ms, in flight/fetch queue/workers busy/results waiting: ${s.series.join(' ')}`);
  }
  const perf = await page.evaluate(() => window.__perf);
  const bench = await page.evaluate(() => window.__mapWorldBench.results.map(r => ({ phase: r.phase, frames: r.bucket.frames })));
  fs.writeFileSync(out, JSON.stringify(profile));

  // --- frames
  const frames = perf.frames.slice().sort((a, b) => a - b);
  const pct = (p) => frames[Math.min(frames.length - 1, Math.floor(frames.length * p))]?.toFixed(1);
  const over = (ms) => frames.filter(f => f > ms).length;
  console.log(`\n${seconds.toFixed(1)} s, phases ${bench.map(b => `${b.phase}:${b.frames}`).join(' ')}`);
  console.log(`frames ${frames.length}: median ${pct(.5)} ms, p90 ${pct(.9)}, p99 ${pct(.99)}, max ${frames.at(-1)?.toFixed(1)}; ` +
    `> 20 ms: ${over(20)}, > 33 ms: ${over(33)}, > 50 ms: ${over(50)}, > 100 ms: ${over(100)}`);
  const loafTotal = perf.loaf.reduce((s, l) => s + l.duration, 0);
  console.log(`long animation frames: ${perf.loaf.length}, ${loafTotal.toFixed(0)} ms together, blocking ${perf.loaf.reduce((s, l) => s + l.blocking, 0).toFixed(0)} ms`);

  // --- profile: self time per function
  const nodes = new Map(profile.nodes.map(n => [n.id, n]));
  const self = new Map();
  const dt = profile.timeDeltas;
  for (let i = 0; i < profile.samples.length; i++) self.set(profile.samples[i], (self.get(profile.samples[i]) || 0) + (dt[i] || 0));
  const byFn = new Map();
  let total = 0;
  for (const [id, us] of self) {
    const cf = nodes.get(id).callFrame;
    const key = `${cf.functionName || '(anonymous)'} ${cf.url.split('/').pop().split('?')[0]}:${cf.lineNumber + 1}`;
    byFn.set(key, (byFn.get(key) || 0) + us);
    total += us;
  }
  const idle = [...byFn].filter(([k]) => k.startsWith('(idle)')).reduce((s, [, v]) => s + v, 0);
  console.log(`\nprofile ${(total / 1e6).toFixed(1)} s, busy ${((total - idle) / 1e6).toFixed(1)} s (${((total - idle) / total * 100).toFixed(0)}%)`);
  console.log('self time:');
  for (const [k, us] of [...byFn].sort((a, b) => b[1] - a[1]).slice(0, 40)) {
    console.log(`  ${(us / 1000).toFixed(0).padStart(7)} ms  ${k}`);
  }

  // inclusive time per function (each sample counted once per function on its stack)
  const parent = new Map();
  for (const n of profile.nodes) for (const c of n.children ?? []) parent.set(c, n.id);
  const incl = new Map();
  for (const [id, us] of self) {
    const seen = new Set();
    for (let n = id; n !== undefined; n = parent.get(n)) {
      const cf = nodes.get(n).callFrame;
      const key = `${cf.functionName || '(anonymous)'} ${cf.url.split('/').pop().split('?')[0]}:${cf.lineNumber + 1}`;
      if (seen.has(key)) continue;
      seen.add(key);
      incl.set(key, (incl.get(key) || 0) + us);
    }
  }
  console.log('inclusive time:');
  for (const [k, us] of [...incl].sort((a, b) => b[1] - a[1]).filter(([k]) => !k.startsWith('(root)')).slice(0, 50)) {
    console.log(`  ${(us / 1000).toFixed(0).padStart(7)} ms  ${k}`);
  }

  // the long stretches of the main thread without idle samples, and where their time went
  const keyOf = (id) => {
    const cf = nodes.get(id).callFrame;
    return `${cf.functionName || '(anonymous)'} ${cf.url.split('/').pop().split('?')[0]}:${cf.lineNumber + 1}`;
  };
  const segments = [];
  let seg = null, t = profile.startTime;
  for (let i = 0; i < profile.samples.length; i++) {
    t += dt[i] || 0;
    const id = profile.samples[i];
    const isIdle = nodes.get(id).callFrame.functionName === '(idle)';
    if (isIdle) {
      if (seg) { segments.push(seg); seg = null; }
      continue;
    }
    seg ??= { start: t, end: t, samples: [] };
    seg.end = t;
    seg.samples.push([id, dt[i + 1] || 0]);
  }
  const longSegments = segments.filter(s => s.end - s.start > 20000).sort((a, b) => (b.end - b.start) - (a.end - a.start));
  console.log(`\nbusy stretches > 20 ms: ${longSegments.length}`);
  for (const s of longSegments.slice(0, 10)) {
    const inc = new Map();
    for (const [id, us] of s.samples) {
      const seen = new Set();
      for (let n = id; n !== undefined; n = parent.get(n)) {
        const key = keyOf(n);
        if (seen.has(key) || key.startsWith('(root)')) continue;
        seen.add(key);
        inc.set(key, (inc.get(key) || 0) + us);
      }
    }
    console.log(`  ${((s.end - s.start) / 1000).toFixed(0)} ms at ${((s.start - profile.startTime) / 1e6).toFixed(2)} s: ` +
      [...inc].sort((a, b) => b[1] - a[1]).slice(0, 12).map(([k, us]) => `${(us / 1000).toFixed(0)} ${k}`).join(', '));
  }

  console.log(`\nslow queue writes: ${perf.slowWrites.length}`);
  for (const w of perf.slowWrites.sort((a, b) => b.ms - a.ms).slice(0, 10)) {
    const recent = perf.pipelines.filter(t => t < w.t && t > w.t - 3000).length;
    console.log(`  ${w.ms.toFixed(0)} ms at ${(w.t / 1000).toFixed(2)} s: ${(w.bytes / 1048576).toFixed(2)} MB to ${w.label}, ` +
      `${(w.before / 1048576).toFixed(1)} MB written in the 200 ms before, ${recent} pipelines made in the 3 s before`);
  }
  console.log(perf.pipelineLabels.join(', '));
  const p = perf.pipelines;
  console.log(`pipelines: ${p.length}, ` + [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20].map(s => `${p.filter(t => t < s * 1000).length} by ${s}s`).join(', '));

  console.log('\nlongest animation frames:');
  for (const l of perf.loaf.sort((a, b) => b.duration - a.duration).slice(0, 12)) {
    const scripts = l.scripts.sort((a, b) => b.d - a.d).slice(0, 4).map(s => `${s.d.toFixed(0)}ms ${s.invoker} ${s.fn}@${s.url}`).join(' | ');
    console.log(`  ${l.duration.toFixed(0)} ms at ${(l.t / 1000).toFixed(2)} s (render ${l.render.toFixed(0)}): ${scripts}`);
  }
  console.log(`\nprofile: ${out}`);
} finally {
  await browser.close();
  await server.close();
}
