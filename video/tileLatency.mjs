// The time from the moment a map tile is wanted (in view and missing) to the moment it is drawn, split into its steps:
// builds the app as it is deployed (vite build), serves it, opens it with ?tileTimings (see TileTiming in
// src/mapWorld/map/WorldTileStreamer.ts) and runs, after the first load of a city, jumps to other cities, a drag of the
// map and a turn of the camera. Per scenario the median and the 90th percentile of each step over the tiles, for the
// ground and the buildings of a tile, and how long tiles were in view while missing ("seen missing": tiles asked for
// ahead of the view, see WorldTileStreamer.PREFETCH_MARGIN, are often there before they come into view). After the
// first load also when its view's tiles were in and when its image converged (the last frame that drew the map).
// A/B: --params "prefetchMargin=1" (none ahead of the view), "jobsPerWorker=3" (see WorldWorkerPool.capacity).
//
//   node video/tileLatency.mjs [--root dir] [--label name] [--city Amsterdam] [--runs 1] [--replay dir] [--latency ms]
//     [--dragSpeed 0.5] [--dragSeconds 4] [--turnSeconds 6] [--out file.json] [--nobuild] [--params "workers=4"] [--port 5189]
//     [--scenarios load,jump,drag,turn]
//
// --replay, --latency, --root: see loadTiming.mjs. --dragSpeed: camera distances per second.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';
import { build, preview } from 'vite';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const root = path.resolve(option('root', repo));
const label = option('label', path.basename(root));
const city = option('city', 'Amsterdam');
// Paris, New York, London
const jumps = option('jump', '48.853,2.3498;40.71139,-74.0125;51.50806,-0.12806').split(';').map((place) => place.split(',').map(Number));
const runs = parseInt(option('runs', '1'));
const replayDir = option('replay', '');
const latency = parseFloat(option('latency', '0'));
const dragSpeed = parseFloat(option('dragSpeed', '0.5'));
const dragSeconds = parseFloat(option('dragSeconds', '4'));
const turnSeconds = parseFloat(option('turnSeconds', '6'));
const port = parseInt(option('port', '5189'));
const outFile = option('out', '');
const extra = option('params', '');
const scenarios = option('scenarios', 'load,jump,drag,turn').split(',');
if (replayDir) fs.mkdirSync(replayDir, { recursive: true });

const outDir = path.join(repo, 'video', 'out', `dist-${label}`);
if (!args.includes('--nobuild') || !fs.existsSync(path.join(outDir, 'index.html'))) await build({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  build: { outDir, emptyOutDir: true },
});
console.log('built');
const server = await preview({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  build: { outDir },
  preview: { host: 'localhost', port, strictPort: true },
});

// --- the tile servers from disk (as loadTiming.mjs)
const TILE_HOSTS = ['tiles.openfreemap.org', 's3.amazonaws.com'];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const replayMemory = new Map();
async function replay(url) {
  let entry = replayMemory.get(url);
  if (entry) return entry;
  const file = path.join(replayDir, crypto.createHash('sha1').update(url).digest('hex'));
  if (fs.existsSync(file + '.json')) {
    entry = { ...JSON.parse(fs.readFileSync(file + '.json', 'utf8')), body: fs.readFileSync(file + '.bin') };
  } else {
    const response = await fetch(url);
    entry = { status: response.status, contentType: response.headers.get('content-type') ?? 'application/octet-stream', body: Buffer.from(await response.arrayBuffer()) };
    if (response.status < 500) {
      fs.writeFileSync(file + '.bin', entry.body);
      fs.writeFileSync(file + '.json', JSON.stringify({ status: entry.status, contentType: entry.contentType, url }));
    }
  }
  replayMemory.set(url, entry);
  return entry;
}

// --- the steps of a tile, from its TileTiming (ms since the epoch)
const STEPS = [
  // [name, from, to]: the step is to - from
  ['queued (main)', (t) => t.wanted, (t) => t.requested],
  ['to fetch worker', (t) => t.requested, (t) => t.times?.received],
  ['data (pbf & heights)', (t) => t.times?.received, (t) => t.times?.ready],
  ['  pbf', (t) => t.times?.received, (t) => t.times?.pbf],
  ['  heights', (t) => t.times?.received, (t) => t.times?.heights || undefined],
  ['wait for build worker', (t) => t.times?.ready, (t) => t.times?.buildStart],
  ['build (worker)', (t) => t.times?.buildStart, (t) => t.times?.buildEnd],
  ['to main thread', (t) => t.times?.buildEnd, (t) => t.arrived],
  ['wait for apply', (t) => t.arrived, (t) => t.applyStart],
  ['  for the frame', (t) => t.arrived, (t) => Math.max(t.applyFrame, t.arrived)],
  ['  frames held back', (t) => Math.max(t.applyFrame, t.arrived), (t) => t.applyStart],
  ['apply (main)', (t) => t.applyStart, (t) => t.applyEnd],
];
const GROUND_STEPS = [
  ...STEPS,
  ['rest of frame (main)', (t) => t.applyEnd, (t) => t.groundSubmitted],
  ['GPU', (t) => t.groundSubmitted, (t) => t.groundGpu],
  ['TOTAL wanted -> GPU done', (t) => t.wanted, (t) => t.groundGpu],
];
const CITY_STEPS = [
  ...STEPS,
  ['wait for finalize', (t) => t.applyEnd, (t) => t.finalizeStart],
  ['finalize (main)', (t) => t.finalizeStart, (t) => t.finalizeEnd],
  ['rest of frame (main)', (t) => t.finalizeEnd, (t) => t.citySubmitted],
  ['GPU', (t) => t.citySubmitted, (t) => t.cityGpu],
  ['TOTAL wanted -> GPU done', (t) => t.wanted, (t) => t.cityGpu],
];

const quantile = (list, q) => {
  if (!list.length) return NaN;
  const sorted = [...list].sort((a, b) => a - b);
  const i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
};
const mean = (list) => list.length ? list.reduce((s, v) => s + v, 0) / list.length : NaN;

function stepStats(tiles, steps) {
  const out = {};
  for (const [name, from, to] of steps) {
    const values = [];
    for (const t of tiles) {
      const a = from(t), b = to(t);
      if (a && b) values.push(b - a);
    }
    out[name] = { n: values.length, mean: mean(values), p50: quantile(values, 0.5), p90: quantile(values, 0.9), max: values.length ? Math.max(...values) : NaN };
  }
  return out;
}

function frameStats(frames) {
  const cpu = frames.filter((f) => f.submitted).map((f) => f.submitted - f.start);
  const gpu = frames.filter((f) => f.gpu).map((f) => f.gpu - f.submitted);
  const intervals = frames.slice(1).map((f, i) => f.start - frames[i].start);
  return {
    frames: frames.length,
    cpuP50: quantile(cpu, 0.5), cpuP90: quantile(cpu, 0.9), cpuMax: cpu.length ? Math.max(...cpu) : NaN,
    gpuP50: quantile(gpu, 0.5), gpuP90: quantile(gpu, 0.9), gpuMax: gpu.length ? Math.max(...gpu) : NaN,
    intervalP50: quantile(intervals, 0.5), intervalP90: quantile(intervals, 0.9), intervalMax: intervals.length ? Math.max(...intervals) : NaN,
  };
}

/** The tiles wanted in [t0, t1] and the frames in it: per scenario. */
function analyze(data, t0, t1) {
  const tiles = data.timings.filter((t) => t.wanted >= t0 && t.wanted <= t1);
  const ground = tiles.filter((t) => t.groundGpu);
  const city = tiles.filter((t) => t.cityGpu);
  const frames = data.frames.filter((f) => f.start >= t0 && f.start <= t1);
  const lastGround = ground.length ? Math.max(...ground.map((t) => t.groundGpu)) - t0 : NaN;
  const lastCity = city.length ? Math.max(...city.map((t) => t.cityGpu)) - t0 : NaN;
  // how long the tiles were in view while missing: from when they were seen in it (their request, or later for a tile
  // asked for ahead of the view) to drawn. 0 for a tile that was there before it came into view
  const missing = (list, drawn) => {
    const waits = list.map((t) => (t.visible ?? t.wanted) ? Math.max(drawn(t) - (t.visible ?? t.wanted), 0) : 0);
    const seen = waits.filter((w) => w > 0);
    const last = list.filter((t, i) => waits[i] > 0).map((t) => drawn(t) - t0);
    return { seen: seen.length, of: list.length, mean: mean(seen), p90: quantile(seen, 0.9), sum: seen.reduce((s, w) => s + w, 0), last: last.length ? Math.max(...last) : NaN };
  };
  return {
    groundMissing: missing(ground, (t) => t.groundGpu),
    cityMissing: missing(city, (t) => t.cityGpu),
    jobs: tiles.length,
    dropped: tiles.filter((t) => !t.groundGpu && !t.cityGpu).length,
    groundTiles: ground.length,
    cityTiles: city.length,
    lastGround, lastCity,
    ground: stepStats(ground, GROUND_STEPS),
    city: stepStats(city, CITY_STEPS),
    frames: frameStats(frames),
  };
}

async function waitIdle(page, start, minMs = 1000) {
  const t0 = Date.now();
  for (;;) {
    const state = await page.evaluate((start, minMs) => {
      const st = window.__mapWorld?.tileStreamer;
      if (!st) return { done: false, state: 'no streamer' };
      const now = performance.now();
      window.__busyAt ??= now;
      if (!st.isIdle() || st.loadedTerrainTiles.size === 0) window.__busyAt = now;
      const done = now - start > minMs && now - window.__busyAt > 500;
      if (done) window.__busyAt = undefined;
      return { done, state: `ground ${st.loadedTerrainTiles.size} buildings ${st.loadedCityTiles.size} in flight ${st.inFlight.size} queued ${st.fetchQueue.length} results ${st.resultQueue.size} city ${st.cityReadyQueue.length}` };
    }, start, minMs);
    if (state.done) return;
    if (Date.now() - t0 > 120_000) throw new Error(`not idle after 120 s: ${state.state}`);
    if ((Date.now() - t0) % 10_000 < 50) console.log(`  waiting: ${state.state}`);
    await sleep(50);
  }
}

async function run() {
  const browser = await puppeteer.launch({
    channel: 'chrome',
    headless: false,
    defaultViewport: null,
    protocolTimeout: 3600_000,
    args: ['--window-size=1280,800', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'],
  });
  try {
    const page = (await browser.pages())[0] ?? await browser.newPage();
    page.on('pageerror', (e) => console.error('[page]', e.message));
    page.on('console', (m) => {
      if (m.type() === 'error' && !m.text().includes('404')) console.error('[page]', m.text());
    });
    if (replayDir) {
      await page.setRequestInterception(true);
      page.on('request', async (request) => {
        const url = request.url();
        if (!TILE_HOSTS.some((host) => url.includes(host))) return void request.continue();
        try {
          const saved = await replay(url);
          if (latency > 0) await sleep(latency);
          await request.respond({ status: saved.status, contentType: saved.contentType, headers: { 'access-control-allow-origin': '*' }, body: saved.body });
        } catch (e) {
          if (!/intercept|abort|Session closed|Target closed/i.test(e.message)) console.error('replay', url, e.message);
          await request.abort().catch(() => {});
        }
      });
    }
    const epoch = () => page.evaluate(() => performance.timeOrigin + performance.now());
    const windows = {};

    console.log('browser up');
    const loadStart = Date.now();
    // the last frame that drew the map (a static view stops drawing when its image converged)
    await page.evaluateOnNewDocument(() => {
      const tick = () => {
        const state = window.__mapWorld?.frameState;
        if (state && (state.renderMap || state.newJitter)) window.__lastDraw = performance.now();
        if (window.__mapWorld?.tileStreamer && !window.__mapWorld.tileStreamer.isViewIdle()) window.__lastViewBusy = performance.now();
        requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    await page.goto(`http://localhost:${port}/?city=${encodeURIComponent(city)}&time=afternoon&tileTimings${extra ? '&' + extra : ''}`);
    console.log('page loaded');
    await page.waitForFunction(() => window.__mapWorld?.tileStreamer, { timeout: 600_000, polling: 50 });
    const pageStart = await page.evaluate(() => performance.timeOrigin);
    await waitIdle(page, 0, 0);
    windows.load = [[pageStart, await epoch()]];
    console.log(`load done after ${((Date.now() - loadStart) / 1000).toFixed(1)} s`);
    // the image of the first view converged: no frame drew the map for 1.5 s (at most 30 s)
    await page.waitForFunction(() => performance.now() - (window.__lastDraw ?? 0) > 1500, { timeout: 30_000, polling: 100 }).catch(() => {});
    const converged = await page.evaluate(() => ({ lastDraw: window.__lastDraw, viewTiles: window.__lastViewBusy }));
    console.log(`first view: its tiles in at ${converged.viewTiles?.toFixed(0)} ms, all tiles at ${(windows.load[0][1] - pageStart).toFixed(0)} ms, image converged at ${converged.lastDraw?.toFixed(0)} ms`);

    if (scenarios.includes('jump')) {
      windows.jump = [];
      for (const jump of jumps) {
        const start = await page.evaluate(([lat, lng]) => {
          const world = window.__mapWorld;
          const t = performance.now();
          void world.loadFromBounds(world.boundsAround(lat, lng));
          return t;
        }, jump);
        await waitIdle(page, start);
        windows.jump.push([pageStart + start, await epoch()]);
      }
    }

    if (scenarios.includes('drag')) {
      // a pan to the right, then back over the place it started from on another line: dragSpeed camera distances per
      // second, every frame by its time
      windows.drag = [];
      for (const sign of [1, -1]) {
        const start = await page.evaluate((speed, seconds, sign) => new Promise((resolve) => {
          const world = window.__mapWorld, camera = world.camera;
          const t0 = performance.now();
          let last = t0;
          const step = (t) => {
            const dt = Math.min(t - last, 100) / 1000;
            last = t;
            const right = camera.pivot.worldRight.clone();
            right.y = 0;
            right.normalize().multiplyScalar(sign * speed * camera.distance * dt);
            camera.pivot.translate(right);
            if (t - t0 < seconds * 1000) requestAnimationFrame(step);
            else resolve(t0);
          };
          requestAnimationFrame(step);
        }), dragSpeed, dragSeconds, sign);
        await waitIdle(page, start, 0);
        windows.drag.push([pageStart + start, await epoch()]);
      }
    }

    if (scenarios.includes('turn')) {
      // a full turn around the pivot, looking around
      windows.turn = [];
      const start = await page.evaluate((seconds) => new Promise((resolve) => {
        const rc = window.__mapWorld.camera.rotationController;
        const t0 = performance.now();
        let last = t0;
        const step = (t) => {
          const dt = Math.min(t - last, 100) / 1000;
          last = t;
          const e = rc.eulerXY.clone();
          e.y += Math.PI * 2 * dt / seconds;
          rc.eulerXY = e;
          if (t - t0 < seconds * 1000) requestAnimationFrame(step);
          else resolve(t0);
        };
        requestAnimationFrame(step);
      }), turnSeconds);
      await waitIdle(page, start, 0);
      windows.turn.push([pageStart + start, await epoch()]);
    }

    // the GPU work of the last frames
    await sleep(200);
    const data = await page.evaluate(() => {
      const st = window.__mapWorld.tileStreamer;
      return { timings: st.timings, frames: st.frameTimings, workers: window.__mapWorld.workerPool.size };
    });
    const result = { workers: data.workers };
    for (const [name, list] of Object.entries(windows)) {
      result[name] = list.map(([t0, t1]) => analyze(data, t0, t1));
    }
    return { result, data, windows };
  } finally {
    await browser.close();
  }
}

const f = (v) => Number.isFinite(v) ? v.toFixed(0) : '-';
function printScenario(name, list) {
  // the tiles of all windows of the scenario together: their steps again over all
  const merge = (kind) => {
    const out = {};
    for (const key of Object.keys(list[0][kind])) {
      const stats = list.map((s) => s[kind][key]).filter((s) => s.n > 0);
      const n = stats.reduce((s, v) => s + v.n, 0);
      out[key] = {
        n,
        mean: stats.reduce((s, v) => s + v.mean * v.n, 0) / Math.max(n, 1),
        // the median of the windows' medians and 90th percentiles: close enough for a few windows
        p50: quantile(stats.map((s) => s.p50), 0.5),
        p90: quantile(stats.map((s) => s.p90), 0.5),
        max: Math.max(...stats.map((s) => s.max)),
      };
    }
    return out;
  };
  const ground = merge('ground'), cityStats = merge('city');
  const sum = (key) => list.reduce((s, v) => s + v[key], 0);
  console.log(`\n=== ${name} (${list.length}x): ${sum('jobs')} jobs, ${sum('dropped')} dropped, ${sum('groundTiles')} ground and ${sum('cityTiles')} building tiles drawn; ` +
    `last ground ${list.map((s) => f(s.lastGround)).join('/')} ms, last buildings ${list.map((s) => f(s.lastCity)).join('/')} ms after the start`);
  for (const kind of ['groundMissing', 'cityMissing']) {
    const all = list.map((s) => s[kind]);
    const seen = all.reduce((s, m) => s + m.seen, 0), of = all.reduce((s, m) => s + m.of, 0), total = all.reduce((s, m) => s + m.sum, 0);
    console.log(`${kind === 'groundMissing' ? 'ground   ' : 'buildings'} seen missing: ${seen} of ${of} tiles, ${f(total / Math.max(seen, 1))} ms mean, ` +
      `p90 ${list.map((s) => f(s[kind].p90)).join('/')} ms, ${(total / 1000).toFixed(1)} tile-seconds missing, last in-view tile ${list.map((s) => f(s[kind].last)).join('/')} ms`);
  }
  console.log(`${''.padEnd(28)}${'ground: mean'.padStart(13)}${'p50'.padStart(7)}${'p90'.padStart(7)}${'max'.padStart(7)}  ${'buildings: mean'.padStart(15)}${'p50'.padStart(7)}${'p90'.padStart(7)}${'max'.padStart(7)}`);
  const names = [...new Set([...Object.keys(ground), ...Object.keys(cityStats)])];
  for (const key of names) {
    const g = ground[key], c = cityStats[key];
    const cells = (s) => s ? `${f(s.mean).padStart(7)}${f(s.p50).padStart(7)}${f(s.p90).padStart(7)}${f(s.max).padStart(7)}` : ''.padStart(28);
    console.log(`${key.padEnd(28)}${''.padStart(6)}${g ? cells(g) : ''.padStart(28)}  ${''.padStart(8)}${cells(c)}`);
  }
  const fr = list.map((s) => s.frames);
  const m = (key) => f(quantile(fr.map((x) => x[key]), 0.5));
  console.log(`frames: ${fr.reduce((s, x) => s + x.frames, 0)}, main thread ms p50/p90/max ${m('cpuP50')}/${m('cpuP90')}/${m('cpuMax')}, ` +
    `submit -> GPU done p50/p90/max ${m('gpuP50')}/${m('gpuP90')}/${m('gpuMax')}, frame interval p50/p90/max ${m('intervalP50')}/${m('intervalP90')}/${m('intervalMax')}`);
}

const all = [];
try {
  for (let i = 0; i < runs; i++) {
    const { result, data, windows } = await run();
    all.push(result);
    console.log(`\n##### run ${i + 1}, ${result.workers} build workers, ${replayDir ? `replayed${latency ? ` +${latency} ms` : ''}` : 'live tile servers'}`);
    for (const name of Object.keys(windows)) printScenario(name, result[name]);
    if (outFile) fs.writeFileSync(outFile.replace(/\.json$/, `-${i + 1}.json`), JSON.stringify({ label, result, data, windows }, null, 1));
  }
} finally {
  await server.close();
}
