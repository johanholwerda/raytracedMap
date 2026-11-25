// Times the load of the map tiles: builds the app as it is deployed (vite build), serves it, opens ?bench (see
// src/mapWorld/MapWorldBench.ts) on a city and records, from the navigation on: when the app starts, the first tile
// request, the first ground and buildings on screen, when the tiles of the view are in, and when the image converged.
// Then jumps to other places with the app warm (no startup in it), for the streaming alone: the mean of the jumps.
//
//   node video/loadTiming.mjs [--root dir] [--label name] [--city Amsterdam] [--jump "48.853,2.3498;40.711,-74.012"] [--runs 3]
//     [--replay dir] [--latency ms] [--profile] [--timeline] [--screenshot file.png] [--out file.json] [--nobuild]
//     [--params "workers=4&earlyWorkers=1"] [--port 5187]
//
// The machine does other things too: for a before and after, run the 2 versions in turns (1 run each, a few rounds) and
// compare the medians.
// --root: the source tree to build (default this repo), for a baseline in a git worktree. A worktree needs the
//   node_modules of the repo: put it in a folder of the repo (video/out/...), the imports resolve upwards
// --replay: the tile and elevation responses are saved in dir on their first request and answered from it after, so the
//   runs don't depend on the network (--latency: each answer after this many ms). Without it: the live servers, with an
//   empty browser cache per run
// --profile: a CPU profile of the main thread and the workers of the first load, the self times printed
// --timeline: per run the first of each step of the start and the long frames
// --screenshot: the page of the last run when its first load converged (file.png) and after the jump (file-jump.png)
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
const runs = parseInt(option('runs', '3'));
const replayDir = option('replay', '');
const latency = parseFloat(option('latency', '0'));
const profiling = args.includes('--profile');
const port = parseInt(option('port', '5187'));
const outFile = option('out', '');
const extra = option('params', '');
if (replayDir) fs.mkdirSync(replayDir, { recursive: true });

const outDir = path.join(repo, 'video', 'out', `dist-${label}`);
// --nobuild: the build of the last run of this label again
if (!args.includes('--nobuild') || !fs.existsSync(path.join(outDir, 'index.html'))) await build({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  resolve: { alias: { 'webGPU-framework': path.join(root, 'src/lib') } },
  // a profile is of the build with its names
  build: { outDir, emptyOutDir: true, minify: !profiling },
});
const server = await preview({
  configFile: false,
  root,
  base: './',
  logLevel: 'warn',
  build: { outDir },
  preview: { host: 'localhost', port, strictPort: true },
});

// --- the tile servers from disk
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
    // not the errors of the moment
    if (response.status < 500) {
      fs.writeFileSync(file + '.bin', entry.body);
      fs.writeFileSync(file + '.json', JSON.stringify({ status: entry.status, contentType: entry.contentType, url }));
    }
  }
  replayMemory.set(url, entry);
  return entry;
}

/** The kind of a request to the tile servers: the map tiles, the elevation tiles, the TileJSON. */
function requestKind(url) {
  if (!TILE_HOSTS.some((host) => url.includes(host))) return '';
  return url.endsWith('.pbf') ? 'pbf' : url.includes('terrarium') ? 'dem' : 'tilejson';
}

// --- in the page, before the app: the frames and the state of the streamer per frame, the workers
function instrument() {
  const perf = window.__load = { fetches: [], frames: [], series: [], marks: {}, loaf: [], workers: [] };
  const NativeWorker = window.Worker;
  window.Worker = class extends NativeWorker {
    constructor(...a) {
      super(...a);
      // map: a worker that builds tiles, not the one of the elevation
      const entry = { created: performance.now(), firstPost: 0, firstMessage: 0, map: String(a[0]).includes('worldMapWorker') };
      perf.workers.push(entry);
      const post = this.postMessage;
      this.postMessage = function () {
        entry.firstPost ||= performance.now();
        return post.apply(this, arguments);
      };
      // a tile: not the answer to a cancel
      this.addEventListener('message', (e) => { if (e.data?.result !== undefined) entry.firstMessage ||= performance.now(); });
    }
  };
  // when the app asks for the GPU (its first code that runs) and gets it
  const requestAdapter = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = function () {
    perf.marks.gpuRequest ??= performance.now();
    return requestAdapter.apply(this, arguments);
  };
  const requestDevice = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = function () {
    return requestDevice.apply(this, arguments).then((device) => {
      perf.marks.gpuDevice ??= performance.now();
      return device;
    });
  };
  const raf = window.requestAnimationFrame.bind(window);
  const tick = () => {
    const now = performance.now();
    perf.frames.push(now);
    const world = window.__mapWorld, st = world?.tileStreamer;
    if (world) perf.marks.app ??= now;
    if (st) {
      perf.marks.streamer ??= now;
      perf.series.push([now, st.loadedTerrainTiles.size, st.loadedCityTiles.size, st.inFlight.size, st.fetchQueue.length,
        st.resultQueue?.size ?? 0, st.cityReadyQueue.length]);
    }
    raf(tick);
  };
  raf(tick);
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) perf.loaf.push({ t: e.startTime, duration: e.duration, blocking: e.blockingDuration });
  }).observe({ type: 'long-animation-frame', buffered: true });
}

/** The milestones of a load from the samples of the streamer between t0 and t1 (ms), relative to origin. */
function milestones(perf, t0, t1, origin) {
  const series = perf.series.filter((s) => s[0] >= t0 && s[0] <= t1);
  const busy = (s) => s[3] + s[4] + s[5] + s[6];
  const rel = (t) => t === undefined ? NaN : t - origin;
  const pbf = perf.fetches.filter((f) => f.kind === 'pbf' && f.t0 >= t0 && f.t0 <= t1);
  const dem = perf.fetches.filter((f) => f.kind === 'dem' && f.t0 >= t0 && f.t0 <= t1);
  // the last sample with work left: the tiles are all in right after it
  let lastBusy = -1;
  series.forEach((s, i) => { if (busy(s) > 0) lastBusy = i; });
  const end = series[Math.min(lastBusy + 1, series.length - 1)];
  const ground = end?.[1] ?? 0, cityTiles = end?.[2] ?? 0;
  const reached = (column, count) => rel(series.find((s) => s[column] >= count)?.[0]);
  const frames = perf.frames.filter((t) => t >= t0 && t <= (end?.[0] ?? t1));
  const intervals = frames.slice(1).map((t, i) => t - frames[i]);
  const mean = (list) => list.length ? list.reduce((s, v) => s + v, 0) / list.length : NaN;
  return {
    firstFetch: rel(pbf.length ? Math.min(...pbf.map((f) => f.t0)) : undefined),
    firstGround: reached(1, 1),
    firstCity: reached(2, 1),
    halfGround: reached(1, Math.ceil(ground / 2)),
    halfCity: reached(2, Math.ceil(cityTiles / 2)),
    allGround: reached(1, ground),
    allCity: reached(2, cityTiles),
    tilesDone: rel(end?.[0]),
    groundTiles: ground,
    cityTiles,
    pbfCount: pbf.length,
    pbfMs: mean(pbf.filter((f) => f.t2).map((f) => f.t2 - f.t0)),
    pbfMB: pbf.reduce((s, f) => s + f.bytes, 0) / 1048576,
    demCount: dem.length,
    demMs: mean(dem.filter((f) => f.t2).map((f) => f.t2 - f.t0)),
    frameMax: intervals.length ? Math.max(...intervals) : NaN,
    framesOver50: intervals.filter((v) => v > 50).length,
    framesOver100: intervals.filter((v) => v > 100).length,
    loafMs: perf.loaf.filter((l) => l.t >= t0 && l.t <= (end?.[0] ?? t1)).reduce((s, l) => s + l.duration, 0),
  };
}

/** The self time per function of a CPU profile, the idle time left out. */
function selfTimes(profile, byFn = new Map()) {
  const nodes = new Map(profile.nodes.map((n) => [n.id, n]));
  for (let i = 0; i < profile.samples.length; i++) {
    const cf = nodes.get(profile.samples[i]).callFrame;
    if (cf.functionName === '(idle)') continue;
    const key = `${cf.functionName || '(anonymous)'} ${cf.url.split('/').pop().split('?')[0]}:${cf.lineNumber + 1}:${cf.columnNumber + 1}`;
    byFn.set(key, (byFn.get(key) || 0) + (profile.timeDeltas[i] || 0));
  }
  return byFn;
}

function printSelfTimes(title, byFn, count = 30) {
  const total = [...byFn.values()].reduce((s, v) => s + v, 0);
  console.log(`${title}: ${(total / 1000).toFixed(0)} ms busy, self time:`);
  for (const [k, us] of [...byFn].sort((a, b) => b[1] - a[1]).slice(0, count)) {
    console.log(`  ${(us / 1000).toFixed(0).padStart(7)} ms  ${(us / total * 100).toFixed(1).padStart(5)}%  ${k}`);
  }
}

async function run(index) {
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
    page.on('pageerror', (e) => console.error('[page]', e.message));
    page.on('console', (m) => {
      if (m.type() === 'error' && !m.text().includes('404')) console.error('[page]', m.text());
    });
    await page.evaluateOnNewDocument(instrument);
    // The requests to the tile servers, of the page and of its workers, timed here (wall clock ms: made times of the
    // page with its performance.timeOrigin after the run) and with --replay answered from disk
    const requests = [];
    const byRequest = new Map();
    if (replayDir) await page.setRequestInterception(true);
    page.on('request', async (request) => {
      const url = request.url();
      const kind = requestKind(url);
      if (kind) {
        const entry = { kind, w0: Date.now(), w2: 0, bytes: 0 };
        requests.push(entry);
        byRequest.set(request, entry);
      }
      if (!replayDir) return;
      if (!kind) return void request.continue();
      try {
        const saved = await replay(url);
        if (latency > 0) await sleep(latency);
        byRequest.get(request).bytes = saved.body.length;
        await request.respond({ status: saved.status, contentType: saved.contentType, headers: { 'access-control-allow-origin': '*' }, body: saved.body });
      } catch (e) {
        // a request the page aborted (a cancelled tile) is no error
        if (!/intercept|abort|Session closed|Target closed/i.test(e.message)) console.error('replay', url, e.message);
        await request.abort().catch(() => {});
      }
    });
    page.on('requestfinished', (request) => {
      const entry = byRequest.get(request);
      if (!entry) return;
      entry.w2 = Date.now();
      entry.bytes ||= Number(request.response()?.headers()['content-length'] ?? 0);
    });

    const workerSessions = [];
    let client = null;
    if (profiling && index === 0) {
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
      client = await page.createCDPSession();
      await client.send('Profiler.enable');
      await client.send('Profiler.setSamplingInterval', { interval: 200 });
      await client.send('Profiler.start');
    }

    // the drags of the bench are not timed here: 1 short one, so it ends
    const url = `http://localhost:${port}/?bench=1&dragFrames=1&dragSpeed=0&city=${encodeURIComponent(city)}${extra ? '&' + extra : ''}`;
    await page.goto(url);
    // the load phase of the bench ends when the tiles are in and the map stopped drawing
    await page.waitForFunction(() => window.__mapWorldBench?.results.length >= 1, { timeout: 600_000, polling: 50 });
    const loadEnd = await page.evaluate(() => performance.now());
    const screenshot = option('screenshot', '');
    if (screenshot && index === runs - 1) await page.screenshot({ path: screenshot });

    if (client) {
      const { profile } = await client.send('Profiler.stop');
      printSelfTimes('\nmain thread', selfTimes(profile));
      const byFn = new Map();
      for (const session of workerSessions) selfTimes((await session.send('Profiler.stop')).profile, byFn);
      printSelfTimes(`\n${workerSessions.length} workers`, byFn, 40);
      console.log('');
    }
    const poolStats = () => page.evaluate(() => ({ ...window.__mapWorld.workerPool.stats, workers: window.__mapWorld.workerPool.size }));
    const statsLoad = await poolStats();

    // the 2nd place, with the app warm
    await page.waitForFunction(() => window.__mapWorldBench?.status === 'done', { timeout: 600_000, polling: 50 });
    const jumpTimes = [];
    for (const jump of jumps) {
      const jumpStart = await page.evaluate(([lat, lng]) => {
        const world = window.__mapWorld;
        const t = performance.now();
        void world.loadFromBounds(world.boundsAround(lat, lng));
        return t;
      }, jump);
      // all tiles in, for half a second
      await page.waitForFunction((start) => {
        const st = window.__mapWorld.tileStreamer;
        const now = performance.now();
        if (!st.isIdle() || st.loadedTerrainTiles.size === 0) window.__load.busyAt = now;
        return now - start > 1000 && now - (window.__load.busyAt ?? now) > 500;
      }, { timeout: 600_000, polling: 50 }, jumpStart);
      jumpTimes.push([jumpStart, await page.evaluate(() => performance.now())]);
    }
    if (screenshot && index === runs - 1) {
      // the image of the last place converged too
      await sleep(4000);
      await page.screenshot({ path: screenshot.replace(/\.png$/, '-jump.png') });
    }
    const statsJump = await poolStats();

    const perf = await page.evaluate(() => window.__load);
    const timeOrigin = await page.evaluate(() => performance.timeOrigin);
    perf.fetches = requests.map((r) => ({ kind: r.kind, t0: r.w0 - timeOrigin, t1: r.w2 ? r.w2 - timeOrigin : 0, t2: r.w2 ? r.w2 - timeOrigin : 0, bytes: r.bytes }));
    const mapWorkers = perf.workers.filter((w) => w.map);
    const load = milestones(perf, 0, loadEnd, 0);
    // the mean of the jumps
    const perJump = jumpTimes.map(([start, end]) => milestones(perf, start, end, start));
    const jumped = {};
    for (const key of Object.keys(perJump[0])) jumped[key] = perJump.reduce((s, j) => s + j[key], 0) / perJump.length;
    const perJob = (a, b) => ({
      jobs: b.jobs - a.jobs,
      workMs: (b.busyMs - a.busyMs) / Math.max(b.jobs - a.jobs, 1),
      queueMs: (b.queueMs - a.queueMs) / Math.max(b.jobs - a.jobs, 1),
      // the fetch worker's: until the tile and its heights were there (a pool that fetches on the main thread has none)
      fetchMs: ((b.fetchMs ?? NaN) - (a.fetchMs ?? NaN)) / Math.max(b.jobs - a.jobs, 1),
    });
    const tilejson = perf.fetches.find((f) => f.kind === 'tilejson');
    if (args.includes('--timeline')) {
      // the first of each event and the long frames of the load, in ms from the navigation
      const first = (list) => list.length ? Math.min(...list).toFixed(0) : '-';
      const pbf = perf.fetches.filter((f) => f.kind === 'pbf'), dem = perf.fetches.filter((f) => f.kind === 'dem');
      console.log(`timeline: GPU asked ${perf.marks.gpuRequest?.toFixed(0)} got ${perf.marks.gpuDevice?.toFixed(0)}, app ${perf.marks.app?.toFixed(0)}, workers made ${first(mapWorkers.map((w) => w.created))}, ` +
        `first tile request ${first(pbf.map((f) => f.t0))} answered ${first(pbf.filter((f) => f.t2).map((f) => f.t2))}, ` +
        `first elevation request ${first(dem.map((f) => f.t0))} answered ${first(dem.filter((f) => f.t2).map((f) => f.t2))}, ` +
        `first tile built ${first(mapWorkers.filter((w) => w.firstMessage).map((w) => w.firstMessage))}`);
      const long = [];
      for (let i = 1; i < perf.frames.length && perf.frames[i] < loadEnd; i++) {
        const d = perf.frames[i] - perf.frames[i - 1];
        if (d > 40) long.push(`${perf.frames[i - 1].toFixed(0)}+${d.toFixed(0)}`);
      }
      console.log(`frames > 40 ms (start+duration): first frame at ${perf.frames[0]?.toFixed(0)}, ${long.join(' ')}`);
      // the streamer at every frame until its tiles are in: ground / building tiles loaded, then requests under way /
      // waiting for one / built and waiting for the frame / buildings waiting for theirs
      const until = perf.series.findLastIndex((s) => s[0] < loadEnd && s[3] + s[4] + s[5] + s[6] > 0);
      console.log(`streamer per frame (ms: ground/buildings in, in flight/queued/built/buildings waiting): ` +
        perf.series.slice(0, until + 2).map((s) => `${s[0].toFixed(0)}: ${s[1]}/${s[2]} ${s[3]}/${s[4]}/${s[5]}/${s[6]}`).join(', '));
    }
    return {
      load: {
        gpuRequest: perf.marks.gpuRequest,
        gpuDevice: perf.marks.gpuDevice,
        app: perf.marks.app,
        tilejson: tilejson ? tilejson.t1 - tilejson.t0 : NaN,
        workersCreated: mapWorkers.length ? Math.min(...mapWorkers.map((w) => w.created)) : NaN,
        firstWorkerResult: Math.min(...mapWorkers.map((w) => w.firstMessage || Infinity)),
        ...load,
        converged: loadEnd,
        ...perJob({ jobs: 0, busyMs: 0, queueMs: 0, fetchMs: 0 }, statsLoad),
      },
      jump: { ...jumped, ...perJob(statsLoad, statsJump) },
      workers: statsLoad.workers,
    };
  } finally {
    await browser.close();
  }
}

const results = [];
try {
  for (let i = 0; i < runs; i++) {
    const result = await run(i);
    results.push(result);
    const l = result.load, j = result.jump;
    const f = (v) => Number.isFinite(v) ? v.toFixed(0) : '-';
    console.log(`run ${i + 1}: load app ${f(l.app)} firstFetch ${f(l.firstFetch)} firstGround ${f(l.firstGround)} firstCity ${f(l.firstCity)} ` +
      `halfCity ${f(l.halfCity)} allGround ${f(l.allGround)} allCity ${f(l.allCity)} tilesDone ${f(l.tilesDone)} converged ${f(l.converged)} | ` +
      `jumps firstGround ${f(j.firstGround)} firstCity ${f(j.firstCity)} halfCity ${f(j.halfCity)} tilesDone ${f(j.tilesDone)}`);
  }
} finally {
  await server.close();
}

// --- the median of each number over the runs
const median = (list) => {
  const sorted = list.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2 : NaN;
};
const summary = {};
for (const phase of ['load', 'jump']) {
  summary[phase] = {};
  for (const key of Object.keys(results[0][phase])) summary[phase][key] = median(results.map((r) => r[phase][key]));
}
const rows = [
  ['gpuRequest', 'the app asks for the GPU (ms)'], ['gpuDevice', 'the GPU device is there'], ['app', 'the renderer is ready'], ['tilejson', 'TileJSON request'], ['firstFetch', 'first tile request'],
  ['firstWorkerResult', 'first worker result'], ['firstGround', 'first ground tile drawn'], ['firstCity', 'first building tile drawn'],
  ['halfGround', 'half of the ground tiles'], ['halfCity', 'half of the building tiles'], ['allGround', 'all ground tiles'],
  ['allCity', 'all building tiles'], ['tilesDone', 'streamer idle'], ['converged', 'image converged'],
  ['groundTiles', 'ground tiles'], ['cityTiles', 'building tiles'], ['pbfCount', 'tile requests'], ['pbfMs', 'ms per tile request'],
  ['pbfMB', 'MB of tiles'], ['demCount', 'elevation requests'], ['demMs', 'ms per elevation request'], ['jobs', 'worker jobs'],
  ['workMs', 'worker ms per job'], ['queueMs', 'ms a job waits for a worker'], ['fetchMs', 'ms until a job has its data'], ['frameMax', 'longest frame (ms)'],
  ['framesOver50', 'frames > 50 ms'], ['framesOver100', 'frames > 100 ms'], ['loafMs', 'long animation frames (ms)'],
];
const cell = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 100 || Number.isInteger(v) ? v.toFixed(0) : v.toFixed(1)) : '-').padStart(9);
console.log(`\n${label}: ${city}, then ${jumps.length} jumps (their mean); median of ${runs} runs, ${results[0].workers} workers, ` +
  `${replayDir ? `tiles replayed from disk${latency ? ` after ${latency} ms` : ''}` : 'live tile servers'}`);
console.log(`${''.padEnd(30)}${'load'.padStart(9)}${'jump'.padStart(9)}`);
for (const [key, name] of rows) console.log(`${name.padEnd(30)}${cell(summary.load[key])}${cell(summary.jump[key])}`);
if (outFile) fs.writeFileSync(outFile, JSON.stringify({ label, city, jumps, runs, replay: !!replayDir, latency, summary, results }, null, 2));
