// Times what the map worker does per tile (buildWorldTile of src/mapWorld/map/worldGeometryCore.ts) in node, without a
// browser: on the tiles and the elevation a run of loadTiming.mjs --replay saved. Writes a hash of everything each tile
// gives, to check that a faster version still makes the same geometry.
//
//   node video/tileBench.mjs --replay dir [--root dir] [--label name] [--passes 4] [--limit n] [--compare other.json]
//     [--threads n] [--heaviest n] [--allocs] [--verbose]
//
// --threads: after the passes, the same tiles in n worker threads at once (as the workers of the map build them, see
//   WorldWorkerPool), each starting cold and then again warm: the ms per tile there, to compare with the 1 thread above
// --heaviest: only the n tiles with the most data. --allocs: what a pass allocates, by function. --verbose: per tile
// --root: the source tree (default this repo), for a baseline in a git worktree
// --compare: the hashes of another run (video/out/tileBench-<label>.json), the tiles that differ are listed
// A CPU profile: node --cpu-prof --cpu-prof-dir=video/out video/tileBench.mjs ...
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker } from 'node:worker_threads';
import { build } from 'esbuild';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const root = path.resolve(option('root', repo));
const label = option('label', path.basename(root));
const replayDir = option('replay', '');
const passes = parseInt(option('passes', '4'));
const limit = parseInt(option('limit', '0'));
const compare = option('compare', '');
if (!replayDir) throw new Error('--replay dir: the folder a run of loadTiming.mjs --replay filled');

const outDir = path.join(repo, 'video', 'out');
fs.mkdirSync(outDir, { recursive: true });
const bundle = path.join(outDir, `tileBench-${label}.bundle.mjs`);
await build({
  stdin: {
    contents: `
      export { buildWorldTile } from './src/mapWorld/map/worldGeometryCore';
      export { default as ElevationService } from './src/mapWorld/map/ElevationService';
      export { default as LogGL } from './src/lib/renderer/core/LogGL';
    `,
    resolveDir: root,
    loader: 'ts',
  },
  bundle: true,
  format: 'esm',
  platform: 'node',
  outfile: bundle,
  alias: { 'webGPU-framework': path.join(root, 'src/lib') },
  logLevel: 'warning',
});
const { buildWorldTile, ElevationService, LogGL } = await import(pathToFileURL(bundle).href + `?${Date.now()}`);
LogGL.ENABLED = false;

// --- the saved responses, by url
const saved = new Map();
for (const file of fs.readdirSync(replayDir)) {
  if (!file.endsWith('.json')) continue;
  const meta = JSON.parse(fs.readFileSync(path.join(replayDir, file), 'utf8'));
  saved.set(meta.url, { ...meta, file: path.join(replayDir, file.replace(/\.json$/, '.bin')) });
}
globalThis.fetch = async (url) => {
  const entry = saved.get(String(url));
  if (!entry) return new Response(null, { status: 404 });
  return new Response(entry.status === 204 ? null : fs.readFileSync(entry.file), { status: entry.status });
};

// the bounds MapWorld.boundsAround gives a place (the span of DEFAULT_BOUNDS of osm/layerDefs.ts)
const boundsAround = (lat, lng) => ({ south: lat - .00814, north: lat + .00814, west: lng - .02311, east: lng + .02311 });
const tileCenter = (z, x, y) => {
  const n = 2 ** z;
  return { lng: (x + .5) / n * 360 - 180, lat: Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + .5) / n))) * 180 / Math.PI };
};

// the map tiles, in groups that are close together: each group is a place, projected around its middle
let tiles = [];
for (const [url, entry] of saved) {
  const m = /\/(\d+)\/(\d+)\/(\d+)\.pbf$/.exec(url);
  if (!m || entry.status !== 200) continue;
  tiles.push({ url, file: entry.file, tile: { z: +m[1], x: +m[2], y: +m[3] } });
}
tiles.sort((a, b) => a.tile.x - b.tile.x || a.tile.y - b.tile.y);
const groups = [];
for (const t of tiles) {
  const group = groups.find((g) => Math.abs(g[0].tile.x - t.tile.x) < 64 && Math.abs(g[0].tile.y - t.tile.y) < 64);
  if (group) group.push(t);
  else groups.push([t]);
}
const jobs = [];
for (const group of groups) {
  const lat = group.reduce((s, t) => s + tileCenter(t.tile.z, t.tile.x, t.tile.y).lat, 0) / group.length;
  const lng = group.reduce((s, t) => s + tileCenter(t.tile.z, t.tile.x, t.tile.y).lng, 0) / group.length;
  // rounded, so the same place gets the same projection whichever of its tiles were saved
  const bounds = boundsAround(Math.round(lat * 100) / 100, Math.round(lng * 100) / 100);
  const elevation = new ElevationService();
  elevation.setOrigin((bounds.south + bounds.north) / 2, (bounds.west + bounds.east) / 2);
  for (const t of group) {
    try {
      const { heights, seaLevel } = await elevation.getTileGrid(t.tile);
      jobs.push({ ...t, bounds, heights, seaLevel, pbf: new Uint8Array(fs.readFileSync(t.file)) });
    } catch (e) {
      console.warn(`no elevation for ${t.url}: ${e.message}`);
    }
  }
}
// --heaviest n: only the n tiles with the most data (the centers of the cities)
const heaviest = parseInt(option('heaviest', '0'));
if (heaviest > 0) {
  jobs.sort((a, b) => b.pbf.length - a.pbf.length);
  jobs.length = Math.min(jobs.length, heaviest);
}
if (limit > 0) jobs.length = Math.min(jobs.length, limit);

/** A hash of all a tile gives: the typed arrays by their bytes, the rest as JSON. */
function hashResult(result) {
  const hash = crypto.createHash('sha1');
  const visit = (value, name) => {
    if (value == null) return void hash.update(`${name}:null;`);
    if (ArrayBuffer.isView(value)) {
      hash.update(`${name}:${value.constructor.name}:${value.length};`);
      return void hash.update(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    }
    if (value instanceof ArrayBuffer) {
      hash.update(`${name}:ArrayBuffer:${value.byteLength};`);
      return void hash.update(new Uint8Array(value));
    }
    if (Array.isArray(value) && value.every((v) => typeof v !== 'object')) return void hash.update(`${name}:${JSON.stringify(value)};`);
    if (typeof value === 'object') {
      for (const key of Object.keys(value).sort()) visit(value[key], `${name}.${key}`);
      return;
    }
    hash.update(`${name}:${JSON.stringify(value)};`);
  };
  visit(result, 'result');
  return hash.digest('hex');
}

/** The hashes of the parts of a tile on their own, to tell which part differs. */
function hashParts(result) {
  const parts = {};
  for (const [key, layer] of Object.entries(result.terrain ?? {})) parts[`terrain.${key}`] = hashResult(layer);
  parts.ground = hashResult(result.ground);
  parts.roads = hashResult(result.roads);
  const b = result.buildings;
  if (b) {
    for (const key of Object.keys(b)) parts[`buildings.${key}`] = hashResult(b[key]);
  }
  return parts;
}

const hashes = {};
const passMs = [];
// per tile its ms in the first pass (a cold JIT at first, as in a worker that just started) and its best after
const firstMs = [], bestMs = jobs.map(() => Infinity);
let bytes = 0, vertices = 0;
for (let pass = 0; pass < passes; pass++) {
  const t0 = performance.now();
  for (const [index, job] of jobs.entries()) {
    const start = performance.now();
    // the worker gets copies too (transferred in)
    const result = buildWorldTile(job.pbf.slice(), job.tile, job.bounds, true, true, true, job.heights.slice(), job.seaLevel);
    const ms = performance.now() - start;
    if (pass === 0) firstMs.push(ms);
    else bestMs[index] = Math.min(bestMs[index], ms);
    if (pass === 0) {
      hashes[`${job.tile.z}/${job.tile.x}/${job.tile.y}`] = hashParts(result);
      for (const layer of [...Object.values(result.terrain), result.ground]) {
        bytes += layer.vertices.byteLength + layer.indices.byteLength;
        vertices += layer.vertices.length / layer.floats;
      }
      bytes += result.buildings.interleaved.byteLength + result.buildings.indices.byteLength + (result.buildings.trace?.words.byteLength ?? 0);
    }
  }
  passMs.push(performance.now() - t0);
}
const perTile = passMs.map((ms) => ms / jobs.length);
console.log(`${label}: ${jobs.length} tiles of ${groups.length} places, ${(bytes / 1048576 / jobs.length).toFixed(2)} MB and ` +
  `${Math.round(vertices / jobs.length)} ground vertices per tile`);
console.log(`ms per tile: first pass (cold) ${perTile[0].toFixed(1)}, then ${perTile.slice(1).map((v) => v.toFixed(1)).join(', ')}` +
  (perTile.length > 1 ? `; best ${Math.min(...perTile.slice(1)).toFixed(1)}` : ''));
if (args.includes('--verbose')) {
  console.log(`the first 12 tiles of the first pass, ms (their best after): ` +
    firstMs.slice(0, 12).map((ms, i) => `${ms.toFixed(0)} (${bestMs[i].toFixed(0)})`).join(', '));
  const heavy = jobs.map((job, i) => ({ key: `${job.tile.z}/${job.tile.x}/${job.tile.y}`, ms: bestMs[i], kb: job.pbf.length / 1024 }))
    .sort((a, b) => b.ms - a.ms).slice(0, 8);
  console.log(`the slowest tiles: ${heavy.map((h) => `${h.key} ${h.ms.toFixed(0)} ms (${h.kb.toFixed(0)} KB)`).join(', ')}`);
}
// --allocs: what a pass allocates, by the function that allocates it (sampled, the garbage too)
if (args.includes('--allocs')) {
  const { Session } = await import('node:inspector/promises');
  const session = new Session();
  session.connect();
  await session.post('HeapProfiler.startSampling', {
    samplingInterval: 16384, includeObjectsCollectedByMajorGC: true, includeObjectsCollectedByMinorGC: true,
  });
  for (const job of jobs) buildWorldTile(job.pbf.slice(), job.tile, job.bounds, true, true, true, job.heights.slice(), job.seaLevel);
  const { profile } = await session.post('HeapProfiler.stopSampling');
  const lines = fs.readFileSync(bundle, 'utf8').split(/\r?\n/);
  const bySite = new Map();
  let total = 0;
  const visit = (node) => {
    const cf = node.callFrame;
    const key = `${cf.functionName || '(anonymous)'}:${cf.lineNumber + 1}`;
    bySite.set(key, (bySite.get(key) ?? 0) + node.selfSize);
    total += node.selfSize;
    node.children.forEach(visit);
  };
  visit(profile.head);
  console.log(`allocated per tile: ${(total / jobs.length / 1048576).toFixed(1)} MB, by function:`);
  for (const [key, size] of [...bySite].sort((a, b) => b[1] - a[1]).slice(0, 32)) {
    const line = parseInt(key.split(':').pop());
    console.log(`  ${(size / jobs.length / 1048576).toFixed(2).padStart(7)} MB  ${key.padEnd(28)} ${(lines[line - 1] ?? '').trim().slice(0, 90)}`);
  }
}
const threads = parseInt(option('threads', '0'));
if (threads > 0) {
  const source = `
    import { parentPort, workerData } from 'node:worker_threads';
    const { buildWorldTile } = await import(workerData.bundle);
    const ms = [];
    // its tiles, then all again with the JIT warm
    for (let pass = 0; pass < 2; pass++) {
      for (const job of workerData.jobs) {
        const start = performance.now();
        buildWorldTile(job.pbf.slice(), job.tile, job.bounds, true, true, true, job.heights.slice(), job.seaLevel);
        ms.push(performance.now() - start);
      }
    }
    parentPort.postMessage(ms);
  `;
  const start = performance.now();
  // each thread its share of the tiles, as the pool of the map deals them out
  const results = await Promise.all(Array.from({ length: threads }, (_, t) => new Promise((resolve, reject) => {
    const share = jobs.filter((_, i) => i % threads === t).map(({ pbf, tile, bounds, heights, seaLevel }) => ({ pbf, tile, bounds, heights, seaLevel }));
    const worker = new Worker(source, { eval: true, workerData: { bundle: pathToFileURL(bundle).href, jobs: share } });
    worker.once('message', resolve);
    worker.once('error', reject);
  })));
  const mean = (list) => list.reduce((s, v) => s + v, 0) / list.length;
  const cold = results.flatMap((r) => r.slice(0, r.length / 2)), warm = results.flatMap((r) => r.slice(r.length / 2));
  console.log(`${threads} threads at once: ${mean(cold).toFixed(1)} ms per tile from cold (${mean(firstMs).toFixed(1)} in the first pass ` +
    `of 1 thread), the first tile of a thread ${mean(results.map((r) => r[0])).toFixed(0)} ms, then warm ${mean(warm).toFixed(1)} ms per tile ` +
    `(${Math.min(...perTile.slice(1)).toFixed(1)} with 1 thread)`);
}
const hashFile = path.join(outDir, `tileBench-${label}.json`);
fs.writeFileSync(hashFile, JSON.stringify(hashes, null, 1));

if (compare) {
  const other = JSON.parse(fs.readFileSync(compare, 'utf8'));
  let same = 0;
  const differing = new Map();
  for (const [key, parts] of Object.entries(hashes)) {
    const theirs = other[key];
    if (!theirs) continue;
    const diff = Object.keys(parts).filter((part) => parts[part] !== theirs[part]);
    if (diff.length === 0) same++;
    for (const part of diff) differing.set(part, (differing.get(part) ?? 0) + 1);
  }
  const compared = Object.keys(hashes).filter((key) => other[key]).length;
  console.log(`compared with ${path.basename(compare)}: ${same} of ${compared} tiles give the same bytes` +
    (differing.size ? `; tiles that differ per part: ${[...differing].map(([part, n]) => `${part} ${n}`).join(', ')}` : ''));
}
