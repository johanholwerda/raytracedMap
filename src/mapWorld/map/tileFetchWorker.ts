import ElevationService from './ElevationService';
import type { TileGrid } from './ElevationService';
import type { WorldBuildMessage, WorldBuildDone, WorldWorkerRequest } from './worldMapWorker';
import { jobClock } from './jobTimes';
import type { JobTimes } from './jobTimes';

/**
 * The tile fetch Web Worker: gets what a map tile is made of, off the main thread, and hands it to the workers that
 * build (worldMapWorker.ts). One of it, next to the pool of those.
 *
 * Per job (a tile to make, from WorldWorkerPool) it fetches the vector tile (.pbf) and makes the heights of the ground
 * under it (an ElevationService: the elevation tiles are fetched, decoded and kept here, once for all workers), both at
 * once. When they are there the job waits for a build worker that is free, the job that was asked for first goes first,
 * and gets the tile and the heights transferred. The build worker sends its result to the main thread itself and tells
 * this worker it is free again.
 *
 * So a tile needs the main thread only to be asked for and to be put on the GPU. Before, the main thread fetched the
 * tile, decoded the elevation tiles (8 ms and more each, in the middle of a drag) and sampled the heights, and handed
 * every job to its worker: while the page started or a frame took long, the workers stood still.
 *
 * Why not in the build workers: a worker that builds can't start a fetch, its thread is taken. The next tile of each
 * worker was then only asked for when the one before it was built, and the worker waited for the network every time
 * (measured: a jump to another city took 10% longer than with the fetches on the main thread). This worker is nearly
 * always free, and a tile it has ready goes to whichever build worker is.
 *
 * To the main thread (ElevationClient) it tells the height of the origin and the range of the heights of every elevation
 * tile it has loaded (for the tile-vs-frustum test of WorldTileStreamer).
 */

/** From the main thread. lat, lng: the projection center, where the ground is at 0 (ElevationService.setOrigin). */
export type FetchWorkerMessage =
  | { type: 'origin'; lat: number; lng: number }
  | { type: 'preload'; lat: number; lng: number }
  // the port the pool sends its jobs on, see FetchJobMessage
  | { type: 'jobs'; port: MessagePort }
  // the port of a build worker
  | { type: 'builder'; port: MessagePort };

/** To the main thread. origin: its height (m above sea level). range: of elevation tile x, y, m above sea level. */
export type FetchWorkerEvent =
  | { type: 'origin'; lat: number; lng: number; height: number }
  | { type: 'range'; x: number; y: number; min: number; max: number };

/** From the pool, on its port. limit: the build workers that build at once (the tiles are fetched whatever it is). */
export type FetchJobMessage =
  | { start: WorldWorkerRequest }
  | { cancel: number }
  // the job, asked for ahead of the view (WorldWorkerRequest.margin), is in view now: built before those that are not
  | { prioritize: number }
  | { limit: number };

/**
 * To the pool, for a job that ends here: it failed, or (empty) the server has no data for the tile and the ground is
 * flat, so there is nothing to build.
 */
export type FetchJobEvent = { id: number; error?: string; empty?: boolean };

// Minimal worker-scope typing, so the `webworker` lib (which clashes with the DOM lib of this project) isn't needed.
type WorkerCtx = {
  onmessage: ((e: MessageEvent<FetchWorkerMessage>) => void) | null;
  postMessage(message: FetchWorkerEvent): void;
};

const ctx = self as unknown as WorkerCtx;
const service = new ElevationService();
service.onTile = (x, y, min, max) => ctx.postMessage({ type: 'range', x, y, min, max });

// --- the heights

let originLat = NaN;
let originLng = NaN;

/**
 * The origin of the heights asked for from now on. A job of a place the map has left can set it back for a moment:
 * every grid is relative to the origin it was asked with, and the main thread only takes the height of the origin it
 * asked for last.
 */
function setOrigin(lat: number, lng: number): void {
  if (lat === originLat && lng === originLng) return;
  originLat = lat;
  originLng = lng;
  service.setOrigin(lat, lng);
  void service.originLoaded.then((height) => ctx.postMessage({ type: 'origin', lat, lng, height }));
}

/** The heights of the tile of a job (ElevationService.getTileGrid), relative to the ground at the center of its bounds. */
function getHeights(job: WorldWorkerRequest): Promise<TileGrid> {
  // the projection center, as MapProjection.setBounds has it: where MapWorld puts the origin
  setOrigin((job.bounds.south + job.bounds.north) * 0.5, (job.bounds.west + job.bounds.east) * 0.5);
  return service.getTileGrid(job.tile);
}

// --- the jobs

type Ready = { job: WorldWorkerRequest; pbf: ArrayBuffer; grid: TileGrid | null; fetchMs: number; ready: number; times: JobTimes };

let jobsPort: MessagePort | null = null;
const builders: { port: MessagePort; busy: boolean }[] = [];
let limit = Infinity;
// the jobs that wait for their data, by id: aborted when they are cancelled
const fetching = new Map<number, AbortController>();
// the jobs that wait for a build worker
const ready: Ready[] = [];
// the jobs of both, by id: for prioritize
const pending = new Map<number, WorldWorkerRequest>();

/** The vector tile, null when the server has none for it (open ocean at high zoom, ...). */
async function fetchTile(url: string, signal: AbortSignal, low: boolean): Promise<ArrayBuffer | null> {
  const response = await fetch(url, { signal, priority: low ? 'low' : 'auto' });
  if (response.status === 204 || response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.arrayBuffer();
}

function report(event: FetchJobEvent): void {
  jobsPort?.postMessage(event);
}

async function start(job: WorldWorkerRequest): Promise<void> {
  const began = performance.now();
  const times: JobTimes = { received: jobClock(), pbf: 0, heights: 0, ready: 0, dispatched: 0, buildStart: 0, buildEnd: 0 };
  const abort = new AbortController();
  fetching.set(job.id, abort);
  pending.set(job.id, job);
  try {
    // the heights of the ground, in parallel with the vector tile
    const [pbf, grid] = await Promise.all([
      fetchTile(job.url, abort.signal, job.margin).then((data) => (times.pbf = jobClock(), data)),
      job.elevation ? getHeights(job).then((data) => (times.heights = jobClock(), data)) : null,
    ]);
    // cancelled meanwhile: the pool has dropped it
    if (abort.signal.aborted) return;
    fetching.delete(job.id);
    // No data for this tile. On the terrain it still gets its ground.
    if (!pbf && !grid) {
      pending.delete(job.id);
      return void report({ id: job.id, empty: true });
    }
    const now = performance.now();
    times.ready = jobClock();
    ready.push({ job, pbf: pbf ?? new ArrayBuffer(0), grid, fetchMs: now - began, ready: now, times });
    dispatch();
  } catch (err: any) {
    if (abort.signal.aborted) return;
    fetching.delete(job.id);
    pending.delete(job.id);
    report({ id: job.id, error: String(err?.message ?? err) });
  }
}

function cancel(id: number): void {
  pending.delete(id);
  const abort = fetching.get(id);
  if (abort) {
    fetching.delete(id);
    abort.abort();
    return;
  }
  const index = ready.findIndex((r) => r.job.id === id);
  if (index >= 0) ready.splice(index, 1);
  // else it is with a build worker (or done): its result is dropped by the pool
}

/**
 * The ready jobs to the build workers that are free: the jobs of the view before those ahead of it (margin), the job that
 * was asked for first (the lowest id) first.
 */
function dispatch(): void {
  while (ready.length > 0) {
    const builder = builders.find((b, i) => i < limit && !b.busy);
    if (!builder) return;
    let first = 0;
    for (let i = 1; i < ready.length; i++) {
      const a = ready[i].job, b = ready[first].job;
      if (a.margin !== b.margin ? !a.margin : a.id < b.id) first = i;
    }
    const [next] = ready.splice(first, 1);
    pending.delete(next.job.id);
    builder.busy = true;
    next.times.dispatched = jobClock();
    const heights = next.grid?.heights ?? null;
    const message: WorldBuildMessage = {
      job: next.job, pbf: next.pbf, heights, seaLevel: next.grid?.seaLevel ?? 0, fetchMs: next.fetchMs,
      waitMs: performance.now() - next.ready, times: next.times,
    };
    builder.port.postMessage(message, heights ? [next.pbf, heights.buffer] : [next.pbf]);
  }
}

ctx.onmessage = (e: MessageEvent<FetchWorkerMessage>) => {
  const msg = e.data;
  if (msg.type === 'origin') setOrigin(msg.lat, msg.lng);
  else if (msg.type === 'preload') service.preload(msg.lat, msg.lng);
  else if (msg.type === 'jobs') {
    jobsPort = msg.port;
    jobsPort.onmessage = (event: MessageEvent<FetchJobMessage>) => {
      const m = event.data;
      if ('start' in m) void start(m.start);
      else if ('cancel' in m) cancel(m.cancel);
      else if ('prioritize' in m) {
        const job = pending.get(m.prioritize);
        if (job) job.margin = false;
      }
      else {
        limit = m.limit;
        dispatch();
      }
    };
  } else {
    const builder = { port: msg.port, busy: false };
    builders.push(builder);
    // it built its tile (and sent it to the main thread)
    builder.port.onmessage = (_event: MessageEvent<WorldBuildDone>) => {
      builder.busy = false;
      dispatch();
    };
    dispatch();
  }
};
