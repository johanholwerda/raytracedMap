import { buildWorldTile } from './worldGeometryCore';
import type { WorldTileResult } from './worldGeometryCore';
import { jobClock } from './jobTimes';
import type { JobTimes } from './jobTimes';
import type { LatLngBounds } from '../osm/osmGeometryCore';
import type { TileCoord } from '../osm/PDOKTileService';

/**
 * World map Web Worker: decodes one OpenMapTiles vector tile (.pbf) and builds its ground layers and/or extruded
 * buildings off the main thread. The tile and the heights of the ground under it come from the tile fetch worker
 * (tileFetchWorker.ts) over a port, transferred in. The result buffers are transferred to the main thread, the heights
 * with them (for TerrainHeightField), and the fetch worker is told this worker is free for the next tile.
 */

/** A tile to make: what WorldWorkerPool asks the fetch worker for. */
export type WorldWorkerRequest = {
  id: number;
  // the vector tile, see OpenFreeMapTileService.tileUrl
  url: string;
  tile: TileCoord;
  bounds: LatLngBounds;
  wantTerrain: boolean;
  wantBuildings: boolean;
  // the trees of the wood areas, with the buildings, see worldTrees.ts
  wantTrees: boolean;
  // the ground has heights (m, see ElevationService.getTileGrid), relative to the ground at the center of bounds. False: flat
  elevation: boolean;
  // asked for ahead of the view (WorldTileStreamer.PREFETCH_MARGIN): fetched at a low priority, built after the jobs of
  // the view, until the pool says it is in view (FetchJobMessage prioritize)
  margin: boolean;
};

/** From the main thread, once: the port to the fetch worker. */
export type WorldWorkerSetup = { port: MessagePort };

/**
 * From the fetch worker: a job with its data. pbf: empty when the server has no tile. heights: null without elevation.
 * For profiling: the ms the data took and the ms the job then waited for this worker.
 */
export type WorldBuildMessage = {
  job: WorldWorkerRequest;
  pbf: ArrayBuffer;
  heights: Float32Array | null;
  // the height of the sea in heights (m, relative to the same origin), see seaMask.ts
  seaLevel: number;
  fetchMs: number;
  waitMs: number;
  times: JobTimes;
};

/** To the fetch worker: the job is built. */
export type WorldBuildDone = { done: number };

/** To the main thread. heights: the ones the tile is built on. For profiling: fetchMs, waitMs and the ms of the build. */
export type WorldWorkerReply = {
  id: number;
  result?: WorldTileResult;
  heights?: Float32Array | null;
  error?: string;
  fetchMs: number;
  waitMs: number;
  buildMs: number;
  times: JobTimes;
};

// Minimal worker-scope typing (see osm/mapWorker.ts for why).
type WorkerCtx = {
  onmessage: ((e: MessageEvent<WorldWorkerSetup>) => void) | null;
  postMessage(message: WorldWorkerReply, transfer?: Transferable[]): void;
};

const ctx = self as unknown as WorkerCtx;

function build(msg: WorldBuildMessage, port: MessagePort): void {
  const { job, heights, times } = msg;
  const start = performance.now();
  times.buildStart = jobClock();
  try {
    const result: WorldTileResult = buildWorldTile(
      new Uint8Array(msg.pbf),
      job.tile,
      job.bounds,
      job.wantTerrain,
      job.wantBuildings,
      job.wantTrees,
      heights,
      msg.seaLevel,
    );
    const transfers: Transferable[] = [];
    if (result.terrain) for (const l of Object.values(result.terrain)) transfers.push(l.vertices.buffer, l.indices.buffer);
    if (result.ground) transfers.push(result.ground.vertices.buffer, result.ground.indices.buffer);
    if (result.roads) transfers.push(result.roads.buffer);
    if (result.buildings) {
      const b = result.buildings;
      transfers.push(b.vertices.buffer, b.lightmap.buffer, b.interleaved, b.indices.buffer);
      if (result.buildings.trace) transfers.push(result.buildings.trace.words.buffer, result.buildings.trace.ids.buffer);
    }
    if (heights) transfers.push(heights.buffer);
    times.buildEnd = jobClock();
    ctx.postMessage({ id: job.id, result, heights, fetchMs: msg.fetchMs, waitMs: msg.waitMs, buildMs: performance.now() - start, times }, transfers);
  } catch (err: any) {
    times.buildEnd = jobClock();
    ctx.postMessage({ id: job.id, error: String(err?.message ?? err), fetchMs: msg.fetchMs, waitMs: msg.waitMs, buildMs: performance.now() - start, times });
  }
  const done: WorldBuildDone = { done: job.id };
  port.postMessage(done);
}

ctx.onmessage = (e: MessageEvent<WorldWorkerSetup>) => {
  const port = e.data.port;
  port.onmessage = (message: MessageEvent<WorldBuildMessage>) => build(message.data, port);
};
