import { MeshoptDecoder } from './vendor/meshopt_decoder.module';
import { CityGeometryBuilder } from './cityGlTFCore';
import { decodePdokTile } from './pdokDecodeCore';
import { buildOSMPositions } from './osmGeometryCore';
import type { LayerKey } from './osmGeometryCore';
import type { LatLngBounds } from './OSMBuilder';
import type { TileCoord } from './PDOKTileService';

/**
 * Unified map Web Worker: a single fungible worker that runs *any* of the
 * heavy, renderer-free map build jobs off the main thread, so one shared pool
 * ({@link MapWorkerPool}) can hand whatever work is queued to whichever worker
 * is free — buildings and terrain compete for the same cores-1 threads instead
 * of each owning a fixed slice.
 *
 *  - 'build'       : 3DBAG building .glb  -> de-indexed position soup + AABB
 *  - 'terrain3D'   : 3D Basisvoorziening terrain .glb -> per-category positions
 *  - 'pdokTerrain' : PDOK BRT-A vector tile .pbf -> per-layer positions
 *
 * Jobs are pure CPU: the .glb / .pbf bytes are fetched on the main thread and
 * transferred in (zero-copy), so a worker is never parked waiting on the
 * network and can't starve the pool. The meshopt WASM is instantiated lazily,
 * so workers that only ever run 'pdokTerrain' never pay for it.
 *
 * One {@link CityGeometryBuilder} per worker so its scratch buffers are reused
 * across tiles (no per-tile multi-MB allocation / GC churn).
 */

export type PdokTerrainResult = {
  // True for out-of-coverage tiles or tiles with no geometry — the main thread
  // records them as loaded so they aren't re-requested.
  empty: boolean;
  positions?: Record<LayerKey, Float32Array>;
};

export type MapWorkerRequest =
  | { type: 'build'; id: number; glb: ArrayBuffer; bounds: LatLngBounds; flatten: boolean }
  | { type: 'terrain3D'; id: number; glb: ArrayBuffer; bounds: LatLngBounds }
  | { type: 'pdokTerrain'; id: number; pbf: ArrayBuffer; tile: TileCoord; bounds: LatLngBounds };

// Minimal worker-scope typing so we don't need to pull in the `webworker` lib
// (which would clash with this project's DOM lib). MessageEvent/Transferable
// come from the DOM lib that is already in scope.
type WorkerCtx = {
  onmessage: ((e: MessageEvent<MapWorkerRequest>) => void) | null;
  postMessage(message: any, transfer?: Transferable[]): void;
};

const ctx = self as unknown as WorkerCtx;
const builder = new CityGeometryBuilder();

ctx.onmessage = async (e: MessageEvent<MapWorkerRequest>) => {
  const msg = e.data;
  try {
    if (msg.type === 'build') {
      // The meshopt WASM is instantiated lazily and shared for this worker's life.
      await MeshoptDecoder.ready;
      const result = builder.buildPositions(msg.glb, msg.bounds, msg.flatten);
      ctx.postMessage({ id: msg.id, result }, [result.vertices.buffer]);
    } else if (msg.type === 'terrain3D') {
      await MeshoptDecoder.ready;
      const result = builder.buildTerrainPositions(msg.glb, msg.bounds);
      const transfers = Object.values(result).map((a) => a.buffer);
      ctx.postMessage({ id: msg.id, result }, transfers);
    } else {
      // PDOK MVT decode + earcut triangulation — no meshopt needed.
      const elements = decodePdokTile(new Uint8Array(msg.pbf), msg.tile);
      if (elements.length === 0) {
        ctx.postMessage({ id: msg.id, result: { empty: true } as PdokTerrainResult });
        return;
      }
      const positions = buildOSMPositions(msg.bounds, elements);
      const result: PdokTerrainResult = { empty: false, positions };
      const transfers = Object.values(positions).map((a) => a.buffer);
      ctx.postMessage({ id: msg.id, result }, transfers);
    }
  } catch (err: any) {
    ctx.postMessage({ id: msg.id, error: String(err?.message ?? err) });
  }
};
