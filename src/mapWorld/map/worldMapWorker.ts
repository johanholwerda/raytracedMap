import { buildWorldTile } from './worldGeometryCore';
import type { WorldTileResult } from './worldGeometryCore';
import type { LatLngBounds } from '../../mapOSM/osmGeometryCore';
import type { TileCoord } from '../../mapOSM/PDOKTileService';

/**
 * World map Web Worker: decodes one OpenMapTiles vector tile (.pbf, fetched on
 * the main thread and transferred in) and builds its ground layers and/or
 * extruded buildings off the main thread. Result buffers are transferred back.
 */

export type WorldWorkerRequest = {
  id: number;
  pbf: ArrayBuffer;
  tile: TileCoord;
  bounds: LatLngBounds;
  wantTerrain: boolean;
  wantBuildings: boolean;
  // the trees of the wood areas, with the buildings, see worldTrees.ts
  wantTrees: boolean;
  // the heights of the ground (m), see ElevationService.getTileGrid. Null: flat
  heights: Float32Array | null;
};

// Minimal worker-scope typing (see mapOSM/mapWorker.ts for why).
type WorkerCtx = {
  onmessage: ((e: MessageEvent<WorldWorkerRequest>) => void) | null;
  postMessage(message: any, transfer?: Transferable[]): void;
};

const ctx = self as unknown as WorkerCtx;

ctx.onmessage = (e: MessageEvent<WorldWorkerRequest>) => {
  const msg = e.data;
  try {
    const result: WorldTileResult = buildWorldTile(
      new Uint8Array(msg.pbf),
      msg.tile,
      msg.bounds,
      msg.wantTerrain,
      msg.wantBuildings,
      msg.wantTrees,
      msg.heights,
    );
    const transfers: Transferable[] = [];
    if (result.terrain) for (const l of Object.values(result.terrain)) transfers.push(l.vertices.buffer, l.indices.buffer);
    if (result.ground) transfers.push(result.ground.vertices.buffer, result.ground.indices.buffer);
    if (result.roads) transfers.push(result.roads.buffer);
    if (result.buildings) {
      transfers.push(result.buildings.vertices.buffer, result.buildings.lightmap.buffer, result.buildings.indices.buffer);
      if (result.buildings.trace) transfers.push(result.buildings.trace.words.buffer, result.buildings.trace.ids.buffer);
    }
    ctx.postMessage({ id: msg.id, result }, transfers);
  } catch (err: any) {
    ctx.postMessage({ id: msg.id, error: String(err?.message ?? err) });
  }
};
