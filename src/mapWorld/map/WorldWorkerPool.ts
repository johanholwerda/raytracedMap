import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type { LatLngBounds } from '../../mapOSM/osmGeometryCore';
import type { TileCoord } from '../../mapOSM/PDOKTileService';
import type { WorldTileResult } from './worldGeometryCore';
import type { WorldWorkerRequest } from './worldMapWorker';

/**
 * Pool of {@link worldMapWorker} Web Workers (`hardwareConcurrency - 1`, one
 * core left for the render thread). Jobs go to the next free worker; the .pbf
 * is fetched by the caller and transferred in, so workers never wait on the
 * network. Same design as mapOSM/MapWorkerPool, with a single job type.
 */

type Pending = { resolve: (v: WorldTileResult) => void; reject: (e: any) => void };
type Queued = { req: WorldWorkerRequest; transfer: Transferable[] };

export default class WorldWorkerPool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Queued[] = [];
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(size?: number) {
    const hw = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
    const count = Math.max(1, size ?? hw - 1);
    for (let i = 0; i < count; i++) {
      const worker = new Worker(new URL('./worldMapWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e: MessageEvent) => this.onMessage(worker, e.data);
      worker.onerror = (e) => LogGL.warn('[MapWorld] worker error:', e.message);
      this.workers.push(worker);
      this.idle.push(worker);
    }
  }

  public get size(): number {
    return this.workers.length;
  }

  /** Outstanding jobs (queued + running). */
  public get activeJobs(): number {
    return this.pending.size;
  }

  /** Decode + build one tile. Transfers `pbf` in. */
  public buildTile(
    pbf: ArrayBuffer,
    tile: TileCoord,
    bounds: LatLngBounds,
    wantTerrain: boolean,
    wantBuildings: boolean,
    wantTrees: boolean,
  ): Promise<WorldTileResult> {
    const id = this.nextId++;
    const req: WorldWorkerRequest = { id, pbf, tile, bounds, wantTerrain, wantBuildings, wantTrees };
    return new Promise<WorldTileResult>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      const worker = this.idle.pop();
      if (worker) worker.postMessage(req, [pbf]);
      else this.queue.push({ req, transfer: [pbf] });
    });
  }

  private onMessage(worker: Worker, data: { id: number; result?: WorldTileResult; error?: string }): void {
    const job = this.pending.get(data.id);
    this.pending.delete(data.id);

    const next = this.queue.shift();
    if (next) worker.postMessage(next.req, next.transfer);
    else this.idle.push(worker);

    if (!job) return;
    if (data.error) job.reject(new Error(data.error));
    else job.resolve(data.result!);
  }

  public destroy(): void {
    for (const worker of this.workers) worker.terminate();
    for (const job of this.pending.values()) job.reject(new Error('worker pool destroyed'));
    this.workers = [];
    this.idle = [];
    this.queue = [];
    this.pending.clear();
  }
}
