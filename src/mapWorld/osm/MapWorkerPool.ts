import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type { LatLngBounds } from './OSMBuilder';
import type { TileCoord } from './PDOKTileService';
import type { CityPositionsResult, TerrainCategory } from './cityGlTFCore';
import type { MapWorkerRequest, PdokTerrainResult } from './mapWorker';

/**
 * One shared pool of fungible {@link mapWorker} Web Workers, sized to leave a
 * single core for the main thread (`hardwareConcurrency - 1`). Every worker can
 * run any job type (building / 3D-terrain / PDOK-terrain), and jobs are handed
 * to the next *free* worker — so the building↔terrain split is purely
 * demand-driven. With N free workers and a queue of mixed jobs you naturally get
 * any combination (6 build + 1 terrain, 1 + 6, …) up to N concurrent, instead of
 * each kind owning a fixed slice of the cores.
 *
 * Workers are pure CPU: the caller fetches the .glb / .pbf bytes (off-frame,
 * async) and transfers them in, so a worker is never blocked on the network and
 * can't starve the pool. Result buffers are transferred back zero-copy.
 */

type JobType = MapWorkerRequest['type'];
type Pending = { resolve: (v: any) => void; reject: (e: any) => void; type: JobType };
type Queued = { req: MapWorkerRequest; transfer: Transferable[] };

export default class MapWorkerPool {
  private workers: Worker[] = [];
  private idle: Worker[] = [];
  private queue: Queued[] = [];
  private pending = new Map<number, Pending>();
  private nextId = 1;
  // Outstanding jobs (queued + running) per type, for the streaming HUD.
  private active: Record<JobType, number> = { build: 0, terrain3D: 0, pdokTerrain: 0 };

  constructor(size?: number) {
    const hw = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
    // Use every core but one (reserved for the main/render thread). Clamp to >=1
    // so single-core devices still get a worker.
    const count = Math.max(1, size ?? hw - 1);
    for (let i = 0; i < count; i++) {
      const worker = new Worker(new URL('./mapWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e: MessageEvent) => this.onMessage(worker, e.data);
      worker.onerror = (e) => LogGL.warn('[MapOSM] map worker error:', e.message);
      this.workers.push(worker);
      this.idle.push(worker);
    }
  }

  /** Total worker count (= cores - 1). */
  public get size(): number {
    return this.workers.length;
  }

  /** Outstanding building-tile jobs (queued + running). */
  public get buildingJobs(): number {
    return this.active.build;
  }

  /** Outstanding terrain jobs (PDOK + 3D Basisvoorziening, queued + running). */
  public get terrainJobs(): number {
    return this.active.pdokTerrain + this.active.terrain3D;
  }

  /** Total outstanding jobs across all types. */
  public get activeJobs(): number {
    return this.pending.size;
  }

  /** Build a city tile's de-indexed positions (+ AABB). Transfers `glb` in. */
  public buildCity(glb: ArrayBuffer, bounds: LatLngBounds, flatten: boolean): Promise<CityPositionsResult> {
    return this.dispatch({ type: 'build', id: 0, glb, bounds, flatten }, [glb]);
  }

  /** Build a 3D Basisvoorziening terrain tile's per-category positions. Transfers `glb` in. */
  public buildTerrain3D(glb: ArrayBuffer, bounds: LatLngBounds): Promise<Record<TerrainCategory, Float32Array>> {
    return this.dispatch({ type: 'terrain3D', id: 0, glb, bounds }, [glb]);
  }

  /** Decode + triangulate a PDOK vector tile's already-fetched .pbf bytes. Transfers `pbf` in. */
  public buildPdokTerrain(pbf: ArrayBuffer, tile: TileCoord, bounds: LatLngBounds): Promise<PdokTerrainResult> {
    return this.dispatch({ type: 'pdokTerrain', id: 0, pbf, tile, bounds }, [pbf]);
  }

  private dispatch<T>(req: MapWorkerRequest, transfer: Transferable[]): Promise<T> {
    const id = this.nextId++;
    req.id = id;
    this.active[req.type]++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, type: req.type });
      const worker = this.idle.pop();
      if (worker) worker.postMessage(req, transfer);
      else this.queue.push({ req, transfer });
    });
  }

  private onMessage(worker: Worker, data: { id: number; result?: any; error?: string }): void {
    const job = this.pending.get(data.id);
    if (job) {
      this.pending.delete(data.id);
      this.active[job.type]--;
    }

    // The worker is now free: hand it the next queued job, else return it to the
    // idle set. (Done before resolving so a synchronous re-dispatch from the
    // resolve callback sees a consistent pool state.)
    const next = this.queue.shift();
    if (next) worker.postMessage(next.req, next.transfer);
    else this.idle.push(worker);

    if (!job) return;
    if (data.error) job.reject(new Error(data.error));
    else job.resolve(data.result);

    // console.log('workers active', this.activeJobs, 'queued', this.queue.length);
  }

  public destroy(): void {
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
    this.idle = [];
    this.queue = [];
    this.pending.clear();
  }
}
