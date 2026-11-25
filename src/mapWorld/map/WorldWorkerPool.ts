import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type { WorldTileResult } from './worldGeometryCore';
import type { WorldWorkerReply, WorldWorkerRequest, WorldWorkerSetup } from './worldMapWorker';
import type { FetchJobEvent, FetchJobMessage } from './tileFetchWorker';
import type ElevationClient from './ElevationClient';

/**
 * Pool of {@link worldMapWorker} Web Workers that build the map tiles, fed by the tile fetch worker
 * (tileFetchWorker.ts, started by the ElevationClient). A job goes to the fetch worker, which fetches the .pbf and makes
 * the heights of the ground, and hands both to the next build worker that is free; that one sends the built tile here.
 * So a tile needs the main thread only to be asked for and to be put on the GPU, and the workers go on while the main
 * thread is held up (the page starting, a long frame).
 *
 * The number of build workers: see {@link defaultWorkerCount}.
 */

/** A tile to make, see WorldWorkerRequest. */
export type WorldTileJob = Omit<WorldWorkerRequest, 'id'>;

/** A made tile: null result when the server has no data for it and the ground is flat. heights: with elevation. */
export type WorldTileBuilt = { result: WorldTileResult | null; heights: Float32Array | null };

type Pending = { resolve: (v: WorldTileBuilt) => void; reject: (e: any) => void };

export default class WorldWorkerPool {
  /** What the promise of a cancelled job rejects with, see cancel. */
  public static readonly CANCELLED = new Error('cancelled');
  // The jobs under way at once per build worker, and at least (see capacity): about half of the time of a job is
  // waiting for its data, with as many jobs as workers they were idle half the time
  private static readonly JOBS_PER_WORKER = 2;
  private static readonly JOBS_MIN = 8;

  private workers: Worker[] = [];
  // the jobs under way, by id
  private pending = new Map<number, Pending>();
  private nextId = 1;
  // the port to the fetch worker the jobs go over
  private jobs: MessagePort;
  // the build workers that build at once, see activeLimit
  private limit = Infinity;
  /**
   * For profiling: the ms the workers built together and the jobs done, the ms the jobs waited for a build worker
   * together once their data was there, and the ms that data took (the fetch and the heights).
   */
  public readonly stats = { busyMs: 0, jobs: 0, queueMs: 0, fetchMs: 0 };

  /**
   * The build workers of a pool without a size: one per core but two (the main thread's and the fetch worker's), 12 at
   * most. Building a tile is mostly writing memory, so they hardly scale with the cores. Measured on 8 cores, a jump
   * to another city (31 tiles, the mean of 3 cities): 1.53 s with 4 workers, 1.53 with 5, 1.38 with 6, 1.46 with 7,
   * while the build of a tile went from 178 ms with 4 to 273 ms with 7. So the count matters little, and what there
   * is to gain goes to what the map is used for most, a new place and a fast drag. While the page starts they are held
   * back, see activeLimit.
   */
  public static get defaultWorkerCount(): number {
    const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
    return Math.min(Math.max(cores - 2, 1), 12);
  }

  /** source: its worker fetches the tiles and makes their heights, see tileFetchWorker.ts. */
  constructor(size: number | undefined, source: ElevationClient) {
    this.jobs = source.createPort('jobs');
    this.jobs.onmessage = (e: MessageEvent<FetchJobEvent>) => {
      // a job that ended in the fetch worker
      if (e.data.error) this.settle(e.data.id, new Error(e.data.error), null);
      else this.settle(e.data.id, null, { result: null, heights: null });
    };
    const count = Math.max(1, size ?? WorldWorkerPool.defaultWorkerCount);
    for (let i = 0; i < count; i++) {
      const worker = new Worker(new URL('./worldMapWorker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (e: MessageEvent<WorldWorkerReply>) => this.onBuilt(e.data);
      worker.onerror = (e) => LogGL.warn('[MapWorld] worker error:', e.message);
      const port = source.createPort('builder');
      worker.postMessage({ port } satisfies WorldWorkerSetup, [port]);
      this.workers.push(worker);
    }
  }

  public get size(): number {
    return this.workers.length;
  }

  /**
   * The jobs to have under way at once: all of them are fetched right away, and built as the workers come free. More
   * wait with the caller, which can still drop them for free when the camera moves on.
   */
  public get capacity(): number {
    return Math.max(WorldWorkerPool.JOBS_MIN, this.workers.length * WorldWorkerPool.JOBS_PER_WORKER);
  }

  /** Outstanding jobs (being fetched, waiting for a worker or being built). */
  public get activeJobs(): number {
    return this.pending.size;
  }

  // For the profilers (video/dragProfile.mjs), which count the workers at work and the jobs that wait for one: the
  // fetch worker deals the jobs out, here only their number is known. As if every job under way had a worker.
  public get idle(): undefined[] {
    return new Array(Math.max(this.workers.length - this.pending.size, 0));
  }

  public get queue(): undefined[] {
    return new Array(Math.max(this.pending.size - this.workers.length, 0));
  }

  /**
   * The most build workers that build at once: fewer than the pool while the page starts (see MapWorld.startTiles),
   * when the GPU process needs the cores. All of them by default, 0: none. The tiles are fetched whatever the limit:
   * when it goes up their data is there.
   */
  public get activeLimit(): number {
    return Math.min(this.limit, this.workers.length);
  }

  public set activeLimit(value: number) {
    this.limit = Math.max(0, value);
    this.send({ limit: this.limit });
  }

  /**
   * Makes one tile. done rejects with CANCELLED when the job is cancelled (by its id), with an Error when the tile
   * can't be fetched or built.
   */
  public start(job: WorldTileJob): { id: number; done: Promise<WorldTileBuilt> } {
    const id = this.nextId++;
    const done = new Promise<WorldTileBuilt>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    this.send({ start: { id, ...job } });
    return { id, done };
  }

  /**
   * The job is not wanted anymore: its fetch is aborted, and it is not built when no worker has started on it. Its
   * promise rejects with CANCELLED. A job that is being built is finished, its result is dropped.
   */
  public cancel(id: number): void {
    if (!this.pending.has(id)) return;
    this.settle(id, WorldWorkerPool.CANCELLED, null);
    this.send({ cancel: id });
  }

  private onBuilt(data: WorldWorkerReply): void {
    this.stats.busyMs += data.buildMs;
    this.stats.queueMs += data.waitMs;
    this.stats.fetchMs += data.fetchMs;
    this.stats.jobs++;
    if (data.error) this.settle(data.id, new Error(data.error), null);
    else this.settle(data.id, null, { result: data.result ?? null, heights: data.heights ?? null });
  }

  /** The promise of a job, when it is still wanted (not cancelled). */
  private settle(id: number, error: Error | null, built: WorldTileBuilt | null): void {
    const job = this.pending.get(id);
    if (!job) return;
    this.pending.delete(id);
    if (error) job.reject(error);
    else job.resolve(built!);
  }

  private send(message: FetchJobMessage): void {
    this.jobs.postMessage(message);
  }

  public destroy(): void {
    for (const worker of this.workers) worker.terminate();
    for (const job of this.pending.values()) job.reject(new Error('worker pool destroyed'));
    this.workers = [];
    this.pending.clear();
  }
}
