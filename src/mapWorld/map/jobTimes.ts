/**
 * When a map tile job got to each step, for the tile timings (WorldTileStreamer.timings): ms since the epoch with the
 * precision of performance.now (see jobClock), so the times of the workers and the main thread compare. 0: not reached
 * (no heights without elevation).
 * - received: the job is in the fetch worker (tileFetchWorker.ts)
 * - pbf, heights: the vector tile and the heights of the ground are there. ready: both
 * - dispatched: sent to a build worker (worldMapWorker.ts), which started on it (buildStart) and sent it on (buildEnd)
 */
export type JobTimes = {
  received: number;
  pbf: number;
  heights: number;
  ready: number;
  dispatched: number;
  buildStart: number;
  buildEnd: number;
};

/** The clock of JobTimes: the same in every worker and the main thread. */
export function jobClock(): number {
  return performance.timeOrigin + performance.now();
}
