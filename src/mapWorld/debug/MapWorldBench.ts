import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Profiler from 'webGPU-framework/renderer/render/Profiler';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';

/** What MapRenderer.render did in a frame, see MapWorldBench.afterFrame. */
export type MapWorldFrameState = {
  // a new jittered gBuffer was drawn (the map or the AO took a frame)
  newJitter: boolean;
  lightmapUpdating: boolean;
  ssaoAccumulating: boolean;
  // the map was shaded and resolved by the TAA
  renderMap: boolean;
  tilesLoading: boolean;
  cameraMoving: boolean;
};

type Phase = 'load' | 'drag' | 'settle' | 'done';

type Bucket = {
  frames: number;
  // frames with each state flag, and the last frame (from 1) it was set in
  flags: Record<string, number>;
  last: Record<string, number>;
  // profiler readbacks that arrived in the phase, and the sum of their pass times per label
  samples: number;
  gpuMs: number;
  passMs: Record<string, number>;
  // the readbacks each label was in
  passN: Record<string, number>;
  // every 16th frame: the part of the cells that still bid on a texel of the lightmap that takes samples, in 1e-4
  wanted: number[];
};

/**
 * A benchmark that runs by itself (?bench, with ?uncapped so it runs in a hidden tab): MapWorld steps its time by
 * 1 / 60 s a frame, waits until the city loaded and the map stopped drawing, then pans the camera for dragFrames
 * frames and counts the frames until it stops drawing again, rounds times, back and forth. The GPU time of each pass
 * comes from the Profiler. The results go to window.__mapWorldBench and the console.
 * Url params: bench=rounds (default 4), dragFrames (30), dragSpeed (camera distances per frame, .01), dragDir=same
 * (every drag the same way, onto new tiles, instead of back and forth), set (see applyParams).
 */
export default class MapWorldBench {
  private renderer: Renderer;
  private frame = 0;
  private phase: Phase = 'load';
  private phaseFrames = 0;
  private idleFrames = 0;
  private round = 0;
  private readonly rounds: number;
  private readonly dragFrames: number;
  private readonly dragSpeed: number;
  //every drag goes the same way, onto ground that wasn't loaded yet
  private readonly sameDirection: boolean;
  private bucket: Bucket = MapWorldBench.createBucket();
  private readonly results: { phase: string; bucket: Bucket }[] = [];
  private readonly startTime = performance.now();

  private readonly params: URLSearchParams;

  constructor(renderer: Renderer, params: URLSearchParams) {
    this.renderer = renderer;
    this.params = params;
    this.rounds = parseInt(params.get('bench') || '4') || 4;
    this.dragFrames = parseInt(params.get('dragFrames') || '30');
    this.dragSpeed = parseFloat(params.get('dragSpeed') || '.01');
    this.sameDirection = params.get('dragDir') === 'same';
    Profiler.setEnabled(true);
    renderer.profiler.addFrameCallback((totalMs, labels, passMs) => {
      const b = this.bucket;
      b.samples++;
      b.gpuMs += totalMs;
      const seen = new Set<string>();
      for (let i = 0; i < labels.length; i++) {
        b.passMs[labels[i]] = (b.passMs[labels[i]] || 0) + passMs[i];
        if (!seen.has(labels[i])) b.passN[labels[i]] = (b.passN[labels[i]] || 0) + 1;
        seen.add(labels[i]);
      }
    });
    (window as any).__mapWorldBench = { status: 'load', results: this.results };
  }

  /**
   * ?set=skyLightmap.traceSamplesMax:1024,ssao.amount:0: param values for A/B runs, by the path of the param (see
   * MapWorld.findParam), set before the first frame.
   */
  public applyParams(findParam: (path: string) => any): void {
    const set = this.params.get('set');
    if (!set) return;
    for (const entry of set.split(',')) {
      const [path, value] = entry.split(':');
      const target = findParam(path);
      if (!target?.setValue) {
        console.warn(`[MapWorldBench] no param ${path}`);
        continue;
      }
      target.setValue(value === 'true' ? true : value === 'false' ? false : parseFloat(value));
    }
  }

  private static createBucket(): Bucket {
    return { frames: 0, flags: {}, last: {}, samples: 0, gpuMs: 0, passMs: {}, passN: {}, wanted: [] };
  }

  /** The time of this frame for renderer.update: 60 fps, so the time based logic doesn't depend on the GPU. */
  public nextTime(): number {
    return ++this.frame * 1000 / 60;
  }

  /** Before camera.update: the pan of the drag. */
  public moveCamera(camera: MapCamera): void {
    if (this.phase !== 'drag') return;
    const right = camera.pivot.worldRight;
    const direction = new Vector3(right.x, 0, right.z).normalize();
    const sign = this.sameDirection || this.round % 2 === 0 ? 1 : -1;
    camera.pivot.translate(direction.multiplyScalar(sign * this.dragSpeed * camera.distance));
  }

  /** wanted: the part of the cells that bid on a texel of the lightmap that takes samples. */
  public afterFrame(state: MapWorldFrameState, tilesIdle: boolean, wanted: number): void {
    const b = this.bucket;
    if (b.frames % 16 === 0) b.wanted.push(Math.round(wanted * 1e4));
    b.frames++;
    for (const [key, value] of Object.entries(state)) {
      if (!value) continue;
      b.flags[key] = (b.flags[key] || 0) + 1;
      b.last[key] = b.frames;
    }
    this.phaseFrames++;
    const idle = !state.newJitter && !state.lightmapUpdating && !state.renderMap;

    switch (this.phase) {
      case 'load':
        this.idleFrames = idle && tilesIdle ? this.idleFrames + 1 : 0;
        if (this.idleFrames >= 30 || this.phaseFrames > 6000) this.next('drag');
        break;
      case 'drag':
        if (this.phaseFrames >= this.dragFrames) this.next('settle');
        break;
      case 'settle':
        if (idle || this.phaseFrames > 3000) {
          this.round++;
          this.next(this.round < this.rounds ? 'drag' : 'done');
        }
        break;
    }
  }

  private next(phase: Phase): void {
    this.results.push({ phase: this.phase === 'load' ? 'load' : `${this.phase}${this.round - (this.phase === 'settle' ? 1 : 0)}`, bucket: this.bucket });
    this.bucket = MapWorldBench.createBucket();
    this.phase = phase;
    this.phaseFrames = 0;
    if (phase === 'done') this.finish();
  }

  private finish(): void {
    const canvas = this.renderer.canvas;
    const summary = this.summarize();
    const out = { canvas: `${canvas.width}x${canvas.height}`, seconds: (performance.now() - this.startTime) / 1000, ...summary };
    (window as any).__mapWorldBench = { status: 'done', summary: out, results: this.results };
    console.log('[MapWorldBench] ' + JSON.stringify(out));
  }

  //the settle phases: frames until the map stopped drawing, the GPU ms per frame and per pass (a mean per frame) over all
  private summarize() {
    const settles = this.results.filter(r => r.phase.startsWith('settle'));
    const drags = this.results.filter(r => r.phase.startsWith('drag'));
    const settleFrames = settles.map(r => r.bucket.frames);
    const perFrame = (buckets: Bucket[]) => {
      const samples = buckets.reduce((s, b) => s + b.samples, 0) || 1;
      const passes: Record<string, number> = {};
      const counts: Record<string, number> = {};
      for (const b of buckets) for (const [k, v] of Object.entries(b.passMs)) passes[k] = (passes[k] || 0) + v;
      for (const b of buckets) for (const [k, v] of Object.entries(b.passN)) counts[k] = (counts[k] || 0) + v;
      //per frame that has the pass
      const sorted = Object.entries(passes).map(([k, v]) => [k, +(v / counts[k]).toFixed(3)] as [string, number]).sort((a, b) => b[1] - a[1]);
      return { gpuMs: +(buckets.reduce((s, b) => s + b.gpuMs, 0) / samples).toFixed(3), passes: Object.fromEntries(sorted) };
    };
    const flags: Record<string, number> = {};
    for (const r of settles) for (const [k, v] of Object.entries(r.bucket.flags)) flags[k] = (flags[k] || 0) + v / settles.length;
    const last: Record<string, number[]> = {};
    for (const r of settles) for (const k of Object.keys(r.bucket.flags)) (last[k] ||= []).push(r.bucket.last[k]);
    return {
      settleFrames,
      // the GPU time of all settle frames together, per settle
      settleGpuMs: settles.map(r => +(r.bucket.gpuMs / Math.max(r.bucket.samples, 1) * r.bucket.frames).toFixed(1)),
      settleFlags: flags,
      settleLast: last,
      settle: perFrame(settles.map(r => r.bucket)),
      drag: perFrame(drags.map(r => r.bucket)),
    };
  }
}
