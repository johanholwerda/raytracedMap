
// The total GPU time of a frame, and the label and the time in ms of each of its passes.
export type ProfilerFrameCallback = (totalMs: number, labels: string[], passMs: number[]) => void;

export default class Profiler {
    // Shared toggle state, so the gui switch survives a renderer (re)creation.
    private static enabledGlobal: boolean = false;
    private static instances: Profiler[] = [];

    public static setEnabled(enabled: boolean) {
        Profiler.enabledGlobal = enabled;
        for (const profiler of Profiler.instances) {
            profiler.enabled = enabled;
            if (!enabled) profiler.lastActiveTime = 0;
        }
    }

    public static isEnabled(): boolean {
        return Profiler.enabledGlobal;
    }

    public enabled: boolean = false;
    private querySet: GPUQuerySet | null = null;
    private queryResolveBuffer: GPUBuffer | null = null;
    private currentQueryIndex: number = 0;
    private queryLabels: string[] = [];
    private readonly MAX_QUERIES = 256;
    private readonly BUFFER_POOL_SIZE = 3;
    private device!: GPUDevice;
    private debugDiv: HTMLDivElement | null = null;
    private lastActiveTime: number = 0;

    private readBufferPool: GPUBuffer[] = [];
    private readBufferAvailable: boolean[] = [];
    private resolveFrameCounter: number = 0;
    private displayedFrameCounter: number = -1;

    // Running average of the total GPU time over a fixed window of frames.
    private readonly AVG_WINDOW = 256;
    private avgSamples: Float32Array = new Float32Array(this.AVG_WINDOW);
    private avgSampleIndex: number = 0;
    private avgSampleCount: number = 0;
    private avgSum: number = 0;
    private wasVisible: boolean = false;
    // The mean time of each pass over the same window: the sum of the frames it ran in (a pass that only runs in some
    // frames is averaged over those) and its share of the mean total. One frame's times jump with the GPU clocks, its
    // share of the frame much less. Keyed by the label and its occurrence in the frame (a label can repeat).
    private passSamples: Map<string, { ms: Float32Array, ran: Uint8Array, sum: number, count: number }> = new Map();

    // Running totals for benchmarks: diff them over a window of frames to get the mean GPU time.
    public totalMsSum: number = 0;
    public totalMsCount: number = 0;
    // Passes that got no timestamps because MAX_QUERIES was reached; a total with dropped passes is incomplete.
    public droppedPassCount: number = 0;
    // Called with the total GPU time of every frame that is read back, for a min and max over a window of frames, and
    // with the label and the time of each pass.
    private frameCallbacks: ProfilerFrameCallback[] = [];

    public init(device: GPUDevice, canvasParent: HTMLElement) {
        this.device = device;
        if (this.device.features.has('timestamp-query')) {
            this.querySet = this.device.createQuerySet({
                type: 'timestamp',
                count: this.MAX_QUERIES,
            });
            this.queryResolveBuffer = this.device.createBuffer({
                size: this.MAX_QUERIES * 8, // 64-bit timestamps
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            });

            // Pre-allocate read buffer pool
            for (let i = 0; i < this.BUFFER_POOL_SIZE; i++) {
                this.readBufferPool.push(this.device.createBuffer({
                    size: this.MAX_QUERIES * 8,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
                    label: `profiler_read_buffer_${i}`
                }));
                this.readBufferAvailable.push(true);
            }
        }

        this.createDebugDiv(canvasParent);

        Profiler.instances.push(this);
        this.enabled = Profiler.enabledGlobal;
    }

    private createDebugDiv(canvasParent: HTMLElement) {
        this.debugDiv = document.createElement('div');
        this.debugDiv.style.position = 'absolute';
        this.debugDiv.style.top = '10px';
        this.debugDiv.style.left = '10px';
        this.debugDiv.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        this.debugDiv.style.color = '#00ff00';
        this.debugDiv.style.padding = '10px';
        this.debugDiv.style.fontFamily = 'monospace';
        // a pass per line: never wrapped (it goes out of view instead), smaller on a narrow screen so it mostly fits
        this.debugDiv.style.fontSize = 'clamp(8px, 2.4vw, 12px)';
        this.debugDiv.style.whiteSpace = 'nowrap';
        this.debugDiv.style.pointerEvents = 'none';
        this.debugDiv.style.zIndex = '1000';
        this.debugDiv.style.display = 'none';

        // Ensure parent has positioning
        if (getComputedStyle(canvasParent).position === 'static') {
            canvasParent.style.position = 'relative';
        }
        canvasParent.appendChild(this.debugDiv);
    }

    public addFrameCallback(callback: ProfilerFrameCallback) {
        this.frameCallbacks.push(callback);
    }

    public removeFrameCallback(callback: ProfilerFrameCallback) {
        const index = this.frameCallbacks.indexOf(callback);
        if (index >= 0) this.frameCallbacks.splice(index, 1);
    }

    //whether passes get timestamps. the renderer then gives every dispatch a compute pass of its own
    public get recording(): boolean {
        return this.enabled && this.querySet !== null;
    }

    public getTimestampWrites(label: string): GPURenderPassTimestampWrites | undefined {
        if (!this.enabled || !this.querySet) {
            return undefined;
        }
        if (this.currentQueryIndex + 2 > this.MAX_QUERIES) {
            this.droppedPassCount++;
            return undefined;
        }

        const startIndex = this.currentQueryIndex;
        const endIndex = this.currentQueryIndex + 1;
        this.currentQueryIndex += 2;
        this.queryLabels.push(label);

        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: startIndex,
            endOfPassWriteIndex: endIndex
        };
    }

    private getAvailableReadBuffer(): { buffer: GPUBuffer; index: number } | null {
        for (let i = 0; i < this.BUFFER_POOL_SIZE; i++) {
            if (this.readBufferAvailable[i]) {
                this.readBufferAvailable[i] = false;
                return { buffer: this.readBufferPool[i], index: i };
            }
        }
        return null;
    }

    public resolve(commandEncoder: GPUCommandEncoder) {
        if (!this.querySet) {
            if (this.debugDiv) this.debugDiv.style.display = 'none';
            return;
        }

        if (this.enabled) {
            this.lastActiveTime = performance.now();
        }

        const keepVisible = (performance.now() - this.lastActiveTime) < 1000;

        if (!keepVisible && !this.enabled) {
            if (this.debugDiv) this.debugDiv.style.display = 'none';
            this.wasVisible = false;
            return;
        }

        if (!this.wasVisible) {
            this.avgSamples.fill(0);
            this.avgSampleIndex = 0;
            this.avgSampleCount = 0;
            this.avgSum = 0;
            this.passSamples.clear();
            this.wasVisible = true;
        }

        if (this.debugDiv) {
            this.debugDiv.style.display = 'block';
        }

        if (!this.enabled) {
            return;
        }

        let queryReadBuffer: GPUBuffer | null = null;
        let queryLabels: string[] = [];
        let bufferIndex: number = -1;

        if (this.querySet && this.currentQueryIndex > 0 && this.queryResolveBuffer) {
            const available = this.getAvailableReadBuffer();
            if (!available) {
                // All buffers are in use, skip this frame's profiling read
                this.currentQueryIndex = 0;
                this.queryLabels = [];
                return;
            }

            const count = this.currentQueryIndex;
            const resolveBuffer = this.queryResolveBuffer;

            commandEncoder.resolveQuerySet(this.querySet, 0, count, resolveBuffer, 0);

            queryLabels = [...this.queryLabels];
            queryReadBuffer = available.buffer;
            bufferIndex = available.index;

            commandEncoder.copyBufferToBuffer(resolveBuffer, 0, queryReadBuffer, 0, count * 8);

            this.currentQueryIndex = 0;
            this.queryLabels = [];
        }

        const frameCounter = this.resolveFrameCounter++;
        return { queryReadBuffer, queryLabels, bufferIndex, frameCounter };
    }

    public async read(queryReadBuffer: GPUBuffer | null, queryLabels: string[], bufferIndex: number = -1, frameCounter: number = -1) {
        if (queryReadBuffer && bufferIndex >= 0) {
            try {
                await queryReadBuffer.mapAsync(GPUMapMode.READ);
                // Only update display if this is the most recent frame
                if (frameCounter >= this.displayedFrameCounter) {
                    const times = new BigInt64Array(queryReadBuffer.getMappedRange());
                    let logText = '';

                    let totalMs = 0;
                    const passMs: number[] = [];
                    for (let i = 0; i < queryLabels.length; i++) {
                        const start = times[i * 2];
                        const end = times[i * 2 + 1];
                        const durationNs = Number(end - start);
                        const durationMs = durationNs / 1000000;
                        totalMs += durationMs;
                        passMs.push(durationMs);
                    }
                    const passAvg = this.updatePassAverages(queryLabels, passMs);

                    this.totalMsSum += totalMs;
                    this.totalMsCount++;
                    this.frameCallbacks.forEach(callback => callback(totalMs, queryLabels, passMs));

                    // Ring buffer: replace the oldest sample and keep a running sum.
                    this.avgSum += totalMs - this.avgSamples[this.avgSampleIndex];
                    this.avgSamples[this.avgSampleIndex] = totalMs;
                    this.avgSampleIndex = (this.avgSampleIndex + 1) % this.AVG_WINDOW;
                    if (this.avgSampleCount < this.AVG_WINDOW) this.avgSampleCount++;
                    const avgTotal = this.avgSum / this.avgSampleCount;
                    for (let i = 0; i < queryLabels.length; i++) {
                        const share = avgTotal > 0 ? passAvg[i] / avgTotal * 100 : 0;
                        logText += `${queryLabels[i]}: ${passMs[i].toFixed(3)}ms (avg ${passAvg[i].toFixed(3)}, ${share.toFixed(1)}%)<br>`;
                    }
                    logText += `<br><b>Total: ${totalMs.toFixed(3)}ms (avg ${avgTotal.toFixed(3)}ms, ${this.AVG_WINDOW}-frame)</b>`;

                    if (this.debugDiv) {
                        this.debugDiv.innerHTML = logText;
                    }
                    this.displayedFrameCounter = frameCounter;
                }

                queryReadBuffer.unmap();
            } catch (e) {
                console.error("Profiler read error", e);
            } finally {
                // Return buffer to pool
                this.readBufferAvailable[bufferIndex] = true;
            }
        }
    }

    // The passes of a frame into their windows (the same AVG_WINDOW frames as the total), returns the mean of each
    private updatePassAverages(labels: string[], passMs: number[]): number[] {
        const slot = this.avgSampleIndex;
        const occurrences = new Map<string, number>();
        const seen = new Set<string>();
        const means: number[] = [];
        for (let i = 0; i < labels.length; i++) {
            const n = occurrences.get(labels[i]) ?? 0;
            occurrences.set(labels[i], n + 1);
            const key = labels[i] + '#' + n;
            seen.add(key);
            let p = this.passSamples.get(key);
            if (!p) {
                p = { ms: new Float32Array(this.AVG_WINDOW), ran: new Uint8Array(this.AVG_WINDOW), sum: 0, count: 0 };
                this.passSamples.set(key, p);
            }
            p.sum += passMs[i] - p.ms[slot];
            p.count += 1 - p.ran[slot];
            p.ms[slot] = passMs[i];
            p.ran[slot] = 1;
            means.push(p.sum / p.count);
        }
        // the passes that didn't run in this frame leave the window at this slot
        for (const [key, p] of this.passSamples) {
            if (seen.has(key)) continue;
            p.sum -= p.ms[slot];
            p.count -= p.ran[slot];
            p.ms[slot] = 0;
            p.ran[slot] = 0;
            if (p.count === 0) this.passSamples.delete(key);
        }
        return means;
    }

    public destruct() {
        const index = Profiler.instances.indexOf(this);
        if (index >= 0) Profiler.instances.splice(index, 1);

        for (const buffer of this.readBufferPool) {
            buffer.destroy();
        }
        this.readBufferPool = [];
        this.readBufferAvailable = [];
        this.querySet?.destroy();
        this.queryResolveBuffer?.destroy();
        if (this.debugDiv && this.debugDiv.parentNode) {
            this.debugDiv.parentNode.removeChild(this.debugDiv);
        }
    }
}
