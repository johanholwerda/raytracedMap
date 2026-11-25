import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import { NO_FEATURE_ID, VERTEX_FLOATS } from './cityGlTFCore';

/**
 * Packs the geometry of many terrain tiles (a single map layer) into one
 * vertex buffer and one index buffer so the whole layer can be drawn in a
 * single draw call, regardless of how many tiles are streamed in. Goal: cut
 * the per-tile draw calls the renderer would otherwise issue.
 *
 * A tile is xyz vertices plus triangle indices (mapWorld), or a non-indexed
 * triangle soup (OSMBuilder / PDOK), which gets the indices 0..n-1. On add,
 * each incoming vertex is expanded into the unified stride-4 layout the map
 * shares — xyz position + a `.w` building id, always {@link NO_FEATURE_ID} for
 * terrain — so terrain draws with the exact same vertex layout as the buildings
 * and ground quad (one g-buffer pipeline, no per-mesh layout churn). Merging
 * tiles is concatenating their (expanded) vertices and their indices, offset
 * by where the tile's vertices start. Each tile keeps its vertex and index
 * range; removing a tile compacts everything after it to the left so the used
 * regions stay contiguous (one draw covers them with no holes), and lowers the
 * indices of the moved tiles by how far their vertices moved. CPU mirrors of
 * both buffers are kept so the compaction only re-uploads what moved.
 *
 * The draw uses drawIndexedIndirect: the index count lives in an indirect
 * buffer that is rewritten on every add/remove. This is required because
 * Renderer caches draw calls in render bundles keyed by mesh id (not index
 * count) — a plain renderer.draw() would bake a stale count into the cached
 * bundle.
 */

// Incoming OSM/PDOK geometry is xyz; stored vertices are the unified vec4.
const SRC_FLOATS_PER_VERTEX = 3;
const FLOATS_PER_VERTEX = VERTEX_FLOATS; // 4: xyz + id
const BYTES_PER_VERTEX = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT; // 16
const BYTES_PER_INDEX = Uint32Array.BYTES_PER_ELEMENT; // always 32 bit: the tiles together pass 65535 vertices
const INITIAL_CAPACITY_VERTS = 1 << 14; // 16384 verts (~256 KB)
const INITIAL_CAPACITY_INDICES = 1 << 15; // 32768 indices (128 KB)

// vertices [start, start + count), indices [indexStart, indexStart + indexCount)
type Range = { start: number; count: number; indexStart: number; indexCount: number };

export default class TerrainLayerBuffer {
  private renderer: Renderer;
  private mesh: Mesh | null = null;
  private indirectBuffer: GPUBuffer | null = null;
  private cpu: Float32Array = new Float32Array(0);
  private capacityVerts = 0;
  private usedVerts = 0;
  private cpuIndices: Uint32Array = new Uint32Array(0);
  private capacityIndices = 0;
  private usedIndices = 0;
  private ranges: Map<string, Range> = new Map();

  constructor(renderer: Renderer) {
    this.renderer = renderer;
  }

  public has(key: string): boolean {
    return this.ranges.has(key);
  }

  public get vertexCount(): number {
    return this.usedVerts;
  }

  public get indexCount(): number {
    return this.usedIndices;
  }

  /** The bytes of the tiles in the buffers: vertices and indices. */
  public get usedBytes(): number {
    return this.usedVerts * BYTES_PER_VERTEX + this.usedIndices * BYTES_PER_INDEX;
  }

  /** The bytes of the GPU buffers, which grow by doubling. */
  public get capacityBytes(): number {
    return this.capacityVerts * BYTES_PER_VERTEX + this.capacityIndices * BYTES_PER_INDEX;
  }

  /** The bytes a tile takes in the buffers, 0 when it isn't there. */
  public tileBytes(key: string): number {
    const range = this.ranges.get(key);
    return range ? range.count * BYTES_PER_VERTEX + range.indexCount * BYTES_PER_INDEX : 0;
  }

  /**
   * Append a tile under `key`: its positions (xyz) and its triangles, 3 indices into `positions` each. Without indices
   * `positions` is a triangle soup. No-op if empty or already present.
   */
  public addTile(key: string, positions: Float32Array, indices?: Uint32Array): void {
    if (positions.length === 0 || this.ranges.has(key)) return;
    const nVerts = positions.length / SRC_FLOATS_PER_VERTEX;
    const nIndices = indices ? indices.length : nVerts;
    if (nIndices === 0) return;

    this.ensureCapacity(this.usedVerts + nVerts, this.usedIndices + nIndices);

    const start = this.usedVerts;
    // Expand xyz -> (xyz, NO_FEATURE_ID) straight into the stride-4 CPU mirror,
    // then upload only this tile's expanded region.
    const base = start * FLOATS_PER_VERTEX;
    for (let v = 0; v < nVerts; v++) {
      const s = v * SRC_FLOATS_PER_VERTEX;
      const d = base + v * FLOATS_PER_VERTEX;
      this.cpu[d] = positions[s];
      this.cpu[d + 1] = positions[s + 1];
      this.cpu[d + 2] = positions[s + 2];
      this.cpu[d + 3] = NO_FEATURE_ID;
    }
    this.renderer.device.queue.writeBuffer(
      this.mesh!.vertexBuffer,
      start * BYTES_PER_VERTEX,
      this.cpu.buffer as ArrayBuffer,
      base * Float32Array.BYTES_PER_ELEMENT,
      nVerts * BYTES_PER_VERTEX,
    );

    // The indices of the tile, offset to where its vertices start in the buffer.
    const indexStart = this.usedIndices;
    const idx = this.cpuIndices;
    if (indices) {
      for (let i = 0; i < nIndices; i++) idx[indexStart + i] = start + indices[i];
    } else {
      for (let i = 0; i < nIndices; i++) idx[indexStart + i] = start + i;
    }
    this.uploadIndices(indexStart, nIndices);

    this.ranges.set(key, { start, count: nVerts, indexStart, indexCount: nIndices });
    this.usedVerts += nVerts;
    this.usedIndices += nIndices;
    this.updateIndirect();
  }

  /** Remove a tile, compacting the tiles after it left so the region stays packed. */
  public removeTile(key: string): void {
    this.removeTiles([key]);
  }

  /**
   * Remove several tiles at once: a single compaction pass + a single
   * re-upload of the moved region. Calling removeTile() per key instead costs
   * one full-tail memmove and one GPU re-upload (from the removed offset to the
   * end) *per tile* — evicting a handful of early tiles on a camera move then
   * re-uploads most of the buffer several times over. This does it once.
   */
  public removeTiles(keys: Iterable<string>): void {
    const removeSet = new Set<string>();
    for (const k of keys) if (this.ranges.has(k)) removeSet.add(k);
    if (removeSet.size === 0) return;

    // Walk survivors in address order, sliding each left to close the gaps left
    // by removed tiles. firstDirtyVert tracks the earliest vertex that moved so
    // we re-upload only [firstDirtyVert, usedVerts).
    const survivors: Range[] = [];
    for (const [key, range] of this.ranges) {
      if (!removeSet.has(key)) survivors.push(range);
    }
    survivors.sort((a, b) => a.start - b.start);

    // The index ranges are in the same order as the vertex ranges (both are appended on add and compacted together),
    // so they close their gaps in the same pass. A tile whose vertices move gets its indices lowered by as much.
    let writeVert = 0;
    let firstDirtyVert = Infinity;
    let writeIndex = 0;
    let firstDirtyIndex = Infinity;
    const idx = this.cpuIndices;
    for (const range of survivors) {
      const shift = range.start - writeVert;
      if (shift !== 0) {
        this.cpu.copyWithin(
          writeVert * FLOATS_PER_VERTEX,
          range.start * FLOATS_PER_VERTEX,
          (range.start + range.count) * FLOATS_PER_VERTEX,
        );
        if (writeVert < firstDirtyVert) firstDirtyVert = writeVert;
        range.start = writeVert;
      }
      if (shift !== 0 || range.indexStart !== writeIndex) {
        // writeIndex <= indexStart, so copying forward never overwrites indices still to be read
        const from = range.indexStart;
        for (let i = 0; i < range.indexCount; i++) idx[writeIndex + i] = idx[from + i] - shift;
        if (writeIndex < firstDirtyIndex) firstDirtyIndex = writeIndex;
        range.indexStart = writeIndex;
      }
      writeVert += range.count;
      writeIndex += range.indexCount;
    }

    for (const key of removeSet) this.ranges.delete(key);
    this.usedVerts = writeVert;
    this.usedIndices = writeIndex;

    if (this.mesh && firstDirtyVert < this.usedVerts) {
      this.renderer.device.queue.writeBuffer(
        this.mesh.vertexBuffer,
        firstDirtyVert * BYTES_PER_VERTEX,
        this.cpu.buffer as ArrayBuffer,
        firstDirtyVert * BYTES_PER_VERTEX,
        (this.usedVerts - firstDirtyVert) * BYTES_PER_VERTEX,
      );
    }
    if (firstDirtyIndex < this.usedIndices) this.uploadIndices(firstDirtyIndex, this.usedIndices - firstDirtyIndex);
    this.updateIndirect();
  }

  /** Drop all tiles. Keeps the allocated GPU buffers for reuse. */
  public clear(): void {
    this.ranges.clear();
    this.usedVerts = 0;
    this.usedIndices = 0;
    this.updateIndirect();
  }

  /** Draw the whole layer in a single (indirect) draw call. */
  public draw(material: VertexFragmentShader): void {
    if (this.usedIndices === 0 || !this.mesh || !this.indirectBuffer) return;
    this.renderer.drawIndirect(this.mesh, material, this.indirectBuffer);
  }

  public destruct(): void {
    this.mesh?.destruct();
    this.indirectBuffer?.destroy();
    this.mesh = null;
    this.indirectBuffer = null;
    this.ranges.clear();
    this.usedVerts = 0;
    this.capacityVerts = 0;
    this.cpu = new Float32Array(0);
    this.usedIndices = 0;
    this.capacityIndices = 0;
    this.cpuIndices = new Uint32Array(0);
  }

  /** Update the indirect draw args: [indexCount, instanceCount, firstIndex, baseVertex, firstInstance]. */
  private updateIndirect(): void {
    if (!this.indirectBuffer) return;
    this.renderer.writeBuffer(this.indirectBuffer, new Uint32Array([this.usedIndices, 1, 0, 0, 0]));
  }

  /** Upload [start, start + count) of the CPU index mirror. */
  private uploadIndices(start: number, count: number): void {
    if (!this.mesh?.indexBuffer || count === 0) return;
    this.renderer.device.queue.writeBuffer(
      this.mesh.indexBuffer,
      start * BYTES_PER_INDEX,
      this.cpuIndices.buffer as ArrayBuffer,
      start * BYTES_PER_INDEX,
      count * BYTES_PER_INDEX,
    );
  }

  /**
   * Make sure the GPU/CPU buffers hold at least `neededVerts` vertices and
   * `neededIndices` indices. Grows by doubling and re-uploads the kept data.
   * Growth is rare; the GPU buffers are recreated (new Mesh -> new bundle), and
   * the old ones are destructed.
   */
  private ensureCapacity(neededVerts: number, neededIndices: number): void {
    if (this.mesh && neededVerts <= this.capacityVerts && neededIndices <= this.capacityIndices) return;

    let newCapacity = this.capacityVerts || INITIAL_CAPACITY_VERTS;
    while (newCapacity < neededVerts) newCapacity *= 2;
    let newIndexCapacity = this.capacityIndices || INITIAL_CAPACITY_INDICES;
    while (newIndexCapacity < neededIndices) newIndexCapacity *= 2;

    const newCpu = new Float32Array(newCapacity * FLOATS_PER_VERTEX);
    newCpu.set(this.cpu.subarray(0, this.usedVerts * FLOATS_PER_VERTEX));
    const newCpuIndices = new Uint32Array(newIndexCapacity);
    newCpuIndices.set(this.cpuIndices.subarray(0, this.usedIndices));

    const newMesh = new Mesh(this.renderer);
    // Unified stride-4 layout (xyz + id), bound as `position` like every other
    // map mesh. Single attribute => no interleave copy on upload.
    newMesh.setAttribute('position', FLOATS_PER_VERTEX, newCpu);
    // Force buffer creation + upload now (capacity-sized, VERTEX | COPY_DST).
    void newMesh.vertexBuffer;
    // The index buffer is made here instead of by setIndices: always 32 bit, capacity-sized and written per tile.
    newMesh.indexBuffer = this.renderer.device.createBuffer({
      size: newIndexCapacity * BYTES_PER_INDEX,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      label: 'terrainLayerIndices',
    });
    newMesh.indexFormat = 'uint32';
    newMesh.indexCount = newIndexCapacity;

    this.mesh?.destruct();
    this.mesh = newMesh;
    this.cpu = newCpu;
    this.capacityVerts = newCapacity;
    this.cpuIndices = newCpuIndices;
    this.capacityIndices = newIndexCapacity;
    this.uploadIndices(0, this.usedIndices);

    if (!this.indirectBuffer) {
      this.indirectBuffer = this.renderer.createBuffer(
        20,
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        false,
        'terrainLayerIndirect',
      );
      this.updateIndirect();
    }
  }
}
