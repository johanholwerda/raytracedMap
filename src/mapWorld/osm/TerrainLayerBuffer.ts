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
 * and ground quad (one g-buffer pipeline, no per-mesh layout churn).
 *
 * Each tile gets a range of the vertex buffer and a range of the index buffer
 * (first fit, see RangeAllocator) and keeps them until it is removed: nothing
 * moves, so adding a tile writes only its own data and removing one writes only
 * zeros over its indices. Zeros are degenerate triangles (all corners vertex 0),
 * which the GPU skips, so the single draw covers the index buffer up to the end
 * of the last tile, holes and all; a later tile fills a hole it fits in. The
 * buffers live on the GPU only (no CPU mirrors). When they are full they grow by
 * doubling, copied on the GPU. A compaction instead (moving the tiles after a
 * removed one to the left) re-uploaded much of the buffer on every eviction:
 * 50-200 ms of main thread per frame while the map was dragged.
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
const INITIAL_CAPACITY_VERTS = 1 << 16; // 65536 verts (1 MB)
const INITIAL_CAPACITY_INDICES = 1 << 17; // 131072 indices (512 KB)

// vertices [start, start + count), indices [indexStart, indexStart + indexCount)
type Range = { start: number; count: number; indexStart: number; indexCount: number };

/** The free ranges of a buffer of `capacity` elements: first fit, neighbours merged. */
class RangeAllocator {
  // sorted by start, never touching each other
  private free: { start: number; size: number }[];

  constructor(public capacity: number) {
    this.free = capacity > 0 ? [{ start: 0, size: capacity }] : [];
  }

  /** The start of a free range of `size`, the lowest one: -1 when none is large enough. */
  public allocate(size: number): number {
    for (let i = 0; i < this.free.length; i++) {
      const range = this.free[i];
      if (range.size < size) continue;
      const start = range.start;
      range.start += size;
      range.size -= size;
      if (range.size === 0) this.free.splice(i, 1);
      return start;
    }
    return -1;
  }

  public release(start: number, size: number): void {
    let i = 0;
    while (i < this.free.length && this.free[i].start < start) i++;
    this.free.splice(i, 0, { start, size });
    if (i + 1 < this.free.length && start + size === this.free[i + 1].start) {
      this.free[i].size += this.free[i + 1].size;
      this.free.splice(i + 1, 1);
    }
    if (i > 0 && this.free[i - 1].start + this.free[i - 1].size === start) {
      this.free[i - 1].size += this.free[i].size;
      this.free.splice(i, 1);
    }
  }

  /** Everything free again. */
  public reset(): void {
    this.free = this.capacity > 0 ? [{ start: 0, size: this.capacity }] : [];
  }

  /** The capacity grows to `capacity`: the new part is free. */
  public grow(capacity: number): void {
    const last = this.free[this.free.length - 1];
    if (last && last.start + last.size === this.capacity) last.size += capacity - this.capacity;
    else this.free.push({ start: this.capacity, size: capacity - this.capacity });
    this.capacity = capacity;
  }

  /** The end of the last used range: all after it is free. */
  public get end(): number {
    const last = this.free[this.free.length - 1];
    return last && last.start + last.size === this.capacity ? last.start : this.capacity;
  }
}

export default class TerrainLayerBuffer {
  private renderer: Renderer;
  private mesh: Mesh | null = null;
  private vertexBuffer: GPUBuffer | null = null;
  private indexBuffer: GPUBuffer | null = null;
  private indirectBuffer: GPUBuffer | null = null;
  private indirectArgs = new Uint32Array([0, 1, 0, 0, 0]);
  private vertices = new RangeAllocator(0);
  private indices = new RangeAllocator(0);
  private usedVerts = 0;
  private usedIndices = 0;
  private ranges: Map<string, Range> = new Map();
  // reused for the expanded vertices and the offset indices of a tile, and the zeros over a removed one's indices
  // (queue.writeBuffer copies the data right away)
  private vertexScratch = new Float32Array(0);
  private indexScratch = new Uint32Array(0);
  private zeros = new Uint32Array(0);

  /** initialVerts, initialIndices: the capacity of the buffers when the first tile comes. */
  constructor(renderer: Renderer, private initialVerts = INITIAL_CAPACITY_VERTS, private initialIndices = INITIAL_CAPACITY_INDICES) {
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
    return this.vertices.capacity * BYTES_PER_VERTEX + this.indices.capacity * BYTES_PER_INDEX;
  }

  /** The bytes a tile takes in the buffers, 0 when it isn't there. */
  public tileBytes(key: string): number {
    const range = this.ranges.get(key);
    return range ? range.count * BYTES_PER_VERTEX + range.indexCount * BYTES_PER_INDEX : 0;
  }

  /**
   * Add a tile under `key`: its positions (xyz, or with srcFloats 4 xyzw: the .w is kept, mapWorld's terrain has its
   * normal there) and its triangles, 3 indices into `positions` each. Without indices `positions` is a triangle soup.
   * No-op if empty or already present.
   */
  public addTile(key: string, positions: Float32Array, indices?: Uint32Array, srcFloats = SRC_FLOATS_PER_VERTEX): void {
    if (positions.length === 0 || this.ranges.has(key)) return;
    const nVerts = positions.length / srcFloats;
    const nIndices = indices ? indices.length : nVerts;
    if (nIndices === 0) return;

    let start = this.vertices.allocate(nVerts);
    let indexStart = this.indices.allocate(nIndices);
    if (start < 0 || indexStart < 0) {
      if (start >= 0) this.vertices.release(start, nVerts);
      if (indexStart >= 0) this.indices.release(indexStart, nIndices);
      this.grow(nVerts, nIndices);
      start = this.vertices.allocate(nVerts);
      indexStart = this.indices.allocate(nIndices);
    }
    const queue = this.renderer.device.queue;

    // xyz -> (xyz, NO_FEATURE_ID), xyzw as it is
    let vertexData = positions;
    if (srcFloats !== FLOATS_PER_VERTEX) {
      vertexData = this.vertexScratch = TerrainLayerBuffer.atLeast(this.vertexScratch, nVerts * FLOATS_PER_VERTEX);
      for (let v = 0; v < nVerts; v++) {
        const s = v * srcFloats;
        const d = v * FLOATS_PER_VERTEX;
        vertexData[d] = positions[s];
        vertexData[d + 1] = positions[s + 1];
        vertexData[d + 2] = positions[s + 2];
        vertexData[d + 3] = NO_FEATURE_ID;
      }
    }
    queue.writeBuffer(this.vertexBuffer!, start * BYTES_PER_VERTEX, vertexData.buffer as ArrayBuffer, vertexData.byteOffset, nVerts * BYTES_PER_VERTEX);

    // The indices of the tile, offset to where its vertices start in the buffer.
    const idx = this.indexScratch = TerrainLayerBuffer.atLeast(this.indexScratch, nIndices);
    if (indices) {
      for (let i = 0; i < nIndices; i++) idx[i] = start + indices[i];
    } else {
      for (let i = 0; i < nIndices; i++) idx[i] = start + i;
    }
    queue.writeBuffer(this.indexBuffer!, indexStart * BYTES_PER_INDEX, idx.buffer as ArrayBuffer, 0, nIndices * BYTES_PER_INDEX);

    this.ranges.set(key, { start, count: nVerts, indexStart, indexCount: nIndices });
    this.usedVerts += nVerts;
    this.usedIndices += nIndices;
    this.updateIndirect();
  }

  /** Remove a tile: its indices become degenerate triangles, its ranges free. */
  public removeTile(key: string): void {
    this.removeTiles([key]);
  }

  public removeTiles(keys: Iterable<string>): void {
    let removed = false;
    for (const key of keys) {
      const range = this.ranges.get(key);
      if (!range) continue;
      this.ranges.delete(key);
      const zeros = this.zeros = TerrainLayerBuffer.atLeast(this.zeros, range.indexCount);
      this.renderer.device.queue.writeBuffer(this.indexBuffer!, range.indexStart * BYTES_PER_INDEX, zeros.buffer as ArrayBuffer, 0, range.indexCount * BYTES_PER_INDEX);
      this.vertices.release(range.start, range.count);
      this.indices.release(range.indexStart, range.indexCount);
      this.usedVerts -= range.count;
      this.usedIndices -= range.indexCount;
      removed = true;
    }
    if (removed) this.updateIndirect();
  }

  /** Drop all tiles. Keeps the allocated GPU buffers for reuse. */
  public clear(): void {
    // The old indices are past the end of the draw. First fit fills from the start, so a hole (which the draw covers) is
    // only ever made by removeTiles, which zeroes it.
    this.ranges.clear();
    this.vertices.reset();
    this.indices.reset();
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
    if (this.indirectBuffer) this.renderer.destroyAfterSubmit(this.indirectBuffer);
    this.mesh = null;
    this.vertexBuffer = null;
    this.indexBuffer = null;
    this.indirectBuffer = null;
    this.ranges.clear();
    this.vertices = new RangeAllocator(0);
    this.indices = new RangeAllocator(0);
    this.usedVerts = 0;
    this.usedIndices = 0;
  }

  /** The indirect draw args: [indexCount, instanceCount, firstIndex, baseVertex, firstInstance], to the last tile's end. */
  private updateIndirect(): void {
    if (!this.indirectBuffer) return;
    this.indirectArgs[0] = this.indices.end;
    this.renderer.device.queue.writeBuffer(this.indirectBuffer, 0, this.indirectArgs);
  }

  private static atLeast<T extends Float32Array | Uint32Array>(array: T, length: number): T {
    if (array.length >= length) return array;
    return new (array.constructor as new (length: number) => T)(Math.max(length, array.length * 2));
  }

  /**
   * Room for a tile of `verts` vertices and `indices` indices after the end of the last one: doubles the buffers until
   * it fits. The new buffers get the contents of the old ones by a copy on the GPU, in a submit of its own: after the
   * queue writes made before it, before the ones after it. A new Mesh, so the render bundles with the old buffers are
   * made again.
   */
  private grow(verts: number, indices: number): void {
    const device = this.renderer.device;
    const vertexEnd = this.vertices.end, indexEnd = this.indices.end;
    let vertexCapacity = this.vertices.capacity || this.initialVerts;
    while (vertexCapacity - vertexEnd < verts) vertexCapacity *= 2;
    let indexCapacity = this.indices.capacity || this.initialIndices;
    while (indexCapacity - indexEnd < indices) indexCapacity *= 2;

    // new buffers are zeros: no indices after the copied ones
    const vertexBuffer = device.createBuffer({
      size: vertexCapacity * BYTES_PER_VERTEX,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'terrainLayerVertices',
    });
    const indexBuffer = device.createBuffer({
      size: indexCapacity * BYTES_PER_INDEX,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'terrainLayerIndices',
    });
    if (this.vertexBuffer && this.indexBuffer && (vertexEnd > 0 || indexEnd > 0)) {
      const encoder = device.createCommandEncoder();
      if (vertexEnd > 0) encoder.copyBufferToBuffer(this.vertexBuffer, 0, vertexBuffer, 0, vertexEnd * BYTES_PER_VERTEX);
      if (indexEnd > 0) encoder.copyBufferToBuffer(this.indexBuffer, 0, indexBuffer, 0, indexEnd * BYTES_PER_INDEX);
      device.queue.submit([encoder.finish()]);
    }

    const mesh = new Mesh(this.renderer);
    // Unified stride-4 layout (xyz + id), bound as `position` like every other map mesh; the data is written per tile.
    mesh.setAttribute('position', FLOATS_PER_VERTEX, new Float32Array(0));
    mesh.setVertexBuffer(vertexBuffer, vertexCapacity);
    mesh.indexBuffer = indexBuffer;
    mesh.indexFormat = 'uint32';
    mesh.indexCount = indexCapacity;
    // destroys the old buffers after the submit of the frame
    this.mesh?.destruct();
    this.mesh = mesh;
    this.vertexBuffer = vertexBuffer;
    this.indexBuffer = indexBuffer;
    if (vertexCapacity > this.vertices.capacity) this.vertices.grow(vertexCapacity);
    if (indexCapacity > this.indices.capacity) this.indices.grow(indexCapacity);

    this.indirectBuffer ??= this.renderer.createBuffer(20, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST, false, 'terrainLayerIndirect');
    this.updateIndirect();
  }
}
