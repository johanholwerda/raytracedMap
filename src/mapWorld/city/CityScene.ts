import Renderer from 'webGPU-framework/renderer/render/Renderer';
import ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import ModelBVH from './ModelBVH';
import { CITY_LEAF, CITY_RECORD_VEC4S, CITY_TREE_VEC4S, TOP_DEPTH_MAX } from './cityLayout';
import type { CityTraceBlock } from './cityLayout';
import { intersectTraceBlock } from './cityPick';

//the box of the buildings of a tile that came or went, and how far around it the sky visibility changes
type Box = { min: number[]; max: number[]; reach: number };
//a part of the map where buildings came or went in frame
type Change = Box & { frame: number };

//the ground of a map tile in world x and z: min.x, min.z, max.x, max.z
export type TileRect = [number, number, number, number];

type Tile = {
  block: CityTraceBlock;
  //vec4 index of the block in the city buffer, -1 while it has no place
  offset: number;
  //the map tile it is of: a cell of the grid, see writeGrid
  rect?: TileRect;
};

/*
The buildings and trees the rays of the sky visibility lightmap hit, see isOccluded in skyVisibilityTrace.wgsl. One storage
buffer of vec4s:
-from 0 the top level BVH over the tiles (TLAS_NODES_MAX nodes at most). Its leaves are the roots of the BVHs of the
 tiles: all are in world space, so the 2 levels are 1 tree and the traversal 1 loop. Made again when a tile comes or goes
-then the grid of the tiles (GRID_VEC4S), see writeGrid: a ray that goes up from a place where it can only hit the
 buildings of 1 tile starts at the root of its BVH instead of that of the top level BVH (getCityStart in
 skyVisibilityTrace.wgsl). With the 17 tiles of a view of Amsterdam 5 of the 13 nodes a water reflection ray visited
 were of the top level BVH, to end up in the tile the ray starts in
-then CHANGES_MAX changes: the boxes of the buildings of the tiles that came or went and the frame they did, 2 vec4s
 each: (min.xyz, frame), (max.xyz, reach). A texel that was last sampled before the frame has the lighting of before
 and is reset (see resetStaleTexel in updateSkyVisibilityLightmap.wgsl): its sky visibility within the reach around
 the box, its sun visibility where a ray to the sun passes the box, as far as the shadows of the buildings go (see
 getChangeFrames there). When there are more, the 2 boxes whose union is the least larger than they are become their
 union, with the later frame and the longer reach: that resets more texels than needed, never less
-after it the blocks of the tiles, see CityTraceBlock. A block gets a place when its tile is added and keeps it until the
 tile is removed (first fit, the free ranges merged), so its indices only have to be moved once, on upload: the
 children of its nodes and the starts of its records
The buffer has a fixed size: growing it would copy in the command encoder of the frame, after the queue writes of the
same frame. A tile that doesn't fit gets no place, and its buildings and trees hide no sky and cast no shadows
 */
export default class CityScene {
  private static readonly TLAS_NODES_MAX = 1024;
  private static readonly NODE_VEC4S = 4;
  //the grid of the tiles, after the top level BVH (CITY_GRID_START of skyVisibilityTrace.wgsl): 2 vec4s, and 2 per
  //cell, see writeGrid
  private static readonly GRID_HEADER_VEC4S = 2;
  private static readonly GRID_CELLS_MAX = 32;
  private static readonly GRID_VEC4S = CityScene.GRID_HEADER_VEC4S + CityScene.GRID_CELLS_MAX * CityScene.GRID_CELLS_MAX * 2;
  //a child of a cell without a tile, CITY_NONE of skyVisibilityTrace.wgsl
  private static readonly NONE = 0xFFFFFFFF;
  //the changes that updateSkyVisibilityLightmap.wgsl reads, from CITY_CHANGE_START of skyVisibilityTrace.wgsl
  private static readonly CHANGES_MAX = 16;
  //how much larger the boxes of the tiles count in the grid, in cells: more than the rounding of a ray's start
  private static readonly GRID_MARGIN = 1e-4;
  //how far around a tile the sky visibility changes when it comes or goes, in times the height of its tallest
  //building: the sky it hides. Not the sun it hides: that is where its shadows fall, see the class comment
  public static readonly CHANGE_REACH = 2;
  private renderer: Renderer;
  private readonly capacity: number;
  public buffer: GPUBuffer | null = null;
  private tiles: Map<string, Tile> = new Map();
  //the free ranges after the top level BVH, in vec4s, sorted by start
  private free: { start: number; size: number }[] = [];
  private tlasDirty = true;
  private shaders: ShaderBase[] = [];
  // the ones of them that read the changes too (changeCount, changeEnd)
  private changeShaders: ShaderBase[] = [];
  //the tiles in the top level BVH, and the vec4 index of its root: a tile's own root when there is 1
  public tileCount = 0;
  public root = 0;
  public usedVec4s = 0;
  //bumped when the set of tiles in the buffer changes
  public version = 0;
  private changes: Change[] = [];
  //the boxes of the tiles that came or went since the last update, which gives them its frame
  private pendingChanges: Box[] = [];

  /**
   * The defines of the shaders that include skyVisibilityTrace.wgsl: CITY_STACK_SIZE, the stack of the traversal. 8 on
   * mobile, much faster there but a ray that needs more misses the buildings of the far children it drops (2-3 hits in a
   * million on the tested rays), 16 elsewhere (none missed). A define, so it is fixed when the shaders are made.
   */
  public static get traceDefines(): Record<string, number> {
    return { CITY_STACK_SIZE: isMobile() ? 8 : 16 };
  }

  constructor(renderer: Renderer, sizeBytes: number) {
    this.renderer = renderer;
    const max = renderer.device.limits.maxStorageBufferBindingSize;
    this.capacity = Math.floor(Math.min(sizeBytes, max) / 16);
  }

  private get gridStart(): number {
    return CityScene.TLAS_NODES_MAX * CityScene.NODE_VEC4S;
  }

  private get changeStart(): number {
    return this.gridStart + CityScene.GRID_VEC4S;
  }

  private get heapStart(): number {
    return this.changeStart + CityScene.CHANGES_MAX * 2;
  }

  /** Creates the buffer when it doesn't exist yet and uploads the tiles that were added before. */
  public enable(): void {
    if (this.buffer) return;
    this.buffer = this.renderer.createBuffer(this.capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, false, 'cityBuffer');
    this.free = [{ start: this.heapStart, size: this.capacity - this.heapStart }];
    for (const [key, tile] of this.tiles) this.place(key, tile);
    for (const shader of this.shaders) shader.setBuffer('cityBuffer', this.buffer);
    this.tlasDirty = true;
    LogGL.log(`[CityScene] city buffer ${(this.capacity * 16 / 1048576).toFixed(0)} MB`);
  }

  /**
   * The shaders that trace the city get the buffer and the root, see update. changes: also the changes (changeCount,
   * changeEnd), for the shaders that reset the texels around them
   */
  public bind(shader: ShaderBase, changes: boolean = true): void {
    this.shaders.push(shader);
    if (this.buffer) shader.setBuffer('cityBuffer', this.buffer);
    shader.setFloat('cityTileCount', this.tileCount);
    shader.setFloat('cityRoot', this.root);
    if (!changes) return;
    this.changeShaders.push(shader);
    shader.setFloat('changeCount', this.changes.length);
    shader.setFloat('changeEnd', this.changes.reduce((end, c) => Math.max(end, c.frame), 0));
  }

  /** rect: the ground of the map tile the block is of, a cell of the grid (see writeGrid). */
  public addTile(key: string, block: CityTraceBlock | null, rect?: TileRect): void {
    this.removeTile(key);
    if (!block) return;
    const tile: Tile = { block, offset: -1, rect };
    this.tiles.set(key, tile);
    if (this.buffer) this.place(key, tile);
  }

  public removeTile(key: string): void {
    const tile = this.tiles.get(key);
    if (!tile) return;
    this.tiles.delete(key);
    if (tile.offset >= 0) {
      this.release(tile.offset, tile.block.words.length / 4);
      this.tlasDirty = true;
      this.addChange(tile.block);
    }
  }

  public clear(): void {
    for (const key of [...this.tiles.keys()]) this.removeTile(key);
  }

  private place(key: string, tile: Tile): void {
    const size = tile.block.words.length / 4;
    const offset = this.allocate(size);
    if (offset < 0) {
      LogGL.warn(`[CityScene] no room for tile ${key} (${(size * 16 / 1048576).toFixed(2)} MB, ${(this.usedVec4s * 16 / 1048576).toFixed(0)} of ${(this.capacity * 16 / 1048576).toFixed(0)} MB used): its buildings are not traced`);
      return;
    }
    tile.offset = offset;
    const block = tile.block;
    //the indices of the block to indices of the buffer
    const words = new Uint32Array(block.words);
    for (let n = 0; n < block.nodeCount; n++) {
      for (const w of [n * 16 + 3, n * 16 + 7]) {
        const child = words[w];
        words[w] = child >= CITY_LEAF ? (((child & ~CITY_LEAF) + offset) | CITY_LEAF) >>> 0 : child + offset;
      }
    }
    //the starts of the edges of the building records, of their atlas words (a word index), their header and their
    //wall bottoms (a half word index), and the header of the tree records
    const recordStart = block.nodeCount * 4;
    for (let r = 0; r < block.buildingCount; r++) {
      const w = (recordStart + r * CITY_RECORD_VEC4S) * 4;
      words[w + 2] += offset;
      words[w + 4] += offset * 4;
      words[w + 6] += offset;
      words[w + 7] += offset * 8;
    }
    const treeStart = recordStart + block.buildingCount * CITY_RECORD_VEC4S;
    for (let r = 0; r < block.treeCount; r++) words[(treeStart + r * CITY_TREE_VEC4S) * 4 + 11] += offset;
    this.renderer.device.queue.writeBuffer(this.buffer!, offset * 16, words.buffer, words.byteOffset, words.byteLength);
    this.tlasDirty = true;
    this.addChange(block);
    if (LogGL.ENABLED) LogGL.log(`[CityScene] tile ${key}: ${block.buildingCount} buildings, ${block.treeCount} trees, ${block.edgeCount} edges, BVH depth ${block.depth}, ${(size * 16 / 1048576).toFixed(2)} MB, ${(this.usedVec4s * 16 / 1048576).toFixed(1)} MB used`);
  }

  //the box of the buildings of the block and how far around them they change the sky visibility
  private addChange(block: CityTraceBlock): void {
    const reach = (block.max[1] - block.min[1]) * CityScene.CHANGE_REACH;
    this.pendingChanges.push({ min: [...block.min], max: [...block.max], reach });
  }

  /** The lightmap starts over: no texel has lighting of before a change. */
  public clearChanges(): void {
    this.changes.length = 0;
    this.pendingChanges.length = 0;
    for (const shader of this.changeShaders) {
      shader.setFloat('changeCount', 0);
      shader.setFloat('changeEnd', 0);
    }
  }

  //merges the 2 changes whose union is the least larger than they are, see the class comment
  private mergeChanges(): void {
    const area = (b: Box) => (b.max[0] - b.min[0]) * (b.max[2] - b.min[2]);
    const union = (a: Box, b: Box): Box => ({
      min: a.min.map((v, k) => Math.min(v, b.min[k])),
      max: a.max.map((v, k) => Math.max(v, b.max[k])),
      reach: Math.max(a.reach, b.reach),
    });
    let best = Infinity, bestI = 0, bestJ = 1;
    for (let i = 0; i < this.changes.length; i++) {
      for (let j = i + 1; j < this.changes.length; j++) {
        const growth = area(union(this.changes[i], this.changes[j])) - area(this.changes[i]) - area(this.changes[j]);
        if (growth < best) { best = growth; bestI = i; bestJ = j; }
      }
    }
    const a = this.changes[bestI], b = this.changes[bestJ];
    this.changes[bestI] = { ...union(a, b), frame: Math.max(a.frame, b.frame) };
    this.changes.splice(bestJ, 1);
  }

  private writeChanges(frame: number): void {
    for (const box of this.pendingChanges) this.changes.push({ ...box, frame });
    this.pendingChanges.length = 0;
    while (this.changes.length > CityScene.CHANGES_MAX) this.mergeChanges();
    const data = new ArrayBuffer(this.changes.length * 32);
    const f = new Float32Array(data), u = new Uint32Array(data);
    let end = 0;
    this.changes.forEach((c, i) => {
      f.set(c.min, i * 8);
      u[i * 8 + 3] = c.frame;
      f.set(c.max, i * 8 + 4);
      f[i * 8 + 7] = c.reach;
      end = Math.max(end, c.frame);
    });
    if (data.byteLength > 0) this.renderer.device.queue.writeBuffer(this.buffer!, this.changeStart * 16, data);
    for (const shader of this.changeShaders) {
      shader.setFloat('changeCount', this.changes.length);
      shader.setFloat('changeEnd', end);
    }
  }

  private allocate(size: number): number {
    for (let i = 0; i < this.free.length; i++) {
      const range = this.free[i];
      if (range.size < size) continue;
      const start = range.start;
      range.start += size;
      range.size -= size;
      if (range.size === 0) this.free.splice(i, 1);
      this.usedVec4s += size;
      return start;
    }
    return -1;
  }

  private release(start: number, size: number): void {
    this.usedVec4s -= size;
    let i = 0;
    while (i < this.free.length && this.free[i].start < start) i++;
    this.free.splice(i, 0, { start, size });
    //merge with the next one and the previous one
    if (i + 1 < this.free.length && this.free[i].start + this.free[i].size === this.free[i + 1].start) {
      this.free[i].size += this.free[i + 1].size;
      this.free.splice(i + 1, 1);
    }
    if (i > 0 && this.free[i - 1].start + this.free[i - 1].size === this.free[i].start) {
      this.free[i - 1].size += this.free[i].size;
      this.free.splice(i, 1);
    }
  }

  /**
   * The top level BVH, when the tiles changed. Sets cityTileCount and cityRoot on the bound shaders, and the changes
   * with the frame (changeCount, changeEnd).
   */
  public update(frame: number): void {
    if (!this.buffer) return;
    if (this.pendingChanges.length > 0) this.writeChanges(frame);
    if (!this.tlasDirty) return;
    this.tlasDirty = false;
    this.version++;

    const placed = [...this.tiles.values()].filter(tile => tile.offset >= 0).slice(0, CityScene.TLAS_NODES_MAX);
    this.tileCount = placed.length;
    this.root = placed.length === 1 ? placed[0].offset : 0;
    if (placed.length > 1) {
      const bvh = new ModelBVH(
        placed.map(t => new Vector3(...t.block.min)),
        placed.map(t => new Vector3(...t.block.max)),
        TOP_DEPTH_MAX,
      );
      const data = new ArrayBuffer(bvh.nodeCount * ModelBVH.bytesPerNode);
      bvh.setNodeData(data);
      const words = new Uint32Array(data);
      for (let n = 0; n < bvh.nodeCount; n++) {
        for (const w of [n * 16 + 3, n * 16 + 7]) {
          const child = words[w];
          //a tile is the root node of its block, not a leaf
          words[w] = child >= ModelBVH.leafFlag ? placed[child & ~ModelBVH.leafFlag].offset : child * CityScene.NODE_VEC4S;
        }
      }
      this.renderer.device.queue.writeBuffer(this.buffer, 0, data);
    }
    this.writeGrid(placed);
    for (const shader of this.shaders) {
      shader.setFloat('cityTileCount', this.tileCount);
      shader.setFloat('cityRoot', this.root);
    }
  }

  /**
   * The grid of the tiles, for the rays that can only hit the buildings of 1 tile (getCityStart in
   * skyVisibilityTrace.wgsl): they start at the root of its BVH, not at the root of the top level BVH.
   * -2 vec4s: (origin.x, origin.z, 1 / cell size.x, 1 / cell size.z), (cells along x, cells along z, the top of the
   *  highest building of all tiles, 0)
   * -2 vec4s per cell, row by row along x: (the part of the cell that only the box of its tile reaches into: min.x,
   *  min.z, max.x, max.z in world units, min above max when there is none), (the vec4 index of the root of the BVH of
   *  that tile, NONE where no box reaches into the cell, 0, 0, 0)
   * The cells are the map tiles (their rects), a whole number of cells from the first one's corner. The buildings of
   * a tile reach a few meters over its edges (the buffer of the vector tiles), into the cells around it: a strip
   * along the sides of a cell is of 2 tiles or more. A ray that goes up from the rest of the cell, and is above the
   * highest building before it leaves it, can only hit the tile of the cell.
   * No cells (every ray takes the top level BVH) without the rects, or with more cells than GRID_CELLS_MAX along a
   * side.
   */
  private writeGrid(placed: Tile[]): void {
    const start = this.gridStart * 16;
    const header = new ArrayBuffer(CityScene.GRID_HEADER_VEC4S * 16);
    const rects = placed.map(tile => tile.rect);
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity, top = -Infinity;
    for (const { block } of placed) {
      minX = Math.min(minX, block.min[0]); maxX = Math.max(maxX, block.max[0]);
      minZ = Math.min(minZ, block.min[2]); maxZ = Math.max(maxZ, block.max[2]);
      top = Math.max(top, block.max[1]);
    }
    const first = rects[0];
    let dimX = 0, dimZ = 0, sizeX = 1, sizeZ = 1, originX = 0, originZ = 0;
    if (first && rects.every(rect => rect)) {
      sizeX = Math.max(rects.reduce((sum, r) => sum + r![2] - r![0], 0) / rects.length, 1e-6);
      sizeZ = Math.max(rects.reduce((sum, r) => sum + r![3] - r![1], 0) / rects.length, 1e-6);
      // on the edges of the tiles, and around every box
      originX = first[0] - Math.ceil((first[0] - minX) / sizeX) * sizeX;
      originZ = first[1] - Math.ceil((first[1] - minZ) / sizeZ) * sizeZ;
      dimX = Math.ceil((maxX - originX) / sizeX);
      dimZ = Math.ceil((maxZ - originZ) / sizeZ);
      if (Math.max(dimX, dimZ) > CityScene.GRID_CELLS_MAX) dimX = dimZ = 0;
    }
    const hf = new Float32Array(header), hu = new Uint32Array(header);
    hf[0] = originX; hf[1] = originZ; hf[2] = 1 / sizeX; hf[3] = 1 / sizeZ;
    hu[4] = dimX; hu[5] = dimZ; hf[6] = top;
    this.renderer.device.queue.writeBuffer(this.buffer!, start, header);
    if (dimX * dimZ === 0) return;

    // the boxes count this much larger: the rounding of the rays
    const grow = CityScene.GRID_MARGIN * Math.max(sizeX, sizeZ);
    // the tiles whose box reaches into each cell
    const cells: Tile[][] = Array.from({ length: dimX * dimZ }, () => []);
    for (const tile of placed) {
      const { min, max } = tile.block;
      const x0 = Math.max(Math.floor((min[0] - grow - originX) / sizeX), 0), x1 = Math.min(Math.floor((max[0] + grow - originX) / sizeX), dimX - 1);
      const z0 = Math.max(Math.floor((min[2] - grow - originZ) / sizeZ), 0), z1 = Math.min(Math.floor((max[2] + grow - originZ) / sizeZ), dimZ - 1);
      for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) cells[z * dimX + x].push(tile);
    }
    const data = new ArrayBuffer(cells.length * 2 * 16);
    const f = new Float32Array(data), u = new Uint32Array(data);
    cells.forEach((cell, c) => {
      const x0 = originX + (c % dimX) * sizeX, z0 = originZ + Math.floor(c / dimX) * sizeZ;
      const x1 = x0 + sizeX, z1 = z0 + sizeZ;
      // the tile of the cell: the one with the most of it
      const area = ({ block: { min, max } }: Tile) =>
        Math.max(Math.min(max[0], x1) - Math.max(min[0], x0), 0) * Math.max(Math.min(max[2], z1) - Math.max(min[2], z0), 0);
      cell.sort((a, b) => area(b) - area(a));
      f.set(CityScene.getAloneRect(cell.slice(1).map(tile => tile.block), x0, z0, x1, z1, grow), c * 8);
      u[c * 8 + 4] = cell.length > 0 ? cell[0].offset : CityScene.NONE;
    });
    this.renderer.device.queue.writeBuffer(this.buffer!, start + header.byteLength, data);
  }

  /**
   * The part of a cell (x0, z0 to x1, z1) that none of the boxes of others reach into, as a rect (min.x, min.z, max.x,
   * max.z): each box cuts off the side of what is left that loses the least to it. Nothing left: min above max.
   * grow: the boxes count this much larger, see writeGrid
   */
  private static getAloneRect(others: CityTraceBlock[], x0: number, z0: number, x1: number, z1: number, grow: number): number[] {
    const rect = [x0, z0, x1, z1];
    for (const { min, max } of others) {
      const bx0 = min[0] - grow, bz0 = min[2] - grow, bx1 = max[0] + grow, bz1 = max[2] + grow;
      if (bx0 >= rect[2] || bx1 <= rect[0] || bz0 >= rect[3] || bz1 <= rect[1]) continue;
      // the part of what is left each cut takes: the west side up to the east of the box, the north, the east, the south
      const width = rect[2] - rect[0], depth = rect[3] - rect[1];
      const cuts = [(bx1 - rect[0]) / width, (bz1 - rect[1]) / depth, (rect[2] - bx0) / width, (rect[3] - bz0) / depth];
      let best = 0;
      for (let k = 1; k < 4; k++) if (cuts[k] < cuts[best]) best = k;
      rect[best] = [bx1, bz1, bx0, bz0][best];
    }
    return rect;
  }

  /**
   * The building the ray hits first, on the CPU (the building picking): the key of its tile, its feature id and the
   * ray distance t (in lengths of rd), null when it hits nothing. A tree in front hides the building behind it: then
   * the feature id is -1, as for a building without one. Takes all tiles, also those without a place in the city buffer.
   */
  public pick(ro: Vector3, rd: Vector3): { key: string; featureId: number; t: number } | null {
    const o = [ro.x, ro.y, ro.z], d = [rd.x, rd.y, rd.z];
    const inv = d.map(v => (v >= 0 ? 1 : -1) / Math.max(Math.abs(v), 1e-12));
    //the tiles the ray passes, nearest first, so the closest hit culls the tiles behind it
    const candidates: { key: string; tile: Tile; near: number }[] = [];
    for (const [key, tile] of this.tiles) {
      const { min, max } = tile.block;
      let near = 0, far = Infinity;
      for (let k = 0; k < 3; k++) {
        const t0 = (min[k] - o[k]) * inv[k], t1 = (max[k] - o[k]) * inv[k];
        near = Math.max(near, Math.min(t0, t1));
        far = Math.min(far, Math.max(t0, t1));
      }
      if (near <= far) candidates.push({ key, tile, near });
    }
    candidates.sort((a, b) => a.near - b.near);
    let best = Infinity;
    let hit: { key: string; featureId: number; t: number } | null = null;
    for (const { key, tile, near } of candidates) {
      if (near >= best) break;
      const result = intersectTraceBlock(tile.block, o, d, best);
      if (!result) continue;
      best = result.t;
      const block = tile.block;
      hit = { key, featureId: result.record < block.buildingCount ? block.ids[result.record] : -1, t: result.t };
    }
    return hit;
  }

  public destruct(): void {
    this.buffer?.destroy();
    this.buffer = null;
    this.tiles.clear();
  }
}
