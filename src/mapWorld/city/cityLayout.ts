import { hypot2 } from 'webGPU-framework/renderer/utils/MathUtils';
import ModelBVH from './ModelBVH';

/*
The lightmap charts and the ray tracing data of the buildings of a map tile, see SkyVisibilityLightmap.ts. Runs in the worker,
next to the geometry (see buildBuildings in worldGeometryCore.ts): no Renderer imports.

Charts (the surfaces of cityLightmapSurface.wgsl):
-the ground is 1 chart for the whole map: world xz, so it needs no layout
-the roofs are world xz too, 1 chart per height (see getRoofChart): roofs of the same height that touch share their
 texels, roofs of another height and the ground don't. Every xz point has only 1 surface that faces up and can be seen,
 but a texel on the outline of a roof has 2: with 1 chart for both it averaged the open roof and the shaded ground at
 the foot of its walls (or a lower roof), and the coarser levels mixed them over a wider band still
-the walls of a tile are unrolled into strips, one per ring, cut into chunks at a corner when they get longer than
 CHUNK_LENGTH, and packed in rows in the wall atlas of the tile: u is along the ring, v is the height above the base.
 The chunks are packed in the order of their building on a Morton curve, so walls that are neighbours in the atlas are
 mostly neighbours in the map too: a coarse texel that covers more than 1 chunk mixes walls that get similar light.
 The empty parts of the atlas take no room in the lightmap, only the pages that pixels use get a place
Positions are in world units, like the vertices.
 */

//the gap between the chunks in the atlas and the most length of a chunk, in m
const GUTTER_M = .5;
const CHUNK_LENGTH_M = 64;
//the height classes of the rows of the wall atlas, see layoutWalls
const ROW_HEIGHT_RATIO = 1.25;
//the most widths layoutWalls tries to get the atlas square
const PACK_TRIES = 4;

//the step of a wall atlas position in the gBuffer of MapWorld, in m, and the most steps it holds (19 bits), see
//mapGBufferPacking.wgsl: 5.24 km. The walls of a map tile of Midtown Manhattan (14/4825/6157, one of the densest
//there are) are 22 km2, an atlas of 5.16 km. 1 / 20 of the default level 0 texel
export const GBUFFER_ATLAS_STEP_M = .01;
export const GBUFFER_ATLAS_MAX = 0x7FFFF;

//the bit of the surface id of a wall atlas, see getWallSurfaceId in cityLightmapSurface.wgsl
export const SURFACE_WALL = 0x40000000;
//the heights of the roof charts, in m: roofs closer than this in height can share a chart
const ROOF_CHART_STEP_M = .25;
//added to the height of a roof before getRoofChart makes it a chart: with the terrain the heights are relative to the
//ground at the projection center (see ElevationService), a roof in a valley below it has a negative one
const ROOF_CHART_OFFSET_M = 10000;
//a child of a BVH node that is a record, see isOccluded in skyVisibilityTrace.wgsl
export const CITY_LEAF = 0x80000000;
//the vec4s of a record of the CityTraceBlock: the shape (r0) and where its lightmap surfaces are (r1)
export const CITY_RECORD_VEC4S = 2;
//an atlas word of a wall that has no place in the wall atlas (inside another part, see findWallBottoms)
export const CITY_ATLAS_NONE = 0xFFFFFFFF;
//the largest u and v of an atlas word, 16 bits each: below the 0xFFFF of CITY_ATLAS_NONE
const CITY_ATLAS_STEPS = 0xFFFE;
//the count of a tree record, see CityTraceBlock
export const CITY_TREE = 0x80000000;
//the most depth of the BVH of a tile, and of the top level BVH over the tiles (1024 tiles at most): together within
//CITY_STACK_SIZE of skyVisibilityTrace.wgsl
export const BLOCK_DEPTH_MAX = 22;
export const TOP_DEPTH_MAX = 10;

/** A footprint as buildBuildings has it: x, north pairs in world units, the outer ring first, CCW outer / CW holes. */
export type BuildingPolygon = {
  coords: Float64Array;
  ringStarts: number[];
  ringEnds: number[];
  base: number;
  top: number;
  //the chart of the roof, see getRoofChart
  roofChart: number;
};

/**
 * The ray tracing data of a tile, in vec4s (4 words each), as one block of the city buffer (see CityScene.ts). The any
 * hit rays of skyVisibilityTrace.wgsl read the shapes, the water reflections also where the point a ray hits is on the
 * charts of the lightmap (getCityHitSurface of cityTraceDistance.wgsl):
 * -nodeCount BVH nodes over the buildings and trees, 4 vec4s each: aMin.xyz, a, aMax.xyz, b, bMin.xyz, 0, bMax.xyz, 0
 *  (the layout of ModelBVH). A child is the vec4 index of a node, or of a record | CITY_LEAF
 * -buildingCount building records, then treeCount tree records, CITY_RECORD_VEC4S each:
 *  r0 (base, top, start, count). The base is the one of the ray tracing: down to the ground for a building on it, see
 *  groundY. start is a vec4 index. On a building count is its number of edges, on a tree it is CITY_TREE
 *  r1 (atlas, roof chart, header, 0): the word index of its first atlas word, the roof chart of its roof (of the crown
 *  on a tree, see getRoofChart) and the vec4 index of the header. Only read for the lightmap of a hit
 * -edgeCount edges, 1 vec4 each: (a.x, a.z, b.x, b.z)
 * -treeCount tree shapes, 2 vec4s each, see TraceTree: (center.x, center.z, u.x, u.z), (crown base, trunk half width /
 *  crown half width, 0, 0). u is the half axis of the crown divided by its length squared: the crown is [-1, 1] on u
 *  and on u turned by 90 degrees, the trunk [-k, k]
 * -the header, 1 vec4: (the surface id of the wall atlas of the tile, see getWallSurfaceId, the step of the atlas words
 *  and their v offset in world units, 0)
 * -the atlas words, 1 u32 each (4 per vec4): 1 per edge of a building, in the order of its edges, and 8 per tree (the 4
 *  walls of its trunk, then of its crown, in the order of the corners of square in worldTrees.ts). Where the wall
 *  starts in the wall atlas: u in the low 16 bits, its v at world height 0 (the v of the atlas is the height over the
 *  base) in the high 16, as u * step and v * step + the v offset of the header. CITY_ATLAS_NONE: the wall has no place
 *  in the atlas. 4 bytes instead of 2 floats: an atlas is at most 5.24 km (GBUFFER_ATLAS_MAX), a step of 8 cm at
 *  worst and some 1-3 cm in a city tile, under the 20 cm of the finest texels
 * The indices are from the start of the block, CityScene adds where the block is. A forest tile has some 12000 trees:
 * as 2 extruded footprints each (2 records of 2 vec4s, 8 edges of 2 vec4s, 2 BVH leaves) they took about 5 MB, as tree
 * records some 1.4 MB
 */
export type CityTraceBlock = {
  words: Uint32Array;
  //per building record the feature id of its vertices (the id of the picking, see CityScene.pick), -1 for none. Only on
  //the CPU, the GPU doesn't get it
  ids: Int32Array;
  nodeCount: number;
  buildingCount: number;
  treeCount: number;
  edgeCount: number;
  //the most nodes from the root to a leaf, see CITY_STACK_SIZE in skyVisibilityTrace.wgsl
  depth: number;
  //the vec4 index of the header, see CityTraceBlock
  headerStart: number;
  min: [number, number, number];
  max: [number, number, number];
};

/**
 * A tree of worldTrees.ts for the ray tracing: a trunk and a crown on it, 2 square boxes around the same center turned by
 * the same angle. x and y (north) in world units, the half widths of the squares and the heights in world units too.
 */
export type TraceTree = {
  x: number;
  y: number;
  //the bottom of the trunk: the ground under the tree
  base: number;
  angle: number;
  trunkHalf: number;
  crownHalf: number;
  crownBase: number;
  top: number;
};

/**
 * The wall atlas of a tile for the atlas words of buildTraceBlock: the layout of layoutWalls and the arrays per point of
 * all footprints it was made of (pointStarts, edgeBottoms, see layoutWalls). The footprints of the trunks and crowns of
 * the trees (the trunk of tree i at 2 i, its crown at 2 i + 1) are footprint treeStart and up in them
 */
export type TraceAtlas = {
  layout: CityLayout;
  pointStarts: Int32Array;
  edgeBottoms: Float32Array;
  treeFootprints: BuildingPolygon[];
  treeStart: number;
};

export type CityLayout = {
  //the atlas position (u, v) of the start of every edge, 2 floats per point of the polygons: those of polygon i from
  //pointStarts[i] (see layoutWalls), in the order of its coords
  edgeAtlas: Float32Array;
  //the surface id of the wall atlas, and the tile coordinates it is made of (the lightmap attribute has them as floats)
  wallSurface: number;
  tileX: number;
  tileY: number;
};

export function getWallSurfaceId(tileX: number, tileY: number): number {
  return (SURFACE_WALL | ((tileX & 0x7FFF) << 15) | (tileY & 0x7FFF)) >>> 0;
}

/**
 * The roof chart of a building of height topM (m, over the projection center's ground): its height (plus
 * ROOF_CHART_OFFSET_M) in steps of ROOF_CHART_STEP_M, an integer, so the roof
 * vertices (lightmap attribute x, a float) and the building record of the ray tracing (a u32) get the same one.
 * getRoofSurface in cityLightmapSurface.wgsl makes it a surface id. Not from the height in world units in the shader: the
 * interpolated height of a roof that lies on a step boundary would flicker between 2 charts.
 */
export function getRoofChart(topM: number): number {
  return Math.min(Math.max(Math.round((topM + ROOF_CHART_OFFSET_M) / ROOF_CHART_STEP_M), 0), 0xFFFFFF);
}

//the bits of x and y interleaved, 16 each
function morton(x: number, y: number): number {
  const spread = (v: number) => {
    v &= 0xFFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
  };
  return (spread(x) | (spread(y) << 1)) >>> 0;
}

/**
 * The chunks of layoutWalls as columns, kept from tile to tile (a worker lays out 1 tile at a time): as an object each,
 * with an array of its edges, and the rows of every try of the packing as objects in a Map, they were some 5 MB of
 * garbage per tile of a city. A chunk is a run of edges of 1 ring, in the order of the ring: its first edge (a point
 * index of all polygons, see layoutWalls) and their count.
 */
class ChunkColumns {
  public capacity = 0;
  public polygon = new Int32Array(0);
  public first = new Int32Array(0);
  public count = new Int32Array(0);
  public heightClass = new Int32Array(0);
  // in the row of its height class while the packing makes it, see pack: the next chunk of the row, -1 for none
  public next = new Int32Array(0);
  public length = new Float64Array(0);
  public height = new Float64Array(0);
  public bottom = new Float64Array(0);
  public order = new Float64Array(0);
  public x = new Float64Array(0);
  public y = new Float64Array(0);
  // the sort keys: order * CHUNKS_MAX + index
  public keys = new Float64Array(0);

  /** Room for this many chunks (the old ones are not kept). */
  public reserve(count: number): void {
    if (count <= this.capacity) return;
    const n = this.capacity = Math.max(count, this.capacity * 2, 1024);
    this.polygon = new Int32Array(n); this.first = new Int32Array(n); this.count = new Int32Array(n);
    this.heightClass = new Int32Array(n); this.next = new Int32Array(n);
    this.length = new Float64Array(n); this.height = new Float64Array(n); this.bottom = new Float64Array(n);
    this.order = new Float64Array(n); this.x = new Float64Array(n); this.y = new Float64Array(n); this.keys = new Float64Array(n);
  }
}

let chunkColumns: ChunkColumns | null = null;
// the most chunks of a tile, 2^21: the sort key of a chunk is its order times this plus its index
const CHUNKS_MAX = 2097152;

/** The wall atlas of a tile: see the comment at the top. */
/**
 * pointStarts: per polygon the index of its first point in all of them, and after the last the count of all (the
 * arrays per point are 1 for all polygons). edgeBottoms: per point (the edge from that ring vertex on) the lowest height
 * of its wall that is drawn, or NaN when the wall is not drawn (see findWallBottoms): a chunk is only as high as from its
 * lowest drawn bottom to the top, a wall that isn't drawn has no place in the atlas and the chunk ends there. The v of an
 * edge is still the one at the base of its polygon (below the chunk when the wall is cut off at the bottom)
 */
export function layoutWalls(polygons: BuildingPolygon[], tileX: number, tileY: number, metersToNorm: number,
  pointStarts: Int32Array, edgeBottoms?: Float32Array): CityLayout {
  const gutter = GUTTER_M * metersToNorm;
  const chunkLength = CHUNK_LENGTH_M * metersToNorm;
  const pointCount = pointStarts[polygons.length];
  const edgeAtlas = new Float32Array(pointCount * 2);

  // at most a chunk per edge
  const ch = chunkColumns ??= new ChunkColumns();
  ch.reserve(pointCount);
  let chunkCount = 0;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const centers = new Float64Array(polygons.length * 2);
  for (let pi = 0; pi < polygons.length; pi++) {
    const p = polygons[pi];
    const s = p.ringStarts[0], e = p.ringEnds[0];
    let cx = 0, cy = 0;
    for (let j = s; j < e; j++) { cx += p.coords[j * 2]; cy += p.coords[j * 2 + 1]; }
    cx /= Math.max(e - s, 1); cy /= Math.max(e - s, 1);
    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
    centers[pi * 2] = cx;
    centers[pi * 2 + 1] = cy;
  }
  const spanX = Math.max(maxX - minX, 1e-9), spanY = Math.max(maxY - minY, 1e-9);

  for (let pi = 0; pi < polygons.length; pi++) {
    const p = polygons[pi], c = p.coords, ps = pointStarts[pi];
    const order = morton(Math.floor((centers[pi * 2] - minX) / spanX * 65535), Math.floor((centers[pi * 2 + 1] - minY) / spanY * 65535));
    for (let r = 0; r < p.ringStarts.length; r++) {
      const s = p.ringStarts[r], e = p.ringEnds[r];
      const n = e - s;
      // the chunk being made, -1: none
      let chunk = -1;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        const len = hypot2(c[ib * 2] - c[ia * 2], c[ib * 2 + 1] - c[ia * 2 + 1]);
        const bottom = edgeBottoms ? edgeBottoms[ps + ia] : p.base;
        //a wall that isn't drawn: the next one starts a chunk of its own, it isn't next to the one before it
        if (len > 0 && Number.isNaN(bottom)) {
          chunk = -1;
          continue;
        }
        if (chunk < 0 || (ch.length[chunk] > 0 && ch.length[chunk] + len > chunkLength)) {
          chunk = chunkCount++;
          ch.polygon[chunk] = pi; ch.first[chunk] = ps + ia; ch.count[chunk] = 0;
          ch.length[chunk] = 0; ch.height[chunk] = 0; ch.bottom[chunk] = p.top; ch.order[chunk] = order;
          ch.heightClass[chunk] = 0; ch.x[chunk] = 0; ch.y[chunk] = 0;
        }
        ch.count[chunk]++;
        //an edge of 0 length still gets an atlas position, it has no wall
        edgeAtlas[(ps + ia) * 2] = ch.length[chunk];
        ch.length[chunk] += len;
        if (len > 0) ch.bottom[chunk] = Math.min(ch.bottom[chunk], bottom);
      }
    }
  }
  let area = 0;
  for (let i = 0; i < chunkCount; i++) ch.height[i] = Math.max(polygons[ch.polygon[i]].top - ch.bottom[i], 0);
  for (let i = 0; i < chunkCount; i++) area += (ch.length[i] + gutter) * (ch.height[i] + gutter);

  //rows (shelves) of chunks in Morton order (stable, so the chunks of a ring stay in their order), 1 open row per
  //height class: a row is as high as its highest chunk, so with 1 row for all a single tower among the trees and low
  //buildings of a row made the whole row its height, and the atlas got many times higher than its area (over
  //GBUFFER_ATLAS_MAX: the walls past it all got the lighting of the last atlas row, stretched up). Stable: by the
  //order, then the index, as 1 key (a sort of numbers, without a function to compare)
  if (chunkCount > CHUNKS_MAX) throw new Error(`layoutWalls: ${chunkCount} chunks, at most ${CHUNKS_MAX}`);
  const keys = ch.keys.subarray(0, chunkCount);
  for (let i = 0; i < chunkCount; i++) keys[i] = ch.order[i] * CHUNKS_MAX + i;
  keys.sort();
  const sorted = new Int32Array(chunkCount);
  for (let k = 0; k < chunkCount; k++) sorted[k] = keys[k] % CHUNKS_MAX;
  let maxLength = 0, maxClass = -1;
  for (let k = 0; k < chunkCount; k++) {
    const i = sorted[k];
    maxLength = Math.max(maxLength, ch.length[i]);
    //classes of a factor ROW_HEIGHT_RATIO: a row wastes at most 1 - 1 / ROW_HEIGHT_RATIO of its height
    const heightClass = ch.heightClass[i] = Math.floor(Math.log(Math.max(ch.height[i], gutter) / gutter) / Math.log(ROW_HEIGHT_RATIO));
    maxClass = Math.max(maxClass, heightClass);
  }
  const widthMin = maxLength + 2 * gutter;
  // the open row of each height class: whether there is one, its x and height, its first and last chunk (see next)
  const classes = maxClass + 1;
  const rowOpen = new Uint8Array(classes), rowHead = new Int32Array(classes), rowTail = new Int32Array(classes);
  const rowX = new Float64Array(classes), rowHeight = new Float64Array(classes);
  //sets the x, y of the chunks for an atlas of this width, returns its height
  const pack = (width: number): number => {
    let y = gutter;
    rowOpen.fill(0);
    // the chunks of a row get the y, the next row is after its height
    const closeRow = (head: number, height: number) => {
      for (let i = head; i >= 0; i = ch.next[i]) ch.y[i] = y;
      y += height + gutter;
    };
    for (let k = 0; k < chunkCount; k++) {
      const i = sorted[k], c = ch.heightClass[i];
      if (rowOpen[c] && rowX[c] + ch.length[i] + gutter > width && rowX[c] > gutter) {
        closeRow(rowHead[c], rowHeight[c]);
        rowOpen[c] = 0;
      }
      if (!rowOpen[c]) {
        rowOpen[c] = 1;
        rowX[c] = gutter;
        rowHeight[c] = 0;
        rowHead[c] = i;
      } else ch.next[rowTail[c]] = i;
      rowTail[c] = i;
      ch.next[i] = -1;
      ch.x[i] = rowX[c];
      rowX[c] += ch.length[i] + gutter;
      rowHeight[c] = Math.max(rowHeight[c], ch.height[i]);
    }
    //the rows that are still open, 1 per height class and mostly empty: a tile of towers has some 30 classes, as rows
    //of their own they were a third of its atlas. Their chunks from high to low share rows, so the chunks of a row are
    //still about as high
    let lastOpen = false, lastHead = -1, lastTail = -1, lastX = 0, lastHeight = 0;
    for (let c = maxClass; c >= 0; c--) {
      if (!rowOpen[c]) continue;
      for (let i = rowHead[c]; i >= 0;) {
        // before it is linked into the row it goes to
        const following = ch.next[i];
        if (lastOpen && lastX + ch.length[i] + gutter > width) {
          closeRow(lastHead, lastHeight);
          lastOpen = false;
        }
        if (!lastOpen) {
          lastOpen = true;
          lastX = gutter;
          lastHeight = 0;
          lastHead = i;
        } else ch.next[lastTail] = i;
        lastTail = i;
        ch.next[i] = -1;
        ch.x[i] = lastX;
        lastX += ch.length[i] + gutter;
        lastHeight = Math.max(lastHeight, ch.height[i]);
        i = following;
      }
    }
    if (lastOpen) closeRow(lastHead, lastHeight);
    return y;
  };
  //the gBuffer stores as many steps of u as of v (GBUFFER_ATLAS_MAX), so the atlas that reaches least far is square:
  //the width from the height of the last try, the rows don't fill up the same at every width
  let width = Math.max(Math.sqrt(area) * 1.1, widthMin);
  let best = width, bestSize = Infinity;
  for (let i = 0; i < PACK_TRIES; i++) {
    const height = pack(width);
    const size = Math.max(width, height);
    if (size < bestSize) { bestSize = size; best = width; }
    const next = Math.max(Math.sqrt(width * height), widthMin);
    if (Math.abs(next - width) < width * .01) break;
    width = next;
  }
  if (width !== best) pack(best);
  for (let i = 0; i < chunkCount; i++) {
    //the v at the base of the polygon: chunk.y is at the bottom of the chunk
    const v = ch.y[i] - (ch.bottom[i] - polygons[ch.polygon[i]].base);
    for (let e = ch.first[i]; e < ch.first[i] + ch.count[i]; e++) {
      edgeAtlas[e * 2] += ch.x[i];
      edgeAtlas[e * 2 + 1] = v;
    }
  }

  return {edgeAtlas, wallSurface: getWallSurfaceId(tileX, tileY), tileX, tileY};
}

/** Box i of boxes (min.xyz, max.xyz each). */
function setBox(boxes: Float64Array, i: number, x0: number, y0: number, z0: number, x1: number, y1: number, z1: number): void {
  const o = i * 6;
  boxes[o] = x0; boxes[o + 1] = y0; boxes[o + 2] = z0;
  boxes[o + 3] = x1; boxes[o + 4] = y1; boxes[o + 5] = z1;
}

/**
 * The BVH, records, edges and tree shapes of the tile (see CityTraceBlock). groundY: the height of the ground plane of the
 * ray tracing, a building or tree that stands on the ground (base 0) goes down to it, so no ray passes under its walls
 * between the ground layers. Null on the terrain: the bases are on the ground already (see terrainGrid.ts). ids: per
 * polygon its feature id, see CityTraceBlock (-1 for all without). atlas: where the walls are in the wall atlas, for
 * the atlas words; the polygons are its first footprints. Null without buildings and trees
 */
export function buildTraceBlock(polygons: BuildingPolygon[], trees: TraceTree[], groundY: number | null, atlas: TraceAtlas,
  ids?: ArrayLike<number>): CityTraceBlock | null {
  const buildingCount = polygons.length;
  const treeCount = trees.length;
  if (buildingCount + treeCount === 0) return null;

  //min.xyz and max.xyz per building, then per tree
  const boxes = new Float64Array((buildingCount + treeCount) * 6);
  let edgeCount = 0;
  for (let pi = 0; pi < buildingCount; pi++) {
    const p = polygons[pi];
    let minX = Infinity, minZ = Infinity, maxX = -Infinity, maxZ = -Infinity;
    for (let r = 0; r < p.ringStarts.length; r++) {
      const s = p.ringStarts[r], e = p.ringEnds[r];
      const n = e - s;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        const x = p.coords[ia * 2], z = -p.coords[ia * 2 + 1];
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        if (p.coords[ia * 2] !== p.coords[ib * 2] || p.coords[ia * 2 + 1] !== p.coords[ib * 2 + 1]) edgeCount++;
      }
    }
    const base = groundY !== null && p.base <= 0 ? groundY : p.base;
    setBox(boxes, pi, minX, base, minZ, maxX, p.top, maxZ);
  }
  //a tree is in the box around its crown: the trunk is under it and narrower. In x, z the half axis of the crown is
  //(cos, -sin) * crownHalf (the corners of square in worldTrees.ts, z = -north), its box reaches |cos| + |sin| of it
  for (let ti = 0; ti < treeCount; ti++) {
    const t = trees[ti];
    const reach = (Math.abs(Math.cos(t.angle)) + Math.abs(Math.sin(t.angle))) * t.crownHalf;
    setBox(boxes, buildingCount + ti, t.x - reach, groundY ?? t.base, -t.y - reach, t.x + reach, t.top, -t.y + reach);
  }

  //the stack of isOccluded holds the path of the top level BVH and of a tile (CITY_STACK_SIZE), see CityScene. A tile of
  //30000 buildings is 20 deep
  const bvh = new ModelBVH(boxes, null, BLOCK_DEPTH_MAX);
  const nodeCount = bvh.nodeCount;
  const recordStart = nodeCount * 4;
  const edgeStart = recordStart + (buildingCount + treeCount) * CITY_RECORD_VEC4S;
  const shapeStart = edgeStart + edgeCount;
  const headerStart = shapeStart + treeCount * 2;
  const atlasStart = headerStart + 1;
  const atlasWords = edgeCount + treeCount * 8;
  const vec4Count = atlasStart + Math.ceil(atlasWords / 4);
  const data = new ArrayBuffer(vec4Count * 16);
  const u = new Uint32Array(data);
  const f = new Float32Array(data);

  //the nodes (at the start of the block), with the children as vec4 indices of the block. The records are in the order
  //of the boxes
  bvh.setNodeData(data);
  for (let n = 0; n < nodeCount; n++) {
    for (let w = n * 16 + 3; w < n * 16 + 8; w += 4) {
      const child = u[w];
      u[w] = child >= ModelBVH.leafFlag
        ? ((recordStart + (child & ~ModelBVH.leafFlag) * CITY_RECORD_VEC4S) | CITY_LEAF) >>> 0
        : child * 4;
    }
  }

  //per atlas word the atlas position of the start of its wall: u, and the v at world height 0. NaN: no place
  const atlasU = new Float64Array(atlasWords).fill(NaN), atlasV = new Float64Array(atlasWords).fill(NaN);
  const {layout, pointStarts, edgeBottoms} = atlas;
  const setAtlas = (word: number, p: BuildingPolygon, point: number) => {
    if (Number.isNaN(edgeBottoms[point])) return;
    atlasU[word] = layout.edgeAtlas[point * 2];
    atlasV[word] = layout.edgeAtlas[point * 2 + 1] - p.base;
  };

  let edge = 0;
  polygons.forEach((p, pi) => {
    const record = (recordStart + pi * CITY_RECORD_VEC4S) * 4;
    const first = edge;
    const ps = pointStarts[pi];
    for (let r = 0; r < p.ringStarts.length; r++) {
      const s = p.ringStarts[r], e = p.ringEnds[r];
      const n = e - s;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        const ax = p.coords[ia * 2], ay = p.coords[ia * 2 + 1];
        const bx = p.coords[ib * 2], by = p.coords[ib * 2 + 1];
        if (ax === bx && ay === by) continue;
        const o = (edgeStart + edge) * 4;
        f[o] = ax; f[o + 1] = -ay; f[o + 2] = bx; f[o + 3] = -by;
        setAtlas(edge, p, ps + ia);
        edge++;
      }
    }
    f[record] = boxes[pi * 6 + 1];
    f[record + 1] = p.top;
    u[record + 2] = edgeStart + first;
    u[record + 3] = edge - first;
    u[record + 4] = atlasStart * 4 + first;
    u[record + 5] = p.roofChart;
    u[record + 6] = headerStart;
  });
  trees.forEach((t, ti) => {
    const record = (recordStart + (buildingCount + ti) * CITY_RECORD_VEC4S) * 4;
    const shape = shapeStart + ti * 2;
    f[record] = groundY ?? t.base;
    f[record + 1] = t.top;
    u[record + 2] = shape;
    u[record + 3] = CITY_TREE;
    const word = edgeCount + ti * 8;
    u[record + 4] = atlasStart * 4 + word;
    //the 4 walls of the trunk, then of the crown, whose roof is the one of the tree
    for (let b = 0; b < 2; b++) {
      const footprint = ti * 2 + b, p = atlas.treeFootprints[footprint];
      const ps = pointStarts[atlas.treeStart + footprint];
      for (let k = 0; k < 4; k++) setAtlas(word + b * 4 + k, p, ps + k);
      u[record + 5] = p.roofChart;
    }
    u[record + 6] = headerStart;
    const o = shape * 4;
    f[o] = t.x;
    f[o + 1] = -t.y;
    f[o + 2] = Math.cos(t.angle) / t.crownHalf;
    f[o + 3] = -Math.sin(t.angle) / t.crownHalf;
    f[o + 4] = t.crownBase;
    f[o + 5] = t.trunkHalf / t.crownHalf;
  });

  //the atlas words: 1 step for u and v (the atlas is about square, see layoutWalls), v from the lowest
  let uMax = 0, vMin = Infinity, vMax = -Infinity;
  for (let i = 0; i < atlasWords; i++) {
    if (Number.isNaN(atlasU[i])) continue;
    uMax = Math.max(uMax, atlasU[i]);
    vMin = Math.min(vMin, atlasV[i]);
    vMax = Math.max(vMax, atlasV[i]);
  }
  if (vMin > vMax) vMin = vMax = 0;
  const step = Math.max(uMax, vMax - vMin, 1e-9) / CITY_ATLAS_STEPS;
  const wordStart = atlasStart * 4;
  for (let i = 0; i < atlasWords; i++) {
    if (Number.isNaN(atlasU[i])) {
      u[wordStart + i] = CITY_ATLAS_NONE;
      continue;
    }
    const qu = Math.min(Math.round(atlasU[i] / step), CITY_ATLAS_STEPS);
    const qv = Math.min(Math.round((atlasV[i] - vMin) / step), CITY_ATLAS_STEPS);
    u[wordStart + i] = (qu | (qv << 16)) >>> 0;
  }
  u[headerStart * 4] = layout.wallSurface;
  f[headerStart * 4 + 1] = step;
  f[headerStart * 4 + 2] = vMin;

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let o = 0; o < boxes.length; o += 6) {
    min[0] = Math.min(min[0], boxes[o]); min[1] = Math.min(min[1], boxes[o + 1]); min[2] = Math.min(min[2], boxes[o + 2]);
    max[0] = Math.max(max[0], boxes[o + 3]); max[1] = Math.max(max[1], boxes[o + 4]); max[2] = Math.max(max[2], boxes[o + 5]);
  }
  const recordIds = new Int32Array(buildingCount).fill(-1);
  if (ids) recordIds.set(Array.from(ids).slice(0, buildingCount));
  return {words: u, ids: recordIds, nodeCount, buildingCount, treeCount, edgeCount, depth: bvh.depth, headerStart, min, max};
}
