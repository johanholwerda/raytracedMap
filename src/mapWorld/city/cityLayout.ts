import Vector3 from 'webGPU-framework/renderer/math/Vector3';
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
//the count of a tree record, see CityTraceBlock
export const CITY_TREE = 0x80000000;
//the most depth of the BVH of a tile, and of the top level BVH over the tiles (1024 tiles at most): together within
//CITY_STACK_SIZE of skyVisibilityTrace.wgsl
export const BLOCK_DEPTH_MAX = 22;
export const TOP_DEPTH_MAX = 10;

/** A footprint as buildBuildings has it: x, north pairs in world units, the outer ring first, CCW outer / CW holes. */
export type BuildingPolygon = {
  coords: number[];
  ringStarts: number[];
  ringEnds: number[];
  base: number;
  top: number;
  //the chart of the roof, see getRoofChart
  roofChart: number;
};

/**
 * The ray tracing data of a tile, in vec4s (4 words each), as one block of the city buffer (see CityScene.ts). Only the
 * any hit rays of skyVisibilityTrace.wgsl read it, so it holds the shapes and nothing about the surfaces:
 * -nodeCount BVH nodes over the buildings and trees, 4 vec4s each: aMin.xyz, a, aMax.xyz, b, bMin.xyz, 0, bMax.xyz, 0
 *  (the layout of ModelBVH). A child is the vec4 index of a node, or of a record | CITY_LEAF
 * -buildingCount building records, then treeCount tree records, 1 vec4 each: (base, top, start, count). The base is the
 *  one of the ray tracing: down to the ground for a building on it, see groundY. start is a vec4 index. On a building
 *  count is its number of edges, on a tree it is CITY_TREE
 * -edgeCount edges, 1 vec4 each: (a.x, a.z, b.x, b.z)
 * -treeCount tree shapes, 2 vec4s each, see TraceTree: (center.x, center.z, u.x, u.z), (crown base, trunk half width /
 *  crown half width, 0, 0). u is the half axis of the crown divided by its length squared: the crown is [-1, 1] on u
 *  and on u turned by 90 degrees, the trunk [-k, k]
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

export type CityLayout = {
  //per polygon the atlas position (u, v) of the start of every edge, 2 floats per ring vertex, in the order of coords
  edgeAtlas: Float32Array[];
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

/** The wall atlas of a tile: see the comment at the top. */
/**
 * edgeBottoms: per polygon per edge (the one from that ring vertex on) the lowest height of its wall that is drawn, or
 * NaN when the wall is not drawn (see findWallBottoms): a chunk is only as high as from its lowest drawn bottom to the
 * top, a wall that isn't drawn has no place in the atlas and the chunk ends there. The v of an edge is still the one
 * at the base of its polygon (below the chunk when the wall is cut off at the bottom)
 */
export function layoutWalls(polygons: BuildingPolygon[], tileX: number, tileY: number, metersToNorm: number,
  edgeBottoms?: Float32Array[]): CityLayout {
  const gutter = GUTTER_M * metersToNorm;
  const chunkLength = CHUNK_LENGTH_M * metersToNorm;
  const edgeAtlas = polygons.map(p => new Float32Array(p.coords.length));

  //the chunks: a run of edges of a ring, the start of its first edge in the coords of its polygon
  //x, y: its place in the atlas, see pack
  //bottom: the lowest drawn height of its walls
  type Chunk = {
    polygon: number; edges: number[]; length: number; height: number; bottom: number; order: number; heightClass: number;
    x: number; y: number;
  };
  const chunks: Chunk[] = [];

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const centers: [number, number][] = polygons.map(p => {
    const s = p.ringStarts[0], e = p.ringEnds[0];
    let cx = 0, cy = 0;
    for (let j = s; j < e; j++) { cx += p.coords[j * 2]; cy += p.coords[j * 2 + 1]; }
    cx /= Math.max(e - s, 1); cy /= Math.max(e - s, 1);
    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
    minY = Math.min(minY, cy); maxY = Math.max(maxY, cy);
    return [cx, cy];
  });
  const spanX = Math.max(maxX - minX, 1e-9), spanY = Math.max(maxY - minY, 1e-9);

  let area = 0;
  polygons.forEach((p, pi) => {
    const c = centers[pi];
    const order = morton(Math.floor((c[0] - minX) / spanX * 65535), Math.floor((c[1] - minY) / spanY * 65535));
    const bottoms = edgeBottoms?.[pi];
    for (let r = 0; r < p.ringStarts.length; r++) {
      const s = p.ringStarts[r], e = p.ringEnds[r];
      const n = e - s;
      let chunk: Chunk | null = null;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        const len = hypot2(p.coords[ib * 2] - p.coords[ia * 2], p.coords[ib * 2 + 1] - p.coords[ia * 2 + 1]);
        const bottom = bottoms ? bottoms[ia] : p.base;
        //a wall that isn't drawn: the next one starts a chunk of its own, it isn't next to the one before it
        if (len > 0 && Number.isNaN(bottom)) {
          chunk = null;
          continue;
        }
        if (!chunk || (chunk.length > 0 && chunk.length + len > chunkLength)) {
          chunk = {polygon: pi, edges: [], length: 0, height: 0, bottom: p.top, order, heightClass: 0, x: 0, y: 0};
          chunks.push(chunk);
        }
        chunk.edges.push(ia);
        //an edge of 0 length still gets an atlas position, it has no wall
        edgeAtlas[pi][ia * 2] = chunk.length;
        chunk.length += len;
        if (len > 0) chunk.bottom = Math.min(chunk.bottom, bottom);
      }
    }
  });
  for (const chunk of chunks) chunk.height = Math.max(polygons[chunk.polygon].top - chunk.bottom, 0);
  for (const chunk of chunks) area += (chunk.length + gutter) * (chunk.height + gutter);

  //rows (shelves) of chunks in Morton order (stable, so the chunks of a ring stay in their order), 1 open row per
  //height class: a row is as high as its highest chunk, so with 1 row for all a single tower among the trees and low
  //buildings of a row made the whole row its height, and the atlas got many times higher than its area (over
  //GBUFFER_ATLAS_MAX: the walls past it all got the lighting of the last atlas row, stretched up)
  chunks.sort((a, b) => a.order - b.order);
  let maxLength = 0;
  for (const chunk of chunks) {
    maxLength = Math.max(maxLength, chunk.length);
    //classes of a factor ROW_HEIGHT_RATIO: a row wastes at most 1 - 1 / ROW_HEIGHT_RATIO of its height
    chunk.heightClass = Math.floor(Math.log(Math.max(chunk.height, gutter) / gutter) / Math.log(ROW_HEIGHT_RATIO));
  }
  const widthMin = maxLength + 2 * gutter;
  //sets the x, y of the chunks for an atlas of this width, returns its height
  const pack = (width: number): number => {
    type Row = {x: number; height: number; chunks: Chunk[]};
    const rows = new Map<number, Row>();
    let y = gutter;
    const closeRow = (row: Row) => {
      for (const chunk of row.chunks) chunk.y = y;
      y += row.height + gutter;
    };
    const add = (row: Row, chunk: Chunk) => {
      row.chunks.push(chunk);
      chunk.x = row.x;
      row.x += chunk.length + gutter;
      row.height = Math.max(row.height, chunk.height);
    };
    for (const chunk of chunks) {
      let row = rows.get(chunk.heightClass);
      if (row && row.x + chunk.length + gutter > width && row.x > gutter) {
        closeRow(row);
        row = undefined;
      }
      if (!row) {
        row = {x: gutter, height: 0, chunks: []};
        rows.set(chunk.heightClass, row);
      }
      add(row, chunk);
    }
    //the rows that are still open, 1 per height class and mostly empty: a tile of towers has some 30 classes, as rows
    //of their own they were a third of its atlas. Their chunks from high to low share rows, so the chunks of a row are
    //still about as high
    const open = [...rows.entries()].sort((a, b) => b[0] - a[0]);
    let last: Row | null = null;
    for (const [, row] of open) {
      for (const chunk of row.chunks) {
        if (last && last.x + chunk.length + gutter > width) {
          closeRow(last);
          last = null;
        }
        last ??= {x: gutter, height: 0, chunks: []};
        add(last, chunk);
      }
    }
    if (last) closeRow(last);
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
  for (const chunk of chunks) {
    const atlas = edgeAtlas[chunk.polygon];
    //the v at the base of the polygon: chunk.y is at the bottom of the chunk
    const v = chunk.y - (chunk.bottom - polygons[chunk.polygon].base);
    for (const ia of chunk.edges) {
      atlas[ia * 2] += chunk.x;
      atlas[ia * 2 + 1] = v;
    }
  }

  return {edgeAtlas, wallSurface: getWallSurfaceId(tileX, tileY), tileX, tileY};
}

/**
 * The BVH, records, edges and tree shapes of the tile (see CityTraceBlock). groundY: the height of the ground plane of the
 * ray tracing, a building or tree that stands on the ground (base 0) goes down to it, so no ray passes under its walls
 * between the ground layers. Null on the terrain: the bases are on the ground already (see terrainGrid.ts). ids: per
 * polygon its feature id, see CityTraceBlock (-1 for all without). Null without buildings and trees
 */
export function buildTraceBlock(polygons: BuildingPolygon[], trees: TraceTree[], groundY: number | null, ids?: ArrayLike<number>): CityTraceBlock | null {
  const buildingCount = polygons.length;
  const treeCount = trees.length;
  if (buildingCount + treeCount === 0) return null;

  const boxMin: Vector3[] = [];
  const boxMax: Vector3[] = [];
  let edgeCount = 0;
  for (const p of polygons) {
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
    boxMin.push(new Vector3(minX, base, minZ));
    boxMax.push(new Vector3(maxX, p.top, maxZ));
  }
  //a tree is in the box around its crown: the trunk is under it and narrower. In x, z the half axis of the crown is
  //(cos, -sin) * crownHalf (the corners of square in worldTrees.ts, z = -north), its box reaches |cos| + |sin| of it
  for (const t of trees) {
    const reach = (Math.abs(Math.cos(t.angle)) + Math.abs(Math.sin(t.angle))) * t.crownHalf;
    boxMin.push(new Vector3(t.x - reach, groundY ?? t.base, -t.y - reach));
    boxMax.push(new Vector3(t.x + reach, t.top, -t.y + reach));
  }

  //the stack of isOccluded holds the path of the top level BVH and of a tile (CITY_STACK_SIZE), see CityScene. A tile of
  //30000 buildings is 20 deep
  const bvh = new ModelBVH(boxMin, boxMax, BLOCK_DEPTH_MAX);
  const nodeCount = bvh.nodeCount;
  const recordStart = nodeCount * 4;
  const edgeStart = recordStart + buildingCount + treeCount;
  const shapeStart = edgeStart + edgeCount;
  const vec4Count = shapeStart + treeCount * 2;
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
        ? ((recordStart + (child & ~ModelBVH.leafFlag)) | CITY_LEAF) >>> 0
        : child * 4;
    }
  }

  let edge = 0;
  polygons.forEach((p, pi) => {
    const record = (recordStart + pi) * 4;
    const first = edge;
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
        edge++;
      }
    }
    f[record] = boxMin[pi].y;
    f[record + 1] = p.top;
    u[record + 2] = edgeStart + first;
    u[record + 3] = edge - first;
  });
  trees.forEach((t, ti) => {
    const record = (recordStart + buildingCount + ti) * 4;
    const shape = shapeStart + ti * 2;
    f[record] = groundY ?? t.base;
    f[record + 1] = t.top;
    u[record + 2] = shape;
    u[record + 3] = CITY_TREE;
    const o = shape * 4;
    f[o] = t.x;
    f[o + 1] = -t.y;
    f[o + 2] = Math.cos(t.angle) / t.crownHalf;
    f[o + 3] = -Math.sin(t.angle) / t.crownHalf;
    f[o + 4] = t.crownBase;
    f[o + 5] = t.trunkHalf / t.crownHalf;
  });

  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < boxMin.length; i++) {
    min[0] = Math.min(min[0], boxMin[i].x); min[1] = Math.min(min[1], boxMin[i].y); min[2] = Math.min(min[2], boxMin[i].z);
    max[0] = Math.max(max[0], boxMax[i].x); max[1] = Math.max(max[1], boxMax[i].y); max[2] = Math.max(max[2], boxMax[i].z);
  }
  const recordIds = new Int32Array(buildingCount).fill(-1);
  if (ids) recordIds.set(Array.from(ids).slice(0, buildingCount));
  return {words: u, ids: recordIds, nodeCount, buildingCount, treeCount, edgeCount, depth: bvh.depth, min, max};
}
