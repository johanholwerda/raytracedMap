import type { VectorTile } from '@mapbox/vector-tile';
import { classifyPolygons, FloatList, readRings } from './tileGeometry';
import type { TileFeatures } from './tileGeometry';

/*
Random trees in the wood areas of a tile (the landcover class wood of OpenMapTiles: OpenFreeMap has no single trees).
A tree is 2 boxes, a trunk and a crown on it, each turned by a random angle, which buildBuildings (worldGeometryCore.ts)
adds as footprints next to the buildings: they get walls in the wall atlas and roofs on the roof charts of the
lightmap, records in the ray tracing (they hide the sky like a building) and their vertices in the building mesh of the
tile, so all the trees of a tile are 1 draw. The vertex .w (the feature id of a building) is TREE_MATERIAL_BASE_ID - k
on a tree: k is the palette entry of TREE_COLORS (the trunk, or one of the shades of the crowns), see mapGBuffer.wgsl.
The trees are on a grid of TREE_SPACING_M with a random offset per cell, the grid in world units of the projection: a
wood polygon cut by 2 tiles gets every tree once, the same each time a tile loads. Not on roads, railways, paths or
waterways (their lines with a margin), which run through the woods without a hole in the wood polygon.
 */

// the vertex .w of a tree with palette entry k is TREE_MATERIAL_BASE_ID - k: under NO_FEATURE_ID (-1), not pickable
export const TREE_MATERIAL_BASE_ID = -2;

// the albedos of the trees (as the params, not squared), see MapGeometry: the trunk, then the shades of the crowns
export const TREE_TRUNK_COLOR: [number, number, number] = [0.36, 0.28, 0.21];
export const TREE_CROWN_COLORS: [number, number, number][] = [
  [0.28, 0.45, 0.22], [0.31, 0.48, 0.24], [0.25, 0.41, 0.21], [0.33, 0.50, 0.25],
  [0.30, 0.44, 0.20], [0.27, 0.47, 0.26], [0.34, 0.47, 0.22], [0.26, 0.43, 0.24],
];
export const TREE_COLORS: [number, number, number][] = [TREE_TRUNK_COLOR, ...TREE_CROWN_COLORS];

// the grid of the trees, the part of a cell its tree can be off its center and the part of the cells with a tree
const TREE_SPACING_M = 9;
const TREE_JITTER = 0.35;
const TREE_FILL = 0.85;
// the most trees of a tile: a tile of only forest has about 25000 cells, the rest are left out at random. Each tree
// is 16 vertices (2 per corner of its 2 boxes) and 1 ray tracing record
const TREES_PER_TILE_MAX = 12000;
// the height, the part of it under the crown, the crown width and the trunk width, in m
const TREE_HEIGHT_M = [9, 20];
const TREE_CROWN_BASE = [0.25, 0.4];
const TREE_CROWN_WIDTH_M = [4, 7.5];
const TREE_TRUNK_WIDTH_M = [0.5, 0.8];
// the space between a crown and the side of a line it has to stay off, in m
const LINE_MARGIN_M = 0.5;
// the widths of the lines the trees stay off, in m, per layer and class (the widths of worldGeometryCore, a bit wider)
const LINE_WIDTHS_M: Record<string, Record<string, number>> = {
  transportation: {
    motorway: 16, trunk: 14, primary: 12, secondary: 10, tertiary: 8, minor: 6, service: 4.5, track: 3.5, raceway: 9,
    busway: 7, bus_guideway: 5, rail: 4, transit: 3, path: 2,
  },
  waterway: { river: 12, canal: 10, stream: 3, drain: 2, ditch: 2 },
};

/**
 * A box of a tree: its footprint (4 corners, x and north in world units, CCW), base and top in m. The footprint is
 * square: x, y (north) its center and half its side in world units, turned by angle (see square), for the ray tracing.
 */
export type TreeBox = { coords: Float64Array; base: number; top: number; material: number; x: number; y: number; half: number; angle: number };

/** The trees of a tile as boxes, trunk and crown of a tree after each other. */
export function buildTrees(vt: VectorTile, features: TileFeatures, metersToNorm: number): TreeBox[] {
  const layer = vt.layers['landcover'];
  if (!layer) return [];
  const extent = layer.extent || 4096;
  const spacing = TREE_SPACING_M * metersToNorm;
  // the cells with a tree, so wood polygons that overlap get it once
  const taken = new Set<string>();
  type Tree = { x: number; y: number; seed: number };
  const trees: Tree[] = [];
  const crossings: number[] = [];

  // the wood polygons, each as it is read (the polygon of features is the next one's after)
  let woods = 0;
  const rings = features.rings;
  for (let i = 0; i < layer.length; i++) {
    const f = layer.feature(i);
    if (f.type !== 3 || String(f.properties.class) !== 'wood') continue;
    readRings(f, rings);
    classifyPolygons(rings);
    for (let p = 0; p < rings.polyCount; p++) {
      if (!features.polygonToMap(p, extent)) continue;
      woods++;
      const coords = features.polygon.coords;
      treesOfWood(coords.data, coords.length, features.polygon.ringStarts, spacing, taken, trees, crossings);
    }
  }
  if (woods === 0) return [];
  const lines = new LineGrid(vt, features, metersToNorm);
  const keep = Math.min(1, TREES_PER_TILE_MAX / Math.max(trees.length, 1));
  const boxes: TreeBox[] = [];
  for (const { x, y, seed } of trees) {
    if (keep < 1 && hash01(seed, 4) >= keep) continue;
    const height = lerp(TREE_HEIGHT_M, hash01(seed, 5));
    const crownBase = height * lerp(TREE_CROWN_BASE, hash01(seed, 6));
    const crownWidth = lerp(TREE_CROWN_WIDTH_M, hash01(seed, 7));
    if (lines.near(x, y, (crownWidth * Math.SQRT1_2 + LINE_MARGIN_M) * metersToNorm)) continue;
    const trunkWidth = lerp(TREE_TRUNK_WIDTH_M, hash01(seed, 8));
    const angle = hash01(seed, 9) * Math.PI * 0.5;
    const shade = 1 + Math.min(Math.floor(hash01(seed, 10) * TREE_CROWN_COLORS.length), TREE_CROWN_COLORS.length - 1);
    const trunkHalf = trunkWidth * 0.5 * metersToNorm, crownHalf = crownWidth * 0.5 * metersToNorm;
    boxes.push({ coords: square(x, y, trunkHalf, angle), base: 0, top: crownBase, material: 0, x, y, half: trunkHalf, angle });
    boxes.push({ coords: square(x, y, crownHalf, angle), base: crownBase, top: height, material: shade, x, y, half: crownHalf, angle });
  }
  return boxes;
}

/**
 * The trees of a wood polygon (the first `length` numbers of coords, x, north per point, its rings from ringStarts) into
 * trees, see buildTrees.
 */
function treesOfWood(
  coords: Float64Array, length: number, ringStarts: number[], spacing: number, taken: Set<string>, trees: { x: number; y: number; seed: number }[],
  crossings: number[],
): void {
  let minY = Infinity, maxY = -Infinity;
  for (let k = 1; k < length; k += 2) { minY = Math.min(minY, coords[k]); maxY = Math.max(maxY, coords[k]); }
  // per row of cells the spans inside the polygon along the line through the cell centers, even-odd over all rings
  for (let iy = Math.floor(minY / spacing); iy * spacing <= maxY; iy++) {
    const y = (iy + 0.5) * spacing;
    crossings.length = 0;
    for (let r = 0; r < ringStarts.length; r++) {
      const s = ringStarts[r], e = r + 1 < ringStarts.length ? ringStarts[r + 1] : length / 2;
      for (let i = s, j = e - 1; i < e; j = i++) {
        const ay = coords[j * 2 + 1], by = coords[i * 2 + 1];
        if ((ay > y) === (by > y)) continue;
        const ax = coords[j * 2], bx = coords[i * 2];
        crossings.push(ax + (y - ay) / (by - ay) * (bx - ax));
      }
    }
    crossings.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    for (let c = 0; c + 1 < crossings.length; c += 2) {
      for (let ix = Math.ceil(crossings[c] / spacing - 0.5); (ix + 0.5) * spacing < crossings[c + 1]; ix++) {
        const key = `${ix},${iy}`;
        if (taken.has(key)) continue;
        taken.add(key);
        const seed = (Math.imul(ix, 73856093) ^ Math.imul(iy, 19349663)) >>> 0;
        if (hash01(seed, 1) > TREE_FILL) continue;
        const x = (ix + 0.5 + (hash01(seed, 2) * 2 - 1) * TREE_JITTER) * spacing;
        const ty = (iy + 0.5 + (hash01(seed, 3) * 2 - 1) * TREE_JITTER) * spacing;
        trees.push({ x, y: ty, seed });
      }
    }
  }
}

// the corners of a square around (x, y), half its side h, turned by angle: CCW
function square(x: number, y: number, h: number, angle: number): Float64Array {
  const c = Math.cos(angle) * h, s = Math.sin(angle) * h;
  return new Float64Array([x - c + s, y - s - c, x + c + s, y + s - c, x + c - s, y + s + c, x - c - s, y - s + c]);
}

function lerp(range: number[], t: number): number {
  return range[0] + (range[1] - range[0]) * t;
}

// a value in [0, 1) per seed and salt
function hash01(seed: number, salt: number): number {
  let h = (seed ^ Math.imul(salt, 0x9E3779B1)) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

// the segments of the LineGrid of a tile, kept from tile to tile: ax, ay, bx, by and the half width per segment
const lineSegments = new FloatList();

/**
 * The segments of the lines the trees stay off (see LINE_WIDTHS_M), in cells of the map, for the distance test. The
 * cells are a grid over the box of the segments, their segments 1 array: the segments of cell c are entries[starts[c]]
 * to entries[starts[c + 1]] (as a Map from a string per cell to an array, they were 1 MB of garbage per tile of a city).
 */
class LineGrid {
  private readonly cellSize: number;
  // the first cell of the grid and its size, in cells
  private cellX0 = 0;
  private cellY0 = 0;
  private width = 0;
  private height = 0;
  private starts = new Int32Array(1);
  private entries = new Int32Array(0);

  constructor(vt: VectorTile, features: TileFeatures, metersToNorm: number) {
    this.cellSize = 32 * metersToNorm;
    const segments = lineSegments;
    segments.length = 0;
    const rings = features.rings;
    for (const layerName of Object.keys(LINE_WIDTHS_M)) {
      const layer = vt.layers[layerName];
      if (!layer) continue;
      const widths = LINE_WIDTHS_M[layerName];
      const extent = layer.extent || 4096;
      for (let i = 0; i < layer.length; i++) {
        const f = layer.feature(i);
        if (f.type !== 2) continue;
        const p = f.properties;
        if (p.brunnel === 'tunnel') continue;
        const width = widths[String(p.class)];
        if (width === undefined) continue;
        const halfW = width * 0.5 * metersToNorm;
        readRings(f, rings);
        for (let r = 0; r < rings.ringCount; r++) {
          if (rings.ringLength(r) < 2) continue;
          const pieces = features.clipLine(r, extent);
          for (let k = 0; k < pieces; k++) {
            const n = features.projectPiece(k, extent);
            const line = features.line.data;
            segments.reserve(n * 5);
            const d = segments.data;
            for (let j = 0; j + 3 < n * 2; j += 2) {
              const o = segments.length;
              d[o] = line[j]; d[o + 1] = line[j + 1]; d[o + 2] = line[j + 2]; d[o + 3] = line[j + 3]; d[o + 4] = halfW;
              segments.length = o + 5;
            }
          }
        }
      }
    }
    this.index();
  }

  /** The cells of segment o (its index in the segments times 5): the box around it, widened by its half width. */
  private cellRange(o: number, range: Int32Array): void {
    const d = lineSegments.data, s = this.cellSize;
    const ax = d[o], ay = d[o + 1], bx = d[o + 2], by = d[o + 3], halfW = d[o + 4];
    range[0] = Math.floor((Math.min(ax, bx) - halfW) / s); range[1] = Math.floor((Math.max(ax, bx) + halfW) / s);
    range[2] = Math.floor((Math.min(ay, by) - halfW) / s); range[3] = Math.floor((Math.max(ay, by) + halfW) / s);
  }

  /** The grid: its box, the segment count per cell, then the segments of each cell. */
  private index(): void {
    const count = lineSegments.length / 5;
    if (count === 0) return;
    const range = new Int32Array(4);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < count; i++) {
      this.cellRange(i * 5, range);
      x0 = Math.min(x0, range[0]); x1 = Math.max(x1, range[1]);
      y0 = Math.min(y0, range[2]); y1 = Math.max(y1, range[3]);
    }
    this.cellX0 = x0;
    this.cellY0 = y0;
    const w = this.width = x1 - x0 + 1, h = this.height = y1 - y0 + 1;
    const starts = this.starts = new Int32Array(w * h + 1);
    for (let i = 0; i < count; i++) {
      this.cellRange(i * 5, range);
      for (let cy = range[2]; cy <= range[3]; cy++) {
        for (let cx = range[0]; cx <= range[1]; cx++) starts[(cy - y0) * w + cx - x0 + 1]++;
      }
    }
    for (let c = 0; c < w * h; c++) starts[c + 1] += starts[c];
    const fill = starts.slice(0, w * h);
    const entries = this.entries = new Int32Array(starts[w * h]);
    for (let i = 0; i < count; i++) {
      this.cellRange(i * 5, range);
      for (let cy = range[2]; cy <= range[3]; cy++) {
        for (let cx = range[0]; cx <= range[1]; cx++) entries[fill[(cy - y0) * w + cx - x0]++] = i;
      }
    }
  }

  /** Whether a point is closer than its half width plus radius to a line. */
  public near(x: number, y: number, radius: number): boolean {
    const s = this.cellSize;
    // the cells around the point, in the grid
    const x0 = Math.max(Math.floor((x - radius) / s), this.cellX0), x1 = Math.min(Math.floor((x + radius) / s), this.cellX0 + this.width - 1);
    const y0 = Math.max(Math.floor((y - radius) / s), this.cellY0), y1 = Math.min(Math.floor((y + radius) / s), this.cellY0 + this.height - 1);
    const seg = lineSegments.data, starts = this.starts, entries = this.entries;
    for (let cy = y0; cy <= y1; cy++) {
      for (let cx = x0; cx <= x1; cx++) {
        const c = (cy - this.cellY0) * this.width + cx - this.cellX0;
        for (let e = starts[c]; e < starts[c + 1]; e++) {
          const o = entries[e] * 5;
          const ax = seg[o], ay = seg[o + 1], dx = seg[o + 2] - ax, dy = seg[o + 3] - ay;
          const len2 = dx * dx + dy * dy;
          const t = len2 > 0 ? Math.min(Math.max(((x - ax) * dx + (y - ay) * dy) / len2, 0), 1) : 0;
          const ex = ax + dx * t - x, ey = ay + dy * t - y;
          const r = radius + seg[o + 4];
          if (ex * ex + ey * ey < r * r) return true;
        }
      }
    }
    return false;
  }
}
