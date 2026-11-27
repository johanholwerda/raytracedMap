import type { VectorTileFeature } from '@mapbox/vector-tile';
import type { PbfReader } from 'pbf';
import type { TileCoord } from './geo';
import type MapProjection from './MapProjection';

/*
The geometry of the features of a vector tile as flat numbers, for the map worker (see worldGeometryCore.ts): the rings
of a feature are decoded from the tile into an array that is kept from feature to feature and from tile to tile, the
rings are clipped to the tile into 2 more and projected into the arrays of the caller. VectorTileFeature.loadGeometry
made a Point per vertex, classifyRings, clipRingToTile and clipLineToTile (the former pdokDecodeCore.ts) arrays and points per
ring and per cut, and the projection a number[] per polygon: some 10 MB of garbage per tile of a city, which the
scavenges of the worker had to copy around the data of the tile being built. Gives the same numbers as those, in the
same order: the tiles are built the same (see video/tileBench.mjs --compare).
 */

/**
 * The rings of a feature, see readRings: tile units, x, y per point. Ring r is the points [starts[r], starts[r + 1]).
 * The polygons of classifyPolygons: polygon p is the rings polyRings[polyStarts[p]] up to polyRings[polyStarts[p + 1]],
 * the first the outer one.
 */
export class FeatureRings {
  public points: Float64Array = new Float64Array(4096);
  public starts: Int32Array = new Int32Array(257);
  public ringCount = 0;
  public polyRings: Int32Array = new Int32Array(256);
  public polyStarts: Int32Array = new Int32Array(257);
  public polyCount = 0;
  private pointCount = 0;

  public ringStart(r: number): number {
    return this.starts[r];
  }

  public ringLength(r: number): number {
    return this.starts[r + 1] - this.starts[r];
  }

  public beginRing(): void {
    if (this.ringCount + 2 > this.starts.length) this.starts = growInt(this.starts, this.ringCount + 2);
    this.starts[this.ringCount++] = this.pointCount;
    this.starts[this.ringCount] = this.pointCount;
  }

  public addPoint(x: number, y: number): void {
    const o = this.pointCount * 2;
    if (o + 2 > this.points.length) this.points = growFloat(this.points, o + 2);
    this.points[o] = x;
    this.points[o + 1] = y;
    this.starts[this.ringCount] = ++this.pointCount;
  }

  public clear(): void {
    this.ringCount = 0;
    this.pointCount = 0;
    this.polyCount = 0;
    this.starts[0] = 0;
  }

  public addPolygonRing(r: number, count: number): void {
    if (count + 1 > this.polyRings.length) this.polyRings = growInt(this.polyRings, count + 1);
    this.polyRings[count] = r;
  }

  public endPolygon(ringsSoFar: number): void {
    if (this.polyCount + 2 > this.polyStarts.length) this.polyStarts = growInt(this.polyStarts, this.polyCount + 2);
    this.polyStarts[++this.polyCount] = ringsSoFar;
  }
}

/** An array of at least `size` numbers (twice as long or more) with the numbers of array. */
function growFloat(array: Float64Array, size: number): Float64Array {
  let length = array.length * 2;
  while (length < size) length *= 2;
  const grown = new Float64Array(length);
  grown.set(array);
  return grown;
}

function growInt(array: Int32Array, size: number): Int32Array {
  let length = array.length * 2;
  while (length < size) length *= 2;
  const grown = new Int32Array(length);
  grown.set(array);
  return grown;
}

/** A Float64Array that grows by doubling, kept between its uses: a count of the numbers in it. */
export class FloatList {
  public data: Float64Array;
  public length = 0;

  constructor(capacity = 4096) {
    this.data = new Float64Array(capacity);
  }

  /** Room for `count` more numbers: data can be written up to length + count. */
  public reserve(count: number): void {
    if (this.length + count > this.data.length) this.data = growFloat(this.data, this.length + count);
  }

  public push2(a: number, b: number): void {
    this.reserve(2);
    this.data[this.length] = a;
    this.data[this.length + 1] = b;
    this.length += 2;
  }
}

/**
 * The rings of a feature into rings (loadGeometry of @mapbox/vector-tile without its points): a moveTo starts a ring,
 * a closePath repeats its first point. Reads the fields of the feature that loadGeometry reads (_pbf, _geometry: private
 * in @mapbox/vector-tile 3, check them when it is updated).
 */
export function readRings(feature: VectorTileFeature, rings: FeatureRings): void {
  rings.clear();
  const f = feature as unknown as { _pbf: PbfReader; _geometry: number };
  if (f._geometry < 0) throw new Error('feature has no geometry');
  const pbf = f._pbf;
  pbf.pos = f._geometry;
  const end = pbf.readVarint() + pbf.pos;
  let cmd = 1, length = 0, x = 0, y = 0;
  let open = false;
  while (pbf.pos < end) {
    if (length <= 0) {
      const cmdLen = pbf.readVarint();
      cmd = cmdLen & 0x7;
      length = cmdLen >> 3;
      if (length === 0) continue;
    }
    length--;
    if (cmd === 1) {
      x += pbf.readSVarint();
      y += pbf.readSVarint();
      rings.beginRing();
      rings.addPoint(x, y);
      open = true;
    } else if (cmd === 2) {
      x += pbf.readSVarint();
      y += pbf.readSVarint();
      if (open) rings.addPoint(x, y);
    } else if (cmd === 7) {
      if (open) {
        const o = rings.ringStart(rings.ringCount - 1) * 2;
        rings.addPoint(rings.points[o], rings.points[o + 1]);
      }
    } else {
      throw new Error(`unknown command ${cmd}`);
    }
  }
}

/**
 * The rings into polygons, as classifyRings (the former pdokDecodeCore.ts) had them: the first ring with an area fixes the winding
 * of the outer rings, a ring with it starts a polygon, the others are its holes. 1 ring or none: 1 polygon of them.
 */
export function classifyPolygons(rings: FeatureRings): void {
  rings.polyCount = 0;
  rings.polyStarts[0] = 0;
  const n = rings.ringCount;
  if (n <= 1) {
    if (n === 1) rings.addPolygonRing(0, 0);
    rings.endPolygon(n);
    return;
  }
  const p = rings.points;
  let count = 0;
  let open = false;
  // 0: not known yet, 1: negative, -1: positive
  let exterior = 0;
  for (let r = 0; r < n; r++) {
    const s = rings.starts[r], e = rings.starts[r + 1];
    let area = 0;
    for (let i = s, j = e - 1; i < e; j = i++) area += (p[j * 2] - p[i * 2]) * (p[i * 2 + 1] + p[j * 2 + 1]);
    area /= 2;
    if (area === 0) continue;
    if (exterior === 0) exterior = area < 0 ? 1 : -1;
    if ((area < 0) === (exterior === 1)) {
      if (open) rings.endPolygon(count);
      rings.addPolygonRing(r, count++);
      open = true;
    } else if (open) {
      rings.addPolygonRing(r, count++);
    }
  }
  if (open) rings.endPolygon(count);
}

// clipRing: the 2 arrays it clips back and forth between, x, y per point
let clipA: Float64Array = new Float64Array(4096);
let clipB: Float64Array = new Float64Array(4096);

/**
 * Ring r of rings clipped to the tile [0, extent]^2, as clipRingToTile (Sutherland-Hodgman, the edges x >= 0,
 * x <= extent, y >= 0, y <= extent in turn). Returns its point count, the points are in clipped.points from
 * clipped.offset (x, y each), valid until the next call.
 */
export function clipRing(rings: FeatureRings, r: number, extent: number): number {
  let input = rings.points, inOff = rings.starts[r] * 2, inN = rings.starts[r + 1] - rings.starts[r];
  clipped.points = input;
  clipped.offset = inOff;
  for (let edge = 0; edge < 4 && inN > 0; edge++) {
    // at most 2 points per point of the input
    if (clipA.length < inN * 4) clipA = new Float64Array(inN * 8);
    if (clipB.length < inN * 4) clipB = new Float64Array(inN * 8);
    const out = edge & 1 ? clipB : clipA;
    let m = 0;
    for (let i = 0; i < inN; i++) {
      const prev = (i + inN - 1) % inN;
      const cx = input[inOff + i * 2], cy = input[inOff + i * 2 + 1];
      const px = input[inOff + prev * 2], py = input[inOff + prev * 2 + 1];
      const curIn = edge === 0 ? cx >= 0 : edge === 1 ? cx <= extent : edge === 2 ? cy >= 0 : cy <= extent;
      const prevIn = edge === 0 ? px >= 0 : edge === 1 ? px <= extent : edge === 2 ? py >= 0 : py <= extent;
      if (curIn !== prevIn) {
        // the intersection of prev to cur with the edge
        const t = edge === 0 ? (0 - px) / (cx - px) : edge === 1 ? (extent - px) / (cx - px) :
          edge === 2 ? (0 - py) / (cy - py) : (extent - py) / (cy - py);
        out[m++] = px + t * (cx - px);
        out[m++] = py + t * (cy - py);
      }
      if (curIn) {
        out[m++] = cx;
        out[m++] = cy;
      }
    }
    input = out;
    inOff = 0;
    inN = m / 2;
    clipped.points = out;
    clipped.offset = 0;
  }
  return inN;
}

/** Where the points of the last clipRing are: an array and the index of the first x (no view of them: an object each). */
export const clipped: { points: Float64Array; offset: number } = { points: new Float64Array(0), offset: 0 };

/**
 * Ring r of rings as a line clipped to the tile, as clipLineToTile: each segment clipped on its own (Liang-Barsky), the
 * clipped segments that continue each other joined. The pieces go into out as x, y per point, starts gets the point
 * index of each piece and of the end after the last: piece k is the points [starts[k], starts[k + 1]).
 */
export function clipLine(rings: FeatureRings, r: number, extent: number, out: FloatList, starts: number[]): void {
  out.length = 0;
  starts.length = 0;
  const p = rings.points, s = rings.starts[r], e = rings.starts[r + 1];
  // the points of the piece being made, 0: none (a piece has 2 at least)
  let current = 0;
  for (let i = s; i + 1 < e; i++) {
    const ax = p[i * 2], ay = p[i * 2 + 1];
    const dx = p[i * 2 + 2] - ax, dy = p[i * 2 + 3] - ay;
    let t0 = 0, t1 = 1, keep = true;
    for (let k = 0; k < 4 && keep; k++) {
      const pk = k === 0 ? -dx : k === 1 ? dx : k === 2 ? -dy : dy;
      const qk = k === 0 ? ax : k === 1 ? extent - ax : k === 2 ? ay : extent - ay;
      if (pk === 0) {
        if (qk < 0) keep = false;
      } else {
        const ratio = qk / pk;
        if (pk < 0) {
          if (ratio > t1) keep = false;
          else if (ratio > t0) t0 = ratio;
        } else {
          if (ratio < t0) keep = false;
          else if (ratio < t1) t1 = ratio;
        }
      }
    }
    // outside: the piece ends (it is in out already, from its start on)
    if (!keep) {
      current = 0;
      continue;
    }
    const x0 = ax + t0 * dx, y0 = ay + t0 * dy, x1 = ax + t1 * dx, y1 = ay + t1 * dy;
    const o = out.length;
    if (current > 0 && out.data[o - 2] === x0 && out.data[o - 1] === y0) {
      out.push2(x1, y1);
      current++;
    } else {
      starts.push(o / 2);
      out.push2(x0, y0);
      out.push2(x1, y1);
      current = 2;
    }
  }
  // the end of the last piece
  starts.push(out.length / 2);
}

/**
 * Tile units of a tile to map (x, north) coords: tileToLatLon and MapProjection.project (the same sums, in the same
 * order), see buildWorldTile. Most points are on the whole numbers of the extent: the north of each is worked out once.
 */
export class TileProjector {
  private readonly tileCount: number;
  private readonly lngMid: number;
  private readonly latMid: number;
  private readonly cosLat: number;
  private readonly scale: number;
  private cacheExtent = 0;
  private northCache = new Float64Array(0);

  constructor(private readonly tile: TileCoord, proj: MapProjection) {
    this.tileCount = Math.pow(2, tile.z);
    this.lngMid = proj.lngMid;
    this.latMid = proj.latMid;
    this.cosLat = proj.cosLat;
    this.scale = proj.scale;
  }

  /**
   * n points of src from index `from` (x, y each, tile units) projected into dst from index `to`: dst must have the
   * room. Arrays and indices in, nothing out: a double passed to or returned from a function that isn't inlined is a
   * number on the heap.
   */
  public project(src: Float64Array, from: number, n: number, extent: number, dst: Float64Array, to: number): void {
    const { tile, tileCount, lngMid, latMid, cosLat, scale } = this;
    if (extent !== this.cacheExtent) {
      this.cacheExtent = extent;
      this.northCache = new Float64Array(extent + 1).fill(NaN);
    }
    const cache = this.northCache;
    for (let k = 0; k < n; k++) {
      const px = src[from + k * 2], py = src[from + k * 2 + 1];
      const lon = ((tile.x + px / extent) / tileCount) * 360 - 180;
      let north: number;
      if (Number.isInteger(py) && py >= 0 && py <= extent) {
        north = cache[py];
        if (north !== north) {
          const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + py / extent)) / tileCount)));
          north = cache[py] = ((latRad * 180) / Math.PI - latMid) * scale;
        }
      } else {
        const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + py / extent)) / tileCount)));
        north = ((latRad * 180) / Math.PI - latMid) * scale;
      }
      dst[to + k * 2] = (lon - lngMid) * cosLat * scale;
      dst[to + k * 2 + 1] = north;
    }
  }
}

/**
 * A polygon of a feature clipped to the tile and projected (polygonToMap of worldGeometryCore.ts): coords x, north per
 * point (its first coords.length numbers), the point index of each hole and of each ring. Kept from polygon to polygon:
 * copy out what has to stay.
 */
export class MapPolygon {
  public coords = new FloatList();
  public holes: number[] = [];
  public ringStarts: number[] = [];
}

/**
 * Polygon p of rings (see classifyPolygons) clipped to the tile and projected into out. False when it is degenerate:
 * its outer ring clips to less than 3 points (holes that do are left out). A duplicate closing point is dropped.
 */
export function polygonToMap(rings: FeatureRings, p: number, extent: number, projector: TileProjector, out: MapPolygon): boolean {
  const first = rings.polyStarts[p], last = rings.polyStarts[p + 1];
  if (first === last) return false;
  const coords = out.coords;
  coords.length = 0;
  out.holes.length = 0;
  out.ringStarts.length = 0;
  for (let k = first; k < last; k++) {
    const n = clipRing(rings, rings.polyRings[k], extent);
    if (n < 3) {
      if (k === first) return false;
      continue;
    }
    const start = coords.length / 2;
    coords.reserve(n * 2);
    projector.project(clipped.points, clipped.offset, n, extent, coords.data, coords.length);
    coords.length += n * 2;
    // a duplicate closing point
    const end = coords.length, d = coords.data;
    if (n > 3 && d[start * 2] === d[end - 2] && d[start * 2 + 1] === d[end - 1]) coords.length = end - 2;
    if (k > first) out.holes.push(start);
    out.ringStarts.push(start);
  }
  return true;
}

/**
 * What the map worker reads the features of a tile with, kept from tile to tile (set the projector per tile): the rings
 * of a feature, a polygon of it, the clipped pieces of a line (clipLine) and a piece projected (projectPiece).
 */
export class TileFeatures {
  public readonly rings = new FeatureRings();
  public readonly polygon = new MapPolygon();
  public readonly clipped = new FloatList();
  public readonly pieces: number[] = [];
  public readonly line = new FloatList();
  public projector!: TileProjector;

  /** Line ring r of the feature read last clipped to the tile, into clipped and pieces. Returns the number of pieces. */
  public clipLine(r: number, extent: number): number {
    clipLine(this.rings, r, extent, this.clipped, this.pieces);
    return this.pieces.length - 1;
  }

  /** Piece k of the last clipLine projected into line (x, north per point). Returns its point count. */
  public projectPiece(k: number, extent: number): number {
    const from = this.pieces[k], n = this.pieces[k + 1] - from;
    const line = this.line;
    line.length = 0;
    line.reserve(n * 2);
    this.projector.project(this.clipped.data, from * 2, n, extent, line.data, 0);
    line.length = n * 2;
    return n;
  }

  /** Polygon p of the feature read last (after classifyPolygons) into polygon, see polygonToMap. */
  public polygonToMap(p: number, extent: number): boolean {
    return polygonToMap(this.rings, p, extent, this.projector, this.polygon);
  }
}
