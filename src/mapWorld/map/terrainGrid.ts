/*
The ground of a map tile as a height field (see ElevationService.getTileGrid): a grid of cells over the tile in map
coordinates (x, north), uniform in both, each cell split in 2 triangles along its diagonal from the south west corner
to the north east one. The ground is planar on each triangle, so a flat polygon lies on it exactly when it is cut along
the grid lines and the diagonals and its vertices are lifted to the ground: drapeTriangles (which leaves the cuts
out where the ground is planar enough). The ground mesh (buildMesh) has the same triangles.

The normals are smooth: at the grid points from the heights around them (central differences; the grid comes with a
ring of the neighbour tiles' heights, so 2 tiles get the same normals on their shared edge), inside a triangle of the
ground interpolated between its 3 corners. A vertex of the ground mesh or of a draped layer has the normal there in its
.w, see encodeTerrainNormal: the interpolation over a piece of a layer is the one over the ground under it.

Pure (no renderer imports): runs in the map worker.
 */

/**
 * A mesh in engine space: x, height, -north and the normal (encodeTerrainNormal, or NO_NORMAL) per vertex, 3 indices
 * per triangle.
 */
export type GridMesh = { vertices: number[]; indices: number[] };

/** Floats per vertex of a GridMesh and of buildMesh. */
export const GRID_VERTEX_FLOATS = 4;

// the steps of each of the 2 encoded components of a normal: 11 bits
const NORMAL_STEPS = 2047;
/** The .w of a vertex without a normal of its own (the flat ground, the water): up, see terrainNormal.wgsl. */
export const NO_NORMAL = -1;

/**
 * A unit normal with y > 0 in the .w of a vertex: x and z in 11 bits each, qx * 2048 + qz, an integer under 2^24, so a
 * float holds it exactly (a bitcast u32 could be a NaN, which a copy may change). y follows from x and z. Decoded by
 * decodeTerrainNormal of terrainNormal.wgsl.
 */
export function encodeTerrainNormal(nx: number, nz: number): number {
  const q = (n: number) => Math.round((Math.min(Math.max(n, -1), 1) * .5 + .5) * NORMAL_STEPS);
  return q(nx) * 2048 + q(nz);
}

export default class TerrainGrid {
  public readonly cells: number;
  /** The heights of the grid points (world units): (cells + 1)^2, row j (from the south) after row, i from the west. */
  public readonly heights: Float32Array;
  // the unit normal of each grid point, xyz in engine space (y up, z south)
  private readonly normals: Float32Array;
  private readonly dx: number;
  private readonly dy: number;

  /**
   * x0, y0: the south west corner of the tile, x1, y1 the north east one (map coordinates). bordered: the heights in
   * world units, (cells + 3)^2 from row and column -1 to cells + 1 (ElevationService.getTileGrid): the outer ring is in
   * the neighbour tiles, for the normals on the edges.
   */
  constructor(
    public readonly x0: number, public readonly y0: number, public readonly x1: number, public readonly y1: number,
    bordered: Float32Array,
  ) {
    const outer = Math.round(Math.sqrt(bordered.length));
    const n = this.cells = outer - 3;
    this.dx = (x1 - x0) / n;
    this.dy = (y1 - y0) / n;
    const side = n + 1;
    this.heights = new Float32Array(side * side);
    this.normals = new Float32Array(side * side * 3);
    const b = (i: number, j: number) => bordered[(j + 1) * outer + i + 1];
    for (let j = 0; j < side; j++) {
      for (let i = 0; i < side; i++) {
        this.heights[j * side + i] = b(i, j);
        // the surface y = h(x, north) at z = -north: the normal is (-dh/dx, 1, dh/dnorth), normalized
        const gx = (b(i + 1, j) - b(i - 1, j)) / (2 * this.dx);
        const gn = (b(i, j + 1) - b(i, j - 1)) / (2 * this.dy);
        const l = Math.hypot(gx, 1, gn);
        const o = (j * side + i) * 3;
        this.normals[o] = -gx / l;
        this.normals[o + 1] = 1 / l;
        this.normals[o + 2] = gn / l;
      }
    }
  }

  /** The x of grid column i: the last one exactly at x1, where the neighbour tile has its first. */
  public gridX(i: number): number {
    return i >= this.cells ? this.x1 : this.x0 + (this.x1 - this.x0) * i / this.cells;
  }

  public gridY(j: number): number {
    return j >= this.cells ? this.y1 : this.y0 + (this.y1 - this.y0) * j / this.cells;
  }

  private cellX(x: number): number {
    return Math.min(Math.max(Math.floor((x - this.x0) / this.dx), 0), this.cells - 1);
  }

  private cellY(y: number): number {
    return Math.min(Math.max(Math.floor((y - this.y0) / this.dy), 0), this.cells - 1);
  }

  /**
   * The height of the ground at (x, north), on the triangle of the mesh there. Outside the tile the nearest cell goes
   * on. The same point always gives the same height, whichever piece of a cut polygon it is a vertex of.
   */
  public heightAt(x: number, y: number): number {
    const i = this.cellX(x), j = this.cellY(y);
    const u = (x - this.gridX(i)) / this.dx, v = (y - this.gridY(j)) / this.dy;
    const side = this.cells + 1, h = this.heights;
    const sw = h[j * side + i], ne = h[(j + 1) * side + i + 1];
    if (u >= v) {
      // (sw, se, ne)
      const se = h[j * side + i + 1];
      return sw + u * (se - sw) + v * (ne - se);
    }
    // (sw, ne, nw)
    const nw = h[(j + 1) * side + i];
    return sw + v * (nw - sw) + u * (ne - nw);
  }

  /**
   * The normal of the ground at (x, north), encoded (encodeTerrainNormal): interpolated between the corners of the
   * triangle of the mesh there, as the rasterizer interpolates the ground mesh.
   */
  public normalAt(x: number, y: number): number {
    const i = this.cellX(x), j = this.cellY(y);
    const u = (x - this.gridX(i)) / this.dx, v = (y - this.gridY(j)) / this.dy;
    const side = this.cells + 1;
    const sw = j * side + i, se = sw + 1, nw = sw + side, ne = nw + 1;
    // the barycentric weights of the triangle (sw, se, ne) or (sw, ne, nw), see heightAt
    let a: number, wa: number, b: number, wb: number;
    const wsw = u >= v ? 1 - u : 1 - v;
    if (u >= v) { a = se; wa = u - v; b = ne; wb = v; } else { a = nw; wa = v - u; b = ne; wb = u; }
    const m = this.normals;
    const nx = m[sw * 3] * wsw + m[a * 3] * wa + m[b * 3] * wb;
    const ny = m[sw * 3 + 1] * wsw + m[a * 3 + 1] * wa + m[b * 3 + 1] * wb;
    const nz = m[sw * 3 + 2] * wsw + m[a * 3 + 2] * wa + m[b * 3 + 2] * wb;
    const l = Math.hypot(nx, ny, nz) || 1;
    return encodeTerrainNormal(nx / l, nz / l);
  }

  /**
   * The lowest ground under a footprint (ring [start, end) of coords): at its points and at the grid points inside it
   * (the ground under a large building can dip below all its corners).
   */
  public minHeight(coords: ArrayLike<number>, start: number, end: number): number {
    let min = Infinity;
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (let k = start; k < end; k++) {
      const x = coords[k * 2], y = coords[k * 2 + 1];
      min = Math.min(min, this.heightAt(x, y));
      bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x);
      by0 = Math.min(by0, y); by1 = Math.max(by1, y);
    }
    const side = this.cells + 1;
    const i0 = Math.max(Math.ceil((bx0 - this.x0) / this.dx), 0), i1 = Math.min(Math.floor((bx1 - this.x0) / this.dx), this.cells);
    const j0 = Math.max(Math.ceil((by0 - this.y0) / this.dy), 0), j1 = Math.min(Math.floor((by1 - this.y0) / this.dy), this.cells);
    for (let j = j0; j <= j1; j++) {
      const y = this.gridY(j);
      for (let i = i0; i <= i1; i++) {
        const h = this.heights[j * side + i];
        if (h < min && insideRing(coords, start, end, this.gridX(i), y)) min = h;
      }
    }
    return min;
  }

  /**
   * The ground mesh of the tile: a vertex per grid point (GRID_VERTEX_FLOATS, with its normal), 2 triangles per cell
   * (counter-clockwise in map space).
   */
  public buildMesh(): { vertices: Float32Array; indices: Uint32Array } {
    const n = this.cells, side = n + 1;
    const vertices = new Float32Array(side * side * GRID_VERTEX_FLOATS);
    for (let j = 0; j < side; j++) {
      const y = this.gridY(j);
      for (let i = 0; i < side; i++) {
        const k = j * side + i, o = k * GRID_VERTEX_FLOATS;
        vertices[o] = this.gridX(i);
        vertices[o + 1] = this.heights[k];
        vertices[o + 2] = -y;
        vertices[o + 3] = encodeTerrainNormal(this.normals[k * 3], this.normals[k * 3 + 2]);
      }
    }
    const indices = new Uint32Array(n * n * 6);
    let o = 0;
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const sw = j * side + i, se = sw + 1, nw = sw + side, ne = nw + 1;
        indices[o++] = sw; indices[o++] = se; indices[o++] = ne;
        indices[o++] = sw; indices[o++] = ne; indices[o++] = nw;
      }
    }
    return { vertices, indices };
  }

  /**
   * Drapes flat triangles (coords: x, north per point; tris: 3 point indices each) on the ground, lift (world units)
   * over it, appended to out (engine space). Where the ground under a triangle is planar within tolerance it stays whole
   * on the vertices of its points (shared with the triangles around it). Elsewhere it is split along the grid lines,
   * halving the cells it covers, until the ground under a piece is planar or the piece is in 1 cell, which its diagonal
   * cuts: then every piece lies on 1 triangle of the ground. The pieces keep the winding of their triangle and share
   * their vertices where the cuts give the same points. normals: the vertices get the smooth normal of the ground, else
   * NO_NORMAL (up: the water, which is flat however the ground under it slopes).
   */
  public drapeTriangles(
    coords: ArrayLike<number>, tris: ArrayLike<number>, lift: number, tolerance: number, normals: boolean, out: GridMesh,
  ): void {
    const drape: Drape = { out, lift, tolerance, normals, points: new Int32Array(coords.length / 2).fill(-1), cut: new PointMap() };
    for (let t = 0; t + 2 < tris.length; t += 3) {
      const ia = tris[t], ib = tris[t + 1], ic = tris[t + 2];
      const ax = coords[ia * 2], ay = coords[ia * 2 + 1], bx = coords[ib * 2], by = coords[ib * 2 + 1];
      const cx = coords[ic * 2], cy = coords[ic * 2 + 1];
      if ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) === 0) continue;
      const i0 = this.cellX(Math.min(ax, bx, cx)), i1 = this.cellX(Math.max(ax, bx, cx));
      const j0 = this.cellY(Math.min(ay, by, cy)), j1 = this.cellY(Math.max(ay, by, cy));
      if (this.planarUnder(i0, i1, j0, j1, tolerance)) {
        out.indices.push(this.pointVertex(drape, ia, ax, ay), this.pointVertex(drape, ib, bx, by), this.pointVertex(drape, ic, cx, cy));
        continue;
      }
      this.drapePolygon(drape, [ax, ay, bx, by, cx, cy], i0, i1, j0, j1);
    }
  }

  /** The vertex of source point k, made on first use. */
  private pointVertex(drape: Drape, k: number, x: number, y: number): number {
    if (drape.points[k] < 0) drape.points[k] = this.pushVertex(drape, x, y);
    return drape.points[k];
  }

  /** A vertex of out on the ground at (x, north) plus the lift, with the normal there. Returns its index. */
  private pushVertex(drape: Drape, x: number, y: number): number {
    const out = drape.out;
    const index = out.vertices.length / GRID_VERTEX_FLOATS;
    out.vertices.push(x, this.heightAt(x, y) + drape.lift, -y, drape.normals ? this.normalAt(x, y) : NO_NORMAL);
    return index;
  }

  /**
   * Whether the ground over cells [i0, i1] x [j0, j1] is planar within tolerance: every grid point of those cells within
   * it of the plane through 3 of their corners. The ground is linear between the grid points, so it is within tolerance
   * of the plane everywhere in the cells, and a flat piece over them on its vertices on the ground is within twice that.
   */
  private planarUnder(i0: number, i1: number, j0: number, j1: number, tolerance: number): boolean {
    i1++; j1++;
    const side = this.cells + 1, h = this.heights;
    const h00 = h[j0 * side + i0];
    const du = (h[j0 * side + i1] - h00) / (i1 - i0), dv = (h[j1 * side + i0] - h00) / (j1 - j0);
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        if (Math.abs(h[j * side + i] - (h00 + (i - i0) * du + (j - j0) * dv)) > tolerance) return false;
      }
    }
    return true;
  }

  /**
   * A convex piece within cells [i0, i1] x [j0, j1] (the cells of the split that made it, narrowed to its box): whole
   * where the ground under it is planar, else cut along the diagonal in 1 cell, else halved along a grid line.
   */
  private drapePolygon(drape: Drape, poly: number[], i0: number, i1: number, j0: number, j1: number): void {
    if (poly.length < 6) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let k = 0; k < poly.length; k += 2) {
      minX = Math.min(minX, poly[k]); maxX = Math.max(maxX, poly[k]);
      minY = Math.min(minY, poly[k + 1]); maxY = Math.max(maxY, poly[k + 1]);
    }
    // a point on the grid line between 2 ranges is in both by its box: the range of the split decides
    i0 = Math.min(Math.max(this.cellX(minX), i0), i1); i1 = Math.max(Math.min(this.cellX(maxX), i1), i0);
    j0 = Math.min(Math.max(this.cellY(minY), j0), j1); j1 = Math.max(Math.min(this.cellY(maxY), j1), j0);
    if (this.planarUnder(i0, i1, j0, j1, drape.tolerance)) {
      this.emitPolygon(drape, poly);
      return;
    }
    if (i0 === i1 && j0 === j1) {
      // in 1 cell, its 2 triangles not coplanar: cut along the diagonal. u - v >= 0 is the south east half, the other
      // half the negated function, so both get the same cut vertices
      const sx = 1 / this.dx, sy = 1 / this.dy;
      const ox = this.gridX(i0), oy = this.gridY(j0);
      const a = sx, b = -sy, c = -ox * sx + oy * sy;
      this.emitPolygon(drape, clipHalf(poly, a, b, c, []));
      this.emitPolygon(drape, clipHalf(poly, -a, -b, -c, []));
      return;
    }
    // halve the cells along the longer side, on a grid line. Both sides of the cut get the same new vertices (clipHalf)
    if (i1 - i0 >= j1 - j0) {
      const m = (i0 + i1 + 1) >> 1, x = this.gridX(m);
      this.drapePolygon(drape, clipHalf(poly, -1, 0, x, []), i0, m - 1, j0, j1);
      this.drapePolygon(drape, clipHalf(poly, 1, 0, -x, []), m, i1, j0, j1);
    } else {
      const m = (j0 + j1 + 1) >> 1, y = this.gridY(m);
      this.drapePolygon(drape, clipHalf(poly, 0, -1, y, []), i0, i1, j0, m - 1);
      this.drapePolygon(drape, clipHalf(poly, 0, 1, -y, []), i0, i1, m, j1);
    }
  }

  /** A convex piece as a fan, its points lifted to the ground plus lift. Slivers without area are dropped. */
  private emitPolygon(drape: Drape, poly: number[]): void {
    const n = poly.length / 2;
    if (n < 3) return;
    let area = 0;
    for (let k = 0; k < n; k++) {
      const q = (k + 1) % n;
      area += poly[k * 2] * poly[q * 2 + 1] - poly[q * 2] * poly[k * 2 + 1];
    }
    if (Math.abs(area) < 1e-18) return;
    const { out, cut } = drape;
    const index = new Array<number>(n);
    for (let k = 0; k < n; k++) {
      const x = poly[k * 2], y = poly[k * 2 + 1];
      let v = cut.get(x, y);
      if (v < 0) {
        v = this.pushVertex(drape, x, y);
        cut.set(x, y, v);
      }
      index[k] = v;
    }
    for (let k = 1; k + 1 < n; k++) out.indices.push(index[0], index[k], index[k + 1]);
  }
}

/** Whether (x, y) is inside ring [s, e) of a flat xy array (even-odd). */
export function insideRing(c: ArrayLike<number>, s: number, e: number, x: number, y: number): boolean {
  let inside = false;
  for (let i = s, j = e - 1; i < e; j = i++) {
    const xi = c[i * 2], yi = c[i * 2 + 1], xj = c[j * 2], yj = c[j * 2 + 1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * One drapeTriangles call: the output, the lift, the tolerance and whether the vertices get normals, the vertices of the source points (-1: none yet) and
 * the vertices of the pieces by position.
 */
type Drape = { out: GridMesh; lift: number; tolerance: number; normals: boolean; points: Int32Array; cut: PointMap };

/**
 * A map from a point (x, y, exactly) to a number, for the vertices of the cuts: by a hash of the bits of the 2 floats,
 * with chains of the points of the same hash. A string key (x + ',' + y) made the floats strings: most of the time of
 * the draping went to that.
 */
class PointMap {
  private static readonly bits = new Float64Array(2);
  private static readonly words = new Uint32Array(PointMap.bits.buffer);
  // per hash the first entry, then per entry the next one of the same hash (-1: none)
  private readonly first = new Map<number, number>();
  private readonly next: number[] = [];
  private readonly xs: number[] = [];
  private readonly ys: number[] = [];
  private readonly values: number[] = [];

  private static hash(x: number, y: number): number {
    // + 0: -0 is 0, as x === y has it
    PointMap.bits[0] = x + 0;
    PointMap.bits[1] = y + 0;
    const w = PointMap.words;
    let h = Math.imul(w[0] ^ Math.imul(w[1], 0x9e3779b1), 0x85ebca6b);
    h = Math.imul(h ^ w[2] ^ Math.imul(w[3], 0xc2b2ae35), 0x27d4eb2f);
    // a small integer (a Smi): the fast keys of a Map
    return (h ^ (h >>> 15)) & 0x3fffffff;
  }

  /** The value of the point, -1 when it has none. */
  public get(x: number, y: number): number {
    for (let e = this.first.get(PointMap.hash(x, y)) ?? -1; e >= 0; e = this.next[e]) {
      if (this.xs[e] === x && this.ys[e] === y) return this.values[e];
    }
    return -1;
  }

  /** For a point that has no value yet. */
  public set(x: number, y: number, value: number): void {
    const h = PointMap.hash(x, y);
    const e = this.values.length;
    this.xs.push(x);
    this.ys.push(y);
    this.values.push(value);
    this.next.push(this.first.get(h) ?? -1);
    this.first.set(h, e);
  }
}

/**
 * The part of the convex polygon poly (x, y pairs) where a * x + b * y + c >= 0 (Sutherland-Hodgman), into out (which
 * is returned). A cut edge gets the same new vertex in the part of the negated function: the factor t is the same
 * (negating a float is exact), so the 2 sides of a cut meet without a crack. A cut along a grid line (a or b 0, the
 * other +-1) puts the new vertex on the line exactly.
 */
function clipHalf(poly: number[], a: number, b: number, c: number, out: number[]): number[] {
  out.length = 0;
  const n = poly.length / 2;
  let inside = 0;
  for (let k = 0; k < n; k++) if (a * poly[k * 2] + b * poly[k * 2 + 1] + c >= 0) inside++;
  if (inside === n) {
    for (let k = 0; k < poly.length; k++) out.push(poly[k]);
    return out;
  }
  if (inside === 0) return out;
  for (let k = 0; k < n; k++) {
    const px = poly[k * 2], py = poly[k * 2 + 1];
    const q = (k + 1) % n;
    const qx = poly[q * 2], qy = poly[q * 2 + 1];
    const fp = a * px + b * py + c, fq = a * qx + b * qy + c;
    if (fp >= 0) out.push(px, py);
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out.push(b === 0 ? -c / a : px + t * (qx - px), a === 0 ? -c / b : py + t * (qy - py));
    }
  }
  return out;
}
