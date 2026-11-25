/*
The ground of a map tile as a height field (see ElevationService.getTileGrid): a grid of cells over the tile in map
coordinates (x, north), uniform in both, each cell split in 2 triangles along its diagonal from the south west corner
to the north east one. The ground is planar on each triangle, so a flat polygon lies on it exactly when it is cut along
the grid lines and the diagonals and its vertices are lifted to the ground: drapeTriangles (which leaves the cuts
out where the ground is planar enough). The ground mesh (buildMesh) has the same triangles.

Pure (no renderer imports): runs in the map worker.
 */

/** A mesh in engine space: xyz per vertex (x, height, -north), 3 indices per triangle. */
export type GridMesh = { vertices: number[]; indices: number[] };

export default class TerrainGrid {
  public readonly cells: number;
  private readonly dx: number;
  private readonly dy: number;

  /**
   * x0, y0: the south west corner of the tile, x1, y1 the north east one (map coordinates). heights: (cells + 1)^2,
   * row j (from the south) after row, column i from the west, in world units.
   */
  constructor(
    public readonly x0: number, public readonly y0: number, public readonly x1: number, public readonly y1: number,
    public readonly heights: Float32Array,
  ) {
    this.cells = Math.round(Math.sqrt(heights.length)) - 1;
    this.dx = (x1 - x0) / this.cells;
    this.dy = (y1 - y0) / this.cells;
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

  /** The ground mesh of the tile: a vertex per grid point, 2 triangles per cell (counter-clockwise in map space). */
  public buildMesh(): { vertices: Float32Array; indices: Uint32Array } {
    const n = this.cells, side = n + 1;
    const vertices = new Float32Array(side * side * 3);
    for (let j = 0; j < side; j++) {
      const y = this.gridY(j);
      for (let i = 0; i < side; i++) {
        const o = (j * side + i) * 3;
        vertices[o] = this.gridX(i);
        vertices[o + 1] = this.heights[j * side + i];
        vertices[o + 2] = -y;
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
   * their vertices where the cuts give the same points.
   */
  public drapeTriangles(coords: ArrayLike<number>, tris: ArrayLike<number>, lift: number, tolerance: number, out: GridMesh): void {
    const drape: Drape = { out, lift, tolerance, points: new Int32Array(coords.length / 2).fill(-1), cut: new Map() };
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
    if (drape.points[k] < 0) {
      drape.points[k] = drape.out.vertices.length / 3;
      drape.out.vertices.push(x, this.heightAt(x, y) + drape.lift, -y);
    }
    return drape.points[k];
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
      const key = x + ',' + y;
      let v = cut.get(key);
      if (v === undefined) {
        v = out.vertices.length / 3;
        out.vertices.push(x, this.heightAt(x, y) + drape.lift, -y);
        cut.set(key, v);
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
 * One drapeTriangles call: the output, the lift and the tolerance, the vertices of the source points (-1: none yet) and
 * the vertices of the pieces by position.
 */
type Drape = { out: GridMesh; lift: number; tolerance: number; points: Int32Array; cut: Map<string, number> };

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
