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
import { hypot3 } from 'webGPU-framework/renderer/utils/MathUtils';

/** Floats per vertex of a GridMesh and of buildMesh. */
export const GRID_VERTEX_FLOATS = 4;

/**
 * A mesh in engine space that drapeTriangles appends to: x, height, -north and the normal (encodeTerrainNormal, or
 * NO_NORMAL) per vertex, 3 indices per triangle. In typed arrays that grow by doubling: as number[] the pushes and the
 * copy into a Float32Array at the end were a tenth of the time of a tile, and garbage of 8 bytes per float. Made once
 * and used for mesh after mesh (reset): a new large array is memory the OS has to hand out and clear, which takes
 * much longer than writing to it, and longer still with all the map workers asking at once.
 */
export class GridMesh {
  // public for TerrainGrid, which writes the vertices in place (see allocVertex)
  public vertexData = new Float32Array(16384 * GRID_VERTEX_FLOATS);
  private indexData = new Uint32Array(16384 * 6);
  public vertexCount = 0;
  public indexCount = 0;

  /** Empty again, the arrays stay. */
  public reset(): void {
    this.vertexCount = 0;
    this.indexCount = 0;
  }

  /**
   * A new vertex: returns where its GRID_VERTEX_FLOATS go in vertexData (read vertexData after the call, it may have
   * grown), its index is that / GRID_VERTEX_FLOATS. Not a push of its 4 floats: numbers passed to a function that isn't
   * inlined each become an object on the heap.
   */
  public allocVertex(): number {
    const o = this.vertexCount * GRID_VERTEX_FLOATS;
    if (o + GRID_VERTEX_FLOATS > this.vertexData.length) {
      const grown = new Float32Array(this.vertexData.length * 2);
      grown.set(this.vertexData);
      this.vertexData = grown;
    }
    this.vertexCount++;
    return o;
  }

  public pushTriangle(a: number, b: number, c: number): void {
    const o = this.indexCount;
    if (o + 3 > this.indexData.length) {
      const grown = new Uint32Array(this.indexData.length * 2);
      grown.set(this.indexData);
      this.indexData = grown;
    }
    const d = this.indexData;
    d[o] = a; d[o + 1] = b; d[o + 2] = c;
    this.indexCount = o + 3;
  }

  /** The vertices and the indices in arrays of their own, of their size. */
  public finish(): { vertices: Float32Array; indices: Uint32Array } {
    return {
      vertices: this.vertexData.slice(0, this.vertexCount * GRID_VERTEX_FLOATS),
      indices: this.indexData.slice(0, this.indexCount),
    };
  }
}

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
  // the heights with the ring of the neighbour tiles' around them, see the constructor and minHeight
  private readonly bordered: Float32Array;
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
    this.bordered = bordered;
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
        const l = hypot3(gx, 1, gn);
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
    const l = hypot3(nx, ny, nz) || 1;
    return encodeTerrainNormal(nx / l, nz / l);
  }

  /**
   * The lowest ground under a footprint (ring [start, end) of coords): at its points and at the grid points inside it
   * (the ground under a large building can dip below all its corners). The footprint may reach a cell past the tile:
   * there the ring of the neighbour tiles' heights is read, so a building the edge of 2 tiles cuts, whose whole
   * footprint each tile has (the buffer of the vector tile, 1/64 of it: one cell at z14), stands on the same ground in
   * both. Farther out the cell of the ring goes on.
   */
  public minHeight(coords: ArrayLike<number>, start: number, end: number): number {
    const n = this.cells, outer = n + 3, b = this.bordered;
    let min = Infinity;
    let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    for (let k = start; k < end; k++) {
      const x = coords[k * 2], y = coords[k * 2 + 1];
      bx0 = Math.min(bx0, x); bx1 = Math.max(bx1, x);
      by0 = Math.min(by0, y); by1 = Math.max(by1, y);
      // heightAt on the cells from -1 to cells (the ring), see its sums
      const i = Math.min(Math.max(Math.floor((x - this.x0) / this.dx), -1), n);
      const j = Math.min(Math.max(Math.floor((y - this.y0) / this.dy), -1), n);
      const u = (x - this.ringX(i)) / this.dx, v = (y - this.ringY(j)) / this.dy;
      const o = (j + 1) * outer + i + 1;
      const sw = b[o], ne = b[o + outer + 1];
      const h = u >= v ? sw + u * (b[o + 1] - sw) + v * (ne - b[o + 1]) : sw + v * (b[o + outer] - sw) + u * (ne - b[o + outer]);
      if (h < min) min = h;
    }
    const i0 = Math.max(Math.ceil((bx0 - this.x0) / this.dx), -1), i1 = Math.min(Math.floor((bx1 - this.x0) / this.dx), n + 1);
    const j0 = Math.max(Math.ceil((by0 - this.y0) / this.dy), -1), j1 = Math.min(Math.floor((by1 - this.y0) / this.dy), n + 1);
    for (let j = j0; j <= j1; j++) {
      const y = this.ringY(j);
      for (let i = i0; i <= i1; i++) {
        const h = b[(j + 1) * outer + i + 1];
        if (h < min && insideRing(coords, start, end, this.ringX(i), y)) min = h;
      }
    }
    return min;
  }

  // gridX and gridY on the ring too: column and row -1 and cells + 1, a cell out
  private ringX(i: number): number {
    return i < 0 ? this.x0 - this.dx : i > this.cells ? this.x1 + this.dx : this.gridX(i);
  }

  private ringY(j: number): number {
    return j < 0 ? this.y0 - this.dy : j > this.cells ? this.y1 + this.dy : this.gridY(j);
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
    const scratch = drapeScratch ??= new DrapeScratch();
    scratch.cut.clear();
    const drape: Drape = { out, lift, tolerance, normals, points: scratch.pointsFor(coords.length / 2), cut: scratch.cut, scratch };
    const triangle = scratch.pieceAt(0);
    for (let t = 0; t + 2 < tris.length; t += 3) {
      const ia = tris[t], ib = tris[t + 1], ic = tris[t + 2];
      const ax = coords[ia * 2], ay = coords[ia * 2 + 1], bx = coords[ib * 2], by = coords[ib * 2 + 1];
      const cx = coords[ic * 2], cy = coords[ic * 2 + 1];
      if ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay) === 0) continue;
      const i0 = this.cellX(Math.min(ax, bx, cx)), i1 = this.cellX(Math.max(ax, bx, cx));
      const j0 = this.cellY(Math.min(ay, by, cy)), j1 = this.cellY(Math.max(ay, by, cy));
      if (this.planarUnder(i0, i1, j0, j1, tolerance)) {
        // in this order: the vertices are made on first use
        const va = this.pointVertex(drape, coords, ia), vb = this.pointVertex(drape, coords, ib);
        out.pushTriangle(va, vb, this.pointVertex(drape, coords, ic));
        continue;
      }
      triangle[0] = ax; triangle[1] = ay; triangle[2] = bx; triangle[3] = by; triangle[4] = cx; triangle[5] = cy;
      this.drapePolygon(drape, triangle, 3, i0, i1, j0, j1, 1);
    }
  }

  /** The vertex of source point k of coords, made on first use. */
  private pointVertex(drape: Drape, coords: ArrayLike<number>, k: number): number {
    if (drape.points[k] < 0) drape.points[k] = this.pushVertex(drape, coords, k);
    return drape.points[k];
  }

  /**
   * A vertex of out on the ground at point k of coords (x, north per point) plus the lift, with the normal there.
   * Returns its index. heightAt, normalAt and encodeTerrainNormal written out (the same sums): the point is read here
   * and nothing is passed on, see GridMesh.allocVertex. This runs per vertex of every ground layer.
   */
  private pushVertex(drape: Drape, coords: ArrayLike<number>, k: number): number {
    const x = coords[k * 2], y = coords[k * 2 + 1];
    const cells = this.cells;
    const i = Math.min(Math.max(Math.floor((x - this.x0) / this.dx), 0), cells - 1);
    const j = Math.min(Math.max(Math.floor((y - this.y0) / this.dy), 0), cells - 1);
    const u = (x - (i >= cells ? this.x1 : this.x0 + (this.x1 - this.x0) * i / cells)) / this.dx;
    const v = (y - (j >= cells ? this.y1 : this.y0 + (this.y1 - this.y0) * j / cells)) / this.dy;
    const side = cells + 1, h = this.heights;
    const sw = j * side + i, se = sw + 1, nw = sw + side, ne = nw + 1;
    // the height on the triangle (sw, se, ne) or (sw, ne, nw), see heightAt
    const hsw = h[sw], hne = h[ne];
    let height: number;
    if (u >= v) {
      const hse = h[se];
      height = hsw + u * (hse - hsw) + v * (hne - hse);
    } else {
      const hnw = h[nw];
      height = hsw + v * (hnw - hsw) + u * (hne - hnw);
    }
    let normal = NO_NORMAL;
    if (drape.normals) {
      // see normalAt
      let a: number, wa: number, b: number, wb: number;
      const wsw = u >= v ? 1 - u : 1 - v;
      if (u >= v) { a = se; wa = u - v; b = ne; wb = v; } else { a = nw; wa = v - u; b = ne; wb = u; }
      const m = this.normals;
      const nx = m[sw * 3] * wsw + m[a * 3] * wa + m[b * 3] * wb;
      const ny = m[sw * 3 + 1] * wsw + m[a * 3 + 1] * wa + m[b * 3 + 1] * wb;
      const nz = m[sw * 3 + 2] * wsw + m[a * 3 + 2] * wa + m[b * 3 + 2] * wb;
      const l = hypot3(nx, ny, nz) || 1;
      // see encodeTerrainNormal
      const qx = Math.round((Math.min(Math.max(nx / l, -1), 1) * .5 + .5) * NORMAL_STEPS);
      const qz = Math.round((Math.min(Math.max(nz / l, -1), 1) * .5 + .5) * NORMAL_STEPS);
      normal = qx * 2048 + qz;
    }
    const out = drape.out;
    const o = out.allocVertex();
    const d = out.vertexData;
    d[o] = x;
    d[o + 1] = height + drape.lift;
    d[o + 2] = -y;
    d[o + 3] = normal;
    return o / GRID_VERTEX_FLOATS;
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
   * A convex piece (n points in poly, x, y each) within cells [i0, i1] x [j0, j1] (the cells of the split that made it,
   * narrowed to its box): whole where the ground under it is planar, else cut along the diagonal in 1 cell, else halved
   * along a grid line. depth: of the halving. Its 2 halves are cut into pieceAt(depth) one after the other (the first
   * is done when the second is cut), so no piece is allocated: as arrays of their own the pieces were a fifth of the
   * time of a tile.
   */
  private drapePolygon(drape: Drape, poly: Float64Array, n: number, i0: number, i1: number, j0: number, j1: number, depth: number): void {
    if (n < 3) return;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let k = 0; k < n * 2; k += 2) {
      minX = Math.min(minX, poly[k]); maxX = Math.max(maxX, poly[k]);
      minY = Math.min(minY, poly[k + 1]); maxY = Math.max(maxY, poly[k + 1]);
    }
    // a point on the grid line between 2 ranges is in both by its box: the range of the split decides
    i0 = Math.min(Math.max(this.cellX(minX), i0), i1); i1 = Math.max(Math.min(this.cellX(maxX), i1), i0);
    j0 = Math.min(Math.max(this.cellY(minY), j0), j1); j1 = Math.max(Math.min(this.cellY(maxY), j1), j0);
    if (this.planarUnder(i0, i1, j0, j1, drape.tolerance)) {
      this.emitPolygon(drape, poly, n);
      return;
    }
    const half = drape.scratch.pieceAt(depth);
    if (i0 === i1 && j0 === j1) {
      // in 1 cell, its 2 triangles not coplanar: cut along the diagonal. u - v >= 0 is the south east half, the other
      // half the negated function, so both get the same cut vertices
      const sx = 1 / this.dx, sy = 1 / this.dy;
      const ox = this.gridX(i0), oy = this.gridY(j0);
      const a = sx, b = -sy, c = -ox * sx + oy * sy;
      clipPlane[0] = a; clipPlane[1] = b; clipPlane[2] = c;
      let m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.emitPolygon(drape, poly, n);
      else this.emitPolygon(drape, half, m);
      clipPlane[0] = -a; clipPlane[1] = -b; clipPlane[2] = -c;
      m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.emitPolygon(drape, poly, n);
      else this.emitPolygon(drape, half, m);
      return;
    }
    // halve the cells along the longer side, on a grid line. Both sides of the cut get the same new vertices (clipHalf)
    if (i1 - i0 >= j1 - j0) {
      const mid = (i0 + i1 + 1) >> 1, x = this.gridX(mid);
      clipPlane[0] = -1; clipPlane[1] = 0; clipPlane[2] = x;
      let m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.drapePolygon(drape, poly, n, i0, mid - 1, j0, j1, depth + 1);
      else this.drapePolygon(drape, half, m, i0, mid - 1, j0, j1, depth + 1);
      // set again: the pieces of the first half cut too
      clipPlane[0] = 1; clipPlane[1] = 0; clipPlane[2] = -x;
      m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.drapePolygon(drape, poly, n, mid, i1, j0, j1, depth + 1);
      else this.drapePolygon(drape, half, m, mid, i1, j0, j1, depth + 1);
    } else {
      const mid = (j0 + j1 + 1) >> 1, y = this.gridY(mid);
      clipPlane[0] = 0; clipPlane[1] = -1; clipPlane[2] = y;
      let m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.drapePolygon(drape, poly, n, i0, i1, j0, mid - 1, depth + 1);
      else this.drapePolygon(drape, half, m, i0, i1, j0, mid - 1, depth + 1);
      clipPlane[0] = 0; clipPlane[1] = 1; clipPlane[2] = -y;
      m = clipHalf(poly, n, half);
      if (m === CLIP_ALL) this.drapePolygon(drape, poly, n, i0, i1, mid, j1, depth + 1);
      else this.drapePolygon(drape, half, m, i0, i1, mid, j1, depth + 1);
    }
  }

  /** A convex piece (n points) as a fan, its points lifted to the ground plus lift. Slivers without area are dropped. */
  private emitPolygon(drape: Drape, poly: Float64Array, n: number): void {
    if (n < 3) return;
    let area = 0;
    for (let k = 0; k < n; k++) {
      const q = k + 1 === n ? 0 : k + 1;
      area += poly[k * 2] * poly[q * 2 + 1] - poly[q * 2] * poly[k * 2 + 1];
    }
    if (Math.abs(area) < 1e-18) return;
    const { out, cut } = drape;
    const index = drape.scratch.fan;
    for (let k = 0; k < n; k++) {
      let v = cut.get(poly, k);
      if (v < 0) {
        v = this.pushVertex(drape, poly, k);
        cut.set(poly, k, v);
      }
      index[k] = v;
    }
    for (let k = 1; k + 1 < n; k++) out.pushTriangle(index[0], index[k], index[k + 1]);
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
type Drape = { out: GridMesh; lift: number; tolerance: number; normals: boolean; points: Int32Array; cut: PointMap; scratch: DrapeScratch };

/**
 * What drapeTriangles works in, made on its first call and kept for all the tiles after (a TerrainGrid is made per
 * tile, on the main thread too, where nothing is draped): the vertices of the cuts by position, the pieces of the
 * halving (1 per depth of drapePolygon), the vertices of the piece that is emitted and the vertices of the source points.
 */
class DrapeScratch {
  public readonly cut = new PointMap();
  public readonly fan = new Int32Array(PIECE_POINTS_MAX);
  private readonly pieces: Float64Array[] = [];
  private points = new Int32Array(8192);

  /** The buffer of the pieces that drapePolygon cuts at depth (the triangle itself at 0): x, y per point. */
  public pieceAt(depth: number): Float64Array {
    return this.pieces[depth] ??= new Float64Array(PIECE_POINTS_MAX * 2);
  }

  /** An array of at least count, its first count -1. */
  public pointsFor(count: number): Int32Array {
    if (this.points.length < count) this.points = new Int32Array(Math.max(count, this.points.length * 2));
    return this.points.fill(-1, 0, count);
  }
}

let drapeScratch: DrapeScratch | null = null;

/**
 * A map from a point (x, y, exactly) to a number, for the vertices of the cuts: by a hash of the bits of the 2 floats,
 * in a table with open addressing (the next slot when one is taken), all in typed arrays that are kept between the
 * layers of a tile. A string key (x + ',' + y) made the floats strings, a Map from the hash to chains in number[] was a
 * sixth of the time of a tile: most of the time of the draping went to them.
 */
class PointMap {
  private static readonly bits = new Float64Array(2);
  private static readonly words = new Uint32Array(PointMap.bits.buffer);
  // per slot the entry in it, -1 for none: a power of 2 slots, at most half of them taken
  private slots = new Int32Array(1 << 13).fill(-1);
  // per entry its point (x, y) and its value
  private keys = new Float64Array(2 << 12);
  private values = new Int32Array(1 << 12);
  private count = 0;

  /** The hash of point k of points (x, y each). The point by its index: see GridMesh.allocVertex. */
  private static hash(points: Float64Array, k: number): number {
    // + 0: -0 is 0, as x === y has it
    PointMap.bits[0] = points[k * 2] + 0;
    PointMap.bits[1] = points[k * 2 + 1] + 0;
    const w = PointMap.words;
    let h = Math.imul(w[0] ^ Math.imul(w[1], 0x9e3779b1), 0x85ebca6b);
    h = Math.imul(h ^ w[2] ^ Math.imul(w[3], 0xc2b2ae35), 0x27d4eb2f);
    return (h ^ (h >>> 15)) >>> 0;
  }

  /** Empty again, the arrays stay. */
  public clear(): void {
    if (this.count > 0) this.slots.fill(-1);
    this.count = 0;
  }

  /** The value of point k of points (x, y each), -1 when it has none. */
  public get(points: Float64Array, k: number): number {
    const x = points[k * 2], y = points[k * 2 + 1];
    const slots = this.slots, mask = slots.length - 1, keys = this.keys;
    for (let s = PointMap.hash(points, k) & mask; ; s = (s + 1) & mask) {
      const e = slots[s];
      if (e < 0) return -1;
      if (keys[e * 2] === x && keys[e * 2 + 1] === y) return this.values[e];
    }
  }

  /** For a point (k of points) that has no value yet. */
  public set(points: Float64Array, k: number, value: number): void {
    if (this.count === this.values.length) this.grow();
    const e = this.count++;
    this.keys[e * 2] = points[k * 2];
    this.keys[e * 2 + 1] = points[k * 2 + 1];
    this.values[e] = value;
    this.insert(e);
  }

  private insert(e: number): void {
    const slots = this.slots, mask = slots.length - 1;
    let s = PointMap.hash(this.keys, e) & mask;
    while (slots[s] >= 0) s = (s + 1) & mask;
    slots[s] = e;
  }

  /** Twice the entries, in twice the slots. */
  private grow(): void {
    const size = this.values.length * 2;
    const keys = new Float64Array(size * 2), values = new Int32Array(size);
    keys.set(this.keys); values.set(this.values);
    this.keys = keys; this.values = values;
    this.slots = new Int32Array(size * 2).fill(-1);
    for (let e = 0; e < this.count; e++) this.insert(e);
  }
}

// The most points of a piece of drapePolygon: a triangle, and 1 more per cut. The cuts halve the cells of the tile along
// x and along y (8 times each for the 256 cells of an elevation tile, see terrainGridCells) and end with the diagonal
const PIECE_POINTS_MAX = 32;
// clipHalf: all of the polygon is in the half, nothing was written
const CLIP_ALL = -1;
// The half plane clipHalf cuts by: a, b and c of a * x + b * y + c >= 0. Set before each call instead of passed: see
// GridMesh.allocVertex, as arguments they were 4.5 MB of garbage per tile
const clipPlane = new Float64Array(3);

/**
 * The part of the convex polygon poly (n points, x, y each) where a * x + b * y + c >= 0 (Sutherland-Hodgman, a, b and c
 * from clipPlane), into out: returns its number of points, or CLIP_ALL when that is all of poly (out is not written then). A cut edge gets the same
 * new vertex in the part of the negated function: the factor t is the same (negating a float is exact), so the 2 sides
 * of a cut meet without a crack. A cut along a grid line (a or b 0, the other +-1) puts the new vertex on the line
 * exactly.
 */
function clipHalf(poly: Float64Array, n: number, out: Float64Array): number {
  const a = clipPlane[0], b = clipPlane[1], c = clipPlane[2];
  let inside = 0;
  for (let k = 0; k < n; k++) if (a * poly[k * 2] + b * poly[k * 2 + 1] + c >= 0) inside++;
  if (inside === n) return CLIP_ALL;
  if (inside === 0) return 0;
  let m = 0;
  for (let k = 0; k < n; k++) {
    const px = poly[k * 2], py = poly[k * 2 + 1];
    const q = k + 1 === n ? 0 : k + 1;
    const qx = poly[q * 2], qy = poly[q * 2 + 1];
    const fp = a * px + b * py + c, fq = a * qx + b * qy + c;
    if (fp >= 0) {
      out[m] = px; out[m + 1] = py;
      m += 2;
    }
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out[m] = b === 0 ? -c / a : px + t * (qx - px);
      out[m + 1] = a === 0 ? -c / b : py + t * (qy - py);
      m += 2;
    }
  }
  return m / 2;
}
