import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import AABox from 'webGPU-framework/renderer/math/AABox';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import { LatLngBounds } from './OSMBuilder';
import MapProjection from './MapProjection';
import type { BuildingInfo } from './cityGlTFCore';
import { NO_FEATURE_ID, VERTEX_FLOATS } from './cityGlTFCore';

/**
 * Parses a CityJSON tile from https://3dbag.nl/ (EPSG:7415 / RD New + NAP) and
 * turns its building geometry into renderable 3D meshes that overlay the OSM
 * map. Only the highest-quality LOD (2.2) is used when present.
 *
 * CityJSON layout used here:
 *  - transform.scale / transform.translate : int vertices -> real RD coords
 *  - CityObjects[id].geometry[] : { type:'Solid', lod:'2.2', boundaries, semantics }
 *  - Solid.boundaries = [ shell ][ surface ][ ring ][ vertexIndex ]
 *  - semantics.surfaces[] + semantics.values[shell][surface] : surface type
 *
 * Geometry is projected into the same normalized space as OSMBuilder so the
 * two line up: RD (x,y) -> WGS84 lat/lng -> normalized map plane, NAP z -> up.
 */

const METERS_PER_DEGREE_LAT = 111_320;

export type CityJSONVertex = [number, number, number];

export type CityJSONGeometry = {
  type: string;
  lod?: string | number;
  boundaries: any;
  semantics?: { surfaces: { type: string }[]; values: any };
};

export type CityObject = {
  type: string;
  geometry?: CityJSONGeometry[];
  parents?: string[];
  children?: string[];
  attributes?: Record<string, unknown>;
};

export type CityJSON = {
  type: string;
  version: string;
  transform: { scale: CityJSONVertex; translate: CityJSONVertex };
  vertices: CityJSONVertex[];
  CityObjects: Record<string, CityObject>;
  metadata?: { geographicalExtent?: number[]; title?: string; referenceSystem?: string };
};

// One renderable surface category (matches the CityJSON semantic types we care
// about). Each becomes a toggleable, individually-colored layer.
export const CITY_SURFACE_KEYS = ['Building'] as const;
export type CitySurfaceKey = (typeof CITY_SURFACE_KEYS)[number];

export type CityBuildResult = {
  layers: Record<CitySurfaceKey, Mesh[]>;
  triangleCount: Record<CitySurfaceKey, number>;
  surfaceCount: Record<CitySurfaceKey, number>;
  objectCount: number;
  lodUsed: string;
  /**
   * Pre-computed building-mesh bounds (engine space), when the builder already
   * knows them (the glTF worker path computes the AABB off the main thread).
   * Absent for builders that don't, so callers fall back to scanning the mesh.
   */
  bounds?: AABox;
  /**
   * Per-building attributes for the click/info panel, indexed by the tile-local
   * `_FEATURE_ID_0` carried on the building mesh's `featureId` vertex attribute.
   * Only the 3DBAG glTF builder populates this; empty otherwise.
   */
  buildings?: BuildingInfo[];
};

type Accumulator = { positions: FloatBuffer; surfaces: number };
type Projector = {
  ensure: (idx: number) => void;
  px: Float32Array;
  py: Float32Array;
  pz: Float32Array;
};

/** Approximate RD New (EPSG:28992) -> WGS84 lat/lng. Accurate to a few cm. */
function rdToWGS84(x: number, y: number): { lat: number; lon: number } {
  const X0 = 155000, Y0 = 463000;
  const PHI0 = 52.15517440, LAM0 = 5.38720621;
  const dX = (x - X0) * 1e-5;
  const dY = (y - Y0) * 1e-5;

  const Kp = [0, 2, 0, 2, 0, 2, 1, 4, 2, 4, 1];
  const Kq = [1, 0, 2, 1, 3, 2, 0, 0, 3, 1, 1];
  const Kpq = [3235.65389, -32.58297, -0.24750, -0.84978, -0.06550, -0.01709, -0.00738, 0.00530, -0.00039, 0.00033, -0.00012];
  let phi = 0;
  for (let i = 0; i < Kpq.length; i++) phi += Kpq[i] * Math.pow(dX, Kp[i]) * Math.pow(dY, Kq[i]);
  const lat = PHI0 + phi / 3600;

  const Lp = [1, 1, 1, 3, 1, 3, 0, 3, 1, 0, 2, 5];
  const Lq = [0, 1, 2, 0, 3, 1, 1, 2, 4, 2, 0, 0];
  const Lpq = [5260.52916, 105.94684, 2.45656, -0.81885, 0.05594, -0.05607, 0.01199, -0.00256, 0.00128, 0.00022, -0.00022, 0.00026];
  let lam = 0;
  for (let i = 0; i < Lpq.length; i++) lam += Lpq[i] * Math.pow(dX, Lp[i]) * Math.pow(dY, Lq[i]);
  const lon = LAM0 + lam / 3600;

  return { lat, lon };
}

export default class CityJSONBuilder {
  private renderer: Renderer;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
  }

  /** Parse + build, projecting into the same normalized frame as the OSM map. */
  public build(json: CityJSON, bounds: LatLngBounds): CityBuildResult {

    console.log('CityJSON', json);

    const t0 = performance.now();
    // LogGL.log('[3DBAG] === CityJSON tile load ===');
    // LogGL.log(`[3DBAG] title: ${json.metadata?.title ?? '?'} version: ${json.version} crs: ${json.metadata?.referenceSystem ?? '?'}`);

    const { scale, translate } = json.transform;
    const verts = json.vertices;
    // LogGL.log(`[3DBAG] vertices: ${verts.length}, CityObjects: ${Object.keys(json.CityObjects).length}`);

    const mapProj = new MapProjection();
    mapProj.setBounds(bounds);
    const { latMid, lngMid, cosLat, scale: projScale } = mapProj;
    const metersToNorm = mapProj.metersToNorm;

    const sx = scale[0], sy = scale[1], sz = scale[2];
    const tx = translate[0], ty = translate[1], tz = translate[2];

    // Cache projected (x, y, zUp) per vertex index to avoid repeat work. The
    // `ensure` closure lazily projects a vertex once; consumers then read the
    // result straight out of px/py/pz by index (no per-call tuple allocation).
    const n = verts.length;
    const px = new Float32Array(n);
    const py = new Float32Array(n);
    const pz = new Float32Array(n);
    const projected = new Uint8Array(n);
    const ensure = (idx: number): void => {
      if (projected[idx]) return;
      const v = verts[idx];
      const { lat, lon } = rdToWGS84(v[0] * sx + tx, v[1] * sy + ty);
      px[idx] = (lon - lngMid) * cosLat * projScale;
      py[idx] = (lat - latMid) * projScale;
      pz[idx] = (v[2] * sz + tz) * metersToNorm;
      projected[idx] = 1;
    };
    const proj: Projector = { ensure, px, py, pz };

    const acc: Record<CitySurfaceKey, Accumulator> = {} as Record<CitySurfaceKey, Accumulator>;
    for (let k = 0; k < CITY_SURFACE_KEYS.length; k++) {
      acc[CITY_SURFACE_KEYS[k]] = { positions: new FloatBuffer(), surfaces: 0 };
    }

    let objectCount = 0;
    let lodUsed = '2.2';

    const ids = Object.keys(json.CityObjects);
    for (let i = 0; i < ids.length; i++) {
      const obj = json.CityObjects[ids[i]];
      if (!obj.geometry || obj.geometry.length === 0) continue;

      const geom = this.pickGeometry(obj.geometry);
      if (!geom) continue;
      lodUsed = String(geom.lod);
      objectCount++;

      // Drop each building to y=0 so it sits flush on the flat terrain plane
      // (the terrain carries no real elevation). Record where this object's
      // triangles start, build it, then subtract its lowest vertex in place.
      // See Option A.
      const startLen = startLenScratch;
      for (let k = 0; k < CITY_SURFACE_KEYS.length; k++) startLen[k] = acc[CITY_SURFACE_KEYS[k]].positions.length;
      const objMinY = this.addSolid(geom, acc, proj);
      if (objMinY !== Infinity && objMinY !== 0) {
        for (let k = 0; k < CITY_SURFACE_KEYS.length; k++) {
          const buf = acc[CITY_SURFACE_KEYS[k]].positions;
          const pd = buf.data;
          for (let j = startLen[k] + 1; j < buf.length; j += 3) pd[j] -= objMinY;
        }
      }
    }

    const layers = {} as Record<CitySurfaceKey, Mesh[]>;
    const triangleCount = {} as Record<CitySurfaceKey, number>;
    const surfaceCount = {} as Record<CitySurfaceKey, number>;
    for (let k = 0; k < CITY_SURFACE_KEYS.length; k++) {
      const key = CITY_SURFACE_KEYS[k];
      layers[key] = this.makeMeshes(acc[key]);
      triangleCount[key] = acc[key].positions.length / 9;
      surfaceCount[key] = acc[key].surfaces;
    }

    const result: CityBuildResult = { layers, triangleCount, surfaceCount, objectCount, lodUsed };

    // LogGL.log('[3DBAG] === geometry summary ===');
    // LogGL.log(`[3DBAG] LOD used: ${lodUsed}, buildings/parts meshed: ${objectCount}`);
    let totalTris = 0;
    for (const key of CITY_SURFACE_KEYS) {
      totalTris += triangleCount[key];
      // LogGL.log(`[3DBAG]   ${key.padEnd(14)} : ${surfaceCount[key]} surfaces, ${triangleCount[key]} tris, ${layers[key].length} mesh(es)`);
    }

    const t1 = performance.now();
    // LogGL.log(`[3DBAG] total triangles: ${totalTris}`);
    // LogGL.log(`[3DBAG] built geometry in ${((t1 - t0) / 1000).toFixed(2)}s`);

    return result;
  }

  /** Highest available LOD: prefer 2.2, else the numerically largest. */
  private pickGeometry(geoms: CityJSONGeometry[]): CityJSONGeometry | null {
    let best: CityJSONGeometry | null = null;
    let bestLod = -Infinity;
    for (const g of geoms) {
      if (g.type !== 'Solid' && g.type !== 'MultiSurface') continue;
      const lod = parseFloat(String(g.lod));
      if (String(g.lod) === '2.2') return g;
      if (!isNaN(lod) && lod > bestLod) {
        bestLod = lod;
        best = g;
      }
    }
    return best;
  }

  /** Walk a Solid/MultiSurface and accumulate triangles per semantic layer.
   *  Returns the lowest engine-space Y written (used to drop the building to y=0). */
  private addSolid(
    geom: CityJSONGeometry,
    acc: Record<CitySurfaceKey, Accumulator>,
    proj: Projector,
  ): number {
    const semSurfaces = geom.semantics?.surfaces ?? [];
    let minY = Infinity;

    // Solid: boundaries[shell][surface][ring]. MultiSurface: boundaries[surface][ring].
    const shells: any[] = geom.type === 'Solid' ? geom.boundaries : [geom.boundaries];
    const semShells: any[] = geom.type === 'Solid' ? geom.semantics?.values ?? [] : [geom.semantics?.values ?? []];

    for (let s = 0; s < shells.length; s++) {
      const surfaces = shells[s];
      const semValues = semShells[s] ?? [];
      for (let f = 0; f < surfaces.length; f++) {
        const rings: number[][] = surfaces[f];
        if (!rings || rings.length === 0) continue;

        const semIdx = semValues[f];
        const type = (semIdx != null && semSurfaces[semIdx]?.type) || 'WallSurface';
        const key = this.surfaceKey(type);
        if (!key) continue;

        const surfMinY = this.addSurface(rings, acc[key], proj);
        if (surfMinY < minY) minY = surfMinY;
      }
    }
    return minY;
  }

  private surfaceKey(type: string): CitySurfaceKey | null {
    if (type === 'GroundSurface') return null;
    return 'Building';
  }

  /** Triangulate a (possibly non-convex) 3D polygon surface with one outer ring.
   *  Returns the lowest engine-space Y of the ring (Infinity if degenerate). */
  private addSurface(rings: number[][], a: Accumulator, proj: Projector): number {
    const outer = rings[0];
    const m = outer ? outer.length : 0;
    if (m < 3) return Infinity;

    const { ensure, px, py, pz } = proj;

    // Project the ring once into flat scratch arrays (grown on demand) so the
    // rest of the routine reads contiguous numbers instead of chasing indices.
    if (m > rx.length) {
      rx = new Float32Array(m);
      ry = new Float32Array(m);
      rz = new Float32Array(m);
      flat = new Float32Array(m * 2);
    }
    let minZ = Infinity;
    for (let i = 0; i < m; i++) {
      const idx = outer[i];
      ensure(idx);
      rx[i] = px[idx];
      ry[i] = py[idx];
      rz[i] = pz[idx];
      if (rz[i] < minZ) minZ = rz[i];
    }

    // Newell's method for the surface normal (inlined, reading scratch).
    let nx = 0, ny = 0, nz = 0;
    for (let i = 0; i < m; i++) {
      const j = i + 1 === m ? 0 : i + 1;
      nx += (ry[i] - ry[j]) * (rz[i] + rz[j]);
      ny += (rz[i] - rz[j]) * (rx[i] + rx[j]);
      nz += (rx[i] - rx[j]) * (ry[i] + ry[j]);
    }
    const invLen = 1 / (Math.hypot(nx, ny, nz) || 1);
    nx *= invLen; ny *= invLen; nz *= invLen;

    // Project to the dominant plane for 2D triangulation.
    const ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
    let su: Float32Array, sv: Float32Array;
    if (az >= ax && az >= ay) { su = rx; sv = ry; }
    else if (ax >= ay) { su = ry; sv = rz; }
    else { su = rx; sv = rz; }
    for (let i = 0; i < m; i++) {
      flat[i * 2] = su[i];
      flat[i * 2 + 1] = sv[i];
    }

    let tris = triangulate(flat, m);
    if (tris.length === 0) {
      tris = [];
      for (let i = 1; i < m - 1; i++) tris.push(0, i, i + 1);
    }

    // Reserve once, then write positions straight into the typed buffer.
    // Normals are no longer uploaded — the fragment shader derives a flat
    // face normal from screen-space derivatives of the world position.
    const posBuf = a.positions;
    posBuf.reserve(tris.length * 3);
    const pd = posBuf.data;
    let pi = posBuf.length;

    for (let t = 0; t < tris.length; t += 3) {
      const i0 = tris[t];
      let i1 = tris[t + 1], i2 = tris[t + 2];
      // The 2D triangulation loses the sign of the dropped axis, so the output
      // winding is decoupled from the surface normal — about half the faces end
      // up reversed and get back-face culled. Re-wind each triangle so its
      // vertices run CCW around the outward normal (matching the engine's
      // front-face convention), keeping the winding consistent. The shader's
      // derivative normal relies on this consistent front-facing winding.
      const e1x = rx[i1] - rx[i0], e1y = ry[i1] - ry[i0], e1z = rz[i1] - rz[i0];
      const e2x = rx[i2] - rx[i0], e2y = ry[i2] - ry[i0], e2z = rz[i2] - rz[i0];
      const cx = e1y * e2z - e1z * e2y;
      const cy = e1z * e2x - e1x * e2z;
      const cz = e1x * e2y - e1y * e2x;
      if (cx * nx + cy * ny + cz * nz < 0) {
        const tmp = i1; i1 = i2; i2 = tmp;
      }
      // Positions in engine-space (x, z, -y).
      pd[pi] = rx[i0]; pd[pi + 1] = rz[i0]; pd[pi + 2] = -ry[i0];
      pd[pi + 3] = rx[i1]; pd[pi + 4] = rz[i1]; pd[pi + 5] = -ry[i1];
      pd[pi + 6] = rx[i2]; pd[pi + 7] = rz[i2]; pd[pi + 8] = -ry[i2];
      pi += 9;
    }
    posBuf.length = pi;
    a.surfaces++;
    return minZ;
  }

  private makeMeshes(a: Accumulator): Mesh[] {
    if (a.positions.length === 0) return [];
    // Expand the xyz soup into the unified stride-4 layout (xyz + `.w` building
    // id) every map mesh shares, so the WFS buildings draw with the same vertex
    // layout as the glTF buildings/terrain/ground. WFS tiles carry no per-vertex
    // feature id, so every vertex is tagged NO_FEATURE_ID (never pickable).
    const xyz = a.positions.toArray();
    const vertexCount = xyz.length / 3;
    const verts = new Float32Array(vertexCount * VERTEX_FLOATS);
    for (let v = 0; v < vertexCount; v++) {
      verts[v * 4]     = xyz[v * 3];
      verts[v * 4 + 1] = xyz[v * 3 + 1];
      verts[v * 4 + 2] = xyz[v * 3 + 2];
      verts[v * 4 + 3] = NO_FEATURE_ID;
    }
    const mesh = new Mesh(this.renderer);
    mesh.setAttribute('position', VERTEX_FLOATS, verts);
    return [mesh];
  }
}

// --- helpers ----------------------------------------------------------------

/**
 * A Float32Array that grows by doubling. Lets us write mesh data straight into
 * a typed buffer (instead of pushing onto a number[] and converting at the end).
 * `toArray()` returns an exact-length standalone copy: a single-attribute mesh
 * uploads its backing ArrayBuffer wholesale, so the buffer must be trimmed to
 * exactly the written region (a subarray view would still carry the oversized
 * backing store). The copy is a fast memcpy of just the used floats.
 */
class FloatBuffer {
  public data: Float32Array;
  public length = 0;

  constructor(initial = 4096) {
    this.data = new Float32Array(initial);
  }

  /** Ensure room for `extra` more floats, growing (and copying) if needed. */
  public reserve(extra: number): void {
    const need = this.length + extra;
    if (need <= this.data.length) return;
    let cap = this.data.length || 1;
    while (cap < need) cap *= 2;
    const next = new Float32Array(cap);
    next.set(this.data);
    this.data = next;
  }

  public toArray(): Float32Array {
    return this.data.slice(0, this.length);
  }
}

// Per-object scratch holding each layer's buffer length before a building is
// built, so its triangles can be dropped to y=0 afterwards (one slot per key).
const startLenScratch = new Array<number>(CITY_SURFACE_KEYS.length).fill(0);

// Per-surface scratch reused across addSurface calls to avoid allocations.
let rx = new Float32Array(64);
let ry = new Float32Array(64);
let rz = new Float32Array(64);
let flat = new Float32Array(128);

/** Ear-clipping for a simple polygon (flat [x,y,...], `n` vertices); returns index triplets. */
function triangulate(poly: Float32Array, n: number): number[] {
  const indices: number[] = [];
  if (n < 3) return indices;
  if (n === 3) return [0, 1, 2];

  const ccw = signedArea(poly, n) > 0;
  const v: number[] = [];
  for (let i = 0; i < n; i++) v.push(ccw ? i : n - 1 - i);

  let guard = 0;
  const maxGuard = n * n + 10;
  while (v.length > 3 && guard++ < maxGuard) {
    let earFound = false;
    for (let i = 0; i < v.length; i++) {
      const i0 = v[(i + v.length - 1) % v.length];
      const i1 = v[i];
      const i2 = v[(i + 1) % v.length];
      const ax = poly[i0 * 2], ay = poly[i0 * 2 + 1];
      const bx = poly[i1 * 2], by = poly[i1 * 2 + 1];
      const cx = poly[i2 * 2], cy = poly[i2 * 2 + 1];
      const cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (cross <= 0) continue;
      let contains = false;
      for (let j = 0; j < v.length; j++) {
        const vj = v[j];
        if (vj === i0 || vj === i1 || vj === i2) continue;
        if (pointInTriangle(poly[vj * 2], poly[vj * 2 + 1], ax, ay, bx, by, cx, cy)) { contains = true; break; }
      }
      if (contains) continue;
      indices.push(i0, i1, i2);
      v.splice(i, 1);
      earFound = true;
      break;
    }
    if (!earFound) break;
  }
  if (v.length === 3) indices.push(v[0], v[1], v[2]);
  return indices;
}

function signedArea(poly: Float32Array, n: number): number {
  let area = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    area += poly[i * 2] * poly[j * 2 + 1] - poly[j * 2] * poly[i * 2 + 1];
  }
  return area * 0.5;
}

function pointInTriangle(
  px: number, py: number,
  ax: number, ay: number, bx: number, by: number, cx: number, cy: number,
): boolean {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
  const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
  return !(hasNeg && hasPos);
}
