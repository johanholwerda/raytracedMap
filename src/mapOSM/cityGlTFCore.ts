import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type { LatLngBounds } from './OSMBuilder';
import { MeshoptDecoder } from './vendor/meshopt_decoder.module';
import MapProjection from './MapProjection';

/**
 * Pure (renderer-free) geometry core shared by {@link CityGlTFBuilder} and the
 * city glTF Web Worker. Turns a 3DBAG **3D Tiles** binary glTF (.glb) tile into
 * a de-indexed position soup projected into the normalized map frame, without
 * touching the GPU — so it can run on the main thread *or* inside a worker.
 *
 * {@link CityGlTFBuilder} wraps the {@link CityPositionsResult} into an engine
 * `Mesh` (which needs a `Renderer`); the worker just transfers the position
 * buffer back. Keep this module free of any `Mesh`/`Renderer`/`CityJSONBuilder`
 * imports so the worker bundle stays small and WebGPU-free.
 *
 * The .glb uses two *required* glTF extensions:
 *  - EXT_meshopt_compression : bufferViews are meshopt-compressed (decoded by
 *    the vendored meshoptimizer WASM decoder).
 *  - KHR_mesh_quantization   : POSITION is stored as normalized int16; the node
 *    matrix carries the dequantization scale plus the ECEF placement.
 *
 * Coordinate chain (validated against a real lod22 tile):
 *   int16 -> /32767 -> node matrix -> glTF (Y-up) coords
 *   glTF (x,y,z) -> ECEF (x,-z,y)            (3D Tiles Y-up -> Z-up rotation)
 *   ECEF -> WGS84 lat/lon/h
 *   lat/lon/h -> normalized map plane (matching CityJSONBuilder)
 */

/**
 * Per-building attributes pulled from the tile's EXT_structural_metadata
 * property table, indexed by the tile-local `_FEATURE_ID_0`. This is the data
 * surfaced in the click panel. `identificatie` is the BAG "pand" id (there is
 * no street address in the tile — that needs a separate BAG lookup).
 */
export type BuildingInfo = {
  /** BAG pand id, e.g. "NL.IMBAG.Pand.0363100012197777" ('' when absent). */
  id: string;
  /** Original construction year (oorspronkelijkbouwjaar), or 0 when absent. */
  year: number;
  /** Footprint / ground area in m² (b3_opp_grond), or 0 when absent. */
  groundArea: number;
  /** BAG status, e.g. "Pand in gebruik" ('' when absent). */
  status: string;
  /** Roof shape (b3_dak_type), e.g. "horizontal" / "slanted" ('' when absent). */
  roofType: string;
  /** Building height in meters, when the source provides it (world map). */
  height?: number;
};

/** Geometry-only result of building one city tile (vertices + metadata). */
export type CityPositionsResult = {
  /**
   * De-indexed triangle soup, interleaved as one `vec4` per vertex:
   * `(x, up, -y, featureId)`. The xyz is the engine-space position; `.w` is the
   * tile-local building id (`_FEATURE_ID_0`), or {@link NO_FEATURE_ID} when the
   * tile carries none. Ready to upload as a single stride-4 vertex attribute
   * (no main-thread interleave), and consumed by the osm shader (under
   * `#ifdef FEATURE_IDS`) to build the GPU picking id / highlight a click.
   */
  vertices: Float32Array;
  /** Per-building attributes, indexed by the feature id packed into each vertex's `.w`. */
  buildings: BuildingInfo[];
  triangleCount: number;
  surfaceCount: number;
  objectCount: number;
  lodUsed: string;
  /**
   * Axis-aligned bounds of {@link vertices} in engine space, computed here (off
   * the main thread) so the main thread doesn't have to re-scan every vertex
   * (and allocate a Vector3 per vertex) just to get a tile's bounding box.
   * Both `[0,0,0]` when the tile is empty.
   */
  min: [number, number, number];
  max: [number, number, number];
};

/**
 * Reserved `.w` value for a vertex with no building id (a tile without a
 * `_FEATURE_ID_0` attribute). Negative so the picking shader maps it to the
 * empty index-map value and never resolves to a building. Shared with terrain
 * and the ground quad, which also ride the unified `vec4` position layout.
 */
export const NO_FEATURE_ID = -1;
/** Floats per vertex in {@link CityPositionsResult.vertices}: x, up, -y, featureId. */
export const VERTEX_FLOATS = 4;

/**
 * The 3D Basisvoorziening "terreinen" tiles split their geometry into a handful
 * of primitives tagged by a coarse `3df_class` (Road / Water / Forest /
 * Terrain), and carry per-surface BGT metadata via EXT_structural_metadata +
 * a `_FEATURE_ID_0` vertex attribute:
 *   - bgt_functie          : road purpose (rijbaan, fietspad, voetpad, inrit,
 *                            parkeervlak, verkeerseiland, berm, …)
 *   - bgt_fysiekvoorkomen  : physical surface (grasland, loofbos, struiken,
 *                            bouwland, open/gesloten verharding, zand, …)
 *   - bgt_type             : water kind (waterloop, watervlakte, greppel, …)
 *
 * We combine these into a fine-grained, individually-colored category per
 * surface so e.g. bicycle lanes, parking, farmland and ditches can each be
 * drawn separately. {@link classifyTerrain} maps a surface to one category;
 * unmapped combinations fall through to 'other' (and are logged once).
 */
export type TerrainCategory =
  | 'water' | 'ditch'
  | 'forest' | 'shrubs' | 'grass' | 'greenAmenity' | 'farmland'
  | 'sand' | 'bank' | 'unpaved' | 'yard' | 'verge' | 'pavedTerrain'
  | 'trafficIsland' | 'parking' | 'driveway' | 'road' | 'footpath' | 'bicycleLane'
  | 'other';

export type TerrainCategoryDef = { key: TerrainCategory; label: string; color: [number, number, number] };

// Draw order: water + natural ground first, then bare/paved terrain, then
// roads, with paths/lanes last so they sit on top where surfaces touch.
// Colors are sRGB-ish 0..1. Labels carry the Dutch BGT term for clarity.
export const TERRAIN_CATEGORY_DEFS: TerrainCategoryDef[] = [
  { key: 'water',         label: 'water',                   color: [0.55, 0.75, 0.92] },
  { key: 'ditch',         label: 'greppel',         color: [0.45, 0.58, 0.66] },
  { key: 'forest',        label: 'bos',            color: [0.18, 0.42, 0.20] },
  { key: 'shrubs',        label: 'struiken',       color: [0.42, 0.58, 0.28] },
  { key: 'grass',         label: 'grasland',        color: [0.56, 0.74, 0.42] },
  { key: 'greenAmenity',  label: 'groenvoorziening',color: [0.64, 0.80, 0.52] },
  { key: 'farmland',      label: 'bouwland',     color: [0.80, 0.71, 0.44] },
  { key: 'sand',          label: 'zand',             color: [0.92, 0.86, 0.66] },
  { key: 'bank',          label: 'oever',            color: [0.56, 0.62, 0.46] },
  { key: 'unpaved',       label: 'onverhard',     color: [0.66, 0.54, 0.40] },
  { key: 'yard',          label: 'erf',              color: [0.82, 0.78, 0.68] },
  { key: 'verge',         label: 'berm',            color: [0.62, 0.66, 0.40] },
  { key: 'pavedTerrain',  label: 'verhard',           color: [0.72, 0.70, 0.68] },
  { key: 'trafficIsland', label: 'verkeerseiland',          color: [0.58, 0.56, 0.58] },
  { key: 'parking',       label: 'parkeervlak',   color: [0.40, 0.46, 0.54] },
  { key: 'driveway',      label: 'inrit',        color: [0.50, 0.47, 0.45] },
  { key: 'road',          label: 'rijbaan',          color: [0.24, 0.24, 0.27] },
  { key: 'footpath',      label: 'voetpad',      color: [0.80, 0.74, 0.64] },
  { key: 'bicycleLane',   label: 'fietspad', color: [1, 0.5, 0.5] },
  { key: 'other',         label: 'overig',                   color: [0.5, 0.5, 0.5] },
];

/**
 * Map a BGT surface to a colored terrain category, combining the coarse
 * `3df_class` with the most specific applicable attribute (road function for
 * roads, physical appearance for green/terrain, water type for water).
 */
function classifyTerrain(class3df: string, functie: string, fysiek: string, type: string): TerrainCategory {
  const fn = functie.toLowerCase();
  const fy = fysiek.toLowerCase();
  const ty = type.toLowerCase();
  switch (class3df) {
    case 'Water':
      if (ty.includes('greppel') || ty.includes('droge')) return 'ditch';
      return 'water';
    case 'Road':
      if (fn.includes('fiets')) return 'bicycleLane'; // fietspad / fiets/bromfietspad
      if (fn.includes('voetpad') || fn.includes('voetganger')) return 'footpath';
      if (fn.includes('parkeer')) return 'parking';
      if (fn.includes('inrit')) return 'driveway';
      if (fn.includes('verkeerseiland')) return 'trafficIsland';
      return 'road'; // rijbaan (lokale/regionale weg), …
    case 'Forest':
      if (fy.includes('grasland')) return 'grass';
      if (fy.includes('bouwland')) return 'farmland';
      if (fy.includes('groenvoorziening')) return 'greenAmenity';
      if (fy.includes('struik')) return 'shrubs';
      return 'forest'; // loofbos, naaldbos, houtwal, boomteelt, fruitteelt, …
    case 'Terrain':
    default:
      if (fn.includes('berm')) return 'verge';
      if (fn.includes('verkeerseiland')) return 'trafficIsland';
      if (ty.includes('oever') || ty.includes('slootkant')) return 'bank';
      if (fy.includes('erf')) return 'yard';
      if (fy.includes('zand')) return 'sand';
      if (fy.includes('onverhard')) return 'unpaved';
      if (fy.includes('verhard')) return 'pavedTerrain'; // open/gesloten/half verharding
      if (fy.includes('grasland') || fy.includes('groenvoorziening')) return 'grass';
      return 'other';
  }
}

// Log each distinct BGT surface combination the first time it is seen, together
// with the terrain layer it was assigned to — a running report of what the data
// actually contains (and a flag for anything landing in 'other').
const seenTerrainCombos = new Set<string>();
function logTerrainCombo(class3df: string, functie: string, fysiek: string, type: string, cat: TerrainCategory): void {
  const combo = `${class3df}|${functie}|${fysiek}|${type}`;
  if (seenTerrainCombos.has(combo)) return;
  seenTerrainCombos.add(combo);
  const parts = [
    `3df=${class3df || '-'}`,
    `functie=${functie || '-'}`,
    `fysiek=${fysiek || '-'}`,
    `type=${type || '-'}`,
  ].join(' ');
  LogGL.log(`[terrain] layer "${cat}"  <-  ${parts}`);
}

// --- TEMP building-tile inspector --------------------------------------------
// One-shot probe that logs what per-building data a 3DBAG glTF (.glb) tile
// carries: the glTF extensions in use, the per-primitive vertex attributes
// (notably _FEATURE_ID_0, which tags each vertex with its building), and the
// EXT_structural_metadata property table (the 3DBAG attributes per building,
// e.g. `identificatie`, `b3_h_dak_50p`, `oorspronkelijkbouwjaar`, status).
//
// 3DBAG reference: https://docs.3dbag.nl/en/delivery/3dtiles/ and the attribute
// list at https://docs.3dbag.nl/en/schema/attributes/ .
// Remove once the click-to-select / index-map feature is wired up.
let loggedBuildingTile = false;
function logBuildingTileContentsOnce(json: GlTF, bin: Uint8Array): void {
  if (loggedBuildingTile) return;
  loggedBuildingTile = true;

  // NOTE: this runs inside the map worker, where LogGL.ENABLED is never set
  // (it's only enabled on the main thread), so LogGL.log is a no-op here.
  // Use console.log directly so the probe is actually visible.
  const j = json as any;
  console.log('[building-tile] ===== 3DBAG glTF tile contents (TEMP probe) =====');
  console.log('[building-tile] extensionsUsed:', j.extensionsUsed ?? []);
  console.log('[building-tile] extensionsRequired:', j.extensionsRequired ?? []);

  // Per-primitive vertex attributes + the _FEATURE_ID_0 range (how many
  // distinct buildings are packed into this primitive).
  const meshes = json.meshes ?? [];
  for (let mi = 0; mi < meshes.length; mi++) {
    const prims = meshes[mi].primitives ?? [];
    for (let pi = 0; pi < prims.length; pi++) {
      const prim = prims[pi];
      const attrs = Object.keys(prim.attributes ?? {});
      console.log(`[building-tile] mesh ${mi} prim ${pi} attributes:`, attrs,
        'featureMetadataExt:', prim.extensions?.EXT_mesh_features ?? prim.extensions?.EXT_instance_features ?? null);
      const fidIdx = prim.attributes?._FEATURE_ID_0;
      if (fidIdx != null) {
        const acc = json.accessors[fidIdx];
        const view = directScalarView(json, bin, acc);
        let min = Infinity, max = -Infinity;
        if (view) {
          for (let i = 0; i < acc.count; i++) { const f = view[i] as number; if (f < min) min = f; if (f > max) max = f; }
        } else {
          const read = makeScalarReader(json, bin, acc);
          for (let i = 0; i < acc.count; i++) { const f = read(i); if (f < min) min = f; if (f > max) max = f; }
        }
        console.log(`[building-tile]   _FEATURE_ID_0 -> ${acc.count} verts, feature ids ${min}..${max} (=> ~${max - min + 1} buildings)`);
      }
    }
  }

  // EXT_structural_metadata: schema (property names/types) + property table
  // (the actual per-building values). This is the data you'd surface on click.
  const meta = j.extensions?.EXT_structural_metadata;
  if (!meta) {
    console.log('[building-tile] no EXT_structural_metadata in this tile (no per-building attributes).');
    console.log('[building-tile] ================================================');
    return;
  }

  const schemaClasses = meta.schema?.classes ?? {};
  for (const className of Object.keys(schemaClasses)) {
    const props = schemaClasses[className].properties ?? {};
    const desc = Object.keys(props).map((p) => {
      const def = props[p];
      return `${p}(${def.type}${def.componentType ? ':' + def.componentType : ''})`;
    });
    console.log(`[building-tile] schema class "${className}" properties:`, desc);
  }

  const tables = meta.propertyTables ?? [];
  for (let ti = 0; ti < tables.length; ti++) {
    const table = tables[ti];
    const propNames = Object.keys(table.properties ?? {});
    console.log(`[building-tile] propertyTable ${ti} class="${table.class}" count=${table.count} props:`, propNames);

    // Decode the first few features so you can see real values.
    const sampleCount = Math.min(3, table.count);
    const classProps = schemaClasses[table.class]?.properties ?? {};
    for (let f = 0; f < sampleCount; f++) {
      const row: Record<string, unknown> = {};
      for (const name of propNames) {
        row[name] = readMetadataValue(json, bin, table.properties[name], classProps[name], f);
      }
      console.log(`[building-tile]   feature ${f}:`, row);
    }
  }
  console.log('[building-tile] ================================================');
}

/** Read one property-table value for feature `i` (string or numeric scalar). */
function readMetadataValue(json: GlTF, bin: Uint8Array, prop: any, classProp: any, i: number): unknown {
  if (!prop || prop.values == null) return undefined;
  const type = classProp?.type;
  if (type === 'STRING' || prop.stringOffsets != null) {
    const valBytes = decodeBufferView(json, bin, prop.values);
    const offBytes = decodeBufferView(json, bin, prop.stringOffsets);
    const offDV = new DataView(offBytes.buffer, offBytes.byteOffset, offBytes.byteLength);
    const s = offDV.getUint32(i * 4, true);
    const e = offDV.getUint32((i + 1) * 4, true);
    return new TextDecoder().decode(valBytes.subarray(s, e));
  }
  if (type === 'BOOLEAN') {
    const bytes = decodeBufferView(json, bin, prop.values);
    return ((bytes[i >> 3] >> (i & 7)) & 1) === 1;
  }
  // SCALAR numeric: read by componentType.
  const ct = metadataComponentType(classProp?.componentType);
  if (ct == null) return '<unsupported>';
  const bytes = decodeBufferView(json, bin, prop.values);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const cs = componentSize(ct);
  return componentReader(dv, ct)(i * cs);
}

/** Map an EXT_structural_metadata componentType name to a glTF componentType. */
function metadataComponentType(name: string | undefined): number | null {
  switch (name) {
    case 'INT8': return 5120;
    case 'UINT8': return 5121;
    case 'INT16': return 5122;
    case 'UINT16': return 5123;
    case 'UINT32': case 'INT32': return 5125;
    case 'FLOAT32': return 5126;
    default: return null;
  }
}

// 3DBAG heights are ellipsoidal (WGS84); CityJSON heights are NAP. In the
// Netherlands the geoid sits ~43 m below the ellipsoid, so subtract a constant
// to line the glTF buildings up vertically with the CityJSON/NAP version.
const ELLIPSOID_TO_NAP = 43.0;

// WGS84 ellipsoid (ECEF -> geodetic).
const WGS84_A = 6378137.0;
const WGS84_F = 1 / 298.257223563;
const WGS84_B = WGS84_A * (1 - WGS84_F);
const WGS84_E2 = WGS84_F * (2 - WGS84_F);

type GlTF = {
  accessors: any[];
  bufferViews: any[];
  buffers: any[];
  meshes: any[];
  nodes?: any[];
};

/**
 * Projection frame for a tile. Buildings/terrain are projected with the SAME
 * equirectangular projection as the OSM ground layers ({@link OSMBuilder}) and
 * the camera ({@link MapProjection}), so the two stay aligned no matter how far
 * the camera pans from the region center.
 *
 * (The previous ENU tangent-plane projection was only metrically exact at the
 * region center and drifted from the equirectangular ground grid as you panned
 * away — it used the true ellipsoidal meridian scale and the per-point cos(lat),
 * while the ground grid freezes cos(lat) at latMid. Buildings visibly slid off
 * the terrain the further you got from where you landed, and snapped back when
 * a region re-center moved latMid/lngMid onto the new view.)
 */
type ProjFrame = {
  lngMid: number;
  latMid: number;
  /** cos(latMid) — frozen longitude scale, matching MapProjection / OSMBuilder. */
  cosLat0: number;
  scale: number;
  metersToNorm: number;
};

function makeProjFrame(proj: MapProjection): ProjFrame {
  return {
    lngMid: proj.lngMid,
    latMid: proj.latMid,
    cosLat0: proj.cosLat,
    scale: proj.scale,
    metersToNorm: proj.metersToNorm,
  };
}

// Second eccentricity squared, derived for the ECEF -> geodetic conversion.
const WGS84_EP2 = WGS84_E2 / (1 - WGS84_E2);
const RAD2DEG = 180 / Math.PI;

// Reused output for ecefToGeodetic. projectPrimitive runs synchronously, so a
// single module-level scratch avoids allocating per vertex.
const geo = { lat: 0, lng: 0, h: 0 };

/**
 * WGS84 ECEF -> geodetic (Bowring's closed-form, accurate to well under a mm at
 * these latitudes). Writes lat/lng (radians) and ellipsoidal height (meters)
 * into the shared {@link geo} scratch.
 */
function ecefToGeodetic(X: number, Y: number, Z: number): void {
  const p = Math.sqrt(X * X + Y * Y);
  const th = Math.atan2(WGS84_A * Z, WGS84_B * p);
  const sth = Math.sin(th), cth = Math.cos(th);
  const lat = Math.atan2(
    Z + WGS84_EP2 * WGS84_B * sth * sth * sth,
    p - WGS84_E2 * WGS84_A * cth * cth * cth,
  );
  const slat = Math.sin(lat);
  const N = WGS84_A / Math.sqrt(1 - WGS84_E2 * slat * slat);
  geo.lat = lat;
  geo.lng = Math.atan2(Y, X);
  geo.h = p / Math.cos(lat) - N;
}

export class CityGeometryBuilder {
  // Reused per-primitive scratch buffers. Re-projecting every tile allocated a
  // fresh ~MB Float32Array per call, which showed up as GC spikes. These grow on
  // demand and are overwritten each call, so they hold no state between tiles.
  // projectPrimitive writes interleaved xyz into posScratch; addPrimitive
  // de-indexes straight out of it.
  private posScratch = new Float32Array(0);
  private fidScratch = new Int32Array(0);

  /**
   * Build a city tile's de-indexed positions. Caller must ensure
   * MeshoptDecoder.ready has resolved before calling.
   */
  public buildPositions(glb: ArrayBuffer, bounds: LatLngBounds, flatten = true): CityPositionsResult {
    const { json, bin } = parseGlb(glb);

    // TEMP: dump what per-building info a 3DBAG glTF tile actually contains.
    // logBuildingTileContentsOnce(json, bin);

    const proj = new MapProjection();
    proj.setBounds(bounds);
    const frame = makeProjFrame(proj);

    const nodes = json.nodes ?? [];

    // Pre-size the output buffer from the total triangle-index count. The vertex
    // count is exact (TRIANGLES index counts are multiples of 3), so we fill one
    // right-sized array and hand it straight to the mesh — no growable buffer
    // and no copy. Only reads accessor.count, so it's effectively free. Each
    // vertex is one interleaved vec4 (x, up, -y, featureId).
    let totalFloats = 0;
    for (let n = 0; n < nodes.length; n++) {
      const node = nodes[n];
      if (node.mesh == null) continue;
      const mesh = json.meshes[node.mesh];
      if (!mesh) continue;
      for (const prim of mesh.primitives ?? []) {
        if (prim.mode != null && prim.mode !== 4) continue;
        if (prim.attributes?.POSITION == null || prim.indices == null) continue;
        totalFloats += json.accessors[prim.indices].count * VERTEX_FLOATS;
      }
    }

    const vertices = new Float32Array(totalFloats);
    let writeOffset = 0;
    let surfaces = 0;
    let objectCount = 0;

    // Surveyed ground height per building (b3_h_maaiveld, NAP metres). Used by
    // addPrimitive's per-building flattening to reject stray vertices collapsed
    // far below the building (a 3DBAG data defect) when picking the base height.
    const groundHeights = flatten ? this.readGroundHeights(json, bin) : null;

    for (let n = 0; n < nodes.length; n++) {
      const node = nodes[n];
      if (node.mesh == null) continue;
      const mesh = json.meshes[node.mesh];
      if (!mesh) continue;
      const matrix: number[] = node.matrix ?? IDENTITY;
      objectCount++;

      for (const prim of mesh.primitives ?? []) {
        if (prim.mode != null && prim.mode !== 4) continue; // TRIANGLES only
        if (prim.attributes?.POSITION == null || prim.indices == null) continue;
        writeOffset = this.addPrimitive(json, bin, prim, matrix, frame, flatten, groundHeights, vertices, writeOffset);
        surfaces++;
      }
    }

    // Per-building attributes (BAG pand id, year, area, …) for the click panel.
    const buildings = this.readBuildingInfo(json, bin);

    // Exact fill -> hand the buffer over directly; only slice in the (degenerate)
    // case where less than the pre-scanned upper bound was written.
    const filled = writeOffset === vertices.length ? vertices : vertices.slice(0, writeOffset);

    // Compute the tile's AABB in one allocation-free pass while we're still off
    // the main thread. The main thread used to do this via meshGetBoundingBox,
    // which allocated a Vector3 per vertex on the frame that finalizes the tile.
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < writeOffset; i += VERTEX_FLOATS) {
      const x = filled[i], y = filled[i + 1], z = filled[i + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const empty = writeOffset === 0;

    return {
      vertices: filled,
      buildings,
      triangleCount: writeOffset / (VERTEX_FLOATS * 3),
      surfaceCount: surfaces,
      objectCount,
      lodUsed: '2.2 (glTF)',
      min: empty ? [0, 0, 0] : [minX, minY, minZ],
      max: empty ? [0, 0, 0] : [maxX, maxY, maxZ],
    };
  }

  /**
   * Decode one primitive's triangles into engine-space positions, writing the
   * de-indexed (x, up, -y) triangle soup into `out` starting at `offset`.
   * Returns the new write offset.
   */
  private addPrimitive(
    json: GlTF,
    bin: Uint8Array,
    prim: any,
    matrix: number[],
    frame: ProjFrame,
    flatten: boolean,
    groundHeights: Float64Array | null,
    out: Float32Array,
    offset: number,
  ): number {
    const idxAcc = json.accessors[prim.indices];
    const indexCount = idxAcc.count;
    const pos = this.projectPrimitive(json, bin, prim, matrix, frame);
    const vertexCount = pos.length / 3;

    // 3DBAG packs every building of a tile into one primitive, tagged per vertex
    // by the _FEATURE_ID_0 attribute. We read it for two purposes: per-building
    // base flattening, and the per-vertex picking id written into each vertex's
    // `.w` below.
    const fidIdx = prim.attributes._FEATURE_ID_0;
    let fids: Int32Array | null = null;
    if (fidIdx != null) {
      const fidAcc = json.accessors[fidIdx];
      const fidView = directScalarView(json, bin, fidAcc);
      if (this.fidScratch.length < vertexCount) this.fidScratch = new Int32Array(vertexCount);
      fids = this.fidScratch;
      let maxFid = 0;
      if (fidView) {
        for (let i = 0; i < vertexCount; i++) {
          const f = fidView[i] | 0;
          fids[i] = f;
          if (f > maxFid) maxFid = f;
        }
      } else {
        const readFid = makeScalarReader(json, bin, fidAcc);
        for (let i = 0; i < vertexCount; i++) {
          const f = readFid(i) | 0;
          fids[i] = f;
          if (f > maxFid) maxFid = f;
        }
      }
      // Per-building base flattening (Option A): drop each building so its ground
      // sits at y=0, since the terrain is a flat plane with no real elevation —
      // otherwise buildings float by their ground height.
      //
      // The base is the building's surveyed ground height (b3_h_maaiveld). Using
      // that attribute rather than the lowest vertex makes this immune to stray
      // vertices 3DBAG sometimes collapses far below a building (e.g. to the tile
      // origin — a data defect), which would otherwise drag the base down and
      // lift the whole building, leaving a spike poking at the ground. It matches
      // the geometry's own ground to ~cm (the tile's ellipsoidal heights vs NAP
      // differ only by the geoid residual left by the constant ELLIPSOID_TO_NAP).
      // Buildings with no valid maaiveld fall back to their lowest vertex.
      if (flatten) {
        const baseY = new Float32Array(maxFid + 1).fill(Infinity);
        let missing = false;
        for (let f = 0; f <= maxFid; f++) {
          const g = groundHeights ? groundHeights[f] : NaN;
          if (Number.isNaN(g)) missing = true;
          else baseY[f] = g * frame.metersToNorm;
        }
        // Only scan vertices for the buildings still lacking a ground height.
        if (missing) {
          for (let i = 0; i < vertexCount; i++) {
            const f = fids[i];
            if (groundHeights && !Number.isNaN(groundHeights[f])) continue;
            const y = pos[i * 3 + 1];
            if (y < baseY[f]) baseY[f] = y;
          }
        }
        for (let i = 0; i < vertexCount; i++) {
          const b = baseY[fids[i]];
          if (b !== Infinity) pos[i * 3 + 1] -= b;
        }
      }
    }

    // Interleave each vertex as (x, up, -y, featureId). `.w` carries the
    // tile-local building id, or NO_FEATURE_ID when this primitive has none, so
    // every vertex shares one stride-4 layout (no separate id attribute, no
    // main-thread interleave).
    const indices = readIndices(json, bin, idxAcc);
    let pi = offset;
    for (let t = 0; t + 2 < indexCount; t += 3) {
      const ia = indices[t], ib = indices[t + 1], ic = indices[t + 2];
      const a = ia * 3, b = ib * 3, c = ic * 3;
      out[pi]      = pos[a]; out[pi + 1]  = pos[a + 1]; out[pi + 2]  = pos[a + 2];
      out[pi + 3]  = fids ? fids[ia] : NO_FEATURE_ID;
      out[pi + 4]  = pos[b]; out[pi + 5]  = pos[b + 1]; out[pi + 6]  = pos[b + 2];
      out[pi + 7]  = fids ? fids[ib] : NO_FEATURE_ID;
      out[pi + 8]  = pos[c]; out[pi + 9]  = pos[c + 1]; out[pi + 10] = pos[c + 2];
      out[pi + 11] = fids ? fids[ic] : NO_FEATURE_ID;
      pi += VERTEX_FLOATS * 3;
    }
    return pi;
  }

  /**
   * Decode the tile's per-building attributes from its EXT_structural_metadata
   * property table, indexed by the tile-local `_FEATURE_ID_0`. Returns an empty
   * array when the tile carries no per-building metadata.
   */
  private readBuildingInfo(json: GlTF, bin: Uint8Array): BuildingInfo[] {
    const meta = (json as any).extensions?.EXT_structural_metadata;
    const table = meta?.propertyTables?.[0];
    if (!table) return [];
    const count: number = table.count;
    const classProps = meta.schema?.classes?.[table.class]?.properties ?? {};

    const ids = this.readStringColumn(json, bin, table.properties?.['identificatie'], count);
    const status = this.readStringColumn(json, bin, table.properties?.['status'], count);
    const roof = this.readStringColumn(json, bin, table.properties?.['b3_dak_type'], count);
    const year = this.readScalarColumn(json, bin, table.properties?.['oorspronkelijkbouwjaar'], classProps['oorspronkelijkbouwjaar'], count);
    const area = this.readScalarColumn(json, bin, table.properties?.['b3_opp_grond'], classProps['b3_opp_grond'], count);

    const out: BuildingInfo[] = new Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = {
        id: ids?.[i] ?? '',
        year: year ? year[i] : 0,
        groundArea: area ? area[i] : 0,
        status: status?.[i] ?? '',
        roofType: roof?.[i] ?? '',
      };
    }
    return out;
  }

  /**
   * Read the per-building surveyed ground height (b3_h_maaiveld, NAP metres) into
   * an array indexed by `_FEATURE_ID_0`, mapping the schema noData sentinel
   * (FLOAT_MAX) to NaN so callers treat it as "unknown". Returns null when the
   * tile carries no metadata table.
   */
  private readGroundHeights(json: GlTF, bin: Uint8Array): Float64Array | null {
    const meta = (json as any).extensions?.EXT_structural_metadata;
    const table = meta?.propertyTables?.[0];
    if (!table) return null;
    const classProps = meta.schema?.classes?.[table.class]?.properties ?? {};
    const col = this.readScalarColumn(json, bin, table.properties?.['b3_h_maaiveld'], classProps['b3_h_maaiveld'], table.count);
    if (!col) return null;
    for (let i = 0; i < col.length; i++) if (col[i] > 1e37) col[i] = NaN;
    return col;
  }

  /** Read a scalar-typed structural-metadata property column into an array. */
  private readScalarColumn(json: GlTF, bin: Uint8Array, prop: any, classProp: any, count: number): Float64Array | null {
    if (!prop || prop.values == null) return null;
    const ct = metadataComponentType(classProp?.componentType);
    if (ct == null) return null;
    const bytes = decodeBufferView(json, bin, prop.values);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const cs = componentSize(ct);
    const read = componentReader(dv, ct);
    const out = new Float64Array(count);
    for (let i = 0; i < count; i++) out[i] = read(i * cs);
    return out;
  }

  /**
   * Project a primitive's vertices into engine-space (x, up, -y), written
   * **interleaved** (xyz xyz …) into the reused {@link posScratch}. Returns a
   * view of length `vertexCount * 3`. Interleaving keeps each vertex's 3 floats
   * on one cache line, so the de-index gather (random vertex access) takes 1
   * cache miss per vertex instead of 3 (it previously read 3 separate arrays).
   */
  private projectPrimitive(
    json: GlTF,
    bin: Uint8Array,
    prim: any,
    matrix: number[],
    frame: ProjFrame,
  ): Float32Array {
    const posAcc = json.accessors[prim.attributes.POSITION];
    const vertexCount = posAcc.count;
    const need = vertexCount * 3;
    if (this.posScratch.length < need) this.posScratch = new Float32Array(need);
    const pos = this.posScratch;
    if (vertexCount === 0) return pos.subarray(0, 0);

    const m = matrix;
    const { lngMid, latMid, cosLat0, scale, metersToNorm } = frame;

    // Per-primitive affine map from local (de-quantized) position space to world.
    // The exact ECEF -> geodetic -> equirectangular projection is affine to well
    // under a millimetre across a single tile (its curvature only matters over
    // kilometres), so instead of running the trig ecefToGeodetic per vertex we
    // evaluate it exactly at the local origin and the three unit axes and fit a
    // 3x4 affine — each vertex is then just 9 mul / 9 add. Anchoring the fit at
    // each tile (not the region centre) is what keeps the residual in the mm
    // range: a linear map's error grows with the square of the distance from its
    // anchor, which is why the old region-centred frame drifted by metres far out.
    const projectLocal = (lx: number, ly: number, lz: number, out: number[]): void => {
      const gx = m[0] * lx + m[4] * ly + m[8] * lz + m[12];
      const gy = m[1] * lx + m[5] * ly + m[9] * lz + m[13];
      const gz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
      // glTF (Y-up) -> ECEF (x, -z, y) -> geodetic -> equirectangular, matching
      // OSMBuilder so buildings stay glued to the terrain.
      ecefToGeodetic(gx, -gz, gy);
      out[0] = (geo.lng * RAD2DEG - lngMid) * cosLat0 * scale;
      out[1] = (geo.h - ELLIPSOID_TO_NAP) * metersToNorm;
      out[2] = -(geo.lat * RAD2DEG - latMid) * scale;
    };
    const c0: number[] = [0, 0, 0], cX: number[] = [0, 0, 0];
    const cY: number[] = [0, 0, 0], cZ: number[] = [0, 0, 0];
    projectLocal(0, 0, 0, c0);
    projectLocal(1, 0, 0, cX);
    projectLocal(0, 1, 0, cY);
    projectLocal(0, 0, 1, cZ);
    // Affine columns: world = o0 + lx*(X-o0) + ly*(Y-o0) + lz*(Z-o0).
    const o0x = c0[0], o0y = c0[1], o0z = c0[2];
    const dXx = cX[0] - o0x, dXy = cX[1] - o0y, dXz = cX[2] - o0z;
    const dYx = cY[0] - o0x, dYy = cY[1] - o0y, dYz = cY[2] - o0z;
    const dZx = cZ[0] - o0x, dZy = cZ[1] - o0y, dZz = cZ[2] - o0z;

    // Fast path for the common 3DBAG case: int16 normalized (KHR_mesh_quantization).
    const posBv = json.bufferViews[posAcc.bufferView];
    const posBytes = decodeBufferView(json, bin, posAcc.bufferView);
    const posBase = posAcc.byteOffset ?? 0;
    const posCT = posAcc.componentType;
    const posNorm = !!posAcc.normalized;
    const posCS = componentSize(posCT);
    const posStride = posBv.byteStride ?? posCS * 3;
    const posInv = posNorm ? 1 / normalizeDivisor(posCT) : 1;
    // Zero-copy Int16Array view (the 3DBAG case): avoids a DataView.getInt16
    // call per component, which dominates this loop. Needs 2-byte alignment.
    const fastInt16 = posCT === 5122 && posNorm
      && (posBytes.byteOffset & 1) === 0 && (posBase & 1) === 0 && (posStride & 1) === 0;

    if (fastInt16) {
      const pos16 = new Int16Array(posBytes.buffer, posBytes.byteOffset, posBytes.byteLength >> 1);
      const strideI = posStride >> 1;
      let o = posBase >> 1;
      let p = 0;
      for (let i = 0; i < vertexCount; i++, o += strideI, p += 3) {
        const vx = pos16[o] * posInv;
        const vy = pos16[o + 1] * posInv;
        const vz = pos16[o + 2] * posInv;
        pos[p] = o0x + vx * dXx + vy * dYx + vz * dZx;
        pos[p + 1] = o0y + vx * dXy + vy * dYy + vz * dZy;
        pos[p + 2] = o0z + vx * dXz + vy * dYz + vz * dZz;
      }
    } else {
      const readPos = makeVec3Reader(json, bin, posAcc);
      let p = 0;
      for (let i = 0; i < vertexCount; i++, p += 3) {
        const v = readPos(i);
        pos[p] = o0x + v[0] * dXx + v[1] * dYx + v[2] * dZx;
        pos[p + 1] = o0y + v[0] * dXy + v[1] * dYy + v[2] * dZy;
        pos[p + 2] = o0z + v[0] * dXz + v[1] * dYz + v[2] * dZz;
      }
    }
    return pos.subarray(0, need);
  }

  /**
   * Build a 3D Basisvoorziening "terreinen" tile, splitting its surfaces into
   * the colored {@link TERRAIN_CATEGORY_DEFS} categories (one buffer per
   * category) instead of merging everything. Surfaces are classified per BGT
   * feature via the EXT_structural_metadata property table referenced by
   * `_FEATURE_ID_0`. Caller must ensure MeshoptDecoder.ready has resolved.
   */
  public buildTerrainPositions(glb: ArrayBuffer, bounds: LatLngBounds): Record<TerrainCategory, Float32Array> {
    const { json, bin } = parseGlb(glb);

    const proj = new MapProjection();
    proj.setBounds(bounds);
    const frame = makeProjFrame(proj);

    // featureId -> category, decoded once from the tile's property table.
    const featureCategory = this.readFeatureCategories(json, bin);

    const buffers = {} as Record<TerrainCategory, FloatBuffer>;
    for (const def of TERRAIN_CATEGORY_DEFS) buffers[def.key] = new FloatBuffer();

    const nodes = json.nodes ?? [];
    for (let n = 0; n < nodes.length; n++) {
      const node = nodes[n];
      if (node.mesh == null) continue;
      const mesh = json.meshes[node.mesh];
      if (!mesh) continue;
      const matrix: number[] = node.matrix ?? IDENTITY;

      for (const prim of mesh.primitives ?? []) {
        if (prim.mode != null && prim.mode !== 4) continue; // TRIANGLES only
        if (prim.attributes?.POSITION == null || prim.indices == null) continue;

        const pos = this.projectPrimitive(json, bin, prim, matrix, frame);
        const idxAcc = json.accessors[prim.indices];
        const indices = readIndices(json, bin, idxAcc);
        const indexCount = idxAcc.count;

        // Each surface (and its triangle fan) shares one feature id, so the
        // first vertex of a triangle decides the whole triangle's category.
        const fidIdx = prim.attributes._FEATURE_ID_0;
        const readFid = featureCategory && fidIdx != null
          ? makeScalarReader(json, bin, json.accessors[fidIdx])
          : null;

        for (let t = 0; t + 2 < indexCount; t += 3) {
          const i0 = indices[t], i1 = indices[t + 1], i2 = indices[t + 2];
          const cat = readFid ? featureCategory![readFid(i0) | 0] ?? 'other' : 'other';
          const out = buffers[cat];
          out.reserve(9);
          const pd = out.data;
          let pi = out.length;
          const a = i0 * 3, b = i1 * 3, c = i2 * 3;
          pd[pi] = pos[a]; pd[pi + 1] = pos[a + 1]; pd[pi + 2] = pos[a + 2];
          pd[pi + 3] = pos[b]; pd[pi + 4] = pos[b + 1]; pd[pi + 5] = pos[b + 2];
          pd[pi + 6] = pos[c]; pd[pi + 7] = pos[c + 1]; pd[pi + 8] = pos[c + 2];
          out.length = pi + 9;
        }
      }
    }

    const layers = {} as Record<TerrainCategory, Float32Array>;
    for (const def of TERRAIN_CATEGORY_DEFS) layers[def.key] = buffers[def.key].toArray();
    return layers;
  }

  /**
   * Decode the tile's EXT_structural_metadata property table into a
   * featureId -> {@link TerrainCategory} lookup. Returns null when the tile
   * carries no per-feature metadata (callers then fall back to 'other').
   */
  private readFeatureCategories(json: GlTF, bin: Uint8Array): TerrainCategory[] | null {
    const meta = (json as any).extensions?.EXT_structural_metadata;
    const table = meta?.propertyTables?.[0];
    if (!table) return null;
    const count: number = table.count;
    const cls = this.readStringColumn(json, bin, table.properties?.['3df_class'], count);
    const fun = this.readStringColumn(json, bin, table.properties?.['bgt_functie'], count);
    const fys = this.readStringColumn(json, bin, table.properties?.['bgt_fysiekvoorkomen'], count);
    const typ = this.readStringColumn(json, bin, table.properties?.['bgt_type'], count);
    if (!cls && !fun && !fys && !typ) return null;
    const out: TerrainCategory[] = new Array(count);
    for (let i = 0; i < count; i++) {
      const c = cls?.[i] ?? '', f = fun?.[i] ?? '', fy = fys?.[i] ?? '', t = typ?.[i] ?? '';
      const cat = classifyTerrain(c, f, fy, t);
      out[i] = cat;
      logTerrainCombo(c, f, fy, t, cat);
    }
    return out;
  }

  /** Read a string-typed structural-metadata property column into an array. */
  private readStringColumn(json: GlTF, bin: Uint8Array, prop: any, count: number): string[] | null {
    if (!prop || prop.values == null || prop.stringOffsets == null) return null;
    const valBytes = decodeBufferView(json, bin, prop.values);
    const offBytes = decodeBufferView(json, bin, prop.stringOffsets);
    const offDV = new DataView(offBytes.buffer, offBytes.byteOffset, offBytes.byteLength);
    const dec = new TextDecoder();
    const out: string[] = new Array(count);
    for (let i = 0; i < count; i++) {
      const s = offDV.getUint32(i * 4, true);
      const e = offDV.getUint32((i + 1) * 4, true);
      out[i] = dec.decode(valBytes.subarray(s, e));
    }
    return out;
  }
}

// --- glTF / GLB plumbing -----------------------------------------------------

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Split a .glb into its JSON chunk and BIN chunk. */
function parseGlb(buffer: ArrayBuffer): { json: GlTF; bin: Uint8Array } {
  const dv = new DataView(buffer);
  const magic = dv.getUint32(0, true);
  if (magic !== 0x46546c67) throw new Error('[3DBAG-glTF] not a .glb (bad magic)');
  let off = 12;
  let json: GlTF | null = null;
  let bin: Uint8Array | null = null;
  while (off < dv.byteLength) {
    const chunkLen = dv.getUint32(off, true);
    const chunkType = dv.getUint32(off + 4, true);
    const start = off + 8;
    if (chunkType === 0x4e4f534a) {
      // 'JSON'
      json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, start, chunkLen)));
    } else if (chunkType === 0x004e4942) {
      // 'BIN\0'
      bin = new Uint8Array(buffer, start, chunkLen);
    }
    off = start + chunkLen;
  }
  if (!json) throw new Error('[3DBAG-glTF] missing JSON chunk');
  return { json, bin: bin ?? new Uint8Array(0) };
}

// Cache of decoded (and meshopt-decompressed) bufferView bytes, keyed by index.
// Reused across accessors that share a bufferView within one glb parse.
const bvCache = new WeakMap<GlTF, Map<number, Uint8Array>>();

/** Return a bufferView's bytes, transparently meshopt-decoding when compressed. */
function decodeBufferView(json: GlTF, bin: Uint8Array, bvIndex: number): Uint8Array {
  let cache = bvCache.get(json);
  if (!cache) { cache = new Map(); bvCache.set(json, cache); }
  const hit = cache.get(bvIndex);
  if (hit) return hit;

  const bv = json.bufferViews[bvIndex];
  const ext = bv.extensions?.EXT_meshopt_compression;
  let result: Uint8Array;
  if (ext) {
    const src = new Uint8Array(bin.buffer, bin.byteOffset + (ext.byteOffset ?? 0), ext.byteLength);
    const out = new Uint8Array(ext.count * ext.byteStride);
    MeshoptDecoder.decodeGltfBuffer(out, ext.count, ext.byteStride, src, ext.mode, ext.filter ?? 'NONE');
    result = out;
  } else {
    result = new Uint8Array(bin.buffer, bin.byteOffset + (bv.byteOffset ?? 0), bv.byteLength);
  }
  cache.set(bvIndex, result);
  return result;
}

/** Build a reader returning the (dequantized) VEC3 for a given vertex index. */
function makeVec3Reader(json: GlTF, bin: Uint8Array, accessor: any): (i: number) => [number, number, number] {
  const bv = json.bufferViews[accessor.bufferView];
  const bytes = decodeBufferView(json, bin, accessor.bufferView);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const base = accessor.byteOffset ?? 0;
  const ct = accessor.componentType;
  const norm = !!accessor.normalized;
  const elemSize = componentSize(ct) * 3;
  const stride = bv.byteStride ?? elemSize;
  const read = componentReader(dv, ct);
  const cs = componentSize(ct);
  const inv = norm ? 1 / normalizeDivisor(ct) : 1;
  const out: [number, number, number] = [0, 0, 0];
  return (i: number) => {
    const o = base + i * stride;
    out[0] = read(o) * inv;
    out[1] = read(o + cs) * inv;
    out[2] = read(o + 2 * cs) * inv;
    return out;
  };
}

/** Build a reader returning the scalar value for a given vertex index. */
function makeScalarReader(json: GlTF, bin: Uint8Array, accessor: any): (i: number) => number {
  const bv = json.bufferViews[accessor.bufferView];
  const bytes = decodeBufferView(json, bin, accessor.bufferView);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const base = accessor.byteOffset ?? 0;
  const ct = accessor.componentType;
  const cs = componentSize(ct);
  const stride = bv.byteStride ?? cs;
  const read = componentReader(dv, ct);
  const inv = accessor.normalized ? 1 / normalizeDivisor(ct) : 1;
  return (i: number) => read(base + i * stride) * inv;
}

/**
 * Zero-copy typed-array view of a tightly-packed, aligned accessor, or null when
 * the layout is interleaved/misaligned and the generic (DataView) path is needed.
 * Normalization is ignored — callers that need it apply the divisor themselves.
 */
function directScalarView(
  json: GlTF, bin: Uint8Array, accessor: any,
): Int8Array | Uint8Array | Int16Array | Uint16Array | Uint32Array | Float32Array | null {
  const bv = json.bufferViews[accessor.bufferView];
  const bytes = decodeBufferView(json, bin, accessor.bufferView);
  const ct = accessor.componentType;
  const cs = componentSize(ct);
  const stride = bv.byteStride ?? cs;
  if (stride !== cs) return null; // interleaved
  const off = bytes.byteOffset + (accessor.byteOffset ?? 0);
  if (off % cs !== 0) return null; // misaligned for a typed-array view
  const count = accessor.count;
  switch (ct) {
    case 5120: return new Int8Array(bytes.buffer, off, count);
    case 5121: return new Uint8Array(bytes.buffer, off, count);
    case 5122: return new Int16Array(bytes.buffer, off, count);
    case 5123: return new Uint16Array(bytes.buffer, off, count);
    case 5125: return new Uint32Array(bytes.buffer, off, count);
    case 5126: return new Float32Array(bytes.buffer, off, count);
    default: return null;
  }
}

/** Read an index buffer into an indexable typed array (zero-copy when possible). */
function readIndices(json: GlTF, bin: Uint8Array, accessor: any): Uint32Array | Uint16Array | Uint8Array {
  const direct = directScalarView(json, bin, accessor);
  if (direct && !(direct instanceof Float32Array) && !(direct instanceof Int8Array) && !(direct instanceof Int16Array)) {
    return direct;
  }
  const bv = json.bufferViews[accessor.bufferView];
  const bytes = decodeBufferView(json, bin, accessor.bufferView);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const base = accessor.byteOffset ?? 0;
  const ct = accessor.componentType;
  const cs = componentSize(ct);
  const stride = bv.byteStride ?? cs;
  const read = componentReader(dv, ct);
  const out = new Uint32Array(accessor.count);
  for (let i = 0; i < accessor.count; i++) out[i] = read(base + i * stride);
  return out;
}

function componentSize(ct: number): number {
  switch (ct) {
    case 5120: case 5121: return 1; // (u)byte
    case 5122: case 5123: return 2; // (u)short
    case 5125: case 5126: return 4; // uint / float
    default: throw new Error(`[3DBAG-glTF] unsupported componentType ${ct}`);
  }
}

function normalizeDivisor(ct: number): number {
  switch (ct) {
    case 5120: return 127;    // byte
    case 5121: return 255;    // ubyte
    case 5122: return 32767;  // short
    case 5123: return 65535;  // ushort
    default: return 1;
  }
}

function componentReader(dv: DataView, ct: number): (o: number) => number {
  switch (ct) {
    case 5120: return (o) => dv.getInt8(o);
    case 5121: return (o) => dv.getUint8(o);
    case 5122: return (o) => dv.getInt16(o, true);
    case 5123: return (o) => dv.getUint16(o, true);
    case 5125: return (o) => dv.getUint32(o, true);
    case 5126: return (o) => dv.getFloat32(o, true);
    default: throw new Error(`[3DBAG-glTF] unsupported componentType ${ct}`);
  }
}

/**
 * A Float32Array that grows by doubling (write mesh data straight into a typed
 * buffer). `toArray()` returns an exact-length copy, since a single-attribute
 * mesh uploads its whole backing store.
 */
class FloatBuffer {
  public data: Float32Array;
  public length = 0;

  constructor(initial = 4096) {
    this.data = new Float32Array(initial);
  }

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
