import earcut from 'earcut';
import { VectorTile } from '@mapbox/vector-tile';
import { PbfReader } from 'pbf';
import { hypot2 } from 'webGPU-framework/renderer/utils/MathUtils';
import MapProjection from '../osm/MapProjection';
import {
  classifyRings,
  clipLineToTile,
  clipRingToTile,
  tileToLatLon,
} from '../osm/pdokDecodeCore';
import type { TilePoint } from '../osm/pdokDecodeCore';
import type { LatLngBounds } from '../osm/osmGeometryCore';
import type { TileCoord } from '../osm/PDOKTileService';
import type { BuildingInfo } from '../osm/cityGlTFCore';
import { buildTraceBlock, GBUFFER_ATLAS_MAX, GBUFFER_ATLAS_STEP_M, getRoofChart, layoutWalls } from '../city/cityLayout';
import type { TraceTree } from '../city/cityLayout';
import type { BuildingPolygon, CityTraceBlock } from '../city/cityLayout';
import { buildTrees, TREE_MATERIAL_BASE_ID } from './worldTrees';
import { findWallBottoms } from '../city/hiddenWalls';
import TerrainGrid, { GRID_VERTEX_FLOATS, GridMesh, insideRing } from './terrainGrid';

/**
 * Pure (renderer-free) geometry core for the world map. Decodes one
 * OpenMapTiles-schema vector tile (as served by OpenFreeMap) into
 *  - flat ground layers (water, green, roads, …) as indexed xyz meshes, one per
 *    render layer, ready for {@link TerrainLayerBuffer}, and
 *  - with the heights of the ground (see ElevationService), the ground mesh, with
 *    the layers draped on it and the buildings and trees standing on it, and
 *  - extruded 3D buildings as one stride-4 `(x, up, -y, featureId)` soup plus
 *    per-building attributes, the same layout the 3DBAG path produces, so the
 *    shared osm shader / GPU picking / highlight work unchanged.
 *
 * Runs inside {@link worldMapWorker}; keep it free of Mesh/Renderer imports.
 *
 * Winding: earcut always emits counter-clockwise triangles in (x, north) map
 * space, which is front-facing after the engine's (x, z, -y) swap (the terrain
 * layers render with back-face culling). Building walls are built from rings
 * normalised to CCW outers / CW holes so their faces point outward too.
 */

// Mirrors cityGlTFCore.NO_FEATURE_ID / VERTEX_FLOATS. Re-declared so this
// worker-side module doesn't pull the meshopt decoder in via cityGlTFCore.
const NO_FEATURE_ID = -1;
const VERTEX_FLOATS = 4;
// Picking packs `slot * 65536 + featureId`; ids beyond that can't be encoded. The building vertices store the id plus
// FEATURE_ID_BIAS (the trees are negative) in 16 bits, see quantizeVertices
const FEATURE_ID_BIAS = 16;
const MAX_FEATURE_ID = 65535 - FEATURE_ID_BIAS;

export type WorldLayerConfig = {
  key: string;
  label: string;
  /** #rrggbb, as the color params show it (parsed with Color.fromHex) */
  color: string;
  defaultOn: boolean;
};

// the layers that keep the normal up on the terrain (see TerrainGrid.drapeTriangles): the water is flat
const FLAT_NORMAL_LAYERS = new Set(['water', 'waterways']);

// Render layers, lowest first: stacked up from the ground quad at 0, LAYER_GAP apart (see LAYER_GAP).
export const WORLD_LAYERS: WorldLayerConfig[] = [
  { key: 'urban',     label: 'urban areas', color: '#9e9994', defaultOn: false },
  { key: 'farmland',  label: 'farmland',    color: '#b3b885', defaultOn: true },
  { key: 'sand',      label: 'sand',        color: '#dbcc9e', defaultOn: true },
  { key: 'green',     label: 'parks & grass', color: '#70965e', defaultOn: true },
  { key: 'forest',    label: 'forest',      color: '#547a4d', defaultOn: true },
  { key: 'water',     label: 'water',       color: '#546579', defaultOn: true },
  // the color of the water: the lines of the rivers and canals lie over the water areas, a lighter color showed them
  { key: 'waterways', label: 'waterways',   color: '#546579', defaultOn: true },
  { key: 'roads',     label: 'roads',       color: '#d1d1d6', defaultOn: true },
  { key: 'railways',  label: 'railways',    color: '#595961', defaultOn: true },
  { key: 'paths',     label: 'paths',       color: '#b3a68c', defaultOn: true },
];

export const WORLD_LAYER_KEYS: string[] = WORLD_LAYERS.map((l) => l.key);

/*
The height between 2 ground layers, and between the ground quad (y 0) and the lowest one, in world units (not meters:
the depth precision is in world units, metersToNorm changes with the bounds; about 9 cm at the default bounds, 1 unit is
about 900 m). The smallest gap at which the layers don't z-fight in view, with MapWorld's camera: near 0.01, far 10,
fov 1, an orbit distance up to 2 and a pitch from .2 to 1.4. 2 planes dy apart, seen from the height h at the distance d,
are dy * d / h apart along the view ray. A depth step of e (in distance) fights below dy = e * h / d.
-the forward pass (sceneColorRT): depth24plus, not reversed, e = 2^-24 * d^2 / near, so dy = 2^-24 * d * h / near. d * h
 is at most about 12 in view (the top of the screen at a pitch of about .63, at the far plane): 7e-5
-the gBuffer: depth32float reversed, e is about 2^-24 * d: dy about 1e-7, negligible
-both: the rasterizer snaps the vertices to 1/256 pixel, each mesh on its own, which tilts the depth plane of a triangle
 by up to the snap. dy = d * (fov / pixel height) / 256: 3.6e-5 at the far plane at 1080 pixels high
The forward pass bounds it: 7e-5, with a margin. With a reversed depth32float forward pass it could be about 5e-5
 */
export const LAYER_GAP = 1e-4;

const LAYER_Y: Record<string, number> = Object.fromEntries(WORLD_LAYERS.map((l, i) => [l.key, (i + 1) * LAYER_GAP]));

// How far (world units) a ground layer may be off the ground where it is draped on the terrain without the cuts of the
// grid (see TerrainGrid.drapeTriangles): twice this. A quarter of the gap, so 2 layers stay half of it apart
const DRAPE_TOLERANCE = LAYER_GAP / 8;

/** The height of the top ground layer (paths), in world units: the cars stand on it. */
export const GROUND_TOP_Y = WORLD_LAYERS.length * LAYER_GAP;

/** Building output of one tile (engine space, stride-4 vertices). */
export type WorldBuildingResult = {
  /**
   * Per vertex x, y, z and the feature id (or the palette entry of a tree, see worldTrees.ts) as unorm16 (0 to 65535):
   * position * quantization[0..3] + quantization[4..7] with the vertex read as 0 to 1, see quantizeVertices.
   */
  vertices: Uint16Array;
  /** The scale (xyzw) and the offset (xyzw) of the vertices, the uniforms positionScale and positionOffset. */
  quantization: number[];
  /** Per vertex the lightmap chart (vec2), see buildBuildings. */
  lightmap: Float32Array;
  /**
   * The vertex buffer: vertices and lightmap interleaved, 16 bytes per vertex (unorm16x4, float32x2), made here so the
   * main thread uploads it as it is.
   */
  interleaved: ArrayBuffer;
  /** The triangles, 3 vertex indices each: 16 bit up to 65536 vertices. */
  indices: Uint16Array | Uint32Array;
  /** The ray tracing data of the buildings, null without buildings. See CityScene. */
  trace: CityTraceBlock | null;
  buildings: BuildingInfo[];
  min: [number, number, number];
  max: [number, number, number];
  /** The size of the wall atlas (u, v) in world units, null without walls. */
  atlasSize: [number, number] | null;
  /** The map tile (x, y) the wall atlas is of: the uniform wallTile of the gBuffer, see getWallSurfaceId. */
  wallTile: [number, number];
};

/**
 * One ground layer of one tile: the vertices, floats (3 or 4) each, and the triangles, 3 vertex indices each. xyz, on
 * the terrain with the encoded normal of the ground in .w (see encodeTerrainNormal).
 */
export type WorldTerrainLayer = {
  vertices: Float32Array;
  indices: Uint32Array;
  floats: number;
};

export type WorldTileResult = {
  /** Per-layer indexed meshes (only when terrain was requested). */
  terrain?: Record<string, WorldTerrainLayer>;
  /** The ground mesh under the layers (only with the terrain and heights), see TerrainGrid.buildMesh. */
  ground?: WorldTerrainLayer;
  /**
   * The center lines of the roads cars drive on (only with the terrain), see packRoad: per line its point count, width
   * (m), speed (m/s) and oneway (1: only from the first point to the last), then the points in map (x, north) coords.
   */
  roads?: Float32Array;
  /** Extruded buildings (only when buildings were requested). */
  buildings?: WorldBuildingResult;
};

// --- OpenMapTiles classification ---------------------------------------------

// Line widths (meters) per OpenMapTiles `transportation` class.
const ROAD_WIDTHS: Record<string, number> = {
  motorway: 14, trunk: 12, primary: 11, secondary: 9, tertiary: 7,
  minor: 5.5, service: 4, track: 3, raceway: 8, busway: 6, bus_guideway: 4,
};
const RAIL_WIDTHS: Record<string, number> = { rail: 3, transit: 2.5 };
// The speed of the cars (m/s) per `transportation` class they drive on. A class without one has no cars.
const ROAD_SPEEDS: Record<string, number> = {
  motorway: 27, trunk: 22, primary: 14, secondary: 12, tertiary: 11, minor: 8, service: 5, busway: 10,
};
// Values written per road line before its points, see WorldTileResult.roads.
export const ROAD_HEADER_FLOATS = 4;
const WATERWAY_WIDTHS: Record<string, number> = { river: 10, canal: 8, stream: 3, drain: 1.5, ditch: 1.5 };
const AEROWAY_WIDTHS: Record<string, number> = { runway: 45, taxiway: 18 };

const URBAN_LANDUSE = new Set([
  'residential', 'commercial', 'industrial', 'retail', 'garages', 'railway', 'military',
  'school', 'university', 'college', 'kindergarten', 'hospital', 'library', 'bus_station',
  'suburb', 'quarter', 'neighbourhood', 'stadium', 'theme_park', 'zoo',
]);
const GREEN_LANDUSE = new Set(['cemetery', 'pitch', 'playground', 'track']);
const GREEN_PARKS = new Set(['park', 'garden', 'dog_park', 'golf_course', 'recreation_ground']);

type Props = Record<string, string | number | boolean>;
type FillClass = { key: string };
type LineClass = { key: string; width: number };

function str(v: unknown): string {
  return v === undefined || v === null ? '' : String(v);
}

function classifyFill(layer: string, p: Props): FillClass | null {
  const cls = str(p.class);
  if (p.brunnel === 'tunnel') return null;
  switch (layer) {
    case 'water':
      return { key: 'water' };
    case 'landcover':
      if (cls === 'wood') return { key: 'forest' };
      if (cls === 'grass' || cls === 'wetland') return { key: 'green' };
      if (cls === 'farmland') return { key: 'farmland' };
      if (cls === 'sand') return { key: 'sand' };
      return null; // ice / rock: rare in capitals, left to the ground colour
    case 'landuse':
      if (GREEN_LANDUSE.has(cls)) return { key: 'green' };
      if (URBAN_LANDUSE.has(cls)) return { key: 'urban' };
      return null;
    case 'park':
      // Only city parks: national parks / protected areas are huge and would
      // tint whole districts.
      return GREEN_PARKS.has(cls) ? { key: 'green' } : null;
    case 'transportation':
      // Area features (pedestrian squares, piers, bridge decks).
      if (cls === 'rail' || cls === 'transit' || cls === 'ferry' || cls === 'aerialway') return null;
      return { key: cls === 'path' ? 'paths' : 'roads' };
    case 'aeroway':
      if (cls === 'aerodrome') return null; // whole-airport outline
      return { key: 'roads' };
    default:
      return null;
  }
}

function classifyLine(layer: string, p: Props): LineClass | null {
  const cls = str(p.class);
  if (p.brunnel === 'tunnel') return null;
  switch (layer) {
    case 'waterway':
      return { key: 'waterways', width: WATERWAY_WIDTHS[cls] ?? 3 };
    case 'transportation': {
      if (RAIL_WIDTHS[cls] !== undefined) return { key: 'railways', width: RAIL_WIDTHS[cls] };
      if (cls === 'path') return { key: 'paths', width: str(p.subclass) === 'cycleway' ? 2 : 1.6 };
      const w = ROAD_WIDTHS[cls];
      if (w === undefined) return null; // ferry, aerialway, …
      return { key: 'roads', width: p.ramp === 1 || p.ramp === true ? w * 0.65 : w };
    }
    case 'aeroway': {
      const w = AEROWAY_WIDTHS[cls];
      return w === undefined ? null : { key: 'roads', width: w };
    }
    default:
      return null;
  }
}

const FILL_LAYERS = ['water', 'landcover', 'landuse', 'park', 'transportation', 'aeroway'];
const LINE_LAYERS = ['waterway', 'transportation', 'aeroway'];

// --- tile build ----------------------------------------------------------------

/**
 * Decode an OpenMapTiles vector tile and build the requested geometry,
 * projected relative to `bounds` (the shared {@link MapProjection} frame).
 * heights: the ground of the tile (m), see ElevationService.getTileGrid. Null: the flat ground at 0.
 */
export function buildWorldTile(
  bytes: Uint8Array,
  tile: TileCoord,
  bounds: LatLngBounds,
  wantTerrain: boolean,
  wantBuildings: boolean,
  wantTrees = false,
  heights: Float32Array | null = null,
): WorldTileResult {
  const scratch = tileScratch ??= new TileScratch();
  const vt = new VectorTile(new PbfReader(bytes));
  const proj = new MapProjection();
  proj.setBounds(bounds);
  const metersToNorm = proj.metersToNorm;
  const terrain = heights ? createTerrainGrid(tile, proj, heights) : null;

  // Tile-local MVT point -> projected map (x, north) coords: tileToLatLon and MapProjection.project (the same sums, in
  // the same order), without the object and the array they return per point. Most points are on the whole numbers of
  // the tile's extent: the north of each is worked out once (an atan and a sinh).
  const tileCount = Math.pow(2, tile.z);
  const { lngMid, latMid, cosLat, scale } = proj;
  let cacheExtent = 0;
  let northCache = new Float64Array(0);
  const toMap = (pt: TilePoint, extent: number, out: number[]): void => {
    const lon = ((tile.x + pt.x / extent) / tileCount) * 360 - 180;
    const py = pt.y;
    let north: number;
    if (Number.isInteger(py) && py >= 0 && py <= extent) {
      if (extent !== cacheExtent) {
        cacheExtent = extent;
        northCache = new Float64Array(extent + 1).fill(NaN);
      }
      north = northCache[py];
      if (north !== north) {
        const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + py / extent)) / tileCount)));
        north = northCache[py] = ((latRad * 180) / Math.PI - latMid) * scale;
      }
    } else {
      const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + py / extent)) / tileCount)));
      north = ((latRad * 180) / Math.PI - latMid) * scale;
    }
    out.push((lon - lngMid) * cosLat * scale, north);
  };

  const result: WorldTileResult = {};

  if (wantTerrain) {
    // Indexed like the buildings: earcut already returns indices, so a polygon point is one vertex instead of one per
    // triangle corner (about 3x fewer vertices), and a ribbon segment is 4 vertices instead of 6. Flat (x, north) first:
    // on the terrain the triangles are cut by the ground grid at the end (see TerrainGrid.drapeTriangles)
    const acc = scratch.flat;
    for (const key of WORLD_LAYER_KEYS) (acc[key] ??= new FlatMesh()).reset();

    for (const layerName of FILL_LAYERS) {
      const layer = vt.layers[layerName];
      if (!layer) continue;
      const extent = layer.extent || 4096;
      for (let i = 0; i < layer.length; i++) {
        const f = layer.feature(i);
        if (f.type !== 3) continue;
        const c = classifyFill(layerName, f.properties as Props);
        if (!c) continue;
        for (const poly of classifyRings(f.loadGeometry())) {
          const flat = polygonToMap(poly, extent, toMap);
          if (!flat) continue;
          const tris = earcut(flat.coords, flat.holes.length ? flat.holes : undefined, 2);
          if (tris.length === 0) continue;
          const out = acc[c.key];
          const first = out.pointCount;
          const coords = flat.coords;
          for (let k = 0; k < coords.length; k += 2) out.pushPoint(coords[k], coords[k + 1]);
          for (let t = 0; t < tris.length; t++) out.pushIndex(first + tris[t]);
        }
      }
    }

    const line: number[] = [];
    const roads: number[] = [];
    for (const layerName of LINE_LAYERS) {
      const layer = vt.layers[layerName];
      if (!layer) continue;
      const extent = layer.extent || 4096;
      for (let i = 0; i < layer.length; i++) {
        const f = layer.feature(i);
        if (f.type !== 2) continue;
        const c = classifyLine(layerName, f.properties as Props);
        if (!c) continue;
        const halfW = c.width * 0.5 * metersToNorm;
        const props = f.properties as Props;
        const speed = layerName === 'transportation' && c.key === 'roads' ? ROAD_SPEEDS[str(props.class)] : undefined;
        // OpenMapTiles: 1 one way along the line, -1 against it
        const oneway = Number(props.oneway) || 0;
        for (const ring of f.loadGeometry()) {
          if (ring.length < 2) continue;
          for (const seg of clipLineToTile(ring, extent)) {
            line.length = 0;
            for (const pt of seg) toMap(pt, extent, line);
            addRibbon(acc[c.key], line, halfW, scratch.ribbon);
            if (speed !== undefined) packRoad(roads, line, c.width, speed, oneway);
          }
        }
      }
    }

    const layers: Record<string, WorldTerrainLayer> = {};
    for (const key of WORLD_LAYER_KEYS) {
      const coords = acc[key].coordsView, indices = acc[key].indicesView;
      const y = LAYER_Y[key];
      if (terrain) {
        const out = scratch.draped;
        out.reset();
        terrain.drapeTriangles(coords, indices, y, DRAPE_TOLERANCE, !FLAT_NORMAL_LAYERS.has(key), out);
        layers[key] = { ...out.finish(), floats: GRID_VERTEX_FLOATS };
      } else {
        const vertices = new Float32Array(coords.length / 2 * 3);
        for (let k = 0, o = 0; k < coords.length; k += 2, o += 3) {
          vertices[o] = coords[k]; vertices[o + 1] = y; vertices[o + 2] = -coords[k + 1];
        }
        layers[key] = { vertices, indices: indices.slice(), floats: 3 };
      }
    }
    result.terrain = layers;
    if (terrain) result.ground = { ...terrain.buildMesh(), floats: GRID_VERTEX_FLOATS };
    result.roads = new Float32Array(roads);
  }

  if (wantBuildings) {
    result.buildings = buildBuildings(vt, tile, proj, toMap, wantTrees, terrain);
  }

  return result;
}

/**
 * A ground layer before the terrain: the points (x, north) and the triangles, 3 point indices each. In typed arrays that
 * grow by doubling and are kept for the next tile (see GridMesh for why): as number[] that grew by push they were new
 * memory for every tile.
 */
class FlatMesh {
  private coords = new Float64Array(8192);
  private indices = new Uint32Array(8192);
  private coordCount = 0;
  private indexCount = 0;

  /** Empty again, the arrays stay. */
  public reset(): void {
    this.coordCount = 0;
    this.indexCount = 0;
  }

  public get pointCount(): number {
    return this.coordCount / 2;
  }

  /** x, north per point. */
  public get coordsView(): Float64Array {
    return this.coords.subarray(0, this.coordCount);
  }

  public get indicesView(): Uint32Array {
    return this.indices.subarray(0, this.indexCount);
  }

  /** Returns the index of the point. */
  public pushPoint(x: number, y: number): number {
    const o = this.coordCount;
    if (o + 2 > this.coords.length) {
      const grown = new Float64Array(this.coords.length * 2);
      grown.set(this.coords);
      this.coords = grown;
    }
    this.coords[o] = x;
    this.coords[o + 1] = y;
    this.coordCount = o + 2;
    return o / 2;
  }

  public pushIndex(index: number): void {
    if (this.indexCount === this.indices.length) {
      const grown = new Uint32Array(this.indices.length * 2);
      grown.set(this.indices);
      this.indices = grown;
    }
    this.indices[this.indexCount++] = index;
  }
}

/**
 * What buildWorldTile works in, kept from tile to tile: a worker builds one tile at a time. Made on the first tile, so
 * the main thread (which imports this file for its types and createTerrainGrid) has none.
 */
class TileScratch {
  // the flat layers by key
  public readonly flat: Record<string, FlatMesh> = {};
  // a layer on the terrain, copied out at its size when it is done
  public readonly draped = new GridMesh();
  public readonly vertices = new BuildingVertices();
  public readonly indices = new IndexList();
  public readonly ribbon = new RibbonScratch();
}

let tileScratch: TileScratch | null = null;

/** The ground grid of the tile, from its heights in m with their ring (ElevationService.getTileGrid). Also TerrainHeightField. */
export function createTerrainGrid(tile: TileCoord, proj: MapProjection, heightsM: Float32Array): TerrainGrid {
  // the same corners as getTileGrid: the edges of 2 neighbours are at the same lat/lng
  const sw = tileToLatLon(0, 1, 1, tile), ne = tileToLatLon(1, 0, 1, tile);
  const [x0, y0] = proj.project(sw.lon, sw.lat);
  const [x1, y1] = proj.project(ne.lon, ne.lat);
  const heights = new Float32Array(heightsM.length);
  for (let k = 0; k < heights.length; k++) heights[k] = heightsM[k] * proj.metersToNorm;
  return new TerrainGrid(x0, y0, x1, y1, heights);
}

/**
 * A road center line into roads (see WorldTileResult.roads), without repeated points. oneway -1 is reversed, so a one
 * way road always runs from its first point to its last.
 */
function packRoad(roads: number[], line: number[], width: number, speed: number, oneway: number): void {
  const header = roads.length;
  roads.push(0, width, speed, oneway !== 0 ? 1 : 0);
  let count = 0;
  const n = line.length / 2;
  for (let k = 0; k < n; k++) {
    const i = oneway < 0 ? n - 1 - k : k;
    const x = line[i * 2], y = line[i * 2 + 1];
    if (count > 0 && roads[roads.length - 2] === x && roads[roads.length - 1] === y) continue;
    roads.push(x, y);
    count++;
  }
  if (count < 2) {
    roads.length = header;
    return;
  }
  roads[header] = count;
}

/** Clip a polygon (outer + holes) to the tile and project it; null when degenerate. */
function polygonToMap(
  poly: TilePoint[][],
  extent: number,
  toMap: (pt: TilePoint, extent: number, out: number[]) => void,
): { coords: number[]; holes: number[]; ringStarts: number[] } | null {
  if (!poly.length || !poly[0]) return null;
  const coords: number[] = [];
  const holes: number[] = [];
  const ringStarts: number[] = [];
  for (let r = 0; r < poly.length; r++) {
    const clipped = clipRingToTile(poly[r], extent);
    if (clipped.length < 3) {
      if (r === 0) return null;
      continue;
    }
    const start = coords.length / 2;
    for (const pt of clipped) toMap(pt, extent, coords);
    // Drop a duplicate closing vertex.
    const n = coords.length;
    if (n / 2 - start > 3 && coords[start * 2] === coords[n - 2] && coords[start * 2 + 1] === coords[n - 1]) {
      coords.length = n - 2;
    }
    if (r > 0) holes.push(start);
    ringStarts.push(start);
  }
  return { coords, holes, ringStarts };
}

// A join whose miter is longer than this times the half width (a turn of more than 120 degrees) isn't mitered: the
// segments end square there, each on its own pair of vertices, so a hairpin doesn't get a long spike.
const MITER_LIMIT = 2;

/**
 * Flat ribbon along a projected polyline (same winding as osmGeometryCore): 2 points (left, right) per point, shared
 * by the segments on both sides of it, offset along the miter so the ribbon keeps its width around the bend. A closed
 * line (first point = last) is mitered where it closes too.
 */
function addRibbon(out: FlatMesh, line: number[], halfW: number, scratch: RibbonScratch): void {
  scratch.reserve(line.length / 2);
  const { xs, ys, dxs, dys, ends, starts } = scratch;
  // the points without repeats (a zero length segment has no direction)
  let n = 0;
  for (let i = 0; i < line.length; i += 2) {
    const x = line[i], y = line[i + 1];
    if (n > 0 && xs[n - 1] === x && ys[n - 1] === y) continue;
    xs[n] = x; ys[n] = y;
    n++;
  }
  if (n < 2) return;
  const closed = n > 3 && xs[0] === xs[n - 1] && ys[0] === ys[n - 1];
  if (closed) n--; // the last point is the first
  const segs = closed ? n : n - 1;

  // the unit direction of segment i, from point i to point i + 1
  for (let i = 0; i < segs; i++) {
    const j = (i + 1) % n;
    const dx = xs[j] - xs[i], dy = ys[j] - ys[i];
    const len = hypot2(dx, dy);
    dxs[i] = dx / len; dys[i] = dy / len;
  }

  // per point the pair the segment before it ends on and the pair the segment after it starts on: the same pair when
  // mitered. A pair: the points p + o (left) and p - o (right), by the index of the left one
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    const prev = closed ? (i + segs - 1) % segs : i - 1;
    const next = closed || i < segs ? i : -1;
    if (prev < 0 || next < 0) {
      // an end of an open line: square to its segment
      const s = prev < 0 ? next : prev;
      const ox = -dys[s] * halfW, oy = dxs[s] * halfW;
      ends[i] = starts[i] = out.pushPoint(x + ox, y + oy);
      out.pushPoint(x - ox, y - oy);
      continue;
    }
    // the left normals of both segments, the miter halfway between them
    const n0x = -dys[prev], n0y = dxs[prev];
    const n1x = -dys[next], n1y = dxs[next];
    let mx = n0x + n1x, my = n0y + n1y;
    const ml = hypot2(mx, my);
    // cos of the angle between the miter and the normals: the miter is halfW / cos long
    const cos = ml > 1e-9 ? (mx * n1x + my * n1y) / ml : 0;
    if (cos >= 1 / MITER_LIMIT) {
      const s = halfW / (cos * ml);
      mx *= s; my *= s;
      ends[i] = starts[i] = out.pushPoint(x + mx, y + my);
      out.pushPoint(x - mx, y - my);
    } else {
      const ax = n0x * halfW, ay = n0y * halfW;
      ends[i] = out.pushPoint(x + ax, y + ay);
      out.pushPoint(x - ax, y - ay);
      const bx = n1x * halfW, by = n1y * halfW;
      starts[i] = out.pushPoint(x + bx, y + by);
      out.pushPoint(x - bx, y - by);
    }
  }

  for (let i = 0; i < segs; i++) {
    const a = starts[i], b = ends[(i + 1) % n];
    // (a left, b right, b left), (a left, a right, b right)
    out.pushIndex(a);
    out.pushIndex(b + 1);
    out.pushIndex(b);
    out.pushIndex(a);
    out.pushIndex(a + 1);
    out.pushIndex(b + 1);
  }
}

/** The arrays addRibbon works in, per point of the line: kept between the lines (a tile has thousands). */
class RibbonScratch {
  public xs = new Float64Array(0);
  public ys = new Float64Array(0);
  // the unit direction of each segment
  public dxs = new Float64Array(0);
  public dys = new Float64Array(0);
  public ends = new Int32Array(0);
  public starts = new Int32Array(0);

  /** Room for a line of points. */
  public reserve(points: number): void {
    if (this.xs.length >= points) return;
    const size = Math.max(points, this.xs.length * 2, 256);
    this.xs = new Float64Array(size);
    this.ys = new Float64Array(size);
    this.dxs = new Float64Array(size);
    this.dys = new Float64Array(size);
    this.ends = new Int32Array(size);
    this.starts = new Int32Array(size);
  }
}

// --- buildings -----------------------------------------------------------------

// Exported for the headless test of the city lightmap.
// Walls of 2 parts this close (in m) touch: the one facing the other is hidden, see findWallBottoms
const WALL_TOUCH_M = .05;

export function buildBuildings(
  vt: VectorTile,
  tile: TileCoord,
  proj: MapProjection,
  toMap: (pt: TilePoint, extent: number, out: number[]) => void,
  wantTrees = false,
  terrain: TerrainGrid | null = null,
): WorldBuildingResult {
  // The vertices: position and id, and the lightmap chart, see BuildingVertices
  const scratch = tileScratch ??= new TileScratch();
  const verts = scratch.vertices;
  verts.reset();
  const buildings: BuildingInfo[] = [];
  const metersToNorm = proj.metersToNorm;

  // The footprints first, so the walls can be laid out in the lightmap atlas of the tile before the vertices are made.
  // id: the vertex .w, the feature id of a building or the palette entry of a tree (see worldTrees.ts). roof: false on
  // a trunk, which its crown covers
  type Footprint = BuildingPolygon & { holes: number[]; id: number; roof: boolean };
  const footprints: Footprint[] = [];

  const layer = vt.layers['building'];
  if (layer) {
    const extent = layer.extent || 4096;
    const parts: BuildingPart[] = [];
    for (let i = 0; i < layer.length; i++) {
      const f = layer.feature(i);
      if (f.type !== 3) continue;
      const p = f.properties as Props;
      // Outlines whose volume is described by building:part features.
      if (p.hide_3d === true || p.hide_3d === 'true') continue;

      const { base, top } = buildingHeights(p, tile, i);

      for (const poly of classifyRings(f.loadGeometry())) {
        const flat = polygonToMap(poly, extent, toMap);
        if (!flat) continue;
        const { coords, holes, ringStarts } = flat;

        // Normalise winding: outer ring CCW, holes CW (in x/north space), so
        // the wall quads below face outward (into courtyards for holes).
        const ringEnds = ringStarts.slice(1).concat(coords.length / 2);
        let area = 0;
        for (let r = 0; r < ringStarts.length; r++) {
          const s = ringStarts[r], e = ringEnds[r];
          const a = signedArea(coords, s, e);
          if (r === 0 ? a < 0 : a > 0) reverseRing(coords, s, e);
          area += Math.abs(a) * (r === 0 ? 1 : -1);
        }
        if (Math.abs(area) <= 0) continue;
        parts.push({ coords, holes, ringStarts, ringEnds, area, base, top });
      }
    }

    // One tile unit in map units: the tolerance of mergeCoincidentParts
    const unit: number[] = [];
    toMap({ x: 0, y: 0 } as TilePoint, extent, unit);
    toMap({ x: 1, y: 1 } as TilePoint, extent, unit);
    const tolerance = Math.max(Math.abs(unit[2] - unit[0]), Math.abs(unit[3] - unit[1]));

    const merged = mergeCoincidentParts(parts, tolerance);
    // the ground under each part (world units): base and top are over it
    const grounds = terrain ? partGrounds(merged, terrain) : new Float64Array(merged.length);
    merged.forEach((part, pi) => {
      const { coords, holes, ringStarts, ringEnds, area, base, top } = part;
      const ground = grounds[pi];
      const id = buildings.length <= MAX_FEATURE_ID ? buildings.length : NO_FEATURE_ID;
      footprints.push({
        coords, holes, ringStarts, ringEnds, base: ground + base * metersToNorm, top: ground + top * metersToNorm,
        roofChart: getRoofChart(ground / metersToNorm + top), id, roof: true,
      });

      if (id !== NO_FEATURE_ID) {
        // Centroid of the outer ring, back to lat/lng (used as the address key).
        const s = ringStarts[0], e = ringEnds[0];
        let cx = 0, cy = 0;
        for (let j = s; j < e; j++) { cx += coords[j * 2]; cy += coords[j * 2 + 1]; }
        cx /= e - s; cy /= e - s;
        const { lat, lng } = proj.worldToLatLng(cx, -cy);
        buildings.push({
          id: `${lat.toFixed(6)}, ${lng.toFixed(6)}`,
          year: -1, // not in OpenMapTiles; hides the row in the info panel
          groundArea: Math.abs(area) / (metersToNorm * metersToNorm),
          status: '',
          roofType: '',
          height: top,
        });
      }
    });
  }

  // The tallest first: seen from above they are in front, so the depth test drops the covered roofs and walls of the
  // lower parts after them before they are shaded (a tower is a stack of parts with roofs inside the parts over them)
  footprints.sort((a, b) => b.top - a.top);

  // The trees: boxes like the buildings, in the same mesh and wall atlas, see worldTrees.ts. The ray tracing block has a
  // record of its own per tree (trunk and crown), see CityTraceBlock: the footprints from treeStart are not in it.
  const treeStart = footprints.length;
  const traceTrees: TraceTree[] = [];
  if (wantTrees) {
    const boxes = buildTrees(vt, toMap, polygonToMap, metersToNorm);
    // the ground at the center of each tree (its trunk and crown are a pair of boxes)
    const grounds = boxes.map((box) => terrain ? terrain.heightAt(box.x, box.y) : 0);
    for (let i = 0; i + 1 < boxes.length; i += 2) {
      const trunk = boxes[i], crown = boxes[i + 1], ground = grounds[i];
      traceTrees.push({
        x: crown.x, y: crown.y, base: ground, angle: crown.angle, trunkHalf: trunk.half, crownHalf: crown.half,
        crownBase: ground + crown.base * metersToNorm, top: ground + crown.top * metersToNorm,
      });
    }
    boxes.forEach((box, i) => {
      const ground = grounds[i];
      footprints.push({
        coords: box.coords, holes: [], ringStarts: [0], ringEnds: [4], base: ground + box.base * metersToNorm,
        top: ground + box.top * metersToNorm, roofChart: getRoofChart(ground / metersToNorm + box.top),
        id: TREE_MATERIAL_BASE_ID - box.material, roof: box.material !== 0,
      });
    });
  }

  // The walls inside other parts, see findWallBottoms. Per edge the bottom of its wall, NaN when it isn't drawn. Per
  // corner the height of its bottom vertex: the bottom of the wall that starts there (whose attribute it has), else the
  // one of the wall that ends there, else the base. A wall that ends at a corner with another bottom gets a bottom
  // vertex of its own there (its end isn't the provoking vertex, its attribute isn't read)
  const wallBottoms = findWallBottoms(footprints, treeStart, WALL_TOUCH_M * metersToNorm);
  const cornerBottoms: Float32Array[] = [];
  const edgeBottoms: Float32Array[] = [];
  footprints.forEach((fp, fi) => {
    const walls = wallBottoms[fi];
    const corners = new Float32Array(walls.length);
    const edges = new Float32Array(walls.length);
    const drawn = (b: number) => !Number.isNaN(b) && b < fp.top;
    for (let r = 0; r < fp.ringStarts.length; r++) {
      const s = fp.ringStarts[r], n = fp.ringEnds[r] - s;
      for (let j = 0; j < n; j++) {
        const k = s + j, prev = s + ((j + n - 1) % n);
        const a = walls[prev], b = walls[k];
        corners[k] = drawn(b) ? b : (drawn(a) ? a : fp.base);
        edges[k] = drawn(b) ? b : NaN;
      }
    }
    cornerBottoms.push(corners);
    edgeBottoms.push(edges);
  });
  const layout = layoutWalls(footprints, tile.x, tile.y, metersToNorm, edgeBottoms);
  // Indexed: the vertex shader of the gBuffer, the shadow maps and the picking runs once per vertex instead of once
  // per corner of every triangle. The normals come from the derivatives of the position.
  // 2 vertices per corner of a footprint, which the walls and the roof share: the bottom one has the lightmap attribute of
  // the wall that starts there, the top one that of the roof. A wall is drawn as (a0, b0, b1), (a0, b1, a1), so its
  // bottom start a0 is the first vertex of its 2 triangles: the gBuffer reads the attribute flat from it (the first
  // vertex is the provoking one) and adds the distance along the wall and the height to it, see mapGBuffer.wgsl. A
  // roof triangle has only top vertices.
  const indices = scratch.indices;
  indices.length = 0;
  // the largest wall atlas position, for the check against what the gBuffer can store, and per axis (the size of the
  // atlas, for the wallAtlas view)
  let atlasMax = 0;
  const atlasSize: [number, number] = [0, 0];
  let walls = false;

  footprints.forEach((fp, fi) => {
    const { coords, holes, ringStarts, ringEnds, base: baseN, top: topN, roofChart, id, roof } = fp;
    const atlas = layout.edgeAtlas[fi];
    const corners = cornerBottoms[fi], edges = edgeBottoms[fi];
    const height = topN - baseN;
    const tris = roof ? earcut(coords, holes.length ? holes : undefined, 2) : [];
    const hasRoof = tris.length > 0;

    // the bottom vertex of point k is first + 2k, its top vertex first + 2k + 1
    const first = verts.count;
    // written in place, see BuildingVertices: x, the height, -north and the id, and the lightmap chart
    verts.reserve(coords.length);
    const positions = verts.positions, charts = verts.lightmap;
    let vertex = first;
    for (let k = 0; k < coords.length; k += 2) {
      const x = coords[k], y = coords[k + 1];
      // the atlas v at the bottom vertex, which can be over the base
      const bottom = corners[k / 2];
      let o = vertex * VERTEX_FLOATS;
      positions[o] = x; positions[o + 1] = bottom; positions[o + 2] = -y; positions[o + 3] = id;
      charts[vertex * 2] = atlas[k];
      charts[vertex * 2 + 1] = atlas[k + 1] + bottom - baseN;
      vertex++;
      o += VERTEX_FLOATS;
      positions[o] = x; positions[o + 1] = topN; positions[o + 2] = -y; positions[o + 3] = id;
      // on a roof (roof chart, -1). Without a roof the top vertex is never the first of a triangle: its attribute is
      // not read
      charts[vertex * 2] = hasRoof ? roofChart : atlas[k];
      charts[vertex * 2 + 1] = hasRoof ? -1 : atlas[k + 1];
      vertex++;
    }
    verts.count = vertex;

    // Roof (earcut output is CCW -> faces up), on the top vertices
    for (let t = 0; t < tris.length; t++) indices.push(first + tris[t] * 2 + 1);

    // Walls for every ring
    for (let r = 0; r < ringStarts.length; r++) {
      const s = ringStarts[r], e = ringEnds[r];
      const n = e - s;
      for (let j = 0; j < n; j++) {
        const pa = s + j, pb = s + ((j + 1) % n);
        const ax = coords[pa * 2], ay = coords[pa * 2 + 1];
        const bx = coords[pb * 2], by = coords[pb * 2 + 1];
        if (ax === bx && ay === by) continue;
        // inside other parts
        if (Number.isNaN(edges[pa])) continue;
        const a0 = first + pa * 2, b1 = first + pb * 2 + 1;
        let b0 = b1 - 1;
        if (edges[pa] !== corners[pb]) b0 = verts.push(bx, by, edges[pa], id, atlas[pb * 2], atlas[pb * 2 + 1]);
        indices.push(a0);
        indices.push(b0);
        indices.push(b1);
        indices.push(a0);
        indices.push(b1);
        indices.push(a0 + 1);
        const u1 = atlas[pa * 2] + hypot2(bx - ax, by - ay), v1 = atlas[pa * 2 + 1] + height;
        atlasMax = Math.max(atlasMax, u1, v1);
        atlasSize[0] = Math.max(atlasSize[0], u1);
        atlasSize[1] = Math.max(atlasSize[1], v1);
        walls = true;
      }
    }
  });

  if (atlasMax > GBUFFER_ATLAS_MAX * GBUFFER_ATLAS_STEP_M * metersToNorm) {
    console.warn(`[MapWorld] the wall atlas of tile ${tile.z}/${tile.x}/${tile.y} is ${(atlasMax / metersToNorm / 1000).toFixed(1)} km, ` +
      'over what the gBuffer stores: its far walls get the lighting of its edge. See GBUFFER_ATLAS_STEP_M');
  }
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  if (verts.count > 0) verts.bounds(min, max);
  // The ground plane of the ray tracing is the ground quad (y 0), the layers lie a few cm over it (see LAYER_GAP). On
  // the terrain the bases are on the ground already
  const buildingFootprints = footprints.slice(0, treeStart);
  const trace = buildTraceBlock(buildingFootprints, traceTrees, terrain ? null : 0, buildingFootprints.map((fp) => fp.id));
  const { vertices, quantization } = quantizeVertices(verts.positions, verts.count * VERTEX_FLOATS, min, max);
  const lightmapData = verts.lightmap.slice(0, verts.count * 2);
  return {
    vertices, quantization, lightmap: lightmapData, interleaved: interleaveBuildingVertices(vertices, lightmapData),
    indices: indices.finish(verts.count <= 65536), trace, buildings, min, max,
    atlasSize: walls ? atlasSize : null, wallTile: [layout.tileX, layout.tileY],
  };
}

/**
 * The vertices of the buildings of a tile while buildBuildings makes them, in typed arrays that grow by doubling (as
 * number[] with a closure that pushed them, each vertex took 6 numbers on the heap for its arguments: 10 MB of garbage
 * per tile, and passed to a method 3 MB still: buildBuildings writes them in place after reserve), kept from tile to
 * tile (reset, see GridMesh for why). positions: x, y, z (engine space) and the id per vertex, as doubles: quantizeVertices rounds them. lightmap:
 * per vertex the lightmap chart, see SkyVisibilityLightmap: (atlas u, atlas v) of the start of a wall, (roof chart, -1)
 * on a roof (see getRoofChart). The tile of the wall atlas is the same for all, see wallTile.
 */
class BuildingVertices {
  public positions = new Float64Array(8192 * VERTEX_FLOATS);
  public lightmap = new Float32Array(8192 * 2);
  public count = 0;

  /** Empty again, the arrays stay. */
  public reset(): void {
    this.count = 0;
  }

  /** Room for this many more vertices: positions and lightmap can be written up to count + vertices after it. */
  public reserve(vertices: number): void {
    const needed = (this.count + vertices) * VERTEX_FLOATS;
    if (needed <= this.positions.length) return;
    let size = this.positions.length * 2;
    while (size < needed) size *= 2;
    const positions = new Float64Array(size);
    positions.set(this.positions);
    this.positions = positions;
    const lightmap = new Float32Array(size / VERTEX_FLOATS * 2);
    lightmap.set(this.lightmap);
    this.lightmap = lightmap;
  }

  /**
   * A vertex at map (x, north) and height h, with the lightmap chart (u, v). Returns its index. For the few vertices
   * that are not written in place.
   */
  public push(x: number, y: number, h: number, id: number, u: number, v: number): number {
    this.reserve(1);
    const index = this.count++;
    const p = this.positions, o = index * VERTEX_FLOATS;
    p[o] = x; p[o + 1] = h; p[o + 2] = -y; p[o + 3] = id;
    this.lightmap[index * 2] = u;
    this.lightmap[index * 2 + 1] = v;
    return index;
  }

  /** The box of the positions into min and max. */
  public bounds(min: number[], max: number[]): void {
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    const p = this.positions;
    for (let o = 0; o < this.count * VERTEX_FLOATS; o += VERTEX_FLOATS) {
      const x = p[o], y = p[o + 1], z = p[o + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    min[0] = minX; min[1] = minY; min[2] = minZ;
    max[0] = maxX; max[1] = maxY; max[2] = maxZ;
  }
}

/** Vertex indices in a typed array that grows by doubling. */
class IndexList {
  private data = new Uint32Array(8192 * 6);
  public length = 0;

  public push(index: number): void {
    if (this.length === this.data.length) {
      const data = new Uint32Array(this.data.length * 2);
      data.set(this.data);
      this.data = data;
    }
    this.data[this.length++] = index;
  }

  /** The indices in an array of their size, 16 bit when asked. */
  public finish(uint16: boolean): Uint16Array | Uint32Array {
    const used = this.data.subarray(0, this.length);
    return uint16 ? new Uint16Array(used) : used.slice();
  }
}

/** The vertex buffer of the buildings, see WorldBuildingResult.interleaved. */
function interleaveBuildingVertices(vertices: Uint16Array, lightmap: Float32Array): ArrayBuffer {
  const count = vertices.length / 4;
  const buffer = new ArrayBuffer(count * 16);
  const u16 = new Uint16Array(buffer), f32 = new Float32Array(buffer);
  for (let v = 0; v < count; v++) {
    u16[v * 8] = vertices[v * 4];
    u16[v * 8 + 1] = vertices[v * 4 + 1];
    u16[v * 8 + 2] = vertices[v * 4 + 2];
    u16[v * 8 + 3] = vertices[v * 4 + 3];
    f32[v * 4 + 2] = lightmap[v * 2];
    f32[v * 4 + 3] = lightmap[v * 2 + 1];
  }
  return buffer;
}

/**
 * The building vertices (x, y, z, feature id) as unorm16: xyz over the box of the mesh (a z14 tile is about 2.4 km
 * at the equator: 3.7 cm steps, half that the most a corner moves; the ray tracing has the exact footprints, a sky ray
 * starts MIN_DIST (~1.8 cm) off the surface), the id plus FEATURE_ID_BIAS. Decoded as position * scale + offset with
 * the unorm read as 0 to 1, the id rounded, see mapGBuffer.wgsl and shading.wgsl (QUANTIZED).
 */
function quantizeVertices(verts: Float64Array, length: number, min: number[], max: number[]): { vertices: Uint16Array; quantization: number[] } {
  const vertices = new Uint16Array(length);
  const scale = [0, 0, 0, 65535], offset = [min[0], min[1], min[2], -FEATURE_ID_BIAS];
  const toUnit = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const size = max[k] - min[k];
    // the unorm is read as 0 to 1
    scale[k] = size;
    toUnit[k] = size > 0 ? 65535 / size : 0;
  }
  for (let i = 0; i < length; i += 4) {
    for (let k = 0; k < 3; k++) vertices[i + k] = Math.round((verts[i + k] - min[k]) * toUnit[k]);
    vertices[i + 3] = verts[i + 3] + FEATURE_ID_BIAS;
  }
  return { vertices, quantization: [...scale, ...offset] };
}

/**
 * [base, top] in meters from OpenMapTiles' render_min_height / render_height,
 * falling back to a jittered default for footprints without height data.
 */
function buildingHeights(p: Props, tile: TileCoord, index: number): { base: number; top: number } {
  let top = Number(p.render_height);
  if (!isFinite(top) || top <= 0) {
    const rnd = hash01(tile.x * 73856093 ^ tile.y * 19349663 ^ index * 83492791);
    top = 8 * (0.9 + 0.2 * rnd);
  }
  let base = Number(p.render_min_height);
  if (!isFinite(base) || base < 0 || base >= top) base = 0;
  return { base, top };
}

/**
 * The ground (world units) under each building part: the lowest under its footprint, so no wall stops above the slope.
 * A part over the ground (render_min_height > 0, the upper part of a tower) stands on the ground of the part on the ground
 * around its center, so the parts of a building stack without a step. Without one, its own.
 */
function partGrounds(parts: BuildingPart[], terrain: TerrainGrid): Float64Array {
  const own = new Float64Array(parts.length);
  parts.forEach((p, i) => own[i] = terrain.minHeight(p.coords, p.ringStarts[0], p.ringEnds[0]));
  if (parts.every((p) => p.base <= 0)) return own;

  // the parts on the ground in a grid of buckets over the tile, by their boxes
  const B = 32;
  const bx = (x: number) => Math.min(Math.max(Math.floor((x - terrain.x0) / (terrain.x1 - terrain.x0) * B), 0), B - 1);
  const by = (y: number) => Math.min(Math.max(Math.floor((y - terrain.y0) / (terrain.y1 - terrain.y0) * B), 0), B - 1);
  const buckets: number[][] = Array.from({ length: B * B }, () => []);
  const boxes = new Float64Array(parts.length * 4);
  parts.forEach((p, i) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = p.ringStarts[0]; k < p.ringEnds[0]; k++) {
      const x = p.coords[k * 2], y = p.coords[k * 2 + 1];
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    boxes.set([x0, y0, x1, y1], i * 4);
    if (p.base > 0) return;
    for (let j = by(y0); j <= by(y1); j++) for (let k = bx(x0); k <= bx(x1); k++) buckets[j * B + k].push(i);
  });

  const grounds = Float64Array.from(own);
  parts.forEach((p, i) => {
    if (p.base <= 0) return;
    const s = p.ringStarts[0], e = p.ringEnds[0];
    let cx = 0, cy = 0;
    for (let k = s; k < e; k++) { cx += p.coords[k * 2]; cy += p.coords[k * 2 + 1]; }
    cx /= e - s; cy /= e - s;
    for (const q of buckets[by(cy) * B + bx(cx)]) {
      if (cx < boxes[q * 4] || cx > boxes[q * 4 + 2] || cy < boxes[q * 4 + 1] || cy > boxes[q * 4 + 3]) continue;
      if (!insideRing(parts[q].coords, parts[q].ringStarts[0], parts[q].ringEnds[0], cx, cy)) continue;
      grounds[i] = own[q];
      break;
    }
  });
  return grounds;
}

/** A building polygon before mergeCoincidentParts: map coords (rings normalised), base and top in meters. */
type BuildingPart = {
  coords: number[]; holes: number[]; ringStarts: number[]; ringEnds: number[]; area: number; base: number; top: number;
};

// Height ranges this close (m) touch, see mergeCoincidentParts
const PART_TOUCH_M = .01;

/**
 * The OpenMapTiles building layer has parts with the same footprint more than once: a feature with the same ring twice
 * (an exact copy with another start, or one shifted a tile unit), and parts of a tower whose height ranges overlap
 * (bands 4-8 m, 7-11 m, ... of the same outline, or a 0-5 m part inside a 0-20 m one). findWallBottoms drops one of
 * 2 coplanar walls, but a copy with the same top also has a coplanar roof, which z-fights. Parts whose rings match within tolerance (map units: every vertex within it of a vertex of the other, both ways)
 * and whose height ranges overlap or touch become one: the footprint of the tallest, from the lowest base. Stacks
 * with a gap between them stay apart.
 */
function mergeCoincidentParts(parts: BuildingPart[], tolerance: number): BuildingPart[] {
  const n = parts.length;
  const box = new Float64Array(n * 4);
  parts.forEach((p, i) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = 0; k < p.coords.length; k += 2) {
      const x = p.coords[k], y = p.coords[k + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    box.set([x0, y0, x1, y1], i * 4);
  });
  // every vertex of a near one of b
  const covered = (a: number[], b: number[]): boolean => {
    for (let i = 0; i < a.length; i += 2) {
      let near = false;
      for (let j = 0; j < b.length && !near; j += 2) {
        near = Math.abs(a[i] - b[j]) <= tolerance && Math.abs(a[i + 1] - b[j + 1]) <= tolerance;
      }
      if (!near) return false;
    }
    return true;
  };
  // union-find over the matching footprints, a sweep along x over the boxes
  const parent = new Int32Array(n).map((_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]];
    return i;
  };
  const order = [...parts.keys()].sort((a, b) => box[a * 4] - box[b * 4]);
  let matched = false;
  for (let oi = 0; oi < n; oi++) {
    const a = order[oi], A = parts[a];
    for (let oj = oi + 1; oj < n; oj++) {
      const b = order[oj], B = parts[b];
      if (box[b * 4] > box[a * 4] + tolerance) break;
      if (Math.abs(box[a * 4 + 1] - box[b * 4 + 1]) > tolerance || Math.abs(box[a * 4 + 2] - box[b * 4 + 2]) > tolerance ||
        Math.abs(box[a * 4 + 3] - box[b * 4 + 3]) > tolerance || A.ringStarts.length !== B.ringStarts.length) continue;
      if (!covered(A.coords, B.coords) || !covered(B.coords, A.coords)) continue;
      parent[find(a)] = find(b);
      matched = true;
    }
  }
  if (!matched) return parts;

  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    const group = groups.get(root);
    if (group) group.push(i);
    else groups.set(root, [i]);
  }
  const keep = new Uint8Array(n);
  for (const group of groups.values()) {
    if (group.length === 1) {
      keep[group[0]] = 1;
      continue;
    }
    // the overlapping or touching height ranges, from low to high: the tallest of each run, from the run's base
    group.sort((a, b) => parts[a].base - parts[b].base);
    let tallest = group[0], base = parts[tallest].base, top = parts[tallest].top;
    const close = () => {
      keep[tallest] = 1;
      parts[tallest].base = base;
    };
    for (let k = 1; k < group.length; k++) {
      const p = parts[group[k]];
      if (p.base > top + PART_TOUCH_M) {
        close();
        tallest = group[k];
        base = p.base;
        top = p.top;
      } else if (p.top > top) {
        tallest = group[k];
        top = p.top;
      }
    }
    close();
  }
  return parts.filter((_, i) => keep[i]);
}

function hash01(seed: number): number {
  let h = seed >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x45d9f3b) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

/** Signed area of ring [s, e) in a flat xy array (positive = CCW, y up). */
function signedArea(c: number[], s: number, e: number): number {
  let a = 0;
  for (let i = s, j = e - 1; i < e; j = i++) {
    a += (c[j * 2] - c[i * 2]) * (c[i * 2 + 1] + c[j * 2 + 1]);
  }
  return a / 2;
}

function reverseRing(c: number[], s: number, e: number): void {
  for (let i = s, j = e - 1; i < j; i++, j--) {
    const x = c[i * 2], y = c[i * 2 + 1];
    c[i * 2] = c[j * 2]; c[i * 2 + 1] = c[j * 2 + 1];
    c[j * 2] = x; c[j * 2 + 1] = y;
  }
}
