import earcut from 'earcut';
import { VectorTile } from '@mapbox/vector-tile';
import { PbfReader } from 'pbf';
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
import TerrainGrid, { encodeNormal, GRID_VERTEX_FLOATS, insideRing } from './terrainGrid';

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

/** The layers whose bridges are lifted over the ground (see buildBridges): their decks are in WorldTileResult.bridges. */
export const BRIDGE_LAYER_KEYS: string[] = ['roads', 'railways', 'paths'];

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
   * The bridge decks per ground layer (roads, railways, paths; only those with bridges): solid 3D meshes over the
   * ground, GRID_VERTEX_FLOATS per vertex with the normal in .w (encodeNormal), see buildBridges.
   */
  bridges?: Record<string, WorldTerrainLayer>;
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
      // Area features (pedestrian squares, piers, bridge decks: those are lifted, see buildBridges).
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
  const vt = new VectorTile(new PbfReader(bytes));
  const proj = new MapProjection();
  proj.setBounds(bounds);
  const metersToNorm = proj.metersToNorm;
  const terrain = heights ? createTerrainGrid(tile, proj, heights) : null;

  // Tile-local MVT point -> projected map (x, north) coords.
  const toMap = (pt: TilePoint, extent: number, out: number[]): void => {
    const { lat, lon } = tileToLatLon(pt.x, pt.y, extent, tile);
    const [x, y] = proj.project(lon, lat);
    out.push(x, y);
  };

  const result: WorldTileResult = {};

  // the bridges: their decks (with the terrain) and their boxes in the ray tracing of the buildings
  let decks: BridgeDecks | null = null;
  if (wantTerrain || wantBuildings) {
    const { lines, areas } = collectBridges(vt, toMap, metersToNorm);
    if (lines.length || areas.length) {
      // the tile's width in map units, for the ramps of the decks
      const corners: number[] = [];
      toMap({ x: 0, y: 0 } as TilePoint, 4096, corners);
      toMap({ x: 4096, y: 0 } as TilePoint, 4096, corners);
      decks = buildBridges(lines, areas, terrain, metersToNorm, Math.abs(corners[2] - corners[0]));
    }
  }

  if (wantTerrain) {
    // Indexed like the buildings: earcut already returns indices, so a polygon point is one vertex instead of one per
    // triangle corner (about 3x fewer vertices), and a ribbon segment is 4 vertices instead of 6. Flat (x, north) first:
    // on the terrain the triangles are cut by the ground grid at the end (see TerrainGrid.drapeTriangles)
    const acc: Record<string, FlatMesh> = {};
    for (const key of WORLD_LAYER_KEYS) acc[key] = { coords: [], indices: [] };

    for (const layerName of FILL_LAYERS) {
      const layer = vt.layers[layerName];
      if (!layer) continue;
      const extent = layer.extent || 4096;
      for (let i = 0; i < layer.length; i++) {
        const f = layer.feature(i);
        if (f.type !== 3) continue;
        const c = classifyFill(layerName, f.properties as Props);
        if (!c) continue;
        // a bridge isn't draped: its deck is lifted, see collectBridges
        if (isBridge(layerName, f.properties as Props)) continue;
        for (const poly of classifyRings(f.loadGeometry())) {
          const flat = polygonToMap(poly, extent, toMap);
          if (!flat) continue;
          const tris = earcut(flat.coords, flat.holes.length ? flat.holes : undefined, 2);
          if (tris.length === 0) continue;
          const out = acc[c.key];
          const first = out.coords.length / 2;
          const coords = flat.coords;
          for (let k = 0; k < coords.length; k++) out.coords.push(coords[k]);
          for (let t = 0; t < tris.length; t++) out.indices.push(first + tris[t]);
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
        // a bridge isn't draped: its deck is lifted, see collectBridges. The cars drive on it (on the ground under it)
        const bridge = isBridge(layerName, props);
        for (const ring of f.loadGeometry()) {
          if (ring.length < 2) continue;
          for (const seg of clipLineToTile(ring, extent)) {
            line.length = 0;
            for (const pt of seg) toMap(pt, extent, line);
            if (!bridge) addRibbon(acc[c.key], line, halfW);
            if (speed !== undefined) packRoad(roads, line, c.width, speed, oneway);
          }
        }
      }
    }

    const layers: Record<string, WorldTerrainLayer> = {};
    for (const key of WORLD_LAYER_KEYS) {
      const { coords, indices } = acc[key];
      const y = LAYER_Y[key];
      if (terrain) {
        const out = { vertices: [] as number[], indices: [] as number[] };
        terrain.drapeTriangles(coords, indices, y, DRAPE_TOLERANCE, !FLAT_NORMAL_LAYERS.has(key), out);
        layers[key] = { vertices: new Float32Array(out.vertices), indices: new Uint32Array(out.indices), floats: GRID_VERTEX_FLOATS };
      } else {
        const vertices = new Float32Array(coords.length / 2 * 3);
        for (let k = 0, o = 0; k < coords.length; k += 2, o += 3) {
          vertices[o] = coords[k]; vertices[o + 1] = y; vertices[o + 2] = -coords[k + 1];
        }
        layers[key] = { vertices, indices: new Uint32Array(indices), floats: 3 };
      }
    }
    result.terrain = layers;
    if (terrain) result.ground = { ...terrain.buildMesh(), floats: GRID_VERTEX_FLOATS };
    if (decks) result.bridges = decks.meshes;
    result.roads = new Float32Array(roads);
  }

  if (wantBuildings) {
    result.buildings = buildBuildings(vt, tile, proj, toMap, wantTrees, terrain, decks?.trace ?? []);
  }

  return result;
}

/** A ground layer before the terrain: the points (x, north) and the triangles, 3 point indices each. */
type FlatMesh = { coords: number[]; indices: number[] };

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
function addRibbon(out: FlatMesh, line: number[], halfW: number): void {
  // the points without repeats (a zero length segment has no direction)
  const xs: number[] = [], ys: number[] = [];
  for (let i = 0; i < line.length; i += 2) {
    const x = line[i], y = line[i + 1];
    const k = xs.length - 1;
    if (k >= 0 && xs[k] === x && ys[k] === y) continue;
    xs.push(x); ys.push(y);
  }
  let n = xs.length;
  if (n < 2) return;
  const closed = n > 3 && xs[0] === xs[n - 1] && ys[0] === ys[n - 1];
  if (closed) n--; // the last point is the first
  const segs = closed ? n : n - 1;

  // the unit direction of segment i, from point i to point i + 1
  const dxs = new Float64Array(segs), dys = new Float64Array(segs);
  for (let i = 0; i < segs; i++) {
    const j = (i + 1) % n;
    const dx = xs[j] - xs[i], dy = ys[j] - ys[i];
    const len = Math.hypot(dx, dy);
    dxs[i] = dx / len; dys[i] = dy / len;
  }

  const points = out.coords;
  // the points p + o (left) and p - o (right), returns the index of the left one
  const pair = (x: number, y: number, ox: number, oy: number): number => {
    const v = points.length / 2;
    points.push(x + ox, y + oy, x - ox, y - oy);
    return v;
  };

  // per point the pair the segment before it ends on and the pair the segment after it starts on: the same pair when
  // mitered
  const ends = new Int32Array(n), starts = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    const prev = closed ? (i + segs - 1) % segs : i - 1;
    const next = closed || i < segs ? i : -1;
    if (prev < 0 || next < 0) {
      // an end of an open line: square to its segment
      const s = prev < 0 ? next : prev;
      ends[i] = starts[i] = pair(x, y, -dys[s] * halfW, dxs[s] * halfW);
      continue;
    }
    // the left normals of both segments, the miter halfway between them
    const n0x = -dys[prev], n0y = dxs[prev];
    const n1x = -dys[next], n1y = dxs[next];
    let mx = n0x + n1x, my = n0y + n1y;
    const ml = Math.hypot(mx, my);
    // cos of the angle between the miter and the normals: the miter is halfW / cos long
    const cos = ml > 1e-9 ? (mx * n1x + my * n1y) / ml : 0;
    if (cos >= 1 / MITER_LIMIT) {
      const s = halfW / (cos * ml);
      mx *= s; my *= s;
      ends[i] = starts[i] = pair(x, y, mx, my);
    } else {
      ends[i] = pair(x, y, n0x * halfW, n0y * halfW);
      starts[i] = pair(x, y, n1x * halfW, n1y * halfW);
    }
  }

  for (let i = 0; i < segs; i++) {
    const a = starts[i], b = ends[(i + 1) % n];
    // (a left, b right, b left), (a left, a right, b right)
    out.indices.push(a, b + 1, b, a, a + 1, b + 1);
  }
}

// --- bridges -------------------------------------------------------------------

/** A bridge line of a tile, see buildBridges. */
type BridgeLine = {
  key: string;
  halfW: number;
  /** The pieces of the line in the tile (map x, north), see clipLineToTile. */
  lines: number[][];
  /** OSM layer: how many levels it is over what it crosses. */
  layer: number;
  /** The first and the last point of the unclipped line, in tiles (0 to 1 is the tile). */
  tileEnds: number[];
  /** The same in map (x, north). */
  ends: number[];
  /** The ends (map x, north) that are ends of the bridge, see findBridgeEnds. */
  realEnds: number[];
};

/** A bridge area of a tile (the outline of a deck), see buildBridges: its points (map x, north) and earcut triangles. */
type BridgeArea = { key: string; layer: number; coords: number[]; tris: number[] };

/** The decks of the bridges of a tile (see buildBridges) and their boxes for the ray tracing (see buildTraceBlock). */
type BridgeDecks = { meshes: Record<string, WorldTerrainLayer>; trace: BuildingPolygon[] };

function isBridge(layerName: string, p: Props): boolean {
  return layerName === 'transportation' && p.brunnel === 'bridge';
}

/** The bridge lines and areas of the tile, of the ground layers classifyLine and classifyFill put them in. */
function collectBridges(
  vt: VectorTile, toMap: (pt: TilePoint, extent: number, out: number[]) => void, metersToNorm: number,
): { lines: BridgeLine[]; areas: BridgeArea[] } {
  const lines: BridgeLine[] = [], areas: BridgeArea[] = [];
  const layer = vt.layers['transportation'];
  if (!layer) return { lines, areas };
  const extent = layer.extent || 4096;
  for (let i = 0; i < layer.length; i++) {
    const f = layer.feature(i);
    const p = f.properties as Props;
    if (!isBridge('transportation', p)) continue;
    const level = Number(p.layer) || 1;
    if (f.type === 3) {
      const c = classifyFill('transportation', p);
      if (!c) continue;
      for (const poly of classifyRings(f.loadGeometry())) {
        const flat = polygonToMap(poly, extent, toMap);
        if (!flat) continue;
        const tris = earcut(flat.coords, flat.holes.length ? flat.holes : undefined, 2);
        if (tris.length) areas.push({ key: c.key, layer: level, coords: flat.coords, tris });
      }
    } else if (f.type === 2) {
      const c = classifyLine('transportation', p);
      if (!c) continue;
      for (const ring of f.loadGeometry()) {
        if (ring.length < 2) continue;
        const pieces: number[][] = [];
        for (const seg of clipLineToTile(ring, extent)) {
          const line: number[] = [];
          for (const pt of seg) toMap(pt, extent, line);
          pieces.push(line);
        }
        if (!pieces.length) continue;
        const a = ring[0], b = ring[ring.length - 1];
        const ends: number[] = [];
        toMap(a, extent, ends);
        toMap(b, extent, ends);
        lines.push({
          key: c.key, halfW: c.width * 0.5 * metersToNorm, lines: pieces, layer: level,
          tileEnds: [a.x / extent, a.y / extent, b.x / extent, b.y / extent], ends, realEnds: [],
        });
      }
    }
  }
  return { lines, areas };
}

// The height (m) a bridge is over what it crosses per OSM layer, where the valley fill doesn't lift it more
const BRIDGE_LIFT_M = 4.5;
// The thickness (m) of a deck
const BRIDGE_THICKNESS_M = 1.2;
// The longest piece (m) of a deck: shorter pieces follow its height profile
const BRIDGE_STEP_M = 6;
/*
The vector tiles have the lines up to 64 units (of 4096) out of the tile (OpenFreeMap). An end within END_MARGIN of the
tile is the end of the line, further out it may be where it was cut off. The ramp of a deck at an end is shorter, so a
point in the tile within the ramp of an end has it within END_MARGIN: the 2 tiles on both sides of an edge find the
same ends near it, and the deck meets itself there. About 16 m at 52 degrees (a z14 tile is 1.5 km)
 */
const BRIDGE_END_MARGIN = 48 / 4096;
const BRIDGE_RAMP = 44 / 4096;
// 2 line ends this close (tiles) are the same point: the joint of 2 lines of 1 bridge, not an end of it
const BRIDGE_JOINT = 1.5 / 4096;
const BRIDGE_MITER_LIMIT = 2;
// a box of the ray tracing is in the plane of a stretch of a deck that strays at most this much from it (m), see
// buildBridges: the box is BRIDGE_THICKNESS_M - this thick at least
const BRIDGE_TRACE_FLAT_M = 0.3;
// and at most this many segments of a line (BRIDGE_STEP_M each): a curved bridge in a few boxes, not 1 wide one
const BRIDGE_TRACE_RUN = 12;

/**
 * Marks the ends of the bridges (realEnds): the ends of their lines in or near the tile that no other bridge line ends at
 * (a long bridge is often a few lines, which meet in the middle of it).
 */
function findBridgeEnds(bridges: BridgeLine[]): void {
  const inside = (x: number, y: number) =>
    x >= -BRIDGE_END_MARGIN && x <= 1 + BRIDGE_END_MARGIN && y >= -BRIDGE_END_MARGIN && y <= 1 + BRIDGE_END_MARGIN;
  bridges.forEach((b, bi) => {
    for (let e = 0; e < 2; e++) {
      const x = b.tileEnds[e * 2], y = b.tileEnds[e * 2 + 1];
      if (!inside(x, y)) continue;
      const joint = bridges.some((o, oi) => {
        for (let k = 0; k < 2; k++) {
          if (oi === bi && k === e) continue;
          if (Math.abs(o.tileEnds[k * 2] - x) <= BRIDGE_JOINT && Math.abs(o.tileEnds[k * 2 + 1] - y) <= BRIDGE_JOINT) return true;
        }
        return false;
      });
      if (!joint) b.realEnds.push(b.ends[e * 2], b.ends[e * 2 + 1]);
    }
  });
}

/**
 * The bridge decks of a tile: per line a box along it, BRIDGE_THICKNESS_M thick, its top at the height of the deck (see
 * deckHeight) plus the lift of its ground layer, so it meets the draped road at its ends. Per point of the line (cut into
 * pieces of BRIDGE_STEP_M) 8 vertices: the top 2 with the normal of the deck (up, tilted along a ramp), the sides 2 + 2
 * with the normal sideways, the bottom 2 with the normal down (encodeNormal). Drawn without culling.
 * tileSize: the width of the tile in map units.
 */
function buildBridges(
  bridges: BridgeLine[], areas: BridgeArea[], terrain: TerrainGrid | null, metersToNorm: number, tileSize: number,
): BridgeDecks {
  findBridgeEnds(bridges);
  const ramp = BRIDGE_RAMP * tileSize;
  const thickness = BRIDGE_THICKNESS_M * metersToNorm;
  const step = BRIDGE_STEP_M * metersToNorm;
  // the most a deck may stray from the plane of its trace box
  const flat = BRIDGE_TRACE_FLAT_M * metersToNorm;

  /*
  The boxes of the ray tracing (see buildTraceBlock): per stretch of a deck that is about planar one sloped prism (see
  BuildingPolygon.slope) in that plane, inside the deck: its top at the lowest the deck top is over the plane (without
  the lift of the ground layer), its base at the highest the deck bottom is. A ray that starts inside a box and leaves it
  through a side is blocked, so every deck surface has to be on or outside every box: on a ramp, and where decks of other
  layers overlap (a road, its sidewalk, the area under both, of the same deckHeight but each lifted by its LAYER_Y). Two
  boxes of a deck overlap in height where they meet (both are in the deck there, at least thickness - 2 flat thick), so
  no light leaks between them. A level span is 1 box, a ramp a few. The ray tracing tests a point in a footprint even-odd
  over all its edges, so a footprint may have any rings, in any order.
  A plane: gx, gn the rise per map unit of x and north, through 0 at the first point of the footprint
   */
  const trace: BuildingPolygon[] = [];
  const box = (coords: number[], ringStarts: number[], ringEnds: number[], gx: number, gn: number, lo: number, hi: number, lift: number) =>
    trace.push({ coords, ringStarts, ringEnds, base: hi - thickness, top: lo - lift, roofChart: 0, slope: [gx, -gn] });

  // a line (its 8 vertices per point in v from vertex first, see below): runs of segments, each its outline
  const traceLine = (v: number[], first: number, hs: Float64Array, lift: number) => {
    const n = hs.length;
    // the corner of point i: left (vertex 0) or right (vertex 1), map x, north
    const cx = (i: number, side: number) => v[(first + i * 8 + side) * GRID_VERTEX_FLOATS];
    const cn = (i: number, side: number) => -v[(first + i * 8 + side) * GRID_VERTEX_FLOATS + 2];
    let s = 0;
    while (s < n - 1) {
      // the plane along the first segment, through 0 at the right corner of point s (the first of the footprint)
      const dx = (cx(s + 1, 0) + cx(s + 1, 1) - cx(s, 0) - cx(s, 1)) / 2;
      const dn = (cn(s + 1, 0) + cn(s + 1, 1) - cn(s, 0) - cn(s, 1)) / 2;
      const len2 = dx * dx + dn * dn || 1;
      const gx = (hs[s + 1] - hs[s]) * dx / len2, gn = (hs[s + 1] - hs[s]) * dn / len2;
      const x0 = cx(s, 1), n0 = cn(s, 1);
      const dev = (i: number, side: number) => hs[i] - gx * (cx(i, side) - x0) - gn * (cn(i, side) - n0);
      let lo = Infinity, hi = -Infinity, e = s;
      for (let i = s; i < n; i++) {
        const a = dev(i, 0), b = dev(i, 1);
        const nlo = Math.min(lo, a, b), nhi = Math.max(hi, a, b);
        if (i > s + 1 && (nhi - nlo > flat || i - s > BRIDGE_TRACE_RUN)) break;
        lo = nlo; hi = nhi; e = i;
      }
      // counter-clockwise: the right side forward, the left side back
      const coords: number[] = [];
      for (let i = s; i <= e; i++) coords.push(cx(i, 1), cn(i, 1));
      for (let i = e; i >= s; i--) coords.push(cx(i, 0), cn(i, 0));
      box(coords, [0], [coords.length / 2], gx, gn, lo, hi, lift);
      s = e;
    }
  };

  // an area: patches of adjacent triangles about in the plane of their first one, each the rings of its outline
  const traceArea = (
    coords: number[], tris: number[], hs: Float64Array, lift: number,
    edges: Map<number, number[]>, edgeKey: (a: number, b: number) => number,
  ) => {
    const count = tris.length / 3;
    const group = new Int32Array(count).fill(-1);
    for (let seed = 0; seed < count; seed++) {
      if (group[seed] >= 0) continue;
      // the plane of the seed triangle
      const a = tris[seed * 3], b = tris[seed * 3 + 1], c = tris[seed * 3 + 2];
      const x1 = coords[b * 2] - coords[a * 2], n1 = coords[b * 2 + 1] - coords[a * 2 + 1], h1 = hs[b] - hs[a];
      const x2 = coords[c * 2] - coords[a * 2], n2 = coords[c * 2 + 1] - coords[a * 2 + 1], h2 = hs[c] - hs[a];
      const det = x1 * n2 - x2 * n1;
      const ok = Math.abs(det) > 1e-18;
      const gx = ok ? (h1 * n2 - h2 * n1) / det : 0, gn = ok ? (x1 * h2 - x2 * h1) / det : 0;
      const dev = (k: number) => hs[k] - gx * coords[k * 2] - gn * coords[k * 2 + 1];
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < 3; k++) { lo = Math.min(lo, dev(tris[seed * 3 + k])); hi = Math.max(hi, dev(tris[seed * 3 + k])); }
      group[seed] = seed;
      const members = [seed];
      for (let m = 0; m < members.length; m++) {
        const t = members[m];
        for (let e = 0; e < 3; e++) {
          for (const u of edges.get(edgeKey(tris[t * 3 + e], tris[t * 3 + (e + 1) % 3]))!) {
            if (group[u] >= 0) continue;
            let ulo = lo, uhi = hi;
            for (let k = 0; k < 3; k++) { ulo = Math.min(ulo, dev(tris[u * 3 + k])); uhi = Math.max(uhi, dev(tris[u * 3 + k])); }
            if (uhi - ulo > flat) continue;
            group[u] = seed; lo = ulo; hi = uhi;
            members.push(u);
          }
        }
      }
      // the outline: the edges (in the winding of their triangle) without a neighbour in the patch, chained into rings
      const next = new Map<number, number[]>();
      for (const t of members) {
        for (let e = 0; e < 3; e++) {
          const p0 = tris[t * 3 + e], p1 = tris[t * 3 + (e + 1) % 3];
          if (edges.get(edgeKey(p0, p1))!.some((u) => u !== t && group[u] === seed)) continue;
          const list = next.get(p0);
          if (list) list.push(p1); else next.set(p0, [p1]);
        }
      }
      const ring: number[] = [], starts: number[] = [], ends: number[] = [];
      for (const [start, list] of next) {
        while (list.length) {
          starts.push(ring.length / 2);
          let p = start;
          do {
            ring.push(coords[p * 2], coords[p * 2 + 1]);
            p = next.get(p)!.pop()!;
          } while (p !== start && next.get(p)?.length);
          ends.push(ring.length / 2);
        }
      }
      if (!ring.length) continue;
      // the devs are over the plane through 0 at (0, 0): over the one through 0 at the first point of the footprint
      const at0 = gx * ring[0] + gn * ring[1];
      box(ring, starts, ends, gx, gn, lo + at0, hi + at0, lift);
    }
  };
  const out: Record<string, { vertices: number[]; indices: number[] }> = {};
  const down = encodeNormal(0, -1, 0);

  /*
  The top of the deck at (x, north): from the ground at an end of the bridge up to the higher of the valley fill (the
  rims of the valley it spans, see TerrainGrid.fillHeightAt) and the lift of its layer over the ground, along a
  smoothstep over the ramp. Only the position and the ends near it count, so 2 tiles agree on their shared edge
   */
  const deckHeight = (ends: number[], layer: number, x: number, y: number): number => {
    const ground = terrain ? terrain.heightAt(x, y) : 0;
    const fill = terrain ? terrain.fillHeightAt(x, y) : 0;
    let d = Infinity;
    for (let k = 0; k < ends.length; k += 2) d = Math.min(d, Math.hypot(x - ends[k], y - ends[k + 1]));
    const t = Math.min(d / ramp, 1);
    const s = t * t * (3 - 2 * t);
    return ground + s * Math.max(fill - ground, Math.max(layer, 0) * BRIDGE_LIFT_M * metersToNorm);
  };

  // The ends of all the bridge lines, for every deck: an area has no ends of its own (the roads on it end where it does),
  // and a road, its sidewalks and the area under them get the same height where they overlap
  const allEnds: number[] = [];
  for (const b of bridges) allEnds.push(...b.realEnds);
  for (const area of areas) {
    const mesh = out[area.key] ??= { vertices: [], indices: [] };
    const { coords, tris } = subdivideTriangles(area.coords, area.tris, step * 2);
    const n = coords.length / 2;
    const lift = LAYER_Y[area.key];
    const hs = new Float64Array(n);
    for (let i = 0; i < n; i++) hs[i] = deckHeight(allEnds, area.layer, coords[i * 2], coords[i * 2 + 1]) + lift;
    const v = mesh.vertices, idx = mesh.indices;
    // the top (up) and the bottom (down, reversed)
    const top = v.length / GRID_VERTEX_FLOATS, up = encodeNormal(0, 1, 0);
    for (let i = 0; i < n; i++) v.push(coords[i * 2], hs[i], -coords[i * 2 + 1], up);
    const bottom = v.length / GRID_VERTEX_FLOATS;
    for (let i = 0; i < n; i++) v.push(coords[i * 2], hs[i] - thickness, -coords[i * 2 + 1], down);
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t], b = tris[t + 1], c = tris[t + 2];
      idx.push(top + a, top + b, top + c);
      idx.push(bottom + a, bottom + c, bottom + b);
    }
    // the triangles of each edge, for the outline (edges of 1 triangle) and the trace patches
    const edges = new Map<number, number[]>();
    const edgeKey = (a: number, b: number) => Math.min(a, b) * n + Math.max(a, b);
    for (let t = 0; t < tris.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const k = edgeKey(tris[t + e], tris[t + (e + 1) % 3]);
        const list = edges.get(k);
        if (list) list.push(t / 3); else edges.set(k, [t / 3]);
      }
    }
    traceArea(coords, tris, hs, lift, edges, edgeKey);
    // the sides: along the edges of 1 triangle only (the outline), facing out (right of the counter-clockwise edge)
    for (let t = 0; t < tris.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = tris[t + e], b = tris[t + (e + 1) % 3];
        if (edges.get(edgeKey(a, b))!.length !== 1) continue;
        const ax = coords[a * 2], ay = coords[a * 2 + 1], bx = coords[b * 2], by = coords[b * 2 + 1];
        const len = Math.hypot(bx - ax, by - ay);
        if (len === 0) continue;
        // out: (dy, -dx) in (x, north), engine (dy, 0, dx)
        const nx = (by - ay) / len, ny = -(bx - ax) / len;
        const side = encodeNormal(nx, 0, -ny);
        const w = v.length / GRID_VERTEX_FLOATS;
        v.push(ax, hs[a], -ay, side, bx, hs[b], -by, side, bx, hs[b] - thickness, -by, side, ax, hs[a] - thickness, -ay, side);
        idx.push(w, w + 1, w + 2, w, w + 2, w + 3);
      }
    }
  }

  for (const b of bridges) {
    const mesh = out[b.key] ??= { vertices: [], indices: [] };
    const lift = LAYER_Y[b.key];
    for (const line of b.lines) {
      // the points, without repeats, cut into pieces of at most step
      const xs: number[] = [], ys: number[] = [];
      for (let i = 0; i < line.length; i += 2) {
        const x = line[i], y = line[i + 1];
        const k = xs.length - 1;
        if (k >= 0 && xs[k] === x && ys[k] === y) continue;
        if (k >= 0) {
          const px = xs[k], py = ys[k];
          const pieces = Math.ceil(Math.hypot(x - px, y - py) / step);
          for (let p = 1; p < pieces; p++) { xs.push(px + (x - px) * p / pieces); ys.push(py + (y - py) * p / pieces); }
        }
        xs.push(x); ys.push(y);
      }
      const n = xs.length;
      if (n < 2) continue;
      const hs = new Float64Array(n);
      for (let i = 0; i < n; i++) hs[i] = deckHeight(allEnds, b.layer, xs[i], ys[i]) + lift;

      const first = mesh.vertices.length / GRID_VERTEX_FLOATS;
      const v = mesh.vertices;
      for (let i = 0; i < n; i++) {
        // the unit directions of the segments before and after the point (the one segment at an end)
        const i0 = Math.max(i - 1, 0), i1 = Math.min(i + 1, n - 1);
        const pa = i > 0 ? i - 1 : 0, pb = i > 0 ? i : 1;
        const na = i < n - 1 ? i : n - 2, nb = na + 1;
        const pl = Math.hypot(xs[pb] - xs[pa], ys[pb] - ys[pa]), nl = Math.hypot(xs[nb] - xs[na], ys[nb] - ys[na]);
        const pdx = (xs[pb] - xs[pa]) / pl, pdy = (ys[pb] - ys[pa]) / pl;
        const ndx = (xs[nb] - xs[na]) / nl, ndy = (ys[nb] - ys[na]) / nl;
        // the left normal mitered between both: halfW / cos long, cos between it and the normal of the next segment
        let mx = -pdy - ndy, my = pdx + ndx;
        const ml = Math.hypot(mx, my);
        if (ml < 1e-9) { mx = -ndy; my = ndx; } else { mx /= ml; my /= ml; }
        const cos = Math.max(mx * -ndy + my * ndx, 1 / BRIDGE_MITER_LIMIT);
        const ox = mx * b.halfW / cos, oy = my * b.halfW / cos;

        // the slope along the line, for the normal of the top: it rises sl per unit along (dx, dn) in (x, north)
        const run = Math.hypot(xs[i1] - xs[i0], ys[i1] - ys[i0]) || 1;
        const sl = (hs[i1] - hs[i0]) / run;
        const dx = (xs[i1] - xs[i0]) / run, dn = (ys[i1] - ys[i0]) / run;
        // engine space: (x, height, -north)
        const tl = Math.hypot(sl, 1);
        const top = encodeNormal(-sl * dx / tl, 1 / tl, sl * dn / tl);
        // the left side faces (mx, my) in (x, north): engine (mx, 0, -my)
        const left = encodeNormal(mx, 0, -my), right = encodeNormal(-mx, 0, my);
        const h = hs[i], lo = h - thickness;
        const lx = xs[i] + ox, ly = ys[i] + oy, rx = xs[i] - ox, ry = ys[i] - oy;
        v.push(
          lx, h, -ly, top, rx, h, -ry, top, // 0, 1: top left, right
          lx, h, -ly, left, lx, lo, -ly, left, // 2, 3: left side top, bottom
          rx, h, -ry, right, rx, lo, -ry, right, // 4, 5: right side top, bottom
          lx, lo, -ly, down, rx, lo, -ry, down, // 6, 7: bottom left, right
        );
      }
      traceLine(v, first, hs, lift);
      const idx = mesh.indices;
      for (let i = 0; i + 1 < n; i++) {
        const a = first + i * 8, c = a + 8;
        // the quads between 2 points (p, q here, then q, p at the next) as 2 triangles: top, sides, bottom
        for (const [p, q] of [[0, 1], [3, 2], [4, 5], [7, 6]]) {
          idx.push(a + p, a + q, c + q, a + p, c + q, c + p);
        }
      }
    }
  }

  const meshes: Record<string, WorldTerrainLayer> = {};
  for (const key in out) {
    const m = out[key];
    meshes[key] = { vertices: new Float32Array(m.vertices), indices: new Uint32Array(m.indices), floats: GRID_VERTEX_FLOATS };
  }
  return { meshes, trace };
}

/**
 * Splits the triangles (coords: x, north per point; tris: 3 point indices each) until no edge is longer than maxLen, so
 * the points can follow a height profile. An edge is halved when it is too long, whichever triangle it is of, at a point
 * the triangles on both sides share: no cracks. A triangle with 1, 2 or 3 halved edges becomes 2, 3 or 4, in its winding.
 */
function subdivideTriangles(coords0: number[], tris0: ArrayLike<number>, maxLen: number): { coords: number[]; tris: number[] } {
  const coords = coords0.slice();
  let tris = Array.from(tris0);
  const maxSq = maxLen * maxLen;
  for (let pass = 0; pass < 12; pass++) {
    const mids = new Map<string, number>();
    const mid = (a: number, b: number): number => {
      const dx = coords[a * 2] - coords[b * 2], dy = coords[a * 2 + 1] - coords[b * 2 + 1];
      if (dx * dx + dy * dy <= maxSq) return -1;
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      let m = mids.get(key);
      if (m === undefined) {
        m = coords.length / 2;
        coords.push((coords[a * 2] + coords[b * 2]) / 2, (coords[a * 2 + 1] + coords[b * 2 + 1]) / 2);
        mids.set(key, m);
      }
      return m;
    };
    const next: number[] = [];
    for (let t = 0; t < tris.length; t += 3) {
      const p = [tris[t], tris[t + 1], tris[t + 2]];
      const m = [mid(p[0], p[1]), mid(p[1], p[2]), mid(p[2], p[0])];
      const split = m.filter((x) => x >= 0).length;
      if (split === 0) { next.push(p[0], p[1], p[2]); continue; }
      if (split === 3) {
        next.push(p[0], m[0], m[2], m[0], p[1], m[1], m[2], m[1], p[2], m[0], m[1], m[2]);
        continue;
      }
      // rotate so that edge 0 (a, b) is halved, and with 2 halved edges also edge 1 (b, c)
      let r = 0;
      if (split === 1) r = m.findIndex((x) => x >= 0);
      else r = m[2] < 0 ? 0 : m[0] < 0 ? 1 : 2;
      const a = p[r], b = p[(r + 1) % 3], c = p[(r + 2) % 3];
      const mab = m[r], mbc = m[(r + 1) % 3];
      if (split === 1) next.push(a, mab, c, mab, b, c);
      else next.push(mab, b, mbc, a, mab, mbc, a, mbc, c);
    }
    const done = next.length === tris.length;
    tris = next;
    if (done) break;
  }
  return { coords, tris };
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
  bridges: BuildingPolygon[] = [],
): WorldBuildingResult {
  const verts: number[] = [];
  // Per vertex the lightmap chart, see SkyVisibilityLightmap: (atlas u, atlas v) of the start of a wall, (roof chart, -1)
  // on a roof (see getRoofChart). The tile of the wall atlas is the same for all, see wallTile.
  const lightmap: number[] = [];
  const buildings: BuildingInfo[] = [];
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
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
  const indices: number[] = [];
  let vertexCount = 0;
  // u: the roof chart on a roof. Returns the index of the vertex
  const push = (x: number, y: number, h: number, id: number, u: number, v: number, roof: boolean): number => {
    const ex = x, ey = h, ez = -y;
    verts.push(ex, ey, ez, id);
    if (roof) lightmap.push(u, -1);
    else lightmap.push(u, v);
    if (ex < min[0]) min[0] = ex; if (ex > max[0]) max[0] = ex;
    if (ey < min[1]) min[1] = ey; if (ey > max[1]) max[1] = ey;
    if (ez < min[2]) min[2] = ez; if (ez > max[2]) max[2] = ez;
    return vertexCount++;
  };
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
    const first = vertexCount;
    for (let k = 0; k < coords.length; k += 2) {
      const x = coords[k], y = coords[k + 1];
      // the atlas v at the bottom vertex, which can be over the base
      const bottom = corners[k / 2];
      push(x, y, bottom, id, atlas[k], atlas[k + 1] + bottom - baseN, false);
      // without a roof the top vertex is never the first of a triangle: its attribute is not read
      if (hasRoof) push(x, y, topN, id, roofChart, 0, true);
      else push(x, y, topN, id, atlas[k], atlas[k + 1], false);
    }

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
        if (edges[pa] !== corners[pb]) b0 = push(bx, by, edges[pa], id, atlas[pb * 2], atlas[pb * 2 + 1], false);
        indices.push(a0, b0, b1, a0, b1, a0 + 1);
        const u1 = atlas[pa * 2] + Math.hypot(bx - ax, by - ay), v1 = atlas[pa * 2 + 1] + height;
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
  if (verts.length === 0) {
    min[0] = min[1] = min[2] = 0;
    max[0] = max[1] = max[2] = 0;
  }
  // The ground plane of the ray tracing is the ground quad (y 0), the layers lie a few cm over it (see LAYER_GAP). On
  // the terrain the bases are on the ground already
  const buildingFootprints = footprints.slice(0, treeStart);
  // the bridge decks are in it too (see buildBridges), without an id: they shadow and occlude the sky, no picking
  const trace = buildTraceBlock([...buildingFootprints, ...bridges], traceTrees, terrain ? null : 0,
    [...buildingFootprints.map((fp) => fp.id), ...bridges.map(() => NO_FEATURE_ID)]);
  const { vertices, quantization } = quantizeVertices(verts, min, max);
  const lightmapData = new Float32Array(lightmap);
  return {
    vertices, quantization, lightmap: lightmapData, interleaved: interleaveBuildingVertices(vertices, lightmapData),
    indices: vertices.length / 4 <= 65536 ? new Uint16Array(indices) : new Uint32Array(indices), trace, buildings, min, max,
    atlasSize: walls ? atlasSize : null, wallTile: [layout.tileX, layout.tileY],
  };
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
function quantizeVertices(verts: number[], min: number[], max: number[]): { vertices: Uint16Array; quantization: number[] } {
  const vertices = new Uint16Array(verts.length);
  const scale = [0, 0, 0, 65535], offset = [min[0], min[1], min[2], -FEATURE_ID_BIAS];
  const toUnit = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const size = max[k] - min[k];
    // the unorm is read as 0 to 1
    scale[k] = size;
    toUnit[k] = size > 0 ? 65535 / size : 0;
  }
  for (let i = 0; i < verts.length; i += 4) {
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
