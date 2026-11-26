import earcut from 'earcut';
import MapProjection from './MapProjection';

/**
 * Pure (renderer-free) geometry core shared by {@link OSMBuilder} and the PDOK
 * terrain Web Worker. Turns decoded {@link OSMElement}s into a merged xyz
 * triangle soup per layer ({@link buildOSMPositions}) — earcut triangulation,
 * ribbon extrusion and (for the Overpass path) building extrusion — without
 * touching the GPU, so it can run on the main thread *or* inside a worker.
 *
 * {@link OSMBuilder} wraps the per-layer position buffers into engine `Mesh`es
 * (which need a `Renderer`); the worker transfers the raw buffers back. Keep
 * this module free of any `Mesh`/`Renderer` imports so the worker bundle stays
 * small and WebGPU-free.
 */

export type LatLngBounds = {
  south: number;
  west: number;
  north: number;
  east: number;
};

// Renderable layers, named after the PDOK BRT-A MVT source-layers they map to
// (so the UI matches what the tiles actually contain). The Overpass fallback
// path maps its semantic OSM tags onto these same keys.
//
// `mode` decides how a layer's geometry is built:
//   - 'fill'     : flat filled polygons (areas)
//   - 'ribbon'   : flat ribbons extruded from poly-lines (line features)
//   - 'building' : extruded 3D volumes (Overpass buildings only)
// `groundOffset` is a small downward z offset (meters) so coplanar ground
// layers don't z-fight (earlier / more negative = lower, drawn first).
export type LayerMode = 'fill' | 'ribbon' | 'building';

export type LayerConfig = {
  key: string;
  mode: LayerMode;
  groundOffset: number;
  ribbonWidth?: number; // meters, used by 'ribbon' layers
};

export const LAYER_CONFIGS: LayerConfig[] = [
  { key: 'nederland', mode: 'fill', groundOffset: -0.8 },
  { key: 'waterdeelvlak', mode: 'fill', groundOffset: -0.6 },
  { key: 'terreinvlak', mode: 'fill', groundOffset: -0.5 },
  { key: 'wegdeelvlak', mode: 'fill', groundOffset: -0.3 },
  { key: 'waterdeellijn', mode: 'ribbon', groundOffset: -0.25, ribbonWidth: 3 },
  { key: 'spoorbaandeellijn', mode: 'ribbon', groundOffset: -0.15, ribbonWidth: 3 },
  { key: 'inrichtingselementlijn', mode: 'ribbon', groundOffset: -0.1, ribbonWidth: 1.5 },
  { key: 'gebouwvlak', mode: 'fill', groundOffset: 0 },
];

export const LAYER_KEYS = LAYER_CONFIGS.map((c) => c.key) as readonly string[];

export type LayerKey = (typeof LAYER_CONFIGS)[number]['key'];

const LAYER_CONFIG_BY_KEY: Record<string, LayerConfig> = Object.fromEntries(
  LAYER_CONFIGS.map((c) => [c.key, c]),
);

export type OSMBuilderOptions = {
  // Finalize the current mesh once it reaches this many vertices.
  chunkMaxVertices: number;
  // Overpass endpoint (public instance by default).
  overpassUrl: string;
  // Default building height (meters) when no height/levels tag is present.
  defaultBuildingHeight: number;
  // Meters per building level (used when only building:levels is tagged).
  metersPerLevel: number;
};

export const defaultOSMBuilderOptions: OSMBuilderOptions = {
  chunkMaxVertices: 1_500_000,
  overpassUrl: 'https://overpass-api.de/api/interpreter',
  defaultBuildingHeight: 8,
  metersPerLevel: 3,
};

// Per-layer downward z offset (meters), derived from LAYER_CONFIGS.
const LAYER_GROUND_OFFSET_METERS: Record<string, number> = Object.fromEntries(
  LAYER_CONFIGS.map((c) => [c.key, c.groundOffset]),
);

// Road / path widths in meters, keyed by OSM highway value.
const ROAD_WIDTHS: Record<string, number> = {
  motorway: 14,
  motorway_link: 9,
  trunk: 12,
  trunk_link: 8,
  primary: 11,
  primary_link: 7,
  secondary: 9,
  tertiary: 7,
  unclassified: 5.5,
  residential: 5.5,
  living_street: 5,
  service: 4,
  track: 3.5,
  road: 5,
  pedestrian: 4,
};

const PATH_WIDTHS: Record<string, number> = {
  footway: 1.5,
  path: 1.8,
  cycleway: 2,
  steps: 1.5,
  bridleway: 1.5,
};

export type OSMElement = {
  type: 'way' | 'relation' | 'node';
  id: number;
  tags?: Record<string, string>;
  geometry?: { lat: number; lon: number }[];
  members?: { type: string; role: string; geometry?: { lat: number; lon: number }[] }[];
  // PDOK source-layer key (e.g. 'wegdeelvlak'). When set, the element is
  // routed directly to that layer; otherwise tags are classified (Overpass).
  layer?: string;
  // Interior rings (holes) cut out of `geometry` for filled polygons, e.g. a
  // pond inside a road surface or a park cut out of a built-up area.
  holes?: { lat: number; lon: number }[][];
};

// On-disk format saved after a fetch and accepted by drag-and-drop. Raw
// Overpass JSON (just { elements }) is also accepted on load.
export type OSMSaveFile = {
  bounds: LatLngBounds;
  elements: OSMElement[];
};

export type LayerAccumulator = {
  positions: number[];
  features: number;
};

// Trivial math helpers, inlined so this module pulls in no renderer-package
// imports (keeps the worker bundle WebGPU-free).
function clamp(v: number, min: number, max: number): number {
  return v < min ? min : v > max ? max : v;
}
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Build merged vertex positions (xyz triangle soup) per layer from decoded OSM
 * elements, without creating Meshes. Used by the batched terrain buffer (which
 * packs many tiles' positions into one GPU buffer) and the terrain Web Worker.
 */
export function buildOSMPositions(
  bounds: LatLngBounds,
  elements: OSMElement[],
  options: OSMBuilderOptions = defaultOSMBuilderOptions,
): Record<LayerKey, Float32Array> {
  const { acc } = accumulateLayers(bounds, elements, options);
  const out = {} as Record<LayerKey, Float32Array>;
  for (const key of LAYER_KEYS) out[key] = new Float32Array(acc[key].positions);
  return out;
}

export function accumulateLayers(
  bounds: LatLngBounds,
  elements: OSMElement[],
  options: OSMBuilderOptions,
): { acc: Record<LayerKey, LayerAccumulator>; mapAspectRatio: number } {
  const proj = new MapProjection();
  proj.setBounds(bounds);
  const { latMid, lngMid, cosLat, scale, mapAspectRatio } = proj;
  const metersToNorm = proj.metersToNorm;

  const project = (lng: number, lat: number): [number, number] => {
    const x = (lng - lngMid) * cosLat * scale;
    const y = (lat - latMid) * scale;
    return [x, y];
  };

  const acc: Record<LayerKey, LayerAccumulator> = {} as Record<LayerKey, LayerAccumulator>;
  for (const key of LAYER_KEYS) acc[key] = { positions: [], features: 0 };

  const ring2D: number[] = [];

  const projectRing = (geometry: { lat: number; lon: number }[]): number => {
    ring2D.length = 0;
    for (let i = 0; i < geometry.length; i++) {
      const [x, y] = project(geometry[i].lon, geometry[i].lat);
      ring2D.push(x, y);
    }
    // Drop duplicate closing vertex if present.
    const n = ring2D.length;
    if (n >= 4 && ring2D[0] === ring2D[n - 2] && ring2D[1] === ring2D[n - 1]) {
      ring2D.length = n - 2;
    }
    return ring2D.length / 2;
  };

  // Project a ring's [lng,lat] into x/y, appended to `out`, dropping a
  // duplicate closing vertex. Returns the number of points appended.
  const projectRingInto = (out: number[], geometry: { lat: number; lon: number }[]): number => {
    const start = out.length;
    for (let i = 0; i < geometry.length; i++) {
      const [x, y] = project(geometry[i].lon, geometry[i].lat);
      out.push(x, y);
    }
    const n = out.length;
    if (n - start >= 4 && out[start] === out[n - 2] && out[start + 1] === out[n - 1]) {
      out.length = n - 2;
    }
    return (out.length - start) / 2;
  };

  const addFlatPolygon = (
    layer: LayerKey,
    geometry: { lat: number; lon: number }[],
    holes?: { lat: number; lon: number }[][],
  ) => {
    const verts: number[] = [];
    if (projectRingInto(verts, geometry) < 3) return;

    // Holes are appended after the outer ring; earcut takes their start
    // vertex indices. Degenerate holes (< 3 points) are dropped.
    const holeIndices: number[] = [];
    if (holes) {
      for (const hole of holes) {
        const startIdx = verts.length / 2;
        if (projectRingInto(verts, hole) < 3) {
          verts.length = startIdx * 2;
          continue;
        }
        holeIndices.push(startIdx);
      }
    }

    const z = LAYER_GROUND_OFFSET_METERS[layer] * metersToNorm;
    const tris = earcut(verts, holeIndices.length ? holeIndices : undefined, 2);
    const a = acc[layer];
    for (let t = 0; t < tris.length; t += 3) {
      const ia = tris[t] * 2;
      const ib = tris[t + 1] * 2;
      const ic = tris[t + 2] * 2;
      pushVertex(a, verts[ia], verts[ia + 1], z);
      pushVertex(a, verts[ib], verts[ib + 1], z);
      pushVertex(a, verts[ic], verts[ic + 1], z);
    }
    a.features++;
  };

  const addBuilding = (geometry: { lat: number; lon: number }[], base: number, top: number) => {
    const pointCount = projectRing(geometry);
    if (pointCount < 3) return;
    const a = acc.gebouwvlak;

    // Roof.
    const tris = triangulate(ring2D);
    for (let t = 0; t < tris.length; t += 3) {
      const ia = tris[t] * 2;
      const ib = tris[t + 1] * 2;
      const ic = tris[t + 2] * 2;
      pushVertex(a, ring2D[ia], ring2D[ia + 1], top);
      pushVertex(a, ring2D[ib], ring2D[ib + 1], top);
      pushVertex(a, ring2D[ic], ring2D[ic + 1], top);
    }

    // Walls (from base to top - base > 0 for building parts sitting on top
    // of another part, e.g. a tower or spire).
    for (let i = 0; i < pointCount; i++) {
      const i0 = i * 2;
      const i1 = ((i + 1) % pointCount) * 2;
      const ax = ring2D[i0];
      const ay = ring2D[i0 + 1];
      const bx = ring2D[i1];
      const by = ring2D[i1 + 1];

      pushVertex(a, ax, ay, base);
      pushVertex(a, bx, by, base);
      pushVertex(a, bx, by, top);

      pushVertex(a, ax, ay, base);
      pushVertex(a, bx, by, top);
      pushVertex(a, ax, ay, top);
    }
    a.features++;
  };

  const addRibbon = (layer: LayerKey, geometry: { lat: number; lon: number }[], widthMeters: number) => {
    const pointCount = projectRing(geometry);
    if (pointCount < 2) return;
    const z = LAYER_GROUND_OFFSET_METERS[layer] * metersToNorm;
    const halfW = widthMeters * 0.5 * metersToNorm;
    const a = acc[layer];

    for (let i = 0; i < pointCount - 1; i++) {
      const i0 = i * 2;
      const i1 = (i + 1) * 2;
      const ax = ring2D[i0];
      const ay = ring2D[i0 + 1];
      const bx = ring2D[i1];
      const by = ring2D[i1 + 1];

      let dx = bx - ax;
      let dy = by - ay;
      const len = Math.hypot(dx, dy) || 1;
      dx /= len;
      dy /= len;
      // Perpendicular offset.
      const px = -dy * halfW;
      const py = dx * halfW;

      const a0x = ax + px, a0y = ay + py;
      const a1x = ax - px, a1y = ay - py;
      const b0x = bx + px, b0y = by + py;
      const b1x = bx - px, b1y = by - py;

      pushVertex(a, a0x, a0y, z);
      pushVertex(a, b1x, b1y, z);
      pushVertex(a, b0x, b0y, z);

      pushVertex(a, a0x, a0y, z);
      pushVertex(a, a1x, a1y, z);
      pushVertex(a, b1x, b1y, z);
    }
    a.features++;
  };

  // Buildings are handled in a second pass so that "building:part" towers /
  // spires (Simple 3D Buildings) can suppress the plain building outline they
  // belong to, matching what blosm imports into Blender.
  const buildingOutlines: OSMElement[] = [];
  const buildingParts: OSMElement[] = [];

  // --- classify + dispatch every element ---
  for (const el of elements) {
    const tags = el.tags || {};

    if (el.type === 'relation') {
      // Only water relations are requested; render their outer members.
      if (tags.natural === 'water' && el.members) {
        for (const m of el.members) {
          if (m.role === 'outer' && m.geometry && m.geometry.length >= 3) {
            addFlatPolygon('waterdeelvlak', m.geometry);
          }
        }
      }
      continue;
    }

    if (el.type !== 'way' || !el.geometry || el.geometry.length < 2) continue;
    const geometry = el.geometry;

    // Overpass buildings extrude in a dedicated second pass (outline /
    // building:part handling). PDOK elements carry an explicit `layer`.
    if (!el.layer && tags['building:part']) {
      buildingParts.push(el);
      continue;
    }
    if (!el.layer && tags.building) {
      buildingOutlines.push(el);
      continue;
    }

    const c = classifyElement(el);
    if (!c) continue;
    if (c.mode === 'ribbon') addRibbon(c.key, geometry, c.width);
    else addFlatPolygon(c.key, geometry, el.holes); // 'fill' (and any explicit building footprint)
  }

  // --- buildings (second pass) ---
  // Project each part's centroid so we can detect which outlines are fully
  // described by parts and should therefore not be extruded as a flat block.
  const partCentroids: [number, number][] = [];
  for (const el of buildingParts) {
    if (!el.geometry || el.geometry.length < 3) continue;
    let cx = 0, cy = 0;
    for (const p of el.geometry) {
      const [x, y] = project(p.lon, p.lat);
      cx += x;
      cy += y;
    }
    partCentroids.push([cx / el.geometry.length, cy / el.geometry.length]);
  }

  for (const el of buildingOutlines) {
    const geometry = el.geometry!;
    // Skip the plain outline if a building:part sits inside it - the parts
    // already describe this building's volume (towers, tiered roofs, etc.).
    if (partCentroids.length && outlineCoversAnyPart(project, geometry, partCentroids)) continue;
    const { base, top } = buildingDimensions(el.tags || {}, el.id, options);
    addBuilding(geometry, base * metersToNorm, top * metersToNorm);
  }

  for (const el of buildingParts) {
    const { base, top } = buildingDimensions(el.tags || {}, el.id, options);
    addBuilding(el.geometry!, base * metersToNorm, top * metersToNorm);
  }

  return { acc, mapAspectRatio };
}

/**
 * Resolves a building (or building:part) to a [base, top] height range in
 * meters, mirroring how blosm derives heights (see the blosm style rule):
 *  - height present            -> use it directly
 *  - building:levels present   -> levels * levelHeight
 *  - nothing                   -> random_weighted levels (4/5/6 @ 10/40/10)
 */
function buildingDimensions(
  tags: Record<string, string>,
  id: number,
  options: OSMBuilderOptions,
): { base: number; top: number } {
  const rng = mulberry32(hashId(id));
  // Per-level height ~ N(metersPerLevel, sd), clamped to a sane range.
  const levelHeight = clamp(randomNormal(rng, options.metersPerLevel, 0.5), 2.2, 4.5);

  let top = parseLengthMeters(tags.height);
  if (top === null || top <= 0) {
    let levels = parseFloat(tags['building:levels']);

    if (!isNaN(levels) && levels > 0) {
      //many houses look better when they have different height than the neighboring houses
      const h = options.metersPerLevel * lerp(.9, 1.1, Math.random());
      top = levels * h;
    } else {
      top = options.defaultBuildingHeight * lerp(.9, 1.1, Math.random());
    }
  }

  let base = parseLengthMeters(tags.min_height);
  if (base === null || base <= 0) {
    const minLevel = parseFloat(tags['building:min_level']);
    if (!isNaN(minLevel) && minLevel > 0) base = minLevel * levelHeight;
  }
  if (base === null || base < 0 || base >= top) base = 0;

  return { base, top };
}

// --- helpers ----------------------------------------------------------------

function pushVertex(a: LayerAccumulator, x: number, y: number, z: number) {
  a.positions.push(x, z, -y);
}

function isClosedRing(geometry: { lat: number; lon: number }[]): boolean {
  const n = geometry.length;
  if (n < 4) return false;
  return geometry[0].lat === geometry[n - 1].lat && geometry[0].lon === geometry[n - 1].lon;
}

type ElementClass = { key: LayerKey; mode: LayerMode; width: number };

/**
 * Resolve which layer an element belongs to and how its geometry is built.
 * PDOK elements carry an explicit `layer` (the MVT source-layer); Overpass
 * elements are classified from their OSM tags onto the same layer keys.
 * Returns null for elements that map to no rendered layer.
 */
function classifyElement(el: OSMElement): ElementClass | null {
  if (el.layer) {
    const cfg = LAYER_CONFIG_BY_KEY[el.layer];
    if (!cfg) return null;
    return { key: cfg.key, mode: cfg.mode, width: cfg.ribbonWidth ?? 1 };
  }

  const tags = el.tags || {};
  const geometry = el.geometry || [];

  if (tags.natural === 'water' || tags.waterway === 'riverbank') {
    return { key: 'waterdeelvlak', mode: 'fill', width: 0 };
  }
  if (tags.waterway) {
    return { key: 'waterdeellijn', mode: 'ribbon', width: LAYER_CONFIG_BY_KEY['waterdeellijn'].ribbonWidth ?? 3 };
  }
  if (tags.railway) {
    return { key: 'spoorbaandeellijn', mode: 'ribbon', width: LAYER_CONFIG_BY_KEY['spoorbaandeellijn'].ribbonWidth ?? 3 };
  }
  if (
    tags.landuse === 'forest' ||
    tags.natural === 'wood' ||
    tags.natural === 'scrub' ||
    ['grass', 'meadow', 'recreation_ground', 'village_green', 'cemetery'].includes(tags.landuse || '') ||
    ['park', 'garden', 'pitch', 'playground'].includes(tags.leisure || '')
  ) {
    return { key: 'terreinvlak', mode: 'fill', width: 0 };
  }
  if (tags.highway) {
    const hw = tags.highway;
    const isArea = tags.area === 'yes' && isClosedRing(geometry);
    if (isArea) return { key: 'wegdeelvlak', mode: 'fill', width: 0 };
    return { key: 'wegdeelvlak', mode: 'ribbon', width: PATH_WIDTHS[hw] ?? ROAD_WIDTHS[hw] ?? 5 };
  }
  return null;
}

// Deterministic 32-bit PRNG (mulberry32) so a building's randomized height is
// stable across rebuilds. Seeded from the OSM id.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashId(id: number): number {
  // Fold a (possibly > 32 bit) OSM id into a 32-bit seed.
  let h = (id ^ (id >>> 16)) >>> 0;
  h = Math.imul(h, 0x45d9f3b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

// Normally distributed value (Box-Muller) with given mean / standard deviation.
function randomNormal(rng: () => number, mean: number, sd: number): number {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Parses an OSM length value (height / min_height) into meters. Accepts a bare
 * number, a metric suffix ("25 m", "25m"), or imperial feet/inches
 * ("82'", "82'6\"", "82 ft"). Returns null when nothing usable is found.
 */
function parseLengthMeters(value: string | undefined): number | null {
  if (!value) return null;
  const s = value.trim();

  // Feet (and optional inches): 82' or 82'6" or 82 ft
  const ftInMatch = s.match(/^(\d+(?:\.\d+)?)\s*(?:'|ft|feet)\s*(?:(\d+(?:\.\d+)?)\s*(?:"|in))?/i);
  if (ftInMatch) {
    const feet = parseFloat(ftInMatch[1]);
    const inches = ftInMatch[2] ? parseFloat(ftInMatch[2]) : 0;
    const meters = feet * 0.3048 + inches * 0.0254;
    return isNaN(meters) ? null : meters;
  }

  // Metric or bare number (ignores a trailing "m" / "meter(s)").
  const n = parseFloat(s);
  return isNaN(n) ? null : n;
}

/**
 * True if the projected outline ring contains the centroid of at least one
 * building:part, meaning the part(s) describe this building's volume and the
 * plain outline should be skipped.
 */
function outlineCoversAnyPart(
  project: (lng: number, lat: number) => [number, number],
  geometry: { lat: number; lon: number }[],
  partCentroids: [number, number][],
): boolean {
  const ring: number[] = [];
  for (const p of geometry) {
    const [x, y] = project(p.lon, p.lat);
    ring.push(x, y);
  }
  for (const [px, py] of partCentroids) {
    if (pointInPolygon(px, py, ring)) return true;
  }
  return false;
}

function pointInPolygon(px: number, py: number, ring: number[]): boolean {
  const n = ring.length / 2;
  let inside = false;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = ring[i * 2], yi = ring[i * 2 + 1];
    const xj = ring[j * 2], yj = ring[j * 2 + 1];
    const intersect = yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/**
 * Triangulate a simple polygon given as flat [x, y, ...] (no holes). Returns
 * vertex-index triplets. Delegates to earcut; `addFlatPolygon` calls earcut
 * directly when it needs holes.
 */
function triangulate(poly: number[]): number[] {
  if (poly.length < 6) return [];
  return earcut(poly, undefined, 2);
}
