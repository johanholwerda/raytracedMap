import { VectorTile } from '@mapbox/vector-tile';
import { PbfReader } from 'pbf';
import type { OSMElement } from './osmGeometryCore';
import type { TileCoord } from './PDOKTileService';

/**
 * Pure (renderer-free) MVT decode shared by {@link PDOKTileService} and the
 * PDOK terrain Web Worker. Decodes one BRT-A Mapbox Vector Tile (.pbf) into the
 * {@link OSMElement} shape that {@link osmGeometryCore} consumes — clipping each
 * feature to the tile box (BRT-A ships large polygons unclipped, repeated in
 * every overlapping tile) and converting tile-local coords to lat/lon.
 *
 * Kept free of any renderer / fetch / cache concerns so the worker can run it
 * off the main thread.
 */

// BRT-A MVT source-layers we decode, in order. Each becomes a render layer of
// the same name (see osmGeometryCore.LAYER_CONFIGS for fill / ribbon mode).
// Text-only layers (straatnamen, waterdeelvlak_label, annotaties) are skipped.
const KEPT_LAYERS: string[] = [
  'nederland',
  'waterdeelvlak',
  'terreinvlak',
  'wegdeelvlak',
  'waterdeellijn',
  'spoorbaandeellijn',
  'inrichtingselementlijn',
  'gebouwvlak',
];

/** Decode one MVT tile's relevant layers into OSM elements. */
export function decodePdokTile(bytes: Uint8Array, tile: TileCoord): OSMElement[] {
  const vt = new VectorTile(new PbfReader(bytes));
  const elements: OSMElement[] = [];
  // Local id counter: ids only need to be unique within this tile's elements.
  let nextId = 1;

  for (const layerName of KEPT_LAYERS) {
    const layer = vt.layers[layerName];
    if (!layer) continue;
    const extent = layer.extent || 4096;

    for (let i = 0; i < layer.length; i++) {
      const feature = layer.feature(i);
      const rings = feature.loadGeometry();

      if (feature.type === 3) {
        // Polygon(s): MVT lists each outer ring followed by its holes; group
        // them by winding (signed area) so holes are punched out instead of
        // filled solid (ponds in road surfaces, parks in built-up areas...).
        for (const poly of classifyRings(rings)) {
          // BRT-A ships large polygons unclipped: the *full* feature geometry
          // (often many tiles wide — a whole lake or the sea) is repeated in
          // every overlapping tile, extending far beyond [0, extent]. Clip each
          // ring to this tile's box so the tile renders only its own slice.
          // This removes the buffer overlap (no doubled geometry at seams,
          // adjacent tiles meet exactly at the boundary) and, crucially, lets a
          // big feature appear in every tile it covers instead of only the one
          // "owner" tile the old centroid test tried — and usually failed — to
          // pick, which is why low-detail water bodies went missing.
          const clippedOuter = clipRingToTile(poly[0], extent);
          const outer = ringToLatLon(clippedOuter, extent, tile);
          if (!outer) continue;
          const holes: { lat: number; lon: number }[][] = [];
          for (let h = 1; h < poly.length; h++) {
            const hole = ringToLatLon(clipRingToTile(poly[h], extent), extent, tile);
            if (hole) holes.push(hole);
          }
          // The source-layer name *is* the render layer; OSMBuilder routes on it.
          elements.push({
            type: 'way',
            id: nextId++,
            layer: layerName,
            geometry: outer,
            holes: holes.length ? holes : undefined,
          });
        }
      } else {
        // Lines / points: clip each polyline to the tile box (same reasoning as
        // for polygons). A line may leave and re-enter the tile, so clipping
        // yields zero or more visible sub-segments.
        for (const ring of rings) {
          if (ring.length < 2) continue;
          for (const seg of clipLineToTile(ring, extent)) {
            const geometry = seg.map((p) => tileToLatLon(p.x, p.y, extent, tile));
            elements.push({ type: 'way', id: nextId++, layer: layerName, geometry });
          }
        }
      }
    }
  }
  return elements;
}

// --- polygon ring helpers ---------------------------------------------------

export type TilePoint = { x: number; y: number };

/** Signed area of a tile-space ring (sign denotes winding / orientation). */
function signedAreaXY(ring: TilePoint[]): number {
  let area = 0;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    area += (ring[j].x - ring[i].x) * (ring[i].y + ring[j].y);
  }
  return area / 2;
}

/**
 * Group an MVT feature's rings into polygons of [outer, ...holes], following
 * the Mapbox Vector Tile convention: the first non-zero ring fixes the
 * exterior winding; rings with that winding start a new polygon, the others
 * are holes of the current polygon.
 */
export function classifyRings(rings: TilePoint[][]): TilePoint[][][] {
  if (rings.length <= 1) return [rings];
  const polygons: TilePoint[][][] = [];
  let polygon: TilePoint[][] | undefined;
  let exteriorIsNeg: boolean | undefined;

  for (const ring of rings) {
    const area = signedAreaXY(ring);
    if (area === 0) continue;
    if (exteriorIsNeg === undefined) exteriorIsNeg = area < 0;
    if (area < 0 === exteriorIsNeg) {
      if (polygon) polygons.push(polygon);
      polygon = [ring];
    } else if (polygon) {
      polygon.push(ring);
    }
  }
  if (polygon) polygons.push(polygon);
  return polygons;
}

/**
 * Clip a tile-space polygon ring to the tile's [0, extent] × [0, extent] box
 * using the Sutherland–Hodgman algorithm (four half-plane clips against the
 * convex box). BRT-A ships large polygons unclipped — the full feature is
 * repeated in every overlapping tile, with coordinates running far beyond the
 * extent — so each tile must clip to its own bounds to render just its slice.
 * Returns the clipped ring, which may be empty if the ring lies fully outside
 * the box. When the box is entirely inside the polygon the four box corners are
 * produced as edge intersections, i.e. the tile is filled solid (e.g. open
 * water), which is exactly what we want.
 */
export function clipRingToTile(ring: TilePoint[], extent: number): TilePoint[] {
  // edge: 0 = x>=0, 1 = x<=extent, 2 = y>=0, 3 = y<=extent.
  const inside = (p: TilePoint, edge: number): boolean =>
    edge === 0 ? p.x >= 0 : edge === 1 ? p.x <= extent : edge === 2 ? p.y >= 0 : p.y <= extent;
  const intersect = (a: TilePoint, b: TilePoint, edge: number): TilePoint => {
    const t =
      edge === 0 ? (0 - a.x) / (b.x - a.x) :
      edge === 1 ? (extent - a.x) / (b.x - a.x) :
      edge === 2 ? (0 - a.y) / (b.y - a.y) :
                   (extent - a.y) / (b.y - a.y);
    return { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };
  };

  let output = ring;
  for (let edge = 0; edge < 4 && output.length > 0; edge++) {
    const input = output;
    output = [];
    for (let i = 0; i < input.length; i++) {
      const cur = input[i];
      const prev = input[(i + input.length - 1) % input.length];
      const curIn = inside(cur, edge);
      const prevIn = inside(prev, edge);
      if (curIn) {
        if (!prevIn) output.push(intersect(prev, cur, edge));
        output.push(cur);
      } else if (prevIn) {
        output.push(intersect(prev, cur, edge));
      }
    }
  }
  return output;
}

/**
 * Clip a single tile-space segment to the [0, extent] box (Liang–Barsky).
 * Returns the visible portion, or null if the segment lies fully outside.
 */
function clipSegmentToTile(a: TilePoint, b: TilePoint, extent: number): [TilePoint, TilePoint] | null {
  const dx = b.x - a.x, dy = b.y - a.y;
  const p = [-dx, dx, -dy, dy];
  const q = [a.x, extent - a.x, a.y, extent - a.y];
  let t0 = 0, t1 = 1;
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return null; // parallel to this edge and outside it
    } else {
      const r = q[i] / p[i];
      if (p[i] < 0) { if (r > t1) return null; if (r > t0) t0 = r; }
      else { if (r < t0) return null; if (r < t1) t1 = r; }
    }
  }
  return [
    { x: a.x + t0 * dx, y: a.y + t0 * dy },
    { x: a.x + t1 * dx, y: a.y + t1 * dy },
  ];
}

/**
 * Clip a tile-space polyline to the [0, extent] box, returning the visible
 * sub-polylines (a line may leave and re-enter the box, producing several).
 */
export function clipLineToTile(line: TilePoint[], extent: number): TilePoint[][] {
  const out: TilePoint[][] = [];
  let current: TilePoint[] = [];
  for (let i = 0; i + 1 < line.length; i++) {
    const seg = clipSegmentToTile(line[i], line[i + 1], extent);
    if (!seg) {
      if (current.length >= 2) out.push(current);
      current = [];
      continue;
    }
    const [a, b] = seg;
    const last = current[current.length - 1];
    if (last && last.x === a.x && last.y === a.y) {
      current.push(b);
    } else {
      if (current.length >= 2) out.push(current);
      current = [a, b];
    }
  }
  if (current.length >= 2) out.push(current);
  return out;
}

/** Convert a tile-space ring to a closed lat/lon ring, or null if degenerate. */
function ringToLatLon(
  ring: TilePoint[],
  extent: number,
  tile: TileCoord,
): { lat: number; lon: number }[] | null {
  if (!ring || ring.length < 3) return null;
  const geometry = ring.map((p) => tileToLatLon(p.x, p.y, extent, tile));
  const first = geometry[0];
  const last = geometry[geometry.length - 1];
  if (first.lat !== last.lat || first.lon !== last.lon) geometry.push({ ...first });
  return geometry.length < 4 ? null : geometry;
}

/** Convert an MVT tile-local point (0..extent) to lat/lon (degrees). */
export function tileToLatLon(px: number, py: number, extent: number, tile: TileCoord): { lat: number; lon: number } {
  const n = Math.pow(2, tile.z);
  const x = tile.x + px / extent;
  const y = tile.y + py / extent;
  const lon = (x / n) * 360 - 180;
  const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n)));
  const lat = (latRad * 180) / Math.PI;
  return { lat, lon };
}
