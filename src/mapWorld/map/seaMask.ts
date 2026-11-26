import type { VectorTileLayer } from '@mapbox/vector-tile';
import { tileToLatLon } from '../osm/pdokDecodeCore';
import type { TileCoord } from '../osm/PDOKTileService';

/*
The sea is flat: the points of the height grid of a map tile (see ElevationService.getTileGrid) in the sea of its
vector tile get the height of the sea. The elevation tiles have false mountains on the sea in places, up to 1000 m
high (ArcticDEM north of 60 degrees, see demSpikes.ts), which removeDemSpikes only finds when they rise steeply enough
to tell from a real island; the vector tile knows where the sea is. The sea polygons of OpenMapTiles (water, class
ocean) are the coastline of OpenStreetMap, so the islands, sea stacks and cliffs keep their heights. The points of the
sea next to the land keep the slope down to the coast (see flattenSea): where the land is too high (ArcticDEM where
arcticDem.ts leaves it, 17 to 33 m: Svalbard, Nuuk) flat sea next to it showed as a wall. The lakes aren't
flattened: their level isn't known (the lowest point of the shore per tile would differ between the tiles of a lake).

The grid points on the edge of a tile are tested in both tiles at the same place in the same coastline (the tiles have
a buffer around them: OpenFreeMap 64 of 4096), so the neighbours agree. The outer ring of the grid (for the normals on
the edges) is 1 cell outside the tile, at z14 exactly on the edge of that buffer: it is tested a little inside it.
 */

/** The ring of the grid is tested this many units of the vector tile's extent inside the tile. */
const RING_INSET = .5;

/**
 * Sets the points of the height grid of a tile in the sea of its vector tile (its water layer) to seaLevel, in place.
 * heights: (cells + 3)^2, the layout of ElevationService.getTileGrid. Returns the number of points set.
 */
export function flattenSea(water: VectorTileLayer | undefined, tile: TileCoord, heights: Float32Array, seaLevel: number): number {
  if (!water) return 0;
  const outer = Math.round(Math.sqrt(heights.length));
  const cells = outer - 3;
  const extent = water.extent || 4096;

  // the grid in the units of the vector tile: the columns (x, from the west) and the rows (y, from the north) as
  // getTileGrid places them, uniform in lat and lng, its rows from the south
  const xs = new Float64Array(outer);
  for (let i = 0; i < outer; i++) xs[i] = (i - 1) / cells * extent;
  xs[0] += RING_INSET;
  xs[outer - 1] -= RING_INSET;
  const sw = tileToLatLon(0, 1, 1, tile), ne = tileToLatLon(1, 0, 1, tile);
  const tileCount = Math.pow(2, tile.z);
  // by row from the north (r = outer - 1 - j), so the y go up
  const ys = new Float64Array(outer);
  for (let r = 0; r < outer; r++) {
    const k = outer - 2 - r;
    // the edges exactly, as the neighbours have them
    if (k === 0) ys[r] = extent;
    else if (k === cells) ys[r] = 0;
    else {
      const lat = sw.lat + (ne.lat - sw.lat) * k / cells;
      const latRad = lat * Math.PI / 180;
      ys[r] = ((1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * tileCount - tile.y) * extent;
    }
  }
  ys[0] += RING_INSET;
  ys[outer - 1] -= RING_INSET;

  // per row the x where the rings of a feature cross it: inside between the 1st and 2nd, the 3rd and 4th, ...
  const crossings: number[][] = [];
  for (let r = 0; r < outer; r++) crossings.push([]);
  const sea = new Uint8Array(heights.length);
  let any = false;
  for (let f = 0; f < water.length; f++) {
    const feature = water.feature(f);
    if (feature.type !== 3 || feature.properties.class !== 'ocean') continue;
    for (const list of crossings) list.length = 0;
    for (const ring of feature.loadGeometry()) {
      for (let e = 0; e < ring.length; e++) {
        const p = ring[e], q = ring[(e + 1) % ring.length];
        if (p.y === q.y) continue;
        const yMin = Math.min(p.y, q.y), yMax = Math.max(p.y, q.y);
        // the rows with yMin <= y < yMax: the edge crosses them (a vertex on a row counts once)
        for (let r = firstAtLeast(ys, yMin); r < outer && ys[r] < yMax; r++) {
          crossings[r].push(p.x + (ys[r] - p.y) * (q.x - p.x) / (q.y - p.y));
        }
      }
    }
    for (let r = 0; r < outer; r++) {
      const list = crossings[r];
      if (list.length < 2) continue;
      list.sort((a, b) => a - b);
      const row = (outer - 1 - r) * outer;
      for (let c = 0; c + 1 < list.length; c += 2) {
        for (let i = firstAtLeast(xs, list[c]); i < outer && xs[i] < list[c + 1]; i++) {
          sea[row + i] = 1;
          any = true;
        }
      }
    }
  }
  if (!any) return 0;

  // A point of the sea next to the land keeps its height, up to the highest land next to it: the coast keeps the slope
  // it had, from the land down to the sea over 1 cell. Snapped to the sea it was a wall of 1 cell, zigzagging along the
  // grid where the land is too high (see above). The other points of the sea are the sea.
  let set = 0;
  for (let j = 0; j < outer; j++) {
    for (let i = 0; i < outer; i++) {
      const k = j * outer + i;
      if (!sea[k]) continue;
      let land = -Infinity;
      if (i > 0 && !sea[k - 1]) land = Math.max(land, heights[k - 1]);
      if (i < outer - 1 && !sea[k + 1]) land = Math.max(land, heights[k + 1]);
      if (j > 0 && !sea[k - outer]) land = Math.max(land, heights[k - outer]);
      if (j < outer - 1 && !sea[k + outer]) land = Math.max(land, heights[k + outer]);
      const h = land === -Infinity ? seaLevel : Math.max(seaLevel, Math.min(heights[k], land));
      if (h !== heights[k]) set++;
      heights[k] = h;
    }
  }
  return set;
}

/** The first index of the ascending values that is >= v (values.length when none is). */
function firstAtLeast(values: Float64Array, v: number): number {
  let lo = 0, hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
