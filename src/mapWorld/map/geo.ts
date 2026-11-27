/** A lat/lng box in degrees. */
export type LatLngBounds = {
  south: number;
  west: number;
  north: number;
  east: number;
};

/** A web mercator (slippy map) tile: zoom, column, row. */
export type TileCoord = { z: number; x: number; y: number };

/**
 * The start region, and the span of the bounds MapWorld centers on a location (see MapManager.boundsAround): its size
 * fixes the world scale of the map (MapProjection).
 */
//https://prochitecture.com/blender-osm/extent/?blender_version=5.1&addon=blosm&addon_version=2.7.27
export const DEFAULT_BOUNDS: LatLngBounds = {
  south: 52.36627,
  west: 4.87739,
  north: 52.38255,
  east: 4.92361,
};

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
