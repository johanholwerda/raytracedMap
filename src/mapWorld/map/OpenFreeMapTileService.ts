import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type { LatLngBounds } from '../../mapOSM/osmGeometryCore';
import type { TileCoord } from '../../mapOSM/PDOKTileService';

/**
 * Worldwide vector tiles from OpenFreeMap (https://openfreemap.org): free, no
 * API key, CORS enabled, OpenMapTiles schema built from OpenStreetMap. One tile
 * carries both the ground layers (water, landcover, roads, …) and the building
 * footprints with heights, so the world map needs a single request per tile.
 *
 * The tile URL contains the planet build date and changes with every weekly
 * rebuild, so it is read from the TileJSON at startup ({@link init}).
 */

const TILEJSON_URL = 'https://tiles.openfreemap.org/planet';
// Only used when the TileJSON can't be fetched; may go stale over time.
const FALLBACK_TILE_URL = 'https://tiles.openfreemap.org/planet/20261004_113936_pt/{z}/{x}/{y}.pbf';

export const ATTRIBUTION = '© OpenFreeMap © OpenMapTiles, data © OpenStreetMap contributors';

export default class OpenFreeMapTileService {
  private template: string = FALLBACK_TILE_URL;
  /** Highest zoom the source serves (buildings exist at z13–14). */
  public maxZoom: number = 14;

  /** Resolve the current tile URL template from the TileJSON. */
  public async init(): Promise<void> {
    try {
      const res = await fetch(TILEJSON_URL);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const json = await res.json();
      if (Array.isArray(json.tiles) && typeof json.tiles[0] === 'string') this.template = json.tiles[0];
      if (typeof json.maxzoom === 'number') this.maxZoom = json.maxzoom;
      LogGL.log(`[MapWorld] OpenFreeMap tiles: ${this.template} (max z${this.maxZoom})`);
    } catch (e) {
      LogGL.warn('[MapWorld] could not read OpenFreeMap TileJSON, using fallback URL:', e);
    }
  }

  /** Tile URL. `x` is wrapped so tiles across the antimeridian resolve. */
  public tileUrl(tile: TileCoord): string {
    const n = Math.pow(2, tile.z);
    const x = ((tile.x % n) + n) % n;
    return this.template.replace('{z}', String(tile.z)).replace('{x}', String(x)).replace('{y}', String(tile.y));
  }

  /** True when the tile row exists (web-mercator has no rows past ±85°). */
  public isValid(tile: TileCoord): boolean {
    return tile.y >= 0 && tile.y < Math.pow(2, tile.z);
  }

  public latLngToTile(lat: number, lng: number, zoom: number): TileCoord {
    const n = Math.pow(2, zoom);
    const x = Math.floor(((lng + 180) / 360) * n);
    const rad = (lat * Math.PI) / 180;
    const y = Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * n);
    return { z: zoom, x, y };
  }

  public tileBounds(tile: TileCoord): LatLngBounds {
    const n = Math.pow(2, tile.z);
    const west = (tile.x / n) * 360 - 180;
    const east = ((tile.x + 1) / n) * 360 - 180;
    const north = (Math.atan(Math.sinh(Math.PI * (1 - (2 * tile.y) / n))) * 180) / Math.PI;
    const south = (Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + 1)) / n))) * 180) / Math.PI;
    return { south, west, north, east };
  }

  public tileKey(tile: TileCoord): string {
    return `${tile.z}/${tile.x}/${tile.y}`;
  }
}
