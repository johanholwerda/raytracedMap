import LogGL from 'webGPU-framework/renderer/core/LogGL';
import { LatLngBounds, OSMElement } from './OSMBuilder';
import { decodePdokTile } from './pdokDecodeCore';

/**
 * Streams terrain (roads, paths, water, land-use) from the PDOK
 * "BRT Achtergrondkaart" OGC API vector tiles (Mapbox Vector Tiles, .pbf),
 * derived from the Dutch TOP10NL topography.
 *
 *   https://api.pdok.nl/kadaster/brt-achtergrondkaart/ogc/v1
 *
 * Tiles are served in the standard WebMercatorQuad (slippy z/x/y) scheme, so
 * the usual web-mercator math applies. The actual MVT decode lives in the
 * renderer-free {@link decodePdokTile} (so it can also run in the terrain Web
 * Worker); this service owns the tile math, URL building, fetch and cache.
 *
 * Tiles are cached by `z/x/y`, which makes this ready to later stream tiles in
 * and out as the camera moves.
 */

const TILE_URL =
  'https://api.pdok.nl/kadaster/brt-achtergrondkaart/ogc/v1/tiles/WebMercatorQuad/{z}/{y}/{x}?f=mvt';

// Web-mercator zoom for the terrain tiles. z16 ~= 600 m tiles at NL latitudes,
// close to the ~500-860 m 3DBAG building tiles.
const DEFAULT_ZOOM = 16;

// Safety cap so an accidentally huge region can't fan out into thousands of
// tile requests. At z16, a ~6 km² Amsterdam selection needs ~160 tiles.
const MAX_TILES = 256;

export type TileCoord = { z: number; x: number; y: number };

export default class PDOKTileService {
  private cache: Map<string, OSMElement[]> = new Map();

  /** Fetch + decode every BRT-A tile covering the bounds into OSM elements. */
  public async fetchRegionElements(bounds: LatLngBounds, zoom: number = DEFAULT_ZOOM): Promise<OSMElement[]> {
    const tiles = this.coverTiles(bounds, zoom);
    LogGL.log(`[PDOK] ${tiles.length} terrain tile(s) at z${zoom} cover the region`);

    // Fetch all tiles in parallel — the browser queues excess requests itself
    // (typically 6–8 concurrent per host), so this is safe even for 200+ tiles.
    const results = await Promise.all(
      tiles.map((tile) =>
        this.fetchTile(tile).catch((e) => {
          LogGL.warn(`[PDOK] failed terrain tile ${tile.z}/${tile.x}/${tile.y}:`, e);
          return [] as OSMElement[];
        }),
      ),
    );

    const elements = results.flat();
    LogGL.log(`[PDOK] decoded ${elements.length} terrain features from ${tiles.length} tile(s)`);
    return elements;
  }

  /** Fetch + decode a single tile (cached by z/x/y). */
  public async fetchTileElements(tile: TileCoord): Promise<OSMElement[]> {
    return this.fetchTile(tile);
  }

  /** Full MVT URL for a tile (used by the terrain worker, which fetches itself). */
  public tileUrl(tile: TileCoord): string {
    return TILE_URL.replace('{z}', String(tile.z)).replace('{x}', String(tile.x)).replace('{y}', String(tile.y));
  }

  /** Tiles covering a lat/lng bounds at the given zoom. */
  public coverTilesForBounds(bounds: LatLngBounds, zoom: number = DEFAULT_ZOOM): TileCoord[] {
    return this.coverTiles(bounds, zoom);
  }

  /** Web-mercator tile coord for a lat/lng at the given zoom. */
  public latLngToTile(lat: number, lng: number, zoom: number): TileCoord {
    return { z: zoom, x: lonToTileX(lng, zoom), y: latToTileY(lat, zoom) };
  }

  /** Lat/lng bounds of a web-mercator tile. */
  public tileBounds(tile: TileCoord): LatLngBounds {
    const n = Math.pow(2, tile.z);
    const west = (tile.x / n) * 360 - 180;
    const east = ((tile.x + 1) / n) * 360 - 180;
    const northRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * tile.y) / n)));
    const southRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * (tile.y + 1)) / n)));
    const north = (northRad * 180) / Math.PI;
    const south = (southRad * 180) / Math.PI;
    return { south, west, north, east };
  }

  /** Cache key for a tile coord. */
  public tileKey(tile: TileCoord): string {
    return `${tile.z}/${tile.x}/${tile.y}`;
  }

  /** Fetch + decode a single tile (cached by z/x/y). */
  private async fetchTile(tile: TileCoord): Promise<OSMElement[]> {
    const key = this.tileKey(tile);
    const cached = this.cache.get(key);
    if (cached) return cached;

    const response = await fetch(this.tileUrl(tile));
    // Empty / out-of-coverage tiles are normal: PDOK returns 204 or 404.
    if (response.status === 204 || response.status === 404) {
      this.cache.set(key, []);
      return [];
    }
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();
    const elements = decodePdokTile(new Uint8Array(buffer), tile);
    this.cache.set(key, elements);
    return elements;
  }

  /** Web-mercator tiles (z/x/y) overlapping the bounds, capped at MAX_TILES. */
  private coverTiles(bounds: LatLngBounds, zoom: number): TileCoord[] {
    const xMin = lonToTileX(bounds.west, zoom);
    const xMax = lonToTileX(bounds.east, zoom);
    const yMin = latToTileY(bounds.north, zoom); // north is the smaller row
    const yMax = latToTileY(bounds.south, zoom);

    const tiles: TileCoord[] = [];
    for (let x = xMin; x <= xMax; x++) {
      for (let y = yMin; y <= yMax; y++) {
        tiles.push({ z: zoom, x, y });
        if (tiles.length >= MAX_TILES) {
          LogGL.warn(`[PDOK] region exceeds ${MAX_TILES} tiles at z${zoom}; truncating`);
          return tiles;
        }
      }
    }
    return tiles;
  }
}

// --- web-mercator helpers ---------------------------------------------------

function lonToTileX(lon: number, z: number): number {
  return Math.floor(((lon + 180) / 360) * Math.pow(2, z));
}

function latToTileY(lat: number, z: number): number {
  const rad = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * Math.pow(2, z));
}
