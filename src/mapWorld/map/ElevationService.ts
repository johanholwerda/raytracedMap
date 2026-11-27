import LogGL from 'webGPU-framework/renderer/core/LogGL';
import decodePngRgba8 from 'webGPU-framework/utils/image/decodePngRgba8';
import { tileToLatLon, type TileCoord } from './geo';
import { demPixelMeters, removeDemSpikes } from './demSpikes';
import { COARSE_ZOOM, correctArcticDem, mayBeArcticDem } from './arcticDem';

/*
The height of the ground, from the terrarium elevation tiles of the AWS Open Data terrain tiles (Mapzen, Tilezen joerd):
256 x 256 png in web mercator, the height in m = R * 256 + G + B / 256 - 32768. Free, no key, CORS enabled. The sources
differ per region (https://github.com/tilezen/joerd/blob/master/docs/data-sources.md): in the US the NED (bare earth, 3
to 10 m), elsewhere mostly SRTM (30 to 90 m) and the like, which are surface models: buildings and forests are in it
as bumps, smoothed at the resolution used here. North of 60 degrees it is ArcticDEM in places (Iceland): heights above
the ellipsoid, not the sea, which correctArcticDem corrects with the tile at COARSE_ZOOM (see arcticDem.ts), and false
mountains on the water, which removeDemSpikes fills in (see demSpikes.ts) and the sea of the map tiles flattens (see
seaMask.ts).

The tiles are fetched and decoded in the tile fetch worker (tileFetchWorker.ts, the main thread talks to it through
ElevationClient), once per ELEVATION_ZOOM tile (a z12 tile covers 4 x 4 of the z14
map tiles). For each map tile getTileGrid samples a grid of terrainGridCells + 1 heights over it, uniform in lat and
lng (the projection is equirectangular, so uniform in world x and z too), which the worker builds the ground mesh from
and drapes the ground layers, buildings and trees on (see terrainGrid.ts). The grids of 2 neighbours sample their
shared edge at the same lat/lng, so they meet without cracks.

Heights are relative to the ground at the projection center (setOrigin): there the ground is at y 0, as before, so the
camera pivot and everything that assumes y 0 stays close to the ground around the center.
 */

const TERRARIUM_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png';
// the sources are listed in the data-sources doc above
export const ELEVATION_ATTRIBUTION = 'Terrain Tiles © Mapzen';
/** The zoom of the elevation tiles: about 38 m per pixel at the equator, 24 m at 50 degrees. */
export const ELEVATION_ZOOM = 12;
const DEM_SIZE = 256;
/**
 * The cells per side of the height grid of a map tile at zoom z, see getTileGrid: 1 per elevation pixel, so the grid
 * has the resolution of the data (64 at z14: about 38 m at the equator, 24 m at 52 degrees). Its points are on the
 * corners of the pixels, each the mean of the 4 pixels around it: the edges of the map tiles are on pixel edges. The
 * draped ground of a tile took 1.7 to 3.4 MB (San Francisco, Amsterdam), 0.3 to 0.4 MB flat; half of that with a cell
 * per 2 x 2 pixels.
 */
export function terrainGridCells(z: number): number {
  return Math.max(1, DEM_SIZE >> Math.max(0, z - ELEVATION_ZOOM));
}
// decoded tiles kept (256 KB each)
const CACHE_MAX = 96;
// the coarse tiles kept for correctArcticDem: 1 covers 4 x 4 tiles
const COARSE_CACHE_MAX = 8;
// The heights are clamped to the sea level: the tiles have the depth of the sea (ETOPO1), the water layer lies on the
// ground. Land below the sea level (polders, the Dead Sea) becomes flat at 0.
const MIN_HEIGHT_M = 0;

type DemTile = { heights: Float32Array; min: number; max: number };
/** The heights of a map tile (see getTileGrid) and the height of the sea in them: both relative to the origin. */
export type TileGrid = { heights: Float32Array; seaLevel: number };

export default class ElevationService {
  // per tile key: the tile, null where there is none (404). LRU: a hit moves the key to the end
  private tiles = new Map<string, Promise<DemTile | null>>();
  private decoded = new Map<string, DemTile | null>();
  // the tiles at COARSE_ZOOM (see arcticDem.ts) by key, LRU
  private coarse = new Map<string, Promise<Float32Array | null>>();
  private origin: Promise<number> = Promise.resolve(0);
  private originM = 0;
  /**
   * Called when an elevation tile (x, y at ELEVATION_ZOOM) is loaded, with the lowest and highest height in it (m above
   * sea level, both 0 where there is no tile): for ElevationClient.tileRange on the main thread.
   */
  public onTile: ((x: number, y: number, min: number, max: number) => void) | null = null;

  /** The origin of the last setOrigin (m above sea level), when it is loaded. */
  public get originLoaded(): Promise<number> {
    return this.origin;
  }

  /** The ground height (m above sea level) the heights are relative to, 0 until it is loaded, see setOrigin. */
  public get originHeight(): number {
    return this.originM;
  }

  /** The ground height (m) the heights are relative to: the one at lat/lng, the projection center. */
  public setOrigin(lat: number, lng: number): void {
    const origin = this.sample(new Float64Array([lat]), new Float64Array([lng]))
      .then((h) => h[0])
      .catch((e) => {
        LogGL.warn('[MapWorld] no elevation at the center, heights from sea level:', e);
        return 0;
      });
    this.origin = origin;
    void origin.then((h) => {
      if (this.origin === origin) this.originM = h;
    });
  }

  /**
   * Starts loading the elevation tiles around lat/lng: the one it is in and rings of them around it. For a place whose
   * map tiles are asked for soon (see MapManager.setPlace): getTileGrid then finds them loading or loaded. On the first
   * load of the page the elevation tiles are what a map tile waits for longest (about 0.4 s each), and they can be
   * asked for before the URL of the map tiles is known (OpenFreeMapTileService.init, another 0.35 s).
   */
  public preload(lat: number, lng: number, rings = 1): void {
    const n = 1 << ELEVATION_ZOOM;
    const latRad = Math.max(Math.min(lat, 85.05), -85.05) * Math.PI / 180;
    const x = Math.floor((lng + 180) / 360 * n);
    const y = Math.floor((1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * n);
    for (let dy = -rings; dy <= rings; dy++) {
      for (let dx = -rings; dx <= rings; dx++) {
        if (y + dy < 0 || y + dy >= n) continue;
        // a tile that fails is asked for again by the map tile that needs it
        this.load(this.wrap(x + dx, n), y + dy).catch(() => {});
      }
    }
  }

  /**
   * The heights (m, relative to the origin) of the grid over a map tile with a ring of the neighbour tiles' around it:
   * (cells + 3)^2, row j (from the south, -1 to cells + 1) after row, column i from the west. Lat and lng are uniform
   * over the tile, row and column 0 and cells at its edges exactly. The ring is for the normals on the edges, see
   * TerrainGrid. With the height of the sea relative to the same origin (see seaMask.ts). Rejects when an elevation
   * tile can't be loaded (the map tile retries later).
   */
  public async getTileGrid(tile: TileCoord, cells = terrainGridCells(tile.z)): Promise<TileGrid> {
    const sw = tileToLatLon(0, 1, 1, tile), ne = tileToLatLon(1, 0, 1, tile);
    const outer = cells + 3;
    const at = (k: number, from: number, to: number) => k === 0 ? from : k === cells ? to : from + (to - from) * k / cells;
    const lats = new Float64Array(outer * outer), lngs = new Float64Array(outer * outer);
    for (let j = 0; j < outer; j++) {
      const lat = at(j - 1, sw.lat, ne.lat);
      for (let i = 0; i < outer; i++) {
        lats[j * outer + i] = lat;
        lngs[j * outer + i] = at(i - 1, sw.lon, ne.lon);
      }
    }
    const origin = this.origin;
    const heights = await this.sample(lats, lngs);
    const h0 = await origin;
    for (let k = 0; k < heights.length; k++) heights[k] -= h0;
    return { heights, seaLevel: MIN_HEIGHT_M - h0 };
  }

  /**
   * The range of the ground heights (m, relative to the origin) around a map tile: of the whole elevation tile it is
   * in. Null while that isn't loaded.
   */
  public tileRange(tile: TileCoord): [number, number] | null {
    const shift = tile.z - ELEVATION_ZOOM;
    const x = shift >= 0 ? tile.x >> shift : tile.x << -shift;
    const y = shift >= 0 ? tile.y >> shift : tile.y << -shift;
    const dem = this.decoded.get(this.key(x, y));
    if (dem === undefined) return null;
    if (dem === null) return [-this.originM, -this.originM];
    return [dem.min - this.originM, dem.max - this.originM];
  }

  /** The heights (m) at the points, bilinear between the pixel centers of the elevation tiles. */
  private async sample(lats: Float64Array, lngs: Float64Array): Promise<Float32Array> {
    const n = 1 << ELEVATION_ZOOM;
    const world = n * DEM_SIZE;
    const count = lats.length;
    // the pixel coordinates, from the pixel centers, and the range of pixels the bilinear samples read
    const px = new Float64Array(count), py = new Float64Array(count);
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let k = 0; k < count; k++) {
      const latRad = Math.max(Math.min(lats[k], 85.05), -85.05) * Math.PI / 180;
      const x = px[k] = (lngs[k] + 180) / 360 * world - .5;
      const y = py[k] = (1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * world - .5;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    // the tiles of that range (by ty * n + tx): the points are close together (a map tile and its ring), so a few
    const tx0 = Math.floor(Math.floor(xMin) / DEM_SIZE), tx1 = Math.floor((Math.floor(xMax) + 1) / DEM_SIZE);
    const ty0 = Math.min(Math.max(Math.floor(Math.floor(yMin) / DEM_SIZE), 0), n - 1);
    const ty1 = Math.min(Math.max(Math.floor((Math.floor(yMax) + 1) / DEM_SIZE), 0), n - 1);
    const needed: [number, number][] = [];
    for (let ty = ty0; ty <= ty1; ty++) {
      for (let tx = tx0; tx <= tx1; tx++) needed.push([this.wrap(tx, n), ty]);
    }
    const tiles = new Map<number, DemTile | null>();
    await Promise.all(needed.map(async ([tx, ty]) => tiles.set(ty * n + tx, await this.load(tx, ty))));

    // the tile of the last pixel is mostly the one of the next
    let lastKey = -1;
    let lastDem: DemTile | null = null;
    const pixel = (x: number, y: number): number => {
      const tileX = Math.floor(x / DEM_SIZE);
      const yc = Math.min(Math.max(y, 0), world - 1);
      const ty = Math.floor(yc / DEM_SIZE);
      const key = ty * n + this.wrap(tileX, n);
      if (key !== lastKey) {
        lastKey = key;
        lastDem = tiles.get(key) ?? null;
      }
      if (!lastDem) return MIN_HEIGHT_M;
      return lastDem.heights[(yc - ty * DEM_SIZE) * DEM_SIZE + x - tileX * DEM_SIZE];
    };
    const out = new Float32Array(count);
    for (let k = 0; k < count; k++) {
      const x0 = Math.floor(px[k]), y0 = Math.floor(py[k]);
      const fx = px[k] - x0, fy = py[k] - y0;
      const top = pixel(x0, y0) * (1 - fx) + pixel(x0 + 1, y0) * fx;
      const bottom = pixel(x0, y0 + 1) * (1 - fx) + pixel(x0 + 1, y0 + 1) * fx;
      out[k] = top * (1 - fy) + bottom * fy;
    }
    return out;
  }

  private wrap(x: number, n: number): number {
    return ((x % n) + n) % n;
  }

  private key(x: number, y: number): string {
    return `${x}/${y}`;
  }

  /** The elevation tile (x, y) at ELEVATION_ZOOM, cached. Null where there is none, rejects on a network error. */
  private load(x: number, y: number): Promise<DemTile | null> {
    const key = this.key(x, y);
    const cached = this.tiles.get(key);
    if (cached) {
      this.tiles.delete(key);
      this.tiles.set(key, cached);
      return cached;
    }
    const promise = (async (): Promise<DemTile | null> => {
      const heights = await fetchHeights(ELEVATION_ZOOM, x, y);
      if (!heights) return null;
      // ArcticDEM: above the ellipsoid, see arcticDem.ts
      if (mayBeArcticDem(x, y, ELEVATION_ZOOM)) {
        const shift = ELEVATION_ZOOM - COARSE_ZOOM;
        const coarse = await this.loadCoarse(x >> shift, y >> shift);
        if (coarse) correctArcticDem(heights, coarse, DEM_SIZE, x, y, ELEVATION_ZOOM);
      }
      // the false mountains on the sea, see demSpikes.ts
      removeDemSpikes(heights, DEM_SIZE, demPixelMeters(y, ELEVATION_ZOOM, DEM_SIZE));
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < heights.length; i++) {
        const h = heights[i];
        if (h < min) min = h;
        if (h > max) max = h;
      }
      return { heights, min, max };
    })();
    this.tiles.set(key, promise);
    promise.then(
      (dem) => {
        if (this.tiles.get(key) === promise) this.decoded.set(key, dem);
        this.onTile?.(x, y, dem ? dem.min : MIN_HEIGHT_M, dem ? dem.max : MIN_HEIGHT_M);
      },
      () => { if (this.tiles.get(key) === promise) this.tiles.delete(key); },
    );
    while (this.tiles.size > CACHE_MAX) {
      const oldest = this.tiles.keys().next().value as string;
      this.tiles.delete(oldest);
      this.decoded.delete(oldest);
    }
    return promise;
  }

  /** The tile (x, y) at COARSE_ZOOM, for correctArcticDem: cached (LRU), null where there is none. */
  private loadCoarse(x: number, y: number): Promise<Float32Array | null> {
    const key = this.key(x, y);
    const cached = this.coarse.get(key);
    if (cached) {
      this.coarse.delete(key);
      this.coarse.set(key, cached);
      return cached;
    }
    const promise = fetchHeights(COARSE_ZOOM, x, y);
    this.coarse.set(key, promise);
    // a network error: asked for again by the next tile
    promise.catch(() => { if (this.coarse.get(key) === promise) this.coarse.delete(key); });
    while (this.coarse.size > COARSE_CACHE_MAX) this.coarse.delete(this.coarse.keys().next().value as string);
    return promise;
  }
}

/** The heights (m) of the terrarium tile z/x/y, clamped to the sea level. Null where there is none, rejects on a network error. */
async function fetchHeights(z: number, x: number, y: number): Promise<Float32Array | null> {
  const url = TERRARIUM_URL.replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y));
  const response = await fetch(url);
  if (response.status === 404 || response.status === 403) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const png = await decodePngRgba8(await response.arrayBuffer());
  if (png.width !== DEM_SIZE || png.height !== DEM_SIZE) throw new Error(`elevation tile ${z}/${x}/${y} is ${png.width} x ${png.height}`);
  const heights = new Float32Array(DEM_SIZE * DEM_SIZE);
  const d = png.data;
  for (let i = 0; i < heights.length; i++) {
    heights[i] = Math.max(d[i * 4] * 256 + d[i * 4 + 1] + d[i * 4 + 2] / 256 - 32768, MIN_HEIGHT_M);
  }
  return heights;
}
