import { ELEVATION_ZOOM } from './ElevationService';
import type { FetchWorkerEvent, FetchWorkerMessage } from './tileFetchWorker';
import type { TileCoord } from '../osm/PDOKTileService';

/**
 * The main thread's side of the elevation: the elevation tiles are fetched, decoded and kept in the tile fetch worker
 * (tileFetchWorker.ts, an ElevationService there), which this class starts. Here: where the heights are relative to
 * (setOrigin), what to load ahead (preload), and what the worker tells back: the height of the origin and the range of
 * the heights of the elevation tiles it has loaded (tileRange, for the tile-vs-frustum test of WorldTileStreamer). No
 * elevation data is on the main thread but the grids that come with the built tiles (see TerrainHeightField).
 *
 * The same worker fetches the map tiles for the WorldWorkerPool, which gets its ports to it here (createPort).
 */
export default class ElevationClient {
  // the elevation tiles kept at most: a tile is 2 numbers here
  private static readonly RANGES_MAX = 4096;

  private worker: Worker;
  // per elevation tile (x/y at ELEVATION_ZOOM) the lowest and highest height in it, m above sea level
  private ranges = new Map<string, [number, number]>();
  private originLat = NaN;
  private originLng = NaN;
  private originM = 0;

  constructor() {
    this.worker = new Worker(new URL('./tileFetchWorker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FetchWorkerEvent>) => {
      const msg = e.data;
      if (msg.type === 'range') {
        this.ranges.set(`${msg.x}/${msg.y}`, [msg.min, msg.max]);
        if (this.ranges.size > ElevationClient.RANGES_MAX) this.ranges.delete(this.ranges.keys().next().value as string);
      } else if (msg.lat === this.originLat && msg.lng === this.originLng) {
        // of the origin asked for last: a tile of the place before can still be on its way
        this.originM = msg.height;
      }
    };
  }

  /**
   * The ground height (m above sea level) the heights are relative to: the one of the last setOrigin once the worker
   * has it, until then the one before (0 at first).
   */
  public get originHeight(): number {
    return this.originM;
  }

  /** The heights are relative to the ground at lat/lng from now on: the projection center. */
  public setOrigin(lat: number, lng: number): void {
    this.originLat = lat;
    this.originLng = lng;
    this.post({ type: 'origin', lat, lng });
  }

  /**
   * Starts loading the elevation tiles around lat/lng (ElevationService.preload), for a place whose map tiles are asked
   * for soon: on the first load of the page before the URL of the map tiles is known.
   */
  public preload(lat: number, lng: number): void {
    this.post({ type: 'preload', lat, lng });
  }

  /**
   * The range of the ground heights (m, relative to the origin) around a map tile: of the whole elevation tile it is
   * in. Null while the worker hasn't loaded that.
   */
  public tileRange(tile: TileCoord): [number, number] | null {
    const shift = tile.z - ELEVATION_ZOOM;
    const x = shift >= 0 ? tile.x >> shift : tile.x << -shift;
    const y = shift >= 0 ? tile.y >> shift : tile.y << -shift;
    const range = this.ranges.get(`${x}/${y}`);
    return range ? [range[0] - this.originM, range[1] - this.originM] : null;
  }

  /**
   * A port to the worker, for the WorldWorkerPool: 'jobs' the one the pool sends its jobs on, 'builder' one for a
   * build worker to get its tiles on (see tileFetchWorker.ts). Returns this side of it.
   */
  public createPort(type: 'jobs' | 'builder'): MessagePort {
    const channel = new MessageChannel();
    this.worker.postMessage({ type, port: channel.port1 } satisfies FetchWorkerMessage, [channel.port1]);
    return channel.port2;
  }

  public destroy(): void {
    this.worker.terminate();
  }

  private post(message: FetchWorkerMessage): void {
    this.worker.postMessage(message);
  }
}
