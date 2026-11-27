import LogGL from 'webGPU-framework/renderer/core/LogGL';
import type Vector3 from 'webGPU-framework/renderer/math/Vector3';
import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import type ParamBool from 'webGPU-framework/utils/params/ParamBool';
import type ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import type ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import { type LatLngBounds, DEFAULT_BOUNDS } from './geo';
import MapProjection from './MapProjection';
import { NOTABLE_BUILDINGS } from './notableBuildings';
import { describeCity, findCity, pickRandomCity, pickStartCity } from './worldCities';
import type { City } from './worldCities';
import { randomNiceView } from './niceViews';
import OpenFreeMapTileService from './OpenFreeMapTileService';
import WorldWorkerPool from './WorldWorkerPool';
import WorldTileStreamer, { PreviewCamera } from './WorldTileStreamer';
import ElevationClient from './ElevationClient';
import TerrainHeightField from './TerrainHeightField';
import type { CityMeshes } from '../rendering/MapGeometry';
import { CAMERA_FOV, CAMERA_NEAR, startCameraView } from '../camera/MapCameraRig';
import { parseUrlCamera, type UrlCamera } from '../MapUrlSync';

// The frames the page draws with 1 map worker before the others join in, see startTiles: the first frames compile
// the pipelines, which the GPU process does on the cores the workers would take
const START_FRAMES = 4;

/** What the MapManager tells the rest of the map. */
export type MapManagerHooks = {
  // a tile failed to load
  onTileError: (msg: string) => void;
  // a param changed what is drawn: the map draws again
  onRedraw: () => void;
  // the buildings or the trees came or went: the map draws again, the AO starts over
  onContentChanged: () => void;
  // the tiles the selected building was in are gone
  onClearSelection: () => void;
  // the elevation param changed: see MapWorld.applyElevation
  onElevationChanged: () => void;
};

/** A place to go to, see pickRandomPlace. */
export type MapPlace = { lat: number; lng: number; label: string; title: string; building?: string };

/**
 * The map data, none of it on the GPU: the place the map is centered on (the projection), the tiles streamed around
 * the camera (WorldTileStreamer on the WorldWorkerPool, from OpenFreeMap), the heights of the ground (ElevationClient,
 * TerrainHeightField) and the params of what is loaded (buildings, trees, elevation). It starts before the GPU, see
 * startTiles: MapWorld attaches the streamer to the renderer once that is there.
 */
export default class MapManager {
  public readonly projection: MapProjection = new MapProjection();
  public workerPool!: WorldWorkerPool;
  public tileStreamer!: WorldTileStreamer;
  // the ground of the loaded tiles, for the cars, the camera pivot and the plane (empty with the elevation off: flat 0)
  public heightField!: TerrainHeightField;
  // The heights of the ground: the tiles are built on it while the elevation param is on, see setElevation
  public readonly elevation: ElevationClient = new ElevationClient();
  private bounds: LatLngBounds = DEFAULT_BOUNDS;
  private tileService: OpenFreeMapTileService = new OpenFreeMapTileService();
  // the frames to draw before all map workers work, see startTiles
  private startFrames: number = START_FRAMES;
  // SSAO "loading" hysteresis across the short idle gaps between tile batches.
  private loadingCooldown: number = 0;

  private drawBuildings?: ParamBool;
  private elevationParam?: ParamBool;
  // the trees of a tile are left out of the gBuffer where the sphere around the largest is smaller than this many
  // pixels, see WorldTileStreamer.cullTrees. The sphere is 23 m at most: 1 pixel is beyond the city tiles of the farthest
  // camera. Measured (video/treeCullBench.mjs, the Veluwe at 1.8 km, 1264x649): 3 leaves out the trees of half the tiles,
  // the gBuffer 0.60 -> 0.36 ms per frame (no trees: 0.22), the image differs 0.2 / 255 on average, at the horizon:
  // their shadows and the sky visibility of the lightmap keep them
  private treeCullPixels!: ParamFloat;

  constructor(private hooks: MapManagerHooks) {}

  /**
   * What the tiles need, none of it the GPU, and the requests for the tiles of the view the map starts with: the
   * workers, the tile service, the place (the projection centered on it) and the streamer, which asks for the tiles
   * of a PreviewCamera with that view (WorldTileStreamer.prefetch). Returns the camera of the URL (null without) and
   * the city the map starts on (undefined with a URL camera).
   * @param viewDistance the viewDistanceMax of the quality the map starts with
   */
  public async startTiles(urlParams: URLSearchParams, canvasParent: HTMLElement, viewDistance: number):
    Promise<{ urlCamera: UrlCamera | null; city?: City; time?: string }> {
    // ?workers=4: the number of map workers, for timings (default: see WorldWorkerPool)
    // ?jobsPerWorker=3: the jobs under way per worker, for timings (see WorldWorkerPool.capacity)
    if (urlParams.has('jobsPerWorker')) WorldWorkerPool.JOBS_PER_WORKER = parseFloat(urlParams.get('jobsPerWorker')!);
    this.workerPool = new WorldWorkerPool(parseInt(urlParams.get('workers') ?? '') || undefined, this.elevation);
    // 1 of them until the map has drawn its first START_FRAMES frames (frameSubmitted lifts it): until then the GPU
    // process starts up and compiles the pipelines next to them. With all cores busy the WebGPU device took 2 s instead
    // of 0.6 s (and now and then over 4 s on a machine that was busy with other things, also with 2 workers), with 1 it
    // comes about as without. And with all workers from the device on, the first tiles were on screen after 1.5 s
    // instead of 1 s: the first frames took longer. ?earlyWorkers=2 for timings
    const early = parseInt(urlParams.get('earlyWorkers') ?? '');
    this.workerPool.activeLimit = Number.isFinite(early) ? early : 1;
    // Reads the current OpenFreeMap tile URL (it changes with each planet build).
    const tileService = this.tileService.init();
    this.heightField = new TerrainHeightField(this.projection);
    this.tileStreamer = new WorldTileStreamer({
      projection: this.projection,
      workerPool: this.workerPool,
      tileService: this.tileService,
      onTileError: (msg) => this.hooks.onTileError(msg),
      heightField: this.heightField,
    });
    // on by default, see the elevation param
    this.tileStreamer.setElevation(this.elevation);
    // ?prefetchMargin=1: no tiles asked for ahead of the view, for timings (see WorldTileStreamer.PREFETCH_MARGIN)
    if (urlParams.has('prefetchMargin')) WorldTileStreamer.PREFETCH_MARGIN = parseFloat(urlParams.get('prefetchMargin')!);
    // ?tileTimings: the way of each tile from wanted to drawn, for video/tileLatency.mjs
    if (urlParams.has('tileTimings')) this.tileStreamer.timings = [];

    // Start on one of a few well-known cities (pickStartCity). The projection
    // is centered on it so vertex coordinates stay small (fp32 precision).
    // ?lat=&lng=(&dist=&pitch=&yaw=): the camera of a shared link, see MapUrlSync. A bench starts its drags there. Not
    // in a video, it moves the camera itself.
    let urlCamera = urlParams.has('video') ? null : parseUrlCamera(urlParams);
    let city: City | undefined;
    // a plain page load: half the time one of the NICE_VIEWS (with its time of day) instead of a start city. Not in a
    // bench, a video, a tile timing run or with ?city=: those want the same start every time
    let time: string | undefined;
    const plainLoad = !['video', 'bench', 'tileTimings', 'city'].some((key) => urlParams.has(key));
    if (!urlCamera && plainLoad && Math.random() < 0.5) {
      const niceView = randomNiceView();
      urlCamera = niceView;
      time = niceView.time;
    }
    if (urlCamera) {
      this.setPlace(this.boundsAround(urlCamera.lat, urlCamera.lng));
      LogGL.log(`[MapWorld] centering on the URL camera (${urlCamera.lat}, ${urlCamera.lng})`);
    } else {
      // ?city=Paris: always the same city, for timings
      const cityName = urlParams.get('city');
      city = (cityName ? await findCity(cityName) : undefined) ?? pickStartCity();
      this.setPlace(this.boundsAround(city.lat, city.lng));
      LogGL.log(`[MapWorld] centering on ${describeCity(city)}`);
    }
    await tileService;

    // The view of the first frame: the camera MapCameraRig starts with and the camera of the URL (applyUrlCamera), the
    // view distance of the quality init starts with (the far plane as updateFarPlane has it), the canvas as wide as
    // its parent
    const view = startCameraView(urlCamera, this.projection.metersToNorm);
    const height = Math.max(view.distance * Math.sin(-view.pitch), 0);
    const size = canvasParent.getBoundingClientRect();
    const preview = new PreviewCamera(CAMERA_FOV, CAMERA_NEAR, Math.hypot(viewDistance, height), view.distance, view.pitch, view.yaw,
      size.width / Math.max(size.height, 1) || 1);
    this.tileStreamer.setViewDistanceMax(viewDistance);
    this.tileStreamer.setTerrainStreamingEnabled(true);
    this.tileStreamer.setCityStreamingEnabled(true);
    this.tileStreamer.prefetch(preview);
    return { urlCamera, city, time };
  }

  /** The params of what is loaded, in the Layers group: the buildings, the trees and the elevation. */
  public initParams(layersGroup: ParamGroup): void {
    const drawBuildings = layersGroup.addParamBool('buildings', true, { callback: () => {
      this.hooks.onContentChanged();
      if (drawBuildings.value) {
        this.tileStreamer.setCityStreamingEnabled(true);
        this.tileStreamer.resetStreamingPosition();
      } else {
        this.tileStreamer.setCityStreamingEnabled(false);
        this.tileStreamer.clearAllCityTiles();
        this.hooks.onClearSelection();
      }
    } });
    this.drawBuildings = drawBuildings;
    // in the building tiles, see worldTrees.ts: they load again
    const trees = layersGroup.addParamBool('trees', true, { callback: () => {
      this.hooks.onContentChanged();
      this.hooks.onClearSelection();
      this.tileStreamer.setTreesEnabled(trees.value);
    } });
    this.tileStreamer.setTreesEnabled(trees.value);
    this.treeCullPixels = layersGroup.addParamFloat('treeCullPixels', 3, 0, 8, { callback: () => this.hooks.onRedraw() });
    // the heights of the ground (ElevationService), off: the flat ground at 0. The tiles load again
    this.elevationParam = layersGroup.addParamBool('elevation', true, { callback: () => this.hooks.onElevationChanged() });
  }

  /** Whether the elevation param is on (also before init: on by default). */
  public get elevationEnabled(): boolean {
    return this.elevationParam?.value !== false;
  }

  /** Whether the buildings param is on (also before init: on by default). */
  public get buildingsEnabled(): boolean {
    return this.drawBuildings?.value !== false;
  }

  /** The building meshes to draw, null while the buildings are hidden. */
  public get visibleCity(): CityMeshes | null {
    if (!this.buildingsEnabled) return null;
    const s = this.tileStreamer;
    return { meshes: s.cityMeshes, slots: s.cityMeshSlots, tiles: s.cityMeshTiles, quantization: s.cityMeshQuantization, indexCounts: s.cityMeshIndexCounts };
  }

  /** Bounds centered on the camera pivot, keeping the current span. */
  public currentViewBounds(pivot: Vector3): LatLngBounds {
    const { lat, lng } = this.projection.worldToLatLng(pivot.x, pivot.z);
    const halfLat = (this.bounds.north - this.bounds.south) * 0.5;
    const halfLng = (this.bounds.east - this.bounds.west) * 0.5;
    return { south: lat - halfLat, north: lat + halfLat, west: lng - halfLng, east: lng + halfLng };
  }

  /** Bounds centered on lat/lng with the DEFAULT_BOUNDS span (same world scale as NL). */
  public boundsAround(lat: number, lng: number): LatLngBounds {
    const halfLat = (DEFAULT_BOUNDS.north - DEFAULT_BOUNDS.south) * 0.5;
    const halfLng = (DEFAULT_BOUNDS.east - DEFAULT_BOUNDS.west) * 0.5;
    return { south: lat - halfLat, north: lat + halfLat, west: lng - halfLng, east: lng + halfLng };
  }

  /**
   * A random city (weighted by sqrt(population)) or, half the time, a uniformly picked notable building. Avoids
   * the place around the pivot.
   */
  public async pickRandomPlace(pivot: Vector3): Promise<MapPlace> {
    const current = this.projection.worldToLatLng(pivot.x, pivot.z);
    const isHere = (p: { lat: number; lng: number }) => Math.abs(p.lat - current.lat) + Math.abs(p.lng - current.lng) < 0.5;
    if (Math.random() < 0.5) {
      const pick = () => NOTABLE_BUILDINGS[Math.floor(Math.random() * NOTABLE_BUILDINGS.length)];
      let notable = pick();
      for (let i = 0; i < 8 && isHere(notable); i++) notable = pick();
      return {
        lat: notable.lat,
        lng: notable.lng,
        label: `${notable.name}, ${notable.city}, ${notable.country}`,
        title: `${notable.city}, ${notable.country}`,
        building: notable.name,
      };
    }
    const city = await pickRandomCity(isHere);
    return { lat: city.lat, lng: city.lng, label: describeCity(city), title: city.name, building: city.landmark || undefined };
  }

  /** The projection centered on a region, for the tiles asked for from now on. */
  public setPlace(bounds: LatLngBounds): void {
    this.projection.setBounds(bounds);
    this.bounds = bounds;
    // the ground at the center is at y 0
    this.elevation.setOrigin(this.projection.latMid, this.projection.lngMid);
    // the elevation of the tiles around it, while the map tiles are still to be asked for
    if (this.elevationEnabled) this.elevation.preload(this.projection.latMid, this.projection.lngMid);
  }

  /** No more tiles, the loaded ones gone: before the projection moves to another place (setPlace). */
  public stopStreaming(): void {
    this.tileStreamer.setTerrainStreamingEnabled(false);
    this.tileStreamer.setCityStreamingEnabled(false);
    this.tileStreamer.clearAllTerrainTiles();
    this.tileStreamer.clearAllCityTiles();
  }

  /** After setPlace: the tiles stream around the new center (the buildings while their param is on). */
  public restartStreaming(): void {
    this.tileStreamer.resetStreamingPosition();
    this.tileStreamer.setTerrainStreamingEnabled(true);
    if (this.buildingsEnabled) this.tileStreamer.setCityStreamingEnabled(true);
  }

  /** The elevation param to the tiles: they all load again, on the heights or on the flat ground. */
  public applyElevation(): void {
    this.tileStreamer.setElevation(this.elevationEnabled ? this.elevation : null);
    this.tileStreamer.clearAllTerrainTiles();
    this.tileStreamer.clearAllCityTiles();
  }

  /**
   * The streaming of this frame: the tiles whose building shadows fall into the view load too.
   * @returns whether tiles of the view are loading, with a bit of hysteresis over the gaps between the batches
   */
  public update(dt: number, sunDirection: Vector3): boolean {
    this.tileStreamer.setSunDirection(sunDirection);
    this.tileStreamer.update(dt);
    // not the tiles ahead of the view: the image converges while they come
    if (!this.tileStreamer.isViewIdle()) {
      this.loadingCooldown = 0.5;
    } else {
      this.loadingCooldown = Math.max(0, this.loadingCooldown - dt);
    }
    return this.loadingCooldown > 0;
  }

  /**
   * The trees too small to see are left out of the gBuffer (not out of the shadow map: there they cost next to nothing,
   * and a change of the cut would draw all cascades again).
   */
  public cullTrees(camera: MapCamera, canvasHeight: number): void {
    // the pixels per radian at the center of the screen
    const pixelsPerRadian = canvasHeight / (2 * Math.tan(camera.projection.FOV * 0.5));
    this.tileStreamer.cullTrees(camera.worldPosition, pixelsPerRadian, this.treeCullPixels.value);
  }

  /** After a frame is drawn. */
  public frameSubmitted(): void {
    this.tileStreamer.frameSubmitted();
    // all map workers from here on, see startTiles
    if (this.startFrames > 0 && --this.startFrames === 0) this.workerPool.activeLimit = this.workerPool.size;
  }

  destruct(): void {
    this.tileStreamer?.destruct();
    this.workerPool?.destroy();
    this.elevation.destroy();
  }
}
