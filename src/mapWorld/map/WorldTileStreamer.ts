import LogGL from 'webGPU-framework/renderer/core/LogGL';
import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import AABox from 'webGPU-framework/renderer/math/AABox';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Frustum from 'webGPU-framework/utils/camera/Frustum';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import MapProjection from '../../mapOSM/MapProjection';
import type { LatLngBounds } from '../../mapOSM/osmGeometryCore';
import type { TileCoord } from '../../mapOSM/PDOKTileService';
import type { BuildingInfo } from '../../mapOSM/cityGlTFCore';
import type { LayerRuntime } from '../../mapOSM/layerDefs';
import type { WorldBuildingResult, WorldTileResult } from './worldGeometryCore';
import type { WallAtlasMeshData } from '../city/WallAtlasView';
import OpenFreeMapTileService from './OpenFreeMapTileService';
import WorldWorkerPool from './WorldWorkerPool';
import type ElevationService from './ElevationService';
import type TerrainHeightField from './TerrainHeightField';
import type CityScene from '../city/CityScene';
import type TerrainLayerBuffer from '../../mapOSM/TerrainLayerBuffer';

export type WorldTileStreamerConfig = {
  renderer: Renderer;
  camera: MapCamera;
  projection: MapProjection;
  layers: LayerRuntime[];
  meshBounds: Map<Mesh, AABox>;
  workerPool: WorldWorkerPool;
  tileService: OpenFreeMapTileService;
  onTileError?: (message: string) => void;
  /** Gets the ray tracing data of the building tiles that come and go, see SkyVisibilityLightmap. */
  cityScene?: CityScene;
  /** Gets the road center lines of the terrain tiles that come and go, see CarTraffic. */
  roads?: RoadTileSink;
  /** Gets the ground meshes of the terrain tiles with heights (see setElevation). */
  ground?: TerrainLayerBuffer;
  /** Gets the heights of the terrain tiles, for what moves over the ground. */
  heightField?: TerrainHeightField;
};

/** Takes the road center lines of the terrain tiles (WorldTileResult.roads), by tile key. */
export interface RoadTileSink {
  addTile(key: string, roads: Float32Array): void;
  removeTile(key: string): void;
  clear(): void;
}

type QueuedTile = { tile: TileCoord; key: string; terrain: boolean; city: boolean };

/**
 * Streams OpenFreeMap vector tiles around the camera for {@link MapWorld}.
 *
 * Each z14 tile holds both ground layers and buildings, so a tile is fetched
 * once and the worker builds whatever is still missing for it (terrain and/or
 * buildings). Terrain goes into the shared per-layer {@link TerrainLayerBuffer}s;
 * buildings become one Mesh per tile with a recycled picking slot, exactly like
 * the 3DBAG tiles in mapOSM/TileStreamer, so picking/highlight/shadows/SSAO in
 * the host work unchanged.
 */
export default class WorldTileStreamer {
  public static readonly FEATURE_STRIDE = 65536;
  private static readonly KEEP_MARGIN_TILES = 1;
  private static readonly RETRY_COOLDOWN_MS = 10000;
  private static readonly MAX_CITY_FINALIZE_PER_FRAME = 1;
  // Concurrent tile fetches. Queued tiles beyond this can still be cancelled
  // when the camera moves away.
  private static readonly MAX_CONCURRENT_FETCHES = 8;
  // Tallest buildings we expect, used for the tile-vs-frustum test.
  private static readonly MAX_BUILDING_HEIGHT_M = 400;
  // The range of the ground (m, relative to the projection center's) of the tile-vs-frustum test while its elevation tile
  // isn't loaded
  private static readonly UNKNOWN_GROUND_M: [number, number] = [-1000, 1000];

  private renderer: Renderer;
  private camera: MapCamera;
  private projection: MapProjection;
  private layers: LayerRuntime[];
  private meshBounds: Map<Mesh, AABox>;
  private pool: WorldWorkerPool;
  private tiles: OpenFreeMapTileService;
  private onTileError?: (message: string) => void;
  private cityScene?: CityScene;
  private roads?: RoadTileSink;
  private ground?: TerrainLayerBuffer;
  private heightField?: TerrainHeightField;
  // the heights of the ground, null: the flat ground at 0. See setElevation
  private elevation: ElevationService | null = null;

  // --- tile state ---
  private loadedTerrainTiles: Set<string> = new Set();
  private loadedCityTiles: Map<string, { mesh: Mesh | null; slot: number; wallTile: [number, number]; quantization: number[] }> = new Map();
  private inFlight: Map<string, { terrain: boolean; city: boolean }> = new Map();
  private fetchQueue: QueuedTile[] = [];
  private failedTiles: Map<string, number> = new Map();
  private cityReadyQueue: { key: string; result: WorldBuildingResult }[] = [];
  // Bumped on every clear so results of requests made for an old region (or an
  // old projection center) are dropped when they arrive.
  private generation = 0;
  private terrainDesired: Set<string> = new Set();
  private cityDesired: Set<string> = new Set();
  private needsRestream = false;

  // --- picking slots (see mapOSM/TileStreamer) ---
  private slotBuildings: (BuildingInfo[] | null)[] = [];
  private freeSlots: number[] = [];
  private nextSlot = 0;

  public cityMeshes: Mesh[] = [];
  public cityMeshSlots: number[] = [];
  // per mesh the map tile of its wall atlas, x and y after each other: the uniform wallTile of the gBuffer
  public cityMeshTiles: number[] = [];
  // per mesh the decoding of its unorm16 vertices, 8 numbers (see setMeshQuantization)
  public cityMeshQuantization: number[] = [];
  // The mesh data of the last finalized city tile with walls: the wallAtlas drawMode, see WallAtlasView, which makes a
  // mesh of its own from it
  public lastCityTile: { key: string; result: WallAtlasMeshData } | null = null;
  private cityTilesChanged = false;

  // --- streaming state ---
  private frustum: Frustum = new Frustum();
  private minScratch = new Vector3();
  private maxScratch = new Vector3();
  private swScratch = new Vector3();
  private neScratch = new Vector3();
  private lastStreamPos: Vector3 = new Vector3(Infinity, 0, Infinity);
  private lastStreamDist = 0;
  private lastStreamEuler: Vector3 = new Vector3(Infinity, Infinity, Infinity);
  private terrainStreamingEnabled = false;
  private cityStreamingEnabled = false;
  private treesEnabled = true;

  // --- batch timing ---
  private batchStart = 0;
  private batchTiles = 0;

  // The direction to the sun, see setSunDirection
  private sunDirection: Vector3 | null = null;

  private radiusScaleParam!: ParamFloat;
  private loadOnlyInFrustum!: ParamBool;
  // How far from the camera the ground loads, in world units: the tiles that reach into the circle around the camera
  // (xz). The buildings load as far as the load distance, but not beyond it. See setViewDistanceMax
  private viewDistanceMax = 8;

  constructor(config: WorldTileStreamerConfig) {
    this.renderer = config.renderer;
    this.camera = config.camera;
    this.projection = config.projection;
    this.layers = config.layers;
    this.meshBounds = config.meshBounds;
    this.pool = config.workerPool;
    this.tiles = config.tileService;
    this.onTileError = config.onTileError;
    this.cityScene = config.cityScene;
    this.roads = config.roads;
    this.ground = config.ground;
    this.heightField = config.heightField;
  }

  public init(paramGroup: ParamGroup): void {
    const pg = paramGroup.addGroup('TileStreamer');
    const params = { callback: () => this.resetStreamingPosition() };
    // the buildings (and trees): times the view footprint of the camera height, see streamRadiusTiles. The ground loads
    // as far as viewDistanceMax, see setViewDistanceMax
    this.radiusScaleParam = pg.addParamFloat('load distance', isMobile() ? 1.5 : 2, 0.1, 8, params);
    this.loadOnlyInFrustum = pg.addParamBool('loadOnlyInFrustum', true, params);
  }

  // --- control -----------------------------------------------------------------

  public setTerrainStreamingEnabled(enabled: boolean): void {
    this.terrainStreamingEnabled = enabled;
  }

  public setCityStreamingEnabled(enabled: boolean): void {
    this.cityStreamingEnabled = enabled;
  }

  /**
   * The heights of the ground the tiles are built on (null: flat at 0). Takes effect for the tiles requested from now
   * on: the host clears the loaded ones.
   */
  public setElevation(elevation: ElevationService | null): void {
    this.elevation = elevation;
  }

  /**
   * The direction to the sun (normalized, y up): the tile-vs-frustum test then also counts a tile whose building shadows
   * fall into the view, so a building just outside the view casts its shadow as soon as the shadow can be seen.
   */
  public setSunDirection(dir: Vector3): void {
    const prev = this.sunDirection;
    if (prev && Math.abs(prev.x - dir.x) + Math.abs(prev.y - dir.y) + Math.abs(prev.z - dir.z) < 0.01) return;
    this.sunDirection = dir.clone();
    this.needsRestream = true;
  }

  /** The distance from the camera where the map ends (world units, see MapWorld.viewDistanceMax): the tiles reload. */
  public setViewDistanceMax(distance: number): void {
    if (this.viewDistanceMax === distance) return;
    this.viewDistanceMax = distance;
    this.resetStreamingPosition();
  }

  public resetStreamingPosition(): void {
    this.lastStreamPos = new Vector3(Infinity, 0, Infinity);
  }

  /** The trees of the wood areas in the building tiles (see worldTrees.ts): the building tiles load again. */
  public setTreesEnabled(enabled: boolean): void {
    if (this.treesEnabled === enabled) return;
    this.treesEnabled = enabled;
    this.clearAllCityTiles();
  }

  public consumeCityTilesChanged(): boolean {
    const changed = this.cityTilesChanged;
    this.cityTilesChanged = false;
    return changed;
  }

  public get activeWorkers(): number {
    return this.pool.activeJobs;
  }

  public isIdle(): boolean {
    return this.inFlight.size === 0 && this.fetchQueue.length === 0 && this.cityReadyQueue.length === 0;
  }

  // --- lifecycle ---------------------------------------------------------------

  public clearAllTerrainTiles(): void {
    for (const layer of this.layers) layer.buffer.clear();
    this.ground?.clear();
    this.heightField?.clear();
    this.roads?.clear();
    this.loadedTerrainTiles.clear();
    this.dropStale();
  }

  public clearAllCityTiles(): void {
    for (const tile of this.loadedCityTiles.values()) {
      if (tile.mesh) { this.meshBounds.delete(tile.mesh); tile.mesh.destruct(); }
    }
    this.loadedCityTiles.clear();
    this.lastCityTile = null;
    this.cityReadyQueue = [];
    this.cityMeshes.length = 0;
    this.cityMeshSlots.length = 0;
    this.cityMeshTiles.length = 0;
    this.cityMeshQuantization.length = 0;
    this.slotBuildings.length = 0;
    this.freeSlots.length = 0;
    this.nextSlot = 0;
    this.cityTilesChanged = true;
    this.cityScene?.clear();
    this.dropStale();
  }

  /** Forget queued / in-flight requests; their results will be ignored. */
  private dropStale(): void {
    this.generation++;
    this.inFlight.clear();
    this.fetchQueue = [];
    this.failedTiles.clear();
    // Requests for still-wanted tiles may have been dropped; re-evaluate.
    this.needsRestream = true;
  }

  /**
   * The global id (slot * FEATURE_STRIDE + feature id) of the building the ray hits first, -1 for none: traced on the
   * CPU through the BVHs of the city scene (see CityScene.pick). Without a city scene nothing is picked.
   */
  public pickBuilding(ro: Vector3, rd: Vector3): number {
    const hit = this.cityScene?.pick(ro, rd);
    if (!hit || hit.featureId < 0) return -1;
    const tile = this.loadedCityTiles.get(hit.key);
    return tile ? tile.slot * WorldTileStreamer.FEATURE_STRIDE + hit.featureId : -1;
  }

  public lookupBuilding(globalId: number): BuildingInfo | null {
    if (globalId < 0) return null;
    const slot = Math.floor(globalId / WorldTileStreamer.FEATURE_STRIDE);
    const featureId = globalId - slot * WorldTileStreamer.FEATURE_STRIDE;
    const buildings = this.slotBuildings[slot];
    return buildings ? buildings[featureId] ?? null : null;
  }

  // --- per frame ---------------------------------------------------------------

  public update(_dt: number): void {
    const pivotPos = this.camera.pivot.position;
    const euler = this.camera.euler;
    const moveDist = Vector3.distance(pivotPos, this.lastStreamPos);
    const elevChanged = Math.abs(this.camera.distance - this.lastStreamDist) > 0.01;
    const rotChanged =
      Math.abs(euler.x - this.lastStreamEuler.x) > 0.01 ||
      Math.abs(euler.y - this.lastStreamEuler.y) > 0.01;

    if (moveDist > 0.01 || elevChanged || rotChanged || this.needsRestream) {
      this.lastStreamPos = pivotPos.clone();
      this.lastStreamDist = this.camera.distance;
      this.lastStreamEuler = euler;
      this.needsRestream = false;
      this.streamTiles();
    }

    this.pumpFetchQueue();
    this.finalizeCityMeshes();
  }

  // --- streaming ---------------------------------------------------------------

  private streamRadiusTiles(elevationWorld: number, zoom: number): number {
    const fov = this.camera.projection.FOV;
    const aspect = this.camera.projection.aspectRatio;
    const halfW = elevationWorld * Math.tan(fov * 0.5) * aspect;
    const halfH = elevationWorld * Math.tan(fov * 0.5);
    const maxDimMeters = Math.max(halfW, halfH) * this.projection.metersPerUnit;
    const scale = this.radiusScaleParam ? this.radiusScaleParam.value : 1;
    return Math.ceil((scale * maxDimMeters) / this.projection.tileSizeMeters(zoom)) + 1;
  }

  /** The distance from p to the nearest point of the tile, in the xz plane (world units). */
  private tileDistance(bounds: LatLngBounds, p: Vector3): number {
    const sw = this.swScratch, ne = this.neScratch;
    this.projection.latLngToWorldInto(bounds.south, bounds.west, sw);
    this.projection.latLngToWorldInto(bounds.north, bounds.east, ne);
    const dx = Math.max(Math.min(sw.x, ne.x) - p.x, 0, p.x - Math.max(sw.x, ne.x));
    const dz = Math.max(Math.min(sw.z, ne.z) - p.z, 0, p.z - Math.max(sw.z, ne.z));
    return Math.sqrt(dx * dx + dz * dz);
  }

  private tileVisible(tile: TileCoord, bounds: LatLngBounds, maxHeightMeters: number): boolean {
    const sw = this.swScratch, ne = this.neScratch;
    this.projection.latLngToWorldInto(bounds.south, bounds.west, sw);
    this.projection.latLngToWorldInto(bounds.north, bounds.east, ne);
    const h = maxHeightMeters / this.projection.metersPerUnit;
    // the ground: from the elevation tile around it, a wide range until that is loaded
    let ground: [number, number] = [0, 0];
    if (this.elevation) ground = this.elevation.tileRange(tile) ?? WorldTileStreamer.UNKNOWN_GROUND_M;
    const lo = ground[0] / this.projection.metersPerUnit, hi = ground[1] / this.projection.metersPerUnit;
    const min = this.minScratch.setValues(Math.min(sw.x, ne.x), lo, Math.min(sw.z, ne.z));
    const max = this.maxScratch.setValues(Math.max(sw.x, ne.x), hi + h, Math.max(sw.z, ne.z));
    const sun = this.sunDirection;
    if (sun && sun.y > 0) {
      // the box grown by the shadow on the ground of its top, away from the sun. Its length is capped (the sun at
      // ~15 degrees): at a sunset the shadows of the tallest building reach many tiles away
      const length = h / Math.max(sun.y, 0.25);
      const dx = -sun.x * length, dz = -sun.z * length;
      if (dx < 0) min.x += dx; else max.x += dx;
      if (dz < 0) min.z += dz; else max.z += dz;
    }
    return this.frustum.boxInFrustum(min, max);
  }

  /** Recompute the wanted tile rings, evict what fell out, queue what's missing. */
  private streamTiles(): void {
    const terrainOn = this.terrainStreamingEnabled;
    const cityOn = this.cityStreamingEnabled;
    if (!terrainOn && !cityOn) return;

    this.frustum.setFromCamera(this.camera);
    const pivot = this.camera.pivot.position;
    const { lat, lng } = this.projection.worldToLatLng(pivot.x, pivot.z);
    const zoom = this.tiles.maxZoom;
    const cityRadius = this.streamRadiusTiles(this.camera.distance, zoom);
    // The ground: the tiles that reach into viewDistanceMax around the camera (xz), the rings around the pivot that
    // hold that circle. The pivot is anywhere in the center tile: 1 ring more
    const eye = this.camera.worldPosition;
    const tileSize = this.projection.tileSizeMeters(zoom) / this.projection.metersPerUnit;
    const viewDistance = this.viewDistanceMax;
    const eyeOffset = Math.hypot(eye.x - pivot.x, eye.z - pivot.z);
    const terrainRadius = Math.ceil((viewDistance + eyeOffset) / tileSize) + 1;
    const keep = WorldTileStreamer.KEEP_MARGIN_TILES;
    const keepDistance = viewDistance + keep * tileSize;
    const outer = Math.max(terrainOn ? terrainRadius : 0, cityOn ? cityRadius : 0) + keep;
    const center = this.tiles.latLngToTile(lat, lng, zoom);
    const now = performance.now();
    const frustumOnly = this.loadOnlyInFrustum?.value ?? true;

    this.terrainDesired.clear();
    this.cityDesired.clear();
    const candidates: (QueuedTile & { visible: number; distSq: number })[] = [];

    for (let dx = -outer; dx <= outer; dx++) {
      for (let dy = -outer; dy <= outer; dy++) {
        const tile: TileCoord = { z: zoom, x: center.x + dx, y: center.y + dy };
        if (!this.tiles.isValid(tile)) continue;
        const key = this.tiles.tileKey(tile);
        const ring = Math.max(Math.abs(dx), Math.abs(dy));
        const bounds = this.tiles.tileBounds(tile);
        // the distance from the camera to the nearest point of the tile, in xz
        const distance = this.tileDistance(bounds, eye);
        const inView = distance <= viewDistance;
        const kept = distance <= keepDistance;
        if (terrainOn && kept) this.terrainDesired.add(key);
        if (cityOn && ring <= cityRadius + keep && kept) this.cityDesired.add(key);

        const needTerrain = terrainOn && inView && !this.loadedTerrainTiles.has(key);
        const needCity = cityOn && ring <= cityRadius && inView && !this.loadedCityTiles.has(key) && !this.cityReadyQueued(key);
        if (!needTerrain && !needCity) continue;
        if (this.inFlight.has(key)) continue;
        const failedAt = this.failedTiles.get(key);
        if (failedAt && now - failedAt < WorldTileStreamer.RETRY_COOLDOWN_MS) continue;

        const visible = this.tileVisible(tile, bounds, needCity ? WorldTileStreamer.MAX_BUILDING_HEIGHT_M : 1) ? 0 : 1;
        if (frustumOnly && visible === 1) continue;
        candidates.push({ tile, key, terrain: needTerrain, city: needCity, visible, distSq: dx * dx + dy * dy });
      }
    }

    // Evict tiles outside the keep rings.
    const terrainRemove: string[] = [];
    for (const key of this.loadedTerrainTiles) if (!this.terrainDesired.has(key)) terrainRemove.push(key);
    if (terrainRemove.length) {
      for (const layer of this.layers) layer.buffer.removeTiles(terrainRemove);
      this.ground?.removeTiles(terrainRemove);
      for (const key of terrainRemove) {
        this.heightField?.removeTile(key);
        this.loadedTerrainTiles.delete(key);
        this.roads?.removeTile(key);
      }
    }
    for (const key of [...this.loadedCityTiles.keys()]) {
      if (!this.cityDesired.has(key)) this.removeCityTile(key);
    }
    this.cityReadyQueue = this.cityReadyQueue.filter((e) => this.cityDesired.has(e.key));

    // Visible first, then nearest first. Replaces the old queue so tiles the
    // camera moved away from are never fetched.
    candidates.sort((a, b) => (a.visible !== b.visible ? a.visible - b.visible : a.distSq - b.distSq));
    this.fetchQueue = candidates.map(({ tile, key, terrain, city }) => ({ tile, key, terrain, city }));
    if (this.fetchQueue.length && this.batchTiles === 0) this.batchStart = performance.now();
  }

  private cityReadyQueued(key: string): boolean {
    for (const e of this.cityReadyQueue) if (e.key === key) return true;
    return false;
  }

  private pumpFetchQueue(): void {
    while (this.fetchQueue.length > 0 && this.inFlight.size < WorldTileStreamer.MAX_CONCURRENT_FETCHES) {
      const entry = this.fetchQueue.shift()!;
      if (this.inFlight.has(entry.key)) continue;
      this.loadTile(entry);
    }
  }

  private loadTile(entry: QueuedTile): void {
    const { tile, key } = entry;
    const generation = this.generation;
    const bounds = this.projection.bounds;
    this.inFlight.set(key, { terrain: entry.terrain, city: entry.city });
    const url = this.tiles.tileUrl(tile);
    // the heights of the ground, in parallel with the vector tile
    const heightsPromise = this.elevation ? this.elevation.getTileGrid(tile) : Promise.resolve(null);
    // rejections are handled where it is awaited, not when the vector tile fails first
    heightsPromise.catch(() => {});

    // the heights the tile is built on, kept for the height field (the worker gets a copy)
    let tileHeights: Float32Array | null = null;
    void (async (): Promise<WorldTileResult | null> => {
      const response = await fetch(url);
      let pbf: ArrayBuffer;
      // No data for this tile (open ocean at high zoom, …). On the terrain it still gets its ground.
      if (response.status === 204 || response.status === 404) {
        if (!this.elevation) return null;
        pbf = new ArrayBuffer(0);
      } else {
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        pbf = await response.arrayBuffer();
      }
      tileHeights = await heightsPromise;
      if (generation !== this.generation) return null;
      return this.pool.buildTile(pbf, tile, bounds, entry.terrain, entry.city, this.treesEnabled, tileHeights?.slice() ?? null);
    })()
      .then((result) => {
        if (generation !== this.generation) return;
        this.inFlight.delete(key);
        this.failedTiles.delete(key);
        this.batchTiles++;
        this.onTileResult(key, entry, result, tileHeights);
        this.maybeLogBatch();
      })
      .catch((e) => {
        if (generation !== this.generation) return;
        this.inFlight.delete(key);
        this.failedTiles.set(key, performance.now());
        LogGL.warn(`[MapWorld] failed tile ${key}:`, e);
        this.onTileError?.(`Tile service error: GET ${url} - ${e?.message ?? e}`);
        this.maybeLogBatch();
      });
  }

  private onTileResult(key: string, entry: QueuedTile, result: WorldTileResult | null, heights: Float32Array | null): void {
    if (entry.terrain && this.terrainStreamingEnabled && this.terrainDesired.has(key) && !this.loadedTerrainTiles.has(key)) {
      const terrain = result?.terrain;
      if (terrain) {
        for (const layer of this.layers) {
          const l = terrain[layer.def.key];
          if (l && l.indices.length) layer.buffer.addTile(key, l.vertices, l.indices, l.floats);
        }
      }
      if (result?.ground) this.ground?.addTile(key, result.ground.vertices, result.ground.indices, result.ground.floats);
      if (heights) this.heightField?.addTile(entry.tile, heights);
      if (terrain && LogGL.ENABLED) this.logGroundMemory(key);
      if (result?.roads?.length) this.roads?.addTile(key, result.roads);
      // Empty tiles are recorded too, so they aren't re-requested.
      this.loadedTerrainTiles.add(key);
    }

    if (entry.city && this.cityStreamingEnabled && this.cityDesired.has(key) && !this.loadedCityTiles.has(key)) {
      const buildings = result?.buildings ?? {
        vertices: new Float32Array(0), lightmap: new Float32Array(0), indices: new Uint32Array(0), trace: null, buildings: [], min: [0, 0, 0], max: [0, 0, 0],
      };
      this.cityReadyQueue.push({ key, result: buildings as WorldBuildingResult });
    }

    // Something toggled on while this request was in flight: stream again.
    if (
      (this.terrainStreamingEnabled && this.terrainDesired.has(key) && !this.loadedTerrainTiles.has(key)) ||
      (this.cityStreamingEnabled && this.cityDesired.has(key) && !this.loadedCityTiles.has(key) && !this.cityReadyQueued(key))
    ) {
      this.needsRestream = true;
    }
  }

  /** The GPU memory of the ground of a tile that came in, and of all of them. */
  private logGroundMemory(key: string): void {
    const buffers = this.layers.map((l) => l.buffer);
    if (this.ground) buffers.push(this.ground);
    let tile = 0, used = 0, capacity = 0;
    for (const b of buffers) {
      tile += b.tileBytes(key);
      used += b.usedBytes;
      capacity += b.capacityBytes;
    }
    const groundTile = this.ground?.tileBytes(key) ?? 0;
    const mb = (bytes: number) => (bytes / 1048576).toFixed(2);
    LogGL.log(`[MapWorld] ground tile ${key}: ${mb(tile)} MB (layers ${mb(tile - groundTile)}, ground mesh ${mb(groundTile)}), ` +
      `all ${this.loadedTerrainTiles.size + 1} ground tiles ${mb(used)} MB in ${mb(capacity)} MB of buffers`);
  }

  private maybeLogBatch(): void {
    if (this.batchTiles > 0 && this.isIdleExceptFinalize()) {
      LogGL.log(`[MapWorld] batch: ${this.batchTiles} tiles in ${(performance.now() - this.batchStart).toFixed()}ms`);
      this.batchTiles = 0;
    }
  }

  private isIdleExceptFinalize(): boolean {
    return this.inFlight.size === 0 && this.fetchQueue.length === 0;
  }

  /** Upload at most a few finished building tiles per frame. */
  private finalizeCityMeshes(): void {
    let budget = WorldTileStreamer.MAX_CITY_FINALIZE_PER_FRAME;
    while (budget-- > 0 && this.cityReadyQueue.length > 0) {
      const { key, result } = this.cityReadyQueue.shift()!;
      if (this.loadedCityTiles.has(key)) continue;

      let mesh: Mesh | null = null;
      if (result.vertices.length > 0) {
        mesh = new Mesh(this.renderer);
        mesh.setAttribute('position', 4, result.vertices);
        // The lightmap chart per vertex, read by the materials that declare it (see SkyVisibilityLightmap).
        if (result.lightmap.length * 2 === result.vertices.length) mesh.setAttribute('lightmap', 2, result.lightmap);
        // Indexed, see buildBuildings. A result without indices (an older worker) is a triangle soup
        if (result.indices?.length) mesh.setIndices(result.indices);
        const { vertices, lightmap, indices, atlasSize } = result;
        if (atlasSize) this.lastCityTile = { key, result: { vertices, quantization: result.quantization, lightmap, indices, atlasSize } };
        this.meshBounds.set(mesh, new AABox(
          new Vector3(result.min[0], result.min[1], result.min[2]),
          new Vector3(result.max[0], result.max[1], result.max[2]),
        ));
      }
      const slot = this.freeSlots.length > 0 ? this.freeSlots.pop()! : this.nextSlot++;
      this.slotBuildings[slot] = result.buildings;
      this.loadedCityTiles.set(key, { mesh, slot, wallTile: result.wallTile, quantization: result.quantization });
      this.cityScene?.addTile(key, result.trace);
      this.cityTilesChanged = true;
      this.rebuildCityMeshes();
    }
  }

  private removeCityTile(key: string): void {
    const tile = this.loadedCityTiles.get(key);
    if (!tile) return;
    if (tile.mesh) { this.meshBounds.delete(tile.mesh); tile.mesh.destruct(); }
    if (this.lastCityTile?.key === key) this.lastCityTile = null;
    this.slotBuildings[tile.slot] = null;
    this.freeSlots.push(tile.slot);
    this.loadedCityTiles.delete(key);
    this.cityScene?.removeTile(key);
    this.cityTilesChanged = true;
    this.rebuildCityMeshes();
  }

  private rebuildCityMeshes(): void {
    this.cityMeshes.length = 0;
    this.cityMeshSlots.length = 0;
    this.cityMeshTiles.length = 0;
    this.cityMeshQuantization.length = 0;
    for (const tile of this.loadedCityTiles.values()) {
      if (!tile.mesh) continue;
      this.cityMeshes.push(tile.mesh);
      this.cityMeshSlots.push(tile.slot);
      this.cityMeshTiles.push(tile.wallTile[0], tile.wallTile[1]);
      this.cityMeshQuantization.push(...tile.quantization);
    }
  }

  public destruct(): void {
    this.clearAllTerrainTiles();
    this.clearAllCityTiles();
  }
}
