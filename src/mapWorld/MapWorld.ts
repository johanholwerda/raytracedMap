import Renderer from 'webGPU-framework/renderer/render/Renderer';
import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Color from 'webGPU-framework/renderer/math/Color';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import ParamColor from 'webGPU-framework/utils/params/ParamColor';
import ParamsGui from 'webGPU-framework/utils/paramsGui/ParamsGui';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import { clamp } from 'webGPU-framework/renderer/utils/MathUtils';
import type { LatLngBounds } from './osm/osmGeometryCore';
import { DEFAULT_BOUNDS } from './osm/layerDefs';
import MapProjection from './osm/MapProjection';
import ReprojectedSSAO, { SSAOMode } from './ssao/ReprojectedSSAO';
import TAA from './taa/TAA';
import MapGBuffer from './deferred/MapGBuffer';
import DeferredShading from './deferred/DeferredShading';
import WaterReflectionHalfRes from './deferred/WaterReflectionHalfRes';
import { NOTABLE_BUILDINGS } from './map/notableBuildings';
import { describeCity, findCity, pickRandomCity, pickStartCity } from './map/worldCities';
import OpenFreeMapTileService from './map/OpenFreeMapTileService';
import WorldWorkerPool from './map/WorldWorkerPool';
import WorldTileStreamer from './map/WorldTileStreamer';
import ElevationService from './map/ElevationService';
import TerrainHeightField from './map/TerrainHeightField';
import TiltShift from './tiltShift/TiltShift';
import SkyVisibilityLightmap, { LightmapView } from './skyVisibilityLightmap/SkyVisibilityLightmap';
import type { LightmapBlur } from './skyVisibilityLightmap/SkyVisibilityLightmap';
import MapWorldBench, { MapWorldFrameState } from './MapWorldBench';
import MapWorldVideo from './MapWorldVideo';
import {
  DRAW_MODES, DrawMode, QUALITIES, Quality, qualitySettings, REFLECTION_BLOCKS, REFLECTION_RAYS, ReflectionRays, SHADOW_VIEWS,
  ShadowMode, SKY_VISIBILITY_VIEWS, randomTimeOfDay, TIMES_OF_DAY, TimeOfDay, timeOfDaySettings, timeOfDaySunAngles,
} from './MapWorldSettings';
import MapGeometry, { CityMeshes } from './MapGeometry';
import SunShadows from './SunShadows';
import WallAtlasView from './city/WallAtlasView';
import type { ShadowMapAhead } from './osm/shadowMap/ShadowMap';
import BuildingPicker from './BuildingPicker';
import CarTraffic from './cars/CarTraffic';
import MapWorldUI from './ui/MapWorldUI';
import AirplaneController from './AirplaneController';
import OrbitKeyRotation from './OrbitKeyRotation';
import { TREE_CROWN_COLORS, TREE_TRUNK_COLOR } from './map/worldTrees';
import WaterRippleTexture from './deferred/waterRipples/WaterRippleTexture';
import KeyListener from 'webGPU-framework/utils/input/KeyListener';
import { KEY_P } from 'webGPU-framework/utils/input/Key';
import { saveCanvasToDisk } from 'webGPU-framework/utils/image/ImageUtils';

// The SSAO (ReprojectedSSAO): off since 2026-11-23, on the sky term alone it is too subtle next to the sky visibility
// lightmap. Not constructed (nor in the GUI) while false. Its params then take their defaults, no quality sets them; the
// qualities had amount 1, rayMarchSteps 6 (1 on low), randomStep on medium. Cost at 4K (3080): 0.55 ms with 6 steps,
// 0.19 ms with 1 (only the contact occlusion), 0.20 ms with randomStep (~44% of the occlusion).
const SSAO_ENABLED = false;
// 1 / seconds: how fast the orbit pivot eases to the ground under it, and how close (m) it snaps onto it, see followGround
const PIVOT_GROUND_RATE = 6;
const PIVOT_GROUND_SNAP_M = .01;
// the orbit camera: the pitch (eulerX, radians, always a bit down) and the distance to the pivot (world units)
const CAMERA_PITCH_MIN = -1.4;
const CAMERA_PITCH_MAX = -.2;
const CAMERA_DISTANCE_MIN = .1;
const CAMERA_DISTANCE_MAX = 2;
// ms the orbit camera has to stand still before its view goes into the URL, see updateUrl
const URL_CAMERA_DELAY_MS = 500;
// ms after the last input (pointer, wheel, key) in which the camera moving counts as the user moving it: the zoom
// easing and the rotation inertia run on after the input, a random city or a searched place loads a bit after the click
const USER_INPUT_WINDOW_MS = 2000;

/**
 * The orbit camera in the URL: ?lat=&lng= the pivot (the ground point it orbits), dist= its distance to the pivot in
 * meters, pitch= and yaw= in degrees (pitch < 0: looking down). The time of day preset goes in as ?time=, see TIMES_OF_DAY.
 */
type UrlCamera = { lat: number; lng: number; dist: number; pitch: number; yaw: number };

/** The camera of the URL, or null without a (valid) lat and lng. The missing others are NaN. */
function parseUrlCamera(params: URLSearchParams): UrlCamera | null {
  const num = (key: string) => {
    const value = params.get(key);
    return value === null || value.trim() === '' ? NaN : Number(value);
  };
  const lat = num('lat');
  const lng = num('lng');
  if (!(Math.abs(lat) <= 85) || !(Math.abs(lng) <= 180)) return null;
  return { lat, lng, dist: num('dist'), pitch: num('pitch'), yaw: num('yaw') };
}

/**
 * The sandbox page loads eruda (the mobile console, a cogwheel + wrench button) for every example on mobile. MapWorld
 * doesn't want it: tapping it opened the browser console. eruda loads async from a CDN, so wait for it, then destroy it.
 */
function removeMobileConsole(): void {
  let tries = 0;
  const check = () => {
    const eruda = (window as any).eruda;
    if (eruda) {
      try { eruda.destroy(); } catch { /* not initialised yet: try again */ tries++; if (tries < 100) setTimeout(check, 100); }
    } else if (++tries < 100) {
      setTimeout(check, 100);
    }
  };
  check();
}

/**
 * World version of MapNetherlands: the same renderer (cascaded VSM sun shadows, click-to-pick buildings, URL camera
 * sync, 2D region picker), but shaded deferred: 1 gBuffer pass with a jittered camera (MapGBuffer), shaded once per
 * pixel (DeferredShading) with the light of the sky visibility lightmap, the
 * reprojected SSAO over it, and TAA instead of
 * MSAA. Fed by worldwide OpenStreetMap data instead of the
 * Dutch PDOK / 3DBAG services:
 *  - ground layers + extruded buildings from OpenFreeMap vector tiles
 *    (OpenMapTiles schema, one request per tile), and
 *  - street addresses for clicked buildings from Nominatim.
 * It starts on a random major world capital.
 *
 * The parts: MapGeometry (what is drawn), SunShadows, BuildingPicker and MapWorldUI (the DOM overlays).
 * This class runs the frame: the gBuffer, the lightmap, the deferred shading, the TAA, the SSAO and the tilt-shift.
 */
export default class MapWorld {
  private renderer!: Renderer;
  private camera!: MapCamera;
  // the airplane button: flies the camera with WASD / the arrows instead of orbiting
  private airplane!: AirplaneController;
  // not flying: WASD / the arrows and the touch joystick rotate the orbit camera
  private orbitKeys!: OrbitKeyRotation;
  private bounds: LatLngBounds = DEFAULT_BOUNDS;
  private workerPool!: WorldWorkerPool;
  private tileService: OpenFreeMapTileService = new OpenFreeMapTileService();
  // The heights of the ground: the tiles are built on it while the elevation param is on, see applyElevation
  private elevation: ElevationService = new ElevationService();
  private elevationParam!: ParamBool;
  // the ground of the loaded tiles, for the cars, the camera pivot and the plane (empty with the elevation off: flat 0)
  private heightField!: TerrainHeightField;
  private drawMode!: ParamSelect;
  private quality!: ParamSelect;
  private timeOfDay!: ParamSelect;
  private backgroundColor!: ParamColor;
  private skyColor!: ParamColor;
  private sunColor!: ParamColor;
  // Renders on every other vsync of a 120 Hz screen: an even 60 fps instead of an uneven 80 (1 and 2 vsyncs mixed).
  private cap60!: ParamBool;
  // The rAF time of the last rendered frame, ms.
  private lastFrameTime: number = -Infinity;
  // the user moved the camera or changed the time of day since the URL was last written, the time (performance.now) the
  // camera last moved and of the last user input, see updateUrl
  private urlPending: boolean = false;
  private cameraStillSince: number = 0;
  private lastInputTime: number = -Infinity;
  // ?time=sunset: the time of day preset to start with, see init
  private urlTimeOfDay?: TimeOfDay;
  // QualitySettings.forward of the quality param
  private forward: boolean = false;
  private drawBuildings!: ParamBool;
  private clearColor = Color.fromHex('#b8dbff');
  private logDrawInfo!: ParamBool;
  private distanceFog!: ParamFloat;
  // Where the map ends, in world units from the camera: the fog is 1 there (see getFog in deferredShading.wgsl), the
  // far plane, the ground quad and the ground tiles reach that far (see applyViewDistanceMax). At least 3: the pivot is
  // up to 2 (setDistanceLimit) from the camera and stays out of the fog
  private viewDistanceMax!: ParamFloat;
  // the water darker where it reflects a building, see deferredShading.wgsl
  private waterReflections!: ParamBool;
  // the water pixels per ray of those reflections, see ReflectionRays
  private reflectionRays!: ParamSelect;
  // traces them when that is more than 1 pixel, before the shading
  private waterReflection!: WaterReflectionHalfRes;
  private reflectionDarkening!: ParamFloat;
  // the water mirrored on the ripples with the sun in it, also without waterReflections (the buildings)
  private waterRipples!: ParamBool;
  // The animated micro normal the reflections are mirrored on, redrawn every frame. Its wavelength is in meters.
  private ripples!: WaterRippleTexture;
  // the slope of the ripples, 0 is a flat mirror (and no animation)
  private rippleSlope!: ParamFloat;
  private rippleSpeed!: ParamFloat;
  // the ripples move: the map is drawn every frame. Off, they are frozen and a still view stops being drawn
  private animateWater!: ParamBool;
  // seconds the ripples moved, only counts while animateWater is on
  private rippleTime: number = 0;
  // the ripple texture needs a redraw: a param changed (or it was never drawn)
  private ripplesDirty: boolean = true;
  private sunReflection!: ParamFloat;
  private sunReflectionPower!: ParamFloat;
  // the gBuffer materials of the water layers, a bit each
  private waterMaterials: number = 0;
  private projection: MapProjection = new MapProjection();

  private geometry!: MapGeometry;
  private sun!: SunShadows;
  private wallAtlasView!: WallAtlasView;
  private picker!: BuildingPicker;
  private ui!: MapWorldUI;
  private tileStreamer!: WorldTileStreamer;

  // null without SSAO_ENABLED: not constructed then
  private ssao: ReprojectedSSAO | null = null;
  // The cars on the roads, drawn over the converged image, see CarTraffic.
  private traffic!: CarTraffic;
  private taa!: TAA;
  private gBuffer!: MapGBuffer;
  private deferred!: DeferredShading;
  private tiltShift!: TiltShift;
  // The sky the points see instead of the sky term, see SkyVisibilityLightmap. It owns the CityScene.
  private skyLightmap!: SkyVisibilityLightmap;
  // ReprojectedSSAO.version the tilt-shift target was composited with, -1 without AO.
  private postSSAOVersion: number = -1;
  // ReprojectedSSAO.restartVersion the shading (its TAA history) holds, -1 without the AO in the shading (skyOnly).
  private shadedSSAORestart: number = -1;
  // The shaded gBuffer of this frame (jittered), the TAA resolves it. A static view stops being drawn once the TAA
  // converged: its history is re-presented, the SSAO composites over it.
  private sceneColorRT?: RenderTexture;
  // The forward quality's image, with the depth its shading tests. Created when the forward quality is first drawn.
  private forwardRT?: RenderTexture;
  private dirty: boolean = true;
  // P saves the canvas as a png, right after the next drawn frame: a WebGPU canvas only holds its image in the task
  // that drew it.
  private screenshotRequested: boolean = false;
  // The forward image in forwardRT is not of the current view: the deferred path drew while it was not.
  private forwardStale: boolean = true;
  // The shadow map was not updated in a frame (ray traced shadows, overdraw): it may miss casters that came or went.
  private shadowMapSkipped: boolean = false;
  // SSAO "loading" hysteresis across the short idle gaps between tile batches.
  private loadingCooldown: number = 0;

  // ?bench: the self-running benchmark, see MapWorldBench
  private bench?: MapWorldBench;
  // ?video: a frame at a time for video/record.mjs instead of on requestAnimationFrame, see MapWorldVideo
  private video?: MapWorldVideo;
  private frameState: MapWorldFrameState = {
    newJitter: false, lightmapUpdating: false, ssaoAccumulating: false, renderMap: false, tilesLoading: false, cameraMoving: false,
  };

  // false when WebGPU isn't available: the Renderer then shows its "WebGPU Not Supported" message and nothing else starts
  async start(canvasParent: HTMLElement, preloader: WebGPUPreLoader, uiVisible: boolean): Promise<boolean> {
    LogGL.ENABLED = true;

    // router5 strips query params on start; bootstrap.ts keeps the original.
    const urlParams = new URLSearchParams(window.location.search || (window as any).__initialSearch || '');

    this.renderer = new Renderer();
    // ?res=2160: the canvas this many pixels high, for timings at a fixed resolution
    // const res = parseFloat(urlParams.get('res') ?? '');
    // const pixelRatio = res > 0 ? res / Math.max(canvasParent.getBoundingClientRect().height, 1) : -1;

    // No MSAA: the TAA anti-aliases with the camera jitter.
    // On desktop the adapter's storage binding limit (the Renderer lowers the request to it), for lightmaps over 4096
    // (very high quality). Phones keep the default 128 MiB: their lightmap falls back to 4096, see getFittingSize of
    // SkyVisibilityLightmap. CityScene stays at its 128 MiB whatever the limit
    const requiredLimits: Record<string, GPUSize64> = isMobile() ? {} : { maxStorageBufferBindingSize: 2 ** 32 };
    await this.renderer.init(canvasParent, { sampleCount: 1, pixelRatio: -1, requiredLimits });
    if (!this.renderer.device) return false;
    // await this.renderer.init(canvasParent, { sampleCount: 1, pixelRatio: 1});
    if (urlParams.has('bench')) {
      this.bench = new MapWorldBench(this.renderer, urlParams);
      // for setting params from the console between runs
      (window as any).__mapWorld = this;
    }
    if (urlParams.has('video')) this.video = this.createVideo();
    this.renderer.setClearColor(0, 0, 0, 1);

    // Read the current OpenFreeMap tile URL (it changes with each planet build).
    await this.tileService.init();

    this.camera = new MapCamera(this.renderer, {
      fov: 1,
      nearPlane: 0.01,
      farPlane: 10,
      zPosition: 1,
    });
    this.camera.rotationController.eulerXY = new Vector2(-.7, 0);
    // always look a bit down, never from under the map
    this.camera.rotationController.rotationLimitX = new Vector2(CAMERA_PITCH_MIN, CAMERA_PITCH_MAX);
    this.camera.setDistanceLimit(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX);
    // Geometry is projected relative to the bounds center (world origin).
    this.camera.pivot.position = new Vector3(0, 0, 0);
    this.airplane = new AirplaneController(this.renderer, this.camera);
    this.orbitKeys = new OrbitKeyRotation(this.renderer, this.camera);

    if (SSAO_ENABLED) this.ssao = new ReprojectedSSAO(this.renderer, preloader);
    this.taa = new TAA(this.renderer);
    this.tiltShift = new TiltShift(this.renderer);
    // The AO converges to fewer samples under the blur, the lightmap texels follow it too
    this.tiltShift.onChange = () => {
      this.ssao?.reset();
      this.dirty = true;
    };
    this.gBuffer = new MapGBuffer(this.renderer);
    this.deferred = new DeferredShading(this.renderer);
    this.sceneColorRT = new RenderTexture(this.renderer, 1, 1, {
      // 10 bits: the TAA reads it, the 1 / 255 steps of 8 bits would show through its average (see TiltShift)
      formats: ['rgb10a2unorm'],
      sampleCount: 1,
      scaleToCanvas: true,
    });

    this.skyLightmap = new SkyVisibilityLightmap(this.renderer, preloader);
    this.skyLightmap.onChange = () => this.dirty = true;
    // The deferred shading reads the lightmap.
    for (const material of this.deferred.materials) this.skyLightmap.addResolveMaterial(material, true);
    // the points on the sun of the soft ray traced shadows
    this.deferred.bindBlueNoise(this.skyLightmap.blueNoise);

    this.traffic = new CarTraffic(this.renderer, this.projection);
    // The cars are lit by the sky visibility of the ground under them (looked up per car).
    for (const shader of this.traffic.resolveShaders) this.skyLightmap.addResolveMaterial(shader);
    // The ray traced shadows on the cars (per car) trace the buildings of the lightmap.
    this.traffic.bindCity(this.skyLightmap.cityScene);

    this.geometry = new MapGeometry(this.renderer, this.projection, this.gBuffer);
    // The water reflections trace the buildings of the lightmap.
    this.deferred.bindCity(this.skyLightmap.cityScene);
    this.ripples = new WaterRippleTexture(this.renderer);
    this.ripples.wavelength = 10;
    this.deferred.bindWaterRipples(this.ripples);
    this.waterReflection = new WaterReflectionHalfRes(this.renderer);
    this.waterReflection.bindCity(this.skyLightmap.cityScene);
    this.waterReflection.bindWaterRipples(this.ripples);
    for (const layer of this.geometry.layers) {
      if (layer.def.key === 'water' || layer.def.key === 'waterways') this.waterMaterials |= 1 << this.gBuffer.materialIndex(layer.material);
    }
    this.workerPool = new WorldWorkerPool();

    // Click-to-select buildings; the address comes from Nominatim.
    this.picker = new BuildingPicker(this.renderer, canvasParent, () => this.dirty = true);

    KeyListener.addKeyHitEvent(KEY_P, () => this.screenshotRequested = true);

    this.sun = new SunShadows(this.renderer);
    this.wallAtlasView = new WallAtlasView(this.renderer);
    for (const material of this.materials) this.sun.bindResources(material);
    // the hills for the ray traced sun shadows, which trace only the buildings and the trees
    for (const material of [...this.deferred.materials, ...this.traffic.materials]) this.sun.terrainSun.bind(material);

    this.ui = new MapWorldUI(canvasParent, {
      getViewBounds: () => this.currentViewBounds(),
      onSelectBounds: (bounds) => void this.loadFromBounds(bounds),
      onRandomCity: () => void this.goToRandomCity(),
      onToggleAirplane: () => this.toggleAirplane(),
      onGoToPlace: (place) => {
        LogGL.log(`[MapWorld] search: ${place.title} (${place.lat.toFixed(5)}, ${place.lng.toFixed(5)})`);
        this.ui.showLocationTitle(place.title, place.subtitle);
        void this.loadFromBounds(this.boundsAround(place.lat, place.lng));
      },
    });

    this.heightField = new TerrainHeightField(this.projection);
    this.traffic.setHeightField(this.heightField);
    this.airplane.ground = (x, z) => this.heightField.heightAt(x, z);
    this.tileStreamer = new WorldTileStreamer({
      renderer: this.renderer,
      camera: this.camera,
      projection: this.projection,
      layers: this.geometry.layers,
      meshBounds: this.geometry.meshBounds,
      workerPool: this.workerPool,
      tileService: this.tileService,
      onTileError: (msg) => this.ui.showTileError(msg),
      cityScene: this.skyLightmap.cityScene,
      roads: this.traffic,
      ground: this.geometry.groundBuffer,
      heightField: this.heightField,
    });
    // on by default, see the elevation param
    this.tileStreamer.setElevation(this.elevation);
    this.geometry.elevation = true;

    // Start on one of a few well-known cities (pickStartCity). The projection
    // is centered on it so vertex coordinates stay small (fp32 precision).
    // ?lat=&lng=(&dist=&pitch=&yaw=): the camera of a shared link, see updateUrl. Not in a bench or a video,
    // they move the camera themselves.
    const urlCamera = this.bench || this.video ? null : parseUrlCamera(urlParams);
    // not in a bench: its frames have to compare between runs
    const timeIndex = TIMES_OF_DAY.indexOf(urlParams.get('time') ?? '');
    if (!this.bench && timeIndex >= 0) this.urlTimeOfDay = timeIndex as TimeOfDay;
    // the user's input, for telling their camera moves from the program's, see updateUrl
    const markInput = () => this.lastInputTime = performance.now();
    const inputOptions = { capture: true, passive: true };
    window.addEventListener('pointerdown', markInput, inputOptions);
    window.addEventListener('pointermove', (e) => { if (e.buttons) markInput(); }, inputOptions);
    window.addEventListener('wheel', markInput, inputOptions);
    window.addEventListener('keydown', markInput, inputOptions);
    if (urlCamera) {
      this.bounds = this.boundsAround(urlCamera.lat, urlCamera.lng);
      LogGL.log(`[MapWorld] centering on the URL camera (${urlCamera.lat}, ${urlCamera.lng})`);
    } else {
      // ?city=Paris: always the same city, for timings
      const cityName = urlParams.get('city');
      const city = (cityName ? await findCity(cityName) : undefined) ?? pickStartCity();
      this.bounds = this.boundsAround(city.lat, city.lng);
      LogGL.log(`[MapWorld] centering on ${describeCity(city)}`);
      this.ui.showLocationTitle(city.name, city.landmark || undefined);
    }
    if (!this.video && !this.bench) this.ui.showControlsHint();

    await this.loadFromBounds(this.bounds);
    if (urlCamera) this.applyUrlCamera(urlCamera);
    return true;
  }

  /** The pitch, yaw and distance of the URL (the pivot is at its lat/lng already, loadFromBounds centered there). */
  private applyUrlCamera(urlCamera: UrlCamera): void {
    const euler = this.camera.rotationController.eulerXY;
    const toRad = Math.PI / 180;
    const pitch = Number.isFinite(urlCamera.pitch) ? clamp(urlCamera.pitch * toRad, CAMERA_PITCH_MIN, CAMERA_PITCH_MAX) : euler.x;
    const yaw = Number.isFinite(urlCamera.yaw) ? urlCamera.yaw * toRad : euler.y;
    this.camera.rotationController.eulerXY = new Vector2(pitch, yaw);
    if (Number.isFinite(urlCamera.dist)) {
      this.camera.distance = clamp(urlCamera.dist * this.projection.metersToNorm, CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX);
    }
  }

  /**
   * Writes the orbit camera (lat, lng, dist, pitch, yaw, see UrlCamera) and the time of day to the URL once the camera
   * has stood still for URL_CAMERA_DELAY_MS, with replaceState: no history entry per view. Only after the user moved
   * the camera (within USER_INPUT_WINDOW_MS of their input) or changed the time of day: a fresh load, the pivot easing
   * onto the loaded ground, keep the URL as it is. Not while flying (the plane never stands still), in a bench or a video.
   */
  private updateUrl(moving: boolean): void {
    if (this.bench || this.video || this.airplane.active) return;
    const now = performance.now();
    if (moving) {
      if (now - this.lastInputTime < USER_INPUT_WINDOW_MS) this.urlPending = true;
      this.cameraStillSince = now;
      return;
    }
    if (!this.urlPending || now - this.cameraStillSince < URL_CAMERA_DELAY_MS) return;
    this.urlPending = false;

    const pivot = this.camera.pivot.position;
    const { lat, lng } = this.projection.worldToLatLng(pivot.x, pivot.z);
    const euler = this.camera.rotationController.eulerXY;
    // the yaw adds up over the turns: into -180..180
    const yaw = euler.y - Math.PI * 2 * Math.round(euler.y / (Math.PI * 2));
    const toDeg = 180 / Math.PI;
    const params = new URLSearchParams(window.location.search);
    // the camera's lat / lng win over it on load
    params.delete('city');
    params.set('lat', lat.toFixed(6));
    params.set('lng', lng.toFixed(6));
    params.set('dist', (this.camera.distance / this.projection.metersToNorm).toFixed(0));
    params.set('pitch', (euler.x * toDeg).toFixed(2));
    params.set('yaw', (yaw * toDeg).toFixed(2));
    params.set('time', TIMES_OF_DAY[this.timeOfDay.value]);
    const url = `${window.location.pathname}?${params}${window.location.hash}`;
    if (url !== window.location.pathname + window.location.search + window.location.hash) {
      window.history.replaceState(window.history.state, '', url);
    }
  }

  /** Bounds centered on the camera pivot, keeping the current span. */
  private currentViewBounds(): LatLngBounds {
    const pivot = this.camera.pivot.position;
    const { lat, lng } = this.projection.worldToLatLng(pivot.x, pivot.z);
    const halfLat = (this.bounds.north - this.bounds.south) * 0.5;
    const halfLng = (this.bounds.east - this.bounds.west) * 0.5;
    return { south: lat - halfLat, north: lat + halfLat, west: lng - halfLng, east: lng + halfLng };
  }

  /** Bounds centered on lat/lng with the DEFAULT_BOUNDS span (same world scale as NL). */
  private boundsAround(lat: number, lng: number): LatLngBounds {
    const halfLat = (DEFAULT_BOUNDS.north - DEFAULT_BOUNDS.south) * 0.5;
    const halfLng = (DEFAULT_BOUNDS.east - DEFAULT_BOUNDS.west) * 0.5;
    return { south: lat - halfLat, north: lat + halfLat, west: lng - halfLng, east: lng + halfLng };
  }

  /**
   * Jump to a random city (weighted by sqrt(population)) or, half the time, a
   * uniformly picked notable building. Avoids landing where we already are.
   */
  private async goToRandomCity(): Promise<void> {
    const pivot = this.camera.pivot.position;
    const current = this.projection.worldToLatLng(pivot.x, pivot.z);
    const isHere = (p: { lat: number; lng: number }) => Math.abs(p.lat - current.lat) + Math.abs(p.lng - current.lng) < 0.5;
    let place: { lat: number; lng: number };
    let label: string;
    let title: string;
    let building: string | undefined;
    if (Math.random() < 0.5) {
      const pick = () => NOTABLE_BUILDINGS[Math.floor(Math.random() * NOTABLE_BUILDINGS.length)];
      let notable = pick();
      for (let i = 0; i < 8 && isHere(notable); i++) notable = pick();
      place = notable;
      label = `${notable.name}, ${notable.city}, ${notable.country}`;
      title = `${notable.city}, ${notable.country}`;
      building = notable.name;
    } else {
      const city = await pickRandomCity(isHere);
      place = city;
      label = describeCity(city);
      title = city.name;
      building = city.landmark || undefined;
    }
    LogGL.log(`[MapWorld] random location: ${label}`);
    this.ui.showLocationTitle(title, building);
    // another time of day too (the callback applies it, the GUI follows); loadFromBounds then sets the sun angles for the new latitude
    let timeOfDay = randomTimeOfDay();
    for (let i = 0; i < 8 && timeOfDay === this.timeOfDay.value; i++) timeOfDay = randomTimeOfDay();
    this.timeOfDay.setValue(timeOfDay);
    await this.loadFromBounds(this.boundsAround(place.lat, place.lng));
  }

  /** The video mode: what MapWorldVideo needs of the map, the streaming and the plane. */
  private createVideo(): MapWorldVideo {
    const defaults = { minHeight: 0, maxHeight: 0 };
    return new MapWorldVideo(this.renderer, {
      example: this,
      step: () => this.update(),
      load: (lat, lng) => this.loadFromBounds(this.boundsAround(lat, lng)),
      place: (east, north, height, yaw, speed, pitch) => {
        const plane = this.airplane;
        if (!plane.active) {
          plane.enter(this.projection.metersToNorm);
          defaults.minHeight = plane.minHeight;
          defaults.maxHeight = plane.maxHeight;
        }
        // a shot may fly lower or higher than the keys can
        plane.minHeight = Math.min(defaults.minHeight, height);
        plane.maxHeight = Math.max(defaults.maxHeight, height);
        plane.speed = speed;
        plane.cruisePitch = pitch;
        plane.place(east, -north, height * this.projection.metersToNorm, yaw);
      },
      metersToNorm: () => this.projection.metersToNorm,
      settled: () => this.tileStreamer.isIdle() && !this.skyLightmap.converging,
      // something over the camera: the roof of the building it is in
      inside: () => this.skyLightmap.cityScene.pick(this.camera.worldPosition, Vector3.UP) != null,
    });
  }

  /** Airplane mode on or off, returns whether it is on now. */
  private toggleAirplane(): boolean {
    if (this.airplane.active) this.airplane.exit();
    else this.airplane.enter(this.projection.metersToNorm);
    this.orbitKeys.reset();
    return this.airplane.active;
  }

  /** Re-center the projection on a region and (re)start tile streaming there. */
  private async loadFromBounds(bounds: LatLngBounds): Promise<void> {
    this.tileStreamer.setTerrainStreamingEnabled(false);
    this.tileStreamer.setCityStreamingEnabled(false);
    this.tileStreamer.clearAllTerrainTiles();
    this.tileStreamer.clearAllCityTiles();
    this.picker.clearSelection();

    this.projection.setBounds(bounds);
    this.bounds = bounds;
    // the ground at the center is at y 0
    this.elevation.setOrigin(this.projection.latMid, this.projection.lngMid);
    // the sun stands elsewhere at another latitude (before init, applyTimeOfDay does it)
    if (this.timeOfDay) this.applySunAngles();
    // Another projection center: everything the lightmap holds is somewhere else now.
    this.skyLightmap.reset();
    this.sun?.shadowMap.reset();
    // the plane keeps flying, around the new center
    if (this.airplane.active) this.airplane.recenter();
    this.camera.pivot.position = new Vector3(0, 0, 0);
    this.tileStreamer.resetStreamingPosition();
    // The projection is recentered: all geometry jumps, no history reprojects.
    this.taa.invalidate();
    this.ssao?.reset();

    this.tileStreamer.setTerrainStreamingEnabled(true);
    if (!this.drawBuildings || this.drawBuildings.value) this.tileStreamer.setCityStreamingEnabled(true);
    this.dirty = true;
  }

  public init(): void {
    const paramGroup = new ParamGroup('controls');
    this.drawMode = paramGroup.addParamSelect('drawMode', DrawMode.Full, DRAW_MODES, { callback: () => this.applyDrawMode() });
    this.quality = paramGroup.addParamSelect('quality', isMobile() ? Quality.Medium : Quality.High, QUALITIES, { callback: () => this.applyQuality() });
    // the one of the URL, else a random one on every load, except in a bench: its frames have to compare between runs
    const timeOfDay = this.bench ? TimeOfDay.Afternoon : this.urlTimeOfDay ?? randomTimeOfDay();
    this.timeOfDay = paramGroup.addParamSelect('timeOfDay', timeOfDay, TIMES_OF_DAY, { callback: () => {
      this.applyTimeOfDay();
      // the GUI or a random city changed it: into the URL (once the camera stands still)
      this.urlPending = true;
    } });
    this.cap60 = paramGroup.addParamBool('cap60fps', isMobile());
    this.logDrawInfo = paramGroup.addParamBool('logDrawInfo', false);
    const setDirty = () => this.dirty = true;
    const layersGroup = paramGroup.addGroup('Layers');
    for (const layer of this.geometry.layers) {
      layer.enabled = layersGroup.addParamBool(layer.def.label, layer.def.defaultOn, { callback: setDirty });
    }
    this.drawBuildings = layersGroup.addParamBool('buildings', true, { callback: () => {
      this.dirty = true;
      this.ssao?.reset();
      if (this.drawBuildings.value) {
        this.tileStreamer.setCityStreamingEnabled(true);
        this.tileStreamer.resetStreamingPosition();
      } else {
        this.tileStreamer.setCityStreamingEnabled(false);
        this.tileStreamer.clearAllCityTiles();
        this.picker.clearSelection();
      }
    } });
    // in the building tiles, see worldTrees.ts: they load again
    const trees = layersGroup.addParamBool('trees', true, { callback: () => {
      this.dirty = true;
      this.ssao?.reset();
      this.picker.clearSelection();
      this.tileStreamer.setTreesEnabled(trees.value);
    } });
    this.tileStreamer.setTreesEnabled(trees.value);
    // the heights of the ground (ElevationService), off: the flat ground at 0. The tiles load again
    this.elevationParam = layersGroup.addParamBool('elevation', true, { callback: () => this.applyElevation() });
    if (this.elevationParam.value !== this.geometry.elevation) this.applyElevation();

    // the albedos of the map (as the params, not squared), in the gBuffer palette and the forward materials
    const pgColors = paramGroup.addGroup('Map colors');
    const toVector3 = (c: [number, number, number]) => new Vector3(c[0], c[1], c[2]);
    const colorParam = (name: string, value: Vector3, apply: (v: Vector3) => void) => {
      pgColors.addParamColor(name, value, { callback: (v) => {
        apply(v);
        this.dirty = true;
      } });
    };
    colorParam('buildings', new Vector3(1, 1, 1), (v) => this.geometry.setBuildingColor(v));
    colorParam('ground', new Vector3(.5, .5, .5), (v) => this.geometry.setGroundColor(v));
    this.geometry.layers.forEach((layer, i) => {
      colorParam(layer.def.label, toVector3(layer.def.color), (v) => this.geometry.setLayerColor(i, v));
    });
    colorParam('treeTrunks', toVector3(TREE_TRUNK_COLOR), (v) => this.geometry.setTreeTrunkColor(v));
    const crownAverage = TREE_CROWN_COLORS.reduce((a, c) => a.add(toVector3(c)), new Vector3(0, 0, 0));
    colorParam('treeCrowns', crownAverage.multiplyScalar(1 / TREE_CROWN_COLORS.length), (v) => this.geometry.setTreeCrownColor(v));

    const pgShading = paramGroup.addGroup('Shading');
    const params = { materials: this.materials, callback: setDirty };
    this.backgroundColor = pgShading.addParamColor('backgroundColor', this.clearColor.vector3, { ...params, callback: (v) => {
      this.clearColor.setValues(v.x, v.y, v.z, 1);
      this.dirty = true;
    } });
    this.skyColor = pgShading.addParamColor('skyColor', Color.fromHex('#96a3b0').vector3, params);
    this.sunColor = pgShading.addParamColor('sunColor', Color.fromHex('#b0a7a0').vector3, params);
    this.distanceFog = pgShading.addParamFloat('distanceFog', .25, 0, 1, params);
    this.viewDistanceMax = pgShading.addParamFloat('viewDistanceMax', 8, 3, 12, { ...params, callback: () => {
      this.applyViewDistanceMax();
      this.dirty = true;
    } });
    this.applyViewDistanceMax();
    pgShading.addParamFloat('contrast', .2, 0, 1, params);
    const toneMap = pgShading.addParamBool('toneMap', true, { callback: (v) => {
      this.deferred.toneMap = v;
      this.dirty = true;
    } });
    this.deferred.toneMap = toneMap.value;
    pgShading.addParamFloat('exposure', 1.5, 0, 4, params);
    // Only with the sky visibility lightmap: it keeps the BVH of the buildings up to date
    this.waterReflections = pgShading.addParamBool('waterReflections', false, { callback: setDirty });
    this.reflectionRays = pgShading.addParamSelect('reflectionRays', ReflectionRays.PerPixel, REFLECTION_RAYS, { callback: setDirty });
    this.reflectionDarkening = pgShading.addParamFloat('reflectionDarkening', .35, 0, 1, { callback: setDirty });
    this.waterRipples = pgShading.addParamBool('waterRipples', true, { callback: setDirty });
    const updateRipples = () => {
      // the bake scales the gradient by the height, which makes the slope of the ripples independent of their wavelength
      this.ripples.material.setFloat('rippleStrength', this.rippleSlope.value * this.ripples.wavelength);
      this.ripples.material.setFloat('speed', this.rippleSpeed.value);
      this.ripplesDirty = true;
      this.dirty = true;
    };
    this.rippleSlope = pgShading.addParamFloat('rippleSlope', .05, 0, .5, { callback: updateRipples });
    pgShading.addParamFloat('rippleWavelength', this.ripples.wavelength, 1, 50, { callback: (v) => {
      this.ripples.wavelength = v;
      updateRipples();
    } });
    this.rippleSpeed = pgShading.addParamFloat('rippleSpeed', .5, 0, 2, { callback: updateRipples });
    this.animateWater = pgShading.addParamBool('animateWater', true, { callback: setDirty });
    updateRipples();
    this.sunReflection = pgShading.addParamFloat('sunReflection', 8, 0, 16, { callback: setDirty });
    this.sunReflectionPower = pgShading.addParamFloat('sunReflectionPower', 200, 10, 1000, { callback: setDirty });
    // the fresnel term of the water: the sky it mirrors (scales the fresnel weight) and the glints at grazing angles
    pgShading.addParamFloat('skyReflection', 1, 0, 4, params);
    pgShading.addParamFloat('sunFresnel', 4, 0, 16, params);
    // the sun in the sky (deferred only, the forward pass clears to the background color)
    pgShading.addParamFloat('sunSize', .02, 0, .1, params);
    pgShading.addParamFloat('sunGlow', .4, 0, 1, params);
    this.sun.initParams(pgShading, this.materials, setDirty);
    this.applyTimeOfDay();

    this.applyDrawMode();
    this.sun.init(paramGroup);
    this.ssao?.init(paramGroup);
    this.skyLightmap.init(paramGroup);
    this.taa.init(paramGroup);
    this.tiltShift.init(paramGroup);
    this.traffic.init(paramGroup);
    this.tileStreamer.init(paramGroup);
    this.applyQuality();

    if (isMobile()) {
      paramGroup.collapsed = true;
      removeMobileConsole();
    }
    paramGroup.collapseChildrenRecursive();
    new ParamsGui(paramGroup);
    this.bench?.applyParams(this);

    this.update();
    if (this.video) this.video.ready = true;
  }

  /** After a change of drawMode: the debug views of the lightmap and the shadows. */
  private applyDrawMode(): void {
    const mode = this.drawMode.value as DrawMode;
    this.skyLightmap.view = SKY_VISIBILITY_VIEWS[mode] ?? LightmapView.Shaded;
    const shadowView = SHADOW_VIEWS[mode] ?? 0;
    this.deferred.shadowView = shadowView;
    // the forward shading (osm/shading.wgsl) only has the cascades view
    for (const m of this.geometry.forwardMaterials) m.setFloat('shadowDebug', mode === DrawMode.ShadowMapCascades ? 1 : 0);
    this.dirty = true;
  }

  /** The qualitySettings of the quality param into the params of the lightmap, the TAA and the tilt-shift (which the GUI follows). */
  private applyQuality(): void {
    const settings = qualitySettings[this.quality.value as Quality];
    this.skyLightmap.enabled.setValue(settings.skyVisibility);
    this.taa.enabled.setValue(settings.TAAEnabled);
    this.tiltShift.enabled.setValue(settings.tiltShiftEnabled);
    this.tiltShift.halfRes.setValue(settings.tiltShiftHalfRes);
    this.traffic.msaa.setValue(settings.carsMSAA);
    this.traffic.perPixelLighting.setValue(settings.carsPerPixelLighting);
    this.sun.mode.setValue(settings.shadowMode);
    this.waterReflections.setValue(settings.waterReflections);
    this.reflectionRays.setValue(settings.reflectionRays);
    this.waterRipples.setValue(settings.waterRipples);
    this.animateWater.setValue(settings.animateWater);
    this.forward = settings.forward;
    this.viewDistanceMax.setValue(settings.viewDistanceMax);
    // the canvas resizes in the next renderer.update()
    this.renderer.canvasManager.pixelRatioMax = settings.pixelRatioMax;
    this.skyLightmap.setQuality(settings.lightMapSize,settings.texelsPerPixel, settings.traceSamplesMax, settings.sampleCount, settings.referenceCellSize, settings.lightmapResolution, settings.searchCount);
    this.dirty = true;
  }

  /** The timeOfDaySettings of the timeOfDay param into the sun angle and the sky, sun and background color params (which the GUI follows). */
  private applyTimeOfDay(): void {
    const settings = timeOfDaySettings[this.timeOfDay.value as TimeOfDay];
    this.applySunAngles();
    this.skyColor.setValue(Color.fromHex(settings.skyColor).vector3);
    this.sunColor.setValue(Color.fromHex(settings.sunColor).vector3);
    this.backgroundColor.setValue(Color.fromHex(settings.backgroundColor).vector3);
    this.dirty = true;
  }

  /**
   * The orbit pivot eases to the height of the ground under it, so the camera orbits a point on the terrain and rides
   * over the hills while panning. Snaps when close: a still camera stops changing, so the map converges.
   */
  private followGround(): void {
    const pivot = this.camera.pivot.position;
    const ground = this.heightField.heightAt(pivot.x, pivot.z) ?? (this.elevationParam?.value === false ? 0 : null);
    if (ground === null || ground === pivot.y) return;
    let y = pivot.y + (ground - pivot.y) * (1 - Math.exp(-PIVOT_GROUND_RATE * Math.min(this.renderer.time.dt, .1)));
    if (Math.abs(ground - y) < PIVOT_GROUND_SNAP_M * this.projection.metersToNorm) y = ground;
    this.camera.pivot.setPositionValues(pivot.x, y, pivot.z);
  }

  /** The elevation param to the tiles (they load again) and the ground (meshes or the flat quad). */
  private applyElevation(): void {
    const on = this.elevationParam.value;
    this.geometry.elevation = on;
    this.tileStreamer.setElevation(on ? this.elevation : null);
    this.tileStreamer.clearAllTerrainTiles();
    this.tileStreamer.clearAllCityTiles();
    this.picker.clearSelection();
    // the ground and the buildings move: what the lightmap and the histories hold is somewhere else now
    this.skyLightmap.reset();
    this.sun?.shadowMap.reset();
    this.taa.invalidate();
    this.ssao?.reset();
    this.tileStreamer.resetStreamingPosition();
    this.dirty = true;
  }

  /** viewDistanceMax to the far plane (see updateFarPlane), the ground quad and the tile streaming. */
  private applyViewDistanceMax(): void {
    const distance = this.viewDistanceMax.value;
    this.updateFarPlane();
    this.geometry.setViewDistanceMax(distance);
    this.tileStreamer.setViewDistanceMax(distance);
  }

  /**
   * The far plane at the 3d distance of the ground at viewDistanceMax in xz, from the camera height. The fog is of the
   * horizontal distance (see getFog in deferredShading.wgsl): a fixed far plane at viewDistanceMax would clip the ground
   * before the fog hides it when the camera is up high, one with room for the highest camera would draw past the fog
   * when it is low. Every frame, after the camera moves: the readers of the depth take depthParams from the camera of
   * the frame, the histories hold camera distances, not depths.
   */
  private updateFarPlane(): void {
    const distance = this.viewDistanceMax.value;
    const height = Math.max(this.camera.worldPosition.y, 0);
    this.camera.projection.far = Math.sqrt(distance * distance + height * height);
  }

  /** The sun angles of the timeOfDay param at the latitude of the projection center, see timeOfDaySunAngles. */
  private applySunAngles(): void {
    const settings = timeOfDaySettings[this.timeOfDay.value as TimeOfDay];
    const { angleX, angleY } = timeOfDaySunAngles(settings, this.projection.latMid);
    this.sun.setSunAngles(angleX, angleY);
  }

  /**
   * The shadowMode param as it can be: the forward shading has only the shadow map (or none), the shadows in the
   * lightmap need the lightmap (else a ray per pixel).
   */
  private get shadowMode(): ShadowMode {
    const mode = this.sun.mode.value as ShadowMode;
    if (this.forward) return mode === ShadowMode.None ? ShadowMode.None : ShadowMode.ShadowMap;
    if (mode === ShadowMode.LightmapRaytrace && !this.skyLightmap.active) return ShadowMode.Raytrace;
    return mode;
  }

  private get showOverdraw(): boolean {
    return this.drawMode.value === DrawMode.Overdraw;
  }

  // The materials the shading params (sky, sun, fog, shadows) go to by name: the deferred shading, the forward ones and
  // the cars.
  private get materials() {
    return [...this.deferred.materials, ...this.geometry.forwardMaterials, ...this.traffic.materials];
  }

  /** The building meshes to draw, null while the buildings are hidden. */
  private get visibleCity(): CityMeshes | null {
    if (!this.drawBuildings.value) return null;
    const s = this.tileStreamer;
    return { meshes: s.cityMeshes, slots: s.cityMeshSlots, tiles: s.cityMeshTiles, quantization: s.cityMeshQuantization };
  }

  public update(time: number = 0): void {
    // Skip the vsyncs less than a 60 fps frame after the last rendered one. The 4 ms slack keeps a frame that is a
    // bit early (rAF times jitter) from being skipped: on a 120 Hz screen that renders every 2nd vsync, at 60 Hz every one.
    // Off while the profiler records: the idle vsyncs let the GPU clock down, so the passes time 1.5-3x too long.
    if (this.cap60.value && !this.bench && !this.video && !this.renderer.profiler.recording && time - this.lastFrameTime < 1000 / 60 - 4) {
      window.requestAnimationFrame((t) => this.update(t));
      return;
    }
    this.lastFrameTime = time;
    this.renderer.update(this.bench ? this.bench.nextTime() : this.video ? this.video.nextTime() : time);
    // Without the jitter of the last frame, so camera.changed only sees the camera motion. draw() jitters it.
    this.camera.projection.resetJitter();
    this.bench?.moveCamera(this.camera);
    if (this.airplane.active) this.airplane.update(this.projection.metersToNorm, this.video?.stick ?? this.ui?.joystickValue);
    else {
      if (!this.video && !this.bench) this.orbitKeys.update(this.ui?.joystickValue);
      // before the update, which sees whether the camera changed
      this.followGround();
      this.camera.update();
      // eases out the bank left over from airplane mode
      this.airplane.settle();
    }
    this.updateFarPlane();

    // the tiles whose building shadows fall into the view load too
    this.tileStreamer.setSunDirection(this.sun.lightDir);
    this.tileStreamer.update(this.renderer.time.dt);
    this.geometry.updateFrustum(this.camera);

    const cameraMoving = this.camera.changed;
    this.updateUrl(cameraMoving);
    if (!this.tileStreamer.isIdle()) {
      this.loadingCooldown = 0.5;
    } else {
      this.loadingCooldown = Math.max(0, this.loadingCooldown - this.renderer.time.dt);
    }
    const tilesLoading = this.loadingCooldown > 0;
    const cityTilesChanged = this.tileStreamer.consumeCityTilesChanged();
    // Only the deferred path draws the cars.
    if (!this.forward && !this.showOverdraw) {
      this.traffic.update(this.renderer.time.dt, this.camera, this.renderer.canvas.height);
    }

    this.draw(this.dirty, cameraMoving, tilesLoading, cityTilesChanged);
    this.dirty = false;
    if (this.screenshotRequested) {
      this.screenshotRequested = false;
      const stamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
      saveCanvasToDisk(this.renderer.canvas, `raytracedmap-${stamp}.png`);
    }
    if (this.bench) {
      const lm = this.skyLightmap;
      this.bench.afterFrame(this.frameState, this.tileStreamer.isIdle(), lm.unconvergedCells / Math.max(lm.cellCount, 1));
    }
    this.camera.projection.resetJitter();

    if (this.drawBuildings.value) this.picker.update(this.camera.viewProjection, this.camera.worldPosition, this.tileStreamer);

    // the video steps the frames itself
    if (!this.video) window.requestAnimationFrame((t) => this.update(t));
  }

  /**
   * In airplane mode the shadow cascades go in front of the plane: it can't look back, so around the pivot most of
   * each cascade would lie behind it, never seen.
   */
  private shadowsAhead(): ShadowMapAhead | null {
    if (!this.airplane.active) return null;
    const origin = this.camera.worldPosition.clone();
    origin.y = this.camera.pivot.worldPosition.y;
    const forward = Vector3.subtract(this.camera.pivot.worldPosition, origin);
    forward.y = 0;
    // the pivot: where the view ray of the screen center hits the ground, the tilt-shift is sharp there
    const focus = forward.length();
    if (focus < 1e-6) return null;
    forward.normalize();
    return { origin, forward, focus };
  }

  private draw(dirty: boolean, cameraMoving: boolean, tilesLoading: boolean, cityTilesChanged: boolean) {
    let shadowsSettled = true;
    // the ground casts shadows too, with elevation (without it the flat ground is at 0, what an empty texel means)
    const ground = this.geometry.elevation ? this.geometry.groundBuffer : null;
    const groundChanged = this.tileStreamer.consumeGroundChanged();
    if (groundChanged) this.sun.terrainSun.setDirty();
    // only the shadowMap mode and the debug views of the shadow map need it
    const mode = this.drawMode.value as DrawMode;
    const shadowMapNeeded = this.shadowMode === ShadowMode.ShadowMap || mode === DrawMode.ShadowMap || mode === DrawMode.ShadowMapCascades;
    if (!this.showOverdraw && shadowMapNeeded) {
      // the casters may have changed while it was skipped
      if (this.shadowMapSkipped) this.sun.shadowMap.reset();
      this.shadowMapSkipped = false;
      shadowsSettled = this.sun.update(this.camera, cityTilesChanged || groundChanged,
        { meshes: this.tileStreamer.cityMeshes, quantization: this.tileStreamer.cityMeshQuantization }, ground, this.shadowsAhead());
    } else {
      this.shadowMapSkipped = true;
    }
    // the rays of the ray traced sun shadows don't trace the ground: the hills come from the sun's view of it
    const rayTracedSun = this.shadowMode === ShadowMode.LightmapRaytrace || this.shadowMode === ShadowMode.Raytrace;
    if (!this.showOverdraw && rayTracedSun) this.sun.updateTerrainSun(this.camera, this.viewDistanceMax.value, ground);

    if (this.drawMode.value === DrawMode.ShadowMap) {
      // instead of the scene, applyDrawMode draws it again when the mode changes back
      this.sun.drawView(this.projection.metersToNorm);
    } else if (this.drawMode.value === DrawMode.WallAtlas) {
      this.wallAtlasView.draw(this.tileStreamer.lastCityTile, this.projection.metersToNorm);
    } else if (this.forward && !this.showOverdraw && this.sceneColorRT) {
      this.drawForward(this.getForwardRT(), cameraMoving, tilesLoading, dirty || cameraMoving || tilesLoading || cityTilesChanged || !shadowsSettled);
    } else {
      this.drawDeferred(dirty, cameraMoving, tilesLoading, cityTilesChanged, shadowsSettled);
    }

    this.renderer.finish();

    if (this.logDrawInfo.value) {
      const stats = this.geometry.stats;
      console.log(
        'meshCount', stats.meshCount,
        'vertexCount', stats.vertexCount,
        'culledCount', stats.culledCount,
        'workers', this.tileStreamer.activeWorkers,
      );
    }
  }

  private getForwardRT(): RenderTexture {
    this.forwardRT ??= new RenderTexture(this.renderer, 1, 1, {
      formats: ['rgb10a2unorm'],
      depth: true,
      // only drawForward's pass tests it, the bounding boxes after it don't
      discardDepth: true,
      sampleCount: 1,
      scaleToCanvas: true,
      toScreen: false,
    });
    return this.forwardRT;
  }

  /**
   * The forward quality: the map shaded forward into sceneColor, without the gBuffer, lightmap, SSAO, TAA or
   * tilt-shift. Only drawn when something changed, else its last image is presented again.
   */
  private drawForward(sceneColor: RenderTexture, cameraMoving: boolean, tilesLoading: boolean, changed: boolean): void {
    const renderMap = changed || this.forwardStale;
    this.forwardStale = false;
    this.frameState = { newJitter: false, lightmapUpdating: false, ssaoAccumulating: false, renderMap, tilesLoading, cameraMoving };
    // the deferred path starts over when it comes back
    this.taa.invalidate();
    this.ssao?.reset();
    this.tiltShift.invalidate();
    this.postSSAOVersion = -1;

    this.geometry.resetStats();
    if (renderMap) {
      // the forward shading has only the shadow map: none turns it down
      const strength = this.shadowMode === ShadowMode.None ? 0 : this.sun.shadowStrength.value;
      for (const m of this.geometry.forwardMaterials) m.setFloat('shadowStrength', strength);
      const city = this.visibleCity;
      this.renderer.startRenderPass(sceneColor, { clearColor: this.clearColor, colorClear: true, depthClear: true }, 0, 'mapForward');
      this.geometry.drawForward(this.camera, city, this.picker.selectedID);
      if (this.drawMode.value === DrawMode.BoundingBoxes) {
        this.renderer.startRenderPass(sceneColor, { colorClear: false, depthClear: false });
        this.geometry.drawBoundingBoxes(this.camera.viewProjection, city);
      }
    }
    this.renderer.blit(sceneColor, null, null, true);
  }

  private drawDeferred(dirty: boolean, cameraMoving: boolean, tilesLoading: boolean, cityTilesChanged: boolean, shadowsSettled: boolean): void {
    // the forward path draws again when it comes back
    this.forwardStale = true;
    // a shadow debug view replaces the scene: no SSAO, tilt-shift or cars over it
    const shadowView = SHADOW_VIEWS[this.drawMode.value as DrawMode] != null;
    const useSceneColor = this.sceneColorRT != null && !this.showOverdraw;
    // The lightmap updates while the camera moves, geometry or params change, and while its texels in view converge.
    const lightmap = this.skyLightmap;
    // before converging is read below: a move of the sun clears the sun visibility of the texels
    const shadowMode = this.shadowMode;
    lightmap.setSun(shadowMode === ShadowMode.LightmapRaytrace, this.sun.lightDir, this.sun.sunRadius.value);
    const lightmapActive = lightmap.active && useSceneColor;
    // The lightmap view doesn't move with the camera: no TAA, tilt-shift or SSAO, drawn every frame
    const showsLightmap = useSceneColor && lightmap.showsLightmap;
    const geometryChanged = cameraMoving || tilesLoading || dirty || cityTilesChanged;
    const lightmapUpdating = lightmapActive && (geometryChanged || lightmap.converging);
    // The water reflections mirrored on moving ripples change every frame: the map is drawn every frame, and the TAA
    // blends it like motion instead of averaging it into a static image.
    // the buildings in the water need the BVH of the lightmap, the ripples and the sun don't
    const reflectBuildings = this.waterReflections.value && lightmap.active;
    const reflections = useSceneColor && (reflectBuildings || this.waterRipples.value);
    const rippling = reflections && this.waterRipples.value && this.rippleSlope.value > 0;
    const waterAnimating = rippling && this.animateWater.value && this.rippleSpeed.value > 0;
    // Also over the lightmap: its texels are too coarse for the contact occlusion in the corners, the SSAO adds it.
    // Not in the sky visibility draw modes: they show the lightmap's own occlusion.
    const skyVisibilityMode = SKY_VISIBILITY_VIEWS[this.drawMode.value as DrawMode] != null;
    const ssao = this.ssao;
    const ssaoActive = ssao != null && useSceneColor && !showsLightmap && !skyVisibilityMode && ssao.active && !shadowView;
    const ssaoMode = cameraMoving ? SSAOMode.Moving : tilesLoading ? SSAOMode.Loading : SSAOMode.Converging;
    // The AO on the sky term only, in the shading (ReprojectedSSAO.skyOnly): the map is drawn while it accumulates, the
    // TAA blends it like motion until it is settling, like the lightmap below. When it starts over or is switched, the
    // TAA history holds the old AO: the TAA starts over too.
    const ssaoInShading = ssaoActive && ssao.shadesSky;
    const shadedSSAORestart = ssaoInShading ? ssao.restartVersion : -1;
    const ssaoRestarted = shadedSSAORestart !== this.shadedSSAORestart;
    this.shadedSSAORestart = shadedSSAORestart;
    const ssaoAccumulating = ssaoActive && ssao.accumulating(ssaoMode);
    const shadedSSAOSettled = !ssaoInShading || !ssaoAccumulating || ssao!.settling;
    // A converging lightmap changes the image every frame: the TAA blends it like motion (which smooths its noise) and
    // starts its static average once the lightmap is settling (the sky visibility, see SkyVisibilityLightmap.settling):
    // its last frames go into the average, which takes the TAA as long as the lightmap needs for them.
    const lightmapSettling = lightmapUpdating && !geometryChanged && lightmap.settling;
    const isStatic = !cameraMoving && !tilesLoading && shadowsSettled && !dirty && !waterAnimating && (!lightmapUpdating || lightmapSettling) &&
      !ssaoRestarted && shadedSSAOSettled;
    if (!useSceneColor) {
      this.taa.invalidate();
    } else if (!isStatic) {
      this.taa.restart();
    }
    // A static view is drawn until the TAA and the lightmap converged, the AO keeps accumulating after that (in the
    // shading: the map is drawn until it converged too). Both take a new jitter offset each frame, and the gBuffer that
    // goes with it.
    const renderMap = !useSceneColor || !this.taa.converged || showsLightmap || lightmapUpdating || waterAnimating ||
      (ssaoInShading && ssaoAccumulating);
    const newJitter = useSceneColor && (renderMap || ssaoAccumulating);
    this.frameState = { newJitter, lightmapUpdating, ssaoAccumulating, renderMap, tilesLoading, cameraMoving };
    this.taa.setCamera(this.camera, newJitter);
    const viewProjection = this.camera.viewProjection;
    // The texels of the lightmap follow the tilt-shift blur, the shading reads them with the same blur. The SSAO
    // converges to fewer samples under it.
    const blur: LightmapBlur = this.tiltShift.active && !shadowView ? this.tiltShift.getBlur(this.camera) : { slope: 0, max: 0 };
    const metersToNorm = this.projection.metersToNorm;
    const city = this.visibleCity;

    if (waterAnimating) this.rippleTime += this.renderer.time.dt;
    if (renderMap && rippling && (waterAnimating || this.ripplesDirty)) {
      this.ripples.update(this.rippleTime);
      this.ripplesDirty = false;
    }

    // The gBuffer with the camera of this frame: the shading, the lightmap, the SSAO and the TAA read it.
    this.geometry.resetStats();
    if (newJitter) {
      this.gBuffer.render(this.camera, metersToNorm, () => this.geometry.drawGBuffer(this.camera, city, this.picker.selectedID));
    }
    if (lightmapUpdating && newJitter) {
      lightmap.update({
        camera: this.camera,
        blur,
        metersToNorm,
        changed: geometryChanged,
        gBuffer: this.gBuffer,
      });
    }
    // The AO, before the shading reads it (skyOnly) and before the scene is presented. Called every frame it is active:
    // the reprojection needs the history of the frame before.
    if (ssaoActive) {
      ssao.update(this.camera, this.taa, ssaoMode, this.gBuffer, blur);
    } else {
      ssao?.reset();
    }

    // The ray traced shadows: soft with the TAA, which averages the points on the sun, hard without (no noise). They
    // trace the city of the lightmap, kept up to date here too for when the lightmap is off. With the sun in the
    // lightmap the shading reads its texels, the rays only until the lightmap has its buffers. The cars take hard rays
    // in both, at the corners of each car or per pixel (CarTraffic.perPixelLighting): they aren't in the lightmap
    const rayTraced = shadowMode === ShadowMode.LightmapRaytrace || shadowMode === ShadowMode.Raytrace;
    this.deferred.setSunShadows(shadowMode, rayTraced && this.taa.enabled.value);
    this.traffic.sunShadows = shadowMode === ShadowMode.None ? 0 : rayTraced ? 2 : 1;
    this.traffic.lightDirection.copy(this.sun.lightDir);
    if (rayTraced && useSceneColor) {
      lightmap.cityScene.enable();
      lightmap.cityScene.update(this.renderer.time.frameNumber);
    }

    if (this.showOverdraw) {
      this.geometry.drawOverdraw(this.camera, city);
    } else if (renderMap && this.sceneColorRT) {
      lightmap.setResolveUniforms(this.camera, blur, metersToNorm, this.taa.enabled.value);
      const rippleTile = rippling ? this.ripples.tileSize * metersToNorm : 0;
      // The rays of the buildings in the water: 1 per block of pixels, traced here for the shading to blend, or a ray per
      // water pixel in the shading. The traced pixel of the blocks rotates for the TAA to average.
      const [blockX, blockY] = REFLECTION_BLOCKS[this.reflectionRays.value];
      const reflectionPass = reflections && reflectBuildings && blockX * blockY > 1 ? this.waterReflection : null;
      if (reflectionPass) {
        reflectionPass.setBlock(blockX, blockY);
        reflectionPass.draw(this.camera, this.gBuffer, this.waterMaterials, rippleTile, this.taa.enabled.value);
      }
      this.deferred.setWaterReflections(reflections ? this.waterMaterials : 0, reflectBuildings, this.reflectionDarkening.value, rippleTile,
        this.sunReflection.value, this.sunReflectionPower.value, reflectionPass);
      if (lightmap.halfResActive) lightmap.halfResPass.draw(this.camera, this.gBuffer);
      this.deferred.setSSAO(ssao, ssaoInShading);
      this.deferred.draw(this.camera, this.gBuffer, this.sceneColorRT, lightmap.halfResPass);
      if (this.drawMode.value === DrawMode.BoundingBoxes) {
        this.renderer.startRenderPass(this.sceneColorRT, { colorClear: false, depthClear: false });
        this.geometry.drawBoundingBoxes(viewProjection, city);
      }
      if (!showsLightmap) this.taa.resolve(this.camera, this.sceneColorRT, this.gBuffer.depthView, isStatic);
    }

    // Composited over the image, the AO's version tells whether the tilt-shift input changed. In the shading it is a
    // part of the map.
    const ssaoVersion = ssaoActive && !ssaoInShading ? ssao.version : -1;

    // The scene and its AO go into the tilt-shift target, which is blurred onto the screen after. Also with tilt-shift
    // off: its composite is then only the dither, the last pass onto the 8 bit canvas, which breaks up the banding of
    // the TAA (it averages the dither of the shading away). While neither the scene nor the AO changed, the target and
    // its blurs still hold this frame: only the tilt-shift composite runs.
    const postTarget = useSceneColor && !showsLightmap && !shadowView ? this.tiltShift.target : null;
    // The cars move every frame: over the image, before the tilt-shift. Not over the debug views of the lightmap, the
    // shadows and the SSAO.
    const drawCars = useSceneColor && !showsLightmap && !shadowView && !(ssaoActive && ssao.showOnly) && this.traffic.active;
    const reusePost = postTarget != null && !renderMap && !drawCars && ssaoVersion === this.postSSAOVersion && this.tiltShift.holdsTarget;
    this.postSSAOVersion = ssaoVersion;

    if (!reusePost) {
      // The AO copies the scene into the target in the same pass. The SSAO debug view replaces the scene, it isn't
      // drawn under it.
      const ssaoDrawn = ssaoActive && !ssaoInShading && ssao.composite(this.distanceFog.value, this.viewDistanceMax.value, this.taa.output, postTarget);
      if (!ssaoDrawn && useSceneColor && !(ssaoActive && ssao.showOnly)) {
        this.renderer.blit(showsLightmap ? this.sceneColorRT : this.taa.output, postTarget, null, true);
      }
      if (drawCars) this.traffic.draw(this.camera, this.taa.viewProjectionNoJitter, this.gBuffer, postTarget);
    }

    if (postTarget) {
      // In a static view only the jitter, the AO and the lightmap's noise change: the blurs may lag a few frames. Not
      // the cars: they'd move in steps in the blurred rows.
      this.tiltShift.draw(this.camera, !reusePost, null, !geometryChanged && !drawCars);
    } else {
      this.tiltShift.invalidate();
    }
    // over everything of the frame, when its param is on
    lightmap.drawRayGraph();
  }

  destruct(): void {
    this.ui?.destruct();
    this.orbitKeys?.destruct();
    this.picker?.destruct();
    this.tileStreamer.destruct();
    this.workerPool?.destroy();
    this.geometry?.destruct();
    this.ssao?.destruct();
    this.traffic?.destruct();
    this.taa?.destruct();
    this.gBuffer?.destruct();
    this.deferred?.destruct();
    this.ripples?.texture.destruct();
    this.waterReflection?.destruct();
    this.skyLightmap?.destruct();
    this.tiltShift?.destruct();
    this.sun?.destruct();
    this.wallAtlasView?.destruct();
    this.sceneColorRT?.destruct();
    this.forwardRT?.destruct();
  }
}
