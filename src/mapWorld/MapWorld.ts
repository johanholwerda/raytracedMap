import Renderer from 'webGPU-framework/renderer/render/Renderer';
import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import ParamsGui from 'webGPU-framework/utils/paramsGui/ParamsGui';
import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import KeyListener from 'webGPU-framework/utils/input/KeyListener';
import { KEY_P } from 'webGPU-framework/utils/input/Key';
import { saveCanvasToDisk } from 'webGPU-framework/utils/image/ImageUtils';
import type { LatLngBounds } from './map/geo';
import MapManager from './map/MapManager';
import MapCameraRig from './camera/MapCameraRig';
import MapRenderer from './rendering/MapRenderer';
import MapUrlSync from './MapUrlSync';
import MapWorldBench from './debug/MapWorldBench';
import MapWorldVideo from './debug/MapWorldVideo';
import BuildingPicker from './ui/BuildingPicker';
import MapWorldUI from './ui/MapWorldUI';
import {
  QUALITIES, Quality, qualitySettings, randomTimeOfDay, TIMES_OF_DAY, TimeOfDay, timeOfDaySettings, timeOfDaySunAngles,
} from './MapWorldSettings';

/** The quality the map starts with, see the quality param. */
function startQuality(): Quality {
  return isMobile() ? Quality.Medium : Quality.High;
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
 * sync, 2D region picker), but shaded deferred, fed by worldwide OpenStreetMap data:
 *  - ground layers + extruded buildings from OpenFreeMap vector tiles
 *    (OpenMapTiles schema, one request per tile), and
 *  - street addresses for clicked buildings from Nominatim.
 * It starts on a random major world capital.
 *
 * The parts:
 *  - MapManager: the map data (the place, the tile streaming, the heights of the ground), it starts before the GPU
 *  - MapCameraRig: the orbit camera riding the ground, and the airplane mode
 *  - MapRenderer: draws the frame (the shadows, the deferred or the forward path, the post), with the shading params
 *  - MapUrlSync, BuildingPicker and MapWorldUI (the DOM overlays)
 * This class starts them, ties them together (a new place, the elevation, the quality, the time of day) and runs the
 * frame loop.
 */
export default class MapWorld {
  private renderer!: Renderer;
  private mapManager: MapManager;
  private cameraRig!: MapCameraRig;
  private mapRenderer!: MapRenderer;
  private urlSync?: MapUrlSync;
  private picker!: BuildingPicker;
  private ui!: MapWorldUI;
  private quality!: ParamSelect;
  private timeOfDay!: ParamSelect;
  // Renders on every other vsync of a 120 Hz screen: an even 60 fps instead of an uneven 80 (1 and 2 vsyncs mixed).
  private cap60!: ParamBool;
  private logDrawInfo!: ParamBool;
  // The rAF time of the last rendered frame, ms.
  private lastFrameTime: number = -Infinity;
  // ?time=sunset: the time of day preset to start with, see init
  private urlTimeOfDay?: TimeOfDay;
  // P saves the canvas as a png, right after the next drawn frame: a WebGPU canvas only holds its image in the task
  // that drew it.
  private screenshotRequested: boolean = false;

  // ?bench: the self-running benchmark, see MapWorldBench
  private bench?: MapWorldBench;
  // ?video: a frame at a time for video/record.mjs instead of on requestAnimationFrame, see MapWorldVideo
  private video?: MapWorldVideo;

  constructor() {
    this.mapManager = new MapManager({
      onTileError: (msg) => this.ui?.showTileError(msg),
      onRedraw: () => this.mapRenderer.markDirty(),
      onContentChanged: () => {
        this.mapRenderer.markDirty();
        this.mapRenderer.ssao?.reset();
      },
      onClearSelection: () => this.picker.clearSelection(),
      onElevationChanged: () => this.applyElevation(),
    });
  }

  // What the scripts in video/ read through window.__mapWorld.
  get camera(): MapCamera { return this.cameraRig.camera; }
  get tileStreamer() { return this.mapManager.tileStreamer; }
  get workerPool() { return this.mapManager.workerPool; }
  get geometry() { return this.mapRenderer.geometry; }
  get frameState() { return this.mapRenderer.frameState; }

  /**
   * A param by its path from MapWorld or one of its parts (the renderer, the water, the snow, the map manager, the
   * camera rig), e.g. 'quality', 'animateWater', 'tiltShift.aperture': for ?set of MapWorldBench and the params of
   * MapWorldVideo. Private fields too. Undefined when no part has it.
   */
  public findParam(path: string): any {
    const roots: object[] = [this, this.mapRenderer, this.mapRenderer.water, this.mapRenderer.snow, this.mapManager, this.cameraRig];
    for (const root of roots) {
      let target: any = root;
      for (const key of path.split('.')) target = target?.[key];
      if (target?.setValue) return target;
    }
    return undefined;
  }

  // false when WebGPU isn't available: the Renderer then shows its "WebGPU Not Supported" message and nothing else starts
  async start(canvasParent: HTMLElement, preloader: WebGPUPreLoader, uiVisible: boolean): Promise<boolean> {
    LogGL.ENABLED = true;

    // router5 strips query params on start; bootstrap.ts keeps the original.
    const urlParams = new URLSearchParams(window.location.search || (window as any).__initialSearch || '');

    // The tiles of the first view, before the GPU: asking for the WebGPU device takes half a second or more of waiting,
    // in which the tiles are fetched and built already (see MapManager.startTiles). The map draws them in its first
    // frames. Of the quality init starts with: its view distance
    const map = this.mapManager;
    const tilesStarted = map.startTiles(urlParams, canvasParent, qualitySettings[startQuality()].viewDistanceMax);

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
    if (!this.renderer.device) {
      map.workerPool.destroy();
      return false;
    }
    // await this.renderer.init(canvasParent, { sampleCount: 1, pixelRatio: 1});
    if (urlParams.has('bench')) {
      this.bench = new MapWorldBench(this.renderer, urlParams);
      // for setting params from the console between runs
      (window as any).__mapWorld = this;
    }
    // for video/tileLatency.mjs, which reads the timings and moves the camera
    if (urlParams.has('tileTimings') || urlParams.has('remote')) (window as any).__mapWorld = this;
    if (urlParams.has('video')) this.video = this.createVideo();
    this.renderer.setClearColor(0, 0, 0, 1);

    this.cameraRig = new MapCameraRig(this.renderer, map.projection, map.heightField, () => !map.elevationEnabled,
      (origin, direction) => map.buildingsEnabled ? this.mapRenderer.skyLightmap.cityScene.pick(origin, direction)?.t ?? null : null);
    this.mapRenderer = new MapRenderer(this.renderer, preloader, this.camera, map.projection, map.heightField);
    this.mapRenderer.onViewDistanceMaxChange = (distance) => {
      this.cameraRig.updateFarPlane(distance);
      map.tileStreamer.setViewDistanceMax(distance);
    };
    // Click-to-select buildings; the address comes from Nominatim.
    this.picker = new BuildingPicker(this.renderer, canvasParent, () => this.mapRenderer.markDirty());

    KeyListener.addKeyHitEvent(KEY_P, () => this.screenshotRequested = true);

    this.ui = new MapWorldUI(canvasParent, {
      getViewBounds: () => map.currentViewBounds(this.camera.pivot.position),
      onSelectBounds: (bounds) => void this.loadFromBounds(bounds),
      onRandomCity: () => void this.goToRandomCity(),
      onToggleAirplane: () => this.cameraRig.toggleAirplane(),
      onGoToPlace: (place) => {
        LogGL.log(`[MapWorld] search: ${place.title} (${place.lat.toFixed(5)}, ${place.lng.toFixed(5)})`);
        this.ui.showLocationTitle(place.title, place.subtitle);
        void this.loadFromBounds(map.boundsAround(place.lat, place.lng));
      },
    });

    // where the tiles startTiles asked for go: the built ones wait for the first frame
    const view = this.mapRenderer;
    map.tileStreamer.attach({
      renderer: this.renderer,
      camera: this.camera,
      layers: view.geometry.layers,
      meshBounds: view.geometry.meshBounds,
      cityScene: view.skyLightmap.cityScene,
      roads: view.traffic,
      ground: view.geometry.groundBuffer,
    });

    const { urlCamera, city, time } = await tilesStarted;
    // not in a bench: its frames have to compare between runs
    const timeIndex = TIMES_OF_DAY.indexOf(urlParams.get('time') ?? time ?? '');
    if (!this.bench && timeIndex >= 0) this.urlTimeOfDay = timeIndex as TimeOfDay;
    this.urlSync = new MapUrlSync();
    if (city) this.ui.showLocationTitle(city.name, city.landmark || undefined);
    if (!this.video && !this.bench) this.ui.showControlsHint();

    // the place is set and its tiles are asked for (startTiles): the rest of loadFromBounds
    this.resetView();
    if (urlCamera) this.cameraRig.applyUrlCamera(urlCamera);
    return true;
  }

  /** Bounds centered on lat/lng with the DEFAULT_BOUNDS span, see MapManager. Also for video/loadTiming.mjs. */
  private boundsAround(lat: number, lng: number): LatLngBounds {
    return this.mapManager.boundsAround(lat, lng);
  }

  /**
   * Jump to a random city (weighted by sqrt(population)) or, half the time, a
   * uniformly picked notable building. Avoids landing where we already are.
   */
  private async goToRandomCity(): Promise<void> {
    const place = await this.mapManager.pickRandomPlace(this.camera.pivot.position);
    LogGL.log(`[MapWorld] random location: ${place.label}`);
    this.ui.showLocationTitle(place.title, place.building);
    // another time of day too (the callback applies it, the GUI follows); loadFromBounds then sets the sun angles for the new latitude
    let timeOfDay = randomTimeOfDay();
    for (let i = 0; i < 8 && timeOfDay === this.timeOfDay.value; i++) timeOfDay = randomTimeOfDay();
    this.timeOfDay.setValue(timeOfDay);
    await this.loadFromBounds(this.boundsAround(place.lat, place.lng));
  }

  /** The video mode: what MapWorldVideo needs of the map, the streaming and the plane. */
  private createVideo(): MapWorldVideo {
    const defaults = { minHeight: 0, maxHeight: 0 };
    const projection = this.mapManager.projection;
    return new MapWorldVideo(this.renderer, {
      findParam: (path) => this.findParam(path),
      step: () => this.update(),
      load: (lat, lng) => this.loadFromBounds(this.boundsAround(lat, lng)),
      place: (east, north, height, yaw, speed, pitch) => {
        const plane = this.cameraRig.airplane;
        if (!plane.active) {
          plane.enter(projection.metersToNorm);
          defaults.minHeight = plane.minHeight;
          defaults.maxHeight = plane.maxHeight;
        }
        // a shot may fly lower or higher than the keys can
        plane.minHeight = Math.min(defaults.minHeight, height);
        plane.maxHeight = Math.max(defaults.maxHeight, height);
        plane.speed = speed;
        plane.cruisePitch = pitch;
        plane.place(east, -north, height * projection.metersToNorm, yaw);
      },
      metersToNorm: () => projection.metersToNorm,
      settled: () => this.mapManager.tileStreamer.isIdle() && !this.mapRenderer.skyLightmap.converging,
      // something over the camera: the roof of the building it is in
      inside: () => this.mapRenderer.skyLightmap.cityScene.pick(this.camera.worldPosition, Vector3.UP) != null,
    });
  }

  /** Re-center the projection on a region and (re)start tile streaming there. */
  private async loadFromBounds(bounds: LatLngBounds): Promise<void> {
    this.mapManager.stopStreaming();
    this.picker.clearSelection();

    this.mapManager.setPlace(bounds);
    this.resetView();
  }

  /** After setPlace: the camera on the center of the new projection, all that was of the old one gone, the tiles streaming. */
  private resetView(): void {
    // the sun stands elsewhere at another latitude (before init, applyTimeOfDay does it)
    if (this.timeOfDay) this.applySunAngles();
    // Another projection center: everything the lightmap and the histories hold is somewhere else now, all geometry
    // jumps, no history reprojects.
    this.mapRenderer.resetHistories();
    // the plane keeps flying, around the new center
    this.cameraRig.recenter();
    this.mapManager.restartStreaming();
    this.mapRenderer.markDirty();
  }

  public init(): void {
    const paramGroup = new ParamGroup('controls');
    this.mapRenderer.initDrawMode(paramGroup);
    this.quality = paramGroup.addParamSelect('quality', startQuality(), QUALITIES, { callback: () => this.applyQuality() });
    // the one of the URL, else a random one on every load, except in a bench: its frames have to compare between runs
    const timeOfDay = this.bench ? TimeOfDay.Afternoon : this.urlTimeOfDay ?? randomTimeOfDay();
    this.timeOfDay = paramGroup.addParamSelect('timeOfDay', timeOfDay, TIMES_OF_DAY, { callback: () => {
      this.applyTimeOfDay();
      // the GUI or a random city changed it: into the URL (once the camera stands still)
      this.urlSync?.markPending();
    } });
    this.cap60 = paramGroup.addParamBool('cap60fps', isMobile());
    this.logDrawInfo = paramGroup.addParamBool('logDrawInfo', false);
    const layersGroup = paramGroup.addGroup('Layers');
    this.mapRenderer.initLayerParams(layersGroup);
    this.mapManager.initParams(layersGroup);
    if (this.mapManager.elevationEnabled !== this.mapRenderer.geometry.elevation) this.applyElevation();

    this.mapRenderer.initShadingParams(paramGroup);
    this.applyTimeOfDay();

    this.mapRenderer.initPasses(paramGroup);
    this.mapManager.tileStreamer.init(paramGroup);
    this.applyQuality();

    if (isMobile()) {
      paramGroup.collapsed = true;
      removeMobileConsole();
    }
    paramGroup.collapseChildrenRecursive();
    new ParamsGui(paramGroup);
    this.bench?.applyParams((path) => this.findParam(path));

    this.update();
    if (this.video) this.video.ready = true;
  }

  /** The qualitySettings of the quality param into the renderer (the GUI follows) and the pixel ratio of the canvas. */
  private applyQuality(): void {
    const settings = qualitySettings[this.quality.value as Quality];
    this.mapRenderer.applyQuality(settings);
    // the canvas resizes in the next renderer.update()
    this.renderer.canvasManager.pixelRatioMax = settings.pixelRatioMax;
  }

  /** The timeOfDaySettings of the timeOfDay param into the sun angle and the sky, sun and background color params (which the GUI follows). */
  private applyTimeOfDay(): void {
    const settings = timeOfDaySettings[this.timeOfDay.value as TimeOfDay];
    this.applySunAngles();
    this.mapRenderer.setSkyColors(settings);
  }

  /** The sun angles of the timeOfDay param at the latitude of the projection center, see timeOfDaySunAngles. */
  private applySunAngles(): void {
    const settings = timeOfDaySettings[this.timeOfDay.value as TimeOfDay];
    const { angleX, angleY } = timeOfDaySunAngles(settings, this.mapManager.projection.latMid);
    this.mapRenderer.sun.setSunAngles(angleX, angleY);
  }

  /** The elevation param to the tiles (they load again) and the ground (meshes or the flat quad). */
  private applyElevation(): void {
    const on = this.mapManager.elevationEnabled;
    this.mapRenderer.geometry.elevation = on;
    this.mapManager.applyElevation();
    this.picker.clearSelection();
    // the ground and the buildings move: what the lightmap and the histories hold is somewhere else now
    this.mapRenderer.resetHistories();
    this.mapManager.tileStreamer.resetStreamingPosition();
    this.mapRenderer.markDirty();
  }

  /** Into the URL once the camera stands still, see MapUrlSync. Not while flying (the plane never stands still), in a bench or a video. */
  private updateUrl(moving: boolean): void {
    if (!this.urlSync || this.bench || this.video || this.cameraRig.airplane.active) return;
    this.urlSync.update(moving, this.camera, this.mapManager.projection, this.timeOfDay.value as TimeOfDay);
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
    const map = this.mapManager;
    const camera = this.camera;
    // Without the jitter of the last frame, so camera.changed only sees the camera motion. The renderer jitters it.
    camera.projection.resetJitter();
    this.bench?.moveCamera(camera);
    this.cameraRig.update(this.video?.stick ?? this.ui?.joystickValue, !this.video && !this.bench);
    this.cameraRig.updateFarPlane(this.mapRenderer.viewDistance);

    const tilesLoading = map.update(this.renderer.time.dt, this.mapRenderer.sun.lightDir);
    this.mapRenderer.updateFrustum();
    const cameraMoving = camera.changed;
    this.updateUrl(cameraMoving);
    map.cullTrees(camera, this.renderer.canvas.height);

    const streamer = map.tileStreamer;
    const pivot = camera.pivot.position;
    this.mapRenderer.render({
      cameraMoving,
      tilesLoading,
      cityTilesChanged: streamer.consumeCityTilesChanged(),
      groundChanged: streamer.consumeGroundChanged(),
      city: map.visibleCity,
      shadowCasters: { meshes: streamer.cityMeshes, quantization: streamer.cityMeshQuantization },
      lastCityTile: streamer.lastCityTile,
      selectedID: this.picker.selectedID,
      shadowsAhead: this.cameraRig.shadowsAhead(),
      pivotLatitude: map.projection.worldToLatLng(pivot.x, pivot.z).lat,
      originHeightM: map.elevation.originHeight,
    });
    if (this.logDrawInfo.value) {
      const stats = this.mapRenderer.geometry.stats;
      console.log(
        'meshCount', stats.meshCount,
        'vertexCount', stats.vertexCount,
        'culledCount', stats.culledCount,
        'workers', streamer.activeWorkers,
      );
    }
    map.frameSubmitted();

    if (this.screenshotRequested) {
      this.screenshotRequested = false;
      const stamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '-');
      saveCanvasToDisk(this.renderer.canvas, `raytracedmap-${stamp}.png`);
    }
    if (this.bench) {
      const lm = this.mapRenderer.skyLightmap;
      this.bench.afterFrame(this.frameState, streamer.isIdle(), lm.unconvergedCells / Math.max(lm.cellCount, 1));
    }
    camera.projection.resetJitter();

    if (map.buildingsEnabled) this.picker.update(camera.viewProjection, camera.worldPosition, streamer);

    // the video steps the frames itself
    if (!this.video) window.requestAnimationFrame((t) => this.update(t));
  }

  destruct(): void {
    this.ui?.destruct();
    this.cameraRig?.destruct();
    this.picker?.destruct();
    this.urlSync?.destruct();
    this.mapManager.destruct();
    this.mapRenderer?.destruct();
  }
}
