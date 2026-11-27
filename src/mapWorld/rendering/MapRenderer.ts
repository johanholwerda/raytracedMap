import type Renderer from 'webGPU-framework/renderer/render/Renderer';
import type WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import type Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Color from 'webGPU-framework/renderer/math/Color';
import type Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import type ParamColor from 'webGPU-framework/utils/params/ParamColor';
import type ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import type ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import type ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import type MapProjection from '../map/MapProjection';
import type TerrainHeightField from '../map/TerrainHeightField';
import type WorldTileStreamer from '../map/WorldTileStreamer';
import { TREE_CROWN_COLORS, TREE_TRUNK_COLOR } from '../map/worldTrees';
import CarTraffic from '../cars/CarTraffic';
import type { MapWorldFrameState } from '../debug/MapWorldBench';
import WallAtlasView from '../debug/WallAtlasView';
import {
  DRAW_MODES, DrawMode, type QualitySettings, SHADOW_VIEWS, ShadowMode, SKY_VISIBILITY_VIEWS, type TimeOfDaySettings,
} from '../MapWorldSettings';
import MapGeometry, { type CityMeshes } from './MapGeometry';
import MapSnow from './MapSnow';
import ForwardPath from './ForwardPath';
import PostProcess from './PostProcess';
import type { DeferredFrame } from './DeferredFrame';
import DeferredShading from './deferred/DeferredShading';
import MapGBuffer from './deferred/MapGBuffer';
import MapWater from './deferred/MapWater';
import ReprojectedSSAO, { SSAOMode } from './ssao/ReprojectedSSAO';
import TAA from './taa/TAA';
import TiltShift from './tiltShift/TiltShift';
import SkyVisibilityLightmap, { LightmapView } from './skyVisibilityLightmap/SkyVisibilityLightmap';
import type { LightmapBlur } from './skyVisibilityLightmap/SkyVisibilityLightmap';
import SunShadows from './sun/SunShadows';
import type { ShadowMapAhead } from './sun/ShadowMap';

// The SSAO (ReprojectedSSAO): off since 2026-11-23, on the sky term alone it is too subtle next to the sky visibility
// lightmap. Not constructed (nor in the GUI) while false. Its params then take their defaults, no quality sets them; the
// qualities had amount 1, rayMarchSteps 6 (1 on low), randomStep on medium. Cost at 4K (3080): 0.55 ms with 6 steps,
// 0.19 ms with 1 (only the contact occlusion), 0.20 ms with randomStep (~44% of the occlusion).
const SSAO_ENABLED = false;

/** What MapRenderer.render needs of the rest of the map in a frame. */
export type MapFrame = {
  cameraMoving: boolean;
  // tiles of the view are loading, see MapManager.update
  tilesLoading: boolean;
  // city tiles came or went (WorldTileStreamer.consumeCityTilesChanged)
  cityTilesChanged: boolean;
  // ground tiles came or went (WorldTileStreamer.consumeGroundChanged)
  groundChanged: boolean;
  // the building meshes to draw, null while the buildings are hidden
  city: CityMeshes | null;
  // the building meshes that cast shadows into the shadow map (also while hidden)
  shadowCasters: { meshes: Mesh[]; quantization: number[] };
  // for the wallAtlas drawMode
  lastCityTile: WorldTileStreamer['lastCityTile'];
  // the building the picker selected, -1 without
  selectedID: number;
  // the shadow cascades in front of the plane, see MapCameraRig.shadowsAhead
  shadowsAhead: ShadowMapAhead | null;
  // for the snow line
  pivotLatitude: number;
  originHeightM: number;
};

/**
 * Draws the map, shaded deferred: 1 gBuffer pass with a jittered camera (MapGBuffer), shaded once per pixel
 * (DeferredShading) with the light of the sky visibility lightmap, the reprojected SSAO over it, and TAA instead of
 * MSAA, then the PostProcess (the AO composite, the cars and the tilt-shift). The forward quality takes the
 * ForwardPath instead. It owns the passes, what is drawn (MapGeometry, CarTraffic), the sun (SunShadows), the water
 * (MapWater), the snow (MapSnow) and the params of the shading.
 *
 * A frame: the shadows (shadow map, terrain sun), then the forward path, the deferred path or a debug view. The
 * deferred path decides what it does at its start (beginDeferredFrame), draws the scene (drawDeferredScene) and the post.
 */
export default class MapRenderer {
  public readonly geometry: MapGeometry;
  public readonly sun: SunShadows;
  // The sky the points see instead of the sky term, see SkyVisibilityLightmap. It owns the CityScene.
  public readonly skyLightmap: SkyVisibilityLightmap;
  // The cars on the roads, drawn over the converged image, see CarTraffic.
  public readonly traffic: CarTraffic;
  public readonly taa: TAA;
  public readonly tiltShift: TiltShift;
  // null without SSAO_ENABLED: not constructed then
  public readonly ssao: ReprojectedSSAO | null = null;
  public readonly water: MapWater;
  public readonly snow: MapSnow;
  private gBuffer: MapGBuffer;
  private deferred: DeferredShading;
  private wallAtlasView: WallAtlasView;
  private forwardPath: ForwardPath;
  private post: PostProcess;
  // The shaded gBuffer of this frame (jittered), the TAA resolves it. A static view stops being drawn once the TAA
  // converged: its history is re-presented, the SSAO composites over it.
  private sceneColorRT: RenderTexture;

  public drawMode!: ParamSelect;
  // QualitySettings.forward of the quality param
  private forward: boolean = false;
  private clearColor = Color.fromHex('#b8dbff');
  private backgroundColor!: ParamColor;
  private skyColor!: ParamColor;
  private sunColor!: ParamColor;
  private distanceFog!: ParamFloat;
  // Where the map ends, in world units from the camera: the fog is 1 there (see getFog in deferredShading.wgsl), the
  // far plane, the ground quad and the ground tiles reach that far (see applyViewDistanceMax). At least 3: the pivot is
  // up to 2 (setDistanceLimit) from the camera and stays out of the fog
  private viewDistanceMax!: ParamFloat;
  // the far plane and the tiles follow viewDistanceMax, see MapWorld
  public onViewDistanceMaxChange?: (distance: number) => void;

  // the map draws again: a param or what is drawn changed
  private dirty: boolean = true;
  // ReprojectedSSAO.restartVersion the shading (its TAA history) holds, -1 without the AO in the shading (skyOnly).
  private shadedSSAORestart: number = -1;
  // The shadow map was not updated in a frame (ray traced shadows, overdraw): it may miss casters that came or went.
  private shadowMapSkipped: boolean = false;
  // what the last frame did, for the bench and video/tileLatency.mjs
  public frameState: MapWorldFrameState = {
    newJitter: false, lightmapUpdating: false, ssaoAccumulating: false, renderMap: false, tilesLoading: false, cameraMoving: false,
  };

  constructor(private renderer: Renderer, preloader: WebGPUPreLoader, private camera: MapCamera, private projection: MapProjection,
    private heightField: TerrainHeightField) {
    if (SSAO_ENABLED) this.ssao = new ReprojectedSSAO(renderer, preloader);
    this.taa = new TAA(renderer);
    this.tiltShift = new TiltShift(renderer);
    // The AO converges to fewer samples under the blur, the lightmap texels follow it too
    this.tiltShift.onChange = () => {
      this.ssao?.reset();
      this.dirty = true;
    };
    this.gBuffer = new MapGBuffer(renderer);
    this.deferred = new DeferredShading(renderer);
    this.sceneColorRT = new RenderTexture(renderer, 1, 1, {
      // 10 bits: the TAA reads it, the 1 / 255 steps of 8 bits would show through its average (see TiltShift)
      formats: ['rgb10a2unorm'],
      sampleCount: 1,
      scaleToCanvas: true,
    });

    this.skyLightmap = new SkyVisibilityLightmap(renderer, preloader);
    this.skyLightmap.onChange = () => this.dirty = true;
    // The deferred shading reads the lightmap.
    for (const material of this.deferred.materials) this.skyLightmap.addResolveMaterial(material, true);
    // the points on the sun of the soft ray traced shadows
    this.deferred.bindBlueNoise(this.skyLightmap.blueNoise);

    this.traffic = new CarTraffic(renderer, projection);
    // The cars are lit by the sky visibility of the ground under them (looked up per car).
    for (const shader of this.traffic.resolveShaders) this.skyLightmap.addResolveMaterial(shader);
    // The ray traced shadows on the cars (per car) trace the buildings of the lightmap.
    this.traffic.bindCity(this.skyLightmap.cityScene);
    this.traffic.setHeightField(heightField);

    this.geometry = new MapGeometry(renderer, projection, this.gBuffer);
    // on by default, see the elevation param
    this.geometry.elevation = true;
    // The water reflections trace the buildings of the lightmap.
    this.deferred.bindCity(this.skyLightmap.cityScene);
    this.water = new MapWater(renderer, this.deferred, this.gBuffer, this.geometry, this.skyLightmap.cityScene, heightField);
    this.snow = new MapSnow(this.geometry, this.water.horizon, projection, heightField);

    this.sun = new SunShadows(renderer);
    this.wallAtlasView = new WallAtlasView(renderer);
    for (const material of this.materials) this.sun.bindResources(material);
    // the hills for the ray traced sun shadows, which trace only the buildings and the trees
    for (const material of [...this.deferred.materials, ...this.traffic.materials, ...this.water.materials]) {
      this.sun.terrainSun.bind(material);
    }
    // and its height map for the ground in the water reflections, with the materials of the ground for its color
    for (const material of [...this.deferred.materials, ...this.water.materials]) this.sun.terrainSun.bindHeights(material);
    for (const material of this.water.materials) this.sun.terrainSun.bindMaterials(material);

    this.forwardPath = new ForwardPath(renderer, this.geometry);
    this.post = new PostProcess(renderer, this.taa, this.tiltShift, this.traffic, this.gBuffer, this.sceneColorRT);
  }

  /** The map draws again in the next frame. */
  public markDirty(): void {
    this.dirty = true;
  }

  /** The ground or the projection moved: what the lightmap, the shadow map and the histories hold is somewhere else now. */
  public resetHistories(): void {
    this.skyLightmap.reset();
    this.sun.shadowMap.reset();
    this.taa.invalidate();
    this.ssao?.reset();
  }

  /** The value of the viewDistanceMax param, see there. */
  public get viewDistance(): number {
    return this.viewDistanceMax.value;
  }

  /** The drawMode param, the first of the controls. */
  public initDrawMode(paramGroup: ParamGroup): void {
    this.drawMode = paramGroup.addParamSelect('drawMode', DrawMode.Full, DRAW_MODES, { callback: () => this.applyDrawMode() });
  }

  /** The layers of the map, on or off. */
  public initLayerParams(layersGroup: ParamGroup): void {
    const setDirty = () => this.dirty = true;
    for (const layer of this.geometry.layers) {
      layer.enabled = layersGroup.addParamBool(layer.def.label, layer.def.defaultOn, { callback: setDirty });
    }
  }

  /** The Map colors and the Shading groups (with the Snow and the Water). */
  public initShadingParams(paramGroup: ParamGroup): void {
    const setDirty = () => this.dirty = true;
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
      // the glaciers take the snowColor param
      if (layer.def.key === 'glacier') return;
      colorParam(layer.def.label, toVector3(layer.def.color), (v) => this.geometry.setLayerColor(i, v));
    });
    colorParam('treeTrunks', toVector3(TREE_TRUNK_COLOR), (v) => this.geometry.setTreeTrunkColor(v));
    const crownAverage = TREE_CROWN_COLORS.reduce((a, c) => a.add(toVector3(c)), new Vector3(0, 0, 0));
    colorParam('treeCrowns', crownAverage.multiplyScalar(1 / TREE_CROWN_COLORS.length), (v) => this.geometry.setTreeCrownColor(v));

    const pgShading = paramGroup.addGroup('Shading');

    const sunAndSky = pgShading.addGroup('Sun and Sky');

    // the snow above the snow line (see map/snowLine.ts), on the ground and the layers of MapGeometry's SNOW_LAYERS
    this.snow.initParams(pgShading, setDirty);

    const params = { materials: this.shadingMaterials, callback: setDirty };
    this.backgroundColor = sunAndSky.addParamColor('backgroundColor', this.clearColor.vector3, { ...params, callback: (v) => {
      this.clearColor.setValues(v.x, v.y, v.z, 1);
      this.dirty = true;
    } });
    this.skyColor = sunAndSky.addParamColor('skyColor', Color.fromHex('#96a3b0').vector3, params);
    this.sunColor = sunAndSky.addParamColor('sunColor', Color.fromHex('#b0a7a0').vector3, params);

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

    this.water.initParams(pgShading, params, setDirty);

    // the sun in the sky (deferred only, the forward pass clears to the background color)
    sunAndSky.addParamFloat('sunSize', .02, 0, .1, params);
    sunAndSky.addParamFloat('sunGlow', .4, 0, 1, params);
    this.sun.initParams(pgShading, this.shadingMaterials, setDirty);
  }

  /** The groups of the passes, after the shading params. */
  public initPasses(paramGroup: ParamGroup): void {
    this.applyDrawMode();
    this.sun.init(paramGroup);
    this.ssao?.init(paramGroup);
    this.skyLightmap.init(paramGroup);
    this.taa.init(paramGroup);
    this.tiltShift.init(paramGroup);
    this.traffic.init(paramGroup);
  }

  /** The qualitySettings into the params of the lightmap, the TAA, the tilt-shift, the cars, the sun and the water (which the GUI follows). */
  public applyQuality(settings: QualitySettings): void {
    this.skyLightmap.enabled.setValue(settings.skyVisibility);
    this.taa.enabled.setValue(settings.TAAEnabled);
    this.tiltShift.enabled.setValue(settings.tiltShiftEnabled);
    this.tiltShift.halfRes.setValue(settings.tiltShiftHalfRes);
    this.traffic.msaa.setValue(settings.carsMSAA);
    this.traffic.perPixelLighting.setValue(settings.carsPerPixelLighting);
    this.sun.mode.setValue(settings.shadowMode);
    this.water.applyQuality(settings);
    this.forward = settings.forward;
    this.viewDistanceMax.setValue(settings.viewDistanceMax);
    this.skyLightmap.setQuality(settings.lightMapSize,settings.texelsPerPixel, settings.traceSamplesMax, settings.sampleCount, settings.referenceCellSize, settings.lightmapResolution, settings.searchCount);
    this.dirty = true;
  }

  /** The sky, sun and background colors of a time of day (which the GUI follows). */
  public setSkyColors(settings: TimeOfDaySettings): void {
    this.skyColor.setValue(Color.fromHex(settings.skyColor).vector3);
    this.sunColor.setValue(Color.fromHex(settings.sunColor).vector3);
    this.backgroundColor.setValue(Color.fromHex(settings.backgroundColor).vector3);
    this.dirty = true;
  }

  /** After a change of drawMode: the debug views of the lightmap and the shadows. */
  private applyDrawMode(): void {
    const mode = this.drawMode.value as DrawMode;
    this.skyLightmap.view = SKY_VISIBILITY_VIEWS[mode] ?? LightmapView.Shaded;
    const shadowView = SHADOW_VIEWS[mode] ?? 0;
    this.deferred.shadowView = shadowView;
    // the forward shading (rendering/shaders/shading.wgsl) only has the cascades view
    for (const m of this.geometry.forwardMaterials) m.setFloat('shadowDebug', mode === DrawMode.ShadowMapCascades ? 1 : 0);
    this.dirty = true;
  }

  /** viewDistanceMax to the ground quad, and to the far plane and the tile streaming (onViewDistanceMaxChange). */
  private applyViewDistanceMax(): void {
    const distance = this.viewDistanceMax.value;
    this.onViewDistanceMaxChange?.(distance);
    this.geometry.setViewDistanceMax(distance);
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

  // the materials the sun and sky params go to by their names: with the one that shades the ground in the water
  // reflections
  private get shadingMaterials() {
    return [...this.materials, ...this.water.materials];
  }

  /** After the camera moved: the meshes out of view are culled. */
  public updateFrustum(): void {
    this.geometry.updateFrustum(this.camera);
  }

  /** Draws the frame. */
  public render(frame: MapFrame): void {
    // Only the deferred path draws the cars.
    if (!this.forward && !this.showOverdraw) {
      this.traffic.update(this.renderer.time.dt, this.camera, this.renderer.canvas.height);
    }
    if (this.snow.update(frame.pivotLatitude, frame.originHeightM)) this.dirty = true;
    if (this.water.updateTerrainReflection(this.skyLightmap.active, this.geometry.elevation, this.projection.metersToNorm)) this.dirty = true;

    const dirty = this.dirty;
    const shadowsSettled = this.updateShadows(frame);
    const mode = this.drawMode.value as DrawMode;
    if (mode === DrawMode.ShadowMap) {
      // instead of the scene, applyDrawMode draws it again when the mode changes back
      this.sun.drawView(this.projection.metersToNorm);
    } else if (mode === DrawMode.WallAtlas) {
      this.wallAtlasView.draw(frame.lastCityTile, this.projection.metersToNorm);
    } else if (this.forward && !this.showOverdraw) {
      this.drawForward(frame, dirty || frame.cameraMoving || frame.tilesLoading || frame.cityTilesChanged || !shadowsSettled);
    } else {
      this.drawDeferred(frame, dirty, shadowsSettled);
    }

    this.renderer.finish();
    this.dirty = false;
  }

  /**
   * The shadow map (when the shadow mode or a debug view needs it) and the sun's view of the ground (for the ray traced
   * sun shadows and the ground in the water reflections).
   * @returns whether the shadow map is settled: its cascades are all up to date
   */
  private updateShadows(frame: MapFrame): boolean {
    let shadowsSettled = true;
    // the ground casts shadows too, with elevation (without it the flat ground is at 0, what an empty texel means)
    const ground = this.geometry.elevation ? this.geometry.groundBuffer : null;
    if (frame.groundChanged) this.sun.terrainSun.setDirty();
    // only the shadowMap mode and the debug views of the shadow map need it
    const mode = this.drawMode.value as DrawMode;
    const shadowMapNeeded = this.shadowMode === ShadowMode.ShadowMap || mode === DrawMode.ShadowMap || mode === DrawMode.ShadowMapCascades;
    if (!this.showOverdraw && shadowMapNeeded) {
      // the casters may have changed while it was skipped
      if (this.shadowMapSkipped) this.sun.shadowMap.reset();
      this.shadowMapSkipped = false;
      shadowsSettled = this.sun.update(this.camera, frame.cityTilesChanged || frame.groundChanged, frame.shadowCasters, ground, frame.shadowsAhead);
    } else {
      this.shadowMapSkipped = true;
    }
    // the rays of the ray traced sun shadows don't trace the ground: the hills come from the sun's view of it
    const rayTracedSun = this.shadowMode === ShadowMode.LightmapRaytrace || this.shadowMode === ShadowMode.Raytrace;
    // the water reflections march the height map it is made from
    const reflectTerrain = this.water.reflectsTerrain;
    if (!this.showOverdraw && (rayTracedSun || reflectTerrain)) {
      // with the materials of the ground, for its color in the water
      const field = this.heightField;
      const metersToNorm = this.projection.metersToNorm;
      const materials = reflectTerrain
        ? { draw: (viewProjection: Matrix4x4) => this.geometry.drawMaterialMap(viewProjection), top: field.maxHeightM * metersToNorm, bottom: field.minHeightM * metersToNorm }
        : null;
      this.sun.updateTerrainSun(this.camera, this.viewDistanceMax.value, ground, materials);
    }
    return shadowsSettled;
  }

  /** The forward quality, see ForwardPath. The deferred path starts over when it comes back. */
  private drawForward(frame: MapFrame, changed: boolean): void {
    this.taa.invalidate();
    this.ssao?.reset();
    this.post.invalidate();
    // the forward shading has only the shadow map: none turns it down
    const strength = this.shadowMode === ShadowMode.None ? 0 : this.sun.shadowStrength.value;
    const renderMap = this.forwardPath.draw(this.camera, frame.city, frame.selectedID, this.clearColor, strength,
      this.drawMode.value === DrawMode.BoundingBoxes, changed);
    this.frameState = {
      newJitter: false, lightmapUpdating: false, ssaoAccumulating: false, renderMap, tilesLoading: frame.tilesLoading, cameraMoving: frame.cameraMoving,
    };
  }

  private drawDeferred(frame: MapFrame, dirty: boolean, shadowsSettled: boolean): void {
    // the forward path draws again when it comes back
    this.forwardPath.markStale();
    const plan = this.beginDeferredFrame(frame, dirty, shadowsSettled);
    const { newJitter, lightmapUpdating, ssaoAccumulating, renderMap } = plan;
    this.frameState = { newJitter, lightmapUpdating, ssaoAccumulating, renderMap, tilesLoading: frame.tilesLoading, cameraMoving: frame.cameraMoving };
    this.drawDeferredScene(plan, frame);
    this.post.draw(plan, this.camera, this.distanceFog.value, this.viewDistanceMax.value);
    // over everything of the frame, when its param is on
    this.skyLightmap.drawRayGraph();
  }

  /**
   * What the deferred path does in this frame. Also tells the lightmap where the sun is, and restarts the TAA when the
   * view is not static: the frame is drawn until it converged again.
   */
  private beginDeferredFrame(frame: MapFrame, dirty: boolean, shadowsSettled: boolean): DeferredFrame {
    const { cameraMoving, tilesLoading, cityTilesChanged } = frame;
    // a shadow debug view replaces the scene: no SSAO, tilt-shift or cars over it
    const shadowView = SHADOW_VIEWS[this.drawMode.value as DrawMode] != null;
    const useSceneColor = !this.showOverdraw;
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
    const water = this.water.plan(useSceneColor, lightmap.active);
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
    const isStatic = !cameraMoving && !tilesLoading && shadowsSettled && !dirty && !water.animating && (!lightmapUpdating || lightmapSettling) &&
      !ssaoRestarted && shadedSSAOSettled;
    if (!useSceneColor) {
      this.taa.invalidate();
    } else if (!isStatic) {
      this.taa.restart();
    }
    // A static view is drawn until the TAA and the lightmap converged, the AO keeps accumulating after that (in the
    // shading: the map is drawn until it converged too). Both take a new jitter offset each frame, and the gBuffer that
    // goes with it.
    const renderMap = !useSceneColor || !this.taa.converged || showsLightmap || lightmapUpdating || water.animating ||
      (ssaoInShading && ssaoAccumulating);
    const newJitter = useSceneColor && (renderMap || ssaoAccumulating);
    return {
      shadowMode, shadowView, useSceneColor, showsLightmap, geometryChanged, lightmapUpdating, water,
      ssao: ssaoActive ? ssao : null, ssaoMode, ssaoInShading, ssaoAccumulating, isStatic, renderMap, newJitter,
    };
  }

  /** The gBuffer, the lightmap, the SSAO, the shading and the TAA of the deferred path. */
  private drawDeferredScene(plan: DeferredFrame, frame: MapFrame): void {
    const camera = this.camera;
    const lightmap = this.skyLightmap;
    const { shadowMode, renderMap, newJitter } = plan;
    this.taa.setCamera(camera, newJitter);
    const viewProjection = camera.viewProjection;
    // The texels of the lightmap follow the tilt-shift blur, the shading reads them with the same blur. The SSAO
    // converges to fewer samples under it.
    const blur: LightmapBlur = this.tiltShift.active && !plan.shadowView ? this.tiltShift.getBlur(camera) : { slope: 0, max: 0 };
    const metersToNorm = this.projection.metersToNorm;
    const city = frame.city;

    this.water.updateRipples(plan.water, renderMap, this.renderer.time.dt);

    // The gBuffer with the camera of this frame: the shading, the lightmap, the SSAO and the TAA read it.
    this.geometry.resetStats();
    if (newJitter) {
      this.gBuffer.render(camera, metersToNorm, () => this.geometry.drawGBuffer(camera, city, frame.selectedID));
    }
    if (plan.lightmapUpdating && newJitter) {
      lightmap.update({
        camera,
        blur,
        metersToNorm,
        changed: plan.geometryChanged,
        gBuffer: this.gBuffer,
      });
    }
    // The AO, before the shading reads it (skyOnly) and before the scene is presented. Called every frame it is active:
    // the reprojection needs the history of the frame before.
    if (plan.ssao) {
      plan.ssao.update(camera, this.taa, plan.ssaoMode, this.gBuffer, blur);
    } else {
      this.ssao?.reset();
    }

    // The ray traced shadows: soft with the TAA, which averages the points on the sun, hard without (no noise). They
    // trace the city of the lightmap, kept up to date here too for when the lightmap is off. With the sun in the
    // lightmap the shading reads its texels, the rays only until the lightmap has its buffers. The cars take hard rays
    // in both, at the corners of each car or per pixel (CarTraffic.perPixelLighting): they aren't in the lightmap
    const rayTraced = shadowMode === ShadowMode.LightmapRaytrace || shadowMode === ShadowMode.Raytrace;
    this.deferred.setSunShadows(shadowMode, rayTraced && this.taa.enabled.value);
    this.traffic.sunShadows = shadowMode === ShadowMode.None ? 0 : rayTraced ? 2 : 1;
    this.traffic.lightDirection.copy(this.sun.lightDir);
    if (rayTraced && plan.useSceneColor) {
      lightmap.cityScene.enable();
      lightmap.cityScene.update(this.renderer.time.frameNumber);
    }

    if (this.showOverdraw) {
      this.geometry.drawOverdraw(camera, city);
    } else if (renderMap) {
      lightmap.setResolveUniforms(camera, blur, metersToNorm, this.taa.enabled.value);
      this.water.draw(plan.water, camera, this.gBuffer, metersToNorm, this.taa.enabled.value && plan.isStatic);
      if (lightmap.halfResActive) lightmap.halfResPass.draw(camera, this.gBuffer);
      this.deferred.setSSAO(this.ssao, plan.ssaoInShading);
      this.deferred.draw(camera, this.gBuffer, this.sceneColorRT, lightmap.halfResPass);
      if (this.drawMode.value === DrawMode.BoundingBoxes) {
        this.renderer.startRenderPass(this.sceneColorRT, { colorClear: false, depthClear: false });
        this.geometry.drawBoundingBoxes(viewProjection, city);
      }
      if (!plan.showsLightmap) this.taa.resolve(camera, this.sceneColorRT, this.gBuffer.depthView, plan.isStatic);
    }
  }

  destruct(): void {
    this.geometry.destruct();
    this.ssao?.destruct();
    this.traffic.destruct();
    this.taa.destruct();
    this.gBuffer.destruct();
    this.deferred.destruct();
    this.water.destruct();
    this.skyLightmap.destruct();
    this.tiltShift.destruct();
    this.sun.destruct();
    this.wallAtlasView.destruct();
    this.sceneColorRT.destruct();
    this.forwardPath.destruct();
  }
}
