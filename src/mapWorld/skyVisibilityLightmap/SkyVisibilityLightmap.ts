import Renderer from 'webGPU-framework/renderer/render/Renderer';
import ComputeShader from 'webGPU-framework/renderer/material/ComputeShader';
import ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import ParamInt from 'webGPU-framework/utils/params/ParamInt';
import ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import ParamLog from 'webGPU-framework/utils/params/ParamLog';
import BlueNoiseTexture2D16Bits from 'webGPU-framework/utils/texture/BlueNoiseTexture2D16Bits';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import r2 from 'webGPU-framework/shaders/r2.wgsl?raw';
import cameraRay from '../cameraRay.wgsl?raw';
import cityLightmapSurface from '../city/cityLightmapSurface.wgsl?raw';
import gBufferPacking from '../deferred/mapGBufferPacking.wgsl?raw';
import cityLightmap from '../city/cityLightmap.wgsl?raw';
import skyVisibilityResolve from './skyVisibilityResolve.wgsl?raw';
import skyVisibilityTexel from './skyVisibilityTexel.wgsl?raw';
import skyVisibilityTrace from './skyVisibilityTrace.wgsl?raw';
import updateSkyVisibilityLightmap from './updateSkyVisibilityLightmap.wgsl?raw';
import CityScene from '../city/CityScene';
import MapGBuffer from '../deferred/MapGBuffer';
import SkyVisibilityHalfRes from './SkyVisibilityHalfRes';
import RayCountGraph from './RayCountGraph';

//the lightmap sizes of the dropdown. Above 4096 the lighting buffer is over the default storage binding limit (128 MiB):
//MapWorld requests the adapter's limit on desktop, and a size that doesn't fit falls back to the largest that does
//(getFittingSize)
export const LIGHTMAP_SIZE_OPTIONS = [1024, 2048, 3072, 4096, 6144];
//the paths of a cell per sun ray while the lightmap has the sun and both take samples, see setSun. And for a cell
//whose texel is in a penumbra so far, where the noise of the sun rays shows: see trace_main for both
const SUN_PATH_RATIO = 8;
const SUN_PENUMBRA_PATH_RATIO = 4;
//the size of the buffer of the buildings the rays hit, see CityScene. The default storage binding limit
const CITY_BUFFER_BYTES = 128 * 1024 * 1024;
//packedTexels: the most paths of a texel: what the 9 bits of the sky rays hold (PACKED_SKY_MAX of
//skyVisibilityTexel.wgsl), as a multiple of PACKED_SUN_SCALE. The sun rays, a quarter of it, then fit their 7 bits too
const PACKED_ACCUMULATION_MAX = 508;
//packedTexels: the sun rays count as this many paths (PACKED_SUN_SCALE of skyVisibilityTexel.wgsl)
const PACKED_SUN_SCALE = 4;

/** The blur of the tilt shift, see TiltShift.getBlur. */
export type LightmapBlur = { slope: number; max: number };

/** The debug views of the deferred shading of the lightmap: the override LIGHTMAP_VIEW of deferredShading.wgsl. */
export enum LightmapView {
  // the scene, lit by the lightmap
  Shaded = 0,
  // the sky visibility alone
  Light = 1,
  // the lightmap level of each pixel in a color
  Levels = 2,
  // the pixels whose texel still takes samples in yellow
  Converging = 3,
  // the lightmap itself instead of the scene
  Lightmap = 4,
  // the tiles of the lightmap in a color each, their texels as a checkerboard
  Tiles = 5,
}

/** The resolution of the lightmap lookup of the shading, see SkyVisibilityHalfRes. */
export enum LightmapResolution {
  // a lookup per pixel, in the shading
  Full,
  // 1 per 2 x 2 pixels, the shading blends those on the surface of a pixel
  Half,
  // 1 per 2 x 2 pixels, the shading samples them bilinearly across the surfaces, without the surface test: cheaper,
  // but the lighting bleeds a pixel across every edge
  HalfBilinear,
}

export const LIGHTMAP_RESOLUTIONS = ['fullRes', 'halfRes', 'halfResBilinear'];

export type LightmapFrame = {
  camera: MapCamera;
  blur: LightmapBlur;
  metersToNorm: number;
  //the camera moved, tiles came or went, or a param changed: the converged state of the last frames no longer holds
  changed: boolean;
  //of this frame, drawn with the camera (jittered or not), see MapGBuffer
  gBuffer: MapGBuffer;
};

/*
A virtual lightmap (see irradianceCache/virtualLightmap/VirtualLightmap.ts) with the albedos of the buildings and the ground at 0: ray traced ambient occlusion. A ray that hits a
surface adds nothing, so a texel holds the part of the sky its point sees (cosine weighted: 1 on an open roof, .5 on an
open wall), and the deferred shading multiplies it by the sky color (deferredShading.wgsl). The same
virtual lightmap, charts (cityLayout.ts), cells and claims as a lightmap with bounces would use, and what that makes simpler:
-1 ray per path, an any hit one (skyVisibilityTrace.wgsl): no hit point, normal, surface or ground plane, no lighting
 cached at the hit, no sun ray from it. A path that goes down would end on the ground: it is mirrored up, or skipped
-the visibility doesn't depend on the sun or the sky, so a change of the light changes nothing here: no light changes to
 follow, no refresh of the resolved texels when the light stops changing, no ageing of a tile that comes back in view
-a texel is 1 word (2 x f16) instead of 2, added to with a compare exchange instead of a lock. Or, with packedTexels, 1
 word of ray counts for the sky and the sun together, added to with 1 compare exchange (skyVisibilityTexel.wgsl)
-a tile that comes or goes still resets the texels around it (CityScene), and the sun visibility of the ones its
 shadows can fall on, but no staleness spreads through the bounces
It owns the CityScene, which the tile streamer fills.
With the sun (setSun) the texels hold the sun visibility too, for the sun shadows of the deferred shading: a second
half of lightingBuffer, the same texels, which the cells that trace add sun rays to. That half depends on the sun: it is
cleared when the sun moves, the sky half is kept.
 */
export default class SkyVisibilityLightmap {
  private renderer: Renderer;
  public readonly cityScene: CityScene;
  private claimShader: ComputeShader;
  private selectShader: ComputeShader;
  private traceShader: ComputeShader;
  //packedTexels: clears the sun fields when the sun moves
  private clearSunShader: ComputeShader;
  private updateShaders: ComputeShader[];
  //the materials that read the lightmap (the deferred shading), see addResolveMaterial
  private resolveMaterials: ShaderBase[] = [];
  //the ones of them with the debug views and the switches as overrides (deferredShading.wgsl), see setResolveUniforms
  private viewMaterials = new Set<ShaderBase>();
  //the ones of them with the override LIGHTMAP_SUN, see setSun
  private sunMaterials: ShaderBase[] = [];
  //the includes of a material that reads the lightmap (deferredShading.wgsl)
  public static readonly resolveIncludes = [hash, cityLightmapSurface, gBufferPacking, cityLightmap, skyVisibilityTexel, skyVisibilityResolve];

  //the size of the gBuffer of the last update, for the pixelSize and pixelScale of the resolve
  private gBufferWidth = 0;
  private gBufferHeight = 0;
  //2 channel blue noise for the paths, also of the points on the sun of the ray traced shadows (DeferredShading)
  public readonly blueNoise: BlueNoiseTexture2D16Bits;

  //the buffers of the lightmap, null until it is first enabled
  private tileBuffer: GPUBuffer | null = null;
  private tileTimeBuffer: GPUBuffer | null = null;
  private lightingBuffer: GPUBuffer | null = null;
  private claimBuffer: GPUBuffer | null = null;
  private tileCountBuffer: GPUBuffer;
  private traceArgsBuffer: GPUBuffer;
  private traceIndirectBuffer: GPUBuffer;
  private traceCellsBuffer: GPUBuffer | null = null;
  //bound to the resolve materials while there is no lightmap
  private dummyBuffer: GPUBuffer;
  private boundResolveBuffers: GPUBuffer | null = null;

  private size = 0;
  private readonly tileSize = 8;
  private tileCount = 0;
  private cellSize = 4;
  private readonly referenceHeight = 1080;
  private readonly workGroupSizeX = 8;
  private readonly workGroupSizeY = 8;
  private readonly traceLanesMax = 8;
  private readonly evictFramesMin = 2;
  private resetPending = true;
  //the texels hold the sun visibility too, see setSun. The sun half is cleared in the next update when sunClearPending
  private sunEnabled = false;
  private readonly sunDirection = new Vector3();
  private sunRadius = 0;
  private sunClearPending = false;
  private cameraHistory: { frame: number; position: Vector3 }[] = [];

  //see converging
  private changeVersion = 0;
  private _converging = true;
  private _settling = false;
  private reading = false;
  private cityVersion = -1;
  private frameCount = 0;
  //the cells that won the bid on a texel that still takes samples, of the last readback, the ones of them whose texel
  //has less than convergedPaths paths, and the cells of the screen
  public wantedCells = 0;
  public unconvergedCells = 0;
  public cellCount = 0;
  //called when a param changed, so the host draws again
  public onChange: (() => void) | null = null;

  public enabled!: ParamBool;
  private sizeParam!: ParamSelect;
  private _view: LightmapView = LightmapView.Shaded;
  private texelsPerPixel!: ParamFloat;
  private texelSizeMin!: ParamFloat;
  private levelAngle!: ParamFloat;
  private blurTexelScale!: ParamFloat;
  private searchCount!: ParamInt;
  private sampleCount!: ParamInt;
  private tailSampleCountMax!: ParamInt;
  private referenceCellSize!: ParamFloat;
  private traceSamplesMax!: ParamInt;
  private accumulationMax!: ParamInt;
  public packedTexels!: ParamBool;
  private ancestorLevels!: ParamInt;
  private seedCount!: ParamFloat;
  private ancestorWeight!: ParamFloat;
  private evictFrames!: ParamFloat;
  private evictDistance!: ParamFloat;
  private convergedFraction!: ParamFloat;
  private convergedPaths!: ParamInt;
  private settleFraction!: ParamFloat;
  private stochasticLevels!: ParamBool;
  private levelBlend!: ParamFloat;
  //the lookup at half resolution, see SkyVisibilityHalfRes
  public resolution!: ParamSelect;
  public halfResNearest!: ParamBool;
  public halfResPass!: SkyVisibilityHalfRes;
  private log!: ParamLog;
  //the debug graph of the rays per frame, see drawRayGraph
  private rayGraph!: ParamBool;
  private rayCountGraph: RayCountGraph;
  //the cells of the screen of the last update, for the top of the graph
  private screenCells = 0;

  constructor(renderer: Renderer, preloader: WebGPUPreLoader) {
    this.renderer = renderer;
    this.cityScene = new CityScene(renderer, CITY_BUFFER_BYTES);

    const includes = [hash, r2, cameraRay, cityLightmapSurface, gBufferPacking, cityLightmap, skyVisibilityTexel, skyVisibilityTrace];
    const source = updateSkyVisibilityLightmap;
    // compiled without blocking the frames (claim and trace take about 300 ms each on a cold shader cache), update waits
    // for all of them
    const options = { includes, defines: CityScene.traceDefines, asyncPipeline: true };
    this.claimShader = new ComputeShader(renderer, 'claimSkyVisibilityLightmap', source, { ...options, kernel: 'claim_main' });
    this.selectShader = new ComputeShader(renderer, 'selectSkyVisibilityLightmap', source, { ...options, kernel: 'select_main' });
    this.traceShader = new ComputeShader(renderer, 'traceSkyVisibilityLightmap', source, { ...options, kernel: 'trace_main' });
    this.clearSunShader = new ComputeShader(renderer, 'clearSunSkyVisibilityLightmap', source, { ...options, kernel: 'clear_sun_main' });
    this.updateShaders = [this.claimShader, this.selectShader, this.traceShader, this.clearSunShader];
    for (const s of this.updateShaders) {
      s.setConstant('WORK_GROUP_SIZE_X', this.workGroupSizeX);
      s.setConstant('WORK_GROUP_SIZE_Y', this.workGroupSizeY);
      s.setConstant('TRACE_GROUP_SIZE', this.workGroupSizeX * this.workGroupSizeY);
      s.setConstant('CELL_SIZE', this.cellSize);
      s.setConstant('TILE_SIZE', this.tileSize);
      this.cityScene.bind(s);
    }

    this.blueNoise = new BlueNoiseTexture2D16Bits(renderer, preloader);
    for (const s of this.updateShaders) s.setTexture('blueNoiseTexture', this.blueNoise);

    this.tileCountBuffer = renderer.createBuffer(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.traceArgsBuffer = renderer.createBuffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    this.traceIndirectBuffer = renderer.createBuffer(16, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.dummyBuffer = renderer.createBuffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    for (const s of this.updateShaders) {
      s.setBuffer('tileCount', this.tileCountBuffer);
      s.setBuffer('traceArgs', this.traceArgsBuffer);
    }
    this.rayCountGraph = new RayCountGraph(renderer, this.traceArgsBuffer);
  }

  public init(paramGroup: ParamGroup): void {
    const changed = () => this.paramChanged();
    const pg = paramGroup.addGroup('virtual lightmap');
    //the sky light of the lightmap instead of the sky term
    this.enabled = pg.addParamBool('enabled', true, { callback: changed });
    pg.addButton('reset', { callback: () => this.reset() });
    //the rays asked for per frame as a graph in the bottom left, see drawRayGraph
    this.rayGraph = pg.addParamBool('rayGraph', false);
    this.sizeParam = pg.addParamSelect('lightmapSize', LIGHTMAP_SIZE_OPTIONS.indexOf(2048), LIGHTMAP_SIZE_OPTIONS.map(String), { callback: changed });
    //a pixel reads 1 of the 2 levels around its level, picked by noise, instead of blending both, when the TAA is on
    //to average it, see getLightmapLightingStochastic in skyVisibilityResolve.wgsl
    this.stochasticLevels = pg.addParamBool('stochasticLevels', true, { callback: changed });
    //the part of a level (the middle) in which a pixel blends it with the next one. Outside it a pixel reads 1 level
    //instead of 2, see getLevelBlend in skyVisibilityResolve.wgsl. 1: blended over the whole level
    this.levelBlend = pg.addParamFloat('levelBlend', .5, .05, 1, { callback: changed });
    //the lookup per pixel or for 1 pixel per 2 x 2, see LightmapResolution. Set by the quality
    this.resolution = pg.addParamSelect('resolution', LightmapResolution.Half, LIGHTMAP_RESOLUTIONS, { callback: changed });
    //the half resolution lookup reads the texel of the point instead of the 4 around it: about 2x faster, blocky (a
    //texel is a few pixels), see NEAREST_TEXEL in skyVisibilityResolve.wgsl
    this.halfResNearest = pg.addParamBool('halfResNearest', false, { callback: changed });
    this.log = pg.addLog('tiles: 0');
    this.halfResPass = new SkyVisibilityHalfRes(this.renderer, SkyVisibilityLightmap.resolveIncludes);
    for (const m of this.halfResPass.materials) this.addResolveMaterial(m);

    const pgTexels = pg.addGroup('texels');
    //lightmap texels per reference pixel along a side (a pixel of a 1920 x 1080 screen, see getPixelScale)
    this.texelsPerPixel = pgTexels.addParamFloat('texelsPerPixel', .25, .05, 1, { callback: changed });
    //texels per sigma of the tilt shift blur
    this.blurTexelScale = pgTexels.addParamFloat('blurTexelScale', 1, 0, 4, { callback: changed });
    //the smallest texels in m
    this.texelSizeMin = pgTexels.addParamFloat('texelSizeMin', .2, .01, 1, { callback: changed });
    this.levelAngle = pgTexels.addParamFloat('levelAngle', .5, 0, 1, { callback: changed });
    this.searchCount = pgTexels.addParamInt('searchCount', 8, 1, 32, { callback: changed });
    pgTexels.collapsed = true;

    const pgUpdate = pg.addGroup('update');
    this.sampleCount = pgUpdate.addParamInt('sampleCount', 8, 8, 64, { step: 8, callback: changed });
    //the paths a cell takes at most in the tail of the convergence, where the few cells that still trace share
    //traceSamplesMax instead of taking sampleCount each, see getSampleCount
    this.tailSampleCountMax = pgUpdate.addParamInt('tailSampleCountMax', 128, 8, 256, { step: 8, callback: changed });
    //the update cell in pixels along a side, on a screen of 1920 x 1080 pixels (scaled by the pixel count, so every
    //aspect gets about the same number of cells). Set by the quality
    this.referenceCellSize = pgUpdate.addParamFloat('referenceCellSize', 4, 1, 16, { callback: changed });
    //the most paths per frame, times 1000, see VirtualLightmap
    this.traceSamplesMax = pgUpdate.addParamInt('traceSamplesMax', 512, 64, 8192, { step: 64, callback: changed });
    this.accumulationMax = pgUpdate.addParamInt('accumulationMax', 1024, 8, 2048, { step: 8, callback: changed });
    //a texel as 1 word of ray counts (sky and sun hits and rays: 9, 9, 7, 7 bits) instead of 2 x f16 per channel: 1 load
    //per texel for both channels, 1 compare exchange per texel for a batch. accumulationMax is PACKED_ACCUMULATION_MAX at most,
    //and only the paths of a wall are folded up (see getPathRay). Starts the lightmap over
    this.packedTexels = pgUpdate.addParamBool('packedTexels', true, { callback: () => this.setPackedTexels() });
    //the update shaders and the half resolution materials were made before the param, and its callback only runs on a
    //change. The resolve materials added from here on get it in addResolveMaterial
    for (const s of [...this.updateShaders, ...this.resolveMaterials]) s.setConstant('PACKED_TEXELS', this.packed);
    this.ancestorLevels = pgUpdate.addParamInt('ancestorLevels', 2, 0, 4, { callback: changed });
    this.seedCount = pgUpdate.addParamFloat('seedCount', 32, 0, 512, { callback: changed });
    this.ancestorWeight = pgUpdate.addParamFloat('ancestorWeight', 64, 0, 512, { callback: changed });
    this.evictFrames = pgUpdate.addParamFloat('evictFrames', 30, 1, 1000);
    //in m: the camera of the map is 90 m to 2 km away
    this.evictDistance = pgUpdate.addParamFloat('evictDistance', 50, 1, 1000);
    //the part of the cells that still bid on a texel that takes samples, under which the lightmap counts as
    //converged and the host stops drawing
    this.convergedFraction = pgUpdate.addParamFloat('convergedFraction', .005, 0, .05, { callback: changed });
    //the paths from which a texel no longer counts for convergedFraction. It still takes samples up to accumulationMax
    //while the host draws. The ones left below accumulationMax at the end are mostly the slivers some of the jitter
    //offsets of the TAA show: a cell rarely picks them, and they took hundreds of frames
    this.convergedPaths = pgUpdate.addParamInt('convergedPaths', 256, 0, 2048, { step: 16, callback: changed });
    //the part of the cells under which the lightmap is settling (see settling): the host may average its last frames
    this.settleFraction = pgUpdate.addParamFloat('settleFraction', .02, 0, .2, { callback: changed });
    pgUpdate.collapsed = true;
    pg.collapsed = true;
  }

  /**
   * The settings of MapWorld's quality dropdown: the lightmap size (one of LIGHTMAP_SIZE_OPTIONS), its sampling (the
   * paths per frame at most, and per cell while the view changes, and the size of its update cells), the resolution of
   * the lookup, and the slots of the tile table a tile is looked for in.
   */
  public setQuality(lightmapSize: number, texelsPerPixel: number, traceSamplesMax: number, sampleCount: number, referenceCellSize: number, resolution: LightmapResolution, searchCount: number): void {
    const sizeIndex = LIGHTMAP_SIZE_OPTIONS.indexOf(lightmapSize);
    if (sizeIndex < 0) throw new Error(`[SkyVisibilityLightmap] no lightmap size ${lightmapSize}`);
    this.sizeParam.setValue(sizeIndex);
    this.texelsPerPixel.setValue(texelsPerPixel);
    this.traceSamplesMax.setValue(traceSamplesMax);
    this.sampleCount.setValue(sampleCount);
    this.referenceCellSize.setValue(referenceCellSize);
    this.resolution.setValue(resolution);
    this.searchCount.setValue(searchCount);
  }

  //the shading samples the half resolution lookup bilinearly, see LightmapResolution.HalfBilinear
  private get halfResBilinear(): boolean {
    return this.resolution.value === LightmapResolution.HalfBilinear;
  }

  /**
   * The shading reads the lightmap from the samples of halfResPass, which the host draws before it (see
   * SkyVisibilityHalfRes). Not in the lightmap view, which doesn't read it.
   */
  public get halfResActive(): boolean {
    return this.enabled.value && this.lightingBuffer != null && this.resolution.value !== LightmapResolution.Full && this._view !== LightmapView.Lightmap;
  }

  public get active(): boolean {
    return this.enabled.value;
  }

  /** The debug view of the shading, set by the host (MapWorld's drawMode), see LightmapView. */
  public get view(): LightmapView {
    return this._view;
  }

  public set view(value: LightmapView) {
    if (value === this._view) return;
    this._view = value;
    this.paramChanged();
  }

  /** The lightmap view: the lightmap itself instead of the scene. It doesn't move with the camera, so the host skips the TAA and the tilt-shift. */
  public get showsLightmap(): boolean {
    return this.enabled.value && this._view === LightmapView.Lightmap;
  }

  /**
   * Whether the visibility may still change without a camera move or new geometry: the last update had texels that
   * took samples, or something changed since the last readback that said otherwise. The host keeps drawing while so.
   */
  public get converging(): boolean {
    return this.enabled.value && this._converging;
  }

  /**
   * Converging, but the last readback (of the view as it is) had less than settleFraction of the cells on a texel with
   * few paths: what still changes is small and scattered. The TAA of MapWorld then already averages the frames, its
   * average of the last ones is an estimate of the visibility as good as the last frame's.
   */
  public get settling(): boolean {
    return this.converging && this._settling;
  }

  //the visibility in view can change: a readback of before this says nothing
  private markChanged(): void {
    this.changeVersion++;
    this._converging = true;
    this._settling = false;
  }

  //and the host has to draw again. Not from update, which the host calls because it draws
  private paramChanged(): void {
    this.markChanged();
    this.onChange?.();
  }

  /**
   * The sun visibility in the texels too, for the sun shadows of the deferred shading (its getLightmapSunShadow), each
   * frame before converging is read. direction: toward the sun, radius: the angular radius of its disc. A change of
   * enabled makes the buffers again (the lightmap starts over), a move of the sun clears the sun texels only: the
   * lightmap converges again either way.
   */
  public setSun(enabled: boolean, direction: Vector3, radius: number): void {
    if (enabled !== this.sunEnabled) {
      this.sunEnabled = enabled;
      //the half resolution pass looks up the sun with the sky
      for (const s of [...this.updateShaders, ...this.halfResPass.materials, ...this.sunMaterials]) s.setConstant('LIGHTMAP_SUN', enabled);
      //made again at the size with or without the sun half in the next update
      this.size = 0;
      this.markChanged();
    }
    if (!enabled) return;
    if (!direction.equals(this.sunDirection) || radius !== this.sunRadius) {
      this.sunDirection.copy(direction);
      this.halfResPass.sunDirection.copy(direction);
      this.sunRadius = radius;
      this.sunClearPending = true;
      this.markChanged();
    }
  }

  public reset(): void {
    this.resetPending = true;
    this.paramChanged();
  }

  /**
   * A material (with resolveIncludes) that reads the lightmap, see deferredShading.wgsl. views: it has the debug views,
   * the overrides LIGHTMAP_VIEW (see view) and HALF_RES_BILINEAR (LightmapResolution.HalfBilinear): a change of them makes its pipeline
   * again. sun: it has the override LIGHTMAP_SUN, whether the texels hold the sun visibility too (see setSun)
   */
  public addResolveMaterial(material: ShaderBase, views = false, sun = false): void {
    this.resolveMaterials.push(material);
    if (sun) {
      this.sunMaterials.push(material);
      material.setConstant('LIGHTMAP_SUN', this.sunEnabled);
    }
    if (views) {
      this.viewMaterials.add(material);
      material.setConstant('LIGHTMAP_VIEW', this._view);
    }
    material.setConstant('TILE_SIZE', this.tileSize);
    material.setConstant('PACKED_TEXELS', this.packed);
    if (this.size > 0) {
      material.setConstant('LIGHTMAP_SIZE', this.size);
      material.setConstant('TILE_COUNT', this.tileCount);
    }
    this.bindResolveBuffers(material);
  }

  private bindResolveBuffers(material: ShaderBase): void {
    const real = this.enabled?.value && this.tileBuffer != null;
    material.setBuffer('tileBuffer', real ? this.tileBuffer! : this.dummyBuffer);
    material.setBuffer('lightingBuffer', real ? this.lightingBuffer! : this.dummyBuffer);
  }

  //the bytes of lightingBuffer at a size: 1 word per texel, twice that with the sun unpacked (see setSun)
  private getLightingBytes(size: number): number {
    return size * size * 4 * (this.sunEnabled && !this.packed ? 2 : 1);
  }

  //the size, or the largest smaller option whose lightingBuffer (the largest buffer) fits in a storage binding of the
  //device: 4096 is the most at the default limit with the sun unpacked
  private getFittingSize(size: number): number {
    const limits = this.renderer.device.limits;
    const max = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    if (this.getLightingBytes(size) <= max) return size;
    const fitting = LIGHTMAP_SIZE_OPTIONS.filter(s => s < size && this.getLightingBytes(s) <= max);
    const fit = fitting.length > 0 ? fitting[fitting.length - 1] : LIGHTMAP_SIZE_OPTIONS[0];
    if (fit !== this.size) console.warn(`[SkyVisibilityLightmap] lightmap ${size} needs ${this.getLightingBytes(size)} bytes, over the storage binding limit ${max}: ${fit} instead`);
    return fit;
  }

  //the buffers of the texels and the tiles at the size of the param. Replaces the ones of a previous size
  private createLightmapBuffers(size: number): void {
    this.tileBuffer?.destroy();
    this.tileTimeBuffer?.destroy();
    this.lightingBuffer?.destroy();
    this.claimBuffer?.destroy();
    this.size = size;
    this.tileCount = Math.pow(size / this.tileSize, 2);
    this.tileBuffer = this.renderer.createBuffer(this.tileCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.tileTimeBuffer = this.renderer.createBuffer(this.tileCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    //1 word per texel, see addTexelSample in updateSkyVisibilityLightmap.wgsl. Twice that with the sun, see setSun
    this.lightingBuffer = this.renderer.createBuffer(this.getLightingBytes(size), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.claimBuffer = this.renderer.createBuffer(size * size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    for (const s of [...this.updateShaders, ...this.resolveMaterials]) {
      s.setConstant('LIGHTMAP_SIZE', size);
      s.setConstant('TILE_COUNT', this.tileCount);
    }
    for (const s of this.updateShaders) {
      s.setBuffer('tileBuffer', this.tileBuffer);
      s.setBuffer('tileTimeBuffer', this.tileTimeBuffer);
      s.setBuffer('lightingBuffer', this.lightingBuffer);
      s.setBuffer('claimBuffer', this.claimBuffer);
    }
    this.boundResolveBuffers = null;
    this.resetPending = true;
    LogGL.log(`[SkyVisibilityLightmap] lightmap ${size} x ${size}, ${this.tileCount} tiles`);
  }

  private setAll(name: string, value: number): void {
    for (const s of this.updateShaders) s.setFloat(name, value);
  }

  /** Before the shading: the uniforms the resolve materials need to read the lightmap, the same as the update's. */
  //temporal: a TAA averages the frames, so a pixel can read 1 level picked by noise, see stochasticLevels
  public setResolveUniforms(camera: MapCamera, blur: LightmapBlur, metersToNorm: number, temporal: boolean): void {
    const real = this.enabled.value && this.lightingBuffer != null ? this.lightingBuffer : null;
    const rebind = real !== this.boundResolveBuffers;
    this.boundResolveBuffers = real;
    for (const m of this.resolveMaterials) {
      if (rebind) this.bindResolveBuffers(m);
      const stochasticLevel = temporal && this.stochasticLevels.value;
      if (this.viewMaterials.has(m)) {
        // the deferred shading has these as constants, so each setting compiles without the code of the others
        m.setConstant('LIGHTMAP_ENABLED', real != null);
        m.setConstant('LIGHTMAP_HALF_RES', this.halfResActive);
        m.setConstant('STOCHASTIC_LEVEL', stochasticLevel);
        m.setConstant('LIGHTMAP_VIEW', this._view);
        m.setConstant('HALF_RES_BILINEAR', this.halfResBilinear);
      } else {
        m.setBool('lightmapEnabled', real != null);
        m.setBool('lightmapHalfRes', this.halfResActive);
        m.setBool('stochasticLevel', stochasticLevel);
      }
      //the size of the gBuffer the update used, not renderer.height: that is of the last pass (a shadow cascade)
      const width = this.gBufferWidth || this.renderer.canvas.width;
      const height = this.gBufferHeight || this.renderer.canvas.height;
      m.setFloat('pixelSize', this.getPixelSize(camera, width, height));
      m.setFloat('pixelScale', this.getPixelScale(width, height));
      m.setFloat('texelsPerPixel', this.texelsPerPixel.value);
      m.setFloat('texelSizeMin', this.texelSizeMin.value * metersToNorm);
      m.setFloat('levelAngle', this.levelAngle.value);
      m.setFloat('blurSlope', blur.slope);
      m.setFloat('blurMax', blur.max);
      m.setFloat('blurTexelScale', this.blurTexelScale.value);
      m.setFloat('searchCount', this.searchCount.value);
      m.setFloat('accumulationMax', this.getAccumulationMax());
      m.setFloat('ancestorWeight', this.ancestorWeight.value);
      m.setFloat('levelBlend', this.levelBlend.value);
      m.setFloat('atlasStep', MapGBuffer.atlasStep(metersToNorm));
    }
    this.halfResPass.bilinear = this.halfResBilinear;
    this.halfResPass.nearest = this.halfResNearest.value;
  }

  /** Independent of the light (sun, sky): the shading multiplies the visibility by the sky color. */
  public update(frame: LightmapFrame): void {
    if (!this.enabled.value) return;
    const { camera, metersToNorm } = frame;
    if (frame.changed) this.markChanged();

    // the buffers right away: the resolve compiles its LIGHTMAP_ENABLED variant once, not again when they come
    const size = this.getFittingSize(LIGHTMAP_SIZE_OPTIONS[this.sizeParam.value]);
    if (size !== this.size) this.createLightmapBuffers(size);
    this.cityScene.enable();
    const gBuffer = frame.gBuffer.target;
    //by the area of the screen, not its height: a portrait screen would get aspect² fewer cells (3.2x at 9:16)
    const cellSize = Math.max(1, Math.round(this.referenceCellSize.value * this.getPixelScale(gBuffer.width, gBuffer.height)));
    if (cellSize !== this.cellSize) {
      this.cellSize = cellSize;
      for (const s of this.updateShaders) s.setConstant('CELL_SIZE', cellSize);
    }
    // nothing more until the pipelines of all the shaders are compiled (see ComputeShader.asyncPipeline), before
    // anything else changes: the shaders work together, a claim without its trace would leave the claimed texels. After
    // the constants, a constant that changes after this check would make its pipeline sync
    let ready = true;
    for (const s of this.updateShaders) ready = s.ready && ready;
    if (!ready) return;
    if (this.resetPending) {
      this.renderer.clearBuffer(this.tileBuffer!);
      this.renderer.clearBuffer(this.tileTimeBuffer!);
      this.renderer.clearBuffer(this.lightingBuffer!);
      this.renderer.clearBuffer(this.claimBuffer!);
      this.renderer.clearBuffer(this.tileCountBuffer);
      this.renderer.clearBuffer(this.traceArgsBuffer);
      this.cameraHistory.length = 0;
      this.cityScene.clearChanges();
      this.resetPending = false;
      this.sunClearPending = false;
    } else if (this.sunClearPending) {
      //the sun moved: the sun half of the texels only, or the sun fields of the packed texels
      if (this.packed) {
        this.renderer.dispatchCompute(this.clearSunShader, Math.ceil(size / this.workGroupSizeX), Math.ceil(size / this.workGroupSizeY), 1);
      } else {
        this.renderer.clearBuffer(this.lightingBuffer!, size * size * 4, size * size * 4);
      }
      this.sunClearPending = false;
    }

    //the tiles that came or went get this frame: the texels around them that were sampled before are reset
    const frameNumber = this.renderer.time.frameNumber;
    this.cityScene.update(frameNumber);
    if (this.cityScene.version !== this.cityVersion) {
      this.cityVersion = this.cityScene.version;
      this.markChanged();
    }

    this.gBufferWidth = gBuffer.width;
    this.gBufferHeight = gBuffer.height;

    const countX = Math.ceil(gBuffer.width / (this.workGroupSizeX * this.cellSize));
    const countY = Math.ceil(gBuffer.height / (this.workGroupSizeY * this.cellSize));
    const cellCount = countX * this.workGroupSizeX * countY * this.workGroupSizeY;
    this.screenCells = Math.ceil(gBuffer.width / this.cellSize) * Math.ceil(gBuffer.height / this.cellSize);
    if (!this.traceCellsBuffer || this.traceCellsBuffer.size < cellCount * 8) {
      this.traceCellsBuffer?.destroy();
      this.traceCellsBuffer = this.renderer.createBuffer(cellCount * 8, GPUBufferUsage.STORAGE);
      for (const s of this.updateShaders) s.setBuffer('traceCells', this.traceCellsBuffer);
    }

    const sampleCount = this.getSampleCount(frame.changed);
    const traceLanes = this.getTraceLanes(sampleCount);
    const traceCellsMax = Math.max(1, Math.floor(this.traceSamplesMax.value * 1000 / sampleCount));
    for (const s of this.updateShaders) {
      s.setVector2('resolution', gBuffer.size);
      s.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
      s.setMatrix('inverseView', camera.viewMatrix.inverse);
      s.setMatrix('viewProjection', camera.viewProjection);
      s.setVector3('cameraPosition', camera.worldPosition);
      s.setTexture('gBufferTexture', frame.gBuffer.view);
      s.setTexture('gBufferAtlasTexture', frame.gBuffer.atlasView);
      s.setTexture('gBufferDepth', frame.gBuffer.depthView);
      s.setVector2('depthParams', MapGBuffer.depthParams(camera));
      s.setFloat('atlasStep', MapGBuffer.atlasStep(metersToNorm));
    }
    this.setAll('frame', frameNumber);
    this.setAll('pixelSize', this.getPixelSize(camera, gBuffer.width, gBuffer.height));
    this.setAll('pixelScale', this.getPixelScale(gBuffer.width, gBuffer.height));
    this.setAll('texelsPerPixel', this.texelsPerPixel.value);
    this.setAll('texelSizeMin', this.texelSizeMin.value * metersToNorm);
    this.setAll('levelAngle', this.levelAngle.value);
    this.setAll('blurSlope', frame.blur.slope);
    this.setAll('blurMax', frame.blur.max);
    this.setAll('blurTexelScale', this.blurTexelScale.value);
    this.setAll('sampleCount', sampleCount);
    this.setAll('accumulationMax', this.getAccumulationMax());
    this.setAll('ancestorLevels', this.ancestorLevels.value);
    this.setAll('seedCount', this.seedCount.value);
    this.setAll('evictFrames', this.getEvictFrames(camera, frameNumber, metersToNorm));
    this.setAll('searchCount', this.searchCount.value);
    this.setAll('traceLanes', traceLanes);
    this.setAll('traceCellsMax', traceCellsMax);
    this.setAll('convergedPaths', this.convergedPaths.value);
    for (const s of this.updateShaders) s.setVector3('sunDirection', this.sunDirection);
    this.setAll('sunRadius', this.sunRadius);
    //the rays of a cell that go to the sun instead of the sky while both take samples, see trace_main: 1 per
    //SUN_PATH_RATIO paths, also in the tail of the convergence where a cell takes more paths. More in a penumbra
    const sunSampleCount = Math.max(1, Math.round(sampleCount / SUN_PATH_RATIO));
    this.setAll('sunSampleCount', sunSampleCount);
    this.setAll('sunSampleCountPenumbra', Math.max(sunSampleCount, Math.round(sampleCount / SUN_PENUMBRA_PATH_RATIO)));

    const wantedIndex = 4 + (frameNumber & 1);
    this.renderer.clearBuffer(this.traceArgsBuffer, 0, 16);
    this.renderer.clearBuffer(this.traceArgsBuffer, 24, 4);
    this.renderer.clearBuffer(this.traceArgsBuffer, wantedIndex * 4, 4);
    this.renderer.dispatchCompute(this.claimShader, countX, countY, 1);
    this.renderer.dispatchCompute(this.selectShader, countX, countY, 1);
    if (this.rayGraph.value) this.rayCountGraph.record(wantedIndex, sampleCount, traceCellsMax);
    this.renderer.copyBufferToBuffer(this.traceArgsBuffer, this.traceIndirectBuffer, 0, 0, 16);
    this.renderer.dispatchComputeIndirect(this.traceShader, this.traceIndirectBuffer);

    this.readBack(wantedIndex, traceCellsMax, cellCount, sampleCount);
    this.frameCount++;
  }

  /**
   * The debug graph of the rays per frame (the rayGraph param), over the screen after everything else of the frame: a
   * bar per frame for the last 512, the rays the cells that won a bid ask for (before the cap, green) and the ones
   * traced (after it, dark green). From 0 to all the
   * cells of the screen times sampleCount (the base paths per cell, the tail of the convergence can ask more per cell),
   * and a red line at traceSamplesMax when it is lower.
   */
  public drawRayGraph(): void {
    if (!this.rayGraph.value || !this.enabled.value || this.screenCells === 0) return;
    this.rayCountGraph.draw(this.screenCells * this.sampleCount.value, this.traceSamplesMax.value * 1000);
  }

  //whether the texels in view still take samples
  private readBack(wantedIndex: number, traceCellsMax: number, cellCount: number, sampleCount: number): void {
    if (this.reading) return;
    this.reading = true;
    const version = this.changeVersion;
    const logNow = this.frameCount % 30 === 0;
    const reads: Promise<Uint32Array>[] = [this.renderer.readBuffer(this.traceArgsBuffer, Uint32Array)];
    if (logNow) reads.push(this.renderer.readBuffer(this.tileCountBuffer, Uint32Array));
    Promise.all(reads).then(([traceArgs, count]) => {
      this.reading = false;
      const wanted = traceArgs[wantedIndex];
      this.wantedCells = wanted;
      this.unconvergedCells = traceArgs[6];
      this.cellCount = cellCount;
      if (count) {
        const city = this.cityScene;
        const traced = Math.min(traceArgs[3], traceCellsMax);
        //the paths of the frame against the budget: it caps them when the cells that won a bid times sampleCount reach it
        const paths = `${sampleCount} paths per cell, ${(traced * sampleCount / 1000).toFixed(0)}k of ${this.traceSamplesMax.value}k max`;
        this.log.setValue(`tiles: ${count[0]} / ${this.tileCount} (${(count[0] / this.tileCount * 100).toFixed(1)}%)\ntracing ${traced} of ${wanted} / ${cellCount} cells\n${paths}\ncity: ${city.tileCount} tiles, ${(city.usedVec4s * 16 / 1048576).toFixed(1)} MB`);
      }
      if (version !== this.changeVersion) return;
      const converging = traceArgs[6] > this.convergedFraction.value * cellCount;
      this._settling = traceArgs[6] <= this.settleFraction.value * cellCount;
      if (converging !== this._converging && LogGL.ENABLED) LogGL.log(`[SkyVisibilityLightmap] ${converging ? 'converging' : 'converged'}`);
      this._converging = converging;
    }).catch(() => {
      this.reading = false;
    });
  }

  /*
  the paths a cell traces this frame. sampleCount, but in the tail of the convergence, when the last readback had few
  cells that won a bid (wantedCells), each of them traces more: traceSamplesMax in all, at most tailSampleCountMax each,
  and at most what a texel holds (accumulationMax). The texels that are left are the ones a cell rarely picks
  (small, or shared by few cells), which took several hundred frames at 16 paths. Not while the view changes: the
  readback is of a few frames before. The budget is the cap of the trace, so the cells it asks for fit under it: a
  bigger one only made fewer of them trace, with more paths each
   */
  private getSampleCount(changed: boolean): number {
    const base = this.sampleCount.value;
    if (changed || this.wantedCells === 0) return base;
    const tail = Math.floor(this.traceSamplesMax.value * 1000 / this.wantedCells / 8) * 8;
    const tailMax = Math.floor(Math.min(this.tailSampleCountMax.value, this.getAccumulationMax()) / 8) * 8;
    return Math.max(base, Math.min(tail, tailMax));
  }

  //the frames a tile has to be unused before another can take its place
  private getEvictFrames(camera: MapCamera, frame: number, metersToNorm: number): number {
    const position = camera.worldPosition.clone();
    const history = this.cameraHistory;
    const framesMax = Math.ceil(this.evictFrames.value);
    while (history.length > 0 && frame - history[0].frame > framesMax) history.shift();
    let evictFrames = framesMax;
    const distance = this.evictDistance.value * metersToNorm;
    for (let i = history.length - 1; i >= 0; i--) {
      if (Vector3.distance(history[i].position, position) > distance) {
        evictFrames = frame - history[i].frame;
        break;
      }
    }
    history.push({ frame, position });
    return Math.min(Math.max(evictFrames, this.evictFramesMin), framesMax);
  }

  //the threads of trace_main per cell, see VirtualLightmap.getTraceLanes
  private getTraceLanes(sampleCount: number): number {
    const occupancy = (lanes: number) => sampleCount / (lanes * Math.ceil(sampleCount / lanes));
    const options: number[] = [];
    for (let lanes = this.traceLanesMax; lanes >= 1; lanes /= 2) options.push(lanes);
    const best = Math.max(...options.map(occupancy));
    return options.find(lanes => occupancy(lanes) >= best - 1 / 8)!;
  }

  private get packed(): boolean {
    return this.packedTexels?.value ?? false;
  }

  //the most paths of a texel: with packedTexels at most PACKED_ACCUMULATION_MAX, and a multiple of PACKED_SUN_SCALE (the
  //sun's cap is a quarter of it)
  private getAccumulationMax(): number {
    const value = this.accumulationMax.value;
    if (!this.packed) return value;
    return Math.floor(Math.min(value, PACKED_ACCUMULATION_MAX) / PACKED_SUN_SCALE) * PACKED_SUN_SCALE;
  }

  //a change of packedTexels: the texels of the other layout mean nothing, so the buffers are made again (the size of the
  //sun half changes) and the lightmap starts over
  private setPackedTexels(): void {
    for (const s of [...this.updateShaders, ...this.resolveMaterials]) s.setConstant('PACKED_TEXELS', this.packed);
    this.size = 0;
    this.paramChanged();
  }

  /*
  the screen pixels per reference pixel along a side: a reference pixel is a pixel of a screen with the pixel count of
  1920 x 1080 and the aspect of this one. By the area, not the height, so texels (texelsPerPixel) and cells
  (referenceCellSize) divide every aspect in about the same number: by the height, the FOV being vertical, a 9:16
  screen got aspect² = 3.2x fewer texels, so fewer cells that win a bid and 3.2x fewer rays than 16:9
  */
  private getPixelScale(width: number, height: number): number {
    return Math.sqrt(width * height / (this.referenceHeight * this.referenceHeight * 16 / 9));
  }

  //the world size of a reference pixel at distance 1 (see getPixelScale)
  private getPixelSize(camera: MapCamera, width: number, height: number): number {
    return Math.tan(camera.projection.FOV * .5) * 2 / height * this.getPixelScale(width, height);
  }

  public destruct(): void {
    this.tileBuffer?.destroy();
    this.tileTimeBuffer?.destroy();
    this.lightingBuffer?.destroy();
    this.claimBuffer?.destroy();
    this.traceCellsBuffer?.destroy();
    this.tileCountBuffer.destroy();
    this.traceArgsBuffer.destroy();
    this.traceIndirectBuffer.destroy();
    this.dummyBuffer.destroy();
    this.halfResPass?.destruct();
    this.rayCountGraph.destruct();
    this.cityScene.destruct();
  }
}
