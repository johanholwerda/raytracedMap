import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import cameraRay from '../cameraRay.wgsl?raw';
import toneMap from 'webGPU-framework/shaders/toneMap.wgsl?raw';
import r2 from 'webGPU-framework/shaders/r2.wgsl?raw';
import Texture2D from 'webGPU-framework/renderer/texture/Texture2D';
import SkyVisibilityLightmap from '../skyVisibilityLightmap/SkyVisibilityLightmap';
import SkyVisibilityHalfRes from '../skyVisibilityLightmap/SkyVisibilityHalfRes';
import MapGBuffer from './MapGBuffer';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import shader from './deferredShading.wgsl?raw';
import skyVisibilityTrace from '../skyVisibilityLightmap/skyVisibilityTrace.wgsl?raw';
import cityTraceDistance from './cityTraceDistance.wgsl?raw';
import CityScene from '../city/CityScene';
import WaterRippleTexture from './waterRipples/WaterRippleTexture';
import { ShadowMode } from '../MapWorldSettings';
import ReprojectedSSAO, { SSAO_FADE_COUNT } from '../ssao/ReprojectedSSAO';
import WaterReflectionHalfRes from './WaterReflectionHalfRes';
import waterReflection from './waterReflection.wgsl?raw';
import waterScreenReflection from './waterScreenReflection.wgsl?raw';
import ScreenReflections from './ScreenReflections';
import terrainSun from '../terrainSun.wgsl?raw';
import terrainReflection from './terrainReflection.wgsl?raw';
import terrainHorizonLookup from './terrainHorizonLookup.wgsl?raw';
import WaterTerrainHorizon from './WaterTerrainHorizon';

/**
 * Shades the gBuffer of MapWorld into the scene color, once per pixel: see deferredShading.wgsl. The materials take
 * the shading params (sky, sun, fog, shadows) by name and read the sky visibility lightmap
 * (SkyVisibilityLightmap.addResolveMaterial).
 * The water reflections trace the city of the lightmap (CityScene), mirrored on the ripples of WaterRippleTexture, see
 * setWaterReflections: a ray per water pixel here, or the rays of WaterReflectionHalfRes. Over the screen first, with the
 * colors of what they hit there, see ScreenReflections and setScreenReflections. The sun shadows are the VSM cascades of SunShadows, or rays traced against the same city (per pixel
 * or the texels of the lightmap), see setSunShadows.
 */
export default class DeferredShading {
  private renderer: Renderer;
  public readonly material: VertexFragmentShader;
  private frame = 0;
  private readonly depthParams = new Vector2();
  // bound as the city until CityScene has its buffer: no tiles, so no ray reads it
  private readonly dummyCityBuffer: GPUBuffer;
  // bound as ssaoTexture without an SSAO (MapWorld only constructs it when a quality turns it on): SSAO_SKY is off then
  private readonly dummySSAOTexture: Texture2D;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'mapDeferredShading', shader, {
      includes: [toneMap, r2, cameraRay, ...SkyVisibilityLightmap.resolveIncludes, waterReflection, waterScreenReflection, skyVisibilityTrace, cityTraceDistance, terrainSun,
        terrainReflection, terrainHorizonLookup],
      defines: CityScene.traceDefines,
      // about 600 ms to compile on a cold shader cache: the first frames are empty instead of stalled
      asyncPipeline: true,
    });
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    this.dummyCityBuffer = renderer.createBuffer(16, GPUBufferUsage.STORAGE, false, 'dummyCityBuffer');
    this.dummySSAOTexture = new Texture2D(renderer, 1, 1);
    for (const m of this.materials) {
      m.setBuffer('cityBuffer', this.dummyCityBuffer);
      m.setTexture('ssaoTexture', this.dummySSAOTexture);
    }
  }

  public get materials(): VertexFragmentShader[] {
    return [this.material];
  }

  /** The city the water reflections trace: CityScene sets its buffer and root on the materials. */
  public bindCity(cityScene: CityScene): void {
    for (const m of this.materials) cityScene.bind(m, false);
  }

  /** The ripples the water reflections are mirrored on, see setWaterReflections. */
  public bindWaterRipples(ripples: WaterRippleTexture): void {
    for (const m of this.materials) {
      m.setTexture('rippleTexture', ripples.texture);
      m.setSampler('rippleSampler', ripples.sampler);
    }
  }

  /** The 2d blue noise of the points on the sun of the soft ray traced shadows, see setRayTracedShadows. */
  public bindBlueNoise(texture: Texture2D): void {
    for (const m of this.materials) m.setTexture('blueNoiseTexture', texture);
  }

  /**
   * Where the sun shadows come from, the override SUN_SHADOWS of deferredShading.wgsl (the values of ShadowMode). The
   * modes with rays (lightmapRaytrace while the lightmap has no buffers, raytrace) need the city bound (bindCity) and its
   * buffer enabled. soft: the rays sample the disc of the sun (SOFT_SHADOWS), which needs the TAA to average them, else
   * they go to its center (hard shadows)
   */
  public setSunShadows(mode: ShadowMode, soft: boolean): void {
    for (const m of this.materials) {
      m.setConstant('SUN_SHADOWS', mode);
      m.setConstant('SOFT_SHADOWS', soft);
    }
  }

  /**
   * The SSAO on the sky term only (the override SSAO_SKY of deferredShading.wgsl), see ReprojectedSSAO.skyOnly. Every
   * frame before draw, after the SSAO's update: its result ping-pongs, and is bound also when off (the binding stays).
   * on false: the SSAO is composited over the image after the TAA, or is off. ssao null: there is none, a 1x1 texture
   * is bound instead.
   */
  public setSSAO(ssao: ReprojectedSSAO | null, on: boolean): void {
    for (const m of this.materials) {
      m.setConstant('SSAO_SKY', on && ssao != null);
      m.setTexture('ssaoTexture', ssao ? ssao.result : this.dummySSAOTexture);
      if (ssao) m.setFloat('ssaoAmount', ssao.amount.value);
      m.setFloat('ssaoFadeCount', SSAO_FADE_COUNT);
    }
  }

  /** The debug view of the sun shadows, the override SHADOW_VIEW of deferredShading.wgsl: 0 off, 1 the shadow lookup, 2 the cascades. */
  public set shadowView(value: number) {
    for (const m of this.materials) m.setConstant('SHADOW_VIEW', value);
  }

  /** ACES fitted tone mapping of the linear color, the override TONE_MAP of deferredShading.wgsl. */
  public set toneMap(value: boolean) {
    for (const m of this.materials) m.setConstant('TONE_MAP', value);
  }

  /** The horizons the ground in the water reflections is read from, see setTerrainReflection. */
  public bindTerrainHorizon(horizon: WaterTerrainHorizon): void {
    for (const m of this.materials) m.setTexture('terrainHorizon', horizon.target);
  }

  /**
   * The ground in the water reflections that trace the buildings (reflectBuildings of setWaterReflections): their rays
   * are compared with the horizon of the ground around the pixel (getTerrainReflection of terrainHorizonLookup.wgsl),
   * which WaterTerrainHorizon has to draw before the shading then: where a ray hits the ground, the water mirrors its
   * color. A pixel without one marches the height map of
   * TerrainSunMap.bindHeights itself. enabled false skips it, where no ground is high enough above the water (most
   * places). top: the highest ground, world y, a ray above it ends. bias: how far above the water the rays start, world
   * units.
   */
  public setTerrainReflection(enabled: boolean, top: number, bias: number): void {
    for (const m of this.materials) {
      m.setFloat('reflectTerrainInWater', enabled ? 1 : 0);
      m.setFloat('terrainReflectionTop', top);
      m.setFloat('terrainReflectionBias', bias);
    }
  }

  /**
   * The water reflections. waterMaterials: the gBuffer materials of the water, a bit each (see
   * MapGBuffer.materialIndex), 0 turns them off. darkening: the part of the light the water loses where it reflects a
   * building. reflectBuildings: trace the rays against the buildings, else only the sun is reflected. rippleTile: the size of the tile of the ripples in world units, 0 for a flat mirror. sun: the brightness
   * of the sun reflected where the ray misses the buildings (0 none), sunPower: the phong exponent of its glints.
   * halfRes: the buildings from the rays this pass traced for the gBuffer of the frame (drawn before, with the same
   * waterMaterials and rippleTile) instead of a ray per water pixel, null traces them here
   */
  public setWaterReflections(waterMaterials: number, reflectBuildings: boolean, darkening: number, rippleTile: number, sun: number, sunPower: number,
    halfRes: WaterReflectionHalfRes | null = null): void {
    for (const m of this.materials) {
      m.setConstant('REFLECTIONS_HALF_RES', halfRes != null);
      if (halfRes) {
        m.setTexture('reflectionTexture', halfRes.view);
        m.setFloat('reflectionPhase', halfRes.phase);
        m.setConstant('REFLECTION_BLOCK_X', halfRes.blockX);
        m.setConstant('REFLECTION_BLOCK_Y', halfRes.blockY);
      }
      // constants: the shader of the water reflections that are off has none of their code
      m.setConstant('WATER_REFLECTIONS', waterMaterials !== 0);
      m.setConstant('REFLECT_BUILDINGS', reflectBuildings);
      m.setConstant('WATER_RIPPLES', rippleTile > 0);
      m.setFloat('sunReflection', sun);
      m.setFloat('sunReflectionPower', sunPower);
      m.setFloat('waterMaterials', waterMaterials);
      m.setFloat('reflectionDarkening', darkening);
      m.setFloat('rippleTile', rippleTile);
    }
  }

  /**
   * The rays of the buildings in the water over the screen first, with the colors of what they hit there
   * (SCREEN_REFLECTIONS). Every frame before draw, with the state ScreenReflections.update made: the ray per water pixel
   * here, and the samples of WaterReflectionHalfRes (setWaterReflections) only blend their colors with it.
   */
  public setScreenReflections(screen: ScreenReflections): void {
    for (const m of this.materials) screen.bind(m);
  }

  /**
   * The camera of the gBuffer (jittered): its matrices reconstruct the positions. halfRes: the samples of the lightmap
   * at half resolution, drawn before when the shading reads them (SkyVisibilityLightmap.halfResActive)
   */
  public draw(camera: Camera, gBuffer: MapGBuffer, destination: RenderTexture, halfRes: SkyVisibilityHalfRes): void {
    const m = this.material;
    halfRes.bindShading(m);
    m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    m.setMatrix('inverseView', camera.viewMatrix.inverse);
    m.setMatrix('viewProjection', camera.viewProjection);
    m.setVector3('cameraPosition', camera.worldPosition);
    m.setVector2('resolution', gBuffer.target.size);
    m.setFloat('frame', this.frame++);
    m.setTexture('gBufferTexture', gBuffer.view);
    m.setTexture('gBufferAtlasTexture', gBuffer.atlasView);
    m.setTexture('gBufferDepth', gBuffer.depthView);
    m.setVector2('depthParams', MapGBuffer.depthParams(camera, this.depthParams));
    m.setBuffer('materialPalette', gBuffer.paletteBuffer);
    this.renderer.blit(null, destination, m);
  }

  public destruct(): void {
    this.dummyCityBuffer.destroy();
    this.dummySSAOTexture.destruct();
  }
}
