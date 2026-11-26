import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import cameraRay from '../cameraRay.wgsl?raw';
import cityLightmapSurface from '../city/cityLightmapSurface.wgsl?raw';
import skyVisibilityTrace from '../skyVisibilityLightmap/skyVisibilityTrace.wgsl?raw';
import cityTraceDistance from './cityTraceDistance.wgsl?raw';
import CityScene from '../city/CityScene';
import WaterRippleTexture from './waterRipples/WaterRippleTexture';
import MapGBuffer from './MapGBuffer';
import waterReflection from './waterReflection.wgsl?raw';
import waterScreenReflection from './waterScreenReflection.wgsl?raw';
import shader from './waterReflectionHalfRes.wgsl?raw';
import ScreenReflections from './ScreenReflections';

/**
 * The buildings in the water reflections of MapWorld with fewer rays (waterReflectionHalfRes.wgsl): the reflection ray
 * of the deferred shading for 1 pixel per block of blockX x blockY pixels of the gBuffer, into an rg32uint target
 * (hit or not, the ray and the color) the shading blends by how alike the rays are (getHalfResReflection in
 * deferredShading.wgsl, DeferredShading.setWaterReflections).
 * The rays were more than half of the deferred shading at 4K with a quarter of the screen water (1.05 of 1.7 ms on an
 * RTX 3080). This pass takes .7 ms for half of them and .4 for a quarter: a ray costs about 1.4x as much here as in the
 * shading, whose other work overlaps with it.
 * The colors of what the rays hit, from the screen (ScreenReflections), go into the samples too,
 * which the shading blends with the same weights.
 * The traced pixel of the blocks rotates per frame (rotate, with the TAA): its history averages them, which in a static
 * view is every pixel traced. Without it the sample is always the first pixel of the block.
 */
export default class WaterReflectionHalfRes {
  private renderer: Renderer;
  private readonly material: VertexFragmentShader;
  private readonly target: RenderTexture;
  // the pixels of a block along x and y, see setBlock
  private _blockX = 2;
  private _blockY = 2;
  private frame = 0;
  private _phase = 0;
  private readonly depthParams = new Vector2();
  // bound as the city until CityScene has its buffer: no tiles, so no ray reads it
  private readonly dummyCityBuffer: GPUBuffer;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'waterReflectionHalfRes', shader, {
      includes: [cameraRay, cityLightmapSurface, ...MapGBuffer.readIncludes, waterReflection, waterScreenReflection, skyVisibilityTrace, cityTraceDistance],
      defines: CityScene.traceDefines,
    });
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    this.dummyCityBuffer = renderer.createBuffer(16, GPUBufferUsage.STORAGE, false, 'dummyCityBuffer');
    this.material.setBuffer('cityBuffer', this.dummyCityBuffer);
    this.target = new RenderTexture(renderer, 1, 1, { formats: ['rg32uint'], sampleCount: 1 });
  }

  /** The materials that take the params of the shading by name (the fog, the contrast, the exposure), see ScreenReflections. */
  public get materials(): VertexFragmentShader[] {
    return [this.material];
  }

  /** The tone map of the shading (DeferredShading.toneMap), which the colors of the screen are turned back from. */
  public set toneMap(value: boolean) {
    this.material.setConstant('TONE_MAP', value);
  }

  /** The city the rays trace: CityScene sets its buffer and root on the material. */
  public bindCity(cityScene: CityScene): void {
    cityScene.bind(this.material, false);
  }

  /** The ripples the rays are mirrored on, as the shading's (DeferredShading.bindWaterRipples). */
  public bindWaterRipples(ripples: WaterRippleTexture): void {
    this.material.setTexture('rippleTexture', ripples.texture);
    this.material.setSampler('rippleSampler', ripples.sampler);
  }

  // the samples, a texture_2d<u32>
  public get view(): GPUTextureView {
    return this.target.views[0];
  }

  /**
   * The pixels that share a ray, along x and y (REFLECTION_BLOCK_X and Y of waterReflection.wgsl): 2 x 2 is a quarter
   * of the rays, 2 x 1 half of them. Before draw.
   */
  public setBlock(x: number, y: number): void {
    this._blockX = x;
    this._blockY = y;
  }

  public get blockX(): number {
    return this._blockX;
  }

  public get blockY(): number {
    return this._blockY;
  }

  /** Which pixel of the blocks the last draw traced, for the shading (getReflectionSampleOffset in waterReflection.wgsl). */
  public get phase(): number {
    return this._phase;
  }

  // The pixel of a block for a frame: its index with the bits reversed, so 2 frames after each other are far apart in
  // the block. A block of 2 x 2 goes over its diagonal first: 2 frames then cover both rows and both columns
  private getPhase(frame: number): number {
    const count = this._blockX * this._blockY;
    const index = frame % count;
    if (this._blockX === 2 && this._blockY === 2) return [0, 3, 1, 2][index];
    const bits = Math.ceil(Math.log2(count));
    let reversed = 0;
    for (let i = 0; i < bits; i++) reversed |= ((index >> i) & 1) << (bits - 1 - i);
    // a count that is not a power of 2: the reversed index can be past the end
    return reversed < count ? reversed : index;
  }

  /**
   * The samples of the gBuffer of this frame, before the shading. waterMaterials and rippleTile: as
   * DeferredShading.setWaterReflections takes them. rotate: another pixel of the blocks every frame, for a TAA to average.
   * screen: the colors of what the rays hit, see ScreenReflections.
   */
  public draw(camera: Camera, gBuffer: MapGBuffer, waterMaterials: number, rippleTile: number, rotate: boolean, screen: ScreenReflections): void {
    const size = gBuffer.target.size;
    // rounded up: the samples cover every pixel
    this.target.setSize(Math.ceil(size.x / this._blockX), Math.ceil(size.y / this._blockY));
    this._phase = rotate ? this.getPhase(this.frame++) : 0;
    const m = this.material;
    m.setConstant('WATER_RIPPLES', rippleTile > 0);
    m.setConstant('REFLECTION_BLOCK_X', this._blockX);
    m.setConstant('REFLECTION_BLOCK_Y', this._blockY);
    m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    m.setMatrix('inverseView', camera.viewMatrix.inverse);
    m.setVector3('cameraPosition', camera.worldPosition);
    m.setVector2('resolution', size);
    m.setVector2('depthParams', MapGBuffer.depthParams(camera, this.depthParams));
    m.setFloat('waterMaterials', waterMaterials);
    m.setFloat('phase', this._phase);
    m.setFloat('rippleTile', rippleTile);
    m.setTexture('gBufferTexture', gBuffer.view);
    m.setTexture('gBufferDepth', gBuffer.depthView);
    m.setMatrix('viewProjection', camera.viewProjection);
    screen.bind(m);
    this.renderer.blit(null, this.target, m);
  }

  public destruct(): void {
    this.target.destruct();
    this.dummyCityBuffer.destroy();
  }
}
