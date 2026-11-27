import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import cameraRay from '../shaders/cameraRay.wgsl?raw';
import MapGBuffer from '../deferred/MapGBuffer';
import shader from './skyVisibilityHalfRes.wgsl?raw';

/**
 * The sky visibility lightmap at half resolution (skyVisibilityHalfRes.wgsl): the lookup of getLightmapLighting for 1
 * pixel per 2 x 2 block of the gBuffer, into an rg32uint target of the visibility and a key of the surface point. With
 * the sun in the lightmap (SkyVisibilityLightmap.setSun) the sun visibility too, in the same lookup. The
 * deferred shading blends the samples around a pixel that are on its surface point (getHalfResLighting in
 * deferredShading.wgsl). A lightmap texel covers several pixels (about 3 at texelsPerPixel .3), so the samples are
 * closer together than the texels.
 * A pixel none of whose samples is on its surface point misses: a thin surface, or the corner of one. It then takes the
 * samples of the other surfaces around it.
 * bilinear: the naive alternative, without the surface test. The visibility alone into an r16float target (filtered),
 * which the shading samples bilinearly across the surfaces (HALF_RES_BILINEAR in both shaders).
 */
export default class SkyVisibilityHalfRes {
  private renderer: Renderer;
  public readonly material: VertexFragmentShader;
  public readonly target: RenderTexture;
  //of bilinear: HALF_RES_BILINEAR
  public readonly bilinearMaterial: VertexFragmentShader;
  public readonly filteredTarget: RenderTexture;
  private readonly sampler: GPUSampler;
  //the bilinear upsample instead of the surface test, set by SkyVisibilityLightmap (LightmapResolution.HalfBilinear)
  public bilinear = false;
  private frame = 0;
  private readonly depthParams = new Vector2();
  //toward the sun, while the lightmap has it (the LIGHTMAP_SUN constant of the materials): see SkyVisibilityLightmap.setSun
  public readonly sunDirection = new Vector3(0, 1, 0);

  constructor(renderer: Renderer, includes: string[]) {
    this.renderer = renderer;
    const create = (name: string, bilinear: boolean) => {
      const m = new VertexFragmentShader(renderer, name, shader, {
        includes: [cameraRay, ...includes],
        defines: bilinear ? { HALF_RES_BILINEAR: true } : {},
      });
      m.depthWriteEnabled = false;
      m.depthCompare = 'always';
      // level + 2 only where level + 1 has no paths, see LEVEL2_PRIOR in skyVisibilityResolve.wgsl
      m.setConstant('LEVEL2_PRIOR', false);
      return m;
    };
    this.material = create('skyVisibilityHalfRes', false);
    this.bilinearMaterial = create('skyVisibilityHalfResBilinear', true);
    this.target = new RenderTexture(renderer, 1, 1, { formats: ['rg32uint'], sampleCount: 1, sizeMultiplier: .5 });
    // the sky and the sun visibility
    this.filteredTarget = new RenderTexture(renderer, 1, 1, { formats: ['rg16float'], sampleCount: 1, sizeMultiplier: .5 });
    this.sampler = renderer.device.createSampler({ magFilter: 'linear', minFilter: 'linear', label: 'skyVisibilityHalfRes' });
  }

  //the materials of the pass, both read the lightmap
  public get materials(): VertexFragmentShader[] {
    return [this.material, this.bilinearMaterial];
  }

  /** The nearest texel instead of the 4 around the point (NEAREST_TEXEL in skyVisibilityResolve.wgsl), set by SkyVisibilityLightmap (halfResNearest). */
  public set nearest(nearest: boolean) {
    for (const m of this.materials) m.setConstant('NEAREST_TEXEL', nearest);
  }

  //the samples, a texture_2d<u32>
  public get view(): GPUTextureView {
    return this.target.views[0];
  }

  /** The samples of the gBuffer of this frame. After SkyVisibilityLightmap.setResolveUniforms, which sets the lightmap's. */
  public draw(camera: Camera, gBuffer: MapGBuffer): void {
    const size = gBuffer.target.size;
    const target = this.bilinear ? this.filteredTarget : this.target;
    // rounded up: the samples cover every pixel
    target.setSize(size.x, size.y);
    const m = this.bilinear ? this.bilinearMaterial : this.material;
    m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    m.setMatrix('inverseView', camera.viewMatrix.inverse);
    m.setMatrix('viewProjection', camera.viewProjection);
    m.setVector3('cameraPosition', camera.worldPosition);
    m.setVector2('resolution', size);
    m.setFloat('frame', this.frame++);
    m.setTexture('gBufferTexture', gBuffer.view);
    m.setTexture('gBufferAtlasTexture', gBuffer.atlasView);
    m.setTexture('gBufferDepth', gBuffer.depthView);
    m.setVector2('depthParams', MapGBuffer.depthParams(camera, this.depthParams));
    m.setVector3('lightDirection', this.sunDirection);
    this.renderer.blit(null, target, m);
  }

  /** Binds the samples to a material of the deferred shading. */
  public bindShading(material: VertexFragmentShader): void {
    material.setTexture('halfResTexture', this.view);
    material.setTexture('halfResFiltered', this.filteredTarget.views[0]);
    material.setSampler('halfResSampler', this.sampler);
  }

  public destruct(): void {
    this.target.destruct();
    this.filteredTarget.destruct();
  }
}
