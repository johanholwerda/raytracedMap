import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Color from 'webGPU-framework/renderer/math/Color';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import cityLightmapSurface from '../city/cityLightmapSurface.wgsl?raw';
import packing from './mapGBufferPacking.wgsl?raw';
import shader from './mapGBuffer.wgsl?raw';
import { GBUFFER_ATLAS_STEP_M } from '../city/cityLayout';

// the materials the 5 bits of the gBuffer can tell apart, see mapGBufferPacking.wgsl
const MATERIAL_MAX = 32;

/**
 * The gBuffer of MapWorld, drawn once per frame with the jittered camera (see TAA). Everything that used to draw the
 * scene again reads it: the deferred shading (DeferredShading), the city lightmap update, the SSAO and the TAA.
 * 2 targets of 12 bytes per pixel together (rg32uint: the lightmap surface id, the normal, the material and the high bits
 * of the wall atlas position, r32uint: its low bits) and the depth, which the readers turn into the camera distance.
 * The depth is reversed (1 at the near plane, 0 far away, cleared to 0) and depth32float, in this pass only: float
 * precision where the depth is small, so the distance is exact to about 1e-7 of itself. A standard depth near 1 was off
 * by up to 1e-4 of the distance (cm), more than the ray offset of the sky visibility trace (MIN_DIST).
 * Packed as in mapGBufferPacking.wgsl (the readers include it). The albedo of a material is in paletteBuffer. 16 bytes
 * written per pixel, and the passes that only need the distance (SSAO, TAA) read 4 bytes per sample. No MSAA.
 */
export default class MapGBuffer {
  private renderer: Renderer;
  public readonly target: RenderTexture;
  private materials: VertexFragmentShader[] = [];
  // the palette index of each material, and the palette entries in use (see addPaletteColors)
  private materialIndices = new Map<VertexFragmentShader, number>();
  private paletteCount = 0;
  private readonly clearColor = new Color(0, 0, 0, 0);
  // the viewProjection of the camera with the depth reversed: clip z' = w - z
  private readonly reverseDepth = new Matrix4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1);
  private readonly viewProjection = new Matrix4x4();
  // the albedo of each material (rgb, as the param, not squared), array<vec4<f32>> for the deferred shading
  public readonly paletteBuffer: GPUBuffer;
  private readonly palette = new Float32Array(MATERIAL_MAX * 4);
  //the includes of a shader that reads the gBuffer
  public static readonly readIncludes = [packing];

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.target = new RenderTexture(renderer, 1, 1, {
      formats: ['rg32uint', 'r32uint'],
      // read after the pass: the camera distance, see getGBufferDistance in mapGBufferPacking.wgsl
      depth: true,
      depthFormat: 'depth32float',
      sampleCount: 1,
      scaleToCanvas: true,
      toScreen: false,
    });
    this.paletteBuffer = renderer.createBuffer(this.palette.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, false, 'mapGBufferPalette');
  }

  //the surface id, the normal, the material and the high bits of the wall atlas position
  public get view(): GPUTextureView {
    return this.target.views[0];
  }

  //the low bits of the wall atlas position
  public get atlasView(): GPUTextureView {
    return this.target.views[1];
  }

  //the depth, a texture_depth_2d
  public get depthView(): GPUTextureView {
    return this.target.depthStencilView!;
  }

  /**
   * The uniform depthParams of a reader of the depth (see getGBufferDistance in mapGBufferPacking.wgsl): 1 / the view
   * depth is depth * x + y. In double precision here, so the reader has no cancellation.
   */
  public static depthParams(camera: Camera, out: Vector2 = new Vector2()): Vector2 {
    const near = camera.projection.near, far = camera.projection.far;
    return out.setValues((far - near) / (near * far), 1 / far);
  }

  /** The step of the wall atlas positions in the gBuffer, in world units. The readers get it as a uniform. */
  public static atlasStep(metersToNorm: number): number {
    return GBUFFER_ATLAS_STEP_M * metersToNorm;
  }

  /**
   * A material that draws into the gBuffer. walls: the buildings, with the lightmap attribute and the selection
   * highlight. cullMode: 'back' for closed meshes, 'none' for the ground quad, whose winding doesn't matter.
   */
  public createMaterial(baseColor: Vector3, walls: boolean, cullMode: GPUCullMode = 'back'): VertexFragmentShader {
    const index = this.addPaletteColors([[baseColor.x, baseColor.y, baseColor.z]]);
    const material = new VertexFragmentShader(this.renderer, 'mapGBuffer', shader, {
      includes: [cityLightmapSurface, packing],
      defines: walls ? { WALLS: true } : {},
    });
    material.cullMode = cullMode;
    // the reversed depth: nearer is larger
    material.depthCompare = 'greater';
    material.setFloat('material', index);
    material.setFloat('selectedID', -1);
    material.setFloat('tileID', 0);
    material.setVector2('wallTile', new Vector2(0, 0));
    material.setFloat('treeMaterial', 0);
    this.materials.push(material);
    this.materialIndices.set(material, index);
    return material;
  }

  /**
   * Palette entries without a material of their own: a material with WALLS picks them per vertex (the trees, see
   * mapGBuffer.wgsl). Returns the index of the first.
   */
  public addPaletteColors(colors: [number, number, number][]): number {
    const index = this.paletteCount;
    if (index + colors.length > MATERIAL_MAX) throw new Error(`MapGBuffer: more than ${MATERIAL_MAX} materials`);
    colors.forEach((c, i) => this.palette.set([c[0], c[1], c[2], 1], (index + i) * 4));
    this.paletteCount += colors.length;
    this.renderer.writeBuffer(this.paletteBuffer, this.palette);
    return index;
  }

  /** Changes the albedo of a palette entry (of createMaterial or addPaletteColors). */
  public setPaletteColor(index: number, color: [number, number, number]): void {
    this.palette.set([color[0], color[1], color[2], 1], index * 4);
    this.renderer.writeBuffer(this.paletteBuffer, this.palette);
  }

  /** The index of a material of createMaterial, as getGBufferMaterial of mapGBufferPacking.wgsl reads it. */
  public materialIndex(material: VertexFragmentShader): number {
    return this.materialIndices.get(material) ?? -1;
  }

  /** Starts the pass into the gBuffer with the camera of this frame, the draws of the callback go into it. */
  public render(camera: Camera, metersToNorm: number, draw: () => void): void {
    const atlasStep = MapGBuffer.atlasStep(metersToNorm);
    Matrix4x4.multiply(this.reverseDepth, camera.viewProjection, this.viewProjection);
    for (const material of this.materials) {
      material.setMatrix('viewProjection', this.viewProjection);
      material.setVector3('cameraPosition', camera.worldPosition);
      material.setFloat('atlasStep', atlasStep);
    }
    this.renderer.startRenderPass(this.target, { clearColor: this.clearColor, colorClear: true, depthClear: true, depthClearValue: 0 }, 0, 'mapGBuffer');
    draw();
  }

  public destruct(): void {
    this.target.destruct();
    this.paletteBuffer.destroy();
  }
}
