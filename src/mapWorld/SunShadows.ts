import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Transform from 'webGPU-framework/renderer/scene/Transform';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import shaderSource from '../mapOSM/shading.wgsl?raw';
import ShadowMap, { ShadowMapAhead } from '../mapOSM/shadowMap/ShadowMap';
import viewShaderSource from './shadowMapView.wgsl?raw';
import { SHADOW_MODES, ShadowMode } from './MapWorldSettings';
import { setMeshQuantization } from './city/meshQuantization';

/**
 * The sun: its direction (sunAngleX/Y params) and its cascaded VSM shadow map, with the buildings as the only casters.
 * The shadowMode param picks where the sun shadows come from (see ShadowMode): this shadow map, or rays against the city
 * of the lightmap (CityScene), per pixel or per lightmap texel, see DeferredShading.setSunShadows. MapWorld skips update
 * when the shadow map isn't used.
 */
export default class SunShadows {
  public readonly shadowMap: ShadowMap;
  public lightDir: Vector3 = new Vector3(0, -1, 0);
  // Depth-only variant writing VSM moments.
  private depthMaterial: VertexFragmentShader;
  private sampler: GPUSampler;
  private sunAngleX!: ParamFloat;
  private sunAngleY!: ParamFloat;
  // where the sun shadows come from, see ShadowMode
  public mode!: ParamSelect;
  // the forward shading has only the shadow map, it turns this down to 0 for ShadowMode.None
  public shadowStrength!: ParamFloat;
  // the angular radius of the sun the ray traced and the lightmap shadows sample
  public sunRadius!: ParamFloat;
  private materials: VertexFragmentShader[] = [];
  // the shadowMap drawMode, see drawView
  private viewMaterial: VertexFragmentShader;
  private onChange: () => void = () => {};

  constructor(private renderer: Renderer) {
    this.shadowMap = new ShadowMap(renderer);
    // the casters are the building meshes, with their unorm16 vertices
    this.depthMaterial = new VertexFragmentShader(renderer, 'osm', shaderSource, { includes: [hash], defines: { QUANTIZED: true } });
    this.depthMaterial.setConstant('DRAW_DEPTH', true);
    this.sampler = renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.bindResources(this.depthMaterial);

    this.viewMaterial = new VertexFragmentShader(renderer, 'shadowMapView', viewShaderSource);
    this.viewMaterial.depthWriteEnabled = false;
    this.viewMaterial.depthCompare = 'always';
    this.viewMaterial.setTexture('shadowMaps', this.shadowMap.cascadeTexture);
  }

  /** The shadow map textures and data of the shading, on a material that shades with them. */
  public bindResources(material: VertexFragmentShader): void {
    material.setSampler('shadowMapSampler', this.sampler);
    material.setTexture('shadowMaps', this.shadowMap.cascadeTexture);
    material.setBuffer('shadowMapData', this.shadowMap.shadowMapDataBuffer);
  }

  /** The sun and shadow params into pgShading; they go to the materials by name. */
  public initParams(pgShading: ParamGroup, materials: VertexFragmentShader[], onChange: () => void): void {
    this.materials = materials;
    this.onChange = onChange;
    const params = { materials, callback: onChange };
    const sunParams = { callback: () => this.updateLightDirection() };
    this.sunAngleX = pgShading.addParamFloat('sunAngleX', 1, -0.05, Math.PI * .5, sunParams);
    this.sunAngleY = pgShading.addParamFloat('sunAngleY', .5, -Math.PI, Math.PI, sunParams);
    this.updateLightDirection();

    this.mode = pgShading.addParamSelect('shadowMode', ShadowMode.ShadowMap, SHADOW_MODES, { callback: onChange });
    this.shadowStrength = pgShading.addParamFloat('shadowStrength', 1, 0, 1, params);
    pgShading.addParamFloat('VSMBleedThreshold', .65, 0, 1, params);
    pgShading.addParamFloat('varianceMin', .01, 0, .05, params);
    // the angular radius of the sun the soft ray traced shadows sample, in radians: the real sun is .0047. Not the sunSize
    // of the sun drawn in the sky, which is larger
    this.sunRadius = pgShading.addParamFloat('shadowSunRadius', .01, 0, .05, params);
  }

  /** Sets the sunAngleX/Y params (the GUI follows), see TimeOfDay. */
  public setSunAngles(angleX: number, angleY: number): void {
    // with the callbacks: they update the light direction and the GUI
    this.sunAngleX.setValue(angleX);
    this.sunAngleY.setValue(angleY);
  }

  /** The ShadowMap params. */
  public init(paramGroup: ParamGroup): void {
    this.shadowMap.init(paramGroup);
  }

  private updateLightDirection(): void {
    const t = new Transform();
    t.rotateY(this.sunAngleY.value);
    t.rotateX(-this.sunAngleX.value);
    const dir = t.forward;
    this.lightDir = dir.clone();
    for (const material of this.materials) material.setVector3('lightDirection', dir);
    this.onChange();
  }

  /**
   * Renders the cascades that need it, returns whether they settled. With ahead the cascades go in front of the camera
   * instead of around its pivot, see ShadowMap.update.
   */
  // casters: the building meshes and the decoding of their vertices, see setMeshQuantization
  public update(camera: MapCamera, castersChanged: boolean, casters: { meshes: Mesh[]; quantization: number[] }, ahead: ShadowMapAhead | null = null): boolean {
    return this.shadowMap.update(
      this.lightDir,
      camera.pivot.worldPosition,
      camera.distance,
      castersChanged,
      (cam) => {
        this.depthMaterial.setMatrix('viewProjection', cam.viewProjection);
        casters.meshes.forEach((mesh, i) => {
          setMeshQuantization(this.depthMaterial, casters.quantization, i);
          this.renderer.draw(mesh, this.depthMaterial);
        });
      },
      ahead,
    );
  }

  /** The 4 cascades as a 2 x 2 grid on the screen, letterboxed to a square (the shadowMap drawMode). */
  public drawView(metersToNorm: number): void {
    // of the canvas: renderer.aspectRatio is that of the current render target, a (square) cascade on the frames one
    // was re-rendered
    const canvas = this.renderer.canvas;
    this.viewMaterial.setFloat('aspectRatio', canvas.width / canvas.height);
    this.viewMaterial.setFloat('metersToNorm', metersToNorm);
    this.renderer.blit(null, null, this.viewMaterial, true);
  }

  destruct(): void {
    this.shadowMap.destruct();
  }
}
