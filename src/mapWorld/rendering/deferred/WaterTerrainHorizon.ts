import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import cameraRay from '../shaders/cameraRay.wgsl?raw';
import cityLightmapSurface from '../skyVisibilityLightmap/cityLightmapSurface.wgsl?raw';
import terrainSun from '../sun/terrainSun.wgsl?raw';
import snow from '../shaders/snow.wgsl?raw';
import type { SnowParams } from '../../map/snowLine';
import MapGBuffer from './MapGBuffer';
import terrainReflection from './terrainReflection.wgsl?raw';
import shader from './waterTerrainHorizon.wgsl?raw';

/**
 * The terrain in the water reflections of MapWorld with a march over the height map (TerrainSunMap.bindHeights) per
 * block of 4 x 4 pixels instead of per water pixel (waterTerrainHorizon.wgsl): the horizon of the ground along the
 * mirror ray of the water pixel of a block, how steep a ray has to go up there to pass over the ground, and the shaded
 * color of the ground the water mirrors there. The shading compares the ray of each water pixel with the horizons
 * around it and takes their color (getTerrainReflection of terrainHorizonLookup.wgsl). The horizon and the color change
 * slowly across the water, the ripples only tilt the ray, which stays per pixel: the edge of a hill in the water is as
 * sharp as with a march per pixel.
 * A march per water pixel took .31 ms at 1080p with half of the screen water under a hill (the rest of the shading
 * .25 ms), most of it in the rays that pass over the ground.
 * The material takes the sun and sky params of the shading by their names (materials), and the textures of
 * TerrainSunMap (bind, bindHeights and bindMaterials).
 */
export default class WaterTerrainHorizon {
  // the pixels of a block along x and y, mirrors BLOCK of waterTerrainHorizon.wgsl and terrainHorizonLookup.wgsl
  private static readonly BLOCK = 4;

  private renderer: Renderer;
  private readonly material: VertexFragmentShader;
  // the horizons and the colors, what the shader of the reflections reads (terrainHorizon)
  public readonly target: RenderTexture;
  private readonly depthParams = new Vector2();

  constructor(renderer: Renderer, gBuffer: MapGBuffer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'waterTerrainHorizon', shader, {
      includes: [cameraRay, cityLightmapSurface, ...MapGBuffer.readIncludes, terrainReflection, terrainSun, snow],
    });
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    this.material.setBuffer('materialPalette', gBuffer.paletteBuffer);
    this.target = new RenderTexture(renderer, 1, 1, { formats: ['rgba16float'], sampleCount: 1 });
  }

  /** For the shading params and the textures of TerrainSunMap. */
  public get materials(): VertexFragmentShader[] {
    return [this.material];
  }

  /** The highest ground (world y) and how far above the water the rays start, as DeferredShading.setTerrainReflection. */
  public setTerrain(top: number, bias: number): void {
    this.material.setFloat('terrainReflectionTop', top);
    this.material.setFloat('terrainReflectionBias', bias);
  }

  /** The snow on the ground in the reflections, as MapGeometry.setSnow. snowMaterial: its palette index. */
  public setSnow(params: SnowParams, snowMaterial: number): void {
    const m = this.material;
    m.setFloat('snowEnabled', params.enabled ? 1 : 0);
    m.setFloat('snowMaterial', snowMaterial);
    m.setFloat('snowLine', params.line);
    m.setFloat('snowNoise', params.noise);
    m.setFloat('snowMetersPerUnit', params.metersPerUnit);
  }

  /** The horizons of the gBuffer of this frame, before the passes that read them. waterMaterials: a bit each. */
  public draw(camera: Camera, gBuffer: MapGBuffer, waterMaterials: number): void {
    const size = gBuffer.target.size;
    const block = WaterTerrainHorizon.BLOCK;
    // rounded up: the blocks cover every pixel
    this.target.setSize(Math.ceil(size.x / block), Math.ceil(size.y / block));
    const m = this.material;
    m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    m.setMatrix('inverseView', camera.viewMatrix.inverse);
    m.setVector3('cameraPosition', camera.worldPosition);
    m.setVector2('resolution', size);
    m.setVector2('depthParams', MapGBuffer.depthParams(camera, this.depthParams));
    m.setFloat('waterMaterials', waterMaterials);
    m.setTexture('gBufferTexture', gBuffer.view);
    m.setTexture('gBufferDepth', gBuffer.depthView);
    this.renderer.blit(null, this.target, m);
  }

  public destruct(): void {
    this.target.destruct();
    this.material.destruct();
  }
}
