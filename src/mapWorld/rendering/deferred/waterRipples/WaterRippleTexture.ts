import Renderer from "webGPU-framework/renderer/render/Renderer";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import shader from './waterRippleBake.wgsl?raw';

export default class WaterRippleTexture {
  private static readonly SIZE = 512;
  private static readonly TILE_CELLS = 12;

  public readonly texture: RenderTexture;
  public readonly sampler: GPUSampler;
  private renderer: Renderer;
  public material: VertexFragmentShader;
  private _wavelength = 1;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'waterRippleBake', shader);
    this.material.depthWriteEnabled = false;
    //a gradient, two channels
    this.texture = new RenderTexture(renderer, WaterRippleTexture.SIZE, WaterRippleTexture.SIZE, {
      formats: ['rg16float'],
      useMips: true,
    });
    this.sampler = renderer.device.createSampler({
      addressModeU: 'repeat', addressModeV: 'repeat',
      magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear',
      // far away a pixel is many times longer along the view than across, see getRippleNormal of waterReflection.wgsl
      maxAnisotropy: 16,
    });
    this.wavelength = this._wavelength;
  }

  //the wavelength of the coarsest octave in world units
  public get wavelength(): number {
    return this._wavelength;
  }

  public set wavelength(value: number) {
    this._wavelength = value;
    this.material.setFloat('wavelength', value);
  }

  //the size of the tile in world units, the water divides its position by it
  public get tileSize(): number {
    return WaterRippleTexture.TILE_CELLS * this._wavelength;
  }

  //every frame, outside of a render pass (it starts its own): the ripples move. time in seconds
  public update(time: number) {
    this.material.setFloat('time', time);
    this.renderer.blit(null, this.texture, this.material);
    this.renderer.generateMipmaps(this.texture);
  }
}
