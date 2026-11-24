import IDestructible from "webGPU-framework/renderer/core/IDestructible";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import Renderer from "webGPU-framework/renderer/render/Renderer";
import Texture from "webGPU-framework/renderer/texture/Texture";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import twoPassBlurShader from "./twoPassBlur.wgsl?raw";

export type TwoPassBlurOptions = {
  format:       GPUTextureFormat,
  //half the number of bilinear taps per pass, the kernel it approximates is 4*filterWidth-1 wide
  filterWidth:  number,
}

export const defaultTwoPassBlurOptions: TwoPassBlurOptions = {
  format:       'rgba8unorm',
  filterWidth:  8,
};

/**
 * Separable gaussian blur. Runs a horizontal and a vertical pass at full resolution, each tap sitting
 * between two texels so bilinear filtering folds two kernel samples into one fetch. The radius is exact
 * but the cost grows with it, so prefer {@link Blur} when the radius is large and its shape does not matter.
 */
export default class TwoPassBlur implements IDestructible {
  private renderer: Renderer;
  private renderTexture: RenderTexture;
  private materialX: VertexFragmentShader;
  private materialY: VertexFragmentShader;
  private weightsBuffer: GPUBuffer;
  private offsetsBuffer: GPUBuffer;
  private options: TwoPassBlurOptions;

  constructor(renderer: Renderer, options: Partial<TwoPassBlurOptions> = {}) {
    this.options = {...defaultTwoPassBlurOptions, ...options};
    this.renderer = renderer;

    //4*n-1 (3, 7, 11, 15, ...)
    const data: Float32Array[] = this.generateGaussShaderKernelWeightsAndOffsets(this.options.filterWidth * 4 - 1);

    this.weightsBuffer = this.renderer.createBuffer(data[0].length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.renderer.writeBuffer(this.weightsBuffer, data[0]);

    this.offsetsBuffer = this.renderer.createBuffer(data[1].length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.renderer.writeBuffer(this.offsetsBuffer, data[1]);

    const samplerClamp = this.renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.materialX = this.createMaterial(samplerClamp);
    this.materialY = this.createMaterial(samplerClamp);

    this.renderTexture = new RenderTexture(this.renderer, 1, 1, {formats: [this.options.format], depth: false});
  }

  private createMaterial(sampler: GPUSampler): VertexFragmentShader {
    const material = new VertexFragmentShader(this.renderer, 'twoPassBlur', twoPassBlurShader);
    material.setConstant('HALF_FILTER_WIDTH', this.options.filterWidth);
    material.depthWriteEnabled = false;
    material.setBuffer('weights', this.weightsBuffer);
    material.setBuffer('offsets', this.offsetsBuffer);
    material.setSampler('textureSampler', sampler);
    return material;
  }

  public draw(source: Texture, destination: RenderTexture | null): void {
    this.renderTexture.setSize(source.width, source.height);

    //blit binds the source as 'texture'
    this.materialX.setUniform('blurDir', [1 / source.width, 0]);
    this.renderer.blit(source, this.renderTexture, this.materialX);

    this.materialY.setUniform('blurDir', [0, 1 / source.height]);
    this.renderer.blit(this.renderTexture, destination, this.materialY);
  }

  public destruct(): void {
    this.renderTexture.destruct();
    this.materialX.destruct();
    this.materialY.destruct();
    this.weightsBuffer.destroy();
    this.offsetsBuffer.destroy();
  }

  //https://github.com/mrdooz/tano/blob/master/gauss_gen.cpp
  private generateSeparableGaussKernel(sigma: number, kernelSize: number): number[] {
    const halfKernelSize: number = Math.floor(kernelSize / 2);
    const kernel: number[] = [];
    const mean = halfKernelSize;
    let sum = 0.0;
    for (let x = 0; x < kernelSize; ++x) {
      kernel[x] = Math.sqrt(Math.exp(-0.5 * (Math.pow((x - mean) / sigma, 2.0) + Math.pow((mean) / sigma, 2.0))));
      sum += kernel[x];
    }
    for (let x = 0; x < kernelSize; ++x) {
      kernel[x] /= sum;
    }
    return kernel;
  }

  private getAppropriateSeparableGauss(kernelSize: number): number[] {
    const epsilon = 0.02 / kernelSize;
    let searchStep = 1.0;
    let sigma = 1.0;

    // binary search to find a signal that gives us roughly 0 at the edge of the kernel
    while (true) {
      const kernelAttempt: number[] = this.generateSeparableGaussKernel(sigma, kernelSize);
      if (kernelAttempt[0] > epsilon) {
        if (searchStep > 0.02) {
          sigma -= searchStep;
          searchStep *= 0.1;
          sigma += searchStep;
          continue;
        }
        const retVal: number[] = [];
        for (let i = 0; i < kernelSize; i++)
          retVal.push(kernelAttempt[i]);
        return retVal;
      }

      sigma += searchStep;

      if (sigma > 1000.0) break;
    }

    throw new Error("getAppropriateSeparableGauss: did not find a value");
  }

  //Acceptable kernel sizes are of 4*n-1 (3, 7, 11, 15, ...)
  private generateGaussShaderKernelWeightsAndOffsets(kernelSize: number): Float32Array[] {
    if ((kernelSize + 1) % 4 == 0) {

      // Gauss filter kernel & offset creation
      const inputKernel = this.getAppropriateSeparableGauss(kernelSize);

      const oneSideInputs: number[] = [];
      const start = Math.floor(kernelSize / 2);
      for (let i = start; i >= 0; i--) {
        if (i == start) {
          oneSideInputs.push(inputKernel[i] * 0.5);
        } else {
          oneSideInputs.push(inputKernel[i]);
        }
      }
      const numSamples = oneSideInputs.length / 2;

      const weights = new Float32Array(numSamples);
      const offsets = new Float32Array(numSamples);

      for (let i = 0; i < numSamples; i++) {
        const sum = oneSideInputs[i * 2 + 0] + oneSideInputs[i * 2 + 1];
        weights[i] = sum;
        offsets[i] = i * 2.0 + oneSideInputs[i * 2 + 1] / sum;
      }
      return [weights, offsets];
    }
    throw new Error("incorrect kernel size");
  }
}
