import IDestructible from "webGPU-framework/renderer/core/IDestructible";
import ParamFloat from "webGPU-framework/utils/params/ParamFloat";
import ParamGroup from "webGPU-framework/utils/params/ParamGroup";
import RenderTexture from "webGPU-framework/renderer/texture/RenderTexture";
import Texture from "webGPU-framework/renderer/texture/Texture";
import Vector2 from "webGPU-framework/renderer/math/Vector2";
import VertexFragmentShader from "webGPU-framework/renderer/material/VertexFragmentShader";
import Renderer from "webGPU-framework/renderer/render/Renderer";
import smolGaussianBlurShader from './smolGaussianBlur.wgsl?raw';

export type SmolGaussianBlurOptions = {
  format:    GPUTextureFormat,
  //downscales at most, each one halves the image
  maxLevels: number,
  //in source pixels, the standard deviation. Used until init() adds the param
  radius:    number,
  //the range of the param
  maxRadius: number,
}

export const defaultSmolGaussianBlurOptions: SmolGaussianBlurOptions = {
  format:    'rgba8unorm',
  maxLevels: 8,
  radius:    20,
  maxRadius: 256,
};

/**
 * Gaussian blur with a continuous radius at a cost that does not grow with it, after Smol Gaussian
 * (Aras Pranckevičius, "Fast blur with animated radius", 2026, https://github.com/aras-p/blur-playground).
 *
 * Halves the source until the radius is 3 to 6 texels of the level, runs a separable gaussian there and enlarges
 * the result back up in one cubic B-spline pass. The downscales and the upscale blur as well, and kernels add up in
 * variance, so the gaussian gets what is left of the variance after taking theirs off. The radius is then exact at
 * every level count and the gaussian's sigma moves continuously with it, so an animated radius stays a gaussian
 * all the way: VariableBlur crossfades two pyramids in between its pass counts instead, which keeps the variance but
 * makes the shape a mix of a narrow and a wide kernel. When the radius crosses into another level count, both sides
 * are close to the same gaussian, so the switch does not show.
 *
 * Differences from Smol Gaussian: the downscale is the 5 tap tent of blur.wgsl instead of a box, which aliases less
 * and costs little at a quarter of the pixels, and odd sizes drop their last row/column instead of integrating the
 * area (as in VariableBlur), so the right and bottom edge clamp up to 2^levels - 1 pixels early.
 *
 * Draws into a destination of the source size, see draw().
 */
export default class SmolGaussianBlur implements IDestructible {
  //per axis, in texels of the level a pass reads, see smolGaussianBlur.wgsl
  private static readonly DOWN_VARIANCE = 0.75;
  private static readonly UP_VARIANCE = 1 / 3;
  //a level is added while the radius is at least this many texels of the level, so the gaussian runs at 3 to 6
  private static readonly LEVEL_SIGMA = 6;

  private readonly levels: RenderTexture[] = [];
  private readonly temp: RenderTexture[] = [];
  private renderer: Renderer;
  private downMaterial: VertexFragmentShader;
  private gaussianMaterial: VertexFragmentShader;
  private upMaterial: VertexFragmentShader;
  private options: SmolGaussianBlurOptions;
  private radiusParam: ParamFloat | null = null;
  private texelStep = new Vector2();
  private uvScale = new Vector2();

  private _radius = 0;
  private _levelCount = 0;
  private _levelSigma = 0;

  constructor(renderer: Renderer, options: Partial<SmolGaussianBlurOptions> = {}) {
    this.options = {...defaultSmolGaussianBlurOptions, ...options};
    this.renderer = renderer;

    const samplerClamp = this.renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });

    this.downMaterial = this.createMaterial(samplerClamp, 0);
    this.gaussianMaterial = this.createMaterial(samplerClamp, 1);
    this.upMaterial = this.createMaterial(samplerClamp, 2);

    //index by level, level 0 is the source. Created at 1x1, draw() sizes them
    for (let i = 0; i <= this.options.maxLevels; i++) {
      this.levels.push(this.createBuffer());
      this.temp.push(this.createBuffer());
    }
  }

  public init(paramGroup: ParamGroup, name: string = 'smolGaussianBlur'): void {
    const pg: ParamGroup = paramGroup.addGroup(name);
    //in source pixels: the standard deviation of the kernel, not its footprint
    this.radiusParam = pg.addParamFloat('blurRadius', this.options.radius, 0, this.options.maxRadius);
  }

  public get radius(): number {
    return this.radiusParam ? this.radiusParam.value : this.options.radius;
  }

  /** In source pixels. Sets the param when init() added one, so one instance can blur at several radii a frame. */
  public set radius(value: number) {
    if (this.radiusParam) this.radiusParam.setValue(value, false);
    else this.options.radius = value;
  }

  /** The radius the last draw blurred with, after clamping to what the source allows. */
  public get drawnRadius(): number {
    return this._radius;
  }

  /** How many times the last draw halved the source before the gaussian. */
  public get levelCount(): number {
    return this._levelCount;
  }

  /** The sigma of the gaussian of the last draw, in texels of the level it ran on. */
  public get levelSigma(): number {
    return this._levelSigma;
  }

  /**
   * Blurs source into destination, which is expected to be the size of the source (null is the canvas).
   * The source and destination can not be the same texture.
   * @param rows  when set, only these rows of the destination are blurred, as [start, end) pairs in pixels; the rest is
   *              cleared. Every pass then only draws the rows its output feeds, padded by what its kernel reaches.
   */
  public draw(source: Texture, destination: RenderTexture | null, rows: [number, number][] | null = null): void {
    const maxLevels = this.maxUsefulLevels(source);
    //past one more level than it may add the gaussian keeps growing, cap it there
    const sigma = Math.min(Math.max(this.radius, 0), SmolGaussianBlur.LEVEL_SIGMA * Math.pow(2, maxLevels + 1));

    let k = 0;
    while (k < maxLevels && sigma / Math.pow(2, k) >= SmolGaussianBlur.LEVEL_SIGMA) k++;
    const scale = Math.pow(2, k);

    //variance in source pixels squared. A downscale into level i blurs by its variance in texels of level i - 1,
    //which are 2^(i-1) source pixels; the upscale by its variance in texels of level k
    let variance = sigma * sigma;
    for (let i = 1; i <= k; i++) variance -= Math.pow(4, i - 1) * SmolGaussianBlur.DOWN_VARIANCE;
    if (k > 0) variance -= scale * scale * SmolGaussianBlur.UP_VARIANCE;
    //below about a third of a texel a gaussian is a copy, and the variance of the ones before can't be undone
    const levelSigma = Math.sqrt(Math.max(variance, 0)) / scale;

    this._radius = sigma;
    this._levelCount = k;
    this._levelSigma = levelSigma;
    const taps = levelSigma > 0.01 ? Math.ceil(4 * levelSigma) : 0;

    //the rows each pass writes, worked back from the destination. null draws everything
    const downRows: ([number, number][] | null)[] = [];
    let gaussianRows: [number, number][] | null = null;
    let tempRows: [number, number][] | null = null;
    if (rows) {
      const heights = [source.height];
      for (let i = 1; i <= k; i++) heights.push(Math.max(1, Math.floor(heights[i - 1] * 0.5)));
      //the B-spline reads 2 texels to either side of the level pixel it lands in
      gaussianRows = k > 0 ? SmolGaussianBlur.mapRows(rows, 1 / scale, 3, heights[k]) : rows;
      //the vertical gaussian reads taps rows to either side, plus one for the paired bilinear fetch
      tempRows = SmolGaussianBlur.mapRows(gaussianRows, 1, taps + 1, heights[k]);
      //the horizontal gaussian reads the rows it writes. The tent into level i reads level i - 1 rows 2r - 1 to 2r + 2
      downRows[k] = tempRows;
      for (let i = k; i > 1; i--) downRows[i - 1] = SmolGaussianBlur.mapRows(downRows[i]!, 2, 2, heights[i - 1]);
    }

    //downscale
    let level: Texture = source;
    for (let i = 1; i <= k; i++) {
      const destinationLevel = this.levels[i];
      destinationLevel.setSize(Math.max(1, Math.floor(level.width * 0.5)), Math.max(1, Math.floor(level.height * 0.5)));

      //a full texel out, the corner taps then each sit on a texel corner of the source
      this.texelStep.setValues(1 / level.width, 1 / level.height);
      this.downMaterial.setVector2('texelStep', this.texelStep);
      //the largest 2:1 rectangle of the source that fits, see vs_main
      this.uvScale.setValues(2 * destinationLevel.width / level.width, 2 * destinationLevel.height / level.height);
      this.downMaterial.setVector2('uvScale', this.uvScale);

      this.blit(level, destinationLevel, this.downMaterial, downRows[i] ?? null);
      level = destinationLevel;
    }

    //gaussian, a horizontal pass into temp and a vertical one back into the level, or into the destination when there
    //is nothing to enlarge
    const temp = this.temp[k];
    temp.setSize(level.width, level.height);
    this.gaussianMaterial.setFloat('sigma', levelSigma);
    this.gaussianMaterial.setFloat('taps', taps);

    this.texelStep.setValues(1 / level.width, 0);
    this.gaussianMaterial.setVector2('texelStep', this.texelStep);
    this.blit(level, temp, this.gaussianMaterial, tempRows);

    this.texelStep.setValues(0, 1 / level.height);
    this.gaussianMaterial.setVector2('texelStep', this.texelStep);
    if (k === 0) {
      this.blit(temp, destination, this.gaussianMaterial, gaussianRows);
      return;
    }
    this.blit(temp, this.levels[k], this.gaussianMaterial, gaussianRows);

    //upscale, a source pixel p is at p / 2^k texels into level k
    this.uvScale.setValues(source.width / (scale * level.width), source.height / (scale * level.height));
    this.upMaterial.setVector2('uvScale', this.uvScale);
    this.blit(this.levels[k], destination, this.upMaterial, rows);
  }

  private blit(source: Texture, destination: RenderTexture | null, material: VertexFragmentShader, rows: [number, number][] | null): void {
    if (rows) this.renderer.blitRows(source, destination, material, rows);
    else this.renderer.blit(source, destination, material);
  }

  //row ranges scaled into another level and padded, clamped to its height, sorted and merged
  private static mapRows(rows: [number, number][], scale: number, pad: number, height: number): [number, number][] {
    const mapped: [number, number][] = [];
    for (const [start, end] of rows) {
      const a = Math.max(0, Math.floor(start * scale) - pad);
      const b = Math.min(height, Math.ceil(end * scale) + pad);
      if (b > a) mapped.push([a, b]);
    }
    mapped.sort((p, q) => p[0] - q[0]);
    const merged: [number, number][] = [];
    for (const r of mapped) {
      const last = merged[merged.length - 1];
      if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
      else merged.push([r[0], r[1]]);
    }
    return merged;
  }

  public destruct() {
    for (const buffer of this.levels) buffer.destruct();
    for (const buffer of this.temp) buffer.destruct();
    this.downMaterial.destruct();
    this.gaussianMaterial.destruct();
    this.upMaterial.destruct();
  }

  private createMaterial(sampler: GPUSampler, pass: number): VertexFragmentShader {
    const material = new VertexFragmentShader(this.renderer, 'smolGaussianBlur', smolGaussianBlurShader);
    material.setConstant('PASS', pass);
    material.depthWriteEnabled = false;
    material.depthCompare = 'always';
    material.setSampler('textureSampler', sampler);
    //uniforms start out at 0, which would collapse every tap onto the first texel
    this.uvScale.setValues(1, 1);
    material.setVector2('uvScale', this.uvScale);
    return material;
  }

  //a level of 2x2 is the smallest that still has a neighbour to blur with
  private maxUsefulLevels(source: Texture): number {
    const levels = Math.floor(Math.log2(Math.min(source.width, source.height))) - 1;
    return Math.max(0, Math.min(this.options.maxLevels, levels));
  }

  private createBuffer(): RenderTexture {
    return new RenderTexture(this.renderer, 1, 1, {formats: [this.options.format], depth: false});
  }
}
