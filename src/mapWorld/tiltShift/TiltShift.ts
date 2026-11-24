import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector4 from 'webGPU-framework/renderer/math/Vector4';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import PivotCamera from 'webGPU-framework/utils/camera/PivotCamera';
import SmolGaussianBlur from 'webGPU-framework/effects/blur/SmolGaussianBlur';
import smolGaussianBlurShader from 'webGPU-framework/effects/blur/smolGaussianBlur.wgsl?raw';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import tiltShiftShader from './tiltShift.wgsl?raw';

//the blurred copies the composite mixes between, each half the radius of the next. Steps of 2x keep a crossfade
//between two of them close to a gaussian, the first step from the sharp image included
const BLUR_COUNT = 5;
//in pixels, a smaller level is not drawn: a crossfade with the sharp image is as good there
const MIN_SIGMA = 1;
//with halfRes: the variance in pixels² of the way down to half resolution and back, which every blurred level has on
//top of its own gaussian. The tent downscale of SmolGaussianBlur (.75 texels² of the full image per axis) and the
//bilinear read at exactly 2:1 (weights .75/.25 a quarter and three quarters of a texel out: .1875 half texels²)
const HALF_RES_VARIANCE = 0.75 + 4 * 0.1875;
//with halfRes, as MIN_SIGMA: the round trip alone is a sigma of 1.22
const HALF_RES_MIN_SIGMA = 1.5;
//frames the blurs of a static view may be older than the sharp image, see draw()
const STALE_FRAMES_MAX = 3;

/**
 * Depth of field of a lens focused on the camera pivot, assuming the scene is the flat ground plane: the miniature
 * look of a tilt-shift photo, without reading scene depth.
 *
 * For a plane, 1 / depth is linear in screen y. With the pivot on the ground at distance d and the camera pitched
 * down by an angle a, 1 / z - 1 / d = -ndcY * tan(fov / 2) * cot(a) / d. The circle of confusion of an aperture A,
 * as a fraction of the screen height, is A * |1 / z - 1 / d| / (2 tan(fov / 2)), so A * |ndcY| * cot(a) / (2 d): the
 * fov drops out. A gaussian of the same variance as a disc of diameter D has sigma D / 4.
 *
 * A lens of a fixed size would blur more the closer the camera gets. The aperture grows with the distance instead
 * (A = aperture * d), like a miniature photographed with a lens to its scale, so the blur does not change with the
 * zoom: A * |ndcY| * cot(a) / (2 d) becomes aperture * |ndcY| * cot(a) / 2. Only the pitch changes it, looking straight
 * down the ground is all in focus.
 * The image is blurred at 5 radii (the largest one the screen edge needs, halving down from it) and every pixel
 * mixes the two around its own radius by variance.
 *
 * halfRes (for mobile) blurs a half resolution copy instead and reads the levels bilinearly. That takes the blurs to
 * a quarter of the pixels: at full resolution the levels below 6 pixels are separable gaussians of up to 25 taps per
 * pass, and the bands of rows of the levels add up to about two screens of full resolution writes. The sharp image
 * stays at full resolution and the variance of the round trip is taken off the gaussians, so the blur keeps its
 * amount. Only the levels below 1.5 pixels are skipped (the sharp image is crossfaded a little further out instead).
 */
export default class TiltShift {
  private renderer: Renderer;
  private blur: SmolGaussianBlur;
  private blurred: RenderTexture[] = [];
  private material: VertexFragmentShader;
  //the composite with HALF_RES, and the downscale into half
  private materialHalf: VertexFragmentShader;
  private downMaterial: VertexFragmentShader;
  private half: RenderTexture;
  private texelStep = new Vector2();
  private uvScale = new Vector2();
  private blurTexel = new Vector2();
  private sigmas = new Vector4();
  //whether target holds the input draw() was last given, and the blurs were made from it with the current params
  private targetDrawn = false;
  private blursValid = false;
  //the frames since the blurs were made, and whether target changed since
  private blurAge = 0;
  private blursStale = false;
  //the material the last draw() presented with, the composite (which also dithers)
  private presentMaterial: VertexFragmentShader | null = null;

  //the scene is drawn into this, draw() presents it
  public readonly target: RenderTexture;
  //called when a param changes the blur (getBlur), which the AO and the lightmap of the host follow
  public onChange: (() => void) | null = null;

  public enabled!: ParamBool;
  //lens diameter as a fraction of the camera distance
  private aperture!: ParamFloat;
  //sigma in pixels, caps the blur at the screen edge
  private maxRadius!: ParamFloat;
  //the blurs at half the resolution, see the class comment
  public halfRes!: ParamBool;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    //the target and the blurs are 10 bits: at 8 every pass rounds the smooth ramps of the sky and the fog to steps of
    //1 / 255 before the composite dithers, which the dither can't break up. Same size as 8 bits, and unlike
    //rg11b10ufloat (6 and 5 bit mantissas) finer than 8 bits at every value of the gamma encoded image. The alpha is
    //not used (2 bits)
    const format: GPUTextureFormat = 'rgb10a2unorm';

    this.target = new RenderTexture(renderer, 1, 1, {formats: [format], depth: false, scaleToCanvas: true, toScreen: false});
    //sized by draw(): to the target, or to half with halfRes
    for (let i = 0; i < BLUR_COUNT; i++) {
      this.blurred.push(new RenderTexture(renderer, 1, 1, {formats: [format], depth: false, toScreen: false}));
    }
    this.half = new RenderTexture(renderer, 1, 1, {formats: [format], depth: false, toScreen: false});
    this.blur = new SmolGaussianBlur(renderer, {format});

    const sampler = renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });
    this.material = this.createMaterial('tiltShift', tiltShiftShader, [hash]);
    this.material.setSampler('blurSampler', sampler);
    this.materialHalf = this.createMaterial('tiltShiftHalf', tiltShiftShader, [hash]);
    this.materialHalf.setSampler('blurSampler', sampler);
    this.materialHalf.setConstant('HALF_RES', 1);
    this.downMaterial = this.createMaterial('tiltShiftDown', smolGaussianBlurShader);
    this.downMaterial.setSampler('textureSampler', sampler);
    this.downMaterial.setConstant('PASS', 0);
  }

  private createMaterial(name: string, shader: string, includes: string[] = []): VertexFragmentShader {
    const material = new VertexFragmentShader(this.renderer, name, shader, {includes});
    material.depthWriteEnabled = false;
    material.depthCompare = 'always';
    return material;
  }

  public init(paramGroup: ParamGroup): void {
    const pg = paramGroup.addGroup('Tilt-shift');
    const params = {callback: () => {
      this.blursValid = false;
      this.onChange?.();
    }};
    //off, draw() still presents target (dithered)
    this.enabled = pg.addParamBool('enabled', true, params);
    this.aperture = pg.addParamFloat('aperture', 0.03, 0, 0.1, params);
    this.maxRadius = pg.addParamFloat('maxRadius', 64, 0, 128, params);
    //the blur amount stays the same, but onChange has the host draw a static view again
    this.halfRes = pg.addParamBool('halfRes', false, params);
  }

  public get active(): boolean {
    return this.enabled.value && this.aperture.value > 0;
  }

  /** Whether target still holds the last input at the canvas size, so it need not be drawn again. */
  public get holdsTarget(): boolean {
    return this.targetDrawn && this.target.width === this.renderer.width && this.target.height === this.renderer.height;
  }

  /**
   * The blur as draw() makes it: the sigma in pixels at |ndc y| = 1 (slope) and the most sigma (max), so a pixel at ndc y
   * gets min(slope * |ndc y|, max). 0 and 0 when there is no blur. For the city lightmap, which needs no finer lighting
   * than the blur leaves.
   */
  public getBlur(camera: PivotCamera): { slope: number; max: number } {
    if (!this.active) return { slope: 0, max: 0 };
    // The target's height, which draw() blurs: renderer.height is the target of the last pass (a shadow cascade).
    const slope = this.getSlope(camera, this.target.height);
    const max = Math.min(slope, this.maxRadius.value);
    return max < 0.25 ? { slope: 0, max: 0 } : { slope, max };
  }

  //sigma in pixels at |ndcY| = 1, for a target of the height
  private getSlope(camera: PivotCamera, height: number): number {
    const distance = Math.max(camera.distance, 1e-6);
    //sin of the pitch, the pivot is on the ground
    const sinPitch = Math.min(Math.max((camera.worldPosition.y - camera.pivot.worldPosition.y) / distance, 0.02), 1);
    const cotPitch = Math.sqrt(1 - sinPitch * sinPitch) / sinPitch;
    //the aperture is aperture * distance, which cancels the distance of the circle of confusion
    return height * this.aperture.value * cotPitch / 8;
  }

  /** Call on a frame target is not drawn into for this effect, the next draw() then needs a new input. */
  public invalidate(): void {
    this.targetDrawn = false;
    this.blursValid = false;
    this.blursStale = false;
  }

  /**
   * Presents target into destination (null is the canvas).
   * @param inputChanged  whether target was drawn this frame. Without a change of it or of the params, the blurs of
   *                      the last draw are presented again. A camera change redraws the scene, so it changes the input.
   * @param staticView  the input changed, but only a little: the jitter of the TAA, the AO or the lighting that
   *                    converge. The blurs are then made every STALE_FRAMES_MAX + 1 frames, the sharp image every
   *                    frame. They are made again on the first frame the input doesn't change, so the image the view
   *                    settles on has its own blurs.
   */
  public draw(camera: PivotCamera, inputChanged: boolean, destination: RenderTexture | null = null, staticView: boolean = false): void {
    const source = this.target;
    if (inputChanged) this.targetDrawn = true;
    const reuse = inputChanged ? staticView && this.blurAge < STALE_FRAMES_MAX : !this.blursStale;
    if (reuse && this.blursValid) {
      if (inputChanged) {
        this.blurAge++;
        this.blursStale = true;
      }
      this.renderer.blit(source, destination, this.presentMaterial);
      return;
    }
    this.blursValid = true;
    this.blurAge = 0;
    this.blursStale = false;

    const slope = this.getSlope(camera, source.height);
    const maxSigma = Math.min(slope, this.maxRadius.value);
    if (!this.active || maxSigma < 0.25) {
      //the composite with slope 0 only copies and dithers (the last pass dithers, see tiltShift.wgsl). It skips the
      //levels, but they need a binding
      const material = this.material;
      for (let i = 0; i < BLUR_COUNT; i++) material.setTexture('blur' + i, source);
      material.setFloat('slope', 0);
      this.presentMaterial = material;
      this.renderer.blit(source, destination, material);
      return;
    }
    const halfRes = this.halfRes.value;
    const material = halfRes ? this.materialHalf : this.material;
    this.presentMaterial = material;

    //the image the levels blur, its pixels per pixel of the target, the variance it already has (in pixels² of the
    //target) and the rows a read of it reaches past a pixel
    let blurSource: RenderTexture = source;
    let scale = 1;
    let baseVariance = 0;
    let rowPad = 1;
    if (halfRes) {
      blurSource = this.drawHalf(source);
      scale = 0.5;
      baseVariance = HALF_RES_VARIANCE;
      rowPad = 2;
      this.blurTexel.setValues(0.5 / blurSource.width, 0.5 / blurSource.height);
      material.setVector2('blurTexel', this.blurTexel);
    }
    const minSigma = halfRes ? HALF_RES_MIN_SIGMA : MIN_SIGMA;

    //a level below minSigma is skipped: sigma 0 and the sharp source in its place, the composite then mixes the
    //source with the first level that is drawn. The largest one is always drawn
    const requested: number[] = [];
    for (let i = 0; i < BLUR_COUNT; i++) {
      const sigma = maxSigma / Math.pow(2, BLUR_COUNT - 1 - i);
      requested.push(sigma < minSigma && i < BLUR_COUNT - 1 ? 0 : sigma);
    }
    //in pixels of the target, like slope
    const sigmas: number[] = [];
    for (let i = 0; i < BLUR_COUNT; i++) {
      const sigma = requested[i];
      if (sigma === 0) {
        sigmas.push(0);
        material.setTexture('blur' + i, source);
        continue;
      }
      //the composite reads a level where the sigma of the pixel lies between the levels on either side of it, which
      //are two bands of rows (one around the center line for the first level drawn)
      const lo = i > 0 ? requested[i - 1] : 0;
      const hi = i < BLUR_COUNT - 1 ? requested[i + 1] : Infinity;
      const blurred = this.blurred[i];
      blurred.setSize(blurSource.width, blurSource.height);
      //what the round trip through half resolution leaves of the variance, in pixels of blurSource. Only the largest
      //level can be below it (when the whole blur is), it then blurs by the round trip alone
      this.blur.radius = Math.sqrt(Math.max(sigma * sigma - baseVariance, 0)) * scale;
      this.blur.draw(blurSource, blurred, TiltShift.getRows(lo / slope, hi / slope, blurSource.height, rowPad));
      const drawn = this.blur.drawnRadius / scale;
      sigmas.push(Math.sqrt(drawn * drawn + baseVariance));
      material.setTexture('blur' + i, blurred);
    }

    this.sigmas.setValues(sigmas[0], sigmas[1], sigmas[2], sigmas[3]);
    material.setVector4('sigmas', this.sigmas);
    material.setFloat('sigma4', sigmas[4]);
    material.setFloat('slope', slope);
    this.renderer.blit(source, destination, material);
  }

  /**
   * The source at half resolution into this.half, by the tent of SmolGaussianBlur: an exact 2:1 rectangle of it from
   * the top left, an odd last row/column falls off. A texel of half is centered on the corner of its 2 x 2 pixels.
   */
  private drawHalf(source: RenderTexture): RenderTexture {
    const half = this.half;
    half.setSize(Math.max(1, Math.floor(source.width / 2)), Math.max(1, Math.floor(source.height / 2)));
    this.texelStep.setValues(1 / source.width, 1 / source.height);
    this.downMaterial.setVector2('texelStep', this.texelStep);
    this.uvScale.setValues(2 * half.width / source.width, 2 * half.height / source.height);
    this.downMaterial.setVector2('uvScale', this.uvScale);
    this.renderer.blit(source, half, this.downMaterial);
    return half;
  }

  //the rows where lo < |ndc y| <= hi, as [start, end) pairs padded by pad pixels
  private static getRows(lo: number, hi: number, height: number, pad: number): [number, number][] {
    lo = Math.min(lo, 1);
    hi = Math.min(hi, 1);
    //ndc y = 1 - 2 uv.y, with uv.y = (row + .5) / height: the top band and its mirror at the bottom
    const topStart = Math.max(0, Math.floor(height * (1 - hi) / 2) - pad);
    const topEnd = Math.ceil(height * (1 - lo) / 2) + pad;
    const bottomStart = height - topEnd;
    const bottomEnd = Math.min(height, height - topStart);
    if (topEnd >= bottomStart) return [[topStart, bottomEnd]];
    return [[topStart, topEnd], [bottomStart, bottomEnd]];
  }

  public destruct(): void {
    this.target.destruct();
    for (const rt of this.blurred) rt.destruct();
    this.half.destruct();
    this.blur.destruct();
    this.material.destruct();
    this.materialHalf.destruct();
    this.downMaterial.destruct();
  }
}
