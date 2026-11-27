import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import TextureLoader from 'webGPU-framework/utils/loaders/TextureLoader';
import WebGPUPreLoader from 'webGPU-framework/renderer/core/WebGPUPreLoader';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import ParamInt from 'webGPU-framework/utils/params/ParamInt';
import {isMobile} from 'webGPU-framework/renderer/utils/Browser';
import r2 from 'webGPU-framework/shaders/r2.wgsl?raw';
import blueNoiseUrl from './blueNoise.png';
import gBufferPacking from '../deferred/mapGBufferPacking.wgsl?raw';
import ssaoShader from './reprojectedSSAO.wgsl?raw';
import compositeShader from './reprojectedSSAOComposite.wgsl?raw';
import TAA, { STATIC_FRAMES } from '../taa/TAA';
import MapGBuffer from '../deferred/MapGBuffer';

//(visibility, sample count, camera distance). The distance is the depth the history was made at, for the reprojection
//test, the sky mask and the fog of the composite. f16 holds the count exactly and the distance to ~0.05%; the running
//average loses updates smaller than half an ulp, which at 256 samples darkened near-white pixels by up to ~3.6 / 255
//(CPU sim), at 128 by ~0.2 / 255. Raise MAX_FRAMES and this has to go back to rgba32float
const ACCUM_FORMAT = 'rgba16float' as GPUTextureFormat;

//samples a pixel averages while the view is static and loaded
const MAX_FRAMES = 64;
//the fewest samples a pixel converges to under the tilt-shift blur
const COUNT_MIN = 8;
//while tiles stream in the history is capped at this, a new frame weighs 1 / 10, so the AO follows new geometry
const LOADING_FRAMES = 9;
//samples over which the AO of a pixel fades in, see reprojectedSSAOComposite.wgsl and getSSAO in deferredShading.wgsl
export const SSAO_FADE_COUNT = 4;
//the blur factors (blurSamples * sigma^2, see getCountLimit in reprojectedSSAO.wgsl) from which the rows trace a ray
//every 2nd and every 4th frame
const HALF_RATE_FACTOR = 2;
const QUARTER_RATE_FACTOR = 4;

/**
 * Temporal accumulation mode, picked each frame from the camera/tile state.
 *
 * - {@link Moving}: the camera moves: the history is reprojected and capped at movingFrames samples.
 * - {@link Loading}: camera static, tiles still streaming in: the history of each pixel is capped at
 *   {@link LOADING_FRAMES} samples, so the AO tracks new geometry.
 * - {@link Converging}: static and loaded: a running average up to {@link MAX_FRAMES} samples, then it stops. Fewer
 *   under the tilt-shift blur.
 */
export enum SSAOMode {
  Moving,
  Loading,
  Converging,
}

/**
 * Screen-space ambient occlusion for MapWorld, after mapOSM/ProgressiveSSAO: at most one ray per pixel per frame,
 * accumulated with reprojection, so the AO also shows while the camera moves. Under the tilt-shift blur the rows trace
 * only every 2nd or 4th frame, in bands that trace together, see reprojectedSSAO.wgsl.
 *
 * It reads the normal and the camera distance of the gBuffer of MapWorld (MapGBuffer), which is drawn on every frame
 * the map or the AO gets a new jitter.
 */
export default class ReprojectedSSAO {
  private renderer: Renderer;
  private accumA: RenderTexture;
  private accumB: RenderTexture;
  private pingPong: boolean = true;

  private ssaoMaterial: VertexFragmentShader;
  private compositeMaterial: VertexFragmentShader;
  //the composite without blending: draws the AO alone, for showOnly
  private aoOnlyMaterial: VertexFragmentShader;
  private blueNoise: TextureLoader;

  public enabled!: ParamBool;
  /** The steps of the ray march, each twice as far as the one before (scale): 1 is only the contact occlusion. */
  public rayMarchSteps!: ParamInt;
  /** 1 depth read a frame at a random radius over the range of the rayMarchSteps steps, instead of the march: lighter. */
  public randomStep!: ParamBool;
  //the AO alone on white, instead of multiplied over the scene, see showOnly
  private _showOnly = false;
  /** Strength of the AO in the composite; at 0 the SSAO is off, like {@link enabled} false. */
  public amount!: ParamFloat;
  /**
   * The AO only on the sky term, in the deferred shading (getSSAO in deferredShading.wgsl): with the lightmap the smaller
   * of the 2 (min, so the occlusion the texels hold isn't counted twice), not on the sun. Else composite() multiplies it
   * over the whole image. The host draws the map again while it accumulates then, and starts the TAA over on a new
   * {@link restartVersion}.
   */
  public skyOnly!: ParamBool;
  private movingFrames!: ParamFloat;
  //pixels the tilt-shift blur averages per sigma^2 (4 pi for white noise), see getCountLimit in reprojectedSSAO.wgsl
  private blurSamples!: ParamFloat;
  //under the tilt-shift blur the rows trace a ray only every 2nd or 4th frame, see getRateBand
  private blurRayRate!: ParamBool;
  private historyValid = false;
  //frames accumulated in Converging mode since the last other mode or reset
  private convergingFrames = 0;
  //for the noise, never reset
  private noiseFrame = 0;
  private _version = 0;
  private _restartVersion = 0;

  constructor(renderer: Renderer, preloader: WebGPUPreLoader) {
    this.renderer = renderer;

    this.accumA = new RenderTexture(renderer, 1, 1, {formats: [ACCUM_FORMAT], scaleToCanvas: true, toScreen: false});
    this.accumB = new RenderTexture(renderer, 1, 1, {formats: [ACCUM_FORMAT], scaleToCanvas: true, toScreen: false});

    this.ssaoMaterial = new VertexFragmentShader(renderer, 'reprojectedSSAO', ssaoShader, {includes: [r2, gBufferPacking]});
    this.ssaoMaterial.depthWriteEnabled = false;
    this.ssaoMaterial.depthCompare = 'always';

    this.compositeMaterial = new VertexFragmentShader(renderer, 'reprojectedSSAOComposite', compositeShader);
    this.compositeMaterial.depthWriteEnabled = false;
    this.compositeMaterial.depthCompare = 'always';
    this.compositeMaterial.setFloat('fadeCount', SSAO_FADE_COUNT);

    this.aoOnlyMaterial = new VertexFragmentShader(renderer, 'reprojectedSSAOOnly', compositeShader);
    this.aoOnlyMaterial.setConstant('SHOW_ONLY', true);
    this.aoOnlyMaterial.depthWriteEnabled = false;
    this.aoOnlyMaterial.depthCompare = 'always';
    this.aoOnlyMaterial.setFloat('fadeCount', SSAO_FADE_COUNT);

    this.blueNoise = new TextureLoader(renderer, preloader, blueNoiseUrl);
  }

  public init(paramGroup: ParamGroup): void {
    // Re-tuning any parameter invalidates the accumulated result.
    const reset = () => this.reset();
    const params = {materials: [this.ssaoMaterial], callback: reset};
    const pg = paramGroup.addGroup('SSAO');
    this.enabled = pg.addParamBool('enabled', !isMobile(), {callback: reset});
    //only the composite reads it: the history stays valid, but the composited result changes
    this.amount = pg.addParamFloat('amount', 1, 0, 1, {
      materials: [this.compositeMaterial, this.aoOnlyMaterial],
      callback: () => this.changed(),
    });
    this.skyOnly = pg.addParamBool('skyOnly', true, {callback: () => this.changed()});
    pg.addParamFloat('radius', 0.001, 0.0001, 0.01, params);
    pg.addParamFloat('bias', 0.0001, 0, 0.001, params);
    pg.addParamFloat('scale', 2, 1.1, 2, params);
    pg.addParamFloat('distanceMax', 0.01, 0, 0.1, params);
    pg.addParamFloat('skyLight', 0, 0, 1, params);
    this.rayMarchSteps = pg.addParamInt('rayMarchSteps', 6, 1, 16, params);
    this.randomStep = pg.addParamBool('randomStep', false, params);
    //the samples the history is capped at while the camera moves: more is less noise, but more smearing
    this.movingFrames = pg.addParamFloat('movingFrames', 16, 1, 64);
    pg.addParamFloat('depthTolerance', 0.002, 0, 0.05, {materials: [this.ssaoMaterial]});
    //0: as many samples under the tilt-shift blur as without it
    this.blurSamples = pg.addParamFloat('blurSamples', 3, 0, 12, {callback: reset});
    this.blurRayRate = pg.addParamBool('blurRayRate', false, {callback: reset});
    pg.collapsed = true;

    this.ssaoMaterial.setTexture('blueNoise', this.blueNoise);
  }

  /** Visibility texture holding the latest accumulated result. */
  private get output(): RenderTexture {
    return this.pingPong ? this.accumA : this.accumB;
  }

  /**
   * The accumulated (visibility, sample count, camera distance, 0) for the shading to read ({@link skyOnly}), valid after
   * update() of this frame. It ping-pongs: bind it every frame.
   */
  public get result(): RenderTexture {
    return this.output;
  }

  /** Previous accumulation, read as history while writing {@link output}. */
  private get input(): RenderTexture {
    return this.pingPong ? this.accumB : this.accumA;
  }

  /** Debug view, set by the host (MapWorld's drawMode): the AO alone on white, instead of multiplied over the scene. */
  public get showOnly(): boolean {
    return this._showOnly;
  }

  public set showOnly(value: boolean) {
    if (value === this._showOnly) return;
    this._showOnly = value;
    this.reset();
  }

  public get active(): boolean {
    if (!this.enabled) return true;
    return this.enabled.value && this.amount.value > 0;
  }

  /** Changes whenever what composite() draws may change: on a reset and on every accumulated frame. */
  public get version(): number {
    return this._version;
  }

  /**
   * Changes when the AO starts over or its look changes (a reset, amount, skyOnly), not on an accumulated frame: with
   * {@link skyOnly} the TAA history of the shading then holds another AO, the host restarts the TAA.
   */
  public get restartVersion(): number {
    return this._restartVersion;
  }

  /** The AO is in the deferred shading ({@link skyOnly}) instead of composited over the image. */
  public get shadesSky(): boolean {
    return this.skyOnly?.value === true && !this._showOnly;
  }

  private changed(): void {
    this._version++;
    this._restartVersion++;
  }

  /**
   * Converging, with the last STATIC_FRAMES (TAA) of its frames to go. With {@link skyOnly} the host starts the static
   * average of the TAA here: earlier it would average the noisy early running averages of the AO into the image too
   * (about half the samples), now it ends on the frame the AO converges.
   */
  public get settling(): boolean {
    return this.historyValid && this.convergingFrames >= MAX_FRAMES - STATIC_FRAMES;
  }

  /** Whether update() would accumulate a frame in this mode. */
  public accumulating(mode: SSAOMode): boolean {
    return this.active && (mode !== SSAOMode.Converging || !this.historyValid || this.convergingFrames < MAX_FRAMES);
  }

  /** Discard the accumulated result. */
  public reset(): void {
    this.historyValid = false;
    this.convergingFrames = 0;
    this.changed();
  }

  /**
   * Accumulates one frame into the history, after the gBuffer was drawn this frame. Call it every frame: a frame without it
   * loses the history, the reprojection needs the camera of the frame before.
   * @param taa  holds the matrices without jitter of this frame and the last
   * @param blur  the tilt-shift blur (TiltShift.getBlur), a pixel under it converges to fewer samples
   */
  public update(camera: Camera, taa: TAA, mode: SSAOMode, gBuffer: MapGBuffer, blur: { slope: number; max: number }): void {
    if (!this.active) {
      this.historyValid = false;
      return;
    }
    // Canvas resize invalidates the G-buffer and accumulation textures.
    if (this.output.width !== this.renderer.width || this.output.height !== this.renderer.height) {
      this.reset();
    }
    if (!this.accumulating(mode)) return;

    let countMax: number;
    switch (mode) {
      case SSAOMode.Moving:
        countMax = this.movingFrames.value;
        break;
      case SSAOMode.Loading:
        countMax = LOADING_FRAMES;
        break;
      default:
        countMax = MAX_FRAMES;
        break;
    }
    if (mode === SSAOMode.Converging) {
      this.convergingFrames++;
    } else {
      this.convergingFrames = 0;
    }

    const m = this.ssaoMaterial;
    m.setMatrix('viewProjection', camera.viewProjection);
    m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
    m.setMatrix('inverseView', camera.viewMatrix.inverse);
    m.setVector3('cameraPosition', camera.worldPosition);
    m.setMatrix('viewProjectionNoJitter', taa.viewProjectionNoJitter);
    m.setMatrix('viewProjectionOldNoJitter', taa.viewProjectionOldNoJitter);
    m.setVector3('cameraPositionOld', taa.cameraPositionOld);
    m.setVector2('resolution', gBuffer.target.size);
    m.setFloat('frame', this.noiseFrame++);
    m.setBool('historyValid', this.historyValid);
    m.setBool('cameraStatic', mode !== SSAOMode.Moving);
    m.setFloat('countMax', countMax);
    m.setFloat('countLimit', MAX_FRAMES);
    m.setBool('converging', mode === SSAOMode.Converging);
    m.setFloat('halfRateFrom', this.getRateBand(blur, HALF_RATE_FACTOR));
    m.setFloat('quarterRateFrom', this.getRateBand(blur, QUARTER_RATE_FACTOR));
    m.setFloat('blurSlope', blur.slope);
    m.setFloat('blurMax', blur.max);
    m.setFloat('blurSamples', this.blurSamples.value);
    m.setFloat('countMin', COUNT_MIN);
    m.setTexture('gBuffer', gBuffer.view);
    m.setTexture('gBufferDepth', gBuffer.depthView);
    m.setVector2('depthParams', MapGBuffer.depthParams(camera));

    // blit binds `input` as `texture` (the history); the shader reads it and writes the new accumulation into `output`.
    this.pingPong = !this.pingPong;
    this.renderer.blit(this.input, this.output, m);
    this.historyValid = true;
    this._version++;
  }

  /**
   * The |ndc y| from which the blur factor blurSamples * sigma^2 reaches factor, 2 (off screen) when it never does.
   * Matches getCountLimit in reprojectedSSAO.wgsl: the count limit there is at most countLimit / factor from here on, so
   * a row tracing 1 in factor frames reaches it in the same MAX_FRAMES frames as the sharp rows.
   */
  private getRateBand(blur: { slope: number; max: number }, factor: number): number {
    const samples = this.blurSamples.value;
    if (!this.blurRayRate.value || samples <= 0 || blur.slope <= 0) return 2;
    const sigma = Math.sqrt(factor / samples);
    return sigma > blur.max ? 2 : sigma / blur.slope;
  }

  /**
   * Copies scene to destination with the accumulated visibility multiplied over it, or with {@link showOnly} draws the
   * visibility alone. False when there is no AO to draw (inactive, or no history yet): the caller copies the scene then.
   * @param distanceFog  fog coefficient matching shading.wgsl, the AO fades out with it
   * @param viewDistanceMax  where that fog is 1, see MapWorld.viewDistanceMax
   * @param scene  the scene the AO darkens, the size of the AO (the canvas)
   * @param destination  null is the screen
   */
  public composite(distanceFog: number, viewDistanceMax: number, scene: RenderTexture, destination: RenderTexture | null = null): boolean {
    if (!this.active || !this.historyValid) return false;
    const material = this._showOnly ? this.aoOnlyMaterial : this.compositeMaterial;
    material.setFloat('distanceFog', distanceFog);
    material.setFloat('viewDistanceMax', viewDistanceMax);
    material.setTexture('scene', scene);
    this.renderer.blit(this.output, destination, material, true);
    return true;
  }

  public destruct(): void {
    this.accumA.destruct();
    this.accumB.destruct();
  }
}
