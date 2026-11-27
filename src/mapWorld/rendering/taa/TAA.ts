import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import gBufferPacking from '../deferred/mapGBufferPacking.wgsl?raw';
import MapGBuffer from '../deferred/MapGBuffer';
import shader from './taa.wgsl?raw';

//static frames the history averages before it counts as converged: 2 cycles of the 16 jitter offsets, the history
//of the motion before it then weighs 1 / 34
export const STATIC_FRAMES = 32;
const JITTER_COUNT = 16;

/**
 * Temporal anti aliasing in place of MSAA, after irradianceCache/taa. The map renders into a 1 sample target with a
 * jittered camera, resolve() blends that into the history, see taa.wgsl.
 *
 * While the view is static the history becomes a running average of the jittered frames. After {@link STATIC_FRAMES}
 * of them it is {@link converged}: the map need not be drawn again, the history is the image to present.
 * setCamera also keeps the matrices without jitter, ReprojectedSSAO reprojects with them too.
 */
export default class TAA {
  public enabled!: ParamBool;
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private histories: RenderTexture[];
  private historyIndex = 0;
  private historyValid = false;
  private staticFrames = 0;
  private readonly resolution = new Vector2();
  private readonly canvasSize = new Vector2();
  public readonly viewProjectionNoJitter = new Matrix4x4();
  public readonly viewProjectionOldNoJitter = new Matrix4x4();
  public readonly cameraPosition = new Vector3();
  public readonly cameraPositionOld = new Vector3();
  private readonly inverseView = new Matrix4x4();
  private readonly inverseProjection = new Matrix4x4();

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(renderer, 'mapTAA', shader, {includes: [gBufferPacking]});
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    this.material.setSampler('linearSampler', renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    }));
    this.histories = [0, 1].map(() => new RenderTexture(renderer, 1, 1, {scaleToCanvas: true, depth: false, toScreen: false, formats: ['rgba16float']}));
  }

  public init(paramGroup: ParamGroup) {
    const params = {materials: [this.material], callback: () => this.restart()};
    const pg = paramGroup.addGroup('TAA');
    //off: no camera jitter, the history is a plain copy of the map
    this.enabled = pg.addParamBool('enabled', true, {callback: () => this.invalidate()});
    pg.addParamFloat('feedback', .9, 0, .99, params);
    pg.addParamFloat('clipGamma', 1, .5, 3, params);
    pg.collapsed = true;
  }

  /**
   * Every frame, after camera.update() with the jitter reset: keeps the camera without jitter for the reprojection,
   * then jitters it when jitter is true.
   */
  public setCamera(camera: Camera, jitter: boolean) {
    this.viewProjectionOldNoJitter.copy(this.viewProjectionNoJitter);
    this.viewProjectionNoJitter.copy(camera.viewProjection);
    this.cameraPositionOld.copy(this.cameraPosition);
    this.cameraPosition.copy(camera.worldPosition);
    if (jitter && this.enabled.value) camera.projection.jitter(this.getCanvasSize(), JITTER_COUNT);
  }

  /** Whether the history holds the converged static view, see {@link STATIC_FRAMES}. */
  public get converged(): boolean {
    return this.historyValid && this.staticFrames >= (this.enabled.value ? STATIC_FRAMES : 1) && this.sizeValid;
  }

  /** The image changes: converge again, from the history. */
  public restart() {
    this.staticFrames = 0;
  }

  /** The history doesn't belong to the next frame: the map was not resolved for a frame, or it all moved. */
  public invalidate() {
    this.historyValid = false;
    this.staticFrames = 0;
  }

  /** The latest resolved image. */
  public get output(): RenderTexture {
    return this.histories[this.historyIndex];
  }

  //renderer.size is the size of the last render target, e.g. the shadow map while the camera moves: the map and the
  //history are drawn at the canvas size
  private getCanvasSize(): Vector2 {
    return this.canvasSize.setValues(this.renderer.canvas.width, this.renderer.canvas.height);
  }

  //the camera of setCamera is the one of the frame before: a pixel reprojects onto itself, see CAMERA_STILL in taa.wgsl
  private get cameraStill(): boolean {
    const m = this.viewProjectionNoJitter.m, old = this.viewProjectionOldNoJitter.m;
    for (let i = 0; i < 16; i++) {
      if (m[i] !== old[i]) return false;
    }
    return true;
  }

  private get sizeValid(): boolean {
    const size = this.getCanvasSize();
    return size.x === this.resolution.x && size.y === this.resolution.y;
  }

  /**
   * Resolves source (drawn with the jittered camera) into the history, see {@link output}.
   * @param depth  the depth of the gBuffer (MapGBuffer.depthView), drawn with the same jitter
   * @param isStatic  the camera and the map didn't change since the last frame
   */
  public resolve(camera: Camera, source: RenderTexture, depth: GPUTextureView, isStatic: boolean) {
    if (!this.sizeValid) {
      this.resolution.copy(this.getCanvasSize());
      this.invalidate();
    }
    if (!isStatic) this.staticFrames = 0;
    //the history from the motion counts as 1 frame, so the first static frame doesn't drop to 1 aliased sample
    const staticWeight = isStatic ? 1 / (this.staticFrames + 2) : 0;

    this.historyIndex = 1 - this.historyIndex;
    const current = this.histories[this.historyIndex];

    //the jittered matrices, they reconstruct the positions of the depth samples
    Matrix4x4.invert(camera.viewMatrix, this.inverseView);
    Matrix4x4.invert(camera.projectionMatrix, this.inverseProjection);
    this.material.setMatrix('inverseView', this.inverseView);
    this.material.setMatrix('inverseProjection', this.inverseProjection);
    this.material.setVector3('cameraPosition', camera.worldPosition);
    this.material.setMatrix('viewProjectionNoJitter', this.viewProjectionNoJitter);
    this.material.setMatrix('viewProjectionOldNoJitter', this.viewProjectionOldNoJitter);
    this.material.setVector2('resolution', this.resolution);
    this.material.setBool('historyValid', this.historyValid && this.enabled.value);
    this.material.setConstant('CAMERA_STILL', this.cameraStill);
    this.material.setFloat('staticWeight', staticWeight);
    this.material.setBinding('depthTexture', depth);
    this.material.setVector2('depthParams', MapGBuffer.depthParams(camera));
    this.material.setTexture('source', source);
    this.material.setTexture('history', this.histories[1 - this.historyIndex]);
    this.renderer.blit(null, current, this.material);

    this.historyValid = true;
    if (isStatic) this.staticFrames++;
  }

  public destruct() {
    for (const history of this.histories) history.destruct();
    this.material.destruct();
  }
}
