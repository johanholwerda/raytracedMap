import Vector3 from '../../renderer/math/Vector3';
import Renderer from '../../renderer/render/Renderer';
import Vector2 from '../../renderer/math/Vector2';
import PivotCamera, {PivotCameraOptions} from './PivotCamera';
import Ray from '../../renderer/math/Ray';

export type MapCameraOptions = PivotCameraOptions;

export const defaultMapCameraOptions: MapCameraOptions = {
  fov: 1,
  nearPlane: 0.01,
  farPlane: 100,
  zPosition: 1,
  mouseInverse: true,
  clampEuler: true,
  scrollZoom: true,
};

export default class MapCamera extends PivotCamera {
  // Pan state: the ground point grabbed under the cursor stays under the cursor
  private panning: boolean = false;
  private panAnchor: Vector3 = new Vector3();
  private panAnchorValid: boolean = false;
  // height of the horizontal plane the anchor is on (the pivot's height when grabbed)
  private panPlaneY: number = 0;
  private panRay: Ray = new Ray();
  // Cursor position (screen space) where the current press started, plus whether
  // we've moved past the dead-zone yet. A click (a near-stationary tap, e.g. to
  // select a building) jitters by a pixel or two between press and release; the
  // dead-zone stops that jitter from panning the camera, which would otherwise
  // mark the camera changed and trigger a needless AO redraw.
  private panStartPosition: Vector2 = new Vector2();
  private panThresholdPassed: boolean = false;
  //prevent moving when clicking
  private readonly panDeadZone: number = 0.002;
  // Max ground movement per screen movement (NDC units), in camera distances. Looking straight
  // down a drag moves the ground ~0.5-1 distance per NDC unit; near the horizon keeping the
  // grabbed point under the cursor would move it far more, so the drag speed is capped there.
  private readonly panMaxSpeed: number = 1;
  private panPrevScreen: Vector2 = new Vector2();

  // Ctrl state
  private ctrlDown: boolean = false;
  private wasRotating: boolean = false;

  // Adjusts the view after the input moved it, before update() sees whether it changed (e.g. keeps it above the ground)
  public constrainView: (() => void) | null = null;

  // Pinch twist state (mobile)
  private pinchPrev0: Vector2 = new Vector2();
  private pinchPrev1: Vector2 = new Vector2();

  constructor(renderer: Renderer, opts: Partial<MapCameraOptions> = {}) {
    const options = {...defaultMapCameraOptions, ...opts};
    super(renderer, options);

    const signal = this.listenerController.signal;
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Control') this.ctrlDown = true;
    }, {signal});
    window.addEventListener('keyup', (e: KeyboardEvent) => {
      if (e.key === 'Control') this.ctrlDown = false;
    }, {signal});
    window.addEventListener('blur', () => {
      this.ctrlDown = false;
    }, {signal});

    // Enable pinch-to-zoom on mobile and
    this.enablePinch();
    // prevent browser/canvas from scaling on pinch (mobile) and ctrl+scroll (desktop)
    this.preventBrowserZoom();
  }

  /** Where the ray through the screen position (NDC) hits the horizontal plane at panPlaneY, or null. */
  private groundPoint(screenPosition: Vector2): Vector3 | null {
    const ray = this.getRay(screenPosition, this.panRay);
    const dy = ray.direction.y;
    if (Math.abs(dy) < 1e-6) return null;
    const t = (this.panPlaneY - ray.origin.y) / dy;
    // behind the camera, or so close to the horizon that a pixel would move the map by miles
    if (t <= 0 || t > this.distance * 50) return null;
    return new Vector3(ray.origin.x + ray.direction.x * t, this.panPlaneY, ray.origin.z + ray.direction.z * t);
  }

  /** Grabs the ground under the cursor; pan() then keeps it under the cursor. */
  private startPan(screenPosition: Vector2): void {
    this.panPlaneY = this.pivot.position.y;
    const hit = this.groundPoint(screenPosition);
    this.panAnchorValid = hit !== null;
    if (hit) this.panAnchor.copy(hit);
    this.panPrevScreen.copy(screenPosition);
  }

  private pan(screenPosition: Vector2): void {
    // the anchor was above the horizon: grab again once the cursor is on the ground
    if (!this.panAnchorValid) {
      this.startPan(screenPosition);
      return;
    }
    const hit = this.groundPoint(screenPosition);
    if (!hit) return;
    // moving the camera by the offset puts the anchor back under the cursor
    let dx = this.panAnchor.x - hit.x;
    let dz = this.panAnchor.z - hit.z;
    const screenMove = Vector2.distance(screenPosition, this.panPrevScreen);
    this.panPrevScreen.copy(screenPosition);
    const maxMove = this.panMaxSpeed * this.distance * screenMove;
    const move = Math.hypot(dx, dz);
    if (move > maxMove) {
      const s = maxMove / move;
      dx *= s;
      dz *= s;
      // the anchor slips: grab the ground that is under the cursor after this move,
      // so the map doesn't keep catching up after the cursor stops
      this.panAnchor.copy(new Vector3(hit.x + dx, this.panPlaneY, hit.z + dz));
    }
    this.pivot.translate(new Vector3(dx, 0, dz));
  }

  public update(): void {
    this.projection.aspectRatio = this.renderer.aspectRatio;

    if (this._zoomTargetDistance === 0) {
      this._zoomTargetDistance = this.distance;
    }

    const isRotating = this.ctrlDown && this.mouseListener.mouseDown;
    const isPanning = !this.ctrlDown && this.mouseListener.mouseDown && !this.pinching;

    if (isRotating) {
      this.wasRotating = true;
      this._rotationController.update(this);
    } else {
      if (this.wasRotating) {
        this._rotationController.resetSpeed();
        this.wasRotating = false;
      }

      if (isPanning) {
        const screenPosition = this.mouseListener.screenPosition;
        if (!this.panning) {
          // First frame of the press: remember where it started and wait for
          // the cursor to leave the dead-zone before treating it as a pan.
          this.panStartPosition = screenPosition.clone();
          this.panThresholdPassed = false;
          this.panning = true;
        } else if (!this.panThresholdPassed) {
          if (Vector2.distance(screenPosition, this.panStartPosition) > this.panDeadZone) {
            // Crossed the dead-zone: start panning from here so the camera
            // doesn't jump by the accumulated jitter.
            this.panThresholdPassed = true;
            this.startPan(screenPosition);
          }
        } else {
          this.pan(screenPosition);
        }
      } else {
        this.panning = false;
        this.panThresholdPassed = false;
        // Skip rotation easing while pinching so only zoom happens.
        if (!this.pinching) {
          this._rotationController.update(this);
        }
      }
    }

    this.updateZoom();
    this.applyControllerRotation();
    this.constrainView?.();

    this.checkChanged();
  }

  /**
   * Enable two-finger pinch-to-zoom on touch devices; twisting the fingers rotates around the pivot.
   * Also prevents the browser's default pinch-to-zoom gesture.
   */
  public enablePinch(): void {
    // Prevent browser default touch gestures (pinch-to-zoom, double-tap zoom).
    this.renderer.canvas.style.touchAction = 'none';
    super.enablePinch();
  }

  protected onPinchStart(): void {
    this.pinchPrev0.copy(this.pinchStart0);
    this.pinchPrev1.copy(this.pinchStart1);
  }

  protected onPinchMove(): void {
    // Two-finger twist rotates around the pivot's Y axis: take the change in
    // the angle of the line between the two fingers since the last move.
    const prevAngle = Math.atan2(this.pinchPrev1.y - this.pinchPrev0.y, this.pinchPrev1.x - this.pinchPrev0.x);
    const angle = Math.atan2(this.pinch1.y - this.pinch0.y, this.pinch1.x - this.pinch0.x);
    let deltaAngle = angle - prevAngle;
    // Wrap into [-PI, PI] so crossing the ±PI seam doesn't spin the camera.
    if (deltaAngle > Math.PI) deltaAngle -= Math.PI * 2;
    else if (deltaAngle < -Math.PI) deltaAngle += Math.PI * 2;
    this._rotationController.rotateBy(new Vector2(0, deltaAngle));

    this.pinchPrev0.copy(this.pinch0);
    this.pinchPrev1.copy(this.pinch1);
  }

  /**
   * Prevent the browser from zooming the page on ctrl+scroll (desktop)
   * and pinch-to-zoom (mobile, via touch-action CSS).
   * The camera's own scroll zoom still works.
   */
  public preventBrowserZoom(): void {
    const element = this.renderer.canvas;
    // Prevent browser pinch-to-zoom on touch devices.
    element.style.touchAction = 'none';
    // Prevent ctrl+scroll page zoom on desktop.
    element.addEventListener(
      'wheel',
      (e: WheelEvent) => {
        if (e.ctrlKey) {
          e.preventDefault();
        }
      },
      { passive: false, signal: this.listenerController.signal }
    );
  }
}
