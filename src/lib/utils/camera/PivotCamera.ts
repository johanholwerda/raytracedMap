import Camera from '../../renderer/camera/Camera';
import Quaternion from '../../renderer/math/Quaternion';
import Transform from '../../renderer/scene/Transform';
import Vector3 from '../../renderer/math/Vector3';
import MouseListener from '../../renderer/input/MouseListener';
import Renderer from '../../renderer/render/Renderer';
import StreetviewRotationController from '../input/StreetviewRotationController';
import {clamp, lerp} from '../../renderer/utils/MathUtils';
import Vector2 from '../../renderer/math/Vector2';

export type PivotCameraOptions = {
  fov: number;
  nearPlane: number;
  farPlane: number;
  zPosition: number;
  mouseInverse: boolean;
  clampEuler: boolean;
  scrollZoom: boolean;
};

/*
 a camera at a distance from a pivot it rotates around, shared by OrbitCamera and MapCamera: the rotation from a
 StreetviewRotationController, zooming to a target distance (scroll wheel and pinch) and the euler and distance setters.
 the subclasses decide what update() does with the input
 */
export default class PivotCamera extends Camera {
  public pivot: Transform = new Transform();
  protected _rotationController: StreetviewRotationController;
  protected renderer: Renderer;
  protected mouseListener: MouseListener;
  protected _euler: Vector3 = new Vector3();
  protected _zoomTargetDistance: number = 0;
  protected distanceLimit: Vector2 = new Vector2(0, 1e20);

  protected pinching: boolean = false;
  protected pinchStart0: Vector2 = new Vector2();
  protected pinchStart1: Vector2 = new Vector2();
  protected pinch0: Vector2 = new Vector2();
  protected pinch1: Vector2 = new Vector2();
  protected zoomTargetDistanceStart: number = 1;

  constructor(renderer: Renderer, options: PivotCameraOptions) {
    super(options);

    this.renderer = renderer;
    this.pivot.setPositionValues(0, 0, 0);
    this.view.parent = this.pivot;
    this.distance = options.zPosition;

    this.mouseListener = renderer.mouseListener;
    if (options.scrollZoom) {
      this.mouseListener.addWheelEvent((e: number) => this.scrollZoom(e));
    }

    this._rotationController = new StreetviewRotationController(this.renderer, options.mouseInverse, options.clampEuler);
  }

  public get zooming(): boolean {
    return this.distance != this._zoomTargetDistance;
  }

  public get rotationController(): StreetviewRotationController {
    return this._rotationController;
  }

  public get eulerX(): number {
    return this._euler.x;
  }

  /**
   * @param value: in radians
   */
  public set eulerX(value: number) {
    this._euler.x = value;
    this._rotationController.setEulerValues(this._euler.x, this._euler.y);
    this.pivot.euler = this._euler;
  }

  public get eulerY(): number {
    return this._euler.y;
  }

  /**
   * @param value: in radians
   */
  public set eulerY(value: number) {
    this._euler.y = value;
    this._rotationController.setEulerValues(this._euler.x, this._euler.y);
    this.pivot.euler = this._euler;
  }

  public get euler(): Vector3 {
    return this._euler.clone();
  }

  public set rotation(rotation: Quaternion) {
    this._rotationController.rotation = rotation;
    this.pivot.euler = this._euler;
  }

  public get distance(): number {
    return this.view.position.z;
  }

  public set distance(value: number) {
    this._zoomTargetDistance = value;
    this.view.setPositionValues(0, 0, value);
  }

  public set zoomTargetDistance(value: number) {
    this._zoomTargetDistance = value;
  }

  public get zoomTargetDistance() {
    return this._zoomTargetDistance;
  }

  public lookAt(pos: Vector3, target: Vector3, up: Vector3 = Vector3.UP): void {
    this.pivot.lookAt(pos, target, up);
    this.pivot.position = target;
    this.view.setPositionValues(0, 0, Vector3.distance(pos, target));
    this._zoomTargetDistance = this.distance;
  }

  public scrollZoom(e: number): void {
    let z: number = this._zoomTargetDistance;
    if (e > 0) {
      z /= 1.05;
    } else {
      z *= 1.05;
    }
    z = clamp(z, this.distanceLimit.x, this.distanceLimit.y);
    this._zoomTargetDistance = z;
  }

  public setDistanceLimit(min: number, max: number) {
    this.distanceLimit.setValues(min, max);
  }

  //moves the camera towards the zoom target distance
  protected updateZoom() {
    if (this.distance != this._zoomTargetDistance) {
      const speed = Math.min(this.renderer.time.dt * 6, 1);
      let d = lerp(this.distance, this._zoomTargetDistance, speed);
      //snap when close, for the zooming getter to be useful
      const ratio = this.distance / this._zoomTargetDistance;
      if (ratio > 0.999 && ratio < 1.001) {
        d = this._zoomTargetDistance;
      }
      this.view.setPositionValues(0, 0, d);
    }
  }

  //rotates the pivot to the rotation of the rotation controller
  protected applyControllerRotation() {
    const mouseRotation = this._rotationController.eulerXY;
    this._euler.x = mouseRotation.x;
    this._euler.y = mouseRotation.y;
    this.pivot.euler = this._euler;
  }

  /**
   * Two-finger pinch-to-zoom on touch devices.
   */
  public enablePinch(): void {
    const element = this.renderer.canvas;
    const signal = this.listenerController.signal;

    element.addEventListener(
      'touchstart',
      (e: TouchEvent) => {
        if (e.touches.length === 2) {
          e.preventDefault();
          this.pinching = true;
          this.pinchStart0.setValues(e.touches[0].pageX, e.touches[0].pageY);
          this.pinchStart1.setValues(e.touches[1].pageX, e.touches[1].pageY);
          this.zoomTargetDistanceStart = this._zoomTargetDistance;
          this.onPinchStart();
        }
      },
      {passive: false, signal}
    );

    element.addEventListener(
      'touchmove',
      (e: TouchEvent) => {
        if (this.pinching) {
          e.preventDefault();
          this.pinch0.setValues(e.touches[0].pageX, e.touches[0].pageY);
          this.pinch1.setValues(e.touches[1].pageX, e.touches[1].pageY);
          const d0 = Vector2.distance(this.pinchStart0, this.pinchStart1);
          const d1 = Vector2.distance(this.pinch0, this.pinch1);
          this._zoomTargetDistance = this.zoomTargetDistanceStart * (d0 / d1);
          this.onPinchMove();
        }
      },
      {passive: false, signal}
    );

    element.addEventListener(
      'touchend',
      () => {
        this.pinching = false;
      },
      {passive: false, signal}
    );
  }

  //for a subclass that does more with the two fingers than zooming
  protected onPinchStart(): void {
  }

  protected onPinchMove(): void {
  }
}
