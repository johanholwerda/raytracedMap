import Renderer from 'webGPU-framework/renderer/render/Renderer';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import { clamp, lerp } from 'webGPU-framework/renderer/utils/MathUtils';

const KEYS = {
  up: ['KeyW', 'ArrowUp'],
  down: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
};

/**
 * Flies the MapCamera like an airplane: it moves forward at a constant speed, W/S (or up/down) climbs and descends
 * between a min and max height, A/D (or left/right) banks and turns.
 *
 * It keeps driving the pivot camera instead of replacing it: the camera looks down at a slant and the pivot is where
 * that view ray hits the ground (y = 0). The tile streaming, the sun shadows and the cars, which all work around the
 * pivot at the camera distance, keep working as in orbit mode.
 */
export default class AirplaneController {
  // heights in meters
  public minHeight = 120;
  public maxHeight = 800;
  // meters per second
  public speed = 140;
  public climbSpeed = 120;
  // meters below the max height (above the min) where the climb (descent) starts to slow down
  public limitEaseBand = 150;
  // radians per second at full bank
  public turnSpeed = 0.7;
  public maxRoll = 0.45;
  // the look down angle (negative), level flight and while climbing / descending
  private readonly cruisePitch = -0.45;
  private readonly pitchRange = 0.12;
  // seconds to reach full speed after entering, so the view doesn't lurch forward
  private readonly accelerationTime = 1.5;
  // 1 / seconds: how fast a height outside min..max (entering from a high or low orbit) eases into it
  private readonly heightSettleRate = 0.8;

  private _active = false;
  // camera position in world units
  private x = 0;
  private z = 0;
  private height = 0;
  private yaw = 0;
  private pitch = this.cruisePitch;
  private roll = 0;
  // 0..1, ramps up after entering
  private throttle = 0;
  // the bank left over after exiting, eased out on the orbit camera by settle()
  private exitRoll = 0;
  // the eased input, -1..1
  private climb = 0;
  private turn = 0;
  private keys = new Set<string>();
  private listeners?: AbortController;

  constructor(private renderer: Renderer, private camera: MapCamera) {
  }

  public get active(): boolean {
    return this._active;
  }

  /**
   * Starts flying from the current orbit view, in its direction. It starts at the exact orbit camera pose (height and
   * pitch), update() eases those into the flight range, so there is no jump.
   */
  public enter(metersToNorm: number): void {
    if (this._active) return;
    this._active = true;
    const pivot = this.camera.pivot.position;
    const euler = this.camera.rotationController.eulerXY;
    const d = this.camera.distance;
    this.yaw = euler.y;
    // the orbit camera position
    const horizontal = d * Math.cos(euler.x);
    this.x = pivot.x + Math.sin(this.yaw) * horizontal;
    this.z = pivot.z + Math.cos(this.yaw) * horizontal;
    // the pitch has to look down for the view ray to hit the ground (the orbit limit keeps it below -0.2)
    this.pitch = Math.min(euler.x, -0.05);
    this.height = pivot.y - d * Math.sin(this.pitch);
    this.roll = this.exitRoll;
    this.exitRoll = 0;
    this.throttle = 0;
    this.climb = 0;
    this.turn = 0;

    this.listeners = new AbortController();
    const signal = this.listeners.signal;
    window.addEventListener('keydown', (e) => {
      if (!this.isFlightKey(e.code) || this.isTyping(e)) return;
      this.keys.add(e.code);
      // no page scrolling with the arrows
      e.preventDefault();
    }, { signal });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code), { signal });
    window.addEventListener('blur', () => this.keys.clear(), { signal });
  }

  /** Back to orbiting the ground point the plane looked at, from the same camera pose. */
  public exit(): void {
    if (!this._active) return;
    this._active = false;
    this.listeners?.abort();
    this.listeners = undefined;
    this.keys.clear();
    // wrapped into -PI..PI for the orbit rotation controller
    const yaw = Math.atan2(Math.sin(this.yaw), Math.cos(this.yaw));
    this.camera.rotationController.resetSpeed();
    this.camera.rotationController.eulerXY = new Vector2(this.pitch, yaw);
    // keep the distance, the orbit zoom eases it into its limits
    const d = this.camera.distance;
    this.camera.distance = d;
    this.camera.zoomTargetDistance = clamp(d, 0.1, 2);
    this.exitRoll = this.roll;
    this.roll = 0;
  }

  /** After camera.update() while not active: eases out the bank the plane had when exiting. */
  public settle(): void {
    if (this.exitRoll === 0) return;
    const dt = Math.min(this.renderer.time.dt, 0.1);
    this.exitRoll = lerp(this.exitRoll, 0, Math.min(dt * 3, 1));
    if (Math.abs(this.exitRoll) < 1e-4) this.exitRoll = 0;
    const euler = this.camera.rotationController.eulerXY;
    this.camera.pivot.euler = new Vector3(euler.x, euler.y, this.exitRoll);
    this.camera.checkChanged();
  }

  /** The projection was recentered on the origin: the pivot moves there, the plane with it. */
  public recenter(): void {
    const pivot = this.camera.pivot.position;
    this.x -= pivot.x;
    this.z -= pivot.z;
  }

  /** Instead of camera.update() while active. stick: the touch joystick, x right (turn), y up (climb), added to the keys. */
  public update(metersToNorm: number, stick?: { x: number; y: number }): void {
    const dt = Math.min(this.renderer.time.dt, 0.1);
    const input = (keys: string[]) => keys.some(k => this.keys.has(k)) ? 1 : 0;
    const ease = Math.min(dt * 3, 1);
    this.climb = lerp(this.climb, clamp(input(KEYS.up) - input(KEYS.down) + (stick?.y ?? 0), -1, 1), ease);
    this.turn = lerp(this.turn, clamp(input(KEYS.right) - input(KEYS.left) + (stick?.x ?? 0), -1, 1), ease);

    const minHeight = this.minHeight * metersToNorm;
    const maxHeight = this.maxHeight * metersToNorm;
    // The climb rate fades out over the last limitEaseBand meters to the max (descending: the min): eased in, then
    // linear to 0 at the limit, so the height approaches it exponentially and the nose levels out with it.
    const room = this.climb > 0 ? maxHeight - this.height : this.height - minHeight;
    const fade = Math.sin(Math.PI * 0.5 * clamp(room / (this.limitEaseBand * metersToNorm), 0, 1));
    const climb = this.climb * fade;
    let height = this.height + climb * this.climbSpeed * metersToNorm * dt;
    // outside the range (just entered from a high or low orbit): ease into it instead of clamping
    const settle = 1 - Math.exp(-this.heightSettleRate * dt);
    if (height > maxHeight) height = Math.max(maxHeight, lerp(height, maxHeight, settle));
    else if (height < minHeight) height = Math.min(minHeight, lerp(height, minHeight, settle));
    this.height = height;

    // bank right is a negative roll (clockwise), and turns right: a negative yaw
    this.roll = lerp(this.roll, -this.turn * this.maxRoll, ease);
    this.yaw += this.roll / this.maxRoll * this.turnSpeed * dt;
    // the nose up while climbing
    this.pitch = lerp(this.pitch, this.cruisePitch + climb * this.pitchRange, ease);

    this.throttle = Math.min(this.throttle + dt / this.accelerationTime, 1);
    // smoothstep: no sudden start
    const throttle = this.throttle * this.throttle * (3 - 2 * this.throttle);
    const step = this.speed * throttle * metersToNorm * dt;
    this.x -= Math.sin(this.yaw) * step;
    this.z -= Math.cos(this.yaw) * step;

    // the pivot is where the view ray hits the ground
    const d = this.height / Math.sin(-this.pitch);
    const horizontal = d * Math.cos(this.pitch);
    const pivot = this.camera.pivot;
    pivot.setPositionValues(this.x - Math.sin(this.yaw) * horizontal, 0, this.z - Math.cos(this.yaw) * horizontal);
    // yaw, pitch, then the roll around the view axis
    pivot.euler = new Vector3(this.pitch, this.yaw, this.roll);
    this.camera.distance = d;

    this.camera.projection.aspectRatio = this.renderer.aspectRatio;
    this.camera.checkChanged();
  }

  private isFlightKey(code: string): boolean {
    return KEYS.up.includes(code) || KEYS.down.includes(code) || KEYS.left.includes(code) || KEYS.right.includes(code);
  }

  private isTyping(e: KeyboardEvent): boolean {
    const target = e.target as HTMLElement | null;
    return !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
  }
}
