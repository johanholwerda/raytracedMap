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
 * that view ray hits the ground (see ground). The height of the plane is over the ground under it. The tile streaming, the sun shadows and the cars, which all work around the
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
  public cruisePitch = -0.45;
  private readonly pitchRange = 0.12;
  // seconds to reach full speed after entering, so the view doesn't lurch forward
  private readonly accelerationTime = 1.5;
  // 1 / seconds: how fast a height outside min..max (entering from a high or low orbit) eases into it
  private readonly heightSettleRate = 0.8;

  /** The height of the ground at world x, z (world units), null where it isn't known: then the last one is kept. */
  public ground: (x: number, z: number) => number | null = () => null;
  // 1 / seconds: how fast the ground under the plane and under the pivot follow the terrain
  private readonly groundRate = 1.5;
  private readonly pivotGroundRate = 4;
  // the least height of the camera over the ground at the pivot, m: on a slope up ahead the view ray reaches it early
  private readonly pivotClearance = 20;

  private _active = false;
  // camera position in world units, the height over planeGround
  private x = 0;
  private z = 0;
  private height = 0;
  // the ground under the plane and under the pivot, eased (world units)
  private planeGround = 0;
  private pivotGround = 0;
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
  // dragging: the look up / down added to the cruise pitch (the turn goes into the yaw)
  private lookPitch = 0;
  // the pitch limits with the look: below -0.1 the view ray has to hit the ground (the pivot)
  private readonly pitchMin = -1.4;
  private readonly pitchMax = -0.1;
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
    this.planeGround = this.ground(this.x, this.z) ?? pivot.y;
    this.pivotGround = pivot.y;
    this.height = pivot.y - d * Math.sin(this.pitch) - this.planeGround;
    this.roll = this.exitRoll;
    this.exitRoll = 0;
    this.throttle = 0;
    this.climb = 0;
    this.turn = 0;
    this.lookPitch = 0;

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

  /**
   * Puts the flying plane somewhere, level and at full speed: x, z and height (over the ground) in world units, yaw in
   * radians (0 flies north, positive turns left). For the video mode of MapWorld, which starts every shot in flight.
   */
  public place(x: number, z: number, height: number, yaw: number): void {
    this.x = x;
    this.z = z;
    this.height = height;
    this.planeGround = this.ground(x, z) ?? 0;
    this.pivotGround = this.planeGround;
    this.yaw = yaw;
    this.pitch = this.cruisePitch;
    this.roll = 0;
    this.throttle = 1;
    this.climb = 0;
    this.turn = 0;
    this.lookPitch = 0;
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
    this.dragLook();
    // the nose up while climbing
    this.pitch = clamp(lerp(this.pitch, this.cruisePitch + this.lookPitch + climb * this.pitchRange, ease), this.pitchMin, this.pitchMax);

    this.throttle = Math.min(this.throttle + dt / this.accelerationTime, 1);
    // smoothstep: no sudden start
    const throttle = this.throttle * this.throttle * (3 - 2 * this.throttle);
    const step = this.speed * throttle * metersToNorm * dt;
    this.x -= Math.sin(this.yaw) * step;
    this.z -= Math.cos(this.yaw) * step;

    // the plane follows the terrain under it, eased so it flies over a hill instead of along every bump
    const planeTarget = this.ground(this.x, this.z);
    if (planeTarget !== null) this.planeGround += (planeTarget - this.planeGround) * (1 - Math.exp(-this.groundRate * dt));
    const y = this.planeGround + this.height;

    // the pivot is where the view ray hits the ground: the ground under the point of the ray at the height of the last
    // one, a few times. The camera stays where it is, the pivot only moves along the ray
    const sin = Math.sin(-this.pitch), cos = Math.cos(this.pitch);
    const pivotAt = (groundY: number) => {
      const horizontal = (y - groundY) / sin * cos;
      return [this.x - Math.sin(this.yaw) * horizontal, this.z - Math.cos(this.yaw) * horizontal];
    };
    let target = this.pivotGround;
    for (let i = 0; i < 3; i++) {
      const [px, pz] = pivotAt(target);
      target = this.ground(px, pz) ?? target;
    }
    this.pivotGround += (target - this.pivotGround) * (1 - Math.exp(-this.pivotGroundRate * dt));
    this.pivotGround = Math.min(this.pivotGround, y - this.pivotClearance * metersToNorm);
    const d = (y - this.pivotGround) / sin;
    const [px, pz] = pivotAt(this.pivotGround);
    const pivot = this.camera.pivot;
    pivot.setPositionValues(px, this.pivotGround, pz);
    // yaw, pitch, then the roll around the view axis
    pivot.euler = new Vector3(this.pitch, this.yaw, this.roll);
    this.camera.distance = d;

    this.camera.projection.aspectRatio = this.renderer.aspectRatio;
    this.camera.checkChanged();
  }

  /**
   * A drag (no Ctrl needed while flying) turns the plane and looks up / down, like Ctrl + drag rotates the orbit camera (StreetviewRotationController):
   * the drag turns the yaw directly, the look goes into lookPitch, which the pitch keeps.
   */
  private dragLook(): void {
    const mouse = this.renderer.mouseListener;
    if (!mouse.mouseDown) return;
    const v = mouse.normalizedVelocity;
    const fov = this.camera.projection.FOV;
    const fovH = Math.atan2(this.renderer.aspectRatio * 0.5, 0.5 / Math.tan(fov * 0.5)) * 2;
    // the orbit camera's speed multiplier and inverted mouse
    const yaw = -3 * v.x * fovH;
    const pitch = -3 * v.y * fov;
    this.yaw += yaw;
    // the look within the pitch limits (around the cruise pitch, without the climb)
    const look = clamp(this.lookPitch + pitch, this.pitchMin - this.cruisePitch + this.pitchRange, this.pitchMax - this.cruisePitch - this.pitchRange);
    this.pitch += look - this.lookPitch;
    this.lookPitch = look;
  }

  private isFlightKey(code: string): boolean {
    return KEYS.up.includes(code) || KEYS.down.includes(code) || KEYS.left.includes(code) || KEYS.right.includes(code);
  }

  private isTyping(e: KeyboardEvent): boolean {
    const target = e.target as HTMLElement | null;
    return !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
  }
}
