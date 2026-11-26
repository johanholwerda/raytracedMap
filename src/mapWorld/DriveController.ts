import Renderer from 'webGPU-framework/renderer/render/Renderer';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import { clamp, lerp } from 'webGPU-framework/renderer/utils/MathUtils';
import type { PlayerCar } from './cars/CarTraffic';

const KEYS = {
  up: ['KeyW', 'ArrowUp'],
  down: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
  brake: ['Space'],
};

/**
 * The first ray hit on the buildings and trees, see CityScene.raycast: the distance along d (in its length), Infinity
 * for none.
 */
export type DriveRaycast = (ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, tMax: number) => number;

/**
 * Drives a car with the MapCamera following it, behind and above (cameraHeight). Entering, the car drops out of the sky
 * onto the road nearest to the point the orbit camera looked at (the pivot), the camera looking down at it on the way,
 * then drives: W/S (up/down) throttle and brake or reverse, A/D (left/right) steer, space the handbrake, a drag looks
 * around the car. CarTraffic draws the car (see car).
 *
 * The ground is the terrain height field, and the roofs: a ray down through the BVH of the buildings (see
 * CityScene.raycast) from a bit above the wheels, so a curb up to stepUp is climbed and a roof can be landed on and
 * driven off. Walls and tree trunks are rays ahead through the same BVH, from the bumper height: the car stops there and
 * bounces back a little, or scrapes along it at a slant.
 *
 * Hills: the wheels stay on the terrain going down a slope as long as it is less steep than maxSlope (only a roof edge
 * or a cliff makes the car fall), and a slope pulls at the car with slopePull of gravity, so the engine gets it up any
 * road.
 *
 * Internally in meters (speeds in m/s), the positions in world units. The camera stays a pivot camera, as for the plane:
 * the pivot is a point a bit ahead of the car, so the tile streaming and the shadows, which work around the pivot,
 * look ahead of it.
 */
export default class DriveController {
  // m/s, m/s^2
  public maxSpeed = 30;
  public maxReverseSpeed = 8;
  public acceleration = 8;
  public braking = 14;
  public handbrake = 20;
  // rolling resistance and air drag, m/s^2 and per m/s
  private readonly rolling = 0.8;
  private readonly drag = 0.012;
  // m/s^2, a bit more than earth, so a jump off a roof doesn't float
  public gravity = 14;
  // the part of gravity a slope pulls at the car with: arcade, so it climbs the steep streets of a hilly city
  public slopePull = 0.5;
  // radians: steeper down than this the car leaves the ground, less steep it follows it
  private readonly maxSlope = 0.9;
  private readonly wheelbase = 2.7;
  // the steer angle at standstill, radians, smaller at speed (see update)
  private readonly maxSteer = 0.6;
  // the highest step the wheels climb, m: higher than that is a wall
  private readonly stepUp = 0.6;
  // the ray ahead: its height over the ground, how far in front of the center of the car the bumper is and half the
  // car width, m. The size of the car CarTraffic draws (see car)
  private readonly bumperHeight = 0.6;
  private readonly bumperOffset = 2.25;
  private readonly halfWidth = 0.9;
  // m: the camera over the ground behind the car, and how far behind it. It looks at a point lookAhead in front of the
  // car, targetHeight over the ground
  public cameraHeight = 10;
  public cameraBehind = 14;
  private readonly lookAhead = 6;
  private readonly targetHeight = 1.5;
  // 1 / s: how fast the camera turns after the car
  private readonly cameraYawRate = 4;
  // the drop: the camera this far from the car (m), looking down this steep at the start (radians)
  private readonly dropCameraDistance = 45;
  private readonly dropCameraPitch = -1.2;
  // the drop out of the sky: from at most this high over the landing point (m), and how long it takes at the most (s)
  private readonly dropHeightMax = 400;
  private readonly dropTimeMax = 3;
  // m: the orbit camera after exiting starts this far from the car and zooms out to exitZoomDistance
  private readonly exitDistance = 30;
  private readonly exitZoomDistance = 300;
  // m, the near plane while driving: the orbit one is ~10 m
  public readonly nearPlane = 0.5;

  /** The height of the ground at world x, z (world units), null where it isn't known. */
  public ground: (x: number, z: number) => number | null = () => null;
  /** The ray hits on the buildings and trees, null without them. */
  public raycast: DriveRaycast | null = null;
  /**
   * The nearest point on a road within radius of x, z (world units) that accept takes, and the direction of the road
   * there; null without one (or without roads).
   */
  public road: ((x: number, z: number, radius: number, accept: (x: number, z: number) => boolean) =>
    { x: number; z: number; dx: number; dz: number } | null) | null = null;
  // m: how far from the pivot the drop looks for a road to land on
  private readonly roadSearchRadius = 250;

  private _active = false;
  // the wheels on the ground (world units): x, z and the height of the ground under them
  private x = 0;
  private y = 0;
  private z = 0;
  // m/s: vertical, forward along yaw
  private vy = 0;
  private speed = 0;
  private yaw = 0;
  // the camera: its yaw, following the car's, and how much it still looks down at the dropping car (1 to 0)
  private cameraYaw = 0;
  private dropLook = 0;
  // 0..1: the part of its distance the camera is from the target, less where a building is in between
  private cameraReach = 1;
  // radians, eased: the slope of the ground under the car
  private slope = 0;
  private grounded = false;
  // The drop, while it is not grounded for the first time: from dropStart to the landing point in dropTime, with its
  // own gravity (m/s^2). The way there is flown in the first DROP_ACROSS of the time, high up, the rest falls straight
  // down: so it lands on a roof instead of coming in through a wall below it.
  private dropping = false;
  private dropStartX = 0;
  private dropStartZ = 0;
  private dropTime = 0;
  private dropElapsed = 0;
  private dropGravity = 0;
  private static readonly DROP_ACROSS = 0.55;
  private landX = 0;
  private landZ = 0;
  // the yaw turns from the one of the camera to the one of the road over the way across
  private dropStartYaw = 0;
  private landYaw = 0;
  // the drop waited for the ground (a new place): the roads came with it, the landing point is looked for again
  private dropWaited = false;
  // the camera bobs on landing and over steps: a damped spring around 0 (m, m/s)
  private bob = 0;
  private bobVelocity = 0;
  // the eased input, -1..1
  private throttle = 0;
  private steer = 0;
  // the look of a drag, added to the pitch and yaw
  private lookPitch = 0;
  private lookYaw = 0;
  private keys = new Set<string>();
  private listeners?: AbortController;
  private orbitNear = 0;

  constructor(private renderer: Renderer, private camera: MapCamera) {
  }

  public get active(): boolean {
    return this._active;
  }

  /**
   * Drops out of the sky onto the road nearest to the pivot of the orbit camera (else onto the pivot), from where the
   * camera is, facing along the road the way it looks.
   */
  public enter(metersToNorm: number): void {
    if (this._active) return;
    this._active = true;
    const pivot = this.camera.pivot.position;
    const euler = this.camera.rotationController.eulerXY;
    const eye = this.camera.worldPosition;
    this.yaw = euler.y;
    this.cameraYaw = this.yaw;
    this.cameraReach = 1;
    this.speed = 0;
    this.throttle = 0;
    this.steer = 0;
    this.lookPitch = 0;
    this.lookYaw = 0;
    this.slope = 0;
    this.bob = 0;
    this.bobVelocity = 0;
    this.startDrop(eye.x, eye.y, eye.z, pivot.x, pivot.z, metersToNorm);

    this.orbitNear = this.camera.projection.near;
    this.camera.projection.near = this.nearPlane * metersToNorm;

    this.listeners = new AbortController();
    const signal = this.listeners.signal;
    window.addEventListener('keydown', (e) => {
      if (!this.isDriveKey(e.code) || this.isTyping(e)) return;
      this.keys.add(e.code);
      // no page scrolling with the arrows and space
      e.preventDefault();
    }, { signal });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code), { signal });
    window.addEventListener('blur', () => this.keys.clear(), { signal });
  }

  /**
   * Back to orbiting, around the car: close behind it, looking down a bit, the orbit zoom eases out from there.
   */
  public exit(metersToNorm: number): void {
    if (!this._active) return;
    this._active = false;
    this.listeners?.abort();
    this.listeners = undefined;
    this.keys.clear();
    this.camera.projection.near = this.orbitNear;
    const yaw = Math.atan2(Math.sin(this.cameraYaw), Math.cos(this.cameraYaw));
    const pitch = -0.35;
    this.camera.rotationController.resetSpeed();
    this.camera.rotationController.eulerXY = new Vector2(pitch, yaw);
    this.camera.pivot.setPositionValues(this.x, this.y, this.z);
    this.camera.pivot.euler = new Vector3(pitch, yaw, 0);
    this.camera.distance = this.exitDistance * metersToNorm;
    this.camera.zoomTargetDistance = clamp(this.exitZoomDistance * metersToNorm, 0.1, 2);
  }

  /** The projection was recentered on another place: drops again out of the sky onto its center. */
  public recenter(metersToNorm: number): void {
    const height = this.dropHeightMax * metersToNorm;
    this.speed = 0;
    this.startDrop(0, height, 0, 0, 0, metersToNorm);
  }

  /**
   * A drop from (x, y, z) onto the ground at landX, landZ, in a straight line seen from above: the horizontal speed and
   * the gravity of the drop make it take dropTime and land there. Held while the ground there isn't known yet.
   */
  private startDrop(x: number, y: number, z: number, landX: number, landZ: number, metersToNorm: number): void {
    this.dropStartYaw = this.landYaw = this.yaw;
    // onto a road in the open, along it
    const spot = this.road?.(landX, landZ, this.roadSearchRadius * metersToNorm, (rx, rz) => this.inTheOpen(rx, rz, metersToNorm));
    if (spot) {
      landX = spot.x;
      landZ = spot.z;
      // forward is (-sin(yaw), -cos(yaw)): the way along the road that is nearest to it
      const along = -Math.sin(this.yaw) * spot.dx - Math.cos(this.yaw) * spot.dz >= 0 ? 1 : -1;
      const roadYaw = Math.atan2(-spot.dx * along, -spot.dz * along);
      // the short way around
      this.landYaw = this.yaw + Math.atan2(Math.sin(roadYaw - this.yaw), Math.cos(roadYaw - this.yaw));
    }
    const groundY = this.ground(landX, landZ) ?? 0;
    const height = clamp((y - groundY) / metersToNorm, 5, this.dropHeightMax);
    this.x = this.dropStartX = x;
    this.y = groundY + height * metersToNorm;
    this.z = this.dropStartZ = z;
    this.landX = landX;
    this.landZ = landZ;
    this.dropTime = Math.min(Math.sqrt(2 * height / this.gravity), this.dropTimeMax);
    this.dropElapsed = 0;
    this.dropGravity = 2 * height / (this.dropTime * this.dropTime);
    this.vy = 0;
    this.dropping = true;
    this.grounded = false;
    this.dropLook = 1;
  }

  /** Instead of camera.update() while active. stick: the touch joystick, x right (steer), y up (throttle), added to the keys. */
  public update(metersToNorm: number, stick?: { x: number; y: number }): void {
    const dt = Math.min(this.renderer.time.dt, 0.05);
    const m = metersToNorm;
    const input = (keys: string[]) => keys.some(k => this.keys.has(k)) ? 1 : 0;
    const ease = Math.min(dt * 6, 1);
    this.throttle = lerp(this.throttle, clamp(input(KEYS.up) - input(KEYS.down) + (stick?.y ?? 0), -1, 1), ease);
    this.steer = lerp(this.steer, clamp(input(KEYS.right) - input(KEYS.left) + (stick?.x ?? 0), -1, 1), Math.min(dt * 4, 1));
    this.dragLook();

    if (this.dropping) this.updateDrop(dt, m);
    else this.updateDriving(dt, m, input(KEYS.brake) > 0);

    // the eye bob: a damped spring back to 0
    const stiffness = 120, damping = 14;
    this.bobVelocity += (-stiffness * this.bob - damping * this.bobVelocity) * dt;
    this.bob = clamp(this.bob + this.bobVelocity * dt, -0.8, 0.5);

    this.placeCamera(dt, m);
  }

  private updateDrop(dt: number, m: number): void {
    const landGround = this.ground(this.landX, this.landZ);
    // wait in the air until the ground under the landing point has come
    if (landGround === null) {
      this.dropWaited = true;
      return;
    }
    if (this.dropWaited) {
      this.dropWaited = false;
      this.startDrop(this.x, this.y, this.z, this.landX, this.landZ, m);
      return;
    }
    this.vy -= this.dropGravity * dt;
    this.dropElapsed += dt;
    const startX = this.x, startY = this.y, startZ = this.z;
    // smoothstep over the first DROP_ACROSS of the time
    const s = clamp(this.dropElapsed / (this.dropTime * DriveController.DROP_ACROSS), 0, 1);
    const across = s * s * (3 - 2 * s);
    this.x = lerp(this.dropStartX, this.landX, across);
    this.z = lerp(this.dropStartZ, this.landZ, across);
    this.yaw = lerp(this.dropStartYaw, this.landYaw, across);
    this.y += this.vy * m * dt;
    // something in the way (a tower on the way there): no further across, down from here
    if (this.raycast && (this.x !== startX || this.z !== startZ)) {
      const t = this.raycast(startX, startY, startZ, this.x - startX, this.y - startY, this.z - startZ, 1);
      if (t < 1) {
        this.x = this.dropStartX = this.landX = startX;
        this.z = this.dropStartZ = this.landZ = startZ;
        this.y = startY;
      }
    }
    const support = this.support(this.x, this.z, startY, m);
    if (support !== null && this.y <= support) this.land(support);

    // the camera looks down at the car on the way, and comes behind it over the last meters
    const landY = this.support(this.landX, this.landZ, this.y, m) ?? landGround;
    const height = (this.y - landY) / m;
    const k = clamp((height - 3) / 80, 0, 1);
    this.dropLook = k * k * (3 - 2 * k);
  }

  private land(support: number): void {
    this.y = support;
    // the eye dips with the impact, more the faster it came down (vy in m/s)
    this.bobVelocity += Math.max(-this.vy, 0) * 0.12 * (this.dropping ? 0.5 : 1);
    this.bobVelocity = -Math.abs(this.bobVelocity);
    this.vy = 0;
    this.grounded = true;
    this.dropping = false;
    this.dropLook = 0;
  }

  private updateDriving(dt: number, m: number, handbrake: boolean): void {
    // the speed: throttle, brake (the throttle against the motion), the resistances
    const sign = Math.sign(this.speed);
    let accel = 0;
    if (this.grounded) {
      const t = this.throttle;
      if (t * this.speed < 0) accel = t * this.braking;
      else accel = t * (t > 0 ? this.acceleration * (1 - Math.max(this.speed, 0) / this.maxSpeed) : this.acceleration * 0.6);
      accel -= sign * (this.rolling + this.drag * this.speed * this.speed);
      if (handbrake) accel -= sign * this.handbrake;
      // a slope pulls the car down it (a part of gravity, see slopePull); a standing car stays, as with its brake on
      if (sign !== 0 || Math.abs(t) > 0.05) accel -= Math.sin(this.slope) * 9.81 * this.slopePull;
    }
    let speed = this.speed + accel * dt;
    // the brakes and the resistances stop the car, they don't turn it around
    if (sign !== 0 && Math.sign(speed) !== sign && this.throttle * sign <= 0.05) speed = 0;
    this.speed = clamp(speed, -this.maxReverseSpeed, this.maxSpeed);

    // the bicycle model: the turn rate from the steer angle, which narrows with the speed
    if (this.grounded) {
      const steer = this.steer * this.maxSteer / (1 + Math.abs(this.speed) * 0.08);
      // steering right (positive) turns the yaw negative
      this.yaw -= this.speed / this.wheelbase * Math.tan(steer) * dt;
    }

    // the move, unless a wall is in the way
    let step = this.speed * dt;
    const fx = -Math.sin(this.yaw), fz = -Math.cos(this.yaw);
    if (Math.abs(step) > 1e-6) {
      const free = this.freeDistance(fx * Math.sign(step), fz * Math.sign(step), m);
      if (free < Math.abs(step)) {
        step = Math.sign(step) * Math.max(free, 0);
        this.hitWall(fx * Math.sign(this.speed), fz * Math.sign(this.speed), dt, m);
      }
    }
    const startY = this.y;
    this.x += fx * step * m;
    this.z += fz * step * m;

    const support = this.support(this.x, this.z, startY, m);
    if (this.grounded) {
      // down a slope the wheels follow the ground, down a roof edge or a cliff they don't
      const follow = this.stepUp + Math.abs(step) * Math.tan(this.maxSlope);
      // what the slope the car is on rises or falls over the step: more than that is a curb, the camera jolts
      const expected = Math.abs(step) * Math.abs(Math.tan(this.slope)) + 0.15;
      if (support === null) {
        // the ground isn't known here: stay on the last height
      } else if (support < this.y - follow * m) {
        // off an edge: falling, with the speed it had
        this.grounded = false;
        this.vy = 0;
      } else {
        if (Math.abs(support - this.y) > expected * m) this.bob -= (support - this.y) / m * 0.3;
        this.y = support;
      }
    }
    if (!this.grounded) {
      this.vy -= this.gravity * dt;
      this.y += this.vy * m * dt;
      if (support !== null && this.y <= support) this.land(support);
    }

    // the slope of the ground along the car, eased: from the ground at the front and the back wheels
    if (this.grounded) {
      const half = this.wheelbase * 0.5 * m;
      const front = this.support(this.x + fx * half, this.z + fz * half, this.y, m);
      const back = this.support(this.x - fx * half, this.z - fz * half, this.y, m);
      const target = front !== null && back !== null ? Math.atan2((front - back) / m, this.wheelbase) : this.slope;
      this.slope = lerp(this.slope, clamp(target, -this.maxSlope, this.maxSlope), Math.min(dt * 8, 1));
    }
  }

  /**
   * The ground the wheels stand on at x, z: the terrain or a roof (or a tree crown) under the point stepUp above
   * fromY, the highest of them. null where the terrain isn't known and nothing is hit.
   */
  private support(x: number, z: number, fromY: number, m: number): number | null {
    const terrain = this.ground(x, z);
    if (!this.raycast) return terrain;
    const top = fromY + this.stepUp * m;
    const t = this.raycast(x, top, z, 0, -1, 0, terrain !== null ? top - terrain : Infinity);
    const roof = t < Infinity ? top - t : null;
    if (roof === null) return terrain;
    return terrain === null ? roof : Math.max(terrain, roof);
  }

  /** Whether no building or tree stands over the ground at x, z (a road through a building, or a tunnel). */
  private inTheOpen(x: number, z: number, m: number): boolean {
    const terrain = this.ground(x, z);
    if (terrain === null || !this.raycast) return true;
    const top = terrain + 1000 * m;
    return this.raycast(x, top, z, 0, -1, 0, top - terrain - 1 * m) === Infinity;
  }

  /** How far the car can move along the direction (dx, dz, normalized) before its bumper hits a wall, m. */
  private freeDistance(dx: number, dz: number, m: number): number {
    if (!this.raycast) return Infinity;
    const y = this.y + this.bumperHeight * m;
    // to the left: (dz, -dx), the rays from the middle and the corners of the bumper, from the middle of the car
    const reach = this.bumperOffset + Math.abs(this.speed) * 0.05 + 0.5;
    let free = Infinity;
    for (const side of [-1, 0, 1]) {
      const sx = this.x + dz * side * this.halfWidth * m, sz = this.z - dx * side * this.halfWidth * m;
      const t = this.raycast(sx, y, sz, dx * m, 0, dz * m, reach);
      free = Math.min(free, t - this.bumperOffset);
    }
    return free;
  }

  /**
   * The car ran into a wall moving along (dx, dz). At a slant it scrapes along: it turns along the wall and keeps the
   * speed along it. Head on it bounces back a bit and the eye jolts.
   */
  private hitWall(dx: number, dz: number, dt: number, m: number): void {
    const tangent = this.wallTangent(dx, dz, m);
    // |cos| of the angle between the motion and the wall: 1 along it, 0 head on
    const along = tangent ? Math.abs(dx * tangent.x + dz * tangent.z) : 0;
    if (tangent && along > 0.25) {
      const sign = Math.sign(dx * tangent.x + dz * tangent.z) * Math.sign(this.speed || 1);
      // the yaw of a car that drives along the wall, the same way (reversing: its back along it)
      const wallYaw = Math.atan2(-tangent.x * sign, -tangent.z * sign);
      const turn = Math.atan2(Math.sin(wallYaw - this.yaw), Math.cos(wallYaw - this.yaw));
      this.yaw += turn * Math.min(dt * 12, 1);
      this.speed *= Math.pow(along, dt * 8);
      this.bobVelocity -= Math.min(Math.abs(this.speed) * (1 - along) * 0.05, 1);
      return;
    }
    this.bobVelocity -= Math.min(Math.abs(this.speed) * 0.15, 3);
    this.speed = -this.speed * 0.25;
  }

  /** The direction of the wall ahead along (dx, dz), from where the rays of the bumper corners hit it; null for no 2 hits. */
  private wallTangent(dx: number, dz: number, m: number): { x: number; z: number } | null {
    if (!this.raycast) return null;
    const y = this.y + this.bumperHeight * m;
    const reach = this.bumperOffset + 6;
    const hits: number[][] = [];
    for (const side of [-1, 0, 1]) {
      const sx = this.x + dz * side * this.halfWidth * m, sz = this.z - dx * side * this.halfWidth * m;
      const t = this.raycast(sx, y, sz, dx * m, 0, dz * m, reach);
      if (t < Infinity) hits.push([sx + dx * t * m, sz + dz * t * m]);
    }
    if (hits.length < 2) return null;
    const a = hits[0], b = hits[hits.length - 1];
    const tx = b[0] - a[0], tz = b[1] - a[1];
    const length = Math.hypot(tx, tz);
    return length > 1e-9 ? { x: tx / length, z: tz / length } : null;
  }

  /** The car for CarTraffic to draw, null while not driving. */
  public get car(): PlayerCar | null {
    if (!this._active) return null;
    return {
      x: this.x,
      z: this.z,
      hx: -Math.sin(this.yaw),
      hz: -Math.cos(this.yaw),
      ground: this.y,
      // the rise per world unit along the heading
      slope: Math.tan(this.slope),
    };
  }

  /**
   * The camera behind the car and above it, looking at a point ahead of it; looking down at it during the drop. It
   * turns after the car (cameraYaw), and looks up and down hills with half their slope.
   */
  private placeCamera(dt: number, m: number): void {
    this.cameraYaw += Math.atan2(Math.sin(this.yaw - this.cameraYaw), Math.cos(this.yaw - this.cameraYaw)) * Math.min(dt * this.cameraYawRate, 1);
    const yaw = this.cameraYaw + this.lookYaw;
    const rise = this.cameraHeight - this.targetHeight;
    const chasePitch = -Math.atan2(rise, this.cameraBehind + this.lookAhead) + this.slope * 0.5;
    const chaseDistance = Math.hypot(rise, this.cameraBehind + this.lookAhead);
    const look = this.dropLook;
    const pitch = clamp(lerp(chasePitch, this.dropCameraPitch, look) + this.lookPitch, -1.5, 0.3);
    let d = lerp(chaseDistance, this.dropCameraDistance, look) * m;
    // the target: ahead of the car (along the camera's yaw, so it doesn't swing with the steering), over the ground
    const ahead = this.lookAhead * (1 - look) * m;
    const tx = this.x - Math.sin(this.cameraYaw) * ahead;
    const ty = this.y + (this.targetHeight + this.bob) * m;
    const tz = this.z - Math.cos(this.cameraYaw) * ahead;
    // the camera is at pivot + (sin(yaw) cos(pitch), -sin(pitch), cos(yaw) cos(pitch)) * d. A building in between
    // pulls it in front of the building at once, it eases back out after
    const cx = Math.sin(yaw) * Math.cos(pitch) * d, cy = -Math.sin(pitch) * d, cz = Math.cos(yaw) * Math.cos(pitch) * d;
    const margin = 1.5 * m / d;
    const free = this.raycast ? this.raycast(tx, ty, tz, cx, cy, cz, 1 + margin) - margin : 1;
    const reach = clamp(free, 0.15, 1);
    this.cameraReach = reach < this.cameraReach ? reach : lerp(this.cameraReach, reach, Math.min(dt * 2, 1));
    d *= this.cameraReach;
    const pivot = this.camera.pivot;
    pivot.setPositionValues(tx, ty, tz);
    pivot.euler = new Vector3(pitch, yaw, 0);
    this.camera.distance = d;
    this.camera.projection.aspectRatio = this.renderer.aspectRatio;
    this.camera.checkChanged();
  }

  /** A drag looks around the car (no Ctrl needed), it springs back to behind it once released. */
  private dragLook(): void {
    const mouse = this.renderer.mouseListener;
    const dt = Math.min(this.renderer.time.dt, 0.05);
    if (!mouse.mouseDown) {
      const back = Math.min(dt * 2, 1);
      this.lookYaw = lerp(this.lookYaw, 0, back);
      this.lookPitch = lerp(this.lookPitch, 0, back);
      return;
    }
    const v = mouse.normalizedVelocity;
    const fov = this.camera.projection.FOV;
    const fovH = Math.atan2(this.renderer.aspectRatio * 0.5, 0.5 / Math.tan(fov * 0.5)) * 2;
    this.lookYaw = clamp(this.lookYaw - 3 * v.x * fovH, -Math.PI, Math.PI);
    this.lookPitch = clamp(this.lookPitch - 3 * v.y * fov, -1, 0.3);
  }

  private isDriveKey(code: string): boolean {
    return Object.values(KEYS).some(keys => keys.includes(code));
  }

  private isTyping(e: KeyboardEvent): boolean {
    const target = e.target as HTMLElement | null;
    return !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
  }
}
