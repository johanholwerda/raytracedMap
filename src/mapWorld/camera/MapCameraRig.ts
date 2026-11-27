import type Renderer from 'webGPU-framework/renderer/render/Renderer';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import { clamp } from 'webGPU-framework/renderer/utils/MathUtils';
import type MapProjection from '../map/MapProjection';
import type TerrainHeightField from '../map/TerrainHeightField';
import type { ShadowMapAhead } from '../rendering/sun/ShadowMap';
import type { UrlCamera } from '../MapUrlSync';
import AirplaneController from './AirplaneController';
import OrbitKeyRotation from './OrbitKeyRotation';

// 1 / seconds: how fast the orbit pivot eases to the ground under it, and how close (m) it snaps onto it, see followGround
const PIVOT_GROUND_RATE = 6;
const PIVOT_GROUND_SNAP_M = .01;
// the orbit camera: the pitch (eulerX, radians, always a bit down) and the distance to the pivot (world units)
const CAMERA_PITCH_MIN = -1.4;
const CAMERA_PITCH_MAX = -.2;
const CAMERA_DISTANCE_MIN = .1;
const CAMERA_DISTANCE_MAX = 2;
// what it starts with: the pitch and the distance (the yaw is 0), and its lens
const CAMERA_PITCH_START = -.7;
const CAMERA_DISTANCE_START = 1;
export const CAMERA_FOV = 1;
export const CAMERA_NEAR = .01;
// The orbit camera stays this far (m) above the ground under it and around it, see keepAboveGround. At least
// CAMERA_GROUND_CLEARANCE_NEAR near planes (world units): the corners of the near plane reach ~1.3 of it from the eye.
const CAMERA_GROUND_CLEARANCE_M = 25;
const CAMERA_GROUND_CLEARANCE_NEAR = 2;

export type CameraView = { pitch: number; yaw: number; distance: number };

/**
 * The pitch, the yaw and the distance the orbit camera starts with: the ones of the URL where it has them. Without a
 * camera: the tiles of the first view are asked for before the GPU is there, see MapManager.startTiles.
 */
export function startCameraView(urlCamera: UrlCamera | null, metersToNorm: number): CameraView {
  const toRad = Math.PI / 180;
  const pitch = urlCamera && Number.isFinite(urlCamera.pitch) ? clamp(urlCamera.pitch * toRad, CAMERA_PITCH_MIN, CAMERA_PITCH_MAX) : CAMERA_PITCH_START;
  const yaw = urlCamera && Number.isFinite(urlCamera.yaw) ? urlCamera.yaw * toRad : 0;
  const distance = urlCamera && Number.isFinite(urlCamera.dist)
    ? clamp(urlCamera.dist * metersToNorm, CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX) : CAMERA_DISTANCE_START;
  return { pitch, yaw, distance };
}

/**
 * The camera of the map and what moves it: the orbit camera (MapCamera) that rides the ground, the airplane mode
 * (the airplane button: flies with WASD / the arrows) and the keys / touch joystick that rotate the orbit camera.
 */
export default class MapCameraRig {
  public readonly camera: MapCamera;
  // the airplane button: flies the camera with WASD / the arrows instead of orbiting
  public readonly airplane: AirplaneController;
  // not flying: WASD / the arrows and the touch joystick rotate the orbit camera
  private orbitKeys: OrbitKeyRotation;

  /**
   * @param flatGround whether the elevation is off: the ground is flat at 0, the height field empty
   */
  constructor(private renderer: Renderer, private projection: MapProjection, private heightField: TerrainHeightField,
    private flatGround: () => boolean) {
    this.camera = new MapCamera(renderer, {
      fov: CAMERA_FOV,
      nearPlane: CAMERA_NEAR,
      farPlane: 10,
      zPosition: CAMERA_DISTANCE_START,
    });
    this.camera.rotationController.eulerXY = new Vector2(CAMERA_PITCH_START, 0);
    // always look a bit down, never from under the map
    this.camera.rotationController.rotationLimitX = new Vector2(CAMERA_PITCH_MIN, CAMERA_PITCH_MAX);
    this.camera.setDistanceLimit(CAMERA_DISTANCE_MIN, CAMERA_DISTANCE_MAX);
    this.camera.constrainView = () => this.keepAboveGround();
    // Geometry is projected relative to the bounds center (world origin).
    this.camera.pivot.position = new Vector3(0, 0, 0);
    this.airplane = new AirplaneController(renderer, this.camera);
    this.airplane.ground = (x, z) => this.heightField.heightAt(x, z);
    this.orbitKeys = new OrbitKeyRotation(renderer, this.camera);
  }

  /** The pitch, yaw and distance of the URL (the pivot is at its lat/lng already, the projection is centered there). */
  public applyUrlCamera(urlCamera: UrlCamera): void {
    const view = startCameraView(urlCamera, this.projection.metersToNorm);
    this.camera.rotationController.eulerXY = new Vector2(view.pitch, view.yaw);
    this.camera.distance = view.distance;
  }

  /** Airplane mode on or off, returns whether it is on now. */
  public toggleAirplane(): boolean {
    if (this.airplane.active) this.airplane.exit();
    else this.airplane.enter(this.projection.metersToNorm);
    this.orbitKeys.reset();
    return this.airplane.active;
  }

  /** The projection is centered elsewhere: the pivot on the new center, the plane keeps flying around it. */
  public recenter(): void {
    if (this.airplane.active) this.airplane.recenter();
    this.camera.pivot.position = new Vector3(0, 0, 0);
  }

  /**
   * Moves the camera of this frame: the plane, or the orbit camera on the keys and riding the ground.
   * @param stick the touch joystick (or the scripted one of a video)
   * @param orbitKeys whether the keys and the joystick rotate the orbit camera (not in a bench or a video)
   */
  public update(stick: { x: number; y: number } | undefined, orbitKeys: boolean): void {
    if (this.airplane.active) this.airplane.update(this.projection.metersToNorm, stick);
    else {
      if (orbitKeys) this.orbitKeys.update(stick);
      // before the update, which sees whether the camera changed
      this.followGround();
      this.camera.update();
      // eases out the bank left over from airplane mode
      this.airplane.settle();
    }
  }

  /**
   * The far plane at the 3d distance of the ground at viewDistanceMax in xz, from the camera height. The fog is of the
   * horizontal distance (see getFog in deferredShading.wgsl): a fixed far plane at viewDistanceMax would clip the ground
   * before the fog hides it when the camera is up high, one with room for the highest camera would draw past the fog
   * when it is low. Every frame, after the camera moves: the readers of the depth take depthParams from the camera of
   * the frame, the histories hold camera distances, not depths.
   */
  public updateFarPlane(viewDistanceMax: number): void {
    const height = Math.max(this.camera.worldPosition.y, 0);
    this.camera.projection.far = Math.sqrt(viewDistanceMax * viewDistanceMax + height * height);
  }

  /**
   * In airplane mode the shadow cascades go in front of the plane: it can't look back, so around the pivot most of
   * each cascade would lie behind it, never seen.
   */
  public shadowsAhead(): ShadowMapAhead | null {
    if (!this.airplane.active) return null;
    const origin = this.camera.worldPosition.clone();
    origin.y = this.camera.pivot.worldPosition.y;
    const forward = Vector3.subtract(this.camera.pivot.worldPosition, origin);
    forward.y = 0;
    // the pivot: where the view ray of the screen center hits the ground, the tilt-shift is sharp there
    const focus = forward.length();
    if (focus < 1e-6) return null;
    forward.normalize();
    return { origin, forward, focus };
  }

  /**
   * The orbit pivot eases to the height of the ground under it, so the camera orbits a point on the terrain and rides
   * over the hills while panning. Snaps when close: a still camera stops changing, so the map converges.
   */
  private followGround(): void {
    const pivot = this.camera.pivot.position;
    const ground = this.heightField.heightAt(pivot.x, pivot.z) ?? (this.flatGround() ? 0 : null);
    if (ground === null || ground === pivot.y) return;
    let y = pivot.y + (ground - pivot.y) * (1 - Math.exp(-PIVOT_GROUND_RATE * Math.min(this.renderer.time.dt, .1)));
    if (Math.abs(ground - y) < PIVOT_GROUND_SNAP_M * this.projection.metersToNorm) y = ground;
    this.camera.pivot.setPositionValues(pivot.x, y, pivot.z);
  }

  /**
   * Keeps the orbit camera out of the terrain: where the eye would be under the ground (a hill behind it, when it
   * looks flat over a valley), the view tilts down until it clears the ground by the clearance. Only the view: the
   * rotation controller keeps the pitch of the user, so the view tilts back up once the ground behind is lower again.
   * The ground is sampled under the eye and around it (the near plane is wider than a point).
   */
  private keepAboveGround(): void {
    if (this.flatGround()) return;
    const pivot = this.camera.pivot.position;
    const eye = this.camera.worldPosition;
    const dx = eye.x - pivot.x;
    const dz = eye.z - pivot.z;
    const horizontal = Math.hypot(dx, dz);
    if (horizontal < 1e-9) return;
    const dirX = dx / horizontal;
    const dirZ = dz / horizontal;
    const distance = this.camera.distance;
    const clearance = Math.max(CAMERA_GROUND_CLEARANCE_M * this.projection.metersToNorm, CAMERA_GROUND_CLEARANCE_NEAR * CAMERA_NEAR);
    const pitch = this.camera.eulerX;
    let p = pitch;
    // the eye moves toward the pivot as the view tilts down, over other ground: a few rounds
    for (let i = 0; i < 4; i++) {
      const h = distance * Math.cos(p);
      const x = pivot.x + dirX * h;
      const z = pivot.z + dirZ * h;
      let ground = -Infinity;
      for (const [ox, oz] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const g = this.heightField.heightAt(x + ox * clearance, z + oz * clearance);
        if (g !== null && g > ground) ground = g;
      }
      if (ground === -Infinity) break;
      const needed = (ground + clearance - pivot.y) / distance;
      if (Math.sin(-p) >= needed - 1e-9) break;
      p = Math.max(-Math.asin(Math.min(needed, 1)), CAMERA_PITCH_MIN);
      if (p === CAMERA_PITCH_MIN) break;
    }
    if (p === pitch) return;
    this.camera.pivot.euler = new Vector3(p, this.camera.eulerY, 0);
  }

  destruct(): void {
    this.orbitKeys.destruct();
  }
}
