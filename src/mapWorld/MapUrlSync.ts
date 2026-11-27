import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import type MapProjection from './map/MapProjection';
import { TIMES_OF_DAY, TimeOfDay } from './MapWorldSettings';

// ms the orbit camera has to stand still before its view goes into the URL, see update
const URL_CAMERA_DELAY_MS = 500;
// ms after the last input (pointer, wheel, key) in which the camera moving counts as the user moving it: the zoom
// easing and the rotation inertia run on after the input, a random city or a searched place loads a bit after the click
const USER_INPUT_WINDOW_MS = 2000;

/**
 * The orbit camera in the URL: ?lat=&lng= the pivot (the ground point it orbits), dist= its distance to the pivot in
 * meters, pitch= and yaw= in degrees (pitch < 0: looking down). The time of day preset goes in as ?time=, see TIMES_OF_DAY.
 */
export type UrlCamera = { lat: number; lng: number; dist: number; pitch: number; yaw: number };

/** The camera of the URL, or null without a (valid) lat and lng. The missing others are NaN. */
export function parseUrlCamera(params: URLSearchParams): UrlCamera | null {
  const num = (key: string) => {
    const value = params.get(key);
    return value === null || value.trim() === '' ? NaN : Number(value);
  };
  const lat = num('lat');
  const lng = num('lng');
  if (!(Math.abs(lat) <= 85) || !(Math.abs(lng) <= 180)) return null;
  return { lat, lng, dist: num('dist'), pitch: num('pitch'), yaw: num('yaw') };
}

/**
 * Writes the orbit camera (lat, lng, dist, pitch, yaw, see UrlCamera) and the time of day to the URL once the camera
 * has stood still for URL_CAMERA_DELAY_MS, with replaceState: no history entry per view. Only after the user moved
 * the camera (within USER_INPUT_WINDOW_MS of their input) or changed the time of day: a fresh load, the pivot easing
 * onto the loaded ground, keep the URL as it is.
 */
export default class MapUrlSync {
  // the user moved the camera or changed the time of day since the URL was last written, the time (performance.now) the
  // camera last moved and of the last user input
  private pending: boolean = false;
  private cameraStillSince: number = 0;
  private lastInputTime: number = -Infinity;
  private readonly markInput = () => this.lastInputTime = performance.now();
  private readonly markDrag = (e: PointerEvent) => { if (e.buttons) this.markInput(); };
  private static readonly inputOptions = { capture: true, passive: true };

  constructor() {
    // the user's input, for telling their camera moves from the program's
    const options = MapUrlSync.inputOptions;
    window.addEventListener('pointerdown', this.markInput, options);
    window.addEventListener('pointermove', this.markDrag, options);
    window.addEventListener('wheel', this.markInput, options);
    window.addEventListener('keydown', this.markInput, options);
  }

  /** The time of day changed (the GUI or a random city): into the URL once the camera stands still. */
  public markPending(): void {
    this.pending = true;
  }

  /** Every frame, with whether the camera moved in it. */
  public update(moving: boolean, camera: MapCamera, projection: MapProjection, timeOfDay: TimeOfDay): void {
    const now = performance.now();
    if (moving) {
      if (now - this.lastInputTime < USER_INPUT_WINDOW_MS) this.pending = true;
      this.cameraStillSince = now;
      return;
    }
    if (!this.pending || now - this.cameraStillSince < URL_CAMERA_DELAY_MS) return;
    this.pending = false;

    const pivot = camera.pivot.position;
    const { lat, lng } = projection.worldToLatLng(pivot.x, pivot.z);
    const euler = camera.rotationController.eulerXY;
    // the yaw adds up over the turns: into -180..180
    const yaw = euler.y - Math.PI * 2 * Math.round(euler.y / (Math.PI * 2));
    const toDeg = 180 / Math.PI;
    const params = new URLSearchParams(window.location.search);
    // the camera's lat / lng win over it on load
    params.delete('city');
    params.set('lat', lat.toFixed(6));
    params.set('lng', lng.toFixed(6));
    params.set('dist', (camera.distance / projection.metersToNorm).toFixed(0));
    params.set('pitch', (euler.x * toDeg).toFixed(2));
    params.set('yaw', (yaw * toDeg).toFixed(2));
    params.set('time', TIMES_OF_DAY[timeOfDay]);
    const url = `${window.location.pathname}?${params}${window.location.hash}`;
    if (url !== window.location.pathname + window.location.search + window.location.hash) {
      window.history.replaceState(window.history.state, '', url);
    }
  }

  destruct(): void {
    const options = MapUrlSync.inputOptions;
    window.removeEventListener('pointerdown', this.markInput, options);
    window.removeEventListener('pointermove', this.markDrag, options);
    window.removeEventListener('wheel', this.markInput, options);
    window.removeEventListener('keydown', this.markInput, options);
  }
}
