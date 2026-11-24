import Renderer from 'webGPU-framework/renderer/render/Renderer';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import { clamp, lerp } from 'webGPU-framework/renderer/utils/MathUtils';

const KEYS = {
  up: ['KeyW', 'ArrowUp'],
  down: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
};
const ALL_KEYS = [...KEYS.up, ...KEYS.down, ...KEYS.left, ...KEYS.right];

/**
 * Rotates the orbit camera around its pivot with WASD / the arrows (and the touch joystick) while not flying: A/D turns
 * the view left / right, W tilts it up towards the horizon, S down towards top-down. The input is eased, so a tap
 * nudges the view and holding a key turns it at a constant rate.
 */
export default class OrbitKeyRotation {
  // radians per second at full input
  public yawSpeed = 1.2;
  public pitchSpeed = 0.8;
  private keys = new Set<string>();
  private listeners = new AbortController();
  // the eased input, -1..1
  private yaw = 0;
  private pitch = 0;

  constructor(private renderer: Renderer, private camera: MapCamera) {
    const signal = this.listeners.signal;
    window.addEventListener('keydown', (e) => {
      if (!ALL_KEYS.includes(e.code) || e.ctrlKey || e.metaKey || e.altKey || this.isTyping(e)) return;
      this.keys.add(e.code);
      // no page scrolling with the arrows
      e.preventDefault();
    }, { signal });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code), { signal });
    window.addEventListener('blur', () => this.keys.clear(), { signal });
  }

  /** Before camera.update(), while not flying. stick: the touch joystick, x right, y up, added to the keys. */
  public update(stick?: { x: number; y: number }): void {
    const dt = Math.min(this.renderer.time.dt, 0.1);
    const input = (keys: string[]) => keys.some(k => this.keys.has(k)) ? 1 : 0;
    const ease = Math.min(dt * 8, 1);
    this.yaw = lerp(this.yaw, clamp(input(KEYS.right) - input(KEYS.left) + (stick?.x ?? 0), -1, 1), ease);
    this.pitch = lerp(this.pitch, clamp(input(KEYS.up) - input(KEYS.down) + (stick?.y ?? 0), -1, 1), ease);
    if (Math.abs(this.yaw) < 1e-3 && Math.abs(this.pitch) < 1e-3) {
      this.yaw = this.pitch = 0;
      return;
    }
    // turning right is a negative yaw (as in AirplaneController), tilting up a less negative pitch
    this.camera.rotationController.rotateBy(new Vector2(this.pitch * this.pitchSpeed * dt, -this.yaw * this.yawSpeed * dt));
  }

  /** Entering airplane mode: the keys steer the plane from then on. */
  public reset(): void {
    this.keys.clear();
    this.yaw = this.pitch = 0;
  }

  private isTyping(e: KeyboardEvent): boolean {
    const target = e.target as HTMLElement | null;
    return !!target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);
  }

  destruct(): void {
    this.listeners.abort();
  }
}
