import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import InformationOverlay from '../mapOSM/InformationOverlay';
import WorldTileStreamer from './map/WorldTileStreamer';
import { fetchNominatimAddress } from './map/nominatimAddress';

/**
 * Click-to-select buildings: a click (not a drag) on the canvas is picked after the next frame, with a ray through the
 * BVHs of the buildings on the CPU (see CityScene.pick), the InformationOverlay shows the building with its address
 * from Nominatim.
 */
export default class BuildingPicker {
  private overlay: InformationOverlay;
  private pendingPick: { x: number; y: number } | null = null;
  private readonly inverseViewProjection = new Matrix4x4();

  constructor(private renderer: Renderer, canvasParent: HTMLElement, private onPicked: () => void) {
    this.overlay = new InformationOverlay(renderer, canvasParent);
    this.overlay.setAddressResolver(fetchNominatimAddress);
    this.listen();
  }

  /** The global id of the selected building, -1 without one. */
  public get selectedID(): number {
    return this.overlay.selectedID;
  }

  public clearSelection(): void {
    this.overlay.clearSelection();
  }

  private listen(): void {
    const canvas = this.renderer.canvas;
    let downX = 0, downY = 0, downTime = 0;
    canvas.addEventListener('pointerdown', (e) => {
      downX = e.clientX; downY = e.clientY; downTime = performance.now();
    });
    canvas.addEventListener('pointerup', (e) => {
      const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
      const dt = performance.now() - downTime;
      if (moved > 6 || dt > 500) return; // a drag / long press, not a click
      const rect = canvas.getBoundingClientRect();
      this.pendingPick = {
        x: (e.clientX - rect.left) / rect.width,
        y: (e.clientY - rect.top) / rect.height,
      };
    });
  }

  /**
   * After the frame: picks the click since the last frame, if any. The ray goes from the camera through the point of
   * the click halfway the depth range of viewProjection.
   */
  public update(viewProjection: Matrix4x4, cameraPosition: Vector3, tileStreamer: WorldTileStreamer): void {
    if (!this.pendingPick) return;
    const { x, y } = this.pendingPick;
    this.pendingPick = null;

    const m = Matrix4x4.invert(viewProjection, this.inverseViewProjection).m;
    const nx = x * 2 - 1, ny = 1 - y * 2, nz = .5;
    const w = m[3] * nx + m[7] * ny + m[11] * nz + m[15];
    const point = new Vector3(
      (m[0] * nx + m[4] * ny + m[8] * nz + m[12]) / w,
      (m[1] * nx + m[5] * ny + m[9] * nz + m[13]) / w,
      (m[2] * nx + m[6] * ny + m[10] * nz + m[14]) / w,
    );
    const direction = Vector3.subtract(point, cameraPosition);

    const globalId = tileStreamer.pickBuilding(cameraPosition, direction);
    this.overlay.select(globalId, tileStreamer.lookupBuilding(globalId));
    this.onPicked();
  }

  destruct(): void {
    this.overlay.destruct();
  }
}
