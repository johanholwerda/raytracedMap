import type Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import type Color from 'webGPU-framework/renderer/math/Color';
import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import type MapGeometry from './MapGeometry';
import type { CityMeshes } from './MapGeometry';

/**
 * The forward quality: the map shaded forward, without the gBuffer, lightmap, SSAO, TAA or tilt-shift. Only drawn
 * when something changed, else its last image is presented again.
 */
export default class ForwardPath {
  // The forward quality's image, with the depth its shading tests. Created when the forward quality is first drawn.
  private forwardRT?: RenderTexture;
  // The forward image in forwardRT is not of the current view: the deferred path drew while it was not.
  private stale: boolean = true;

  constructor(private renderer: Renderer, private geometry: MapGeometry) {}

  /** The deferred path drew: draw again when the forward path comes back. */
  public markStale(): void {
    this.stale = true;
  }

  /**
   * @param shadowStrength of the shadow map, the only shadows the forward shading has (0: none)
   * @param changed something changed since the last frame: the map is drawn again
   * @returns whether the map was drawn (else the last image was presented)
   */
  public draw(camera: MapCamera, city: CityMeshes | null, selectedID: number, clearColor: Color, shadowStrength: number,
    boundingBoxes: boolean, changed: boolean): boolean {
    const sceneColor = this.getForwardRT();
    const renderMap = changed || this.stale;
    this.stale = false;

    this.geometry.resetStats();
    if (renderMap) {
      for (const m of this.geometry.forwardMaterials) m.setFloat('shadowStrength', shadowStrength);
      this.renderer.startRenderPass(sceneColor, { clearColor, colorClear: true, depthClear: true }, 0, 'mapForward');
      this.geometry.drawForward(camera, city, selectedID);
      if (boundingBoxes) {
        this.renderer.startRenderPass(sceneColor, { colorClear: false, depthClear: false });
        this.geometry.drawBoundingBoxes(camera.viewProjection, city);
      }
    }
    this.renderer.blit(sceneColor, null, null, true);
    return renderMap;
  }

  private getForwardRT(): RenderTexture {
    this.forwardRT ??= new RenderTexture(this.renderer, 1, 1, {
      formats: ['rgb10a2unorm'],
      depth: true,
      // only draw's pass tests it, the bounding boxes after it don't
      discardDepth: true,
      sampleCount: 1,
      scaleToCanvas: true,
      toScreen: false,
    });
    return this.forwardRT;
  }

  destruct(): void {
    this.forwardRT?.destruct();
  }
}
