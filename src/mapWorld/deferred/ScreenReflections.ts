import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import TAA from '../taa/TAA';

/**
 * The colors of the buildings and trees in the water reflections, from the screen (waterScreenReflection.wgsl): the
 * world trace of a mirror ray gives where it hits (getOcclusionHit of cityTraceDistance.wgsl), and where the screen
 * shows that point the water mirrors its color, from the image of the frame before (the TAA history, at where the point
 * was in it). A hit off the screen takes the color where the ray leaves it, one the screen doesn't show (an underside)
 * the color of an unseen surface. Without it a building in the water only darkens it.
 * A march of the rays over the depth of the gBuffer first (screen space reflections) stepped over buildings between its
 * samples and gave the same image for more time (+.27 instead of +.18 ms at 1080p with a ray per water pixel, +1.2
 * instead of +.6 at 4K, RTX 3080 Ti).
 * The state of a frame for the passes that trace the rays: the deferred shading (a ray per water pixel) or
 * WaterReflectionHalfRes. update() before them, then bind() on their material.
 */
export default class ScreenReflections {
  private enabled = false;
  private history: GPUTextureView | null = null;
  private historyValid = false;
  private readonly historyViewProjection = new Matrix4x4();
  private readonly sampler: GPUSampler;

  constructor(renderer: Renderer) {
    this.sampler = renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  /**
   * Every frame before the rays are traced, after TAA.setCamera and before TAA.resolve: its output is then the image of
   * the frame before, its camera without the jitter viewProjectionOldNoJitter. enabled false: the buildings in the water
   * have no colors, they only darken it.
   */
  public update(enabled: boolean, taa: TAA): void {
    this.enabled = enabled;
    this.history = taa.output.views[0];
    this.historyValid = taa.hasHistory;
    this.historyViewProjection.copy(taa.viewProjectionOldNoJitter);
  }

  /** The uniforms and bindings of waterScreenReflection.wgsl, on a material that includes it. */
  public bind(m: VertexFragmentShader): void {
    m.setConstant('SCREEN_REFLECTIONS', this.enabled);
    if (this.history) m.setTexture('reflectionHistory', this.history);
    m.setSampler('reflectionHistorySampler', this.sampler);
    m.setMatrix('historyViewProjection', this.historyViewProjection);
    m.setFloat('screenReflectionHistory', this.historyValid ? 1 : 0);
  }
}
