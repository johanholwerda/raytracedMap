import type Renderer from 'webGPU-framework/renderer/render/Renderer';
import type RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import type MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import type CarTraffic from '../cars/CarTraffic';
import type MapGBuffer from './deferred/MapGBuffer';
import type TAA from './taa/TAA';
import type TiltShift from './tiltShift/TiltShift';
import type { DeferredFrame } from './DeferredFrame';

/**
 * After the TAA of the deferred path: the AO composited over the image (when it is not in the shading), the cars and
 * the tilt-shift onto the screen.
 */
export default class PostProcess {
  // ReprojectedSSAO.version the tilt-shift target was composited with, -1 without AO.
  private postSSAOVersion: number = -1;

  constructor(private renderer: Renderer, private taa: TAA, private tiltShift: TiltShift, private traffic: CarTraffic,
    private gBuffer: MapGBuffer, private sceneColor: RenderTexture) {}

  /** Another path drew: the tilt-shift composites again when the deferred path comes back. */
  public invalidate(): void {
    this.tiltShift.invalidate();
    this.postSSAOVersion = -1;
  }

  /** @param distanceFog, viewDistanceMax of the shading params, for the AO composite */
  public draw(frame: DeferredFrame, camera: MapCamera, distanceFog: number, viewDistanceMax: number): void {
    const { useSceneColor, showsLightmap, shadowView, ssao, ssaoInShading, renderMap, geometryChanged } = frame;
    // Composited over the image, the AO's version tells whether the tilt-shift input changed. In the shading it is a
    // part of the map.
    const ssaoVersion = ssao && !ssaoInShading ? ssao.version : -1;

    // The scene and its AO go into the tilt-shift target, which is blurred onto the screen after. Also with tilt-shift
    // off: its composite is then only the dither, the last pass onto the 8 bit canvas, which breaks up the banding of
    // the TAA (it averages the dither of the shading away). While neither the scene nor the AO changed, the target and
    // its blurs still hold this frame: only the tilt-shift composite runs.
    const postTarget = useSceneColor && !showsLightmap && !shadowView ? this.tiltShift.target : null;
    // The cars move every frame: over the image, before the tilt-shift. Not over the debug views of the lightmap, the
    // shadows and the SSAO.
    const drawCars = useSceneColor && !showsLightmap && !shadowView && !ssao?.showOnly && this.traffic.active;
    const reusePost = postTarget != null && !renderMap && !drawCars && ssaoVersion === this.postSSAOVersion && this.tiltShift.holdsTarget;
    this.postSSAOVersion = ssaoVersion;

    if (!reusePost) {
      // The AO copies the scene into the target in the same pass. The SSAO debug view replaces the scene, it isn't
      // drawn under it.
      const ssaoDrawn = ssao != null && !ssaoInShading && ssao.composite(distanceFog, viewDistanceMax, this.taa.output, postTarget);
      if (!ssaoDrawn && useSceneColor && !ssao?.showOnly) {
        this.renderer.blit(showsLightmap ? this.sceneColor : this.taa.output, postTarget, null, true);
      }
      if (drawCars) this.traffic.draw(camera, this.taa.viewProjectionNoJitter, this.gBuffer, postTarget);
    }

    if (postTarget) {
      // In a static view only the jitter, the AO and the lightmap's noise change: the blurs may lag a few frames. Not
      // the cars: they'd move in steps in the blurred rows.
      this.tiltShift.draw(camera, !reusePost, null, !geometryChanged && !drawCars);
    } else {
      this.tiltShift.invalidate();
    }
  }
}
