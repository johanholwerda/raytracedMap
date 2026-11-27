import type ReprojectedSSAO from './ssao/ReprojectedSSAO';
import type { SSAOMode } from './ssao/ReprojectedSSAO';
import type { WaterFrame } from './deferred/MapWater';
import type { ShadowMode } from '../MapWorldSettings';

/**
 * What the deferred path does in a frame, decided once at its start (MapRenderer.planDeferredFrame) and followed by
 * its passes: the scene (gBuffer, lightmap, SSAO, shading, TAA) and the post (PostProcess).
 */
export type DeferredFrame = {
  // the shadowMode param as it can be, see MapRenderer.shadowMode
  shadowMode: ShadowMode;
  // a shadow debug view replaces the scene: no SSAO, tilt-shift or cars over it
  shadowView: boolean;
  // the map is shaded into the scene color (not the overdraw view)
  useSceneColor: boolean;
  // The lightmap view doesn't move with the camera: no TAA, tilt-shift or SSAO, drawn every frame
  showsLightmap: boolean;
  // the camera moved, tiles came, or a param changed
  geometryChanged: boolean;
  lightmapUpdating: boolean;
  water: WaterFrame;
  // the SSAO when it runs this frame, else null
  ssao: ReprojectedSSAO | null;
  ssaoMode: SSAOMode;
  // the AO on the sky term only, in the shading (ReprojectedSSAO.skyOnly), instead of composited over the image
  ssaoInShading: boolean;
  ssaoAccumulating: boolean;
  // nothing changes: the TAA averages the frames into a static image
  isStatic: boolean;
  // the map is drawn (gBuffer and shading), else the TAA history is presented again
  renderMap: boolean;
  // a new jitter offset, and the gBuffer that goes with it
  newJitter: boolean;
};
