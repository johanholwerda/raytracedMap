import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Color from 'webGPU-framework/renderer/math/Color';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import { GBUFFER_ATLAS_MAX, GBUFFER_ATLAS_STEP_M } from '../city/cityLayout';
import type { WorldBuildingResult } from '../map/worldGeometryCore';

/** The part of a tile's building mesh data the view needs. */
export type WallAtlasMeshData = Pick<WorldBuildingResult, 'interleaved' | 'vertexCount' | 'quantization' | 'indices' | 'atlasSize'>;
import shaderSource from './wallAtlasView.wgsl?raw';

// the margin around the atlas, a fraction of the screen
const MARGIN = .03;
const GRID_STEP_M = 100;

/**
 * The wallAtlas drawMode: the wall atlas of a tile (see layoutWalls in cityLayout.ts) fitted to the screen, see
 * wallAtlasView.wgsl. MapWorld shows the one of the last city tile that finished (WorldTileStreamer.lastCityTile).
 * The walls of the building mesh share their corners, the lightmap attribute of a vertex is the start of the wall it
 * starts (see buildBuildings): this view makes a mesh of its own with every corner of a wall at its atlas position.
 */
export default class WallAtlasView {
  private material: VertexFragmentShader;
  private clearColor = new Color(.08, .08, .1, 1);
  private loggedKey = '';
  private scale = new Vector2();
  private offset = new Vector2();
  // the mesh of the walls at their atlas positions and the tile it is of
  private mesh: Mesh | null = null;
  private meshKey = '';

  constructor(private renderer: Renderer) {
    this.material = new VertexFragmentShader(renderer, 'wallAtlasView', shaderSource);
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    // the atlas flips the winding of the walls of the holes
    this.material.cullMode = 'none';
  }

  /** Null tile: the empty screen. */
  public draw(tile: { key: string; result: WallAtlasMeshData } | null, metersToNorm: number): void {
    this.renderer.startRenderPass(null, { clearColor: this.clearColor, colorClear: true, depthClear: true }, 0, 'wallAtlasView');
    if (!tile?.result.atlasSize) return;
    if (tile.key !== this.meshKey) {
      this.mesh?.destruct();
      this.mesh = this.createMesh(tile.result);
      this.meshKey = tile.key;
    }
    if (!this.mesh) return;
    const [su, sv] = tile.result.atlasSize;
    if (tile.key !== this.loggedKey) {
      this.loggedKey = tile.key;
      LogGL.log(`[MapWorld] wall atlas of tile ${tile.key}: ${(su / metersToNorm).toFixed()} x ${(sv / metersToNorm).toFixed()} m`);
    }
    // of the canvas, renderer.width is that of the current render target
    const w = this.renderer.canvas.width, h = this.renderer.canvas.height;
    // pixels per world unit
    const p = Math.min(w / su, h / sv) * (1 - 2 * MARGIN);
    this.scale.setValues(2 * p / w, -2 * p / h);
    this.offset.setValues(-su * p / w, sv * p / h);
    this.material.setVector2('scale', this.scale);
    this.material.setVector2('offset', this.offset);
    this.material.setFloat('limit', GBUFFER_ATLAS_MAX * GBUFFER_ATLAS_STEP_M * metersToNorm);
    this.material.setFloat('gridStep', GRID_STEP_M * metersToNorm);
    this.renderer.draw(this.mesh, this.material);
  }

  // per triangle of a wall its corners at their atlas positions: the attribute of its first vertex plus the distance
  // along the wall and the height, like mapGBuffer.wgsl. position: atlas u, v, 0, the feature id
  private createMesh(result: WallAtlasMeshData): Mesh | null {
    const { interleaved, vertexCount, quantization: s, indices } = result;
    // the vertices decoded (see interleaveVertices in worldGeometryCore.ts), and their lightmap charts
    const u16 = new Uint16Array(interleaved), f32 = new Float32Array(interleaved);
    const p = new Float32Array(vertexCount * 4), lightmap = new Float32Array(vertexCount * 2);
    for (let v = 0; v < vertexCount; v++) {
      for (let k = 0; k < 4; k++) p[v * 4 + k] = u16[v * 8 + k] / 65535 * s[k] + s[4 + k];
      lightmap[v * 2] = f32[v * 4 + 2];
      lightmap[v * 2 + 1] = f32[v * 4 + 3];
    }
    const out: number[] = [];
    for (let t = 0; t + 2 < indices.length; t += 3) {
      const a = indices[t];
      // a roof
      if (lightmap[a * 2 + 1] < 0) continue;
      for (let k = 0; k < 3; k++) {
        const i = indices[t + k] * 4;
        const along = Math.hypot(p[i] - p[a * 4], p[i + 2] - p[a * 4 + 2]);
        out.push(lightmap[a * 2] + along, lightmap[a * 2 + 1] + p[i + 1] - p[a * 4 + 1], 0, p[i + 3]);
      }
    }
    if (out.length === 0) return null;
    const mesh = new Mesh(this.renderer);
    mesh.setAttribute('position', 4, new Float32Array(out));
    return mesh;
  }

  destruct(): void {
    this.mesh?.destruct();
    this.material.destruct();
  }
}
