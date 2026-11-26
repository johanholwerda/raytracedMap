import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import MapProjection from './MapProjection';
import shaderSource from './shading.wgsl?raw';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import MapCamera from "webGPU-framework/utils/camera/MapCamera";
import { NO_FEATURE_ID, VERTEX_FLOATS } from './cityGlTFCore';

export default class GroundQuad {
  private static readonly SCALE = 10;

  // Unit quad in the XZ plane (y = 0), two triangles. Both the main material and
  // the SSAO g-buffer material draw it with cullMode 'none', so winding is
  // irrelevant. xyz only; the .w id is added in vertexData.
  private static readonly BASE = new Float32Array([
    -0.5, 0, -0.5,   0.5, 0,  0.5,   0.5, 0, -0.5,
    -0.5, 0, -0.5,  -0.5, 0,  0.5,   0.5, 0,  0.5,
  ]);

  private renderer: Renderer;
  private mesh: Mesh;
  public material: VertexFragmentShader;
  private projection: MapProjection;

  // Interleaved vertex data (xyz + id), reused on every update so the underlying
  // GPU buffer (and therefore the cached render bundle) stays the same object.
  private vertexData: Float32Array;
  private readonly vertCount: number;
  // The ground position and size currently baked into the vertices (NaN forces a first update).
  private groundPosition = new Vector3(NaN, NaN, NaN);
  private bakedScale = NaN;

  /** The width of the quad in world units, SCALE by default. */
  public size = GroundQuad.SCALE;
  /** Centered under the camera instead of under the pivot: with size 2 * r it covers r around the camera. */
  public centerOnCamera = false;

  // the height of the quad in meters
  private readonly heightM: number;

  /** heightM: the height of the quad, in meters, under the ground layers (MapWorld stacks its layers up from 0) */
  constructor(renderer: Renderer, projection: MapProjection, heightM: number = -1) {
    this.renderer = renderer;
    this.projection = projection;
    this.heightM = heightM;

    this.vertCount = GroundQuad.BASE.length / 3;
    // Unified stride-4 layout (xyz position + NO_FEATURE_ID), so the ground draws
    // with the same vertex layout as terrain and buildings (one g-buffer pipeline).
    this.vertexData = new Float32Array(this.vertCount * VERTEX_FLOATS);
    for (let v = 0; v < this.vertCount; v++) this.vertexData[v * VERTEX_FLOATS + 3] = NO_FEATURE_ID;

    this.mesh = new Mesh(renderer);
    this.mesh.setAttribute('position', VERTEX_FLOATS, this.vertexData);
    // Keep the CPU-side data so updates can re-bake positions into the buffer.
    this.mesh.keepCPUData = true;

    this.material = new VertexFragmentShader(renderer, 'osm', shaderSource, {
      includes: [hash],
    });
    this.material.setVector3('lightDirection', new Vector3(0.5, 0.5, 0.5).normalize());
    this.material.setVector3('baseColor', new Vector3(0.5, 0.5, 0.5));
  }

  // Bake the ground position (and the fixed scale) into the vertex positions and
  // re-upload them into the existing vertex buffer. No-op when nothing changed.
  private updateGroundPosition(camera: MapCamera): void {
    const y = this.heightM * this.projection.metersToNorm;
    const pivot = this.centerOnCamera ? camera.worldPosition : camera.pivot.worldPosition;
    const scale = this.size;

    if (this.groundPosition.x === pivot.x && this.groundPosition.y === y && this.groundPosition.z === pivot.z && this.bakedScale === scale) {
      return;
    }
    this.groundPosition.setValues(pivot.x, y, pivot.z);
    this.bakedScale = scale;

    const base = GroundQuad.BASE;
    for (let v = 0; v < this.vertCount; v++) {
      const dst = v * VERTEX_FLOATS;
      const src = v * 3;
      this.vertexData[dst] = base[src] * scale + pivot.x;
      this.vertexData[dst + 1] = base[src + 1] * scale + y;
      this.vertexData[dst + 2] = base[src + 2] * scale + pivot.z;
      // dst + 3 (id) stays NO_FEATURE_ID.
    }
    // Writes into the same GPUBuffer, so the cached render bundle stays valid.
    this.renderer.writeBuffer(this.mesh.vertexBuffer, this.vertexData);
  }

  public draw(camera: MapCamera, material?: VertexFragmentShader): void {
    this.updateGroundPosition(camera);

    if (!material) {
      this.material.setMatrix('viewProjection', camera.viewProjection);
      this.material.setVector3('cameraPosition', camera.worldPosition);
    }
    this.renderer.draw(this.mesh, material ?? this.material);
  }

  public drawGBuffer(camera: MapCamera, material: VertexFragmentShader): void {
    this.updateGroundPosition(camera);
    this.renderer.draw(this.mesh, material);
  }

  get vertexCount(): number {
    return this.vertCount;
  }

  destruct(): void {
    this.mesh.destruct();
  }
}
