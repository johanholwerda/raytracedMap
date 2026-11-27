import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import boundingBoxShader from './boundingBox.wgsl?raw';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';

export type BBoxEntry = {
  min: [number, number, number];
  max: [number, number, number];
  color: [number, number, number];
};

// 48 bytes per entry: vec4 min + vec4 max + vec4 color
const BBOX_BYTES = 48;
const MAX_BBOXES = 10000;

export default class BoundingBoxViewer {
  private renderer: Renderer;
  private material: VertexFragmentShader;
  private mesh: Mesh;
  private bboxBuffer: GPUBuffer;

  constructor(renderer: Renderer) {
    this.renderer = renderer;
    this.material = new VertexFragmentShader(this.renderer, 'boundingBox', boundingBoxShader, {includes:[hash]});
    this.material.setAlphaBlending();
    this.material.cullMode = 'none';
    this.material.depthCompare = 'always';
    this.material.depthWriteEnabled = false;

    this.bboxBuffer = renderer.createBuffer(BBOX_BYTES * MAX_BBOXES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.material.setBuffer('boundingBoxes', this.bboxBuffer);

    this.mesh = this.createAABBEdges();
  }

  private createAABBEdges(): Mesh {
    const mesh = new Mesh(this.renderer);
    const numEdges = 12;
    const numVertices = numEdges * 4;
    const numIndices = numEdges * 6;

    const positions = new Float32Array(numVertices * 3);
    const normals = new Float32Array(numVertices * 3);
    const uvs = new Float32Array(numVertices * 2);
    const indices = new Array<number>(numIndices);

    let vIdx = 0;
    let uvIdx = 0;
    let iIdx = 0;

    for (let axis = 0; axis < 3; axis++) {
      for (let i = 0; i < 4; i++) {
        const c1 = (i & 1) ? 1 : -1;
        const c2 = (i & 2) ? 1 : -1;

        const p1 = [0, 0, 0];
        const p2 = [0, 0, 0];
        const dir = [0, 0, 0];

        p1[axis] = -1;
        p1[(axis + 1) % 3] = c1;
        p1[(axis + 2) % 3] = c2;

        p2[axis] = 1;
        p2[(axis + 1) % 3] = c1;
        p2[(axis + 2) % 3] = c2;

        dir[axis] = 1;

        const baseV = vIdx / 3;

        positions[vIdx] = p1[0]; positions[vIdx + 1] = p1[1]; positions[vIdx + 2] = p1[2];
        normals[vIdx] = dir[0]; normals[vIdx + 1] = dir[1]; normals[vIdx + 2] = dir[2];
        uvs[uvIdx] = 0; uvs[uvIdx + 1] = 0;
        vIdx += 3; uvIdx += 2;

        positions[vIdx] = p2[0]; positions[vIdx + 1] = p2[1]; positions[vIdx + 2] = p2[2];
        normals[vIdx] = dir[0]; normals[vIdx + 1] = dir[1]; normals[vIdx + 2] = dir[2];
        uvs[uvIdx] = 0; uvs[uvIdx + 1] = 1;
        vIdx += 3; uvIdx += 2;

        positions[vIdx] = p2[0]; positions[vIdx + 1] = p2[1]; positions[vIdx + 2] = p2[2];
        normals[vIdx] = dir[0]; normals[vIdx + 1] = dir[1]; normals[vIdx + 2] = dir[2];
        uvs[uvIdx] = 1; uvs[uvIdx + 1] = 1;
        vIdx += 3; uvIdx += 2;

        positions[vIdx] = p1[0]; positions[vIdx + 1] = p1[1]; positions[vIdx + 2] = p1[2];
        normals[vIdx] = dir[0]; normals[vIdx + 1] = dir[1]; normals[vIdx + 2] = dir[2];
        uvs[uvIdx] = 1; uvs[uvIdx + 1] = 0;
        vIdx += 3; uvIdx += 2;

        indices[iIdx++] = baseV;
        indices[iIdx++] = baseV + 1;
        indices[iIdx++] = baseV + 2;
        indices[iIdx++] = baseV;
        indices[iIdx++] = baseV + 2;
        indices[iIdx++] = baseV + 3;
      }
    }

    mesh.setPositions(positions);
    mesh.setNormals(normals);
    mesh.setUV0(uvs);
    mesh.setIndices(indices);
    return mesh;
  }

  public draw(viewProjection: Matrix4x4, entries: BBoxEntry[]): void {
    if (entries.length === 0) return;
    const count = Math.min(entries.length, MAX_BBOXES);

    const data = new Float32Array(count * 12);
    for (let i = 0; i < count; i++) {
      const e = entries[i];
      data[i * 12 + 0] = e.min[0];
      data[i * 12 + 1] = e.min[1];
      data[i * 12 + 2] = e.min[2];
      data[i * 12 + 3] = 0;
      data[i * 12 + 4] = e.max[0];
      data[i * 12 + 5] = e.max[1];
      data[i * 12 + 6] = e.max[2];
      data[i * 12 + 7] = 0;
      data[i * 12 + 8] = e.color[0];
      data[i * 12 + 9] = e.color[1];
      data[i * 12 + 10] = e.color[2];
      data[i * 12 + 11] = 0;
    }

    this.renderer.writeBuffer(this.bboxBuffer, data);
    this.material.setVector2('resolution', this.renderer.size);
    this.material.setMatrix('viewProjection', viewProjection);
    this.renderer.draw(this.mesh, this.material, { instanceCount: count });
  }

  public destruct(): void {
    this.mesh.destruct();
    this.bboxBuffer.destroy();
  }
}
