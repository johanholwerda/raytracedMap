import type ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import Vector4 from 'webGPU-framework/renderer/math/Vector4';

const scale = new Vector4();
const offset = new Vector4();

/**
 * Sets the decoding of the unorm16 vertices of a building mesh (see quantizeVertices in worldGeometryCore.ts) on a
 * material that draws it: the uniforms positionScale and positionOffset of mapGBuffer.wgsl (WALLS), shading.wgsl and
 * overdraw.wgsl (QUANTIZED). quantization: 8 numbers per mesh (scale xyzw, offset xyzw), index: the mesh.
 */
export function setMeshQuantization(material: ShaderBase, quantization: number[], index = 0): void {
  const q = index * 8;
  material.setVector4('positionScale', scale.setValues(quantization[q], quantization[q + 1], quantization[q + 2], quantization[q + 3]));
  material.setVector4('positionOffset', offset.setValues(quantization[q + 4], quantization[q + 5], quantization[q + 6], quantization[q + 7]));
}
