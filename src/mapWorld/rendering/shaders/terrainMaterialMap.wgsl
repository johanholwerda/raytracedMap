// The material of the ground top down, for TerrainSunMap.ts (its materialMap): the ground and its layers drawn from
// above with a material each (MapGeometry.drawMaterialMap), the highest wins the depth test. The terrain in the water
// reflections is shaded from it (waterTerrainHorizon.wgsl).
// Out, r32uint: uniforms.material, 0 where no ground is (the clear color)

// Unified map vertex: xyz position + `.w` building id (unused here).
struct VertexInput {
  @location(0) position : vec4<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
};

struct Uniforms {
  viewProjection : mat4x4<f32>,
  // the palette index of the layer + 1, and TERRAIN_MATERIAL_SNOW where the snow lies on it
  material : u32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = uniforms.viewProjection * vec4<f32>(input.position.xyz, 1.0);
  return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<u32> {
  return vec4<u32>(uniforms.material, 0u, 0u, 0u);
}
