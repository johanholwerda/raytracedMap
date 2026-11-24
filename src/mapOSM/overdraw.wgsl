// Unified map vertex: xyz position + `.w` building id (unused here).
struct VertexInput {
#ifdef QUANTIZED
  // the building meshes of mapWorld, see shading.wgsl
  @location(0) position : vec4<f32> /* unorm16 */,
#else
  @location(0) position : vec4<f32>,
#endif
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) world : vec3<f32>,
};

struct Uniforms {
  viewProjection : mat4x4<f32>,
  // QUANTIZED: the decoding of the vertices
  positionScale : vec4<f32>,
  positionOffset : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
#ifdef QUANTIZED
  let world = input.position.xyz * uniforms.positionScale.xyz + uniforms.positionOffset.xyz;
#else
  let world = input.position.xyz;
#endif

  output.position = uniforms.viewProjection * vec4<f32>(world, 1.0);
  output.world = world;
  return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(.1);
}
