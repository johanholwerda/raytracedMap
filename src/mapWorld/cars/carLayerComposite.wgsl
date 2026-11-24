// The layer of the cars (see cars.wgsl) over the image, with premultiplied alpha blending: the cars over it, their
// shadows (black with the darkening as the alpha) darken it.

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
}

@group(0) @binding(0) var texture : texture_2d<f32>;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  return textureLoad(texture, vec2<i32>(input.position.xy), 0);
}
