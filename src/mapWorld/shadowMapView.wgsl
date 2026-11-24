// The shadowMap drawMode: the 4 cascades of the sun shadow map (SunShadows) as a 2 x 2 grid, cascade 0 top left,
// 1 top right, 2 bottom left, 3 bottom right. The grid is a square in the middle of the screen with black borders
// beside it. A texel shows the height of its caster (the first moment of the VSM is -height), the ground is black.

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

struct Uniforms {
  // width / height of the screen
  aspectRatio : f32,
  metersToNorm : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// the 4 cascades, one layer each (ShadowMap.cascadeTexture)
@group(0) @binding(1) var shadowMaps : texture_2d_array<f32>;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  output.uv = input.uv;
  return output;
}

fn loadMoments(i : u32, uv : vec2<f32>) -> vec2<f32> {
  let size = vec2<f32>(textureDimensions(shadowMaps));
  let coord = vec2<i32>(min(uv * size, size - 1.0));
  return textureLoad(shadowMaps, coord, i, 0).xy;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  // -1..1 across the square of the grid
  var p = input.uv * 2.0 - 1.0;
  if (uniforms.aspectRatio > 1.0) {
    p.x *= uniforms.aspectRatio;
  } else {
    p.y /= uniforms.aspectRatio;
  }
  if (any(abs(p) > vec2<f32>(1.0))) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let grid = (p * 0.5 + 0.5) * 2.0;
  let i = min(u32(grid.x), 1u) + min(u32(grid.y), 1u) * 2u;
  let uv = fract(grid);

  // the color of the cascade in shadowDebug 1, on the lines between the cells
  var col = vec3<f32>(1.0, 0.3, 0.3);
  if (i == 1u) { col = vec3<f32>(0.3, 1.0, 0.3); }
  else if (i == 2u) { col = vec3<f32>(0.3, 0.5, 1.0); }
  else if (i == 3u) { col = vec3<f32>(1.0, 1.0, 0.3); }
  let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  if (edge < 0.004) {
    return vec4<f32>(col, 1.0);
  }

  let moments = loadMoments(i, uv);
  let meters = max(-moments.x, 0.0) / uniforms.metersToNorm;
  // 50 m is about .5, 200 m nearly white
  let h = 1.0 - exp(-meters / 70.0);
  // the spread of the blurred moments (the penumbra of the VSM) tinted with the cascade color
  let sigma = sqrt(max(moments.y - moments.x * moments.x, 0.0)) / uniforms.metersToNorm;
  let spread = 1.0 - exp(-sigma / 20.0);
  return vec4<f32>(mix(vec3<f32>(h), col, spread * 0.5), 1.0);
}
