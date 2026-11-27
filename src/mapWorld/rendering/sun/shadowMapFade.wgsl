// One step of the fade of a shadow cascade to its latest render (ShadowMap): the moments it shows, reprojected to the
// latest center and size, blended towards the latest render. The VSM moments are linear, so a blend of them is a
// partial caster: a building that is new in the cascade fades in instead of popping in.
// Drawn over the full cascade: uv is the texel of the latest render.

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

struct Uniforms {
  // ndc in the shown cascade = ndc in the latest render * scale + offset (the same sun direction, so no rotation)
  offset : vec2<f32>,
  scale : f32,
  // towards the latest render, 1 is all of it
  alpha : f32,
  // 1: the ground the shown cascade didn't cover fades in from lit (the last cascade: nothing covered it)
  fadeUncovered : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var fadeSampler : sampler;
// a copy of the cascade as it was shown
@group(0) @binding(2) var shownMoments : texture_2d<f32>;
// the latest render, blurred
@group(0) @binding(3) var latestMoments : texture_2d<f32>;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  output.uv = input.uv;
  return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let size = vec2<f32>(textureDimensions(latestMoments));
  let latest = textureLoad(latestMoments, vec2<i32>(min(input.uv * size, size - 1.0)), 0);
  if (uniforms.alpha >= 1.0) {
    return latest;
  }
  // texture v runs down, ndc y up (as in the shadow lookup of shading.wgsl)
  let ndc = vec2<f32>(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
  let shownNdc = ndc * uniforms.scale + uniforms.offset;
  // Ground the shown cascade didn't cover yet. A cascade behind it covered it until now, it has the latest render at
  // once: lit while it fades in would be a pop too. Past the last one the shadows fade in from lit (moments 0).
  if (any(abs(shownNdc) > vec2<f32>(1.0))) {
    return latest * mix(1.0, uniforms.alpha, uniforms.fadeUncovered);
  }
  let shownUV = vec2<f32>(shownNdc.x * 0.5 + 0.5, 0.5 - shownNdc.y * 0.5);
  let shown = textureSampleLevel(shownMoments, fadeSampler, shownUV, 0.0);
  return mix(shown, latest, uniforms.alpha);
}
