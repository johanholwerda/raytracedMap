// Tilt-shift composite: mixes the sharp image with five blurred copies of it by a circle of confusion that grows
// linearly with the distance to the horizontal center line, see TiltShift.ts.
// Needs hash.wgsl.

// 1: the blurs are at half the resolution (the halfRes param of TiltShift), read bilinearly
override HALF_RES: u32 = 0;

struct Uniforms {
  // the gaussian sigmas of blur0..3 in pixels, ascending. 0 for a level that was skipped, its texture is the source
  sigmas : vec4<f32>,
  // the sigma of blur4, the largest
  sigma4 : f32,
  // the sigma in pixels at the top and bottom edge of the screen (|ndc y| = 1)
  slope : f32,
  // HALF_RES: a pixel position times this is the uv of the blurs, 1 / (2 * their size). They are an exact 2:1
  // rectangle of the image from the top left (an odd last row/column falls off), see TiltShift.draw
  blurTexel : vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var texture : texture_2d<f32>;
@group(0) @binding(2) var blur0 : texture_2d<f32>;
@group(0) @binding(3) var blur1 : texture_2d<f32>;
@group(0) @binding(4) var blur2 : texture_2d<f32>;
@group(0) @binding(5) var blur3 : texture_2d<f32>;
@group(0) @binding(6) var blur4 : texture_2d<f32>;
@group(0) @binding(7) var blurSampler : sampler;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  output.uv = input.uv;
  return output;
}

//level 0 is the sharp image, level i the blur i - 1
fn level(i : u32, coord : vec2<i32>, uv : vec2<f32>) -> vec4<f32> {
  if (HALF_RES != 0u) {
    // a skipped level (sigma 0) is not read, fs_main reads the sharp image for it
    switch (i) {
      case 0u: { return textureLoad(texture, coord, 0); }
      case 1u: { return textureSampleLevel(blur0, blurSampler, uv, 0.0); }
      case 2u: { return textureSampleLevel(blur1, blurSampler, uv, 0.0); }
      case 3u: { return textureSampleLevel(blur2, blurSampler, uv, 0.0); }
      case 4u: { return textureSampleLevel(blur3, blurSampler, uv, 0.0); }
      default: { return textureSampleLevel(blur4, blurSampler, uv, 0.0); }
    }
  }
  switch (i) {
    case 0u: { return textureLoad(texture, coord, 0); }
    case 1u: { return textureLoad(blur0, coord, 0); }
    case 2u: { return textureLoad(blur1, coord, 0); }
    case 3u: { return textureLoad(blur2, coord, 0); }
    case 4u: { return textureLoad(blur3, coord, 0); }
    default: { return textureLoad(blur4, coord, 0); }
  }
}

//this is the last pass onto the 8 bit canvas: triangle noise of 1 step of 255 each way (the sum of 2 uniforms) breaks
//up the banding of the smooth gradients (the sky, the fog, the blurs), and its error doesn't depend on the color
fn dither(c : vec4<f32>, coord : vec2<i32>) -> vec4<f32> {
  let r = hash22uf(vec2<u32>(coord));
  return vec4<f32>(saturate(c.rgb + (r.x + r.y - 1.0) / 255.0), c.a);
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(input.position.xy);
  return dither(composite(input, coord), coord);
}

fn composite(input : VertexOutput, coord : vec2<i32>) -> vec4<f32> {
  // no blur (see TiltShift.draw): only the dither
  if (uniforms.slope == 0.0) { return textureLoad(texture, coord, 0); }
  let ndcY = 1.0 - 2.0 * input.uv.y;
  let s = min(uniforms.slope * abs(ndcY), uniforms.sigma4);
  let v = s * s;

  // The first level i that blurs at least as much as this pixel needs, with the variances of the levels on either
  // side of it: a of level i - 1, b of level i. Not a loop over an array of the sigmas: the array, indexed by a per
  // pixel i, went to local memory on D3D12 and took half the time of this pass at 4K.
  let sv = uniforms.sigmas * uniforms.sigmas;
  var i = 5u;
  var a = sv.w;
  var b = uniforms.sigma4 * uniforms.sigma4;
  if (v <= sv.x) { i = 1u; a = 0.0; b = sv.x; }
  else if (v <= sv.y) { i = 2u; a = sv.x; b = sv.y; }
  else if (v <= sv.z) { i = 3u; a = sv.y; b = sv.z; }
  else if (v <= sv.w) { i = 4u; a = sv.z; b = sv.w; }

  // A mix of two zero mean kernels has the weighted mean of their variances, so lerping by variance keeps the
  // variance of the mix at s^2 between the radii that were blurred.
  let t = clamp((v - a) / max(b - a, 1e-6), 0.0, 1.0);
  if (HALF_RES == 0u) {
    return mix(level(i - 1u, coord, vec2<f32>()), level(i, coord, vec2<f32>()), t);
  }
  // A skipped level holds the full resolution image, which the half resolution uv would misplace: read the sharp
  // image itself for it.
  let uv = input.position.xy * uniforms.blurTexel;
  let lo = select(i - 1u, 0u, a == 0.0);
  let hi = select(i, 0u, b == 0.0);
  return mix(level(lo, coord, uv), level(hi, coord, uv), t);
}
