// Copies the scene to the target with the reprojected SSAO visibility multiplied over it, after
// mapOSM/progressiveSSAOComposite.wgsl: one pass instead of a copy and a blended pass that reads the target back.
// SHOW_ONLY: the AO alone on white, the scene isn't read.
// The AO of a pixel fades in over its first fadeCount samples, so disoccluded pixels don't pop in with the noise
// of 1 ray.

override SHOW_ONLY : bool = false;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

struct Uniforms {
  fadeCount : f32,
  // Matches shading.wgsl's fog coefficient. The AO fades out with the distance fog, so it doesn't darken
  // geometry the fog has already washed into the background.
  distanceFog : f32,
  // where the fog is 1, see getFog
  viewDistanceMax : f32,
  // strength of the AO, 0 is none (ReprojectedSSAO then skips it)
  amount : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// (visibility, sample count, camera distance, 0), see reprojectedSSAO.wgsl
@group(0) @binding(1) var texture : texture_2d<f32>;
// the scene the AO darkens, the size of texture
@group(0) @binding(2) var scene : texture_2d<f32>;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  output.uv = input.uv;
  return output;
}

// The distance fog: exponential (distanceFog per world unit), with a smoothstep from .7 of viewDistanceMax that takes
// it to 1 at viewDistanceMax, where the loaded map ends (see MapWorld.viewDistanceMax). 0 (not set): only the exp.
// The same in deferredShading.wgsl, cars.wgsl, reprojectedSSAOComposite.wgsl and mapOSM/shading.wgsl.
fn getFog(d : f32) -> f32 {
  let end = select(1., 1. - smoothstep(.7 * uniforms.viewDistanceMax, uniforms.viewDistanceMax, d), uniforms.viewDistanceMax > 0.);
  return 1. - exp(-d * uniforms.distanceFog) * end;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let coord = vec2<i32>(input.uv * vec2<f32>(textureDimensions(texture)));
  let s = textureLoad(texture, coord, 0);
  let ao = clamp(s.x, 0.0, 1.0);
  let fog = getFog(s.z);
  let strength = clamp(s.y / uniforms.fadeCount, 0.0, 1.0) * (1.0 - fog) * uniforms.amount;
  let faded = mix(1.0, ao, strength);
  if (SHOW_ONLY) {
    return vec4(faded, faded, faded, 1.0);
  }
  let color = textureLoad(scene, coord, 0);
  return vec4(color.rgb * faded, color.a);
}
