// The sky visibility lightmap at half resolution, see SkyVisibilityHalfRes.ts: 1 sample per 2 x 2 pixels of the gBuffer,
// at the top left pixel of the block. The lookup (getLightmapLighting: hash table probes and texel loads, for 1 to 3
// levels) is the most expensive part of the deferred shading; the shading blends these samples instead
// (getHalfResLighting in deferredShading.wgsl). With LIGHTMAP_SUN also the sun visibility of the texels (the shadows of
// SUN_SHADOWS 2), in the same lookup (getLightmapChannels shares the tile probes), where the surface faces the sun.
// Out, rg32uint:
// x: the sky visibility and the sun visibility (2 x f16), -1 where the texels have no paths (no sun: always -1)
// y: the key of the surface point of the sample, see packHalfResKey in skyVisibilityResolve.wgsl. 0 on the sky
// With HALF_RES_BILINEAR, rg16float, which the shading samples bilinearly (getHalfResLightingBilinear): the sky
// visibility, that of an open sky (n.y * .5 + .5, as the shading's) while the texels have no paths yet, 1 on the sky,
// and the sun visibility, 1 without paths (also where it faces away: it would darken the edges of the lit faces)
// Needs the includes of the lightmap resolve (SkyVisibilityLightmap.resolveIncludes), its uniforms are set with the
// deferred shading's (SkyVisibilityLightmap.setResolveUniforms)

// the texels hold the sun visibility too, see SkyVisibilityLightmap.setSun
override LIGHTMAP_SUN : bool = false;

struct Uniforms {
  inverseProjection : mat4x4<f32>,
  inverseView : mat4x4<f32>,
  viewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  // of the gBuffer, the full resolution
  resolution : vec2<f32>,
  // for the noise of stochasticLevel
  frame : f32,
  // set with the other resolve materials, see SkyVisibilityLightmap.setResolveUniforms. Not all are used here
  lightmapEnabled : u32,
  stochasticLevel : u32,
  pixelSize : f32,
  pixelScale : f32,
  texelsPerPixel : f32,
  texelSizeMin : f32,
  levelAngle : f32,
  blurSlope : f32,
  blurMax : f32,
  blurTexelScale : f32,
  searchCount : u32,
  accumulationMax : f32,
  ancestorWeight : f32,
  levelBlend : f32,
  atlasStep : f32,
  lightmapHalfRes : u32,
  // see getGBufferDistance
  depthParams : vec2<f32>,
  // toward the sun, with LIGHTMAP_SUN: a surface facing away gets no sun lookup
  lightDirection : vec3<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> tileBuffer : array<u32>;
// the sky visibility lightmap: 1 word per texel, see SkyVisibilityLightmap
@group(0) @binding(2) var<storage, read> lightingBuffer : array<u32>;
// the gBuffer: its 2 targets and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(3) var gBufferTexture : texture_2d<u32>;
@group(0) @binding(4) var gBufferAtlasTexture : texture_2d<u32>;
@group(0) @binding(5) var gBufferDepth : texture_depth_2d;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  return output;
}

// as getNoise in deferredShading.wgsl, of the full resolution pixel
fn getNoise(coord : vec2<u32>) -> f32 {
  let p = vec2<f32>(coord) + 5.588238 * (uniforms.frame % 64.);
  return fract(52.9829189 * fract(dot(p, vec2(.06711056, .00583715))));
}

@fragment
#ifdef HALF_RES_BILINEAR
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
#else
fn fs_main(input : VertexOutput) -> @location(0) vec2<u32> {
#endif
  // the top left pixel of the 2 x 2 block. The half resolution rounds up, so it is always in the gBuffer
  let coord = vec2<u32>(input.position.xy) * 2u;
  let g0 = textureLoad(gBufferTexture, coord, 0).xy;
  if (g0.x == SURFACE_NONE) {
#ifdef HALF_RES_BILINEAR
    return vec4<f32>(1.0);
#else
    // no paths, no surface
    return vec2<u32>(pack2x16float(vec2(-1.)), 0u);
#endif
  }
  let g = vec3(g0, textureLoad(gBufferAtlasTexture, coord, 0).x);
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  let d = getGBufferDistance(textureLoad(gBufferDepth, coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
  let world = depthToPosition(d, uv * 2. - 1.);
  let n = getGBufferNormal(g);
  let surface = getGBufferSurface(g, world, uniforms.atlasStep);
  let levelF = getLevel(world, n);
#ifdef HALF_RES_BILINEAR
  let sun = LIGHTMAP_SUN;
#else
  // the shading takes no sun where the surface faces away
  let sun = LIGHTMAP_SUN && dot(n, uniforms.lightDirection) > 0.;
#endif
  // the sky (xy) and the sun (zw): the visibility and the paths
  var c : vec4<f32>;
  if (uniforms.stochasticLevel != 0u) {
    c = getLightmapChannelsStochastic(surface, levelF, getNoise(coord), sun);
  } else {
    c = getLightmapChannels(surface, levelF, sun);
  }
#ifdef HALF_RES_BILINEAR
  return vec4<f32>(select(n.y * .5 + .5, c.x, c.y > 0.0), select(1., c.z, c.w > 0.0), 0., 1.);
#else
  return vec2(pack2x16float(select(vec2(-1.), c.xz, c.yw > vec2(0.))), packHalfResKey(surface, u32(levelF)));
#endif
}
