// The buildings the water of MapWorld reflects, with fewer rays, see WaterReflectionHalfRes.ts: the ray of the
// deferred shading (getWaterReflectionRay of waterReflection.wgsl, traced with isOccluded of skyVisibilityTrace.wgsl)
// for 1 pixel per block of the gBuffer (REFLECTION_BLOCK_X x Y pixels), the one at getReflectionSampleOffset(phase). The
// shading blends the samples around a pixel (getHalfResReflection in deferredShading.wgsl) instead of tracing a ray per
// water pixel, which was more than half of its time at 4K.
// With REFLECTION_COLORS the trace gives where the ray hits, which is lit by the lightmap there (see
// waterReflectionColor.wgsl).
// Out, rg32uint. x: 4 x 8 bits (pack4x8unorm), its first 0 no water (no sample), .5 the ray reaches the sky, 1 it hits a
// building, then the ray (* .5 + .5): the shading weighs a sample by how alike its ray and the ray of the pixel are.
// y: the color of the building the ray hits (linear like the shading, packReflectionColor), 0 without one.
// 1 target: the shading has a texture less to bind, it has the 16 sampled textures a stage has by default
// Needs cameraRay.wgsl, the includes of the lightmap resolve (SkyVisibilityLightmap.resolveIncludes, its uniforms are set
// with the deferred shading's: SkyVisibilityLightmap.setResolveUniforms), waterReflection.wgsl,
// waterReflectionColor.wgsl, skyVisibilityTrace.wgsl and cityTraceDistance.wgsl

// the texels hold the sun visibility too, see SkyVisibilityLightmap.setSun
override LIGHTMAP_SUN : bool = false;

struct Uniforms {
  inverseProjection : mat4x4<f32>,
  inverseView : mat4x4<f32>,
  // the jittered camera of the gBuffer
  viewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  // of the gBuffer, the full resolution
  resolution : vec2<f32>,
  // see getGBufferDistance
  depthParams : vec2<f32>,
  // the gBuffer materials of the water, a bit each
  waterMaterials : u32,
  // which pixel of a block is traced, see getReflectionSampleOffset
  phase : u32,
  // the size of the tile of rippleTexture in world units (WATER_RIPPLES)
  rippleTile : f32,
  // the city the rays trace, see CityScene.bind
  cityTileCount : u32,
  cityRoot : u32,
  // the end of the shading, which the background color the colors are fogged to is turned back from (see
  // getLinearFromDisplay)
  backgroundColor : vec3<f32>,
  distanceFog : f32,
  contrast : f32,
  exposure : f32,
  // the light and the albedos of what the rays hit, see waterReflectionColor.wgsl
  skyColor : vec3<f32>,
  reflectionUnseen : f32,
  sunColor : vec3<f32>,
  shadowStrength : f32,
  lightDirection : vec3<f32>,
  reflectionLevelBias : f32,
  reflectionBuildingAlbedo : vec3<f32>,
  reflectionTreeAlbedo : vec3<f32>,
  // the lightmap, set with the other resolve materials, see SkyVisibilityLightmap.setResolveUniforms. Not all are used
  lightmapEnabled : u32,
  lightmapHalfRes : u32,
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
};

// the rays are mirrored on the ripples of rippleTexture, else on the flat water: as the shading's
override WATER_RIPPLES : bool = false;
// the tone map of the shading (deferredShading.wgsl), for getLinearFromDisplay
override TONE_MAP : bool = true;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// the surface id, the normal and the material of the gBuffer, and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(1) var gBufferTexture : texture_2d<u32>;
@group(0) @binding(2) var gBufferDepth : texture_depth_2d;
// the buildings, see CityScene and skyVisibilityTrace.wgsl
@group(0) @binding(3) var<storage, read> cityBuffer : array<vec4<f32>>;
// the ripples of the water, see WaterRippleTexture.ts
@group(0) @binding(4) var rippleTexture : texture_2d<f32>;
@group(0) @binding(5) var rippleSampler : sampler;
// the sky visibility lightmap, see SkyVisibilityLightmap
@group(0) @binding(6) var<storage, read> tileBuffer : array<u32>;
@group(0) @binding(7) var<storage, read> lightingBuffer : array<u32>;

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

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec2<u32> {
  // the size of the target rounds up: a block that is cut off by the edge has its sample at the last pixel
  let block = vec2(REFLECTION_BLOCK_X, REFLECTION_BLOCK_Y);
  let coord = min(vec2<u32>(input.position.xy) * block + getReflectionSampleOffset(uniforms.phase), vec2<u32>(uniforms.resolution) - 1u);
  let g = vec3(textureLoad(gBufferTexture, coord, 0).xy, 0u);
  if (g.x == SURFACE_NONE || ((uniforms.waterMaterials >> getGBufferMaterial(g)) & 1u) == 0u) {
    return vec2(0u);
  }
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  let d = getGBufferDistance(textureLoad(gBufferDepth, coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
  let world = depthToPosition(d, uv * 2. - 1.);
  let rd = getWaterReflectionRay(getGBufferNormal(g), world, d);
  if (!REFLECTION_COLORS) {
    return vec2(pack4x8unorm(vec4(select(.5, 1., isOccluded(world, rd)), rd * .5 + .5)), 0u);
  }
  let hit = getOcclusionHit(world, rd);
  if (hit.t >= 0.) {
    let color = getReflectionHitColor(world, rd, hit, d, uniforms.lightmapEnabled != 0u, LIGHTMAP_SUN && uniforms.lightmapEnabled != 0u);
    return vec2(pack4x8unorm(vec4(1., rd * .5 + .5)), packReflectionColor(color));
  }
  return vec2(pack4x8unorm(vec4(.5, rd * .5 + .5)), 0u);
}
