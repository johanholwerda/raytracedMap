// The shading of MapWorld from its gBuffer (see MapGBuffer.ts), once per pixel: the forward shading of
// shaders/shading.wgsl (sun with the VSM cascades, sky times the albedo, fog, dither, selection highlight) with the
// sky visibility lightmap in place of the sky term when it is enabled. The forward pass did the lightmap
// gather for every fragment of the overdraw, which made it the most expensive pass.
// The water reflects the buildings: a ray from a water pixel, mirrored on its normal, traced against the city
// (isOccluded of skyVisibilityTrace.wgsl) darkens the pixel when it hits a building. The ray is mirrored on the animated
// micro normal of the ripples (rippleTexture, see WaterRippleTexture.ts), which makes the reflections wobble. A ray
// that misses the buildings reflects the sun where it points at it: the glints on the ripples. The rays are traced here, a
// ray per water pixel, or by the half resolution pass before it (REFLECTIONS_HALF_RES, see WaterReflectionHalfRes.ts).
// What a ray hits the water mirrors in its color, lit by the lightmap there (REFLECTION_COLORS, waterReflectionColor.wgsl).
// The sun shadows (SUN_SHADOWS) are the VSM cascades, a ray to the sun traced against the same city (getRayTracedShadow),
// or the sun visibility the lightmap texels hold (the zw of getLightmapLookup: from the half resolution samples, or in
// the lookup of the sky visibility).
// Needs hash.wgsl, r2.wgsl, toneMap.wgsl, cameraRay.wgsl, waterReflection.wgsl, waterReflectionColor.wgsl, skyVisibilityTrace.wgsl and the includes of the lightmap resolve (see SkyVisibilityLightmap.resolveIncludes):
// the lightmap holds the part of the sky a point sees, times the sky color here

struct Uniforms {
  inverseProjection : mat4x4<f32>,
  inverseView : mat4x4<f32>,
  viewProjection : mat4x4<f32>,
  // the sun's view of the ground for the ray traced shadows (SUN_SHADOWS 2 and 3), see terrainSun.wgsl
  terrainSunViewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  resolution : vec2<f32>,
  lightDirection : vec3<f32>,
  skyColor : vec3<f32>,
  sunColor : vec3<f32>,
  backgroundColor : vec3<f32>,
  distanceFog : f32,
  // where the fog is 1, see getFog
  viewDistanceMax : f32,
  contrast : f32,
  // scales the linear color before the tone map (TONE_MAP)
  exposure : f32,
  shadowStrength : f32,
  VSMBleedThreshold : f32,
  varianceMin : f32,
  // for the noise of stochasticLevel
  frame : f32,
  // the sky visibility lightmap (LIGHTMAP_ENABLED), see SkyVisibilityLightmap.setResolveUniforms
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
  // the part of a level in which the 2 levels around it blend, see getLevelBlend
  levelBlend : f32,
  // the step of the wall atlas positions in the gBuffer, see MapGBuffer.atlasStep
  atlasStep : f32,
  // see getGBufferDistance
  depthParams : vec2<f32>,
  // the water reflections (WATER_REFLECTIONS): the gBuffer materials of the water, a bit each
  waterMaterials : u32,
  // how much darker the water is where its reflection hits a building
  reflectionDarkening : f32,
  // the city the reflections trace, see CityScene.bind
  cityTileCount : u32,
  cityRoot : u32,
  // the size of the tile of rippleTexture in world units (WATER_RIPPLES), see getRippleNormal
  rippleTile : f32,
  // the strength of the ripples 4 times smaller that fade in close up (WATER_RIPPLES), see getRippleNormal
  rippleDetail : f32,
  // the brightness of the sun reflected in the water, times the sun color
  sunReflection : f32,
  // the phong exponent of the reflection of the sun: higher is a smaller, sharper glint per ripple
  sunReflectionPower : f32,
  // the sun in the sky: the angular radius of its disc in radians, and the brightness of the glow around it
  sunSize : f32,
  sunGlow : f32,
  // the angular radius of the sun the soft ray traced shadows sample (SOFT_SHADOWS), in radians
  shadowSunRadius : f32,
  // the SSAO on the sky term (SSAO_SKY), see getSSAO: its amount, and the samples over which it fades in
  ssaoAmount : f32,
  ssaoFadeCount : f32,
  // which pixel of its 2 x 2 blocks the half resolution reflection pass traced (REFLECTIONS_HALF_RES), see
  // getReflectionSampleOffset
  reflectionPhase : u32,
  // the sky the water mirrors (WATER_REFLECTIONS), weighted by the fresnel term of the water (Schlick, WATER_F0): scales
  // the fresnel weight, 0 no sky in the water
  skyReflection : f32,
  // how much brighter the glints of the sun get at grazing angles, by the same fresnel term: 0 the same at every angle
  sunFresnel : f32,
  // how far a point goes off its surface before it is compared with terrainSunMap, see getTerrainSunShadow
  terrainSunOffset : f32,
  // the ground in the water reflections (REFLECT_TERRAIN), see terrainHorizonLookup.wgsl
  terrainReflectionTop : f32,
  terrainReflectionBias : f32,
  // how much of the light the water loses where it reflects a building (reflectionDarkening) the color of the building
  // gives back (REFLECTION_COLORS): 0 dark as without it, 1 all of it
  reflectionColor : f32,
  // the sky light of a building in the water that the lightmap has no texels of, and the albedos of the buildings and
  // the trees in the water (as the map color params, not squared), see getUnseenReflectionColor
  reflectionUnseen : f32,
  reflectionBuildingAlbedo : vec3<f32>,
  reflectionTreeAlbedo : vec3<f32>,
  // the lightmap levels the reflections read coarser than the screen claims, see getReflectionLevel
  reflectionLevelBias : f32,
};

// the reflectance of water seen straight down (index of refraction 1.33)
const WATER_F0 = .02;

// the debug view of the lightmap, see LightmapView in SkyVisibilityLightmap.ts: 0 shaded, 1 the indirect light only, 2
// the levels, 3 the texels still converging, 4 the lightmap itself (see getLightmapViewColor), 5 the tiles and texels.
// A constant, so the shaded view has none of the branches of the others
override LIGHTMAP_VIEW : u32 = 0u;
// the half resolution lightmap from halfResFiltered, bilinearly across the surfaces (getHalfResLightingBilinear), instead
// of the samples of halfResTexture on the surface of the pixel (getHalfResLighting)
override HALF_RES_BILINEAR : bool = false;
// The feature switches: constants (set by the host, a pipeline per combination), so the shader of a setting has none of
// the code of the others.
// the sky visibility lightmap in place of the sky term, see SkyVisibilityLightmap.setResolveUniforms
override LIGHTMAP_ENABLED : bool = false;
// the lightmap from the half resolution samples of halfResTexture, see getHalfResLighting
override LIGHTMAP_HALF_RES : bool = false;
// a pixel reads 1 of the 2 levels around its level, picked by noise: the TAA averages it, see getLightmapLightingStochastic
override STOCHASTIC_LEVEL : bool = false;
// the water reflections, of the materials in waterMaterials, see DeferredShading.setWaterReflections
override WATER_REFLECTIONS : bool = false;
// the reflection rays are traced against the buildings, else only the sun is reflected
override REFLECT_BUILDINGS : bool = false;
// the reflections are mirrored on the ripples of rippleTexture, else on the flat water
override WATER_RIPPLES : bool = false;
// the ground in the reflections of the buildings (REFLECT_BUILDINGS), where it is high enough above the water to show
// (most places it isn't): see DeferredShading.setTerrainReflection and getTerrainReflection
override REFLECT_TERRAIN : bool = false;
// REFLECT_BUILDINGS from the samples of reflectionTexture (with their colors with REFLECTION_COLORS), 1 traced ray per
// block of pixels (see getHalfResReflection), instead of a ray per water pixel here
override REFLECTIONS_HALF_RES : bool = false;
// the debug views of the sun shadows (MapWorld's drawMode): 0 off, 1 the shadow lookup (getSunShadow) alone, black on the
// faces away from the sun, 2 the
// cascade of each pixel in a color (cascadeDebugColor)
override SHADOW_VIEW : u32 = 0u;
// the linear color, times the exposure, goes through acesTonemap (toneMap.wgsl) before the gamma: the sun, the sky and the glints add up
// to more than 1, which the output would clip. Else it is only clamped by the contrast's smoothstep
override TONE_MAP : bool = true;
// where the sun shadows come from, ShadowMode of MapWorldSettings.ts: 0 none, 1 the VSM cascades (getShadowMap), 2 the sun
// visibility of the lightmap texels (getLightmapLookup, the rays while the lightmap has no buffers: LIGHTMAP_ENABLED
// false), 3 a ray per pixel against the city (getRayTracedShadow)
override SUN_SHADOWS : u32 = 1u;
// the rays sample the disc of the sun, a point per pixel and frame, which the TAA averages into a penumbra. Else they go
// to its center: hard shadows without noise, for the image without the TAA
override SOFT_SHADOWS : bool = false;
// the SSAO (ssaoTexture) darkens only the sky term, see getSSAO. Else it is multiplied over the whole image after the TAA
// (reprojectedSSAOComposite.wgsl), the sun too
override SSAO_SKY : bool = false;

struct ShadowMapData {
  shadowMapViewProjection : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var shadowMapSampler : sampler;
// the 4 cascades, one layer each (ShadowMap.cascadeTexture)
@group(0) @binding(2) var shadowMaps : texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> shadowMapData : array<ShadowMapData>;
@group(0) @binding(7) var<storage, read> tileBuffer : array<u32>;
// the sky visibility lightmap: 1 word per texel, see SkyVisibilityLightmap
@group(0) @binding(8) var<storage, read> lightingBuffer : array<u32>;
// the gBuffer: its 2 targets and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(10) var gBufferTexture : texture_2d<u32>;
@group(0) @binding(12) var gBufferAtlasTexture : texture_2d<u32>;
@group(0) @binding(13) var gBufferDepth : texture_depth_2d;
// the albedo of each material of the gBuffer, see MapGBuffer.paletteBuffer
@group(0) @binding(11) var<storage, read> materialPalette : array<vec4<f32>>;
// the sky visibility lightmap at half resolution, see skyVisibilityHalfRes.wgsl
@group(0) @binding(14) var halfResTexture : texture_2d<u32>;
// the sky and the sun visibility alone (rg16float), for HALF_RES_BILINEAR. Bilinear, clamped to the edge
@group(0) @binding(18) var halfResFiltered : texture_2d<f32>;
@group(0) @binding(19) var halfResSampler : sampler;
// the buildings the water reflections trace, see CityScene and skyVisibilityTrace.wgsl
@group(0) @binding(17) var<storage, read> cityBuffer : array<vec4<f32>>;
// the ripples of the water, the tiling gradient of their height (dh/dx, dh/dz), see WaterRippleTexture.ts
@group(0) @binding(20) var rippleTexture : texture_2d<f32>;
@group(0) @binding(21) var rippleSampler : sampler;
// 2d blue noise for the points on the sun of the soft ray traced shadows (rg16unorm), see BlueNoiseTexture2D16Bits
@group(0) @binding(22) var blueNoiseTexture : texture_2d<f32>;
// the SSAO of this frame (SSAO_SKY): (visibility, sample count, camera distance, 0), see reprojectedSSAO.wgsl
@group(0) @binding(23) var ssaoTexture : texture_2d<f32>;
// the reflection rays of the water at half resolution (REFLECTIONS_HALF_RES): x whether there is a sample, the part of
// the ray that hits a building and the ray (packReflectionSample), y the color of what it hits (REFLECTION_COLORS), see
// waterReflectionHalfRes.wgsl
@group(0) @binding(24) var reflectionTexture : texture_2d<u32>;
// the ground as the sun sees it (TerrainSunMap.ts): the hills for the ray traced sun shadows, see terrainSun.wgsl
@group(0) @binding(25) var terrainSunMap : texture_2d<f32>;
// the height map of the ground it is made from, for the ground in the water reflections, see TerrainSunMap.bindHeights
@group(0) @binding(26) var terrainHeights : texture_2d<f32>;
@group(0) @binding(27) var terrainMaxHeights : texture_2d<f32>;
@group(0) @binding(28) var terrainMaxHeightsFine : texture_2d<f32>;
// the horizon of the ground per block of pixels and the color of it, see WaterTerrainHorizon.ts
@group(0) @binding(29) var terrainHorizon : texture_2d<f32>;

// SSAO_SKY: the part of the sky the SSAO sees around the pixel, relative to an open point (1), like the composite
// (reprojectedSSAOComposite.wgsl) applies it: faded in over its first samples, times its amount. No fog fade: the fog
// is mixed in after the shading. 1 without SSAO_SKY
fn getSSAO(coord : vec2<u32>) -> f32 {
  if (!SSAO_SKY) {
    return 1.0;
  }
  let s = textureLoad(ssaoTexture, coord, 0);
  let strength = clamp(s.y / uniforms.ssaoFadeCount, 0.0, 1.0) * uniforms.ssaoAmount;
  return mix(1.0, clamp(s.x, 0.0, 1.0), strength);
}

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

// --- Variance shadow map sampling, as in shading.wgsl -------------------------

fn shadowUVDepthStrength(p : vec3<f32>, i : u32) -> vec4<f32> {
  let screen = shadowMapData[i].shadowMapViewProjection * vec4<f32>(p, 1.0);
  let uv = screen.xy / screen.w;
  let depth = -p.y;
  let s2 = vec2<f32>(1.) - smoothstep(vec2<f32>(0.4), vec2<f32>(1.0), abs(uv));
  return vec4<f32>(uv, depth, min(s2.x, s2.y));
}

fn sampleShadowMoments(i : u32, uv : vec2<f32>) -> vec2<f32> {
  return textureSampleLevel(shadowMaps, shadowMapSampler, uv, i, 0.0).xy;
}

fn shadowCascade(uv : vec2<f32>, i : u32, dist : f32) -> f32 {
  var shadowMapUV = uv * 0.5 + 0.5;
  shadowMapUV.y = 1.0 - shadowMapUV.y;
  let moments = sampleShadowMoments(i, shadowMapUV);
  let lit = step(dist, moments.x);
  var variance = moments.y - moments.x * moments.x;
  variance = max(variance, uniforms.varianceMin * uniforms.varianceMin);
  let d = dist - moments.x;
  let s = variance / (variance + d * d);
  return smoothstep(uniforms.VSMBleedThreshold, 1.0, max(lit, s));
}

// how many texels of a cascade a point goes off its surface along its normal before the lookup: the ground at a low sun
// spans many texels along the sun (tens of meters each in the far cascades), its own heights in them would shadow it
const SHADOW_NORMAL_OFFSET = 1.5;

// the size of a texel of cascade i in world units: its half-extent is 1 over the length of the first row of its matrix
fn shadowTexelSize(i : u32) -> f32 {
  let m = shadowMapData[i].shadowMapViewProjection;
  return 2.0 / (f32(textureDimensions(shadowMaps).x) * length(vec3<f32>(m[0].x, m[1].x, m[2].x)));
}

// The cascades front to back, each taking its weight (1 inside, fading out over its edge) of what the ones before it
// left. Each cascade is re-rendered around a center of its own, so the next one need not cover the edge of the one
// before it: that point goes on to the first one that does. What no cascade covers is lit, so past the last cascade the
// shadows fade out instead of ending at a hard edge.
fn getShadowMap(p : vec3<f32>, n : vec3<f32>) -> f32 {
  var shadow = 0.0;
  var remaining = 1.0;
  for (var i = 0u; i < 4u; i++) {
    let u = shadowUVDepthStrength(p + n * (SHADOW_NORMAL_OFFSET * shadowTexelSize(i)), i);
    if (u.w > 0.0) {
      shadow += remaining * u.w * shadowCascade(u.xy, i, u.z);
      remaining *= 1.0 - u.w;
      if (remaining <= 0.0) {
        break;
      }
    }
  }
  return shadow + remaining;
}

// the offset of the origin of a shadow ray from its surface along the normal, times the camera distance: the position from
// the depth is exact to about 1e-7 of the distance, the normal of the gBuffer to about 1e-4
const SHADOW_RAY_BIAS = 1e-4;

// SUN_SHADOWS 3: 1 where the sun reaches the point world (with normal n, d from the camera), 0 where a building
// or a tree is in the way: a ray traced against the city (isOccluded of skyVisibilityTrace.wgsl), the buildings and the
// trees the shadow map draws. SOFT_SHADOWS: to a point on the disc of the sun, shadowSunRadius around lightDirection,
// from 2d blue noise per pixel moved along the R2 sequence every frame. Each pixel takes 1 ray per frame, the TAA
// averages them: blended like motion while the view changes, its static average converges the penumbra (STATIC_FRAMES
// of TAA.ts). Else to the center of the sun: the hard shadow, the same every frame
fn getRayTracedShadow(world : vec3<f32>, n : vec3<f32>, d : f32, coord : vec2<u32>) -> f32 {
  let l = uniforms.lightDirection;
  var rd = l;
  if (SOFT_SHADOWS) {
    let size = textureDimensions(blueNoiseTexture);
    let r = fract(textureLoad(blueNoiseTexture, coord % size, 0).xy + getR2(u32(uniforms.frame)));
    let t = normalize(cross(l, select(vec3<f32>(0., 1., 0.), vec3<f32>(1., 0., 0.), abs(l.y) > .5)));
    let b = cross(l, t);
    // uniform on the disc. Small angles: the tangent offset is the angle
    let radius = uniforms.shadowSunRadius * sqrt(r.x);
    let phi = 6.28318531 * r.y;
    rd = normalize(l + (t * cos(phi) + b * sin(phi)) * radius);
  }
  let ro = world + n * (d * SHADOW_RAY_BIAS);
  return select(1., 0., isOccluded(ro, rd));
}

// the sun the point gets, 0 to 1, see SUN_SHADOWS. lightmapSun: with SUN_SHADOWS 2 and LIGHTMAP_ENABLED, the sun
// visibility of the lightmap texels of the surface point and the paths it holds (zw of getLightmapLookup), lit where it
// has none yet. The rays trace only the buildings and the trees: the hills come from the sun's view of the ground
// (getTerrainSunShadow). The cascades have the ground in them
fn getSunShadow(world : vec3<f32>, n : vec3<f32>, d : f32, coord : vec2<u32>, lightmapSun : vec2<f32>) -> f32 {
  if (SUN_SHADOWS == 0u) {
    return 1.;
  }
  if (SUN_SHADOWS == 2u && LIGHTMAP_ENABLED) {
    return select(1., lightmapSun.x, lightmapSun.y > 0.) * getTerrainSunShadow(world, n);
  }
  if (SUN_SHADOWS >= 2u) {
    return getRayTracedShadow(world, n, d, coord) * getTerrainSunShadow(world, n);
  }
  return getShadowMap(world, n);
}

fn cascadeDebugColor(p : vec3<f32>) -> vec3<f32> {
  for (var i = 0u; i < 4u; i++) {
    let u = shadowUVDepthStrength(p, i);
    if (u.w > 0.0) {
      var col = vec3<f32>(0.4);
      if (i == 0u) { col = vec3<f32>(1.0, 0.3, 0.3); }
      else if (i == 1u) { col = vec3<f32>(0.3, 1.0, 0.3); }
      else if (i == 2u) { col = vec3<f32>(0.3, 0.5, 1.0); }
      else { col = vec3<f32>(1.0, 1.0, 0.3); }
      return col * (0.4 + 0.6 * u.w);
    }
  }
  return vec3<f32>(0.25);
}

// interleaved gradient noise, moved every frame: a repeat of 64 frames
fn getNoise(coord : vec2<u32>) -> f32 {
  let p = vec2<f32>(coord) + 5.588238 * (uniforms.frame % 64.);
  return fract(52.9829189 * fract(dot(p, vec2(.06711056, .00583715))));
}

// the sky and the sun as getLightmapChannels returns them (the paths only as more than 0 or not), from the samples of the
// half resolution pass around the pixel (skyVisibilityHalfRes.wgsl), bilinearly: the pixel of the sample of a 2 x 2
// block reads only that one, the others 2 or 4 at 1 pixel from them. Only the samples of the surface point of the pixel
// (matchesHalfResKeys) count, and per channel only the ones with paths, like the texels of gather. When none of the
// samples is on the surface point, the ones with paths count, whatever their surface: the lighting bleeds across
// the edge. Straight-line: the 4 loads first (a sample of weight 0 is loaded too, and doesn't count), then the tests and
// the sums on all 4 at once. The loop with a branch per sample was 1.27x slower for the same result
fn getHalfResLighting(coord : vec2<u32>, surface : Surface) -> vec4<f32> {
  let last = textureDimensions(halfResTexture) - 1u;
  let h0 = min(coord / 2u, last);
  let odd = coord & vec2(1u);
  let h1 = min(h0 + odd, last);
  let f = vec2<f32>(odd) * .5;
  let s00 = textureLoad(halfResTexture, h0, 0).xy;
  let s10 = textureLoad(halfResTexture, vec2(h1.x, h0.y), 0).xy;
  let s01 = textureLoad(halfResTexture, vec2(h0.x, h1.y), 0).xy;
  let s11 = textureLoad(halfResTexture, h1, 0).xy;
  let w = vec4((1. - f.x) * (1. - f.y), f.x * (1. - f.y), (1. - f.x) * f.y, f.x * f.y);
  let onSurface = matchesHalfResKeys(vec4(s00.y, s10.y, s01.y, s11.y), surface) & (w > vec4(0.));
  // the sky and the sun visibility per sample, -1 without paths
  let v0 = unpack2x16float(s00.x);
  let v1 = unpack2x16float(s10.x);
  let v2 = unpack2x16float(s01.x);
  let v3 = unpack2x16float(s11.x);
  let skyV = vec4(v0.x, v1.x, v2.x, v3.x);
  let sunV = vec4(v0.y, v1.y, v2.y, v3.y);
  let matched = any(onSurface);
  // the weights of the samples on the surface, or of all of them on a miss, per channel of the ones with paths
  let wm = select(w, select(vec4(0.), w, onSurface), matched);
  let wSky = select(vec4(0.), wm, skyV >= vec4(0.));
  let wSun = select(vec4(0.), wm, sunV >= vec4(0.));
  let skyW = dot(wSky, vec4(1.));
  let sunW = dot(wSun, vec4(1.));
  return vec4(select(0., dot(skyV, wSky) / skyW, skyW > 0.), skyW, select(0., dot(sunV, wSun) / sunW, sunW > 0.), sunW);
}

// HALF_RES_BILINEAR: the sky (x) and the sun visibility (y) of the half resolution samples (skyVisibilityHalfRes.wgsl),
// bilinearly, whatever their surface. Of an open sky and lit where the texels have no paths yet. A sample is of the top
// left pixel of its 2 x 2 block: the pixel at full resolution position x is at (x - .5) / 2 + .5 in the half resolution
// texels
fn getHalfResLightingBilinear(coord : vec2<u32>) -> vec2<f32> {
  let uv = (vec2<f32>(coord) * .5 + .5) / vec2<f32>(textureDimensions(halfResFiltered));
  return textureSampleLevel(halfResFiltered, halfResSampler, uv, 0.0).xy;
}

// The lightmap at the pixel: the sky visibility (x) and the paths it holds (y, 0 without), and with SUN_SHADOWS 2 the same
// of the sun (zw): from the half resolution samples, or looked up here, both channels in 1 lookup (getLightmapChannels:
// the same tile probes). sun: the sun is needed (the surface faces it)
fn getLightmapLookup(coord : vec2<u32>, surface : Surface, levelF : f32, sun : bool) -> vec4<f32> {
  if (LIGHTMAP_HALF_RES) {
    if (HALF_RES_BILINEAR) {
      let s = getHalfResLightingBilinear(coord);
      return vec4<f32>(s.x, 1., s.y, 1.);
    }
    return getHalfResLighting(coord, surface);
  }
  let lookupSun = SUN_SHADOWS == 2u && sun;
  if (STOCHASTIC_LEVEL) {
    return getLightmapChannelsStochastic(surface, levelF, getNoise(coord), lookupSun);
  }
  return getLightmapChannels(surface, levelF, lookupSun);
}

// LIGHTMAP_VIEW 4: the lightmap itself instead of the scene, as a square with black borders next to it. The tile slots
// laid out in a grid, an empty slot dark red. As in resolveVirtualLightmap.wgsl of the irradiance cache
fn getLightmapViewColor(pixel : vec2<f32>) -> vec3<f32> {
  let side = min(uniforms.resolution.x, uniforms.resolution.y);
  let p = (pixel - (uniforms.resolution - side) * .5) / side;
  if (any(p < vec2(0.)) || any(p >= vec2(1.))) {
    return vec3(0.);
  }
  let tilesPerSide = LIGHTMAP_SIZE / TILE_SIZE;
  let texel = vec2<u32>(p * f32(LIGHTMAP_SIZE));
  let tile = texel / TILE_SIZE;
  let tileIndex = tile.y * tilesPerSide + tile.x;
  if (tileIndex >= TILE_COUNT || tileBuffer[tileIndex] == 0u) {
    return vec3(.5, 0., 0.);
  }
  let t = texel % TILE_SIZE;
  let v = lightingBuffer[(tileIndex * TILE_SIZE + t.y) * TILE_SIZE + t.x];
  let c = vec3(select(unpack2x16float(v).x, decodePackedTexel(v).x, PACKED_TEXELS));
  // gamma like the shading, clamped like its output
  return clamp(sqrt(max(c, vec3(0.))), vec3(0.), vec3(1.));
}

// how fast the weight of a half resolution reflection sample falls with the difference between its ray and the ray of the
// pixel: exp(-REFLECTION_RAY_MATCH * |difference|^2), at 256 half at a difference of 3 degrees
const REFLECTION_RAY_MATCH = 256.;

// the weight of a sample of reflectionTexture for a pixel with the reflection ray rd, before its bilinear weight: 0 off the
// water (no sample), else by how alike the 2 rays are. Never 0 on the water, so samples that all differ still count
fn getReflectionSampleWeight(s : vec4<f32>, rd : vec3<f32>) -> f32 {
  let difference = s.yzw * 2. - 1. - rd;
  return step(.5 / 255., s.x) * (exp(-REFLECTION_RAY_MATCH * dot(difference, difference)) + 1e-6);
}

// REFLECTIONS_HALF_RES: the part of the reflection of the water pixel (its ray rd) that hits a building, 0 to 1, from the
// rays the reduced pass traced around it (waterReflectionHalfRes.wgsl): its samples are at the pixel
// getReflectionSampleOffset of each block, so a pixel is on a sample or between 2 or 4 of them. Bilinearly, times
// how alike the ray of the sample and rd are: the ripples turn the rays of neighbouring pixels degrees apart, and what a
// ray hits follows its direction more than its pixel. Without it the reflections are a pixel softer. Only the samples on
// the water count, none: no building. A sample hits by the part of its hit the fog leaves (getReflectionSampleHit). The
// colors of the samples (times that part, 0 where a ray hit no building) with the same weights: the rgb of the result,
// its a the part that hits
fn getHalfResReflection(coord : vec2<u32>, rd : vec3<f32>) -> vec4<f32> {
  let last = vec2<i32>(textureDimensions(reflectionTexture)) - 1;
  let block = vec2<i32>(vec2(REFLECTION_BLOCK_X, REFLECTION_BLOCK_Y));
  // from the sample of the first block, which the first pixels are before: + block keeps the division rounding down
  let p = vec2<i32>(coord) - vec2<i32>(getReflectionSampleOffset(uniforms.reflectionPhase)) + block;
  let h = p / block - 1;
  let between = p % block;
  let h0 = clamp(h, vec2(0), last);
  let h1 = clamp(h + min(between, vec2(1)), vec2(0), last);
  let f = vec2<f32>(between) / vec2<f32>(block);
  let t00 = textureLoad(reflectionTexture, h0, 0).xy;
  let t10 = textureLoad(reflectionTexture, vec2(h1.x, h0.y), 0).xy;
  // blocks of 1 pixel along y: a pixel is between 2 samples of its row, the other 2 have no weight
  var t01 = t00;
  var t11 = t10;
  if (REFLECTION_BLOCK_Y > 1u) {
    t01 = textureLoad(reflectionTexture, vec2(h0.x, h1.y), 0).xy;
    t11 = textureLoad(reflectionTexture, h1, 0).xy;
  }
  let s00 = unpack4x8unorm(t00.x);
  let s10 = unpack4x8unorm(t10.x);
  let s01 = unpack4x8unorm(t01.x);
  let s11 = unpack4x8unorm(t11.x);
  let w = vec4((1. - f.x) * (1. - f.y), f.x * (1. - f.y), (1. - f.x) * f.y, f.x * f.y) * vec4(
    getReflectionSampleWeight(s00, rd), getReflectionSampleWeight(s10, rd), getReflectionSampleWeight(s01, rd), getReflectionSampleWeight(s11, rd));
  let sum = w.x + w.y + w.z + w.w;
  if (sum <= 0.) {
    return vec4(0.);
  }
  let occluded = dot(w, vec4(getReflectionSampleHit(s00), getReflectionSampleHit(s10), getReflectionSampleHit(s01),
    getReflectionSampleHit(s11))) / sum;
  if (!REFLECTION_COLORS) {
    return vec4(0., 0., 0., occluded);
  }
  let color = unpackReflectionColor(t00.y) * w.x + unpackReflectionColor(t10.y) * w.y + unpackReflectionColor(t01.y) * w.z +
    unpackReflectionColor(t11.y) * w.w;
  return vec4(color / sum, occluded);
}

// The distance fog: exponential (distanceFog per world unit), with a smoothstep from .7 of viewDistanceMax that takes
// it to 1 at viewDistanceMax, where the loaded map ends (see MapRenderer.viewDistanceMax). 0 (not set): only the exp.
// d is the horizontal (xz) distance to the camera, not the 3d one: the map is loaded in a radius around the camera in
// xz, and a camera that goes up and looks down gets no more fog (the far plane reaches past it, see
// MapWorld.applyViewDistanceMax).
// The same in deferredShading.wgsl, cars.wgsl, reprojectedSSAOComposite.wgsl and shading.wgsl.
fn getFog(d : f32) -> f32 {
  let end = select(1., 1. - smoothstep(.7 * uniforms.viewDistanceMax, uniforms.viewDistanceMax, d), uniforms.viewDistanceMax > 0.);
  return 1. - exp(-d * uniforms.distanceFog) * end;
}

// The background color with the sun disc and a glow around it, in the sun color brightened to its strongest channel.
// The edge of the disc is soft over a fixed angle (no fwidth: the sky branch is not uniform control flow), the TAA
// smooths it further.
fn getSky(coord : vec2<u32>) -> vec3<f32> {
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  return getSkyColor(getRay(uv * 2. - 1.), true);
}

// the sky along rd, as getSky draws it (gamma space): with the disc of the sun, or only its glow (the water: the glints
// are its disc)
fn getSkyColor(rd : vec3<f32>, withDisc : bool) -> vec3<f32> {
  let angle = acos(clamp(dot(rd, uniforms.lightDirection), -1., 1.));
//  let sunTint = uniforms.sunColor / max(max(max(uniforms.sunColor.r, uniforms.sunColor.g), uniforms.sunColor.b), 1e-4);
  let sunTint = uniforms.sunColor * 4.;
  let disc = select(0., 1. - smoothstep(uniforms.sunSize * .85, uniforms.sunSize, angle), withDisc);
  let glow = uniforms.sunGlow * exp(-angle / max(uniforms.sunSize * 4., 1e-4));
  let c = uniforms.backgroundColor + sunTint * glow;
  return min(mix(c, sunTint, disc), vec3<f32>(1.));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  if (LIGHTMAP_VIEW == 4u && LIGHTMAP_ENABLED) {
    return vec4<f32>(getLightmapViewColor(input.position.xy), 1.0);
  }
  let coord = vec2<u32>(input.position.xy);
  let g0 = textureLoad(gBufferTexture, coord, 0).xy;
  // the sky: the background color, like the clear color of the forward pass, with the sun in it
  if (g0.x == SURFACE_NONE) {
    return vec4<f32>(getSky(coord), 1.0);
  }
  let g = vec3(g0, textureLoad(gBufferAtlasTexture, coord, 0).x);
  let materialIndex = getGBufferMaterial(g);
  let albedoSample = materialPalette[materialIndex].rgb;
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  let d = getGBufferDistance(textureLoad(gBufferDepth, coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
  let world = depthToPosition(d, uv * 2. - 1.);
  // the fog is of the horizontal distance, see getFog
  let fog = getFog(distance(world.xz, uniforms.cameraPosition.xz));
  // all fog (beyond viewDistanceMax): the background color, which the full shading ends on too (the contrast is before
  // the fog). Skips the lightmap lookup, the shadows and the reflections. Not in the debug views
  if (SHADOW_VIEW == 0u && LIGHTMAP_VIEW == 0u && fog >= 1.) {
    let fogged = uniforms.backgroundColor;
    return vec4<f32>(select(fogged, mix(fogged, vec3<f32>(0.9, 0.1, 0.1), 0.6), isGBufferSelected(g)), 1.0);
  }
  let n = getGBufferNormal(g);
  let l = uniforms.lightDirection.xyz;
  let nDotL = max(dot(n, l), 0.0);
  // the lightmap: the sky visibility (xy) and with SUN_SHADOWS 2 the sun's (zw), see getLightmapLookup. A surface that
  // faces away from the sun gets none of it: no sun lookup
  var surface = Surface(0u, vec2<f32>(0.));
  var lightmap = vec4<f32>(0.);
  if (LIGHTMAP_ENABLED) {
    surface = getGBufferSurface(g, world, uniforms.atlasStep);
    lightmap = getLightmapLookup(coord, surface, getLevel(world, n), nDotL > 0.0);
  }

  if (SHADOW_VIEW == 1u) {
    // a face away from the sun is unlit: black, the same at full and at half resolution (the half resolution pass has no
    // sun there, see skyVisibilityHalfRes.wgsl)
    return vec4<f32>(vec3<f32>(select(0.0, getSunShadow(world, n, d, coord, lightmap.zw), nDotL > 0.0)), 1.0);
  } else if (SHADOW_VIEW == 2u) {
    return vec4<f32>(cascadeDebugColor(world), 1.0);
  }

  let skyColor = uniforms.skyColor * uniforms.skyColor;
  let sunColor = uniforms.sunColor * uniforms.sunColor;
  let albedo = albedoSample * albedoSample;
  var shadow = 1.0;
  if (nDotL > 0.0) {
    shadow = mix(1.0, getSunShadow(world, n, d, coord, lightmap.zw), uniforms.shadowStrength);
  }

  var c = albedo * sunColor * (nDotL * shadow);
  if (LIGHTMAP_ENABLED) {
    // the part of the sky the point sees, cosine weighted. No texel yet: that of an open sky, as in the sky term
    let openSky = n.y * .5 + .5;
    var sky = select(openSky, lightmap.x, lightmap.y > 0.0);
    if (SSAO_SKY) {
      // the SSAO is relative to the open sky. The smaller of the 2, not their product: where the texels are fine enough
      // they hold the occlusion the SSAO sees too, the SSAO only adds the contact occlusion they are too coarse for
      sky = min(sky, openSky * getSSAO(coord));
    }
    let indirect = skyColor * sky;
    c += albedo * indirect;
    // the debug views, see LightmapView in SkyVisibilityLightmap.ts. A constant: the shaded view (0) has none of this
    if (LIGHTMAP_VIEW != 0u) {
      let view = LIGHTMAP_VIEW;
      if (view == 1u) {
//        c = indirect;
        c = vec3(lightmap.x);
      } else if (view == 2u) {
        c = mix(c, fract(vec3<f32>(f32(u32(getLevel(world, n)))) * vec3<f32>(.31, .57, .73)), .5);
      } else if (view == 3u && getLightmapTexelCount(surface, u32(getLevel(world, n))) < uniforms.accumulationMax) {
        c = mix(c, vec3<f32>(1.0, 1.0, 0.0), .5);
      } else if (view == 5u) {
        // the tiles of the level of the pixel in a color each, and its texels as a checkerboard
        let texel = getTexel(surface, u32(getLevel(world, n)));
        let tile = texel / TILE_SIZE;
        let tileColor = vec3<f32>((vec3(tileHash(tile.x ^ tileHash(tile.y ^ tileHash(surface.id)))) >> vec3(0u, 8u, 16u)) & vec3(255u)) / 255.;
        c = mix(c, tileColor * select(.7, 1., (texel.x + texel.y) % 2u == 0u), .5);
      }
    }
  } else {
    // the sky term, with the albedo like the lightmap's indirect light. shading.wgsl adds it without: a light sky then
    // washes out the sun, the shadows and the colors of the layers
    c += albedo * ((n.y * .5 + .5) * getSSAO(coord)) * skyColor;
  }

  // the water reflects a building or the ground: darker. The ray mirrored on the normal of the water goes up, it can only
  // hit a building, or the ground where that is higher than the water (isTerrainOccluded)
  if (WATER_REFLECTIONS && ((uniforms.waterMaterials >> materialIndex) & 1u) != 0u) {
    let rd = getWaterReflectionRay(n, world, d);
    // the part of the reflection that hits a building: the ray of the pixel (0 or 1), or the rays of the half resolution
    // pass around it. With REFLECTION_COLORS the trace gives where it hits, which is lit by the lightmap there, and
    // fades with the fog on the way to it: the part the fog leaves, the rest of the mirror is the sky's as without a hit
    var occluded = 0.;
    // the color of the buildings in the mirror (REFLECTION_COLORS), linear, times the part of them the fog leaves
    var reflected = vec3<f32>(0.);
    if (REFLECT_BUILDINGS) {
      if (REFLECTIONS_HALF_RES) {
        let samples = getHalfResReflection(coord, rd);
        occluded = samples.a;
        reflected = samples.rgb;
      } else {
        if (REFLECTION_COLORS) {
          let hit = getOcclusionHit(world, rd);
          if (hit.t >= 0.) {
            let color = getReflectionHitColor(world, rd, hit, LIGHTMAP_ENABLED && SUN_SHADOWS == 2u);
            occluded = color.a;
            reflected = color.rgb;
          }
        } else {
          occluded = select(0., 1., isOccluded(world, rd));
        }
      }
    }
    // the ground in it, where no building is: the buildings and the trees are in front of the hills more often than
    // behind them. Its color (terrainColor.a: how much of the mirror has one) goes in with the sky, below
    var terrainColor = vec4<f32>(0.);
    if (REFLECT_BUILDINGS && REFLECT_TERRAIN) {
      let terrain = getTerrainReflection(coord, world, rd);
      if (terrain.occluded) {
        terrainColor = vec4(terrain.color.rgb, 1.) * (terrain.color.a * (1. - occluded));
        occluded = 1.;
      }
    }
    // the light the water loses to a building in the mirror, of which its color gives some back
    c = c * (1. - uniforms.reflectionDarkening * occluded) + reflected * (uniforms.reflectionDarkening * uniforms.reflectionColor);
    // the fresnel term (Schlick) of the facet that mirrors the view into rd: its normal is halfway between rd and the
    // direction to the eye, the micro normal of the ripples with WATER_RIPPLES, which makes it vary per ripple
    let toEye = normalize(uniforms.cameraPosition - world);
    let cosTheta = clamp(dot(rd, normalize(rd + toEye)), 0., 1.);
    let schlick = pow(1. - cosTheta, 5.);
    let fresnel = WATER_F0 + (1. - WATER_F0) * schlick;
    // the sky in the mirror where it misses the buildings: the water goes to the sky color at grazing angles. Linear,
    // like the shading (getSkyColor is gamma space)
    let sky = getSkyColor(rd, false);
    // and the ground: it is in the mirror in place of the sky, by the same fresnel term
    let mirror = min(fresnel * uniforms.skyReflection, 1.);
    c = c * (1. - mirror * (1. - occluded + terrainColor.a)) + (sky * sky * (1. - occluded) + terrainColor.rgb) * mirror;
    if (uniforms.sunReflection > 0. && occluded < 1.) {
      // the sun in the mirror where it sees the sky, in the shadows of the buildings like the sun light of the water.
      // Brighter at grazing angles: 1 straight down, 1 + sunFresnel at the horizon
      let glint = pow(max(dot(rd, l), 0.), uniforms.sunReflectionPower) * (1. + uniforms.sunFresnel * schlick);
      c += sunColor * (glint * shadow * uniforms.sunReflection * (1. - occluded));
    }
  }

  c *= uniforms.exposure;
  if (TONE_MAP) {
//    c = acesTonemapFitted(c);
    c = acesTonemap(c);
  }
  c = sqrt(c);
  // the contrast before the fog: the fog ends on the background color of the sky (getSky), without a seam
  c = mix(c, smoothstep(vec3<f32>(0.), vec3<f32>(1.), c), uniforms.contrast);
  c = mix(c, uniforms.backgroundColor, fog);

/*  let h = hash22ff(input.position.xy);
  c += (h.y + h.z - 1.) / 255.;
  c = clamp(c, vec3(0.), vec3(1.));*/

  //red highlight for selected building
  if (isGBufferSelected(g)) {
    c = mix(c, vec3<f32>(0.9, 0.1, 0.1), 0.6);
  }
  return vec4<f32>(c, 1.0);
}
