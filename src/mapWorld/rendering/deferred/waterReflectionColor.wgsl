// The colors of the buildings and trees in the water reflections: the world trace of a mirror ray (getOcclusionHit of
// cityTraceDistance.wgsl) gives the face it hits, which is lit as the deferred shading lights the gBuffer: by the sky
// visibility lightmap at that point (getCityFaceSurface: the chart and the position the gBuffer has there, from the atlas
// words of the city buffer), with the sun visibility of its texels when the lightmap has the sun. The lightmap only has
// the tiles of the texels the pixels of the screen use: a face it has no texels of (an underside, a side away from the
// camera, one behind another, a part off the screen) is shaded from its normal instead, see getUnseenReflectionColor.
// The colors came from the image of the frame before (the TAA history at the hit point) before: a frame late, and only
// where the screen showed the hit.
// A hit fades with the fog on the way from it to the water (getReflectionFog): the colors come with the part of the hit
// the fog leaves, and the rest of the mirror is the sky's, as where a ray hits nothing. The fog has the color of the
// background, which is what the sky in the water is. The colors were mixed to the background color before, as the
// linear color the shading turns into it (the contrast, the gamma and the tone map backwards): of a light sky that is
// many times brighter than any face (a channel near 1 is at the end of the tone map), and at the part of it the
// water takes it came out as a saturated color: far woods in the water were bright blue under an afternoon sky.
// For the deferred shading (a ray per water pixel) and the reduced pass (waterReflectionHalfRes.wgsl, a ray per block).
// Needs cityTraceDistance.wgsl, the lightmap resolve (SkyVisibilityLightmap.resolveIncludes, with its uniforms and
// bindings) and the uniforms distanceFog, skyColor, sunColor, lightDirection, shadowStrength (see deferredShading.wgsl),
// reflectionUnseen, reflectionBuildingAlbedo, reflectionTreeAlbedo (see getUnseenReflectionColor) and
// reflectionLevelBias (see getReflectionLevel)

// the colors of what the rays hit. Else the buildings only darken the water
override REFLECTION_COLORS : bool = false;
// what the rays hit is lit by the lightmap (reflectionType lightmap, and the lightmap has its buffers: see
// MapWater.draw), else every hit has the color of a face the lightmap has no texels of (getUnseenReflectionColor): no
// lookup, no tie, no atlas words or wall bottoms read
override REFLECTION_LIGHTMAP : bool = true;

// A linear color (not negative) in 32 bits, for the samples of the reduced pass (waterReflectionHalfRes.wgsl): the half
// floats of its channels without their sign and the low bits of their mantissa, 11, 11 and 10 bits (as rg11b10ufloat)
fn packReflectionColor(c : vec3<f32>) -> u32 {
  let rg = pack2x16float(c.rg);
  let b = pack2x16float(vec2(c.b, 0.));
  return ((rg >> 4u) & 0x7FFu) | (((rg >> 20u) & 0x7FFu) << 11u) | (((b >> 5u) & 0x3FFu) << 22u);
}

fn unpackReflectionColor(bits : u32) -> vec3<f32> {
  let rg = unpack2x16float(((bits & 0x7FFu) << 4u) | (((bits >> 11u) & 0x7FFu) << 20u));
  return vec3(rg, unpack2x16float(((bits >> 22u) & 0x3FFu) << 5u).x);
}

// The level of the lightmap texels of a hit at p (normal n): the one the pixels of the screen claim for p (getLevel, from
// the camera to p), not one for the way through the mirror. A face in the mirror is mostly one the screen shows too, and
// the lightmap only has the texels the screen claims: at the level of its pixels and ancestorLevels above it. A level
// for the mirrored way (longer, at another angle to the face) was levels finer or coarser than those, which read no
// texels, or the coarser ancestors around them, an average of other points. reflectionLevelBias levels coarser (within
// the ancestors): smoother, the ripples blur the mirror anyway
fn getReflectionLevel(p : vec3<f32>, n : vec3<f32>) -> f32 {
  return min(getLevel(p, n) + uniforms.reflectionLevelBias, f32(LEVEL_MAX));
}

// The part of a hit at p the fog leaves in the mirror of the water point ro: by the way between them across the ground,
// as getFog of the shading fogs a pixel by its distance to the camera
fn getReflectionFog(ro : vec3<f32>, p : vec3<f32>) -> f32 {
  return exp(-distance(p.xz, ro.xz) * uniforms.distanceFog);
}

// The linear color of a face of a building or a tree at p (normal n, albedo as the map color params, not squared) that a
// mirror ray from the water point ro hit: lit as the deferred shading lights it, the sky by sky (the part of it the
// point sees, cosine weighted) and the sun by n.l times sun (its visibility). a: the part of the hit the fog leaves
// (getReflectionFog), the color is times that
fn shadeReflectionHit(ro : vec3<f32>, p : vec3<f32>, n : vec3<f32>, albedo : vec3<f32>, sky : f32, sun : f32) -> vec4<f32> {
  let skyLight = uniforms.skyColor * uniforms.skyColor * sky;
  let sunLight = uniforms.sunColor * uniforms.sunColor * (max(dot(n, uniforms.lightDirection), 0.) * sun);
  let surface = albedo * albedo * (skyLight + sunLight);
  return vec4(surface, 1.) * getReflectionFog(ro, p);
}

// The color of a face the lightmap has no texels of (normal n, albedo as shadeReflectionHit): the sky of the open sky
// the normal faces (n.y * .5 + .5, the shading's without the lightmap) times reflectionUnseen, which stands in for the
// occlusion of the lightmap, and the sun by n.l, without its shadows
fn getUnseenReflectionColor(ro : vec3<f32>, p : vec3<f32>, n : vec3<f32>, albedo : vec3<f32>) -> vec4<f32> {
  return shadeReflectionHit(ro, p, n, albedo, (n.y * .5 + .5) * uniforms.reflectionUnseen, 1.);
}

// The lightmap at a surface (getCityFaceSurface of a face with normal n at p): the sky visibility and its paths, and
// with sun the sun visibility and its paths (getLightmapChannels). Paths 0 where it has no texels, or is on no chart
fn getReflectionLightmap(s : CityHitSurface, p : vec3<f32>, n : vec3<f32>, sun : bool) -> vec4<f32> {
  if (!s.valid) {
    return vec4(0.);
  }
  // a face away from the sun gets none of it: no sun lookup, as the shading's
  return getLightmapChannels(s.surface, getReflectionLevel(p, n), sun && dot(n, uniforms.lightDirection) > 0.);
}

// The linear color of what the mirror ray from the water point ro along rd hit (getOcclusionHit): lit by the lightmap at
// the hit where it has texels there (REFLECTION_LIGHTMAP), else getUnseenReflectionColor: then only the face of the
// hit is worked out, not where it is on the charts. Of 2 walls that coincide there (hit.tie) the one the lightmap has
// the texels of: the one the screen shows. With the part of the hit the fog leaves in its a, see shadeReflectionHit.
// sun: the texels of the lightmap hold the sun visibility too (see SkyVisibilityLightmap.setSun), else the sun is not
// shadowed
fn getReflectionHitColor(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit, sun : bool) -> vec4<f32> {
  var face = getCityHitFace(ro, rd, hit);
  let p = ro + rd * hit.t;
  let albedo = select(uniforms.reflectionBuildingAlbedo, uniforms.reflectionTreeAlbedo, isCityHitTree(hit));
  if (REFLECTION_LIGHTMAP) {
    var l = getReflectionLightmap(getCityFaceSurface(hit.record, face, p), p, face.normal, sun);
    if (l.y == 0. && hit.tie != hit.record) {
      let tieFace = getCityHitTieFace(ro, rd, hit);
      let lt = getReflectionLightmap(getCityFaceSurface(hit.tie, tieFace, p), p, tieFace.normal, sun);
      if (lt.y > 0.) {
        face = tieFace;
        l = lt;
      }
    }
    if (l.y > 0.) {
      let sunVisibility = mix(1., select(1., l.z, l.w > 0.), uniforms.shadowStrength);
      return shadeReflectionHit(ro, p, face.normal, albedo, l.x, sunVisibility);
    }
  }
  return getUnseenReflectionColor(ro, p, face.normal, albedo);
}
