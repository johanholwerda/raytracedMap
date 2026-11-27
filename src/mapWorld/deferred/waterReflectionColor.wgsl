// The colors of the buildings and trees in the water reflections: the world trace of a mirror ray (getOcclusionHit of
// cityTraceDistance.wgsl) gives the face it hits, which is lit as the deferred shading lights the gBuffer: by the sky
// visibility lightmap at that point (getCityHitSurface: the chart and the position the gBuffer has there, from the atlas
// words of the city buffer), with the sun visibility of its texels when the lightmap has the sun. The lightmap only has
// the tiles of the texels the pixels of the screen use: a face it has no texels of (an underside, a side away from the
// camera, one behind another, a part off the screen) is shaded from its normal instead, see getUnseenReflectionColor.
// The colors came from the image of the frame before (the TAA history at the hit point) before: a frame late, and only
// where the screen showed the hit.
// For the deferred shading (a ray per water pixel) and the reduced pass (waterReflectionHalfRes.wgsl, a ray per block).
// Needs cityTraceDistance.wgsl, the lightmap resolve (SkyVisibilityLightmap.resolveIncludes, with its uniforms and
// bindings) and the uniforms backgroundColor, distanceFog, contrast, exposure (see getLinearFromDisplay), skyColor,
// sunColor, lightDirection, shadowStrength (see deferredShading.wgsl), reflectionUnseen, reflectionBuildingAlbedo,
// reflectionTreeAlbedo (see getUnseenReflectionColor), reflectionLevelBias (see getReflectionLevel), and the override
// TONE_MAP

// the colors of what the rays hit. Else the buildings only darken the water
override REFLECTION_COLORS : bool = false;

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

// The inverse of acesTonemap (toneMap.wgsl): its curve per channel between its 2 matrices is a ratio of 2 quadratics,
// solved for its input. Exact up to the clamp to 1 of its output
fn getInverseAcesTonemap(c : vec3<f32>) -> vec3<f32> {
  let m1Inverse = mat3x3(
    1.764741, -0.147028, -0.036337,
    -0.675778, 1.160252, -0.162436,
    -0.088963, -0.013224, 1.198773
  );
  let m2Inverse = mat3x3(
    0.643038, 0.059269, 0.005962,
    0.311187, 0.931436, 0.063929,
    0.045775, 0.009295, 0.930118
  );
  // below the asymptote of the curve (1.0165)
  let r = min(m2Inverse * c, vec3(1.));
  // v * (v + .0245786) - .000090537 = r * (v * (.983729 * v + .432951) + .238081)
  let a = 1. - .983729 * r;
  let b = .0245786 - .432951 * r;
  let k = -(.000090537 + .238081 * r);
  let v = (-b + sqrt(max(b * b - 4. * a * k, vec3(0.)))) / (2. * a);
  return max(m1Inverse * v, vec3(0.));
}

// The linear color of the shading (before the exposure) that it turned into the color c of the image, of a pixel that is
// fogged by fog: the end of the shading (deferredShading.wgsl) backwards. The fog, the contrast (a smoothstep mixed in,
// increasing: a few Newton steps), the gamma, the tone map and the exposure. For the background color the reflections
// are fogged to
fn getLinearFromDisplay(c : vec3<f32>, fog : f32) -> vec3<f32> {
  let x = clamp((c - uniforms.backgroundColor * fog) / max(1. - fog, .01), vec3(0.), vec3(1.));
  var y = x;
  for (var i = 0; i < 3; i++) {
    let f = mix(y, y * y * (3. - 2. * y), uniforms.contrast) - x;
    let slope = mix(vec3(1.), 6. * y * (1. - y), uniforms.contrast);
    y = clamp(y - f / slope, vec3(0.), vec3(1.));
  }
  var l = y * y;
  if (TONE_MAP) {
    l = getInverseAcesTonemap(l);
  }
  // a clamped bright pixel would come back far too bright
  return min(l, vec3(4.)) / max(uniforms.exposure, 1e-4);
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

// The linear color of a face of a building or a tree at p (normal n, albedo as the map color params, not squared) that a
// mirror ray from the water point ro hit: lit as the deferred shading lights it, the sky by sky (the part of it the
// point sees, cosine weighted) and the sun by n.l times sun (its visibility). Then fogged by the way from p to ro, as
// the shading fogs the water pixel by its own distance after
fn shadeReflectionHit(ro : vec3<f32>, p : vec3<f32>, n : vec3<f32>, albedo : vec3<f32>, sky : f32, sun : f32) -> vec3<f32> {
  let skyLight = uniforms.skyColor * uniforms.skyColor * sky;
  let sunLight = uniforms.sunColor * uniforms.sunColor * (max(dot(n, uniforms.lightDirection), 0.) * sun);
  let surface = albedo * albedo * (skyLight + sunLight);
  let background = getLinearFromDisplay(uniforms.backgroundColor, 0.);
  let way = 1. - exp(-distance(p.xz, ro.xz) * uniforms.distanceFog);
  return mix(surface, background, way);
}

// The color of a face the lightmap has no texels of (normal n, albedo as shadeReflectionHit): the sky of the open sky
// the normal faces (n.y * .5 + .5, the shading's without the lightmap) times reflectionUnseen, which stands in for the
// occlusion of the lightmap, and the sun by n.l, without its shadows
fn getUnseenReflectionColor(ro : vec3<f32>, p : vec3<f32>, n : vec3<f32>, albedo : vec3<f32>) -> vec3<f32> {
  return shadeReflectionHit(ro, p, n, albedo, (n.y * .5 + .5) * uniforms.reflectionUnseen, 1.);
}

// The lightmap at a hit surface (getCityHitSurface at p): the sky visibility and its paths, and with sun the sun
// visibility and its paths (getLightmapChannels). Paths 0 where it has no texels, or is on no chart
fn getReflectionLightmap(s : CityHitSurface, p : vec3<f32>, sun : bool) -> vec4<f32> {
  if (!s.valid) {
    return vec4(0.);
  }
  // a face away from the sun gets none of it: no sun lookup, as the shading's
  return getLightmapChannels(s.surface, getReflectionLevel(p, s.normal), sun && dot(s.normal, uniforms.lightDirection) > 0.);
}

// The linear color of what the mirror ray from the water point ro along rd hit (getOcclusionHit): lit by the lightmap at
// the hit where it has texels there, else getUnseenReflectionColor. Of 2 walls that coincide there (hit.tie) the one the
// lightmap has the texels of: the one the screen shows.
// lightmap: the lightmap can be read (its buffers are bound); sun: its texels hold the sun visibility too (see
// SkyVisibilityLightmap.setSun), else the sun is not shadowed
fn getReflectionHitColor(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit, lightmap : bool, sun : bool) -> vec3<f32> {
  var s = getCityHitSurface(ro, rd, hit);
  let p = ro + rd * hit.t;
  let albedo = select(uniforms.reflectionBuildingAlbedo, uniforms.reflectionTreeAlbedo, isCityHitTree(hit));
  if (lightmap) {
    var l = getReflectionLightmap(s, p, sun);
    if (l.y == 0. && hit.tie != hit.record) {
      let tie = getCityHitTieSurface(ro, rd, hit);
      let lt = getReflectionLightmap(tie, p, sun);
      if (lt.y > 0.) {
        s = tie;
        l = lt;
      }
    }
    if (l.y > 0.) {
      let sunVisibility = mix(1., select(1., l.z, l.w > 0.), uniforms.shadowStrength);
      return shadeReflectionHit(ro, p, s.normal, albedo, l.x, sunVisibility);
    }
  }
  return getUnseenReflectionColor(ro, p, s.normal, albedo);
}
