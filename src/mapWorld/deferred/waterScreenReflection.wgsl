// The colors of the buildings and trees in the water reflections of MapWorld, from the screen. The mirror ray of a
// water pixel (getWaterReflectionRay of waterReflection.wgsl) is traced through the city with its hit distance
// (getOcclusionHit of cityTraceDistance.wgsl). Where the screen shows the hit point (the depth of the gBuffer at its
// pixel is its own), the water mirrors its color: of the image of the frame before (the TAA history), at where the
// point was in it, turned back into the linear color of the shading (getLinearFromDisplay), see
// getWorldHitReflectionColor. A hit off the screen takes the color where the ray leaves it (a tower that goes on above
// its top), one the screen doesn't show (an underside, a side away from the camera) is shaded from the normal of the
// face the ray hits, see getUnseenReflectionColor.
// A march of the rays over the depth of the gBuffer first (screen space reflections) stepped over buildings between its
// samples (dark bands in the reflections, lost hits) and took .1 ms more at 1080p for the same image: the hit point of
// the world trace is what the color of a ray needs.
// For the deferred shading (a ray per water pixel) and the reduced pass (waterReflectionHalfRes.wgsl, a ray per block).
// Needs the uniforms
//   viewProjection (the jittered camera of the gBuffer), resolution, depthParams (see getGBufferDistance), cameraPosition,
//   inverseProjection and inverseView (of the same camera, see cameraRay.wgsl), waterMaterials, backgroundColor,
//   distanceFog, viewDistanceMax, contrast, exposure, skyColor, sunColor, lightDirection (see deferredShading.wgsl),
//   reflectionUnseen, reflectionBuildingAlbedo and reflectionTreeAlbedo (see getUnseenReflectionColor), historyViewProjection (the camera of the history, without the jitter),
//   screenReflectionHistory (u32, 0: the history isn't of the view, no colors)
// the bindings gBufferTexture, gBufferDepth, reflectionHistory (texture_2d<f32>) and reflectionHistorySampler (bilinear,
// clamped), mapGBufferPacking.wgsl, cityTraceDistance.wgsl and the override TONE_MAP

// the colors from the screen. Else the buildings only darken the water
override SCREEN_REFLECTIONS : bool = false;

// The ray that leaves the screen takes the color of the surface behind it at the edge for a building off the screen it
// hits (getScreenReflectionFallbackColor) when it is at most this part of the camera distance of that surface in front
// of it: a facade it goes up along, which goes on above the top of the screen. Not a roof far below it, which has
// nothing to do with the building off the screen it hits
const SCREEN_REFLECTION_FALLBACK_GAP = .15;

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

// the screen position of a point in clip space: xy in pixels (y down), z 1 / its view depth, which is linear along a
// line on the screen
fn getScreenReflectionPoint(clip : vec4<f32>) -> vec3<f32> {
  let ndc = clip.xy / clip.w;
  return vec3(vec2(ndc.x * .5 + .5, .5 - ndc.y * .5) * uniforms.resolution, 1. / clip.w);
}

// 1 / the view depth of the surface at a pixel, 0 on the sky (the depth is reversed: 0 there)
fn getScreenReflectionDepth(pixel : vec2<f32>) -> f32 {
  let p = clamp(vec2<i32>(pixel), vec2(0), vec2<i32>(uniforms.resolution) - 1);
  let depth = textureLoad(gBufferDepth, p, 0);
  return select(0., depth * uniforms.depthParams.x + uniforms.depthParams.y, depth > 0.);
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
// increasing: a few Newton steps), the gamma, the tone map and the exposure
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

// the color the water mirrors of the surface the ray from ro hits at p: of the image of the frame before at p, which
// is fogged by the horizontal camera distance (getFog), and fogged here by the way from p to ro instead. a 0 where p
// was off the screen of the frame before, or the history isn't of this view.
// atEdge: p is the surface at the edge of the screen (getScreenReflectionFallbackColor), which is off the screen of the
// frame before as soon as the camera moves towards it: it takes the edge of the history there instead. Without that
// the reflections of buildings above the screen flickered to the unseen color while the camera moved
fn getScreenReflectionColor(ro : vec3<f32>, p : vec3<f32>, atEdge : bool) -> vec4<f32> {
  let h = uniforms.historyViewProjection * vec4(p, 1.);
  var uv = vec2(h.x / h.w * .5 + .5, .5 - h.y / h.w * .5);
  if (atEdge) {
    uv = clamp(uv, vec2(0.), vec2(1.));
  }
  if (uniforms.screenReflectionHistory == 0u || h.w <= 0. || any(uv < vec2(0.)) || any(uv > vec2(1.))) {
    return vec4(0.);
  }
  let display = textureSampleLevel(reflectionHistory, reflectionHistorySampler, uv, 0.).rgb;
  let fogEnd = select(1., 1. - smoothstep(.7 * uniforms.viewDistanceMax, uniforms.viewDistanceMax, distance(p.xz, uniforms.cameraPosition.xz)),
    uniforms.viewDistanceMax > 0.);
  let fog = 1. - exp(-distance(p.xz, uniforms.cameraPosition.xz) * uniforms.distanceFog) * fogEnd;
  let surface = getLinearFromDisplay(display, fog);
  // the background of the fog as the shading's linear color, and the fog of the way to the water
  let background = getLinearFromDisplay(uniforms.backgroundColor, 0.);
  let way = 1. - exp(-distance(p.xz, ro.xz) * uniforms.distanceFog);
  return vec4(mix(surface, background, way), 1.);
}

// The pixel where the ray from ro to its hit point p (off the screen) leaves the screen, for the color of what it hits
// (getScreenReflectionFallbackColor), when the ray there is in front of the surface the screen shows and close to it
// (SCREEN_REFLECTION_FALLBACK_GAP). x < 0: none
fn getScreenExitPixel(ro : vec3<f32>, p : vec3<f32>) -> vec2<f32> {
  let c0 = uniforms.viewProjection * vec4(ro, 1.);
  let c1 = uniforms.viewProjection * vec4(p, 1.);
  if (c1.w <= 0.) {
    return vec2(-1.);
  }
  let s0 = getScreenReflectionPoint(c0);
  let s1 = getScreenReflectionPoint(c1);
  let delta = s1.xy - s0.xy;
  let edge = select(vec2(0.), uniforms.resolution, delta > vec2(0.));
  let toEdge = select(vec2(1e9), (edge - s0.xy) / delta, abs(delta) > vec2(1e-6));
  let fOut = min(toEdge.x, toEdge.y);
  if (fOut >= 1. || fOut <= 0.) {
    return vec2(-1.);
  }
  // just inside the screen
  let s = mix(s0, s1, fOut - .5 / max(max(abs(delta.x), abs(delta.y)), 1.));
  let scene = getScreenReflectionDepth(s.xy);
  let gap = 1. / max(scene, 1e-12) - 1. / s.z;
  return select(vec2(-1.), s.xy, scene > 0. && gap > 0. && gap < SCREEN_REFLECTION_FALLBACK_GAP / scene);
}

// The color the water mirrors where the world trace of a ray hits a building after it left the screen: the color of
// what the screen shows at the pixel where it left (getScreenExitPixel, as getScreenReflectionColor), which is most of the time
// the building it hits, or one beside it. Only a surface that faces the ray (rd): a facade, not a roof it passes over.
// Else (on the sky and the water, a ray that went behind something, a point the frame before didn't show) the color of
// a surface the screen doesn't show (of the hit, see getUnseenReflectionColor), a 0. Without SCREEN_REFLECTIONS 0, dark
// as before
fn getScreenReflectionFallbackColor(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit, pixel : vec2<f32>) -> vec4<f32> {
  if (!SCREEN_REFLECTIONS) {
    return vec4(0.);
  }
  if (pixel.x >= 0.) {
    let p = clamp(vec2<i32>(pixel), vec2(0), vec2<i32>(uniforms.resolution) - 1);
    let depth = textureLoad(gBufferDepth, p, 0);
    let g = vec3(textureLoad(gBufferTexture, p, 0).xy, 0u);
    if (depth > 0. && ((uniforms.waterMaterials >> getGBufferMaterial(g)) & 1u) == 0u && dot(getGBufferNormal(g), rd) < 0.) {
      let uv = (vec2<f32>(p) + .5) / uniforms.resolution;
      let d = getGBufferDistance(depth, uv, uniforms.inverseProjection, uniforms.depthParams);
      let c = getScreenReflectionColor(ro, depthToPosition(d, uv * 2. - 1.), true);
      if (c.a > 0.) {
        return c;
      }
    }
  }
  return vec4(getUnseenReflectionColor(ro, rd, hit), 0.);
}

// how far the hit point of the world trace may be from the surface the screen shows at its pixel, a part of its camera
// distance, and still be that surface (getWorldHitReflectionColor)
const SCREEN_REFLECTION_VISIBLE = .02;

// The color of what the world trace of a ray from ro along rd hit (getOcclusionHit of cityTraceDistance.wgsl): of the
// screen at that point where it is visible there (the surface of its pixel is at its distance). Else the color at the
// edge of the screen or of an unseen surface, see getScreenReflectionFallbackColor
fn getWorldHitReflectionColor(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> vec3<f32> {
  if (!SCREEN_REFLECTIONS) {
    return vec3(0.);
  }
  let p = ro + rd * hit.t;
  let clip = uniforms.viewProjection * vec4(p, 1.);
  var fallback = vec2(-1.);
  if (clip.w > 0.) {
    let s = getScreenReflectionPoint(clip);
    let onScreen = all(s.xy >= vec2(0.)) && all(s.xy < uniforms.resolution);
    if (!onScreen) {
      fallback = getScreenExitPixel(ro, p);
    } else {
      let scene = getScreenReflectionDepth(s.xy);
      if (scene > 0. && abs(1. / scene - 1. / s.z) < SCREEN_REFLECTION_VISIBLE / scene) {
        let c = getScreenReflectionColor(ro, p, false);
        if (c.a > 0.) {
          return c.rgb;
        }
      }
    }
  }
  return getScreenReflectionFallbackColor(ro, rd, hit, fallback).rgb;
}

// The color of a building or tree in the water that the screen doesn't show (its underside, a side away from the
// camera, one behind another) that the ray from ro along rd hit: shaded like the deferred shading from the normal of the
// face it hit, linear. Its albedo (reflectionBuildingAlbedo or reflectionTreeAlbedo, the map colors), the sky of the
// open sky the normal faces (n.y * .5 + .5, the shading's without the lightmap) times reflectionUnseen, which stands in
// for the occlusion of the lightmap, and the sun by n.l, without its shadows. Then fogged by the way from p to ro, as
// the colors of the screen. Only the sky light before: a sunlit facade behind another one was a dark blue band below
// the orange of the part the screen shows
fn getUnseenReflectionColor(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> vec3<f32> {
  let n = getCityHitNormal(ro, rd, hit);
  let p = ro + rd * hit.t;
  let albedo = select(uniforms.reflectionBuildingAlbedo, uniforms.reflectionTreeAlbedo, isCityHitTree(hit));
  let sky = uniforms.skyColor * uniforms.skyColor * ((n.y * .5 + .5) * uniforms.reflectionUnseen);
  let sun = uniforms.sunColor * uniforms.sunColor * max(dot(n, uniforms.lightDirection), 0.);
  let surface = albedo * albedo * (sky + sun);
  let background = getLinearFromDisplay(uniforms.backgroundColor, 0.);
  let way = 1. - exp(-distance(p.xz, ro.xz) * uniforms.distanceFog);
  return mix(surface, background, way);
}
