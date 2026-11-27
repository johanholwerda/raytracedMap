// The cars of CarTraffic, drawn over the finished image of MapWorld (after the TAA and the SSAO, before the
// tilt-shift) into a layer of their own, which carLayerComposite.wgsl blends over it. Nothing else of the frame
// sees them: not the gBuffer, the lightmap, the shadow map, the SSAO or the TAA, which all keep converging.
//
// Without SHADOW: the cars as boxes (a body and a cabin), shaded like deferredShading.wgsl: the sun with the VSM
// cascades (the buildings shadow the cars), the sky visibility of the ground under them (the lightmap) in place of
// their own, the fog and the contrast. With SHADOW: a quad on the ground under each car, drawn first, that darkens what
// is under it: the shadow the car casts in the sun (a ray to the sun through the box) and a contact occlusion around
// it. Its alpha is how much darker, the color the fog's light it keeps, so the layer blends over the image as
// premultiplied alpha.
// The buildings shadow the cars (SUN_SHADOWS) through the VSM cascades, or rays to the center of the sun through the
// city: hard, as no TAA averages the cars. The rays and the sky visibility of the ground are per car (carLight, see
// carLighting.wgsl), interpolated over the vertices. With PER_PIXEL (the high quality) they are per fragment instead: a
// ray to the sun and a lookup of the lightmap at the ground under each pixel of a car and of a shadow quad, without the
// carLighting.wgsl pass. Needs skyVisibilityTrace.wgsl then.
//
// The buildings in front: the cars are not in the depth of the gBuffer, so each fragment compares its camera distance
// with the one of the gBuffer at its pixel and is dropped behind it (a depth test per pixel, not per sample: the
// edge of a building over a car is not anti aliased, the edges of the cars themselves are, by the MSAA of the layer).
// The depth of the layer orders the cars among themselves.

struct Uniforms {
  // without the jitter of the TAA, the image the cars go over is the resolved one
  viewProjection : mat4x4<f32>,
  // of the camera the gBuffer was drawn with, for getGBufferDistance
  inverseProjection : mat4x4<f32>,
  // the sun's view of the ground for the rays (SUN_SHADOWS 2), see terrainSun.wgsl
  terrainSunViewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  resolution : vec2<f32>,
  depthParams : vec2<f32>,
  // the height of the bottom of the cars and of the shadow quads over the ground under them (Car.ground)
  groundY : f32,
  shadowY : f32,
  // the margin of the contact occlusion around a car, and how dark it is
  aoMargin : f32,
  aoStrength : f32,
  // how dark the shadow of a car is, times the sun it takes away
  carShadowStrength : f32,
  // the shading params of MapWorld, by name
  lightDirection : vec3<f32>,
  skyColor : vec3<f32>,
  sunColor : vec3<f32>,
  backgroundColor : vec3<f32>,
  distanceFog : f32,
  // where the fog is 1, see getFog
  viewDistanceMax : f32,
  contrast : f32,
  shadowStrength : f32,
  VSMBleedThreshold : f32,
  varianceMin : f32,
  shadowDebug : u32,
  // the sky visibility lightmap, see SkyVisibilityLightmap.setResolveUniforms
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
  // how far a point goes off its surface before it is compared with terrainSunMap, see getTerrainSunShadow
  terrainSunOffset : f32,
#ifdef PER_PIXEL
  // the city of the rays, see CityScene.bind
  cityTileCount : u32,
  cityRoot : u32,
#endif
};

// where the sun shadows of the buildings on the cars come from: 0 none, 1 the VSM cascades, 2 rays, see getSunShadow
override SUN_SHADOWS : u32 = 1u;

struct ShadowMapData {
  shadowMapViewProjection : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var shadowMapSampler : sampler;
// the 4 cascades, one layer each (ShadowMap.cascadeTexture)
@group(0) @binding(2) var shadowMaps : texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> shadowMapData : array<ShadowMapData>;
@group(0) @binding(7) var<storage, read> tileBuffer : array<u32>;
@group(0) @binding(8) var<storage, read> lightingBuffer : array<u32>;
@group(0) @binding(9) var gBufferDepth : texture_depth_2d;
// 3 per car: (x, z, heading x, heading z), (half length, half width, height, type), (ground, slope, 0, 0), see CarTraffic
@group(0) @binding(10) var<storage, read> cars : array<vec4<f32>>;
// the ground as the sun sees it (TerrainSunMap.ts): the hills, which the rays don't trace, see terrainSun.wgsl
@group(0) @binding(13) var terrainSunMap : texture_2d<f32>;
#ifdef PER_PIXEL
// the buildings and trees of the rays, see CityScene and skyVisibilityTrace.wgsl
@group(0) @binding(11) var<storage, read> cityBuffer : array<vec4<f32>>;
#else
// the light of each car: the sun at the corners of its roof and the sky of the ground under it, see carLighting.wgsl
@group(0) @binding(12) var<storage, read> carLight : array<vec2<u32>>;
#endif

// the car types, see CarTraffic: a car has a body and a cabin, a van or a bus is one box
const TYPE_CAR = 0u;

// the paint of all cars and the windows, before they are squared like the palette of the gBuffer
const PAINT = vec3(.78, .78, .79);
const GLASS = vec3(.10, .12, .14);

struct Car {
  center : vec2<f32>,
  // the unit heading (along the length) and the right side (along the width), in xz
  u : vec2<f32>,
  v : vec2<f32>,
  halfLength : f32,
  halfWidth : f32,
  height : f32,
  kind : u32,
  // the height of the ground under the center (world units) and its rise per world unit along u: the car is sheared
  // along it, its bottom on the line of the road
  ground : f32,
  slope : f32,
};

fn loadCar(i : u32) -> Car {
  let a = cars[i * 3u];
  let b = cars[i * 3u + 1u];
  let c = cars[i * 3u + 2u];
  return Car(a.xy, a.zw, vec2(-a.w, a.z), b.x, b.y, b.z, u32(b.w), c.x, c.y);
}

// --- Variance shadow map sampling, as in deferredShading.wgsl -----------------

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

// --- shared --------------------------------------------------------------------

// the sun the point p gets, 0 to 1, see SUN_SHADOWS. raySun: the rays of carLighting.wgsl at p, interpolated (not
// used with PER_PIXEL: a ray from p, off the surface along its normal n by a part of its camera distance d, as the rays
// of the deferred shading)
fn getSunShadow(p : vec3<f32>, n : vec3<f32>, d : f32, raySun : f32) -> f32 {
  if (SUN_SHADOWS == 0u) {
    return 1.;
  }
  if (SUN_SHADOWS == 2u) {
#ifdef PER_PIXEL
    return select(1., 0., isOccluded(p + n * (d * 1e-4), uniforms.lightDirection)) * getTerrainSunShadow(p, n);
#else
    return raySun * getTerrainSunShadow(p, n);
#endif
  }
  return getShadowMap(p, n);
}

#ifdef PER_PIXEL

// The sky visibility of the ground at p (its xz) at the height groundY, as the deferred shading lights the ground there.
// Without the lightmap or a texel there: an open sky, as the sky term of the shading
fn getGroundSky(p : vec3<f32>, groundY : f32) -> f32 {
  if (uniforms.lightmapEnabled == 0u) {
    return 1.;
  }
  let g = vec3(p.x, groundY, p.z);
  let l = getLightmapLighting(getGroundSurface(g), getLevel(g, vec3(0., 1., 0.)));
  return select(1., l.x, l.w > 0.);
}

#else

// The light of a car (carLighting.wgsl): the sun at the 4 corners of its roof and the sky visibility of the ground under
// it, as the deferred shading lights the ground there
fn loadCarLight(i : u32) -> vec2<u32> {
  return carLight[i];
}

fn getCarSky(light : vec2<u32>) -> f32 {
  return bitcast<f32>(light.y);
}

// the sun of the corners at car space x (along the length, -1 to 1) and z (along the width)
fn getCarSun(light : vec2<u32>, x : f32, z : f32) -> f32 {
  let corners = unpack4x8unorm(light.x);
  let along = saturate(x * .5 + .5);
  return mix(mix(corners.x, corners.y, along), mix(corners.z, corners.w, along), saturate(z * .5 + .5));
}

#endif

// The pixel is behind the gBuffer surface there (a building in front): the cars aren't in its depth. d: the camera
// distance of the fragment
fn isHidden(position : vec4<f32>, d : f32) -> bool {
  let coord = vec2<u32>(position.xy);
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  let gd = getGBufferDistance(textureLoad(gBufferDepth, coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
  // the sky is 0: nothing in front
  return gd > 0.0 && gd < d * (1.0 - 2e-4);
}

// The distance fog: exponential (distanceFog per world unit), with a smoothstep from .7 of viewDistanceMax that takes
// it to 1 at viewDistanceMax, where the loaded map ends (see MapWorld.viewDistanceMax). 0 (not set): only the exp.
// d is the horizontal (xz) distance to the camera, not the 3d one: the map is loaded in a radius around the camera in
// xz, and a camera that goes up and looks down gets no more fog (the far plane reaches past it, see
// MapWorld.applyViewDistanceMax).
// The same in deferredShading.wgsl, cars.wgsl, reprojectedSSAOComposite.wgsl and shading.wgsl.
fn getFog(d : f32) -> f32 {
  let end = select(1., 1. - smoothstep(.7 * uniforms.viewDistanceMax, uniforms.viewDistanceMax, d), uniforms.viewDistanceMax > 0.);
  return 1. - exp(-d * uniforms.distanceFog) * end;
}

fn finish(linear : vec3<f32>, d : f32) -> vec3<f32> {
  var c = sqrt(max(linear, vec3(0.)));
  // the contrast before the fog, like deferredShading.wgsl
  c = mix(c, smoothstep(vec3<f32>(0.), vec3<f32>(1.), c), uniforms.contrast);
  return mix(c, uniforms.backgroundColor, getFog(d));
}

fn luminance(c : vec3<f32>) -> f32 {
  return dot(c, vec3(.2126, .7152, .0722));
}

// the corner of the 6 of a quad, 0 to 1
fn quadCorner(i : u32) -> vec2<f32> {
  var corners = array<vec2<f32>, 6>(vec2(0., 0.), vec2(1., 0.), vec2(1., 1.), vec2(0., 0.), vec2(1., 1.), vec2(0., 1.));
  return corners[i];
}

#ifdef SHADOW

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) world : vec3<f32>,
  @location(1) @interpolate(flat) car : u32,
#ifndef PER_PIXEL
  // the sun on the car (the average of its corners) and the sky of the ground under it, see loadCarLight
  @location(2) @interpolate(flat) sun : f32,
  @location(3) @interpolate(flat) sky : f32,
#endif
};

// The sun direction for the shadow of a car: toward the sun, raised to at least this, so a low sun doesn't throw a
// shadow longer than about 5 times the height
const SUN_Y_MIN = .2;

fn getSun() -> vec3<f32> {
  let l = uniforms.lightDirection;
  return normalize(vec3(l.x, max(l.y, SUN_Y_MIN), l.z));
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32, @builtin(instance_index) instance : u32) -> VertexOutput {
  let car = loadCar(instance);
  let l = getSun();
  // where the shadow of the top falls, in car space (along the length, along the width)
  let offset = -l.xz * (car.height / l.y);
  let o = vec2(dot(offset, car.u), dot(offset, car.v));
  let halfSize = vec2(car.halfLength, car.halfWidth);
  let lo = min(-halfSize, o - halfSize) - uniforms.aoMargin;
  let hi = max(halfSize, o + halfSize) + uniforms.aoMargin;
  let q = mix(lo, hi, quadCorner(vertexIndex));
  let xz = car.center + car.u * q.x + car.v * q.y;
  var output : VertexOutput;
  output.world = vec3(xz.x, car.ground + q.x * car.slope + uniforms.shadowY, xz.y);
  output.position = uniforms.viewProjection * vec4(output.world, 1.);
  output.car = instance;
#ifndef PER_PIXEL
  let light = loadCarLight(instance);
  output.sun = dot(unpack4x8unorm(light.x), vec4(.25));
  output.sky = getCarSky(light);
#endif
  return output;
}

// The part of the ray from the ground point q (car space: x along the length, y up from the bottom, z along the
// width) to the sun l that is in the box of the car, in world units: 0 when it misses
fn getSunOcclusion(q : vec3<f32>, l : vec3<f32>, car : Car) -> f32 {
  let boxMin = vec3(-car.halfLength, 0., -car.halfWidth);
  let boxMax = vec3(car.halfLength, car.height, car.halfWidth);
  let inv = 1. / select(l, vec3(1e-6), abs(l) < vec3(1e-6));
  let t0 = (boxMin - q) * inv;
  let t1 = (boxMax - q) * inv;
  let tNear = max(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)), 0.);
  let tFar = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
  return max(tFar - tNear, 0.);
}

// the distance from q to the footprint of the car, 0 inside it
fn getFootprintDistance(q : vec2<f32>, car : Car) -> f32 {
  let d = abs(q) - vec2(car.halfLength, car.halfWidth);
  return length(max(d, vec2(0.)));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let d = distance(input.world, uniforms.cameraPosition);
  if (isHidden(input.position, d)) {
    discard;
  }
  let car = loadCar(input.car);
  let p = input.world;
  let rel = p.xz - car.center;
  let q = vec2(dot(rel, car.u), dot(rel, car.v));

  let l = getSun();
  let sunLocal = vec3(dot(l.xz, car.u), l.y, dot(l.xz, car.v));
  // soft at the edge, where the ray only grazes the box
  let sunShadow = saturate(getSunOcclusion(vec3(q.x, uniforms.groundY - uniforms.shadowY, q.y), sunLocal, car) / (car.halfWidth * .5));
  let ao = 1. - smoothstep(0., uniforms.aoMargin, getFootprintDistance(q, car));
#ifdef PER_PIXEL
  // nothing to darken (most of the quad under a low sun): no ray or lookup
  if (sunShadow == 0. && ao == 0.) {
    return vec4(0.);
  }
#endif

  // the ground lit as the deferred shading lights it (its albedo cancels): the sun the car takes away and the sky
  // the occlusion takes away, as a part of the light there
#ifdef PER_PIXEL
  let raySun = 1.;
  let sky = getGroundSky(p, p.y - (uniforms.shadowY - uniforms.groundY));
#else
  let raySun = input.sun;
  let sky = input.sky;
#endif
  var sunLight = 0.;
  if (uniforms.lightDirection.y > 0.) {
    sunLight = luminance(uniforms.sunColor * uniforms.sunColor) * uniforms.lightDirection.y *
      mix(1., getSunShadow(p, vec3(0., 1., 0.), d, raySun), uniforms.shadowStrength);
  }
  let skyLight = luminance(uniforms.skyColor * uniforms.skyColor) * sky;
  let lit = sunLight + skyLight;
  let left = sunLight * (1. - sunShadow * uniforms.carShadowStrength) + skyLight * (1. - ao * uniforms.aoStrength);
  // after the sqrt of the shading. The image there is mix(ground, backgroundColor, fog): only the ground part darkens,
  // the fog keeps its light. ground * keep mixed with the fog = image * keep + fog * backgroundColor * (1 - keep), the
  // premultiplied blend of this color and alpha
  let keep = sqrt(select(1., left / lit, lit > 0.));
  let fog = getFog(distance(input.world.xz, uniforms.cameraPosition.xz));
  return vec4(fog * uniforms.backgroundColor * (1. - keep), 1. - keep);
}

#else

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) world : vec3<f32>,
  @location(1) @interpolate(flat) normal : vec3<f32>,
  @location(2) @interpolate(flat) albedo : vec3<f32>,
  // the height above the bottom of the car, as a part of its height
  @location(3) height : f32,
  // 1 on the windows
  @location(4) @interpolate(flat) glass : f32,
  // the height of the ground under the vertex
  @location(7) ground : f32,
#ifndef PER_PIXEL
  // the sun of the corners of the roof at the vertex and the sky of the ground under the car, see loadCarLight
  @location(5) sun : f32,
  @location(6) @interpolate(flat) sky : f32,
#endif
};

// the outward normal of the 5 faces of a box (no bottom), and the 2 axes along the face, cross(a, b) = normal so its
// triangles wind counter clockwise seen from outside
fn faceNormal(face : u32) -> vec3<f32> {
  var n = array<vec3<f32>, 5>(vec3(1., 0., 0.), vec3(-1., 0., 0.), vec3(0., 0., 1.), vec3(0., 0., -1.), vec3(0., 1., 0.));
  return n[face];
}

fn faceAxisA(face : u32) -> vec3<f32> {
  var a = array<vec3<f32>, 5>(vec3(0., 1., 0.), vec3(0., 0., 1.), vec3(1., 0., 0.), vec3(0., 1., 0.), vec3(0., 0., 1.));
  return a[face];
}

fn faceAxisB(face : u32) -> vec3<f32> {
  var b = array<vec3<f32>, 5>(vec3(0., 0., 1.), vec3(0., 1., 0.), vec3(0., 1., 0.), vec3(1., 0., 0.), vec3(1., 0., 0.));
  return b[face];
}

// 30 vertices per box, 2 boxes per car: the body and the cabin (empty for a van or a bus)
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32, @builtin(instance_index) instance : u32) -> VertexOutput {
  let car = loadCar(instance);
  let boxIndex = vertexIndex / 30u;
  let face = (vertexIndex % 30u) / 6u;
  let corner = quadCorner(vertexIndex % 6u) * 2. - 1.;
  let n = faceNormal(face);
  // the corner in a box of -1 to 1, y 0 to 1
  let c = n + faceAxisA(face) * corner.x + faceAxisB(face) * corner.y;
  let unitPos = vec3(c.x, c.y * .5 + .5, c.z);

  // the box in car space: its center along the length, its half length, the bottom and top (parts of the height) and
  // its half width (part of the half width)
  var boxCenter = 0.;
  var halfLength = 1.;
  var bottom = 0.;
  var top = 1.;
  var halfWidth = 1.;
  let isCar = car.kind == TYPE_CAR;
  if (boxIndex == 0u) {
    top = select(1., .55, isCar);
  } else {
    boxCenter = -.12;
    halfLength = select(0., .52, isCar);
    bottom = .55;
    halfWidth = .88;
  }
  let carPos = vec3(
    (boxCenter + unitPos.x * halfLength) * car.halfLength,
    mix(bottom, top, unitPos.y) * car.height,
    unitPos.z * halfWidth * car.halfWidth,
  );
  let xz = car.center + car.u * carPos.x + car.v * carPos.z;

  var output : VertexOutput;
  // sheared along the slope: the vertical faces stay vertical, the normal of the shear is (n.x - slope * n.y, n.y, n.z)
  output.ground = car.ground + carPos.x * car.slope + uniforms.groundY;
  output.world = vec3(xz.x, output.ground + carPos.y, xz.y);
  output.position = uniforms.viewProjection * vec4(output.world, 1.);
  let ns = normalize(vec3(n.x - car.slope * n.y, n.y, n.z));
  let nxz = car.u * ns.x + car.v * ns.z;
  output.normal = vec3(nxz.x, ns.y, nxz.y);
  // the sides of the cabin are the windows
  let glass = boxIndex == 1u && face < 4u;
  output.albedo = select(PAINT, GLASS, glass);
  output.glass = select(0., 1., glass);
  output.height = carPos.y / max(car.height, 1e-9);
#ifndef PER_PIXEL
  let light = loadCarLight(instance);
  output.sun = getCarSun(light, carPos.x / max(car.halfLength, 1e-9), carPos.z / max(car.halfWidth, 1e-9));
  output.sky = getCarSky(light);
#endif
  return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let d = distance(input.world, uniforms.cameraPosition);
  if (isHidden(input.position, d)) {
    discard;
  }
  let p = input.world;
  let n = input.normal;
  let l = uniforms.lightDirection;
  let nDotL = max(dot(n, l), 0.0);
  let skyColor = uniforms.skyColor * uniforms.skyColor;
  let sunColor = uniforms.sunColor * uniforms.sunColor;
  let albedo = input.albedo * input.albedo;
#ifdef PER_PIXEL
  let raySun = 1.;
  let groundSky = getGroundSky(p, input.ground);
#else
  let raySun = input.sun;
  let groundSky = input.sky;
#endif
  var shadow = 1.0;
  if (nDotL > 0.0) {
    shadow = mix(1.0, getSunShadow(p, n, d, raySun), uniforms.shadowStrength);
  }
  // the sides get less of the sky lower down: the ground and the cars around them
  let occlusion = mix(.55, 1., saturate(input.height * 1.6));
  let sky = groundSky * (n.y * .5 + .5) * occlusion;
  var c = albedo * (sunColor * (nDotL * shadow) + skyColor * sky);
  // the windows reflect the sky a little
  c += input.glass * .12 * skyColor * sky;
  return vec4(finish(c, distance(input.world.xz, uniforms.cameraPosition.xz)), 1.0);
}

#endif
