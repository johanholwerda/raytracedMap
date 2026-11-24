// The light of each car of CarTraffic, every frame before they're drawn: cars.wgsl reads it per vertex instead of tracing
// a ray and looking up the lightmap per fragment (per pixel of every car and shadow quad). 4 threads per car, a ray each,
// so the long rays of a low sun run side by side instead of one after the other. A car out of view (most of them: they
// live in a radius around the pivot) gets no ray or lookup.
// Out per car, carLight (2 words):
// x: the sun at the 4 corners of its roof (pack4x8unorm; back left, front left, back right, front right: along the
//    length, then the width), which the vertices of the car interpolate. With SUN_SHADOWS 2 a ray each to the center of
//    the sun through the city (isOccluded of skyVisibilityTrace.wgsl), hard; else 1 (cars.wgsl takes the shadow map or
//    nothing)
// y: the sky visibility of the ground under its center (f32 bits), as the deferred shading lights
//    the ground there: the lightmap, or an open sky (1) without it or a texel there
// Needs the includes of the lightmap resolve (SkyVisibilityLightmap.resolveIncludes) and skyVisibilityTrace.wgsl

struct Uniforms {
  // see getLevel of cityLightmap.wgsl
  viewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  // toward the sun
  lightDirection : vec3<f32>,
  // the height of the bottom of the cars
  groundY : f32,
  carCount : u32,
  // the sky visibility lightmap, see SkyVisibilityLightmap.setResolveUniforms. Not all are used here
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
  // the city of the rays, see CityScene.bind
  cityTileCount : u32,
  cityRoot : u32,
};

// where the sun shadows of the buildings on the cars come from, see cars.wgsl: only 2 (rays) is done here
override SUN_SHADOWS : u32 = 1u;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> tileBuffer : array<u32>;
@group(0) @binding(2) var<storage, read> lightingBuffer : array<u32>;
// 2 per car, see loadCar in cars.wgsl
@group(0) @binding(3) var<storage, read> cars : array<vec4<f32>>;
// the buildings and trees of the rays, see CityScene and skyVisibilityTrace.wgsl
@group(0) @binding(4) var<storage, read> cityBuffer : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> carLight : array<vec2<u32>>;

// the threads of a workgroup, 4 per car
const GROUP_SIZE = 64u;
// a car whose center is this far outside the screen (in parts of its half size) still counts as in view: its body and
// its shadow reach in
const VIEW_MARGIN = 1.25;

// the sun of the corner of each thread, which the first thread of its car packs
var<workgroup> cornerSun : array<f32, GROUP_SIZE>;

@compute @workgroup_size(GROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) id : vec3<u32>, @builtin(local_invocation_index) local : u32) {
  let i = id.x / 4u;
  // the corner of the thread: bit 0 along the length, bit 1 along the width
  let corner = id.x % 4u;
  var lit = false;
  var sun = 1.;
  var center = vec2<f32>(0.);
  if (i < uniforms.carCount) {
    let a = cars[i * 2u];
    let b = cars[i * 2u + 1u];
    center = a.xy;
    let height = b.z;
    let clip = uniforms.viewProjection * vec4(center.x, uniforms.groundY + height, center.y, 1.);
    lit = clip.w > 0. && all(abs(clip.xy) <= vec2(clip.w * VIEW_MARGIN));
    if (lit && SUN_SHADOWS == 2u) {
      // along the length and along the width
      let u = a.zw;
      let v = vec2(-a.w, a.z);
      let along = select(-1., 1., (corner & 1u) != 0u) * b.x;
      let across = select(-1., 1., (corner & 2u) != 0u) * b.y;
      let xz = center + u * along + v * across;
      let p = vec3(xz.x, uniforms.groundY + height, xz.y);
      // off the roof by a part of the distance, as the rays of the deferred shading
      let d = distance(p, uniforms.cameraPosition);
      sun = select(1., 0., isOccluded(p + vec3(0., 1., 0.) * (d * 1e-4), uniforms.lightDirection));
    }
  }
  cornerSun[local] = sun;
  workgroupBarrier();
  if (i >= uniforms.carCount || corner != 0u) {
    return;
  }
  var sky = 1.;
  if (lit && uniforms.lightmapEnabled != 0u) {
    let g = vec3(center.x, uniforms.groundY, center.y);
    let l = getLightmapLighting(getGroundSurface(g), getLevel(g, vec3(0., 1., 0.)));
    sky = select(1., l.x, l.w > 0.);
  }
  let corners = vec4(cornerSun[local], cornerSun[local + 1u], cornerSun[local + 2u], cornerSun[local + 3u]);
  carLight[i] = vec2(pack4x8unorm(corners), bitcast<u32>(sky));
}
