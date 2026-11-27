// The mirror ray of the water of MapWorld, for the deferred shading (deferredShading.wgsl) and the half resolution pass
// of its reflections (waterReflectionHalfRes.wgsl): both trace the same ray for a pixel.
// Needs uniforms.inverseProjection, resolution (of the gBuffer), cameraPosition and rippleTile, the bindings
// rippleTexture and rippleSampler, and the override WATER_RIPPLES

// The ripples are sampled over this many pixels, see getRippleNormal. Over 1 they change from pixel to pixel far away,
// and what a mirror ray hits with them: the edge of a reflection is single pixels there, which flicker when the camera
// moves (the TAA only averages them in a static view). Over 3 the ripples of a few pixels are left, the edge wobbles
const RIPPLE_FOOTPRINT = 3.;

// the normal n of the water tilted by the ripples at world, d from the eye along view. Sampled over the footprint of the
// pixel on the water, with explicit gradients (the branch of the water isn't uniform control flow): its side across the
// view, and the one along it, which is 1 / cos longer. Far away that is many times the other: the sampler is
// anisotropic, it averages the ripples along the view and keeps them across. One mip for both (the geometric mean of the
// sides, as this was) is too sharp along the view there: the ripples alias, the reflections flicker when the camera
// moves. The ripples smaller than RIPPLE_FOOTPRINT pixels average out to the flat water
fn getRippleNormal(n : vec3<f32>, world : vec3<f32>, view : vec3<f32>, d : f32) -> vec3<f32> {
  let pixelAngle = 2. * uniforms.inverseProjection[1][1] / uniforms.resolution.y;
  // the size of a pixel at the water, in tiles
  let pixel = RIPPLE_FOOTPRINT * d * pixelAngle / uniforms.rippleTile;
  let across = length(view.xz);
  // straight down: no direction along the view, and both sides the same
  let along = select(vec2(1., 0.), view.xz / max(across, 1e-6), across > 1e-4);
  let g = textureSampleGrad(rippleTexture, rippleSampler, world.xz / uniforms.rippleTile, vec2(-along.y, along.x) * pixel,
    along * (pixel / max(abs(view.y), .02))).xy;
  //the normal of the height field y = h(x, z) is (-dh/dx, 1, -dh/dz)
  return normalize(n + vec3(-g.x, 0., -g.y));
}

// the ray the water at world (normal n, d from the eye) mirrors the view along: on the micro normal of the ripples
// (WATER_RIPPLES), else on the flat water. It goes up, so it can only hit a building
fn getWaterReflectionRay(n : vec3<f32>, world : vec3<f32>, d : f32) -> vec3<f32> {
  let view = normalize(world - uniforms.cameraPosition);
  var rd = reflect(view, n);
  if (WATER_RIPPLES) {
    rd = reflect(view, getRippleNormal(n, world, view, d));
    // a ripple facing away at a grazing angle mirrors the ray into the water: back up, as the next wave would
    rd.y = abs(rd.y);
  }
  return rd;
}

// the block of pixels the reduced pass traces 1 ray for (WaterReflectionHalfRes.ts), along x and y
override REFLECTION_BLOCK_X : u32 = 2u;
override REFLECTION_BLOCK_Y : u32 = 2u;

// the pixel of each block the reduced pass traces, from its phase (0 to the pixels of a block - 1)
fn getReflectionSampleOffset(phase : u32) -> vec2<u32> {
  return vec2(phase % REFLECTION_BLOCK_X, phase / REFLECTION_BLOCK_X);
}

// The first word of a sample of the reduced pass, 4 x 8 bits (pack4x8unorm). Its first: 0 no water (no sample), else
// REFLECTION_SAMPLE_SKY to 255 for the part of the ray that hits a building (hit, 0 to 1: less than 1 where the fog
// takes a part of the hit, see waterReflectionColor.wgsl). Then the ray rd (* .5 + .5): the shading weighs a sample by
// how alike its ray and the ray of the pixel are
const REFLECTION_SAMPLE_SKY = 64.;

fn packReflectionSample(hit : f32, rd : vec3<f32>) -> u32 {
  return pack4x8unorm(vec4((REFLECTION_SAMPLE_SKY + (255. - REFLECTION_SAMPLE_SKY) * hit) / 255., rd * .5 + .5));
}

// the part of the ray of a sample that hits a building. s: its first word unpacked (unpack4x8unorm), not 0
fn getReflectionSampleHit(s : vec4<f32>) -> f32 {
  return saturate((s.x * 255. - REFLECTION_SAMPLE_SKY) / (255. - REFLECTION_SAMPLE_SKY));
}
