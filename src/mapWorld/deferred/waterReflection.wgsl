// The mirror ray of the water of MapWorld, for the deferred shading (deferredShading.wgsl) and the half resolution pass
// of its reflections (waterReflectionHalfRes.wgsl): both trace the same ray for a pixel.
// Needs uniforms.inverseProjection, resolution (of the gBuffer), cameraPosition and rippleTile, the bindings
// rippleTexture and rippleSampler, and the override WATER_RIPPLES

// the normal n of the water tilted by the ripples at world, d from the eye along view. Sampled at an explicit mip (the
// branch of the water isn't uniform control flow): the footprint of the pixel on the water, the geometric mean of its
// sides, the one along the view stretched by 1 / cos. The ripples smaller than a pixel average out to the flat water
fn getRippleNormal(n : vec3<f32>, world : vec3<f32>, view : vec3<f32>, d : f32) -> vec3<f32> {
  let pixelAngle = 2. * uniforms.inverseProjection[1][1] / uniforms.resolution.y;
  let footprint = d * pixelAngle * inverseSqrt(max(abs(view.y), .02));
  let texel = uniforms.rippleTile / f32(textureDimensions(rippleTexture).x);
  let lod = log2(max(footprint / texel, 1.));
  let g = textureSampleLevel(rippleTexture, rippleSampler, world.xz / uniforms.rippleTile, lod).xy;
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
