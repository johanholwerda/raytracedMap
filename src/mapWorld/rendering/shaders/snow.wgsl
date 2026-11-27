// The snow on the ground above the snow line (see map/snowLine.ts), with a hard edge: getSnowField is signed, the snow
// is where it is above 0. Included with the define SNOW by deferred/mapGBuffer.wgsl, where the pixels in the snow take
// the snow material (a hard edge, the TAA antialiases it), and by shading.wgsl, the forward shading without a TAA:
// getSnowCoverAt antialiases the edge over SNOW_EDGE_PIXELS there.
// world: the point, n: its normal. line: the snow line in world y (relative to the ground at the projection center, like
// the heights). noise: how far (m) value noise (a few hundred meters wide patches, a smaller octave on top) moves the
// line up and down, so it isn't a contour line. metersPerUnit: the noise is in meters, the same at every map scale.
// Steep slopes hold no snow (the rock shows), less steep ones higher up.

// the width of the antialiased edge, in pixels
const SNOW_EDGE_PIXELS = 1.5;
// how far (m) below the snow line at the top of its noise a point is when isSnowPossible skips it: the field is at most
// that height, the margin leaves room for the antialiased edge (the field changing that much per pixel). The same as
// SNOW_SKIP_MARGIN_M of map/snowLine.ts, which skips the whole view
const SNOW_SKIP_MARGIN = 50.;
// the steepest slope (its n.y) with snow at the snow line: 45 degrees. SNOW_SLOPE_HEIGHT above the line SNOW_SLOPE_BOOST
// less (64 degrees)
const SNOW_SLOPE = .7;
const SNOW_SLOPE_BOOST = .26;
const SNOW_SLOPE_HEIGHT = 1000.;
// meters of the field per unit of n.y: how the slope edge and the height edge compare where they meet (the min)
const SNOW_SLOPE_SCALE = 500.;
// the noise of the snow line: two octaves of value noise, the size (m) of their cells and the weight of the first
const SNOW_NOISE_SIZE0 = 400.;
const SNOW_NOISE_SIZE1 = 90.;
const SNOW_NOISE_WEIGHT0 = .65;
// the most snowValueNoise changes per unit along an axis: the slope of its smoothstep
const SNOW_NOISE_MAX_SLOPE = 1.5;

fn snowHash(p : vec2<i32>) -> f32 {
  var h = (u32(p.x) * 0x8da6b343u) ^ (u32(p.y) * 0xd8163841u);
  h = (h ^ (h >> 16u)) * 0x7feb352du;
  h = (h ^ (h >> 15u)) * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return f32(h) * (1. / 4294967295.);
}

// value noise, 0 to 1, smooth between the integer points
fn snowValueNoise(p : vec2<f32>) -> f32 {
  let i = vec2<i32>(floor(p));
  let f = fract(p);
  let u = f * f * (3. - 2. * f);
  let a = snowHash(i);
  let b = snowHash(i + vec2(1, 0));
  let c = snowHash(i + vec2(0, 1));
  let d = snowHash(i + vec2(1, 1));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// > 0 in the snow, < 0 outside, in meters (about): the height above the noisy snow line, or how much flatter the slope
// is than the steepest that holds snow there, whichever is less
fn getSnowField(world : vec3<f32>, n : vec3<f32>, line : f32, noise : f32, metersPerUnit : f32) -> f32 {
  let p = world.xz * metersPerUnit;
  let v = snowValueNoise(p * (1. / SNOW_NOISE_SIZE0)) * SNOW_NOISE_WEIGHT0
    + snowValueNoise(p * (1. / SNOW_NOISE_SIZE1) + 17.) * (1. - SNOW_NOISE_WEIGHT0);
  let h = (world.y - line) * metersPerUnit + (v * 2. - 1.) * noise;
//  let h = (world.y - line) * metersPerUnit;
  let slope = n.y - SNOW_SLOPE + SNOW_SLOPE_BOOST * clamp(h / SNOW_SLOPE_HEIGHT, 0., 1.);
  return min(h, slope * SNOW_SLOPE_SCALE);
}

// false: no snow at a point at world y, whatever its noise and slope (getSnowField is below -SNOW_SKIP_MARGIN). The cheap
// test before the field: one multiply-add
fn isSnowPossible(worldY : f32, line : f32, noise : f32, metersPerUnit : f32) -> bool {
  return (worldY - line) * metersPerUnit + noise > -SNOW_SKIP_MARGIN;
}

// the part of the pixel in the snow, 0 to 1: field over its change per pixel (the sum of its differences to the pixels
// right of and below it, as fwidth) is the distance to the edge in pixels
fn getSnowCover(field : f32, perPixel : f32) -> f32 {
  return clamp(field / max(perPixel * SNOW_EDGE_PIXELS, 1e-6) + .5, 0., 1.);
}

// An upper bound of how much getSnowField changes from a point to one dWorld away with a normal dNormal different,
// without its noise: the height part changes with the y and at most SNOW_NOISE_MAX_SLOPE per noise cell along x and z,
// the slope part with the normal and the clamped height
fn getSnowFieldChangeBound(dWorld : vec3<f32>, dNormal : vec3<f32>, noise : f32, metersPerUnit : f32) -> f32 {
  let dxz = (abs(dWorld.x) + abs(dWorld.z)) * metersPerUnit;
  let noisePerMeter = SNOW_NOISE_MAX_SLOPE * (SNOW_NOISE_WEIGHT0 / SNOW_NOISE_SIZE0 + (1. - SNOW_NOISE_WEIGHT0) / SNOW_NOISE_SIZE1);
  let dh = abs(dWorld.y) * metersPerUnit + 2. * abs(noise) * noisePerMeter * dxz;
  return max(dh, SNOW_SLOPE_SCALE * (abs(dNormal.y) + SNOW_SLOPE_BOOST / SNOW_SLOPE_HEIGHT * dh));
}

// The part of the pixel at world, normal n, in the snow (getSnowCover), with the pixels right of and below it at world +
// dWorldX, n + dNormalX and world + dWorldY, n + dNormalY. Away from the edge, where the field is more than the half edge
// width from 0 at the most it can change per pixel (getSnowFieldChangeBound), the cover is 0 or 1 without the field of
// the neighbours: only the pixels near the edge evaluate it 3 times
fn getSnowCoverAt(world : vec3<f32>, n : vec3<f32>, dWorldX : vec3<f32>, dNormalX : vec3<f32>, dWorldY : vec3<f32>,
  dNormalY : vec3<f32>, line : f32, noise : f32, metersPerUnit : f32) -> f32 {
  let f = getSnowField(world, n, line, noise, metersPerUnit);
  let bound = getSnowFieldChangeBound(dWorldX, dNormalX, noise, metersPerUnit)
    + getSnowFieldChangeBound(dWorldY, dNormalY, noise, metersPerUnit);
  if (abs(f) > bound * SNOW_EDGE_PIXELS * .5) {
    return select(0., 1., f > 0.);
  }
  let fx = getSnowField(world + dWorldX, n + dNormalX, line, noise, metersPerUnit);
  let fy = getSnowField(world + dWorldY, n + dNormalY, line, noise, metersPerUnit);
  return getSnowCover(f, abs(fx - f) + abs(fy - f));
}
