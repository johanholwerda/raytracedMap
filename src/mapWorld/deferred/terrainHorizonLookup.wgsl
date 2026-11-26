// The terrain in the water reflections for a pixel, from the horizons of WaterTerrainHorizon.ts (1 per block of
// TERRAIN_HORIZON_BLOCK pixels, see waterTerrainHorizon.wgsl) instead of a march over the height map per pixel.
// Needs terrainReflection.wgsl, the binding
//   terrainHorizon : texture_2d<f32> (r: the horizon of the water pixel of the block, -1 where it isn't water)
// and the uniform reflectTerrainInWater (u32, 0: no ground above the water in reach, most places)

// the pixels of a block along x and y, and the one that has the horizon: mirrors WaterTerrainHorizon.ts
const TERRAIN_HORIZON_BLOCK = 4;
const TERRAIN_HORIZON_OFFSET = 2;

// whether the ray of the water pixel coord, from ro along rd (normalized), hits the ground: where it goes up less
// steep than the horizon around the pixel, of the blocks around it that are water, bilinearly. The horizon is the one
// along the mirror ray of the flat water, the ripples turn rd a few degrees off it: a hill is wider than that. It
// changes slowly across the water, unlike the ray, which the ripples tilt per pixel: that is compared per pixel.
// Without a water block around it (a creek, the last pixels at a shore) the pixel marches its own ray
fn isTerrainOccluded(coord : vec2<u32>, ro : vec3<f32>, rd : vec3<f32>) -> bool {
  if (uniforms.reflectTerrainInWater == 0u || rd.y <= 0.0) {
    return false;
  }
  let last = vec2<i32>(textureDimensions(terrainHorizon)) - 1;
  let f = (vec2<f32>(coord) - f32(TERRAIN_HORIZON_OFFSET)) / f32(TERRAIN_HORIZON_BLOCK);
  let i = vec2<i32>(floor(f));
  let w = f - floor(f);
  var sum = 0.0;
  var weight = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2<i32>(k & 1, k >> 1);
    let horizon = textureLoad(terrainHorizon, clamp(i + o, vec2<i32>(0), last), 0).x;
    let wk = select(1.0 - w.x, w.x, o.x == 1) * select(1.0 - w.y, w.y, o.y == 1);
    if (horizon >= 0.0) {
      sum += horizon * wk;
      weight += wk;
    }
  }
  if (weight < 1e-3) {
    return traceTerrain(ro, rd);
  }
  return rd.y < sum / weight * length(rd.xz);
}
