// The terrain in the water reflections of MapWorld: the mirror ray of the water (getWaterReflectionRay) traces only the
// buildings and the trees (isOccluded of skyVisibilityTrace.wgsl), this marches it over the height map of the ground
// that TerrainSunMap.ts renders top down for the sun shadows. In world space, so the hills beside and behind the view are
// in the water too. The including shader declares
//   terrainHeights : texture_2d<f32> (per texel -height, 1e9 where no ground)
//   terrainMaxHeights : texture_2d<f32> (the highest ground per cell of TERRAIN_COARSE texels, with a texel around it)
//   terrainMaxHeightsFine : texture_2d<f32> (the same per cell of TERRAIN_FINE texels)
// and the uniforms terrainSunViewProjection (mat4x4<f32>, world to the map), reflectTerrainInWater (u32, 0: no ground
// above the water in reach, the rays skip the march: most places), terrainReflectionTop (f32, the highest ground, world
// y) and terrainReflectionBias (f32, world units), see TerrainSunMap.bindHeights and MapWorld.

// texels of the height map per cell of terrainMaxHeights and of terrainMaxHeightsFine, mirror COARSE and FINE of
// TerrainSunMap.ts
const TERRAIN_COARSE = 16;
const TERRAIN_FINE = 4;
// steps of a texel near the water, longer ones further away: by this part of the distance. The ripples scatter the
// rays, a hill far away has no sharp edge in the water
const TERRAIN_STEP_GROWTH = 0.04;
// the steps and the skipped cells together
const TERRAIN_MAX_STEPS = 96;

// the ground height at p (texel coordinates, the centers at .5), bilinear
fn getTerrainHeight(p : vec2<f32>, size : vec2<i32>) -> f32 {
  let t = p - 0.5;
  let i = vec2<i32>(floor(t));
  let f = t - floor(t);
  let last = size - 1;
  let h00 = textureLoad(terrainHeights, clamp(i, vec2<i32>(0), last), 0).x;
  let h10 = textureLoad(terrainHeights, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), last), 0).x;
  let h01 = textureLoad(terrainHeights, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), last), 0).x;
  let h11 = textureLoad(terrainHeights, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), last), 0).x;
  return -mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// how far along d (in texels) the ray at p leaves its cell of cellSize texels, a little past its edge
fn getTerrainCellExit(p : vec2<f32>, d : vec2<f32>, cell : vec2<i32>, cellSize : i32) -> f32 {
  let lo = vec2<f32>(cell * cellSize);
  let edge = select(lo, lo + f32(cellSize), d > vec2<f32>(0.0));
  let t = select(vec2<f32>(1e9), (edge - p) / d, abs(d) > vec2<f32>(1e-6));
  return max(min(t.x, t.y), 0.0) + 0.01;
}

// whether the ray from ro (on the water) along rd (normalized, going up) hits the ground. The ray starts
// terrainReflectionBias above the water: the water lies on the ground of the map, and the map has it a little off the
// mesh. It only rises, so it ends where it is above all the ground, and it skips the cells that are all below it: the
// coarse ones over the open water, the fine ones along a slope, which leaves the 4 texels of a bilinear height for the
// places where the ray is close to the ground
fn isTerrainOccluded(ro : vec3<f32>, rd : vec3<f32>) -> bool {
  if (uniforms.reflectTerrainInWater == 0u || rd.y <= 0.0) {
    return false;
  }
  let size = vec2<i32>(textureDimensions(terrainHeights));
  let sizeF = vec2<f32>(size);
  // the map is an orthographic view: ro in texels, and the texels the ray crosses per unit of its length. Texture v
  // runs down, ndc y up
  let c0 = (uniforms.terrainSunViewProjection * vec4<f32>(ro, 1.0)).xy;
  let c1 = (uniforms.terrainSunViewProjection * vec4<f32>(rd, 0.0)).xy;
  let p0 = vec2<f32>(c0.x * 0.5 + 0.5, 0.5 - c0.y * 0.5) * sizeF;
  let dp = vec2<f32>(c1.x, -c1.y) * 0.5 * sizeF;
  let speed = length(dp);
  if (speed < 1e-4) {
    // straight up
    return false;
  }
  let d = dp / speed;
  // the rise of the ray per texel it goes across
  let rise = rd.y / speed;
  let y0 = ro.y + uniforms.terrainReflectionBias;
  let coarseLast = vec2<i32>(textureDimensions(terrainMaxHeights)) - 1;
  let fineLast = vec2<i32>(textureDimensions(terrainMaxHeightsFine)) - 1;
  var s = 1.0;
  for (var i = 0; i < TERRAIN_MAX_STEPS; i++) {
    let p = p0 + d * s;
    let y = y0 + s * rise;
    if (y >= uniforms.terrainReflectionTop || any(p < vec2<f32>(0.0)) || any(p >= sizeF)) {
      return false;
    }
    let c = clamp(vec2<i32>(floor(p / f32(TERRAIN_COARSE))), vec2<i32>(0), coarseLast);
    if (textureLoad(terrainMaxHeights, c, 0).x <= y) {
      // all the ground of the cell is below the ray: to where it leaves the cell
      s += getTerrainCellExit(p, d, c, TERRAIN_COARSE);
      continue;
    }
    let cf = clamp(vec2<i32>(floor(p / f32(TERRAIN_FINE))), vec2<i32>(0), fineLast);
    if (textureLoad(terrainMaxHeightsFine, cf, 0).x <= y) {
      // at least as far as a step: the steps far away are longer than a fine cell
      s += max(getTerrainCellExit(p, d, cf, TERRAIN_FINE), max(1.0, s * TERRAIN_STEP_GROWTH));
      continue;
    }
    if (getTerrainHeight(p, size) > y) {
      return true;
    }
    s += max(1.0, s * TERRAIN_STEP_GROWTH);
  }
  return false;
}
