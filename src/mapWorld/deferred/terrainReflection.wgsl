// The terrain in the water reflections of MapWorld: the mirror ray of the water (getWaterReflectionRay) traces only the
// buildings and the trees (isOccluded of skyVisibilityTrace.wgsl), these march it over the height map of the ground
// that TerrainSunMap.ts renders top down for the sun shadows. In world space, so the hills beside and behind the view are
// in the water too. A march per water pixel was more than the rest of the shading (.3 ms of .55 at 1080p under a hill),
// so the pixels share one: getTerrainHorizon, 1 per block of pixels (waterTerrainHorizon.wgsl), is how steep a ray has
// to go up to pass over the ground, and a pixel compares its ray with it (getTerrainReflection of
// terrainHorizonLookup.wgsl). traceTerrain is the march of 1 ray, for the pixels without a horizon near them.
// The including shader declares
//   terrainHeights : texture_2d<f32> (per texel -height, 1e9 where no ground)
//   terrainMaxHeights : texture_2d<f32> (the highest ground per cell of TERRAIN_COARSE texels, with a texel around it)
//   terrainMaxHeightsFine : texture_2d<f32> (the same per cell of TERRAIN_FINE texels)
// and the uniforms terrainSunViewProjection (mat4x4<f32>, world to the map), terrainReflectionTop (f32, the highest
// ground, world y) and terrainReflectionBias (f32, world units), see TerrainSunMap.bindHeights and MapWorld.

// texels of the height map per cell of terrainMaxHeights and of terrainMaxHeightsFine, mirror COARSE and FINE of
// TerrainSunMap.ts
const TERRAIN_COARSE = 16;
const TERRAIN_FINE = 4;
// steps of a texel near the water, longer ones further away: by this part of the distance. The ripples scatter the
// rays, a hill far away has no sharp edge in the water
const TERRAIN_STEP_GROWTH = 0.04;
// the steps and the skipped cells together
const TERRAIN_MAX_STEPS = 96;
// A horizon is marched in steps of this many texels, at every distance: with the longer steps far away of traceTerrain
// the samples miss the top of a ridge by a part of a step, another part when the camera moves, and the edge of the
// reflection and the ground in it jump. It is 1 march per block of pixels, see waterTerrainHorizon.wgsl
const TERRAIN_HORIZON_STEP = 1.0;
// it goes on to the edge of the map over flat water, a coarse cell a step, and a texel a step over a slope
const TERRAIN_HORIZON_MAX_STEPS = 512;
// the halvings of the step in which the mirror ray hits the ground: to 1 / 64 of a texel
const TERRAIN_HIT_REFINE_STEPS = 6;

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

// a ray across the height map
struct TerrainRay {
  // where it starts, in texels, and its direction across the map (normalized, in texels)
  p : vec2<f32>,
  d : vec2<f32>,
  // the texels it crosses per world unit along rd. 0: straight up
  speed : f32,
}

// the map is an orthographic view: ro in texels, and the texels a ray along rd crosses per unit of its length. Texture
// v runs down, ndc y up
fn getTerrainRay(ro : vec3<f32>, rd : vec3<f32>, sizeF : vec2<f32>) -> TerrainRay {
  let c0 = (uniforms.terrainSunViewProjection * vec4<f32>(ro, 1.0)).xy;
  let c1 = (uniforms.terrainSunViewProjection * vec4<f32>(rd, 0.0)).xy;
  let dp = vec2<f32>(c1.x, -c1.y) * 0.5 * sizeF;
  let speed = length(dp);
  var ray : TerrainRay;
  ray.p = vec2<f32>(c0.x * 0.5 + 0.5, 0.5 - c0.y * 0.5) * sizeF;
  ray.d = dp / max(speed, 1e-9);
  ray.speed = select(speed, 0.0, speed < 1e-4);
  return ray;
}

// whether the ray from ro (on the water) along rd (normalized, going up) hits the ground. The ray starts
// terrainReflectionBias above the water: the water lies on the ground of the map, and the map has it a little off the
// mesh. It only rises, so it ends where it is above all the ground, and it skips the cells that are all below it: the
// coarse ones over the open water, the fine ones along a slope, which leaves the 4 texels of a bilinear height for the
// places where the ray is close to the ground
fn traceTerrain(ro : vec3<f32>, rd : vec3<f32>) -> bool {
  if (rd.y <= 0.0) {
    return false;
  }
  let size = vec2<i32>(textureDimensions(terrainHeights));
  let sizeF = vec2<f32>(size);
  let ray = getTerrainRay(ro, rd, sizeF);
  if (ray.speed == 0.0) {
    return false;
  }
  let d = ray.d;
  // the rise of the ray per texel it goes across
  let rise = rd.y / ray.speed;
  let y0 = ro.y + uniforms.terrainReflectionBias;
  let coarseLast = vec2<i32>(textureDimensions(terrainMaxHeights)) - 1;
  let fineLast = vec2<i32>(textureDimensions(terrainMaxHeightsFine)) - 1;
  var s = 1.0;
  for (var i = 0; i < TERRAIN_MAX_STEPS; i++) {
    let p = ray.p + d * s;
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

struct TerrainHorizon {
  // of the steepest angle at which the ground is in the way. 0: none is
  tangent : f32,
  // the ground the water mirrors, for its color: how far along direction it is (world units), and its height (world y)
  distance : f32,
  height : f32,
}

// The horizon of the ground from ro (on the water) toward direction (across the ground, normalized): the tangent of
// the steepest angle at which the ground is in the way, the highest (height above ro) / (distance to it). 0: open water
// to the edge of the map. A ray from there along rd hits the ground where rd.y < horizon * length(rd.xz). As
// traceTerrain, with the line to the highest ground so far for the ray: only the ground above it raises the horizon.
// And the ground the mirror ray itself hits, which rises by mirrorTangent: the first ground above it, which is where
// the horizon first goes above the ray, found between the last sample below the ground and that one by bisection (a
// hit at the sample itself moves a step when the samples do: with the camera). Where the ray passes over all the
// ground: the ground of the horizon, what the rays the ripples tilt down hit about
fn getTerrainHorizon(ro : vec3<f32>, direction : vec2<f32>, mirrorTangent : f32) -> TerrainHorizon {
  let size = vec2<i32>(textureDimensions(terrainHeights));
  let sizeF = vec2<f32>(size);
  let ray = getTerrainRay(ro, vec3<f32>(direction.x, 0.0, direction.y), sizeF);
  if (ray.speed == 0.0) {
    return TerrainHorizon(0.0, 0.0, 0.0);
  }
  // the rise of the mirror ray per texel
  let mirrorRise = mirrorTangent / ray.speed;
  // of the ground the water mirrors: in texels along d, and its height. hit: the mirror ray hit it
  var mirrored = vec2<f32>(0.0);
  var hit = false;
  let d = ray.d;
  let y0 = ro.y + uniforms.terrainReflectionBias;
  let coarseLast = vec2<i32>(textureDimensions(terrainMaxHeights)) - 1;
  let fineLast = vec2<i32>(textureDimensions(terrainMaxHeightsFine)) - 1;
  // the rise of the line per texel
  var rise = 0.0;
  var s = 0.5 * TERRAIN_HORIZON_STEP;
  // the last place the mirror ray was above the ground, while it hasn't hit: the line of the horizon is below it until
  // then, so the ground of a skipped cell is too
  var above = 0.0;
  for (var i = 0; i < TERRAIN_HORIZON_MAX_STEPS; i++) {
    let p = ray.p + d * s;
    let y = y0 + s * rise;
    if (y >= uniforms.terrainReflectionTop || any(p < vec2<f32>(0.0)) || any(p >= sizeF)) {
      break;
    }
    let c = clamp(vec2<i32>(floor(p / f32(TERRAIN_COARSE))), vec2<i32>(0), coarseLast);
    if (textureLoad(terrainMaxHeights, c, 0).x <= y) {
      above = s;
      s += getTerrainCellExit(p, d, c, TERRAIN_COARSE);
      continue;
    }
    let cf = clamp(vec2<i32>(floor(p / f32(TERRAIN_FINE))), vec2<i32>(0), fineLast);
    if (textureLoad(terrainMaxHeightsFine, cf, 0).x <= y) {
      above = s;
      s += max(getTerrainCellExit(p, d, cf, TERRAIN_FINE), TERRAIN_HORIZON_STEP);
      continue;
    }
    let h = getTerrainHeight(p, size);
    let slope = (h - y0) / s;
    if (!hit) {
      if (slope > mirrorRise) {
        hit = true;
        // the ray goes into the ground between above and s
        var lo = above;
        var hi = s;
        for (var k = 0; k < TERRAIN_HIT_REFINE_STEPS; k++) {
          let mid = 0.5 * (lo + hi);
          if (getTerrainHeight(ray.p + d * mid, size) > y0 + mid * mirrorRise) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        mirrored = vec2<f32>(hi, y0 + hi * mirrorRise);
      } else {
        above = s;
        if (slope > rise) {
          mirrored = vec2<f32>(s, h);
        }
      }
    }
    rise = max(rise, slope);
    s += TERRAIN_HORIZON_STEP;
  }
  // per texel to per world unit
  return TerrainHorizon(rise * ray.speed, mirrored.x / ray.speed, mirrored.y);
}
