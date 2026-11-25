// The passes of TerrainSunMap.ts over the height map of the ground (rendered top down, per texel -height: the depth
// material of the cascades, -1e9 where no ground). Two kernels, one per material (the entry points are picked by the
// defines):
//   MAX_HEIGHTS: per cell of COARSE x COARSE texels the highest ground in it and a texel around it (a bilinear sample in
//     the cell reads one texel past it), so the march skips the cells it passes over
//   SHADOW_HEIGHT: per texel the height a point there has to be at to see the sun, see fs_main

// texels of the height map per cell of maxHeights, mirrors COARSE of TerrainSunMap.ts
const COARSE = 16;
// the march takes steps of a texel near the texel, longer ones further away: by this part of the distance. A hill far
// away shadows with a wide penumbra, its exact place matters less
const STEP_GROWTH = 0.02;
const MAX_STEPS = 1024;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
}

struct Uniforms {
  // the direction toward the sun across the map, in texels (x along +x, y along +z), normalized. 0: the sun is overhead
  sunDirection : vec2<f32>,
  // how much a ray toward the sun rises per texel it goes across, world units. Negative: the sun is below the horizon
  risePerTexel : f32,
  // the half width of the penumbra per texel of distance to the hill that casts the shadow (the angular radius of the
  // sun), and at the least, world units of height
  penumbraPerTexel : f32,
  penumbraMin : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var heights : texture_2d<f32>;
#ifdef SHADOW_HEIGHT
@group(0) @binding(2) var maxHeights : texture_2d<f32>;
#endif

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  return output;
}

#ifdef MAX_HEIGHTS
@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let size = vec2<i32>(textureDimensions(heights));
  let c = vec2<i32>(input.position.xy) * COARSE;
  var m = -1e30;
  for (var y = c.y - 1; y <= c.y + COARSE; y++) {
    for (var x = c.x - 1; x <= c.x + COARSE; x++) {
      let t = clamp(vec2<i32>(x, y), vec2<i32>(0), size - 1);
      m = max(m, -textureLoad(heights, t, 0).x);
    }
  }
  return vec4<f32>(m, 0.0, 0.0, 1.0);
}
#endif

#ifdef SHADOW_HEIGHT
// the ground height at p (texel coordinates, the centers at .5), bilinear
fn heightAt(p : vec2<f32>, size : vec2<i32>) -> f32 {
  let t = p - 0.5;
  let i = vec2<i32>(floor(t));
  let f = t - floor(t);
  let last = size - 1;
  let h00 = textureLoad(heights, clamp(i, vec2<i32>(0), last), 0).x;
  let h10 = textureLoad(heights, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), last), 0).x;
  let h01 = textureLoad(heights, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), last), 0).x;
  let h11 = textureLoad(heights, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), last), 0).x;
  return -mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// r: the height H a point at this texel has to be at to see the sun over the ground toward it,
//   H = max over s of (ground(x + s * sunDirection) - s * risePerTexel), s > 0 in texels.
//   A max of continuous functions of x, so H is continuous: interpolated bilinearly, the edge of a shadow (where a point
//   is at H) is a smooth curve, not the texel grid as with a map rendered from the sun. And the texels lie on the
//   ground, they don't stretch at a low sun.
// g: the half width of the penumbra there, of the distance to the ground that sets H
@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  let size = vec2<i32>(textureDimensions(heights));
  let sizeF = vec2<f32>(size);
  let d = uniforms.sunDirection;
  let rise = uniforms.risePerTexel;
  if (rise < 0.0) {
    // the sun is below the horizon: all in shadow
    return vec4<f32>(1e30, uniforms.penumbraMin, 0.0, 1.0);
  }
  if (dot(d, d) < 0.25) {
    // the sun is overhead: no ground in the way
    return vec4<f32>(-1e30, uniforms.penumbraMin, 0.0, 1.0);
  }
  let p0 = input.position.xy;
  let coarseLast = vec2<i32>(textureDimensions(maxHeights)) - 1;
  var best = -1e30;
  var bestS = 0.0;
  var s = 1.0;
  for (var i = 0; i < MAX_STEPS; i++) {
    let p = p0 + d * s;
    if (any(p < vec2<f32>(0.0)) || any(p >= sizeF)) {
      break;
    }
    // the ground further along raises H only where it is above best + s * rise, which only grows with s: skip to where
    // the ray leaves a cell whose ground is all below that
    let c = clamp(vec2<i32>(floor(p / f32(COARSE))), vec2<i32>(0), coarseLast);
    if (textureLoad(maxHeights, c, 0).x <= best + s * rise) {
      let lo = vec2<f32>(c * COARSE);
      let edge = select(lo, lo + f32(COARSE), d > vec2<f32>(0.0));
      let t = select(vec2<f32>(1e9), (edge - p) / d, abs(d) > vec2<f32>(1e-6));
      s += max(min(t.x, t.y), 0.0) + 0.01;
      continue;
    }
    let h = heightAt(p, size) - s * rise;
    if (h > best) {
      best = h;
      bestS = s;
    }
    s += max(1.0, s * STEP_GROWTH);
  }
  return vec4<f32>(best, uniforms.penumbraMin + bestS * uniforms.penumbraPerTexel, 0.0, 1.0);
}
#endif
