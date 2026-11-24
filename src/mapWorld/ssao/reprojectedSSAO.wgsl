// SSAO accumulated with reprojection, after mapOSM/progressiveSSAO.wgsl. Each frame shoots at most ONE ray per pixel in
// a random hemisphere direction and folds it into the history of the same surface point, also while the camera moves.
// The output is (visibility, sample count, camera distance, 0); the distance is 0 on the sky.
//
// Moving: the world position of the pixel is projected with the previous camera (without jitter) and the history is
// read there bilinearly, each of the 4 taps only when its camera distance matches the one the point had then.
// Disoccluded pixels start again at 0 samples. Static: the history of the same pixel, without the depth test: the
// jitter of the TAA flips the edge pixels between surfaces, a test would restart them every frame.
// The sample count is capped at countMax first, which makes the running average 1 / (count + 1) an exponential one
// while moving or while tiles load, and is kept up to countLimit.
// Converging (static and loaded, countMax = countLimit): countLimit is reached in countLimit frames. Under the tilt-shift
// blur a pixel needs fewer samples: the blur averages its noise with the pixels around it, about blurSamples * sigma^2
// of them. Its limit is countLimit divided by that (at least countMin). A pixel at its limit is left as it is.
//
// Rays under the blur, in all modes: from |ndc y| = halfRateFrom the rows trace every 2nd frame, from quarterRateFrom
// every 4th, where the limit is at most a half and a quarter, so they converge in the same frames. Only the ray is
// skipped, the history is reprojected every frame. A band of rate 1 / k is cut into k pieces of equal height that each
// trace on their own frame of the k, so every frame traces the same number of rows, and the decision is the same for
// all pixels of a piece: the waves stay coherent, only the ones on the edges of a piece diverge.

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

struct Uniforms {
  viewProjection : mat4x4<f32>,
  inverseProjection : mat4x4<f32>,
  inverseView : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  viewProjectionNoJitter : mat4x4<f32>,
  viewProjectionOldNoJitter : mat4x4<f32>,
  cameraPositionOld : vec3<f32>,
  radius : f32,
  bias : f32,
  scale : f32,
  distanceMax : f32,
  rayMarchSteps : u32,
  skyLight : f32,
  frame : u32,
  resolution : vec2<f32>,
  historyValid : u32,
  cameraStatic : u32,
  countMax : f32,
  countLimit : f32,
  //relative camera distance difference at which a history tap is a different surface, on top of the depth slope
  depthTolerance : f32,
  //static and loaded: the count stops at the limit of getCountLimit
  converging : u32,
  //the |ndc y| from which the rows trace every 2nd and every 4th frame, 2 for none, see getTrace
  halfRateFrom : f32,
  quarterRateFrom : f32,
  //the tilt-shift blur: its sigma in pixels at |ndc y| = 1 and at most, see TiltShift.getBlur
  blurSlope : f32,
  blurMax : f32,
  //pixels the blur averages per sigma^2, see getCountLimit
  blurSamples : f32,
  //the fewest samples a pixel converges to under the blur
  countMin : f32,
  // see getGBufferDistance
  depthParams : vec2<f32>,
  //1 step a frame at a random radius over the range of the rayMarchSteps steps, see fs_main
  randomStep : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// the gBuffer of MapWorld: its depth (the camera distance) and its target 0 (the normal), see mapGBufferPacking.wgsl
@group(0) @binding(1) var gBuffer : texture_2d<u32>;
@group(0) @binding(2) var gBufferDepth : texture_depth_2d;
@group(0) @binding(3) var blueNoise : texture_2d<f32>;
@group(0) @binding(4) var texture : texture_2d<f32>; // accumulation history

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  output.uv = input.uv;
  return output;
}

const PI : f32 = 3.14159265359;
const TAU : f32 = 6.28318530718;
const timePeriod = 1024u;

// the view depth of a pixel, 0 on the sky. 1 / the view depth is depth * x + y, see getGBufferDistance. Times the
// camera distance per view depth of the pixel (k in fs_main) it is the camera distance: the neighbours and the ray
// march samples use the k of the pixel, which differs from their own by ~1% at most, and saves the inverse projection
fn loadViewDepth(coord : vec2<i32>) -> f32 {
  let depth = textureLoad(gBufferDepth, coord, 0);
  return select(0.0, 1.0 / (depth * uniforms.depthParams.x + uniforms.depthParams.y), depth > 0.0);
}

//the samples the pixel converges to: fewer under the tilt-shift blur, which averages it with the pixels around it
fn getCountLimit(uv : vec2<f32>) -> f32 {
  let ndcY = abs(1.0 - 2.0 * uv.y);
  let sigma = min(uniforms.blurSlope * ndcY, uniforms.blurMax);
  return clamp(uniforms.countLimit / max(uniforms.blurSamples * sigma * sigma, 1.0), uniforms.countMin, uniforms.countLimit);
}

//(trace this frame, index of the sample of this pixel) for the rows at uv.y, see the top. Rate 1 / 2: the top half
//traces on even frames, the bottom on odd ones. Rate 1 / 4: the near and far half of the top and of the bottom band.
fn getTrace(uv : vec2<f32>, frame : u32) -> vec2<u32> {
  let ndcY = 1.0 - 2.0 * uv.y;
  let a = abs(ndcY);
  let bottom = select(0u, 1u, ndcY < 0.0);
  var period = 1u;
  var piece = 0u;
  if (a >= uniforms.quarterRateFrom) {
    period = 4u;
    let far = select(0u, 1u, a >= 0.5 * (uniforms.quarterRateFrom + 1.0));
    piece = bottom + 2u * far;
  } else if (a >= uniforms.halfRateFrom) {
    period = 2u;
    piece = bottom;
  }
  return vec2(select(0u, 1u, frame % period == piece), frame / period);
}

//whether the depth at p0 is in front of it, by less than distanceMax. k: see loadViewDepth
fn isOccluded(p0 : vec3<f32>, k : f32) -> bool {
  // clip.w is the view depth of the sample: the depth difference in view depth, scaled to camera distance with k
  let clip = uniforms.viewProjection * vec4(p0, 1.0);
  let uv = vec2(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
  let delta = (loadViewDepth(vec2<i32>(uv * uniforms.resolution)) - clip.w) * k + uniforms.bias;
  return delta < 0.0 && -delta < uniforms.distanceMax;
}

fn hemiSphereRandom(r : vec2<f32>, n : vec3<f32>) -> vec3<f32> {
  let x = r.x * 2.0 - 1.0;
  let y = r.y * TAU;
  let v = vec3(sqrt(1.0 - x * x) * vec2(sin(y), cos(y)), x);
  return v * sign(dot(v, n));
}

//the golden ratio sequence in 0.32 fixed point, the 1d version of R2: n * GOLDEN_FIXED wraps around exactly as well
const GOLDEN_FIXED = 2654435769u;

fn goldenRatio(n: u32) -> f32 {
	return f32(n * GOLDEN_FIXED) * 2.3283064e-10;
}

//the uv of a world position, 0 to 1, y down
fn projectToUV(viewProjection : mat4x4<f32>, p : vec3<f32>) -> vec2<f32> {
  let clip = viewProjection * vec4(p, 1.);
  let ndc = clip.xy / clip.w;
  return vec2(ndc.x * .5 + .5, .5 - ndc.y * .5);
}

//the smaller camera distance step to the 2 neighbours along an axis: the slope of the surface, not of an edge next to it.
//A sky neighbour has no step
fn depthStep(d : f32, a : f32, b : f32) -> f32 {
  var s = 1e30;
  if (a > 0.0) { s = abs(d - a); }
  if (b > 0.0) { s = min(s, abs(d - b)); }
  return select(s, 0.0, s > 1e29);
}

//(visibility, sample count) of the history at the point p, count 0 without one. k: see loadViewDepth
fn reprojectHistory(coord : vec2<i32>, uv : vec2<f32>, p : vec3<f32>, d : f32, k : f32) -> vec2<f32> {
  if (uniforms.historyValid == 0u) {
    return vec2(0.0);
  }
  if (uniforms.cameraStatic != 0u) {
    let h = textureLoad(texture, coord, 0);
    return select(vec2(0.0), h.xy, h.z > 0.0);
  }

  let uvOld = uv + projectToUV(uniforms.viewProjectionOldNoJitter, p) - projectToUV(uniforms.viewProjectionNoJitter, p);
  let expected = distance(p, uniforms.cameraPositionOld);

  //the taps are up to a pixel away from the point: allow the distance step of the surface over that
  let maxCoord = vec2<i32>(uniforms.resolution) - 1;
  let sx = depthStep(d, k * loadViewDepth(clamp(coord - vec2(1, 0), vec2(0), maxCoord)), k * loadViewDepth(clamp(coord + vec2(1, 0), vec2(0), maxCoord)));
  let sy = depthStep(d, k * loadViewDepth(clamp(coord - vec2(0, 1), vec2(0), maxCoord)), k * loadViewDepth(clamp(coord + vec2(0, 1), vec2(0), maxCoord)));
  let allowed = expected * uniforms.depthTolerance + 1.5 * (sx + sy);

  let position = uvOld * uniforms.resolution - 0.5;
  let base = vec2<i32>(floor(position));
  let f = position - floor(position);
  var sum = vec2(0.0);
  var weightSum = 0.0;
  for (var i = 0; i < 4; i++) {
    let o = vec2(i & 1, i >> 1);
    let c = base + o;
    if (any(c < vec2(0)) || any(c > maxCoord)) {
      continue;
    }
    let h = textureLoad(texture, c, 0);
    if (h.z <= 0.0 || abs(h.z - expected) > allowed) {
      continue;
    }
    let w = select(1.0 - f.x, f.x, o.x == 1) * select(1.0 - f.y, f.y, o.y == 1);
    sum += h.xy * w;
    weightSum += w;
  }
  if (weightSum < 1e-3) {
    return vec2(0.0);
  }
  return sum / weightSum;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  let coord = vec2<u32>(in.uv * uniforms.resolution);
  let frameU = uniforms.frame;

  let viewDepth = loadViewDepth(vec2<i32>(coord));
  // Background (no geometry): visibility 1 and no samples, so the composite never darkens empty pixels.
  if (viewDepth <= 0.0) {
    return vec4(1.0, 0.0, 0.0, 0.0);
  }

  // the eye ray of the pixel at view depth 1, and the camera distance per view depth along it
  let eye = uniforms.inverseProjection * vec4(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0, 1.0, 1.0);
  let k = length(eye.xyz) / abs(eye.z);
  let d = viewDepth * k;
  let ray = normalize((uniforms.inverseView * vec4(eye.xyz, 0.0)).xyz);
  let p = uniforms.cameraPosition + ray * d;
  let n = unpackGBufferNormal(textureLoad(gBuffer, coord, 0).y);

  let history = reprojectHistory(vec2<i32>(coord), in.uv, p, d, k);
  let count = min(history.y, uniforms.countMax);
  var countLimit = uniforms.countLimit;
  if (uniforms.converging != 0u) {
    countLimit = getCountLimit(in.uv);
    if (count >= countLimit) {
      return vec4(history.x, count, d, 0.0);
    }
  }
  let trace = getTrace(in.uv, frameU);
  if (trace.x == 0u) {
    return vec4(history.x, count, d, 0.0);
  }

  let steps = uniforms.rayMarchSteps;
  // Per-pixel blue noise decorrelates neighbours; the golden-ratio offset by the index of the sample (consecutive
  // for a pixel, also under the blur where it skips frames) gives each ray a fresh, low-discrepancy direction.
  let h = textureLoad(blueNoise, coord % 256, 0).r;
  let index = trace.y;
  let t = u32(h * f32(timePeriod)) + index;
  let dir = hemiSphereRandom(getR2(t % timePeriod), n);
  let light = 1.0 + dir.y * uniforms.skyLight;
  let u = fract(h + goldenRatio(index));
  // randomStep: 1 depth read, at a log-uniform radius over the range the steps of the march cover together (radius * .5
  // up to radius * .5 * scale^steps), the history averages them. ~44% of the occlusion of the march (an occluded step
  // makes all the ones past it count as occluded, 1 step doesn't see that): the composite amount makes up for it.
  // (The march up to a random last step has the mean of the march, but its tiles of a shared last step showed.)
  // 1 loop for both: with the test in 2 branches the march came out different (~.011 lighter) on dawn/D3D12
  let random = uniforms.randomStep != 0u;
  var rad = select(uniforms.radius * (0.5 + 0.5 * u), uniforms.radius * 0.5 * pow(uniforms.scale, u * f32(steps)), random);
  let marchSteps = select(steps, 1u, random);
  // the index of the first occluded step, marchSteps for none
  var hit = marchSteps;
  for (var j = 0u; j < marchSteps; j += 1u) {
    if (isOccluded(p + dir * rad, k)) {
      hit = j;
      break;
    }
    rad *= uniforms.scale;
  }
  let visibility = f32(hit) * light / f32(marchSteps);

  let ao = mix(history.x, visibility, 1.0 / (count + 1.0));
  return vec4(ao, min(count + 1.0, countLimit), d, 0.0);
}
