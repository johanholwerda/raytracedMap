// The terrain in the way of the sun, from TerrainSunMap.ts: the ray traced sun shadows trace only the buildings and the
// trees (skyVisibilityTrace.wgsl), this adds the hills. The including shader declares
//   terrainSunMap : texture_2d<f32> (per texel how far toward the sun the ground nearest to it is: the dot with
//   lightDirection; -1e9 where none)
// and the uniforms terrainSunViewProjection (mat4x4<f32>), terrainSunOffset (f32) and lightDirection (toward the sun),
// see TerrainSunMap.bind.

// 1 where no ground is between the point p (with normal n) and the sun, 0 behind a hill: the 4 texels around it compared
// and blended bilinearly, a soft edge of a texel. p goes off the surface along n by terrainSunOffset first (a texel and a
// half): the map is too coarse for the ground to compare with itself, a slope would shadow its own lit side. Lit outside
// the map. A wall goes up by it too: off a wall along its normal is sideways, which over flat ground gains nothing
// toward the sun, and the foot of every wall was shadowed by the ground texels next to it (up to a texel high)
fn getTerrainSunShadow(p : vec3<f32>, n : vec3<f32>) -> f32 {
  let q = p + (n + vec3<f32>(0.0, 1.0 - abs(n.y), 0.0)) * uniforms.terrainSunOffset;
  let clip = uniforms.terrainSunViewProjection * vec4<f32>(q, 1.0);
  // texture v runs down, ndc y up (as in the cascades)
  let uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
  if (any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) {
    return 1.0;
  }
  let size = vec2<f32>(textureDimensions(terrainSunMap));
  let texel = uv * size - 0.5;
  let i = vec2<i32>(floor(texel));
  let f = texel - floor(texel);
  let last = vec2<i32>(size) - 1;
  // lit where the point is at least as far toward the sun as the ground the texel has
  let t = dot(q, uniforms.lightDirection);
  let s00 = step(textureLoad(terrainSunMap, clamp(i, vec2<i32>(0), last), 0).x, t);
  let s10 = step(textureLoad(terrainSunMap, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), last), 0).x, t);
  let s01 = step(textureLoad(terrainSunMap, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), last), 0).x, t);
  let s11 = step(textureLoad(terrainSunMap, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), last), 0).x, t);
  return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}
