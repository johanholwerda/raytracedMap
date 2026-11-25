// The terrain in the way of the sun, from TerrainSunMap.ts: the ray traced sun shadows trace only the buildings and the
// trees (skyVisibilityTrace.wgsl), this adds the hills. The including shader declares
//   terrainSunMap : texture_2d<f32> (a map of the ground top down, per texel r: the height a point there has to be at to
//   see the sun, g: the half width of the penumbra there, world units, see terrainSunHeight.wgsl)
// and the uniforms terrainSunViewProjection (mat4x4<f32>, world to the map), terrainSunOffset (f32), see
// TerrainSunMap.bind.

// 1 where no ground is between the point p (with normal n) and the sun, 0 behind a hill, soft over the penumbra: the
// height of p against the height the map has there, interpolated bilinearly. That height is continuous, so the edge is
// a smooth curve and not the texel grid. p goes off the surface along n by terrainSunOffset first (a part of a texel):
// the map has the ground interpolated between its texels, the triangles of the ground are a little off that. A wall goes
// up by it too: off a wall along its normal is sideways. Lit outside the map
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
  let m00 = textureLoad(terrainSunMap, clamp(i, vec2<i32>(0), last), 0).xy;
  let m10 = textureLoad(terrainSunMap, clamp(i + vec2<i32>(1, 0), vec2<i32>(0), last), 0).xy;
  let m01 = textureLoad(terrainSunMap, clamp(i + vec2<i32>(0, 1), vec2<i32>(0), last), 0).xy;
  let m11 = textureLoad(terrainSunMap, clamp(i + vec2<i32>(1, 1), vec2<i32>(0), last), 0).xy;
  let m = mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
  // smoothstep(m.x - m.y, m.x + m.y, q.y), but without the edges falling together where m.x is +-1e30 (no ground, the
  // sun below the horizon): a width of 0 there would be a division by 0
  let t = clamp((q.y - m.x) / max(2.0 * m.y, 1e-9) + 0.5, 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}
