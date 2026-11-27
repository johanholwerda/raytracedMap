// The smooth normal of the ground in the .w of a vertex of the terrain (encodeTerrainNormal in terrainGrid.ts): x and z
// in 11 bits each, w = qx * 2048 + qz, y from them (the ground faces up). Negative (NO_NORMAL, NO_FEATURE_ID): up, the
// flat ground quad, the flat layers and the water.
fn decodeTerrainNormal(w : f32) -> vec3<f32> {
  if (w < 0.0) {
    return vec3<f32>(0.0, 1.0, 0.0);
  }
  let q = u32(w + 0.5);
  let xz = vec2<f32>(f32(q >> 11u), f32(q & 2047u)) / 2047.0 * 2.0 - 1.0;
  return vec3<f32>(xz.x, sqrt(max(1.0 - dot(xz, xz), 0.0)), xz.y);
}
