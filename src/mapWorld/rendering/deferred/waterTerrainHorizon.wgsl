// The horizon of the ground over the water of MapWorld, for the terrain in its reflections, see WaterTerrainHorizon.ts:
// for 1 pixel per block of TERRAIN_HORIZON_BLOCK x TERRAIN_HORIZON_BLOCK pixels of the gBuffer, how steep a ray along
// its mirror ray has to go up to pass over the ground (getTerrainHorizon of terrainReflection.wgsl, a march over the
// height map), and the color of the ground the water mirrors there: shaded as the deferred shading does the ground (the
// sun with the shadows of the hills, the sky term of an open sky, the snow), from the height map and the materials of
// the ground top down (TerrainSunMap.ts). The shading compares the ray of each water pixel with the horizons around it
// and takes their color (getTerrainReflection of terrainHorizonLookup.wgsl) instead of marching and shading it.
// Out, rgba16float. a: the horizon (a tangent, 0 open water), -1 where the pixel isn't water. rgb: the color, linear
// (as the shading's before its exposure), with a horizon above 0
// Needs cameraRay.wgsl, cityLightmapSurface.wgsl, mapGBufferPacking.wgsl, terrainReflection.wgsl, terrainSun.wgsl and
// snow.wgsl

struct Uniforms {
  inverseProjection : mat4x4<f32>,
  inverseView : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  // of the gBuffer, the full resolution
  resolution : vec2<f32>,
  // see getGBufferDistance
  depthParams : vec2<f32>,
  // the gBuffer materials of the water, a bit each
  waterMaterials : u32,
  // the height map, see terrainReflection.wgsl
  terrainSunViewProjection : mat4x4<f32>,
  terrainReflectionTop : f32,
  terrainReflectionBias : f32,
  // the shading of the ground, the params of the deferred shading by the same names (gamma space colors)
  lightDirection : vec3<f32>,
  skyColor : vec3<f32>,
  sunColor : vec3<f32>,
  shadowStrength : f32,
  // see getTerrainSunShadow
  terrainSunOffset : f32,
  // the snow, see snow.wgsl and MapGeometry.setSnow: 0 none in the scene, the material of it, the snow line in world y,
  // how far (m) the noise moves it, and the meters per world unit
  snowEnabled : u32,
  snowMaterial : u32,
  snowLine : f32,
  snowNoise : f32,
  snowMetersPerUnit : f32,
};

// the pixels of a block along x and y, and the one that is traced: mirror terrainHorizonLookup.wgsl
const BLOCK = 4u;
const OFFSET = 2u;
// the texels of terrainMaterials, see MapGeometry.drawMaterialMap: the palette index + 1 in the low bits, and the snow
// lies on it
const TERRAIN_MATERIAL_MASK = 255u;
const TERRAIN_MATERIAL_SNOW = 256u;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// the surface id, the normal and the material of the gBuffer, and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(1) var gBufferTexture : texture_2d<u32>;
@group(0) @binding(2) var gBufferDepth : texture_depth_2d;
// the height map of the ground, see TerrainSunMap.bindHeights
@group(0) @binding(3) var terrainHeights : texture_2d<f32>;
@group(0) @binding(4) var terrainMaxHeights : texture_2d<f32>;
@group(0) @binding(5) var terrainMaxHeightsFine : texture_2d<f32>;
// the materials of the ground on the same map, see TerrainSunMap.bindMaterials
@group(0) @binding(6) var terrainMaterials : texture_2d<u32>;
// the albedo of each material, see MapGBuffer.paletteBuffer
@group(0) @binding(7) var<storage, read> materialPalette : array<vec4<f32>>;
// the hills in the way of the sun, see terrainSun.wgsl
@group(0) @binding(8) var terrainSunMap : texture_2d<f32>;

struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) uv : vec2<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
}

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.position = vec4<f32>(input.position, 1.0);
  return output;
}

// the color of the ground at world (on it), shaded: linear, as the deferred shading has it before its exposure. The
// normal is of the height map, the material of the map of them, with the snow where it lies. The sun is in the shadow
// of the hills (not of the buildings and the trees: no ray), the sky term is that of an open sky (no lightmap there)
fn getTerrainColor(world : vec3<f32>) -> vec3<f32> {
  let size = vec2<i32>(textureDimensions(terrainHeights));
  let sizeF = vec2<f32>(size);
  let clip = (uniforms.terrainSunViewProjection * vec4<f32>(world, 1.0)).xy;
  let p = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5) * sizeF;
  // a texel in world units. The map has x along +x and v along +z
  let texel = 2.0 / (uniforms.terrainSunViewProjection[0][0] * sizeF.x);
  let dx = getTerrainHeight(p + vec2<f32>(1.0, 0.0), size) - getTerrainHeight(p - vec2<f32>(1.0, 0.0), size);
  let dz = getTerrainHeight(p + vec2<f32>(0.0, 1.0), size) - getTerrainHeight(p - vec2<f32>(0.0, 1.0), size);
  let n = normalize(vec3<f32>(-dx, 2.0 * texel, -dz));

  let material = textureLoad(terrainMaterials, clamp(vec2<i32>(floor(p)), vec2<i32>(0), size - 1), 0).x;
  // no ground in the map there (it has a height: a tile whose layers aren't there yet): grey
  var albedo = vec3<f32>(0.5);
  if (material != 0u) {
    var index = (material & TERRAIN_MATERIAL_MASK) - 1u;
    if (uniforms.snowEnabled != 0u && (material & TERRAIN_MATERIAL_SNOW) != 0u
      && isSnowPossible(world.y, uniforms.snowLine, uniforms.snowNoise, uniforms.snowMetersPerUnit)
      && getSnowField(world, n, uniforms.snowLine, uniforms.snowNoise, uniforms.snowMetersPerUnit) > 0.0) {
      index = uniforms.snowMaterial;
    }
    albedo = materialPalette[index].rgb;
  }
  albedo *= albedo;
  let nDotL = max(dot(n, uniforms.lightDirection), 0.0);
  var shadow = 1.0;
  if (nDotL > 0.0) {
    shadow = mix(1.0, getTerrainSunShadow(world, n), uniforms.shadowStrength);
  }
  return albedo * (uniforms.sunColor * uniforms.sunColor * (nDotL * shadow) + uniforms.skyColor * uniforms.skyColor * (n.y * 0.5 + 0.5));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  // the size of the target rounds up: a block that is cut off by the edge has its sample at the last pixel
  let coord = min(vec2<u32>(input.position.xy) * BLOCK + OFFSET, vec2<u32>(uniforms.resolution) - 1u);
  let g = vec3(textureLoad(gBufferTexture, coord, 0).xy, 0u);
  if (g.x == SURFACE_NONE || ((uniforms.waterMaterials >> getGBufferMaterial(g)) & 1u) == 0u) {
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
  }
  let uv = (vec2<f32>(coord) + .5) / uniforms.resolution;
  let d = getGBufferDistance(textureLoad(gBufferDepth, coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
  let world = depthToPosition(d, uv * 2. - 1.);
  // the mirror ray of the flat water, across the ground
  let rd = reflect(normalize(world - uniforms.cameraPosition), getGBufferNormal(g));
  let across = length(rd.xz);
  if (across < 1e-4) {
    // straight up: no ground in the way
    return vec4<f32>(0.0);
  }
  let direction = rd.xz / across;
  let horizon = getTerrainHorizon(world, direction, rd.y / across);
  if (horizon.tangent <= 0.0) {
    return vec4<f32>(0.0);
  }
  let hit = vec3<f32>(world.x + direction.x * horizon.distance, horizon.height, world.z + direction.y * horizon.distance);
  return vec4<f32>(getTerrainColor(hit), horizon.tangent);
}
