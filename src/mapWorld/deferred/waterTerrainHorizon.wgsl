// The horizon of the ground over the water of MapWorld, for the terrain in its reflections, see WaterTerrainHorizon.ts:
// for 1 pixel per block of TERRAIN_HORIZON_BLOCK x TERRAIN_HORIZON_BLOCK pixels of the gBuffer, how steep a ray along
// its mirror ray has to go up to pass over the ground (getTerrainHorizon of terrainReflection.wgsl, a march over the
// height map). The shading and the half resolution pass of the reflections compare the ray of each water pixel with the
// horizons around it (isTerrainOccluded of terrainHorizonLookup.wgsl) instead of marching it.
// Out, r16float: the horizon (a tangent, 0 open water), -1 where the pixel isn't water
// Needs cameraRay.wgsl, cityLightmapSurface.wgsl, mapGBufferPacking.wgsl and terrainReflection.wgsl

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
};

// the pixels of a block along x and y, and the one that is traced: mirror terrainHorizonLookup.wgsl
const BLOCK = 4u;
const OFFSET = 2u;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// the surface id, the normal and the material of the gBuffer, and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(1) var gBufferTexture : texture_2d<u32>;
@group(0) @binding(2) var gBufferDepth : texture_depth_2d;
// the height map of the ground, see TerrainSunMap.bindHeights
@group(0) @binding(3) var terrainHeights : texture_2d<f32>;
@group(0) @binding(4) var terrainMaxHeights : texture_2d<f32>;
@group(0) @binding(5) var terrainMaxHeightsFine : texture_2d<f32>;

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

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  // the size of the target rounds up: a block that is cut off by the edge has its sample at the last pixel
  let coord = min(vec2<u32>(input.position.xy) * BLOCK + OFFSET, vec2<u32>(uniforms.resolution) - 1u);
  let g = vec3(textureLoad(gBufferTexture, coord, 0).xy, 0u);
  if (g.x == SURFACE_NONE || ((uniforms.waterMaterials >> getGBufferMaterial(g)) & 1u) == 0u) {
    return vec4<f32>(-1.0);
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
  return vec4<f32>(getTerrainHorizon(world, rd.xz / across), 0.0, 0.0, 1.0);
}
