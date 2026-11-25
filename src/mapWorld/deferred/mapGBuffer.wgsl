// The gBuffer of MapWorld, see MapGBuffer.ts and the packing in mapGBufferPacking.wgsl. The targets are cleared to 0
// and the depth to 1, which is the sky. The camera distance is the depth.
// WALLS: the buildings and the trees, with the lightmap attribute (see city/cityLayout.ts) and in .w the feature id for
// the selection highlight, or on a tree its palette entry (see map/worldTrees.ts), with the flat normal of the triangle.
// Without it the ground layers and the ground quad, on the ground chart of the lightmap, with the smooth normal of the
// terrain in .w (see terrainNormal.wgsl).

struct VertexInput {
#ifdef WALLS
  // unorm16: position * positionScale + positionOffset, see quantizeVertices in worldGeometryCore.ts
  @location(0) position : vec4<f32> /* unorm16 */,
  // a wall: its atlas position (u, v); a roof: (roof chart, -1). See buildBuildings in worldGeometryCore.ts
  @location(1) lightmap : vec2<f32>,
#else
  @location(0) position : vec4<f32>,
#endif
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  // the position relative to the camera: the difference of 2 close f32 positions is exact, the world position far from
  // the projection center is not (the atlas position along a wall below)
  @location(0) relative : vec3<f32>,
#ifdef WALLS
  // flat: the attribute of the first vertex of the triangle, on a wall the start of it (its bottom corner, in corner),
  // see buildBuildings in worldGeometryCore.ts
  @location(1) @interpolate(flat) lightmap : vec2<f32>,
  @location(2) @interpolate(flat) selected : u32,
  @location(3) @interpolate(flat) material : u32,
  // relative to the camera, like relative
  @location(4) @interpolate(flat) corner : vec3<f32>,
#else
  // the smooth normal of the ground, see terrainNormal.wgsl
  @location(1) normal : vec3<f32>,
#endif
};

struct Uniforms {
  viewProjection : mat4x4<f32>,
  cameraPosition : vec3<f32>,
  // the index of the albedo in the palette of MapGBuffer
  material : u32,
  // the step of the wall atlas positions, see MapGBuffer.atlasStep
  atlasStep : f32,
  // the global id of the selected building (slot * 65536 + feature id), -1 for none. tileID: the slot of the mesh
  selectedID : i32,
  tileID : u32,
  // the map tile of the mesh (x, y), whose wall atlas the walls are on, see getWallSurfaceId
  wallTile : vec2<f32>,
  // the palette index of the first tree color (the trunk), see MapGeometry
  treeMaterial : u32,
  // WALLS: the decoding of the unorm16 vertices of the mesh, see quantizeVertices in worldGeometryCore.ts
  positionScale : vec4<f32>,
  positionOffset : vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

// the .w of a tree vertex with palette entry k is TREE_MATERIAL_BASE_ID - k, see map/worldTrees.ts
const TREE_MATERIAL_BASE_ID = -2.0;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
#ifdef WALLS
  let q = input.position * uniforms.positionScale;
  // the offset minus the camera first: the 2 are close, so that is exact
  output.relative = q.xyz + (uniforms.positionOffset.xyz - uniforms.cameraPosition);
  output.position = uniforms.viewProjection * vec4<f32>(q.xyz + uniforms.positionOffset.xyz, 1.0);
  output.lightmap = input.lightmap;
  output.corner = output.relative;
  // like shading.wgsl with FEATURE_IDS
  let featureId = round(q.w + uniforms.positionOffset.w);
  output.selected = 0u;
  if (featureId >= 0.0 && uniforms.selectedID >= 0) {
    let globalId = uniforms.tileID * 65536u + u32(featureId + 0.5);
    output.selected = select(0u, 1u, globalId == u32(uniforms.selectedID));
  }
  output.material = uniforms.material;
  if (featureId < TREE_MATERIAL_BASE_ID + 0.5) {
    output.material = uniforms.treeMaterial + u32(TREE_MATERIAL_BASE_ID - featureId + 0.5);
  }
#else
  output.position = uniforms.viewProjection * vec4<f32>(input.position.xyz, 1.0);
  output.relative = input.position.xyz - uniforms.cameraPosition;
  output.normal = decodeTerrainNormal(input.position.w);
#endif
  return output;
}

struct FragmentOutput {
  @location(0) surface : vec2<u32>,
  @location(1) atlas : u32,
}

@fragment
fn fs_main(input : VertexOutput) -> FragmentOutput {
#ifdef WALLS
  // the flat normal facing the camera
  var n = normalize(cross(dpdx(input.relative), dpdy(input.relative)));
  if (dot(n, input.relative) > 0.0) {
    n = -n;
  }
#else
  let n = normalize(input.normal);
#endif
  var id = SURFACE_GROUND;
  var atlas = vec2<u32>(0u);
  var selected = false;
  var material = uniforms.material;
#ifdef WALLS
  // on a wall the atlas position of its start corner plus the distance along the wall and the height over the corner
  var lightmap = vec4<f32>(input.lightmap.x, 0.0, -1.0, -1.0);
  if (input.lightmap.y >= 0.0) {
    lightmap = vec4<f32>(input.lightmap + vec2<f32>(length(input.relative.xz - input.corner.xz), input.relative.y - input.corner.y), uniforms.wallTile);
  }
  let surface = getAttributeSurface(lightmap, input.relative + uniforms.cameraPosition);
  id = surface.id;
  // a roof is on world xz, like the ground: only a wall stores its position
  if ((id & SURFACE_WALL) != 0u) {
    atlas = vec2<u32>(clamp(round(surface.position / uniforms.atlasStep), vec2(0.), vec2(f32(GBUFFER_ATLAS_MAX))));
  }
  selected = input.selected != 0u;
  material = input.material;
#endif
  let g = packGBuffer(id, n, material, selected, atlas);
  return FragmentOutput(g.xy, g.z);
}
