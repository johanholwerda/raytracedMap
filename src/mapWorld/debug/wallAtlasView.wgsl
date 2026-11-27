// The wallAtlas drawMode: the walls of a tile's building mesh at their wall atlas position (see layoutWalls in
// cityLayout.ts) instead of in the world, the whole atlas fitted to the screen, v down: the mesh WallAtlasView makes of
// them, without the roofs (they're on the roof charts). A building gets a color of its feature id, the trees are green.
// A grid line every 100 m, the part past what the gBuffer stores (GBUFFER_ATLAS_MAX steps) is striped red.

struct VertexInput {
  // xy: the atlas position, w: the feature id
  @location(0) position : vec4<f32>,
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) atlas : vec2<f32>,
  @location(1) @interpolate(flat) featureId : f32,
}

struct Uniforms {
  // clip = atlas * scale + offset, see WallAtlasView.draw
  scale : vec2<f32>,
  offset : vec2<f32>,
  // the most atlas position the gBuffer stores, in world units
  limit : f32,
  // 100 m in world units
  gridStep : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

// the .w of a tree vertex is TREE_MATERIAL_BASE_ID - k, see map/worldTrees.ts
const TREE_MATERIAL_BASE_ID = -2.0;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
  output.atlas = input.position.xy;
  output.featureId = input.position.w;
  output.position = vec4<f32>(input.position.xy * uniforms.scale + uniforms.offset, 0.5, 1.0);
  return output;
}

fn hashU(x : u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}

fn hue(h : f32) -> vec3<f32> {
  return clamp(abs(fract(h + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
  var col = vec3<f32>(0.6);
  if (input.featureId < TREE_MATERIAL_BASE_ID + 0.5) {
    col = mix(vec3<f32>(0.1, 0.45, 0.1), vec3<f32>(0.4, 0.75, 0.2), f32(hashU(u32(TREE_MATERIAL_BASE_ID - input.featureId + 0.5)) & 255u) / 255.0);
  } else if (input.featureId >= 0.0) {
    let h = hashU(u32(input.featureId + 0.5));
    col = mix(vec3<f32>(0.35), hue(f32(h & 1023u) / 1024.0), 0.75) * (0.7 + 0.3 * f32((h >> 10u) & 255u) / 255.0);
  }

  // the grid, about 1 pixel wide
  let cell = input.atlas / uniforms.gridStep;
  let line = abs(fract(cell + 0.5) - 0.5) / max(fwidth(cell), vec2<f32>(1e-6));
  col = mix(col, vec3<f32>(1.0), 0.5 * (1.0 - clamp(min(line.x, line.y), 0.0, 1.0)));

  if (any(input.atlas > vec2<f32>(uniforms.limit))) {
    let stripe = fract((input.position.x + input.position.y) / 16.0) < 0.5;
    col = mix(col, vec3<f32>(1.0, 0.0, 0.0), select(0.3, 0.7, stripe));
  }
  return vec4<f32>(col, 1.0);
}
