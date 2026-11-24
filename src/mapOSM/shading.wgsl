override DRAW_DEPTH: bool = false;

// Unified map vertex: xyz position + `.w` tile-local building id (or a negative
// sentinel for geometry with no id). One stride-4 layout shared by terrain,
// ground and buildings so they all draw through one pipeline.
struct VertexInput {
#ifdef QUANTIZED
  // the building meshes of mapWorld: unorm16, position * positionScale + positionOffset (see quantizeVertices in
  // mapWorld/map/worldGeometryCore.ts)
  @location(0) position : vec4<f32> /* unorm16 */,
#else
  @location(0) position : vec4<f32>,
#endif
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) world : vec3<f32>,
#ifdef FEATURE_IDS
  @location(1) @interpolate(flat) globalId : u32,
#endif
#ifdef TREE_COLORS
  // the palette index of a tree vertex, 0xFFFFFFFF for baseColor
  @location(2) @interpolate(flat) treeMaterial : u32,
#endif
};

struct Uniforms {
  viewProjection : mat4x4<f32>,
  lightDirection : vec3<f32>,
  baseColor : vec3<f32>,
  cameraPosition : vec3<f32>,
  skyColor : vec3<f32>,
  sunColor : vec3<f32>,
  ambient : f32,
  distanceFog : f32,
  // where the fog is 1 (mapWorld), 0 only the exp, see getFog
  viewDistanceMax : f32,
  shadowStrength : f32,
  VSMBleedThreshold : f32,
  varianceMin : f32,
  shadowDebug : u32,
  contrast : f32,
  drawIndex : u32,
  selectedID : i32,
  tileID : u32,
  backgroundColor : vec3<f32>,
  // TREE_COLORS: the palette index of the first tree color, see mapWorld/map/worldTrees.ts
  treeMaterial : u32,
  // QUANTIZED: the decoding of the vertices
  positionScale : vec4<f32>,
  positionOffset : vec4<f32>,
};

struct ShadowMapData {
  shadowMapViewProjection : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var shadowMapSampler : sampler;
// the 4 cascades, one layer each (ShadowMap.cascadeTexture)
@group(0) @binding(2) var shadowMaps : texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> shadowMapData : array<ShadowMapData>;
#ifdef TREE_COLORS
// the albedos of the gBuffer of mapWorld (MapGBuffer.paletteBuffer): a vertex with a .w of -2 - k (a tree, see
// mapWorld/map/worldTrees.ts) has the one at treeMaterial + k
@group(0) @binding(7) var<storage, read> materialPalette : array<vec4<f32>>;
#endif

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
  var output : VertexOutput;
#ifdef QUANTIZED
  let vertex = input.position * uniforms.positionScale + uniforms.positionOffset;
  let featureId = round(vertex.w);
#else
  let vertex = input.position;
  let featureId = input.position.w;
#endif
  let world = vertex.xyz;

  output.position = uniforms.viewProjection * vec4<f32>(world, 1.0);
  output.world = world;
#ifdef FEATURE_IDS
  // `.w` is the tile-local feature id, or negative (NO_FEATURE_ID) for untagged
  // geometry. Map "no id" to 0xFFFFFFFF so the index map writes 0 (empty) and it
  // can never be picked; otherwise pack (tileID, featureId) into the global id.
  if (featureId < 0.0) {
    output.globalId = 0xFFFFFFFFu;
  } else {
    output.globalId = uniforms.tileID * 65536u + u32(featureId + 0.5);
  }
#endif
#ifdef TREE_COLORS
  output.treeMaterial = 0xFFFFFFFFu;
  if (featureId < -1.5) {
    output.treeMaterial = uniforms.treeMaterial + u32(-2.0 - featureId + 0.5);
  }
#endif
  return output;
}

// --- Variance shadow map sampling ------------------------------------------

fn shadowUVDepthStrength(p : vec3<f32>, i : u32) -> vec4<f32> {
	let screen = shadowMapData[i].shadowMapViewProjection * vec4<f32>(p, 1.0);
	let uv = screen.xy / screen.w;
	// Compare in world height (-y), not view-space depth. The cascade matrix is
	// only used for the UV lookup; the occluder selection (nearest to the sun) is
	// done by the depth test when rendering the shadow map. Height is a correct
	// occlusion order along a sun ray and keeps the moments at a small, well-
	// conditioned scale (NDC depth squashed everything near 0.5 / 1.0, which broke
	// the VSM variance on part of the view -> the hard lit/shadow seam).
	let depth = -p.y * 1.;
	// Fade out over the outer 60% of the cascade, so the transition to the next one is gradual instead of a seam where
	// the shadows change thickness. The next one is twice as large: this is its inner 30%, where it has its full weight.
	let s2 = vec2<f32>(1.) - smoothstep(vec2<f32>(0.4), vec2<f32>(1.0), abs(uv));
	return vec4<f32>(uv, depth, min(s2.x, s2.y));
}

fn sampleShadowMoments(i : u32, uv : vec2<f32>) -> vec2<f32> {
	return textureSampleLevel(shadowMaps, shadowMapSampler, uv, i, 0.0).xy;
}

fn shadowCascade(uv : vec2<f32>, i : u32, dist : f32) -> f32 {
	var shadowMapUV = uv * 0.5 + 0.5;
	shadowMapUV.y = 1.0 - shadowMapUV.y;
	let moments = sampleShadowMoments(i, shadowMapUV);
	let lit = step(dist, moments.x);
	var variance = moments.y - moments.x * moments.x;
	variance = max(variance, uniforms.varianceMin * uniforms.varianceMin);
	let d = dist - moments.x;
	let s = variance / (variance + d * d);
	// The smoothstep clamps the upper variance tail, removing most VSM light
	// bleeding on the blurred depth.
	return smoothstep(uniforms.VSMBleedThreshold, 1.0, max(lit, s));
}

// The cascades front to back, each taking its weight (1 inside, fading out over its edge) of what the ones before it
// left. Each cascade is re-rendered around a center of its own, so the next one need not cover the edge of the one
// before it: that point goes on to the first one that does. What no cascade covers is lit, so past the last cascade the
// shadows fade out instead of ending at a hard edge.
fn getShadowMap(p : vec3<f32>) -> f32 {
	var shadow = 0.0;
	var remaining = 1.0;
	for (var i = 0u; i < 4u; i++) {
		let u = shadowUVDepthStrength(p, i);
		if (u.w > 0.0) {
			shadow += remaining * u.w * shadowCascade(u.xy, i, u.z);
			remaining *= 1.0 - u.w;
			if (remaining <= 0.0) {
				break;
			}
		}
	}
	return shadow + remaining;
}

// Debug: tint each cascade's coverage region a distinct colour so it's obvious
// which part of the view each cascade covers (and where coverage runs out).
fn cascadeDebugColor(p : vec3<f32>) -> vec3<f32> {
	for (var i = 0u; i < 4u; i++) {
		let u = shadowUVDepthStrength(p, i);
		if (u.w > 0.0) {
			var col = vec3<f32>(0.4);
			if (i == 0u) { col = vec3<f32>(1.0, 0.3, 0.3); }
			else if (i == 1u) { col = vec3<f32>(0.3, 1.0, 0.3); }
			else if (i == 2u) { col = vec3<f32>(0.3, 0.5, 1.0); }
			else { col = vec3<f32>(1.0, 1.0, 0.3); }
			return col * (0.4 + 0.6 * u.w);
		}
	}
	return vec3<f32>(0.25);
}

// The distance fog: exponential (distanceFog per world unit), with a smoothstep from .7 of viewDistanceMax that takes
// it to 1 at viewDistanceMax, where the loaded map ends (see MapWorld.viewDistanceMax). 0 (not set): only the exp.
// d is the horizontal (xz) distance to the camera, not the 3d one: the map is loaded in a radius around the camera in
// xz, and a camera that goes up and looks down gets no more fog (the far plane reaches past it, see
// MapWorld.applyViewDistanceMax).
// The same in deferredShading.wgsl, cars.wgsl, reprojectedSSAOComposite.wgsl and mapOSM/shading.wgsl.
fn getFog(d : f32) -> f32 {
	let end = select(1., 1. - smoothstep(.7 * uniforms.viewDistanceMax, uniforms.viewDistanceMax, d), uniforms.viewDistanceMax > 0.);
	return 1. - exp(-d * uniforms.distanceFog) * end;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
#ifdef FEATURE_IDS
  if (uniforms.drawIndex != 0u) {
    let v = input.globalId + 1u;
    return vec4<f32>(
      f32(v & 0xFFu) / 255.0,
      f32((v >> 8u) & 0xFFu) / 255.0,
      f32((v >> 16u) & 0xFFu) / 255.0,
      f32((v >> 24u) & 0xFFu) / 255.0,
    );
  }
#endif
 var n = normalize(cross(dpdy(input.world), dpdx(input.world)));

  if (DRAW_DEPTH) {
    let d = -input.world.y * 1.;
    return vec4<f32>(d, d * d, 0.0, 1.0);
  }

  if (uniforms.shadowDebug > 0u) {
    return vec4<f32>(cascadeDebugColor(input.world), 1.0);
  }

	let l = uniforms.lightDirection.xyz;
	let nDotL = max(dot(n, l), 0.0);

	let skyColor = uniforms.skyColor * uniforms.skyColor;
	var sunColor = uniforms.sunColor * uniforms.sunColor;

	var baseColor = uniforms.baseColor;
#ifdef TREE_COLORS
	if (input.treeMaterial != 0xFFFFFFFFu) {
		baseColor = materialPalette[input.treeMaterial].rgb;
	}
#endif
	let albedo = baseColor * baseColor;

	let shadow = mix(1.0, getShadowMap(input.world), uniforms.shadowStrength);

//	var c = albedo * mix(nDotL * shadow, 1., uniforms.ambient) * (vec3(1.) - skyColor);
	var c = albedo * sunColor * (nDotL * shadow);
#ifdef SKY_ALBEDO
	// with the albedo, like the deferred shading of mapWorld: without it a light sky washes out the sun, the shadows
	// and the colors of the layers
	c += albedo * (n.y * .5 + .5) * skyColor;
#else
	c += (n.y * .5 + .5) * skyColor;
#endif

	c = sqrt(c);
	//contrast
//	c = mix(c, smoothstep(vec3(0.), vec3(1.), c), uniforms.contrast);

	c = mix(c,uniforms.backgroundColor, getFog(distance(input.world.xz, uniforms.cameraPosition.xz)));

 	var h = hash23ff(input.position.xy);
    c += (h.y + h.z - 1.) / 255.;
    c = clamp(c, vec3(0.), vec3(1.));
#ifdef FEATURE_IDS
  if (uniforms.selectedID >= 0 && input.globalId == u32(uniforms.selectedID)) {
    c = mix(c, vec3<f32>(0.9, 0.1, 0.1), 0.6);
  }
#endif

	return vec4<f32>(c, 1.0);
}
