//the gBuffer of MapWorld (see MapGBuffer.ts): 2 targets of 12 bytes per pixel together, and the hardware depth (4 bytes)
//for the camera distance. A reader loads the targets into a vec3<u32> g (see MapGBuffer.readIncludes):
// x (target 0 .x): the lightmap surface id (see cityLightmapSurface.wgsl), SURFACE_NONE (0) on the sky
// y (target 0 .y): the normal facing the camera, octahedral 2 x 10 bits | the low 4 bits of the material << 20 (5 bits,
//    its albedo is in the palette of MapGBuffer) | selected << 24 | the high bits of the wall atlas position: u << 25,
//    v << 28 (3 bits each) | the high bit of the material << 31
// z (target 1): the low 16 bits of the wall atlas position: u | v << 16
//The surface position is only stored on a wall: on the ground and a roof it is world xz, which the distance gives (see
//getGBufferSurface). The wall atlas position is in steps of atlasStep (see MapGBuffer.atlasStep), 19 bits each.
//The depth is reversed and depth32float (see MapGBuffer.ts), cleared to 0, the sky: getGBufferDistance makes that a
//distance of 0

const GBUFFER_ATLAS_MAX = 0x7FFFFu;

fn octWrap(v: vec2<f32>) -> vec2<f32> {
	return (1. - abs(v.yx)) * select(vec2(-1.), vec2(1.), v >= vec2(0.));
}

//n: normalized
fn packGBufferNormal(n: vec3<f32>) -> u32 {
	var e = n.xz / (abs(n.x) + abs(n.y) + abs(n.z));
	if(n.y < 0.){
		e = octWrap(e);
	}
	let q = vec2<u32>(round(clamp(e * .5 + .5, vec2(0.), vec2(1.)) * 1023.));
	return q.x | (q.y << 10u);
}

fn unpackGBufferNormal(bits: u32) -> vec3<f32> {
	let e = vec2<f32>(vec2(bits, bits >> 10u) & vec2(1023u)) / 1023. * 2. - 1.;
	var n = vec3(e.x, 1. - abs(e.x) - abs(e.y), e.y);
	if(n.y < 0.){
		let w = octWrap(n.xz);
		n = vec3(w.x, n.y, w.y);
	}
	return normalize(n);
}

//the 3 words: target 0 is xy, target 1 is z. atlas: the wall atlas position in atlasSteps, 0 off a wall
fn packGBuffer(surfaceId: u32, n: vec3<f32>, material: u32, selected: bool, atlas: vec2<u32>) -> vec3<u32> {
	let a = min(atlas, vec2(GBUFFER_ATLAS_MAX));
	let y = packGBufferNormal(n) | ((material & 15u) << 20u) | (select(0u, 1u, selected) << 24u) |
		((a.x >> 16u) << 25u) | ((a.y >> 16u) << 28u) | (((material >> 4u) & 1u) << 31u);
	return vec3(surfaceId, y, (a.x & 0xFFFFu) | ((a.y & 0xFFFFu) << 16u));
}

//the camera distance of a pixel from its reversed depth, 0 on the sky. uv: 0 to 1, y down like the texture coordinates.
//inverseProjection: of the (jittered) camera the gBuffer was drawn with, not reversed. depthParams: see
//MapGBuffer.depthParams, 1 / the view depth is depth * x + y: no cancellation, unlike the inverse projection of the depth
fn getGBufferDistance(depth: f32, uv: vec2<f32>, inverseProjection: mat4x4<f32>, depthParams: vec2<f32>) -> f32 {
	if(depth <= 0.){
		return 0.;
	}
	let viewDepth = 1. / (depth * depthParams.x + depthParams.y);
	//the eye ray of the pixel, at view depth 1
	let ray = inverseProjection * vec4(uv.x * 2. - 1., 1. - uv.y * 2., 1., 1.);
	return viewDepth * length(ray.xyz) / abs(ray.z);
}

fn getGBufferNormal(g: vec3<u32>) -> vec3<f32> {
	return unpackGBufferNormal(g.y);
}

fn getGBufferMaterial(g: vec3<u32>) -> u32 {
	return ((g.y >> 20u) & 15u) | ((g.y >> 27u) & 16u);
}

fn isGBufferSelected(g: vec3<u32>) -> bool {
	return ((g.y >> 24u) & 1u) != 0u;
}

//the wall atlas position in atlasSteps
fn getGBufferAtlas(g: vec3<u32>) -> vec2<u32> {
	return vec2(g.z & 0xFFFFu, g.z >> 16u) | (vec2((g.y >> 25u) & 7u, (g.y >> 28u) & 7u) << vec2(16u));
}
