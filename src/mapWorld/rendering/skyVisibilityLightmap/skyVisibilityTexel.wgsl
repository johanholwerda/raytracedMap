//PACKED_TEXELS (SkyVisibilityLightmap.packedTexels): a texel of the sky visibility lightmap is 1 word that counts rays,
//also with the sun: the sky rays that reached the sky (bits 0-8), the sky rays (9-17), the sun rays that reached the sun
//(18-24) and the sun rays (25-31). The visibility is hits / rays, exact (or of before, see STALE_COUNT), and trace_main adds the batch of a cell with 1
//compare exchange per texel (addPackedBatch in updateSkyVisibilityLightmap.wgsl). Without it a texel is the running average and
//the paths as 2 x f16, and the sun is a second word in the second half of lightingBuffer (getSunTexelOffset).
//Included by the update and by the resolve (SkyVisibilityLightmap.resolveIncludes)
override PACKED_TEXELS : bool = false;

const PACKED_SKY_RAYS_SHIFT = 9u;
const PACKED_SUN_HITS_SHIFT = 18u;
const PACKED_SUN_RAYS_SHIFT = 25u;
//the most a field holds. The rays of a texel stop at their cap, which is at most this (getPackedCaps in
//updateSkyVisibilityLightmap.wgsl), and the hits are never more than the rays
const PACKED_SKY_MAX = 511u;
const PACKED_SUN_MAX = 127u;
//the bits of the sky fields, which the clear of the sun keeps
const PACKED_SKY_MASK = 0x3FFFFu;
//the paths a sun ray counts as: the sun count compares with the same thresholds as the sky's (accumulationMax, the
//prior, convergedPaths). The sun takes accumulationMax / PACKED_SUN_SCALE rays at most
const PACKED_SUN_SCALE = 4u;

/*
A stale texel: a tile came or went near it since its rays (resetStaleTexel in updateSkyVisibilityLightmap.wgsl), so its
visibility is of the buildings of before. It is kept to show until the texel is sampled again, but as STALE_COUNT
paths: next to nothing, so the first rays replace it, and where a texel next to it or coarser has rays, the lookup
shows those. A texel that is never sampled again (under a building that came) so adds nothing to the pixels around it.
Packed, a stale channel has no rays and its visibility in the hits: 1 for 0 up to the most the field holds for 1 (a
channel has no hits without rays otherwise). Kept as rays, the old rays stayed in the count for good: 2 of the 127 sun
rays lit in a shadow that came with a tile, a dark gray. The f16 texels hold STALE_COUNT as their paths
 */
const STALE_COUNT = .0625;

//the fields of a word: sky hits, sky rays, sun hits, sun rays
fn unpackTexelRays(word: u32) -> vec4<u32> {
	return vec4(word & PACKED_SKY_MAX, (word >> PACKED_SKY_RAYS_SHIFT) & PACKED_SKY_MAX, (word >> PACKED_SUN_HITS_SHIFT) & PACKED_SUN_MAX, word >> PACKED_SUN_RAYS_SHIFT);
}

fn packTexelRays(r: vec4<u32>) -> u32 {
	return r.x | (r.y << PACKED_SKY_RAYS_SHIFT) | (r.z << PACKED_SUN_HITS_SHIFT) | (r.w << PACKED_SUN_RAYS_SHIFT);
}

//the visibility and the paths of a channel (its hits and rays, and the most its hits field holds): of its rays, or of
//before when it is stale
fn decodePackedChannel(hits: f32, rays: f32, hitsMax: f32) -> vec2<f32> {
	let stale = vec2(max(hits - 1., 0.) / (hitsMax - 1.), select(0., STALE_COUNT, hits > 0.));
	return select(stale, vec2(hits / rays, rays), rays > 0.);
}

//the visibility and the paths of the sky (xy) and of the sun (zw, its rays times PACKED_SUN_SCALE), as the f16 texels
//hold them
fn decodePackedTexel(word: u32) -> vec4<f32> {
	let r = vec4<f32>(unpackTexelRays(word));
	let sun = decodePackedChannel(r.z, r.w, f32(PACKED_SUN_MAX));
	return vec4(decodePackedChannel(r.x, r.y, f32(PACKED_SKY_MAX)), sun.x, sun.y * f32(PACKED_SUN_SCALE));
}
