//the sky visibility of the sky visibility lightmap at a point, for the deferred shading (deferredShading.wgsl).
//Texels of 1 word, the visibility and the paths it holds as 2 x f16, see
//updateSkyVisibilityLightmap.wgsl. The functions return the visibility in rgb, and the shading
//multiplies it by the sky color. With the sun in the lightmap (SkyVisibilityLightmap.setSun) the sun half of the texels
//holds the sun visibility: getLightmapChannels looks up both with the same tile probes. Needs cityLightmap.wgsl, the
//bindings tileBuffer (array<u32>) and lightingBuffer (array<u32>), and uniforms.searchCount, accumulationMax,
//ancestorWeight and levelBlend

//the slot of a tile, TILE_NONE when it doesn't exist
fn findTile(surface: u32, level: u32, tile: vec2<u32>) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = tileBuffer[index];
		if(existing == key){
			return index;
		}
		if(existing == 0u){
			break;
		}
		index = (index + 1u) % TILE_COUNT;
	}
	return TILE_NONE;
}

//above this part of accumulationMax the texels of gather count the same, see resolveVirtualLightmap.wgsl
const GATHER_COUNT_MAX = .25;

//level + 2 is the prior of level + 1 while that has few paths (see getLightmapChannels). false: only where level + 1
//has no paths at all, so it doesn't show as a hole. The half resolution pass (SkyVisibilityHalfRes), where the gather
//of level + 2 cost 1.3x while flying and changed next to nothing once the texels converge
override LEVEL2_PRIOR: bool = true;
//gather reads the texel the point is in instead of the 4 around it, bilinearly: 1 hash table lookup and 1 load instead
//of up to 4 and 4. Blocky, a texel is a few pixels. See SkyVisibilityLightmap.halfResNearest
override NEAREST_TEXEL: bool = false;

//the visibility and the paths of a texel: of the sky (xy), and of the sun (zw, see getSunTexelOffset) when sun, else 0
fn loadTexel(tileIndex: u32, tileTexel: vec2<u32>, sun: bool) -> vec4<f32> {
	if(tileIndex == TILE_NONE){
		return vec4(0.);
	}
	let index = (tileIndex * TILE_SIZE + tileTexel.y) * TILE_SIZE + tileTexel.x;
	if(PACKED_TEXELS){
		//both channels in 1 word, see skyVisibilityTexel.wgsl
		let t = decodePackedTexel(lightingBuffer[index]);
		return select(vec4(t.xy, 0., 0.), t, sun);
	}
	var sunTexel = vec2(0.);
	if(sun){
		sunTexel = unpack2x16float(lightingBuffer[index + getSunTexelOffset()]);
	}
	return vec4(unpack2x16float(lightingBuffer[index]), sunTexel);
}

//1 channel of gather: the 4 texels (visibility, paths) bilinearly and weighted by their path counts
fn gatherChannel(bilinear: vec4<f32>, l00: vec2<f32>, l10: vec2<f32>, l01: vec2<f32>, l11: vec2<f32>) -> vec2<f32> {
	let counts = vec4(l00.y, l10.y, l01.y, l11.y);
	let weights = bilinear * min(counts, vec4(uniforms.accumulationMax * GATHER_COUNT_MAX));
	let weight = weights.x + weights.y + weights.z + weights.w;
	if(weight == 0.){
		return vec2(0.);
	}
	let count = dot(bilinear, counts);
	let visibility = dot(vec4(l00.x, l10.x, l01.x, l11.x), weights) / weight;
	return vec2(visibility * count, count);
}

//the 4 texels of a level nearest to the surface point, bilinearly and weighted by their path counts. The visibility
//times the path count, and the path count: of the sky (xy), and with sun of the sun texels (zw, else 0). The hash table
//lookups (up to 4) are the same for both. NEAREST_TEXEL: the texel of the point only
fn gather(surface: Surface, level: u32, sun: bool) -> vec4<f32> {
	if(NEAREST_TEXEL){
		let texel = getTexel(surface, level);
		let l = loadTexel(findTile(surface.id, level, texel / TILE_SIZE), texel % TILE_SIZE, sun);
		return vec4(l.x * l.y, l.y, l.z * l.w, l.w);
	}
	let q = surface.position / getTexelSize(level) - .5;
	let f = fract(q);
	let texel0 = vec2<u32>(vec2<i32>(floor(q)) + vec2(TEXEL_OFFSET));
	let texel1 = texel0 + 1u;

	let t0 = texel0 / TILE_SIZE;
	let t1 = texel1 / TILE_SIZE;
	let i00 = findTile(surface.id, level, t0);
	var i10 = i00;
	var i01 = i00;
	var i11 = i00;
	if(t1.x != t0.x){
		i10 = findTile(surface.id, level, vec2(t1.x, t0.y));
		i11 = i10;
	}
	if(t1.y != t0.y){
		i01 = findTile(surface.id, level, vec2(t0.x, t1.y));
		i11 = i01;
		if(t1.x != t0.x){
			i11 = findTile(surface.id, level, t1);
		}
	}

	let p0 = texel0 % TILE_SIZE;
	let p1 = texel1 % TILE_SIZE;
	let l00 = loadTexel(i00, p0, sun);
	let l10 = loadTexel(i10, vec2(p1.x, p0.y), sun);
	let l01 = loadTexel(i01, vec2(p0.x, p1.y), sun);
	let l11 = loadTexel(i11, p1, sun);

	let bilinear = vec4((1. - f.x) * (1. - f.y), f.x * (1. - f.y), (1. - f.x) * f.y, f.x * f.y);
	let sky = gatherChannel(bilinear, l00.xy, l10.xy, l01.xy, l11.xy);
	var sunGather = vec2(0.);
	if(sun){
		sunGather = gatherChannel(bilinear, l00.zw, l10.zw, l01.zw, l11.zw);
	}
	return vec4(sky, sunGather);
}

const PRIOR_FADE = 8.;
//below convergedPaths (SkyVisibilityLightmap, 256 of the 508 of packedTexels), so a texel that counts as converged needs
//no gather of its coarser level. At .75 most converged texels still gathered it, for a weight of a few percent: the half
//resolution pass was 1.2x slower for a difference of 1e-5
const PRIOR_FADE_ACCUMULATION = .5;

//the paths from which a level is shown as it is, without the coarser level
fn getPriorEnd() -> f32 {
	return min(uniforms.ancestorWeight * PRIOR_FADE, uniforms.accumulationMax * PRIOR_FADE_ACCUMULATION);
}

//a level, leaning on the coarser level while it has few paths. y 0 when neither has visibility
fn combine(fine: vec2<f32>, coarse: vec2<f32>) -> vec2<f32> {
	let priorEnd = getPriorEnd();
	if(fine.y >= priorEnd || coarse.y == 0.){
		return fine;
	}
	let priorWeight = min(coarse.y, uniforms.ancestorWeight) * (1. - fine.y / priorEnd);
	return vec2(fine.x + coarse.x / coarse.y * priorWeight, fine.y + priorWeight);
}

//combine of both channels of gather. A channel with priorEnd paths or more stays as it is, so a gather that only the
//other channel needed changes nothing
fn combine2(fine: vec4<f32>, coarse: vec4<f32>) -> vec4<f32> {
	return vec4(combine(fine.xy, coarse.xy), combine(fine.zw, coarse.zw));
}

//a channel of l (gather) has fewer paths than priorEnd: the sky, or with sun the sun
fn needsPrior(l: vec4<f32>, sun: bool, priorEnd: f32) -> bool {
	return l.y < priorEnd || (sun && l.w < priorEnd);
}

//a channel of c (gather) has no paths, see needsPrior
fn hasEmpty(c: vec4<f32>, sun: bool) -> bool {
	return c.y == 0. || (sun && c.w == 0.);
}

//per component of c (gather): its channel has no paths
fn emptyMask(c: vec4<f32>) -> vec4<bool> {
	return vec4(c.y == 0., c.y == 0., c.w == 0., c.w == 0.);
}

//the weight of the coarser of the 2 levels around levelF: it goes from 0 to 1 in the middle uniforms.levelBlend part of
//a level only. A pixel outside it reads 1 level (and its prior while that has few paths) instead of 2. levelBlend 1:
//fract(levelF), blended over the whole level
fn getLevelBlend(levelF: f32) -> f32 {
	return saturate((fract(levelF) - .5) / max(uniforms.levelBlend, 1e-3) + .5);
}

//the sky visibility between the 2 levels around levelF (rgb) and the paths it holds (a, 0 when there is none). Gathers
//a level only when the blend or a prior needs it: each gather is up to 4 hash table lookups and 4 texel loads
fn getLightmapLighting(surface: Surface, levelF: f32) -> vec4<f32> {
	let c = getLightmapChannels(surface, levelF, false);
	return vec4(vec3(c.x), c.y);
}

//1 channel of getLightmapChannels: the 2 levels (each combined with its prior) blended by t, the visibility and the paths
fn blendLevels(c0: vec2<f32>, c1: vec2<f32>, t: f32) -> vec2<f32> {
	let visibility0 = select(0., c0.x / c0.y, c0.y > 0.);
	let visibility1 = select(0., c1.x / c1.y, c1.y > 0.);
	let w0 = select(0., 1. - t, c0.y > 0.);
	let w1 = select(0., t, c1.y > 0.);
	let count = max(c0.y, c1.y);
	if(w0 + w1 == 0.){
		return vec2(select(visibility1, visibility0, c0.y > 0.), count);
	}
	return vec2((visibility0 * w0 + visibility1 * w1) / (w0 + w1), count);
}

//the sky visibility (x) and the paths it holds (y, 0 when there is none), and with sun the same of the sun visibility
//(zw, else 0; see SkyVisibilityLightmap.setSun), in 1 lookup: the gathers of both share their tile probes. A level is
//gathered when the blend or the prior of either channel needs it. The visibilities are those of separate lookups, a path
//count can be higher (it is only compared with 0)
fn getLightmapChannels(surface: Surface, levelF: f32, sun: bool) -> vec4<f32> {
	let level = u32(levelF);
	let t = getLevelBlend(levelF);
	let priorEnd = getPriorEnd();
	var l0 = vec4(0.);
	if(t < 1.){
		l0 = gather(surface, level, sun);
	}
	//the coarser level of the blend, or the prior of the finer one
	var l1 = vec4(0.);
	if(t > 0. || needsPrior(l0, sun, priorEnd)){
		l1 = gather(surface, level + 1u, sun);
	}
	var c0 = combine2(l0, l1);
	//also when the finer level has no visibility, so it doesn't show as a hole
	var c1 = vec4(0.);
	if(t > 0. || hasEmpty(c0, sun)){
		c1 = l1;
		//its prior, without LEVEL2_PRIOR only where it has no visibility
		if(select(hasEmpty(l1, sun), needsPrior(l1, sun, priorEnd), LEVEL2_PRIOR)){
			c1 = combine2(l1, gather(surface, level + 2u, sun));
		}
	}
	//the coarser level has no visibility: the finer one, when the blend skipped it (of those channels only)
	if(t == 1. && hasEmpty(c1, sun)){
		c0 = select(c0, combine2(gather(surface, level, sun), l1), emptyMask(c1));
	}
	return vec4(blendLevels(c0.xy, c1.xy, t), blendLevels(c0.zw, c1.zw, t));
}

//1 of the 2 levels around levelF instead of both, see getLightmapLightingStochastic in cityLightmapResolve.wgsl.
//noise: 0 to 1, different per pixel and per frame
fn getLightmapLightingStochastic(surface: Surface, levelF: f32, noise: f32) -> vec4<f32> {
	let c = getLightmapChannelsStochastic(surface, levelF, noise, false);
	return vec4(vec3(c.x), c.y);
}

//getLightmapChannels with 1 level picked by noise, the same for both channels
fn getLightmapChannelsStochastic(surface: Surface, levelF: f32, noise: f32, sun: bool) -> vec4<f32> {
	let levelBase = u32(levelF);
	let coarser = noise < getLevelBlend(levelF);
	let level = levelBase + select(0u, 1u, coarser);
	var c = gather(surface, level, sun);
	//the level above it is only its prior
	if(needsPrior(c, sun, getPriorEnd())){
		c = combine2(c, gather(surface, level + 1u, sun));
	}
	//a level without visibility doesn't show as a hole when the other one has it (of those channels only)
	if(hasEmpty(c, sun)){
		//the finer level, or the one above the prior: where the lightmap had no room for the tiles of 2 levels, the
		//cells sample the third (claim_main in updateSkyVisibilityLightmap.wgsl)
		c = select(c, gather(surface, select(levelBase + 2u, levelBase, coarser), sun), emptyMask(c));
	}
	return vec4(select(0., c.x / c.y, c.y > 0.), c.y, select(0., c.z / c.w, c.w > 0.), c.w);
}

//the path count of the texel of the point at its level, for the converging view
fn getLightmapTexelCount(surface: Surface, level: u32) -> f32 {
	let texel = getTexel(surface, level);
	let tileIndex = findTile(surface.id, level, texel / TILE_SIZE);
	return loadTexel(tileIndex, texel % TILE_SIZE, false).y;
}

//the half resolution resolve (see SkyVisibilityHalfRes.ts): a sample per 2 x 2 pixels holds the lighting and a key of the
//surface point it was looked up at. A full resolution pixel uses the samples around it whose key matches its own surface
//point: the same surface (a hash of the id) and within HALF_RES_MATCH_TEXELS texels. The position matters: the walls of
//the buildings of a map tile share a surface id, and so does all of the ground

//along each axis, at the level of the sample. Neighbouring pixels on a surface are less than a texel apart (texelsPerPixel
//is below 1), more at a grazing angle. The ground in front of a building and behind it are many texels apart
const HALF_RES_MATCH_TEXELS = 2u;
//the position in the key: in 1 / 4 texel steps, modulo 64 texels (8 bits)
const HALF_RES_STEPS_PER_TEXEL = 4.;
const HALF_RES_WRAP_TEXELS = 64.;

//the position of a surface point in the key, at a level
fn getHalfResPosition(position: vec2<f32>, level: u32) -> vec2<u32> {
	let q = fract(position / (getTexelSize(level) * HALF_RES_WRAP_TEXELS)) * (HALF_RES_WRAP_TEXELS * HALF_RES_STEPS_PER_TEXEL);
	return vec2<u32>(q) & vec2(255u);
}

//the hash of a surface id in the key, 1 to 2047: 0 is no surface
fn getHalfResSurfaceHash(id: u32) -> u32 {
	return tileHash(id) % 2047u + 1u;
}

//the key of the surface point a sample was looked up at: the hash of the surface id (11 bits), its level (5 bits) and
//its position at that level (2 x 8 bits). 0: no surface (the sky)
fn packHalfResKey(surface: Surface, level: u32) -> u32 {
	let p = getHalfResPosition(surface.position, level);
	return getHalfResSurfaceHash(surface.id) | (min(level, 31u) << 11u) | (p.x << 16u) | (p.y << 24u);
}

//per key of 4 samples: whether it is of the surface point of a pixel, see packHalfResKey. The position compares at the
//level of the sample, wrapped (getHalfResPosition, the same arithmetic per component). All 4 at once, without branches:
//the loop of 1 test per sample (with an early out on the hash) made the deferred shading 1.27x slower
fn matchesHalfResKeys(keys: vec4<u32>, surface: Surface) -> vec4<bool> {
	let levels = (keys >> vec4(11u)) & vec4(31u);
	let wrap = uniforms.texelSizeMin * vec4<f32>(vec4(1u) << levels) * HALF_RES_WRAP_TEXELS;
	let steps = HALF_RES_WRAP_TEXELS * HALF_RES_STEPS_PER_TEXEL;
	let px = vec4<i32>(vec4<u32>(fract(surface.position.x / wrap) * steps) & vec4(255u));
	let py = vec4<i32>(vec4<u32>(fract(surface.position.y / wrap) * steps) & vec4(255u));
	let dx = ((px - vec4<i32>((keys >> vec4(16u)) & vec4(255u)) + 128) & vec4(255)) - 128;
	let dy = ((py - vec4<i32>(keys >> vec4(24u)) + 128) & vec4(255)) - 128;
	let reach = vec4(i32(HALF_RES_MATCH_TEXELS * u32(HALF_RES_STEPS_PER_TEXEL)));
	return ((keys & vec4(0x7FFu)) == vec4(getHalfResSurfaceHash(surface.id))) & (abs(dx) <= reach) & (abs(dy) <= reach);
}
