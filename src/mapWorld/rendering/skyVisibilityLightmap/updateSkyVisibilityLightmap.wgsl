//the update of the sky visibility lightmap: claim_main, select_main and trace_main, see SkyVisibilityLightmap.ts.
//updateCityLightmap.wgsl with albedo 0: a ray that hits a surface adds nothing, so a texel holds the part of the sky its
//point sees, cosine weighted (1 on an open roof, .5 on an open wall). The shading multiplies it by the sky color. So
//there is no light that changes, no cached lighting at a ray hit, no sun ray, and 1 any hit ray per path at most.
//LIGHTMAP_SUN: the texels hold the sun visibility too, in the second half of lightingBuffer (getSunTexelOffset): of the
//sampleCount rays of a cell that traces, sunSampleCount go to points on the disc of the sun instead of the sky (see
//trace_main). It depends on the sun, so the host clears that half when the sun moves (SkyVisibilityLightmap.setSun). The
//claims go by the fewer paths of the 2
//PACKED_TEXELS: a texel is 1 word of ray counts instead, with the sun in it too, see skyVisibilityTexel.wgsl and
//addPackedBatch
//Charts of cityLightmapSurface.wgsl, rays of skyVisibilityTrace.wgsl

override WORK_GROUP_SIZE_X: u32;
override WORK_GROUP_SIZE_Y: u32;
override CELL_SIZE: u32;
//the threads of a workgroup of trace_main
override TRACE_GROUP_SIZE: u32;
//the texels hold the sun visibility too, see the top
override LIGHTMAP_SUN: bool = false;

struct Uniforms {
	resolution : vec2<f32>,
	inverseProjection : mat4x4<f32>,
	inverseView : mat4x4<f32>,
	viewProjection : mat4x4<f32>,
	cameraPosition : vec3<f32>,
	frame : u32,
	//the world size of a reference pixel at distance 1 (1920 x 1080 pixel count, see SkyVisibilityLightmap.getPixelScale)
	pixelSize : f32,
	//screen pixels per reference pixel along a side
	pixelScale : f32,
	//lightmap texels per screen pixel, along a side
	texelsPerPixel : f32,
	//the size of the level 0 texels, in world units
	texelSizeMin : f32,
	//how much the texels grow at a grazing view angle, see getLevel
	levelAngle : f32,
	//the tilt shift blur: its sigma in pixels at |ndc y| = 1 and at most, 0 without blur. See getBlurPixels
	blurSlope : f32,
	blurMax : f32,
	//texels per blur sigma, see getLevel
	blurTexelScale : f32,
	//the tiles of the top level BVH and its root, see CityScene.update
	cityTileCount : u32,
	cityRoot : u32,
	sampleCount : u32,
	//the paths the running average of a texel holds, see addTexelSample
	accumulationMax : f32,
	//coarser levels that accumulate the samples too, at most ANCESTOR_LEVELS_MAX
	ancestorLevels : u32,
	//the paths of the coarser visibility a new texel starts with
	seedCount : f32,
	//the frames a tile has to be unused before another one can take its place, when there is no empty slot
	evictFrames : u32,
	searchCount : u32,
	//the threads of trace_main per cell, a divisor of TRACE_GROUP_SIZE
	traceLanes : u32,
	//the most cells that trace in a frame, see select_main
	traceCellsMax : u32,
	//the boxes where tiles came or went (in cityBuffer), and the latest frame of them, 0 when none. See getChangeFrames
	changeCount : u32,
	changeEnd : u32,
	//the paths from which a texel no longer counts in traceArgs[6], see select_main
	convergedPaths : f32,
	//the step of the wall atlas positions in the gBuffer, see MapGBuffer.atlasStep
	atlasStep : f32,
	//see getGBufferDistance
	depthParams : vec2<f32>,
	//LIGHTMAP_SUN: toward the sun, the angular radius of its disc (radians), and how many of the sampleCount rays of a cell
	//go to the sun while the sky takes samples too, see trace_main
	sunDirection : vec3<f32>,
	sunRadius : f32,
	sunSampleCount : u32,
	//and for a texel in a penumbra (its sun visibility between 0 and 1 so far), see trace_main
	sunSampleCountPenumbra : u32,
}

const ANCESTOR_LEVELS_MAX = 4u;
//the offset of the origin of a sun ray from its surface along the normal, times the camera distance, as the ray traced
//shadows of deferredShading.wgsl
const SUN_RAY_BIAS = 1e-4;
const TAU : f32 = 6.28318530718;
//a path that goes down is mirrored up, see getPathRay. false: it is 0 without a ray. With PACKED_TEXELS, which counts
//whole hits, only on a wall: a folded path weighs .5 there, so its ray counts as 2 paths (see trace_main)
override FOLD_DOWN_PATHS: bool = true;
//PACKED_TEXELS: the most rays of a cell to the sun while only the sun takes samples (after it moved). They are all from
//1 pixel, and what varies in a texel is where the pixel is: not the up to tailSampleCountMax rays a cell has in the
//tail of the convergence
const PACKED_SUN_RAYS_MAX = 16u;

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//per tile slot: the checksum of the tile, 0 when empty
@group(0) @binding(1) var<storage, read_write> tileBuffer : array<atomic<u32>>;
//the frame a tile was last used in
@group(0) @binding(2) var<storage, read_write> tileTimeBuffer : array<atomic<u32>>;
//per tile slot TILE_SIZE x TILE_SIZE texels: the visibility and the paths it holds, as 2 x f16 in 1 word. See
//addTexelSample
@group(0) @binding(3) var<storage, read_write> lightingBuffer : array<atomic<u32>>;
//the slots in use, for the log
@group(0) @binding(4) var<storage, read_write> tileCount : atomic<u32>;
//the gBuffer of MapWorld (see MapGBuffer.ts): its 2 targets and its depth, see mapGBufferPacking.wgsl
@group(0) @binding(6) var gBufferTexture: texture_2d<u32>;
@group(0) @binding(5) var gBufferAtlasTexture: texture_2d<u32>;
@group(0) @binding(9) var gBufferDepth: texture_depth_2d;
//a tiling 2 channel blue noise texture, see getBlueNoise
@group(0) @binding(8) var blueNoiseTexture: texture_2d<f32>;
//the buildings the rays hit, see CityScene and skyVisibilityTrace.wgsl
@group(0) @binding(7) var<storage, read> cityBuffer : array<vec4<f32>>;
//per texel: the highest bid of the cells that sample it this frame, see claim_main
@group(0) @binding(11) var<storage, read_write> claimBuffer : array<atomic<u32>>;
//first the cells that won the bid of their texel (x | y << 16), then per cell the texel it bids on, see getCellTexelSlot
@group(0) @binding(12) var<storage, read_write> traceCells : array<u32>;
//the workgroups of trace_main (x, y, z), the count of traceCells, in 4 and 5 the cells that won a bid in the even
//and the odd frames, and in 6 the ones of them whose texel has less than convergedPaths paths. 7 is unused
@group(0) @binding(13) var<storage, read_write> traceArgs : array<atomic<u32>, 8>;

/*
The compare exchanges of the word at index of tileBuffer, lightingBuffer and claimBuffer (no pointer parameters: core
WGSL only takes them to function and private memory). Without HAS_ATOMIC_CAS (WebKit of iOS 27.2, see Renderer) an
atomicMax toward a larger value, an atomicMin toward a smaller one. Exact when the word is compare (it becomes value) or
outside compare..value (it stays). Only a word that another thread changed to between the 2 (since the caller loaded
compare) is overwritten: the value goes in over that update, and it counts as exchanged. So at worst a sample is lost
or 2 tiles race for 1 slot in a frame, never a hang: a spinlock on atomicExchange lost the device on an iPhone
 */
struct CasResult {
	old_value : u32,
	exchanged : bool,
}

fn emulatedCasResult(old: u32, compare: u32, value: u32) -> CasResult {
	return CasResult(old, old == compare || (old > min(compare, value) && old < max(compare, value)));
}

fn casTile(index: u32, compare: u32, value: u32) -> CasResult {
#ifdef HAS_ATOMIC_CAS
	let r = atomicCompareExchangeWeak(&tileBuffer[index], compare, value);
	return CasResult(r.old_value, r.exchanged);
#else
	if(value == compare){
		let old = atomicLoad(&tileBuffer[index]);
		return CasResult(old, old == compare);
	}
	var old : u32;
	if(value > compare){
		old = atomicMax(&tileBuffer[index], value);
	} else {
		old = atomicMin(&tileBuffer[index], value);
	}
	return emulatedCasResult(old, compare, value);
#endif
}

fn casLighting(index: u32, compare: u32, value: u32) -> CasResult {
#ifdef HAS_ATOMIC_CAS
	let r = atomicCompareExchangeWeak(&lightingBuffer[index], compare, value);
	return CasResult(r.old_value, r.exchanged);
#else
	if(value == compare){
		let old = atomicLoad(&lightingBuffer[index]);
		return CasResult(old, old == compare);
	}
	var old : u32;
	if(value > compare){
		old = atomicMax(&lightingBuffer[index], value);
	} else {
		old = atomicMin(&lightingBuffer[index], value);
	}
	return emulatedCasResult(old, compare, value);
#endif
}

fn casClaim(index: u32, compare: u32, value: u32) -> CasResult {
#ifdef HAS_ATOMIC_CAS
	let r = atomicCompareExchangeWeak(&claimBuffer[index], compare, value);
	return CasResult(r.old_value, r.exchanged);
#else
	if(value == compare){
		let old = atomicLoad(&claimBuffer[index]);
		return CasResult(old, old == compare);
	}
	var old : u32;
	if(value > compare){
		old = atomicMax(&claimBuffer[index], value);
	} else {
		old = atomicMin(&claimBuffer[index], value);
	}
	return emulatedCasResult(old, compare, value);
#endif
}

//whether a ray from p to a point on the disc of the sun can pass through a box: the buildings in it can be in front of
//the sun there. The box as wide as the disc at the farthest such a ray can enter it, under its top
fn sunPassesBox(p: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> bool {
	let l = uniforms.sunDirection;
	if(l.y <= 0.){
		return false;
	}
	let margin = uniforms.sunRadius * max(boxMax.y - p.y, 0.) / l.y;
	let inverse = 1. / select(l, vec3(1e-6), abs(l) < vec3(1e-6));
	let t0 = (boxMin - margin - p) * inverse;
	let t1 = (boxMax + margin - p) * inverse;
	let near = min(t0, t1);
	let far = max(t0, t1);
	return max(max(near.x, near.y), max(near.z, 0.)) <= min(far.x, min(far.y, far.z));
}

/*
the latest frame a tile came or went in that changed the sky visibility at p (x) and its sun visibility (y, with
LIGHTMAP_SUN), 0 when none did: the visibility sampled before that frame has the buildings of before. See CityScene: a
change is the box of the buildings of the tile and its reach. The sky changes within the reach around the box. The sun
changes where a ray to it passes the box, however far: at a sunset the shadows are 10x as long as the buildings are
high. With the reach for both, the ground a tile away kept the sun it got before the tile with the tower came.
after: only the changes later than this frame, the others are not tested (the texels at p were sampled since)
 */
fn getChangeFrames(p: vec3<f32>, after: u32) -> vec2<u32> {
	var frames = vec2(0u);
	for(var i = 0u; i < uniforms.changeCount; i++){
		let a = cityBuffer[CITY_CHANGE_START + i * 2u];
		let frame = bitcast<u32>(a.w);
		if(frame <= after){
			continue;
		}
		let b = cityBuffer[CITY_CHANGE_START + i * 2u + 1u];
		if(all(p >= a.xyz - vec3(b.w, 0., b.w)) && all(p <= b.xyz + b.w)){
			frames.x = max(frames.x, frame);
		}
		if(LIGHTMAP_SUN && sunPassesBox(p, a.xyz, b.xyz)){
			frames.y = max(frames.y, frame);
		}
	}
	return frames;
}

//the tile is used this frame. No light that changes, so no ageing of a tile that comes back in view, see
//updateCityLightmap.wgsl
fn markUsed(index: u32, frame: u32) {
	atomicMax(&tileTimeBuffer[index], frame);
}

//the slots from its first one a new tile can take the place of an unused tile in, while there still is an empty slot
//further on. See findOrInsert in updateVirtualLightmap.wgsl
const EVICT_NEAR = 4u;

//the slot of a tile, inserted when it is not there: see findOrInsert in updateVirtualLightmap.wgsl
fn findOrInsert(surface: u32, level: u32, tile: vec2<u32>, stamp: u32) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	let frame = uniforms.frame;
	var index = keySlot.y;
	var i = 0u;
	for(; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&tileBuffer[index]);
		if(existing == key){
			markUsed(index, stamp);
			return index;
		}
		if(existing == 0u){
			break;
		}
		index = (index + 1u) % TILE_COUNT;
	}

	let near = min(EVICT_NEAR, uniforms.searchCount);
	var evictIndex = TILE_NONE;
	var evictKey = 0u;
	if(i >= near){
		var evictAge = uniforms.evictFrames;
		let scanCount = select(near, uniforms.searchCount, i >= uniforms.searchCount);
		var scanIndex = keySlot.y;
		for(var s = 0u; s < scanCount; s++){
			if(evictIndex != TILE_NONE && s == near){
				break;
			}
			let existing = atomicLoad(&tileBuffer[scanIndex]);
			if(existing == key){
				markUsed(scanIndex, stamp);
				return scanIndex;
			}
			if(existing != 0u){
				let age = frame - min(atomicLoad(&tileTimeBuffer[scanIndex]), frame);
				if(age > evictAge){
					evictAge = age;
					evictIndex = scanIndex;
					evictKey = existing;
				}
			}
			scanIndex = (scanIndex + 1u) % TILE_COUNT;
		}
	}

	if(evictIndex != TILE_NONE){
		for(var j = 0u; j < 4u; j++){
			let cmp = casTile(evictIndex, evictKey, key);
			if(cmp.exchanged){
				markUsed(evictIndex, stamp);
				let start = evictIndex * TILE_SIZE * TILE_SIZE;
				for(var t = 0u; t < TILE_SIZE * TILE_SIZE; t++){
					atomicStore(&lightingBuffer[start + t], 0u);
					if(LIGHTMAP_SUN && !PACKED_TEXELS){
						atomicStore(&lightingBuffer[start + t + getSunTexelOffset()], 0u);
					}
				}
				return evictIndex;
			}
			if(cmp.old_value == key){
				markUsed(evictIndex, stamp);
				return evictIndex;
			}
			if(cmp.old_value != evictKey){
				break;
			}
		}
	}

	for(; i < uniforms.searchCount; i++){
		let cmp = casTile(index, 0u, key);
		if(cmp.exchanged){
			atomicAdd(&tileCount, 1u);
		}
		if(cmp.exchanged || cmp.old_value == key){
			markUsed(index, stamp);
			return index;
		}
		if(cmp.old_value != 0u){
			index = (index + 1u) % TILE_COUNT;
		}
	}
	return TILE_NONE;
}

fn find(surface: u32, level: u32, tile: vec2<u32>) -> u32 {
	let keySlot = getTileKey(surface, level, tile);
	let key = keySlot.x;
	var index = keySlot.y;
	for(var i = 0u; i < uniforms.searchCount; i++){
		let existing = atomicLoad(&tileBuffer[index]);
		if(existing == key){
			return index;
		}
		if(existing == 0u){
			return TILE_NONE;
		}
		index = (index + 1u) % TILE_COUNT;
	}
	return TILE_NONE;
}

//the pixel a cell samples this frame and the surface point it shows. valid is false for the sky and outside the screen
struct CellPoint {
	valid : bool,
	coord : vec2<u32>,
	worldPosition : vec3<f32>,
	n : vec3<f32>,
	surface : Surface,
}

//the visibility (x) and the paths it holds (y) of a texel
fn loadTexel(index: u32) -> vec2<f32> {
	if(PACKED_TEXELS){
		return decodePackedTexel(atomicLoad(&lightingBuffer[index])).xy;
	}
	return unpack2x16float(atomicLoad(&lightingBuffer[index]));
}

//the sun visibility (x) and the paths it holds (y) of a texel, LIGHTMAP_SUN
fn loadSunTexel(index: u32) -> vec2<f32> {
	if(PACKED_TEXELS){
		return decodePackedTexel(atomicLoad(&lightingBuffer[index])).zw;
	}
	return unpack2x16float(atomicLoad(&lightingBuffer[index + getSunTexelOffset()]));
}

//the paths of a texel the claims go by: with the sun the fewer of the sky's and the sun's
fn getTexelCount(index: u32) -> f32 {
	if(PACKED_TEXELS){
		//1 load for both
		let t = decodePackedTexel(atomicLoad(&lightingBuffer[index]));
		return select(t.y, min(t.y, t.w), LIGHTMAP_SUN);
	}
	let count = loadTexel(index).y;
	if(LIGHTMAP_SUN){
		return min(count, loadSunTexel(index).y);
	}
	return count;
}

//the tries of addTexelSample, against the other cells that add to the texel (the coarser texels are shared)
const TEXEL_CAS_TRIES = 64u;

//adds a sample to a texel and returns the texel. index: of the sky texel, or + getSunTexelOffset() of the sun one. The texel is 1 word, so a compare exchange replaces the lock of
//updateCityLightmap.wgsl. A new texel starts from prior, the texel of the level above, as at most seedCount paths.
//weight: the paths the sample is the average of. The count is capped at countMax
fn addTexelSample(index: u32, visibility: f32, weight: f32, prior: vec2<f32>, countMax: f32) -> vec2<f32> {
	let texelWord = &lightingBuffer[index];
	var old = atomicLoad(texelWord);
	var result = prior;
	for(var i = 0u; i < TEXEL_CAS_TRIES; i++){
		var texel = unpack2x16float(old);
		if(texel.y == 0. && prior.y > 0.){
			texel = vec2(prior.x, min(prior.y, uniforms.seedCount));
		}
		texel = vec2(mix(texel.x, visibility, weight / (texel.y + weight)), min(texel.y + weight, countMax));
		let cmp = casLighting(index, old, pack2x16float(texel));
		if(cmp.exchanged){
			result = texel;
			break;
		}
		old = cmp.old_value;
	}
	return result;
}

//2 uniform random numbers for sample index of the seed, see getBlueNoise in pathTrace.wgsl of the irradiance cache
fn getBlueNoise(coord: vec2<u32>, index: u32, dimension: u32) -> vec2<f32> {
	let size = textureDimensions(blueNoiseTexture);
	let offset = vec2<u32>(getR2(dimension + 1u) * vec2<f32>(size));
	let start = textureLoad(blueNoiseTexture, (coord + offset) % size, 0).xy;
	return fract(start + getR2(index));
}

//the paths the texel took, see getTexelPathIndex. -1: none, the paths are numbered by the frame
var<private> pathIndex: i32 = -1;

fn cosWeightedRandomHemisphereDirection(n: vec3<f32>, r: vec2<f32>) -> vec3<f32> {
	let uu = normalize(cross(n, select(vec3(0., 1., 0.), vec3(1., 0., 0.), abs(n.y) > .5)));
	let vv = cross(uu, n);
	let ra = sqrt(r.y);
	let rx = ra * cos(TAU * r.x);
	let ry = ra * sin(TAU * r.x);
	let rz = sqrt(1. - r.y);
	return normalize(rx * uu + ry * vv + rz * n);
}

/*
the visibility of path i of the sampleCount paths of a texel: 0 or the weight of a ray in a cosine weighted direction
around n that reaches the sky. A ray down from a point above the ground ends on the ground or on a building, which add
nothing, so it needs no ray. With FOLD_DOWN_PATHS it is mirrored up (y = -y) instead: the pdf of a direction d that
goes up is then p(d) + p(m(d)), with p the cosine pdf and m the mirror, so the path weighs p(d) / (p(d) + p(m(d))).
That is .5 on a wall (its lobe is symmetric in y) and 1 on a roof, exact for any normal. A wall then traces twice the
rays for a visibility of 0 or .5 instead of 0 or 1 per path: less noise per path, a lot less on a wall that sees much sky
 */
//the direction (xyz) and the weight (w) of path i: the ray reaches the sky or it is 0. weight 0: no ray needed.
//PACKED_TEXELS: only a wall folds (n: its normal without y), with weight 1: a hit in the 2 paths the ray counts as
fn getPathRay(n: vec3<f32>, seed: vec2<u32>, i: u32, sampleCount: u32, wall: bool) -> vec4<f32> {
	//the paths of the texel so far: its sequence without gaps, also when sampleCount changes between frames
	let index = select(uniforms.frame * sampleCount, u32(pathIndex), pathIndex >= 0) + i;
	let fold = FOLD_DOWN_PATHS && (wall || !PACKED_TEXELS);
	var r = getBlueNoise(seed, index, 0u);
	if(fold && PACKED_TEXELS){
		//the half of the lobe that goes up (its angle is TAU * r.x, from the horizontal tangent of a wall), so the rays
		//are spread as their numbers are. Mirrored (abs below), the rays of the 2 halves fell next to each other
		r.x *= .5;
	}
	var rd = cosWeightedRandomHemisphereDirection(n, r);
	var weight = 1.;
	if(fold){
		rd.y = abs(rd.y);
		if(!PACKED_TEXELS){
			let cosine = max(dot(n, rd), 0.);
			let mirrored = max(dot(n, vec3(rd.x, -rd.y, rd.z)), 0.);
			weight = cosine / max(cosine + mirrored, 1e-20);
		}
	}else if(rd.y < 0.){
		weight = 0.;
	}
	return vec4(rd, weight);
}

//the direction of sun ray index of a texel (the sun rays it took before, see sunIndex of trace_main): toward a point on
//the disc of the sun, sunRadius around sunDirection, from the blue noise of the texel moved along the R2 sequence
//(another dimension than the sky paths). It reaches the sun or it is 0
fn getSunRay(seed: vec2<u32>, index: u32) -> vec3<f32> {
	let l = uniforms.sunDirection;
	let r = getBlueNoise(seed, index, 1u);
	let t = normalize(cross(l, select(vec3(0., 1., 0.), vec3(1., 0., 0.), abs(l.y) > .5)));
	let b = cross(l, t);
	//uniform on the disc. Small angles: the tangent offset is the angle
	let radius = uniforms.sunRadius * sqrt(r.x);
	let phi = TAU * r.y;
	return normalize(l + (t * cos(phi) + b * sin(phi)) * radius);
}

fn getCellPoint(cellID: vec2<u32>) -> CellPoint {
	var point: CellPoint;
	point.valid = false;
	let rand = hash34uf(vec3<u32>(cellID, uniforms.frame)).zw;
	//a random pixel in the cell, the same one in every pass of the frame
	point.coord = cellID * CELL_SIZE + vec2<u32>(rand * f32(CELL_SIZE));
	if(point.coord.x >= u32(uniforms.resolution.x) || point.coord.y >= u32(uniforms.resolution.y)){
		return point;
	}
	let g0 = textureLoad(gBufferTexture, point.coord, 0).xy;
	point.surface = Surface(g0.x, vec2(0.));
	if(g0.x == SURFACE_NONE){
		return point;
	}
	let g = vec3(g0, textureLoad(gBufferAtlasTexture, point.coord, 0).x);
	point.n = getGBufferNormal(g);
	let uv = (vec2<f32>(point.coord) + .5) / uniforms.resolution;
	let d = getGBufferDistance(textureLoad(gBufferDepth, point.coord, 0), uv, uniforms.inverseProjection, uniforms.depthParams);
	point.worldPosition = depthToPosition(d, uv * 2. - 1.);
	point.surface = getGBufferSurface(g, point.worldPosition, uniforms.atlasStep);
	point.valid = true;
	return point;
}

//the texels of the point of a cell at its level and the coarser ones, TILE_NONE where the lightmap is full.
//insert: claim_main adds the tiles that don't exist yet and marks them used, trace_main only finds them
fn getCellTexels(point: CellPoint, insert: bool) -> array<u32, ANCESTOR_LEVELS_MAX + 1u> {
	let level = u32(getLevel(point.worldPosition, point.n));
	var indices: array<u32, ANCESTOR_LEVELS_MAX + 1u>;
	for(var i = 0u; i <= ANCESTOR_LEVELS_MAX; i++){
		indices[i] = TILE_NONE;
	}
	for(var i = 0u; i <= min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX); i++){
		let texel = getTexel(point.surface, level + i);
		var tileIndex: u32;
		if(insert){
			tileIndex = findOrInsert(point.surface.id, level + i, texel / TILE_SIZE, uniforms.frame);
		}else{
			tileIndex = find(point.surface.id, level + i, texel / TILE_SIZE);
		}
		indices[i] = select(getTexelIndex(tileIndex, texel), TILE_NONE, tileIndex == TILE_NONE);
	}
	return indices;
}

//the seed of the random numbers of the paths of a texel: its place in the lightmap, see updateVirtualLightmap.wgsl
fn getTexelSeed(texelIndex: u32) -> vec2<u32> {
	let t = vec2(texelIndex % TILE_SIZE, (texelIndex / TILE_SIZE) % TILE_SIZE);
	let tile = texelIndex / (TILE_SIZE * TILE_SIZE);
	return t + vec2(tile % 32u, tile / 32u) * TILE_SIZE;
}

//the paths a texel took, so it walks its R2 sequence without gaps. -1 when its count is at its cap
fn getTexelPathIndex(texelIndex: u32) -> i32 {
	let count = loadTexel(texelIndex).y;
	if(count + f32(uniforms.sampleCount) > uniforms.accumulationMax){
		return -1;
	}
	return i32(count);
}

//the slot in traceCells of the texel a cell bids on: after the slots of the cells that trace
fn getCellTexelSlot(cellID: vec2<u32>) -> u32 {
	let groupSize = vec2(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y) * CELL_SIZE;
	let cellCount = (vec2<u32>(uniforms.resolution) + groupSize - 1u) / groupSize * vec2(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y);
	return cellCount.x * cellCount.y + cellID.y * cellCount.x + cellID.x;
}

//the bits of a claim below the frame, and the mask of the frame: 20 bits, so it wraps after 4.8 hours at 60 fps
const CLAIM_FRAME_SHIFT = 12u;
const CLAIM_FRAME_MASK = 0xFFFFFu;

//the frame in the high bits, so a claim of an earlier frame never wins, random below it
fn getClaimKey(cellID: vec2<u32>) -> u32 {
	let frame = uniforms.frame;
	return (frame << CLAIM_FRAME_SHIFT) | (tileHash(cellID.x ^ tileHash(cellID.y ^ tileHash(frame))) & ((1u << CLAIM_FRAME_SHIFT) - 1u));
}

//the frame a texel was last sampled in, 0 when never: the frame part of its claim, see claim_main
fn getLastSampled(index: u32) -> u32 {
	let frame = uniforms.frame;
	let claim = atomicLoad(&claimBuffer[index]);
	if(claim == 0u){
		return 0u;
	}
	return frame - ((frame - (claim >> CLAIM_FRAME_SHIFT)) & CLAIM_FRAME_MASK);
}

/*
a tile came or went since the texel was last sampled that changed its sky visibility or its sun visibility
(changeFrames, see getChangeFrames): that visibility is of the buildings of before. It turns stale (STALE_COUNT in
skyVisibilityTexel.wgsl): kept to show, without paths, so the texel is no longer resolved, the cell bids on it and its
next rays replace it.
The frame part of claimBuffer is when the texel was last sampled, see resetStaleTexel in updateCityLightmap.wgsl.
Called for the texels of a cell, by claim_main. Returns whether this call made the texel stale: 1 of the cells that
share it
 */
fn resetStaleTexel(index: u32, changeFrames: vec2<u32>) -> bool {
	let lastSampled = getLastSampled(index);
	let sky = lastSampled < changeFrames.x;
	let sun = LIGHTMAP_SUN && lastSampled < changeFrames.y;
	if(!sky && !sun){
		return false;
	}
	if(PACKED_TEXELS){
		return setPackedTexelStale(index, sky, sun);
	}
	var changed = false;
	if(sky){
		changed = setTexelStale(index);
	}
	if(sun){
		changed = setTexelStale(index + getSunTexelOffset()) || changed;
	}
	return changed;
}

/*
The texels around the one of a surface point at a level that are stale too, when that one was (resetStaleTexel). The
lookup of a pixel blends the 4 texels around its point, and 1 of them can be under a building that came: no pixel shows
it, so no cell ever resets it, and it kept the sky and the sun of the open ground with all its paths. That lit the
pixels along the foot of the building
 */
fn resetStaleNeighbours(surface: Surface, level: u32, changeFrames: vec2<u32>) {
	let texel = vec2<i32>(getTexel(surface, level));
	let centerTile = vec2<u32>(texel) / TILE_SIZE;
	let centerIndex = find(surface.id, level, centerTile);
	for(var y = -1; y <= 1; y++){
		for(var x = -1; x <= 1; x++){
			if(x == 0 && y == 0){
				continue;
			}
			let neighbour = vec2<u32>(texel + vec2(x, y));
			let tile = neighbour / TILE_SIZE;
			var tileIndex = centerIndex;
			if(any(tile != centerTile)){
				tileIndex = find(surface.id, level, tile);
			}
			if(tileIndex != TILE_NONE){
				resetStaleTexel(getTexelIndex(tileIndex, neighbour), changeFrames);
			}
		}
	}
}

//the texel word at index (a sky or a sun texel of 2 x f16) stale: its visibility kept as STALE_COUNT paths. Returns
//whether it had more
fn setTexelStale(index: u32) -> bool {
	let texelWord = &lightingBuffer[index];
	var old = atomicLoad(texelWord);
	for(var i = 0u; i < TEXEL_CAS_TRIES; i++){
		let v = unpack2x16float(old);
		if(v.y <= STALE_COUNT){
			return false;
		}
		let cmp = casLighting(index, old, pack2x16float(vec2(v.x, STALE_COUNT)));
		if(cmp.exchanged){
			return true;
		}
		old = cmp.old_value;
	}
	return false;
}

//the fields of a packed count (rays, hits) scaled down to rays rayMax at most, the visibility kept (rounded)
fn capPackedChannel(hitsRays: vec2<u32>, rayMax: u32) -> vec2<u32> {
	if(hitsRays.y <= rayMax){
		return hitsRays;
	}
	let hits = u32(round(f32(hitsRays.x) * f32(rayMax) / f32(hitsRays.y)));
	return vec2(min(hits, rayMax), rayMax);
}

//a packed channel (hits, rays) as stale: no rays, its visibility in the hits, 1 to hitsMax (see STALE_COUNT in
//skyVisibilityTexel.wgsl). One without rays stays as it is: empty, or stale already
fn getStaleChannel(hitsRays: vec2<u32>, hitsMax: u32) -> vec2<u32> {
	if(hitsRays.y == 0u){
		return hitsRays;
	}
	return vec2(1u + u32(round(f32(hitsRays.x) * f32(hitsMax - 1u) / f32(hitsRays.y))), 0u);
}

//the rays of a packed channel (hits, rays) to add to: none when it is stale or empty, the visibility of a stale one
//goes with its first rays
fn getPackedRays(hitsRays: vec2<u32>) -> vec2<u32> {
	return select(vec2(0u), hitsRays, hitsRays.y > 0u);
}

//setTexelStale of a packed texel (PACKED_TEXELS): its sky and its sun channel, each when asked. Returns whether it
//had rays there. Rare, so a compare exchange
fn setPackedTexelStale(index: u32, sky: bool, sun: bool) -> bool {
	let texelWord = &lightingBuffer[index];
	var old = atomicLoad(texelWord);
	for(var i = 0u; i < TEXEL_CAS_TRIES; i++){
		var r = unpackTexelRays(old);
		if(sky){
			r = vec4(getStaleChannel(r.xy, PACKED_SKY_MAX), r.zw);
		}
		if(sun){
			r = vec4(r.xy, getStaleChannel(r.zw, PACKED_SUN_MAX));
		}
		let stale = packTexelRays(r);
		if(stale == old){
			return false;
		}
		let cmp = casLighting(index, old, stale);
		if(cmp.exchanged){
			return true;
		}
		old = cmp.old_value;
	}
	return false;
}

//PACKED_TEXELS: the sky rays (x) and the sun rays (y) a texel takes at most. accumulationMax (the host keeps it at most
//PACKED_ACCUMULATION_MAX and a multiple of PACKED_SUN_SCALE, see SkyVisibilityLightmap.getAccumulationMax), so both fit
//their field
fn getPackedCaps() -> vec2<u32> {
	let skyCap = u32(uniforms.accumulationMax);
	return vec2(skyCap, skyCap / PACKED_SUN_SCALE);
}

//a channel of a texel (hits, rays) after a batch: as it was when it is full (its rays at the cap), else with the batch,
//scaled down to the room under the cap. A channel without rays starts from the coarser texel (prior) as at most
//seedMax rays, as addTexelSample, and a stale one loses its visibility of before (getPackedRays)
fn addPackedSky(texel: vec2<u32>, batch: vec2<u32>, prior: vec2<u32>, cap: u32, seedMax: u32) -> vec2<u32> {
	if(batch.y == 0u || texel.y >= cap){
		return texel;
	}
	var add = batch;
	if(texel.y == 0u){
		add += capPackedChannel(getPackedRays(prior), seedMax);
	}
	return getPackedRays(texel) + capPackedChannel(add, cap - texel.y);
}

//a texel with sun rays that reached the sun and ones that didn't: in a penumbra, or on the edge of a shadow
fn isPenumbra(sun: vec2<u32>) -> bool {
	return sun.x > 0u && sun.x < sun.y;
}

/*
The sun fields (hits, rays) of a texel after the sun rays of a batch (its hits and rays as they were traced), see
trace_main for why. While all the sun rays of a texel agree it is most likely fully lit or fully in shadow, and a ray
counts as weight rays: the count goes up as the one of the sky does, and stops at the cap after a quarter or an eighth
of the rays. The first batch that disagrees shows it is in a penumbra: its count goes back to the rays that were traced
(/ weight: they all agreed, so the visibility is kept exactly), also when it was full, and from then on a ray counts
once, up to the cap in rays that were traced.
A texel without rays starts from the coarser one (prior) as at most seedMax rays, as addPackedSky: under a coarser
texel in a penumbra it mostly starts as one. Tried and not kept: a ray that also counts once while the coarser texel
is in a penumbra (it gets the rays of all the cells in it, so it knows first). The same noise, 1 more frame of 240
 */
fn addPackedSun(texel: vec2<u32>, batch: vec2<u32>, weight: u32, prior: vec2<u32>, cap: u32, seedMax: u32) -> vec2<u32> {
	if(batch.y == 0u){
		return texel;
	}
	var t = texel;
	if(t.y == 0u){
		//a stale one loses its visibility of before
		t = capPackedChannel(getPackedRays(prior), seedMax);
	}
	var add = batch;
	if(!isPenumbra(t)){
		//an empty texel agrees with a batch that agrees with itself
		let agrees = (batch.x == 0u && t.x == 0u) || (batch.x == batch.y && t.x == t.y);
		if(agrees){
			add = batch * weight;
		}else if(!agrees && t.y > 0u){
			let rays = max((t.y + weight / 2u) / weight, 1u);
			t = vec2(select(0u, rays, t.x > 0u), rays);
		}
	}
	if(t.y >= cap){
		return t;
	}
	return t + capPackedChannel(add, cap - t.y);
}

/*
PACKED_TEXELS: adds the batch of a cell (its sky hits and paths, its sun hits and rays, and the rays a sun ray counts
as, see addPackedSun) to its texel and the coarser ones above it, 1
compare exchange per texel (more when another cell added in between). A texel with both channels full gets no atomic at
all, only the load. The rays of a channel never go over its cap (addPackedSky, addPackedSun), and the exchange only goes in on the
word the add was made for, so a field never overflows into the next one, whatever the size of the batch.
Not an atomicAdd that subtracts itself again when it went over: the coarser texels get the batches of many cells in the
same frame, and an add that read the word while another one had wrapped a field (before that one's subtract) saw a
small count and kept its batch. The sum stayed over the field, carried into the next one: hits above rays, a texel too
bright or too dark for good (the spots on the walls). dawn.node, RTX 3080, 4K Very High, Amsterdam: 433 such reads in
the first 50 frames (most with fresh coarse texels), 100 texels with hits > rays after converging, and
the rest wrong without showing it. The compare exchange made the trace pass no slower
 */
fn addPackedBatch(indices: array<u32, ANCESTOR_LEVELS_MAX + 1u>, levels: u32, batch: vec4<u32>, sunWeight: u32) {
	let caps = getPackedCaps();
	let seedMax = vec2(u32(uniforms.seedCount), max(u32(uniforms.seedCount) / PACKED_SUN_SCALE, 1u));
	//the coarser texel after the batch, for the seed
	var prior = vec4(0u);
	for(var i = i32(levels); i >= 0; i--){
		let index = indices[i];
		if(index == TILE_NONE){
			continue;
		}
		let texelWord = &lightingBuffer[index];
		var old = atomicLoad(texelWord);
		var texel = unpackTexelRays(old);
		for(var t = 0u; t < TEXEL_CAS_TRIES; t++){
			texel = unpackTexelRays(old);
			let next = vec4(
				addPackedSky(texel.xy, batch.xy, prior.xy, caps.x, seedMax.x),
				addPackedSun(texel.zw, batch.zw, sunWeight, prior.zw, caps.y, seedMax.y));
			if(all(next == texel)){
				break;
			}
			let cmp = casLighting(index, old, packTexelRays(next));
			if(cmp.exchanged){
				texel = next;
				break;
			}
			old = cmp.old_value;
		}
		prior = texel;
	}
}

/*
1 claim_main: every cell inserts its tiles and bids on its texel, unless the texel is resolved
2 select_main: the cell with the highest bid of a texel goes in traceCells, at most traceCellsMax of them
3 trace_main: only those cells trace
See updateVirtualLightmap.wgsl
 */
@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn claim_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let cellID = GlobalInvocationID.xy;
	let texelSlot = getCellTexelSlot(cellID);
	traceCells[texelSlot] = TILE_NONE;
	let point = getCellPoint(cellID);
	if(!point.valid){
		return;
	}
	let indices = getCellTexels(point, true);
	let levels = min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX);
	//before the count of the texel is read below: a texel that is reset is not resolved
	if(uniforms.changeCount > 0u){
		//the changes since the finest texel of the cell was last sampled only, none while it takes samples: its coarser
		//texels were sampled with it (their claims get its frame below). So most cells load 1 claim and test no box
		var finest = 0u;
		while(finest < levels && indices[finest] == TILE_NONE){
			finest++;
		}
		var changeFrames = vec2(0u);
		if(indices[finest] != TILE_NONE){
			let lastSampled = getLastSampled(indices[finest]);
			if(lastSampled < uniforms.changeEnd){
				changeFrames = getChangeFrames(point.worldPosition, lastSampled);
			}
		}
		if(any(changeFrames != vec2(0u))){
			for(var i = 0u; i <= levels; i++){
				if(indices[i] != TILE_NONE && resetStaleTexel(indices[i], changeFrames)){
					//the level as getCellTexels has it
					let level = u32(getLevel(point.worldPosition, point.n));
					resetStaleNeighbours(point.surface, level + i, changeFrames);
				}
			}
		}
	}
	//the texel the cell bids on: the one of its level, or the finest coarser one where the lightmap has no room for the
	//tile of that (all the slots findOrInsert looks in hold tiles in use, 4% of the tiles at 70% full). Without a bid
	//the coarser texels there, which are inside that 1 tile, got no samples either: nothing for the resolve to show
	//but an open sky, white blocks on the ground. With it the resolve finds the coarser texel
	var own = 0u;
	while(own < levels && indices[own] == TILE_NONE){
		own++;
	}
	let texelIndex = indices[own];
	if(texelIndex == TILE_NONE){
		return;
	}
	if(getTexelCount(texelIndex) >= uniforms.accumulationMax){
		return;
	}
	traceCells[texelSlot] = texelIndex;

	//the texel gets a sample this frame, from this cell or the one that wins the bid, and so do the coarser texels
	//above it: the frame of their claim is when they were last sampled, see resetStaleTexel. Not over a claim of this
	//frame, which can be the bid of a cell that has the texel as its own
	let stamp = (uniforms.frame & CLAIM_FRAME_MASK) << CLAIM_FRAME_SHIFT;
	for(var i = own + 1u; i <= levels; i++){
		if(indices[i] != TILE_NONE){
			let ancestorClaim = &claimBuffer[indices[i]];
			let ancestorOld = atomicLoad(ancestorClaim);
			if((ancestorOld >> CLAIM_FRAME_SHIFT) != (stamp >> CLAIM_FRAME_SHIFT)){
				casClaim(indices[i], ancestorOld, stamp);
			}
		}
	}

	let key = getClaimKey(cellID);
	let claim = &claimBuffer[texelIndex];
	var old = atomicLoad(claim);
	for(var i = 0u; i < 16u; i++){
		if((old >> CLAIM_FRAME_SHIFT) == (key >> CLAIM_FRAME_SHIFT) && old >= key){
			return;
		}
		let cmp = casClaim(texelIndex, old, key);
		if(cmp.exchanged){
			return;
		}
		old = cmp.old_value;
	}
}

/*
The counters of select_main are summed per workgroup first, then added to traceArgs with 1 atomic per counter and
workgroup, and traceArgs[5 - parity] is read once per workgroup. Per cell, all the cells of the screen did their atomics
on the same few words, and those run one after the other: on D3D12 even an atomicLoad is an atomic read-modify-write.
That took select_main from .25 to .07 ms (dawn.node, RTX 3080, 1080p High). The winners of a workgroup get consecutive
places in traceCells: the order was the one of the atomics before, so it was not fixed either
 */
var<workgroup> selectWins: atomic<u32>;
var<workgroup> selectNoisy: atomic<u32>;
var<workgroup> selectSurvivors: atomic<u32>;
var<workgroup> selectPrevious: u32;
var<workgroup> selectBase: u32;

@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn select_main(
	@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>,
	@builtin(local_invocation_index) localIndex : u32
) {
	let cellID = GlobalInvocationID.xy;
	let parity = uniforms.frame & 1u;
	if(localIndex == 0u){
		atomicStore(&selectWins, 0u);
		atomicStore(&selectNoisy, 0u);
		atomicStore(&selectSurvivors, 0u);
		selectPrevious = atomicLoad(&traceArgs[5u - parity]);
	}
	workgroupBarrier();
	let texelIndex = traceCells[getCellTexelSlot(cellID)];
	let key = getClaimKey(cellID);
	var won = texelIndex != TILE_NONE && atomicLoad(&claimBuffer[texelIndex]) == key;
	//the place of the cell among the ones of its workgroup that trace
	var local = 0u;
	if(won){
		//over traceCellsMax in the previous frame: a random part of the winners is left out first, see updateVirtualLightmap.wgsl
		atomicAdd(&selectWins, 1u);
		//the ones that still look noisy: the host keeps drawing while there are many, see SkyVisibilityLightmap.readBack
		if(getTexelCount(texelIndex) < uniforms.convergedPaths){
			atomicAdd(&selectNoisy, 1u);
		}
		let previous = selectPrevious;
		if(previous > uniforms.traceCellsMax){
			let random = f32(tileHash(key ^ 0x9E3779B9u) >> 8u) / 16777216.;
			if(random * f32(previous) >= f32(uniforms.traceCellsMax)){
				won = false;
			}
		}
		if(won){
			local = atomicAdd(&selectSurvivors, 1u);
		}
	}
	workgroupBarrier();
	if(localIndex == 0u){
		let wins = atomicLoad(&selectWins);
		if(wins > 0u){
			atomicAdd(&traceArgs[4u + parity], wins);
		}
		let noisy = atomicLoad(&selectNoisy);
		if(noisy > 0u){
			atomicAdd(&traceArgs[6], noisy);
		}
		let survivors = atomicLoad(&selectSurvivors);
		if(survivors > 0u){
			selectBase = atomicAdd(&traceArgs[3], survivors);
		}
	}
	workgroupBarrier();
	if(!won){
		return;
	}
	let index = selectBase + local;
	if(index >= uniforms.traceCellsMax){
		return;
	}
	traceCells[index] = cellID.x | (cellID.y << 16u);
	if(index % (TRACE_GROUP_SIZE / uniforms.traceLanes) == 0u){
		atomicAdd(&traceArgs[0], 1u);
	}
	if(index == 0u){
		atomicStore(&traceArgs[1], 1u);
		atomicStore(&traceArgs[2], 1u);
	}
}

//the paths of a cell summed per thread, see trace_main, and its sun rays (LIGHTMAP_SUN)
var<workgroup> pathSums: array<f32, TRACE_GROUP_SIZE>;
var<workgroup> sunSums: array<f32, TRACE_GROUP_SIZE>;
//the cells that trace, read from traceArgs once per workgroup instead of by every thread: on D3D12 an atomicLoad is an
//atomic read-modify-write, and all the threads of the pass on 1 word ran one after the other. That took trace_main from
//1.27 to .85 ms (dawn.node, RTX 3080, 1080p High, Amsterdam), more where the rays are short (a forest 1.17 to .45 ms)
var<workgroup> groupTraceCells: u32;

/*
traceLanes threads per cell, which each trace every traceLanes-th of its sampleCount rays. Lane 0 adds the sums to the
texel and to the coarser ones above it.
LIGHTMAP_SUN: the first rays of the cell go to the sun, the others to the sky. sunSampleCount of them while both take
samples (1 per 8 paths), all of them while only the sun does (after the sun moved), none while only the sky does. So the
threads of a cell always have the same rays to trace, and the rays of a cell stay sampleCount (the budget of
traceCellsMax). 1 call of isOccluded for both: with 2, the threads of a subgroup with a sun ray and the ones with a sky
ray would run the 2 traversals one after the other.
A texel in a penumbra (some of its sun rays so far reached the sun, some didn't) takes more of them (without
PACKED_TEXELS sunSampleCountPenumbra, 1 per 4 paths), and the sun rays of a texel walk its sequence of points on the
sun without gaps (sunIndex), as its sky paths do. With a low sun the penumbras are wide (a sunset: 20 m per 100 m of
shadow along it, the shadows 10x the height of what casts them), so many texels are in one, and there the noise is
which points on the sun the rays went to.
A sun sample weighs sampleCount, as much as the sky paths of the cell, so the counts of the 2 go up alike: the variance
of the sun visibility of a texel is mostly where in it the cell's pixel is, not the point on the sun.
PACKED_TEXELS: a texel counts rays, so a sample weighs its rays. A sky ray of a wall is folded up (getPathRay) and
counts as 2 paths: the one that went up and its mirror image down, which is 0. And next to sky rays a sun ray counts as
the paths of those sky rays / PACKED_SUN_SCALE (rounded up, sunRayCount), so the 2 counts of a texel go up alike here
too. But only while all the sun rays of the texel agree. From the first one that doesn't, the texel is in a penumbra:
its count goes back to the rays that were traced and a ray counts once (addPackedSun), so it ends with the 127 rays
the field holds. The rays of its cell then split as the rays the sun and the sky still take (a fifth to the sun on the
ground and a roof, a third on a wall, whose sky paths count twice), so both are full together.
Every sun ray counted once, the sun is at half of its rays when the sky is full (a quarter on a wall), and the cells
after that have all their rays to the sun, for every texel, also the 92% whose rays all agree: 316 frames to
converge instead of 241, for the same noise. Counted as the sky paths in every texel, a texel has 32 sun rays on a
wall and 64 on the ground (16 and 32 with 1 sun ray per 16 paths, and numbered by the frame, so as good as random).
The noise on the shadow edges (New York, sunset, 1080p, High, converged, frames to converge): 1 per 16 paths by the
frame .057 (223), in sequence .040 (223), 1 per 8 paths .027 (236), and a quarter of the rays in a penumbra .023
(238), a penumbra counted once .018 (241), every texel counted once .0175 (316).
Tried and not kept: the sun rays counted for the part of the room of the sun that the sky rays
are of the room of the sky, so both are full together also when the sun was cleared with the sky half full. 1.25x fewer
frames after such a sun move, 1.13x the noise on the shadow edges.
A sun ray into the surface (a face away from the sun, or the part of the disc below its horizon) is a miss without a
trace, still counted. Traced, 13% of them reached the sun: it starts SUN_RAY_BIAS in front of the wall and only gets
back to its plane after bias / |n.rd|, metres at a grazing angle, by then over the roof or past the corner. The shading
reads no sun there, but the coarser texels and the gather across a corner brought it to the lit faces
 */
@compute @workgroup_size(TRACE_GROUP_SIZE, 1, 1)
fn trace_main(
	@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>,
	@builtin(local_invocation_index) localIndex : u32
) {
	let lanes = uniforms.traceLanes;
	let cell = GlobalInvocationID.x / lanes;
	let lane = GlobalInvocationID.x % lanes;
	if(localIndex == 0u){
		groupTraceCells = min(atomicLoad(&traceArgs[3]), uniforms.traceCellsMax);
	}
	workgroupBarrier();
	var hasCell = cell < groupTraceCells;
	var cellID = vec2(0u);
	var point: CellPoint;
	if(hasCell){
		let packed = traceCells[cell];
		cellID = vec2(packed & 0xFFFFu, packed >> 16u);
		point = getCellPoint(cellID);
		hasCell = point.valid;
	}
	var sum = 0.;
	var sunSum = 0.;
	var skyNeeded = false;
	var sunNeeded = false;
	//how the sampleCount rays of the cell split, see above. Every lane of the cell gets the same, lane 0 divides by them
	var sunRays = 0u;
	var skyRays = 0u;
	//PACKED_TEXELS: the paths a sky ray counts as and the sun rays a sun ray counts as, see above
	var skyRayCount = 1u;
	var sunRayCount = 1u;
	var wall = false;
	var skyNormal = vec3(0.);
	//the sun rays the texel took (about: it counts them weighted), so they walk its sequence without gaps, see above
	var sunIndex = 0u;
	if(hasCell){
		let texelIndex = traceCells[getCellTexelSlot(cellID)];
		wall = (point.surface.id & SURFACE_WALL) != 0u;
		skyNormal = point.n;
		if(PACKED_TEXELS){
			//1 load
			let r = unpackTexelRays(atomicLoad(&lightingBuffer[texelIndex]));
			let caps = getPackedCaps();
			skyNeeded = r.y < caps.x;
			sunNeeded = LIGHTMAP_SUN && r.w < caps.y;
			if(FOLD_DOWN_PATHS && wall){
				skyRayCount = 2u;
				//the weight of a folded path is .5 for a normal without y only, and the one of the gBuffer is rounded
				skyNormal = normalize(point.n * vec3(1., 0., 1.));
			}
			//no more rays than the texel still takes
			let sunRoom = caps.y - min(r.w, caps.y);
			let skyRoom = (caps.x - min(r.y, caps.x) + skyRayCount - 1u) / skyRayCount;
			let sunAlone = min(uniforms.sampleCount, PACKED_SUN_RAYS_MAX);
			//in a penumbra so far: its sun rays count once (addPackedSun), so the rays of the cell split as the rays the
			//sun and the sky still take, and both are full together
			let penumbra = isPenumbra(r.zw);
			var sunJoint = uniforms.sunSampleCount;
			if(penumbra){
				let split = u32(round(f32(uniforms.sampleCount * sunRoom) / f32(max(sunRoom + skyRoom, 1u))));
				sunJoint = clamp(split, uniforms.sunSampleCount, max(uniforms.sampleCount, uniforms.sunSampleCount + 1u) - 1u);
			}
			sunRays = select(0u, min(select(sunAlone, sunJoint, skyNeeded), sunRoom), sunNeeded);
			skyRays = select(0u, min(uniforms.sampleCount - sunRays, skyRoom), skyNeeded);
			if(sunRays > 0u && skyRays > 0u){
				//the sun count goes up as the sky count does (rounded up)
				let sunPaths = PACKED_SUN_SCALE * sunRays;
				sunRayCount = (skyRays * skyRayCount + sunPaths - 1u) / sunPaths;
			}
			//the rays it took, so its sequence has no gaps (getTexelPathIndex)
			pathIndex = i32(r.y / skyRayCount);
			sunIndex = select(r.w / sunRayCount, r.w, penumbra);
		}else{
			skyNeeded = loadTexel(texelIndex).y < uniforms.accumulationMax;
			var sunTexel = vec2(0.);
			if(LIGHTMAP_SUN){
				sunTexel = loadSunTexel(texelIndex);
			}
			sunNeeded = LIGHTMAP_SUN && sunTexel.y < uniforms.accumulationMax;
			pathIndex = getTexelPathIndex(texelIndex);
			//a running average: not exactly 0 or 1 where all the rays agreed (pack2x16float)
			let penumbra = sunTexel.y > 0. && sunTexel.x > .01 && sunTexel.x < .99;
			let sunJoint = select(uniforms.sunSampleCount, uniforms.sunSampleCountPenumbra, penumbra);
			sunRays = select(0u, select(sunJoint, uniforms.sampleCount, !skyNeeded), sunNeeded);
			skyRays = select(0u, uniforms.sampleCount - sunRays, skyNeeded);
			//a visit weighs sampleCount
			sunIndex = u32(sunTexel.y / f32(uniforms.sampleCount)) * sunRays;
		}
		let seed = getTexelSeed(texelIndex);
		let skyRo = point.worldPosition + point.n * MIN_DIST;
		let d = distance(point.worldPosition, uniforms.cameraPosition);
		let sunRo = point.worldPosition + point.n * max(MIN_DIST, d * SUN_RAY_BIAS);
		for(var i = lane; i < sunRays + skyRays; i += lanes){
			let sun = i < sunRays;
			var ro = skyRo;
			var ray: vec4<f32>;
			if(sun){
				ro = sunRo;
				ray = vec4(getSunRay(seed, sunIndex + i), 1.);
				//into the surface: a miss without a trace, see above
				ray.w = select(0., 1., dot(point.n, ray.xyz) > 0.);
			}else{
				ray = getPathRay(skyNormal, seed, i - sunRays, skyRays, wall);
			}
			if(ray.w > 0. && !isOccluded(ro, ray.xyz)){
				if(sun){
					sunSum += ray.w;
				}else{
					sum += ray.w;
				}
			}
		}
	}
	pathSums[localIndex] = sum;
	sunSums[localIndex] = sunSum;
	workgroupBarrier();
	if(!hasCell || lane != 0u){
		return;
	}
	var total = 0.;
	var sunTotal = 0.;
	for(var i = 0u; i < lanes; i++){
		total += pathSums[localIndex + i];
		sunTotal += sunSums[localIndex + i];
	}
	if(PACKED_TEXELS){
		//the rays weigh 0 or 1, so the sums are the hits
		let batch = vec4(u32(round(total)), skyRays * skyRayCount, u32(round(sunTotal)), sunRays);
		addPackedBatch(getCellTexels(point, false), min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX), batch, sunRayCount);
		return;
	}
	let visibility = total / f32(max(skyRays, 1u));
	let sunVisibility = sunTotal / f32(max(sunRays, 1u));
	//the sky as the paths it took, the sun as the cell's sampleCount (see above)
	let skyWeight = f32(skyRays);
	let sunWeight = f32(uniforms.sampleCount);
	let indices = getCellTexels(point, false);

	//from coarse to fine, a new texel starts from the coarser one
	let ancestorLevels = min(uniforms.ancestorLevels, ANCESTOR_LEVELS_MAX);
	var prior = vec2(0.);
	var sunPrior = vec2(0.);
	for(var i = i32(ancestorLevels); i >= 0; i--){
		let index = indices[i];
		if(index == TILE_NONE){
			continue;
		}
		if(skyNeeded){
			prior = addTexelSample(index, visibility, skyWeight, prior, uniforms.accumulationMax);
		}
		if(sunNeeded){
			sunPrior = addTexelSample(index + getSunTexelOffset(), sunVisibility, sunWeight, sunPrior, uniforms.accumulationMax);
		}
	}
}

//PACKED_TEXELS: the sun moved, the sun fields of every texel are cleared (SkyVisibilityLightmap.setSun). Without it the
//host clears the sun half of lightingBuffer. 1 thread per texel, the texels as a LIGHTMAP_SIZE x LIGHTMAP_SIZE grid
@compute @workgroup_size(WORK_GROUP_SIZE_X, WORK_GROUP_SIZE_Y, 1)
fn clear_sun_main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
	let p = GlobalInvocationID.xy;
	if(p.x >= LIGHTMAP_SIZE || p.y >= LIGHTMAP_SIZE){
		return;
	}
	atomicAnd(&lightingBuffer[p.y * LIGHTMAP_SIZE + p.x], PACKED_SKY_MASK);
}
