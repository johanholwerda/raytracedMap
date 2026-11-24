//the levels, texels and tiles of the city lightmap, see SkyVisibilityLightmap.ts. Like virtualLightmap.wgsl of the irradiance
//cache, with the charts of cityLightmapSurface.wgsl. Needs uniforms.cameraPosition, viewProjection, pixelSize,
//pixelScale, texelsPerPixel, texelSizeMin, levelAngle, blurSlope, blurMax and blurTexelScale

//texels along the side of the lightmap, which is square. The defaults are for the materials that are not set yet
override LIGHTMAP_SIZE: u32 = 2048u;
//the tiles the lightmap has room for
override TILE_COUNT: u32 = 65536u;
//texels along the side of a tile
override TILE_SIZE: u32 = 8u;

//no tile: not found, or the probe sequence is full of tiles that are in use
const TILE_NONE = 0xFFFFFFFFu;
//the texel size doubles per level. Keeps 1 << level in range
const LEVEL_MAX = 20u;
//added to the texel coordinates, so the ones left of or below 0 are u32 too. The top chart is the whole map, so it has
//no size and no clamp: a texel is any integer coordinate
const TEXEL_OFFSET = 0x40000000;

fn tileHash(x: u32) -> u32 {
	var a = x;
	a = (a ^ 61u) ^ (a >> 16u);
	a = a + (a << 3u);
	a = a ^ (a >> 4u);
	a = a * 0x27d4eb2du;
	a = a ^ (a >> 15u);
	return a;
}

//independent of tileHash, for the slot of getTileKey
fn tilePcg(x: u32) -> u32 {
	let state = x * 747796405u + 2891336453u;
	let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

//the most levels a grazing view angle adds, see getLevel
const LEVEL_ANGLE_MAX = 2.;

//the sigma of the tilt shift blur at p in reference pixels (see SkyVisibilityLightmap.getPixelScale), like tiltShift.wgsl: it grows with
//the distance to the horizontal center line of the screen. Behind the camera the most, a ray hit can be there
fn getBlurPixels(p: vec3<f32>) -> f32 {
	if(uniforms.blurSlope <= 0.){
		return 0.;
	}
	let clip = uniforms.viewProjection * vec4(p, 1.);
	var ndcY = 1e6;
	if(clip.w > 1e-6){
		ndcY = abs(clip.y / clip.w);
	}
	return min(uniforms.blurSlope * ndcY, uniforms.blurMax) / uniforms.pixelScale;
}

//the level of the texels at p with normal n, so there are texelsPerPixel texels per screen pixel, or a texel per
//blurTexelScale times the sigma of the tilt shift blur where that is more: it would blur the finer lighting away.
//Level 0 texels are texelSizeMin, each level doubles the size. Fractional, so the resolve can blend between 2 levels.
//levelAngle: see getLevel in virtualLightmap.wgsl
fn getLevel(p: vec3<f32>, n: vec3<f32>) -> f32 {
	let toCamera = uniforms.cameraPosition - p;
	let d = length(toCamera);
	let cosAngle = abs(dot(n, toCamera)) / max(d, 1e-9);
	let angleLevels = min(-log2(max(cosAngle, 1e-6)) * uniforms.levelAngle, LEVEL_ANGLE_MAX);
	let texelPixels = max(1. / uniforms.texelsPerPixel, getBlurPixels(p) * uniforms.blurTexelScale);
	let size = uniforms.pixelSize * d * texelPixels;
	return clamp(log2(size / uniforms.texelSizeMin) + angleLevels, 0., f32(LEVEL_MAX));
}

fn getTexelSize(level: u32) -> f32 {
	return uniforms.texelSizeMin * f32(1u << level);
}

//the texel of a chart position at a level, offset by TEXEL_OFFSET. A texel of a level is 2 x 2 texels of the level below it
fn getTexelAt(position: vec2<f32>, level: u32) -> vec2<i32> {
	return vec2<i32>(floor(position / getTexelSize(level))) + vec2(TEXEL_OFFSET);
}

fn getTexel(surface: Surface, level: u32) -> vec2<u32> {
	return vec2<u32>(getTexelAt(surface.position, level));
}

//the checksum and the first slot of a tile of a chart at a level. The checksum is never 0, which is an empty slot
fn getTileKey(surface: u32, level: u32, tile: vec2<u32>) -> vec2<u32> {
	let key = max(tileHash(tile.x ^ tileHash(tile.y ^ tileHash(surface ^ tileHash(level)))), 1u);
	let slot = tilePcg(tile.x + tilePcg(tile.y + tilePcg(surface + tilePcg(level))));
	return vec2(key, slot % TILE_COUNT);
}

//the index in lightingBuffer of a texel, in the tile at tileIndex
fn getTexelIndex(tileIndex: u32, texel: vec2<u32>) -> u32 {
	let t = texel % TILE_SIZE;
	return (tileIndex * TILE_SIZE + t.y) * TILE_SIZE + t.x;
}

//the sun visibility of a texel is this far after its sky visibility in lightingBuffer, when the lightmap has the sun
//(SkyVisibilityLightmap.setSun): the buffer is twice the size then, the second half the sun texels in the same layout
fn getSunTexelOffset() -> u32 {
	return LIGHTMAP_SIZE * LIGHTMAP_SIZE;
}

//the running average of the lighting (rgb) and the sample count it holds (a), as 4 x f16
fn unpackLighting(v: vec2<u32>) -> vec4<f32> {
	return vec4(unpack2x16float(v.x), unpack2x16float(v.y));
}

fn packLighting(l: vec4<f32>) -> vec2<u32> {
	return vec2(pack2x16float(l.rg), pack2x16float(l.ba));
}
