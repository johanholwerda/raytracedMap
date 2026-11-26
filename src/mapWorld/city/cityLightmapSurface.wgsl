//the charts of the city lightmap, see cityLayout.ts. No uniforms, so the gBuffer pass can include it too

//no surface: the sky, or a pixel nothing was drawn in
const SURFACE_NONE = 0u;
//the ground: world xz
const SURFACE_GROUND = 1u;
//the bit of the roofs of a height (world xz), with the height bucket in the bits below it. See getRoofChart in
//cityLayout.ts. Not the chart of the ground: a texel on the outline of a roof would average the open roof and the
//shaded ground next to its walls, and light the ground there
const SURFACE_ROOF = 0x20000000u;
//the bit of the wall atlas of a map tile, with the tile coordinates in the bits below it. See getWallSurfaceId in cityLayout.ts
const SURFACE_WALL = 0x40000000u;

//a point on a chart: the chart and where the point is on it, in world units
struct Surface {
	id : u32,
	position : vec2<f32>,
}

fn getGroundSurface(p: vec3<f32>) -> Surface {
	return Surface(SURFACE_GROUND, p.xz);
}

//the charts of the bridge decks (see buildBridges in worldGeometryCore.ts): world xz like the ground, but not the ground
//chart, where the deck, its sides and the water under it would share texels and average the open sky over the deck
//with the shade under it. Roof charts above the ones of getRoofChart (24 bits), per facing (up, down, sideways: the top,
//the bottom and the sides of 1 deck are over each other) and per BRIDGE_CHART_STEP of height (decks over each other)
const BRIDGE_CHART_BASE = 0x1000000u;
//about 2 m (a world unit is about 900 m, see MapProjection)
const BRIDGE_CHART_STEP = 2e-3;
fn getBridgeSurface(p: vec3<f32>, n: vec3<f32>) -> Surface {
	let facing = select(select(2u, 1u, n.y > .5), 3u, abs(n.y) <= .5);
	let level = u32(clamp(floor(p.y / BRIDGE_CHART_STEP) + 32768., 0., 65535.));
	return getRoofSurface(BRIDGE_CHART_BASE * facing + level, p);
}

//chart: the height bucket of the roof, see getRoofChart in cityLayout.ts
fn getRoofSurface(chart: u32, p: vec3<f32>) -> Surface {
	return Surface(SURFACE_ROOF | (chart & 0x1FFFFFFFu), p.xz);
}

//tile: the map tile coordinates, as the floats of the lightmap attribute
fn getWallSurfaceId(tile: vec2<f32>) -> u32 {
	let t = vec2<u32>(max(tile, vec2(0.)) + .5) & vec2(0x7FFFu);
	return SURFACE_WALL | (t.x << 15u) | t.y;
}

//the surface of a point of a building, from the interpolated lightmap attribute of its vertices: the atlas position (xy)
//and the map tile (zw) on a wall, the roof chart (x) and zw -1 on a roof. The 3 vertices of a roof triangle have the
//same chart, so the rounding undoes the error of the interpolation
fn getAttributeSurface(lightmap: vec4<f32>, p: vec3<f32>) -> Surface {
	if(lightmap.z < 0.){
		return getRoofSurface(u32(max(round(lightmap.x), 0.)), p);
	}
	return Surface(getWallSurfaceId(lightmap.zw), lightmap.xy);
}

//the surface of a pixel of the gBuffer (see mapGBufferPacking.wgsl, which this needs then): the wall atlas position
//stored on a wall, else world xz. world: the position the camera distance of the pixel gives. atlasStep: see
//MapGBuffer.atlasStep
fn getGBufferSurface(g: vec3<u32>, world: vec3<f32>, atlasStep: f32) -> Surface {
	if((g.x & SURFACE_WALL) != 0u){
		return Surface(g.x, vec2<f32>(getGBufferAtlas(g)) * atlasStep);
	}
	return Surface(g.x, world.xz);
}
