//any hit rays against the buildings of the map for the sky visibility lightmap: whether a ray reaches the sky. The city
//buffer of CityScene.ts (see CityTraceBlock in cityLayout.ts), without the ground plane (the rays go up, see
//getPathVisibility in updateSkyVisibilityLightmap.wgsl), the hit point, its normal and its surface: the first building
//the ray hits ends it. Needs uniforms.cityTileCount and cityRoot, and cityBuffer. Distances in world units

//the offset of a ray from the surface it starts on, about 2 cm (a world unit is about 900 m, see MapProjection)
const MIN_DIST: f32 = 2e-5;
//a child of a node that is a record, and the bit of the last word of r0 that a tree record has set (the sign of its
//-k), see CityTraceBlock in cityLayout.ts
const CITY_LEAF = 0x80000000u;
const CITY_TREE = 0x80000000u;
//the top level BVH (the tiles) and the BVH of a tile (a dense one is about 16 deep) on 1 stack. Only 32 is safe (the
//most depth, see BLOCK_DEPTH_MAX and TOP_DEPTH_MAX in cityLayout.ts): a push past the end drops the far child, which
//misses its buildings. On real rays of Amsterdam and New York 8 missed 2-3 hits in a million, 16 none, and on an S23
//(Adreno) 8 instead of 32 took the frame rate from 47 to 60 fps. A define, not an override (a function scope array
//can't have an override size): CityScene.traceDefines, 8 on mobile and 16 elsewhere, fixed when the shaders are made
#ifndef CITY_STACK_SIZE
#define CITY_STACK_SIZE 16
#endif
//the grid of the tiles of CityScene (see writeGrid there): after the top level BVH (TLAS_NODES_MAX nodes of 4 vec4s).
//2 header vec4s, then 2 per cell
const CITY_GRID_START = 4096u;
//the changes of CityScene: after the grid (GRID_VEC4S), 2 vec4s each
const CITY_CHANGE_START = 6146u;
//no node: a ray that can't hit anything, see getCityStart
const CITY_NONE = 0xFFFFFFFFu;

//the nodes, records and edges (see CityTraceBlock) are in cityBuffer: array<vec4<f32>>, which the including shader
//declares at a binding of its own (updateSkyVisibilityLightmap.wgsl, deferredShading.wgsl)

//large instead of infinite for axis aligned rays, so a ray in the plane of a box side gives no NaN
fn getInverseDirection(rd: vec3<f32>) -> vec3<f32> {
	return select(vec3(-1.), vec3(1.), rd >= vec3(0.)) / max(abs(rd), vec3(1e-8));
}

//the distances to where the ray enters (x) and leaves (y) the box, x > y when it misses. roInvRd: ro * invRd
fn intersectAABB(roInvRd: vec3<f32>, invRd: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
	let t0 = fma(boxMin, invRd, -roInvRd);
	let t1 = fma(boxMax, invRd, -roInvRd);
	let tNear = min(t0, t1);
	let tFar = max(t0, t1);
	return vec2(max(max(tNear.x, tNear.y), tNear.z), min(min(tFar.x, tFar.y), tFar.z));
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 {
	return a.x * b.y - a.y * b.x;
}

/*
whether the ray hits a building: its footprint (with its holes) extruded from its base to its flat roof, see
intersectBuilding in cityTrace.wgsl. A wall the ray crosses between the base and the roof ends it in the loop over the
edges. Else the point where the ray crosses the plane of the roof (from above) or of the floor (from below) is inside the
footprint, even-odd, which takes all edges.
r0: the building record (base, top, edgeStart, edgeCount)
 */
fn hitsBuilding(ro: vec3<f32>, rd: vec3<f32>, r0: vec4<f32>) -> bool {
	let base = r0.x;
	let top = r0.y;
	let edgeStart = bitcast<u32>(r0.z);
	let edgeCount = bitcast<u32>(r0.w);

	var tCap = -1.;
	if(rd.y < 0. && ro.y >= top){
		tCap = (top - ro.y) / rd.y;
	}else if(rd.y > 0. && ro.y <= base){
		tCap = (base - ro.y) / rd.y;
	}
	let testCap = tCap > MIN_DIST;
	let capPoint = ro.xz + rd.xz * tCap;
	var inside = false;
	for(var i = 0u; i < edgeCount; i++){
		let e = cityBuffer[edgeStart + i];
		let a = e.xy;
		let b = e.zw;
		let d = b - a;
		let denom = cross2(rd.xz, d);
		if(abs(denom) > 1e-20){
			let w = a - ro.xz;
			let t = cross2(w, d) / denom;
			let s = cross2(w, rd.xz) / denom;
			if(t > MIN_DIST && s >= 0. && s <= 1.){
				let y = ro.y + rd.y * t;
				if(y >= base && y <= top){
					return true;
				}
			}
		}
		if(testCap && (a.y > capPoint.y) != (b.y > capPoint.y)){
			let x = a.x + (capPoint.y - a.y) / (b.y - a.y) * d.x;
			if(capPoint.x < x){
				inside = !inside;
			}
		}
	}
	return inside;
}

/*
whether the ray hits a tree: its trunk or its crown, 2 square boxes around the same vertical axis, turned by the same
angle (see TraceTree in cityLayout.ts). In the frame of the crown, where it is [-1, 1] in x and z (the trunk [-k, k]) and
y stays: the distances along the ray are the same as in the world. Like hitsBuilding a ray that leaves a box from its
side or top, or misses it, doesn't hit it: the distance where it leaves has to be over MIN_DIST.
r0, r1: the tree record (base, top, crown base, -k), (center.xz, u.xz)
 */
fn hitsTree(ro: vec3<f32>, rd: vec3<f32>, r0: vec4<f32>, r1: vec4<f32>) -> bool {
	let u = r1.zw;
	let v = vec2(-u.y, u.x);
	let w = ro.xz - r1.xy;
	let p = vec3(dot(w, u), ro.y, dot(w, v));
	let d = vec3(dot(rd.xz, u), rd.y, dot(rd.xz, v));
	let invD = getInverseDirection(d);
	let pInvD = p * invD;
	let crownBase = r0.z;
	let k = -r0.w;
	let crown = intersectAABB(pInvD, invD, vec3(-1., crownBase, -1.), vec3(1., r0.y, 1.));
	let trunk = intersectAABB(pInvD, invD, vec3(-k, r0.x, -k), vec3(k, crownBase, k));
	return (crown.y > MIN_DIST && crown.x <= crown.y) || (trunk.y > MIN_DIST && trunk.x <= trunk.y);
}

//whether the ray hits the building or tree of a record
fn hitsRecord(ro: vec3<f32>, rd: vec3<f32>, record: u32) -> bool {
	let r0 = cityBuffer[record];
	if(bitcast<u32>(r0.w) >= CITY_TREE){
		return hitsTree(ro, rd, r0, cityBuffer[record + 1u]);
	}
	return hitsBuilding(ro, rd, r0);
}

//The node a ray from ro along rd starts at (invRd: getInverseDirection): the root of the top level BVH, or the root of
//the BVH of 1 tile for a ray that can only hit its buildings, CITY_NONE where that is no tile. That is a ray that goes
//up and is above the highest building of the city before it leaves the part of the cell of the grid (the map tile it
//starts over) that no other tile reaches into, see CityScene.writeGrid. It saves those rays the top level BVH: 85% of
//the water reflection rays of a view over the IJ in Amsterdam (17 tiles, the highest building 120 m), which made the
//shading with a ray per water pixel .1 ms faster at 4K (1.31 to 1.23 ms). The trace pass of the lightmap, whose paths go
//all ways over the hemisphere, took as long with it as without
fn getCityStart(ro: vec3<f32>, rd: vec3<f32>, invRd: vec3<f32>) -> u32 {
	let h0 = cityBuffer[CITY_GRID_START];
	let h1 = cityBuffer[CITY_GRID_START + 1u];
	let size = vec2<i32>(bitcast<vec2<u32>>(h1.xy));
	let cell = vec2<i32>(floor((ro.xz - h0.xy) * h0.zw));
	if(rd.y <= 0. || any(cell < vec2(0)) || any(cell >= size)){
		return uniforms.cityRoot;
	}
	let index = CITY_GRID_START + 2u + u32(cell.y * size.x + cell.x) * 2u;
	let alone = cityBuffer[index];
	//where the ray is above the highest building
	let end = ro.xz + rd.xz * (max(h1.z - ro.y, 0.) * invRd.y);
	if(any(min(ro.xz, end) < alone.xy) || any(max(ro.xz, end) > alone.zw)){
		return uniforms.cityRoot;
	}
	return bitcast<u32>(cityBuffer[index + 1u].x);
}

//whether the ray hits a building or tree, through the BVH of the tiles and of the buildings in a tile (see CityScene). Any hit
//ends it, so there is no closest distance to cull the boxes with. The near child is still visited first: the buildings
//nearby hide the most sky
fn isOccluded(ro: vec3<f32>, rd: vec3<f32>) -> bool {
	if(uniforms.cityTileCount == 0u){
		return false;
	}
	let invRd = getInverseDirection(rd);
	let roInvRd = ro * invRd;
	var stack: array<u32, CITY_STACK_SIZE>;
	var stackSize = 0u;
	var node = getCityStart(ro, rd, invRd);
	if(node == CITY_NONE){
		return false;
	}
	loop {
		let n0 = cityBuffer[node];
		let n1 = cityBuffer[node + 1u];
		let n2 = cityBuffer[node + 2u];
		let n3 = cityBuffer[node + 3u];
		let a = bitcast<u32>(n0.w);
		let b = bitcast<u32>(n1.w);
		let tA = intersectAABB(roInvRd, invRd, n0.xyz, n1.xyz);
		let tB = intersectAABB(roInvRd, invRd, n2.xyz, n3.xyz);
		var visitA = tA.y >= max(tA.x, 0.);
		var visitB = tB.y >= max(tB.x, 0.);
		if(visitA && a >= CITY_LEAF){
			if(hitsRecord(ro, rd, a & ~CITY_LEAF)){
				return true;
			}
			visitA = false;
		}
		if(visitB && b >= CITY_LEAF){
			if(hitsRecord(ro, rd, b & ~CITY_LEAF)){
				return true;
			}
			visitB = false;
		}
		if(visitA && visitB){
			let aIsNear = tA.x <= tB.x;
			node = select(b, a, aIsNear);
			if(stackSize < CITY_STACK_SIZE){
				stack[stackSize] = select(a, b, aIsNear);
				stackSize++;
			}
			continue;
		}
		if(visitA || visitB){
			node = select(b, a, visitA);
			continue;
		}
		if(stackSize == 0u){
			break;
		}
		stackSize--;
		node = stack[stackSize];
	}
	return false;
}
