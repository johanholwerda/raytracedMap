// isOccluded of skyVisibilityTrace.wgsl with the distance to the hit, for the water reflections: the color of what a
// ray hits is looked up on the screen at that point (see waterScreenReflection.wgsl). The nearest hit: the traversal
// goes on after a hit, through the boxes the ray enters before it. The first hit found (as isOccluded) was often a part
// of a building behind another part of it (their boxes overlap), whose point the screen doesn't show. With the record
// it hits, whose normal there (getCityHitNormal) shades a hit the screen doesn't show (getUnseenReflectionColor of
// waterScreenReflection.wgsl). Needs skyVisibilityTrace.wgsl

// what a ray hits: t -1 when it reaches the sky, record: the building or tree record (a vec4 index of cityBuffer). Not
// the normal: kept from the trace through the lookups on the screen it made the deferred shading .1 ms slower at 4K
// with a ray per water pixel (registers), getCityHitNormal works it out only for the hits that need it
struct CityHit {
  t : f32,
  record : u32,
}

const CITY_HIT_NONE = CityHit(-1., 0u);

// where the ray enters the building of r0 (its walls, or its roof or floor plane inside the footprint), -1 when it
// misses: as hitsBuilding, the nearest of the walls instead of the first
fn getBuildingHitDistance(ro : vec3<f32>, rd : vec3<f32>, r0 : vec4<f32>) -> f32 {
  let base = r0.x;
  let top = r0.y;
  let edgeStart = bitcast<u32>(r0.z);
  let edgeCount = bitcast<u32>(r0.w);
  var tCap = -1.;
  if (rd.y < 0. && ro.y >= top) {
    tCap = (top - ro.y) / rd.y;
  } else if (rd.y > 0. && ro.y <= base) {
    tCap = (base - ro.y) / rd.y;
  }
  let testCap = tCap > MIN_DIST;
  let capPoint = ro.xz + rd.xz * tCap;
  var inside = false;
  var tWall = 1e30;
  for (var i = 0u; i < edgeCount; i++) {
    let e = cityBuffer[edgeStart + i];
    let a = e.xy;
    let b = e.zw;
    let d = b - a;
    let denom = cross2(rd.xz, d);
    if (abs(denom) > 1e-20) {
      let w = a - ro.xz;
      let t = cross2(w, d) / denom;
      let s = cross2(w, rd.xz) / denom;
      if (t > MIN_DIST && s >= 0. && s <= 1.) {
        let y = ro.y + rd.y * t;
        if (y >= base && y <= top) {
          tWall = min(tWall, t);
        }
      }
    }
    if (testCap && (a.y > capPoint.y) != (b.y > capPoint.y)) {
      let x = a.x + (capPoint.y - a.y) / (b.y - a.y) * d.x;
      if (capPoint.x < x) {
        inside = !inside;
      }
    }
  }
  // the cap comes before every wall: the ray crosses its plane before it is between the base and the top
  if (inside) {
    return tCap;
  }
  return select(-1., tWall, tWall < 1e30);
}

// where the ray enters the tree of r0 (its crown or trunk box), -1 when it misses: as hitsTree
fn getTreeHitDistance(ro : vec3<f32>, rd : vec3<f32>, r0 : vec4<f32>) -> f32 {
  let shape = bitcast<u32>(r0.z);
  let s0 = cityBuffer[shape];
  let s1 = cityBuffer[shape + 1u];
  let u = s0.zw;
  let v = vec2(-u.y, u.x);
  let w = ro.xz - s0.xy;
  let p = vec3(dot(w, u), ro.y, dot(w, v));
  let d = vec3(dot(rd.xz, u), rd.y, dot(rd.xz, v));
  let invD = getInverseDirection(d);
  let pInvD = p * invD;
  let crownBase = s1.x;
  let k = s1.y;
  let crown = intersectAABB(pInvD, invD, vec3(-1., crownBase, -1.), vec3(1., r0.y, 1.));
  let trunk = intersectAABB(pInvD, invD, vec3(-k, r0.x, -k), vec3(k, crownBase, k));
  var t = 1e30;
  if (crown.y > MIN_DIST && crown.x <= crown.y) {
    t = max(crown.x, 0.);
  }
  if (trunk.y > MIN_DIST && trunk.x <= trunk.y) {
    t = min(t, max(trunk.x, 0.));
  }
  return select(-1., t, t < 1e30);
}

// The normal of the building of r0 at p, a point on it where the ray along rd enters it (getBuildingHitDistance): of
// its roof or floor plane, or of the wall nearest to p, whichever p is closer to. Against the ray
fn getBuildingNormal(p : vec3<f32>, rd : vec3<f32>, r0 : vec4<f32>) -> vec3<f32> {
  let edgeStart = bitcast<u32>(r0.z);
  let edgeCount = bitcast<u32>(r0.w);
  var nearest = min(abs(p.y - r0.y), abs(p.y - r0.x));
  var normal = vec3(0., -sign(rd.y), 0.);
  for (var i = 0u; i < edgeCount; i++) {
    let e = cityBuffer[edgeStart + i];
    let d = e.zw - e.xy;
    let w = p.xz - e.xy;
    let distance = length(w - d * clamp(dot(w, d) / max(dot(d, d), 1e-20), 0., 1.));
    if (distance < nearest) {
      nearest = distance;
      let n = normalize(vec3(d.y, 0., -d.x));
      normal = select(n, -n, dot(n, rd) > 0.);
    }
  }
  return normal;
}

// The normal of the tree of r0 where the ray enters it at distance t (getTreeHitDistance): of the face of the crown or
// trunk box it enters, on the axis it enters last. Against the ray
fn getTreeNormal(ro : vec3<f32>, rd : vec3<f32>, t : f32, r0 : vec4<f32>) -> vec3<f32> {
  let shape = bitcast<u32>(r0.z);
  let s0 = cityBuffer[shape];
  let s1 = cityBuffer[shape + 1u];
  let u = s0.zw;
  let v = vec2(-u.y, u.x);
  let w = ro.xz - s0.xy;
  let p = vec3(dot(w, u), ro.y, dot(w, v));
  let d = vec3(dot(rd.xz, u), rd.y, dot(rd.xz, v));
  let invD = getInverseDirection(d);
  let crownBase = s1.x;
  let k = s1.y;
  // the box the hit is in: the crown when the point is at its height
  let crown = ro.y + rd.y * t >= crownBase;
  let bMin = select(vec3(-k, r0.x, -k), vec3(-1., crownBase, -1.), crown);
  let bMax = select(vec3(k, crownBase, k), vec3(1., r0.y, 1.), crown);
  let tNear = min((bMin - p) * invD, (bMax - p) * invD);
  let axis = select(select(vec3(0., 0., 1.), vec3(0., 1., 0.), tNear.y >= tNear.z), vec3(1., 0., 0.),
    tNear.x >= tNear.y && tNear.x >= tNear.z);
  let local = -sign(d) * axis;
  // the x and z axes of the boxes are u and v divided by their length squared: back along their directions
  let uDir = normalize(u);
  return vec3(local.x * uDir.x - local.z * uDir.y, local.y, local.x * uDir.y + local.z * uDir.x);
}

fn getRecordHitDistance(ro : vec3<f32>, rd : vec3<f32>, record : u32) -> f32 {
  let r0 = cityBuffer[record];
  if (bitcast<u32>(r0.w) == CITY_TREE) {
    return getTreeHitDistance(ro, rd, r0);
  }
  return getBuildingHitDistance(ro, rd, r0);
}

// the nearest building or tree along rd (normalized): t the distance, -1 when the ray reaches the sky
fn getOcclusionHit(ro : vec3<f32>, rd : vec3<f32>) -> CityHit {
  if (uniforms.cityTileCount == 0u) {
    return CITY_HIT_NONE;
  }
  let invRd = getInverseDirection(rd);
  let roInvRd = ro * invRd;
  var stack : array<u32, CITY_STACK_SIZE>;
  var stackSize = 0u;
  var node = uniforms.cityRoot;
  var nearest = 1e30;
  var nearestRecord = 0u;
  loop {
    let n0 = cityBuffer[node];
    let n1 = cityBuffer[node + 1u];
    let n2 = cityBuffer[node + 2u];
    let n3 = cityBuffer[node + 3u];
    let a = bitcast<u32>(n0.w);
    let b = bitcast<u32>(n1.w);
    let tA = intersectAABB(roInvRd, invRd, n0.xyz, n1.xyz);
    let tB = intersectAABB(roInvRd, invRd, n2.xyz, n3.xyz);
    // only the boxes the ray enters before the nearest hit so far
    var visitA = tA.y >= max(tA.x, 0.) && tA.x < nearest;
    var visitB = tB.y >= max(tB.x, 0.) && tB.x < nearest;
    if (visitA && a >= CITY_LEAF) {
      let t = getRecordHitDistance(ro, rd, a & ~CITY_LEAF);
      if (t >= 0. && t < nearest) {
        nearest = t;
        nearestRecord = a & ~CITY_LEAF;
      }
      visitA = false;
    }
    if (visitB && b >= CITY_LEAF) {
      let t = getRecordHitDistance(ro, rd, b & ~CITY_LEAF);
      if (t >= 0. && t < nearest) {
        nearest = t;
        nearestRecord = b & ~CITY_LEAF;
      }
      visitB = false;
    }
    if (visitA && visitB) {
      let aIsNear = tA.x <= tB.x;
      node = select(b, a, aIsNear);
      if (stackSize < CITY_STACK_SIZE) {
        stack[stackSize] = select(a, b, aIsNear);
        stackSize++;
      }
      continue;
    }
    if (visitA || visitB) {
      node = select(b, a, visitA);
      continue;
    }
    if (stackSize == 0u) {
      break;
    }
    stackSize--;
    node = stack[stackSize];
  }
  if (nearest >= 1e30) {
    return CITY_HIT_NONE;
  }
  return CityHit(nearest, nearestRecord);
}

// whether the hit of getOcclusionHit is a tree
fn isCityHitTree(hit : CityHit) -> bool {
  return bitcast<u32>(cityBuffer[hit.record].w) == CITY_TREE;
}

// the normal of the face the ray from ro along rd enters at the hit of getOcclusionHit, against the ray
fn getCityHitNormal(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> vec3<f32> {
  let r0 = cityBuffer[hit.record];
  if (bitcast<u32>(r0.w) == CITY_TREE) {
    return getTreeNormal(ro, rd, hit.t, r0);
  }
  return getBuildingNormal(ro + rd * hit.t, rd, r0);
}
