// isOccluded of skyVisibilityTrace.wgsl with the distance to the hit, for the water reflections: the color of what a
// ray hits is looked up on the screen at that point (see waterScreenReflection.wgsl). The nearest hit: the traversal
// goes on after a hit, through the boxes the ray enters before it. The first hit found (as isOccluded) was often a part
// of a building behind another part of it (their boxes overlap), whose point the screen doesn't show. Needs
// skyVisibilityTrace.wgsl

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

fn getRecordHitDistance(ro : vec3<f32>, rd : vec3<f32>, record : u32) -> f32 {
  let r0 = cityBuffer[record];
  if (bitcast<u32>(r0.w) == CITY_TREE) {
    return getTreeHitDistance(ro, rd, r0);
  }
  return getBuildingHitDistance(ro, rd, r0);
}

// the distance along rd (normalized) to the nearest building or tree, -1 when the ray reaches the sky
fn getOcclusionDistance(ro : vec3<f32>, rd : vec3<f32>) -> f32 {
  if (uniforms.cityTileCount == 0u) {
    return -1.;
  }
  let invRd = getInverseDirection(rd);
  let roInvRd = ro * invRd;
  var stack : array<u32, CITY_STACK_SIZE>;
  var stackSize = 0u;
  var node = uniforms.cityRoot;
  var nearest = 1e30;
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
      if (t >= 0.) {
        nearest = min(nearest, t);
      }
      visitA = false;
    }
    if (visitB && b >= CITY_LEAF) {
      let t = getRecordHitDistance(ro, rd, b & ~CITY_LEAF);
      if (t >= 0.) {
        nearest = min(nearest, t);
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
  return select(-1., nearest, nearest < 1e30);
}
