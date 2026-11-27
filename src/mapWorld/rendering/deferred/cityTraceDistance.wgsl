// isOccluded of skyVisibilityTrace.wgsl with the distance to the hit, for the water reflections: what a ray hits is
// lit by the lightmap at that point (see waterReflectionColor.wgsl). The nearest hit: the traversal goes on after a
// hit, through the boxes the ray enters before it. The first hit found (as isOccluded) was often a part of a building
// behind another part of it (their boxes overlap). With the record it hits, whose face there (getCityHitSurface) is on
// a chart of the lightmap: the atlas words of the CityTraceBlock (cityLayout.ts). Needs skyVisibilityTrace.wgsl and
// cityLightmapSurface.wgsl

// what a ray hits: t -1 when it reaches the sky, record: the building or tree record (a vec4 index of cityBuffer), tie:
// the record of a wall that coincides with the one hit (see addCityHit), else record too. Not
// the normal: kept from the trace through the lookups on the screen it made the deferred shading .1 ms slower at 4K
// with a ray per water pixel (registers), getCityHitSurface works out the face only for the hits that need it
struct CityHit {
  t : f32,
  record : u32,
  tie : u32,
}

const CITY_HIT_NONE = CityHit(-1., 0u, 0u);

// an atlas word of a wall that has no place in the wall atlas, see CityTraceBlock in cityLayout.ts
const CITY_ATLAS_NONE = 0xFFFFFFFFu;
// the face of CityFace that is the roof of a building (the top of the crown of a tree), and a face that is on no chart
// of the lightmap: a floor, the underside of a crown, the top of a trunk (inside the crown)
const CITY_FACE_ROOF = 0xFFFFFFFEu;
const CITY_FACE_NONE = 0xFFFFFFFFu;

// The face of a building or tree a ray enters: its normal (against the ray), which face it is (a wall: the index of its
// atlas word from the first one of the record, see CityTraceBlock; or CITY_FACE_ROOF or CITY_FACE_NONE), and on a wall
// how far the point is from where the wall starts, in world units: its u from the u of the wall's atlas word
struct CityFace {
  normal : vec3<f32>,
  face : u32,
  along : f32,
}

// a word of cityBuffer, at a word index (4 per vec4)
fn getCityWord(index : u32) -> u32 {
  return bitcast<u32>(cityBuffer[index >> 2u][index & 3u]);
}

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

// The face of the building of r0 at p, a point on it where the ray along rd enters it (getBuildingHitDistance): its
// roof or floor plane, or the wall nearest to p, whichever p is closer to. A wall is its edge, whose atlas word has the
// same index
fn getBuildingFace(p : vec3<f32>, rd : vec3<f32>, r0 : vec4<f32>) -> CityFace {
  let edgeStart = bitcast<u32>(r0.z);
  let edgeCount = bitcast<u32>(r0.w);
  var nearest = min(abs(p.y - r0.y), abs(p.y - r0.x));
  // a cap: the roof when the ray comes down on it
  var face = CityFace(vec3(0., -sign(rd.y), 0.), select(CITY_FACE_NONE, CITY_FACE_ROOF, rd.y < 0.), 0.);
  for (var i = 0u; i < edgeCount; i++) {
    let e = cityBuffer[edgeStart + i];
    let d = e.zw - e.xy;
    let w = p.xz - e.xy;
    let distance = length(w - d * clamp(dot(w, d) / max(dot(d, d), 1e-20), 0., 1.));
    if (distance < nearest) {
      nearest = distance;
      let n = normalize(vec3(d.y, 0., -d.x));
      face = CityFace(select(n, -n, dot(n, rd) > 0.), i, length(w));
    }
  }
  return face;
}

// The face of the tree of r0 where the ray enters it at distance t (getTreeHitDistance): the face of the crown or trunk
// box it enters, on the axis it enters last. Its walls are the edges of square in worldTrees.ts, which go around the
// corners (-1, 1), (1, 1), (1, -1), (-1, -1) of the box (x on u, z on v, in half widths of the crown): +z, +x, -z, -x.
// Their atlas words are those of the trunk (0 to 3), then of the crown (4 to 7)
fn getTreeFace(ro : vec3<f32>, rd : vec3<f32>, t : f32, r0 : vec4<f32>) -> CityFace {
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
  let normal = vec3(local.x * uDir.x - local.z * uDir.y, local.y, local.x * uDir.y + local.z * uDir.x);
  // the point in the box, and the half width of the box in world units (u has the length 1 / the crown half width)
  let q = p + d * t;
  let halfWidth = select(k, 1., crown);
  let scale = 1. / length(u);
  var face = CityFace(normal, CITY_FACE_NONE, 0.);
  if (axis.y > 0.) {
    if (crown && local.y > 0.) {
      face.face = CITY_FACE_ROOF;
    }
    return face;
  }
  if (axis.x > 0.) {
    face.face = select(3u, 1u, local.x > 0.);
    face.along = select(q.z + halfWidth, halfWidth - q.z, local.x > 0.) * scale;
  } else {
    face.face = select(2u, 0u, local.z > 0.);
    face.along = select(halfWidth - q.x, q.x + halfWidth, local.z > 0.) * scale;
  }
  face.face += select(0u, 4u, crown);
  return face;
}

fn getRecordHitDistance(ro : vec3<f32>, rd : vec3<f32>, record : u32) -> f32 {
  let r0 = cityBuffer[record];
  if (bitcast<u32>(r0.w) == CITY_TREE) {
    return getTreeHitDistance(ro, rd, r0);
  }
  return getBuildingHitDistance(ro, rd, r0);
}

// 2 hits this close (a part of their distance) are on the same wall: walls of 2 parts that coincide, see addCityHit
const CITY_HIT_TIE = 1.002;

// a hit of record at t (-1: none) added to the nearest so far. Walls of 2 parts that coincide are both drawn, and which
// one the gBuffer shows at a pixel is up to the rounding of their depths: the hit keeps the other one as its tie, whose
// atlas position the lightmap may have the texels of instead (getReflectionHitColor tries both). The trace kept the
// first one it found only: the wall of the other part, at another place in the wall atlas, read no texels, or the
// coarser texels around that place, of other walls
fn addCityHit(nearest : CityHit, t : f32, record : u32) -> CityHit {
  if (t < 0.) {
    return nearest;
  }
  if (t * CITY_HIT_TIE < nearest.t) {
    return CityHit(t, record, record);
  }
  if (t <= nearest.t * CITY_HIT_TIE) {
    if (t < nearest.t) {
      return CityHit(t, record, nearest.record);
    }
    return CityHit(nearest.t, nearest.record, record);
  }
  return nearest;
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
  var nearest = CityHit(1e30, 0u, 0u);
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
    // (and the ones it enters at a tie with it, see addCityHit)
    let reach = nearest.t * CITY_HIT_TIE;
    var visitA = tA.y >= max(tA.x, 0.) && tA.x <= reach;
    var visitB = tB.y >= max(tB.x, 0.) && tB.x <= reach;
    if (visitA && a >= CITY_LEAF) {
      nearest = addCityHit(nearest, getRecordHitDistance(ro, rd, a & ~CITY_LEAF), a & ~CITY_LEAF);
      visitA = false;
    }
    if (visitB && b >= CITY_LEAF) {
      nearest = addCityHit(nearest, getRecordHitDistance(ro, rd, b & ~CITY_LEAF), b & ~CITY_LEAF);
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
  if (nearest.t >= 1e30) {
    return CITY_HIT_NONE;
  }
  return nearest;
}

// whether the hit of getOcclusionHit is a tree
fn isCityHitTree(hit : CityHit) -> bool {
  return bitcast<u32>(cityBuffer[hit.record].w) == CITY_TREE;
}

// the face the ray from ro along rd enters at the hit of getOcclusionHit
fn getCityHitFace(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> CityFace {
  let r0 = cityBuffer[hit.record];
  if (bitcast<u32>(r0.w) == CITY_TREE) {
    return getTreeFace(ro, rd, hit.t, r0);
  }
  return getBuildingFace(ro + rd * hit.t, rd, r0);
}

// where the hit of getOcclusionHit is on the charts of the lightmap (the surfaces of getAttributeSurface, the same the
// gBuffer has there), and the normal of its face. valid false: a face on no chart (CITY_FACE_NONE, a wall without a
// place in the atlas)
struct CityHitSurface {
  surface : Surface,
  normal : vec3<f32>,
  valid : bool,
}

// The same of the wall that coincides with the one hit (hit.tie), where there is one
fn getCityHitTieSurface(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> CityHitSurface {
  return getCityHitSurface(ro, rd, CityHit(hit.t, hit.tie, hit.tie));
}

fn getCityHitSurface(ro : vec3<f32>, rd : vec3<f32>, hit : CityHit) -> CityHitSurface {
  let face = getCityHitFace(ro, rd, hit);
  let p = ro + rd * hit.t;
  // the second vec4 of the record: its first atlas word, its roof chart, its header and its first wall bottom, see
  // CityTraceBlock
  let r1 = bitcast<vec4<u32>>(cityBuffer[hit.record + 1u]);
  var result = CityHitSurface(Surface(SURFACE_NONE, vec2(0.)), face.normal, false);
  if (face.face == CITY_FACE_ROOF) {
    result.surface = getRoofSurface(r1.y, p);
    result.valid = true;
  } else if (face.face != CITY_FACE_NONE) {
    let word = getCityWord(r1.x + face.face);
    // the step of the words and the offset of their v, and the surface of the wall atlas of the tile
    let header = cityBuffer[r1.z];
    // how far above the base of the record the wall is drawn: below it the wall is inside another part, and has no
    // place in the atlas (a v there is in another chunk)
    let bottomIndex = r1.w + face.face;
    let bottom = f32((getCityWord(bottomIndex >> 1u) >> ((bottomIndex & 1u) * 16u)) & 0xFFFFu) * header.y;
    if (word != CITY_ATLAS_NONE && p.y >= cityBuffer[hit.record].x + bottom - header.y) {
      let start = vec2(f32(word & 0xFFFFu), f32(word >> 16u)) * header.y + vec2(0., header.z);
      result.surface = Surface(bitcast<u32>(header.x), start + vec2(face.along, p.y));
      result.valid = true;
    }
  }
  return result;
}
