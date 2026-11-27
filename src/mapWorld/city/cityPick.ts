import { CITY_LEAF, CITY_RECORD_VEC4S, CITY_TREE } from './cityLayout';
import type { CityTraceBlock } from './cityLayout';

/*
The closest hit of a ray on the buildings and trees of a tile, on the CPU: the building picking (see CityScene.pick). The
same shapes as isOccluded of skyVisibilityTrace.wgsl reads (see CityTraceBlock), but the closest hit instead of any hit:
a building is the closest of its walls and its roof (or floor, from below), a tree the closest of its crown and trunk.
The block is the one of the worker, its indices from the start of the block (CityScene moves only its GPU copy)
 */

//the distances along the ray where it enters (near) and leaves (far) the box, near > far when it misses
const span = { near: 0, far: 0 };

function intersectBox(
  ox: number, oy: number, oz: number, ix: number, iy: number, iz: number,
  minX: number, minY: number, minZ: number, maxX: number, maxY: number, maxZ: number,
): void {
  const x0 = (minX - ox) * ix, x1 = (maxX - ox) * ix;
  const y0 = (minY - oy) * iy, y1 = (maxY - oy) * iy;
  const z0 = (minZ - oz) * iz, z1 = (maxZ - oz) * iz;
  span.near = Math.max(Math.min(x0, x1), Math.min(y0, y1), Math.min(z0, z1));
  span.far = Math.min(Math.max(x0, x1), Math.max(y0, y1), Math.max(z0, z1));
}

//large instead of infinite for axis aligned rays, as getInverseDirection in skyVisibilityTrace.wgsl
function inverse(d: number): number {
  return (d >= 0 ? 1 : -1) / Math.max(Math.abs(d), 1e-12);
}

//the distance to the first wall, roof or floor of the building of record r0 (a vec4 index), Infinity when it misses
function intersectBuilding(f: Float32Array, u: Uint32Array, r0: number, ro: number[], rd: number[]): number {
  const base = f[r0 * 4], top = f[r0 * 4 + 1];
  const edgeStart = u[r0 * 4 + 2], edgeCount = u[r0 * 4 + 3];
  let tCap = -1;
  if (rd[1] < 0 && ro[1] >= top) tCap = (top - ro[1]) / rd[1];
  else if (rd[1] > 0 && ro[1] <= base) tCap = (base - ro[1]) / rd[1];
  const capX = ro[0] + rd[0] * tCap, capZ = ro[2] + rd[2] * tCap;
  let inside = false;
  let best = Infinity;
  for (let i = 0; i < edgeCount; i++) {
    const e = (edgeStart + i) * 4;
    const ax = f[e], az = f[e + 1], bx = f[e + 2], bz = f[e + 3];
    const dx = bx - ax, dz = bz - az;
    const denom = rd[0] * dz - rd[2] * dx;
    if (Math.abs(denom) > 1e-20) {
      const wx = ax - ro[0], wz = az - ro[2];
      const t = (wx * dz - wz * dx) / denom;
      const s = (wx * rd[2] - wz * rd[0]) / denom;
      if (t > 0 && t < best && s >= 0 && s <= 1) {
        const y = ro[1] + rd[1] * t;
        if (y >= base && y <= top) best = t;
      }
    }
    if (tCap > 0 && (az > capZ) !== (bz > capZ)) {
      const x = ax + (capZ - az) / (bz - az) * dx;
      if (capX < x) inside = !inside;
    }
  }
  return inside ? Math.min(best, tCap) : best;
}

//the distance to the crown or trunk of the tree of record r0, Infinity when it misses. See hitsTree in skyVisibilityTrace.wgsl
function intersectTree(f: Float32Array, u: Uint32Array, r0: number, ro: number[], rd: number[]): number {
  const s = u[r0 * 4 + 2] * 4;
  const cx = f[s], cz = f[s + 1], ux = f[s + 2], uz = f[s + 3];
  const crownBase = f[s + 4], k = f[s + 5];
  //the frame of the crown: x along u, z along u turned by 90 degrees, y stays
  const vx = -uz, vz = ux;
  const wx = ro[0] - cx, wz = ro[2] - cz;
  const px = wx * ux + wz * uz, pz = wx * vx + wz * vz;
  const ix = inverse(rd[0] * ux + rd[2] * uz), iy = inverse(rd[1]), iz = inverse(rd[0] * vx + rd[2] * vz);
  let best = Infinity;
  intersectBox(px, ro[1], pz, ix, iy, iz, -1, crownBase, -1, 1, f[r0 * 4 + 1], 1);
  if (span.far > 0 && span.near <= span.far) best = Math.max(span.near, 0);
  intersectBox(px, ro[1], pz, ix, iy, iz, -k, f[r0 * 4], -k, k, crownBase, k);
  if (span.far > 0 && span.near <= span.far) best = Math.min(best, Math.max(span.near, 0));
  return best;
}

/**
 * The closest building or tree of the block the ray hits before tMax: its distance and the record index (0 for the
 * first building record, buildingCount and up for the trees), or null. rd doesn't have to be normalized, t is in its
 * length.
 */
export function intersectTraceBlock(block: CityTraceBlock, ro: number[], rd: number[], tMax: number): { t: number; record: number } | null {
  const u = block.words;
  const f = new Float32Array(u.buffer, u.byteOffset, u.length);
  const recordStart = block.nodeCount * 4;
  const ix = inverse(rd[0]), iy = inverse(rd[1]), iz = inverse(rd[2]);
  let best = tMax;
  let bestRecord = -1;
  //vec4 indices of nodes, the root first
  const stack: number[] = [0];
  while (stack.length > 0) {
    //its first word
    const node = stack.pop()! * 4;
    let nearChild = -1, farChild = -1, nearT = 0;
    //aMin.xyz, a, aMax.xyz, b, bMin.xyz, 0, bMax.xyz, 0: the boxes 8 words apart, the children 4
    for (let side = 0; side < 2; side++) {
      const o = node + side * 8;
      const child = u[node + 3 + side * 4];
      intersectBox(ro[0], ro[1], ro[2], ix, iy, iz, f[o], f[o + 1], f[o + 2], f[o + 4], f[o + 5], f[o + 6]);
      const near = Math.max(span.near, 0);
      if (span.far < near || near >= best) continue;
      if (child >= CITY_LEAF) {
        const record = (child & ~CITY_LEAF) >>> 0;
        const t = u[record * 4 + 3] === CITY_TREE ? intersectTree(f, u, record, ro, rd) : intersectBuilding(f, u, record, ro, rd);
        if (t < best) {
          best = t;
          bestRecord = (record - recordStart) / CITY_RECORD_VEC4S;
        }
      } else if (nearChild < 0) {
        nearChild = child;
        nearT = near;
      } else if (near < nearT) {
        farChild = nearChild;
        nearChild = child;
      } else {
        farChild = child;
      }
    }
    //the near child on top, it is visited first and makes best shorter for the far one
    if (farChild >= 0) stack.push(farChild);
    if (nearChild >= 0) stack.push(nearChild);
  }
  return bestRecord < 0 ? null : { t: best, record: bestRecord };
}
