import { hypot2 } from 'webGPU-framework/renderer/utils/MathUtils';
import type { BuildingPolygon } from './cityLayout';

/**
 * The walls of the building parts of a tile that are inside other parts, which no camera outside the buildings sees:
 * OpenMapTiles has a tower as a stack of parts (a podium 0 - 199 m, crowns 200 - 227 m, 200 - 242 m, ...) and touching
 * buildings as separate footprints, so in Midtown Manhattan some 45% of the wall area is inside another part.
 *
 * Per wall (the edge from ring vertex k to the next one) the height from which it can be seen: the lowest height above
 * which the space just outside it (eps outward) is not filled by the other parts, over the whole length of the wall.
 * Below it the outside of the wall is in other parts over its whole length. Coplanar walls of 2 parts (the space just
 * inside the wall is in the other part, the space outside isn't) z-fight: they count as hidden for one of them, see
 * coplanarCovers. polygons[i].top when it is hidden altogether, NaN for an edge of 0 length (no wall).
 *
 * Only the polygons before `count` take part (the buildings, not the trees). eps: the offset of the test, in world
 * units: walls of touching parts that are that close count as touching
 */
export function findWallBottoms(polygons: BuildingPolygon[], count: number, eps: number): Float32Array[] {
  const out = polygons.map(p => new Float32Array(p.coords.length / 2).fill(p.base));
  if (count < 2) {
    fillZeroEdges(polygons, out, count);
    return out;
  }

  // the bounds of the polygons, and a grid over them for the candidates of a wall
  const bounds = new Float64Array(count * 4);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < count; i++) {
    const c = polygons[i].coords;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = 0; k < c.length; k += 2) {
      x0 = Math.min(x0, c[k]); x1 = Math.max(x1, c[k]);
      y0 = Math.min(y0, c[k + 1]); y1 = Math.max(y1, c[k + 1]);
    }
    bounds.set([x0, y0, x1, y1], i * 4);
    minX = Math.min(minX, x0); maxX = Math.max(maxX, x1);
    minY = Math.min(minY, y0); maxY = Math.max(maxY, y1);
  }
  const grid = Math.max(1, Math.min(256, Math.ceil(Math.sqrt(count))));
  const cellX = Math.max(maxX - minX, 1e-12) / grid, cellY = Math.max(maxY - minY, 1e-12) / grid;
  const cellOf = (v: number, min: number, size: number) => Math.min(grid - 1, Math.max(0, Math.floor((v - min) / size)));
  const cells: number[][] = Array.from({ length: grid * grid }, () => []);
  for (let i = 0; i < count; i++) {
    const cx0 = cellOf(bounds[i * 4], minX, cellX), cx1 = cellOf(bounds[i * 4 + 2], minX, cellX);
    const cy0 = cellOf(bounds[i * 4 + 1], minY, cellY), cy1 = cellOf(bounds[i * 4 + 3], minY, cellY);
    for (let y = cy0; y <= cy1; y++) for (let x = cx0; x <= cx1; x++) cells[y * grid + x].push(i);
  }

  // per candidate the stamp of the wall it was last tested for, so a polygon in several cells is tested once
  const stamp = new Int32Array(count).fill(-1);
  let wallIndex = 0;
  // the covered pieces of the wall: t0, t1 along it (0 to 1), h0, h1
  // Of 2 coplanar walls (of parts p and q) only the bottom of one can go up (a wall is drawn from its bottom to its top):
  // the one with the higher base, of equal bases the later one (the parts are sorted tallest first). In Midtown the
  // parts of a tower often all start at the ground and share facades: the walls of the shorter ones are dropped there.
  const coplanarCovers = (pi: number, qi: number): boolean => {
    const pb = polygons[pi].base, qb = polygons[qi].base;
    return qb < pb - eps || (qb <= pb + eps && qi < pi);
  };

  for (let pi = 0; pi < count; pi++) {
    const p = polygons[pi];
    const c = p.coords;
    for (let r = 0; r < p.ringStarts.length; r++) {
      const s = p.ringStarts[r], e = p.ringEnds[r];
      const n = e - s;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        const dx = c[ib * 2] - c[ia * 2], dy = c[ib * 2 + 1] - c[ia * 2 + 1];
        const len = hypot2(dx, dy);
        if (len === 0) continue;
        // outward: the outer ring is CCW and the holes CW, the solid is on the left of every edge
        const ox = dy / len * eps, oy = -dx / len * eps;
        const ax = c[ia * 2] + ox, ay = c[ia * 2 + 1] + oy, bx = ax + dx, by = ay + dy;
        // The same eps / 2 inward: a part that has it but not the point outside has a wall of its own along this one
        // (within eps / 2, facing the same way), which z-fights with it. It covers this wall when coplanarCovers. Less
        // than eps: of 2 parallel walls further apart the one behind is inside the other part (the test outward drops
        // it), this one must stay.
        const ix = ax - 1.5 * ox, iy = ay - 1.5 * oy;
        // the box of both test segments
        const sx0 = Math.min(ax, bx, ix, ix + dx), sx1 = Math.max(ax, bx, ix, ix + dx);
        const sy0 = Math.min(ay, by, iy, iy + dy), sy1 = Math.max(ay, by, iy, iy + dy);
        // most walls are seen from their base on: when no part is just outside (or inside, coplanar) the middle of the
        // wall at its base, the intervals are not needed
        const mx = ax + dx * .5, my = ay + dy * .5, imx = ix + dx * .5, imy = iy + dy * .5;
        const midCell = cellOf(my, minY, cellY) * grid + cellOf(mx, minX, cellX);
        const innerCell = cellOf(imy, minY, cellY) * grid + cellOf(imx, minX, cellX);
        let midCovered = false;
        // the cell of the middle, then the one of the inner middle when it is another
        for (let side = 0; side < 2 && !midCovered; side++) {
          if (side === 1 && innerCell === midCell) break;
          const list = cells[side === 0 ? midCell : innerCell];
          for (let k = 0; k < list.length; k++) {
            const qi = list[k];
            const q = polygons[qi];
            if (qi === pi || q.base > p.base + eps || q.top <= p.base) continue;
            if (pointInBounds(bounds, qi, mx, my)) {
              point[0] = mx; point[1] = my;
              if (pointInPolygon(q)) { midCovered = true; break; }
            }
            if (coplanarCovers(pi, qi) && pointInBounds(bounds, qi, imx, imy)) {
              point[0] = imx; point[1] = imy;
              if (pointInPolygon(q)) { midCovered = true; break; }
            }
          }
        }
        if (!midCovered) continue;
        rectFloats = 0;
        wallIndex++;
        const cx0 = cellOf(sx0, minX, cellX), cx1 = cellOf(sx1, minX, cellX);
        const cy0 = cellOf(sy0, minY, cellY), cy1 = cellOf(sy1, minY, cellY);
        for (let gy = cy0; gy <= cy1; gy++) for (let gx = cx0; gx <= cx1; gx++) {
          const list = cells[gy * grid + gx];
          for (let k = 0; k < list.length; k++) {
            const qi = list[k];
            if (qi === pi || stamp[qi] === wallIndex) continue;
            stamp[qi] = wallIndex;
            const q = polygons[qi];
            if (q.top <= p.base || q.base >= p.top) continue;
            if (bounds[qi * 4] > sx1 || bounds[qi * 4 + 2] < sx0 || bounds[qi * 4 + 1] > sy1 || bounds[qi * 4 + 3] < sy0) continue;
            segment[0] = ax; segment[1] = ay; segment[2] = dx; segment[3] = dy;
            segment[4] = Math.max(q.base, p.base); segment[5] = Math.min(q.top, p.top);
            insideIntervals(q);
            if (coplanarCovers(pi, qi)) {
              segment[0] = ix; segment[1] = iy;
              insideIntervals(q);
            }
          }
        }
        if (rectFloats > 0) {
          cover[0] = p.base; cover[1] = p.top; cover[2] = eps / len; cover[3] = eps;
          coveredBottom();
          out[pi][ia] = cover[4];
        } else out[pi][ia] = p.base;
      }
    }
  }
  fillZeroEdges(polygons, out, count);
  return out;
}

/** NaN for the edges of 0 length of the buildings */
function fillZeroEdges(polygons: BuildingPolygon[], out: Float32Array[], count: number): void {
  for (let pi = 0; pi < count; pi++) {
    const { coords: c, ringStarts, ringEnds } = polygons[pi];
    for (let r = 0; r < ringStarts.length; r++) {
      const s = ringStarts[r], n = ringEnds[r] - s;
      for (let j = 0; j < n; j++) {
        const ia = s + j, ib = s + ((j + 1) % n);
        if (c[ia * 2] === c[ib * 2] && c[ia * 2 + 1] === c[ib * 2 + 1]) out[pi][ia] = NaN;
      }
    }
  }
}

/*
The numbers the functions below take and give, set before a call and read after it instead of passed and returned: a
number passed to (or returned by) a function that isn't inlined becomes an object on the heap, and these run per wall
per part near it (5 MB of garbage per tile).
-point: x, y of pointInPolygon
-segment: a.x, a.y, d.x, d.y, h0, h1 of insideIntervals
-cover: base, top, gap, hGap of coveredBottom, and its result
 */
const point = new Float64Array(2);
const segment = new Float64Array(6);
const cover = new Float64Array(5);

/*
What they work in, kept from wall to wall and from tile to tile, in typed arrays with a count next to them: an array
that is set to length 0 lets go of its storage, the next push allocates it again (3 MB per tile for the few numbers
here).
-rects: the covered pieces of the wall of findWallBottoms, t0, t1 along it (0 to 1), h0, h1 each; rectFloats of them
-crossings: where a wall crosses the edges of a part, as t along the wall (insideIntervals)
-coveredOrder: the rects at a height of coveredBottom
 */
let rects = new Float64Array(256);
let rectFloats = 0;
let crossings = new Float64Array(64);
let coveredOrder = new Int32Array(64);

/** A rect after the others. */
function addRect(t0: number, t1: number): void {
  if (rectFloats + 4 > rects.length) {
    const grown = new Float64Array(rects.length * 2);
    grown.set(rects);
    rects = grown;
  }
  rects[rectFloats] = t0;
  rects[rectFloats + 1] = t1;
  rects[rectFloats + 2] = segment[4];
  rects[rectFloats + 3] = segment[5];
  rectFloats += 4;
}

/**
 * The intervals of t (0 to 1) of the segment a + t d that are inside polygon q (even-odd over its rings), added to rects
 * as (t0, t1, h0, h1): a, d, h0 and h1 from segment. Without a callback per interval and without Array.sort (which
 * allocates on every call, also for the 2 to 4 numbers here): this runs per wall per part near it.
 */
function insideIntervals(q: BuildingPolygon): void {
  const ax = segment[0], ay = segment[1], dx = segment[2], dy = segment[3];
  const c = q.coords;
  let ts = crossings;
  ts[0] = 0;
  ts[1] = 1;
  let count = 2;
  for (let r = 0; r < q.ringStarts.length; r++) {
    const s = q.ringStarts[r], n = q.ringEnds[r] - s;
    for (let j = 0; j < n; j++) {
      const ia = s + j, ib = s + ((j + 1) % n);
      const cx = c[ia * 2], cy = c[ia * 2 + 1];
      const ex = c[ib * 2] - cx, ey = c[ib * 2 + 1] - cy;
      const den = dx * ey - dy * ex;
      if (den === 0) continue;
      const fx = cx - ax, fy = cy - ay;
      const t = (fx * ey - fy * ex) / den, u = (fx * dy - fy * dx) / den;
      if (t > 0 && t < 1 && u >= 0 && u <= 1) {
        if (count === ts.length) {
          ts = new Float64Array(count * 2);
          ts.set(crossings);
          crossings = ts;
        }
        ts[count++] = t;
      }
    }
  }
  // ascending, by insertion: a few numbers
  for (let k = 1; k < count; k++) {
    const t = ts[k];
    let j = k - 1;
    for (; j >= 0 && ts[j] > t; j--) ts[j + 1] = ts[j];
    ts[j + 1] = t;
  }
  let open = -1;
  for (let k = 0; k + 1 < count; k++) {
    if (ts[k + 1] <= ts[k]) continue;
    const m = (ts[k] + ts[k + 1]) * .5;
    point[0] = ax + dx * m; point[1] = ay + dy * m;
    const inside = pointInPolygon(q);
    if (inside && open < 0) open = ts[k];
    if (!inside && open >= 0) { addRect(open, ts[k]); open = -1; }
  }
  if (open >= 0) addRect(open, 1);
}

function pointInBounds(bounds: Float64Array, i: number, x: number, y: number): boolean {
  return bounds[i * 4] <= x && bounds[i * 4 + 2] >= x && bounds[i * 4 + 1] <= y && bounds[i * 4 + 3] >= y;
}

/** Whether point (x, y) is inside q, even-odd over its rings. */
function pointInPolygon(q: BuildingPolygon): boolean {
  const x = point[0], y = point[1];
  const c = q.coords;
  let inside = false;
  for (let r = 0; r < q.ringStarts.length; r++) {
    const s = q.ringStarts[r], e = q.ringEnds[r];
    for (let i = s, j = e - 1; i < e; j = i++) {
      const xi = c[i * 2], yi = c[i * 2 + 1], xj = c[j * 2], yj = c[j * 2 + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
  }
  return inside;
}

/**
 * The lowest height h from base on such that [0, 1] x [base, h] is covered by the rects (t0, t1, h0, h1), gaps along t
 * up to gap and steps in height up to hGap ignored: going up from base, at every height the rects there cover all of t until the lowest top of them.
 * base, top, gap and hGap from cover, the result into cover[4]
 */
function coveredBottom(): void {
  const base = cover[0], top = cover[1], gap = cover[2], hGap = cover[3];
  const n = rectFloats / 4;
  if (coveredOrder.length < n) coveredOrder = new Int32Array(n * 2);
  const order = coveredOrder;
  let h = base;
  for (;;) {
    let count = 0;
    let next = Infinity;
    for (let i = 0; i < n; i++) {
      if (rects[i * 4 + 2] <= h + hGap && rects[i * 4 + 3] > h) {
        order[count++] = i;
        next = Math.min(next, rects[i * 4 + 3]);
      }
    }
    if (count === 0) {
      cover[4] = h;
      return;
    }
    // by their start, equal ones in their order (as a stable sort has them), by insertion: a few rects
    for (let k = 1; k < count; k++) {
      const i = order[k], t = rects[i * 4];
      let j = k - 1;
      for (; j >= 0 && rects[order[j] * 4] > t; j--) order[j + 1] = order[j];
      order[j + 1] = i;
    }
    let reach = 0;
    for (let k = 0; k < count; k++) {
      const i = order[k];
      if (rects[i * 4] > reach + gap) break;
      reach = Math.max(reach, rects[i * 4 + 1]);
    }
    if (reach < 1 - gap) {
      cover[4] = h;
      return;
    }
    h = next;
    if (h >= top) {
      cover[4] = top;
      return;
    }
  }
}
