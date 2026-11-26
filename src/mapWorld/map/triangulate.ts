import earcut from 'earcut';

/*
earcut with the triangles into an array that is kept from polygon to polygon, and without its linked list of objects
for the polygons where it finds an ear at every step on its first try: then its run is fixed, the ears come one after
the other from where it starts, and that run is played here on arrays of indices. The ground layers and the roofs of a
tile are thousands of polygons, and earcut made an object per point of each (Node: 9 fields) and an array of the
triangles: some 4 MB of garbage per tile of a city. The rest (holes, more than MAX_POINTS points, a step where earcut
would go on to the next point) goes to earcut itself: the same triangles, in the same order, either way.

What earcut 3 does for a polygon without holes of at most 80 points (80 * dim coords: above it, it orders the points on a
z-order curve to find them faster, see isEarHashed): it links the points in a ring, forward when signedArea is positive
(else backward), and drops the last one when it equals the first (a closed ring). It starts at the last one: the ear
there is cut when area(prev, ear, next) < 0 and isEar (no other point of the ring in its triangle), then it goes on at
next. When the first try at every step is an ear, that is all of it.
 */

/** The triangles of triangulate: 3 point indices each, in data (from 0 to length). */
export class Triangles {
  public data = new Uint32Array(8192);
  public length = 0;

  public push3(a: number, b: number, c: number): void {
    if (this.length + 3 > this.data.length) {
      const grown = new Uint32Array(this.data.length * 2);
      grown.set(this.data);
      this.data = grown;
    }
    const d = this.data, o = this.length;
    d[o] = a; d[o + 1] = b; d[o + 2] = c;
    this.length = o + 3;
  }
}

// the most points of the polygons played here: earcut's limit without the z-order curve (data.length > 80 * dim)
const MAX_POINTS = 80;
// the ring of the polygon being played: per node its point and its neighbours
const ringPoint = new Int32Array(MAX_POINTS);
const ringPrev = new Int32Array(MAX_POINTS);
const ringNext = new Int32Array(MAX_POINTS);

/**
 * The triangles of earcut(data, holes, 2) for the first `length` numbers of data (x, y per point; holes: the point index
 * of each hole) into out (from 0): the same triangles in the same order.
 */
export function triangulate(data: Float64Array, length: number, holes: number[] | null, out: Triangles): void {
  out.length = 0;
  if ((!holes || holes.length === 0) && length <= MAX_POINTS * 2 && playEars(data, length, out)) return;
  out.length = 0;
  const tris = earcut(length === data.length ? data : data.subarray(0, length), holes && holes.length ? holes : undefined, 2);
  for (let t = 0; t + 2 < tris.length; t += 3) out.push3(tris[t], tris[t + 1], tris[t + 2]);
}

/**
 * earcut's run on a polygon without holes, as long as its first try at each step is an ear: false as soon as it isn't
 * (out is then incomplete). See the comment at the top.
 */
function playEars(data: Float64Array, length: number, out: Triangles): boolean {
  const n = length >> 1;
  if (n === 0) return true;
  // linkedList(data, 0, length, 2, clockwise true): the points in their order when signedArea > 0, else reversed
  let sum = 0;
  for (let i = 0, j = length - 2; i < length; i += 2) {
    sum += (data[j] - data[i]) * (data[i + 1] + data[j + 1]);
    j = i;
  }
  const forward = sum > 0;
  for (let k = 0; k < n; k++) {
    ringPoint[k] = forward ? k : n - 1 - k;
    ringPrev[k] = k === 0 ? n - 1 : k - 1;
    ringNext[k] = k === n - 1 ? 0 : k + 1;
  }
  let ear = n - 1;
  // the last point equal to the next one (the first): it goes
  if (equal(data, ear, ringNext[ear])) {
    remove(ear);
    ear = ringNext[ear];
  }
  // under 3 points: no triangles
  while (ringPrev[ear] !== ringNext[ear]) {
    const a = ringPrev[ear], c = ringNext[ear];
    if (!(area(data, a, ear, c) < 0) || !isEar(data, a, ear, c)) return false;
    out.push3(ringPoint[a], ringPoint[ear], ringPoint[c]);
    remove(ear);
    ear = c;
  }
  return true;
}

function remove(node: number): void {
  ringPrev[ringNext[node]] = ringPrev[node];
  ringNext[ringPrev[node]] = ringNext[node];
}

/** earcut's equals of 2 nodes. */
function equal(data: Float64Array, p: number, q: number): boolean {
  const i = ringPoint[p] * 2, j = ringPoint[q] * 2;
  return data[i] === data[j] && data[i + 1] === data[j + 1];
}

/** earcut's area of 3 nodes: (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y). */
function area(data: Float64Array, p: number, q: number, r: number): number {
  const pi = ringPoint[p] * 2, qi = ringPoint[q] * 2, ri = ringPoint[r] * 2;
  return (data[qi + 1] - data[pi + 1]) * (data[ri] - data[qi]) - (data[qi] - data[pi]) * (data[ri + 1] - data[qi + 1]);
}

/** earcut's isEar: no other node of the ring in the triangle (a, b, c) (with earcut's tests, in the order of the ring). */
function isEar(data: Float64Array, a: number, b: number, c: number): boolean {
  const ai = ringPoint[a] * 2, bi = ringPoint[b] * 2, ci = ringPoint[c] * 2;
  const ax = data[ai], ay = data[ai + 1], bx = data[bi], by = data[bi + 1], cx = data[ci], cy = data[ci + 1];
  const x0 = Math.min(ax, bx, cx), y0 = Math.min(ay, by, cy), x1 = Math.max(ax, bx, cx), y1 = Math.max(ay, by, cy);
  for (let p = ringNext[c]; p !== a; p = ringNext[p]) {
    const pi = ringPoint[p] * 2;
    const px = data[pi], py = data[pi + 1];
    if (px >= x0 && px <= x1 && py >= y0 && py <= y1 && !(ax === px && ay === py) &&
      (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
      (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
      (bx - px) * (cy - py) >= (cx - px) * (by - py) &&
      area(data, ringPrev[p], p, ringNext[p]) >= 0) return false;
  }
  return true;
}
