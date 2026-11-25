/*
Removes the false mountains of the elevation tiles: in some regions the terrarium tiles (see ElevationService) have
blobs and streaks hundreds of meters high on the sea and along the coast. Around Reykjavik for one: north of 60
degrees the source is ArcticDEM, a surface model from satellite stereo images, whose matching fails on the water. They
are 300 to 900 m high on water at 0 m, so the sea there looked like a field of needles that cast shadows across the
city at sunset.

What gives them away is how they rise: straight up from the ground around them, by up to 50 times the width of a pixel
in 1 pixel (over 800 m in 17 m). Real terrain doesn't, not even at the DEM's resolution: the steepest steps in the data
are along the walls of Yosemite (El Capitan and Half Dome: up to 13 times the pixel width, 394 m in 30 m), the fjords,
sea cliffs, karst towers and the Alps stay under 10. So:
 - a seed is a pixel more than CLIFF_PX pixel widths (and at least CLIFF_MIN_M) above a neighbour, its base the lowest
   such neighbour;
 - its region is the pixels connected to it that are more than ABOVE_BASE_M above that base: the blob, with the ramp
   some of them have on one side down to the ground. A region of more than 1/16 of the tile is a real massif with a
   real wall (El Capitan's region is the whole of it) and is kept;
 - that leaves the lower part of the blob, which still showed as low mounds on the water: so the region grows over all
   the pixels above the base around it, when they are at most about twice the region (a blob on the sea), else (it
   touches land, or an island: about 65 m above the sea in ArcticDEM, see ElevationService) only over the pixels that
   go down steeply from it, the foot of the ramp but not the land it ends on;
 - those pixels are filled in from the heights around them.
Measured on z12 tiles of Reykjavik (30 tiles around it: blobs of up to 1010 m gone, about 23000 pixels filled) and of
steep places that must stay as they are (Geirangerfjord, Lysefjord, Kjerag, Yosemite, the Eiger, the Matterhorn, the
Faroe cliffs, Vestmannaeyjar, Na Pali, Table Mountain, the Grand Canyon, Lofoten, Sugarloaf, Ha Long Bay, Railay,
Meteora, the Dolomites, the Cliffs of Moher, St Kilda, Hornstrandir, Nuuk, Svalbard: not a pixel changed).

What it can't see: the smaller blobs that don't rise steeply enough to tell from a real islet or sea stack (around
Reykjavik 32 of up to 250 m high and 60 pixels wide are left), and the smooth bumps of 100 to 200 m the same noise left
on land in places, which look like hills. And a blob that a tile edge cuts is handled per tile, so a sliver of it can
stay along the edge.
 */

/** A seed rises more than this many pixel widths in 1 pixel (13 is the steepest real wall seen, see above)... */
const CLIFF_PX = 15;
/** ...and at least this much (m): where the pixels are small, near the poles. */
const CLIFF_MIN_M = 200;
/** The region of a seed: the pixels connected to it this much (m) above its base. */
const ABOVE_BASE_M = 150;
/** The whole blob of a region: the pixels connected to it this much (m) above its base... */
const ABOVE_BASE_MIN_M = 2;
/** ...when they are at most this many times the region, plus BLOB_EXTRA pixels. */
const BLOB_GROWTH = 2;
const BLOB_EXTRA = 64;
/** Else the foot of a region: the pixels around it that go down by more than this many pixel widths per pixel. */
const FOOT_SLOPE = .5;

/** The width (m) of a pixel of a web mercator tile row y (at its center) at zoom z, of size pixels. */
export function demPixelMeters(y: number, z: number, size: number): number {
  const n = 1 << z;
  const lat = Math.atan(Math.sinh(Math.PI * (1 - 2 * (y + .5) / n)));
  return 40075016.7 / (n * size) * Math.cos(lat);
}

/**
 * Fills in the false mountains of a square elevation tile in place, see above. Heights in m, row after row, size
 * pixels per side, pixelM the width of a pixel (demPixelMeters). Returns the number of pixels filled in.
 */
export function removeDemSpikes(heights: Float32Array, size: number, pixelM: number): number {
  const count = size * size;
  const cliff = Math.max(CLIFF_MIN_M, CLIFF_PX * pixelM);
  // the seeds and their bases
  const base = new Float32Array(count).fill(Infinity);
  const seeds: number[] = [];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const k = y * size + x;
      const top = heights[k] - cliff;
      let b = Infinity;
      if (x > 0 && heights[k - 1] < top) b = Math.min(b, heights[k - 1]);
      if (x < size - 1 && heights[k + 1] < top) b = Math.min(b, heights[k + 1]);
      if (y > 0 && heights[k - size] < top) b = Math.min(b, heights[k - size]);
      if (y < size - 1 && heights[k + size] < top) b = Math.min(b, heights[k + size]);
      if (b !== Infinity) {
        base[k] = b;
        seeds.push(k);
      }
    }
  }
  if (seeds.length === 0) return 0;

  // the regions, from the seed with the lowest base up: a pixel is in 1 region at most, a kept one included, so the
  // seeds on a kept massif don't grow it again
  seeds.sort((a, b) => base[a] - base[b]);
  const inRegion = new Uint8Array(count);
  // 1: filled in. Later the pixels whose height is known (0) or not yet (1) while filling in.
  const bad = new Uint8Array(count);
  const queue = new Int32Array(count);
  // the pixels of the whole blob of a region (see below) by the region's number, so it needn't be cleared per region
  const inBlob = new Int32Array(count);
  const blob = new Int32Array(count);
  const maxRegion = count >> 4;
  const footStep = FOOT_SLOPE * pixelM;
  let filled = 0, regions = 0;
  for (const seed of seeds) {
    if (inRegion[seed]) continue;
    const b = base[seed];
    const above = b + ABOVE_BASE_M;
    let head = 0, tail = 0;
    queue[tail++] = seed;
    inRegion[seed] = 1;
    while (head < tail) {
      const k = queue[head++];
      const x = k % size;
      if (x > 0 && !inRegion[k - 1] && heights[k - 1] > above) { inRegion[k - 1] = 1; queue[tail++] = k - 1; }
      if (x < size - 1 && !inRegion[k + 1] && heights[k + 1] > above) { inRegion[k + 1] = 1; queue[tail++] = k + 1; }
      if (k >= size && !inRegion[k - size] && heights[k - size] > above) { inRegion[k - size] = 1; queue[tail++] = k - size; }
      if (k < count - size && !inRegion[k + size] && heights[k + size] > above) { inRegion[k + size] = 1; queue[tail++] = k + size; }
    }
    if (tail > maxRegion) continue;

    // The region leaves the lower part of the blob, up to ABOVE_BASE_M high: the whole of what is above the base
    // around it, when that is at most BLOB_GROWTH times as large (a blob on the sea). Else (it reaches the land, or an
    // island) only the pixels that go down steeply from the region: the foot of a ramp, not the land at its foot.
    const id = ++regions;
    const cap = Math.min(maxRegion, Math.floor(BLOB_GROWTH * tail) + BLOB_EXTRA);
    const grow = (foot: boolean): number => {
      let n = 0;
      for (let i = 0; i < tail; i++) {
        blob[n++] = queue[i];
        inBlob[queue[i]] = foot ? -id : id;
      }
      const mark = foot ? -id : id;
      const take = (k: number, from: number): boolean => inBlob[k] !== mark && !bad[k] && heights[k] > b + ABOVE_BASE_MIN_M &&
        (!foot || heights[from] - heights[k] > footStep);
      for (let i = 0; i < n; i++) {
        const k = blob[i];
        const x = k % size;
        if (x > 0 && take(k - 1, k)) { inBlob[k - 1] = mark; blob[n++] = k - 1; }
        if (x < size - 1 && take(k + 1, k)) { inBlob[k + 1] = mark; blob[n++] = k + 1; }
        if (k >= size && take(k - size, k)) { inBlob[k - size] = mark; blob[n++] = k - size; }
        if (k < count - size && take(k + size, k)) { inBlob[k + size] = mark; blob[n++] = k + size; }
        if (!foot && n > cap) return -1;
      }
      return n;
    };
    let n = grow(false);
    if (n < 0) n = grow(true);
    for (let i = 0; i < n; i++) {
      bad[blob[i]] = 1;
      inRegion[blob[i]] = 1;
    }
    filled += n;
  }
  if (filled === 0) return 0;

  // fill in from the outside in, ring after ring: a pixel gets the mean of its neighbours known before its ring
  let ring: number[] = [];
  const queued = new Uint8Array(count);
  const knownNeighbour = (k: number): boolean => {
    const x = k % size;
    return (x > 0 && !bad[k - 1]) || (x < size - 1 && !bad[k + 1]) || (k >= size && !bad[k - size]) ||
      (k < count - size && !bad[k + size]);
  };
  for (let k = 0; k < count; k++) {
    if (bad[k] && knownNeighbour(k)) {
      ring.push(k);
      queued[k] = 1;
    }
  }
  const values = new Float32Array(count);
  while (ring.length > 0) {
    for (const k of ring) {
      const x = k % size;
      let sum = 0, n = 0;
      if (x > 0 && !bad[k - 1]) { sum += heights[k - 1]; n++; }
      if (x < size - 1 && !bad[k + 1]) { sum += heights[k + 1]; n++; }
      if (k >= size && !bad[k - size]) { sum += heights[k - size]; n++; }
      if (k < count - size && !bad[k + size]) { sum += heights[k + size]; n++; }
      values[k] = sum / n;
    }
    const next: number[] = [];
    for (const k of ring) {
      heights[k] = values[k];
      bad[k] = 0;
    }
    for (const k of ring) {
      const x = k % size;
      if (x > 0 && bad[k - 1] && !queued[k - 1]) { queued[k - 1] = 1; next.push(k - 1); }
      if (x < size - 1 && bad[k + 1] && !queued[k + 1]) { queued[k + 1] = 1; next.push(k + 1); }
      if (k >= size && bad[k - size] && !queued[k - size]) { queued[k - size] = 1; next.push(k - size); }
      if (k < count - size && bad[k + size] && !queued[k + size]) { queued[k + size] = 1; next.push(k + size); }
    }
    ring = next;
  }
  return filled;
}
