import { geoidHeight } from './geoid';

/*
North of 60 degrees the elevation tiles (see ElevationService) are ArcticDEM in places: Iceland, the Faroes, Greenland,
Svalbard, the Kola peninsula, ... Its heights are above the WGS84 ellipsoid, not above the sea (the geoid): the land
there is as much too high as the geoid is above the ellipsoid, 66 m in Iceland, 57 m on the Faroes, 25 to 33 m in
Greenland and on Svalbard. Where ArcticDEM has no data (the water, gaps) the tiles have a coarse source instead (GMTED,
ETOPO1: above the sea), so the coasts and lake shores of that land were walls of 66 m, the harbour of Reykjavik a pit.

What tells the sources apart is the z10 tile: its source north of 60 degrees is the coarse one everywhere (ArcticDEM
starts at z11). So per pixel of a z12 tile, d = its height - the height of the z10 tile there (bilinear):
 - the coarse source (filler) has d of about 0: the same data at another resolution (Thingvallavatn: 83.0 at z10, z11
   and z12). Within FILLER_M;
 - ArcticDEM has d of about the geoid height plus 14 m (the coarse source is lower than the real height by that much
   in Iceland), on the flat ground. Where the ground is steep, d is whatever the smoothing of the z10 tile makes of the
   slopes, + or - 50 m and more on the walls of a fjord: only the flat ground (FLAT_SLOPE) decides;
 - the other sources of the tiles there (Norway, Sweden, Finland, Vestmannaeyjar, Hornstrandir, ...) are above the sea
   and have d of about 0, up to 20 m where they have the forests and the buildings in them.
So a tile is ArcticDEM when the median of d over its flat ground that isn't filler is above half the geoid height plus
MARGIN_M, and its pixels that aren't filler get the geoid height off (EGM96, see geoid.ts). Per tile, not per part of
it: the coarse source is too far off in places for less (on the Faroes the median per 64 x 64 pixels is anything from
22 to 160 m, the threshold 39 m, so the parts of a tile came out differently, with steps of 57 m between them; per
tile 48 to 106 m). Measured: the airports of Reykjavik (15 m), Keflavik (52 m) and Akureyri (2 m), Hallgrimskirkja
(41 m) and Myvatn (277 m) were 63 to 65 m too high, are within 3 m now (12, 49, 0, 39 and 272 m). Not a pixel changed in
Geiranger, Romsdalen, the Sognefjord, Lofoten, Tromso, Trondheim, Kebnekaise and Helsinki; Iceland is corrected but for
Vestmannaeyjar and Hornstrandir (another source).

Only where the geoid is at least MIN_GEOID_M: Iceland, the Faroes, the south and east of Greenland. How far the coarse
source is off differs per region (Svalbard 5 m too high, Iceland 14 m too low, Murmansk 18 m, Narvik 30 m), so where
the geoid is about 30 m (Svalbard, Nuuk, Scandinavia) ArcticDEM and a coarse source that is off look the same: Narvik
(not ArcticDEM, its neighbours have about 0) has a median of 30 m, the threshold there is 26 m. Of the 184 tiles with
land north of 60 degrees where the geoid is 40 m or more (every half degree of lat, 1 of lng) all of Norway's came out
not ArcticDEM, Iceland's ArcticDEM but for 3 (Hornstrandir and 2 more of another source).

What it can't: where the sources meet inside a tile, part of it is corrected wrongly. Svalbard, the west of Greenland,
the Kola peninsula, Alaska and Arctic Canada keep their offset (17 to 33 m, negative in Canada). The lakes keep the
height of the coarse source (Thingvallavatn 83 m, real 101 m), below the corrected land around them.
 */

/** The zoom of the coarse tiles: the coarse source everywhere north of 60 degrees. */
export const COARSE_ZOOM = 10;
/** Only north of this latitude (ArcticDEM)... */
const MIN_LAT = 60;
/** ...where the geoid is at least this high (m): lower, ArcticDEM and a coarse source that is off can't be told apart. */
const MIN_GEOID_M = 40;
/** A pixel within this (m) of the coarse tile is of the coarse source. */
const FILLER_M = 3;
/**
 * Smaller patches of such pixels are ArcticDEM (on a slope, by chance), and so are the patches whose pixels are further
 * off on average (m): the coarse source is within 0.6 m (the lakes 0 to 0.14 m, the sea 0.03 m), ArcticDEM by chance
 * 1 to 2 m (on the hills of the Faroes, patches of 100 pixels: they stood 57 m above the corrected land around them).
 */
const MIN_FILLER = 16;
const FILLER_MEAN_M = .75;
/** The flat ground: the slope of the coarse tile below this (m per m). */
const FLAT_SLOPE = .08;
/** The pixels of flat ground the tile needs to decide on them. */
const MIN_FLAT = 64;
/** ArcticDEM: the median of d is above half the geoid height plus this (m). */
const MARGIN_M = 10;

/** The lat of the north edge of tile row y at zoom z, in degrees. */
function tileLat(y: number, z: number): number {
  return Math.atan(Math.sinh(Math.PI * (1 - 2 * y / (1 << z)))) * 180 / Math.PI;
}

function tileLng(x: number, z: number): number {
  return x / (1 << z) * 360 - 180;
}

/** Whether the elevation tile (x, y at zoom z) can be ArcticDEM: then correctArcticDem needs its coarse tile. */
export function mayBeArcticDem(x: number, y: number, z: number): boolean {
  if (tileLat(y, z) < MIN_LAT) return false;
  return geoidHeight(tileLat(y + .5, z), tileLng(x + .5, z)) >= MIN_GEOID_M;
}

/**
 * Takes the geoid height off the ArcticDEM pixels of a square elevation tile, in place, see above. heights: m, size x
 * size, row after row, of tile x, y at zoom z. coarse: the tile at COARSE_ZOOM it is in, the same size. Returns the
 * number of pixels corrected.
 */
export function correctArcticDem(heights: Float32Array, coarse: Float32Array, size: number, x: number, y: number, z: number): number {
  const count = size * size;
  const scale = 1 << (z - COARSE_ZOOM);
  const step = size / scale;
  const ox = (x % scale) * step, oy = (y % scale) * step;

  // the geoid height per pixel, bilinear between the corners of the tile (it changes little over a tile, and so the
  // edges of 2 neighbours have the same)
  const latN = tileLat(y, z), latS = tileLat(y + 1, z);
  const lngW = tileLng(x, z), lngE = tileLng(x + 1, z);
  const gNW = geoidHeight(latN, lngW), gNE = geoidHeight(latN, lngE);
  const gSW = geoidHeight(latS, lngW), gSE = geoidHeight(latS, lngE);
  const geoidAt = (i: number, j: number): number => {
    const u = (i + .5) / size, v = (j + .5) / size;
    return (gNW * (1 - u) + gNE * u) * (1 - v) + (gSW * (1 - u) + gSE * u) * v;
  };

  // the coarse tile at the pixel centers (bilinear, clamped to its edge) and d
  const up = new Float32Array(count);
  const at = new Int32Array(size), frac = new Float32Array(size);
  const prepare = (o: number, out: Int32Array, f: Float32Array) => {
    for (let i = 0; i < size; i++) {
      const p = Math.min(Math.max(o + (i + .5) / scale - .5, 0), size - 1);
      out[i] = Math.min(Math.floor(p), size - 2);
      f[i] = p - out[i];
    }
  };
  const atY = new Int32Array(size), fracY = new Float32Array(size);
  prepare(ox, at, frac);
  prepare(oy, atY, fracY);
  for (let j = 0; j < size; j++) {
    const r0 = atY[j] * size, r1 = r0 + size, fy = fracY[j];
    for (let i = 0; i < size; i++) {
      const c = at[i], fx = frac[i];
      const top = coarse[r0 + c] * (1 - fx) + coarse[r0 + c + 1] * fx;
      const bottom = coarse[r1 + c] * (1 - fx) + coarse[r1 + c + 1] * fx;
      up[j * size + i] = top * (1 - fy) + bottom * fy;
    }
  }

  // filler: of the coarse source. 1: filler, 2: flat ground that isn't (decides), 0: the rest
  const lat = (latN + latS) / 2;
  const pixelM = 40075016.7 / ((1 << z) * size) * Math.cos(lat * Math.PI / 180);
  const kind = new Uint8Array(count);
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const k = j * size + i;
      const d = heights[k] - up[k];
      if (Math.abs(d) <= FILLER_M) {
        kind[k] = 1;
        continue;
      }
      if (heights[k] <= 1) continue;
      // the slope of the coarse tile, as numpy.gradient: central differences, one sided at the edges
      const l = i > 0 ? k - 1 : k, r = i < size - 1 ? k + 1 : k;
      const t = j > 0 ? k - size : k, b = j < size - 1 ? k + size : k;
      const gx = (up[r] - up[l]) / ((r - l) * pixelM), gy = (up[b] - up[t]) / ((b - t) / size * pixelM);
      if (Math.hypot(gx, gy) < FLAT_SLOPE) kind[k] = 2;
    }
  }

  // the median of d over the deciding pixels: null with too few
  const values = new Float32Array(count);
  const median = (wanted: (k: number) => boolean): number | null => {
    let n = 0;
    for (let k = 0; k < count; k++) if (wanted(k)) values[n++] = heights[k] - up[k];
    if (n < MIN_FLAT) return null;
    const sorted = values.subarray(0, n).sort();
    return n % 2 ? sorted[n >> 1] : (sorted[(n >> 1) - 1] + sorted[n >> 1]) / 2;
  };
  // no flat ground: all the ground that isn't filler
  const m = median((k) => kind[k] === 2) ?? median((k) => kind[k] !== 1 && heights[k] > 1);
  if (m === null || m <= geoidAt(size / 2 - .5, size / 2 - .5) / 2 + MARGIN_M) return 0;

  // the filler patches too small or too far off to be (see MIN_FILLER, FILLER_MEAN_M) aren't
  const queue = new Int32Array(count);
  const seen = new Uint8Array(count);
  for (let s = 0; s < count; s++) {
    if (kind[s] !== 1 || seen[s]) continue;
    let head = 0, tail = 0, off = 0;
    queue[tail++] = s;
    seen[s] = 1;
    while (head < tail) {
      const k = queue[head++];
      off += Math.abs(heights[k] - up[k]);
      const i = k % size;
      if (i > 0 && kind[k - 1] === 1 && !seen[k - 1]) { seen[k - 1] = 1; queue[tail++] = k - 1; }
      if (i < size - 1 && kind[k + 1] === 1 && !seen[k + 1]) { seen[k + 1] = 1; queue[tail++] = k + 1; }
      if (k >= size && kind[k - size] === 1 && !seen[k - size]) { seen[k - size] = 1; queue[tail++] = k - size; }
      if (k < count - size && kind[k + size] === 1 && !seen[k + size]) { seen[k + size] = 1; queue[tail++] = k + size; }
    }
    if (tail < MIN_FILLER || off / tail > FILLER_MEAN_M) for (let q = 0; q < tail; q++) kind[queue[q]] = 0;
  }

  let corrected = 0;
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const k = j * size + i;
      if (kind[k] === 1) continue;
      heights[k] = Math.max(heights[k] - geoidAt(i, j), 0);
      corrected++;
    }
  }
  return corrected;
}
