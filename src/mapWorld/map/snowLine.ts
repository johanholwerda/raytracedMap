import type Vector3 from 'webGPU-framework/renderer/math/Vector3';

/*
The height (m above sea level) above which the ground is snow covered, by latitude: roughly the climatic snow line of
the mountains at that latitude, a bit lower so the high peaks show white the year round. Highest in the dry subtropics
(the Andes, Tibet), lower toward the poles: about 4 km in the Himalayas, 2.6 km in the Alps, 1 km in western Norway,
0.6 km in Iceland, at sea level in Greenland and Antarctica. The southern hemisphere is cooler at the same latitude (more
ocean, Antarctica): its latitudes count as SOUTH_SCALE times as far from the equator (Patagonia, the Southern Alps of New
Zealand).

Approximate by design: the real line differs by hundreds of meters with the side of a range (wet or dry), the aspect of
a slope and the season. The shading breaks it up with noise and slope, see snow.wgsl.
 */

// |latitude| (degrees), height (m), linear between
const SNOW_LINE_TABLE: [number, number][] = [
  [0, 4800],
  [18, 4900],
  [28, 4100],
  [36, 3500],
  [42, 3000],
  [46, 2600],
  [52, 1900],
  [57, 1400],
  [61, 1000],
  [64, 700],
  [67, 450],
  [72, 150],
  [76, 0],
];

const SOUTH_SCALE = 1.1;

/**
 * How far (m) below the snow line at the top of its noise the highest ground in view is when there is no snow in view,
 * see SNOW_SKIP_MARGIN of snow.wgsl (the same per point).
 */
export const SNOW_SKIP_MARGIN_M = 50;

/** The snow line (m above sea level) at a latitude (degrees, south negative), see SNOW_LINE_TABLE. */
export function snowLineMeters(lat: number): number {
  const a = lat < 0 ? -lat * SOUTH_SCALE : lat;
  const table = SNOW_LINE_TABLE;
  if (a <= table[0][0]) return table[0][1];
  for (let i = 1; i < table.length; i++) {
    const [lat1, h1] = table[i];
    if (a <= lat1) {
      const [lat0, h0] = table[i - 1];
      return h0 + (h1 - h0) * (a - lat0) / (lat1 - lat0);
    }
  }
  return table[table.length - 1][1];
}

/** The snow the shading puts on the ground, see snow.wgsl. */
export type SnowParams = {
  enabled: boolean;
  /** the snow line in world y: relative to the ground at the projection center, like the heights */
  line: number;
  /** how far the noise moves the snow line up and down, m */
  noise: number;
  metersPerUnit: number;
  /** the albedo, as the color param (not squared) */
  color: Vector3;
};
