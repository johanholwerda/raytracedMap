import { LightmapResolution, LightmapView } from './skyVisibilityLightmap/SkyVisibilityLightmap';

// The quality param: the renderer (forward or deferred) and the settings of the sky visibility lightmap, the TAA and
// the tilt-shift it sets.
export enum Quality {
  VeryHigh,
  High,
  Medium,
  Low,
  VeryLow,
  // forward shaded instead of deferred: the sun with its shadows and the sky term only
  Forward,
}

export const QUALITIES = ['very high', 'high', 'medium', 'low', 'very low', 'forward'];

// The shadowMode param: where the sun shadows come from. The values are the SUN_SHADOWS override of deferredShading.wgsl.
export enum ShadowMode {
  None,
  // the cascaded VSM shadow map of SunShadows
  ShadowMap,
  // the sun visibility the texels of the sky visibility lightmap hold (SkyVisibilityLightmap.setSun): rays per texel,
  // fewer than 1 per pixel. Needs the lightmap, else it is raytrace
  LightmapRaytrace,
  // a ray per pixel against the city of the lightmap, soft with the TAA and hard without
  Raytrace,
}

export const SHADOW_MODES = ['none', 'shadowMap', 'lightmapRaytrace', 'raytrace'];

// The reflectionRays param: the water pixels that share a ray of the reflections of the buildings. The rays are over half
// of the deferred shading at 4K with a quarter of the screen water (RTX 3080: 1.7 ms, 1.4 with half of the rays, 1.1
// with a quarter, their pass included). With fewer, the reflections are a little softer: the streaks of the ripples 1
// pixel thin the most
export enum ReflectionRays {
  // a ray per water pixel, in the deferred shading
  PerPixel,
  // 1 per 2 x 1 pixels, traced before the shading (WaterReflectionHalfRes), which blends them
  Half,
  // 1 per 2 x 2 pixels
  Quarter,
}

export const REFLECTION_RAYS = ['per pixel', '1 per 2 pixels', '1 per 4 pixels'];

// the block of pixels of a ray along x and y, per ReflectionRays
export const REFLECTION_BLOCKS: [number, number][] = [[1, 1], [2, 1], [2, 2]];

export type QualitySettings = {
  // where the map ends, in world units from the camera (the viewDistanceMax param of MapWorld): the fog is 1 there, the
  // ground tiles load that far and the pixels beyond skip the shading
  viewDistanceMax: number;
  // forward shaded (MapGeometry.drawForward) instead of the gBuffer and the deferred shading
  forward: boolean;
  // the sky visibility lightmap, else the sky term
  skyVisibility: boolean;
  // the TAA, else no jitter and no history (no anti-aliasing)
  TAAEnabled: boolean;
  tiltShiftEnabled: boolean;
  // the tilt-shift blurs at half resolution (the halfRes param of TiltShift), a quarter of their cost
  tiltShiftHalfRes: boolean;
  // one of LIGHTMAP_SIZE_OPTIONS of SkyVisibilityLightmap
  lightMapSize: number;
  // lightmap texels per screen pixel along a side, at a screen height of 1080
  texelsPerPixel: number;
  // the most paths per frame, times 1000
  traceSamplesMax: number;
  // the paths a cell traces per frame while the view changes (the sampleCount param of SkyVisibilityLightmap): the
  // rays of a moving view are the cells that won a texel times this, mostly under traceSamplesMax
  sampleCount: number;
  // the update cell of the lightmap in pixels along a side, on a 1920 x 1080 screen, scaled by pixel count (the referenceCellSize param of
  // SkyVisibilityLightmap). Smaller cells converge in fewer frames (a cell picks the thin texels sooner), but the
  // claim, select and trace cost per frame while the view changes grows with the cell count
  referenceCellSize: number;
  // the slots of the tile table of the lightmap a tile is looked for in (the searchCount param of
  // SkyVisibilityLightmap). The table is full in practice at about 70%: a new tile then has no room where all of its
  // slots hold tiles in use (4% of the tiles of a view at 8), and its cells sample a coarser texel. Very high, with a
  // texel per pixel, fills the table: at 16 a tenth of those are left, for 7% more claim time
  searchCount: number;
  // the lightmap lookup per pixel or for 1 pixel per 2 x 2 (SkyVisibilityHalfRes), see LightmapResolution. Where a
  // texel covers about 3 pixels or more (texelsPerPixel .3 and less) the half resolution samples are closer together
  // than the texels
  lightmapResolution: LightmapResolution;
  // the water darker where it reflects a building, ray traced against the city of the lightmap (needs skyVisibility)
  waterReflections: boolean;
  // the water pixels per ray of those reflections, see ReflectionRays
  reflectionRays: ReflectionRays;
  // the water mirrored on animated ripples, with the sun reflected in it (no skyVisibility needed). The buildings are
  // only reflected with waterReflections
  waterRipples: boolean;
  // the ripples move, which draws the map every frame. Off, they're frozen and a still view stops being drawn
  animateWater: boolean;
  // caps the device pixel ratio of the canvas, -1 means no cap
  pixelRatioMax: number;
  // MSAA on the cars (CarTraffic), off where the map has no TAA either, and on medium for its cost on mobile
  carsMSAA: boolean;
  // the sun rays and the lightmap lookup of the cars per pixel, else per car (a compute pass, interpolated over the car)
  carsPerPixelLighting: boolean;
  // where the sun shadows come from, see ShadowMode. The forward shading only has the shadow map
  shadowMode: ShadowMode;
};

export const qualitySettings: Record<Quality, QualitySettings> = {
  [Quality.VeryHigh]: { viewDistanceMax: 8, forward: false, skyVisibility: true, TAAEnabled: true, tiltShiftEnabled: true, tiltShiftHalfRes: true, lightMapSize: 6144, texelsPerPixel: 1, traceSamplesMax: 4096, sampleCount: 32, referenceCellSize: 1.5, searchCount: 16,lightmapResolution: LightmapResolution.Full, waterReflections: true, reflectionRays: ReflectionRays.PerPixel, waterRipples: true, animateWater: false, pixelRatioMax: -1, carsMSAA: true, carsPerPixelLighting: true, shadowMode: ShadowMode.LightmapRaytrace },
  [Quality.High]: { viewDistanceMax: 8, forward: false, skyVisibility: true, TAAEnabled: true, tiltShiftEnabled: true, tiltShiftHalfRes: true, lightMapSize: 4096, texelsPerPixel: .6, traceSamplesMax: 1536, sampleCount: 16, referenceCellSize: 2, searchCount: 8,lightmapResolution: LightmapResolution.Full, waterReflections: true, reflectionRays: ReflectionRays.PerPixel, waterRipples: true, animateWater: false, pixelRatioMax: -1, carsMSAA: true, carsPerPixelLighting: true, shadowMode: ShadowMode.LightmapRaytrace },
  [Quality.Medium]: { viewDistanceMax: 6, forward: false, skyVisibility: true, TAAEnabled: true, tiltShiftEnabled: true, tiltShiftHalfRes: true, lightMapSize: 3072, texelsPerPixel: .3, traceSamplesMax: 256, sampleCount: 8, referenceCellSize: 4, searchCount: 8,lightmapResolution: LightmapResolution.Half, waterReflections: false, reflectionRays: ReflectionRays.Quarter, waterRipples: true, animateWater: false, pixelRatioMax: 2, carsMSAA: false, carsPerPixelLighting: false, shadowMode: ShadowMode.LightmapRaytrace },
  [Quality.Low]: { viewDistanceMax: 4, forward: false, skyVisibility: true, TAAEnabled: true, tiltShiftEnabled: true, tiltShiftHalfRes: true, lightMapSize: 2048, texelsPerPixel: .15, traceSamplesMax: 256, sampleCount: 8, referenceCellSize: 4, searchCount: 8,lightmapResolution: LightmapResolution.HalfBilinear, waterReflections: false, reflectionRays: ReflectionRays.Quarter, waterRipples: false, animateWater: false, pixelRatioMax: 1, carsMSAA: false, carsPerPixelLighting: false, shadowMode: ShadowMode.ShadowMap },
  // the lightmap settings only count when skyVisibility is turned on by hand
  [Quality.VeryLow]: { viewDistanceMax: 4, forward: false, skyVisibility: false, TAAEnabled: false, tiltShiftEnabled: true, tiltShiftHalfRes: true, lightMapSize: 1024, texelsPerPixel: .1, traceSamplesMax: 256, sampleCount: 8, referenceCellSize: 4, searchCount: 8,lightmapResolution: LightmapResolution.HalfBilinear, waterReflections: false, reflectionRays: ReflectionRays.Quarter, waterRipples: false, animateWater: false, pixelRatioMax: 1, carsMSAA: false, carsPerPixelLighting: false, shadowMode: ShadowMode.ShadowMap },
  // no SSAO, sky visibility, AA or tilt-shift: they need the gBuffer. The lightmap settings are those of very low
  // no cars either: they're drawn over the deferred image
  [Quality.Forward]: { viewDistanceMax: 4, forward: true, skyVisibility: false, TAAEnabled: false, tiltShiftEnabled: false, tiltShiftHalfRes: false, lightMapSize: 1024, texelsPerPixel: .1, traceSamplesMax: 256, sampleCount: 8, referenceCellSize: 4, searchCount: 8,lightmapResolution: LightmapResolution.HalfBilinear, waterReflections: false, reflectionRays: ReflectionRays.Quarter, waterRipples: false, animateWater: false, pixelRatioMax: 1, carsMSAA: false, carsPerPixelLighting: false, shadowMode: ShadowMode.ShadowMap },
};

// The drawMode param: the full shading, or a debug view.
export enum DrawMode {
  Full,
  // everything additively, straight to the screen
  Overdraw,
  // the full shading with the bounding boxes of the building meshes over it
  BoundingBoxes,
  // the sky visibility lightmap with one of its views, see LightmapView
  SkyLight,
  VirtualLightmap,
  LightmapTiles,
  LightmapLevels,
  LightmapConverging,
  // the 4 cascades of the sun shadow map, see SunShadows.drawView. Not in DRAW_MODES
  ShadowMap,
  // the shadow lookup of each pixel alone (getSunShadow of deferredShading.wgsl, the rays or the shadow map), white is lit
  Shadows,
  // the cascade of each pixel in a color, see cascadeDebugColor. Not in DRAW_MODES
  ShadowMapCascades,
  // the wall atlas of the last city tile that finished, see WallAtlasView
  WallAtlas,
}

// The options of the drawMode param, label to DrawMode. Without ShadowMap and ShadowMapCascades
export const DRAW_MODES: Record<string, DrawMode> = {
  'full': DrawMode.Full,
  'overdraw': DrawMode.Overdraw,
  'bounding boxes': DrawMode.BoundingBoxes,
  'sky light': DrawMode.SkyLight,
  'virtual lightmap': DrawMode.VirtualLightmap,
  'lightmap tiles': DrawMode.LightmapTiles,
  'lightmap levels': DrawMode.LightmapLevels,
  'lightmap converging': DrawMode.LightmapConverging,
  'shadows': DrawMode.Shadows,
  'wall atlas': DrawMode.WallAtlas,
};

// The view of the sky visibility lightmap per draw mode, the others shade with it.
// The SHADOW_VIEW of deferredShading.wgsl per draw mode, the others shade.
export const SHADOW_VIEWS: Partial<Record<DrawMode, number>> = {
  [DrawMode.Shadows]: 1,
  [DrawMode.ShadowMapCascades]: 2,
};

export const SKY_VISIBILITY_VIEWS: Partial<Record<DrawMode, LightmapView>> = {
  [DrawMode.SkyLight]: LightmapView.Light,
  [DrawMode.VirtualLightmap]: LightmapView.Lightmap,
  [DrawMode.LightmapTiles]: LightmapView.Tiles,
  [DrawMode.LightmapLevels]: LightmapView.Levels,
  [DrawMode.LightmapConverging]: LightmapView.Converging,
};

// The timeOfDay param: the sun direction (the sunAngleX/Y params of SunShadows, see timeOfDaySunAngles) and the sky and
// sun colors it sets.
export enum TimeOfDay {
  Sunrise,
  Morning,
  Noon,
  Afternoon,
  Sunset,
}

export const TIMES_OF_DAY = ['sunrise', 'morning', 'noon', 'afternoon', 'sunset'];

export type TimeOfDaySettings = {
  // the elevation of the sun the time of day aims for, radians above the horizon; noon takes the highest the sun gets
  sunElevation: number;
  // -1 before noon (sun in the east), 1 after (west), 0 at noon (on the meridian)
  sunSide: number;
  skyColor: string;
  sunColor: string;
  // the clear color, which the distance fog fades to as well
  backgroundColor: string;
};

export const timeOfDaySettings: Record<TimeOfDay, TimeOfDaySettings> = {
  [TimeOfDay.Sunrise]: { sunElevation: .12, sunSide: -1, skyColor: '#4f5663', sunColor: '#ffa866', backgroundColor: '#c9b8b0' },
  [TimeOfDay.Morning]: { sunElevation: .55, sunSide: -1, skyColor: '#657995', sunColor: '#d8b794', backgroundColor: '#bcd4ec' },
  [TimeOfDay.Noon]: { sunElevation: 1., sunSide: 0, skyColor: '#b2c1d2', sunColor: '#60503d', backgroundColor: '#a9bbcc' },
  [TimeOfDay.Afternoon]: { sunElevation: .3, sunSide: 1, skyColor: '#596b80', sunColor: '#dfba9f', backgroundColor: '#b8dbff' },
  [TimeOfDay.Sunset]: { sunElevation: .1, sunSide: 1, skyColor: '#697398', sunColor: '#ff7e33', backgroundColor: '#d4a088' },
};

// the relative odds of each time of day being picked at init: sunset twice as likely as the others
const TIME_OF_DAY_WEIGHTS: Record<TimeOfDay, number> = {
  [TimeOfDay.Sunrise]: 1,
  [TimeOfDay.Morning]: 1,
  [TimeOfDay.Noon]: 1,
  [TimeOfDay.Afternoon]: 1,
  [TimeOfDay.Sunset]: 2,
};

export function randomTimeOfDay(): TimeOfDay {
  const weights = TIMES_OF_DAY.map((_, i) => TIME_OF_DAY_WEIGHTS[i as TimeOfDay]);
  let r = Math.random() * weights.reduce((a, b) => a + b, 0);
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r < 0) return i;
  }
  return weights.length - 1;
}

// the declination of the sun at the summer solstice, radians
const SUN_DECLINATION_SUMMER = 23.44 * Math.PI / 180;

/**
 * The sunAngleX/Y params of SunShadows (elevation, and heading: 0 = +z = south, positive toward +x = east, see
 * MapProjection) for a time of day at a latitude (degrees), on the midsummer day of that hemisphere (the most daylight,
 * no polar night). The hour angle is the one at which the sun reaches settings.sunElevation on settings.sunSide; where
 * the sun never gets that high (polar regions at noon, morning) it takes noon, where it never gets that low (midnight
 * sun) midnight. So the sun is north of the zenith at noon south of the tropics, in the northwest at a northern
 * summer sunset, low in the north at a polar 'sunset', etc.
 */
export function timeOfDaySunAngles(settings: TimeOfDaySettings, latitude: number): { angleX: number; angleY: number } {
  const lat = latitude * Math.PI / 180;
  const dec = latitude < 0 ? -SUN_DECLINATION_SUMMER : SUN_DECLINATION_SUMMER;
  const sinLat = Math.sin(lat), cosLat = Math.cos(lat);
  const sinDec = Math.sin(dec), cosDec = Math.cos(dec);
  // sin(elevation) = sinLat sinDec + cosLat cosDec cos(hourAngle)
  let hourAngle = 0;
  if (settings.sunSide !== 0) {
    const cosH = (Math.sin(settings.sunElevation) - sinLat * sinDec) / (cosLat * cosDec);
    hourAngle = settings.sunSide * Math.acos(Math.min(1, Math.max(-1, cosH)));
  }
  const cosH = Math.cos(hourAngle);
  // the direction to the sun in east/north/up, hour angle positive toward the west
  const east = -cosDec * Math.sin(hourAngle);
  const north = cosLat * sinDec - sinLat * cosDec * cosH;
  const up = sinLat * sinDec + cosLat * cosDec * cosH;
  // world: x = east, z = -north; SunShadows: x = cos(angleX) sin(angleY), z = cos(angleX) cos(angleY)
  return {
    angleX: Math.asin(Math.min(1, Math.max(-1, up))),
    angleY: Math.atan2(east, -north),
  };
}
