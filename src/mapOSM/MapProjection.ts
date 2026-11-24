import type { LatLngBounds } from './osmGeometryCore';

/**
 * Equirectangular projection shared by MapOSM, OSMBuilder, CityJSONBuilder and
 * CityGlTFBuilder. Maps lat/lng into a normalized world space where the
 * latitude span of the selected region covers y in [-1, 1], with a cos(lat)
 * correction so longitude keeps its real metric proportions.
 *
 * Previously the same math (latMid, lngMid, cosLat, scale, metersToNorm) was
 * copy-pasted in every builder and in MapOSM.storeProjectionParams /
 * worldToLatLng / latLngToWorld.
 */
const METERS_PER_DEGREE_LAT = 111_320;

export default class MapProjection {
  public latMid = 0;
  public lngMid = 0;
  public cosLat = 1;
  /** 2 / latSpan — maps latitude delta to normalized y. */
  public scale = 1;
  /** Meters per normalized world unit (for elevation / radius calculations). */
  public metersPerUnit = 1;
  /** Width / height aspect ratio of the projected region. */
  public mapAspectRatio = 1;
  public bounds: LatLngBounds = { south: 0, west: 0, north: 0, east: 0 };

  /** (Re)compute projection parameters from a lat/lng bounds rectangle. */
  public setBounds(bounds: LatLngBounds): void {
    this.bounds = bounds;
    this.latMid = (bounds.south + bounds.north) * 0.5;
    this.lngMid = (bounds.west + bounds.east) * 0.5;
    this.cosLat = Math.cos((this.latMid * Math.PI) / 180);
    const latSpan = Math.max(bounds.north - bounds.south, 1e-9);
    const lngSpan = Math.max(bounds.east - bounds.west, 1e-9);
    this.scale = 2 / latSpan;
    this.metersPerUnit = (METERS_PER_DEGREE_LAT * latSpan) / 2;
    this.mapAspectRatio = (lngSpan * this.cosLat) / latSpan;
  }

  /** Scale factor: meters → normalized world units. */
  public get metersToNorm(): number {
    return this.scale / METERS_PER_DEGREE_LAT;
  }

  /** Project lng/lat to normalized world x/y (before the engine's x,z,-y swap). */
  public project(lng: number, lat: number): [number, number] {
    const x = (lng - this.lngMid) * this.cosLat * this.scale;
    const y = (lat - this.latMid) * this.scale;
    return [x, y];
  }

  /** Convert world-space ground coords (x, z) back to lat/lng. */
  public worldToLatLng(worldX: number, worldZ: number): { lat: number; lng: number } {
    const lng = worldX / (this.cosLat * this.scale) + this.lngMid;
    const lat = -worldZ / this.scale + this.latMid;
    return { lat, lng };
  }

  /** Convert lat/lng to world-space ground coords (x, z). */
  public latLngToWorld(lat: number, lng: number): { x: number; z: number } {
    const x = (lng - this.lngMid) * this.cosLat * this.scale;
    const z = -(lat - this.latMid) * this.scale;
    return { x, z };
  }

  /**
   * Allocation-free variant of latLngToWorld: writes x/z into `out` (any object
   * with x/z fields, e.g. a scratch Vector3) instead of returning a new object.
   * Used in hot streaming paths to avoid per-call garbage.
   */
  public latLngToWorldInto(lat: number, lng: number, out: { x: number; z: number }): void {
    out.x = (lng - this.lngMid) * this.cosLat * this.scale;
    out.z = -(lat - this.latMid) * this.scale;
  }

  /** Approximate web-mercator tile size in meters at a given zoom (at the projection latitude). */
  public tileSizeMeters(zoom: number): number {
    const n = Math.pow(2, zoom);
    const latRad = (this.latMid * Math.PI) / 180;
    return (40075000 * Math.cos(latRad)) / n;
  }
}
