import MapProjection from '../osm/MapProjection';
import type { TileCoord } from '../osm/PDOKTileService';
import TerrainGrid from './terrainGrid';
import { createTerrainGrid } from './worldGeometryCore';

/**
 * The ground of the loaded terrain tiles on the main thread: per tile the same TerrainGrid the worker built its ground
 * mesh from (see ElevationService.getTileGrid), so heightAt is the height of the ground mesh, which the layers are draped
 * on. For what moves over the ground: the cars, the camera pivot and the plane.
 *
 * heightAt runs a few times per car per frame (thousands of times): the grids are found by a number, and the grid of
 * the last call is tried first, by its bounds (the front, center and back of a car are mostly on the same tile).
 */
export default class TerrainHeightField {
  // by x * 2^zoom + y
  private grids = new Map<number, TerrainGrid>();
  private zoom = 14;
  private last: TerrainGrid | null = null;

  constructor(private projection: MapProjection) {
  }

  /** The grid of a tile, from its heights in m (relative to the origin of the ElevationService). */
  public addTile(tile: TileCoord, heightsM: Float32Array): void {
    if (tile.z !== this.zoom) this.clear();
    this.zoom = tile.z;
    this.grids.set(this.key(tile.x, tile.y), createTerrainGrid(tile, this.projection, heightsM));
  }

  /** key: z/x/y, as OpenFreeMapTileService.tileKey. */
  public removeTile(key: string): void {
    const [z, x, y] = key.split('/').map(Number);
    if (z !== this.zoom) return;
    const k = this.key(x, y);
    if (this.grids.get(k) === this.last) this.last = null;
    this.grids.delete(k);
  }

  public clear(): void {
    this.grids.clear();
    this.last = null;
  }

  /** The height of the ground at world x, z (world units), null where no tile with heights is loaded. */
  public heightAt(x: number, z: number): number | null {
    // map coordinates (x, north)
    const north = -z;
    const last = this.last;
    if (last && x >= last.x0 && x < last.x1 && north >= last.y0 && north < last.y1) return last.heightAt(x, north);

    const p = this.projection;
    const lng = x / (p.cosLat * p.scale) + p.lngMid;
    const lat = north / p.scale + p.latMid;
    const n = 2 ** this.zoom;
    const tx = Math.floor((lng + 180) / 360 * n);
    const latRad = Math.max(Math.min(lat, 85.05), -85.05) * Math.PI / 180;
    const ty = Math.floor((1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * n);
    const grid = this.grids.get(this.key(((tx % n) + n) % n, ty));
    if (!grid) return null;
    this.last = grid;
    return grid.heightAt(x, north);
  }

  private key(x: number, y: number): number {
    return x * 2 ** this.zoom + y;
  }
}
