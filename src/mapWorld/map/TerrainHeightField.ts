import MapProjection from '../../mapOSM/MapProjection';
import type { TileCoord } from '../../mapOSM/PDOKTileService';
import TerrainGrid from './terrainGrid';
import { createTerrainGrid } from './worldGeometryCore';

/**
 * The ground of the loaded terrain tiles on the main thread: per tile the same TerrainGrid the worker built its ground
 * mesh from (see ElevationService.getTileGrid), so heightAt is the height of the ground mesh, which the layers are draped
 * on. For what moves over the ground: the cars, the camera pivot and the plane.
 */
export default class TerrainHeightField {
  private grids = new Map<string, TerrainGrid>();
  private zoom = 14;

  constructor(private projection: MapProjection) {
  }

  /** The grid of a tile, from its heights in m (relative to the origin of the ElevationService). */
  public addTile(tile: TileCoord, heightsM: Float32Array): void {
    this.zoom = tile.z;
    this.grids.set(this.key(tile.z, tile.x, tile.y), createTerrainGrid(tile, this.projection, heightsM));
  }

  /** key: z/x/y, as OpenFreeMapTileService.tileKey. */
  public removeTile(key: string): void {
    this.grids.delete(key);
  }

  public clear(): void {
    this.grids.clear();
  }

  /** The height of the ground at world x, z (world units), null where no tile with heights is loaded. */
  public heightAt(x: number, z: number): number | null {
    const { lat, lng } = this.projection.worldToLatLng(x, z);
    const n = 2 ** this.zoom;
    const tx = Math.floor((lng + 180) / 360 * n);
    const latRad = Math.max(Math.min(lat, 85.05), -85.05) * Math.PI / 180;
    const ty = Math.floor((1 - Math.asinh(Math.tan(latRad)) / Math.PI) / 2 * n);
    const grid = this.grids.get(this.key(this.zoom, ((tx % n) + n) % n, ty));
    return grid ? grid.heightAt(x, -z) : null;
  }

  private key(z: number, x: number, y: number): string {
    return `${z}/${x}/${y}`;
  }
}
