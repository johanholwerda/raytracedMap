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
  // the highest height (m, relative to the origin) of each grid, by its key, and of them all (maxHeightM)
  private maxHeights = new Map<number, number>();
  private maxHeight = -Infinity;
  private maxDirty = false;
  // the lowest of each grid and of them all, as the highest
  private minHeights = new Map<number, number>();
  private minHeight = Infinity;

  constructor(private projection: MapProjection) {
  }

  /** The grid of a tile, from its heights in m (relative to the origin of the ElevationService). */
  public addTile(tile: TileCoord, heightsM: Float32Array): void {
    if (tile.z !== this.zoom) this.clear();
    this.zoom = tile.z;
    const k = this.key(tile.x, tile.y);
    this.grids.set(k, createTerrainGrid(tile, this.projection, heightsM));
    let max = -Infinity;
    let min = Infinity;
    for (let i = 0; i < heightsM.length; i++) {
      if (heightsM[i] > max) max = heightsM[i];
      if (heightsM[i] < min) min = heightsM[i];
    }
    this.maxHeights.set(k, max);
    this.minHeights.set(k, min);
    this.maxDirty = true;
  }

  /**
   * The highest ground (m, relative to the origin of the ElevationService) of the loaded tiles, with their rings:
   * -Infinity without tiles.
   */
  public get maxHeightM(): number {
    this.updateExtremes();
    return this.maxHeight;
  }

  /** The lowest ground (m, relative to the origin of the ElevationService) of the loaded tiles, Infinity without. */
  public get minHeightM(): number {
    this.updateExtremes();
    return this.minHeight;
  }

  /**
   * How much higher the highest ground of the loaded tiles is than the lowest, m: about 0 where the land is flat, 0
   * without tiles.
   */
  public get reliefM(): number {
    this.updateExtremes();
    return this.maxHeight > this.minHeight ? this.maxHeight - this.minHeight : 0;
  }

  private updateExtremes(): void {
    if (!this.maxDirty) return;
    this.maxDirty = false;
    this.maxHeight = -Infinity;
    this.minHeight = Infinity;
    for (const h of this.maxHeights.values()) if (h > this.maxHeight) this.maxHeight = h;
    for (const h of this.minHeights.values()) if (h < this.minHeight) this.minHeight = h;
  }

  /** key: z/x/y, as OpenFreeMapTileService.tileKey. */
  public removeTile(key: string): void {
    const [z, x, y] = key.split('/').map(Number);
    if (z !== this.zoom) return;
    const k = this.key(x, y);
    if (this.grids.get(k) === this.last) this.last = null;
    this.grids.delete(k);
    this.maxHeights.delete(k);
    this.minHeights.delete(k);
    this.maxDirty = true;
  }

  public clear(): void {
    this.grids.clear();
    this.maxHeights.clear();
    this.minHeights.clear();
    this.maxDirty = true;
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
