import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import AABox from 'webGPU-framework/renderer/math/AABox';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import { LatLngBounds } from './OSMBuilder';
import { CityBuildResult, CITY_SURFACE_KEYS, CitySurfaceKey } from './CityJSONBuilder';
import {
  CityGeometryBuilder,
  CityPositionsResult,
  TerrainCategory,
  TerrainCategoryDef,
  TERRAIN_CATEGORY_DEFS,
  VERTEX_FLOATS,
} from './cityGlTFCore';
import MapWorkerPool from './MapWorkerPool';

// Re-export the terrain category types/defs (which now live in the pure core)
// so existing importers (layerDefs, MapNetherlands) keep importing them here.
export { TERRAIN_CATEGORY_DEFS };
export type { TerrainCategory, TerrainCategoryDef };

/**
 * Renderer-aware façade over the pure {@link CityGeometryBuilder} core. Turns a
 * 3DBAG **3D Tiles** binary glTF (.glb) tile into renderable building meshes,
 * projected into the normalized map frame so it overlays the OSM map.
 *
 * The geometry-heavy work (parseGlb + meshopt WASM decode + projection +
 * de-index) lives in {@link cityGlTFCore}, which is free of any GPU deps so it
 * can run either:
 *  - on the main thread via {@link buildSync} (kept for fallback/comparison), or
 *  - in a Web Worker via {@link build} (the shared {@link MapWorkerPool}), which keeps
 *    the ~10-25ms/tile decode+build off the frame. Either way the only
 *    main-thread, renderer-bound step is wrapping the position buffer into a
 *    `Mesh` for GPU upload.
 */
export default class CityGlTFBuilder {
  private renderer: Renderer;
  private core = new CityGeometryBuilder();
  private pool: MapWorkerPool;

  constructor(renderer: Renderer, pool: MapWorkerPool) {
    this.renderer = renderer;
    this.pool = pool;
  }

  /** Number of building-tile builds outstanding in the shared worker pool. */
  public get activeWorkers(): number {
    return this.pool.buildingJobs;
  }

  /**
   * Build a city tile in a Web Worker (off the main thread). Only the GPU mesh
   * creation runs here, on resolution. No MeshoptDecoder.ready await needed —
   * the worker owns its own decoder instance.
   */
  public async build(glb: ArrayBuffer, bounds: LatLngBounds, options?: { flatten?: boolean }): Promise<CityBuildResult> {
    const flatten = options?.flatten ?? true;
    const result = await this.pool.buildCity(glb, bounds, flatten);
    return this.wrapPositions(result);
  }

  /**
   * Synchronous main-thread build (fallback / comparison). Caller must ensure
   * MeshoptDecoder.ready has resolved before calling (e.g. via await
   * MeshoptDecoder.ready at init).
   */
  public buildSync(glb: ArrayBuffer, bounds: LatLngBounds, options?: { flatten?: boolean }): CityBuildResult {
    const flatten = options?.flatten ?? true;
    return this.wrapPositions(this.core.buildPositions(glb, bounds, flatten));
  }

  /** Wrap a geometry-only result into a renderable CityBuildResult (creates the GPU mesh). */
  private wrapPositions(result: CityPositionsResult): CityBuildResult {
    const layers = {} as Record<CitySurfaceKey, Mesh[]>;
    const triangleCount = {} as Record<CitySurfaceKey, number>;
    const surfaceCount = {} as Record<CitySurfaceKey, number>;
    for (const key of CITY_SURFACE_KEYS) {
      layers[key] = [];
      triangleCount[key] = 0;
      surfaceCount[key] = 0;
    }
    // No semantic split: everything lands in the single 'Building' layer.
    const buildingKey = CITY_SURFACE_KEYS[0];
    let bounds: AABox | undefined;
    if (result.vertices.length > 0) {
      const m = new Mesh(this.renderer);
      // One interleaved stride-4 attribute: xyz position + `.w` tile-local
      // building id (_FEATURE_ID_0, or NO_FEATURE_ID). Bound as `position` so
      // every map mesh (terrain, ground, buildings) shares one vertex layout;
      // the osm shader reads `.xyz` and (under FEATURE_IDS) `.w`. Single
      // attribute => Mesh skips the interleave copy and uploads directly.
      m.setAttribute('position', VERTEX_FLOATS, result.vertices);
      layers[buildingKey] = [m];
      // AABB computed in the core (off the main thread for the worker path), so
      // the caller doesn't have to re-scan every vertex to get the tile bounds.
      bounds = new AABox(
        new Vector3(result.min[0], result.min[1], result.min[2]),
        new Vector3(result.max[0], result.max[1], result.max[2]),
      );
    }
    triangleCount[buildingKey] = result.triangleCount;
    surfaceCount[buildingKey] = result.surfaceCount;
    return {
      layers,
      triangleCount,
      surfaceCount,
      objectCount: result.objectCount,
      lodUsed: result.lodUsed,
      bounds,
      buildings: result.buildings,
    };
  }

  /** Build a 3D Basisvoorziening terrain tile in a Web Worker (off the main thread). */
  public buildTerrain(glb: ArrayBuffer, bounds: LatLngBounds): Promise<Record<TerrainCategory, Float32Array>> {
    return this.pool.buildTerrain3D(glb, bounds);
  }

  /**
   * Synchronous main-thread terrain build (fallback). Caller must ensure
   * MeshoptDecoder.ready has resolved before calling.
   */
  public buildTerrainSync(glb: ArrayBuffer, bounds: LatLngBounds): Record<TerrainCategory, Float32Array> {
    return this.core.buildTerrainPositions(glb, bounds);
  }

}
