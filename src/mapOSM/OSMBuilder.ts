import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import LogGL from 'webGPU-framework/renderer/core/LogGL';
import {
  accumulateLayers,
  buildOSMPositions,
  defaultOSMBuilderOptions,
  LAYER_KEYS,
} from './osmGeometryCore';
import type {
  LatLngBounds,
  LayerAccumulator,
  LayerKey,
  OSMBuilderOptions,
  OSMElement,
} from './osmGeometryCore';

/**
 * Fetches an OpenStreetMap extract for a lat/lng rectangle from the public
 * Overpass API and turns it into renderable geometry, grouped by "layer"
 * (buildings, water, forest, vegetation, roads, paths, pedestrian areas) -
 * similar in spirit to Blender's blosm add-on.
 *
 * The renderer-free geometry build (projection / triangulation / extrusion)
 * lives in {@link osmGeometryCore} so it can run on the main thread *or* inside
 * the PDOK terrain Web Worker; this class only adds the Overpass fetch and the
 * GPU `Mesh` wrapping.
 *
 * No API key is required. Keep the requested area small (a few hundred
 * meters) to avoid Overpass gateway timeouts.
 */

// Re-export the geometry-core types/constants other modules still import from
// here (the pure definitions now live in osmGeometryCore).
export type {
  LatLngBounds,
  LayerConfig,
  LayerKey,
  LayerMode,
  OSMBuilderOptions,
  OSMElement,
  OSMSaveFile,
} from './osmGeometryCore';
export { LAYER_CONFIGS, LAYER_KEYS, defaultOSMBuilderOptions } from './osmGeometryCore';

export type OSMBuildResult = {
  layers: Record<LayerKey, Mesh[]>;
  mapAspectRatio: number;
  bounds: LatLngBounds;
  counts: Record<LayerKey, number>;
};

export default class OSMBuilder {
  private renderer: Renderer;
  private options: OSMBuilderOptions;

  constructor(renderer: Renderer, options: Partial<OSMBuilderOptions> = {}) {
    this.renderer = renderer;
    this.options = { ...defaultOSMBuilderOptions, ...options };
  }

  /** Fetch from Overpass, then build geometry. Convenience wrapper. */
  public async build(bounds: LatLngBounds): Promise<OSMBuildResult> {
    const elements = await this.fetchElements(bounds);
    return this.buildFromElements(bounds, elements);
  }

  /** Download the OSM extract for the given bounds from the Overpass API. */
  public async fetchElements(bounds: LatLngBounds): Promise<OSMElement[]> {
    LogGL.log('[MapOSM] === OpenStreetMap area load ===');
    LogGL.log(
      `[MapOSM] bbox: south ${bounds.south} west ${bounds.west} north ${bounds.north} east ${bounds.east}`,
    );
    const elements = await this.fetchOverpass(bounds);
    LogGL.log(`[MapOSM] fetched ${elements.length} elements`);
    return elements;
  }

  /** Build renderable geometry from already-fetched (or loaded) elements. */
  public buildFromElements(bounds: LatLngBounds, elements: OSMElement[]): OSMBuildResult {
    const { acc, mapAspectRatio } = accumulateLayers(bounds, elements, this.options);
    const layers = {} as Record<LayerKey, Mesh[]>;
    const counts = {} as Record<LayerKey, number>;
    for (const key of LAYER_KEYS) {
      layers[key] = this.makeMeshes(acc[key]);
      counts[key] = acc[key].features;
    }
    return { layers, mapAspectRatio, bounds, counts };
  }

  /**
   * Build merged vertex positions (xyz triangle soup) per layer, without
   * creating Meshes. Used by the batched terrain buffer, which packs many
   * tiles' positions into one GPU buffer to cut draw calls.
   */
  public buildPositions(bounds: LatLngBounds, elements: OSMElement[]): Record<LayerKey, Float32Array> {
    return buildOSMPositions(bounds, elements, this.options);
  }

  // --- Overpass fetch -------------------------------------------------------

  private buildQuery(b: LatLngBounds): string {
    const bbox = `${b.south},${b.west},${b.north},${b.east}`;
    return [
      '[out:json][timeout:30];',
      '(',
      `  way["building"](${bbox});`,
      `  way["building:part"](${bbox});`,
      `  way["natural"="water"](${bbox});`,
      `  relation["natural"="water"](${bbox});`,
      `  way["waterway"="riverbank"](${bbox});`,
      `  way["landuse"="forest"](${bbox});`,
      `  way["natural"="wood"](${bbox});`,
      `  way["landuse"~"grass|meadow|recreation_ground|village_green|cemetery"](${bbox});`,
      `  way["leisure"~"park|garden|pitch|playground"](${bbox});`,
      `  way["natural"="scrub"](${bbox});`,
      `  way["highway"](${bbox});`,
      ');',
      'out geom;',
    ].join('\n');
  }

  private async fetchOverpass(bounds: LatLngBounds): Promise<OSMElement[]> {
    const query = this.buildQuery(bounds);
    LogGL.log('[MapOSM] Overpass query:\n' + query);

    const response = await fetch(this.options.overpassUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: 'data=' + encodeURIComponent(query),
    });

    if (!response.ok) {
      throw new Error(`[MapOSM] Overpass request failed: ${response.status} ${response.statusText}`);
    }

    const json = await response.json();

    console.log('loaded', json);

    return (json.elements || []) as OSMElement[];
  }

  // --- mesh creation --------------------------------------------------------

  private makeMeshes(a: LayerAccumulator): Mesh[] {
    if (a.positions.length === 0) return [];

    const meshes: Mesh[] = [];
    const maxVerts = this.options.chunkMaxVertices;
    const totalVerts = a.positions.length / 3;

    for (let start = 0; start < totalVerts; start += maxVerts) {
      const end = Math.min(start + maxVerts, totalVerts);
      const mesh = new Mesh(this.renderer);
      mesh.setPositions(new Float32Array(a.positions.slice(start * 3, end * 3)));
      meshes.push(mesh);
    }
    return meshes;
  }
}
