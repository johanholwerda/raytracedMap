import Renderer from 'webGPU-framework/renderer/render/Renderer';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Color from 'webGPU-framework/renderer/math/Color';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import AABox from 'webGPU-framework/renderer/math/AABox';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import Frustum from 'webGPU-framework/utils/camera/Frustum';
import hash from 'webGPU-framework/shaders/hash.wgsl?raw';
import overdraw from './osm/overdraw.wgsl?raw';
import terrainNormal from './map/terrainNormal.wgsl?raw';
import forwardShading from './osm/shading.wgsl?raw';
import { LayerRuntime } from './osm/layerDefs';
import TerrainLayerBuffer from './osm/TerrainLayerBuffer';
import BoundingBoxViewer, { BBoxEntry } from './osm/BoundingBoxViewer';
import MapProjection from './osm/MapProjection';
import GroundQuad from './osm/GroundQuad';
import MapGBuffer from './deferred/MapGBuffer';
import { WORLD_LAYERS } from './map/worldGeometryCore';
import { setMeshQuantization } from './city/meshQuantization';
import { TREE_COLORS, TREE_CROWN_COLORS, TREE_TRUNK_COLOR } from './map/worldTrees';
import snow from './snow.wgsl?raw';
import terrainMaterialMap from './terrainMaterialMap.wgsl?raw';
import type { SnowParams } from './map/snowLine';

// the ground layers the snow lies on (and the ground under them), see snow.wgsl. Not the water, the roads (cleared) or
// the railways, nor the urban areas
const SNOW_LAYERS = new Set(['farmland', 'sand', 'green', 'forest', 'paths']);
// the layer whose material the snow takes in the gBuffer: the glaciers, in the snow color
const SNOW_MATERIAL_LAYER = 'glacier';

/** The building meshes to draw, with their picking slots (parallel), see WorldTileStreamer. */
// tiles: per mesh the map tile of its wall atlas, x and y after each other (the uniform wallTile of the gBuffer).
// quantization: per mesh the decoding of its unorm16 vertices, see setMeshQuantization
// indexCounts: per mesh the indices to draw, without its trees when they are too small, see WorldTileStreamer.cullTrees
export type CityMeshes = { meshes: Mesh[]; slots: number[]; tiles: number[]; quantization: number[]; indexCounts: number[] };

export type DrawStats = { meshCount: number; vertexCount: number; culledCount: number };

/**
 * The geometry of the map and how it is drawn: the ground layers (one packed buffer + gBuffer material each), the
 * ground quad under them and the buildings (meshes of the WorldTileStreamer, frustum culled with meshBounds). Draws
 * them into the gBuffer, forward shaded (osm/shading.wgsl: sun with shadows and the sky term, see forwardMaterials),
 * or as debug views (overdraw, bounding boxes).
 */
export default class MapGeometry {
  public readonly layers: LayerRuntime[] = [];
  // The ground meshes of the tiles with heights (see ElevationService), filled by the WorldTileStreamer. Drawn instead of
  // the ground quad while elevation is on
  public readonly groundBuffer: TerrainLayerBuffer;
  public elevation = false;
  // The bounds of the building meshes, filled by the WorldTileStreamer.
  public readonly meshBounds: Map<Mesh, AABox> = new Map();
  public stats: DrawStats = { meshCount: 0, vertexCount: 0, culledCount: 0 };
  private frustum: Frustum = new Frustum();
  private groundQuad: GroundQuad;
  // The gBuffer materials of the ground quad and the buildings, the layers have theirs in LayerRuntime.material.
  private groundGBufferMaterial: VertexFragmentShader;
  // Has the lightmap charts and the selection highlight.
  private cityGBufferMaterial: VertexFragmentShader;
  private wallTile = new Vector2();
  // The forward materials: per layer (parallel to layers), the ground quad and the buildings (with the selection
  // highlight). The host gives them the shading params and the shadow map, see forwardMaterials.
  private forwardLayerMaterials: VertexFragmentShader[] = [];
  private forwardGroundMaterial: VertexFragmentShader;
  private forwardCityMaterial: VertexFragmentShader;
  private overdrawMaterial: VertexFragmentShader;
  // the building meshes, with their unorm16 vertices
  private overdrawCityMaterial: VertexFragmentShader;
  // the materials of drawMaterialMap: per layer (parallel to layers), and the ground
  private materialMapLayerMaterials: VertexFragmentShader[] = [];
  private materialMapGroundMaterial: VertexFragmentShader;
  private boundingBoxViewer: BoundingBoxViewer;
  // the first palette entry of TREE_COLORS (the trunk, then the crowns)
  private treeMaterial: number;

  constructor(private renderer: Renderer, projection: MapProjection, private gBuffer: MapGBuffer) {
    // at 0, under the ground layers (see LAYER_GAP of worldGeometryCore.ts)
    this.groundQuad = new GroundQuad(renderer, projection, 0);
    // under the camera, as far as the view reaches, see setViewDistanceMax
    this.groundQuad.centerOnCamera = true;
    this.groundBuffer = new TerrainLayerBuffer(renderer);
    this.groundGBufferMaterial = gBuffer.createMaterial(new Vector3(.5, .5, .5), false, 'none', true);
    this.cityGBufferMaterial = gBuffer.createMaterial(new Vector3(1, 1, 1), true);

    // One gBuffer material (with the albedo of the layer) + packed buffer per ground layer (one draw call each).
    for (const def of WORLD_LAYERS) {
      const color = Color.fromHex(def.color);
      const material = gBuffer.createMaterial(color.vector3, false, 'back', SNOW_LAYERS.has(def.key));
      this.layers.push({
        def: { key: def.key, label: def.label, color: [color.r, color.g, color.b], defaultOn: def.defaultOn },
        material,
        buffer: new TerrainLayerBuffer(renderer),
        enabled: null as unknown as ParamBool,
      });
    }

    // The colors of the trees, which the building meshes have (see map/worldTrees.ts): palette entries the city
    // materials pick per vertex.
    const treeMaterial = this.treeMaterial = gBuffer.addPaletteColors(TREE_COLORS);
    this.cityGBufferMaterial.setFloat('treeMaterial', treeMaterial);

    const createForwardMaterial = (color: Vector3, cullMode: GPUCullMode, city = false, snowy = false) => {
      const material = new VertexFragmentShader(renderer, 'osm', forwardShading, {
        includes: city ? [hash] : snowy ? [hash, terrainNormal, snow] : [hash, terrainNormal],
        // SKY_ALBEDO: the sky term with the albedo, like the deferred shading. TREE_COLORS: the albedos of the trees.
        // TERRAIN_NORMALS: the ground's smooth normal from .w, the buildings keep the flat one of the triangle. SNOW: the
        // snow above the snow line, see setSnow
        defines: city ? { FEATURE_IDS: true, SKY_ALBEDO: true, TREE_COLORS: true, QUANTIZED: true }
          : snowy ? { SKY_ALBEDO: true, TERRAIN_NORMALS: true, SNOW: true } : { SKY_ALBEDO: true, TERRAIN_NORMALS: true },
      });
      if (city) {
        material.setBuffer('materialPalette', gBuffer.paletteBuffer);
        material.setFloat('treeMaterial', treeMaterial);
      }
      material.cullMode = cullMode;
      material.setVector3('baseColor', color);
      return material;
    };
    for (const def of WORLD_LAYERS) {
      this.forwardLayerMaterials.push(createForwardMaterial(Color.fromHex(def.color).vector3, 'back', false, SNOW_LAYERS.has(def.key)));
    }
    this.forwardGroundMaterial = createForwardMaterial(new Vector3(.5, .5, .5), 'none', false, true);
    this.forwardCityMaterial = createForwardMaterial(new Vector3(1, 1, 1), 'back', true);
    this.forwardCityMaterial.setFloat('selectedID', -1);
    this.forwardCityMaterial.setFloat('tileID', 0);
    this.forwardCityMaterial.setFloat('drawIndex', 0);

    this.overdrawMaterial = new VertexFragmentShader(renderer, 'overdraw', overdraw);
    this.overdrawCityMaterial = new VertexFragmentShader(renderer, 'overdraw', overdraw, { defines: { QUANTIZED: true } });
    for (const material of [this.overdrawMaterial, this.overdrawCityMaterial]) {
      material.setAdditiveBlending();
      material.depthWriteEnabled = false;
      material.depthCompare = 'always';
    }

    // what drawMaterialMap writes: the palette index of the gBuffer material + 1, and TERRAIN_MATERIAL_SNOW (256) where
    // the snow lies on it, see waterTerrainHorizon.wgsl
    const createMaterialMapMaterial = (gBufferMaterial: VertexFragmentShader, snowy: boolean) => {
      const material = new VertexFragmentShader(renderer, 'terrainMaterialMap', terrainMaterialMap);
      // seen from above, with a mirrored viewProjection
      material.cullMode = 'none';
      material.setFloat('material', gBuffer.materialIndex(gBufferMaterial) + 1 + (snowy ? 256 : 0));
      return material;
    };
    this.materialMapLayerMaterials = this.layers.map(layer => createMaterialMapMaterial(layer.material, SNOW_LAYERS.has(layer.def.key)));
    this.materialMapGroundMaterial = createMaterialMapMaterial(this.groundGBufferMaterial, true);

    this.boundingBoxViewer = new BoundingBoxViewer(renderer);
  }

  /**
   * The ground and its layers with their materials into the current pass, for a map of the materials of the ground top
   * down (TerrainSunMap.update, its viewProjection): the highest wins the depth test. Only with elevation.
   */
  public drawMaterialMap(viewProjection: Matrix4x4): void {
    if (!this.elevation) return;
    this.layers.forEach((layer, i) => {
      if (!layer.enabled.value || layer.buffer.vertexCount === 0) return;
      this.materialMapLayerMaterials[i].setMatrix('viewProjection', viewProjection);
      layer.buffer.draw(this.materialMapLayerMaterials[i]);
    });
    if (this.groundBuffer.vertexCount === 0) return;
    this.materialMapGroundMaterial.setMatrix('viewProjection', viewProjection);
    this.groundBuffer.draw(this.materialMapGroundMaterial);
  }

  /** The palette index of the snow, for what shades the ground itself (WaterTerrainHorizon.setSnow). */
  public get snowMaterial(): number {
    return this.gBuffer.materialIndex(this.layers.find(layer => layer.def.key === SNOW_MATERIAL_LAYER)!.material);
  }

  /** The albedo of a ground layer (index into layers), in the gBuffer and the forward shading. */
  public setLayerColor(index: number, color: Vector3): void {
    this.setMaterialColor(this.layers[index].material, this.forwardLayerMaterials[index], color);
  }

  /** The ground quad reaches distance (world units) around the camera, see MapWorld.viewDistanceMax. */
  public setViewDistanceMax(distance: number): void {
    this.groundQuad.size = distance * 2;
  }

  /** The albedo of the ground quad under the layers. */
  public setGroundColor(color: Vector3): void {
    this.setMaterialColor(this.groundGBufferMaterial, this.forwardGroundMaterial, color);
  }

  /** The albedo of the buildings. */
  public setBuildingColor(color: Vector3): void {
    this.setMaterialColor(this.cityGBufferMaterial, this.forwardCityMaterial, color);
  }

  /** The albedo of the tree trunks. */
  public setTreeTrunkColor(color: Vector3): void {
    this.gBuffer.setPaletteColor(this.treeMaterial, [color.x, color.y, color.z]);
  }

  /**
   * The albedo of the tree crowns: the shades of TREE_CROWN_COLORS keep their variation, scaled per channel so that
   * their average is the color.
   */
  public setTreeCrownColor(color: Vector3): void {
    const avg = [0, 0, 0];
    for (const c of TREE_CROWN_COLORS) for (let i = 0; i < 3; i++) avg[i] += c[i] / TREE_CROWN_COLORS.length;
    const target = [color.x, color.y, color.z];
    TREE_CROWN_COLORS.forEach((c, k) => {
      this.gBuffer.setPaletteColor(this.treeMaterial + 1 + k, [0, 1, 2].map(i => c[i] * target[i] / avg[i]) as [number, number, number]);
    });
  }

  /** The palette entry of a gBuffer material (what the deferred shading reads) and the baseColor of its forward material. */
  private setMaterialColor(gBufferMaterial: VertexFragmentShader, forwardMaterial: VertexFragmentShader, color: Vector3): void {
    this.gBuffer.setPaletteColor(this.gBuffer.materialIndex(gBufferMaterial), [color.x, color.y, color.z]);
    forwardMaterial.setVector3('baseColor', color);
  }

  /**
   * The snow above the snow line on the ground and SNOW_LAYERS, which have the define SNOW: in the gBuffer their pixels
   * in the snow take the material of the glaciers (a hard edge, the TAA antialiases it), the forward shading mixes in
   * params.color over an antialiased edge. Off: the line out of reach, and the gBuffer pipelines without the snow
   * (the override SNOW_IN_SCENE of mapGBuffer.wgsl, see snowPipelinePending).
   */
  public setSnow(params: SnowParams): void {
    const line = params.enabled ? params.line : 1e9;
    const snowMaterial = this.gBuffer.materialIndex(this.layers.find(layer => layer.def.key === SNOW_MATERIAL_LAYER)!.material);
    const gBufferMaterials = this.snowGBufferMaterials;
    for (const m of gBufferMaterials) {
      m.setConstant('SNOW_IN_SCENE', params.enabled);
      m.setFloat('snowMaterial', snowMaterial);
    }
    for (const m of [...gBufferMaterials, ...this.forwardMaterials]) {
      m.setFloat('snowLine', line);
      m.setFloat('snowNoise', params.noise);
      m.setFloat('snowMetersPerUnit', params.metersPerUnit);
    }
    for (const m of this.forwardMaterials) m.setVector3('snowColor', params.color);
  }

  /**
   * Whether the gBuffer pipeline of SNOW_IN_SCENE that setSnow switched to is still being made (async): the frames draw
   * with the one before (with or without the snow) until it is there.
   */
  public get snowPipelinePending(): boolean {
    return this.snowGBufferMaterials.some(m => m.pipelinePending);
  }

  // the gBuffer materials with SNOW: the ground and SNOW_LAYERS
  private get snowGBufferMaterials(): VertexFragmentShader[] {
    return [this.groundGBufferMaterial, ...this.layers.filter(layer => SNOW_LAYERS.has(layer.def.key)).map(layer => layer.material)];
  }

  /** The materials of drawForward: they take the shading params (sky, sun, fog, shadows) by name, like the deferred shading. */
  public get forwardMaterials(): VertexFragmentShader[] {
    return [...this.forwardLayerMaterials, this.forwardGroundMaterial, this.forwardCityMaterial];
  }

  public updateFrustum(camera: MapCamera): void {
    this.frustum.setFromCamera(camera);
  }

  public resetStats(): void {
    this.stats = { meshCount: 0, vertexCount: 0, culledCount: 0 };
  }

  /**
   * The buildings (null: hidden) and the ground (terrain layers and quad) in view, front to back, with their gBuffer
   * materials.
   */
  public drawGBuffer(camera: MapCamera, city: CityMeshes | null, selectedID: number): void {
    if (city) this.drawBuildings(camera, city, selectedID, this.cityGBufferMaterial, true);
    this.drawGround(camera, this.layers.map(layer => layer.material), this.groundGBufferMaterial);
  }

  /**
   * Like drawGBuffer, forward shaded into the current pass (the host starts it). The shading params and the shadow
   * map are on the materials already, see forwardMaterials.
   */
  public drawForward(camera: MapCamera, city: CityMeshes | null, selectedID: number): void {
    for (const material of this.forwardMaterials) {
      material.setMatrix('viewProjection', camera.viewProjection);
      material.setVector3('cameraPosition', camera.worldPosition);
    }
    if (city) this.drawBuildings(camera, city, selectedID, this.forwardCityMaterial);
    this.drawGround(camera, this.forwardLayerMaterials, this.forwardGroundMaterial);
  }

  /** The layers top first (layerMaterials parallel to layers), the ground (quad or meshes) under them last. */
  private drawGround(camera: MapCamera, layerMaterials: VertexFragmentShader[], groundMaterial: VertexFragmentShader): void {
    const stats = this.stats;
    for (let j = this.layers.length - 1; j >= 0; j--) {
      const layer = this.layers[j];
      if (!layer.enabled.value || layer.buffer.vertexCount === 0) continue;
      layer.buffer.draw(layerMaterials[j]);
      stats.meshCount++;
      stats.vertexCount += layer.buffer.vertexCount;
    }
    if (this.elevation) {
      if (this.groundBuffer.vertexCount === 0) return;
      this.groundBuffer.draw(groundMaterial);
      stats.meshCount++;
      stats.vertexCount += this.groundBuffer.vertexCount;
      return;
    }
    this.groundQuad.drawGBuffer(camera, groundMaterial);
    stats.meshCount++;
    stats.vertexCount += this.groundQuad.vertexCount;
  }

  /** The meshes in the frustum, sorted by the distance of their bounds to the eye. */
  // wallTile: the material has the uniform wallTile (the gBuffer)
  private drawBuildings(camera: MapCamera, city: CityMeshes, selectedID: number, material: VertexFragmentShader, wallTile = false): void {
    const stats = this.stats;
    material.setFloat('selectedID', selectedID);
    const highlight = selectedID >= 0;
    const { meshes, slots, tiles } = city;
    const visible: { index: number; distance: number }[] = [];
    const eye = camera.worldPosition;
    for (let i = 0; i < meshes.length; i++) {
      const bounds = this.meshBounds.get(meshes[i]);
      if (bounds && !this.frustum.aaBoxInFrustum(bounds)) {
        stats.culledCount++;
        continue;
      }
      let distance = 0;
      if (bounds) {
        const dx = Math.max(bounds.min.x - eye.x, 0, eye.x - bounds.max.x);
        const dy = Math.max(bounds.min.y - eye.y, 0, eye.y - bounds.max.y);
        const dz = Math.max(bounds.min.z - eye.z, 0, eye.z - bounds.max.z);
        distance = dx * dx + dy * dy + dz * dz;
      }
      visible.push({ index: i, distance });
    }
    visible.sort((a, b) => a.distance - b.distance);
    for (const { index } of visible) {
      const mesh = meshes[index];
      if (highlight) material.setFloat('tileID', slots[index]);
      if (wallTile) material.setVector2('wallTile', this.wallTile.setValues(tiles[index * 2], tiles[index * 2 + 1]));
      setMeshQuantization(material, city.quantization, index);
      stats.meshCount++;
      stats.vertexCount += mesh.vertexCount;
      this.renderer.draw(mesh, material, { indexCount: city.indexCounts[index] });
    }
  }

  /** DrawMode.Overdraw: everything additively with the overdraw material, straight to the screen. */
  public drawOverdraw(camera: MapCamera, city: CityMeshes | null): void {
    this.overdrawMaterial.setMatrix('viewProjection', camera.viewProjection);
    this.overdrawCityMaterial.setMatrix('viewProjection', camera.viewProjection);
    this.renderer.startRenderPass();
    if (this.elevation) this.groundBuffer.draw(this.overdrawMaterial);
    else this.groundQuad.draw(camera, this.overdrawMaterial);
    for (const layer of this.layers) {
      if (!layer.enabled.value || layer.buffer.vertexCount === 0) continue;
      layer.buffer.draw(this.overdrawMaterial);
    }
    if (city) {
      city.meshes.forEach((mesh, i) => {
        const bounds = this.meshBounds.get(mesh);
        if (bounds && !this.frustum.aaBoxInFrustum(bounds)) return;
        setMeshQuantization(this.overdrawCityMaterial, city.quantization, i);
        this.renderer.draw(mesh, this.overdrawCityMaterial, { indexCount: city.indexCounts[i] });
      });
    }
  }

  /** In the current pass, without the depth of the scene: a debug view. */
  public drawBoundingBoxes(viewProjection: Matrix4x4, city: CityMeshes | null): void {
    const entries: BBoxEntry[] = [];
    const red: [number, number, number] = [.5, 0, 0];
    if (city) {
      for (const mesh of city.meshes) {
        const b = this.meshBounds.get(mesh);
        if (b) entries.push({ min: [b.min.x, b.min.y, b.min.z], max: [b.max.x, b.max.y, b.max.z], color: red });
      }
    }
    this.boundingBoxViewer.draw(viewProjection, entries);
  }

  destruct(): void {
    this.groundQuad.destruct();
    this.groundBuffer.destruct();
    for (const layer of this.layers) layer.buffer.destruct();
    this.boundingBoxViewer.destruct();
  }
}
