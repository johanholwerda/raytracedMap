import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import Color from 'webGPU-framework/renderer/math/Color';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import TerrainLayerBuffer from './osm/TerrainLayerBuffer';
import heightShaderSource from './terrainSunHeight.wgsl?raw';

/**
 * The materials of the ground for TerrainSunMap.update: draw draws the ground and its layers with the viewProjection of
 * the map, a material each (MapGeometry.drawMaterialMap). top and bottom: the highest and the lowest ground, world y.
 */
export type TerrainMaterials = {
  draw: (viewProjection: Matrix4x4) => void;
  top: number;
  bottom: number;
};

/**
 * The hills in the way of the sun, for the ray traced sun shadows: their rays trace only the buildings and the trees of
 * CityScene, so without this a hill casts no shadow while the trees on it do. A map of the ground top down, centered
 * under the camera and as large as the view distance, per texel the height a point there has to be at to see the sun
 * (terrainSunHeight.wgsl); the shading compares the height of a point with it (getTerrainSunShadow of terrainSun.wgsl).
 * That height is continuous over the map, so interpolated the edge of a shadow is a smooth curve. A depth map from the
 * sun (as this was before) jumps at every ridge, its edges are the texel grid, stretched to long stairs at a low sun.
 * The passes: the ground meshes (MapGeometry.groundBuffer) top down with the depth material of the cascades (-height),
 * the highest ground per cell of 4, 16 (COARSE), 64 and all texels, and the march toward the sun per texel, which skips
 * the cells of COARSE below the ray and stops where it is above all the ground.
 * Rendered again only when the sun moves, the ground tiles change or the camera drifts off its center.
 */
export default class TerrainSunMap {
  private static readonly RESOLUTION = 1024;
  // texels per cell of the highest ground, mirrors COARSE of terrainSunHeight.wgsl
  private static readonly COARSE = 16;
  // and of the first level of it, which the terrain in the water reflections skips by too (TERRAIN_FINE of
  // terrainReflection.wgsl)
  private static readonly FINE = 4;
  // the half-extent of the map is this many times the view distance: the ground in view stays in it while the camera
  // drifts up to REDRAW_DRIFT of the view distance from its center
  private static readonly SIZE_SCALE = 1.25;
  private static readonly REDRAW_DRIFT = 0.2;
  // the offset of the point off its surface along the normal, in texels, see getTerrainSunShadow
  private static readonly NORMAL_OFFSET_TEXELS = 0.25;
  // the least half width of the penumbra: the rise of a ray toward the sun over this many texels, an edge about twice
  // this wide across the ground behind a hill. Below a texel the interpolation of the map shows
  private static readonly PENUMBRA_MIN_TEXELS = 0.5;
  // no ground: -height far below anything
  private static readonly CLEAR = new Color(1e9, 0, 0, 0);

  // the ground top down: -height, and the depth to keep the highest
  private readonly heights: RenderTexture;
  // the highest ground per cell of 4, 16 (COARSE), 64 and all texels of heights, each level from the one before
  private readonly maxHeights: RenderTexture[] = [];
  // the material of the ground top down (terrainMaterialMap.wgsl), for the color of the ground in the water
  // reflections: only drawn with the materials of update
  private readonly materialMap: RenderTexture;
  private readonly materialViewProjection = new Matrix4x4();
  private hasMaterials = false;
  private static readonly MATERIAL_CLEAR = new Color(0, 0, 0, 0);
  // the height to see the sun at and the penumbra, what the shading reads
  private readonly target: RenderTexture;
  private readonly maxHeightsMaterials: VertexFragmentShader[] = [];
  private readonly shadowHeightMaterial: VertexFragmentShader;
  private readonly viewProjection = new Matrix4x4();
  private readonly materials: ShaderBase[] = [];
  // the materials that march the height map itself, see bindHeights
  private readonly heightsMaterials: ShaderBase[] = [];
  private center = new Vector3(1e9, 1e9, 1e9);
  private size = 0;
  private lightKey = '';
  private dirty = true;

  /** heightMaterial: the depth material of the cascades (DRAW_DEPTH, DEPTH_CLAMP, no culling), it writes -height. */
  constructor(private renderer: Renderer, private heightMaterial: VertexFragmentShader) {
    const resolution = TerrainSunMap.RESOLUTION;
    this.heights = new RenderTexture(renderer, resolution, resolution, { formats: ['r32float'], depth: true, depthFormat: 'depth32float' });
    this.target = new RenderTexture(renderer, resolution, resolution, { formats: ['rg32float'] });
    this.materialMap = new RenderTexture(renderer, resolution, resolution, { formats: ['r32uint'], depth: true, depthFormat: 'depth32float' });

    // 1024 / 4 = 256 / 4 = 64 (COARSE: 16 texels) / 4 = 16 / 16 = 1
    let source = this.heights;
    let sourceSize = resolution;
    for (const factor of [4, 4, 4, sourceSize / 64]) {
      const size = sourceSize / factor;
      const level = new RenderTexture(renderer, size, size, { formats: ['r32float'] });
      const material = new VertexFragmentShader(renderer, 'terrainSunMaxHeights', heightShaderSource, { defines: { MAX_HEIGHTS: true } });
      material.depthWriteEnabled = false;
      material.setTexture('source', source);
      material.setFloat('factor', factor);
      // the first level: a texel around its block too (see terrainSunHeight.wgsl), from -height
      material.setFloat('apron', source === this.heights ? 1 : 0);
      material.setFloat('sign', source === this.heights ? -1 : 1);
      this.maxHeights.push(level);
      this.maxHeightsMaterials.push(material);
      source = level;
      sourceSize = size;
    }
    if (resolution / TerrainSunMap.COARSE !== this.maxHeights[1].width || resolution / TerrainSunMap.FINE !== this.maxHeights[0].width ||
      this.maxHeights[3].width !== 1) {
      throw new Error('TerrainSunMap: the levels of maxHeights do not fit RESOLUTION');
    }
    this.shadowHeightMaterial = new VertexFragmentShader(renderer, 'terrainSunShadowHeight', heightShaderSource, { defines: { SHADOW_HEIGHT: true } });
    this.shadowHeightMaterial.depthWriteEnabled = false;
    this.shadowHeightMaterial.setTexture('heights', this.heights);
    this.shadowHeightMaterial.setTexture('maxHeights', this.maxHeights[1]);
    this.shadowHeightMaterial.setTexture('maxHeight', this.maxHeights[3]);
  }

  /** The map on a material that reads it with getTerrainSunShadow: its texture now, its matrix on every render. */
  public bind(material: ShaderBase): void {
    material.setTexture('terrainSunMap', this.target);
    this.materials.push(material);
    this.setUniforms(material);
  }

  /**
   * The height map itself on a material that marches it (isTerrainOccluded of terrainReflection.wgsl, the terrain in the
   * water reflections): its textures now, its matrix on every render.
   */
  public bindHeights(material: ShaderBase): void {
    material.setTexture('terrainHeights', this.heights);
    material.setTexture('terrainMaxHeights', this.maxHeights[1]);
    material.setTexture('terrainMaxHeightsFine', this.maxHeights[0]);
    this.heightsMaterials.push(material);
    material.setMatrix('terrainSunViewProjection', this.viewProjection);
  }

  /**
   * The materials of the ground on the same map (materials of update) on a material that shades the ground from it
   * (waterTerrainHorizon.wgsl): the texture now.
   */
  public bindMaterials(material: ShaderBase): void {
    material.setTexture('terrainMaterials', this.materialMap);
  }

  /** Render again in the next update: the ground tiles changed. */
  public setDirty(): void {
    this.dirty = true;
  }

  /**
   * Renders the map again when it is stale. lightDirection: toward the sun. center: under the camera. viewDistance: how
   * far from it (world units, xz) the shading needs it. sunRadius: the angular radius of the sun (radians), the
   * penumbra. ground null: no elevation, nothing in the way. materials: the materials of the ground are drawn too, for
   * bindMaterials, null when nothing reads them.
   */
  public update(lightDirection: Vector3, center: Vector3, viewDistance: number, sunRadius: number, ground: TerrainLayerBuffer | null,
    materials: TerrainMaterials | null = null): void {
    // they weren't drawn with the map there is
    if (materials && !this.hasMaterials) this.dirty = true;
    const lightKey = `${lightDirection.x.toFixed(4)},${lightDirection.y.toFixed(4)},${lightDirection.z.toFixed(4)},${sunRadius}`;
    if (lightKey !== this.lightKey) {
      this.lightKey = lightKey;
      this.dirty = true;
    }
    const size = viewDistance * TerrainSunMap.SIZE_SCALE;
    if (size !== this.size) this.dirty = true;
    const dx = center.x - this.center.x;
    const dz = center.z - this.center.z;
    if (Math.sqrt(dx * dx + dz * dz) > viewDistance * TerrainSunMap.REDRAW_DRIFT) this.dirty = true;
    if (!this.dirty) return;
    this.dirty = false;
    this.size = size;
    this.center = center.clone();
    this.render(lightDirection, sunRadius, ground, materials);
  }

  private render(toSun: Vector3, sunRadius: number, ground: TerrainLayerBuffer | null, materials: TerrainMaterials | null): void {
    const size = this.size;
    const texel = (2 * size) / TerrainSunMap.RESOLUTION;
    // the center snapped to the texel grid, as the cascades do (ShadowMap.renderCascade): a render around a new center
    // puts the ground on the same texels, the edges of the shadows don't move
    const cx = Math.round(this.center.x / texel) * texel;
    const cz = Math.round(this.center.z / texel) * texel;
    // top down: clip x along +x, clip y along -z (texture v runs down, so v along +z), depth 0 at top, growing down. The
    // ground above top is clamped onto it (DEPTH_CLAMP), so top is far above any: the highest ground has to win the depth
    // test, a tie at 0 would keep the first one drawn
    const top = this.center.y + 100 * size;
    const range = 200 * size;
    this.viewProjection.setValues(
      1 / size, 0, 0, 0,
      0, 0, -1 / range, 0,
      0, -1 / size, 0, 0,
      -cx / size, cz / size, top / range, 1,
    );

    this.renderer.startRenderPass(this.heights, { clearColor: TerrainSunMap.CLEAR });
    if (ground) {
      this.heightMaterial.setMatrix('viewProjection', this.viewProjection);
      ground.draw(this.heightMaterial);
    }
    this.maxHeights.forEach((level, i) => this.renderer.blit(null, level, this.maxHeightsMaterials[i]));

    this.hasMaterials = materials != null;
    if (materials) {
      // the same map, with the depth from the highest to the lowest ground: the layers are centimeters above each other
      // (LAYER_GAP of worldGeometryCore.ts), less than the depth of the height map above tells apart
      const margin = 0.01 + 0.01 * Math.max(materials.top - materials.bottom, 0);
      const materialTop = materials.top + margin;
      const materialRange = materials.top - materials.bottom + 2 * margin;
      this.materialViewProjection.setValues(
        1 / size, 0, 0, 0,
        0, 0, -1 / materialRange, 0,
        0, -1 / size, 0, 0,
        -cx / size, cz / size, materialTop / materialRange, 1,
      );
      this.renderer.startRenderPass(this.materialMap, { clearColor: TerrainSunMap.MATERIAL_CLEAR, colorClear: true, depthClear: true });
      if (ground && Number.isFinite(materialRange)) materials.draw(this.materialViewProjection);
    }

    // the sun across the map in texels: a texel is a step of texel world units in x or z
    const horizontal = Math.hypot(toSun.x, toSun.z);
    const m = this.shadowHeightMaterial;
    m.setVector2('sunDirection', horizontal < 1e-5 ? new Vector2(0, 0) : new Vector2(toSun.x / horizontal, toSun.z / horizontal));
    const rise = horizontal < 1e-5 ? 0 : (toSun.y / horizontal) * texel;
    m.setFloat('risePerTexel', rise);
    // a sun higher or lower by sunRadius changes the rise over s texels by about s * texel * sunRadius / cos^2
    const cos = Math.max(horizontal / toSun.length(), 0.05);
    m.setFloat('penumbraPerTexel', (texel * sunRadius) / (cos * cos));
    m.setFloat('penumbraMin', Math.max(rise, 0) * TerrainSunMap.PENUMBRA_MIN_TEXELS + texel * 1e-3);
    this.renderer.blit(null, this.target, m);

    for (const material of this.materials) this.setUniforms(material);
    for (const material of this.heightsMaterials) material.setMatrix('terrainSunViewProjection', this.viewProjection);
  }

  private setUniforms(material: ShaderBase): void {
    material.setMatrix('terrainSunViewProjection', this.viewProjection);
    material.setFloat('terrainSunOffset', TerrainSunMap.NORMAL_OFFSET_TEXELS * (2 * this.size) / TerrainSunMap.RESOLUTION);
  }

  public destruct(): void {
    this.heights.destruct();
    for (const level of this.maxHeights) level.destruct();
    this.target.destruct();
    this.materialMap.destruct();
    for (const material of this.maxHeightsMaterials) material.destruct();
    this.shadowHeightMaterial.destruct();
  }
}
