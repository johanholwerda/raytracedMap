import Renderer from 'webGPU-framework/renderer/render/Renderer';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import ProjectionOrthographic from 'webGPU-framework/renderer/camera/ProjectionOrthographic';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import Color from 'webGPU-framework/renderer/math/Color';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Transform from 'webGPU-framework/renderer/scene/Transform';
import TerrainLayerBuffer from './osm/TerrainLayerBuffer';

/**
 * The ground as the sun sees it, for the ray traced sun shadows: their rays trace only the buildings and the trees of
 * CityScene, so without this a hill casts no shadow while the trees on it do. One orthographic view down the sun,
 * centered under the camera and as large as the view distance, of the ground meshes (MapGeometry.groundBuffer) with the
 * depth material of the cascades with SUN_DEPTH: per texel how far toward the sun the ground nearest to it is. The
 * shading compares that of a point with it (getTerrainSunShadow of terrainSun.wgsl). Not the height, as the cascades:
 * a point compared with the texels next to its own would be up to a texel higher or lower than them, which at a low sun
 * is many texels along it, and the slopes shadowed themselves in stripes. The ground is about 30 m per elevation sample and
 * smooth, a map of it is as good as tracing it.
 * Rendered again only when the sun moves, the ground tiles change or the camera drifts off its center.
 */
export default class TerrainSunMap {
  private static readonly RESOLUTION = 2048;
  // the half-extent of the map is this many times the view distance: the ground in view stays in it while the camera
  // drifts up to REDRAW_DRIFT of the view distance from its center
  private static readonly SIZE_SCALE = 1.25;
  private static readonly REDRAW_DRIFT = 0.2;
  // the offset of the point off its surface along the normal, in texels, see getTerrainSunShadow
  private static readonly NORMAL_OFFSET_TEXELS = 1.5;
  // world units of depth past the ground footprint along the sun: the casters in front of the near plane are clamped onto
  // it (DEPTH_CLAMP of shading.wgsl), the far plane only has to be past the points it shades
  private static readonly DEPTH_MARGIN = 1;
  // no ground: far away from the sun, lit
  private static readonly CLEAR = new Color(-1e9, 0, 0, 0);

  private readonly target: RenderTexture;
  private readonly camera: Camera;
  private readonly pivot = new Transform();
  private readonly materials: ShaderBase[] = [];
  private center = new Vector3(1e9, 1e9, 1e9);
  private size = 0;
  private lightKey = '';
  private dirty = true;

  constructor(private renderer: Renderer, private depthMaterial: VertexFragmentShader) {
    const resolution = TerrainSunMap.RESOLUTION;
    this.target = new RenderTexture(renderer, resolution, resolution, { formats: ['r32float'], depth: true });
    this.camera = new Camera({ nearPlane: -1, farPlane: 1, orthographic: true, fov: 0.5 });
    this.camera.parent = this.pivot;
  }

  /** The map on a material that reads it with getTerrainSunShadow: its texture now, its matrix on every render. */
  public bind(material: ShaderBase): void {
    material.setTexture('terrainSunMap', this.target);
    this.materials.push(material);
    this.setUniforms(material);
  }

  /** Render again in the next update: the ground tiles changed. */
  public setDirty(): void {
    this.dirty = true;
  }

  /**
   * Renders the map again when it is stale. lightDirection: toward the sun. center: under the camera. viewDistance: how
   * far from it (world units, xz) the shading needs it. ground null: no elevation, the map is cleared (all lit).
   */
  public update(lightDirection: Vector3, center: Vector3, viewDistance: number, ground: TerrainLayerBuffer | null): void {
    const lightKey = `${lightDirection.x.toFixed(4)},${lightDirection.y.toFixed(4)},${lightDirection.z.toFixed(4)}`;
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
    this.render(lightDirection, ground);
  }

  private render(toSun: Vector3, ground: TerrainLayerBuffer | null): void {
    const size = this.size;
    // the center snapped to the texel grid across the sun, as the cascades do (ShadowMap.renderCascade): a render around
    // a new center puts the ground on the same texels, the edges of the shadows don't move
    const center = this.center;
    this.pivot.lookAt(center, Vector3.subtract(center, toSun));
    const texel = (2 * size) / TerrainSunMap.RESOLUTION;
    const snapped = center.clone();
    for (const axis of [this.pivot.right, this.pivot.up]) {
      const d = Vector3.dot(center, axis);
      snapped.add(axis.multiplyScalar(Math.round(d / texel) * texel - d));
    }
    this.pivot.lookAt(snapped, Vector3.subtract(snapped, toSun));

    // the depth range: the ground within size of the center (√2 size at the corners), along the sun at most that
    const extent = size * Math.SQRT2 + TerrainSunMap.DEPTH_MARGIN;
    const projection = this.camera.projection as ProjectionOrthographic;
    projection.setSize(size, size);
    projection.near = -extent;
    projection.far = extent;

    this.renderer.startRenderPass(this.target, { clearColor: TerrainSunMap.CLEAR });
    if (ground) {
      this.depthMaterial.setMatrix('viewProjection', this.camera.viewProjection);
      this.depthMaterial.setVector3('lightDirection', toSun);
      ground.draw(this.depthMaterial);
    }
    for (const m of this.materials) this.setUniforms(m);
  }

  private setUniforms(material: ShaderBase): void {
    material.setMatrix('terrainSunViewProjection', this.camera.viewProjection);
    material.setFloat('terrainSunOffset', TerrainSunMap.NORMAL_OFFSET_TEXELS * (2 * this.size) / TerrainSunMap.RESOLUTION);
  }

  public destruct(): void {
    this.target.destruct();
  }
}
