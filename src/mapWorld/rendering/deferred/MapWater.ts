import type Renderer from 'webGPU-framework/renderer/render/Renderer';
import type Camera from 'webGPU-framework/renderer/camera/Camera';
import type VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import type ParamBool from 'webGPU-framework/utils/params/ParamBool';
import type ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import type ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import type ParamSelect from 'webGPU-framework/utils/params/ParamSelect';
import type CityScene from '../../city/CityScene';
import type TerrainHeightField from '../../map/TerrainHeightField';
import {
  type QualitySettings, REFLECTION_BLOCKS, REFLECTION_RAYS, REFLECTION_TYPES, ReflectionRays, ReflectionType,
} from '../../MapWorldSettings';
import type MapGeometry from '../MapGeometry';
import type DeferredShading from './DeferredShading';
import type MapGBuffer from './MapGBuffer';
import WaterReflectionHalfRes from './WaterReflectionHalfRes';
import WaterTerrainHorizon from './WaterTerrainHorizon';
import WaterRippleTexture from './waterRipples/WaterRippleTexture';

// The water reflects the ground (updateTerrainReflection) only where the highest loaded ground is this much above the
// lowest, m: below it no hill is high enough to show in the water, and the rays skip the height map
const TERRAIN_REFLECTION_RELIEF_M = 20;
// the rays start this far above the water, m: the water lies on the ground of the height map, see traceTerrain
const TERRAIN_REFLECTION_BIAS_M = .5;

/** What the water does in a frame, see MapWater.plan. */
export type WaterFrame = {
  // the buildings in the water (they need the BVH of the lightmap)
  reflectBuildings: boolean;
  // the water reflects anything: the buildings, or the sky and the sun on the ripples
  reflections: boolean;
  // the reflections are mirrored on the ripples
  rippling: boolean;
  // the ripples move: the map is drawn every frame
  animating: boolean;
};

/**
 * The water of the deferred shading: the reflections of the buildings (traced in the city of the lightmap, per pixel
 * in the shading or per block of pixels in WaterReflectionHalfRes), the ground in them (WaterTerrainHorizon), the
 * animated ripples they are mirrored on (WaterRippleTexture) and the sun in the water. With their params.
 */
export default class MapWater {
  // what the buildings in the water look like (none: not reflected), see ReflectionType and waterReflectionColor.wgsl
  private reflectionType!: ParamSelect;
  // the water pixels per ray of those reflections, see ReflectionRays
  private reflectionRays!: ParamSelect;
  // traces them when that is more than 1 pixel, before the shading
  private waterReflection: WaterReflectionHalfRes;
  // the ground in the reflections of the buildings, where it is in reach: see updateTerrainReflection. From its horizon
  // over the water, per block of pixels
  private terrainHorizon: WaterTerrainHorizon;
  private reflectTerrain!: ParamBool;
  private reflectTerrainInWater = false;
  private reflectionDarkening!: ParamFloat;
  // the water mirrored on the ripples with the sun in it, also with reflectionType None (no buildings)
  private waterRipples!: ParamBool;
  // The animated micro normal the reflections are mirrored on, redrawn every frame. Its wavelength is in meters.
  private ripples: WaterRippleTexture;
  // the slope of the ripples, 0 is a flat mirror (and no animation)
  private rippleSlope!: ParamFloat;
  private rippleSpeed!: ParamFloat;
  // the strength of the ripples 4 times smaller laid over them close up, see getRippleNormal of waterReflection.wgsl
  private rippleDetail!: ParamFloat;
  // the ripples move: the map is drawn every frame. Off, they are frozen and a still view stops being drawn
  private animateWater!: ParamBool;
  // seconds the ripples moved, only counts while animateWater is on
  private rippleTime: number = 0;
  // the ripple texture needs a redraw: a param changed (or it was never drawn)
  private ripplesDirty: boolean = true;
  private sunReflection!: ParamFloat;
  private sunReflectionPower!: ParamFloat;
  // the gBuffer materials of the water layers, a bit each
  private waterMaterials: number = 0;

  constructor(renderer: Renderer, private deferred: DeferredShading, gBuffer: MapGBuffer, geometry: MapGeometry, cityScene: CityScene,
    private heightField: TerrainHeightField) {
    this.ripples = new WaterRippleTexture(renderer);
    this.ripples.wavelength = 10;
    this.deferred.bindWaterRipples(this.ripples);
    this.waterReflection = new WaterReflectionHalfRes(renderer);
    this.waterReflection.bindCity(cityScene);
    this.waterReflection.bindWaterRipples(this.ripples);
    this.terrainHorizon = new WaterTerrainHorizon(renderer, gBuffer);
    this.deferred.bindTerrainHorizon(this.terrainHorizon);
    for (const layer of geometry.layers) {
      if (layer.def.key === 'water' || layer.def.key === 'waterways') this.waterMaterials |= 1 << gBuffer.materialIndex(layer.material);
    }
  }

  /** The materials that shade the ground in the water reflections, for the shading params and the sun. */
  public get materials(): VertexFragmentShader[] {
    return this.terrainHorizon.materials;
  }

  /**
   * The materials of the reduced pass of the reflections: they read the lightmap (with its sun) for the light of what
   * the rays hit, take the shading params by name (the light, the fog) and the albedos of the buildings and trees
   * (reflectionBuildingAlbedo, reflectionTreeAlbedo).
   */
  public get reflectionMaterials(): VertexFragmentShader[] {
    return this.waterReflection.materials;
  }

  // the buildings are in the water (a reflectionType that shows them)
  private get reflectsBuildings(): boolean {
    return this.reflectionType.value !== ReflectionType.None;
  }

  /** See MapSnow. */
  public get horizon(): WaterTerrainHorizon {
    return this.terrainHorizon;
  }

  /** Whether the water reflections have ground in them, see updateTerrainReflection. */
  public get reflectsTerrain(): boolean {
    return this.reflectTerrainInWater;
  }

  /**
   * The Water group.
   * @param shadingParams the options of the params that go to the shading materials by name
   */
  public initParams(pgShading: ParamGroup, shadingParams: { materials: VertexFragmentShader[]; callback: () => void }, setDirty: () => void): void {
    const waterGroup = pgShading.addGroup('Water');

    this.reflectTerrain = waterGroup.addParamBool('reflectTerrain', true, { callback: setDirty });
    this.reflectionRays = waterGroup.addParamSelect('reflectionRays', ReflectionRays.PerPixel, REFLECTION_RAYS, { callback: setDirty });
    this.reflectionDarkening = waterGroup.addParamFloat('reflectionDarkening', .5, 0, 1, { callback: setDirty });
    // Only with the sky visibility lightmap: it keeps the BVH of the buildings up to date.
    // the buildings and trees in the water: none, only darkening it, in their colors shaded as a face the lightmap has
    // no texels of (reflectionUnseen), or lit by the lightmap where the rays hit them
    this.reflectionType = waterGroup.addParamSelect('reflectionType', ReflectionType.Lightmap, REFLECTION_TYPES, { callback: setDirty });
    // how much of the darkening the color of a building gives back, see deferredShading.wgsl
    waterGroup.addParamFloat('reflectionColor', 1, 0, 1, shadingParams);
    // the sky light of a building in the water that the lightmap has no texels of (an underside, a side away from the
    // camera), a part of the open sky's: in place of the occlusion of the lightmap. Its sun is by the normal, see
    // getUnseenReflectionColor
    waterGroup.addParamFloat('reflectionUnseen', .5, 0, 1, shadingParams);
    // the lightmap levels the reflections read coarser than the screen claims for the point they hit (within the
    // ancestor levels of the lightmap), see getReflectionLevel
    waterGroup.addParamFloat('reflectionLevelBias', 0, 0, 4, shadingParams);
    this.waterRipples = waterGroup.addParamBool('waterRipples', true, { callback: setDirty });
    const updateRipples = () => {
      // the bake scales the gradient by the height, which makes the slope of the ripples independent of their wavelength
      this.ripples.material.setFloat('rippleStrength', this.rippleSlope.value * this.ripples.wavelength);
      this.ripples.material.setFloat('speed', this.rippleSpeed.value);
      this.ripplesDirty = true;
      setDirty();
    };
    this.rippleSlope = waterGroup.addParamFloat('rippleSlope', .05, 0, .5, { callback: updateRipples });
    waterGroup.addParamFloat('rippleWavelength', this.ripples.wavelength, 1, 50, { callback: (v) => {
      this.ripples.wavelength = v;
      updateRipples();
    } });
    this.rippleSpeed = waterGroup.addParamFloat('rippleSpeed', .5, 0, 2, { callback: updateRipples });
    this.rippleDetail = waterGroup.addParamFloat('rippleDetail', .8, 0, 1, shadingParams);
    this.animateWater = waterGroup.addParamBool('animateWater', true, { callback: setDirty });
    updateRipples();
    this.sunReflection = waterGroup.addParamFloat('sunReflection', 8, 0, 16, { callback: setDirty });
    this.sunReflectionPower = waterGroup.addParamFloat('sunReflectionPower', 200, 10, 1000, { callback: setDirty });

    // the fresnel term of the water: the sky it mirrors (scales the fresnel weight) and the glints at grazing angles
    waterGroup.addParamFloat('skyReflection', 1, 0, 4, shadingParams);
    waterGroup.addParamFloat('sunFresnel', 4, 0, 16, shadingParams);
    waterGroup.collapsed = true;
  }

  /** The water params of a quality (which the GUI follows). */
  public applyQuality(settings: QualitySettings): void {
    this.reflectionType.setValue(settings.reflectionType);
    this.reflectionRays.setValue(settings.reflectionRays);
    this.waterRipples.setValue(settings.waterRipples);
    this.animateWater.setValue(settings.animateWater);
  }

  /**
   * Whether the water reflections can have ground in them, into their shaders (reflectTerrainInWater): the rays of the
   * water march the height map of the ground only where the loaded ground has relief, which most cities on the water
   * don't. With the highest ground, above which a ray ends.
   * @returns whether that switched: the map draws again
   */
  public updateTerrainReflection(lightmapActive: boolean, elevation: boolean, metersToNorm: number): boolean {
    const field = this.heightField;
    const enabled = this.reflectTerrain.value && this.reflectsBuildings && lightmapActive && elevation &&
      field.reliefM > TERRAIN_REFLECTION_RELIEF_M;
    const changed = enabled !== this.reflectTerrainInWater;
    this.reflectTerrainInWater = enabled;
    const top = enabled ? field.maxHeightM * metersToNorm : 0;
    const bias = TERRAIN_REFLECTION_BIAS_M * metersToNorm;
    this.deferred.setTerrainReflection(enabled, top, bias);
    this.terrainHorizon.setTerrain(top, bias);
    return changed;
  }

  /**
   * What the water does in this frame.
   * @param useSceneColor the deferred shading draws (not the overdraw view)
   * @param lightmapActive the lightmap is on: the buildings in the water need its BVH, the ripples and the sun don't
   */
  public plan(useSceneColor: boolean, lightmapActive: boolean): WaterFrame {
    const reflectBuildings = this.reflectsBuildings && lightmapActive;
    const reflections = useSceneColor && (reflectBuildings || this.waterRipples.value);
    const rippling = reflections && this.waterRipples.value && this.rippleSlope.value > 0;
    const animating = rippling && this.animateWater.value && this.rippleSpeed.value > 0;
    return { reflectBuildings, reflections, rippling, animating };
  }

  /** The ripples move on (while animated) and are baked again when the map draws and they changed. */
  public updateRipples(frame: WaterFrame, renderMap: boolean, dt: number): void {
    if (frame.animating) this.rippleTime += dt;
    if (renderMap && frame.rippling && (frame.animating || this.ripplesDirty)) {
      this.ripples.update(this.rippleTime);
      this.ripplesDirty = false;
    }
  }

  /**
   * Before the shading: the passes of the reflections, and their params into the shading.
   * @param rotate the traced pixel of the blocks rotates, for the TAA to average (in a static view only)
   * @param lightmapReadable the lightmap has its buffers (SkyVisibilityLightmap.readable): what the rays hit can be lit by it
   */
  public draw(frame: WaterFrame, camera: Camera, gBuffer: MapGBuffer, metersToNorm: number, rotate: boolean, lightmapReadable: boolean): void {
    const { reflections, reflectBuildings, rippling } = frame;
    const rippleTile = rippling ? this.ripples.tileSize * metersToNorm : 0;
    // The rays of the buildings in the water: 1 per block of pixels, traced here for the shading to blend, or a ray per
    // water pixel in the shading. The traced pixel of the blocks rotates for the TAA to average, in a static view only:
    // while the water or the camera moves the TAA blends its history like motion, and the rotation shows as jitter.
    const [blockX, blockY] = REFLECTION_BLOCKS[this.reflectionRays.value];
    const reflectionPass = reflections && reflectBuildings && blockX * blockY > 1 ? this.waterReflection : null;
    // the ground in them: its horizon over the water, which the rays are compared with
    if (reflections && reflectBuildings && this.reflectTerrainInWater) this.terrainHorizon.draw(camera, gBuffer, this.waterMaterials);
    else this.terrainHorizon.release();
    // their colors, lit by the lightmap where they hit
    const reflectionType = this.reflectionType.value as ReflectionType;
    const reflectionColors = reflections && reflectBuildings && reflectionType !== ReflectionType.SolidColor;
    const litReflections = reflectionColors && reflectionType === ReflectionType.Lightmap && lightmapReadable;
    this.deferred.setReflectionColors(reflectionColors, litReflections);
    if (reflectionPass) {
      reflectionPass.setBlock(blockX, blockY);
      reflectionPass.draw(camera, gBuffer, this.waterMaterials, rippleTile, rotate, reflectionColors, litReflections);
    } else this.waterReflection.release();
    this.deferred.setWaterReflections(reflections ? this.waterMaterials : 0, reflectBuildings, this.reflectionDarkening.value, rippleTile,
      this.sunReflection.value, this.sunReflectionPower.value, this.waterReflection, reflectionPass != null);
  }

  destruct(): void {
    this.ripples.texture.destruct();
    this.waterReflection.destruct();
    this.terrainHorizon.destruct();
  }
}
