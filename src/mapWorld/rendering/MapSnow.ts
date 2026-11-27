import Color from 'webGPU-framework/renderer/math/Color';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import type ParamBool from 'webGPU-framework/utils/params/ParamBool';
import type ParamColor from 'webGPU-framework/utils/params/ParamColor';
import type ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import type ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import type MapProjection from '../map/MapProjection';
import type TerrainHeightField from '../map/TerrainHeightField';
import { SNOW_SKIP_MARGIN_M, snowLineMeters, SnowParams } from '../map/snowLine';
import type MapGeometry from './MapGeometry';
import type WaterTerrainHorizon from './deferred/WaterTerrainHorizon';

/**
 * The snow on the ground above the snow line of the latitude of the pivot (see map/snowLine.ts), on the ground and the
 * layers of MapGeometry's SNOW_LAYERS, and on the ground in the water reflections (WaterTerrainHorizon).
 */
export default class MapSnow {
  private snow!: ParamBool;
  // added to the snow line of snowLine.ts, m
  private snowLineOffset!: ParamFloat;
  // how far (m) the noise moves the snow line up and down
  private snowLineNoise!: ParamFloat;
  private snowColor!: ParamColor;
  // see update
  private snowPipelineWasPending = false;
  private snowParams: SnowParams = { enabled: false, line: 0, noise: 0, metersPerUnit: 1, color: new Vector3(1, 1, 1) };

  constructor(private geometry: MapGeometry, private terrainHorizon: WaterTerrainHorizon, private projection: MapProjection,
    private heightField: TerrainHeightField) {}

  public initParams(pgShading: ParamGroup, setDirty: () => void): void {
    const pgSnow = pgShading.addGroup('Snow');
    this.snow = pgSnow.addParamBool('snow', true, { callback: setDirty });
    this.snowLineOffset = pgSnow.addParamFloat('snowLineOffset', 0, -2000, 2000, { callback: setDirty });
    this.snowLineNoise = pgSnow.addParamFloat('snowLineNoise', 150, 0, 500, { callback: setDirty });
    // also the albedo of the glaciers (the layer of the landcover class ice)
    const glacier = this.geometry.layers.findIndex((layer) => layer.def.key === 'glacier');
    this.snowColor = pgSnow.addParamColor('snowColor', Color.fromHex('#eef2f7').vector3, { callback: (v) => {
        this.geometry.setLayerColor(glacier, v);
        setDirty();
      } });
    this.geometry.setLayerColor(glacier, this.snowColor.value);
    pgSnow.collapsed = true;
  }

  /**
   * The snow line at the latitude of the pivot (snowLineMeters, it hardly changes across the view) into the shading, in
   * world y: relative to the ground at the projection center. Without elevation the ground is flat at 0: no snow.
   * @param originHeightM the height of the ground at the projection center (ElevationClient.originHeight)
   * @returns whether the map has to draw again: the line moved (e.g. when the height of the center came in), or the
   * pipeline of the snow switched
   */
  public update(pivotLatitude: number, originHeightM: number): boolean {
    let dirty = false;
    const p = this.snowParams;
    const metersToNorm = this.projection.metersToNorm;
    const lineM = snowLineMeters(pivotLatitude) + this.snowLineOffset.value - originHeightM;
    // off while even the highest loaded ground, at the top of the noise, is below the line: no snow in view, the shading
    // skips it (most cities)
    const reachable = this.heightField.maxHeightM + this.snowLineNoise.value > lineM - SNOW_SKIP_MARGIN_M;
    const enabled = this.snow.value && this.geometry.elevation && reachable;
    const line = lineM * metersToNorm;
    if (enabled !== p.enabled || Math.abs(line - p.line) > .5 * metersToNorm) dirty = true;
    p.enabled = enabled;
    p.line = line;
    p.noise = this.snowLineNoise.value;
    p.metersPerUnit = 1 / metersToNorm;
    p.color = this.snowColor.value;
    this.geometry.setSnow(p);
    this.terrainHorizon.setSnow(p, this.geometry.snowMaterial);
    // a still view draws again when the pipeline of the snow switched on or off is there (the frames before used the old
    // one): while it is made, and once after
    const pending = this.geometry.snowPipelinePending;
    if (pending || this.snowPipelineWasPending) dirty = true;
    this.snowPipelineWasPending = pending;
    return dirty;
  }
}
