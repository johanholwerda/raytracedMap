import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import { LatLngBounds, LayerKey } from './OSMBuilder';
import { TerrainCategory, TerrainCategoryDef } from './CityGlTFBuilder';
import TerrainLayerBuffer from './TerrainLayerBuffer';

export type LayerDef = {
  key: LayerKey;
  label: string;
  color: [number, number, number];
  defaultOn: boolean;
};

// One UI toggle per PDOK BRT-A source-layer. Labels match the MVT layer names.
// Order matches OSMBuilder.LAYER_CONFIGS draw order (lowest first).
export const LAYER_DEFS: LayerDef[] = [
  { key: 'nederland', label: 'nederland', color: [0.90, 0.90, 0.87], defaultOn: false },
  { key: 'waterdeelvlak', label: 'waterdeelvlak', color: [0.42, 0.60, 0.78], defaultOn: true },
  { key: 'terreinvlak', label: 'terreinvlak', color: [0.40, 0.55, 0.35], defaultOn: true },
  { key: 'wegdeelvlak', label: 'wegdeelvlak', color: [0.82, 0.82, 0.84], defaultOn: true },
  { key: 'waterdeellijn', label: 'waterdeellijn', color: [0.45, 0.62, 0.80], defaultOn: true },
  { key: 'spoorbaandeellijn', label: 'spoorbaandeellijn', color: [0.35, 0.35, 0.38], defaultOn: true },
  { key: 'inrichtingselementlijn', label: 'inrichtingselementlijn', color: [0.70, 0.65, 0.55], defaultOn: true },
  { key: 'gebouwvlak', label: 'gebouwvlak', color: [0.84, 0.82, 0.78], defaultOn: false },
];

export type LayerRuntime = {
  def: LayerDef;
  material: VertexFragmentShader;
  buffer: TerrainLayerBuffer;
  enabled: ParamBool;
  /** The bridge decks of the layer (MapWorld: roads, railways, paths), drawn with bridgeMaterial. */
  bridges?: TerrainLayerBuffer;
  bridgeMaterial?: VertexFragmentShader;
};

export type Terrain3DLayerRuntime = {
  def: TerrainCategoryDef;
  material: VertexFragmentShader;
  buffer: TerrainLayerBuffer;
  enabled: ParamBool;
};

export type LoadedTerrainTile = {
  key: string;
  zoom: number;
};

//https://prochitecture.com/blender-osm/extent/?blender_version=5.1&addon=blosm&addon_version=2.7.27
export const DEFAULT_BOUNDS: LatLngBounds = {
  south: 52.36627,
  west: 4.87739,
  north: 52.38255,
  east: 4.92361,
};
