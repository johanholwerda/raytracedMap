import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import TerrainLayerBuffer from './TerrainLayerBuffer';

/** A ground layer as the GUI shows it, see WORLD_LAYERS (worldGeometryCore.ts). */
export type LayerDef = {
  key: string;
  label: string;
  color: [number, number, number];
  defaultOn: boolean;
};

/** A ground layer as MapGeometry draws it: its packed geometry, its material and its GUI toggle. */
export type LayerRuntime = {
  def: LayerDef;
  material: VertexFragmentShader;
  buffer: TerrainLayerBuffer;
  enabled: ParamBool;
};
