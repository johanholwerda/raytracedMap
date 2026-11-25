// Minimal type declaration for the vendored meshoptimizer decoder
// (meshopt_decoder.module.js, MIT, meshoptimizer 0.20). Only the subset of the
// API used by CityGlTFBuilder is declared.

export type MeshoptFilter = 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL';
export type MeshoptMode = 'ATTRIBUTES' | 'TRIANGLES' | 'INDICES';

export declare const MeshoptDecoder: {
  /** Resolves once the embedded WASM module is instantiated. */
  ready: Promise<void>;
  supported: boolean;
  /**
   * Decode one EXT_meshopt_compression bufferView into `target`.
   * @param target destination bytes (length = count * size)
   * @param count  element count from the extension
   * @param size   byteStride from the extension
   * @param source compressed source bytes
   * @param mode   'ATTRIBUTES' | 'TRIANGLES' | 'INDICES'
   * @param filter 'NONE' | 'OCTAHEDRAL' | 'QUATERNION' | 'EXPONENTIAL'
   */
  decodeGltfBuffer(
    target: Uint8Array,
    count: number,
    size: number,
    source: Uint8Array,
    mode: MeshoptMode,
    filter: MeshoptFilter,
  ): void;
};
