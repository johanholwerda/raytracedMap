import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Mesh from 'webGPU-framework/renderer/mesh/Mesh';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Color from 'webGPU-framework/renderer/math/Color';
import type { BuildingInfo } from './cityGlTFCore';

/**
 * Resolves a BAG "pand" id (e.g. "NL.IMBAG.Pand.0363100012197777") to a human
 * readable address. The 3DBAG tiles carry no address — only the pand id — so
 * this is an optional async hook the host can wire to the PDOK Locatieserver /
 * BAG API later. Returns null when no address is available.
 */
export type AddressResolver = (pandId: string) => Promise<string | null>;

/** Inputs for a single pick (one canvas click). */
export type PickRequest = {
  /** Click position in render-texture pixels (canvas backing-store space). */
  x: number;
  y: number;
  /** The building material (osm shader, compiled with FEATURE_IDS). */
  material: VertexFragmentShader;
  viewProjection: Matrix4x4;
  cameraPosition: Vector3;
  /** Building meshes to rasterize into the index map. */
  meshes: Mesh[];
  /** Per-mesh picking slot, parallel to {@link meshes}. */
  slots: number[];
  /** Called before mesh i is drawn, for uniforms of its own (mapWorld: the decoding of its vertices). */
  beforeDraw?: (index: number) => void;
  /** Resolve a packed global id to its building attributes. */
  lookup: (globalId: number) => BuildingInfo | null;
};

/**
 * Click-to-select building info overlay.
 *
 * On a canvas click the host calls {@link pick}, which rasterizes the building
 * meshes into an offscreen rgba8 *index map* (the osm shader packs each
 * fragment's 32-bit picking id — `slot * 65536 + featureId`, +1 so 0 means
 * empty — into the 4 channels), reads back the single clicked pixel, decodes
 * the id and resolves it to a {@link BuildingInfo}. The selected building is
 * highlighted in the main pass via the shader's `selectedID` uniform (this
 * class only owns {@link selectedID}; the host feeds it to the material).
 *
 * A small DOM panel shows the BAG pand id (address — pending a BAG lookup, see
 * {@link setAddressResolver}), construction year and footprint area.
 */
export default class InformationOverlay {
  private renderer: Renderer;
  /** rgba8 offscreen the building picking ids are rendered into (no MSAA). Made on the first {@link pick}. */
  private indexRT: RenderTexture | null = null;
  private panel: HTMLDivElement;
  private addressEl: HTMLDivElement;
  private bodyEl: HTMLDivElement;
  private sizeMultiplier: number = .3;

  /** Packed global id of the selected building, or -1 when nothing is selected. */
  public selectedID: number = -1;

  private addressResolver: AddressResolver | null = null;
  // Guards against a slow address lookup landing after a newer selection.
  private resolveToken = 0;
  // A 256-byte readback buffer (one row; copyTextureToBuffer needs bytesPerRow
  // to be a multiple of 256). Reused across picks. Made on the first {@link pick}.
  private readbackBuffer: GPUBuffer | null = null;

  constructor(renderer: Renderer, parent: HTMLElement) {
    this.renderer = renderer;
    const { panel, addressEl, bodyEl } = this.buildPanel();
    this.panel = panel;
    this.addressEl = addressEl;
    this.bodyEl = bodyEl;
    parent.appendChild(panel);
  }

  /** The offscreen the host should NOT render to except via {@link pick}. */
  public get indexRenderTexture(): RenderTexture {
    if (!this.indexRT) {
      this.indexRT = new RenderTexture(this.renderer, 1, 1, {
        formats: ['rgba8unorm'],
        depth: true,
        sampleCount: 1,
        scaleToCanvas: true,
        sizeMultiplier: this.sizeMultiplier,
        toScreen: false,
      });
    }
    return this.indexRT;
  }

  /**
   * Select a building picked elsewhere (mapWorld: on the CPU, see CityScene.pick): globalId -1 or a null info clears
   * the selection.
   */
  public select(globalId: number, info: BuildingInfo | null): void {
    this.selectedID = info ? globalId : -1;
    this.showInfo(info);
  }

  /** Wire an async pand-id -> address resolver (e.g. PDOK BAG). */
  public setAddressResolver(resolver: AddressResolver): void {
    this.addressResolver = resolver;
  }

  /**
   * Render the index map, read back the clicked pixel and select the building
   * under it (updating {@link selectedID} and the info panel). Resolves once the
   * pixel has been read; an empty pixel clears the selection.
   */
  public async pick(req: PickRequest): Promise<void> {
    const m = req.material;
    // The index map shares the building material; switch it into id-writing mode
    // and feed the per-mesh tile slot so the shader can pack (slot, featureId).
    m.setMatrix('viewProjection', req.viewProjection);
    m.setVector3('cameraPosition', req.cameraPosition);
    m.setFloat('drawIndex', 1);

    this.renderer.startRenderPass(this.indexRenderTexture, { clearColor: new Color(0, 0, 0, 0) });
    for (let i = 0; i < req.meshes.length; i++) {
      m.setFloat('tileID', req.slots[i]);
      req.beforeDraw?.(i);
      this.renderer.draw(req.meshes[i], m);
    }
    m.setFloat('drawIndex', 0);
    this.renderer.endPasses();

    const globalId = await this.readPixel(req.x, req.y);
    this.selectedID = globalId;
    const info = globalId >= 0 ? req.lookup(globalId) : null;
    // TEMP debug: log the picked building so it can be identified (e.g. to
    // inspect a bad vertex in the source 3DBAG tile).
    console.log('[pick] globalId', globalId, 'building', info);
    this.showInfo(info);
  }

  /** Clear the current selection and hide the panel. */
  public clearSelection(): void {
    this.selectedID = -1;
    this.showInfo(null);
  }

  /**
   * Copy the single pixel at (x, y) from the index map and decode its packed
   * id. Returns the global building id, or -1 for an empty pixel.
   */
  private async readPixel(x: number, y: number): Promise<number> {
    const rt = this.indexRenderTexture;
    const tex = rt.gpuTexture;
    // pick coords are in full-res canvas backing-store pixels, but the index
    // map is rendered at sizeMultiplier scale — scale the click to match.
    const scale = rt.sizeMultiplier;
    this.readbackBuffer ??= this.renderer.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const readback = this.readbackBuffer;
    const px = Math.max(0, Math.min(tex.width - 1, Math.round(x * scale)));
    const py = Math.max(0, Math.min(tex.height - 1, Math.round(y * scale)));

    this.renderer.commandEncoder.copyTextureToBuffer(
      { texture: tex, origin: { x: px, y: py, z: 0 } },
      { buffer: readback, bytesPerRow: 256 },
      { width: 1, height: 1 },
    );
    this.renderer.finish();

    await readback.mapAsync(GPUMapMode.READ);
    const b = new Uint8Array(readback.getMappedRange());
    // Little-endian unpack of the 32-bit id the shader wrote (value = id + 1).
    const value = (b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)) >>> 0;
    readback.unmap();
    return value === 0 ? -1 : value - 1;
  }

  // --- panel ---------------------------------------------------------------

  private showInfo(info: BuildingInfo | null): void {
    // A new selection invalidates any in-flight address lookup.
    this.resolveToken++;
    if (!info) {
      this.panel.style.display = 'none';
      return;
    }
    this.panel.style.display = 'block';

    const year = info.year > 0 ? String(info.year) : 'unknown';
    const area = info.groundArea > 0 ? `${info.groundArea.toFixed(1)} m²` : 'unknown';

    this.bodyEl.innerHTML = '';
    // A negative year means the source has no construction year at all (world map).
    if (info.year >= 0) this.bodyEl.appendChild(this.row('Year of construction', year));
    if (info.height !== undefined && info.height > 0) this.bodyEl.appendChild(this.row('Height', `${info.height.toFixed(1)} m`));
    this.bodyEl.appendChild(this.row('Footprint area', area));

    // Address: there's no address in the tile, so show the pand id and, if a
    // resolver is wired, replace it with the looked-up street address.
    this.addressEl.textContent = info.id || 'unknown';
    if (info.id && this.addressResolver) {
      const token = this.resolveToken;
      this.addressResolver(info.id)
        .then((address) => {
          if (token === this.resolveToken && address) this.addressEl.textContent = address;
        })
        .catch(() => {});
    }
  }

  private row(label: string, value: string): HTMLDivElement {
    const row = document.createElement('div');
    Object.assign(row.style, { display: 'flex', justifyContent: 'space-between', gap: '16px' } as CSSStyleDeclaration);
    const l = document.createElement('span');
    l.textContent = label;
    l.style.opacity = '0.7';
    const v = document.createElement('span');
    v.textContent = value;
    v.style.textAlign = 'right';
    row.appendChild(l);
    row.appendChild(v);
    return row;
  }

  private buildPanel(): { panel: HTMLDivElement; addressEl: HTMLDivElement; bodyEl: HTMLDivElement } {
    const panel = document.createElement('div');
    Object.assign(panel.style, {
      position: 'absolute',
      top: '20px',
      left: '20px',
      zIndex: '600',
      minWidth: '240px',
      maxWidth: '320px',
      background: 'rgba(20, 24, 30, 0.92)',
      color: '#fff',
      fontFamily: 'system-ui, sans-serif',
      fontSize: '13px',
      lineHeight: '1.5',
      padding: '14px 16px',
      borderRadius: '8px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.45)',
      display: 'none',
    } as CSSStyleDeclaration);

    const close = document.createElement('div');
    close.textContent = '×';
    Object.assign(close.style, {
      position: 'absolute',
      top: '8px',
      right: '12px',
      cursor: 'pointer',
      fontSize: '18px',
      opacity: '0.7',
    } as CSSStyleDeclaration);
    close.addEventListener('click', () => this.clearSelection());
    panel.appendChild(close);

    const title = document.createElement('div');
    title.textContent = 'Address';
    Object.assign(title.style, { fontWeight: '600', opacity: '0.7', marginBottom: '2px' } as CSSStyleDeclaration);
    panel.appendChild(title);

    const addressEl = document.createElement('div');
    Object.assign(addressEl.style, { fontWeight: '600', marginBottom: '10px', wordBreak: 'break-word' } as CSSStyleDeclaration);
    panel.appendChild(addressEl);

    const bodyEl = document.createElement('div');
    Object.assign(bodyEl.style, { display: 'flex', flexDirection: 'column', gap: '4px' } as CSSStyleDeclaration);
    panel.appendChild(bodyEl);

    return { panel, addressEl, bodyEl };
  }

  public destruct(): void {
    this.indexRT?.destruct();
    this.readbackBuffer?.destroy();
    this.panel.remove();
  }
}
