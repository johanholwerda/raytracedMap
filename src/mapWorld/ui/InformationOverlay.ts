import type { BuildingInfo } from '../map/worldGeometryCore';

/** Resolves a building id to a human readable address (see nominatimAddress.ts), null when there is none. */
export type AddressResolver = (id: string) => Promise<string | null>;

/**
 * The info panel of the selected building (picked on the CPU, see BuildingPicker): a small DOM panel with its address
 * (looked up with the AddressResolver), height and footprint area. The selected building is highlighted via the
 * shader's `selectedID` uniform (this class only owns {@link selectedID}; the host feeds it to the material).
 */
export default class InformationOverlay {
  private panel: HTMLDivElement;
  private addressEl: HTMLDivElement;
  private bodyEl: HTMLDivElement;

  /** Packed global id of the selected building, or -1 when nothing is selected. */
  public selectedID: number = -1;

  private addressResolver: AddressResolver | null = null;
  // Guards against a slow address lookup landing after a newer selection.
  private resolveToken = 0;

  constructor(parent: HTMLElement) {
    const { panel, addressEl, bodyEl } = this.buildPanel();
    this.panel = panel;
    this.addressEl = addressEl;
    this.bodyEl = bodyEl;
    parent.appendChild(panel);
  }

  /**
   * Select a building (picked on the CPU, see CityScene.pick): globalId -1 or a null info clears the selection.
   */
  public select(globalId: number, info: BuildingInfo | null): void {
    this.selectedID = info ? globalId : -1;
    this.showInfo(info);
  }

  /** Wire an async id -> address resolver. */
  public setAddressResolver(resolver: AddressResolver): void {
    this.addressResolver = resolver;
  }

  /** Clear the current selection and hide the panel. */
  public clearSelection(): void {
    this.selectedID = -1;
    this.showInfo(null);
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

    // Address: there's no address in the tile, so show the id and, if a
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
    this.panel.remove();
  }
}
