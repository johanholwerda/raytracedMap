import { LatLngBounds } from '../map/geo';

/**
 * Fullscreen OpenStreetMap (Leaflet) location picker.
 *
 * Shows the slippy map
 * and a single "Create 3D map" button. Panning the map moves the geographic
 * location under the crosshair; clicking the button reports that location (as a
 * lat/lng bbox sized by the initial span) through the onCreate callback, which
 * becomes the camera pivot / projection center. There is no resizable
 * rectangle — the 3D view streams tiles around the chosen center.
 *
 * Leaflet is loaded from a CDN on demand so it does not need to be added to the
 * project's dependencies, matching the renderer used by the reference site.
 */

const LEAFLET_VERSION = '1.9.4';
const LEAFLET_JS = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.js`;
const LEAFLET_CSS = `https://unpkg.com/leaflet@${LEAFLET_VERSION}/dist/leaflet.css`;

/**
 * Bounding box of the (European) Netherlands. We only have terrain and building
 * data for this area, so the picker is constrained to it: the map cannot be
 * panned (far) outside this box and cannot be zoomed out past the point where
 * the box still fills the screen.
 */
const NETHERLANDS_BOUNDS: LatLngBounds = {
  south: 50.75,
  west: 3.36,
  north: 53.56,
  east: 7.23,
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

let leafletPromise: Promise<any> | null = null;

function loadLeaflet(): Promise<any> {
  const w = window as any;
  if (w.L) return Promise.resolve(w.L);
  if (leafletPromise) return leafletPromise;

  leafletPromise = new Promise<any>((resolve, reject) => {
    if (!document.querySelector(`link[data-leaflet]`)) {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = LEAFLET_CSS;
      css.setAttribute('data-leaflet', '');
      document.head.appendChild(css);
    }
    const script = document.createElement('script');
    script.src = LEAFLET_JS;
    script.async = true;
    script.onload = () => resolve((window as any).L);
    script.onerror = () => reject(new Error('[MapSelector] failed to load Leaflet from CDN'));
    document.head.appendChild(script);
  });
  return leafletPromise;
}

export default class MapSelector {
  private parent: HTMLElement;
  private bounds: LatLngBounds;
  private initialZoom: number;
  private onCreate: (bounds: LatLngBounds) => void;
  private onCancel?: () => void;
  // Area the crosshair is kept inside (where we have data), or null for the
  // whole world.
  private clampBounds: LatLngBounds | null;

  // Span of the reported bbox, taken from the initial bounds so the 3D view
  // keeps a consistent world scale regardless of where the user pans.
  private halfLatSpan: number;
  private halfLngSpan: number;

  private overlay!: HTMLDivElement;
  private mapEl!: HTMLDivElement;
  private L: any;
  private map: any;

  // Guards against re-entrancy: re-centering the map fires 'move' again, which
  // would otherwise recurse into clampCenterToArea until the stack blows.
  private clamping = false;

  constructor(
    parent: HTMLElement,
    initialBounds: LatLngBounds,
    initialZoom: number,
    onCreate: (bounds: LatLngBounds) => void,
    onCancel?: () => void,
    clampBounds: LatLngBounds | null = NETHERLANDS_BOUNDS,
  ) {
    this.parent = parent;
    this.clampBounds = clampBounds;
    this.bounds = { ...initialBounds };
    this.initialZoom = initialZoom;
    this.onCreate = onCreate;
    this.onCancel = onCancel;
    this.halfLatSpan = Math.max(initialBounds.north - initialBounds.south, 1e-4) * 0.5;
    this.halfLngSpan = Math.max(initialBounds.east - initialBounds.west, 1e-4) * 0.5;
  }

  public async show(): Promise<void> {
    this.L = await loadLeaflet();
    this.buildDom();
    this.initMap();
  }

  public hide(): void {
    if (this.map) {
      this.map.remove();
      this.map = null;
    }
    if (this.overlay && this.overlay.parentElement) {
      this.overlay.parentElement.removeChild(this.overlay);
    }
  }

  // --- dom ------------------------------------------------------------------

  private buildDom(): void {
    this.overlay = document.createElement('div');
    Object.assign(this.overlay.style, {
      position: 'absolute',
      inset: '0',
      zIndex: '1000',
      background: '#fff',
    } as CSSStyleDeclaration);

    this.mapEl = document.createElement('div');
    Object.assign(this.mapEl.style, {
      position: 'absolute',
      inset: '0',
    } as CSSStyleDeclaration);
    this.overlay.appendChild(this.mapEl);

    const buttonBar = document.createElement('div');
    Object.assign(buttonBar.style, {
      position: 'absolute',
      bottom: '60px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '1001',
      display: 'flex',
      gap: '12px',
    } as CSSStyleDeclaration);

    const createButton = this.makeButton('go', '#2d7ef7', '#1b6ae0');
    createButton.addEventListener('click', () => this.onCreate(this.boundsFromMapCenter()));
    buttonBar.appendChild(createButton);

    const cancelButton = this.makeButton('cancel', '#6b7280', '#4b5563');
    cancelButton.addEventListener('click', () => this.onCancel?.());
    buttonBar.appendChild(cancelButton);

    this.overlay.appendChild(buttonBar);
    this.parent.appendChild(this.overlay);
  }

  private makeButton(label: string, bg: string, bgHover: string): HTMLButtonElement {
    const button = document.createElement('button');
    button.textContent = label;
    Object.assign(button.style, {
      padding: '12px 24px',
      fontSize: '15px',
      fontWeight: '600',
      color: '#fff',
      background: bg,
      border: 'none',
      borderRadius: '6px',
      cursor: 'pointer',
      boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    } as CSSStyleDeclaration);
    button.addEventListener('mouseenter', () => (button.style.background = bgHover));
    button.addEventListener('mouseleave', () => (button.style.background = bg));
    return button;
  }

  // --- leaflet map ----------------------------------------------------------

  private initMap(): void {
    const L = this.L;
    const b = this.bounds;
    const area = this.clampBounds;

    this.map = L.map(this.mapEl, { zoomControl: true, worldCopyJump: !area });

    if (area) {
      const nl = L.latLngBounds([area.south, area.west], [area.north, area.east]);

      // Constrain the *center* (the crosshair) to the data area rather than the
      // whole viewport, so the picker can be centered on remote/border areas like
      // Maastricht at every zoom level. The surrounding view may spill over into
      // Belgium/Germany, but the reported location always stays within our data.
      this.map.on('move', () => this.clampCenterToArea());

      // Don't let the user zoom out past the point where the area fills the
      // viewport, so it always covers a significant portion of the screen (and
      // panning stays within data coverage).
      const minZoom = Math.ceil(this.map.getBoundsZoom(nl));
      this.map.setMinZoom(minZoom);

      // Seed center is clamped into the area in case the camera was panned past
      // the edge of our data coverage before the picker was reopened.
      const center: [number, number] = [
        clamp((b.south + b.north) / 2, area.south, area.north),
        clamp((b.west + b.east) / 2, area.west, area.east),
      ];
      this.map.setView(center, Math.max(this.initialZoom, minZoom));
    } else {
      this.map.setMinZoom(2);
      this.map.setView([(b.south + b.north) / 2, (b.west + b.east) / 2], this.initialZoom);
    }

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(this.map);
  }

  /** Current Leaflet zoom level, so the caller can restore it next time. */
  public getZoom(): number {
    return this.map ? this.map.getZoom() : this.initialZoom;
  }

  /** Keep the map center (crosshair) within the data area (clampBounds). */
  private clampCenterToArea(): void {
    const area = this.clampBounds;
    if (this.clamping || !area) return;
    const c = this.map.getCenter();
    const lat = clamp(c.lat, area.south, area.north);
    const lng = clamp(c.lng, area.west, area.east);
    if (lat === c.lat && lng === c.lng) return;
    // Recenter without animation; the guard swallows the 'move' this re-fires so
    // we don't recurse. stop() halts any in-flight pan/inertia driving us out.
    this.clamping = true;
    this.map.stop();
    this.map.setView([lat, lng], this.map.getZoom(), { animate: false });
    this.clamping = false;
  }

  /** Bbox centered on the current map center, sized by the initial span. */
  private boundsFromMapCenter(): LatLngBounds {
    // wrap(): the world view can be panned past ±180° longitude.
    const c = this.map.getCenter().wrap();
    // The 'move' handler already keeps the center inside the data area; clamp
    // here too as a safety net so the reported center never leaves our data.
    const area = this.clampBounds;
    const lat = area ? clamp(c.lat, area.south, area.north) : c.lat;
    const lng = area ? clamp(c.lng, area.west, area.east) : c.lng;
    return {
      south: lat - this.halfLatSpan,
      north: lat + this.halfLatSpan,
      west: lng - this.halfLngSpan,
      east: lng + this.halfLngSpan,
    };
  }
}
