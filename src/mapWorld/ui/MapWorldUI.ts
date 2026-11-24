import type { LatLngBounds } from '../../mapOSM/osmGeometryCore';
import MapSelector from '../../mapOSM/MapSelector';
import { ATTRIBUTION } from '../map/OpenFreeMapTileService';
import { searchCities } from '../map/worldCities';
import { searchNotableBuildings } from '../map/notableBuildings';
import { searchNominatim } from '../map/nominatimSearch';
import Joystick from '../../deferredRenderer/scene/Joystick';

/** A place picked in the search box. */
export type SearchedPlace = { lat: number; lng: number; title: string; subtitle?: string };

// a row of the search suggestions
type Suggestion = { label: string; detail?: string; pick: () => void };

export type MapWorldUICallbacks = {
  // the bounds the 2D map opens on
  getViewBounds: () => LatLngBounds;
  // a region picked on the 2D map
  onSelectBounds: (bounds: LatLngBounds) => void;
  onRandomCity: () => void;
  // a place picked in the search box
  onGoToPlace: (place: SearchedPlace) => void;
  // returns whether airplane mode is on now
  onToggleAirplane: () => boolean;
};

/**
 * The DOM overlays of MapWorld: the button bar (map, city search, random city, fly / orbit), the 2D (Leaflet) region picker, the
 * OpenStreetMap attribution and the tile error toasts.
 */
export default class MapWorldUI {
  private static readonly MAX_TILE_ERROR_TOASTS = 5;
  private static readonly TILE_ERROR_TIMEOUT_MS = 8000;

  private selector?: MapSelector;
  private selectorZoom: number = 15;
  private buttonBar: HTMLDivElement;
  private attributionEl: HTMLDivElement;
  private tileErrorContainer?: HTMLDivElement;
  private searchInput!: HTMLInputElement;
  private suggestionList!: HTMLDivElement;
  private suggestions: Suggestion[] = [];
  private highlighted: number = 0;
  // bumped on every new search: an older async result is dropped
  private searchSeq: number = 0;
  private static readonly LOCATION_TITLE_MS = 4000;
  private locationTitleEl?: HTMLDivElement;
  private locationTitleTimer?: number;
  // touch screens: centered above the button bar. Steers the plane while flying, else rotates the camera
  private joystick?: Joystick;
  private static readonly CONTROLS_HINT_MS = 8000;
  private controlsHintEl?: HTMLDivElement;

  constructor(private parent: HTMLElement, private callbacks: MapWorldUICallbacks) {
    this.buttonBar = this.createButtonBar();
    this.attributionEl = this.createAttribution();
    if (Joystick.wanted) this.joystick = new Joystick(this.parent, 'mwui-joystick');
  }

  /**
   * The sizes of the bar live in a stylesheet (not inline) so a media query can shrink them: on a narrow (portrait phone) screen the bar
   * spans the width, the buttons get smaller and the search field takes what is left.
   */
  private static injectStyles(): void {
    const id = 'map-world-ui-styles';
    if (document.getElementById(id)) return;
    const style = document.createElement('style');
    style.id = id;
    style.textContent = `
      .mwui-bar { position: absolute; bottom: 60px; left: 50%; transform: translateX(-50%); z-index: 500; display: flex; gap: 8px; }
      .mwui-button { padding: 12px 24px; font-size: 15px; white-space: nowrap; flex: none; }
      .mwui-search { position: relative; display: flex; }
      .mwui-input { width: 220px; padding: 12px 14px; font-size: 15px; }
      .mwui-list { width: max-content; max-width: min(420px, calc(100vw - 32px)); }
      .mwui-joystick { position: absolute; left: 50%; margin-left: -60px; bottom: 120px; z-index: 499; }
      .mwui-hint { bottom: 120px; }
      .mwui-attribution { font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: calc(100% - 16px); }
      @media (max-width: 600px) {
        .mwui-bar { left: 8px; right: 8px; bottom: calc(28px + env(safe-area-inset-bottom, 0px)); transform: none; gap: 6px; }
        .mwui-button { padding: 9px 10px; font-size: 13px; }
        .mwui-search { position: static; flex: 1 1 auto; min-width: 0; }
        .mwui-input { width: 100%; min-width: 0; padding: 9px 10px; font-size: 16px; }
        .mwui-list { width: auto; max-width: none; right: 0; }
        .mwui-joystick { bottom: calc(84px + env(safe-area-inset-bottom, 0px)); }
        .mwui-hint { bottom: calc(84px + env(safe-area-inset-bottom, 0px)); }
        .mwui-attribution { font-size: 9px; }
      }
    `;
    document.head.appendChild(style);
  }

  private createButtonBar(): HTMLDivElement {
    MapWorldUI.injectStyles();
    const bar = document.createElement('div');
    bar.className = 'mwui-bar';
    this.parent.appendChild(bar);
    this.createButton(bar, 'Map', () => void this.openSelector());
    this.createSearchBox(bar);
    this.createButton(bar, 'Random', () => this.callbacks.onRandomCity());
    const airplane = this.createButton(bar, 'Fly', () => {
      const flying = this.callbacks.onToggleAirplane();
      airplane.textContent = flying ? 'Orbit' : 'Fly';
      // the arrow keys and space would otherwise act on the focused button
      airplane.blur();
    });
    return bar;
  }

  private createButton(parent: HTMLElement, label: string, onClick: () => void): HTMLButtonElement {
    const button = document.createElement('button');
    button.textContent = label;
    button.className = 'mwui-button';
    Object.assign(button.style, {
      fontWeight: '600',
      color: '#fff',
      background: '#2d7ef7',
      border: 'none',
      borderRadius: '6px',
      cursor: 'pointer',
      boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    } as CSSStyleDeclaration);
    button.addEventListener('mouseenter', () => (button.style.background = '#1b6ae0'));
    button.addEventListener('mouseleave', () => (button.style.background = '#2d7ef7'));
    button.addEventListener('click', onClick);
    parent.appendChild(button);
    return button;
  }

  // --- city search -------------------------------------------------------------

  /**
   * A text field that suggests the notable buildings and the cities of worldCitiesData as you type. Its last row searches the whole world with
   * Nominatim (only on demand: its usage policy forbids autocomplete), for towns, streets and buildings.
   */
  private createSearchBox(parent: HTMLElement): void {
    const wrapper = document.createElement('div');
    // on a narrow screen it is position: static, so the suggestions span the whole bar
    wrapper.className = 'mwui-search';
    parent.appendChild(wrapper);

    const input = document.createElement('input');
    input.type = 'search';
    input.placeholder = 'Search';
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.className = 'mwui-input';
    Object.assign(input.style, {
      fontFamily: 'system-ui, sans-serif',
      color: '#222',
      background: 'rgba(255,255,255,0.95)',
      border: 'none',
      borderRadius: '6px',
      outline: 'none',
      boxShadow: '0 2px 8px rgba(0,0,0,0.3)',
    } as CSSStyleDeclaration);
    wrapper.appendChild(input);
    this.searchInput = input;

    // above the field: the bar is at the bottom of the screen
    const list = document.createElement('div');
    list.className = 'mwui-list';
    Object.assign(list.style, {
      position: 'absolute',
      bottom: 'calc(100% + 6px)',
      left: '0',
      minWidth: '100%',
      display: 'none',
      flexDirection: 'column',
      background: 'rgba(255,255,255,0.97)',
      borderRadius: '6px',
      boxShadow: '0 2px 12px rgba(0,0,0,0.35)',
      overflow: 'hidden',
      fontFamily: 'system-ui, sans-serif',
    } as CSSStyleDeclaration);
    wrapper.appendChild(list);
    this.suggestionList = list;

    input.addEventListener('input', () => void this.updateLocalSuggestions());
    input.addEventListener('focus', () => void this.updateLocalSuggestions());
    input.addEventListener('blur', () => this.closeSuggestions());
    input.addEventListener('keydown', (e) => {
      // the map's own key handlers (KeyListener hotkeys) shouldn't see the typing
      e.stopPropagation();
      const n = this.suggestions.length;
      if (e.key === 'ArrowDown' && n) {
        this.setHighlighted((this.highlighted + 1) % n);
        e.preventDefault();
      } else if (e.key === 'ArrowUp' && n) {
        this.setHighlighted((this.highlighted - 1 + n) % n);
        e.preventDefault();
      } else if (e.key === 'Enter') {
        this.suggestions[this.highlighted]?.pick();
        e.preventDefault();
      } else if (e.key === 'Escape') {
        input.value = '';
        input.blur();
      }
    });
    input.addEventListener('keyup', (e) => e.stopPropagation());
  }

  /** The cities matching the typed text, and the worldwide search row. */
  private async updateLocalSuggestions(): Promise<void> {
    const query = this.searchInput.value.trim();
    const seq = ++this.searchSeq;
    if (!query) {
      this.showSuggestions([]);
      return;
    }
    const cities = await searchCities(query, 8);
    if (seq !== this.searchSeq) return;
    // notable buildings first: there are few, so a match is likely what was meant
    const rows: Suggestion[] = searchNotableBuildings(query, 3).map((b) => ({
      label: b.name,
      detail: `${b.city}, ${b.country}`,
      pick: () => this.goTo({ lat: b.lat, lng: b.lng, title: `${b.city}, ${b.country}`, subtitle: b.name }, b.name),
    }));
    rows.push(...cities.map((city) => ({
      label: city.name,
      detail: city.landmark || undefined,
      pick: () => this.goTo({ lat: city.lat, lng: city.lng, title: city.name, subtitle: city.landmark || undefined }),
    })));
    rows.push({ label: `Search worldwide for “${query}”`, pick: () => void this.searchWorldwide(query) });
    this.showSuggestions(rows);
  }

  /** The Nominatim results for the query in place of the city suggestions; a single result is gone to directly. */
  private async searchWorldwide(query: string): Promise<void> {
    const seq = ++this.searchSeq;
    this.showSuggestions([], 'Searching…');
    try {
      const places = await searchNominatim(query);
      if (seq !== this.searchSeq) return;
      if (places.length === 1) {
        const p = places[0];
        this.goTo({ lat: p.lat, lng: p.lng, title: p.name, subtitle: p.context || undefined });
        return;
      }
      this.showSuggestions(places.map((p) => ({
        label: p.name,
        detail: p.context,
        pick: () => this.goTo({ lat: p.lat, lng: p.lng, title: p.name, subtitle: p.context || undefined }),
      })), places.length ? undefined : `No places found for “${query}”`);
    } catch (err) {
      if (seq === this.searchSeq) this.showSuggestions([], `Search failed: ${(err as Error).message}`);
    }
  }

  /** inputText: what the field shows afterwards, the title by default. */
  private goTo(place: SearchedPlace, inputText: string = place.title): void {
    this.searchInput.value = inputText;
    this.searchInput.blur();
    this.closeSuggestions();
    this.callbacks.onGoToPlace(place);
  }

  /** The rows of the dropdown, or only a message (searching, nothing found). */
  private showSuggestions(rows: Suggestion[], message?: string): void {
    this.suggestions = rows;
    const list = this.suggestionList;
    list.replaceChildren();
    rows.forEach((row, i) => {
      const el = document.createElement('div');
      Object.assign(el.style, {
        padding: '8px 14px',
        cursor: 'pointer',
        color: '#222',
        fontSize: '14px',
        borderTop: i ? '1px solid rgba(0,0,0,0.08)' : 'none',
      } as CSSStyleDeclaration);
      const label = document.createElement('div');
      label.textContent = row.label;
      label.style.fontWeight = '600';
      el.appendChild(label);
      if (row.detail) {
        const detail = document.createElement('div');
        detail.textContent = row.detail;
        Object.assign(detail.style, { fontSize: '12px', opacity: '0.65', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } as CSSStyleDeclaration);
        el.appendChild(detail);
      }
      // mousedown, not click: the input's blur would close the list first
      el.addEventListener('mousedown', (e) => {
        e.preventDefault();
        row.pick();
      });
      el.addEventListener('mouseenter', () => this.setHighlighted(i));
      list.appendChild(el);
    });
    if (message) {
      const el = document.createElement('div');
      el.textContent = message;
      Object.assign(el.style, { padding: '8px 14px', color: '#666', fontSize: '14px' } as CSSStyleDeclaration);
      list.appendChild(el);
    }
    list.style.display = rows.length || message ? 'flex' : 'none';
    this.setHighlighted(0);
  }

  private setHighlighted(index: number): void {
    this.highlighted = index;
    Array.from(this.suggestionList.children).forEach((el, i) => {
      (el as HTMLElement).style.background = i === index && i < this.suggestions.length ? 'rgba(45,126,247,0.15)' : 'transparent';
    });
  }

  private closeSuggestions(): void {
    this.searchSeq++;
    this.suggestions = [];
    this.suggestionList.replaceChildren();
    this.suggestionList.style.display = 'none';
  }

  private setButtonsVisible(visible: boolean): void {
    this.buttonBar.style.display = visible ? 'flex' : 'none';
    if (this.joystick) this.joystick.element.style.display = visible ? '' : 'none';
  }

  /** The joystick (x right, y up), undefined without one: a mouse has WASD and the arrows. */
  public get joystickValue(): { x: number; y: number } | undefined {
    return this.joystick?.value;
  }

  /** Open the 2D (Leaflet) location picker, unrestricted: the whole world. */
  private async openSelector(): Promise<void> {
    if (this.selector) return;
    this.setButtonsVisible(false);

    const close = () => {
      this.selectorZoom = this.selector?.getZoom() ?? this.selectorZoom;
      this.selector?.hide();
      this.selector = undefined;
      this.setButtonsVisible(true);
    };
    this.selector = new MapSelector(this.parent, this.callbacks.getViewBounds(), this.selectorZoom, (bounds) => {
      close();
      this.callbacks.onSelectBounds(bounds);
    }, close, null);
    await this.selector.show();
  }

  /** OpenStreetMap / OpenMapTiles attribution (required by the data licence). */
  private createAttribution(): HTMLDivElement {
    const el = document.createElement('div');
    el.textContent = ATTRIBUTION;
    // one line: the font shrinks on a narrow screen (stylesheet)
    el.className = 'mwui-attribution';
    Object.assign(el.style, {
      position: 'absolute',
      left: '8px',
      bottom: '6px',
      zIndex: '500',
      fontFamily: 'system-ui, sans-serif',
      color: 'rgba(0,0,0,0.75)',
      background: 'rgba(255,255,255,0.6)',
      padding: '2px 6px',
      borderRadius: '3px',
      pointerEvents: 'none',
    } as CSSStyleDeclaration);
    this.parent.appendChild(el);
    return el;
  }

  // --- onscreen tile error toast -------------------------------------------

  private ensureTileErrorContainer(): HTMLDivElement {
    if (this.tileErrorContainer) return this.tileErrorContainer;
    const container = document.createElement('div');
    Object.assign(container.style, {
      position: 'absolute',
      bottom: '20px',
      right: '20px',
      zIndex: '500',
      display: 'flex',
      flexDirection: 'column',
      gap: '8px',
      maxWidth: '420px',
      pointerEvents: 'none',
    } as CSSStyleDeclaration);
    this.parent.appendChild(container);
    this.tileErrorContainer = container;
    return container;
  }

  public showTileError(message: string): void {
    const container = this.ensureTileErrorContainer();
    while (container.childElementCount >= MapWorldUI.MAX_TILE_ERROR_TOASTS) {
      container.firstChild?.remove();
    }
    const toast = document.createElement('div');
    toast.textContent = message;
    Object.assign(toast.style, {
      background: 'rgba(180, 40, 40, 0.92)',
      color: '#fff',
      fontSize: '12px',
      fontFamily: 'monospace',
      padding: '10px 14px',
      borderRadius: '6px',
      boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
      wordBreak: 'break-all',
      lineHeight: '1.4',
      opacity: '0',
      transition: 'opacity 0.3s ease',
    } as CSSStyleDeclaration);
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.style.opacity = '1'; });
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, MapWorldUI.TILE_ERROR_TIMEOUT_MS);
  }

  // --- location title (top center) -------------------------------------------

  /**
   * Show a place name centered at the top of the screen for LOCATION_TITLE_MS,
   * with an optional second line (the building in focus). Replaces any title
   * still showing.
   */
  public showLocationTitle(title: string, subtitle?: string): void {
    this.hideLocationTitle();
    const el = document.createElement('div');
    Object.assign(el.style, {
      position: 'absolute',
      top: '24px',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '500',
      maxWidth: 'calc(100% - 32px)',
      textAlign: 'center',
      fontFamily: 'system-ui, sans-serif',
      color: '#fff',
      background: 'rgba(0,0,0,0.45)',
      padding: '10px 22px',
      borderRadius: '8px',
      textShadow: '0 1px 3px rgba(0,0,0,0.6)',
      pointerEvents: 'none',
      opacity: '0',
      transition: 'opacity 0.4s ease',
    } as CSSStyleDeclaration);
    const titleEl = document.createElement('div');
    titleEl.textContent = title;
    Object.assign(titleEl.style, { fontSize: '26px', fontWeight: '700' } as CSSStyleDeclaration);
    el.appendChild(titleEl);
    if (subtitle) {
      const subEl = document.createElement('div');
      subEl.textContent = subtitle;
      Object.assign(subEl.style, { fontSize: '16px', fontWeight: '500', marginTop: '2px', opacity: '0.9' } as CSSStyleDeclaration);
      el.appendChild(subEl);
    }
    this.parent.appendChild(el);
    this.locationTitleEl = el;
    requestAnimationFrame(() => { el.style.opacity = '1'; });
    this.locationTitleTimer = window.setTimeout(() => {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
      if (this.locationTitleEl === el) this.locationTitleEl = undefined;
    }, MapWorldUI.LOCATION_TITLE_MS);
  }

  // --- controls hint (above the button bar) ------------------------------------

  /** How to rotate and fly, for CONTROLS_HINT_MS after the start. Desktop only: a touch screen has the joystick. */
  public showControlsHint(): void {
    if (Joystick.wanted || this.controlsHintEl) return;
    const el = document.createElement('div');
    el.className = 'mwui-hint';
    el.textContent = 'Ctrl + drag or WASD to rotate';
    Object.assign(el.style, {
      position: 'absolute',
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: '500',
      maxWidth: 'calc(100% - 32px)',
      textAlign: 'center',
      fontFamily: 'system-ui, sans-serif',
      fontSize: '15px',
      fontWeight: '500',
      color: '#fff',
      background: 'rgba(0,0,0,0.45)',
      padding: '8px 18px',
      borderRadius: '8px',
      textShadow: '0 1px 3px rgba(0,0,0,0.6)',
      pointerEvents: 'none',
      opacity: '0',
      transition: 'opacity 0.4s ease',
    } as CSSStyleDeclaration);
    this.parent.appendChild(el);
    this.controlsHintEl = el;
    requestAnimationFrame(() => { el.style.opacity = '1'; });
    window.setTimeout(() => {
      el.style.opacity = '0';
      setTimeout(() => el.remove(), 400);
      if (this.controlsHintEl === el) this.controlsHintEl = undefined;
    }, MapWorldUI.CONTROLS_HINT_MS);
  }

  private hideLocationTitle(): void {
    clearTimeout(this.locationTitleTimer);
    this.locationTitleEl?.remove();
    this.locationTitleEl = undefined;
  }

  destruct(): void {
    this.hideLocationTitle();
    this.controlsHintEl?.remove();
    this.joystick?.destruct();
    this.selector?.hide();
    this.tileErrorContainer?.remove();
    this.attributionEl.remove();
    this.buttonBar.remove();
  }
}
