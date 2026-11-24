/**
 * Forward geocoding with OpenStreetMap Nominatim: any place name (a town under 100k, a street, a building, a
 * country...) to a lat/lng, for what the city search box doesn't find in worldCitiesData.
 *
 * Nominatim's usage policy forbids autocomplete: this is only called when the user submits the query (Enter or the
 * "search worldwide" row), at most once per second, and the results are cached.
 * https://operations.osmfoundation.org/policies/nominatim/
 */

const SEARCH_URL = 'https://nominatim.openstreetmap.org/search';
const MIN_INTERVAL_MS = 1100;

export interface GeocodedPlace {
  /** The short name, e.g. "Giethoorn". */
  name: string;
  /** The rest of the display name, e.g. "Steenwijkerland, Overijssel, Nederland". */
  context: string;
  lat: number;
  lng: number;
}

let lastRequest = 0;
const cache = new Map<string, GeocodedPlace[]>();

export async function searchNominatim(query: string, limit: number = 6): Promise<GeocodedPlace[]> {
  const q = query.trim();
  if (!q) return [];
  const key = q.toLowerCase();
  const cached = cache.get(key);
  if (cached) return cached;

  const wait = lastRequest + MIN_INTERVAL_MS - performance.now();
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastRequest = performance.now();

  const lang = (typeof navigator !== 'undefined' && navigator.language) || 'en';
  const url = `${SEARCH_URL}?format=jsonv2&limit=${limit}&q=${encodeURIComponent(q)}&accept-language=${encodeURIComponent(lang)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Nominatim ${res.status}`);
  const json = await res.json() as { name?: string; display_name?: string; lat: string; lon: string }[];
  const places = json.map((r) => {
    const parts = (r.display_name ?? '').split(',').map(s => s.trim());
    const name = r.name || parts[0] || q;
    return {
      name,
      context: parts.filter((p, i) => !(i === 0 && p === name)).join(', '),
      lat: parseFloat(r.lat),
      lng: parseFloat(r.lon),
    };
  }).filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lng));
  cache.set(key, places);
  return places;
}
