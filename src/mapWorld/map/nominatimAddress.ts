/**
 * Reverse-geocodes a clicked building to a street address with OpenStreetMap
 * Nominatim (worldwide, no API key, CORS enabled). World buildings carry no
 * address or id in the vector tiles, so {@link worldGeometryCore} stores the
 * footprint centroid as the BuildingInfo id ("lat, lng") and this resolver
 * turns it into an address for the info panel.
 *
 * Nominatim's usage policy allows at most one request per second; clicks are
 * user-driven, and requests are additionally spaced out here.
 * https://operations.osmfoundation.org/policies/nominatim/
 */

const REVERSE_URL = 'https://nominatim.openstreetmap.org/reverse';
const MIN_INTERVAL_MS = 1100;

let lastRequest = 0;
const cache = new Map<string, string | null>();

type NominatimAddress = Record<string, string | undefined>;

function formatAddress(address: NominatimAddress, displayName: string): string {
  const street = address.road ?? address.pedestrian ?? address.footway ?? address.square ?? address.neighbourhood;
  const house = address.house_number;
  const place = address.city ?? address.town ?? address.village ?? address.municipality ?? address.state;
  const line1 = [address.building ?? address.amenity ?? address.tourism, street ? [street, house].filter(Boolean).join(' ') : '']
    .filter(Boolean)
    .join(', ');
  const line2 = [address.postcode, place].filter(Boolean).join(' ');
  const out = [line1, line2, address.country].filter(Boolean).join(', ');
  return out || displayName;
}

export async function fetchNominatimAddress(id: string): Promise<string | null> {
  const m = id.match(/^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$/);
  if (!m) return null;
  const key = `${m[1]},${m[2]}`;
  if (cache.has(key)) return cache.get(key)!;

  const wait = lastRequest + MIN_INTERVAL_MS - performance.now();
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastRequest = performance.now();

  const lang = (typeof navigator !== 'undefined' && navigator.language) || 'en';
  const url =
    `${REVERSE_URL}?format=jsonv2&addressdetails=1&zoom=18` +
    `&lat=${encodeURIComponent(m[1])}&lon=${encodeURIComponent(m[2])}` +
    `&accept-language=${encodeURIComponent(lang)}`;

  try {
    const res = await fetch(url);
    if (!res.ok) return null;
    const json = await res.json();
    if (!json || json.error) {
      cache.set(key, null);
      return null;
    }
    const address = formatAddress(json.address ?? {}, json.display_name ?? '');
    cache.set(key, address || null);
    return address || null;
  } catch {
    return null;
  }
}
