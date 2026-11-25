/**
 * Nice views, to be used as the landing views. Paste them as copied from the url (without the leading ?):
 * lat=..&lng=..&dist=..&pitch=..&yaw=..&time=..
 */
export const NICE_VIEWS: string[] = [
  'lat=4.609368&lng=-74.058333&dist=960&pitch=-25.77&yaw=11.12&time=sunset',
  'lat=1.285584&lng=103.859607&dist=915&pitch=-27.27&yaw=25.48&time=afternoon',
  'lat=55.720917&lng=37.895266&dist=535&pitch=-30.87&yaw=-68.53&time=noon'
];

export type NiceView = { lat: number; lng: number; dist: number; pitch: number; yaw: number; time?: string };

/** Parses one url query string into a view. */
export function parseNiceView(query: string): NiceView {
  const p = new URLSearchParams(query.replace(/^[?#]/, ''));
  return {
    lat: Number(p.get('lat')),
    lng: Number(p.get('lng')),
    dist: Number(p.get('dist')),
    pitch: Number(p.get('pitch')),
    yaw: Number(p.get('yaw')),
    time: p.get('time') ?? undefined,
  };
}

/** A random nice view, for the landing. */
export function randomNiceView(): NiceView {
  return parseNiceView(NICE_VIEWS[Math.floor(Math.random() * NICE_VIEWS.length)]);
}
