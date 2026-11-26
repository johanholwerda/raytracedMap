/**
 * Nice views, to be used as the landing views. Paste them as copied from the url (without the leading ?):
 * lat=..&lng=..&dist=..&pitch=..&yaw=..&time=..
 */
export const NICE_VIEWS: string[] = [
  'lat=4.609368&lng=-74.058333&dist=960&pitch=-25.77&yaw=11.12&time=sunset',
  'lat=1.285584&lng=103.859607&dist=915&pitch=-27.27&yaw=25.48&time=afternoon',
  'lat=55.720917&lng=37.895266&dist=535&pitch=-30.87&yaw=-68.53&time=noon',
  'lat=52.513973&lng=13.378825&dist=137&pitch=-26.03&yaw=48.48&time=afternoon',
  'lat=52.515823&lng=13.376545&dist=148&pitch=-20.80&yaw=-79.47&time=sunset',
  'lat=50.089779&lng=14.401372&dist=440&pitch=-25.70&yaw=171.34&time=afternoon',
  'lat=39.914328&lng=116.390864&dist=560&pitch=-30.96&yaw=3.73&time=sunrise',
  'lat=-31.963102&lng=115.847630&dist=533&pitch=-23.96&yaw=-55.05&time=afternoon',
  'lat=52.376002&lng=4.909186&dist=483&pitch=-22.36&yaw=81.57&time=sunset',
  'lat=41.890147&lng=12.491649&dist=222&pitch=-29.53&yaw=37.39&time=afternoon',
  'lat=43.637305&lng=-79.384320&dist=1812&pitch=-33.47&yaw=69.49&time=afternoon',
  'lat=38.897527&lng=-77.057341&dist=535&pitch=-26.20&yaw=83.70&time=sunset',
  'lat=37.589069&lng=126.974844&dist=961&pitch=-29.37&yaw=172.27&time=morning',
  'lat=22.275807&lng=114.164348&dist=1406&pitch=-30.30&yaw=-11.80&time=noon',
  'lat=68.240545&lng=14.542637&dist=961&pitch=-17.13&yaw=-66.78&time=afternoon',
  'lat=22.330710&lng=114.150867&dist=1060&pitch=-35.63&yaw=17.14&time=morning'
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
