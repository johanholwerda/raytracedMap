// The shots of the video, recorded by record.mjs (npm run video). Change a shot, then
//   npm run video -- --stills --shots tokyo     a few stills of the shot in out/stills, to check the flight
//   npm run video -- --shots tokyo              records the shot again and joins all shots into out/raytracedMap.mp4

export const video = {
  width: 1920,
  height: 1080,
  fps: 60,
  // seconds the plane already flies before the first frame of a shot, so a shot starts in flight
  leadIn: .5,
  // of the joined video (x264): lower is better and larger, 16 is close to lossless to the eye. X encodes it again
  crf: 16,
  // The params of the GUI by their path (see MapWorld.findParam), in this order: the quality sets
  // most of the others. A select takes the index of its option: quality 0 is very high, 1 high, 2 medium.
  params: {
    quality: 0,
    // the qualities turn it off (a still view then stops being drawn), in a flight it costs nothing extra
    animateWater: true,
    // half the blur of the tilt-shift (.03 in the app): the plane looks ahead more than down, the skyline it flies to
    // is far behind the focus
    'tiltShift.aperture': .012,
  },
};

// A shot is a flight of the airplane mode (the airplane button), on a scripted stick instead of the keys:
//   name       of its file in out/shots, and for --shots
//   seconds    its length
//   timeOfDay  sunrise, morning, noon, afternoon or sunset
//   lat, lng   where the camera is in the first frame
//   height     of the camera above the ground, m. The airplane mode flies at 120 to 800
//   heading    the course in compass degrees: 0 north, 90 east, 180 south, 270 west
//   speed      m/s, 140 (the airplane mode) when left out
//   pitch      the look down angle in radians, -.45 (the airplane mode) when left out. Closer to 0 shows more skyline
//   stick      [seconds, turn, climb] keys, interpolated: turn -1 (full left) to 1 (full right), climb -1 (down) to 1
//              (up). A full turn is .7 radians (40 degrees) per second. Left out: straight and level
//   params     param values of this shot only, like video.params (they stay set for the shots after it)
// The video plays them in this order.
export const shots = [
  // over the harbor toward the Battery and the towers of Lower Manhattan
  { name: 'newyork', seconds: 5.5, timeOfDay: 'sunset', lat: 40.6935, lng: -74.0205, height: 150, heading: 27, pitch: -.25 },
  // down the Amstel toward the canal belt
  { name: 'amsterdam', seconds: 5.5, timeOfDay: 'afternoon', lat: 52.3625, lng: 4.9040, height: 90, heading: 320, pitch: -.4 },
  // over the skyscrapers of Nishi-Shinjuku toward Shinjuku Gyoen
  { name: 'tokyo', seconds: 5.5, timeOfDay: 'afternoon', lat: 35.6915, lng: 139.6850, height: 280, heading: 110, pitch: -.35 },
  // over the trees of the Ramble and the Lake in Central Park, toward the towers of 59th Street
  { name: 'centralpark', seconds: 5.5, timeOfDay: 'sunrise', lat: 40.7835, lng: -73.9659, height: 130, heading: 205, pitch: -.3 },
  // up the Thames to Tower Bridge, the City on the right and the Shard on the left
  { name: 'london', seconds: 5.5, timeOfDay: 'afternoon', lat: 51.5035, lng: -.0640, height: 120, heading: 292, pitch: -.3 },
  // over Victoria Harbour to the skyline of Central
  { name: 'hongkong', seconds: 5.5, timeOfDay: 'sunrise', lat: 22.2935, lng: 114.1680, height: 280, heading: 225, pitch: -.22 },
  // from the Bund over the Huangpu to the towers of Lujiazui
  { name: 'shanghai', seconds: 5.5, timeOfDay: 'sunset', lat: 31.2360, lng: 121.4810, height: 170, heading: 85, pitch: -.2 },
];
