import Renderer from 'webGPU-framework/renderer/render/Renderer';

/** A flight of the video: the plane of the airplane mode on a scripted course, see video/shots.mjs. */
export type VideoShot = {
  // where the camera is in the first recorded frame
  lat: number;
  lng: number;
  // of the camera above the ground, m
  height: number;
  // the course in compass degrees: 0 north, 90 east
  heading: number;
  // m/s, the 140 of the airplane mode when left out
  speed?: number;
  // the look down angle in radians (negative), the -.45 of the airplane mode when left out
  pitch?: number;
  // the stick over the time of the shot, [seconds, turn (-1 left .. 1 right), climb (-1 down .. 1 up)], interpolated
  // linearly between the keys. Left out: straight and level
  stick?: [number, number, number][];
  // param values of the shot by their path from MapWorld, see setParams
  params?: Record<string, number | boolean>;
};

/** What MapWorld does for the video. */
export type VideoHost = {
  // the MapWorld, for the params by path
  example: object;
  // one frame: MapWorld.update()
  step: () => void;
  // recenters the map on a place and starts streaming its tiles
  load: (lat: number, lng: number) => Promise<void>;
  // the plane at world units east / north of the map center, height and speed in m, yaw and pitch in radians
  place: (east: number, north: number, height: number, yaw: number, speed: number, pitch: number) => void;
  metersToNorm: () => number;
  // no tiles on their way and the lightmap of the view converged
  settled: () => boolean;
  // whether the camera is inside a building
  inside: () => boolean;
};

/**
 * Records a video offline (?video, driven by video/record.mjs through window.__mapWorldVideo): MapWorld doesn't run on
 * requestAnimationFrame but a frame at a time, each a step of 1 / fps s, so a recorded frame is what a screen at that
 * rate shows however long the frame and the saving of its image take. Nothing else is different: a frame of the video
 * is 1 frame of the app, the lightmap gets the rays of 1 frame. The image is read from the canvas, at a fixed size
 * whatever the window is, without the DOM over it (the GUI, the buttons, the location title).
 * Before a shot the map loads and converges at the start of the flight (not recorded), then the plane flies the lead
 * in to the first recorded frame, so the shot starts in flight.
 */
export default class MapWorldVideo {
  private frameNumber = 0;
  private fps = 60;
  // the frame of the shot, 0 the first recorded one, negative in the lead in
  private shotFrame = 0;
  private stickKeys: [number, number, number][] = [];
  // the stick of this frame for AirplaneController.update
  public stick = { x: 0, y: 0 };
  // the frames of the shot so far in which the camera was inside a building
  public insideFrames = 0;

  constructor(private renderer: Renderer, private host: VideoHost) {
    (window as any).__mapWorldVideo = this;
  }

  /** MapWorld drew its first frame: record.mjs waits for this. */
  public ready = false;

  /** The time of this frame for renderer.update, ms. */
  public nextTime(): number {
    return ++this.frameNumber * 1000 / this.fps;
  }

  /** The size of the canvas in pixels and the frame rate, then the params (see setParams). */
  public setup(width: number, height: number, fps: number, params: Record<string, number | boolean> = {}): void {
    this.renderer.canvasManager.fixedSize = { width, height };
    this.fps = fps;
    this.setParams(params);
    this.host.step();
  }

  /**
   * Param values by the path of the param from MapWorld (private fields too), like ?set of MapWorldBench:
   * { quality: 0, animateWater: true, 'tiltShift.enabled': false }. A select takes the index of its option. In the
   * order of the object: quality first, it sets most of the others.
   */
  public setParams(params: Record<string, number | boolean>): void {
    for (const [path, value] of Object.entries(params)) {
      let target: any = this.host.example;
      for (const key of path.split('.')) target = target?.[key];
      if (!target?.setValue) throw new Error(`[MapWorldVideo] no param ${path}`);
      target.setValue(value);
    }
  }

  /**
   * Loads the map at the start of a shot and waits until it is settled (at most maxFrames frames), then flies the
   * lead in. The next frame() is the first of the shot. Returns the frames it waited, -1 when it didn't settle.
   */
  public async loadShot(shot: VideoShot, leadIn: number = .5, maxFrames: number = 3600): Promise<number> {
    if (shot.params) this.setParams(shot.params);
    await this.host.load(shot.lat, shot.lng);

    const speed = shot.speed ?? 140;
    const leadFrames = Math.round(leadIn * this.fps);
    // the lead in is straight: back from the start along the course. The map center is the start
    const course = shot.heading * Math.PI / 180;
    const back = leadFrames / this.fps * speed * this.host.metersToNorm();
    this.host.place(-Math.sin(course) * back, -Math.cos(course) * back, shot.height, -course, 0, shot.pitch ?? -.45);
    this.stickKeys = [];
    this.stick = { x: 0, y: 0 };

    // standing still at the start of the lead in: the tiles load (the first are only asked for after a frame) and the
    // lightmap converges
    let frames = 0, settledFrames = 0;
    while (settledFrames < 30 && frames < maxFrames) {
      this.host.step();
      frames++;
      settledFrames = frames > 30 && this.host.settled() ? settledFrames + 1 : 0;
      // the tiles arrive in tasks of their own, the GPU keeps up
      await this.renderer.device.queue.onSubmittedWorkDone();
    }
    const settled = settledFrames >= 30;

    this.host.place(-Math.sin(course) * back, -Math.cos(course) * back, shot.height, -course, speed, shot.pitch ?? -.45);
    this.stickKeys = shot.stick ?? [];
    this.shotFrame = -leadFrames;
    for (let i = 0; i < leadFrames; i++) this.advance();
    this.insideFrames = 0;
    return settled ? frames : -1;
  }

  /** The next frame of the shot as an image. */
  public frame(type: string = 'image/png', quality?: number): Promise<Blob> {
    this.advance();
    // in the task that drew it: a WebGPU canvas only holds its image until the task ends
    return new Promise((resolve, reject) => {
      this.renderer.canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('no image of the canvas')), type, quality);
    });
  }

  /**
   * The next frame of the shot as its pixels: RGBA, 8 bits, rows from the top. For the recording instead of frame():
   * no png to encode, and the blobs of a few hundred frames fill the blob storage of the browser before it frees them.
   */
  public async framePixels(): Promise<ArrayBuffer> {
    this.advance();
    // a copy of the canvas, taken in the task that drew it
    const image = new VideoFrame(this.renderer.canvas, { timestamp: 0 });
    const options = { format: 'RGBA' } as VideoFrameCopyToOptions;
    const pixels = new Uint8Array(image.allocationSize(options));
    await image.copyTo(pixels, options);
    image.close();
    return pixels.buffer;
  }

  /** count frames of the shot without their image. */
  public async skip(count: number): Promise<void> {
    for (let i = 0; i < count; i++) {
      this.advance();
      if (i % 8 === 7) await this.renderer.device.queue.onSubmittedWorkDone();
    }
  }

  private advance(): void {
    const time = this.shotFrame++ / this.fps;
    const keys = this.stickKeys;
    let x = 0, y = 0;
    if (keys.length > 0 && time >= 0) {
      let i = 0;
      while (i < keys.length - 1 && keys[i + 1][0] <= time) i++;
      const a = keys[i], b = keys[Math.min(i + 1, keys.length - 1)];
      const t = b[0] > a[0] ? Math.min(Math.max((time - a[0]) / (b[0] - a[0]), 0), 1) : 0;
      // before the first key: its value
      x = a[1] + (b[1] - a[1]) * t;
      y = a[2] + (b[2] - a[2]) * t;
    }
    this.stick.x = x;
    this.stick.y = y;
    this.host.step();
    if (this.host.inside()) this.insideFrames++;
  }
}
