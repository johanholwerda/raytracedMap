import Renderer from 'webGPU-framework/renderer/render/Renderer';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import ComputeShader from 'webGPU-framework/renderer/material/ComputeShader';
import ShaderBase from 'webGPU-framework/renderer/material/ShaderBase';
import Color from 'webGPU-framework/renderer/math/Color';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Matrix4x4 from 'webGPU-framework/renderer/math/Matrix4x4';
import MapCamera from 'webGPU-framework/utils/camera/MapCamera';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamBool from 'webGPU-framework/utils/params/ParamBool';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import MapProjection from '../map/MapProjection';
import SkyVisibilityLightmap from '../rendering/skyVisibilityLightmap/SkyVisibilityLightmap';
import MapGBuffer from '../rendering/deferred/MapGBuffer';
import type { RoadTileSink } from '../map/WorldTileStreamer';
import RoadGraph, { RoadEdge } from './RoadGraph';
import { GROUND_TOP_Y, LAYER_GAP } from '../map/worldGeometryCore';
import skyVisibilityTrace from '../rendering/skyVisibilityLightmap/skyVisibilityTrace.wgsl?raw';
import terrainSun from '../rendering/sun/terrainSun.wgsl?raw';
import CityScene from '../city/CityScene';
import type TerrainHeightField from '../map/TerrainHeightField';
import carsShader from './cars.wgsl?raw';
import carLightingShader from './carLighting.wgsl?raw';
import compositeShader from './carLayerComposite.wgsl?raw';

// the car types of cars.wgsl: a car (a body and a cabin), a van and a bus or truck (one box each)
const TYPE_CAR = 0;
const TYPE_VAN = 1;
const TYPE_BUS = 2;
// floats per car in the instance buffer, see loadCar in cars.wgsl
const CAR_FLOATS = 12;
// bytes per car of the light the cars are drawn with, see carLighting.wgsl
const CAR_LIGHT_BYTES = 8;
// the workgroup size of carLighting.wgsl, and its threads per car (a ray each)
const LIGHTING_GROUP_SIZE = 64;
const LIGHTING_THREADS_PER_CAR = 4;
// vertices per car (2 boxes of 5 faces) and per shadow quad
const CAR_VERTICES = 60;
const SHADOW_VERTICES = 6;
// seconds a car takes to grow when it spawns and to shrink when it goes
const GROW_TIME = .6;
// how fast the drawn position and heading follow the lane (1 / s): rounds the corners and the lane changes off
const POSITION_RATE = 10;
const HEADING_RATE = 8;
// how fast a car gets to the speed of its road (1 / s)
const SPEED_RATE = 1.5;
// seconds between the counts of the road length around the camera
const COUNT_INTERVAL = .25;
// cars spawned per frame at most, so a new area fills up over a few frames
const SPAWN_PER_FRAME = 64;
// cars further than this times the radius are taken away at once (out of view), see radiusScale
const KEEP_RADIUS = 1.3;
// the width of a lane in meters, for the lane count of a road
const LANE_WIDTH_M = 3.3;
// world units over the ground under a car (Car.ground): its bottom, on the top ground layer (the paths), and the plane of
// its shadow, a layer gap above it, see LAYER_GAP of worldGeometryCore.ts
const GROUND_Y = GROUND_TOP_Y;
const SHADOW_Y = GROUND_TOP_Y + LAYER_GAP;
// the margin of the contact occlusion around a car, meters
const AO_MARGIN_M = .8;
// the length of an average car, for minPixels
const CAR_LENGTH_M = 4.5;

type CarMaterials = {
  car: VertexFragmentShader;
  shadow: VertexFragmentShader;
};

type Car = {
  edge: RoadEdge;
  // driving from a to b of the edge
  forward: boolean;
  // the distance driven along the edge
  s: number;
  speed: number;
  speedFactor: number;
  // picks the lane of a road, modulo its lane count
  lane: number;
  // world units
  halfLength: number;
  halfWidth: number;
  height: number;
  // see loadCar in cars.wgsl
  type: number;
  // the drawn position and heading, following the lane
  x: number;
  z: number;
  hx: number;
  hz: number;
  // the ground under the car (world units): the height at its center and the rise per world unit along the heading,
  // see setHeightField
  ground: number;
  slope: number;
  placed: boolean;
  // 0 to 1: grows in when it spawns, shrinks when dying
  grow: number;
  dying: boolean;
};

/**
 * Cars that drive along the roads of the streamed tiles, drawn over the finished image of MapWorld: see cars.wgsl for
 * the drawing and RoadGraph for the roads.
 *
 * Nothing else of the frame sees them, so the map stays as it converged: the gBuffer, the lightmap, the shadow map, the
 * SSAO and the TAA don't have them. They're drawn into a layer of their own every frame (with MSAA, which the TAA can't
 * give them: they move), and blended over the image after the SSAO, before the tilt-shift. The cars are lit by the sun
 * with the shadows of the buildings and by the sky visibility of the ground under them; their own shadows and a contact
 * occlusion darken the ground in that layer. The rays of those shadows and the lightmap lookup are per car, in a compute
 * pass before the draw (carLighting.wgsl), or per pixel with perPixelLighting (PER_PIXEL of cars.wgsl, no compute pass).
 *
 * The simulation is on the CPU, a car at a time: it drives along an edge at the speed of its road (in a lane, on the
 * right or the left side) and at the end of it picks one of the edges of the node, mostly the straightest one, or turns
 * around at a dead end. The cars are spawned around the camera pivot, in proportion to the road length there
 * (density), up to maxCars. Below minPixels the cars are neither moved nor drawn, the frame then doesn't change.
 */
export default class CarTraffic implements RoadTileSink {
  private renderer: Renderer;
  private projection: MapProjection;
  private readonly graph = new RoadGraph();
  private heightField: TerrainHeightField | null = null;
  private cars: Car[] = [];
  private instanceData = new Float32Array(0);
  private instanceBuffer: GPUBuffer | null = null;
  // the light of each car, written by lightingShader and read by the vertices of the materials, see carLighting.wgsl
  private carLightBuffer: GPUBuffer | null = null;
  private readonly lightingShader: ComputeShader;
  private layer: RenderTexture | null = null;
  private layerSampleCount = 0;
  // the cars and their shadow quads lit per car (carLighting.wgsl) and per pixel (PER_PIXEL of cars.wgsl), see
  // perPixelLighting
  private readonly perCar: CarMaterials;
  private readonly perPixel: CarMaterials;
  private readonly compositeMaterial: VertexFragmentShader;
  // bound as the city of the rays of lightingShader and the perPixel materials until CityScene has its buffer, see
  // bindCity
  private readonly dummyCityBuffer: GPUBuffer;
  private readonly clearColor = new Color(0, 0, 0, 0);
  private readonly depthParams = new Vector2();
  private readonly resolution = new Vector2();
  private desired = 0;
  private countAge = Infinity;
  private countVersion = -1;
  private countX = 0;
  private countZ = 0;
  private countRadius = 0;
  // the pick of turn(), reused
  private readonly candidates: RoadEdge[] = [];
  private readonly weights: number[] = [];
  private _active = false;
  // toward the sun, for the rays of lightingShader: set by MapWorld before the draw
  public readonly lightDirection = new Vector3(0, 1, 0);

  public enabled!: ParamBool;
  // cars per km of road around the camera
  private density!: ParamFloat;
  private maxCars!: ParamFloat;
  // the radius cars live in, times the camera distance
  private radiusScale!: ParamFloat;
  private speedScale!: ParamFloat;
  private driveOnLeft!: ParamBool;
  // set by the quality of MapWorld: off where the map has no TAA either
  public msaa!: ParamBool;
  // set by the quality of MapWorld: the sun rays and the lightmap lookup per pixel instead of per car
  public perPixelLighting!: ParamBool;
  // the length of a car in pixels at the pivot below which there are no cars
  private minPixels!: ParamFloat;
  private aoStrength!: ParamFloat;
  private shadowStrength!: ParamFloat;

  constructor(renderer: Renderer, projection: MapProjection) {
    this.renderer = renderer;
    this.projection = projection;
    this.dummyCityBuffer = renderer.createBuffer(16, GPUBufferUsage.STORAGE, false, 'carsDummyCityBuffer');
    this.perCar = this.createMaterials('', false);
    this.perPixel = this.createMaterials('PerPixel', true);
    this.lightingShader = new ComputeShader(renderer, 'carLighting', carLightingShader, {
      includes: [...SkyVisibilityLightmap.resolveIncludes, skyVisibilityTrace],
      defines: CityScene.traceDefines,
      kernel: 'main',
    });
    this.lightingShader.setBuffer('cityBuffer', this.dummyCityBuffer);
    this.compositeMaterial = new VertexFragmentShader(renderer, 'carLayerComposite', compositeShader, { asyncPipeline: true });
    this.compositeMaterial.setPremultipliedAlphaBlending();
    this.compositeMaterial.depthWriteEnabled = false;
    this.compositeMaterial.depthCompare = 'always';
  }

  //the car and shadow materials. perPixel: PER_PIXEL of cars.wgsl, the rays and the lightmap lookup in the fragments
  private createMaterials(suffix: string, perPixel: boolean): CarMaterials {
    const create = (name: string, shadow: boolean) => new VertexFragmentShader(this.renderer, name + suffix, carsShader, {
      // getGBufferDistance (mapGBufferPacking.wgsl is one of them); per car the lightmap lookup is in lightingShader
      // terrainSun.wgsl: the hills in the way of the sun, which the rays don't trace
      includes: perPixel ? [...SkyVisibilityLightmap.resolveIncludes, skyVisibilityTrace, terrainSun] : [...SkyVisibilityLightmap.resolveIncludes, terrainSun],
      defines: { SHADOW: shadow, PER_PIXEL: perPixel, ...(perPixel ? CityScene.traceDefines : {}) },
      // drawn every frame: the cars are missing for the frames that the pipeline compiles in, instead of stalling them
      asyncPipeline: true,
    });
    const car = create('cars', false);
    car.cullMode = 'back';
    const shadow = create('carShadows', true);
    shadow.cullMode = 'none';
    shadow.depthWriteEnabled = false;
    shadow.depthCompare = 'always';
    // the darkest of the shadows that overlap, not their product: the color is the fog of the ground point (the same for
    // both) times the alpha, so the max of each picks the same shadow
    shadow.setBlendState({
      color: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
      alpha: { operation: 'max', srcFactor: 'one', dstFactor: 'one' },
    });
    if (perPixel) {
      car.setBuffer('cityBuffer', this.dummyCityBuffer);
      shadow.setBuffer('cityBuffer', this.dummyCityBuffer);
    }
    return { car, shadow };
  }

  /** The materials that take the shading params (sky, sun, fog, shadows) by name, the shadow map and the lightmap. */
  public get materials(): VertexFragmentShader[] {
    return [this.perCar.car, this.perCar.shadow, this.perPixel.car, this.perPixel.shadow];
  }

  /** The shaders that read the lightmap (SkyVisibilityLightmap.addResolveMaterial): the materials and the lighting pass. */
  public get resolveShaders(): ShaderBase[] {
    return [...this.materials, this.lightingShader];
  }

  /** The city the ray traced shadows trace: CityScene sets its buffer and root on the lighting pass and the perPixel materials. */
  public bindCity(cityScene: CityScene): void {
    cityScene.bind(this.lightingShader, false);
    cityScene.bind(this.perPixel.car, false);
    cityScene.bind(this.perPixel.shadow, false);
  }

  /**
   * Where the shadows of the buildings on the cars come from, the override SUN_SHADOWS of cars.wgsl: 0 none, 1 the shadow
   * map, 2 rays to the sun per car or per pixel (bindCity, its buffer enabled)
   */
  public set sunShadows(value: number) {
    for (const m of [...this.materials, this.lightingShader]) m.setConstant('SUN_SHADOWS', value);
  }

  public init(paramGroup: ParamGroup): void {
    const pg = paramGroup.addGroup('Cars');
    const reset = { callback: () => this.clearCars() };
    this.enabled = pg.addParamBool('enabled', true, reset);
    this.density = pg.addParamFloat('density', 12.5, 1, 100);
    this.maxCars = pg.addParamFloat('maxCars', isMobile() ? 750 : 2000, 100, 20000, { step: 100 });
    this.radiusScale = pg.addParamFloat('radius', 2.5, .5, 6);
    this.speedScale = pg.addParamFloat('speed', 1, 0, 4);
    this.driveOnLeft = pg.addParamBool('driveOnLeft', false);
    this.msaa = pg.addParamBool('msaa', true);
    this.perPixelLighting = pg.addParamBool('perPixelLighting', true);
    this.minPixels = pg.addParamFloat('minPixels', 2, 0, 10);
    this.aoStrength = pg.addParamFloat('aoStrength', .45, 0, 1);
    this.shadowStrength = pg.addParamFloat('shadowStrength', .85, 0, 1);
    pg.collapsed = true;
  }

  // --- RoadTileSink ---------------------------------------------------------------

  public addTile(key: string, roads: Float32Array): void {
    this.graph.setMetersToNorm(this.projection.metersToNorm);
    this.graph.addTile(key, roads);
  }

  public removeTile(key: string): void {
    this.graph.removeTile(key);
  }

  public clear(): void {
    this.graph.clear();
    this.clearCars();
  }

  // --- simulation -------------------------------------------------------------------

  /** Whether the last update moved the cars: they are drawn this frame, and the frame changes because of them. */
  public get active(): boolean {
    return this._active;
  }

  private clearCars(): void {
    this.cars.length = 0;
    this.countAge = Infinity;
  }

  /** Moves the cars by dt seconds, spawns and removes them around the camera pivot. viewHeight: in pixels. */
  public update(dt: number, camera: MapCamera, viewHeight: number): void {
    this._active = false;
    if (!this.enabled?.value) return;
    const mtn = this.projection.metersToNorm;
    // pixels per meter at the pivot
    const pixelsPerMeter = viewHeight / (2 * Math.tan(camera.projection.FOV * .5) * camera.distance / mtn);
    if (pixelsPerMeter * CAR_LENGTH_M < this.minPixels.value) return;
    this._active = true;

    const pivot = camera.pivot.worldPosition;
    const px = pivot.x, pz = pivot.z;
    const radius = this.radiusScale.value * camera.distance;
    this.updateDesired(dt, px, pz, radius);

    // gone with their tile, or far out of view
    const keep2 = (radius * KEEP_RADIUS) ** 2;
    let living = 0;
    for (let i = this.cars.length - 1; i >= 0; i--) {
      const car = this.cars[i];
      if (!car.edge.alive || (car.x - px) ** 2 + (car.z - pz) ** 2 > keep2) {
        this.removeCar(i);
      } else if (!car.dying) {
        living++;
      }
    }
    // too many for the area (zoomed in, a param): the extra ones shrink away
    if (living > this.desired * 1.1 + 2) {
      let excess = living - this.desired;
      for (let i = 0; i < this.cars.length && excess > 0; i++) {
        const car = this.cars[i];
        if (car.dying || Math.random() < .5) continue;
        car.dying = true;
        excess--;
      }
    }
    const spawn = Math.min(SPAWN_PER_FRAME, Math.floor(this.desired) - living);
    for (let i = 0; i < spawn; i++) {
      const edge = this.graph.sampleEdge(px, pz, radius);
      if (!edge) break;
      this.cars.push(this.createCar(edge, mtn));
    }

    for (let i = this.cars.length - 1; i >= 0; i--) {
      if (!this.moveCar(this.cars[i], dt, mtn)) this.removeCar(i);
    }
  }

  //how many cars the area around the pivot holds, counted again a few times a second when it changed
  private updateDesired(dt: number, x: number, z: number, radius: number): void {
    this.countAge += dt;
    const moved = Math.hypot(x - this.countX, z - this.countZ) > radius * .1 || Math.abs(radius / this.countRadius - 1) > .1;
    if (this.countAge < COUNT_INTERVAL || (!moved && this.countVersion === this.graph.version)) return;
    this.countAge = 0;
    this.countVersion = this.graph.version;
    this.countX = x;
    this.countZ = z;
    this.countRadius = radius;
    const meters = this.graph.lengthWithin(x, z, radius) / this.projection.metersToNorm;
    this.desired = Math.min(this.maxCars.value, this.density.value * meters / 1000);
  }

  private removeCar(index: number): void {
    const last = this.cars.pop()!;
    if (index < this.cars.length) this.cars[index] = last;
  }

  private createCar(edge: RoadEdge, mtn: number): Car {
    const r = Math.random();
    const type = r < .04 ? TYPE_BUS : r < .14 ? TYPE_VAN : TYPE_CAR;
    const between = (a: number, b: number) => (a + Math.random() * (b - a)) * mtn;
    let length: number, width: number, height: number;
    if (type === TYPE_BUS) {
      length = between(10, 12); width = 2.5 * mtn; height = between(3, 3.3);
    } else if (type === TYPE_VAN) {
      length = between(5, 5.6); width = 2 * mtn; height = between(2.1, 2.5);
    } else {
      length = between(4, 4.8); width = between(1.75, 1.85); height = between(1.4, 1.55);
    }
    const speedFactor = (type === TYPE_CAR ? .85 : .75) + Math.random() * .3;
    return {
      edge,
      forward: edge.oneway || Math.random() < .5,
      s: Math.random() * edge.length,
      speed: edge.speed * speedFactor,
      speedFactor,
      lane: Math.floor(Math.random() * 8),
      halfLength: length * .5,
      halfWidth: width * .5,
      height,
      type,
      x: 0, z: 0, hx: 1, hz: 0,
      ground: 0, slope: 0,
      placed: false,
      grow: 0,
      dying: false,
    };
  }

  //false when the car is gone: shrunk away, or at the end of a one way road that goes nowhere
  private moveCar(car: Car, dt: number, mtn: number): boolean {
    if (car.dying) {
      car.grow -= dt / GROW_TIME;
      if (car.grow <= 0) return false;
    } else {
      car.grow = Math.min(1, car.grow + dt / GROW_TIME);
    }
    const target = car.edge.speed * car.speedFactor * this.speedScale.value;
    car.speed += (target - car.speed) * Math.min(1, dt * SPEED_RATE);
    car.s += car.speed * dt;
    for (let k = 0; k < 16 && car.s > car.edge.length; k++) {
      car.s -= car.edge.length;
      if (!this.turn(car)) return false;
    }
    car.s = Math.min(car.s, car.edge.length);

    const e = car.edge;
    const dx = car.forward ? e.dx : -e.dx;
    const dz = car.forward ? e.dz : -e.dz;
    const start = car.forward ? e.a : e.b;
    const offset = this.getLaneOffset(car, e, mtn) * (this.driveOnLeft.value ? -1 : 1);
    // the right side of the heading is (-dz, dx)
    const tx = start.x + dx * car.s - dz * offset;
    const tz = start.z + dz * car.s + dx * offset;
    if (!car.placed) {
      car.placed = true;
      car.x = tx; car.z = tz; car.hx = dx; car.hz = dz;
      this.placeOnGround(car);
      return true;
    }
    const a = 1 - Math.exp(-dt * POSITION_RATE);
    car.x += (tx - car.x) * a;
    car.z += (tz - car.z) * a;
    const b = 1 - Math.exp(-dt * HEADING_RATE);
    let hx = car.hx + (dx - car.hx) * b;
    let hz = car.hz + (dz - car.hz) * b;
    const l = Math.hypot(hx, hz);
    // turned around: through 0
    if (l < 1e-3) {
      hx = dx; hz = dz;
    } else {
      hx /= l; hz /= l;
    }
    car.hx = hx; car.hz = hz;
    this.placeOnGround(car);
    return true;
  }

  /**
   * The ground under the car from the height field: the slope from the ground under its front and back, its bottom on
   * the line between them, not under the ground at its center (on a crest the ends float a bit instead). Flat at 0 where
   * there are no heights (the elevation off).
   */
  private placeOnGround(car: Car): void {
    const field = this.heightField;
    if (!field) {
      car.ground = car.slope = 0;
      return;
    }
    const ox = car.hx * car.halfLength, oz = car.hz * car.halfLength;
    const center = field.heightAt(car.x, car.z) ?? 0;
    const front = field.heightAt(car.x + ox, car.z + oz) ?? center;
    const back = field.heightAt(car.x - ox, car.z - oz) ?? center;
    car.slope = car.halfLength > 0 ? (front - back) / (2 * car.halfLength) : 0;
    car.ground = Math.max(center, (front + back) * .5);
  }

  /** The heights of the ground the cars drive on, null: flat at 0. */
  public setHeightField(field: TerrainHeightField | null): void {
    this.heightField = field;
  }

  //the distance of the lane of the car right of the center line: on a two way road in the right half
  private getLaneOffset(car: Car, e: RoadEdge, mtn: number): number {
    const widthM = e.width / mtn;
    if (e.oneway) {
      const lanes = Math.max(1, Math.round(widthM / LANE_WIDTH_M));
      return ((car.lane % lanes) + .5) * (e.width / lanes) - e.width * .5;
    }
    const lanes = Math.max(1, Math.round(widthM * .5 / LANE_WIDTH_M));
    return ((car.lane % lanes) + .5) * (e.width * .5 / lanes);
  }

  //onto the next edge at the end of this one: mostly the straightest, a one way road only along it. Turns around at a
  //dead end, false at the end of a one way road
  private turn(car: Car): boolean {
    const e = car.edge;
    const node = car.forward ? e.b : e.a;
    const inX = car.forward ? e.dx : -e.dx;
    const inZ = car.forward ? e.dz : -e.dz;
    const candidates = this.candidates;
    const weights = this.weights;
    candidates.length = 0;
    weights.length = 0;
    let total = 0;
    for (const c of node.edges) {
      if (c === e || !c.alive) continue;
      const fromA = c.a === node;
      if (!fromA && c.oneway) continue;
      const dot = fromA ? inX * c.dx + inZ * c.dz : -(inX * c.dx + inZ * c.dz);
      const straight = Math.max(0, dot + .35);
      const w = (.08 + straight * straight) * (c.speed >= e.speed * .6 ? 1 : .4);
      candidates.push(c);
      weights.push(w);
      total += w;
    }
    if (candidates.length === 0) {
      if (e.oneway) return false;
      car.forward = !car.forward;
      return true;
    }
    let pick = Math.random() * total;
    let i = 0;
    while (i < candidates.length - 1 && (pick -= weights[i]) > 0) i++;
    const next = candidates[i];
    car.edge = next;
    car.forward = next.a === node;
    return true;
  }

  // --- drawing ----------------------------------------------------------------------

  /**
   * The cars of the last update over destination (null: the canvas), after the image is in it. viewProjection: of the
   * camera without the jitter of the TAA. camera: the one the gBuffer was drawn with, for its distances.
   */
  public draw(camera: MapCamera, viewProjection: Matrix4x4, gBuffer: MapGBuffer, destination: RenderTexture | null): void {
    const count = this.writeInstances();
    if (count === 0) return;
    const layer = this.getLayer();
    const mtn = this.projection.metersToNorm;
    MapGBuffer.depthParams(camera, this.depthParams);
    this.resolution.setValues(gBuffer.target.width, gBuffer.target.height);
    const perPixel = this.perPixelLighting.value;
    const materials = perPixel ? this.perPixel : this.perCar;
    for (const m of [materials.car, materials.shadow]) {
      m.setMatrix('viewProjection', viewProjection);
      m.setMatrix('inverseProjection', camera.projectionMatrix.inverse);
      m.setVector3('cameraPosition', camera.worldPosition);
      m.setVector2('resolution', this.resolution);
      m.setVector2('depthParams', this.depthParams);
      m.setFloat('groundY', GROUND_Y);
      m.setFloat('shadowY', SHADOW_Y);
      m.setFloat('aoMargin', AO_MARGIN_M * mtn);
      m.setFloat('aoStrength', this.aoStrength.value);
      m.setFloat('carShadowStrength', this.shadowStrength.value);
      m.setTexture('gBufferDepth', gBuffer.depthView);
      m.setBuffer('cars', this.instanceBuffer!);
      if (!perPixel) m.setBuffer('carLight', this.carLightBuffer!);
    }
    if (!perPixel) {
      // the light of each car before the draw reads it
      const lighting = this.lightingShader;
      lighting.setMatrix('viewProjection', viewProjection);
      lighting.setVector3('cameraPosition', camera.worldPosition);
      lighting.setVector3('lightDirection', this.lightDirection);
      lighting.setFloat('groundY', GROUND_Y);
      lighting.setFloat('carCount', count);
      lighting.setBuffer('cars', this.instanceBuffer!);
      lighting.setBuffer('carLight', this.carLightBuffer!);
      this.renderer.dispatchCompute(lighting, Math.ceil(count * LIGHTING_THREADS_PER_CAR / LIGHTING_GROUP_SIZE));
    }
    this.renderer.startRenderPass(layer, { clearColor: this.clearColor, colorClear: true, depthClear: true }, 0, 'cars');
    this.renderer.drawVertices(SHADOW_VERTICES, materials.shadow, null, { instanceCount: count });
    this.renderer.drawVertices(CAR_VERTICES, materials.car, null, { instanceCount: count });
    this.renderer.blit(layer, destination, this.compositeMaterial, false);
  }

  //the cars into the instance buffer, grown to fit. Returns their count
  private writeInstances(): number {
    const cars = this.cars;
    if (cars.length === 0) return 0;
    if (this.instanceData.length < cars.length * CAR_FLOATS) {
      const capacity = Math.max(cars.length, Math.ceil(this.instanceData.length / CAR_FLOATS * 1.5), 256);
      this.instanceData = new Float32Array(capacity * CAR_FLOATS);
      this.instanceBuffer?.destroy();
      this.instanceBuffer = this.renderer.createBuffer(this.instanceData.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, false, 'cars');
      this.carLightBuffer?.destroy();
      this.carLightBuffer = this.renderer.createBuffer(capacity * CAR_LIGHT_BYTES, GPUBufferUsage.STORAGE, false, 'carLight');
    }
    const data = this.instanceData;
    for (let i = 0; i < cars.length; i++) {
      const car = cars[i];
      // eased, so a car pops less at its start and its end
      const g = car.grow * car.grow * (3 - 2 * car.grow);
      const o = i * CAR_FLOATS;
      data[o] = car.x;
      data[o + 1] = car.z;
      data[o + 2] = car.hx;
      data[o + 3] = car.hz;
      data[o + 4] = car.halfLength * g;
      data[o + 5] = car.halfWidth * g;
      data[o + 6] = car.height * g;
      data[o + 7] = car.type;
      data[o + 8] = car.ground;
      data[o + 9] = car.slope;
    }
    this.renderer.writeBuffer(this.instanceBuffer!, data.subarray(0, cars.length * CAR_FLOATS));
    return cars.length;
  }

  //the layer the cars are drawn into, at the canvas size, made again when the msaa param changed
  private getLayer(): RenderTexture {
    const sampleCount = this.msaa.value ? 4 : 1;
    if (!this.layer || this.layerSampleCount !== sampleCount) {
      this.layer?.destruct();
      this.layerSampleCount = sampleCount;
      this.layer = new RenderTexture(this.renderer, 1, 1, {
        formats: ['rgba8unorm'],
        depth: true,
        sampleCount,
        scaleToCanvas: true,
        toScreen: false,
        // only the resolved color is read: the samples and the depth stay on chip on a tile based GPU
        discardMSAA: true,
        discardDepth: true,
      });
    }
    return this.layer;
  }

  public destruct(): void {
    this.layer?.destruct();
    this.instanceBuffer?.destroy();
    this.carLightBuffer?.destroy();
    this.dummyCityBuffer.destroy();
    this.graph.clear();
  }
}
