import Renderer from 'webGPU-framework/renderer/render/Renderer';
import ParamGroup from 'webGPU-framework/utils/params/ParamGroup';
import ParamFloat from 'webGPU-framework/utils/params/ParamFloat';
import Camera from 'webGPU-framework/renderer/camera/Camera';
import RenderTexture from 'webGPU-framework/renderer/texture/RenderTexture';
import Texture from 'webGPU-framework/renderer/texture/Texture';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import Vector3 from 'webGPU-framework/renderer/math/Vector3';
import Transform from 'webGPU-framework/renderer/scene/Transform';
import Color from 'webGPU-framework/renderer/math/Color';
import ProjectionOrthographic from 'webGPU-framework/renderer/camera/ProjectionOrthographic';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import TwoPassBlur from 'webGPU-framework/effects/blur/TwoPassBlur';
import fadeShaderSource from './shadowMapFade.wgsl?raw';

/**
 * One orthographic shadow cascade: a top-of-sun camera centred on the map view
 * point. The camera is a child of `pivot` so a single lookAt aims it down the
 * sun direction; `baseSize` is the half-extent (world units) it covers.
 */
export class ShadowMapCascade {
  public readonly camera: Camera;
  public pivot: Transform = new Transform();
  // What the shading samples: layer `layer` of ShadowMap.cascadeTexture, rendered into through this target. After a
  // render it fades to latest, see ShadowMap.stepFade
  public readonly renderTexture: RenderTexture;
  public readonly layer: number;
  // the latest render, blurred
  public readonly latest: RenderTexture;
  public readonly baseSize: number;
  public size: number;
  /** Last view point this cascade was rendered around (world space). */
  public center: Vector3 = new Vector3(1e9, 1e9, 1e9);
  /** Set when the cascade must be re-rendered (sun/center moved, tiles loaded). */
  public dirty: boolean = true;
  /** The next render replaces what is shown at once instead of fading to it (the sun moved, the world recentered). */
  public instant: boolean = true;
  /** Seconds the fade to the latest render has run, Infinity when it's done. */
  public fadeTime: number = Infinity;
  /** The view projection and half-extent of renderTexture: what the shading has in shadowMapData. */
  public shownMatrix: Float32Array = new Float32Array(16);
  public shownSize: number = 0;

  constructor(renderer: Renderer, resolution: number, baseSize: number, textureFormat: GPUTextureFormat, cascadeTexture: Texture, layer: number) {
    this.baseSize = baseSize;
    this.size = baseSize;
    this.layer = layer;

    this.renderTexture = new RenderTexture(renderer, resolution, resolution, {
      formats: [textureFormat],
      depth: false,
      colorTarget: cascadeTexture.getMipView(0, layer),
    });
    this.latest = new RenderTexture(renderer, resolution, resolution, {
      formats: [textureFormat],
      depth: false,
    });

    this.camera = new Camera({ nearPlane: -1, farPlane: 1, orthographic: true, fov: 0.5 });
    this.camera.parent = this.pivot;

    this.applySize(baseSize);
  }

  /** Resize the cascade footprint (half-extent, world units). */
  public applySize(size: number): void {
    this.size = size;
    (this.camera.projection as ProjectionOrthographic).setSize(size, size);
  }

  public destruct(): void {
    // the layer belongs to ShadowMap.cascadeTexture
    this.latest.destruct();
  }
}

/**
 * Where the cascades go when they are placed ahead of a view instead of around a point (see ShadowMap.update): the
 * camera position on the ground and the horizontal direction it looks in.
 */
export type ShadowMapAhead = {
  origin: Vector3;
  forward: Vector3;
  // how far ahead of origin the view looks at the ground (the pivot): the screen center, where the tilt-shift is sharp
  focus: number;
};

/**
 * Cascaded variance shadow map for the map's sun light.
 * At most one cascade is re-rendered per frame, and only when something that affects the shadows changed. A cascade
 * fades to a new render over fadeDuration, so what is new in it (a building that loaded, the ground it just started to
 * cover) fades in instead of popping in.
 */
export default class ShadowMap {
  private renderer: Renderer;
  private shadowMapResolution: number = 512;
  public cascades: ShadowMapCascade[] = [];
  // the shown cascades, one layer each, sampled by the shading as a texture_2d_array
  public readonly cascadeTexture: Texture;
  public lightDirection: Vector3 = new Vector3(0, -1, 0);
  public readonly cascadeCount: number = 4;
  private stride = 16;
  private shadowMapData: Float32Array = new Float32Array(this.cascadeCount * this.stride);
  public shadowMapDataBuffer: GPUBuffer;
  // Scratch target the scene is rendered into before it is blurred into the
  // cascade's own texture.
  private renderTexture: RenderTexture;
  private twoPassBlur: TwoPassBlur;
  private textureFormat: GPUTextureFormat = 'rg16float';
  // Clear to "far" (1) so texels with no caster never occlude (they read as the
  // most distant possible depth, i.e. fully lit).
  private shadowMapClearColor: Color = new Color(0, 0, 0, 0);
  private updateIndex: number = 0;
  private prevLightKey: string = '';
  // User size multiplier (param) and the last total scale (user * zoom) the
  // cascades were sized for, so cascades only resize when it meaningfully drifts.
  private userScale: number = 2;
  private appliedScale: number = 0;
  // the fade of a cascade to its latest render, see stepFade
  private fadeMaterial: VertexFragmentShader;
  // a copy of the shown cascade, which the fade reads while it writes the cascade
  private shownCopy: RenderTexture;
  private fadeDuration?: ParamFloat;

  // Half-extent (world units) of each cascade at scale 1 (i.e. camera distance 1
  // and shadowSize 1). Cascades are scaled by the camera distance every frame so
  // the covered ground always tracks the view as you zoom; without that the
  // fixed footprint only covers part of the view when zoomed out, and which part
  // depends on the sun azimuth (the cascade box projects to a sun-aligned
  // ellipse on the ground).
  private static readonly BASE_SIZES = [0.25, 0.5, 1.0, 2.0];
  // Ahead of a view, the center of a cascade is on the focus of the view, or this many times its half-extent in front
  // of the camera when that is further: the camera is then just inside its back edge.
  private static readonly AHEAD_OFFSET = 0.8;
  // The view point drifting this many times the half-extent from the center of a cascade re-renders it. Ahead of a
  // view it is less: the plane flies towards the front edge.
  private static readonly REDRAW_DRIFT = 0.5;
  private static readonly REDRAW_DRIFT_AHEAD = 0.25;
  // World units above and below the center along the sun the depth range of a cascade has besides the depth of its
  // ground footprint: a unit is 900 m in the map worlds, more than the tallest building.
  private static readonly DEPTH_MARGIN = 1;

  constructor(renderer: Renderer) {
    this.renderer = renderer;

    const resolution = this.shadowMapResolution;
    this.cascadeTexture = new Texture(renderer, resolution, resolution, this.cascadeCount, {
      format: this.textureFormat,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST
        | GPUTextureUsage.COPY_SRC,
      dimension: '2d',
      viewDimension: '2d-array',
    });
    for (let i = 0; i < this.cascadeCount; i++) {
      this.cascades.push(
        new ShadowMapCascade(renderer, resolution, ShadowMap.BASE_SIZES[i], this.textureFormat, this.cascadeTexture, i),
      );
    }

    this.shadowMapDataBuffer = renderer.createBuffer(
      this.shadowMapData.byteLength,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    );

    this.renderTexture = new RenderTexture(renderer, this.shadowMapResolution, this.shadowMapResolution, {
      formats: [this.textureFormat],
      depth: true,
    });

    this.twoPassBlur = new TwoPassBlur(renderer, {filterWidth: 3, format: this.textureFormat});

    this.shownCopy = new RenderTexture(renderer, this.shadowMapResolution, this.shadowMapResolution, {
      formats: [this.textureFormat],
      depth: false,
      usage: GPUTextureUsage.COPY_DST,
    });
    this.fadeMaterial = new VertexFragmentShader(renderer, 'shadowMapFade', fadeShaderSource);
    this.fadeMaterial.depthWriteEnabled = false;
    this.fadeMaterial.depthCompare = 'always';
    this.fadeMaterial.setSampler('fadeSampler', renderer.device.createSampler({
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
    }));
    this.fadeMaterial.setTexture('shownMoments', this.shownCopy);
  }

  public init(paramGroup: ParamGroup): void {
    const pg = paramGroup.addGroup('ShadowMap');
    pg.addParamFloat('shadowSize', this.userScale, 0.25, 4, { callback: (v) => {
      this.userScale = v;
    } });
    // seconds a cascade takes to fade to a new render, 0 is at once
    this.fadeDuration = pg.addParamFloat('fadeDuration', .6, 0, 2);
  }

  /** Force every cascade to be re-rendered (e.g. on a relevant param change). With instant there is no fade to it. */
  public setDirty(instant: boolean = false): void {
    for (const c of this.cascades) {
      c.dirty = true;
      if (instant) c.instant = true;
    }
  }

  /** All geometry moved (the world was recentered): re-render every cascade, without a fade from the old place. */
  public reset(): void {
    this.setDirty(true);
  }

  /**
   * Resize the cascades to `userScale * zoomScale` (zoomScale = camera distance)
   * when that drifts enough to matter. Returns true if the cascades were resized
   * (and hence marked dirty).
   */
  private applyScale(zoomScale: number): boolean {
    const total = this.userScale * Math.max(zoomScale, 0.001);
    // Only resize on a >5% change so a slowly drifting zoom doesn't re-dirty the
    // cascades every single frame.
    if (Math.abs(total - this.appliedScale) <= this.appliedScale * 0.05) return false;
    this.appliedScale = total;
    for (const c of this.cascades) {
      c.applySize(c.baseSize * total);
      c.dirty = true;
    }
    return true;
  }

  /**
   * Update the shadow cascades around the current view point. Re-renders at most
   * one (dirty) cascade per call, round-robin, and only renders a cascade when
   * the sun moved, the view point drifted, or `forceDirty` (tiles streaming in).
   * Returns true once every cascade reflects the current state (and is done fading to it).
   *
   * With `ahead` the cascades are not centered on `center` but put in front of the camera, for a view that can't look
   * back (the airplane): otherwise most of each cascade lies behind the camera.
   */
  public update(
    lightDirection: Vector3,
    center: Vector3,
    zoomScale: number,
    forceDirty: boolean,
    drawScene: (camera: Camera) => void,
    ahead: ShadowMapAhead | null = null,
  ): boolean {
    this.lightDirection = lightDirection;

    // Resize cascades to the current zoom so the footprint tracks the view.
    this.applyScale(zoomScale);

    // Sun direction changed -> all cascades stale. At once: the reprojection of the fade can't rotate, and the shadows
    // follow the sun slider directly.
    const lightKey = `${lightDirection.x.toFixed(4)},${lightDirection.y.toFixed(4)},${lightDirection.z.toFixed(4)}`;
    if (lightKey !== this.prevLightKey) {
      this.prevLightKey = lightKey;
      this.setDirty(true);
    }
    if (forceDirty) this.setDirty();

    // View point (camera pivot) panned far enough that the shadow would lag.
    // Use a generous threshold so cascades don't re-render on every small camera
    // movement — only when the view point has drifted a meaningful distance from
    // where the cascade was last drawn.
    const drift = ahead ? ShadowMap.REDRAW_DRIFT_AHEAD : ShadowMap.REDRAW_DRIFT;
    for (const c of this.cascades) {
      if (Vector3.distance(this.cascadeCenter(c, center, ahead), c.center) > c.size * drift) c.dirty = true;
    }

    // Render a single dirty cascade per frame, round-robin so none starve.
    let rendered = -1;
    for (let n = 0; n < this.cascades.length; n++) {
      const i = (this.updateIndex + n) % this.cascades.length;
      if (this.cascades[i].dirty) {
        this.renderCascade(i, this.cascadeCenter(this.cascades[i], center, ahead), drawScene);
        this.updateIndex = (i + 1) % this.cascades.length;
        rendered = i;
        break;
      }
    }

    // the other cascades that are still fading (renderCascade did the first step of its own)
    const dt = this.renderer.time.dt;
    for (let i = 0; i < this.cascades.length; i++) {
      if (i !== rendered && this.cascades[i].fadeTime !== Infinity) this.stepFade(i, dt);
    }

    return !this.cascades.some((c) => c.dirty || c.fadeTime !== Infinity);
  }

  /** Where a cascade wants its center: on the view point, or ahead of the camera. */
  private cascadeCenter(cascade: ShadowMapCascade, center: Vector3, ahead: ShadowMapAhead | null): Vector3 {
    if (!ahead) return center;
    const distance = Math.max(ahead.focus, cascade.size * ShadowMap.AHEAD_OFFSET);
    return Vector3.add(ahead.origin, ahead.forward.clone().multiplyScalar(distance));
  }

  /**
   * One frame of the fade of a cascade to its latest render: exponential with a time constant of a fifth of
   * fadeDuration, the rest of it at once when that is over. The first step after a render also reprojects what is
   * shown to the new center and size (offset and scale, see shadowMapFade.wgsl), the steps after it have the identity.
   */
  private stepFade(index: number, dt: number, offset: Vector2 = new Vector2(0, 0), scale: number = 1): void {
    const cascade = this.cascades[index];
    const duration = this.fadeDuration?.value ?? 0;
    cascade.fadeTime += dt;
    let alpha = 1;
    if (cascade.fadeTime < duration) alpha = 1 - Math.exp(-dt * 5 / duration);
    if (alpha >= 1) cascade.fadeTime = Infinity;

    if (alpha < 1) this.renderer.copyTextureLayer(this.cascadeTexture.gpuTexture, cascade.layer, this.shownCopy.gpuTexture, 0);
    const m = this.fadeMaterial;
    m.setTexture('latestMoments', cascade.latest);
    m.setVector2('offset', offset);
    m.setFloat('scale', scale);
    m.setFloat('alpha', alpha);
    m.setFloat('fadeUncovered', index === this.cascades.length - 1 ? 1 : 0);
    this.renderer.blit(null, cascade.renderTexture, m, true);
  }

  private renderCascade(index: number, center: Vector3, drawScene: (camera: Camera) => void): void {
    const cascade = this.cascades[index];
    cascade.dirty = false;
    cascade.center = center.clone();

    // The center snapped to the texel grid of the cascade, across the sun: a cascade re-rendered around a new center
    // then rasterizes the buildings to the same texels, instead of half a texel off. Otherwise a building smaller than a
    // texel (tens of meters in the far cascades) pops in and out, and the shadows change thickness.
    const toSun = this.lightDirection;
    cascade.pivot.lookAt(center, Vector3.subtract(center, toSun));
    const right = cascade.pivot.right;
    const up = cascade.pivot.up;
    const texel = (2 * cascade.size) / this.shadowMapResolution;
    const snapped = center.clone();
    for (const axis of [right, up]) {
      const d = Vector3.dot(center, axis);
      snapped.add(axis.multiplyScalar(Math.round(d / texel) * texel - d));
    }
    cascade.pivot.lookAt(snapped, Vector3.subtract(snapped, toSun));

    // The depth range along the sun: all of the ground footprint of the cascade (longer the lower the sun is) and the
    // buildings on it. A fixed range clipped the casters at the far side of the far cascades at a low sun.
    const sinSun = Math.max(Math.abs(toSun.y), 0.1);
    const extent = cascade.size * Math.sqrt(2) * Math.sqrt(1 - sinSun * sinSun) / sinSun + ShadowMap.DEPTH_MARGIN;
    const projection = cascade.camera.projection as ProjectionOrthographic;
    projection.near = -extent;
    projection.far = extent;

    this.renderer.startRenderPass(this.renderTexture, { clearColor: this.shadowMapClearColor });
    drawScene(cascade.camera);
    this.twoPassBlur.draw(this.renderTexture, cascade.latest);

    // Where the new center is in the shown cascade (its ndc: an orthographic projection with the same orientation), and
    // the ratio of the sizes, for the reprojection of the fade.
    const shown = cascade.shownMatrix;
    const offset = new Vector2(
      shown[0] * snapped.x + shown[4] * snapped.y + shown[8] * snapped.z + shown[12],
      shown[1] * snapped.x + shown[5] * snapped.y + shown[9] * snapped.z + shown[13],
    );
    const scale = cascade.size / Math.max(cascade.shownSize, 1e-9);
    const fade = !cascade.instant && (this.fadeDuration?.value ?? 0) > 0;
    cascade.instant = false;

    // the shading samples the cascade with the new matrix from now on, the first fade step moves its content there
    const m = cascade.camera.viewProjection.m;
    cascade.shownMatrix.set(m);
    cascade.shownSize = cascade.size;
    this.shadowMapData.set(m, index * this.stride);
    this.renderer.writeBuffer(this.shadowMapDataBuffer, this.shadowMapData);

    if (fade) {
      cascade.fadeTime = 0;
      this.stepFade(index, this.renderer.time.dt, offset, scale);
    } else {
      cascade.fadeTime = Infinity;
      this.renderer.copyTextureLayer(cascade.latest.gpuTexture, 0, this.cascadeTexture.gpuTexture, cascade.layer);
    }
  }

  public destruct(): void {
    for (const c of this.cascades) c.destruct();
    this.cascadeTexture.destruct();
    this.renderTexture.destruct();
    this.shownCopy.destruct();
  }
}
