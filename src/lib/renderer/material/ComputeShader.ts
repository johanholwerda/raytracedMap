import Renderer from '../render/Renderer';
import ShaderBase, {ShaderBaseOptions} from './ShaderBase';
import {PipelineConstants} from './shaderUtils/OverrideUtils';
import LogGL from '../core/LogGL';

export type ComputeShaderOptions = ShaderBaseOptions & {
  kernel?: string;
  /*
  ready makes a missing pipeline with createComputePipelineAsync, so the driver compiles it without blocking the frames
  (a sync pipeline that is slow to compile stalls the whole GPU process). The caller skips its dispatches while ready is
  false. Dispatching anyway (pipeline) makes it sync, a dispatch is never skipped on its own: shaders that work together
  (one writes what the next reads) are checked together, see SkyVisibilityLightmap.update
   */
  asyncPipeline?: boolean;
}

export default class ComputeShader extends ShaderBase {
  //a constant that keeps changing (a value per frame) would otherwise grow the cache without end
  private static readonly MAX_PIPELINES = 16;
  private dirty = true;
  private kernel: string;
  private _pipeline: GPUComputePipeline | null = null;
  private shaderModule: GPUShaderModule | null = null;
  //by the values of the constants, so switching back to values used before reuses their pipeline
  private pipelines: Map<string, GPUComputePipeline> = new Map();
  private readonly asyncPipeline: boolean;
  //the constants keys of the pipelines that are being made by createComputePipelineAsync
  private pendingPipelines: Set<string> = new Set();

  public constructor(renderer: Renderer, name: string, shaderCode: string, options: ComputeShaderOptions = {}) {
    super(renderer, name, shaderCode, options);
    this.kernel = options.kernel ?? 'compute_main';
    this.asyncPipeline = options.asyncPipeline ?? false;

    this.constants = new PipelineConstants(this.shaderCode, false);
    LogGL.log('overrides', name, this.constants.names);

    this.initBindGroups();
  }

  protected constantsChanged() {
    this.dirty = true;
  }

  private getDescriptor(): GPUComputePipelineDescriptor {
    //a constant only changes the pipeline: the module and the layout are created once
    if (!this.shaderModule) {
      this.shaderModule = this.renderer.device.createShaderModule({
        code: this.shaderCode,
        label: this.name + 'Module',
      });
    }
    return {
      label:   this.name + 'Pipeline',
      layout:  this.pipelineLayout,
      compute: {
        module: this.shaderModule,
        entryPoint: this.kernel,
        constants: this.constants.values,
      },
    };
  }

  private storePipeline(key: string, pipeline: GPUComputePipeline) {
    if (this.pipelines.size >= ComputeShader.MAX_PIPELINES) {
      this.pipelines.delete(this.pipelines.keys().next().value!);
    }
    this.pipelines.set(key, pipeline);
  }

  /*
  whether the pipeline for the current constants is there. With asyncPipeline, false starts making it async (once), skip
  the dispatches until it is true. Without, always true: the pipeline getter makes it sync
   */
  public get ready(): boolean {
    if (!this.asyncPipeline) return true;
    const key = this.constants.key;
    if (this.pipelines.has(key)) return true;
    if (this.pendingPipelines.has(key)) return false;

    const descriptor = this.getDescriptor();
    this.pendingPipelines.add(key);
    const pending = this.pendingPipelines;
    this.renderer.device.createComputePipelineAsync(descriptor).then(
      (pipeline) => {
        //destructed, or made sync by a dispatch in the meantime
        if (pending !== this.pendingPipelines || !pending.delete(key) || this.pipelines.has(key)) return;
        this.storePipeline(key, pipeline);
      },
      (error) => {
        if (pending !== this.pendingPipelines || !pending.delete(key) || this.pipelines.has(key)) return;
        //a sync one reports the validation error the usual way (an uncaptured error)
        LogGL.warn('ComputeShader: createComputePipelineAsync failed, creating it sync', this.name, error);
        this.storePipeline(key, this.renderer.device.createComputePipeline(descriptor));
      },
    );
    return false;
  }

  public get pipeline(): GPUComputePipeline {
    if (this.dirty || !this._pipeline) {
      const key = this.constants.key;
      let pipeline = this.pipelines.get(key);
      if (pipeline) {
        //to the end: a Map iterates in insertion order, so the first entry is the least recently used one
        this.pipelines.delete(key);
        this.pipelines.set(key, pipeline);
      } else {
        pipeline = this.renderer.device.createComputePipeline(this.getDescriptor());
        this.storePipeline(key, pipeline);
      }
      this._pipeline = pipeline;

      this.dirty = false;
    }
    return this._pipeline;
  }

  public destruct(): void {
    super.destruct();
    this.pipelines.clear();
    //the async pipelines still being made are dropped when they are done
    this.pendingPipelines = new Set();
    this._pipeline = null;
    this.dirty = true;
  }
}
