import Renderer from 'webGPU-framework/renderer/render/Renderer';
import ComputeShader from 'webGPU-framework/renderer/material/ComputeShader';
import VertexFragmentShader from 'webGPU-framework/renderer/material/VertexFragmentShader';
import Vector2 from 'webGPU-framework/renderer/math/Vector2';
import { isMobile } from 'webGPU-framework/renderer/utils/Browser';
import recordRayCount from './recordRayCount.wgsl?raw';
import rayCountGraph from './rayCountGraph.wgsl?raw';

//the frames the graph shows, a bar each
const FRAME_COUNT = 512;
//the size of the graph and its distance to the left and the bottom of the screen, in CSS pixels. Above the attribution
//of MapWorldUI, and on a phone above its button bar, which spans the screen there
const WIDTH = 512;
const HEIGHT = 128;
const MARGIN_LEFT = 8;
const MARGIN_BOTTOM = isMobile() ? 100 : 32;

/*
A debug graph of the rays the sky visibility lightmap asks for per frame, in the bottom left of the screen: the cells
that won a bid on their texel times the paths per cell, before the cap of traceSamplesMax (green), and the ones it
traced, after the cap (dark green). The counts stay on the GPU:
recordRayCount.wgsl copies the one of the frame from traceArgs into a ring of FRAME_COUNT frames, rayCountGraph.wgsl
draws that ring. Every frame draw is called in gets a bar, 0 when the lightmap didn't update (see record). After a frame
without a draw (the graph off, another draw path) the graph starts empty.
 */
export default class RayCountGraph {
  private renderer: Renderer;
  private historyBuffer: GPUBuffer;
  private recordShader: ComputeShader;
  private material: VertexFragmentShader;
  private readonly canvasSize = new Vector2();
  private head = 0;
  //the frame number of the slot at head
  private frame = -2;
  //record was called in this frame
  private recorded = false;

  constructor(renderer: Renderer, traceArgsBuffer: GPUBuffer) {
    this.renderer = renderer;
    //2 words per frame: the requested and the traced rays
    this.historyBuffer = renderer.createBuffer(FRAME_COUNT * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.recordShader = new ComputeShader(renderer, 'recordRayCount', recordRayCount);
    this.recordShader.setBuffer('traceArgs', traceArgsBuffer);
    this.recordShader.setBuffer('history', this.historyBuffer);
    this.material = new VertexFragmentShader(renderer, 'rayCountGraph', rayCountGraph);
    this.material.depthWriteEnabled = false;
    this.material.depthCompare = 'always';
    //the triangles of the rect turn clockwise on the screen
    this.material.cullMode = 'none';
    this.material.setAlphaBlending();
    this.material.setBuffer('history', this.historyBuffer);
  }

  //the slot of this frame: the next one, or the first one of an empty graph after a gap
  private startFrame(): void {
    const frame = this.renderer.time.frameNumber;
    if (frame === this.frame) return;
    if (frame === this.frame + 1) {
      this.head = (this.head + 1) % FRAME_COUNT;
    } else {
      this.renderer.clearBuffer(this.historyBuffer);
      this.head = 0;
    }
    this.frame = frame;
    this.recorded = false;
  }

  private dispatchRecord(wantedIndex: number, sampleCount: number, traceCellsMax: number): void {
    this.recordShader.setFloat('head', this.head);
    this.recordShader.setFloat('wantedIndex', wantedIndex);
    this.recordShader.setFloat('sampleCount', sampleCount);
    this.recordShader.setFloat('traceCellsMax', traceCellsMax);
    this.renderer.dispatchCompute(this.recordShader);
  }

  /**
   * The rays of this frame, after select_main of the lightmap update: traceArgs[wantedIndex] (the cells that won a bid)
   * and traceArgs[3] up to traceCellsMax (the ones that trace), times sampleCount (the paths per cell of this frame).
   */
  public record(wantedIndex: number, sampleCount: number, traceCellsMax: number): void {
    this.startFrame();
    this.dispatchRecord(wantedIndex, sampleCount, traceCellsMax);
    this.recorded = true;
  }

  /**
   * Over the screen, after everything else of the frame. raysMax: the rays at the top of the graph, raysLimit: the cap
   * of the rays per frame, a red line when it is below raysMax.
   */
  public draw(raysMax: number, raysLimit: number): void {
    this.startFrame();
    //the lightmap didn't update this frame
    if (!this.recorded) this.dispatchRecord(4, 0, 0);

    const canvas = this.renderer.canvas;
    const ratio = this.renderer.canvasManager.pixelRatio;
    const left = MARGIN_LEFT * ratio;
    const width = Math.max(1, Math.min(WIDTH * ratio, canvas.width - 2 * left));
    const height = Math.max(1, Math.min(HEIGHT * ratio, canvas.height / 3));
    const top = Math.max(0, canvas.height - MARGIN_BOTTOM * ratio - height);
    this.material.setUniform('rect', [left, top, width, height]);
    this.material.setVector2('canvasSize', this.canvasSize.setValues(canvas.width, canvas.height));
    this.material.setFloat('head', this.head);
    this.material.setFloat('raysMax', Math.max(raysMax, 1));
    this.material.setFloat('raysLimit', raysLimit < raysMax ? raysLimit : 0);
    this.material.setFloat('lineWidth', Math.max(1, Math.round(ratio)));
    this.renderer.startRenderPass(null, { colorClear: false, depthClear: false }, 0, 'rayCountGraph');
    this.renderer.drawVertices(6, this.material);
  }

  public destruct(): void {
    this.historyBuffer.destroy();
  }
}
