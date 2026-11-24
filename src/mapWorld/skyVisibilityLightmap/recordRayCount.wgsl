//the rays the lightmap asked for and traced this frame into the history of the ray count graph, see RayCountGraph.ts.
//1 thread. Asked for: the cells that won a bid on their texel (counted by select_main of
//updateSkyVisibilityLightmap.wgsl, before the cap), traced: the ones of them that got a slot in traceCells, at most
//traceCellsMax. Both times the paths per cell

struct Uniforms {
	//the slot of this frame in history
	head : u32,
	//the slot of traceArgs with the cells that won a bid this frame: 4 + the parity of the frame
	wantedIndex : u32,
	//the paths per cell of this frame, 0 when the lightmap didn't update: the bar is 0
	sampleCount : u32,
	//the most cells that trace in a frame, see select_main
	traceCellsMax : u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//traceArgs of updateSkyVisibilityLightmap.wgsl, read only here
@group(0) @binding(1) var<storage, read> traceArgs : array<u32, 8>;
//the requested (x) and the traced (y) rays per frame, a ring of the frames of the graph
@group(0) @binding(2) var<storage, read_write> history : array<vec2<u32>>;

@compute @workgroup_size(1, 1, 1)
fn compute_main() {
	let requested = traceArgs[uniforms.wantedIndex];
	//traceArgs[3] counts every winner that tried to get a slot, also the ones past the cap
	let traced = min(traceArgs[3], uniforms.traceCellsMax);
	history[uniforms.head] = vec2(requested, traced) * uniforms.sampleCount;
}
