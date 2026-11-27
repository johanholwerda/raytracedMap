//the ray count graph of the sky visibility lightmap, see RayCountGraph.ts: a bar per frame for the last frames of
//history, the newest on the right, from 0 at the bottom to raysMax at the top (clamped there). Green: the rays the
//lightmap asked for, dark green: the part of them it traced (the rest fell over the cap). Red: the cap of the rays per
//frame (traceSamplesMax), when it is below raysMax

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
}

struct Uniforms {
	//the graph in canvas pixels: left, top, width, height, y down
	rect : vec4<f32>,
	canvasSize : vec2<f32>,
	//the slot of the newest bar in history
	head : u32,
	//the rays at the top of the graph
	raysMax : f32,
	//the cap of the rays, 0: no line
	raysLimit : f32,
	//of the line, in pixels
	lineWidth : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
//the requested (x) and the traced (y) rays per frame, see recordRayCount.wgsl
@group(0) @binding(1) var<storage, read> history : array<vec2<u32>>;

//the 2 triangles of the rect
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
	var corners = array<vec2<f32>, 6>(vec2(0., 0.), vec2(1., 0.), vec2(0., 1.), vec2(0., 1.), vec2(1., 0.), vec2(1., 1.));
	let pixel = uniforms.rect.xy + corners[vertexIndex] * uniforms.rect.zw;
	var output : VertexOutput;
	output.position = vec4(pixel.x / uniforms.canvasSize.x * 2. - 1., 1. - pixel.y / uniforms.canvasSize.y * 2., 0., 1.);
	return output;
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
	let frameCount = arrayLength(&history);
	let p = input.position.xy - uniforms.rect.xy;
	let column = min(u32(p.x / uniforms.rect.z * f32(frameCount)), frameCount - 1u);
	//the last column is the newest
	let index = (uniforms.head + 1u + column) % frameCount;
	//the rays at this pixel, from the bottom
	let pixelRays = (uniforms.rect.w - p.y) / uniforms.rect.w * uniforms.raysMax;
	let pixelSize = uniforms.raysMax / uniforms.rect.w;

	if(uniforms.raysLimit > 0. && abs(pixelRays - uniforms.raysLimit) < uniforms.lineWidth * .5 * pixelSize){
		return vec4(1., .15, .1, 1.);
	}
	let rays = vec2<f32>(history[index]);
	if(pixelRays < rays.y){
		return vec4(.05, .4, .12, 1.);
	}
	if(pixelRays < rays.x){
		return vec4(.2, .9, .3, 1.);
	}
	return vec4(0., 0., 0., .55);
}
