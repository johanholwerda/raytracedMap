struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>,
	@location(2) normal : vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
	@location(0) side : f32,
	@location(1) color : vec3<f32>,
}

struct Uniforms {
	resolution: vec2<f32>,
	viewProjection : mat4x4<f32>,
}

struct BoundingBox {
	bbMin: vec4<f32>,   // .xyz = min, .w = unused
	bbMax: vec4<f32>,   // .xyz = max, .w = unused
	color: vec4<f32>,   // .xyz = color, .w = unused
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> boundingBoxes: array<BoundingBox>;

@vertex
fn vs_main(
	input: VertexInput,
	@builtin(instance_index) instanceIndex : u32
) -> VertexOutput {
    var output: VertexOutput;

	let box = boundingBoxes[instanceIndex];

	let bbMin = box.bbMin.xyz;
	let bbMax = box.bbMax.xyz;

	if(bbMin.x > bbMax.x){
		output.position = vec4(0.0);
	}else{
		let p0 = mix(bbMin, bbMax, input.position * .5 + .5);
		let clipPos = uniforms.viewProjection * vec4(p0, 1.0);

		let edgeDir = input.normal;
		let clipPos2 = uniforms.viewProjection * vec4(p0 + edgeDir * 0.1, 1.0);

		let screenPos1 = clipPos.xy / clipPos.w;
		let screenPos2 = clipPos2.xy / clipPos2.w;

		var screenDir = normalize(screenPos2 - screenPos1);
		if (length(screenPos2 - screenPos1) < 0.0001) {
			screenDir = vec2<f32>(1.0, 0.0);
		}

		let sideDir = vec2<f32>(-screenDir.y, screenDir.x);
		let side = input.uv.x * 2. - 1.0;

		let thickness = (2. * clipPos.w) / uniforms.resolution.y;

		output.position = vec4<f32>(clipPos.xy + sideDir * side * thickness, clipPos.z, clipPos.w);
		output.side = side;
	}

//	output.color = box.color.xyz;
	output.color = hash13uf(instanceIndex);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	let a = smoothstep(1., 0., abs(input.side));
	return vec4<f32>(input.color, a * .5);
}
