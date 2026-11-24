//for now, only use overrides in the @vertex and @fragment functions. Not in custom functions
override HALF_FILTER_WIDTH: u32 = 4;

struct Uniforms {
	blurDir : vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read> offsets : array<f32>;
@group(0) @binding(3) var textureSampler: sampler;
@group(0) @binding(4) var texture: texture_2d<f32>;

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>
}

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(input:VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	output.uv = input.uv;
  return output;
}

fn ts(uv: vec2<f32>) -> vec2<f32>
{
  return textureSample(texture, textureSampler, uv).xy;
}

//https://software.intel.com/en-us/blogs/2014/07/15/an-investigation-of-fast-real-time-gpu-based-image-blur-algorithms
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {

	var sum = vec2(0.);
	for( var i = 0u; i < HALF_FILTER_WIDTH; i++ )
	{
		let o = offsets[i] * uniforms.blurDir;
		sum += weights[i] * (ts(input.uv + o ) + ts(input.uv - o ));
	}
	return vec4(sum, 0., 0.);
}
