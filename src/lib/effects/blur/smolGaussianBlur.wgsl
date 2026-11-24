//0 downscale, 1 separable gaussian, 2 cubic B-spline upscale, see SmolGaussianBlur
override PASS: u32 = 0;

struct Uniforms {
	//downscale: one texel of the source. gaussian: one texel of the source along the blur direction, 0 across it
	texelStep: vec2<f32>,
	//maps this pass onto an exact rectangle of the source, see vs_main
	uvScale: vec2<f32>,
	//gaussian: the standard deviation in texels of the level it runs on
	sigma: f32,
	//gaussian: the texels per side the kernel reaches, 4 sigma rounded up
	taps: u32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var texture: texture_2d<f32>;

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	//The levels are exact 2:1 rectangles of their parent from the top left (the leftover row/column of an odd size
	//falls off), so the downscale reads 2x its own size and the upscale reads a level as covering 2^k of its texels
	//per source pixel. uvScale keeps every pixel on the same spot of the texel grid it reads, without it the grid
	//drifts across the image and the kernel degenerates in bands.
	output.uv = input.uv * uniforms.uvScale;
	return output;
}

fn ts(uv: vec2<f32>) -> vec4<f32> {
	return textureSampleLevel(texture, textureSampler, uv, 0.);
}

//The tent of blur.wgsl: the destination pixel sits on a texel corner of the source, the 4 corner taps a texel out
//each average 2x2 texels on their own corner and the center tap (x4) the 2x2 around the pixel. Per axis that weighs
//the 4 texels .125/.375/.375/.125, a variance of .75 texels. A plain 2x2 box (.25) would be one tap, but passes
//most of what lies above the nyquist frequency of the level, and moving highlights then pump in brightness.
fn downscale(uv: vec2<f32>) -> vec4<f32> {
	let d = uniforms.texelStep;
	let c = ts(uv + d * vec2(-1., -1.)) + ts(uv + d * vec2(1., -1.)) + ts(uv + d * vec2(-1., 1.)) + ts(uv + d * vec2(1., 1.));
	return (c + ts(uv) * 4.) * .125;
}

//One axis of a gaussian, its texels paired so one bilinear fetch reads two of them. The weights come from the
//incremental gaussian (GPU Gems 3, ch. 40): g(x + 1) / g(x) = exp(-(2x + 1) / 2s^2), so stepping that ratio by
//exp(-1 / s^2) per texel needs no exp in the loop. Between 3 and 4 sigma the weights taper to 0, so a kernel that
//reaches one more texel when sigma grows does not pop.
fn gaussian(uv: vec2<f32>) -> vec4<f32> {
	let s = max(uniforms.sigma, 1e-3);
	let r = exp(-.5 / (s * s));
	let rr = r * r;
	var g = 1.;
	var ratio = r;
	var sum = ts(uv);
	var weightSum = 1.;
	let n = uniforms.taps;
	for (var x = 1u; x <= n; x += 2u) {
		//texel x
		g *= ratio;
		ratio *= rr;
		let ga = g * clamp(4. - f32(x) / s, 0., 1.);
		//texel x + 1, which is past the end of the kernel when n is odd: the taper has it at 0 there
		g *= ratio;
		ratio *= rr;
		let gb = g * clamp(4. - f32(x + 1u) / s, 0., 1.);
		let w = ga + gb;
		if (w > 0.) {
			let o = uniforms.texelStep * (f32(x) + gb / w);
			sum += (ts(uv + o) + ts(uv - o)) * w;
			weightSum += 2. * w;
		}
	}
	return sum / weightSum;
}

//Cubic B-spline in 4 bilinear fetches (Sigg & Hadwiger, GPU Gems 2, ch. 20). The kernel is positive, so unlike
//bilinear it shows no slope steps on bright HDR content when it enlarges a lot, and unlike Catmull-Rom it does not
//ring. It blurs by a variance of 1/3 texel per axis at every phase, which SmolGaussianBlur takes off the gaussian.
fn upscale(uv: vec2<f32>) -> vec4<f32> {
	let size = vec2<f32>(textureDimensions(texture));
	let st = uv * size - .5;
	let i = floor(st);
	let f = st - i;
	let f2 = f * f;
	let f3 = f2 * f;
	let w0 = (1. - f) * (1. - f) * (1. - f) / 6.;
	let w1 = (4. - 6. * f2 + 3. * f3) / 6.;
	let w2 = (1. + 3. * f + 3. * f2 - 3. * f3) / 6.;
	let w3 = f3 / 6.;
	let g0 = w0 + w1;
	let g1 = w2 + w3;
	//texel centers on i + .5, the pairs (i - 1, i) and (i + 1, i + 2) each read with one bilinear fetch
	let p0 = (i + .5 - 1. + w1 / g0) / size;
	let p1 = (i + .5 + 1. + w3 / g1) / size;
	return (ts(vec2(p0.x, p0.y)) * g0.x + ts(vec2(p1.x, p0.y)) * g1.x) * g0.y +
	       (ts(vec2(p0.x, p1.y)) * g0.x + ts(vec2(p1.x, p1.y)) * g1.x) * g1.y;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
	if (PASS == 0u) {
		return downscale(input.uv);
	}
	if (PASS == 1u) {
		return gaussian(input.uv);
	}
	return upscale(input.uv);
}
