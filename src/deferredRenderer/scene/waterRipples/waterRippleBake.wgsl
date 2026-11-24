//the micro normal of the water (see WaterRippleTexture.ts): three octaves of gradient noise summed into one tiling
//gradient, redrawn every frame. The water reads it with a single tap instead of summing the noise per pixel.
//rg the gradient of the height, (dh/dx, dh/dz) per world unit, scaled by rippleStrength. The gBuffer shader turns it into
//a normal. A gradient rather than a normal, so the mips average the slopes right
//
//It tiles because the hash is wrapped to TILE_CELLS lattice cells and every octave is sampled in a frame that is an
//integer matrix: that maps the lattice onto itself, so a step of the tile is a whole number of cells in every octave.
//It is redrawn rather than baked once because the octaves drift at their own speeds in their own directions (the
//dispersion of water waves, sqrt(g * lambda / 2pi)), so the sum depends on time and isn't a rigid pattern sliding by

const TAU = 6.283185307179586;
const GRAVITY = 9.81;
//octave 0 lattice cells across the tile, the period of the hash. Mirrors TILE_CELLS of WaterRippleTexture.ts
const TILE_CELLS = 12.;
//the frame of octave i is this matrix applied i times: a rotation of about 27 degrees and a scale of sqrt(5). Different
//frames so the lattices of the octaves don't line up into a visible grid, integer entries so they still tile
const OCTAVE = mat2x2<f32>(2., -1., 1., 2.);

struct VertexInput {
	@location(0) position : vec3<f32>,
	@location(1) uv : vec2<f32>,
}

struct VertexOutput {
	@builtin(position) position : vec4<f32>,
	@location(0) uv : vec2<f32>,
}

struct Uniforms {
	//seconds
	time : f32,
	//the wavelength of the coarsest octave in world units, the tile is TILE_CELLS of them
	wavelength : f32,
	//multiplies the drift of the dispersion relation
	speed : f32,
	//the height of the ripples, scales their gradient
	rippleStrength : f32,
}

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn vs_main(input : VertexInput) -> VertexOutput {
	var output : VertexOutput;
	output.position = vec4(input.position, 1.);
	output.uv = input.uv;
	return output;
}

//x mod y with the sign of y, like glsl mod (the % of wgsl truncates)
fn modFloor(x : vec2<f32>, y : f32) -> vec2<f32> {
	return x - floor(x / y) * y;
}

//a lattice corner gradient, uniform on the circle. Wrapped to the tile before hashing, which makes the field periodic
fn gradientHash(cell : vec2<f32>) -> vec2<f32> {
	let p = modFloor(cell, TILE_CELLS);
	var p3 = fract(p.xyx * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx + 33.33);
	let a = fract((p3.x + p3.y) * p3.z) * TAU;
	return vec2(cos(a), sin(a));
}

//the derivatives of gradient noise, https://iquilezles.org/articles/gradientnoise/
fn noiseGradient(x : vec2<f32>) -> vec2<f32> {
	let i = floor(x);
	let f = fract(x);

	let u = f * f * f * (f * (f * 6. - 15.) + 10.);
	let du = 30. * f * f * (f * (f - 2.) + 1.);

	let ga = gradientHash(i);
	let gb = gradientHash(i + vec2(1., 0.));
	let gc = gradientHash(i + vec2(0., 1.));
	let gd = gradientHash(i + vec2(1., 1.));

	let va = dot(ga, f);
	let vb = dot(gb, f - vec2(1., 0.));
	let vc = dot(gc, f - vec2(0., 1.));
	let vd = dot(gd, f - vec2(1., 1.));

	let k1 = vb - va;
	let k2 = vc - va;
	let k3 = va - vb - vc + vd;

	return ga + u.x * (gb - ga) + u.y * (gc - ga) + u.x * u.y * (ga - gb - gc + gd) + du * (u.yx * k3 + vec2(k1, k2));
}

@fragment
fn fs_main(input : VertexOutput) -> @location(0) vec4<f32> {
	let wavelength = uniforms.wavelength;
	let tile = TILE_CELLS * wavelength;
	//world x, z inside one tile
	let p = input.uv * tile;

	var dirs = array<vec2<f32>, 3>(vec2(.80, .60), vec2(-.57, .82), vec2(.95, -.31));

	var g = vec2(0.);
	var frame = mat2x2<f32>(1., 0., 0., 1.);
	//the wavelength of this octave, for the dispersion relation, the frame applies the scale
	var lambda = wavelength;
	var amp = 1.;
	for (var i = 0; i < 3; i++) {
		let speed = sqrt(GRAVITY * lambda / TAU) * uniforms.speed;
		//a translation by a whole tile changes nothing, so the drift is wrapped to keep the precision as time grows
		let drift = modFloor(dirs[i] * (speed * uniforms.time), tile);
		let n = noiseGradient(frame * (p - drift) / wavelength);
		//the chain rule back out of the frame: the transpose of the frame times the gradient. Divided by the base
		//wavelength, the frame carries the scale of the octave
		g += (n * frame) * (amp / wavelength);

		frame = OCTAVE * frame;
		lambda /= sqrt(5.);
		amp *= .62;
	}
	//no fade of the octaves that get smaller than a pixel, the mips average them away
	return vec4(g * (.7 * uniforms.rippleStrength), 0., 1.);
}
