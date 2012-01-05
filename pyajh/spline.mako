<%!
	import math
%>

<%
	zp = math.sqrt(3) - 2.
	n, k = 2 * (ntheta - 1), nphi / 2
%>

float2 splineval(__read_only image2d_t, const float2, const float2);

/* Given an image representation of the spline coefficients in a polar-ring
 * arrangement, return the interpolated function value at the coordinate
 * specified with fraction distance frac between index idx and (idx + 1). The
 * first component of idx and frac represents the polar coordinate, while the
 * second respresents the azimuthal coordinate. */
float2 splineval(__read_only image2d_t img, const float2 idx, const float2 frac) {
	/* Define a linear interpolation sampler to read the image. */
	const sampler_t lin = CLK_NORMALIZED_COORDS_FALSE |
		CLK_FILTER_LINEAR | CLK_ADDRESS_NONE;

	/* Precompute the spline function values. */
	const float2 one_frac = (float2) (1.) - frac;
	const float2 one_frac2 = one_frac * one_frac;
	const float2 frac2 = frac * frac;

	const float2 w0 = (float2) (1. / 6.) * one_frac2 * one_frac;
	const float2 w1 = (float2) (2. / 3.) - 
		(float2) (0.5) * frac2 * ((float2) (2.) - frac);
	const float2 w2 = (float2) (2. / 3.) - 
		(float2) (0.5) * one_frac2 * ((float2) (1.) + frac);
	const float2 w3  = (float2) (1. / 6.) * frac2 * frac;

	/* Compute the linear weights. */
	const float4 g = (float4) (w0 + w1, w2 + w3);

	/* The fractional coordinates are increased by half an index work
	 * properly with OpenCL image addressing, and again by a full index to
	 * skip the repeated lower boundary. */
	const float4 h = (float4) ((w1 / g.lo) + (float2) (0.5) + idx, 
		(w3 / g.hi) + (float2) (2.5) + idx);

	/* Grab the four constituent linear interpolations. */
	const float4 tex00 = read_imagef(img, lin, h.lo);
	const float4 tex11 = read_imagef(img, lin, h.hi);
	const float4 tex01 = read_imagef(img, lin, h.xw);
	const float4 tex10 = read_imagef(img, lin, h.zy);

	/* Linearly mix along the y-direction. */
	const float2 ty0 = mix(tex01.lo, tex00.lo, g.y);
	const float2 ty1 = mix(tex11.lo, tex10.lo, g.y);

	/* Linearly mix along the x-direction. */
	return mix(ty1, ty0, g.x);
}

/* Transform a regularly-sampled grid of function values, with the polar angle
 * most rapidly varying, into cubic b-spline coefficients in the polar angle.
 * For function values defined on an ntheta-by-nphi grid, the array mat stores
 * values in a 2 * (ntheta - 1) by nphi / 2 grid representing a full 2-pi polar
 * circle for each azimuthal sample in [0, pi).
 *
 * The global work grid should be one dimensional, with a size equal to the
 * number (nphi / 2) of constant-azimuth polar rings. Thus, each work item
 * operates on a column of the matrix. */
__kernel void polcoeff(__global float2 * const mat) {
	/* Get the global index corresponding to a fixed azimuthal angle. */
	const uint idx = get_global_id(0);

	/* Point to the column of data for this work item. */
	__global float2 * const column = mat + idx * ${n};

	const float2 l = (float2) (6.);
	const float2 zp = (float2) (${zp});

	float2 val = (float2) (0.);

	/* Compute the sum contribution to the causal coefficient. One power of
	 * zp is missing, and the coefficient l is included so the recursion
	 * can be applied directly to yield the first value. */
	% for i in range(p):
		val += (float2) (${6. * zp**i}) * column[${n - i - 1}];
	% endfor

	/* Compute the causal coefficients, recycling the last value. */
	% for i in range(n):
		column[${i}] = val = l * column[${i}] + zp * val;
	% endfor

	/* Compute the negative of the anti-causal sum so the recurrence can be
	 * applied to establish the first output term. */
	val = (float2) (0.);
	% for i in range(p):
		val += - (float2) (${zp**(i + 1)}) * column[${i}];
	% endfor

	/* Compute the anti-causal coefficients. */
	% for i in reversed(range(n)):
		column[${i}] = val = zp * (val - column[${i}]);
	% endfor
}

/* Transform a regularly-sampled grid of polar b-spline coefficients, with the
 * polar angle most rapidly varying, into cubic b-spline coefficients in the
 * azimuthal angle. For coefficients defined on an ntheta-by-nphi grid, the
 * array mat stores values in a 2 * (ntheta - 1) by nphi / 2 grid representing
 * a full 2-pi polar circle for each azimuthal sample in [0, pi).
 *
 * The global work grid should be one dimensional, with a size equal to the
 * number of polar samples in a hemisphere (including poles). Thus, each work
 * item operates on two rows of the matrix that correspond to a constant
 * latitude in the sample grid. */
__kernel void azicoeff(__global float2 * const mat) {
	/* Get the global index corresponding to a fixed polar angle. */
	const uint idx = get_global_id(0);

	/* Point to the low-azimuth hemisphere of data. */
	__global float2 * const lower = mat + idx;
	/* Point to the high-azimuth hemisphere of data. */
	__global float2 * const upper = mat + (${n} - idx) % ${n};

	const float2 l = (float2) (6.);
	const float2 zp = (float2) (${zp});

	float2 val = (float2) (0.);

	/* Compute the sum contribution to the causal coefficient. */
	% for i in range(p):
		val += (float2) (${6. * zp**i}) * upper[${(k - i - 1) * n}];
	% endfor

	/* Compute the low-azimuth causal coefficients. */
	% for i in range(k):
		lower[${i * n}] = val = l * lower[${i * n}] + zp * val;
	% endfor

	/* Update the high-azimuth causal coefficients away from the poles. */
	if (idx > 0 && idx < ${ntheta - 1}) {
		% for i in range(k):
			upper[${i * n}] = val = l * upper[${i * n}] + zp * val;
		% endfor
	}

	/* Compute the negative of the anti-causal sum. */
	val = (float2) (0.);
	% for i in range(p):
		val += - (float2) (${zp**(i + 1)}) * lower[${i * n}];
	% endfor

	/* Compute the remaining high-azimuth anti-causal coefficients. */
	% for i in reversed(range(k)):
		upper[${i * n}] = val = zp * (val - upper[${i * n}]);
	% endfor

	/* Update the low-azimuth anti-causal coefficients away from the poles. */
	if (idx > 0 && idx < ${ntheta - 1}) {
		% for i in reversed(range(k)):
			lower[${i * n}] = val = zp * (val - lower[${i * n}]);
		% endfor
	}
}

/* Convert the polar-ring representation of a function or b-spline
 * coefficients, stored in a buffer, to an OpenCL float image of format CL_RG.
 *
 * The azimuthal boundary values must be manually repeated because of the
 * complicated mirroring that occurs. However, the polar boundary values (rows)
 * should automatically be repeated with CLK_ADDRESS_REPEAT. Thus, the image
 * has three extra columns (one to the left of the samples and two to the
 * right) to account for the repetition but only as many rows as there are
 * polar samples.
 *
 * The work grid should be two dimensional, with the first index representing
 * the total number of rows in the matrix (2 * (ntheta - 1)) and the second
 * index representing the total number of columns in the matrix (nphi / 2). */
__kernel void mat2img(__write_only image2d_t img, __global float2 * const mat) {
	/* Figure out which pixel to duplicate. */
	const uint2 idx = { get_global_id(0), get_global_id(1) };
	const uint inidx = idx.s1 * ${n} + idx.s0;

	/* The output index has to be shifted for boundary repeats. */
	const int2 outidx = (int2) (idx.x, idx.y) + (int2) (1);

	/* Represent the indexed value as a float4 pixel. */
	const float4 pval = (float4) (mat[inidx], 0., 1.);

	/* Store the pixel value in the shifted location. */
	write_imagef(img, outidx, pval);

	/* Wrap the top and bottom boundaries. */
	if (idx.s0 == ${n - 1}) write_imagef(img, (int2) (0, outidx.y), pval);
	if (idx.s0 < 2) write_imagef(img, (int2) (${n + 1} + idx.s0, outidx.y), pval);

	/* This is the wrapped index for cloning the left and right boundaries. */
	uint y = (${n} - idx.s0) % ${n} + 1;

	/* Copy the right boundary values to the left side. */
	if (idx.s1 == ${k - 1}) {
		const int col = 0;
		write_imagef(img, (int2) (y, col), pval);
		/* Duplicate the corner values. */
		if (idx.s0 < 2) {
			const int row = (${n + 1} + idx.s0) % ${n + 2};
			write_imagef(img, (int2) (row, col), pval);
		}
		if (idx.s0 == ${n - 1})
			write_imagef(img, (int2) (${n + 2}, col), pval);
	}
	/* Copy the left boundary values to the right side. */
	if (idx.s1 < 2) {
		const int col = ${k + 1} + idx.s1;
		write_imagef(img, (int2) (y, col), pval);
		/* Duplicate the corner values. */
		if (idx.s0 < 2) {
			const int row = (${n + 1} + idx.s0) % ${n + 2};
			write_imagef(img, (int2) (row, col), pval);
		}
		if (idx.s0 == ${n - 1})
			write_imagef(img, (int2) (${n + 2}, col), pval);
	}
}

/* Given a polar-ring image of b-spline coefficients for a harmonic function
 * defined on a sphere, produce an ordinary representation (0 < theta < pi, 0
 * <= phi < 2 * pi) on an interpolated grid that avoids the poles.
 *
 * The azimuthal angle is most rapidly varying in the output.
 *
 * The first workgroup dimension is the number of polar samples away from the
 * poles; the second dimension is the number of azimuthal samples. */
__kernel void radinterp(__global float2 * const mat, __read_only image2d_t img) {
	/* Figure the output grid size and the place in the fine grid. */
	const uint2 widx = { get_global_id(0), get_global_id(1) };
	const uint2 size = { get_global_size(0), get_global_size(1) };

	/* Figure out the linearized work index. Azimuth varies most rapidly! */
	const uint outidx = widx.y + size.y * widx.x;

	/* The scale that converts fine coordinates to coarse coordinates. */
	const float2 scale = (float2) (${ntheta - 1}, ${nphi}) / 
		(float2) (size.x + 1, size.y);

	/* Represent the coordinates on the coarse grid. */
	const float2 c = scale * (float2) (widx.x + 1, widx.y);

	/* Compute the fractional coordinates and spline weights. */
	float2 idx, frac;
	/* Note that modf only works if coordinates are positive.
	 * Otherwise, the floor function must be used. */
	frac = modf(c, &idx);

	/* Wrap the coordinates in the right hemisphere. */
	if (c.y >= ${k}) {
		idx = (float2) (${n - 1} - idx.x, idx.y - ${k});
		frac.x = 1.0 - frac.x;
	}

	mat[outidx] = splineval(img, idx, frac);
}
