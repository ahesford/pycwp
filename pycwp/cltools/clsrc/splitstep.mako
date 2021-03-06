## Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
## Restrictions are listed in the LICENSE file distributed with this package.

<%!
	import math, operator as op
%>

<%
	nx, ny = grid
	gfc = '__global float2 * const'
	cfl = 'const float'

	if d:
		dn = reduce(op.add, [dv**2 for dv in d[:-1]])
		dirax = [dv / dn for dv in d[:-1]]
		dirmag = d[-1]
%>

## Wrap the coordinate index in accordance with the FFT.
<%def name="wrap(n,i)">
	(float) ((${i} <= ${int(n - 1) / 2}) ? (int) ${i} : (int) ${i} - ${n})
</%def>

## Print the tuple as a floating-point number
<%def name="prtuple(p)">
	${'(' + ', '.join('%0.8f' % pv for pv in p) + ')'}
</%def>

## Compute the coordinates for grid indices
<%def name="crd(n,i)">
	(${h}f * (${i} - ${0.5 * (n - 1.)}f))
</%def>

## Compute the spatial frequencies for unwrapped indices
<%def name="k(n,i)">
	(${2. * math.pi / n / h}f * ${wrap(n, i)})
</%def>

## Roll a two-dimensional grid index into a linear memory index
<%def name="index(i,j)">
	(${i} + ${j} * ${nx})
</%def>

## Grab the 2-D and 1-D grid indices for the current work item
<%def name="getindices(i,j,idx)">
	const uint ${i} = get_global_id(0);
	const uint ${j} = get_global_id(1);
	const uint ${idx} = ${index(i, j)};
</%def>

## Grab the transverse component wave numbers and the combined square
<%def name="getkxy(i, j)">
	/* Compute the transverse wave numbers. */
	const float kx = ${k(nx, i)};
	const float ky = ${k(ny, j)};
	/* Compute the square of the combined transverse wave number. */
	const float kxy = mad(kx, kx, ky * ky);
</%def>

/* Compute the value for a Hann window of width l at point t. */
float hann(const uint);
float hann(const uint t) {
	/* sinpi(x) computes sin(pi * x). */
	float sv = sinpi((float) t / ${2. * l - 1.}f);
	return sv * sv;
}

/* Compute i * r * c for real r and complex c. */
float2 imulr(const float, const float2);
float2 imulr(const float r, const float2 c) {
	return (float2) (r) * (float2) (-c.y, c.x);
}

/* Compute the square root of a complex value v. */
float2 csqrt(const float2);
float2 csqrt(const float2 v) {
	const float ang = 0.5f * atan2(v.y, v.x);
	const float mag = sqrt(hypot(v.x, v.y));
	/* Return the sine of the angle and store the cosine in cosa. */
	float cosa;
	const float sina = sincos(ang, &cosa);

	return (float2) (mag) * (float2) (cosa, sina);
}

/* Compute the complex square root of a real value v. */
float2 csqrtr(const float);
float2 csqrtr(const float v) {
	return (v < 0) ? (float2) (0.0f, sqrt(-v)) : (float2) (sqrt(v), 0.0f);
}

/* Compute the complex division a / b. */
float2 cdiv(const float2, const float2);
float2 cdiv(const float2 a, const float2 b) {
	float bn = mad(b.x, b.x, b.y * b.y);
	return (float2) (mad(a.x, b.x, a.y * b.y) / bn, mad(a.y, b.x, -a.x * b.y) / bn);
}

/* Compute the complex multiplication a * b. */
float2 cmul(const float2, const float2);
float2 cmul(const float2 a, const float2 b) {
	return (float2) (mad(a.x, b.x, -a.y * b.y), mad(a.y, b.x, a.x * b.y));
}

/* Compute the exponential of a complex value v. */
float2 cexp(const float2);
float2 cexp(const float2 v) {
	/* Return the sine of the angle and store the cosine in cosa. */
	float cosa;
	const float sina = sincos(v.y, &cosa);
	return (float2) (exp(v.x)) * (float2) (cosa, sina);
}

/* Compute the reflection coefficient for the interface between the current and
 * next indices of refraction. */
float2 rcoeff(const float2, const float2);
float2 rcoeff(const float2 cur, const float2 next) {
	const float2 one = (float2) (1.0f, 0.0f);
	const float2 ec = csqrt(cur + one);
	const float2 en = csqrt(next + one);
	return cdiv(ec - en, ec + en);
}

<%
	# The axes that will be attenuated
	axes = ['x', 'y']
	# The offset x indices for right-edge attenuation
	xoff = ['(-x - 1)', 'x']
	# The offset y indices for right-edge attenuation
	yoff = ['(y + 1)', '(%d - y - 1)' % ny]
%>

/* Apply Hann windowing of width l to the left and right x or y edges of the
 * domain. For attenuation along the x edges, the global work grid should have
 * dimensions (l, ny); for attenuation along the y edges, the global work grid
 * should be (nx, l). */
% for c, xo, yo in zip(axes, xoff, yoff):
	__kernel void atten${c}(${gfc} fld) {
		/* Grab the position in the work array.
		 * This includes the left-edge linear index. */
		${getindices('x', 'y', 'lidx')}
		/* Compute the offset for the right edge. */
		const uint ridx = ${index(xo, yo)};

		/* Grab the value of the Hann multiplier. */
		const float2 hv = (float2) (hann(${c}));

		fld[lidx] *= hv;
		fld[ridx] *= hv;
	}
% endfor

## All subsequent kernels should use a global work grid size of (nx, ny) unless
## otherwise noted. The local work grid size is not significant.

/* Apply the homogeneous propagator to the field in the spectral domain. */
__kernel void propagate(${gfc} fld, const float dz) {
	/* Grab the spectral sample to be propagated. */
	${getindices('i', 'j', 'idx')}

	/* Compute the transverse and axial wave numbers. */
	${getkxy('i', 'j')}
	const float2 kz = csqrtr(${k0**2}f - kxy);
	/* Compute the propagator, exp(i * kz * dz). */
	const float2 prop = cexp(imulr(dz, kz));

	fld[idx] = cmul(fld[idx], prop);
}

/* In u, apply JPA's high-order spectral operator PP to the input field f. */
__kernel void hospec(${gfc} u, ${gfc} f) {
	/* Grab the spectral sample for the current work item. */
	${getindices('i', 'j', 'idx')}
	/* Compute the transverse wave numbers. */
	${getkxy('i', 'j')}

	/* Normalize the transverse wave number. */
	const float2 kzn = (float2) (${k0**2}f - 0.5f * kxy, 0.0f)
				- csqrtr(${k0**4}f - ${k0**2}f * kxy);
	/* Avoid dividing by zero spatial frequency.
	 * The numerator vanishes faster anyway. */
	const float kzd = ((i != 0 || j != 0) ? kxy : 1.0f);

	/* Multiply the operator by the field. */
	u[idx] = cmul(kzn / (float2) kzd, f[idx]);
}

/* In u, apply the spectral Laplacian operator P to the input field f. */
__kernel void laplacian(${gfc} u, ${gfc} f) {
	/* Grab the spectral sample for the current work item. */
	${getindices('i', 'j', 'idx')}
	/* Compute the transverse wave numbers. */
	${getkxy('i', 'j')}

	u[idx] = (float2) (-kxy / ${k0**2}f) * f[idx];
}

/* In u, apply the high-order spatial operator QQ to the input field f. */
__kernel void hospat(${gfc} u, ${gfc} obj, ${gfc} f) {
	/* Grab the spatial sample  for the current work item. */
	${getindices('i', 'j', 'idx')}

	/* Grab the contrast value for the work item. */
	const float2 oval = obj[idx];
	const float2 one = (float2) (1.0f, 0.0f);
	const float2 eval = cdiv(one, csqrt(oval + one) + one);
	/* Compute 1 / (eta + 1) + 0.5 + obj / 8. */
	const float2 qval = eval - (float2) (0.5f, 0.0f) + (float2) (0.125f) * oval;

	u[idx] = cmul(qval, f[idx]);
}

/* In u, store the product of the object contrast with the field f. */
__kernel void ctmul(${gfc} u, ${gfc} obj, ${gfc} f) {
	/* Grab the spatial sample for the current work item. */
	${getindices('i', 'j', 'idx')}

	const float2 qval = obj[idx];

	u[idx] = cmul(f[idx], qval);
}

/* To the field, apply the phase screen resulting from medium variations. */
__kernel void screen(${gfc} fld, ${gfc} obj, const float dz) {
	${getindices('i', 'j', 'idx')};

	const float2 oval = obj[idx];
	const float2 one = (float2) (1.0f, 0.0f);
	const float2 eval = csqrt(oval + one) - one;

	/* Compute i * k0 * dz * (eta - 1). */
	const float2 arg = imulr(${k0}f * dz, eval);
	/* Multiply the phase shift by the field. */
	fld[idx] = cmul(cexp(arg), fld[idx]);
}

/* Compute z = a * x + y for vectors x, y, z and real scalar a. */
__kernel void caxpy(${gfc} z, const float a, ${gfc} x, ${gfc} y) {
	${getindices('i', 'j', 'idx')}
	z[idx] = y[idx] + (float2) (a) * x[idx];
}

/* Given field corrections u, v and x, compute
 * f = f + delta * (u + v + x) for delta = 1j * k0 * dz.
 * Each of the arguments u, v and x will be ignored if NULL. */
__kernel void corrfld(${gfc} f, ${gfc} u, ${gfc} v, ${gfc} x, const float dz) {
	${getindices('i', 'j', 'idx')}

	const float2 uval = (u == 0)? (float2) (0.0f) : u[idx];
	const float2 vval = (v == 0)? (float2) (0.0f) : v[idx];
	const float2 xval = (x == 0)? (float2) (0.0f) : x[idx];
	f[idx] += imulr(${k0}f * dz, uval + vval + xval);
}

/* Compute the value of the Green's function throughout a slab. The source has
 * transverse coordinates (sx, sy) and has a height dz above the slab. */
__kernel void green3d(${gfc} fld, ${cfl} sx, ${cfl} sy, ${cfl} dz) {
	${getindices('i', 'j', 'idx')}

	/* Compute the position of the observer, assumed to be z = 0. */
	const float3 obs = (float3) (${crd(nx, 'i')}, ${crd(ny, 'j')}, 0.0f);
	/* Compute the vector separation between source and observer. */
	const float3 rv = obs - (float3) (sx, sy, dz);
	/* Compute the scalar distance between source and observer. */
	const float r = length(rv);
	const float kr = ${k0}f * r;

	/* Return the sine of the argument and store in ckr the cosine. */
	float ckr;
	const float skr = sincos(kr, &ckr);
	/* Compute the value of the Green's function. */
	const float2 grf = (float2) (ckr, skr) / (float2) (${4. * math.pi}f * r);

	% if d:
		const float ctheta = dot(rv / (float3) r, (float3) ${prtuple(dirax)});
		const float sthsq = 1.0f - ctheta * ctheta;
		const float mag = ctheta * exp(${-dirmag}f * sthsq);
	% endif

	fld[idx] = ${'' if d is None else '(float2) mag *'} grf;
}

/* Transmit a field through an interface with reflection coefficients rc.
 * Also store the reflected field in bck (if the pointer is not NULL). */
__kernel void txreflect(${gfc} fwd, ${gfc} bck, ${gfc} ocur, ${gfc} onxt) {
	${getindices('i', 'j', 'idx')}

	const float2 cval = ocur[idx];
	const float2 nval = onxt[idx];
	const float2 fval = fwd[idx];
	const float2 rval = rcoeff(cval, nval);
	const float2 reflect = cmul(rval, fval);
	fwd[idx] = fval + reflect;
	if (bck) bck[idx] = reflect;
}

/* Perform a Goertzel iteration to effect a Fourier transform along the
 * propagation (z) direction, with k_z restricted to the unit sphere. The step
 * N inputs are the previous two iterations (N - 1 in pn1 and N - 2 in pn2) of
 * the Goertzel algorithm and the induced current source crt, already Fourier
 * transformed in x and y. On return, pn2 is replaced with the N-th iteration.
 *
 * When crt is NULL, a final iteration is performed. In this case, the values
 * in pn1 are replaced with the upper hemisphere, and the values of pn2 are
 * replaced with the lower hemisphere, of the Fourier transform of the induced
 * volume current, restricted to the unit sphere. Each plane has k_x and k_y
 * values according to the standard FFT ordering, with the value of k_z implied
 * such that
 *
 *      k0**2 = k_x**2 + k_y**2 + k_z**2.
 *
 * For evanescent values of k_z, the value of the Fourier transform is zero.
 *
 * The integer n specifies the number of points involved in the FFT and is used
 * only in the final iteration (when crt is NULL) to properly scale the
 * results. */
__kernel void goertzelfft(${gfc} pn1, ${gfc} pn2, ${gfc} crt, const float dz, int n) {
	/* Grab the recursion values for the work item. */
	${getindices('i', 'j', 'idx')}
	const float2 pn1v = pn1[idx];
	const float2 pn2v = pn2[idx];

	const float2 zero = (float2) (0.0f);

	/* The field might be NULL in the last step. */
	const float2 crtv = crt ? crt[idx] : zero;

	/* Compute the transverse and axial wave numbers. */
	${getkxy('i', 'j')}
	/* This is either pure real or pure imaginary. */
	const float2 kz = csqrtr(${k0**2}f - kxy);

	/* The weight w = exp(-i * kz * dz); recursion uses weight
	 * h = (w + w*) = 2 cos(kz * dz). Only real kz is of interest. */
	const float h = (${k0**2}f >= kxy) ? (2.0f * cos(kz.x * dz)) : 0.0f;

	/* Compute the next value of the Goertzel iteration. */
	const float2 nv = (${k0**2}f >= kxy) ? (h * pn1v - pn2v + crtv) : zero;

	if (crt) {
		/* Overwrite the earliest step with the next iteration. */
		pn2[idx] = nv;
	} else if (${k0**2}f >= kxy) {
		const float2 ikz = imulr(dz, kz);
		const float n2 = (float) (n / 2);
		/* Compute final non-evanescent Fourier transform values. */
		const float2 w = cexp(ikz);
		const float2 wn = cexp((float2) (n2) * ikz);
		const float2 pfft = nv - cmul(w, pn1v);
		const float2 nfft = nv - cmul((float2)(w.x, -w.y), pn1v);
		/* Scale the FFT values appropriately. */
		pn1[idx] = cmul(wn, pfft);
		pn2[idx] = cmul((float2)(wn.x, -wn.y), nfft);
	} else {
		/* For evanescent waves, just fill with zeros. */
		pn1[idx] = zero;
		pn2[idx] = zero;
	}
}
