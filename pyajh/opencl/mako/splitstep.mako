<%!
	import math, operator as op
%>

<%
	nx, ny = grid
	gfc = '__global float2 * const'

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
	${'(' + ', '.join(str(pv) + 'f' for pv in p) + ')'}
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
	float sv = sin(${math.pi}f * (float) t / ${2. * l - 1.}f);
	return sv * sv;
}

/* Compute i * r * c for real r and complex c. */
float2 imulr(const float2 r, const float2 c) {
	return (float2) (r) * (float2) (-c.y, c.x);
}

/* Compute the square root of a complex value v. */
float2 csqrt(const float2);
float2 csqrt(const float2 v) {
	const float ang = 0.5f * atan2(v.y, v.x);
	const float mag = sqrt(sqrt(mad(v.y, v.y, v.x * v.x)));

	return (float2) (mag) * (float2) (cos(ang), sin(ang));
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
	return (float2) (exp(v.x)) * (float2) (cos(v.y), sin(v.y));
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

/* Convert, in place, an object contrast into an index of refraction. The
 * global work grid should be (nx, ny). */
__kernel void obj2eta(${gfc} obj) {
	${getindices('i', 'j', 'idx')}

	const float2 eval = csqrt(obj[idx] + (float2) (1.0f, 0.0f));
	obj[idx] = eval;
}

/* Compute, in place, the average index of refraction. On input, eta stores the
 * index of refraction for the previous slab; aug stores the index of
 * refraction for the next slab. On output, eta stores the average index of
 * refraction for the two slabs. */
__kernel void avgeta(${gfc} eta, ${gfc} aug) {
	${getindices('i', 'j', 'idx')}

	/* Compute the next half contribution to the average. */
	const float2 nval = aug[idx];
	const float2 oval = eta[idx];
	/* Update the average index of refraction. */
	eta[idx] = (float2) (0.5f) * (nval + oval);
}

/* Compute the reflection coefficients for the current interface. */
__kernel void rcoeff(${gfc} rc, ${gfc} cur, ${gfc} next) {
	${getindices('i', 'j', 'idx')}

	const float2 nval = next[idx];
	const float2 cval = cur[idx];

	rc[idx] = cdiv(cval - nval, nval + cval);
}

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
__kernel void hospat(${gfc} u, ${gfc} eta, ${gfc} f) {
	/* Grab the spatial sample  for the current work item. */
	${getindices('i', 'j', 'idx')}

	const float2 eval = eta[idx];
	const float2 one = (float2) (1.0f, 0.0f);
	const float2 qval = cdiv(one, eval + one) - (float2) (0.5f, 0.0f) +
				(float2) (0.125f) * (cmul(eval, eval) - one);

	u[idx] = cmul(qval, f[idx]);
}

/* In u, store the product of the object contrast with the field f. */
__kernel void ctmul(${gfc} u, ${gfc} eta, ${gfc} f) {
	/* Grab the spatial sample for the current work item. */
	${getindices('i', 'j', 'idx')}

	const float2 eval = eta[idx];
	const float2 qval = cmul(eval,eval) - (float2) (1.0f, 0.0f);

	u[idx] = cmul(f[idx], qval);
}

/* To the field, apply the phase screen resulting from medium variations. */
__kernel void screen(${gfc} fld, ${gfc} eta, const float dz) {
	${getindices('i', 'j', 'idx')};

	const float2 eval = eta[idx] - (float2) (1.0f, 0.0f);

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

/* Given field corrections u and v, compute
 * f = f + delta * (u + v) for delta = 1j * k0 * dz. */
__kernel void corrfld(${gfc} f, ${gfc} u, ${gfc} v, const float dz) {
	${getindices('i', 'j', 'idx')}

	const float2 upv = u[idx] + v[idx];
	f[idx] += imulr(${k0}f * dz, upv);
}

/* Compute the value of the Green's function at a slab with height zoff. */
__kernel void green3d(${gfc} fld, const float zoff) {
	${getindices('i', 'j', 'idx')}

	/* Compute the position of the observer. */
	const float3 obs = (float3) (${crd(nx, 'i')}, ${crd(ny, 'j')}, zoff);
	/* Compute the vector separation between source and observer. */
	const float3 rv = obs - (float3) ${prtuple(src)};
	/* Compute the scalar distance between source and observer. */
	const float r = length(rv);
	const float kr = ${k0}f * r;

	/* Compute the value of the Green's function. */
	const float2 grf = (float2) (cos(kr), sin(kr)) / (float2) (${4. * math.pi}f * r);

	% if d:
		const float ctheta = dot(rv / (float3) r, (float3) ${prtuple(dirax)});
		const float stheta = sin(acos(clamp(ctheta, -1.0f, 1.0f)));
		const float mag = ctheta * exp(${-dirmag}f * stheta * stheta);
	% endif

	fld[idx] = ${'' if d is None else '(float2) mag *'} grf;
}

/* Transmit a field through an interface with reflection coefficients rc. */
__kernel void transmit(${gfc} fwd, ${gfc} rc) {
	${getindices('i', 'j', 'idx')}

	const float2 fval = fwd[idx];
	const float2 rval = rc[idx];
	fwd[idx] = fval + cmul(rval, fval);
}

/* Transmit a field through an interface with reflection coefficients rc.
 * Also reflect the backward-traveling wave bck and add to the field. */
__kernel void txreflect(${gfc} fwd, ${gfc} bck, ${gfc} rc) {
	${getindices('i', 'j', 'idx')}

	const float2 fval = fwd[idx];
	const float2 bval = bck[idx];
	const float2 rval = rc[idx];
	const float2 cor = cmul(cdiv(rval, (float2) (1.0f, 0.0f) - rval), bval);
	fwd[idx] = fval + cmul(rval, fval) - cor;
}
