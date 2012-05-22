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

#define cmul(a, b) (float2)(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
#define cdiv(a, b) (float2)(mad((a).x, (b).x, (a).y * (b).y), mad((a).y, (b).x, -(a).x * (b).y)) / (float2)((b).x * (b).x + (b).y * (b).y)
#define eikr(a) (float2) (cos((float) (a)), sin((float) (a)))
#define imul(a) (float2) (-(a).y, (a).x)

## Wrap the coordinate index in accordance with the FFT.
<%def name="wrap(n,i)">
	(float) ((${i} < ${n / 2}) ? (int) ${i} : (int) ${i} - ${n})
</%def>

## Compute the coordinates for grid indices
<%def name="crd(n,i)">
	(${h} * (${i} - ${0.5 * (n - 1.)}))
</%def>

## Compute the spatial frequencies for unwrapped indices
<%def name="k(n,i)">
	(${2. * math.pi / n / h} * ${wrap(n, i)})
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

/* Compute the square root of a complex value v. */
float2 csqrt(const float2);
float2 csqrt(const float2 v) {
	const float ang = 0.5 * atan2(v.y, v.x);
	const float mag = sqrt(sqrt(mad(v.y, v.y, v.x * v.x)));

	return (float2) (mag) * (float2) (cos(ang), sin(ang));
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

	const float2 eval = csqrt(obj[idx] + (float2) (1., 0.));
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
	eta[idx] = (float2) (0.5) * (nval + oval);
}

/* Compute the ratio of the current to the next average indices of refraction. */
__kernel void etafrac(${gfc} efrac, ${gfc} cur, ${gfc} next) {
	${getindices('i', 'j', 'idx')}

	const float2 nval = next[idx];
	const float2 cval = cur[idx];

	efrac[idx] = cdiv(cval, nval);
}

/* Apply the homogeneous propagator to the field in the spectral domain. */
__kernel void propagate(${gfc} fld) {
	/* Grab the spectral sample to be propagated. */
	${getindices('i', 'j', 'idx')}

	/* Compute the transverse and axial wave numbers. */
	${getkxy('i', 'j')}
	const float2 kz = csqrt((float2) (${k0**2} - kxy, 0.));
	/* Compute the propagator, exp(i * kz * dz). */
	const float2 prop = cexp(imul(kz) * (float2) (${dz}));

	fld[idx] = cmul(fld[idx], prop);
}

/* Compute the Laplacian of the field in the spectral domain and divide by the
 * square of the wave number. */
__kernel void laplacian(${gfc} lap, ${gfc} fld) {
	/* Grab the spectral sample for the current work item. */
	${getindices('i', 'j', 'idx')}

	/* Compute the transverse wave numbers. */
	${getkxy('i', 'j')}

	/* Compute the scaled Laplacian in the spectral domain. */
	lap[idx] = (float2) (kxy / ${k0**2}) * fld[idx];
}

/* Add the wide-angle correction term to the field. */
__kernel void wideangle(${gfc} fld, ${gfc} lap, ${gfc} eta) {
	/* Grab the spatial sample for the current work item. */
	${getindices('i', 'j', 'idx')}

	const float2 eval = eta[idx];
	const float2 etafrac = cdiv(eval - (float2) (1., 0.), (float2) (2.) * eval);

	/* Compute i * k0 * dz * etafrac. */
	const float2 cor = (float2) (${k0 * dz}) * imul(etafrac);
	/* Multiply the correction by the Laplacian and add to the field. */
	fld[idx] += cmul(cor, lap[idx]);
}

/* To the field, apply the phase screen resulting from medium variations. */
__kernel void screen(${gfc} fld, ${gfc} eta) {
	${getindices('i', 'j', 'idx')};

	const float2 eval = eta[idx] - (float2) (1., 0.);

	/* Compute i * k0 * dz * (eta - 1). */
	const float2 arg = (float2) (${k0 * dz}) * imul(eval);
	/* Compute phase = exp(arg). */
	const float2 ampl = (float2) (exp(arg.x));
	const float2 phase = ampl * (float2) (cos(arg.y), sin(arg.y));

	fld[idx] = cmul(phase, fld[idx]);
}

/* Compute the value of the Green's function at a slab with height zoff. */
__kernel void green3d(${gfc} fld, const float zoff) {
	${getindices('i', 'j', 'idx')}

	/* Compute the position of the observer. */
	const float3 obs = (float3) (${crd(nx, 'i')}, ${crd(ny, 'j')}, zoff);
	/* Compute the vector separation between source and observer. */
	const float3 rv = obs - (float3) ${tuple(src)};
	/* Compute the scalar distance between source and observer. */
	const float r = length(rv);

	/* Compute the value of the Green's function. */
	const float2 grf = eikr(${k0} * r) / (float2) (${4. * math.pi} * r);

	% if d:
		const float ctheta = dot(rv / (float3) r, (float3) ${tuple(dirax)});
		const float stheta = sin(acos(clamp(ctheta, -1.0f, 1.0f)));
		const float mag = ctheta * exp((float) ${-dirmag} * stheta * stheta);
	% endif

	fld[idx] = ${'' if d is None else '(float2) mag *'} grf;
}

/* Update the guess of the field in a slab based on the propagation of the
 * preceding field and the back-propagated field using a relaxation method with
 * parameter tau = 2 (no memory of past guesses). */
__kernel void update(${gfc} fwd, ${gfc} back, ${gfc} efrac) {
	${getindices('i', 'j', 'idx')}

	const float2 fval = fwd[idx];
	const float2 bval = back[idx];

	const float2 eval = efrac[idx];
	const float2 one = (float2) (1., 0.);

	const float2 ep1 = one + eval;
	const float2 em1 = one - eval;

	const float2 nval = (float2) (0.5) * (cmul(ep1, fval) + cmul(em1, bval));
	fwd[idx] = nval;
}
