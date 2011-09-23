<%
	# Determine the strides of the 3-D grids in FORTRAN order.
	stride = [1, dim[0], dim[0] * dim[1]]

	# The indices of the computational grid
	indices = ['x', 'y']

	# Convenient Python string arrays referring to corresponding kernel variables
	gi = ['gi.' + s for s in indices]
	li = ['li.' + s for s in indices]
	ldim = ['ldim.' + s for s in indices]
	ldw = ['ldw.' + s for s in indices]
%>

<%def name="funcVec(tp, dim, func)">
	(${tp}${dim}) (${','.join(func + '(%d)' % d for d in range(dim))})
</%def>

<%def name="vec(tp, items)">
	(${tp}) (${','.join(str(i) for i in items)})
</%def>

<%def name="zipreduce(outop, inop, left, right)">
	${outop.join('(%s %s %s)' % (str(l), inop, str(r)) for l, r in zip(left, right))}
</%def>

/* Helmholtz OpenCL kernel for a single time step. Arguments:
 * pn, in/out, the 3-D grid representing the next and previous time steps.
 * pc, input, the 3-D grid representing the current time step.
 * rsq, input, the global 3-D grid of r-squared parameters.
 * srcval, input, the source value multiplied by the squared spatial step.
 * ltile, input, a local cache for the current time step in a work group.
 *
 * The pn, pc and rsq 3-D arrays should be stored in FORTRAN order.
 *
 * The global work grid should be 2-D and have as many work items in each
 * dimension as there are non-boundary elements in a constant-z slab. */
__kernel void helm(__global float * const pn, __global float * const pc,
			    __global float * const rsq, const float srcval,
			    __local float * const tile) {
	/* The target cell in local and global coordinates. Avoids boundaries. */
	const uint3 gi = ${funcVec("uint", 3, "get_global_id")} + (uint3) (1);
	const uint3 li = ${funcVec("uint", 3, "get_local_id")} + (uint3) (1);

	/* The local compute grid. */
	uint3 ldim = ${funcVec("uint", 3, "get_local_size")};

	/* The strides of the local work tile. */
	const uint3 ldw = (uint3) (1, ldim.x + 2, (ldim.x + 2) * (ldim.y + 2));

	/* The local target cell of the work item. Avoids boundary layers. */
	const uint ltgt = ${zipreduce('+', '*', li, ldw)};

	float rv, value, zr, zl, xr, xl, yr, yl, cur, prev;

	/* Truncate the local cache dimension to avoid input overruns. */
	ldim = min(ldim, max((uint3) (0), ${vec("uint3", [d - 1 for d in dim])} - gi));

	/* Check if the target cell contains the source in some slab. */
	const bool updsrc = ${zipreduce('&&', '==', gi, srcidx)};

	/* Check if the target cell is within the boundaries. */
	const bool inbounds = ${zipreduce('&&', '<', gi, [d - 1 for d in dim])};

	/* The current slab is the first one. */
	uint idx = ${zipreduce('+', '*', gi, stride)};

	cur = pc[idx];
	zr = pc[idx + ${stride[2]}];

% for k in range(2, dim[2]):
	/* Cycle to the next slab. */
	idx += ${stride[2]};
	zl = cur;
	cur = zr;

	/* Only read the values if they are in bounds. */
	zr = pc[idx + ${stride[2]}];
	prev = pn[idx];
	rv = rsq[idx];

	barrier (CLK_LOCAL_MEM_FENCE);

	/* Add the value to the buffer tile. */
	tile[ltgt] = cur;

	/* Also grab needed boundary values. */
	% for (lidx, llen, lstr, str) in zip(li, ldim, ldw, stride):
		if (${lidx} == 1) {
			tile[ltgt - ${lstr}] = pc[idx - ${str}];
			tile[ltgt + ${llen} * ${lstr}] = pc[idx + ${llen} * ${str}];
		}
	% endfor

	/* Ensure buffer is filled. */
	barrier (CLK_LOCAL_MEM_FENCE);

	/* Perform the temporal update. */
	value = (2. - 6. * rv) * cur - prev;

	/* Grab the in-plane shifted cells. */
	% for ax, step in zip(indices, ldw):
		${ax}l = tile[ltgt - ${step}];
		${ax}r = tile[ltgt + ${step}];
	% endfor

	/* Perfrom the spatial updates. */
	value += rv * (zr + zl + xr + xl + yr + yl);

	% if k - 1 == srcidx[2]:
		/* Update the source location. */
		if (updsrc) value += srcval * rv;
	% endif

	/* Only write the output for cells inside the boundaries. */
	if (inbounds) pn[idx] = value;
% endfor
}

<%
	# Local grid and stride arrays describe the quicker-varying index
	# first, the slower-varying dimension second, and the constant-index
	# dimension last
	bdygrids, bdystrides = [[[a[i] for i in range(len(a)) if i != j] + [a[j]]
				for j in range(len(a))] for a in [dim, stride]]
	bdynames = [d + 'bdy' for d in ['x', 'y', 'z']]
%>

% for n, g, s in zip(bdynames, bdygrids, bdystrides):
	__kernel void ${n} (__global float * const pc,
		__global float * const left, __global float * const right) {
		/* Get the work-item position in the boundary slice. */
		const uint2 pos = (uint2) (get_global_id(0), get_global_id(1));

		/* Precompute the offset of the right slab. */
		const uint roff = ${(g[-1] - 1) * s[-1]};

		/* Store the in-plane linear index of the cell. */
		uint ipidx;

		/* Only work in the boundaries of the slab. */
		if (pos.s0 < ${g[0]} && pos.s1 < ${g[1]}) {
			/* Build the index into the boundary slab. */
			ipidx = pos.s1 * ${g[0]} + pos.s0;
			/* Grab the left and right boundary values. */
			const float2 vals = (float2) (left[ipidx], right[ipidx]);
			/* Now compute the base index in the global grid. */
			ipidx = pos.s1 * ${s[1]} + pos.s0 * ${s[0]};
			pc[ipidx] = vals.s0;
			pc[ipidx + roff] = vals.s1;
		}
	}
% endfor
