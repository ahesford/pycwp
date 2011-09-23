<%
	# Determine the strides of the 3-D grids in FORTRAN order.
	stride = [1, dim[0], dim[0] * dim[1]]
%>

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
	/* The target cell of the work item. Avoids boundary layers. */
	const uint i = get_global_id(0) + 1;
	const uint j = get_global_id(1) + 1;

	/* The local compute position. */
	const uint li = get_local_id(0) + 1;
	const uint lj = get_local_id(1) + 1;

	/* The local compute grid. */
	uint2 ldim = (uint2) (get_local_size(0), get_local_size(1));

	/* The leading dimension of the local work tile. */
	const uint ldw = ldim.x + 2;

	/* The local target cell of the work item. Avoids boundary layers. */
	const uint ltgt = lj * ldw + li;

	float rv, value, zr, zl, xr, xl, yr, yl, cur, prev;
	uint idx = 0;
	bool updsrc, inbounds;

	/* Truncate the local cache dimension to avoid input overruns. */
	ldim = min(ldim, max((uint2) (0),
		(uint2) (${dim[0] - 1}, ${dim[1] - 1}) - (uint2) (i, j)));

	/* Check if the target cell contains the source in some slab. */
	updsrc = (i == ${srcidx[0]}) && (j == ${srcidx[1]});

	/* Check if the target cell is within the boundaries. */
	inbounds = (i < ${dim[0] - 1}) && (j < ${dim[1] - 1});

	/* The current slab is the first one. */
	idx = i * ${stride[0]} + j * ${stride[1]};

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
	% for i, (li, ls, st) in enumerate(zip(['li', 'lj'], ['1', 'ldw'], stride[:2])):
		if (${li} == 1) {
			tile[ltgt - ${ls}] = pc[idx - ${st}];
			tile[ltgt + ldim.s${i} * ${ls}] = pc[idx + ldim.s${i} * ${st}];
		}
	% endfor

	/* Ensure buffer is filled. */
	barrier (CLK_LOCAL_MEM_FENCE);

	/* Perform the temporal update. */
	value = (2. - 6. * rv) * cur - prev;

	/* Grab the in-plane shifted cells. */
	% for ax, step in zip(['x', 'y'], ['1', 'ldw']):
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
