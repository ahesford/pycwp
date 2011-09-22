import numpy as np, pyopencl as cl, pyopencl.array as cla
from mako.template import Template
from . import fdtd

helmsrc = '''
<%
	# Determine the strides of the 3-D grids in FORTRAN order.
	stride = [1, dim[0], dim[0] * dim[1]]
%>

/* Helmholtz OpenCL kernel for a single time step. Arguments:
 * pn, in/out, the 3-D grid representing the next and previous time steps.
 * pc, input, the 3-D grid representing the current time step.
 * csq, input, the global 3-D grid of squared sound-speed values.
 * srcval, input, the value of the point source term.
 * ltile, input, a local cache for the current time step in a work group.
 *
 * The pn, pc and csq 3-D arrays should be stored in FORTRAN order.
 *
 * The global work grid should be 2-D and have as many work items in each
 * dimension as there are non-boundary elements in a constant-z slab. */
__kernel void helm(__global float * const pn, __global float * const pc,
			    __global float * const csq, const float srcval,
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

	/* The scaling for the squared sound speed. */
	const float cscale = ${(dt / dh)**2};

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
	rv = cscale * csq[idx];

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
		if (updsrc) value += srcval * rv * ${dh**2};
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
'''

class Helmholtz(fdtd.Helmholtz):
	'''
	A class that works identically to the standard Helmholtz class but that
	uses PyOpenCL to accelerate computations. A desired context can be
	passed in, but one will be created if none is provided.
	'''

	def __init__(self, c, dt, h, srcfunc, srcidx, context=None):
		'''
		Initialize the sound-speed c, time step dt and spatial step h.
		The coroutine srcfunc should provide a time-dependent value that
		describes the incident pressure at index srcidx. The context, if
		provided, is a PyOpenCL context for a single device. If it is
		not provided, a default context will be created.
		'''

		# Copy the finite-difference parameters
		self.dt, self.h = dt, h
		self.grid = c.shape[:]

		# Copy the source location and initialize the generator
		self.srcidx = srcidx[:]
		self.source = srcfunc()

		# Copy or create the PyOpenCL context
		if context is not None: self.context = context
		else: self.context = cl.Context(dev_type=cl.device_type.DEFAULT)

		# Create a command queue for the context
		self.queue = cl.CommandQueue(self.context)

		# Build a Mako template for the source code
		t = Template(helmsrc, output_encoding='ascii')

		# Render the source template for the specific problem
		# and compile the OpenCL kernels in the program
		self.fdcl = cl.Program(self.context, t.render(dim=self.grid,
			srcidx=self.srcidx, dt=self.dt, dh = self.h)).build()

		# Build the 2-D compute grid size
		self.gsize = tuple(g - 2 for g in self.grid[:2])

		mf = cl.mem_flags
		# Allocate the CL buffer for the squared sound speed
		self.csq = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
				hostbuf=(c.flatten('F').astype(np.float32))**2)
		# Allocate the CL buffers for current and off-time pressure
		# The size is the same as the sound-speed map
		self.pa = [cl.Buffer(self.context, mf.READ_WRITE,
			size=self.csq.size) for i in range(2)]

		# Determine the maximum work items for the kernel
		maxwork = self.fdcl.helm.get_work_group_info(
				cl.kernel_work_group_info.WORK_GROUP_SIZE,
				self.context.devices[0])

		# Conservatively allocate a local memory tile
		self.ltile = cl.LocalMemory(5 * (maxwork + 2) * np.float32().nbytes)


	def boundary(self, xl, xr, yl, yr, zl, zr):
		'''
		Enforce boundary conditions at the left (0-index) and right
		(-1-index) boundaries in each plane.
		'''

		# Determine the dimensions of the boundary slices
		slabdims = [[self.grid[i] for i in range(len(self.grid)) if i != j]
				for j in range(len(self.grid))]

		# Bundle the functions used to set the boundary values
		slabfuncs = [self.fdcl.xbdy, self.fdcl.ybdy, self.fdcl.zbdy]

		# Bundle the left and right slab values for each dimension
		slabvals = [(xl, xr), (yl, yr), (zl, zr)]

		mf = cl.mem_types

		# Create the buffers and copy data in place for each dimension
		for d, v, f in zip(slabdims, slabvals, slabfuncs):
			# Copy the boundary values into OpenCL buffers
			l = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
					hostbuf=v[0].flatten('F').astype(np.float32))
			r = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
					hostbuf=v[1].flatten('F').astype(np.float32))

			# Invoke the value-copying function
			f(self.queue, d, None, self.pa[0], l, r)


	def update(self):
		'''
		Update the pressure everywhere away from the boundary. The
		boundaries are forced separately.
		'''

		# Grab the next source value
		v = np.float32(self.source.next())

		# Invoke the OpenCL kernel
		self.fdcl.helm(self.queue, self.gsize, None, self.pa[1],
				self.pa[0], self.csq, v, self.ltile)

		# Cycle the next and current pressures
		self.pa = [self.pa[1], self.pa[0]]


	def p(self):
		'''Return the current pressure field.'''

		# Create an array to store the current pressure
		p = np.empty(self.grid, dtype=np.float32, order='F')
		# Enqueue and wait for the copy
		cl.enqueue_copy(self.queue, p, self.pa[0]).wait()

		return p
