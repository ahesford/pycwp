import numpy as np, pyopencl as cl, pyopencl.array as cla

from . import fdtd

helmsrc = '''
/* This OpenCL kernel computes a single time step of the acoustic FDTD using
 * shared memory for efficiency. It is based on the NVIDIA sample for
 * general-purpose finite-difference schemes but has been specialized and is
 * intended for use with PyOpenCL and the pyajh Helmholtz solver class.*/

#define MIN(x,y) (((x) > (y)) ? (y) : (x))
#define MAX(x,y) (((x) > (y)) ? (x) : (y))

/* Arguments: pn, in/out, the 3-D grid representing the next and previous time steps.
 * pc, input, the 3-D grid representing the current time step.
 * csq, input, the global 3-D grid of squared sound-speed values.
 * dt, input, the time step.
 * dh, input, the isotropic spatial step.
 * dim, input, the (x,y,z) dimensions of the global grid.
 * srcval, input, the value of the point source term.
 * srcidx, input, the (x,y,z) coordinates of the point source.
 * ltile, input, a local cache for the current time step in a work group. 
 *
 * The global work grid should be 2-D and have at least as many work items in
 * each dimension as the number of non-boundary elements in the corresponding
 * dimension of the global pressure grid. */
__kernel void helm(__global float * const pn, __global float * const pc,
			    __global float * const csq, const float dt,
			    const float dh, const uint3 dim,
			    const float srcval, const uint3 srcidx,
			    __local float * const tile) {
	/* The target cell of the work item. Avoids boundary layers. */
	const uint i = get_global_id(0) + 1;
	const uint j = get_global_id(1) + 1;

	/* The local compute position and grid size. */
	const uint2 lpos = (uint2) (get_local_id(0), get_local_id(1));
	const uint2 ldim = (uint2) (get_local_size(0), get_local_size(1));

	/* Determine the strides of the grids in FORTRAN order. */
	const uint3 strides = (uint3) (1, dim.x, dim.x * dim.y);

	/* The leading dimension of the local work tile. */
	const uint ldw = ldim.x + 2;

	/* The local target cell of the work item. Avoids boundary layers. */
	const uint ltgt = (lpos.y + 1) * ldw + lpos.x + 1;

	/* The scaling for the squared sound speed. */
	const float cscale = (dt / dh) * (dt / dh);

	float rv, value, zr, zl, xr, xl, yr, yl, cur, prev;
	uint idx = 0, k;
	bool updsrc, inbounds;

	/* Limit the local cache dimension to avoid input overruns. */
	ldim.y = MIN(ldim.y, MAX(0, dim.y - 1 - j));
	ldim.x = MIN(ldim.x, MAX(0, dim.x - 1 - i));

	/* Check if the target cell contains the source. */
	updsrc = (i == srcidx.x) && (j == srcidx.y);

	/* Check if the target cell is within the boundaries. */
	inbounds = (i < dim.x - 1) && (j < dim.y - 1);

	/* The current slab is the first one. */
	idx = i * strides.x + j * strides.y;

	/* Only read the value if it is in bounds. */
	if (inbounds) {
		cur = pc[idx];
		zr = pc[idx + strides.z];
	}

	/* Loop through all non-boundary slices sequentially. */
	for (k = 2; k < dim.z; ++k) {
		/* Cycle to the next slab. */
		idx += strides.z;
		zl = cur;
		cur = zr;

		/* Only read the values if they are in bounds. */
		if (inbounds) {
			zr = pc[idx + strides.z];
			prev = pn[idx];
			rv = cscale * csq[idx];
		}

		barrier (CLK_LOCAL_MEM_FENCE);

		/* Add the value to the buffer tile. */
		if (inbounds) tile[ltgt] = cur;

		/* Also grab needed boundary values. */
		if (lpos.y == 0 && inbounds) {
			tile[ltgt - ldw] = pc[idx - strides.y];
			tile[ltgt + ldim.y * ldw] = pc[idx + ldim.y * strides.y];
		}
		if (lpos.x == 0 && inbounds) {
			tile[ltgt - 1] = pc[idx - strides.x];
			tile[ltgt + ldim.x] = pc[idx + ldim.x * strides.x];
		}

		/* Ensure buffer is filled. */
		barrier (CLK_LOCAL_MEM_FENCE);

		/* Perform the temporal update. */
		value = (2. - 6. * rv) * cur - prev;

		/* Grab the in-plane shifted cells. */
		if (inbounds) {
			xl = tile[ltgt - 1];
			xr = tile[ltgt + 1];
			yl = tile[ltgt - ldw];
			yr = tile[ltgt + ldw];
		}
		
		/* Perfrom the spatial updates. */
		value += rv * (zr + zl + xr + xl + yr + yl);

		/* Update the source location. */
		if (updsrc && ((k - 1) == srcidx.z))
			value += srcval * rv * dh * dh;

		if (inbounds) pn[idx] = value;
	}
}

/* Copy the left (low-index) and right (high-index) values in the z boundary
 * planes to the pressure array pc. The global compute grid (x,y) should equal
 * or exceed the (x,y) dimensions of the pressure grid. */
__kernel void zbdy(__global float * const pc, __global float * const left,
			    __global float * const right, const uint3 dim) {
	/* The position of the work item in the global grid. */
	const uint xpos = get_global_id(0);
	const uint ypos = get_global_id(1);

	/* Precompute the starting offset of the last slab. */
	const uint zoff = (dim.z - 1) * dim.x * dim.y;

	/* The in-plane linear index for the cell. */
	const uint ipidx = ypos * dim.x + xpos;

	/* Copy the values if the item points to an in-bounds cell. */
	if (xpos < dim.x && ypos < dim.y) {
		const float2 vals = (float2) (left[ipidx], right[ipidx]);
		pc[ipidx] = vals.s0;
		pc[ipidx + zoff] = vals.s1;
	}
}

/* Copy the left (low-index) and right (high-index) values in the y boundary
 * planes to the pressure array pc. The global compute grid (x,y) should equal
 * or exceed the (x,z) dimensions of the pressure grid. */
__kernel void ybdy(__global float * const pc, __global float * const left,
			    __global float * const right, const uint3 dim) {
	/* The position of the work item in the global grid. */
	const uint xpos = get_global_id(0);
	const uint zpos = get_global_id(1);

	/* Precompute the starting offset of the last y-row. */
	const uint yoff = (dim.y - 1) * dim.x;

	/* The in-plane linear index for each cell. */
	uint ipidx;

	/* Private storage of the values to copy. */
	if (xpos < dim.x && zpos < dim.z) {
		ipidx = zpos * dim.x + xpos;
		const float2 vals = (float2) (left[ipidx], right[ipidx]);
		ipidx = zpos * dim.x * dim.y + xpos;
		pc[ipidx] = vals.s0;
		pc[ipidx + yoff] = vals.s1;
	}
}

/* Copy the left (low-index) and right (high-index) values in the x boundary
 * planes to the pressure array pc. The global compute grid (x,y) should equal
 * or exceed the (y,z) dimensions of the pressure grid. */
__kernel void xbdy(__global float * const pc, __global float * const left,
			    __global float * const right, const uint3 dim) {
	/* The position of the work item in the global grid. */
	const uint ypos = get_global_id(0);
	const uint zpos = get_global_id(1);

	/* Precompute the starting offset of the last x-row. */
	const uint xoff = dim.x - 1;

	/* The in-plane linear index for each cell. */
	uint ipidx;

	/* Private storage of the values to copy. */
	if (ypos < dim.y && zpos < dim.z) {
		ipidx = zpos * dim.y + ypos;
		const float2 vals = (float2) (left[ipidx], right[ipidx]);
		ipidx *= dim.x;
		pc[ipidx] = vals.s0;
		pc[ipidx + xoff] = vals.s1;
	}
}
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
		# They must be of type float32 to work with the OpenCL kernel
		self.dt, self.h = np.float32(dt), np.float32(h)

		# Copy or create the PyOpenCL context
		if context is not None: self.context = context
		else: self.context = cl.Context(dev_type=cl.device_type.DEFAULT)

		# Create a command queue for the context
		self.queue = cl.CommandQueue(self.context)

		# Compile the OpenCL kernels in the program
		self.fdcl = cl.Program(self.context, helmsrc).build()

		mf = cl.mem_flags
		# Allocate the CL buffer for the squared sound speed
		self.csq = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
				hostbuf=(c.flatten('F').astype(np.float32))**2)
		# Allocate the CL buffers for current and off-time pressure
		# The size is the same as the sound-speed map
		self.pa = [cl.Buffer(self.context, mf.READ_WRITE,
			size=self.csq.size) for i in range(2)]

		# Make the source generator and a uint3 vector for the location
		self.source = srcfunc()
		self.srcidx = cla.vec.make_uint3(*srcidx)

		# Grab the problem dimensions
		self.grid = c.shape[:]
		self.dimvec = cla.vec.make_uint3(*self.grid)
		self.gsize = tuple(g - 2 for g in self.grid[:2])

		# Determine the maximum work items for the kernel
		maxwork = self.fdcl.helm.get_work_group_info(
				cl.kernel_work_group_info.WORK_GROUP_SIZE,
				self.context.devices[0])

		# Conservatively allocate a local memory tile
		self.ltile = cl.LocalMemory(5 * (maxwork + 2))


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
			f(self.queue, d, None, self.pa[0], l, r, self.dimvec)


	def update(self):
		'''
		Update the pressure everywhere away from the boundary. The
		boundaries are forced separately.
		'''

		# Grab the next source value
		v = np.float32(self.source.next())

		# Invoke the OpenCL kernel
		self.fdcl.helm (self.queue, self.gsize, None, self.pa[1],
				self.pa[0], self.csq, self.dt, self.h,
				self.dimvec, v, self.srcidx, self.ltile)

		# Cycle the next and current pressures
		self.pa = [self.pa[1], self.pa[0]]


	def p(self):
		'''Return the current pressure field.'''

		# Create an array to store the current pressure
		p = np.empty(self.grid, dtype=np.float32, order='F')
		# Enqueue and wait for the copy
		cl.enqueue_copy(self.queue, p, self.pa[0]).wait()

		return p
