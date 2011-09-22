import numpy as np, pyopencl as cl, pyopencl.array as cla

def findiff(a, axis=0):
	'''
	Computes the finite difference along the specified axis.
	'''

	if axis < 0 or axis > 2: raise ValueError('Axis must be 0, 1, or 2')

	# Set up the slicing objects for pulling out the desired difference
	slices = [[slice(None)] * 3] * 2

	# Now modify the +1 and -1 slices along the desired axis
	slices[0][axis] = slice(1, None)
	slices[1][axis] = slice(-1)

	# Compute the difference
	return a[slices[0]] - a[slices[1]]



class PML:
	'''
	A class to encapsulate first-order pressure and particle velocity
	updates in a perfectly matched layer where the boundaries are forced,
	either by boundary conditions or exchange with another medium.
	'''

	def __init__(self, c, dim, dt, h, sig, smap):
		'''
		Initialize a PML with a homogeneous soudn speed c. The tuple
		dim gives the 3-dimensional shape of the region, including each
		boundary face. The time step dt and the spatial step h are
		scalars. The list sig should contain the 1-D attenuation
		profile for the boundary layer in one side of the PML. The
		i-element list smap of lists specifies which edges of the PM
		will be attenuated; smap[i][0] == True to attenuate the left
		edge along axis i, and smap[i][1] == True to attenuate the
		right edge along axis i.
		'''

		# Copy the parameters of the PML
		self.csq = c**2
		self.h = h
		self.dt = dt

		# Initialize the total pressure everywhere in the PML
		self.p = np.zeros(dim)

		# The pressure components are defined away from the boundary
		dml = [d - 2 for d in dim]
		self.px, self.py, self.pz = [np.zeros(dml) for i in range(3)]

		# Initialize the velocity components in the PML
		self.ux = np.zeros([dim[0] - 1] + dml[1:])
		self.uy = np.zeros([dml[0], dim[1] - 1, dml[2]])
		self.uz = np.zeros(dml[:2] + [dim[2] - 1])

		# Initialize the attenuation profiles
		sigma = [np.zeros(dim) for i in range(3)]
		# Append a zero value to the inside of the sigma profile
		sig = list(sig) + [0.]

		# Compute the indices starting from the left edges
		left = np.mgrid[:dim[0], :dim[1], :dim[2]]
		# Compute the indices starting from the right edges
		right = np.mgrid[dim[0]-1:-1:-1, dim[1]-1:-1:-1, dim[2]-1:-1:-1]

		# Add in the PML profiles as specified in the map
		for s, m, cl, cr in zip(sigma, smap, left, right):
			if m[0]: s += cl.choose(sig, mode='clip')
			if m[1]: s += cr.choose(sig, mode='clip')

		# Functions to compute the scaling factors
		stm = lambda t, s : 1. / t - 0.5 * s
		stp = lambda t, s : 1. / t + 0.5 * s

		# Compute the scaling factors for pressure
		# Include the boundary terms for averaging the velocity scaling
		self.txp = [f(dt, sigma[0][:, 1:-1, 1:-1]) for f in [stm, stp]]
		self.typ = [f(dt, sigma[1][1:-1, :, 1:-1]) for f in [stm, stp]]
		self.tzp = [f(dt, sigma[2][1:-1, 1:-1, :]) for f in [stm, stp]]

		# The velocity factors are averaged from the pressure factors
		self.txu = [0.5 * (t[1:,:,:] + t[:-1,:,:]) for t in self.txp]
		self.tyu = [0.5 * (t[:,1:,:] + t[:,:-1,:]) for t in self.typ]
		self.tzu = [0.5 * (t[:,:,1:] + t[:,:,:-1]) for t in self.tzp]

		# Strip away boundary values from the pressure factors
		self.txp = [t[1:-1,:,:] for t in self.txp]
		self.typ = [t[:,1:-1,:] for t in self.typ]
		self.tzp = [t[:,:,1:-1] for t in self.tzp]


	def update(self):
		'''
		Update the particle velocity everywhere in the PML and the
		pressure everywhere away from each boundary. The boundaries are
		all forced separately.
		'''

		# Scale the previous velocity and pressure components
		self.ux *= self.txu[0] / self.txu[1]
		self.uy *= self.tyu[0] / self.tyu[1]
		self.uz *= self.tzu[0] / self.tzu[1]

		self.px *= self.txp[0] / self.txp[1]
		self.py *= self.typ[0] / self.typ[1]
		self.pz *= self.tzp[0] / self.tzp[1]

		# Add pressure derivatives to the velocity components
		self.ux -= findiff(self.p[:, 1:-1, 1:-1], 0) / (self.txu[1] * self.h)
		self.uy -= findiff(self.p[1:-1, :, 1:-1], 1) / (self.tyu[1] * self.h)
		self.uz -= findiff(self.p[1:-1, 1:-1, :], 2) / (self.tzu[1] * self.h)

		# Add velocity derivatives to the pressure components
		self.px -= self.csq * findiff(self.ux, 0) / (self.txp[1] * self.h)
		self.py -= self.csq * findiff(self.uy, 1) / (self.typ[1] * self.h)
		self.pz -= self.csq * findiff(self.uz, 2) / (self.tzp[1] * self.h)

		# Update the total pressure samples away from the boundaries
		self.p[1:-1, 1:-1, 1:-1] = self.px + self.py + self.pz

	
	def boundary(self, xl = 0., xr = 0., yl = 0., yr = 0., zl = 0., zr = 0.):
		'''
		Enforce pressure boundary conditions at the left (0-index) and
		right (-1-index) boundaries in each plane.
		'''

		# Enforce the left and right x-boundaries
		self.p[0,:,:] = xl
		self.p[-1,:,:] = xr

		# Enforce the left and right y-boundaries for each component
		self.p[:,0,:] = yl
		self.p[:,-1,:] = yr

		# Enforce the left and right z-boundaries for each component
		self.p[:,:,0] = zl
		self.p[:,:,-1] = zr



class Helmholtz:
	'''
	A class to encapsulate second-order pressure updates in a bounded
	medium. The boundaries are independently forced directly or through
	exchanges with other media.
	'''

	def __init__(self, c, dt, h, srcfunc, srcidx):
		'''
		Initialize the sound-speed c, time step dt and spatial step h.
		The coroutine srcfunc should provide a time-dependent value
		that describes the incident pressure at index srcidx.
		'''

		# Make a copy of the sound-speed map and parameters
		self.c = c.copy()
		self.dt = dt
		self.h = h

		# The source is a generator created by a coroutine
		self.source = srcfunc()
		self.srcslice = [slice(i,i+1) for i in srcidx]
		self.srcscale = (self.dt * self.c[srcidx[0], srcidx[1], srcidx[2]])**2

		# The pre-computed scale factors make updates more efficient
		# Strip out the unnecessary boundaries
		self.rsq = (self.c[1:-1,1:-1,1:-1] * self.dt / self.h)**2

		# Initialize the pressure at two time steps
		# The current time step is pa[0]; the previous/next is pa[1]
		self.pa = [np.zeros_like(self.c) for i in range(2)]


	def update(self):
		'''
		Update the pressure everywhere away from the boundary. The
		boundaries are forced separately.
		'''

		# Remember that rsq has had the boundaries stripped

		# Grab the next source value away from the boundaries
		srcval = self.srcscale * self.source.next()

		# Perform the time updates, overwriting previous time with next 
		self.pa[1][1:-1,1:-1,1:-1] = ((2. - 6. * self.rsq) * 
				self.pa[0][1:-1,1:-1,1:-1] - self.pa[1][1:-1,1:-1,1:-1])

		# Add in the source value at the desired location
		self.pa[1][self.srcslice] += srcval

		# Perform the spatial updates
		self.pa[1][1:-1,1:-1,1:-1] += self.rsq * (
				self.pa[0][2:,1:-1,1:-1] + self.pa[0][:-2,1:-1,1:-1] + 
				self.pa[0][1:-1,2:,1:-1] + self.pa[0][1:-1,:-2,1:-1] + 
				self.pa[0][1:-1,1:-1,2:] + self.pa[0][1:-1,1:-1,:-2])

		# Cycle the current and next/previous times
		self.pa = [self.pa[1], self.pa[0]]


	def p(self):
		'''Return the current pressure field.'''
		return self.pa[0]


	def boundary(self, xl = 0., xr = 0., yl = 0., yr = 0., zl = 0., zr = 0.):
		'''
		Enforce boundary conditions at the left (0-index) and right
		(-1-index) boundaries in each plane.
		'''

		# Enforce the left and right x-boundaries for each component
		self.pa[0][0,:,:] = xl
		self.pa[0][-1,:,:] = xr

		# Enforce the left and right y-boundaries for each component
		self.pa[0][:,0,:] = yl
		self.pa[0][:,-1,:] = yr

		# Enforce the left and right z-boundaries for each component
		self.pa[0][:,:,0] = zl
		self.pa[0][:,:,-1] = zr



class HelmholtzCL(Helmholtz):
	'''
	A class that works identically to the standard Helmholtz class but that
	uses PyOpenCL to accelerate computations. A desired context can be
	passed in, but one will be created if none is provided.
	'''

	fdtdsrc = '''
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
		self.fdcl = cl.Program(self.context, HelmholtzCL.fdtdsrc).build()

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



class FDTD:
	'''
	A simple FDTD engine that uses a hybrid scalar/vector formulation to
	efficiently compute the propagation of acoustic waves in a medium
	bounded by a perfectly matched layer.
	'''

	def __init__(self, c, cbg, dt, h, sigma, srcfunc, srcidx):
		'''
		Initialize the Helmholtz and first-order solvers with variable
		sound-speed grid c, background PML sound speed cbg, time step
		dt, spatial step h, and a 1-D profile sigma describing
		attenuation from the outer edge to the inner edge of the PML.
		The coroutine srcfunc specifies the source term over the same
		grid as the sound speed.
		'''

		# Force boundary overlaps to have background sound speed
		c[:2,:,:] = cbg
		c[-2:,:,:] = cbg
		c[:,:2,:] = cbg
		c[:,-2:,:] = cbg
		c[:,:,:2] = cbg
		c[:,:,-2:] = cbg

		# Initialize the Helmholtz region
		# Note that the boundary overlaps the PML
		self.helmholtz = Helmholtz(c, dt, h, srcfunc, srcidx)

		# Copy the PML thickness
		self.l = len(sigma)

		# Note the size of the total grid, excluding overlap
		self.tsize = tuple(d + 2 * (self.l - 1) for d in c.shape)
		# Note the size of the Helmholtz grid
		self.hsize = c.shape[:]

		tg, hg, lp = self.tsize, self.hsize, self.l + 1

		# The shapes of the PMLs along each axis
		shapes = [list(hg[:i]) + [lp] + list(tg[i+1:]) for i in range(3)]

		# Placeholders for the PML attenuation maps
		left, right = [True, False], [False, True]
		tedges, fedges = [[True]*2]*3, [[False]*2]*3

		# Note which edges will be attenuated using PMLs
		# Don't attenuate edges shared with other regions
		amaps = [[fedges[:i] + [s] + tedges[i+1:]
			for s in [left, right]] for i in range(3)]

		# The PML list contains three lists, holding, respectively, the
		# left and right sides of the x, y and z PMLs
		self.pml = [[PML(cbg, dim, dt, h, sigma, a) for a in aax]
				for dim, aax in zip(shapes, amaps)]


	def exchange(self):
		'''
		Exchange the boundary values between the PML and Helmholtz regions.
		'''

		# Shorthand for PML thickness offset indices
		l = self.l
		lm = self.l - 1
		lp = self.l + 1

		# Create array to hold total pressure value
		p = self.pressure()

		# Set the boundary values for each PML

		# The x-axis PMLs: interior x surface set from Helmholtz
		self.pml[0][0].boundary(xr = p[l,:,:])
		self.pml[0][1].boundary(xl = p[-lp,:,:])

		# The y-axis PMLs: interior y surface set from Helmholtz
		# The x surfaces are set from the x-axis PMLs
		self.pml[1][0].boundary(xl = p[lm, :lp, :], xr = p[-l, :lp, :],
				yr = p[lm:-lm, l, :])
		self.pml[1][1].boundary(xl = p[lm, -lp:, :], xr = p[-l, -lp:, :],
				yl = p[lm:-lm, -lp, :])

		# The z-axis PMLs: interior z surface set from Helmholtz
		# The x and y surfaces are set from the x- and y-axis PMLs
		self.pml[2][0].boundary(
				xl = p[lm, lm:-lm, :lp],  xr = p[-l, lm:-lm, :lp], 
				yl = p[lm:-lm, lm, :lp],  yr = p[lm:-lm, -l, :lp], 
				zr = p[lm:-lm, lm:-lm, l])
		self.pml[2][1].boundary(
				xl = p[lm, lm:-lm, -lp:],  xr = p[-l, lm:-lm, -lp:], 
				yl = p[lm:-lm, lm, -lp:],  yr = p[lm:-lm, -l, -lp:], 
				zl = p[lm:-lm, lm:-lm, -lp])

		# Copy the PML pressures to the Helmholtz boundary
		self.helmholtz.boundary (
				p[lm, lm:-lm, lm:-lm], p[-l, lm:-lm, lm:-lm],
				p[lm:-lm, lm, lm:-lm], p[lm:-lm, -l, lm:-lm],
				p[lm:-lm, lm:-lm, lm], p[lm:-lm, lm:-lm, -l])

		# Return the total pressure
		return p


	def pressure(self):
		'''
		Return an array containing the total pressure over the union of
		the Helmholtz and PML grids.
		'''

		# Shorthand for PML thickness offset
		l = self.l

		# Create array to hold total pressure value
		p = np.zeros(self.tsize)

		# Fill the Helmholtz pressure in the center
		p[l:-l,l:-l,l:-l] = self.helmholtz.p()[1:-1,1:-1,1:-1]
		# Copy the PML pressure from the left and right x edges
		p[:l,:,:] = self.pml[0][0].p[:l,:,:]
		p[-l:,:,:] = self.pml[0][1].p[-l:,:,:]
		# Copy the PML pressure from the left and right y edges
		p[l:-l,:l,:] = self.pml[1][0].p[1:-1,:l,:]
		p[l:-l,-l:,:] = self.pml[1][1].p[1:-1,-l:,:]
		# Copy the PML pressure from the left and right z edges
		p[l:-l,l:-l,:l] = self.pml[2][0].p[1:-1,1:-1,:l]
		p[l:-l,l:-l,-l:] = self.pml[2][1].p[1:-1,1:-1,-l:]
		
		return p


	def update(self):
		'''
		Update the Helmholtz and PML fields.
		'''

		# Update the Helmholtz pressure away from the forced boundaries
		self.helmholtz.update()

		# Update the PML fields away from the forced boundaries
		for pleft, pright in self.pml:
			pleft.update()
			pright.update()

		# Return the pressure after exchanging boundary values
		return self.exchange()
