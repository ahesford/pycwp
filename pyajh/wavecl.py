import numpy as np, scipy.special as spec, math
import pyopencl as cl, pyopencl.array as cla, os.path as path

from itertools import product, chain
from mako.template import Template
from . import fdtd

class SplineInterpolator(object):
	'''
	Use OpenCL to quickly interpolate harmonic functions defined at regular
	sample points on the sphere using cubic b-splines.
	'''

	_kernel = path.join(path.split(path.abspath(__file__))[0], 'spline.mako')

	def __init__(self, ntheta, nphi, tol = 1e-7, context = None):
		'''
		Create OpenCL kernels to convert samples of a harmonic
		function, sampled at regular points on the unit sphere, into
		cubic b-spline coefficients. These coefficients can be used for
		rapid, GPU-based interpolation at arbitrary locations.
		'''

		if nphi % 2 != 0:
			raise ValueError('The number of azimuthal samples must be even')

		self.ntheta = ntheta
		self.nphi = nphi
		# This is the polar-ring grid shape
		self.grid = 2 * (ntheta - 1), nphi / 2

		# Set the desired precision of the filter coefficients
		if tol > 0:
			zp = math.sqrt(3) - 2.
			self.precision = int(math.log(tol) / math.log(abs(zp)))
		else: self.precision = ntheta

		# Don't let the precision exceed the number of samples!
		self.precision = min(self.precision, min(ntheta, nphi))

		# Build an OpenCL context if one hasn't been provided
		if context is None:
			self.context = cl.Context(dev_type=cl.device_type.DEFAULT)
		else: self.context = context

		# Build the program for the context
		t = Template(filename=SplineInterpolator._kernel, output_encoding='ascii')
		self.prog = cl.Program(self.context, t.render(ntheta = ntheta, 
			nphi = nphi, p = self.precision)).build()

		# Create a command queue for the context
		self.queue = cl.CommandQueue(self.context)

		# Create an image that will store the spline coefficients
		# Remember to pad the columns to account for repeated boundaries
		mf = cl.mem_flags
		self.coeffs = cl.Image(self.context, mf.READ_WRITE,
				cl.ImageFormat(cl.channel_order.RG, cl.channel_type.FLOAT),
				[g + 3 for g in self.grid])

		# The poles will be stored so they need not be interpolated
		self.poles = 0., 0.


	def buildcoeff(self, f):
		'''
		Convert a harmonic function, sampled on a regular grid, into
		cubic b-spline coefficients that will be stored in the OpenCL
		image self.coeffs.
		'''
		# Store the exact polar values
		self.poles = f[0], f[-1]
		# Reshape the array, polar angle along the rows
		f = np.reshape(f[1:-1], (self.ntheta - 2, self.nphi), order='C')

		# Rearrange the data in the polar-ring format
		c = np.empty(self.grid, dtype=np.complex64, order='F')

		c[1:self.ntheta - 1, :] = f[:, :self.grid[1]]
		c[self.ntheta:, :] = f[-1::-1, self.grid[1]:]

		# Duplicate the polar values
		c[0, :] = self.poles[0]
		c[self.ntheta - 1, :] = self.poles[-1]

		# Copy the polar-ring data to the GPU
		mf = cl.mem_flags
		buf = cl.Buffer(self.context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=c)

		# Invoke the kernels to compute the spline coefficients
		self.prog.polcoeff(self.queue, (self.grid[1],), None, buf)
		self.prog.azicoeff(self.queue, (self.ntheta,), None, buf)

		# Now copy the coefficients into the float image
		self.prog.mat2img(self.queue, self.grid, None, self.coeffs, buf)


	def interpolate(self, ntheta, nphi):
		'''
		Interpolate the previously-established spline representation of
		a function on a regular grid containing ntheta polar samples
		(including the poles) and nphi azimuthal samples.
		'''
		if nphi % 2 != 0:
			raise ValueError('The number of azimuthal samples must be even.')

		# The number of output samples
		nsamp = (ntheta - 2) * nphi + 2;
		# Allocate a GPU buffer for the output
		buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY,
				size = nsamp * np.complex64().nbytes)

		# Call the kernel to interpolate values away from poles
		grid = ntheta - 2, nphi
		self.prog.radinterp(self.queue, grid, None, buf, self.coeffs)

		# Copy the interpolated grid
		f = np.empty((nsamp,), dtype=np.complex64)
		cl.enqueue_copy(self.queue, f, buf).wait()

		# Copy the exact polar values
		f[0] = self.poles[0]
		f[-1] = self.poles[-1]

		return f



class FarMatrix:
	'''
	Compile an OpenCL kernel that will populate a far-field matrix.
	'''

	_kernel = path.join(path.split(path.abspath(__file__))[0], 'farmat.mako')

	def __init__(self, theta, dc = 0.1, n = 4, context = None):
		'''
		Create an OpenCL kernel to build a far-field matrix.

		The polar angular samples are specified in the list theta. The
		first and last entries in the list theta are polar samples that
		correspond to only one unique azimuthal angle.

		The azimuthal samples are equally spaced and number

			2 * (len(theta) - 2).

		The (square) source cells have edge length dc. Gauss-Legendre
		quadrature of order n is used to integrate the cells.

		A desired OpenCL context may be specified in context, or else a
		default context is used.
		'''

		# Build an OpenCL context if one hasn't been provided
		if context is None:
			self.context = cl.Context(dev_type=cl.device_type.DEFAULT)
		else: self.context = context

		# Compute the quadrature points and weights
		self.pts, self.wts = spec.legendre(n).weights.T.tolist()[:2]

		# Copy the integration order and the cell dimensions
		self.n, self.dc = n, dc

		# Read the Mako template for the kernel
		t = Template(filename=FarMatrix._kernel, output_encoding='ascii')

		# Build the program for the context
		self.prog = cl.Program(self.context, t.render(pts = self.pts,
			wts = self.wts, dc = dc)).build()

		# Build a queue for the context
		self.queue = cl.CommandQueue(self.context)

		# Build the wave number
		self.k = np.float32(2 * math.pi)

		# Compute the angular sizes
		ntheta = len(theta)

		# Separate the poles for later
		pv = (theta[0], 0.0), (theta[-1], 0.0)
		theta = theta[1:-1]

		# Count the azimuthal samples and build a generator for them
		nphi = 2 * len(theta)
		phigen = (2. * math.pi * i / float(nphi) for i in range(nphi))

		# Make a generator of angular coordinates with polar samples
		anggen = chain([pv[0]], product(theta, phigen), [pv[1]])

		# Build an array of the coordinates using complex types
		self.angles = np.array([complex(*t) for t in anggen], dtype=np.complex64)

		# Grab the number of angular samples
		self.nsamp = len(self.angles)


	def fillrow(self, src):
		'''
		Use the precompiled kernel to fill a row of the far-field
		matrix corresponding to a source cell with a center specified
		in the list src.

		The azimuthal angle phi most rapidly varies.
		'''
		mf = cl.mem_flags
		# Check for or create device buffers used in the row calculations
		try: getattr(self, 'rowbuf')
		except AttributeError:
			self.rowbuf = cl.Buffer(self.context, mf.WRITE_ONLY,
					size=self.nsamp * np.complex64().nbytes)
		try: getattr(self, 'angbuf')
		except AttributeError:
			self.angbuf = cl.Buffer(self.context, mf.READ_ONLY |
					mf.COPY_HOST_PTR, hostbuf=self.angles)

		# Create the source vector and invoke the kernel
		srcloc = cla.vec.make_float3(*src)
		self.prog.farmat(self.queue, (self.nsamp,), None,
				self.rowbuf, self.k, srcloc, self.angbuf)
		# Copy the result from the device to a NumPy array
		row = np.empty((self.nsamp,), dtype=np.complex64)
		cl.enqueue_copy(self.queue, row, self.rowbuf).wait()
		return row


	def fill(self, srclist):
		'''
		Use the precompiled kernel to fill the far-field matrix for
		sources with centers specified in a list srclist of
		three-element coordinate lists.
		'''
		# Build the matrix row-by-row
		# Device buffers will be created when necessary
		return np.array([fillrow(s) for s in srclist], dtype=np.complex64)



class Helmholtz(fdtd.Helmholtz):
	'''
	A class that works identically to the standard Helmholtz class but that
	uses PyOpenCL to accelerate computations. A desired context can be
	passed in, but one will be created if none is provided.
	'''

	_kernel = path.join(path.split(path.abspath(__file__))[0], 'helmkern.mako')

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
		t = Template(filename=Helmholtz._kernel, output_encoding='ascii')

		# Render the source template for the specific problem
		# and compile the OpenCL kernels in the program
		self.fdcl = cl.Program(self.context, t.render(dim=self.grid,
			srcidx=self.srcidx, dt=self.dt, dh = self.h)).build()

		# Build the 2-D compute grid size
		self.gsize = tuple(g - 2 for g in self.grid[:2])

		# Precompute the r-squared array
		rsq = (c * self.dt / self.h)**2

		mf = cl.mem_flags
		# Allocate the CL buffer for the r-squared parameter
		self.rsq = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
				hostbuf=rsq.flatten('F').astype(np.float32))
		# Allocate the CL buffers for current and off-time pressure
		# The size is the same as the sound-speed map
		self.pa = [cl.Buffer(self.context, mf.READ_WRITE,
			size=self.rsq.size) for i in range(2)]

		# Determine the maximum work items for the kernel
		maxwork = self.fdcl.helm.get_work_group_info(
				cl.kernel_work_group_info.WORK_GROUP_SIZE,
				self.context.devices[0])

		# Conservatively allocate a local memory tile
		self.ltile = cl.LocalMemory(3 * (maxwork + 2) * np.float32().nbytes)


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
		# Needs to be scaled by the square of the time step
		v = np.float32(self.h**2 * self.source.next())

		# Invoke the OpenCL kernel
		self.fdcl.helm(self.queue, self.gsize, None, self.pa[1],
				self.pa[0], self.rsq, v, self.ltile)

		# Cycle the next and current pressures
		self.pa = [self.pa[1], self.pa[0]]


	def p(self):
		'''Return the current pressure field.'''

		# Create an array to store the current pressure
		p = np.empty(self.grid, dtype=np.float32, order='F')
		# Enqueue and wait for the copy
		cl.enqueue_copy(self.queue, p, self.pa[0]).wait()

		return p
