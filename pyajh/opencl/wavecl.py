import numpy as np, scipy.special as spec, math
import pyopencl as cl, pyopencl.array as cla

from pyfft.cl import Plan
from itertools import product, chain
from mako.template import Template

from . import util
from .. import fdtd

class SplineInterpolator(object):
	'''
	Use OpenCL to quickly interpolate harmonic functions defined at regular
	sample points on the sphere using cubic b-splines.
	'''

	_kernel = util.srcpath(__file__, 'mako', 'spline.mako')

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

		# Grab the provided context or create a default
		self.context = util.grabcontext(context)

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

	_kernel = util.srcpath(__file__, 'mako', 'farmat.mako')

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

		# Grab the provided context or create a default
		self.context = util.grabcontext(context)

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

	_kernel = util.srcpath(__file__, 'mako', 'helmkern.mako')

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

		# Grab the provided context or create a default
		self.context = util.grabcontext(context)

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
				hostbuf=rsq.astype(np.float32).ravel('F'))
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
					hostbuf=v[0].astype(np.float32).ravel('F'))
			r = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR,
					hostbuf=v[1].astype(np.float32).ravel('F'))

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


class SplitStep(object):
	'''
	An OpenCL version of the split-step parabolic equation solver.
	'''

	_kernel = util.srcpath(__file__, 'mako', 'splitstep.mako')

	def __init__(self, k0, nx, ny, h, src, d = None,
			l = 10, dz = None, w = 0.32, context = None):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0. The wave
		is advanced in steps of dz or (if dz is not provided) h.

		The source location is specified in a three-tuple src with
		units of wavelengths. If d is provided, it is a 4-tuple that
		describes the directivity of the source as d = (dx, dy, dz, w),
		where (dx, dy, dz) is the directivity axis and w is the beam
		width parameter.

		If l is specified and greater than zero, it is the width of a
		Hann window used to attenuate the field along each edge.

		The parameter w (as a multiplier 1 / w**2) governs the
		high-order spectral cross term.
		'''
		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0, self.l = h, k0, l
		self.w = np.float32(1. / w**2)
		# Set the step length
		self.dz = dz if dz else h

		# Grab the provided context or create a default
		self.context = util.grabcontext(context)

		# Build the program for the context
		t = Template(filename=SplitStep._kernel, output_encoding='ascii')
		self.prog = cl.Program(self.context, t.render(grid = self.grid,
			k0=k0, h=h, src=src, d=d, l=l)).build()

		# Create a command queue for forward propagation calculations
		self.fwdque = cl.CommandQueue(self.context)
		# Create a command queue for transmissions
		self.tranque = cl.CommandQueue(self.context)

		# Create an FFT plan in the OpenCL propagation queue
		self.fftplan = Plan((nx, ny), queue=self.fwdque)

		grid = self.grid
		def newbuffer():
			nbytes = int(np.prod(grid)) * np.complex64().nbytes
			flags = cl.mem_flags.READ_WRITE
			return util.SyncBuffer(self.context, flags, size=nbytes)
		# Buffers to store the propagating (twice) and backward fields
		self.fld = [newbuffer() for i in range(3)]
		# Scratch space used during computations
		self.scratch = [newbuffer() for i in range(3)]
		# The index of refraction gets two buffers for transmission
		self.obj = [newbuffer() for i in range(2)]
		# Initialize buffer to hold results of advance()
		self.result = newbuffer()

		# Initialize refractive index and fields
		self.reset()

		# By default, device exchange happens on the full grid
		self.rectxfer = util.RectangularTransfer(grid, grid, np.complex64, alloc_host=False)


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.ogrid[:g[0], :g[1]]
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	def reset(self):
		'''
		Reset the propagating and backward fields to zero and the prior
		refractive index buffer to unity.
		'''
		grid = self.grid
		z = np.zeros(grid, dtype=np.complex64)
		for a in self.fld + self.obj: 
			cl.enqueue_copy(self.fwdque, a, z, is_blocking=False)


	def setroi(self, rgrid):
		'''
		Set a region of interest that will limit device transfers
		within the computational grid.
		'''
		self.rectxfer = util.RectangularTransfer(self.grid, rgrid, np.complex64, alloc_host=False)


	def setincident(self, zoff, idx = 0):
		'''
		Set the value of the CL field buffer at index idx to the
		incident field at a height zoff.
		'''
		inc = self.fld[idx]
		zoff = np.float32(zoff)
		self.prog.green3d(self.fwdque, self.grid, None, inc, zoff)


	def objupdate(self, obj):
		'''
		Update the rolling buffer with the index of refraction,
		corresponding to an object contrast obj, for the next slab.

		The transmission queue is used for updates to facilitate
		concurrency with forward propagation algorithms.  '''
		# Transfers occur in the transmission queue for concurrency
		prog, queue, grid = self.prog, self.tranque, self.grid
		# Roll the buffer so the next slab is second
		nxt, cur = self.obj
		self.obj = [cur, nxt]

		# Ensure that the next slab is free for reuse
		nxt.sync(queue)
		# Transfer the object contrast into the next-slab buffer
		evt = self.rectxfer.todevice(queue, nxt, obj)

		# Return buffers of the current and next slabs and a transfer event
		return cur, nxt, evt


	def propagate(self, fld = None, dz = None, idx = 0):
		'''
		Propagate the field stored in the device buffer fld (or, if fld
		is None, the current in-device field at index idx) a step dz
		(or, if dz is None, the default step size) through the
		currently represented medium.
		'''
		prog, grid = self.prog, self.grid
		fwdque = self.fwdque

		# Point to the field, scratch buffers, and refractive index
		if fld is None: fld = self.fld[idx]
		u, v, x = [s for s in self.scratch]
		obj = self.obj[0]

		# These constants will be used in field computations
		one = np.float32(1.)
		dz = np.float32(dz if dz is not None else self.dz)

		# Attenuate the boundaries using a Hann window, if desired
		if self.l > 0:
			prog.attenx(fwdque, (self.l, grid[1]), None, fld)
			prog.atteny(fwdque, (grid[0], self.l), None, fld)

		# Multiply, in v, the field by the contrast
		prog.ctmul(fwdque, grid, None, v, obj, fld)
		# Multiply, in u, the field by the high-order spatial operator
		prog.hospat(fwdque, grid, None, u, obj, fld)

		# From here, the field, u and v should be spectral
		self.fftplan.execute(fld)
		self.fftplan.execute(v)
		self.fftplan.execute(u)

		# Compute the scaled, spectral Laplacians of u and the field (in x)
		prog.laplacian(fwdque, grid, None, u, u)
		prog.laplacian(fwdque, grid, None, x, fld)
		# Apply the high-order spatial operator to x
		self.fftplan.execute(x, inverse=True)
		prog.hospat(fwdque, grid, None, x, obj, x)
		self.fftplan.execute(x)
		# Add x to u to get the high-order spatial corrections
		prog.caxpy(fwdque, grid, None, x, one, u, x)

		# Apply, in u, the high-order spectral operator to the field
		prog.hospec(fwdque, grid, None, u, fld)
		# Multiply u by the contrast in the spatial domain
		self.fftplan.execute(u, inverse=True)
		prog.ctmul(fwdque, grid, None, u, obj, u)
		self.fftplan.execute(u)
		# Apply the high-order spectral operator to v = v + u / w**2
		prog.caxpy(fwdque, grid, None, v, self.w, u, v)
		prog.hospec(fwdque, grid, None, v, v)
		# Now add to v the second-order term in u
		prog.caxpy(fwdque, grid, None, v, one, u, v)

		# Add the high-order corrections to the field
		prog.corrfld(fwdque, grid, None, fld, x, v, dz)

		# Propagate the field through a homogeneous slab
		prog.propagate(fwdque, grid, None, fld, dz)

		# Take the inverse FFT of the field and the Laplacian
		self.fftplan.execute(fld, inverse=True)

		# Multiply by the phase screen, returning the event
		return prog.screen(fwdque, grid, None, fld, obj, dz)


	def advance(self, obj, bfld=None, shift=False):
		'''
		Propagate a field through the current slab and transmit it
		through an interface with the next slab characterized by object
		contrast obj. The transmission overwrites the refractive index
		of the current slab with the interface reflection coefficients.

		The field bfld, if provided, runs opposite the direction of
		propagation that will be reflected and added to the propagated
		field after transmission across the interface.

		If shift is True, the forward and backward field are also
		shifted by half a slab to agree with full-wave solutions.

		The relevant result (either the forward field or the
		half-shifted combined field) is copied into a device-side
		buffer for later retrieval and handling.

		Note that getresult() or wait() should be called between calls
		to advance() to ensure that the buffers used in advance() are
		available for use.
		'''
		prog, grid = self.prog, self.grid
		fwdque, tranque = self.fwdque, self.tranque

		# Point to the field components
		fwd, bck, buf = [f for f in self.fld]

		# Copy the forward field for shifting if necessary
		if shift: cl.enqueue_copy(fwdque, buf, fwd)

		# Push the next slab to its buffer
		ocur, onxt, obevt = self.objupdate(obj)

		if bfld is not None:
			# Copy the backward-traveling field
			bfevt = self.rectxfer.todevice(tranque, bck, bfld)
			# Attach the copy event to the backward buffer
			bck.attachevent(bfevt)

		# Propagate the forward field a whole step
		self.propagate(fwd)

		if shift:
			if bfld is not None:
				# Ensure the copy has finished
				bck.sync(fwdque)
				# Add the forward and backward fields
				prog.caxpy(fwdque, grid, None, buf, 
						np.float32(1.), buf, bck)
			# Propagate the combined field a half step
			# Save the propagation event for delaying result copies
			pevt = self.propagate(buf, 0.5 * self.dz)
			self.result.sync(tranque)
			# Copy the shifted field into the result buffer
			evt = cl.enqueue_copy(tranque, self.result, buf, wait_for=[pevt])
			# Attach the copy event to the result buffer
			self.result.attachevent(evt)

		# Ensure the next slab has been received
		cl.enqueue_marker(fwdque, wait_for=[obevt])
		# Compute transmission through the interface
		if bfld is None:
			evt = prog.transmit(fwdque, grid, None, fwd, ocur, onxt)
		# Include reflection of backward field if appropriate
		else:
			# Ensure the backward field has been received
			bck.sync(fwdque)
			evt = prog.txreflect(fwdque, grid, None, fwd, bck, ocur, onxt)

		# Attach the transmission event to the current slab contrast
		ocur.attachevent(evt)

		# If shifted fields were desired, the result copy has already begun
		if shift: return

		# Make sure the result buffer is available
		self.result.sync(fwdque)
		# Copy the forward field into the result buffer
		# Use the forward queue to ensure transmissions have completed
		evt = cl.enqueue_copy(fwdque, self.result, fwd)
		# Attach the copy event to the result buffer
		self.result.attachevent(evt)


	def getresult(self, hbuf):
		'''
		Wait for the intra-device transfer of the previous result to
		the result buffer, and initiate a device-to-host copy of the
		valid result buffer into hbuf.

		An event corresponding to the transfer is returned. This event
		also becomes the "ready condition" for recycling of the result
		buffer in subsequent calls to advance().
		'''
		tranque = self.tranque
		# Make sure the result buffer is available
		self.result.sync(tranque)
		# Initiate the rectangular transfer on the transfer queue
		evt = self.rectxfer.fromdevice(tranque, self.result, hbuf)[1]
		# Attach the copy event to the result buffer
		self.result.attachevent(evt)
		return evt


	def wait(self):
		'''
		Block until all pending transfers and computations are finished.
		'''
		self.fwdque.finish()
		self.tranque.finish()
