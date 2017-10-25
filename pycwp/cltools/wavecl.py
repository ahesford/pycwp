# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, scipy.special as spec, math
import pyopencl as cl, pyopencl.array as cla

from pyfft.cl import Plan
from itertools import product, chain
from mako.template import Template

from . import util
from .. import fdtd, cutil

class FarMatrix:
	'''
	Compile an OpenCL kernel that will populate a far-field matrix.
	'''

	_kernel = util.srcpath(__file__, 'clsrc', 'farmat.mako')

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
		t = Template(filename=self._kernel, output_encoding='ascii')

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

	_kernel = util.srcpath(__file__, 'clsrc', 'helmkern.mako')

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
		t = Template(filename=self._kernel, output_encoding='ascii')

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
		v = np.float32(self.h**2 * next(self.source))

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

	_kernel = util.srcpath(__file__, 'clsrc', 'splitstep.mako')

	def __init__(self, k0, nx, ny, h, d=None, l=10, dz=None,
			w=0.39, propcorr=None, phasetol=None, context=None):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0. The wave
		is advanced in steps of dz or (if dz is not provided) h.

		If d is provided, it is a 4-tuple that describes the
		directivity of any source as d = (dx, dy, dz, w), where (dx,
		dy, dz) is the directivity axis and w is the beam width
		parameter. Otherwise, all sources are treated as point sources.

		If l is specified and greater than zero, it is the width of a
		Hann window used to attenuate the field along each edge.

		The parameter w (as a multiplier 1 / w**2) governs the
		high-order spectral cross term.

		If propcorr is specified, it should be a tuple of Booleans of
		the form (hospec, hospat) that determines whether corrections
		involving high-order spectral or spatial terms, respectively,
		are used in the propagator by default. If propcorr is
		unspecified, both corrections are used.

		The parameter phasetol specifies the maximum permissible phase
		deviation (in fractions of pi), relative to propagation through
		the homogeneous background, incurred by propagating through the
		inhomogeneous medium. The number of steps per slab will be
		adjusted so that the phase shift incurred by propagating
		through materials with the most extreme sound speeds will never
		exceed phasetol.
		'''
		# Ensure that the phase tolerance is not too small
		# Otherwise, number of propagation steps will blow up uncontrollably
		if phasetol is not None and abs(phasetol) < 1e-6:
			raise ValueError('Phase tolerance must be greater than 1e-6')

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0, self.l = h, k0, l
		self.w = np.float32(1. / w**2)
		self.phasetol = phasetol
		# Specify the use of corrective terms
		if propcorr is not None:
			self.propcorr = tuple(propcorr)
		else: self.propcorr = (True, True)

		# Set the step length
		self.dz = dz if dz else h

		# Grab the provided context or create a default
		self.context = util.grabcontext(context)

		# Build the program for the context
		t = Template(filename=self._kernel, output_encoding='ascii')
		src = t.render(grid = self.grid, k0=k0, h=h, d=d, l=l)
		self.prog = cl.Program(self.context, src).build()

		# Create a command queue for forward propagation calculations
		self.fwdque = cl.CommandQueue(self.context)
		# Create command queues for transfers
		self.recvque = cl.CommandQueue(self.context)
		self.sendque = cl.CommandQueue(self.context)

		# Create an FFT plan in the OpenCL propagation queue
		# Reorder the axes to conform with row-major ordering
		self.fftplan = Plan((ny, nx), queue=self.fwdque)

		grid = self.grid
		def newbuffer():
			nbytes = cutil.prod(grid) * np.complex64().nbytes
			flags = cl.mem_flags.READ_WRITE
			return util.SyncBuffer(self.context, flags, size=nbytes)
		# Buffers to store the propagating (twice) and backward fields
		self.fld = [newbuffer() for i in range(3)]
		# Scratch space used during computations
		self.scratch = [newbuffer() for i in range(3)]
		# The index of refraction gets two buffers for transmission
		self.obj = [newbuffer() for i in range(2)]
		# Two buffers are used for the Goertzel FFT of the contrast source
		self.goertzbuf = [newbuffer() for i in range(2)]
		# The sound speed extrema for the current slab are stored here
		self.speedlim = [1., 1.]
		# Initialize buffer to hold results of advance()
		self.result = newbuffer()

		# By default, volume fields will be transfered from the device
		self._goertzel = False

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


	def reset(self, propcorr = None, goertzel = False):
		'''
		Reset the propagating and backward fields to zero and the prior
		refractive index buffer to unity.

		If propcorr is provided, it should be tuple as described in the
		docstring for __init__(). This will change the default behavior
		of corrective terms in the propagator.

		If goertzel is True, calls to advance() with shift=True will
		not copy the shifted, combined field from the device. Instead,
		the field will be used in a Goertzel algorithm to compute
		(slice by slice) the Fourier transform, restricted to the unit
		sphere, of induced scattering sources.
		'''
		if propcorr is not None: self.propcorr = tuple(propcorr)
		grid = self.grid
		z = np.zeros(grid, dtype=np.complex64)
		for a in self.fld + self.obj + self.goertzbuf:
			cl.enqueue_copy(self.fwdque, a, z, is_blocking=False)
		self.speedlim = [1., 1.]
		self._goertzel = goertzel


	def setroi(self, rgrid):
		'''
		Set a region of interest that will limit device transfers
		within the computational grid.
		'''
		self.rectxfer = util.RectangularTransfer(self.grid, rgrid, np.complex64, alloc_host=False)


	def setincident(self, srcloc, idx = 0):
		'''
		Set the value of the CL field buffer at index idx to the
		incident field at a location srcloc represented as a 3-tuple.

		The field plane is always assumed to have a z-height of 0; the
		z coordinate of srcloc is therefore the height of the source
		above the field plane. The transverse origin (x, y) = (0, 0)
		corresponds to the midpoint of the field plane.
		'''
		inc = self.fld[idx]
		sx, sy, dz = [np.float32(s) for s in srcloc]
		self.prog.green3d(self.fwdque, self.grid, None, inc, sx, sy, dz)


	def objupdate(self, obj):
		'''
		Update the rolling buffer with the index of refraction,
		corresponding to an object contrast obj, for the next slab.

		The transmission queue is used for updates to facilitate
		concurrency with forward propagation algorithms.
		'''
		# Transfers occur in the transmission queue for concurrency
		prog, queue, grid = self.prog, self.recvque, self.grid
		# Roll the buffer so the next slab is second
		nxt, cur = self.obj
		self.obj = [cur, nxt]

		# Figure approximate sound speed extrema in the upcoming slab
		# If the imaginary part of the wave number is negligible
		# compared to the real, maximum sound speed corresponds to the
		# minimum contrast and vice versa
		if self.phasetol:
			ctextrema = [np.max(obj.real), np.min(obj.real)]
			self.speedlim = [1 / np.sqrt(ctv + 1.) for ctv in ctextrema]

		# Ensure buffer is not used by prior calculations
		nxt.sync(queue)
		# Transfer the object contrast into the next-slab buffer
		evt = self.rectxfer.todevice(queue, nxt, obj)

		# Return buffers of the current and next slabs and a transfer event
		return cur, nxt, evt


	def propagate(self, fld = None, dz = None, idx = 0, corr = None):
		'''
		Propagate the field stored in the device buffer fld (or, if fld
		is None, the current in-device field at index idx) a step dz
		(or, if dz is None, the default step size) through the
		currently represented medium.

		If corr is not None, it should be a tuple as described in the
		reset() docstring to control the use of corrective terms in the
		spectral propagator. Otherwise, the instance default is used.
		'''
		prog, grid = self.prog, self.grid
		fwdque = self.fwdque
		hospec, hospat = corr if corr is not None else self.propcorr

		# Point to the field, scratch buffers, and refractive index
		if fld is None: fld = self.fld[idx]
		u, v, x = self.scratch
		obj = self.obj[0]

		# These constants will be used in field computations
		one = np.float32(1.)
		dz = np.float32(dz if dz is not None else self.dz)

		# Attenuate the boundaries using a Hann window, if desired
		if self.l > 0:
			prog.attenx(fwdque, (self.l, grid[1]), None, fld)
			prog.atteny(fwdque, (grid[0], self.l), None, fld)

		# Multiply, in v, the field by the contrast
		if hospec: prog.ctmul(fwdque, grid, None, v, obj, fld)
		# Multiply, in u, the field by the high-order spatial operator
		if hospat: prog.hospat(fwdque, grid, None, u, obj, fld)

		# From here, the field should be spectral
		self.fftplan.execute(fld)

		# Compute high-order spatial corrections or set the buffer to NULL
		if hospat:
			# With high-order spatial terms, transform u as well
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
		else: x = None

		# Compute high-order spectral corrections or set buffers to NULL
		if hospec:
			# Apply, in u, the high-order spectral operator to the field
			prog.hospec(fwdque, grid, None, u, fld)
			# Multiply u by the contrast in the spatial domain
			self.fftplan.execute(u, inverse=True)
			prog.ctmul(fwdque, grid, None, u, obj, u)
			# Let v = v + u / w**2 in the spatial domain
			prog.caxpy(fwdque, grid, None, v, self.w, u, v)
			# Transform u and v into the spectral domain
			self.fftplan.execute(u)
			self.fftplan.execute(v)
			# Apply the high-order spectral operator to the new v
			prog.hospec(fwdque, grid, None, v, v)
		else: u, v = None, None

		# Add all appropriate high-order corrections to the field
		if hospat or hospec:
			prog.corrfld(fwdque, grid, None, fld, u, v, x, dz)

		# Propagate the field through a homogeneous slab
		prog.propagate(fwdque, grid, None, fld, dz)

		# Take the inverse FFT of the field and the Laplacian
		self.fftplan.execute(fld, inverse=True)

		# Multiply by the phase screen, returning the event
		return prog.screen(fwdque, grid, None, fld, obj, dz)


	def advance(self, obj, shift=False, corr=None, shcorr=None):
		'''
		Propagate a field through the current slab and transmit it
		through an interface with the next slab characterized by object
		contrast obj. The transmission overwrites the refractive index
		of the current slab with the interface reflection coefficients.

		If shift is True, the forward is shifted by half a slab to
		agree with full-wave solutions and includes a
		backward-traveling contribution caused by reflection from the
		interface with the next slab.

		The relevant result (either the forward field or the
		half-shifted combined field) is copied into a device-side
		buffer for later retrieval and handling.

		If corr is not None, it should be a tuple as specified in the
		reset() docstring to override the default use of corrective
		terms in the spectral propagator.

		The argument shcorr is interpreted exactly as corr, but is used
		instead of corr for the propagation used to shift the field to
		the center of the slab.
		'''
		prog, grid = self.prog, self.grid
		fwdque, recvque, sendque = self.fwdque, self.recvque, self.sendque

		# Point to the field components
		fwd, bck, buf = [f for f in self.fld]

		if shift:
			# Ensure that a prior copy isn't using the buffer
			buf.sync(fwdque)
			# Copy the forward field for shifting if necessary
			cl.enqueue_copy(fwdque, buf, fwd)

		# Copy the sound speed extrema for the current slab
		speedlim = list(self.speedlim)
		# Push the next slab to its buffer (overwrites speed extrema)
		ocur, onxt, obevt = self.objupdate(obj)

		if self.phasetol is not None:
			# Figure maximum propagation distance to not
			# exceed maximum permissible phase deviation
			dzl = []
			for spd in speedlim:
				# Sign governs the sign of the phase deviation,
				# which is irrelevant, so ignore it here
				spdiff = max(abs(spd - 1.), 1e-8)
				# Preventing spdiff from reaching zero limits
				# maximum permissible propagation distance
				dzl.append(abs(0.5 * self.phasetol * spd / spdiff))
			# Subdivide the slab into maximum propagation distance
			nsteps = max(1, int(np.round(self.dz / min(dzl))))
		else: nsteps = 1
		dz = self.dz / nsteps

		# Ensure that no prior copy is using the field buffer
		fwd.sync(fwdque)

		# Propagate the forward field through the slab on the fwdque
		for i in range(nsteps): self.propagate(fwd, dz, corr=corr)

		# Ensure next slab has been received before handling interface
		cl.enqueue_barrier(fwdque, wait_for=[obevt])

		# Compute transmission through the interface
		# The reflected field is only of interest if a shift is desired
		transevt = prog.txreflect(fwdque, grid, None,
				fwd, bck if shift else None, ocur, onxt)
		# Hold the current contrast slab until the transmission is done
		ocur.attachevent(transevt)

		if shift:
			# Add the forward and backward fields
			prog.caxpy(fwdque, grid, None, buf, np.float32(1.), buf, bck)
			# Propagate the combined field a half step
			# Save the propagation event for delaying result copies
			pevt = self.propagate(buf, 0.5 * self.dz, corr=shcorr)

			# Handle Goertzel iterations to compute the Fourier
			# transform of the contrast source on the unit sphere
			if self._goertzel:
				# Compute the FFT of the source in the XY plane
				crt = self.scratch[0]
				prog.ctmul(fwdque, grid, None, crt, ocur, buf)
				self.fftplan.execute(crt)
				# Compute the next Goertzel iteration
				pn1, pn2 = self.goertzbuf
				dz = np.float32(self.dz)
				# The final argument (slab count) is not yet used
				nz = np.int32(0)
				prog.goertzelfft(fwdque, grid, None, pn1, pn2, crt, dz, nz)
				# Cycle the Goertzel buffers
				self.goertzbuf = [pn2, pn1]
			else:
				# Copy the shifted field into the result buffer
				# No result sync necessary, all mods occur on sendque
				evt = cl.enqueue_copy(sendque, self.result, buf, wait_for=[pevt])
				# Attach the copy event to the source buffer
				buf.attachevent(evt)
		else: 
			# Copy the forward field into the result buffer
			# Wait for transmissions to finish for consistency
			evt = cl.enqueue_copy(sendque, self.result, fwd, wait_for=[transevt])
			# Attach the copy event to the field buffer
			fwd.attachevent(evt)


	def getresult(self, hbuf):
		'''
		Wait for the intra-device transfer of the previous result to
		the result buffer, and initiate a device-to-host copy of the
		valid result buffer into hbuf.

		An event corresponding to the transfer is returned.
		'''
		sendque = self.sendque
		# Initiate the rectangular transfer on the transfer queue
		# No sync necessary, all mods to result buffer occur on sendque
		evt = self.rectxfer.fromdevice(sendque, self.result, hbuf)[1]
		# Attach the copy event to the result buffer
		self.result.attachevent(evt)
		return evt


	def goertzelfft(self, nz = 0):
		'''
		Finish Goertzel iterations carried out in repeated calls to
		advance() and copy the positive and negative hemispheres of the
		Fourier transform of the contrast source to successive planes
		of a Numpy array.

		If nz is specified, it is the number of slabs involved in the
		Fourier transform and is used to properly scale the output of
		the Goertzel algorithm. When nz = 0, no scaling is performed.

		Copies are synchronous and are done on the forward propagation
		queue.
		'''
		prog, grid = self.prog, self.grid
		fwdque = self.fwdque
		hemispheres = np.zeros(list(grid) + [2], dtype=np.complex64, order='F')
		# If the spectral field hasn't been computed, just return zeros
		if not self._goertzel: return hemispheres
		# Finalize the Goertzel iteration
		pn1, pn2 = self.goertzbuf
		dz = np.float32(self.dz)
		nz = np.int32(nz)
		# Pass None as the contrast current to signal final iteration
		# After this, pn1 is the positive hemisphere, pn2 is the negative
		prog.goertzelfft(fwdque, grid, None, pn1, pn2, None, dz, nz)
		# Copy the two hemispheres into planes of an array
		cl.enqueue_copy(fwdque, hemispheres[:,:,0:1], pn1, is_blocking=False)
		cl.enqueue_copy(fwdque, hemispheres[:,:,1:2], pn2, is_blocking=True)
		return hemispheres
