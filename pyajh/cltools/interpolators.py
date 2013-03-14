import numpy as np, math
import pyopencl as cl

from mako.template import Template

from . import util

class HarmonicSpline(object):
	'''
	Use OpenCL to quickly interpolate harmonic functions defined at regular
	sample points on the sphere using cubic b-splines.
	'''

	_kernel = util.srcpath(__file__, 'clsrc', 'harmspline.mako')

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
		t = Template(filename=self._kernel, output_encoding='ascii')
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


class InterpolatingRotator(object):
	'''
	Use OpenCL on a GPU device to linearly interpolate or rotate a 2-D
	image of complex float values.
	'''

	_kernel = util.srcpath(__file__, 'clsrc', 'rotinterp.cl')

	def __init__(self, dstshape, srcshape, context=None):
		'''
		Build an OpenCL kernel that will interpolate an image with
		shape srcshape (which should be a tuple with 2 elements) into
		an image with shape dstshape.
		'''
		# Grab the provided context or create a default
		self.context = util.grabcontext(context)
		# Create a command queue for the program execution
		self.queue = cl.CommandQueue(self.context)

		# Create the OpenCL images
		self.setshapes(dstshape, srcshape)

		# Build the program
		src = '\n'.join(open(self._kernel, 'r').readlines())
		self.prog = cl.Program(self.context, src).build()


	def setshapes(self, dstshape, srcshape):
		'''
		Set the destination and source image shapes and allocate OpenCL
		image types for use by the rotating and interpolating kernel.
		'''
		if len(dstshape) != len(srcshape):
			raise ValueError('Source and destination shapes must have same length')
		if len(dstshape) != 2:
			raise ValueError('Only 2-D image interpolations are supported')

		# Copy the source and destination shapes
		self.srcshape = tuple(srcshape[:])
		self.dstshape = tuple(dstshape[:])

		# Build the image format
		fmt = cl.ImageFormat(cl.channel_order.RG, cl.channel_type.FLOAT)
		mf = cl.mem_flags

		# Create the images used for interpolation
		context = self.context
		self._srcim = cl.Image(context, mf.READ_ONLY, fmt, shape=self.srcshape)
		self._dstim = cl.Image(context, mf.WRITE_ONLY, fmt, shape=self.dstshape)


	def loadimg(self, img):
		'''
		Enqueue a copy of the image img, interpreted as an array of
		complex floats, into the OpenCL source image for rotation or
		interpolation.
		'''
		if img.shape != self.srcshape:
			raise ValueError('Image shape does not match configured shape')
		# Convert the data type if necessary
		if img.dtype != np.complex64:
			img = img.astype(np.complex64)
		# Copy the source image into the device storage
		cl.enqueue_copy(self.queue, self._srcim, img.ravel('F'),
				origin=(0,0), region=self.srcshape, is_blocking=False)


	def interpolate(self, img, res=None, rotate=0., contract=(1.,1.)):
		'''
		Given the image img (which will be interpreted as an image of
		complex floats), return the linearly interpolated image of the
		previously configured shape.

		The source will be contracted by factors contract[0] in x and
		contract[1] in y and rotated an angle rotate (in radians). The
		destination grid is rotated an angle rotate (in radians) about
		the center of the grid of the source image.
		'''
		self.loadimg(img)
		# Run the kernel to perform the interpolation
		cx, cy = [np.float32(c) for c in contract]
		theta = np.float32(rotate)
		self.prog.rotinterp(self.queue, self.dstshape, None,
				self._dstim, self._srcim, theta, cx, cy)
		# Create a host-side array to store the result
		if res is None:
			res = np.empty(self.dstshape, np.complex64, order='F')
		evt = cl.enqueue_copy(self.queue, res, self._dstim, origin=(0,0),
				region=self.dstshape, is_blocking=False)
		return res, evt
