"General-purpose OpenCL routines."
import pyopencl as cl, os.path as path, numpy as np
from .. import cutil

def srcpath(fname, subdir, sname):
	'''
	Given a file with an absolute path fname and another file with a local
	name sname, return the full path to sname by assuming it resides in the
	subdirectory subdir of the directory that contains fname.
	'''
	return path.join(path.split(path.abspath(fname))[0], subdir, sname)


def grabcontext(context = None):
	'''
	If context is unspecified or is None, create and return the default
	context. Otherwise, context must either be an PyOpenCL Context instance
	or an integer. If context is a PyOpenCL Context, do nothing but return
	the argument. Otherwise, return the device at the corresponding
	(zero-based) index of the first platform available on the system.
	'''
	# Return a default context if nothing was specified
	if context is None: return cl.Context(dev_type = cl.device_type.DEFAULT)

	# The provided argument is a context, return it
	if isinstance(context, cl.Context): return context

	# Try to return the specified device
	return cl.Context(devices=[cl.get_platforms()[0].get_devices()[context]])


class RectangularTransfer(object):
	'''
	A reusable parameter store (and host-side buffer) for rectangular
	buffer transfers between a host and device.
	'''
	def __init__(self, bufshape, hostshape, dtype, alloc_host=True):
		'''
		Establish a transfer window that is the intersection of the
		host and device shapes, and determine the host and device
		origins for the transfer.

		The transfer window is always centered along each axis of both
		device and host arrays.

		FORTRAN ordering is always assumed.
		'''
		# Ensure that all dimensions are compatible
		if len(hostshape) != len(bufshape):
			raise ValueError('Dimensionality of arrays buffers must agree')
		# Ensure that the dimensionality is 2 or 3
		if len(bufshape) != 2 and len(bufshape) != 3:
			raise ValueError('Rectangular transfers require 2-D or 3-D arrays')

		# Copy the transfer parameters and the data type
		self.hostshape = hostshape[:]
		self.bufshape = bufshape[:]
		self.dtype = dtype

		# Grab the number of bytes in each device record
		byterec = dtype().nbytes

		# Compute the transfer region and buffer and host origins
		region, buffer_origin, host_origin = cutil.commongrid(bufshape, hostshape)

		# Scale the first dimension by the number of bytes in each record
		region[0] *= byterec
		host_origin[0] *= byterec
		buffer_origin[0] *= byterec

		# Set the buffer and host pitches
		buffer_pitches = [byterec * bufshape[0]]
		host_pitches = [byterec * hostshape[0]]
		# The slice pitch is required for 3-D copies
		if len(bufshape) > 2:
			buffer_pitches += [buffer_pitches[0] * bufshape[1]]
			host_pitches += [host_pitches[0] * hostshape[1]]

		# Save the transfer parameters
		self.region = region
		self.host_origin = host_origin
		self.buffer_origin = buffer_origin
		self.buffer_pitches = buffer_pitches
		self.host_pitches = host_pitches

		# Optionally allocate a host-side buffer to receive transfers
		if alloc_host: self.rcvbuffer = np.zeros(hostshape, dtype, order='F')
		else: self.rcvbuffer = None


	def fromdevice(self, queue, clbuffer, is_blocking=False):
		'''
		Initiate a transfer on the specified queue from the specified
		device buffer to the internal host-side buffer.
		'''
		if self.rcvbuffer is None:
			raise IOError('Device-to-host transfer requires host-side buffer')

		cl.enqueue_copy(queue, self.rcvbuffer, clbuffer,
				region=self.region,
				buffer_origin=self.buffer_origin,
				host_origin=self.host_origin,
				buffer_pitches=self.buffer_pitches,
				host_pitches=self.host_pitches,
				is_blocking=is_blocking)
		return self.rcvbuffer


	def todevice(self, queue, clbuffer, hostbuf, is_blocking=False):
		'''
		Initiate a transfer on the specified queue from the specified
		host buffer to the specified device buffer.

		If necessary, the data type of the host array will be
		reinterpreted to the expected type before transfer.
		'''
		if list(self.hostshape) != list(hostbuf.shape):
			raise ValueError('Host buffer dimensions differ from expected values')

		# Transfer to the device if a host buffer was provided
		# Reinterpret the data type of the host buffer if necessary
		if hostbuf.dtype != self.dtype: hostbuf = hostbuf.astype(buftype)
		cl.enqueue_copy(queue, clbuffer, hostbuf,
				region=self.region,
				buffer_origin=self.buffer_origin,
				host_origin=self.host_origin,
				buffer_pitches=self.buffer_pitches,
				host_pitches=self.host_pitches,
				is_blocking=is_blocking)
