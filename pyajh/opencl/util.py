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


def rectxfer(queue, buffer, bufshape, buftype, hostbuf = None, hostshape = None):
	'''
	Execute a transfer between host and device memory on the specified
	queue. The OpenCL device buffer is organized as a multi-dimensional
	grid with elements of type buftype (which must be a Numpy type) and
	dimensions of bufshape.

	If hostbuf is provided, hostshape is ignored and the transfer copies
	the contents of hostbuf into the GPU buffer. The host buffer must be a
	Numpy array and will be interpreted to be of the same type as the
	OpenCL device buffer. Any host dimensions shorter than the
	corresponding dimensions of the device dimensions will be centered in
	the device buffer; host dimensions longer than corresponding device
	dimensions will be symmetrically clipped.

	If hostbuf is not provided, hostshape must be provided. The transfer
	will copy the contents of the device buffer into a newly created array
	with the specified shape and a Numpy type that matches the device
	buffer. As with host-to-device transfers, device dimensions that are
	smaller than the corresponding host dimension will be centered in the
	host buffer, and device dimensions that are larger than the
	corresponding host dimensions will be symetrically clipped.

	Multi-dimensional arrays are always interpreted in FORTRAN order.

	All transfers are non-blocking.
	'''
	# Ensure that one of hostbuf and hostshape is specified
	if hostbuf is None and hostshape is None:
		raise ValueError('One of hostbuf or hostshape must be specified')
	# If the host buffer was provided, get its shape
	if hostbuf is not None: hostshape = hostbuf.shape
	# Ensure that all dimensions are compatible
	if len(hostshape) != len(bufshape):
		raise ValueError('The dimensionality of both buffers must agree')
	# Ensure that the dimensionality is 2 or 3
	if len(bufshape) != 2 and len(bufshape) != 3:
		raise ValueError('Rectangular transfers are supported for 2-D or 3-D only')

	# Grab the number of bytes in each device record
	byterec = buftype().nbytes

	# Compute the transfer region along with the buffer and host origins
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

	if hostbuf is not None:
		# Transfer to the device if a host buffer was provided
		cl.enqueue_copy(queue, buffer, hostbuf.astype(buftype).ravel('F'),
				region=region,
				buffer_origin=buffer_origin,
				host_origin=host_origin,
				buffer_pitches=buffer_pitches,
				host_pitches=host_pitches,
				is_blocking=False)
	else:
		# Create a flat buffer to receive data
		hostbuf = np.zeros(np.prod(hostshape), dtype=buftype)
		cl.enqueue_copy(queue, hostbuf, buffer,
				region=region,
				buffer_origin=buffer_origin,
				host_origin=host_origin,
				buffer_pitches=buffer_pitches,
				host_pitches=host_pitches,
				is_blocking=False)
		# Reshape the flat buffer into the expected form
		return hostbuf.reshape(hostshape, order='F')
