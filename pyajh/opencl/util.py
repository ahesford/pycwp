"General-purpose OpenCL routines."
import pyopencl as cl, os.path as path, numpy as np, time
from threading import Thread, Lock, ThreadError
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
	def __init__(self, bufshape, hostshape, dtype, alloc_host=True, context=None):
		'''
		Establish a transfer window that is the intersection of the
		host and device shapes, and determine the host and device
		origins for the transfer.

		The transfer window is always centered along each axis of both
		device and host arrays.

		If alloc_host is True, a host-side buffer will be allocated for
		receivers. Otherwise, requests for copies from the device must
		provide an external buffer. If, additionally, context is a
		valid OpenCL context, the buffer will map to device memory for
		more efficient transfers.

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
		if alloc_host:
			if context is None:
				self.rcvbuffer = np.zeros(hostshape, dtype, order='F')
			else:
				queue = cl.CommandQueue(context)
				dflags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR
				nbytes = byterec * int(np.prod(hostshape))
				self.devbuffer = cl.Buffer(context, dflags, nbytes)
				self.rcvbuffer = cl.enqueue_map_buffer(queue, self.devbuffer,
						cl.map_flags.READ, 0, hostshape, dtype, order='F')[0]
		else: self.rcvbuffer = None


	def fromdevice(self, queue, clbuffer, hostbuf=None, is_blocking=False):
		'''
		Initiate a transfer on the specified queue from the specified
		device buffer to the internal host-side buffer.
		'''
		# If no buffer was provided, use the default
		if hostbuf is None: hostbuf = self.rcvbuffer
		else:
			# Check that the host buffer is compatible with the transfer
			if list(self.hostshape) != list(hostbuf.shape):
				raise ValueError('Incompatible host buffer shape')
			if self.dtype != hostbuf.dtype:
				raise TypeError('Incompatible host buffer type')

		# If a default was not allocated, fail
		if hostbuf is None:
			raise IOError('Device-to-host transfer requires host-side buffer')

		cl.enqueue_copy(queue, hostbuf, clbuffer,
				region=self.region,
				buffer_origin=self.buffer_origin,
				host_origin=self.host_origin,
				buffer_pitches=self.buffer_pitches,
				host_pitches=self.host_pitches,
				is_blocking=is_blocking)
		return hostbuf


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


class BufferedSlices(Thread):
	'''
	Allocate a number of buffers on the device, using CL_MEM_ALLOC_HOST_PTR
	to hopefully page-lock host-side memory, that correspond to slices (in
	FORTRAN order) of an associated Numpy array. A separate thread watches
	for "pending" buffers and, depending on the mode, either fills these
	buffers with the contents of the associated array or fills the
	associated array slices with their contents.
	'''
	def __init__(self, ctx, array, nbufs, read=True, reversed=False):
		'''
		Allocate the desired number of slice buffers in device context
		ctx. Hopefully these will correspond to host-side "pinned"
		memory to allow for rapid copies between host and device.
		Associate the buffers with successive slices of the specified
		Numpy array, either reading (if read=True) from the array to
		the device or writing (if read=False) from the device to the
		array. If reversed=True, slices of array are processed in
		reverse order.

		The array (and its slices) are always interpreted in FORTRAN
		order.
		'''
		# Call the Thread constructor
		Thread.__init__(self)
		# Reference the Numpy array and the device context
		self._array = array
		self._context = ctx
		# Create the host-side and pending queues
		self._hostq = []
		self._pendq = []
		# Record whether the mode is read-write
		self._read = read
		# Record whether array access is reversed
		self._reversed = reversed

		# Create a list of slice objects to refer to one array slice
		self._grid = [slice(None) for d in range(len(array.shape) - 1)]

		# Figure the size of the buffers to allocate
		nbytes = array[self._grid + [0]].nbytes
		# Set some default memory flags depending on the read/write mode
		if read:
			dflags = cl.mem_flags.READ_ONLY | cl.mem_flags.ALLOC_HOST_PTR
			hflags = cl.map_flags.WRITE
		else:
			dflags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR
			hflags = cl.map_flags.READ

		# Create a queue for mapping the device memory
		queue = cl.CommandQueue(ctx)

		# Allocate the device and host buffer pairs
		for b in range(nbufs):
			dbuf = cl.Buffer(ctx, dflags, size=nbytes)
			hbuf = cl.enqueue_map_buffer(queue, dbuf, hflags, 0,
					array.shape[:-1], array.dtype, order='F')[0]
			# Add the buffers to the pending queue in read mode
			if read: self._pendq.append((dbuf, hbuf))
			# Otherwise, add the buffers to the host queue
			else: self._hostq.append((dbuf, hbuf))

		# Allocate a lock to serialize modification to the queues
		self._qlock = Lock()


	def kill(self):
		'''
		Kill the thread by clearing both buffers.
		'''
		self._qlock.acquire()
		self._pendq = []
		self._hostq = []
		self._qlock.release()
		self.join()


	def getslice(self):
		'''
		Return the host-side buffer at the head of the host queue.

		This function blocks until a buffer is available.
		'''
		while True:
			try: return self._hostq[0][1]
			except IndexError:
				if self.isAlive(): time.sleep(1e-4)
				else: raise ValueError('No buffered slice to return')


	def nextslice(self):
		'''
		Mark the head of the host queue as no longer needed. The buffer
		is moved to the pending queue.

		It is not an error to ask for the next slice if the host queue
		is empty.
		'''
		try:
			# Try to move the head of the host queue to the pending queue
			self._qlock.acquire()
			self._pendq.append(self._hostq.pop(0))
		except IndexError: pass
		finally:
			# Always make sure that the queue lock is released
			try: self._qlock.release()
			except ThreadError: pass


	def frontload(self, sl):
		'''
		In read mode, place the contents of the array sl at the head of
		the host queue.

		In write mode, place the contents of the array at the head of
		the pending queue.

		This will throw an IndexError if there are no available buffers
		for the desired operation.
		'''
		try:
			# Lock the queue for destructive operations
			self._qlock.acquire()
			if self._read:
				# Grab the first available pending object
				buf = self._pendq.pop(0)
				# Fill the host-side buffer as desired
				buf[1][self._grid] = sl
				# Inject the buffer into the head of the host queue
				self._hostq.insert(0, buf)
			else:
				# Grab the first available host object
				buf = self._hostq.pop(0)
				# Fill the host-side buffer as desired
				buf[1][self._grid] = sl
				# Inject the buffer into the head of the pending queue
				self._pendq.insert(0, buf)
		finally:
			try: self._qlock.release()
			except ThreadError: pass


	def run(self):
		'''
		Whenever there are items in the pending queue, either write
		them to or read them from the associated array in order.
		'''
		if self._reversed: iter = reversed(self._array.T)
		else: iter = self._array.T

		# Loop through all slices in the associated array
		for idx, av in enumerate(iter):
			if len(self._pendq) + len(self._hostq) == 0: break
			# Keep trying to process head events as they come available
			while True:
				try:
					self._qlock.acquire()
					# Check if the queues have been killed
					if len(self._pendq) + len(self._hostq) == 0: break
					buf = self._pendq.pop(0)
					if self._read: buf[1][self._grid] = av.T
					else: av.T[self._grid] = buf[1]
					self._hostq.append(buf)
				except IndexError: pass
				else: break
				finally:
					try: self._qlock.release()
					except ThreadError: pass
