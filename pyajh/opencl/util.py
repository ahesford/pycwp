"General-purpose OpenCL routines."
import pyopencl as cl, os.path as path, numpy as np, time
from threading import Thread, Condition, ThreadError
from itertools import chain
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


class SyncBuffer(cl.Buffer):
	'''
	This extends a standard PyOpenCL Buffer object by providing an attached
	event. The event may be injected into a queue to force a
	synchronization point, or it may be replaced with another event.
	'''
	def __init__(self, *args, **kwargs):
		'''
		See documentation for pyopencl.Buffer for argument descriptions.
		'''
		# Initialize the underlying buffer object
		cl.Buffer.__init__(self, *args, **kwargs)
		# Create an empty event
		self._event = None


	def attachevent(self, evt):
		'''
		Attach the event evt to the buffer.
		'''
		self._event = evt


	def detachevent(self):
		'''
		Clear an event attached to the buffer.
		'''
		self._event = None


	def sync(self, queue):
		'''
		Inject a pyopencl marker into the specified queue that waits
		for the attached event.
		'''
		if self._event is None: return
		cl.enqueue_marker(queue, wait_for=[self._event])


	def wait(self):
		'''
		Block until the associated event is clear.
		'''
		try: self._event.wait()
		except AttributeError: pass


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

		# Nothing more to do if no host-side buffer was requested
		if not alloc_host:
			self.rcvbuf = None
			return

		# Allocate a host-side buffer...
		if context is not None:
			# As a map of a device buffer if a context was given
			queue = cl.CommandQueue(context)
			dflags = cl.mem_flags.WRITE_ONLY | cl.mem_flags.ALLOC_HOST_PTR
			hflags = cl.map_flags.READ
			nbytes = byterec * int(np.prod(hostshape))
			self.devbuf = cl.Buffer(context, dflags, nbytes)
			self.rcvbuf = cl.enqueue_map_buffer(queue, self.devbuf,
					hflags, 0, hostshape, dtype, order='F')[0]
		else: self.rcvbuf = np.zeros(hostshape, dtype, order='F')


	def fromdevice(self, queue, clbuffer, hostbuf=None, is_blocking=False, **kwargs):
		'''
		Initiate a transfer on the specified queue from the specified
		device buffer to the internal host-side buffer.

		The host-side buffer and an event corresponding to the transfer
		are returned.
		'''
		# If no buffer was provided, use the default
		if hostbuf is None: hostbuf = self.rcvbuf
		else:
			# Check that the host buffer is compatible with the transfer
			if list(self.hostshape) != list(hostbuf.shape):
				raise ValueError('Incompatible host buffer shape')
			if self.dtype != hostbuf.dtype:
				raise TypeError('Incompatible host buffer type')

		# If a default was not allocated, fail
		if hostbuf is None:
			raise IOError('Device-to-host transfer requires host-side buffer')

		event = cl.enqueue_copy(queue, hostbuf, clbuffer,
				region=self.region,
				buffer_origin=self.buffer_origin,
				host_origin=self.host_origin,
				buffer_pitches=self.buffer_pitches,
				host_pitches=self.host_pitches,
				is_blocking=is_blocking, **kwargs)
		return hostbuf, event


	def todevice(self, queue, clbuffer, hostbuf, is_blocking=False, **kwargs):
		'''
		Initiate a transfer on the specified queue from the specified
		host buffer to the specified device buffer.

		If necessary, the data type of the host array will be
		reinterpreted to the expected type before transfer.

		An event corresponding to the transfer is returned.
		'''
		if list(self.hostshape) != list(hostbuf.shape):
			raise ValueError('Host buffer dimensions differ from expected values')

		# Transfer to the device if a host buffer was provided
		# Reinterpret the data type of the host buffer if necessary
		if hostbuf.dtype != self.dtype: hostbuf = hostbuf.astype(buftype)
		event = cl.enqueue_copy(queue, clbuffer, hostbuf,
				region=self.region,
				buffer_origin=self.buffer_origin,
				host_origin=self.host_origin,
				buffer_pitches=self.buffer_pitches,
				host_pitches=self.host_pitches,
				is_blocking=is_blocking, **kwargs)
		return event


class BufferedSlices(Thread):
	'''
	Allocate a number of buffers on the device, using CL_MEM_ALLOC_HOST_PTR
	to hopefully page-lock host-side memory, that correspond to slices (in
	FORTRAN order) of an associated Numpy array. A separate thread watches
	for "pending" buffers and, depending on the mode, either fills these
	buffers with the contents of the associated array or fills the
	associated array slices with their contents.

	A device context of None may also be provided to only allocate
	host-side buffers.
	'''
	def __init__(self, array, nbufs, read=True, reversed=False, context=None):
		'''
		Allocate the desired number of slice buffers in a device
		context. Hopefully these will correspond to host-side "pinned"
		memory to allow for rapid copies between host and device.
		Associate the buffers with successive slices of the specified
		Numpy array, either reading (if read=True) from the array to
		the device or writing (if read=False) from the device to the
		array. If reversed=True, slices of array are processed in
		reverse order.

		If context is None, allocate host-side buffers only.

		The array (and its slices) are always interpreted in FORTRAN
		order.
		'''
		# Call the Thread constructor
		Thread.__init__(self)
		# Reference the Numpy array and the device context
		self._array = array
		self._context = context
		# Create the ready, idle, and active queues
		self._idleq = []
		self._activeq = []
		self._readyq = []
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
		if context is not None: queue = cl.CommandQueue(context)

		# Allocate the device and host buffer pairs
		for b in range(nbufs):
			if context is not None:
				dbuf = cl.Buffer(context, dflags, size=nbytes)
				hbuf = cl.enqueue_map_buffer(queue, dbuf, hflags, 0,
						array.shape[:-1], array.dtype, order='F')[0]
			else:
				dbuf = None
				hbuf = np.zeros(array.shape[:-1], array.dtype, order='F')
			# Store the device and host buffers and a placeholder for an
			# event that will block until flushes or fills can occur
			record = (dbuf, hbuf, None)
			# Add the buffers to the pending queue in read mode
			if read: self._idleq.append(record)
			# Otherwise, add the buffers to the host queue
			else: self._readyq.append(record)

		# This condition watches for queue changes
		self._qwatch = Condition()
		# Allocate lists of frontloaded and backloaded slices
		self._frontload = []
		self._backload = []
		# Mark this thread a daemon to allow it to die on exit
		self.daemon = True


	def kill(self):
		'''
		Kill the thread by clearing both buffers.
		'''
		with self._qwatch:
			self._readyq = []
			self._idleq = []
			self._activeq = []
			# Notify the parent thread that something has happened
			self._qwatch.notifyAll()
		self.join()


	def queuedepth(self):
		'''
		Return the depth of the queue.
		'''
		return len(self._readyq) + len(self._idleq) + len(self._activeq)


	def queueavail(self):
		'''
		Return the depth of the ready and idle queues. This is the
		number of items ready to be manipulated or flushed/filled.
		'''
		return len(self._readyq) + len(self._idleq)


	def flush(self):
		'''
		Block until all idle buffers are processed.
		'''
		with self._qwatch:
			while self.isAlive() and len(self._idleq) > 0:
				self._qwatch.wait()


	def getslice(self):
		'''
		Return the host-side buffer at the head of the ready queue.

		This function blocks until a buffer is available.
		'''
		# Acquire the access ready condition
		with self._qwatch:
			# As long as the queue is working, wait until an item is ready
			# To avoid deadlocks, waiting stops if all buffers are "active"
			while self.queueavail() > 0 and len(self._readyq) < 1:
				self._qwatch.wait()
			try:
				# Move the ready item to the active queue
				buf = self._readyq.pop(0)
				self._activeq.append(buf)
				# Return the host buffer for the item
				return buf[1]
			except IndexError:
				raise ValueError('No buffered slice to return')
			# Even if no change was possible (because an all-active queue
			# was just marked for flush), notify others because this thread
			# already consumed a notify meant for another task.
			finally: self._qwatch.notify()


	def nextslice(self, evt=None):
		'''
		Mark the head of the active queue as no longer needed. The buffer
		is moved to the idle queue.

		If evt is not None, it should provide a function wait() that
		will block until the buffer can be flushed or filled.

		It is not an error to ask for the next slice if the host queue
		is empty.
		'''
		# Acquire the flush/fill notification condition
		with self._qwatch:
			try:
				# Grab the head of the active queue
				buf = self._activeq.pop(0)
				# Attach the wait event, if it was specified
				buf = (buf[0], buf[1], evt)
				# Put the record at the end of the idle queue
				self._idleq.append(buf)
				# Notify any watchers of the change
				self._qwatch.notify()
			except IndexError: pass


	def frontload(self, sl):
		'''
		Act as if the array sl is a slice preceding the first slice in
		the backer array. The array must have the same dimensions and
		datatype as one slice of the backer.

		These calls may be chained to produce multiple frontloaded
		slices.
		'''
		if sl.dtype != self._array.dtype:
			raise TypeError('Extra slice is not type-compatible with backer')
		if list(sl.shape) != list(self._array.shape[:-1]):
			raise ValueError('Extra slice shape is not compatible with backer')
		# Note that this has to be transposed because the array
		# iterator is transposed for easy indexing
		self._frontload.append(sl.T)


	def backload(self, sl):
		'''
		Act as if the array sl is a slice following the last slice in
		the backer array. The array must have the same dimensions and
		datatype as one slice of the backer.

		These calls may be chained to produce multiple backloaded
		slices.
		'''
		if sl.dtype != self._array.dtype:
			raise TypeError('Extra slice is not type-compatible with backer')
		if list(sl.shape) != list(self._array.shape[:-1]):
			raise ValueError('Extra slice shape is not compatible with backer')
		# Note that this has to be transposed because the array
		# iterator is transposed for easy indexing
		self._backload.append(sl.T)


	def run(self):
		'''
		Whenever there are items in the pending queue, either write
		them to or read them from the associated array in order.
		'''
		if self._reversed: iter = reversed(self._array.T)
		else: iter = self._array.T

		iter = chain(self._frontload, iter, self._backload)

		# Loop through all slices in the associated array
		for idx, av in enumerate(iter):
			# Acquire the flush condition
			with self._qwatch:
				# Wait until an item is available for action
				while self.queuedepth() > 0 and len(self._idleq) < 1:
					self._qwatch.wait()
	
				# Give up if the queue has been killed
				# All others have been notified
				if self.queuedepth() < 1: break
	
				# Grab the first idle object, existence guaranteed
				buf = self._idleq.pop(0)

				# Attempt to block until flush or fill can occur
				try: buf[2].wait()
				except AttributeError: pass
	
				# Process the idle event
				if self._read: buf[1][self._grid] = av.T
				else: av.T[self._grid] = buf[1]

				# Kill the wait event for recycling
				buf = (buf[0], buf[1], None)
				# Return the event back to the ready queue
				self._readyq.append(buf)
				# Notify anybody watching for ready items
				self._qwatch.notify()
