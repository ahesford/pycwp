"General-purpose OpenCL routines."
import pyopencl as cl, os.path as path, numpy as np, time
from threading import Thread, Condition, RLock, ThreadError
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


def mapbuffer(queue, shape, dtype, dflags = cl.mem_flags.ALLOC_HOST_PTR,
		hflags = cl.map_flags.READ | cl.map_flags.WRITE, order='F'):
	'''
	Create a device buffer in the context associated with queue that
	corresponds to a Numpy array of the specified shape and dtype. By
	default, the device buffer and host map may be both read and written.
	All accepted flags arguments to pyopencl.Buffer and
	pyopencl.enqueue_map_buffer may be passed in dflags and hflags,
	respectively. Note that dflags will ALWAYS be ORed with
	CL_MEM_ALLOC_HOST_PTR to ensure that the memory is mappable. Some
	combinations of dflags (like CL_MEM_USE_HOST_PTR) may therefore be
	incompatible.

	If queue is a device context instead of a queue, a new queue will be
	created for the buffer map.

	The return is a tuple containing the Buffer object and the Numpy array
	that maps to the buffer.
	'''
	# Make sure queue is a CommandQueue, otherwise treat it as a context
	try: context = queue.context
	except AttributeError:
		context = queue
		queue = cl.CommandQueue(context)

	# Figure out the desired number of bytes in the map
	nbytes = np.dtype(dtype).type().nbytes * cutil.prod(shape)
	# Ensure that the device flags always contain CL_MEM_ALLOC_HOST_PTR
	dflags |= cl.mem_flags.ALLOC_HOST_PTR
	# Create the device buffer
	dbuf = cl.Buffer(context, dflags, size=nbytes)
	# Map the device buffer to the host with no offset
	hbuf, evt = cl.enqueue_map_buffer(queue, dbuf, hflags, 0, shape, dtype, order)
	# Ensure that the enqueued mapping has finished
	evt.wait()
	return dbuf, hbuf


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
		cl.enqueue_barrier(queue, wait_for=[self._event])


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
		byterec = np.dtype(dtype).type().nbytes

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
			nbytes = byterec * cutil.prod(hostshape)
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
	def __init__(self, backer, nbufs, read=True, context=None):
		'''
		Allocate the desired number of slice buffers in a device
		context. Hopefully these will correspond to host-side "pinned"
		memory to allow for rapid copies between host and device.
		Associate the buffers with successive slices of the specified
		Slicer object backer, either reading (if read=True) from the
		backer to the device or writing (if read=False) from the device
		to the backer.

		If context is None, allocate host-side buffers only.

		The backer (and its slices) are always interpreted in FORTRAN
		order.
		'''
		# Call the Thread constructor
		Thread.__init__(self)
		# Reference the backer and the device context
		self._backer = backer
		self._context = context
		# Create the ready, idle, and active queues
		self._idleq = []
		self._activeq = []
		self._readyq = []
		# Record whether the mode is read-write
		self._read = read

		# Set some default memory flags depending on the read/write mode
		if read:
			dflags = cl.mem_flags.READ_ONLY
			hflags = cl.map_flags.WRITE
		else:
			dflags = cl.mem_flags.WRITE_ONLY
			hflags = cl.map_flags.READ

		# Create a queue for mapping the device memory
		if context is not None: queue = cl.CommandQueue(context)

		# Allocate the device and host buffer pairs
		for b in range(nbufs):
			if context is not None:
				dbuf, hbuf = mapbuffer(queue, backer.shape[:-1],
						backer.dtype, dflags, hflags)
			else:
				dbuf = None
				hbuf = np.zeros(backer.shape[:-1], backer.dtype, order='F')
			# Store the device and host buffers and a placeholder for an
			# event that will block until flushes or fills can occur
			record = (dbuf, hbuf, None)
			# Add the buffers to the pending queue in read mode
			if read: self._idleq.append(record)
			# Otherwise, add the buffers to the host queue
			else: self._readyq.append(record)

		# Two conditions watch item ready and recycle events
		# The lock is shared to ensure queue alterations are atomic
		queuelock = RLock()
		self._qrecycle = Condition(queuelock)
		self._qwatch = Condition(queuelock)
		# Mark this thread a daemon to allow it to die on exit
		self.daemon = True

		# By default, iterate through the backer in order
		self._iter = range(len(self._backer))


	def kill(self):
		'''
		Kill the thread by clearing both buffers.
		'''
		with self._qwatch, self._qrecycle:
			self._readyq = []
			self._idleq = []
			self._activeq = []
			# Notify all watching threads that the queue has changed
			self._qwatch.notifyAll()
			self._qrecycle.notifyAll()
		self.join()


	def setiter(self, iter):
		'''
		Use the iterator iter to step through slices in the backer,
		instead of the default forward sequential iterator.
		'''
		if self.isAlive():
			raise ThreadError('Cannot set iterator while thread is running')
		self._iter = iter


	def queuedepth(self):
		'''
		Return the depth of the queue.
		'''
		return len(self._readyq) + len(self._idleq) + len(self._activeq)


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
		# Lock the queue
		with self._qwatch:
			# Wait until an item is ready as long as there is
			# a non-empty idle queue
			while len(self._idleq) > 0 and len(self._readyq) < 1:
				self._qwatch.wait()
			try:
				# Move the ready item to the active queue
				buf = self._readyq.pop(0)
				self._activeq.append(buf)
				# Return the host buffer for the item
				return buf[1]
			except IndexError:
				raise IndexError('No remaining slices')


	def nextslice(self, evt=None):
		'''
		Mark the head of the active queue as no longer needed. The buffer
		is moved to the idle queue.

		If evt is not None, it should provide a function wait() that
		will block until the buffer can be flushed or filled.

		It is not an error to ask for the next slice if the host queue
		is empty.
		'''
		# Lock the queue
		with self._qrecycle:
			try:
				# Grab the head of the active queue
				buf = self._activeq.pop(0)
				# Attach the wait event, if it was specified
				buf = (buf[0], buf[1], evt)
				# Put the record at the end of the idle queue
				self._idleq.append(buf)
				# Notify any watchers of the change
				self._qrecycle.notify()
			except IndexError: pass


	def run(self):
		'''
		Whenever there are items in the pending queue, either write
		them to or read them from the associated backer in order.
		'''
		# Create a list of slice objects to refer to one backer slice
		grid = [slice(None) for d in range(len(self._backer.shape) - 1)]

		# Loop through the desired slices in the associated backer
		for idx in self._iter:
			# Lock the queue
			with self._qrecycle:
				# Wait until an item is available for recycling
				while self.queuedepth() > 0 and len(self._idleq) < 1:
					self._qrecycle.wait()
	
				# Give up if the queue has been killed
				if self.queuedepth() < 1: break
	
				# Grab the first idle object, existence guaranteed
				buf = self._idleq.pop(0)

				# Attempt to block until flush or fill can occur
				try: buf[2].wait()
				except AttributeError: pass
	
				# Process the idle event
				if self._read: buf[1][grid] = self._backer[idx]
				else: self._backer[idx] = buf[1]

				# Kill the wait event for recycling
				buf = (buf[0], buf[1], None)
				# Return the event back to the ready queue
				self._readyq.append(buf)
				# Notify all watchers for ready events
				with self._qwatch: self._qwatch.notifyAll()

		# Stay alive to retire any outstanding recylced events
		while True:
			with self._qrecycle:
				# Wait until an item is available for recycling
				while self.queuedepth() > 0 and len(self._idleq) < 1:
					self._qrecycle.wait()
				# Give up if the queue has been killed
				if self.queuedepth() < 1: break
				# Throw away the idle event, existence guaranteed
				self._idleq.pop(0)
				# Notify all watchers that the queue has changed
				with self._qwatch: self._qwatch.notifyAll()
