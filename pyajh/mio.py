'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import os, math, numpy as np
from . import cutil

def getmattype (infile, dim = None, dtype = None):
	'''
	Read a header from the provided array data file (or a path to a file)
	to determine the size and data type of the array therein. The file will
	be left pointing to the data beyond the header. The data type and
	dimension may be provided to avoid auto detection.
	'''

	# Open the input file if it isn't already open
	if isinstance(infile, (str, unicode)): infile = open (infile, mode='rb')

	# The maximum allowed dimension
	maxdim = max(dim, 3)

	# Record the size of the file
	fsize = os.fstat(infile.fileno())[6]

	# Read the maximum-length header, which is two more integeris
	# than the max dimension for grouped files
	hdr = np.fromfile(infile, dtype=np.int32, count=maxdim+2)

	# Try to limit the pool of auto-sensed data types, if possible
	# Ensure that the datatype is a Numpy type
	if dtype is not None:
		dtype = np.dtype(dtype).type
		datatypes = { dtype().nbytes : dtype }
	else: datatypes = { np.float32().nbytes : np.float32,
			np.complex64().nbytes : np.complex64,
			np.complex128().nbytes : np.complex128 }

	# Set the range of dimensions to check
	if dim is None: dimrange = range(1, maxdim + 1)
	else: dimrange = [dim]

	# Loop through each possible dimension, checking data types
	for dimen in dimrange:
		# Set the appropriate header size and matrix size
		if hdr[0] == 0: 
			hdrlen = dimen + 2
			matsize = np.array([hdr[dimen+1] * hv for hv in hdr[1:dimen+1]])
		else: 
			hdrlen = dimen
			matsize = hdr[:dimen]

		# Check the size of the file, minus the header
		dsize = fsize - hdrlen * np.int32().nbytes;

		# Grab the number of bytes per record and
		# check that the record size lines up
		nbytes = dsize / np.prod(matsize)
		if nbytes * np.prod(matsize) != dsize: continue

		# Try to grab the data type of the records
		try: 
			dtype = datatypes[nbytes]
			break
		except KeyError: continue
	else: raise TypeError('Could not determine data type in file.')

	# Seek to the end of the header
	infile.seek(hdrlen * np.int32().nbytes)

	# Ensure that the type and shape are compatible with array types
	dtype = np.dtype(dtype)
	matsize = tuple(int(d) for d in matsize)

	# Return the matrix size and data type
	return (matsize, dtype)


def writebmat (mat, outfile):
	'''
	Write a binary matrix file in the provided precision. The parameter
	outfile may be a string, in which case it is the name of the output
	file to be overwritten; or it may be an already-open file object.
	'''

	# Open the output file if it isn't already open
	if isinstance(outfile, (str, unicode)): outfile = open (outfile, mode='wb')

	# This will close the file when writing is done
	with outfile:
		# Write the size header to the matrix
		np.array(mat.shape, dtype=np.int32).tofile(outfile)

		# Write the matrix body in FORTRAN order by transposing first
		mat.T.tofile(outfile)


def writemmap(infile, shape, dtype):
	'''
	Create a file representing a binary matrix file and memmap the file
	into a Numpy array.
	'''
	outhdr = np.array(shape, np.int32)
	hdrbytes = outhdr.nbytes
	# Create the memmap
	arr = np.memmap(infile, mode='w+', dtype=dtype,
			offset=hdrbytes, order='F', shape=shape)
	# Open the file and write the header
	f = open(infile, 'rb+')
	f.seek(0)
	outhdr.tofile(f)
	return arr


def readbmat (infile, dim = None, dtype = None):
	'''
	Memory map a binary, complex matrix file, auto-sensing the precision
	and dimension. The dimension and precision can be manually specified if
	desired. The map is copy-on-write, so changes will not be saved. The
	parameter infile may be a string, in which case it is the name of the
	input file; or it may be an already-open file object.
	'''

	# Open the input file if it isn't already open
	if isinstance(infile, (str, unicode)): infile = open (infile, mode='rb')

	# This will close the file when writing is done
	with infile:
		# Read the matrix header and determine the data type
		matsize, dtype = getmattype(infile, dim, dtype)

		# Create the read-only memory map and close the source file
		datamap = np.memmap(infile, offset=infile.tell(),
				mode='c', dtype=dtype, shape=matsize, order='F')
	return datamap


def createbmat(outfile, shape, dtype):
	'''
	Create a file-backed array using numpy.memmap for a matrix with the
	specified shape and data type.
	'''
	# Open the output file if it isn't already open
	if isinstance(outfile, (str, unicode)): outfile = open(outfile, mode='w+b')

	# This will close the file object when creation is done
	with outfile:
		# Truncate the file and write the header
		outfile.truncate(0)
		np.array(shape, dtype=np.int32).tofile(outfile)
		# Create the memmap
		datamap = np.memmap(outfile, offset=outfile.tell(), mode='w+',
				dtype=dtype, shape=tuple(shape), order='F')

	return datamap


class Slicer(object):
	'''
	This class opens a data file that can be read or written to one slice
	at a time. A slice is defined as a chunk of data with a dimension one
	less than the whole set, i.e., one column of a FORTRAN-ordered matrix
	or one slab of a FORTRAN-ordered three-dimensional grid.
	'''
	def __init__(self, f, dim = None, dtype = None, trunc = False):
		'''
		If f is a file object or a string corresponding to an existing
		file name, create a Slicer object referring to the data stored
		in the file. In this case, dim and dtype are optional arguments
		used to avoid automatic dimensionality and type detection. In
		this case, dim must be the number of dimensions in the file.

		If trunc is True, any existing file is treated as if it did not
		exist and is overwritten.

		If f is a string corresponding to a file name that does not
		exist, an empty data file is created. In this case, dim and
		dtype MUST be specified. The argument dim must be an iterable
		object specifying the length of the data in each dimension.
		'''
		try:
			# Try to open for read/write an existing file,
			# unless truncation is desired
			if trunc: raise IOError('Ignoring any existing file')

			# Open a file if a name was provided
			if isinstance(f, (str, unicode)): f = open(f, mode='rb+')

			# Grab the matrix header, size, and data type
			self.shape, self.dtype = getmattype(f, dim, dtype)

			# The file was not created or truncated by the slicer
			self._created = False
		except IOError:
			# The file does not exist or should be clobbered
			# Make sure the dimensions and data type are specified
			if dim is None or dtype is None:
				raise ValueError('Grid dimensions and data type must be specified to create file')

			# Open a file if a name was provided
			if isinstance(f, (str, unicode)): f = open(f, mode='wb+')
			else: f.truncate(0)

			# Note that the file was created or truncated
			self._created = True

			# Copy the matrix shape and data type
			self.shape = tuple(int(d) for d in dim)
			self.dtype = np.dtype(dtype)

			# Open the file and write the header
			np.array(self.shape, dtype=np.int32).tofile(f)

			# Set the desired size of the output
			nbytes = cutil.prod(self.shape) * self.dtype.type().nbytes
			f.truncate(f.tell() + nbytes)

		# Copy the backer file
		self._backer = f

		# The number of elements and bytes per slice
		self.nelts = cutil.prod(self.shape[:-1])
		self.slicebytes = self.nelts * self.dtype.type().nbytes

		# Store the start of the data block
		self.fstart = self._backer.tell()


	def __enter__(self):
		'''
		Return self to allow the Slicer to act as a context manager.
		'''
		return self


	def __exit__(self, exc_type, exc_val, exc_tb):
		'''
		Close the backer file on context manager exit. If an exception
		was raised and the file was created or truncated, truncate to
		zero length to avoid lengthy write-outs on OS X.
		'''
		clean = (exc_type is None and exc_val is None and exc_tb is None)

		# Truncate the file on exception
		if not clean and self._created is True:
			self._backer.truncate(0)

		# Close the backer
		self._backer.close()

		# Tell the interpreter whether to raise an exception
		return clean


	def __del__(self):
		'''
		Close the open data file on delete.
		'''
		self._backer.close()


	def __len__(self):
		'''
		Return the number of slices.
		'''
		return self.shape[-1]


	def __iter__(self):
		'''
		Create a generator to read each desired slice in succession.
		Returns a tuple of the slice index and its data values.
		'''
		# Point to the first slice in the desired range
		self.setslice(0)

		# Yield each slice along the last index
		for idx in range(len(self)):
			yield (idx, self[idx])

		return


	def clobber(self):
		'''
		Truncate the backer to zero length.
		'''
		self._backer.truncate(0)


	def setslice(self, i):
		'''
		Point the file to the start of slice i.
		'''
		# Ensure the requested index is valid
		if -self.shape[-1] > i or i >= self.shape[-1]:
			raise IndexError('Requested slice is out of bounds')

		# Wrap negative indices in the Python fashion
		if i < 0: i = i + self.shape[-1]

		# Point to the start of the desired slice
		self._backer.seek(self.fstart + i * self.slicebytes)


	def readslice(self):
		'''
		Read the slice at the current file position.
		'''
		data = np.fromfile(self._backer, dtype=self.dtype, count=self.nelts)
		if data.size != self.nelts or data is None:
			raise ValueError('Failed to read current slice')
		return data.reshape(self.shape[:-1], order='F')


	def writeslice(self, slab):
		'''
		Write the slab at the current file position, converting the
		data type to that specified for the file. The array will be
		reshaped with FORTRAN ordering into the expected slab shape.
		'''
		sflat = slab.ravel('F')

		if sflat.shape[0] != self.nelts:
			raise ValueError('Slice size does not agree with output')

		# Convert the data type if necessary
		if self.dtype != sflat.dtype:
			sflat = sflat.astype(self.dtype)
		sflat.tofile(self._backer)
		self._backer.flush()


	def __getitem__(self, key):
		'''
		Grab slices from the data file using list-style indexing.
		'''
		try:
			# Treat the key as a slice to pull out multiple slabs
			idx = key.indices(self.shape[-1])
			# Compute the number of slabs to be read
			nslab = (idx[1] - idx[0]) / idx[2]
			if idx[0] + idx[2] * nslab < idx[1]: nslab += 1
			# Allocate storage for the requested slices
			data = np.empty(list(self.shape[:-1]) + [nslab], dtype=self.dtype)
			# Create slice objects corresponding to a single slab
			sl = [slice(None) for s in self.shape[:-1]]

			# Read all of the requested slices
			for li, i in enumerate(range(*idx)):
				self.setslice(i)
				data[sl + [li]] = self.readslice()
		except AttributeError:
			# A slice was not provided, grab a single slab
			self.setslice(key)
			data = self.readslice()

		return data


	def __setitem__(self, key, value):
		'''
		Write slices to the file using list-style indexing.
		'''
		try:
			# Treat the key as a slice to write multiple slabs
			idx = key.indices(self.shape[-1])
			# Compute the number of slabs to be written
			nslab = (idx[1] - idx[0]) / idx[2]
			if idx[0] + idx[2] * nslab < idx[1]: nslab += 1
			# Check that the number of slices matches
			if value.shape[-1] != nslab:
				raise IndexError('Numbers of input and output slices do not agree')

			# Create slice objects corresponding to a single slab
			sl = [slice(None) for s in value.shape[:-1]]

			for li, i in enumerate(range(*idx)):
				self.setslice(i)
				self.writeslice(value[sl + [li]])
		except AttributeError:
			# A slice was not provided, write a single slab
			self.setslice(key)
			self.writeslice(value)
