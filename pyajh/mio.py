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
	try: datatypes = { dtype().nbytes : dtype }
	except: datatypes = { np.float32().nbytes : np.float32,
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
		datamap = np.memmap(infile, offset=infile.tell(), dtype=dtype, mode='c')

	# Reshape the map in FORTRAN order
	return datamap.reshape(matsize, order='F')


class Slicer(object):
	'''
	This class opens a data file that can be read or written to one slice
	at a time. A slice is defined as a chunk of data with a dimension one
	less than the whole set, i.e., one column of a FORTRAN-ordered matrix
	or one slab of a FORTRAN-ordered three-dimensional grid.
	'''
	def __init__(self, f, dim = None, dtype = None):
		'''
		If f is a file object or a string corresponding to an existing
		file name, create a Slicer object backed by the existing file.
		In this case, optional arguments dim and dtype may be used to
		avoid automatic dimensionality and type detection. In this
		case, dim must be the number of dimensions in the file.

		If f is a string corresponding to a file name that does not
		exist, an empty data file is created. In this case, dim and
		dtype MUST be specified. The argument dim must be an iterable
		object specifying the length of the data in each dimension.
		'''
		try:
			# Treat the file as if it already exists
			# If the name is a string, open the file for reading
			if isinstance(f, (str, unicode)):
				f = open(f, mode='rb+')

			# Grab the matrix header, size, and data type
			self.shape, self.dtype = getmattype(f, dim, dtype)
		except IOError:
			# If the file does not exist, the name must be a string
			if not isinstance(f, (str, unicode)):
				raise TypeError('Specify a string to create a Slicer backing file')

			# Make sure the dimensions and data type are specified
			if dim is None or dtype is None:
				raise ValueError('Grid dimensions and data type must be specified to create file')

			# Copy the matrix shape and data type
			self.shape = dim[:]
			self.dtype = dtype

			# Open the file and write the header
			f = open(f, mode='wb+')
			np.array(self.shape, dtype=np.int32).tofile(f)

			# Truncate the file to the desired size
			f.truncate(f.tell() + cutil.prod(self.shape) * self.dtype().nbytes)

		# Copy the backer file
		self.backer = f

		# The number of elements and bytes per slice
		self.nelts = cutil.prod(self.shape[:-1])
		self.slicebytes = self.nelts * self.dtype().nbytes

		# Store the start of the data block
		self.fstart = self.backer.tell()


	def __del__(self):
		'''
		Close the open data file on delete.
		'''
		self.backer.close()


	def __iter__(self):
		'''
		Create a generator to read each desired slice in succession.
		Returns a tuple of the slice index and its data values.
		'''
		# Point to the first slice in the desired range
		self.setslice(0)

		# Yield each slice along the last index
		for idx in range(self.shape[-1]):
			yield (idx, self.readslice())

		return


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
		self.backer.seek(self.fstart + i * self.slicebytes)


	def readslice(self):
		'''
		Read the slice at the current file position.
		'''
		data = np.fromfile(self.backer, dtype=self.dtype, count=self.nelts)
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

		sflat.astype(self.dtype).tofile(self.backer)


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
