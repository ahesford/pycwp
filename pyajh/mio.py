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
		np.array(mat.shape, dtype='int32').tofile(outfile)

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


class ReadSlicer(object):
	'''
	This class opens a data file that can be read one slice at a time. A
	slice is defined as a chunk of data with a dimension one less than the
	input, i.e., one column of a FORTRAN-ordered matrix or one slab of a
	FORTRAN-ordered three-dimensional grid.
	'''

	def __init__(self, infile, dim = None, dtype = None, slices = None):
		'''
		Open the input file, possibly with the specified dimension and
		data type (to avoid automatic detection), and prepare to read
		the file one slice at a time. The parameter infile may be a
		string containing the path of a file to open, or it may be an
		already-open file. The optional two-element list slices
		specifies the first and last indices (inclusive) of the slices
		to be read.
		'''

		# Open the input file if it isn't already open
		if isinstance(infile, (str, unicode)):
			infile = open(infile, mode='rb')

		# Store the file
		self.infile = infile

		# Grab the matrix header, size, and data type
		self.shape, self.dtype = getmattype(infile, dim, dtype)

		# The number of elements to read per slice
		self.nelts = cutil.prod(self.shape[:-1])

		# The number of bytes per slice
		self.slicebytes = self.nelts * self.dtype().nbytes

		# Store the start of the data block
		self.fstart = self.infile.tell()

		# If a subset of slices are desired, restrict the read
		if slices is not None:
			if len(slices) != 2:
				raise ValueError('Slices list must contain two elements.')
			if (0 > slices[0] >= self.shape[-1]) or (0 > slices[1] >= self.shape[-1]):
				raise ValueError('Slice indices outside of valid range.')
			# Copy the slice range
			self.slices = slices[:]
			# Move the starting position to the first desired slice
			self.fstart += self.slices[0] * self.slicebytes
		else: self.slices = [0, self.shape[-1] - 1]

		# Store the total number of slices to be read
		self.nslices = self.slices[1] - self.slices[0] + 1


	def __del__(self):
		'''
		Close the open data file on delete.
		'''
		self.infile.close()


	def __iter__(self):
		'''
		Create a generator to read each desired slice in succession.
		Returns a tuple of the slice index and its data values.
		'''
		# Point to the first slice in the desired range
		self.setslice(0)

		# Yield each slice along the last index
		for idx in range(self.slices[0], self.slices[1] + 1):
			yield (idx, self.readslice())

		return


	def readslice(self):
		'''
		Read the slice at the current file position and reshape it to
		the slice dimensions.
		'''
		data = np.fromfile(self.infile, dtype=self.dtype, count=self.nelts)
		if data.size != self.nelts or data is None:
			raise ValueError('Failed to read current slice')
		return data.reshape(self.shape[:-1], order='F')


	def setslice(self, i):
		'''
		Point the file to the start of slice i, relative to the
		starting index.
		'''
		# Ensure the requested index is valid
		if -self.nslices > i or i >= self.nslices:
			raise IndexError('Requested slice is out of bounds')

		# Wrap negative indices in the Python fashion
		if i < 0: i = i + self.nslices

		# Point to the start of the desired slice
		self.infile.seek(self.fstart + i * self.slicebytes)


	def __getitem__(self, key):
		'''
		Grab slices from the data file using list-style indexing.
		'''
		try:
			# Treat the key as a slice to pull out multiple slabs
			idx = key.indices(self.nslices)
			# Compute the number of slabs to be read
			nslab = (idx[1] - idx[0]) / idx[2]
			if idx[0] + idx[2] * nslab < idx[1]: nslab += 1
			# Allocate storage for the requested slices
			shape = list(self.shape[:-1]) + [nslab]
			data = np.empty(shape, dtype=self.dtype)

			# Read all of the requested slices
			for li, i in enumerate(range(*idx)):
				self.setslice(i)
				data[:,:,li] = self.readslice()
		except AttributeError:
			# A slice was not provided, grab a single slab
			self.setslice(key)
			data = self.readslice()

		return data
