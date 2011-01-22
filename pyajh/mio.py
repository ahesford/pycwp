'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import numpy as np
import math
import os

def getmattype (infile, dim = None, dtype = None):
	'''
	Read a header from the provided array data file to determine the size
	and data type of the array therein. The file will be left pointing to
	the data beyond the header. The data type and dimension may be provided
	to avoid auto detection.
	'''

	# The maximum allowed dimension
	maxdim = max(dim, 3)

	# Record the size of the file
	fsize = os.stat(infile.name)[6]

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
		

def writebmat (mat, fname):
	'''
	Write a binary matrix file in the provided precision.
	'''
	outfile = open (fname, mode='wb')

	# Pull the shape for writing
	mshape = np.array (mat.shape, dtype='int32')

	# Write the size header
	mshape.tofile (outfile)
	# Write the matrix body in FORTRAN order
	mat.flatten('F').tofile (outfile)
	outfile.close ()


def readbmat (fname, dim = None, dtype = None):
	'''
	Memory map a binary, complex matrix file, auto-sensing the precision
	and dimension. The dimension and precision can be manually specified if
	desired. The map is read-only and must be copied to modify.
	'''

	# Open the file
	infile = open (fname, mode='rb')

	# Read the matrix header and determine the data type
	matsize, dtype = getmattype (infile, dim, dtype)

	# Create the read-only memory map and close the source file
	datamap = np.memmap(infile, offset=infile.tell(), dtype=dtype, mode='r')
	infile.close()

	# Reshape the map in FORTRAN order
	return datamap.reshape (matsize, order = 'F')


def readslicer (fname, dim = None, dtype = None, slices = None):
	'''
	A generator that will read and return each slice of a file one-by-one.
	An optional (inclusive) range limits the slices read.
	'''

	# Open the file
	infile = open(fname, mode='rb')

	# Read the matrix header and determine the data type
	matsize, dtype = getmattype(infile, dim, dtype)

	# The number of elements to read per slice
	nelts = np.prod(matsize[:-1])

	if slices is not None:
		# Seek to the desired starting slice
		infile.seek(nelts * slices[0] * dtype().nbytes, 1)
		# Set the number of slices to read
		slrange = range(slices[0], slices[1] + 1)
	else: slrange = range(matsize[-1])

	# Slice along the last index
	for idx in slrange:
		data = np.fromfile(infile, dtype = dtype, count = nelts)
		if data.size != nelts or data is None:
			raise ValueError('Failed to read requested data.')
		yield (idx, data.reshape(matsize[:-1], order = 'F'))

	# Close the file and return
	infile.close()
	return
