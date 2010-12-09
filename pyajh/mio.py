'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import numpy as np
import math
import os

def getmattype (fname, dim = None, dtype = None):
	'''
	Attempt to determine the datatype and dimension of a binary matrix file.
	The dimension and datatype may also be fixed independently to either
	identify the other, unspecified parameter or to check file validity.
	'''

	# Set the range of dimensions if one wasn't provided
	if dim is None: dimrange = [1, 3]
	else: dimrange = [dim] * 2

	# Attempt to read the header
	infile = open (fname, mode='rb')
	dimspec = np.fromfile (infile, dtype=np.int32, count=dimrange[-1])
	infile.close()

	# A dictionary describing auto-sensed data types.
	# Override this if a default datatype was provided.
	if dtype is None:
		datatypes = { np.float32().nbytes : np.float32,
				np.complex64().nbytes : np.complex64,
				np.complex128().nbytes : np.complex128 }
	else: datatypes = { dtype().nbytes : dtype }

	for dimen in range(dimrange[0], dimrange[-1] + 1):
		# Grab the file size, minus the header size
		fsize = os.stat(fname)[6] - dimen * np.int32().nbytes

		# Grab the number of bytes per record
		nbytes = fsize / np.prod(dimspec[:dimen])

		# Try next dimension if the number of records doesn't line up
		if nbytes * np.prod(dimspec[:dimen]) != fsize: continue

		# Return the record type if it exists, otherwise try again
		try:
			dtype = datatypes[nbytes]
			return (dimen, dtype)
		except KeyError: continue

	# Successful identification should have short circuited this
	raise TypeError('Could not determine data type in file.')


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

def readbmat (fname, dim = None, dtype = None, size = None):
	'''
	Read a binary, complex matrix file, auto-sensing the precision
	'''

	# The type was provided, so do the read
	infile = open (fname, mode='rb')

	# Check the validity of the file contents,
	# or auto-sense the dimension and data record length
	dim, dtype = getmattype (fname, dim, dtype)
	
	# Read the dimension specification from the file
	dimspec = np.fromfile (infile, dtype=np.int32, count=dim)

	# Override the read size with the provided size
	if size is not None: dimspec = size
	
	# The number of elements to be read
	nelts = np.prod (dimspec)

	# Read the number of elements provided and close the file
	data = np.fromfile (infile, dtype = dtype, count = nelts)
	infile.close()

	# The read failed to capture all requested elements, raise an exception
	if data.size != nelts or data is None:
		return ValueError('Failed to read requested data.')
	
	# Rework the array as a np array with the proper shape
	# Fortran order is assumed
	data = data.reshape (dimspec, order = 'F')

	return data
