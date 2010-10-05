'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import numpy
import math
import os

def writebmat (mat, fname):
	'''
	Write a binary matrix file in the provided precision.
	'''
	outfile = open (fname, mode='wb')

	# Pull the shape for writing
	mshape = numpy.array (mat.shape, dtype='int32')

	# Write the size header
	mshape.tofile (outfile)
	# Write the matrix body in FORTRAN order
	mat.flatten('F').tofile (outfile)
	outfile.close ()

def readbmat (fname, dimen = 2, dtype = None, size = None):
	'''
	Read a binary, complex matrix file, auto-sensing the precision
	'''

	# The type was provided, so do the read
	infile = open (fname, mode='rb')
	
	# Read the dimension specification from the file
	dimspec = numpy.fromfile (infile, dtype=numpy.int32, count=dimen)

	# A dictionary describing auto-sensed data types.
	datatypes = { numpy.float32().nbytes : numpy.float32,
			numpy.complex64().nbytes : numpy.complex64, 
			numpy.complex128().nbytes : numpy.complex128 }

	# Attempt to determine the data type.
	if dtype is None:
		# Grab the file size, minus the header size
		fsize = os.stat(fname)[6] - 4 * dimen
		# Grab the number of bytes per record
		nbytes = fsize / numpy.prod(dimspec)
		# Make sure the sizes match up
		if nbytes * numpy.prod(dimspec) != fsize:
			raise TypeError('Could not determine data type in file.')
		# Grab the requested data type or fail
		try:
			dtype = datatypes[nbytes]
		except KeyError:
			raise TypeError('Could not determine data type in file.')

	# Override the read size with the provided size
	if size is not None: dimspec = size
	
	# The number of elements to be read
	nelts = numpy.prod (dimspec)

	# Read the number of elements provided and close the file
	data = numpy.fromfile (infile, dtype = dtype, count = nelts)
	infile.close()

	# The read failed to capture all requested elements, raise an exception
	if data.size != nelts or data is None:
		return ValueError('Failed to read requested data.')
	
	# Rework the array as a numpy array with the proper shape
	# Fortran order is assumed
	data = data.reshape (dimspec, order = 'F')

	return data
