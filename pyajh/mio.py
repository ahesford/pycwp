'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import numpy
import math

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

def readbmat (fname, dimen = 2, type = None, size = None):
	'''
	Read a binary, complex matrix file, auto-sensing the precision
	'''
	# If the type was not provided, try a sequence of values
	if type is None:
		# The default list of types to try
		typelist = (numpy.complex128, numpy.complex64,
				numpy.float64, numpy.float32)
		# Try each type and short-circuit if the correct one is found
		for tp in typelist:
			data = readbmat(fname, dimen, tp, size)
			if data is not None: return data

	# The type was provided, so do the read
	infile = open (fname, mode='rb')
	
	# Read the dimension specification from the file
	dimspec = numpy.fromfile (infile, dtype=numpy.int32, count=dimen)
	# Override the read size with the provided size
	if size is not None: dimspec = size
	
	# The number of elements to be read
	nelts = numpy.prod (dimspec)

	# Read the number of elements provided and close the file
	data = numpy.fromfile (infile, dtype = type, count = nelts)
	infile.close()

	# The read failed to capture all requested elements, return nothing
	if data.size != nelts or data is None: return None
	
	# Rework the array as a numpy array with the proper shape
	# Fortran order is assumed
	data = data.reshape (dimspec, order = 'F')

	return data
