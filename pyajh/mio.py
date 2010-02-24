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
	infile = open (fname, mode='rb')

	# Read the dimension specification from the file
	dimspec = numpy.fromfile (infile, dtype=numpy.int32, count=dimen)
	if size is not None: dimspec = size

	# Conveniently precompute the number of elements to read
	nelts = numpy.prod (dimspec)

	# Store the location of the start of the data
	floc = infile.tell()

	# If the type was provided, don't try to auto-sense
	if type is not None:
		data = numpy.fromfile (infile, dtype = type, count = -1)
		if data.size != nelts: data = None;
	else:
		# Try complex doubles
		data = numpy.fromfile (infile, dtype = numpy.complex128, count = -1)
		
		# If the read didn't pick up enough elements, try complex float
		if data.size != nelts:
			# Back to the start of the data
			infile.seek (floc)
			data = numpy.fromfile (infile, dtype = numpy.complex64, count = -1)
			
			# Read still failed to agree with header, return nothing
			if data.size != nelts: data = None;
			else: data = numpy.array (data, dtype = numpy.complex128)
			
	infile.close ()

	# Rework the array as a numpy array with the proper shape
	if data is not None: data = data.reshape (dimspec, order = 'F')

	return data
