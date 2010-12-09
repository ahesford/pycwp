'''
Routines for reading and writing binary matrix files in FORTRAN-order.
'''

import numpy as np
import math
import os

def getmattype (infile, dim = None, dtype = None):
	'''
	Try to read the header for the provided matrix file, returning an array
	depicting the size and the data type of the records. The file will
	point to the data beyond the header. The data type and dimension may
	be provided to avoid auto detection.
	'''

	# Read the maximum header, which has five elements for a grouped FMM map
	hdr = np.fromfile (infile, dtype=np.int32, count=5)

	# Make sure that the grouped FMM map has dimension three, if specified
	if hdr[0] == 0 and dim and dim != 3:
		raise ValueError('Matrix is a grouped FMM map but dimension is not 3.')
	elif hdr[0] == 0: dim = 3

	# Try to limit the pool of auto-sensed data types, if possible
	try: datatypes = { dtype().nbytes : dtype }
	except: datatypes = { np.float32().nbytes : np.float32,
			np.complex64().nbytes : np.complex64,
			np.complex128().nbytes : np.complex128 }

	# Set the range of dimensions to check
	if dim is None: dimrange = range(1, 4)
	else: dimrange = [dim]

	# Loop through each possible dimension, checking data types
	for dimen in dimrange:
		# Set the appropriate header size and matrix size
		if hdr[0] == 0: 
			hdrlen = 5
			matsize = np.array([hdr[4] * hv for hv in hdr[1:dimen+1]])
		else: 
			hdrlen = dimen
			matsize = hdr[:dimen]

		# Check the size of the file, minus the header
		fsize = os.stat(infile.name)[6] - hdrlen * np.int32().nbytes;

		# Grab the number of bytes per record and
		# check that the record size lines up
		nbytes = fsize / np.prod(matsize)
		if nbytes * np.prod(matsize) != fsize: continue

		# Try to grab the data type of the records
		try: 
			dtype = datatypes[nbytes]
			break
		except KeyError: continue
	else: raise TypeError('Could not determine data type in file.')

	# Seek to the end of the header if it isn't five elements long
	if hdr[0] != 0: infile.seek(len(matsize) * np.int32().nbytes)

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


def readbmat (fname, dim = None, dtype = None, size = None):
	'''
	Read a binary, complex matrix file, auto-sensing the precision
	'''

	# Open the file
	infile = open (fname, mode='rb')

	# Read the matrix header and determine the data type
	matsize, dtype = getmattype (infile, dim, dtype)
	
	# Override the read size with the provided size
	if size is not None: matsize = size
	
	# The number of elements to be read
	nelts = np.prod (matsize)

	# Read the number of elements provided and close the file
	data = np.fromfile (infile, dtype = dtype, count = nelts)
	infile.close()

	# The read failed to capture all requested elements, raise an exception
	if data.size != nelts or data is None:
		return ValueError('Failed to read requested data.')
	
	# Rework the array as a np array with the proper shape
	# Fortran order is assumed
	data = data.reshape (matsize, order = 'F')

	return data
