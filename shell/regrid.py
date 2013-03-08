#!/usr/bin/env python

import sys, os, numpy as np, getopt

from pyajh import mio, cltools

def usage (execname = 'gridup.py'):
	binfile = os.path.basename(execname)
	print "Usage:", binfile, "[-h] [-g g] Nx,Ny <input> <output>"
	print '''
  Use linear interpolation on an OpenCL device to resample each x-y slice in
  the 3-D matrix file input (stored in FORTRAN order) to an (Nx,Ny) grid. The
  number of slices is unchanged. The OpenCL device must support image types.

  The input and output files will have the same data type. Note that the
  interpolator always uses complex floats, do double types will lose precision.

  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -g: Use OpenCL computing device g on the first platform (default: first device)
	'''

if __name__ == "__main__":
	# Grab the executable name
	execname = sys.argv[0]

	# Set the default device
	ctx = 0

	# Process optional arguments
	optlist, args = getopt.getopt(sys.argv[1:], 'hg:')
	for opt in optlist:
		if opt[0] == '-g': ctx = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	# Make sure the mandatory arguments have been specified
	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Grab the source and destination slice shapes
	dstshape = tuple(int(s) for s in args[0].split(','))

	# Open the input file and determine its size
	input = mio.Slicer(args[1])
	# Make sure that the input is three-dimensional
	if len(input.shape) != 3:
		raise ValueError('A three-dimensional input is required')

	# Create or truncate the output file
	outsize = tuple(list(dstshape) + [input.shape[-1]])
	output = mio.Slicer(args[2], outsize, input.dtype, True)

	# Make the linear interpolator
	lint = cltools.LinearInterpolator(dstshape, input.shape[:-1], ctx)

	# Interpolate each of the slices successively
	for idx, slab in input:
		output[idx] = lint.interpolate(slab)
