#!/usr/bin/env python

import sys, os, numpy as np, getopt

from pyajh import mio, cltools

def usage (execname = 'regrid.py'):
	binfile = os.path.basename(execname)
	print "Usage:", binfile, "[-h] [-g g] [-d dx,dy] [-s sx,sy] [-r r] Nx,Ny <input> <output>"
	print '''
  Use linear interpolation on an OpenCL device to resample each x-y slice in
  the 3-D matrix file input (stored in FORTRAN order) to an (Nx,Ny) grid. The
  number of slices is unchanged. The OpenCL device must support image types.

  The input and output files will have the same data type. Note that the
  interpolator always uses complex floats, do double types will lose precision.

  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -g: Use OpenCL device g on the first platform (default: first device)
  -d: Set the output grid spacing (default: 1,1)
  -s: Set the input grid spacing (default: 1,1)
  -r: Rotate output grid r radians about the center of the input (default: 0)
	'''

if __name__ == "__main__":
	# Grab the executable name
	execname = sys.argv[0]

	# Set the default device
	ctx = 0
	# Set the default grid spacings
	dgrid, sgrid = (1., 1.), (1., 1.)
	# Set the default rotation
	theta = 0.

	# Process optional arguments
	optlist, args = getopt.getopt(sys.argv[1:], 'hg:r:s:d:')
	for opt in optlist:
		if opt[0] == '-g': ctx = int(opt[1])
		elif opt[0] == '-d':
			dgrid = tuple(float(d) for d in opt[1].split(','))
		elif opt[0] == '-s':
			dgrid = tuple(float(s) for s in opt[1].split(','))
		elif opt[0] == '-r':
			theta = float(opt[1])
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

	# Make the linear interpolator
	lint = cltools.InterpolatingRotator(dstshape, input.shape[:-1], ctx)

	# Create or truncate the output file
	outsize = tuple(list(dstshape) + [input.shape[-1]])
	output = mio.Slicer(args[2], outsize, input.dtype, True)

	# Make BufferedSlices objects to read input and write output
	src = cltools.BufferedSlices(input, 5, context=lint.context)
	src.start()
	dst = cltools.BufferedSlices(output, 5, read=False, context=lint.context)
	dst.start()

	# Interpolate each of the slices successively
	for idx in range(len(input)):
		s = src.getslice()
		d = dst.getslice()
		# Interpolate the slice, grabbing the result and the copy event
		res, evt = lint.interpolate(s, d, theta=theta, sgrid=sgrid, dgrid=dgrid)
		# Advance the slice buffers
		src.nextslice()
		dst.nextslice(evt)

	dst.flush()
	dst.kill()
