#!/usr/bin/env python

import sys, os, numpy as np, getopt

from pyajh import mio, cltools, wavetools, cutil, util

def pullitems(seq, axes):
	'''
	Given a sequence seq, return a tuple consisting of items given by the
	indices in the sequence axes.
	'''
	return tuple(seq[a] for a in axes)


def rotslices(outmat, inmat, theta, lints,
		ogrid=(1.,1.,1.), igrid=(1.,1.,1.), verbose=False):
	'''
	Linearly interpolate each slice of inmat into the slices of outmat
	rotated an angle theta relative to the center of the input slice. The
	grid spacing of outmat is specified in ogrid, while the grid spacing of
	inmat is specified in igrid.

	A list of instances of cltools.InterpolatingRotator must be provided in
	lints to perform the rotation and interpolation. Slices of outmat and
	inmat will be approximately evenly divided among the instances.
	'''
	if len(inmat) != len(outmat):
		raise ValueError('Number of slices of input and output must agree')

	# Make BufferedSlices objects to read input and write output
	nbuf = 10
	srcs = [cltools.BufferedSlices(inmat, nbuf, context=lint.context) for lint in lints]
	dsts = [cltools.BufferedSlices(outmat, nbuf, read=False, context=lint.context) for lint in lints]

	# Divide workload of rotators, start buffer processing and configure slices
	nlints = len(lints)
	nslice = len(inmat)
	for i, (src, dst, lint) in enumerate(zip(srcs, dsts, lints)):
		# Set the slice shapes
		lint.setshapes(outmat.sliceshape, inmat.sliceshape)
		# Alternate slices between each interpolator
		src.setiter(range(i, nslice, nlints))
		dst.setiter(range(i, nslice, nlints))
		# Start the slicer buffering
		src.start()
		dst.start()

	# Compute the grid spacings for the input and output slices
	dgrid = pullitems(ogrid, inmat.axes[:-1])
	sgrid = pullitems(igrid, outmat.axes[:-1])

	if verbose:
		# Make a progress bar for display if the user desires
		bar = util.ProgressBar([0, nslice], width=50)
		# Reset and print the progress bar
		bar.reset()
		util.printflush(str(bar) + '\r')

	# Interpolate each of the slices successively
	# This aligns with the slicing previously configured
	for idx in range(0, nslice, nlints):
		# Figure out the workload of the current iteration
		# Take into account remainder slices at the end of the iteration
		wload = min(nlints, nslice - idx)
		# Limit this loop to the workload size
		for src, dst, lint, g in zip(srcs, dsts, lints, range(wload)):
			s, d = src.getslice(), dst.getslice()
			# Interpolate the slice, grabbing the result and the copy event
			res, evt = lint.interpolate(s, d, theta=theta, sgrid=sgrid, dgrid=dgrid)
			# Advance the slice buffers
			src.nextslice()
			dst.nextslice(evt)
			
			# Increment and print the progress bar
			if verbose:
				bar.increment()
				util.printflush(str(bar) + '\r')

	if verbose: print

	# Ensure that all output has finished
	for dst in dsts:
		dst.flush()
		dst.kill()


def rectbounds(n, h, theta, phi, hd=None):
	'''
	For a rectangular grid of size n = (nx, ny) with spacing h = (hx, hy)
	or h a scalar, isotropic spacing, find the size of a
	grid rotated by an angle theta (with respect to the center of the
	original grid) necessary to contain the original grid.

	If hd is not None, it represents the spacing of the rotated grid.
	Otherwise, hd is assumed to be equal to h.

	Sizes are constrained to be multiples of 10.
	'''
	from itertools import product

	# Perform two Givens rotations, azimuthal first
	# The polar rotation is done around the y axis
	def rotator(c, theta, phi):
		return cutil.givens(cutil.givens(c, phi), theta, axes=(2, 0))

	# Figure all of the rotated corner coordinates
	coords = [rotator(wavetools.gridtocrd(c, n, h), theta, phi)
			for c in product(*[[0, nv - 1] for nv in n])]
	# Find the most extreme coordinates in each dimension
	cmax = [max(map(abs, c)) for c in zip(*coords)]

	# Check that the rotated grid spacing dimension matches the coordinates
	if hd is None: hd = h
	cmax, hd = cutil.matchdim(cmax, hd)

	# Figure the size of the new grid, discard fractional pixels
	# Also round up to the nearest multiple of 10
	rsize = tuple(int(cutil.roundn(int(2 * cmv / hdv), 10))
			for cmv, hdv in zip(cmax, hd))
	return rsize


def usage (execname = 'regrid.py'):
	binfile = os.path.basename(execname)
	print "Usage:", binfile, "[-h] [-g g] [-v] [-d dx,dy,dz] [-s sx,sy,dz] [-n nx,ny,nz] [-r] [-t t] [-p p] <input> <output>"
	print '''
  Use linear interpolation on an OpenCL device to resample and rotate the 3-D
  matrix file input (stored in FORTRAN order) into the matrix file output. The
  OpenCL device must support image types.

  If an output grid size is not specified, a minimum grid size necessary to
  contain the input grid in the new rotated frame will be chosen.

  In forward mode, an azimuthal rotation is performed about the z axis before
  the polar rotation is performed about the rotated y axis. In reverse mode,
  the polar rotation is performed about the y axis before the azimuthal
  rotation is performed about the new z axis.

  The input and output files will have the same data type. Note that the
  interpolator always uses complex floats, so double types will lose precision.

  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -g: Use OpenCL device g on the first platform (default: first device)
  -v: Display a progress bar to track regrid status
  -d: Set the output grid spacing (default: 1,1,1)
  -s: Set the input grid spacing (default: 1,1,1)
  -n: Set the output grid dimensions (default: minimum necessary)
  -t: Perform a polar rotation of t radians about the y axis (default: 0)
  -p: Perform an azimuthal rotation of p radians about the z axis (default: 0)
  -r: Reverse the order of rotation
	'''


if __name__ == "__main__":
	# Grab the executable name
	execname = sys.argv[0]

	# Set the default device
	ctx = [0]
	# Set the default grid spacings
	dgrid, sgrid = (1., 1., 1.), (1., 1., 1.)
	# There is no default rotation
	theta, phi = 0., 0.
	# Determine the order of rotations
	reversed = False
	# The shape is determined by default
	dsize = None
	# By default, supress the progress bar
	verbose = False

	# Process optional arguments
	optlist, args = getopt.getopt(sys.argv[1:], 'hn:g:rs:d:t:p:v')
	for opt in optlist:
		if opt[0] == '-g':
			ctx = tuple(int(g) for g in opt[1].split(','))
		elif opt[0] == '-d':
			dgrid = tuple(float(d) for d in opt[1].split(','))
		elif opt[0] == '-s':
			sgrid = tuple(float(s) for s in opt[1].split(','))
		elif opt[0] == '-t':
			theta = float(opt[1])
		elif opt[0] == '-p':
			phi = float(opt[1])
		elif opt[0] == '-r':
			reversed = True
		elif opt[0] == '-n':
			dsize = tuple(int(s) for s in opt[1].split(','))
		elif opt[0] == '-v':
			verbose = True
		else:
			usage(execname)
			sys.exit(128)

	# Make sure the mandatory arguments have been specified
	if len(args) < 2:
		usage(execname)
		sys.exit(128)

	# Open the input file and determine its size
	input = mio.readbmat(args[0])
	# Make sure that the input is three-dimensional
	if len(input.shape) != 3:
		print >> sys.stderr, 'A three-dimensional input is required'
		sys.exit(128)

	# If necessary, pick a size to contain the domain
	if dsize is None: dsize = rectbounds(input.shape, sgrid, theta, phi, dgrid)

	# Check if the grid is changing or a rotation is being performed
	flt_eps = sys.float_info.epsilon
	norot = (abs(theta) < flt_eps and abs(phi) < flt_eps)
	nogrid = (dsize == input.shape and dgrid == sgrid)
	if norot and nogrid:
		print >> sys.stderr, 'Output file is just a copy of input'
		sys.exit(128)

	# Format and print the dimensions of the rotation
	def gridprint(x): return ' x '.join(str(xv) for xv in x)
	message = 'Regrid: ({:s}, {:s}) -> ({:s}, {:s})'
	print message.format(*[gridprint(x) for x in [input.shape, sgrid, dsize, dgrid]])

	# Make the linear interpolators
	# The shapes aren't important because they will be set when called
	lints = [cltools.InterpolatingRotator(input.shape[:-1], input.shape[:-1], g) for g in ctx]

	# The intermediate grid spacing is the minimum of input and output
	igrid = map(min, sgrid, dgrid)

	# Determine parameters for the first rotation
	if reversed:
		# The y grid spacing must equal the source
		igrid[1] = sgrid[1]
		# The z grid spacing must be as desired
		igrid[2] = dgrid[2]
		# Find the size of the grid necessary to accomodate the y rotation
		isize = list(rectbounds(input.shape, sgrid, theta, 0., igrid))
		# Ensure the number of slices does not change
		isize[1] = input.shape[1]
		# Clip the z axis to the desired range
		isize[2] = dsize[2]
		# Note the slicing axis and the rotation angle
		slax, angle = 1, theta
	else:
		# The z grid spacing must equal the source
		igrid[2] = sgrid[2]
		# The y grid spacing must be as desired
		igrid[1] = dgrid[1]
		# Find the size of the grid necessary to accomodate the z rotation
		isize = list(rectbounds(input.shape, sgrid, 0., phi, igrid))
		# Ensure the number of slices does not change
		isize[2] = input.shape[2]
		# Clip the y axis to the desired range
		isize[1] = dsize[1]
		# Note the slicing axis and the rotation angle
		slax, angle = 2, phi

	# Convert the size and spacing to tuples
	igrid = tuple(igrid)
	isize = tuple(isize)

	# Create an intermediate output array in memory
	intermed = np.empty(isize, input.dtype, order='F')
	# Create the slicer objects
	inslicer = mio.CoordinateShifter(input, axis=slax)
	outslicer = mio.CoordinateShifter(intermed, axis=slax)

	if verbose:
		message = 'First rotation to grid {:s}, spacing {:s}'
		print message.format(gridprint(isize), gridprint(igrid))

	# Perform the first slicewise rotation
	rotslices(outslicer, inslicer, angle, lints, igrid, sgrid, verbose)

	# The input file can be closed
	del input

	# Choose the axis and angle for the second rotation
	if reversed: slax, angle = 2, phi
	else: slax, angle = 1, theta

	# Create the slicer for the intermediate matrix
	inslicer = mio.CoordinateShifter(intermed, axis=slax)
	# Create or truncate the output file and its slicer
	output = mio.writemmap(args[1], dsize, inslicer.dtype)
	outslicer = mio.CoordinateShifter(output, axis=slax)

	if verbose:
		message = 'Second rotation to grid {:s}, spacing {:s}'
		print message.format(gridprint(dsize), gridprint(dgrid))

	# Perform the final slicewise rotation
	rotslices(outslicer, inslicer, angle, lints, dgrid, igrid, verbose)
