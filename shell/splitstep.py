#!/usr/bin/env python

import numpy as np, math, sys, getopt, os
from pyajh import mio, wavetools

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-a <a,t>] [-f f] [-h h] [-c c] [-w] [-d x,y,z,w]', '<src> <infile> <outfmt>'
	print '''
	Using the split-step method, compute the field induced in a contrast
	medium specified in infile by a point source at location src = x,y,z.
	The coordinate origin is at the center of the contrast file.
	Propagation happens downward; the topmost slab (the last in the file)
	first receives the incident field. A configurable attenuation profile
	is applied to each edge of each slice.

	Each slab is written to a file whose name is given by outfmt % d, where
	outfmt is a Python format string and d is the slab index.

	OPTIONAL ARGUMENTS:
	-a: Use a maximum attenuation a, increased over t cells (default: 7.0, 10)
	-f: Specify the incident frequency, f, in MHz (default: 3.0)
	-h: Specify the grid spacing, h, in mm (default: 0.05)
	-c: Specify the sound speed, c, in mm/us (default: 1.5)
	-w: Disable wide-angle corrections
	-d: Specify a directivity axis x,y,z with width parameter w (default: none)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Store the default parameters
	a, h, f, k0, d, w = (7.0, 10), 0.05, 3.0, 2 * math.pi, None, True

	optlist, args = getopt.getopt(sys.argv[1:], 'wa:f:h:c:d:')

	for opt in optlist:
		if opt[0] == '-a':
			av = opt[1].split(',')
			a = float(av[0]), int(av[1])
		elif opt[0] == '-d': d = [float(ds) for ds in opt[1].split(',')]
		elif opt[0] == '-f': f = float(opt[1])
		elif opt[0] == '-h': h = float(opt[1])
		elif opt[0] == '-c': c = float(opt[1])
		elif opt[0] == '-w': w = False
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Compute the step size in wavelengths
	h *= f / c

	print 'Split-step simulation, frequency %g MHz, background %g mm/us' % (f, c)
	print 'Step size in wavelengths is %g, attenuation is %g over %d pixels' % (h, a[0], a[1])

	# Grab the source location in wavelengths
	src = tuple(float(s) * f / c for s in args[0].split(','))

	# Create a generator to read the matrix slab by slab
	inmat = mio.ReadSlicer(args[1])
	objdim = inmat.matsize

	outfmt = args[2]

	# Pad the domain with the attenuation borders
	grid = [m + 2 * a[1] for m in objdim[:-1]]
	sse = wavetools.SplitStepEngine(k0, grid[0], grid[1], h)

	# Create a slice tuple to strip out the padding when writing
	sl = [slice(a[1], -a[1]) for i in range(2)]

	# Create a propagator for an isotropic step size
	sse.propagator = sse.h
	# Create the attenuation screen
	sse.attenuator = a

	# Compute the z-offset of the slab before the first computed slab
	zoff = 0.5 * float(objdim[-1] + 1) * sse.h
	# Compute the x, y (array) coordinates of the start slab
	crd = sse.slicecoords() + [zoff]
	# Compute and write the values of the Green's function in this slab
	r = np.sqrt(reduce(np.add, map(lambda (x, y): (x - y)**2, zip(crd, src))))
	fld = np.exp(1j * k0 * r) / (4. * math.pi * r)
	# Include a directivity pattern if desired
	if d is not None: fld *= wavetools.directivity(crd, src, d, 4.3458)
	mio.writebmat(fld[sl].astype(inmat.dtype), outfmt % objdim[-1])

	# Create a buffer to store the current, padded contrast
	obj = np.zeros(grid, inmat.dtype)

	# Loop through all slices and compute the propagated field
	for idx in range(objdim[-1] - 1, -1, -1):
		obj[sl] = inmat.getslice(idx)
		fld = sse.advance(fld, obj, w)
		mio.writebmat(fld[sl].astype(inmat.dtype), outfmt % idx)
