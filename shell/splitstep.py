#!/usr/bin/env python

import numpy as np, math, sys, getopt, os
from pyajh import mio, wavetools, wavecl, util, cutil

def printflush(string):
	'''
	Print a string, without a newline, and flush stdout.
	'''
	print string,
	sys.stdout.flush()


def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-a a] [-g g] [-f f] [-s s] [-c c] [-p nx,ny] [-d x,y,z,w]', '<src> <infile> <outfile>'
	print '''
  Using the split-step method, compute the field induced in a contrast
  medium specified in infile by a point source at location src = x,y,z.
  The coordinate origin is at the center of the contrast file.
  Propagation happens downward; the topmost slab (the last in the file)
  first receives the incident field. A configurable attenuation profile
  is applied to each edge of each slice.
  
  The solution is written to outfile.

  The computational domain is padded to the next larger power of 2 for
  artificial attenuation.
  
  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -a: Specify the width of the Hann attenuating window at each edge (default: 50)
  -f: Specify the incident frequency, f, in MHz (default: 3.0)
  -s: Specify the grid spacing, s, in mm (default: 0.05)
  -c: Specify the sound speed, c, in mm/us (default: 1.5)
  -p: Pad the domain to [nx,ny] pixels for attenuation (default: domain plus Hann window)
  -d: Specify a directivity axis x,y,z with width parameter w (default: none)
  -g: Use OpenCL computing device g on the first platform (default: system default device)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Store the default parameters
	a, c, s, f, k0, = 50, 1.5, 0.05, 3.0, 2 * math.pi
	d, p, ctx = [None]*3

	optlist, args = getopt.getopt(sys.argv[1:], 'ha:f:s:c:d:p:g:')

	for opt in optlist:
		if opt[0] == '-a': a = int(opt[1])
		elif opt[0] == '-d': d = [float(ds) for ds in opt[1].split(',')]
		elif opt[0] == '-p': p = [int(ps) for ps in opt[1].split(',')]
		elif opt[0] == '-f': f = float(opt[1])
		elif opt[0] == '-s': s = float(opt[1])
		elif opt[0] == '-c': c = float(opt[1])
		elif opt[0] == '-g': ctx = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Compute the step size in wavelengths
	h = s * f / c

	print 'Split-step simulation, frequency %g MHz, background %g mm/us' % (f, c)
	print 'Step size in wavelengths is %g, maximum attenuation is %g' % (h, a)

	# Set up a slice-wise input reader
	inmat = mio.Slicer(args[1])
	# Automatically pad the domain, if necessary
	if p is None: p = [cutil.ceilpow2(g) for g in inmat.shape[:-1]]

	print 'Computing on expanded grid', p

	# Grab the source location in wavelengths
	src = tuple(float(s) * f / c for s in args[0].split(','))

	printflush('Creating split-step engine... ')
	sse = wavecl.SplitStep(k0, p[0], p[1], h, a, context = ctx)
	print 'finished'

	# Create a slice tuple to strip out the padding when writing
	lpad = [(pv - gv) / 2 for pv, gv in zip(p, inmat.shape)]
	sl = [slice(lv, -(pv - gv - lv)) for pv, gv, lv in zip(p, inmat.shape, lpad)]

	printflush('Computing incident field... ')
	# Compute the z-offset of the slab before the first computed slab
	zoff = 0.5 * float(inmat.shape[-1] + 1) * sse.h
	# Compute the x, y (array) coordinates of the start slab
	crd = sse.slicecoords() + [zoff]
	# Compute and write the values of the Green's function in this slab
	r = np.sqrt(reduce(np.add, map(lambda (x, y): (x - y)**2, zip(crd, src))))
	fld = np.exp(1j * k0 * r) / (4. * math.pi * r)
	# Include any specified directivity pattern
	if d is not None: fld *= wavetools.directivity(crd, src, d[:3], d[3])
	sse.setfield(fld)
	print 'finished'

	# This buffer will store the average contrast value on an expanded grid
	obj = np.zeros(p, inmat.dtype)

	print 'Stepping through slabs...'
	# Create a progress bar and print a blank
	bar = util.ProgressBar([0, inmat.shape[-1]], width=50)
	bar.makebar()
	printflush(str(bar) + '\r')

	# Set up a slice-wise output writer, clobbering any existing file
	outmat = mio.Slicer(args[2], inmat.shape, inmat.dtype, True)

	try:
		# Loop through all slices and compute the propagated field
		for idx in reversed(range(inmat.shape[-1])):
			# Read the current contrast slab into the buffer
			obj[sl] = inmat[idx]
			# Advance and write the field
			sse.advance(obj)
			outmat[idx] = sse.getfield()[sl]
			# Increment and print the progress bar
			bar.increment()
			printflush(str(bar) + '\r')

		print
	except:
		# Truncate the output file to quickly end
		outmat.backer.truncate(0)
