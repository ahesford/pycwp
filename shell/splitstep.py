#!/usr/bin/env python

import numpy as np, math, sys, getopt, os
from tempfile import TemporaryFile
from pyajh import mio, wavetools, util, cutil
from pyajh.opencl import wavecl
from pyajh.f2py import splitstep

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-a a] [-g g] [-f f] [-s s] [-c c] [-i i] [-t t] [-p nx,ny] [-d x,y,z,w]', '<src> <infile> <outfile>'
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
  -i: Specify the number of relaxation updates (default: 1)
  -t: Specify the parameter for relaxation updates (default: 2)
  -p: Pad the domain to [nx,ny] pixels for attenuation (default: domain plus Hann window)
  -d: Specify a directivity axis x,y,z with width parameter w (default: none)
  -g: Use OpenCL computing device g on the first platform (default: system default device)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Store the default parameters
	a, c, s, f, k0, steps, tau = 50, 1.5, 0.05, 3.0, 2 * math.pi, 1, 2.
	d, p, ctx = [None]*3

	optlist, args = getopt.getopt(sys.argv[1:], 'ha:f:s:c:d:p:g:i:t:')

	for opt in optlist:
		if opt[0] == '-a': a = int(opt[1])
		elif opt[0] == '-d': d = [float(ds) for ds in opt[1].split(',')]
		elif opt[0] == '-p': p = [int(ps) for ps in opt[1].split(',')]
		elif opt[0] == '-f': f = float(opt[1])
		elif opt[0] == '-s': s = float(opt[1])
		elif opt[0] == '-c': c = float(opt[1])
		elif opt[0] == '-g': ctx = int(opt[1])
		elif opt[0] == '-i': steps = int(opt[1])
		elif opt[0] == '-t': tau = float(opt[1])
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

	util.printflush('Creating split-step engine... ')
	if ctx is not None:
		sse = wavecl.SplitStep(k0, p[0], p[1], h, src=src, d=d, l=a, context=ctx)
	else:
		sse = wavetools.SplitStep(k0, p[0], p[1], h, l=a)
		# Set the incident field generator
		def srcfld(obs):
			r = np.sqrt(reduce(np.add, ((s - o)**2 for s, o in zip(src, obs))))
			ptsrc = wavetools.green3d(k0, r)
			if d is None: return ptsrc
			else: return ptsrc * wavetools.directivity(obs, src, d[:-1], d[-1])
		sse.setincgen(srcfld)
		# Initialize the underlying FFTW library to use threads
		print 'Using %d threads for calculation' % splitstep.initfft()

	print 'finished'

	# Create a slice tuple to strip out the padding when writing
	lpad = [(pv - gv) / 2 for pv, gv in zip(p, inmat.shape)]
	sl = [slice(lv, -(pv - gv - lv)) for pv, gv, lv in zip(p, inmat.shape, lpad)]

	# Compute the z height of a specified slab
	zoff = lambda i: sse.h * (float(i) - 0.5 * float(inmat.shape[-1] - 1.))

	# Augment the grid with the third (extended) dimension
	# An extra slice is required to turn around the field
	p += [inmat.shape[-1] + 1]

	# This buffer will store the average contrast value on an expanded grid
	obj = np.zeros(p[:2], inmat.dtype, order='F')

	# Create a progress bar to display computation progress
	bar = util.ProgressBar([0, p[-1]], width=50)

	# Open temporary files to store the forward and backward fields 
	fmat = mio.Slicer(TemporaryFile(), p, inmat.dtype, True)
	bmat = mio.Slicer(TemporaryFile(), p, inmat.dtype, True)

	for step in range(1, steps + 1):
		print 'Iteration %d of %d' % (step, steps)

		# Compute the initial forward-traveling field
		sse.setincident(zoff(inmat.shape[-1] + 0.5))

		# Reset and print the progress bar
		bar.reset()
		util.printflush(str(bar) + ' (forward) \r')

		# Propagate the forward field through each slice
		for idx in reversed(range(p[-1])):
			# Grab the contrast for the next slab
			try: obj[sl] = inmat[idx - 1]
			except IndexError: obj[:,:] = 0.
			# Read any existing forward and backward fields
			ffld = fmat[idx - 1]
			bfld = bmat[idx]
			# Advance and write the forward-traveling field
			sse.advance(obj, bfld, ffld, tau if step > 1 else 2.)
			ffld = sse.copyfield()
			fmat[idx - 1] = ffld
			# Increment and print the progress bar
			bar.increment()
			util.printflush(str(bar) + ' (forward) \r')

		# Reset progress bar and propagating field
		sse.reset()
		bar.reset()
		util.printflush(str(bar) + ' (backward)\r')

		for idx in range(p[-1]):
			# Grab the contrast for the next slab
			try: obj[sl] = inmat[idx]
			except IndexError: obj[:,:] = 0.
			# Read any existing forward and backward fields
			ffld = fmat[idx - 1]
			bfld = bmat[idx]
			# Advance and write the backward-traveling field
			sse.advance(obj, ffld, bfld, tau if step > 1 else 2.)
			bfld = sse.copyfield()
			bmat[idx] = bfld
			# Increment and print the progress bar
			bar.increment()
			util.printflush(str(bar) + ' (backward)\r')

		print

	# Create a combined output file; errors will truncate the file
	with mio.Slicer(args[2], inmat.shape, inmat.dtype, True) as outmat:
		bar = util.ProgressBar([0, inmat.shape[-1]], width=50)
		sse.reset()
		# Cut the step size in half and initialize the slab contrast
		sse.dz *= 0.5
		obj[sl] = inmat[-1]
		sse.etaupdate(obj)
		print 'Combining forward and backward fields'
		util.printflush(str(bar) + '\r')
		for idx in reversed(range(inmat.shape[-1])):
			# Combine the forward and reversed fields in one slice
			ffld = fmat[idx]
			bfld = bmat[idx]
			sse.copyfield(ffld + bfld)
			# Grab the next slab contrast
			try: obj[sl] = inmat[idx - 1]
			except IndexError: obj[:,:] = 0.
			# Propagate the field to the midpoint
			sse.advance(obj)
			outmat[idx] = sse.copyfield()[sl]
			bar.increment()
			util.printflush(str(bar) + '\r')

		print
