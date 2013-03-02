#!/usr/bin/env python

import numpy as np, math, sys, getopt, os
from tempfile import TemporaryFile
from pyajh import mio, wavetools, util, cutil
from pyajh.opencl import SplitStep, BufferedSlices

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-q q] [-p p] [-a a] [-g g] [-f f] [-s s] [-c c] [-e nx,ny] [-w w] [-d x,y,z,w]', '<src> <infile> <outfile>'
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

  Computations happen in three passes: a forward pass through the medium, a
  backward pass to pick up backward scatter, and a third pass (computed along
  with the second pass) to shift the fields in each slice to the center for
  comparison to reference methods.
  
  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -q: Use high-order spatial terms in the first q passes (default: 0)
  -p: Use high-order spectral terms in the first p passes (default: 2)
  -a: Specify the width of the Hann attenuating window at each edge (default: 100)
  -f: Specify the incident frequency, f, in MHz (default: 3.0)
  -s: Specify the grid spacing, s, in mm (default: 0.05)
  -c: Specify the sound speed, c, in mm/us (default: 1.507)
  -p: Pad the domain to [nx,ny] pixels (default: next power of two)
  -w: Specify the high-order spectral correction weight (default: 0.39)
  -d: Specify a directivity axis x,y,z with width parameter w (default: none)
  -g: Use OpenCL computing device g on the first platform (default: system default device)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Store the default parameters
	a, c, s, f, k0, w = 100, 1.507, 0.05, 3.0, 2 * math.pi, 0.39
	d, dom = [None]*2
	# Determine the number of slabs that use high-order corrections
	hospat, hospec = 0, 2

	ctx = 0

	optlist, args = getopt.getopt(sys.argv[1:], 'hq:a:f:s:c:d:p:g:w:e:')

	for opt in optlist:
		if opt[0] == '-a': a = int(opt[1])
		elif opt[0] == '-d': d = [float(ds) for ds in opt[1].split(',')]
		elif opt[0] == '-e': dom = [int(ps) for ps in opt[1].split(',')]
		elif opt[0] == '-f': f = float(opt[1])
		elif opt[0] == '-s': s = float(opt[1])
		elif opt[0] == '-c': c = float(opt[1])
		elif opt[0] == '-g': ctx = int(opt[1])
		elif opt[0] == '-w': w = float(opt[1])
		elif opt[0] == '-q': hospat = int(opt[1])
		elif opt[0] == '-p': hospec = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Compute the step size in wavelengths
	h = s * f / c

	print 'Split-step simulation, frequency %g MHz, background %g mm/us' % (f, c)
	print 'Step size in wavelengths is %g, Hann window thickness is %d' % (h, a)

	# Set up a slice-wise input reader
	inmat = mio.Slicer(args[1])
	# Automatically pad the domain, if necessary
	if dom is None: dom = [cutil.ceilpow2(g) for g in inmat.shape[:-1]]

	print 'Computing on expanded grid', dom

	# Grab the source location in wavelengths
	src = tuple(float(s) * f / c for s in args[0].split(','))

	print 'Creating split-step engine... '
	sse = SplitStep(k0, dom[0], dom[1], h, src=src, d=d, l=a, w=w, 
			propcorr=(hospec > 0, hospat > 0), context=ctx)

	# Restrict device transfers to the object grid
	sse.setroi(inmat.shape[:-1])

	# Compute the z height of a specified slab
	zoff = lambda i: sse.h * (float(i) - 0.5 * float(inmat.shape[-1] - 1.))

	# Augment the grid with the third (extended) dimension
	# An extra slice is required to turn around the field
	dom = list(inmat.shape[:-1]) + [inmat.shape[-1] + 1]

	# Create a progress bar to display computation progress
	bar = util.ProgressBar([0, dom[-1]], width=50)

	# Compute the initial forward-traveling field
	sse.setincident(zoff(inmat.shape[-1] + 0.5))

	# Reset and print the progress bar
	bar.reset()
	util.printflush(str(bar) + ' (forward) \r')

	# Create an empty array to store the forward field
	fmat = mio.Slicer(TemporaryFile(dir='.'), dom, inmat.dtype, True)

	# An empty slab is needed at the end of the medium
	zeroslab = np.zeros(inmat.shape[:-1], inmat.dtype, order='F')
	# Create buffered slice objects for the contrast and forward fields
	obj = BufferedSlices(inmat, 5, reversed=True, context=sse.context)
	obj.start()

	# The forward field buffer does not require pinned memory
	fbuf = BufferedSlices(fmat, 5, read=False, context=sse.context)
	fbuf.start()

	# Propagate the forward field through each slice
	for idx in reversed(range(dom[-1])):
		# Try to grab the next slice, or reuse the prior buffer
		# to write a bunch of zeros
		try: obval = obj.getslice()
		except IndexError: obval[:,:] = zeroslab
		# Advance and write the forward-traveling field
		sse.advance(obval)
		# Copy the device result buffer to the host queue
		resevt = sse.getresult(fbuf.getslice())
		# Advance the slice buffers
		obj.nextslice()
		# Tell the buffer to wait until the copy is finished
		fbuf.nextslice(resevt)
		# Increment and print the progress bar
		bar.increment()
		util.printflush(str(bar) + ' (forward) \r')

	# Recreate the object buffer for backward propagation
	obj.kill()
	obj = BufferedSlices(inmat, 5, context=sse.context)
	obj.start()

	# Flush the forward field buffer to ensure consistency
	fbuf.flush()
	fbuf.kill()
	# Now the forward field needs pinned memory for fast transfers
	fbuf = BufferedSlices(fmat, 5, reversed=True, context=sse.context)
	fbuf.start()

	# Reset the split-step state and change the default corrective terms
	sse.reset((hospec > 1, hospat > 1))
	# Reset and reprint the progress bar
	bar.reset()
	util.printflush(str(bar) + ' (backward)\r')

	# Allocate a memory-mapped output array and an associated buffer
	outmat = mio.Slicer(args[2], inmat.shape, inmat.dtype, True)
	obuf = BufferedSlices(outmat, 5, read=False, context=sse.context)
	obuf.start()

	# Determine the propagator terms to include in the shifting pass
	shcorr = (hospec > 2, hospat > 2)

	# Propagate the reverse field through each slice
	for idx in range(dom[-1]):
		# Try to grab the next slice, or reuse the prior buffer
		try: obval = obj.getslice()
		except IndexError: obval[:,:] = zeroslab
		# Advance the backward traveling field
		# This requires the forward field in the slab
		# Also compute a half-shifted, combined field in the result buffer
		sse.advance(obval, fbuf.getslice(), True, shcorr=shcorr)
		# Copy the device result buffer to the host queue
		# The first slab is outside the domain and is not recorded
		if idx > 0:
			resevt = sse.getresult(obuf.getslice())
			# Tell the buffer to wait until the copy is finished
			obuf.nextslice(resevt)
		# Advance the slice buffers
		obj.nextslice()
		fbuf.nextslice()
		# Increment and print the progress bar
		bar.increment()
		util.printflush(str(bar) + ' (backward)\r')

	print

	# Kill the I/O buffers
	obj.kill()
	fbuf.kill()

	# Flush and kill the output buffer
	obuf.flush()
	obuf.kill()

	# Truncate the file to kill pending disk flushes
	fmat.clobber()
