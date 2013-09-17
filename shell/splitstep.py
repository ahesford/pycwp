#!/usr/bin/env python

import numpy as np, math, sys, getopt, os, pyopencl as cl
from numpy.linalg import norm
from tempfile import TemporaryFile
from multiprocessing import Process

from pyajh import mio, wavetools, util, cutil
from pyajh.cltools import SplitStep, BufferedSlices, mapbuffer

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-q q] [-p p] [-a a] [-g g0,g1,...] [-v] [-f f] [-z z] [-s s] [-c c] [-e nx,ny] [-w w] [-b b] [-d x,y,z,w]', '<srcs> <infile> <outfile>'
	print '''
  Using the split-step method, compute the fields induced in a contrast
  medium specified in infile by each of a collection of point sources at
  locations srcs = x0,y0,z0,x1,y1,z1,...,xN,yN,zN. The coordinate origin is at
  the center of the contrast file. Propagation happens downward; the topmost
  slab (the last in the file) first receives the incident field. A configurable
  attenuation profile is applied to each edge of each slice.

  Each solution is written to outfile with a unique identifier appended.

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
  -z: Specify the propagation step, z, in mm (default: transverse grid spacing)
  -s: Specify the transverse grid spacing, s, in mm (default: 0.05)
  -c: Specify the sound speed, c, in mm/us (default: 1.507)
  -e: Pad the domain to [nx,ny] pixels (default: next power of two)
  -w: Specify the high-order spectral correction weight (default: 0.39)
  -d: Specify a directivity axis x,y,z with width parameter w (default: none)
  -b: Specify sound speed bin width governing additional steps per slab (default: none)
  -g: Use OpenCL devices g0,g1,... on the first platform (default: first device)
  -v: Output volume field instead of Fourier transform of contrast source
	'''

def facetnormal(f):
	'''
	For an N x 3 array of coordinates of transducer elements on a planar
	facet, in which the i-th row represents the coordinates (x, y, z)
	coordinates (across the columns) of the i-th element, find the outward
	facet normal. Outward, in this case, means pointing to the side of the
	plane opposite the origin.
	'''
	# This reference vector should point from one corner to another
	ref = f[-1] - f[0]
	nref = norm(ref)
	# Enumerate local coordinates of all other elements
	vecs = [fv - f[0] for fv in f[1:-1]]
	# Find the vector most orthogonal to the reference
	v = min(vecs, key = lambda x: abs(np.dot(x, ref)) / norm(x) / nref)
	# Compute and normalize the normal
	n = np.cross(v, ref)
	n /= np.norm(n)
	# Find length of normal segment connecting origin to facet plane
	d = np.dot(f[0], n)
	# If d is positive, the normal already points outward
	return n if (d > 0) else -n


def propagator(srclist, contrast, output, c=1.507, f=3.0, s=0.05, w=0.39,
		p=2, q=0, a=None, d=None, e=None, z=None, b=None, v=False, g=0):
	'''
	For each source in a sequence of sources (each specified as a
	three-element sequence), propagate its field through the specified
	contrast file, storing the result in output (which may either be a
	volume field or the spectral field on the unit sphere, computed with
	the Goertzel algorithm).

	The remaining arguments have the same meanings as command-line
	arguments and defined in usage():

	c: Background sound speed, mm/us
	f: Excitation frequency, MHz
	s: Transverse grid spacing, mm
	w: High-order spatial correction weight
	p: Number of passes in which high-order spectral corrections are used
	q: Number of passes in which high-order spatial corrections are sued
	a: Width of Hann attenuating window at each edge
	d: Sequence [x,y,z,w] specifying directivity axis (x,y,z) and width w
	e: 2-D sequence specifying the extended transverse domain dimensions
	z: Axial grid spacing, mm
	b: "Speed bin" width, governs number of steps through high-speed slabs
	v: When True, write volume fields instead of spectral field
	g: GPU device to use for computations
	'''
	k0 = 2 * math.pi
	goertzel = not v

	# There is no work to be done if there are no sources
	if len(srclist) < 1: return

	# Convert distance units to wavelengths
	s *= f / float(c)
	# Convert or copy the axial step
	if z is not None: z *= f / float(c)
	else: z = s

	# Set up a slice-wise input reader
	ctmat = mio.Slicer(contrast)

	# Automatically pad the domain, if necessary
	if e is None: e = [cutil.ceilpow2(crd) for crd in ctmat.sliceshape]

	# Pick a default Hann window thickness that will not encroach on the domain
	if a is None:
		a = max(0, min((dcrd - gcrd) / 2
			for dcrd, gcrd in zip(e, ctmat.sliceshape)))

	print 'Hann window thickness: %d pixels' % a
	print 'Computing on expanded grid', e, 'with context', g

	# Convert source locations to wavelengths
	srclist = [tuple(crd * f / float(c) for crd in src) for src in srclist]

	sse = SplitStep(k0, e[0], e[1], s, d=d, l=a, w=w, dz=z,
			propcorr=(p > 0, q > 0), spdbin=b, context=g)

	# Restrict device transfers to the object grid
	sse.setroi(ctmat.sliceshape)

	# Augment the grid with the third (extended) dimension
	# An extra slice is required to turn around the field
	dom = list(ctmat.sliceshape) + [len(ctmat) + 1]
	print 'Propagating through %d slices' % dom[-1]

	# Create a progress bar to display computation progress
	bar = util.ProgressBar([0, dom[-1]], width=50)

	for srcidx, src in enumerate(srclist):
		# Ensure that the split-step engine is consistent
		sse.reset((p > 0, q > 0), goertzel=goertzel)

		# Create a unique output name
		outname = output + '.src%d.ctx%d' % (srcidx, g)

		# Compute the forward-traveling field half a slab before the start
		# Remember that dom[-1] is already a full slab before start
		zoff = wavetools.gridtocrd(dom[-1] - 0.5, len(ctmat), z)
		sse.setincident((src[0], src[1], src[2] - zoff))

		# Reset and print the progress bar
		bar.reset()
		util.printflush(str(bar) + ' (forward) \r')

		fmat = mio.Slicer(TemporaryFile(dir='.'), dom, ctmat.dtype, True)

		# An empty slab is needed for propagation beyond the medium
		zeroslab = mapbuffer(sse.context, ctmat.sliceshape, ctmat.dtype,
				cl.mem_flags.READ_ONLY, cl.map_flags.WRITE)[1]
		zeroslab.fill(0.)

		# Create buffered slice objects to read the contrast
		obj = BufferedSlices(ctmat, 5, context=sse.context)
		obj.setiter(reversed(range(len(ctmat))))
		obj.start()

		# Create a buffered slice object to write the forward field
		fbuf = BufferedSlices(fmat, 5, read=False, context=sse.context)
		fbuf.start()

		# Propagate the forward field through each slice
		for idx in reversed(range(dom[-1])):
			# Try to grab the next contrast, or default to zero
			try: obval = obj.getslice()
			except IndexError: obval = zeroslab
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

		# Recreate the object buffers for backward propagation
		obj.kill()
		obj = BufferedSlices(ctmat, 5, context=sse.context)
		obj.start()

		# Flush the forward field buffers to ensure consistency
		fbuf.flush()
		fbuf.kill()

		# Read the forward fields in reverse
		fbuf = BufferedSlices(fmat, 5, context=sse.context)
		fbuf.setiter(reversed(range(len(fmat))))
		fbuf.start()

		# Reset the split-step state and change the default corrective terms
		sse.reset((p > 1, q > 1), goertzel=goertzel)
		# Reset and reprint the progress bar
		bar.reset()
		util.printflush(str(bar) + ' (backward)\r')

		if v:
			# Only store volume output if volume fields are desired
			# Create a buffered slice object to write the output
			outmat = mio.Slicer(outname, ctmat.shape, ctmat.dtype, True)
			obuf = BufferedSlices(outmat, 5, read=False, context=sse.context)
			# Store first result (don't care) in last slice to be overwritten
			obuf.setiter(range(-1, len(outmat)))
			obuf.start()

		# Determine the propagator terms to include in the shifting pass
		shcorr = (p > 2, q > 2)

		# Propagate the reverse field through each slice
		for idx in range(dom[-1]):
			# Try to grab the next contrast, or default to zero
			try: obval = obj.getslice()
			except IndexError: obval = zeroslab
			# Advance the backward traveling field and reflect forward field
			# Also compute half-shifted, combined field in result buffer
			sse.advance(obval, fbuf.getslice(), True, shcorr=shcorr)
			# Copy the device result buffer to the host queue if desired
			if v: resevt = sse.getresult(obuf.getslice())
			# Advance the slice buffers
			obj.nextslice()
			fbuf.nextslice()
			# Tell the buffer to wait until the copy is finished
			if v: obuf.nextslice(resevt)
			# Increment and print the progress bar
			bar.increment()
			util.printflush(str(bar) + ' (backward)\r')

		print

		# Kill the I/O buffers
		obj.kill()
		fbuf.kill()

		if v:
			# Flush and kill the output buffers for volume writes
			obuf.flush()
			obuf.kill()
		else:
			# Fetch and store spectral fields if desired
			mio.writebmat(sse.goertzelfft(), outname)

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# These arguments will be passed to each propagator
	propargs = {}
	# By default, use only the first GPU for calculations
	gpuctx = [0]

	optlist, args = getopt.getopt(sys.argv[1:], 'hq:a:f:s:z:c:d:p:g:w:e:b:v')

	for opt in optlist:
		if opt[0] == '-a':
			propargs['a'] = int(opt[1])
		elif opt[0] == '-d':
			propargs['d'] = [float(ds) for ds in opt[1].split(',')]
		elif opt[0] == '-e':
			propargs['e'] = [int(ps) for ps in opt[1].split(',')]
		elif opt[0] == '-f':
			propargs['f'] = float(opt[1])
		elif opt[0] == '-s':
			propargs['s'] = float(opt[1])
		elif opt[0] == '-z':
			propargs['z'] = float(opt[1])
		elif opt[0] == '-c':
			propargs['c'] = float(opt[1])
		elif opt[0] == '-g':
			gpuctx = [int(gs) for gs in opt[1].split(',')]
		elif opt[0] == '-w':
			propargs['w'] = float(opt[1])
		elif opt[0] == '-q':
			propargs['q'] = int(opt[1])
		elif opt[0] == '-p':
			propargs['p'] = int(opt[1])
		elif opt[0] == '-b':
			propargs['b'] = float(opt[1])
		elif opt[0] == '-v':
			propargs['v'] = True
		else:
			usage(execname)
			sys.exit(128)

	# Ensure that a source list, contrast file and output file were specified
	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Group comma-separated source coordinates into (x,y,z) triples
	srclist = list(cutil.grouplist([float(s) for s in args[0].split(',')], 3))
	# Grab the file names for input and output
	contrast = args[1]
	output = args[2]

	# If fewer sources than GPUs, truncate the GPU list
	nsrc, ngpu = len(srclist), len(gpuctx)
	if nsrc < ngpu:
		gpuctx = gpuctx[:nsrc]
		ngpu = nsrc

	# Divide the source list into groups for each GPU context
	srcgroups = []
	share = nsrc / ngpu
	remainder = nsrc % ngpu
	for i in range(ngpu):
		start = i * share + min(remainder, i)
		srcgroups.append(srclist[start:start+share+(1 if i < remainder else 0)])

	try:
		# Spawn processes to handle GPU computations
		procs = []
		for srcgrp, g in zip(srcgroups, gpuctx):
			# Set the function arguments
			args = (srcgrp, contrast, output)
			kwargs = dict(propargs)
			# Establish the GPU context argument
			kwargs['g'] = g
			p = Process(target=propagator, args=args, kwargs=kwargs)
			p.start()
			procs.append(p)
		for p in procs: p.join()
	except:
		for p in procs:
			p.terminate()
			p.join()
		raise
