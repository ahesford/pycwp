#!/usr/bin/env python

import numpy as np, math, sys, getopt, os, pyopencl as cl
from numpy.linalg import norm
from tempfile import TemporaryFile
from multiprocessing import Process
from subprocess import check_call
from mpi4py import MPI

from pyajh import mio, wavetools, util, cutil, geom
from pyajh.cltools import SplitStep, BufferedSlices, mapbuffer

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-q q] [-p p] [-a a] [-r regrid.py] [-t tmpdir] [-g g0,g1,...] [-v] [-V] [-f f] [-s s] [-c c] [-e nx,ny] [-w w] [-P P] [-d [x,y,z,]w]', '<srclist> <infile> <outfile>'
	print '''
  Using the split-step method, compute the fields induced in a contrast medium
  specified in infile by each of a collection of sources listed in the file
  srclist. The file describes each source on a separate line and, because
  numpy.loadtxt is used to parse the file, supports hash comments. Each source
  is described by four columns; the x, y, and z coordinates of the source (in
  that order) followed by a facet identifier integer. The identifier is
  arbitrary, but sources sharing an identifier are assumed to belong to the
  same planar facet.

  If a path to regrid.py is specified, the contrast will be rotated so that the
  z axis is parallel to a reference direction unless the z axis is already
  within 5 degrees of that direction. If a directivity axis is specified, this
  is always the negative of the reference direction. Otherwise, if there are at
  least three non-collinear elements in a facet, the reference direction is the
  normal of the facet pointing away from the origin. If the facet is degenerate
  (with only one or two elements, or any number of collinear elements), the
  reference direction is toward the midpoint of the sources.

  The coordinate origin is at the center of the contrast file. Propagation
  happens downward; the topmost slab (the last in the file) first receives the
  incident field. A configurable attenuation profile is applied to each edge of
  each slice.

  Each solution is written to outfile with a unique identifier appended.

  The computational domain is padded to the next larger power of 2 for
  artificial attenuation.

  Computations happen in three passes: a forward pass through the contrast, a
  backward pass to pick up backward scatter, and a third pass (computed along
  with the second pass) to shift the fields in each slice to the center for
  comparison to reference methods.

  OPTIONAL ARGUMENTS:
  -h: Display this message and exit
  -q: Use high-order spatial terms in the first q passes (default: 0)
  -p: Use high-order spectral terms in the first p passes (default: 2)
  -a: Width of the Hann attenuating window at each edge (default: 100)
  -r: Full path to regrid.py for contrast rotation (default: no rotation)
  -t: Full path to temporary directory to store intermediate files (default: CWD)
  -g: Use OpenCL devices g0,g1,... on the first platform (default: first device)
  -v: Display a progress bar to track the status of each propagation
  -V: Output volume field instead of Fourier transform of contrast source
  -f: Incident frequency, f, in MHz (default: 3.0)
  -s: Isotropic grid spacing, s, in mm (default: 0.05)
  -c: Sound speed, c, in mm/us (default: 1.507)
  -e: Pad domain to [nx,ny] pixels (default: next power of two)
  -w: High-order spectral correction weight (default: 0.39)
  -P: Use extra steps to avoid phase deviations greater than (P * pi) per slab
  Incremental speed threshold for additional propagation steps (default: none)
  -d: Directivity width parameter w and optional axis (x,y,z) (default: none)
	'''

def facetnormal(f):
	'''
	For an N x 3 array of coordinates of transducer elements on a planar
	facet, in which the i-th row represents the coordinates (x, y, z)
	coordinates (across the columns) of the i-th element, find the outward
	facet normal. Outward, in this case, means pointing to the side of the
	plane opposite the origin.
	'''
	# If there aren't enough points on the facet, pick the midpoint
	if len(f) < 3:
		n = np.mean(f, axis=0)
		n /= norm(n)
		return n

	# The first reference vector should point from one corner to another
	ref = f[-1] - f[0]
	nref = norm(ref)
	# Enumerate local coordinates of all other elements
	vecs = [fv - f[0] for fv in f[1:-1]]
	# Find the vector most orthogonal to the reference
	v = min(vecs, key = lambda x: abs(np.dot(x, ref)) / norm(x) / nref)
	# Compute the normal and its length
	n = np.cross(v, ref)
	nnrm = norm(n)

	# If the cross product is very small, treat the elements as collinear
	if nnrm < 1.0e-6:
		n = np.mean(f, axis=0)
		n /= norm(n)
		return n

	# Otherwise, normalize the vector and pick the right direction
	n /= nnrm
	# Find length of normal segment connecting origin to facet plane
	d = np.dot(f[0], n)
	# If d is positive, the normal already points outward
	return n if (d > 0) else -n


def propagator(contrast, output, srclist, start=0, share=-1, c=1.507, f=3.0,
		s=0.05, w=0.39, p=2, q=0, a=None, d=None, e=None, g=0, V=False,
		phasetol=None, tmpdir=None, verbose=False):
	'''
	For a subset of sources in a sequence of sources (each specified as a
	three-element sequence), propagate its field through the specified
	contrast file, storing the result in output (which may either be a
	volume field or the spectral field on the unit sphere, computed with
	the Goertzel algorithm).

	The subset of sources is specified by the arguments start and share,
	which specify the starting index and number of subsequent contiguous
	sources to use. If share is less than 0, all sources following and
	including start are used.

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
	g: GPU device to use for computations
	V: When True, write volume fields instead of spectral field
	phasetol: Maximum phase deviation allowed for each propagation
	tmpdir: Create temporary files in this directory (CWD if None)
	verbose: When True, print a progress bar tracking each propagation
	'''
	k0 = 2 * math.pi
	goertzel = not V

	# Ensure that the starting index is valid or else do no work
	if start < 0 or start >= len(srclist): return

	# Adjust negative share values
	if share < 0: share = len(srclist) - start

	# There is no work to be done if there is no share
	if share < 1: return

	# If no temporary directory was specified, use the current directory
	if tmpdir is None: tmpdir = '.'

	# Convert distance units to wavelengths
	s *= f / float(c)

	# Set up a slice-wise input reader
	ctmat = mio.Slicer(contrast)

	# Automatically pad the domain, if necessary
	if e is None: e = [cutil.ceilpow2(crd) for crd in ctmat.sliceshape]

	# Pick a default Hann window thickness that will not encroach on the domain
	if a is None:
		a = max(0, min((dcrd - gcrd) / 2
			for dcrd, gcrd in zip(e, ctmat.sliceshape)))

	# Note the rank of this process
	mpirank = MPI.COMM_WORLD.Get_rank()
	# Construct a unique identifier for this rank and context
	procid = 'MPI rank {:d}({:d})'.format(mpirank, g)

	# Convert source locations to wavelengths
	srclist = [tuple(crd * f / float(c) for crd in src) for src in srclist]

	sse = SplitStep(k0, e[0], e[1], s, d=d, l=a, w=w,
			propcorr=(p > 0, q > 0), phasetol=phasetol, context=g)

	# Restrict device transfers to the object grid
	sse.setroi(ctmat.sliceshape)

	# Augment the grid with the third (extended) dimension
	# An extra slice is required to turn around the field
	dom = list(ctmat.sliceshape) + [len(ctmat) + 1]

	if verbose:
		# Note the propagation parameters
		gridstr = ' x '.join(str(crd) for crd in list(e) + dom[-1:])
		print '{:s}: {:s} grid, {:d}-pixel Hann window'.format(procid, gridstr, a)
		# Create a progress bar to track propagation status
		bar = util.ProgressBar([0, dom[-1]], width=50)

	for srcidx in range(start, start + share):
		print '{:s}: Propagating source {:d}'.format(procid, srcidx)

		if verbose:
			# Reinitialize the progress bar
			bar.reset()
			util.printflush(str(bar) + ' (forward) \r')

		# Convert the current source to wavelengths
		src = tuple(crd * f / float(c) for crd in srclist[srcidx])
		# Create a unique output name
		srcstring = util.zeropad(srcidx, len(srclist))
		outname = '{:s}.src{:s}'.format(output, srcstring)

		# Ensure that the split-step engine is consistent
		sse.reset((p > 0, q > 0), goertzel=goertzel)

		# Compute the forward-traveling field half a slab before the start
		# Remember that dom[-1] is already a full slab before start
		zoff = wavetools.gridtocrd(dom[-1] - 0.5, len(ctmat), s)
		sse.setincident((src[0], src[1], src[2] - zoff))

		fmat = mio.Slicer(TemporaryFile(dir=tmpdir), dom, ctmat.dtype, True)

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

			if verbose:
				bar.increment()
				util.printflush(str(bar) + ' (forward) \r')

		if verbose:
			# Finalize forward printing and reset the bar
			print
			bar.reset()
			util.printflush(str(bar) + ' (backward)\r')

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

		if V:
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
			if V: resevt = sse.getresult(obuf.getslice())
			# Advance the slice buffers
			obj.nextslice()
			fbuf.nextslice()
			# Tell the buffer to wait until the copy is finished
			if V: obuf.nextslice(resevt)

			if verbose:
				bar.increment()
				util.printflush(str(bar) + ' (backward)\r')

		if verbose: print

		# Kill the I/O buffers
		obj.kill()
		fbuf.kill()

		if V:
			# Flush and kill the output buffers for volume writes
			obuf.flush()
			obuf.kill()
		else:
			# Fetch and store spectral fields if desired
			mio.writebmat(sse.goertzelfft(), outname)

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# By default, the rotation program, regrid.py, is unspecified
	rotprog = None
	# Rotate the medium if the reference direction deviates from z by five degrees
	rottol = 5. * math.pi / 180.

	# These arguments will be passed to each propagator
	propargs = {}
	# By default, use only the first GPU for calculations
	gpuctx = [0]

	# By default, don't print verbose status
	verbose = False

	optlist, args = getopt.getopt(sys.argv[1:], 'hq:a:f:s:c:d:p:g:w:e:P:vVr:t:')

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
		elif opt[0] == '-P':
			propargs['phasetol'] = float(opt[1])
		elif opt[0] == '-V':
			propargs['V'] = True
		elif opt[0] == '-v':
			verbose = True
			propargs['verbose'] = True
		elif opt[0] == '-r':
			rotprog = opt[1]
		elif opt[0] == '-t':
			propargs['tmpdir'] = opt[1]
		else:
			usage(execname)
			sys.exit(128)

	# Ensure that a source list, contrast file and output file were specified
	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Find MPI task parameters
	rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()

	# Read elements coordinates and facet indices from facet file
	elements = np.loadtxt(args[0])
	# Ensure that the array has at least two dimensions
	if len(elements.shape) < 2: elements = elements[np.newaxis,:]
	# Find the total number of facets for labeling purposes
	numfacets = max(elements[:,-1].astype(int)) + 1

	# Figure the share of elements for this MPI task
	nelts = len(elements)
	share = nelts / size
	remainder = nelts % size
	start = rank * share + min(remainder, rank)
	if rank < remainder: share += 1

	# Pull out the elements handled by this process
	elements = elements[start:start+share,:]
	# Pull out a list of unique facet indices
	facets = np.unique(elements[:,-1].astype(int)).tolist()

	# Grab the file names for input and output
	contrast = args[1]
	output = args[2]

	# Loop over all facet indices
	for fidx in facets:
		# Pull all elements belonging to the current facet
		srclist = [el[:-1] for el in elements.tolist() if int(el[-1]) == fidx]

		# Determine if rotation is necessary
		try:
			# If a directivity axis was specified, use its negative
			directivity = propargs['d']
			if len(directivity) < 4:
				raise KeyError('Directivity axis was not specified')
			propax = -np.array(directivity[:-1])
			r, theta, phi = geom.cart2sph(*propax)
		except KeyError:
			# If no directivity axis was specified, determine the axis
			propax = facetnormal(np.array(srclist))
			r, theta, phi = geom.cart2sph(*propax)

		# The output file always gets a unique identifier
		appendage = '.facet{:s}'.format(util.zeropad(fidx, numfacets))
		outunique = output + appendage

		if (rotprog is None) or (theta < rottol): rotcontrast = contrast
		else:
			# Create the rotated contrast and output filenames
			rotcontrast = contrast + appendage
			# Put rotated contrast in desired temporary directory
			try: tmpdir = propargs['tmpdir']
			except KeyError: tmpdir = '.'
			rotcontrast = os.path.join(tmpdir, os.path.basename(rotcontrast))

			# Determine the input grid, which is also the output grid
			grid = mio.Slicer(contrast).shape
			gridstr = ','.join('%d' % gv for gv in grid)

			# Rotate the axes of the contrast grid parallel to normal
			rotargs = ['-g', '%d' % gpuctx[0], '-n', gridstr,
					'-t', '%g' % -theta, '-p', '%g' % -phi,
					contrast, rotcontrast]
			# If verbosity is desired, make regridding verbose too
			if verbose: rotargs = ['-v'] + rotargs
			check_call([rotprog] + rotargs)

			# Rotate the element coordinates to the new system
			srclist = [geom.rotate3d(src, -theta, -phi) for src in srclist]
			# Rotate the propagation axis (it should become 0,0,1)
			propax = np.array(geom.rotate3d(propax, -theta, -phi))

			try:
				# Attempt to replace the directivity axis
				beamwidth = propargs['d'][-1]
				propargs['d'] = list(-propax) + [beamwidth]
			except KeyError: pass

		# If fewer sources than GPUs, truncate the GPU list
		nsrc, ngpu = len(srclist), len(gpuctx)
		if nsrc < ngpu:
			gpuctx = gpuctx[:nsrc]
			ngpu = nsrc

		try:
			# Spawn processes to handle GPU computations
			procs = []
			for i, g in enumerate(gpuctx):
				# Figure the work load
				share = nsrc / ngpu
				remainder = nsrc % ngpu
				start = i * share + min(remainder, i)
				if i < remainder: share += 1
				# Set the function arguments
				args = (rotcontrast, outunique, srclist, start, share)
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
		finally:
			# Remove the rotated contrast if necessary
			if rotcontrast != contrast: os.remove(rotcontrast)
