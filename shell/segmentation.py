#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os, numpy as np, getopt
from pycwp import mio, segmentation, process

def usage(progname = 'segmentation.py'):
	binfile = os.path.basename(progname)
	print("Usage:", binfile, "[-h] [-n] [-p p] [-d scatden] [-s scatsd] [-g w,s]")
	print("\t[-c chunk] <segfile> <paramfile> <sndfile> <atnfile> <denfile>")
	print()
	print("\t-n: Disable random variations")
	print("\t-p: Use p processors (default: CPU count)")
	print("\t-d: Assume a random scatterer fractional density scatden")
	print("\t-s: Smooth random scatterers with Gaussian of standard deviation scatsd")
	print("\t-g: Smooth tissue with Gaussian of width w and standard deviation s")
	print("\t-c: Process output chunk slices at a time (default: 8)")

def mapblks(segfile, outputs, params, start, stride, chunk, **kwargs):
	'''
	Open the segmentation file segfile, output files in the list outputs,
	and loop through the segmentation in strides to produce output
	parameter maps. kwargs are optional arguments to be passed to the
	segmentation routine.

	The arguments start and stride refer to chunks rather than slices.
	'''
	# Open the files
	seg = mio.Slicer(segfile)
	sndfile, atnfile, denfile = [mio.Slicer(o) for o in outputs]

	# Add the chunk size to the kwargs for convenience
	kwargs['chunk'] = chunk

	# Loop through the chunks to process output
	for n in range(start * chunk, seg.shape[-1], stride * chunk):
		print('Processing chunk', n)
		snd, atn, den = segmentation.maptissueblk(seg, params, n, **kwargs)
		# Figure out how many slices need to be written
		oend = min(seg.shape[-1], n + snd.shape[-1])
		# Write the outputs
		sndfile[n:oend] = snd
		atnfile[n:oend] = atn
		denfile[n:oend] = den


def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Default values
	random = True
	nproc = process.preferred_process_count()
	chunk = 8

	optlist, args = getopt.getopt (argv, 'p:nd:s:c:g:h')

	# Extra arguments are added as kwargs
	kwargs = {}

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-n': random = False
		elif opt[0] == '-p': nproc = int(opt[1])
		elif opt[0] == '-d': kwargs['scatden'] = float(opt[1])
		elif opt[0] == '-s': kwargs['scatsd'] = float(opt[1])
		elif opt[0] == '-c': chunk = int(opt[1])
		elif opt[0] == '-g':
			kstr = opt[1].split(',')
			kwargs['smoothp'] = [int(kstr[0]), float(kstr[1])]
		else:
			usage (progname)
			return 128

	# The segmentation file and the parameter file must be specified
	if len(args) < 5:
		usage (progname)
		return 128

	# Read the tissue parameters
	pmat = np.loadtxt(args[1])
	# Split the parameters into sound speed, attenuation and density
	params = [p.tolist() for p in [pmat[:,:2], pmat[:,2:4], pmat[:,4:6]]]
	# Eliminate the standard deviation if random scatterers are not desired
	if not random: params = [[[p[0], None] for p in pv] for pv in params]

	# Grab the shape of the segmentation file and the number of slices
	segfile = mio.Slicer(args[0])
	# The output files need to be created and truncated
	outputs = args[2:]
	outfiles = [mio.Slicer(o, segfile.shape, segfile.dtype, True) for o in outputs]

	try:
		with process.ProcessPool() as pool:
			for n in range(nproc):
				args = (args[0], outputs, params, n, nproc, chunk)
				pool.addtask(target=mapblks, args=args, kwargs=kwargs)
			pool.start()
			pool.wait()
	except:
		for f in outfiles: f._backer.truncate(0)
		raise

	return 0

if __name__ == "__main__":
	sys.exit (main ())
