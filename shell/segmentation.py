#!/usr/bin/env python

import sys, os, numpy as np, getopt
from pyajh import mio, segmentation

def usage(progname = 'segmentation.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-n <nproc>] <segfile> <paramfile>"

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Default values
	random, nproc = True, 1

	optlist, args = getopt.getopt (argv, 'p:nh')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-n':
			random = False
		elif opt[0] == '-p':
			nproc = int(opt[1])
		else:
			usage (progname)
			return 128

	# The segmentation file and the parameter file must be specified
	if len(args) < 2:
		usage (progname)
		return 128

	# Read the tissue parameters
	pmat = np.loadtxt(args[-1])
	# Split the parameters into sound speed, attenuation and density
	params = [p.tolist() for p in [pmat[:,:2], pmat[:,2:4], pmat[:,4:6]]]
	# Eliminate the standard deviation if random scatterers are not desired
	if not random: params = [[[p[0], None] for p in pv] for pv in params]

	# Grab the shape of the segmentation file and the number of slices
	segfile = mio.Slicer(args[0])
	# The output files need to be created and truncated
	outputs = ['SoundSpeed.dat', 'Attenuation.dat', 'Density.dat']
	outfiles = [mio.Slicer(o, segfile.shape, segfile.dtype, True) for o in outputs]

	# Perform the segmentation
	segmentation.maptissue(args[0], outputs, params)

	return 0

if __name__ == "__main__":
	sys.exit (main ())
