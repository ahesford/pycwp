#!/usr/bin/env python

import numpy as np, math, sys, getopt
from pyajh import mio, cutil, wavecl, harmonic


def usage(execname):
	print 'USAGE: %s [-h] [-p] [-i <iord>] <ncell> <ntheta> <dc> <outfile>' % execname

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Set some default values
	iord, poles = 4, False

	optlist, args = getopt.getopt(sys.argv[1:], 'hpi:')

	for opt in optlist:
		if opt[0] == '-p': poles = True
		elif opt[0] == '-i': iord = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 4:
		usage(execname)
		sys.exit(128)

	# Grab the variables
	n = int(args[0])
	nt = int(args[1])
	dc = float(args[2])

	# Compute the polar samples as Gauss-Lobatto nodes or regular samples
	if poles: theta = harmonic.polararray(nt)
	else: theta = math.pi * np.arange(1., nt + 1.) / (nt + 1.)

	# Build a flattened array of cell coordinates
	gsl = [slice(-n / 2. + 0.5, n / 2. + 0.5) for i in range(3)]
	grid = np.mgrid[gsl]
	coords = [[dc * co for co in crd] for crd in zip(*[c.flat for c in grid])]

	# Use a default context to build and write the far-field matrix
	mio.writebmat(wavecl.FarMatrix(theta, dc, iord, poles).fill(coords), args[3])
