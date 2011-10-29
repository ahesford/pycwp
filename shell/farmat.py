#!/usr/bin/env python

import numpy as np, math, sys, getopt, itertools
from pyajh import mio, cutil, wavecl, harmonic


def usage(execname):
	print 'USAGE: %s [-h] [-p] [-i <iord>] [-d <length>] <ncell> <ntheta> <outfile>' % execname

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Set some default values
	iord, poles, dc = 4, False, 0.1

	optlist, args = getopt.getopt(sys.argv[1:], 'hpi:d:')

	for opt in optlist:
		if opt[0] == '-p': poles = True
		elif opt[0] == '-i': iord = int(opt[1])
		elif opt[0] == '-d': iord = float(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Grab the variables
	n = int(args[0])
	nt = int(args[1])

	# Compute the polar samples as Gauss-Lobatto nodes or regular samples
	if poles: theta = harmonic.polararray(nt)
	else: theta = math.pi * np.arange(1., nt + 1.) / (nt + 1.)

	# Build a generator of cell coordinates
	hc = n / 2. + 0.5
	coords = itertools.product(np.mgrid[-hc+1:hc], repeat=3)

	# Use a default context to build and write the far-field matrix
	mio.writebmat(wavecl.FarMatrix(theta, dc, iord, poles).fill(coords), args[2])
