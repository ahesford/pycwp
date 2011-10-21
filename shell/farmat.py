#!/usr/bin/env python

import numpy as np, math, sys
from pyajh import mio, cutil, wavecl, harmonic


if __name__ == '__main__':
	if len(sys.argv) < 6:
		print 'USAGE: %s <ncell> <ntheta> <dc> <iord> <outfile>' % sys.argv[0]
		sys.exit(128)

	# Grab the variables
	n = int(sys.argv[1])
	nt = int(sys.argv[2])
	dc = float(sys.argv[3])
	iord = int(sys.argv[4])
	outfile = sys.argv[5]

	# Compute the theta samples as Gauss-Legendre values
	theta = harmonic.polararray(nt)

	# Build a flattened array of cell coordinates
	gsl = [slice(-n / 2. + 0.5, n / 2. + 0.5) for i in range(3)]
	grid = np.mgrid[gsl]
	coords = [[dc * co for co in crd] for crd in zip(*[c.flat for c in grid])]

	# Use a default to build the far-field matrix
	ffd = wavecl.FarMatrix(theta, dc, iord).fill(coords)
	mio.writebmat(ffd, outfile)
