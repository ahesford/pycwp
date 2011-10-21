#!/usr/bin/env python

import numpy as np, math, sys
from pyajh import mio, cutil, harmonic

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print 'USAGE: %s <ntheta> <iord> <infile> <outfile>' % sys.argv[0]
		sys.exit(128)

	# Grab the variables
	ntf = int(sys.argv[1])
	npf = 2 * (ntf - 2)
	iord = int(sys.argv[2])
	outname = sys.argv[4]

	# Read the input file
	inmat = mio.readbmat(sys.argv[3])

	# Determine the number of theta samples for the input
	ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.shape[1] - 10.)))
	npc = 2 * (ntc - 2)

	# Build the coarse and fine theta samples
	thetas = [[math.pi] + list(reversed(cutil.gaussleg(n - 2)[0])) + [0.]
			for n in [ntc, ntf]]
	phis = [[2. * math.pi * i / n for i in range(n)] for n in [npc, npf]]

	# Create the interpolation matrix
	a = harmonic.SphericalInterpolator(thetas, phis, iord)

	# Interpolate the input
	outmat = np.array([a.applymat(row.tolist()) for row in inmat])

	# Write the output as a 64-bit complex matrix
	mio.writebmat(outmat.astype(np.complex64), outname)
