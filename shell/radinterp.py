#!/usr/bin/env python

import numpy as np, math, sys
from pyajh import mio, cutil, harmonic

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print 'USAGE: %s <ntheta> <iord> <infile> <outfile>' % sys.argv[0]
		sys.exit(128)

	# Grab the variables
	ntf = int(sys.argv[1])
	iord = int(sys.argv[2])
	outname = sys.argv[4]

	# Read the input file
	inmat = mio.readbmat(sys.argv[3])

	# Determine the number of theta samples for the input
	ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.shape[1] - 10.)))

	# Build the coarse and fine theta samples
	thetas = [harmonic.polararray(n) for n in [ntc, ntf]]

	# Create the interpolation matrix
	a = harmonic.SphericalInterpolator(thetas, iord)

	# Interpolate the input
	outmat = np.array([a.applymat(row.tolist()) for row in inmat])

	# Write the output as a 64-bit complex matrix
	mio.writebmat(outmat.astype(np.complex64), outname)
