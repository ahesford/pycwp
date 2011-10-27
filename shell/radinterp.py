#!/usr/bin/env python

import numpy as np, math, sys, getopt
from pyajh import mio, cutil, harmonic

def usage(execname):
	print 'USAGE: %s [-h] [-p] [-i <iord>] <ntheta> <infile> <outfile>' % execname

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

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Grab the number of interpolated polar samples
	ntf = int(args[0])

	# Read the input file
	inmat = mio.readbmat(args[1])

	# Grab the output file
	outname = args[2]

	if poles:
		# Compute the input number of polar samples
		ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.shape[1] - 10.)))

		# Build coarse and fine polar samples using Lobatto rules
		thetas = [harmonic.polararray(n) for n in [ntc, ntf]]
	else:
		# Compute the input number of polar samples
		ntc = int(math.sqrt(inmat.shape[1] / 2.))

		# Build coarse and fine polar samples using regular spacing
		thetas = [list(np.pi * np.arange(1., n+1) / (n + 1.)) for n in [ntc, ntf]]

	# Create the interpolation matrix
	a = harmonic.SphericalInterpolator(thetas, iord, poles)

	# Interpolate the input
	outmat = np.array([a.applymat(row.tolist()) for row in inmat])

	# Write the output as a 64-bit complex matrix
	mio.writebmat(outmat.astype(np.complex64), outname)
