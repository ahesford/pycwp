#!/usr/bin/env python

import numpy as np, math, sys, getopt
from pyajh import mio, cutil, harmonic

def usage(execname):
	print 'USAGE: %s [-h] [-p] [-w] [-i <iord>] <ntheta> <infile> <outfile>' % execname
	print '''
	Write to outfile the far-field matrix, characterized by ntheta samples
	of the polar angle, obtained by interpolating the far-field matrix
	specified in infile.

	OPTIONAL ARGUMENTS:
	-h: Display this message and exit
	-p: Sample polar angle at regular intervals away from the poles
	    By default, the samples correspond to Gauss-Lobatto quadrature nodes
	-w: Prohibit interpolation intervals from wrapping around poles
	-i: Use quadrature order iord for integration (default: 4)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Set some default values
	poles, wrap, iord = True, True, 4

	optlist, args = getopt.getopt(sys.argv[1:], 'hpi:w')

	for opt in optlist:
		if opt[0] == '-p': poles = False
		elif opt[0] == '-w': wrap = False
		elif opt[0] == '-i': iord = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Grab the number of interpolated polar samples
	ntf = int(args[0])

	# Create a generator to read the matrix column by column
	inmat = mio.ReadSlicer(args[1])

	# Compute the input number of samples of the polar angle
	if poles: ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.matsize[0] - 10.)))
	else: ntc = int(math.sqrt(inmat.matsize[0] / 2.))

	# Build coarse and fine polar samples using Lobatto rules
	thetas = [harmonic.polararray(n, poles) for n in [ntc, ntf]]

	# Create the interpolation matrix
	a = harmonic.SphericalInterpolator(thetas, iord, poles, wrap)

	# Interpolate each column of the matrix and write it to a file
	with open(args[2], 'wb') as output:
		# Write the output matrix size
		np.array([a.matrix.shape[0], inmat.matsize[-1]],
				dtype=np.int32).tofile(output)

		# Read each row in the input, interpolate it, and write it
		for row in inmat:
			a.applymat(row[1]).astype(inmat.dtype).tofile(output)
