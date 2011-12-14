#!/usr/bin/env python

import numpy as np, math, sys, getopt
from pyajh import mio, cutil, harmonic

def usage(execname):
	print 'USAGE: %s [-h] [-r] [-i <iord>] <ntheta> <infile> <outfile>' % execname
	print '''
	Write to outfile the far-field matrix, characterized by ntheta samples
	of the polar angle, obtained by interpolating the far-field matrix
	specified in infile.

	OPTIONAL ARGUMENTS:
	-h: Display this message and exit
	-r: Sample polar angle at regular intervals instead of Gauss-Lobatto nodes
	-i: Use Lagrange interpolation of order iord (default: use cubic b-splines)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Set some default values
	regular, iord = False, 0

	optlist, args = getopt.getopt(sys.argv[1:], 'hri:')

	for opt in optlist:
		if opt[0] == '-r': regular = True
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
	ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.matsize[0] - 10.)))

	# Build coarse and fine polar samples
	thetas = [harmonic.polararray(n, not regular) for n in [ntc, ntf]]

	# Create the interpolation matrix
	if iord > 0: a = harmonic.SphericalInterpolator(thetas, iord)
	else: a = harmonic.HarmonicSpline(thetas)

	# Interpolate each column of the matrix and write it to a file
	with open(args[2], 'wb') as output:
		# Write the output matrix size
		np.array([a.matrix.shape[0], inmat.matsize[-1]],
				dtype=np.int32).tofile(output)

		# Read each row in the input, interpolate it, and write it
		for row in inmat:
			a.interpolate(row[1]).astype(inmat.dtype).tofile(output)
