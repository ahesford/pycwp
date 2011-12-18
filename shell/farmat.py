#!/usr/bin/env python

import numpy as np, math, sys, getopt, itertools
from pyajh import mio, cutil, wavecl, harmonic


def usage(execname):
	print 'USAGE: %s [-h] [-r] [-i <iord>] [-d <length>] [-n <ncell>] <ntheta> <outfile>' % execname
	print '''
	Write to outfile the far-field matrix for a group of ncell elements per
	dimension using ntheta samples of the polar angle.

	OPTIONAL ARGUMENTS:
	-h: Display this message and exit
	-r: Sample polar angle at regular intervals instead of Gauss-Lobatto nodes
	-i: Use quadrature order iord for integration (default: 4)
	-d: Specify the edge length of each cubic cell in wavelengths (default: 0.1)
	-n: Specify the number of cells per group per dimension (default: 10)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Create an empty dictionary for optional arguments
	regular, iord, dc, n = False, 4, 0.1, 10

	optlist, args = getopt.getopt(sys.argv[1:], 'hri:d:n:')

	for opt in optlist:
		if opt[0] == '-r': regular = True
		elif opt[0] == '-i': iord = int(opt[1])
		elif opt[0] == '-d': dc = float(opt[1])
		elif opt[0] == '-n': n = int(opt[1])
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 2:
		usage(execname)
		sys.exit(128)

	# Grab the number of samples of the polar angle
	nt = int(args[0])

	# Compute the polar samples as Gauss-Lobatto nodes or regular samples
	theta = harmonic.polararray(nt, not regular)

	# Build a generator of cell coordinates
	hc = n / 2. + 0.5
	coords = itertools.product(dc * np.mgrid[-hc+1:hc], repeat=3)

	# Build the matrix class
	f = wavecl.FarMatrix(theta, dc, iord)

	print "Building %d-by-%d far-field matrix" % (f.nsamp, n**3)

	# Create the output file
	with open(args[1], 'wb') as output:
		# Write the final matrix size
		np.array([f.nsamp, n**3], dtype=np.int32).tofile(output)

		# Build the matrix row-by-row and write it to output
		for c in coords: f.fillrow(c).tofile(output)
