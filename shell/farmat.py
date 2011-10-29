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
	coords = itertools.product(dc * np.mgrid[-hc+1:hc], repeat=3)

	# Build the matrix class
	f = wavecl.FarMatrix(theta, dc, iord, poles)

	print "Building %d-by-%d far-field matrix" % (f.nsamp, n**3)

	# Create the output file
	with open(args[2], 'wb') as output:
		# Write the final matrix size
		np.array([f.nsamp, n**3], dtype=np.int32).tofile(output)

		# Build the matrix row-by-row and write it to output
		for c in coords: f.fillrow(c).tofile(output)
