#!/usr/bin/env python

import numpy as np, math, sys, getopt, os
from pyajh import mio, cutil, harmonic
from pyajh.cltools import interpolators as clinterp

def usage(execname):
	binfile = os.path.basename(execname)
	print 'USAGE:', binfile, '[-h] [-r] [-i <iord>] [-t <tol>] <ntheta> <infile> <outfile>'
	print '''
	Write to outfile the far-field matrix, characterized by ntheta samples
	of the polar angle, obtained by interpolating the far-field matrix
	specified in infile.

	OPTIONAL ARGUMENTS:
	-h: Display this message and exit
	-r: Sample polar angle at regular intervals instead of Gauss-Lobatto nodes
	-i: Use Lagrange interpolation of order iord (default: use cubic b-splines)
	-t: Use a tolerance of tol for spline coefficient computation (default: 1e-7)
	-g: Use the GPU for interpolation (implies -r and not -i)
	'''

if __name__ == '__main__':
	# Grab the executable name
	execname = sys.argv[0]

	# Set some default values
	regular, gpu, iord, tol = False, False, 0, 1e-7

	optlist, args = getopt.getopt(sys.argv[1:], 'hrgi:t:')

	for opt in optlist:
		if opt[0] == '-r': regular = True
		elif opt[0] == '-i': iord = int(opt[1])
		elif opt[0] == '-t': tol = float(opt[1])
		elif opt[0] == '-g': gpu = regular = True
		else:
			usage(execname)
			sys.exit(128)

	if len(args) < 3:
		usage(execname)
		sys.exit(128)

	# Grab the number of interpolated polar samples
	ntf = int(args[0])

	# Create a generator to read the matrix column by column
	inmat = mio.Slicer(args[1])

	# Compute the input number of samples of the polar angle
	ntc = int(2. + math.sqrt(4. + 0.5 * (inmat.shape[0] - 10.)))

	# The total number of output samples
	nsamp = 2 * (ntf - 2)**2 + 2

	if not gpu:
		# Build coarse and fine polar samples
		thetas = [harmonic.polararray(n, not regular) for n in [ntc, ntf]]

		# Create the interpolation matrix
		if iord > 0: a = harmonic.SphericalInterpolator(thetas, iord)
		else: a = harmonic.HarmonicSpline(thetas, tol)
	else: a = clinterp.HarmonicSpline(ntc, 2 * (ntc - 2), tol)

	# Interpolate each column of the matrix and write it to a file
	# Truncate any existing output file
	output = mio.Slicer(args[2], [nsamp, inmat.shape[-1]], inmat.dtype, True)
	
	# Interpolate to the finer grid
	if not gpu:
		for i, row in enumerate(inmat): output[i] = a.interpolate(row)
	else:
		for i, row in enumerate(inmat):
			# Build the spline coefficients for the row
			a.buildcoeff(row)
			# Interpolate to the finer grid and write to output
			output[i] = a.interpolate(ntf, 2 * (ntf - 2))
