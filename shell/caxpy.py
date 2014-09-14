#!/usr/bin/env python

import sys, os, getopt, numpy as np, multiprocessing
from itertools import izip

from pycwp import mio, cutil

def usage(progname = 'caxpy.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-p nproc] [-a a] [-b b] <z> <x> <y>"
	print "  Compute z = a * x + b * y for matrices in files z, x, and y"
	print "  Both a and b are complex constants specified in the form c+dj"
	print "  By default, a = b = 1"


def caxpby(args):
	'''
	For an argument list args = (i, z, a, x, b, y) for integer i; matrix
	file names z, x, y; and constants a, b; compute 

		z[i] = a * x[i] + b * y[i].

	Nothing is returned.
	'''
	# Grab the constants from the argument list
	i, a, b = args[::2]
	# Open the files in the argument list as Slicer objects
	z, x, y = [mio.Slicer(arg) for arg in args[1::2]]
	# Compute and store the output
	z[i] = a * x[i] + b * y[i]


def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Set default options
	a = b = 1.
	try: nproc = multiprocessing.cpu_count()
	except NotImplementedError: nproc = 1

	optlist, args = getopt.getopt (argv, 'p:a:b:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-a': a = complex(opt[1])
		elif opt[0] == '-b': b = complex(opt[1])
		elif opt[0] == '-p': nproc = int(opt[1])
		else:
			usage (progname)
			return 128

	# There must be at least three file names
	if len(args) < 2:
		usage (progname)
		return 128

	# If the coefficients have zero imaginary parts, cast them as real
	a, b = [c if np.iscomplex(c) else c.real for c in [a, b]]

	# Make sure that the shapes of all of the inputs agree
	inputs = [mio.Slicer(arg) for arg in args[1:]]
	if tuple(inputs[0].shape) != tuple(inputs[1].shape):
		raise ValueError('Array sizes must agree')

	# Determine the output type based on the greater of the two precisions
	chtypes = [np.dtype(i.dtype).char for i in inputs]
	# Determine if a complex value (uppercase code) exists
	if True in [c.isupper() for c in chtypes]: cplx = True
	elif np.iscomplex(a) or np.iscomplex(b): cplx = True
	else: cplx = False
	# Grab the highest precision and render it complex if necessary
	otype = sorted(c.lower() for c in chtypes)[0]
	if cplx: otype = otype.upper()
	# Now create the output file, truncating if necessary
	output = mio.Slicer(args[0], inputs[0].shape, np.dtype(otype).type, True)
	nslice = output.shape[-1]

	# Compute, in parallel, the slice sums
	p = multiprocessing.Pool(processes=nproc)
	p.map(caxpby, ([i, args[0], a, args[1], b, args[2]] for i in range(nslice)))

	return 0


if __name__ == "__main__":
	sys.exit (main ())
