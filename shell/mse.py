#!/usr/bin/env python

import sys, os, getopt, numpy as np, multiprocessing
from numpy import linalg as la
from itertools import izip

from pyajh import mio, cutil

def usage(progname = 'mse.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-s] [-p nproc] <cmpfile> [...] <reffile>"


def slicerr(args):
	'''
	For an argument list
	
		args = (i, <file1>, [...], <ref>),

	return the Frobenius error of the differences between slice i in each
	file and the reference. The last returned value is the Frobenius norm
	of slice i of the reference.
	'''
	i = args[0]
	files = [mio.Slicer(a) for a in args[1:]]
	# Compute the squared magnitude sums
	df = [la.norm(f[i] - files[-1][i]) for f in files[:-1]]
	# Add the reference norm
	df += [la.norm(files[-1][i])]

	return df


def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	perslice = False
	try: nproc = multiprocessing.cpu_count()
	except NotImplementedError: nproc = 1

	optlist, args = getopt.getopt (argv, 'p:sh')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-s': perslice = True
		elif opt[0] == '-p': nproc = int(opt[1])
		else:
			usage (progname)
			return 128

	# There must be at least two files to compare
	if len(args) < 2:
		usage (progname)
		return 128

	p = multiprocessing.Pool(processes=nproc)

	# Grab the number of slices in the reference file
	nslice = mio.Slicer(args[-1]).shape[-1]

	# Compute the error norms
	errs = np.array(p.map(slicerr, (tuple([i] + args) for i in range(nslice))))

	errs = errs[:,:-1] / la.norm(errs[:,-1])

	if perslice:
		for erow in errs: print ' '.join('%-11.6e' % ev for ev in erow)
	else:
		for i, ecol in enumerate(errs.T):
			print '%4d %11.6e' % (i, la.norm(ecol))

	return 0

if __name__ == "__main__":
	sys.exit (main ())
