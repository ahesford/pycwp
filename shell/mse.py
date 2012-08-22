#!/usr/bin/env python

import sys, os
import getopt
import numpy as np
import operator
from itertools import izip
from multiprocessing import Pool

from pyajh import mio, cutil

def usage(progname = 'mse.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-n] [-p nproc] <cmpfile> [...] <reffile>"

def pool_apply(f, args, p, start, end):
	'''
	For a thread pool p, asynchronously apply the function f to the
	arguments args for each pair (s, e) in zip(start, end). The pair (s, e)
	will be appended to args.
	'''
	res = []
	for s, e in zip(start, end):
		exargs = tuple(list(args) + [s, e])
		res.append(p.apply_async(f, exargs))
	return [r.get() for r in res]


def pool_reduce(a, op):
	'''
	Reduce per-process lists of results to a global list of results by
	repeatedly applying the reduction operator op to corresponding entries
	of each process list.
	'''
	return reduce(lambda x, y: map(op, x, y), a)


def complexmax(fname, start, end):
	'''
	For a binary matrix stored in the file fname, find the complex value
	with the largest magnitude in the slices start:end (with end excluded).
	'''
	# Open all of the matrices
	m = mio.Slicer(fname)
	# This stores the complex max of all slices in the file
	cmx = [cutil.complexmax(m[i]) for i in range(start, end)]
	return cutil.complexmax(np.array(cmx))


def sqerr(fnames, nfacts, start, end):
	'''
	For binary matrices stored in files with names fnames, compute the 
	sum of the squares of the magnitudes of the differences between the
	chunk start:end (with end excluded) of each matrix with the reference
	stored in the final entry of fnames. The "difference" between the last
	entry of fnames and the reference (which is the same matrix) is really
	the sum of the squares of the magnitudes of the elements of the
	reference in the chunk.

	Elements of the list nfacts that are not None are assumed to be
	normalizing factors for the corresponding matrices in fnames.
	'''
	# Open all of the matrices
	mats = [mio.Slicer(f) for f in fnames]
	# The initial sums are all zero
	df = [0.] * len(mats)
	for i in range(start, end):
		# Grab the reference slice
		rs = mats[-1][i]
		# Normalize the reference if appropriate
		if nfacts[-1] is not None: rs /= nfacts[-1]
		# Add the square sum of the reference
		df[-1] += np.sum(np.abs(rs)**2)
		for idx, (m, nf) in enumerate(zip(mats[:-1], nfacts)):
			# Read the matrix slice
			ms = m[i]
			# Normalize if appropriate
			if nf is not None: ms /= nf
			# Add the square sum of the difference
			df[idx] += np.sum(np.abs(ms - rs)**2)
	return df


def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	normalize, nproc = False, 1

	optlist, args = getopt.getopt (argv, 'p:nh')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-n':
			normalize = True
		elif opt[0] == '-p':
			nproc = int(opt[1])
		else:
			usage (progname)
			return 128

	# There must be at least two files to compare
	if len(args) < 2:
		usage (progname)
		return 128

	# Grab the shape of the reference file and the number of slices
	rshape = list(mio.Slicer(args[-1]).shape)
	nslice = rshape[-1]
	# Check that the matrix shapes match
	for mf in args[:-1]:
		if list(mio.Slicer(mf).shape) != rshape:
			raise IndexError('File %s has wrong shape' % mf)

	# Create the worker pool
	p = Pool(processes=nproc)
	# Determine the work share, starting and ending indices
	share = lambda i: (nslice / nproc) + (1 if i < nslice % nproc else 0)
	# Compute the starting and ending slice counts
	starts = [0]
	ends = [share(0)]
	for i in range(1, nproc):
		starts.append(ends[i - 1])
		ends.append(starts[i] + share(i))

	# Find the normalizing factors, if desired
	if normalize:
		maxvals = []
		for a in args:
			mv = pool_apply(complexmax, (a,), p, starts, ends)
			maxvals.append(cutil.complexmax(np.array(mv)))
	else: maxvals = [None] * len(args)

	# Compute the numerators and denominators for the RMS errors
	df = pool_apply(sqerr, (args, maxvals), p, starts, ends)
	df = reduce(lambda x, y: map(operator.add, x, y), df)

	for idx, dfn in enumerate(df[:-1]):
		# Compute the numerator
		print idx, np.sqrt(dfn / df[-1])

	return 0

if __name__ == "__main__":
	sys.exit (main ())
