#!/usr/bin/env python

import sys, os, math, getopt, numpy as np, operator
from itertools import izip
from multiprocessing import Process, Queue

from pyajh import mio, cutil

def usage(progname = 'mse.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-n] [-p nproc] <cmpfile> [...] <reffile>"


def complexmax(mat, start, stride):
	'''
	For a binary matrix stored Slicer mat with N slices, find the complex
	value of largest magnitude in the slices in range(start, N, stride).
	'''
	N = mat.shape[-1]
	cmx = [cutil.complexmax(mat[i]) for i in range(start, N, stride)]
	return cutil.complexmax(np.array(cmx))


def sqerr(mats, nfacts, start, stride):
	'''
	For binary matrix Slicers mats with N slices each, compute the sum of
	the squares of the magnitudes of the differences between the slice
	given by range(start, N, stride) of each matrix with the reference
	stored in the final entry.  The "difference" between the last entry and
	the reference (which is the same matrix) is really the sum of the
	squares of the magnitudes of the elements of the reference in the
	chunk.

	Elements of the list nfacts that are not None are assumed to be
	normalizing factors for the corresponding matrices.
	'''
	N = mats[-1].shape[-1]
	# The initial sums are all zero
	df = [0.] * len(mats)
	for i in range(start, N, stride):
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


def errchunk(fnames, start, stride, qsol, qnorm = None):
	'''
	Compute the squared error contribution from one process for each of the
	matrices stored in the files fnames.

	Each process handles the slice in range(start, N, stride), where N is
	the number of Slices. The multiprocessing queue qsol is used to store
	the output for aggregation by the master process. If the queue qnorm is
	not None, each matrix is normalized by its largest (in the sense of
	magnitude) complex value. The queue qnorm is used to hold the
	process-local maximum value, which is reduced by the master process and
	placed on the qsol queue before computing the error.
	'''
	# Open all of the files
	mats = [mio.Slicer(f) for f in fnames]

	# Check that the file sizes agree
	for m, f in zip(mats[:-1], fnames[:-1]):
		if list(m.shape) != list(mats[-1].shape):
			# Put a None on the solution queue
			qsol.put(None)
			# Raise an exception
			raise IndexError('Shape of file %s differs from reference' % f)

	# Find the complex value with the largest magnitude in each file
	if qnorm is not None:
		fmax = [complexmax(m, start, stride) for m in mats]
		qsol.put(fmax)
		fmax = qnorm.get()
	else: fmax = [None] * len(mats)

	df = sqerr(mats, fmax, start, stride)
	qsol.put(df)


def erreduce(fnames, normalize = False, nproc = 1):
	'''
	Fork nproc processes to handle local portions of the RMS error
	calculation, normalizing the files if desired. This requires reduction
	of the process-local file maxima through a two-queue exchange.

	The final error contribution of each process is collected in a queue
	and reduced to a single solution.
	'''
	# Create the appropriate queues
	if normalize: qnorm = Queue()
	else: qnorm = None
	qsol = Queue()
	# Create and start the processes
	procs = [Process(target=errchunk,
		args=(fnames, s, nproc, qsol, qnorm,)) for s in range(nproc)]
	for p in procs: p.start()

	# Read the qsol for normalization factors or differences
	df = []
	for s in range(nproc):
		df.append(qsol.get())
		# Check for failures from the children
		if df[-1] is None: raise ValueError('Unable to compute RMS errors')

	if normalize:
		# Reduce the normalization factors
		fmax = reduce(lambda x, y: [xv if abs(xv) > abs(yv) else yv
			for xv, yv in zip(x, y)], df)
		# Put the result on the normalization queue for each process
		for s in range(nproc): qnorm.put(fmax)
		# Read the normalized differences
		df = [qsol.get() for s in range(nproc)]

	# Reduce the process-local file errors
	df = reduce(lambda x, y: map(operator.add, x, y), df)
	# Now compute the per-file errors
	errs = [math.sqrt(dfn / df[-1]) for dfn in df[:-1]]

	# Join the processes to make sure that they've quit
	for p in procs: p.join()
	return errs


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

	# Start computation
	errs = erreduce(args, normalize, nproc)
	for idx, err in enumerate(errs): print idx, err

	return 0

if __name__ == "__main__":
	sys.exit (main ())
