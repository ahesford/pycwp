#!/usr/bin/env python

import sys
import getopt
import numpy as np

from pyajh import mio, cutil

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-n] <cmpfile> [...] <reffile>" % progname


def chunkmax (infile, dtype, csize, nchunk):
	'''
	Perform a chunk-by-chunk scan of the file infile to identify the
	maximum value. Seek to the beginning of the data block when finished.
	'''

	maxval = 0.0

	fpos = infile.tell()

	for i in range(nchunk):
		vals = np.fromfile(infile, dtype=dtype, count=csize)
		maxval = cutil.complexmax(np.array([maxval, cutil.complexmax(vals)]))

	infile.seek(fpos)
	return maxval


def chunkerr (infiles, nfacts, csize, nchunk):
	'''
	Perform a chunk-by-chunk comparison of the files. If nfacts is
	provided for each file and is not None, the data is normalized
	by the corresponding factor. The last file is the reference.
	'''

	err = np.array([0.] * len(infiles[:-1]))
	den = 0.

	for i in range(nchunk):
		# Read data chunks from all of the files
		vals = [np.fromfile(mf, dtype=mt, count=csize) for ms, mt, mf in infiles]

		# Normalize each chunk
		if nfacts is not None:
			vals = [v / nf for v, nf in zip(vals, nfacts)]

		# Update the numerator and denominator
		err += np.array([np.sum(np.abs(v - vals[-1])**2) for v in vals[:-1]])
		den += np.sum(np.abs(vals[-1])**2)

	# Return the error
	return np.sqrt(err / den)

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	normalize = False

	optlist, args = getopt.getopt (argv, 'nh')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-n':
			normalize = True
		else:
			usage (progname)
			return 128

	# There must be at least two files to compare
	if len(args) < 2:
		usage (progname)
		return 128

	# Attempt to open all files for comparison and grab the matrix types
	mfiles = [open(name, 'rb') for name in args]
	mtypes = [mio.getmattype(f) for f in mfiles]

	# Stitch together the file and type information
	mtypes = [(t[0], t[1], f) for f, t in zip(mfiles, mtypes)]

	# Check that the matrix sizes match
	for ms, mt, mf in mtypes[:-1]:
		if ms.tolist() != mtypes[-1][0].tolist():
			raise IndexError('File %s does not match shape of file %s' %
					(mf.name, mtypes[-1][-1].name))

	# Grab the file chunk size as a slab in three dimensions or a column in two
	csize, nchunk = np.prod(mtypes[-1][0][:-1]), mtypes[-1][0][-1]

	# Find the normalizing factors, if desired
	if normalize: nfacts = [chunkmax(mf, mt, csize, nchunk) for ms, mt, mf in mtypes]
	else: nfacts = None

	# Compute the MSE for each file relative to the reference
	err = chunkerr (mtypes, nfacts, csize, nchunk)

	# Report the MSE for each pair
	for idx, e in enumerate (err):
		print idx, e

	return 0

if __name__ == "__main__":
	sys.exit (main ())
