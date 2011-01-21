#!/usr/bin/env python

import sys
import getopt
import numpy as np
from itertools import izip

from pyajh import mio, cutil

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-n] <cmpfile> [...] <reffile>" % progname


def filemax (infile):
	'''
	Perform a slice-by-slice scan of infile and identify the maximum value.
	'''

	return cutil.complexmax(np.array([cutil.complexmax(s)
		for idx, s in mio.readslicer(infile)]))


def errslice (infiles, nfacts):
	'''
	Perform a slice-by-slice comparison of the inputs infiles. If nfacts is
	provided for each file and is not None, the data is normalized by the
	corresponding factor. The last file is the reference.
	'''

	err = np.array([0.] * len(infiles[:-1]))
	den = 0.

	# Prepare the file-slicer generators
	slgens = [mio.readslicer(fn) for fn in infiles]

	# Loop through each slice as it is read
	for slices in izip(*slgens):
		# Strip out the slice index
		data = [s[1] for s in slices]

		# Normalize each chunk, if desired
		if nfacts is not None:
			data = [d / nf for d, nf in izip(data, nfacts)]

		# The last file is the reference
		ref = data[-1]

		# Update the numerator and denominator
		err += np.array([np.sum(np.abs(d - ref)**2) for d in data[:-1]])
		den += np.sum(np.abs(ref)**2)

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

	# Attempt to open all files and grab the matrix dimensions and types
	msizes = [mio.getmattype(open(name, 'rb'))[0] for name in args]

	# Check that the matrix shapes match
	for mf, ms in zip(args, msizes[:-1]):
		if ms.tolist() != msizes[-1].tolist():
			raise IndexError('File %s has wrong shape' % mf)

	# Find the normalizing factors, if desired
	if normalize: nfacts = [filemax(mf) for mf in args]
	else: nfacts = None

	# Compute the MSE for each file relative to the reference
	err = errslice (args, nfacts)

	# Report the MSE for each pair
	for idx, e in enumerate (err):
		print idx, e

	return 0

if __name__ == "__main__":
	sys.exit (main ())
