#!/usr/bin/env python

import sys, os
import getopt
import numpy as np
from itertools import izip

from pyajh import mio, cutil

def usage (progname = 'mse.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-n] <cmpfile> [...] <reffile>"


def filemax (mat):
	'''
	Perform a slice-by-slice scan of mat and identify the maximum value.
	'''
	return cutil.complexmax(np.array([cutil.complexmax(s) for i, s in mat]))


def errslice (mats, nfacts):
	'''
	Perform a slice-by-slice comparison of the matices mats. If nfacts is
	provided for each file and is not None, the data is normalized by the
	corresponding factor. The last file is the reference.
	'''

	err = np.array([0.] * (len(mats) - 1))
	den = 0.

	# Perform no normalization if nfacts was omitted
	if nfacts is None: nfacts = [1.] * len(mats)

	# Loop through each slice as it is read
	for slices in izip(*mats):
		# Strip out the slice index and normalize if appropriate
		data = [s[1] / nf for s, nf in izip(slices, nfacts)]

		# Update the numerator and denominator
		err += np.array([np.sum(np.abs(d - data[-1])**2) for d in data[:-1]])
		den += np.sum(np.abs(data[-1])**2)

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

	# Attempt to open all files
	mats = [mio.Slicer(name) for name in args]

	# Check that the matrix shapes match
	for mf, ms in zip(args, mats):
		if list(ms.shape) != list(mats[-1].shape):
			raise IndexError('File %s has wrong shape' % mf)

	# Find the normalizing factors, if desired
	if normalize: nfacts = [filemax(m) for m in mats]
	else: nfacts = None

	# Compute the MSE for each file relative to the reference
	err = errslice (mats, nfacts)

	# Report the MSE for each pair
	for idx, e in enumerate (err):
		print idx, e

	return 0

if __name__ == "__main__":
	sys.exit (main ())
