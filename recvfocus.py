#!/usr/bin/env python

import sys
import numpy
import scipy

import scipy.interpolate as intp

import fastsphere as fs

def recvfocus (mat, thr, coeffs, theta):
	'''
	Apply receive focusing to a 3-D samples, with polar angles given 
	by thr, to generate a 2-D pattern. The focusing coefficients are
	defined over the polar angles theta.
	'''

	# Reverse the coefficients if theta is non-increasing.
	if theta[1] < theta[0]:
		theta.reverse()
		coeffs.reverse()

	# Build the interpolation function.
	cfunc = intp.interp1d (theta, coeffs, kind='quadratic',
			bounds_error=False, fill_value=0.+0j)

	# Compute the new coefficients.
	nc = cfunc (thr)

	# Return the focused array.
	return numpy.dot(mat, nc)

def usage (progname = 'recvfocus.py'):
	print "Usage: %s <input> <output>"
	print "\t<input> is a scattering pattern matrix"
	print "\tFocusing coefficients specified on stdin as output of elevbeam.py"

if __name__ == "__main__":
	if len(sys.argv) < 3:
		usage(sys.argv[0])
		sys.exit (128)

	# Grab the scattering matrix and polar samples.
	# The azimuthal angles don't matter.
	ref, thr = fs.readradpat(sys.argv[1])[0:2]

	# Read the coefficients from stdin.
	theta, coeffs = [], []
	for line in sys.stdin.readlines():
		svals = line.split()
		theta.append (float(svals[2]))
		# Coefficients should be conjugated for receive.
		coeffs.append (complex(float(svals[0]), -float(svals[1])))

	# Focus the field.
	fld = recvfocus (ref, thr, coeffs, theta)

	# Write the field to the output.
	fs.writebmat (fld, sys.argv[2])
