#!/usr/bin/env python

import sys

from ajh import mio, focusing

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
	ref, thr = mio.readradpat(sys.argv[1])[0:2]

	# Read the coefficients from stdin.
	theta, coeffs = [], []
	for line in sys.stdin.readlines():
		svals = line.split()
		theta.append (float(svals[2]))
		# Coefficients should be conjugated for receive.
		coeffs.append (complex(float(svals[0]), -float(svals[1])))

	# Focus the field.
	fld = focusing.recvfocus (ref, thr, coeffs, theta)

	# Write the field to the output.
	mio.writebmat (fld, sys.argv[2])
