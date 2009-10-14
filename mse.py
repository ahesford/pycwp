#!/usr/bin/env python

import sys
import getopt
import fastsphere
import numpy

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-d dimension] <-c|-C> <cmpfile> [...] <-r|-R> <reffile>" % progname
	print "\t Lowercase flags for double precision, uppercase for single precision"

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	reffile = None
	cmpfile = []
	dim = 2

	optlist, args = getopt.getopt (argv, 'r:R:c:C:d:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-r':
			reffile = (opt[1], numpy.complex128)
		elif opt[0] == '-R':
			reffile = (opt[1], numpy.complex64)
		elif opt[0] == '-c':
			cmpfile.append ((opt[1], numpy.complex128))
		elif opt[0] == '-C':
			cmpfile.append ((opt[1], numpy.complex64))
		elif opt[0] == '-d':
			dim = int(opt[1])
		else:
			usage (progname)
			return 128

	if reffile is None or len(cmpfile) == 0 or len(args) > 0:
		usage (progname)
		return 128

	ref = fastsphere.readbmat (reffile[0], type=reffile[1], dimen=dim)

	for idx, fpair in enumerate (cmpfile):
		cmp = fastsphere.readbmat (fpair[0], type=fpair[1])
		print "MSE for file %d: %0.6g" % (idx, fastsphere.mse (cmp, ref))

	return 0

if __name__ == "__main__":
	sys.exit (main ())
