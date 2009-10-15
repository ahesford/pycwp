#!/usr/bin/env python

import sys
import getopt
import fastsphere
import numpy

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-d dimension] -c <cmpfile> [...] -r <reffile>" % progname

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	reffile = None
	cmpfile = []
	dim = 2

	optlist, args = getopt.getopt (argv, 'r:c:d:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-r':
			reffile = opt[1]
		elif opt[0] == '-c':
			cmpfile.append (opt[1])
		elif opt[0] == '-d':
			dim = int(opt[1])
		else:
			usage (progname)
			return 128

	if reffile is None or len(cmpfile) == 0:
		usage (progname)
		return 128

	# Read the reference file
	ref = fastsphere.readbmat (reffile, dimen=dim)

	# Read each comparison file and report the MSE
	for idx, fpair in enumerate (cmpfile):
		cmp = fastsphere.readbmat (fpair, dimen=dim)
		print "MSE for file %d: %0.6g" % (idx, fastsphere.mse (cmp, ref))

	return 0

if __name__ == "__main__":
	sys.exit (main ())
