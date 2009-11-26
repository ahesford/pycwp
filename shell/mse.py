#!/usr/bin/env python

import sys
import getopt
import numpy

from pyajh import mio, cutil

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-d dimension] [-n] -c <cmpfile> [...] -r <reffile>" % progname

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	reffile = None
	cmpfile = []
	dim = 2
	normalize = False

	optlist, args = getopt.getopt (argv, 'nr:c:d:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-r':
			reffile = opt[1]
		elif opt[0] == '-c':
			cmpfile.append (opt[1])
		elif opt[0] == '-d':
			dim = int(opt[1])
		elif opt[0] == '-n':
			normalize = True
		else:
			usage (progname)
			return 128

	if reffile is None or len(cmpfile) == 0:
		usage (progname)
		return 128

	# Read the reference file
	ref = mio.readbmat (reffile, dimen=dim)
	if normalize: ref = ref / cutil.complexmax (ref)

	# Read each comparison file and report the MSE
	for idx, fpair in enumerate (cmpfile):
		cmp = mio.readbmat (fpair, dimen=dim)
		if normalize: cmp = cmp / cutil.complexmax (cmp)
		print cutil.mse (cmp, ref),

	print ""

	return 0

if __name__ == "__main__":
	sys.exit (main ())
