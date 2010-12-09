#!/usr/bin/env python

import sys
import getopt
import numpy

from pyajh import mio, cutil

def usage (progname = 'mse.py'):
	print "Usage: %s [-h] [-n] <cmpfile> [...] <reffile>" % progname

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	reffile = None
	cmpfile = None
	normalize = False

	optlist, args = getopt.getopt (argv, 'nh')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-n':
			normalize = True
		else:
			usage (progname)
			return 128

	# The last argument, if it exists, is the reference file
	try: reffile = args[-1]
	except IndexError: reffile = None

	# All other non-flag arguments are for comparison
	cmpfile = args[:-1]

	if reffile is None or len(cmpfile) == 0:
		usage (progname)
		return 128

	# Read the reference file
	ref = mio.readbmat (reffile)
	if normalize: ref = ref / cutil.complexmax (ref)

	# Read each comparison file and report the MSE
	for idx, fpair in enumerate (cmpfile):
		cmp = mio.readbmat (fpair)
		if normalize: cmp = cmp / cutil.complexmax (cmp)
		print idx, cutil.mse (cmp, ref)

	print ""

	return 0

if __name__ == "__main__":
	sys.exit (main ())
