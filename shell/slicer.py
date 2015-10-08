#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os
import getopt
import numpy as np

from pycwp import mio

def usage (progname = 'slicer.py'):
	binfile = os.path.basename(progname)
	print >> sys.stderr, "Usage:", binfile, "<slice> <inputfile> [outputfile]"


def slicer(slstr):
	'''
	Parse a slice string and return either an integer (for single indices)
	or a slice object.
	'''

	iargs = [int(s) if len(s) > 0 else None for s in slstr.split(':')]
	if len(iargs) < 2: return iargs[0]
	return slice(*iargs)


def main (argv = None):
	if argv is None: argv = sys.argv[:]
	
	progname = argv.pop(0)

	if len(argv) < 2:
		usage(progname)
		return 128

	# Grab the desired slice notation
	slicestr = argv.pop(0)
	# Convert the slice notation to actual slice types
	slices = [slicer(s) for s in slicestr.split(',')]

	# Grab and slice the matrix from the desired input file
	mat = mio.readbmat(argv.pop(0))[slices]

	# Write the output to the desired file or to stdout
	mio.writebmat(mat, argv[0] if len(argv) > 0 else sys.stdout)

	return 0

if __name__ == "__main__":
	sys.exit (main ())
