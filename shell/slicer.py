#!/usr/bin/env python

import sys
import getopt
import numpy as np

from pyajh import mio

def usage (progname = 'slicer.py'):
	print >> sys.stderr, "Usage: %s <slice> <inputfile> [outputfile]" % progname


def slicer(slstr):
	'''
	Parse a slice string and return either an integer (for single indices)
	or a slice object.
	'''

	iargs = [(len(s) > 0 and [int(s)] or [None])[0] for s in slstr.split(':')]
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
	mio.writebmat(mat, (len(argv) > 0 and [argv[0]] or [sys.stdout])[0])

	return 0

if __name__ == "__main__":
	sys.exit (main ())
