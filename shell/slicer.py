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

	# Grab the input matrix data and slice as desired
	mat = mio.readbmat(argv.pop(0))[slices]

	if len(argv) > 0:
		# Write the output file...
		mio.writebmat(mat, argv[0])
	else: 
		# ...or write to stdout
		np.array(mat.shape, dtype='int32').tofile(sys.stdout)
		mat.flatten('F').tofile(sys.stdout)
		sys.stdout.close()

	return 0

if __name__ == "__main__":
	sys.exit (main ())
