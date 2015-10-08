#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os
import numpy

from pycwp import mio

def usage (progname = 'patsplit.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] <file> [x,y]"
	print "\tSplit columns of two-dimensional matrix <file> into separate files"
	print "\tOptional shape specification x,y reshapes each column as a 2-D matrix."

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# At least the
	if len(argv) < 1:
		usage (progname)
		return 128

	# Don't specify a shape by default
	shape = None

	# Pull the file name
	fname = argv.pop (0)

	if len(argv) > 0:
		shape = tuple ([int(x) for x in argv[0].split(',')])

	# Read the matrix
	mat = mio.readbmat (fname)

	# Reshape each column and write it
	for i, col in enumerate(mat.transpose()):
		if shape is not None:
			col = col.reshape (shape, order = 'F')
		mio.writebmat (col, fname + '.%03d' % i)

	return 0

if __name__ == "__main__":
	sys.exit (main ())
