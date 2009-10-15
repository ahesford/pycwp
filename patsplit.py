#!/usr/bin/env python

import sys
import fastsphere
import numpy

def usage (progname = 'patsplit.py'):
	print "Usage: %s [-h] <file> [x,y]" % progname
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
	mat = fastsphere.readbmat (fname)

	# Reshape each column and write it
	for i in range(mat.shape[1]):
		col = mat[:,i]
		if shape is not None:
			col = col.reshape (shape, order = 'F')
		fastsphere.writebmat (col, fname + '.%03d' % i)

	return 0

if __name__ == "__main__":
	sys.exit (main ())
