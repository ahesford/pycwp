#!/usr/bin/env python

import sys
import getopt
import numpy as np

from pyajh import mio

def usage (progname = 'gridup.py'):
	print >> sys.stderr, "Usage: %s <inputfile> [outputfile]" % progname


def main (argv = None):
	if argv is None: argv = sys.argv[:]
	
	progname = argv.pop(0)

	if len(argv) < 1:
		usage(progname)
		return 128

	# Open the input file and get the matrix size and type
	infile = open(argv.pop(0), 'rb')
	msize, dtype = mio.getmattype(infile)
	# Rewind the file for slicing
	infile.seek(0)

	# Compute the size of the larger grid
	bsize = msize * 2

	if len(argv) < 1 or argv[0] == '-': outfile = sys.stdout
	else: outfile = open(argv[0], 'wb')

	# Write the larger size to the file
	bsize.astype(np.int32).tofile(outfile)

	bslab = np.zeros(bsize[:-1], dtype=dtype)
	dims = len(bslab.shape)

	# Prepare the slice specifiers to duplicate the grid values
	sldim = [slice(None, None, 2), slice(1, None, 2)]
	slices = [[sldim[(i & 2**d) >> d] for d in range(dims)] for i in range(2**dims)]

	for slab in mio.ReadSlicer(infile):
		# Duplicate the slab values along each possible dimension
		for sl in slices:
			bslab[sl] = slab[1]

		# Write the slab twice
		bslab.flatten('F').tofile(outfile)
		bslab.flatten('F').tofile(outfile)
	
	outfile.close()

	return 0

if __name__ == "__main__":
	sys.exit (main ())
