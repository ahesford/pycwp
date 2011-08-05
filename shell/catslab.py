#!/usr/bin/env python

import sys
import getopt
import numpy as np

from pyajh import mio

def usage (progname = 'catslab.py'):
	print >> sys.stderr, "Usage: %s [-h] [-t xl,xh,yl,yh] <slab1> [...] <slabN>" % progname

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	optlist, args = getopt.getopt (argv, 'ht:')
	trunc = False

	# Parse the option list
	for opt in optlist:
		if opt[0] == '-t':
			trunc = True
			xmin, xmax, ymin, ymax = tuple([int(l) 
				for l in opt[1].split(',')])
		else:
			usage(progname)
			return 128

	# There must be at least two files to concatenate
	if len(args) < 2:
		usage (progname)
		return 128

	# Grab the number of slabs, the slab dimensions
	# and data type from the first file.
	nslabs = len(args)

	# Grab the first slab and its data type, truncating if necessary
	slab = mio.readbmat(args.pop(0))
	if trunc: slab = slab[xmin:xmax,ymin:ymax]
	dtype = slab.dtype

	# Get the overall matrix size
	shape = list(slab.shape)
	shape.append(nslabs)

	# Write the matrix size to the output
	np.array(shape, dtype=np.int32).tofile(sys.stdout)

	# Write the first slab to the output
	slab.flatten('F').tofile(sys.stdout)

	# Loop through all remaining slabs, converting data types along the way
	for name in args:
		slab = np.array(mio.readbmat(name), dtype=dtype)
		if trunc: slab = slab[xmin:xmax,ymin:ymax]
		slab.flatten('F').tofile(sys.stdout)

	sys.stdout.close()
	return 0

if __name__ == "__main__":
	sys.exit (main ())
