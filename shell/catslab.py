#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, os
import getopt
import numpy as np

from pycwp import mio

def usage (progname = 'catslab.py'):
	binfile = os.path.basename(progname)
	print >> sys.stderr, "Usage:", binfile, "[-h] [-t xl,xh,yl,yh] [-o output] <slab1> [...] <slabN>"

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	optlist, args = getopt.getopt (argv, 'ht:o:')
	trunc = False
	# By default, use stdout
	output = None

	# Parse the option list
	for opt in optlist:
		if opt[0] == '-t':
			trunc = True
			xmin, xmax, ymin, ymax = [int(l) for l in opt[1].split(',')]
		if opt[0] == '-o':
			output = opt[1]
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

	# If an output file was specified, create it
	if output: output = open(output, 'wb')
	# Otherwise, just use stdout
	else: output = sys.stdout

	# Write the matrix size to the output
	np.array(shape, dtype=np.int32).tofile(output)

	# Write the first slab to the output
	slab.flatten('F').tofile(output)

	# Loop through all remaining slabs, converting data types along the way
	for name in args:
		slab = np.array(mio.readbmat(name), dtype=dtype)
		if trunc: slab = slab[xmin:xmax,ymin:ymax]
		slab.flatten('F').tofile(output)

	output.close()
	return 0

if __name__ == "__main__":
	sys.exit (main ())
