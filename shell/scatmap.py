#!/usr/bin/env python

import sys, os, re, numpy as np
from mpi4py import MPI

from pyajh import mio

def localfiles(dir):
	'''
	Return a list of tuples corresponding to all files in the directory dir
	whose name matches the regexp "^.*facet[0-9]+\.src[0-9]+\.ScattMeas$".
	Each tuple contains the number after facet, the number after src, and
	the full path to the file.
	'''
	# Build the regexp and filter the list of files in the directory
	regexp = re.compile(r'^.*facet([0-9]+)\.src([0-9]+)\.ScattMeas$')
	return [(int(m.group(1)), int(m.group(2)), os.path.join(dir, f))
			for f in os.listdir(dir) for m in [regexp.match(f)] if m]


if __name__ == "__main__":
	# Ensure the argument list is appropriate
	if len(sys.argv) < 3:
		sys.exit('USAGE: %s <srcdir> <scatmap>' % sys.argv[0])

	srcdir = sys.argv[1]
	output = sys.argv[2]

	# Grab the list of locally available files, sort by index
	locfiles = localfiles(srcdir)

	# Grab all of the input files and concatenate them into a 2-D matrix
	a = np.concatenate([mio.readbmat(f[-1])[:,np.newaxis] for f in locfiles], axis=-1)

	# Gather a list of source indices on the root
	sources = MPI.COMM_WORLD.gather([f[:-1] for f in locfiles])
	# Flatten the per-host list of sources
	if sources: sources = [src for hsrcs in sources for src in hsrcs]

	# Gather all of the local matrices on the root
	fullmat = MPI.COMM_WORLD.gather(a)
	if fullmat is not None:
		# Sort the columns of the combined matrix by facet-element pair
		idx = sorted(range(len(sources)), key=lambda i: sources[i])
		# Concatenate the local matrices to build a scattering map
		# Use the sorted indices to reorder the columns
		fullmat = np.concatenate(fullmat, axis=-1)[:,idx]
		# Write the matrix
		mio.writebmat(fullmat, output)
