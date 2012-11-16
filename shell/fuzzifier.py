#!/usr/bin/env python

import sys, os, numpy as np, getopt
from multiprocessing import Process, cpu_count
from pyajh import mio, cutil

def usage(progname = 'fuzzifier.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-p p] [-n n] [-c c] <input> <output>"
	print
	print "\t-h: Display this message and exit"
	print "\t-p: Use p processors (default: CPU count)"
	print "\t-g: Use a neighborhood of n (default: 5)"
	print "\t-c: Process output c slices at a time (default: 8)"

def fuzzyblks(infile, outfile, nbr, start, stride, chunk):
	'''
	Open the input file infile, output file outfile (which should exist)
	and loop through the input in strides to fuzzify boundarys in the input
	using cutil.fuzzyimg with a neighborhood nbr.

	The arguments start and stride refer to chunks rather than slices.
	'''
	# Open the files
	inmat = mio.Slicer(infile)
	outmat = mio.Slicer(outfile)
	# Compute the one-sided pad depth
	pad = (nbr - 1) / 2

	# Loop through the chunks to process output
	for n in range(start * chunk, inmat.shape[-1], stride * chunk):
		print 'Processing chunk', n
		# Read the chunk
		start = max(0, n - pad)
		finish = min(n + chunk + pad, inmat.shape[-1])
		block = inmat[start:finish]
		# Fuzzify the block
		outblk = cutil.fuzzyimg(block, nbr)
		# Figure out the proper slices of the output block
		istart = n - start
		iend = min(chunk + istart, istart + outmat.shape[-1] - n)
		# Figure out how many slices need to be written
		oend = n + iend - istart
		# Write the block to output, automatically converting types
		outmat[n:oend] = outblk[:,:,istart:iend]


def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Default values
	try: nproc = cpu_count()
	except NotImplementedError: nproc = 1
	chunk, nbr = 8, 5

	optlist, args = getopt.getopt (argv, 'p:c:n:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-p': nproc = int(opt[1])
		elif opt[0] == '-c': chunk = int(opt[1])
		elif opt[0] == '-n': nbr = int(opt[1])
		else:
			usage (progname)
			return 128

	# The input and output files must be specified
	if len(args) < 2:
		usage (progname)
		return 128

	# Grab the shape of the input file and the number of slices
	infile = mio.Slicer(args[0])
	# The output file must be created and truncated
	outfile = mio.Slicer(args[1], infile.shape, infile.dtype, True)

	try:
		procs = []
		for n in range(nproc):
			args = (args[0], args[1], nbr, n, nproc, chunk)
			p = Process(target=fuzzyblks, args=args)
			p.start()
			procs.append(p)
		for p in procs: p.join()
	except:
		outfile._backer.truncate(0)
		for p in procs:
			p.terminate()
			p.join()
		raise

	return 0

if __name__ == "__main__":
	sys.exit (main ())
