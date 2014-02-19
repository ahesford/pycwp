#!/usr/bin/env python

import sys, os, numpy as np, getopt
from scipy.ndimage.filters import gaussian_filter1d
from pyajh import mio, process

def usage(progname = 'gaussfilt.py'):
	binfile = os.path.basename(progname)
	print "Usage:", binfile, "[-h] [-p p] [-g w,s] [-c c] [-b b] <input> <output>"
	print
	print "\t-h: Display this message and exit"
	print "\t-p: Use p processors (default: CPU count)"
	print "\t-g: Use Gaussian of half width w, stdev. s (default: 24,8)"
	print "\t-b: Assume a constant background value b (default: 0)"
	print "\t-c: Process output c slices at a time (default: 8)"

def filtblks(infile, outfile, stdev, pad, bgv, start, stride, chunk):
	'''
	Open the input file infile, output file outfile (which should exist)
	and loop through the input in strides to filter with a Fourier Gaussian
	of standard deviation stdev. The input is padded by twice the value pad
	and is assumed to have a homogeneous background value bgv.

	The arguments start and stride refer to chunks rather than slices.
	'''
	# Open the files
	inmat = mio.Slicer(infile)
	outmat = mio.Slicer(outfile)

	# Loop through the chunks to process output
	for n in range(start * chunk, inmat.shape[-1], stride * chunk):
		print 'Processing chunk', n
		# Read the chunk
		start = max(0, n - pad)
		finish = min(n + chunk + pad, inmat.shape[-1])
		block = inmat[start:finish]
		outblk = np.zeros_like(block)
		# Filter the block along the three dimensions successively
		gaussian_filter1d(block, stdev, axis=0, output=outblk, 
				mode='constant', cval=bgv)
		gaussian_filter1d(outblk, stdev, axis=1, output=block, 
				mode='constant', cval=bgv)
		gaussian_filter1d(outblk, stdev, axis=2, output=outblk, 
				mode='constant', cval=bgv)
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
	nrpoc = process.preferred_process_count()
	chunk, stdev, pad, bgval = 8, 8, 24, 0.

	optlist, args = getopt.getopt (argv, 'p:c:g:b:h')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-p': nproc = int(opt[1])
		elif opt[0] == '-c': chunk = int(opt[1])
		elif opt[0] == '-b': bgval = float(opt[1])
		elif opt[0] == '-g':
			kstr = opt[1].split(',')
			pad = int(kstr[0])
			stdev = float(kstr[1])
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
		with process.ProcessPool() as pool:
			for n in range(nproc):
				args = (args[0], args[1], stdev,
						pad, bgval, n, nproc, chunk)
				pool.addtask(target=filtblks, args=args)
			pool.start()
			pool.wait()
	except:
		outfile._backer.truncate(0)
		raise

	return 0

if __name__ == "__main__":
	sys.exit (main ())
