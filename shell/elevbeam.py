#!/usr/bin/env python

import sys, os
import getopt
from numpy import *

import pylab

from pycwp import focusing, mio, scattering

def usage (progname = 'elevbeam.py'):
	binfile = os.path.basename(progname)
	print 'Usage:', binfile, '[-h] [-p] [-f freq] [-c speed] [-a phi] [-w aperture] [-l length] [-z offset] [-s shift] [-r scatpat] [-o output]'
	print '\t-h: Print this message'
	print '\t-p: Plot the amplitudes'
	print '\t-f: Excitation frequency in MHz (default: 2.5)'
	print '\t-c: Sound speed in m/s (default: 1509)'
	print '\t-a: Azimuthal angle phi in degrees (default: 0.0)'
	print '\t-w: Aperture width in m (default: 15e-3)'
	print '\t-l: Focus length in m (default: 75e-3)'
	print '\t-z: Focus elevation offset in m (default: 0)'
	print '\t-s: Phase shift to move focus in m (default: 0)'
	print '\t-r: Receive focusing on pattern listed in scatpat'
	print '\t-o: Specify an output file (default: stdout)'

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Set some defaults
	f, c, w, l, z, s = 2.5e6, 1509, 15e-3, 75e-3, 0.0, 0.0
	phi, plot, scatpat, output = 0.0, False, None, None

	optlist, args = getopt.getopt (argv, 'hpf:c:w:l:z:s:r:o:')

	# Parse the options list
	for opt in optlist:
		if opt[0] == '-f':
			f = float(opt[1]) * 1e6
		elif opt[0] == '-p':
			plot = True
		elif opt[0] == '-c':
			c = float(opt[1])
		elif opt[0] == '-w':
			w = float(opt[1])
		elif opt[0] == '-l':
			l = float(opt[1])
		elif opt[0] == '-z':
			z = float(opt[1])
		elif opt[0] == '-a':
			phi = float(opt[1])
		elif opt[0] == '-s':
			s = float(opt[1])
		elif opt[0] == '-r':
			scatpat = opt[1]
		elif opt[0] == '-o':
			output = opt[1]
		else:
			usage (progname)
			return 128
	
	(theta, T) = focusing.focusedbeam (f, c, w, l, s, z)
	theta = [thr * 180 / pi for thr in theta]

	if scatpat is None:
		if output: output = file (output, 'w')
		else: output = sys.stdout

		# Print the angles in ascending order
		for c, t in zip(T[::-1], theta[::-1]):
			print >>output, "%0.10f %0.10f %0.10f %0.10f" % (real(c), imag(c), t, phi)
	else:
		# Read the scattering pattern in the provided file
		ref, thr = scattering.readradpat(scatpat)[0:2]
		# Conjugate the focusing coefficients
		R = [c.conjugate() for c in T]
		fld = focusing.recvfocus (ref, thr, R, theta)

		# Print the focused pattern
		if output is None:
			print fld
		else:
			mio.writebmat (fld, output)

	if plot:
		pylab.plot (theta, real(T), theta, imag(T), '--', theta, abs(T), '--o')
		pylab.xlabel (r'$\theta$, rad')
		pylab.ylabel ('Amplitude')
		pylab.grid (True)
		pylab.legend (('Real', 'Imaginary', 'Magnitude'), loc='best')
		pylab.show ()

if __name__ == "__main__":
	sys.exit (main())
