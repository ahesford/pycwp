#!/usr/bin/env python

import sys
import getopt
from numpy import *

import pylab
import fastsphere

def usage (progname = 'elevbeam.py'):
	print 'Usage: %s [-h] [-p] [-f freq] [-c speed] [-a phi]' % progname + \
		'[-w aperture] [-l length] [-z offset]'
	print '\t-h: Print this message'
	print '\t-p: Plot the amplitudes'
	print '\t-f: Excitation frequency in MHz (default: 2.5)'
	print '\t-c: Sound speed in m/s (default: 1509)'
	print '\t-a: Azimuthal angle phi in degrees (default: 0.0)'
	print '\t-w: Aperture width in m (default: 15e-3)'
	print '\t-l: Focus length in m (default: 75e-3)'
	print '\t-z: Focus elevation offset in m (default: 0)'

def main (argv = None):
	if argv is None:
		argv = sys.argv[1:]
		progname = sys.argv[0]

	# Set some defaults
	f, c, w, l, z, plot, phi = 2.5e6, 1509, 15e-3, 75e-3, 0, False, 0.0

	optlist, args = getopt.getopt (argv, 'hpf:c:w:l:z:')

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
		else:
			usage (progname)
			return 128
	
	(theta, T) = fastsphere.focusedbeam (f, c, w, l, z)
	theta = [thr * 180 / pi for thr in theta]

	# Print the angles in ascending order
	for c, t in zip(T[::-1], theta[::-1]):
		print "%0.10f %0.10f %0.10f %0.10f" % (real(c), imag(c), t, phi)

	if plot:
		pylab.plot (theta, real(T), theta, imag(T), '--', theta, abs(T), '--o')
		pylab.xlabel (r'$\theta$, rad')
		pylab.ylabel ('Amplitude')
		pylab.grid (True)
		pylab.legend (('Real', 'Imaginary', 'Magnitude'), loc='best')
		pylab.show ()

if __name__ == "__main__":
	sys.exit (main())
