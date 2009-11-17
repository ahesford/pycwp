'''
Routines for reading and displaying scattering patterns in two or three
dimensions.
'''

import numpy
import math
import pylab

from ajh.mio import readbmat
from ajh.cutil import gaussleg

from matplotlib import ticker

def readradpat (fname, uniform = False):
	'''
	Read a complex double radiation pattern matrix and return the matrix with
	theta and phi samples. Theta can be uniform or Gaussian (default).
	'''
	# Grab the pattern
	pat = readbmat (fname)

	# Grab the number of angular samples in each dimension
	(nphi, ntheta) = pat.shape

	# Uniform samples in phi
	phi = 360.0 * numpy.arange (0, nphi) / float(nphi)

	if uniform:
		# Uniform samples in theta
		theta = (2.0 * numpy.arange (0, ntheta) + 1.0) * 180.0 / (2.0 * ntheta)
	else:
		# Gauss-Legendre samples in theta
		theta = gaussleg (ntheta)[0] * 180.0 / math.pi

	return pat, theta, phi

def linpat (angle, pattern, column):
	'''
	Plot the pattern of a specified column using provided angle indices.
	'''
	majorLocator = ticker.MultipleLocator (90)
	majorFormatter = ticker.FormatStrFormatter ('%d')
	minorLocator = ticker.MultipleLocator (15)

	# Normalize the pattern by the global maximum
	b = numpy.abs(pattern[:,column]) / numpy.amax (numpy.abs (pattern))

	# Plot the line and grab the line structure
	line = pylab.semilogy (angle, b)
	ax = pylab.gca ()

	# Set fancy ticking on the axis
	ax.xaxis.set_major_locator (majorLocator)
	ax.xaxis.set_major_formatter (majorFormatter)
	ax.xaxis.set_minor_locator (minorLocator)

	return line
