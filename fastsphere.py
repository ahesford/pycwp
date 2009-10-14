'''
Routines for parsing and manipulating the radiation pattern files produced by
the fastsphere solver. Data files are complex double matrices in two
dimensions, with uniform phi samples along the row dimension and Gaussian
theta samples along the column dimension.
'''

import numpy
import math

import matplotlib.pyplot as plt

def readbmat (fname, type = numpy.complex128, dimen = 2):
	'''
	Read a binary, complex matrix file of the specified type and dimensionality.
	'''
	infile = open (fname, mode='rb')

	# Read the dimension specification from the file
	dimspec = numpy.fromfile (infile, dtype=numpy.int32, count=dimen)

	# Conveniently precompute the number of elements to read
	nelts = numpy.prod (dimspec)

	# Read the binary values into the array
	data = numpy.fromfile (infile, dtype = type, count = nelts)

	# Rework the array as a numpy array with the proper shape
	data = data.reshape (dimspec, order = 'F')

	return data

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

def gaussleg (m, tol = 1e-9):
	'''
	Compute the Gaussian nodes in the interval [0,pi] and corresponding weights
	for a specified order.
	'''
	def legendre (t, m):
		p0 = 1.0; p1 = t
		for k in range(1,m):
			p = ((2.0*k + 1.0) * t * p1 - k * p0) / (1.0 + k)
			p0 = p1; p1 = p
		dp = m * (p0 - t * p1) / (1.0 - t**2)
		return p, dp

	weights = numpy.zeros ((m), dtype=numpy.float64)
	nodes = numpy.zeros ((m), dtype=numpy.float64)

	nRoots = (m + 1) / 2

	for i in range(nRoots):
		t = math.cos (math.pi * (i + 0.75) / (m + 0.5))
		for j in range(30):
			p,dp = legendre (t, m)
			dt = -p/dp; t += dt
			if abs(dt) < tol:
				nodes[i] = math.acos(t)
				nodes[m - i - 1] = math.acos(-t)
				weights[i] = 2.0 / (1.0 - t**2) / (dp**2)
				weights[m - i - 1] = weights[i]
				break
	return nodes, weights

def complexmax (a, order = 'C'):
	'''
	Compute the maximum element of a complex array like MATLAB does.
	'''
	maxidx = 0
	maxval = 0
	b = a.flatten (order)
	
	for i,v in enumerate(b):
		if abs(v) > maxval:
			maxval = abs(v)
			maxidx = i
	
	return b[maxidx]

def linpat (angle, pattern, column):
	'''
	Plot the pattern of a specified column using provided angle indices.
	'''
	# Normalize the pattern by the global maximum
	b = numpy.abs(pattern[:,column]) / numpy.amax (numpy.abs (pattern))

	return plt.semilogy (angle, b)

def mse (x, y):
	'''
	Report the mean squared error between the matrix x and the matrix y.
	'''
	err = numpy.sum (numpy.abs(x - y).flatten()**2)
	err /= numpy.sum (numpy.abs(y).flatten()**2)

	return math.sqrt(err)
