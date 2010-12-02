'''
General-purpose numerical routines used in other parts of the module.
'''

import numpy
import math
from scipy import special as spec
from pyajh.geom import sph2cart

def translator (r, s, phi, theta, l):
	'''
	Compute the diagonal translator for a translation distance r, a
	translation direction s, azimuthal samples specified in the array phi,
	polar samples specified in the array theta, and a truncation point l.
	'''

	# The radial argument
	kr = 2. * math.pi * r

	# Compute the radial component
	hl = spec.sph_jn(l, kr)[0] + 1j * spec.sph_yn(l, kr)[0]

	# Define a dot-product function between the translation direction
	# and the sample direction with specified angular coordinates
	sd = lambda t, p: numpy.dot(s, sph2cart(1., t, p))

	# Compute the translator at the specified angular samples
	tr = [[sum([(1j)**li * (2. * li + 1.) * v * h
		for li, (h, v) in enumerate(zip(hl, legpoly(sd(th, ph), l)))])
		for th in theta] for ph in phi]

	return tr

def legpoly (x, n = 0):
	'''
	A coroutine to generate the Legendre polynomials up to order n,
	evaluated at argument x.
	'''

	if abs(x) > 1.0:
		raise ValueError("Argument must be in the range [-1,1]")
	if n < 0:
		raise ValueError("Order must be nonnegative")

	# Yield the zero-order value
	yield 1.0

	# Yield the first-order value, if that order is desired
	if n < 1: raise StopIteration
	else: yield x

	# Set some recursion values
	cur = x
	prev = 1.0

	# Now compute the subsequent values
	for i in xrange(1, n):
		next = ((2. * i + 1.) * x * cur - i * prev) / (i + 1.)
		prev = cur
		cur = next
		yield cur

	raise StopIteration


def exband (a, tol = 1e-6):
	'''
	Compute the excess bandwidth estimation for an object with radius a
	to a tolerance tol.
	'''

	# Compute the number of desired accurate digits
	d0 = -math.log10(tol)

	# Return the estimated bandwidth
	return int(2. * math.pi * a + 1.8 * (d0**2 * 2. * math.pi * a)**(1./3.))


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

def complexmax (a):
	'''
	Compute the maximum element of a complex array like MATLAB does.
	'''
	# Return the maximum value of a numpy matrix.
	return a.flat[numpy.argmax(numpy.abs(a))]

def psnr (x, y):
	'''
	The peak SNR, in dB, of a matrix x relative to the matrix y.
	This assumes x = y + N, where N is noise and y is signal.
	'''
	# Compute the average per-pixel squared error
	err = numpy.sum (numpy.abs(x - y)**2) / numpy.prod(x.shape)
	# Compute the square of the maximum signal value
	maxval = numpy.max(numpy.abs(y))**2

	# Compute the peak SNR in dB
	return 10. * math.log10(maxval / err)

def mse (x, y):
	'''
	Report the mean squared error between the matrix x and the matrix y.
	'''
	err = numpy.sum (numpy.abs(x - y)**2)
	err /= numpy.sum (numpy.abs(y)**2)

	return math.sqrt(err)
