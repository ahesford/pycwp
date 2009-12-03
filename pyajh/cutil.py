'''
General-purpose numerical routines used in other parts of the module.
'''

import numpy
import math
from scipy import special as spec

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
	# Use the flattened representation afforded by the matrix.
	return reduce (lambda x, y: abs(x) > abs(y) and x or y, a.flat)

def mse (x, y):
	'''
	Report the mean squared error between the matrix x and the matrix y.
	'''
	err = numpy.sum (numpy.abs(x - y)**2)
	err /= numpy.sum (numpy.abs(y)**2)

	return math.sqrt(err)

def sph2cart (r, t, p):
	'''
	Mimics the sph2cart function of Matlab, converting spherical
	coordinates (r, t, p) to Cartesian coordinates (x, y, z). The
	variables t and p are the polar and azimuthal angles, respectively.
	'''
	st = numpy.sin(t)
	return r * st * numpy.cos(p), r * st * numpy.sin(p), r * numpy.cos(t)

def cart2sph (x, y, z):
	'''
	Mimics the cart2sph function of Matlab, converting Cartesian
	coordinates (x, y, z) to spherical coordinates (r, t, p). The
	variables t and p are the polar and azimuthal angles, respectively.
	'''
	r = numpy.sqrt(x**2 + y**2 + z**2)
	t = numpy.arccos (z / r)
	p = numpy.arctan2 (y, x)
	return r, t, p

def pol2cart (r, t):
	'''
	Mimics the pol2cart function of Matlab, converting polar coordinates
	(r, t) to Cartesian coordinates (x, y).
	'''
	return r * numpy.cos(t), r * numpy.sin(t)

def cart2pol (x, y):
	'''
	Mimics the cart2pol function of Matlab, converting Cartesian
	coordinates (x, y) to polar coordinates (r, t).
	'''
	r = numpy.sqrt (x**2 + y**2)
	t = numpy.arctan2 (y, x)
	return r, t

def shnormfact (m, n):
	'''
	Normalization factor for the associated Legendre function of degree n
	and order m.
	'''

	if m > n:
		return 0.

	nf = math.sqrt((2. * n + 1.) / 4. / math.pi)
	nf *= math.exp(0.5 * (spec.gammaln(n-m+1) - spec.gammaln(n+m+1)))
	return nf

def spharm (m, n, theta, phi = None):
	'''
	Return the spherical harmonics with degrees 0,...,n and orders
	0,...,m in an array of size (m+1,n+1) for polar angle theta and
	azimuthal angle phi. If phi is unspecified, treat it as 0, which
	returns the normalized associated Legendre polynomial for cos(theta).
	'''
	# Precompute the normalization factors.
	nf = numpy.array([[shnormfact(m, n) for n in xrange(n+1)]
		for m in xrange(m+1)])

	# Only grab the associated Legendre function, not the derivative.
	# Multiply the normalizing factors.
	lp = spec.lpmn(m, n, math.cos(theta))[0] * nf

	# There is no azimuthal angle to consider
	if phi is None:
		return lp

	# Scale the rows function by the azimuthal term
	for m, row in enumerate(lp):
		row *= numpy.exp(1j * m * pi)

	return lp
