'''
General-purpose numerical routines used in other parts of the module.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy, math


def legassoc (n, m, th):
	'''
	Compute the normalized associated Legendre functions up to degree
	n and order n for an argument cos(th), where th is a polar angle.
	Output is an array with shape (m+1,n+1).
	'''
	t = math.cos(th)
	u = -math.sin(th)

	# Initialize the output array
	lgp = np.zeros((m+1,n+1))

	# The initial value
	lgp[0,0] = math.sqrt(1. / 4. / math.pi)

	# Set up the diagonal elements
	for l in xrange(1,min(m,n)+1):
		lgp[l,l] = math.sqrt((2. * l + 1.) / (2. * l)) * u * lgp[l-1,l-1]

	# Set up the upper diagonal
	for l in xrange(0,min(m,n)):
		lgp[l,l+1] = math.sqrt(2. * l + 3.) * t * lgp[l,l]

	# Now fill in the rest of the matrix
	for p in xrange(0,m+1):
		for l in xrange(1+p,n):
			# Precompute the recursion coefficients
			an = math.sqrt((2. * l + 1.) * (2. * l + 3) /
					(1. + l + p) / (1. + l - p))
			bn = math.sqrt((2. * l + 3.) * (l - p) * (l + p) /
					(2. * l - 1.) / (1. + l + p) / (1. + l - p))
			lgp[p,l+1] = an * t * lgp[p,l] - bn * lgp[p,l-1]

	return lgp


def lagrange(x, pts):
	'''
	For sample points pts and a test point x, return the Lagrange
	polynomial weights.
	'''

	# Compute a single weight term
	def lgwt(x, pi, pj): return float(x - pj) / float(pi - pj)

	# Compute all the weights
	wts = [prod(lgwt(x, pi, pj) for j, pj in enumerate(pts) if i != j)
			for i, pi in enumerate(pts)]

	return wts


def legpoly (x, n = 0):
	'''
	A coroutine to generate the Legendre polynomials up to order n,
	evaluated at argument x.
	'''

	if numpy.any(numpy.abs(x) > 1.0):
		raise ValueError("Arguments must be in [-1,1]")
	if n < 0:
		raise ValueError("Order must be nonnegative")

	# Set some recursion values
	cur = x
	prev = numpy.ones_like(x)

	# Yield the zero-order value
	yield prev

	# Yield the first-order value, if that order is desired
	if n < 1: raise StopIteration
	else: yield cur

	# Now compute the subsequent values
	for i in xrange(1, n):
		next = ((2. * i + 1.) * x * cur - i * prev) / (i + 1.)
		prev = cur
		cur = next
		yield cur

	raise StopIteration


def legendre (t, m):
	'''
	Return the value of the Legendre polynomial of order m, along with its
	first and second derivatives, at a point t.
	'''
	# Set function values explicitly for orders less than 2
	if m < 1: return 1., 0., 0.
	elif m < 2: return t, 1., 0.

	p0, p1 = 1.0, t

	for k in range(1, m):
		# This value is necessary for the second derivative
		dpl = k * (p0 - t * p1) / (1. - t**2)
		p = ((2.*k + 1.0) * t * p1 - k * p0) / (1. + k)
		p0 = p1; p1 = p

	# Compute the value of the derivative
	dp = m * (p0 - t * p1) / (1.0 - t**2)

	# Compute the value of the second derivative
	ddp = ((m - 2.) * t * dp + m * (p1 - dpl)) / (t**2 - 1.)

	return p, dp, ddp
