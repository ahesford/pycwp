'''
General-purpose numerical routines used in other parts of the module.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy, math
from . import poly


def clenshaw (m):
	'''
	Compute the Clenshaw-Curtis quadrature nodes and weights in the
	interval [0,pi] for a specified order m.
	'''
	n = int(m - 1)
	idx = numpy.arange(m)
	# Nodes are equally spaced in the interval
	nodes = idx * math.pi / float(n)
	# Endpoint weights should be halved to avoid aliasing
	cj = (numpy.mod(idx, n) == 0).choose(2., 1.)
	k = numpy.arange(1, n // 2 + 1)[:,numpy.newaxis]
	bk = (k < n / 2.).choose(1., 2.)
	# The weights are defined according to Waldvogel (2003)
	cos = numpy.cos(2. * k * nodes[numpy.newaxis,:])
	scl = bk / (4. * k**2 - 1.)
	weights = (cj / n) * (1. - numpy.sum(scl * cos, axis=0))
	return nodes, weights


def fejer2 (m):
	'''
	Compute the quadrature nodes and weights, in the interval [0, pi],
	using Fejer's second rule for an order m.
	'''
	n = int(m - 1)
	idx = numpy.arange(m)
	nodes = idx * math.pi / float(n)
	k = numpy.arange(1, n // 2 + 1)[:,numpy.newaxis]
	sin = numpy.sin((2. * k - 1.) * nodes[numpy.newaxis,:])
	weights = (4. / n) * numpy.sin(nodes) * numpy.sum(sin / (2. * k - 1.), axis=0)
	return nodes, weights


def gaussleg (m, tol = 1e-9, itmax=100):
	'''
	Compute the Gauss-Legendre quadrature nodes and weights in the interval
	[0,pi] for a specified order m. The Newton-Raphson method is used to
	find the roots of Legendre polynomials (the nodes) with a maximum of
	itmax iterations and an error tolerance of tol.
	'''
	weights = numpy.zeros((m), dtype=numpy.float64)
	nodes = numpy.zeros((m), dtype=numpy.float64)

	nRoots = int(m + 1) // 2

	for i in range(nRoots):
		# The initial guess is the (i+1) Chebyshev root
		t = math.cos (math.pi * (i + 0.75) / m)
		for j in range(itmax):
			# Grab the Legendre polynomial and its derviative
			p, dp = poly.legendre(t, m)[:2]

			# Perform a Newton-Raphson update
			dt = -p/dp
			t += dt

			# Update the node and weight estimates
			nodes[i] = math.acos(t)
			nodes[-(i + 1)] = math.acos(-t)
			weights[i] = 2.0 / (1.0 - t**2) / (dp**2)
			weights[-(i + 1)] = weights[i]

			# Nothing left to do if tolerance was achieved
			if abs(dt) < tol: break

	return nodes, weights


def gausslob (m, tol = 1e-9, itmax=100):
	'''
	Compute the Gauss-Lobatto quadrature nodes and weights in the interval
	[0,pi] for a specified order m. The Newton-Raphson method is used to
	find the roots of derivatives of Legendre polynomials (the nodes) with
	a maximum of itmax iterations and an error tolerance of tol.
	'''
	weights = numpy.zeros((m), dtype=numpy.float64)
	nodes = numpy.zeros((m), dtype=numpy.float64)

	# This is the number of roots away from the endpoints
	nRoots = int(m - 1) / 2

	# First compute the nodes and weights at the endpoints
	nodes[0] = 0.
	nodes[-1] = math.pi
	weights[0] = 2. / m / (m - 1.)
	weights[-1] = weights[0]

	for i in range(1, nRoots + 1):
		# The initial guess is halfway between subsequent Chebyshev roots
		t = 0.5 * sum(math.cos(math.pi * (i + j - 0.25) / m) for j in range(2));
		for j in range(itmax):
			# Grab the Legendre polynomial and its derviative
			p, dp, ddp = poly.legendre(t, m - 1)

			# Perform a Newton-Raphson update
			dt = -dp / ddp
			t += dt

			# Update the node and weight estimates
			nodes[i] = math.acos(t)
			nodes[-(i + 1)] = math.acos(-t)
			weights[i] = 2. / m / (m - 1.) / (p**2)
			weights[-(i + 1)] = weights[i]

			# Nothing left to do if tolerance was achieved
			if abs(dt) < tol: break

	return nodes, weights
