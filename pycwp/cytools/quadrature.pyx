# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from libc.math cimport cos, M_PI

from fastgl cimport qpstruct, glpair as cglpair

cdef class QuadPair:
	'''
	A simple class to encapsulate a quadrature node and weight.
	'''
	cdef readonly double theta, weight

	@cython.embedsignature(True)
	def __init__(self, double theta, double weight):
		'''
		Initialize the quadrature pair with node cos(theta) and weight.
		'''
		if self.theta < 0 or self.theta > M_PI:
			raise ValueError('Value of theta must be in range [0, pi]')
		self.theta = theta
		self.weight = weight

	@cython.embedsignature(True)
	cpdef double x(self):
		'''
		The quadture node cos(self.theta).
		'''
		return cos(self.theta)

	def __repr__(self):
		'''
		Return a pretty-print version of the quadrature pair.
		'''
		return '%s(%r, %r)' % (self.__class__.__name__, self.theta, self.weight)


@cython.embedsignature(True)
def glpair(size_t n, size_t k):
	'''
	Compute and return, as a QuadPair instance, the k-th node and weight of
	a Gauss-Legendre quadrature rule of order n.
	'''
	cdef qpstruct result
	cdef int rcode = cglpair(&result, n, k + 1)

	if not rcode:
		raise ValueError('Node index k must be in range [0, n) for quadature order n')

	return QuadPair(result.theta, result.weight)
