# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from libc.math cimport cos, M_PI, fabs
from libc.float cimport DBL_EPSILON

from fastgl cimport qpstruct, glpair as cglpair

from quadrature cimport *

cdef extern from "<alloca.h>":
	void *alloca (size_t) nogil

cdef inline double isimpson(double fa, double fb, double fc, double h) nogil:
	'''
	Return the Simpson integral, over an interval h = b - a for some
	endpoints a and b, of a function with values fa = f(a), fb = f(b) and
	f(c) = f(0.5 * (a + b)).
	'''
	return h * (fa + 4 * fc + fb) / 6.


cdef class QuadPair:
	'''
	A simple class to encapsulate a quadrature node and weight.
	'''
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
	cdef int rcode = cglpair(&result, n, k)

	if not rcode:
		raise ValueError('Node index k must be in range [0, n) for quadature order n')

	return QuadPair(result.theta, result.weight)


cdef class Integrable:
	'''
	A class to represent integrable functions, and an adaptive Simpson
	quadrature to integrate them.
	'''
	@cython.embedsignature(True)
	cdef bint _simpson(self, double *results, unsigned int nval,
					double tol, void *ctx, double *ends) nogil:
		'''
		Perform adaptive Simpson quadrature of self._integrand along
		the interval [0, 1]. The function is evaluated by calling

			self._integrand(output, u, ctx),

		where output is a pointer to a length-nval (in range [1, 32])
		array of doubles that stores the function values to be
		integrated, u in [0, 1] is the variable of integration, and ctx
		is a user-provided (and possibly NULL) pointer to some
		"context".

		The integration is performed independently for each element in
		output; the results are stored in the "results" array, which
		should hold at least nval doubles.

		The function self._integrand must return True for a successful
		call and return False if the call cannot be completed
		successfully. If self._integrand returns False at any point,
		this routine will abort by returning False, leaving results in
		an indeterminate state. If self._integrand never fails, this
		routine will return True, and the values stored in results are
		guranteed to be valid.

		If ends is not NULL, it should point to an array of length
		2 * nval that holds, in elements 0 through (nval - 1), the
		precomputed value of self._integrand at u == 0 and, in elements
		nval through (2 * nval - 1), the precomputed value of
		self._integrand at u == 1. If ends is NULL, these values will
		be computed on demand.
		'''
		cdef:
			double *fa
			double *fb
			double *fc
			unsigned int i

		if not 1 <= nval <= 32: return False

		fa = <double *>alloca(3 * nval * sizeof(double));
		if fa == <double *>NULL: return False
		fb = &(fa[nval])
		fc = &(fb[nval])

		# Allocate storage for function evaluations
		if ends == <double *>NULL:
			# Compute the endpoint values
			if not self._integrand(fa, 0., ctx): return False
			if not self._integrand(fb, 1., ctx): return False
		else:
			# Copy precomputed endpoint values
			for i in range(nval):
				fa[i] = ends[i]
				fb[i] = ends[i + nval]

		# Evaluate at the midpoint, if all other values can be evaluated
		if not self._integrand(fc, 0.5, ctx): return False

		# Compute the Simpson integral over the whole interval
		for i in range(nval):
			results[i] = isimpson(fa[i], fb[i], fc[i], 1.)

		return self._simpaux(results, nval, tol, 0., 1., ctx, fa, fb, fc)


	@cython.embedsignature(True)
	cdef bint _integrand(self, double *results, double u, void *ctx) nogil:
		'''
		A dummy integrand that returns False (no value is computed).
		'''
		return False


	@cython.embedsignature(True)
	cdef bint _simpaux(self, double *results, unsigned int nval,
			double tol, double ua, double ub, void *ctx,
			double *fa, double *fb, double *fc) nogil:
		'''
		A recursive helper for _simpson.
		'''
		cdef:
			# Find midpoint and interval lengths
			double uc = 0.5 * (ua + ub)
			double h = ub - ua
			double h2 = 0.5 * h

			# Midpoints for left and right half intervals
			double ud = 0.75 * ua + 0.25 * ub
			double ue = 0.25 * ua + 0.75 * ub

			double *fd
			double *fe
			double *sl
			double *sr

			double errmax = 0., scomp, err

			unsigned int i

		if not 1 <= nval <= 32: return False

		fd = <double *>alloca(4 * nval * sizeof(double))
		if fd == <double *>NULL: return False
		fe = &(fd[nval])
		sl = &(fe[nval])
		sr = &(sl[nval])

		# Evaluate the function at the left and right midpoints
		if not self._integrand(fd, ud, ctx): return False
		if not self._integrand(fe, ue, ctx): return False

		for i in range(nval):
			# Evaluate the sub-interval integrals
			sl[i] = isimpson(fa[i], fc[i], fd[i], h2)
			sr[i] = isimpson(fc[i], fb[i], fe[i], h2)
			# Combine to form the whole-interval integrals
			scomp = sl[i] + sr[i]
			# Find the maximum error component
			err = scomp - results[i]
			errmax = max(errmax, fabs(err))
			# Update the best-available solution
			results[i] = scomp + err / 15.

		# Tolerances below epsilon are problematic
		tol = max(tol, DBL_EPSILON)

		# If converged or interval collapsed, integration is done
		if h2 <= 2 * DBL_EPSILON or errmax <= 15 * tol: return True

		# Otherwise, drill down left and right
		tol /= 2
		if not self._simpaux(sl, nval, tol, ua, uc, ctx, fa, fc, fd):
			return False
		if not self._simpaux(sr, nval, tol, uc, ub, ctx, fc, fb, fe):
			return False

		# Merge two-sided integrals
		for i in range(nval): results[i] = sl[i] + sr[i]

		return True
