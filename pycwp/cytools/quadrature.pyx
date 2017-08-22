# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from libc.math cimport cos, M_PI, fabs
from libc.float cimport DBL_EPSILON, FLT_EPSILON

from quadrature cimport *

cdef extern from "<alloca.h>":
	void *alloca (size_t) nogil

cdef extern from "fastgl.c":
	ctypedef struct qpstruct:
		double theta, weight
	int cglpair(qpstruct *, size_t, size_t) nogil

cdef extern from "kronrod.c":
	int kronrod(int, double, double *, double *, double *) nogil
	void kronrod_adjust(double, double, int, double *, double *, double *) nogil

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
	A class to represent integrable functions, and adaptive quadratures to
	integrate them.
	'''
	@classmethod
	def errmsg(cls, IntegrableStatus code):
		'''
		Produce a nice error message from an IntegrableStatus code.
		'''
		return {
				OK: 'Success',
				GK_WRONG_ORDER: 'Invalid Gauss-Kronrod order',
				GK_NO_CONVERGE: 'Gauss-Kronrod weights failed to converge',
				INTEGRAND_EVALUATION_FAILED: 'Unable to evaluate integrand',
				INTEGRAND_TOO_MANY_DIMS: 'Integrand has too many dimensions',
				INTEGRAND_MISSING_CONTEXT: 'Integrand requires context variable',
				WORK_ALLOCATION_FAILED: 'Unable to allocate work space',
				NOT_IMPLEMENTED: 'Abstract method not implemented in subclass',
			}.get(code, 'No known error message')

	@staticmethod
	cdef IntegrableStatus _gkweights(double *ndwt, unsigned int N, double tol) nogil:
		'''
		Prepare a set of Gauss-Kronrod abcissae and weights with
		Gaussian order N and Kronrod order 2 * N + 1. Results are
		stored in ndwt, which must hold at least 3 * (N + 1) doubles.
		The order N must be in the range [1, 128].

		The first (N + 1) values in ndwt will be the abcissae in the
		interval [0, 1] and represent half of the quadrature interval
		[-1, 1]; the negative abcissae are the negatives of those
		returned, with identical weights.

		The second (N + 1) values in ndwt will hold the Kronrod weights
		for the (N + 1) returned abcissae. The third (N + 1) values in
		ndwt will hold the corresponding Gaussian weights, which will
		be 0 for even-numbered (Kronrod-only) indices.

		Abcissae and weights are computed to a tolerance

			eps = min(FLT_EPSILON, max(DBL_EPSILON, 0.001 * tol)).

		This method returns True if the weights can be computed, and
		False otherwise.
		'''
		cdef:
			double *kwts
			double *gwts
			unsigned int i

		if not 1 <= N <= 128: return GK_WRONG_ORDER

		kwts = &(ndwt[N + 1])
		gwts = &(kwts[N + 1])

		eps = min(FLT_EPSILON, max(DBL_EPSILON, 0.001 * tol))
		if not kronrod(N, eps, ndwt, kwts, gwts): return GK_NO_CONVERGE
		return OK


	@staticmethod
	@cython.embedsignature(True)
	def gkweights(unsigned int order, double tol):
		'''
		Return a list of elements (x, kw, gw), where x is a
		Gauss-Kronrod abcissa in the range [0, 1] (half of the
		symmetric integration interval [-1, 1]), kw is the Kronrod
		weight, and gw is the Gauss weight (which is 0 if x is a
		Kronrod abcissa only).

		The value of tol should be on the order of the desired integral
		accuracy; the accuracy of the nodes and weights will be
		determined based on this value.
		'''
		cdef:
			double *nodes
			double *kweights
			double *gweights
			unsigned int i
			double eps
			IntegrableStatus rcode

		if not 1 <= order <= 128:
			raise ValueError('Order must be in range [1, 128]')

		nodes = <double *>alloca(3 * (order + 1) * sizeof(double));
		if not nodes:
			raise MemoryError('Could not allocate temporary storage')
		kweights = &(nodes[order + 1])
		gweights = &(kweights[order + 1])


		rcode = Integrable._gkweights(nodes, order, tol)
		if rcode != OK:
			raise ValueError('Cannot evaluate weights: %s' % (Integrable.errmsg(rcode),))

		wtlist = [ ]
		for i in range(order + 1):
			wtlist.append((nodes[i], kweights[i], gweights[i]))

		return wtlist


	cdef IntegrableStatus gausskron(self, double *results, unsigned int nval,
				unsigned int order, double tol, void *ctx) nogil:
		'''
		Exactly as self.simpson, except adaptive Gauss-Kronrod
		quadrature of the specified order (in range [0, 128]) is used
		in place of adapative Simpson quadrature.
		'''
		cdef:
			double *nodes
			double *kweights
			double *gweights
			double eps
			IntegrableStatus rcode

		if not 1 <= nval <= 32: return INTEGRAND_TOO_MANY_DIMS
		if not 1 <= order <= 128: return GK_WRONG_ORDER

		nodes = <double *>alloca(3 * (order + 1) * sizeof(double));
		if nodes == <double *>NULL: return WORK_ALLOCATION_FAILED

		kweights = &(nodes[order + 1])
		gweights = &(kweights[order + 1])

		# Compute weights and adjust to [0, 1] interval
		rcode = Integrable._gkweights(nodes, order, tol)
		if rcode != OK: return rcode
		kronrod_adjust(0., 1., order, nodes, kweights, gweights)

		return self._gkaux(results, nval, tol, nodes, order, 0., 1., ctx)


	cdef IntegrableStatus _geval(self, double *kres, double *gres,
			double *fv, unsigned int nval, double u,
			double h, double kwt, double gwt, void *ctx) nogil:
		'''
		Helper for _geval; updates Gauss and Kronrod integrals and
		stores the integrand at a specific evaluation point.
		'''
		cdef unsigned int i
		cdef IntegrableStatus rcode

		rcode = self.integrand(fv, u, ctx)
		if rcode != OK: return rcode

		for i in range(nval):
			kres[i] += h * kwt * fv[i]
			gres[i] += h * gwt * fv[i]

		return OK


	cdef IntegrableStatus _gkaux(self, double *results, unsigned int nval,
				double tol, double *ndwt, unsigned int order, 
				double ua, double ub, void *ctx) nogil:
		'''
		Recursive helper for gausskron.
		'''
		cdef:
			# Find midpoint and interval lengths
			double uc = 0.5 * (ua + ub)
			double h = ub - ua

			double *gwts
			double *kwts

			double *fv, *gint
			double errmax = 0., u

			unsigned int i
			IntegrableStatus rcode

		if not 1 <= nval <= 32: return INTEGRAND_TOO_MANY_DIMS

		kwts = &(ndwt[order + 1])
		gwts = &(kwts[order + 1])

		fv = <double *>alloca(2 * nval * sizeof(double))
		if fv == <double *>NULL: return WORK_ALLOCATION_FAILED

		gint = &(fv[nval])

		# Clear the integration results
		for i in range(nval):
			results[i] = 0.
			gint[i] = 0.

		# Compute the left-half integral
		for i in range(order):
			# Find point in increasing order
			u = ua + h * (1.0 - ndwt[i])
			# Update the integrals with the evaluated function
			rcode = self._geval(results, gint, fv, nval,
						u, h, kwts[i], gwts[i], ctx)
			if rcode != OK: return rcode

		# Add contribution from central point
		rcode = self._geval(results, gint, fv, nval, uc,
					h, kwts[order], gwts[order], ctx)
		if rcode != OK: return rcode

		# Compute right-half integral
		for i in range(order - 1, -1, -1):
			# Find point in increasing order
			u = ua + h * ndwt[i]
			# Update the integrals
			rcode = self._geval(results, gint, fv, nval,
						u, h, kwts[i], gwts[i], ctx)
			if rcode != OK: return rcode

		# Estimate the error
		for i in range(nval):
			errmax = max(errmax, fabs(results[i] - gint[i]))

		tol = max(tol, DBL_EPSILON)

		# If converged or interval collapsed, integration is done
		if h <= DBL_EPSILON or errmax <= tol: return OK

		# Otherwise, drill down left and right
		tol /= 2
		rcode = self._gkaux(fv, nval, tol, ndwt, order, ua, uc, ctx)
		if rcode != OK: return rcode
		rcode = self._gkaux(gint, nval, tol, ndwt, order, uc, ub, ctx)
		if rcode != OK: return rcode

		for i in range(nval): results[i] = fv[i] + gint[i]

		return OK


	cdef IntegrableStatus simpson(self, double *results, unsigned int nval,
				double tol, void *ctx, double *ends) nogil:
		'''
		Perform adaptive Simpson quadrature of self.integrand along
		the interval [0, 1]. The function is evaluated by calling

			self.integrand(output, u, ctx),

		where output is a pointer to a length-nval (in range [1, 32])
		array of doubles that stores the function values to be
		integrated, u in [0, 1] is the variable of integration, and ctx
		is a user-provided (and possibly NULL) pointer to some
		"context".

		The integration is performed independently for each element in
		output; the results are stored in the "results" array, which
		should hold at least nval doubles.

		The function self.integrand must return IntegrableStatus.OK for
		a successful call and another IntegrableStatus if the call
		cannot be completed successfully. If self.integrand returns
		anything but OK at any point, this routine will abort by
		returning the failure code, leaving results in an indeterminate
		state. If self.integrand never fails, this routine will return
		OK, and the values stored in results are guranteed to be valid.

		If ends is not NULL, it should point to an array of length
		2 * nval that holds, in elements 0 through (nval - 1), the
		precomputed value of self.integrand at u == 0 and, in elements
		nval through (2 * nval - 1), the precomputed value of
		self.integrand at u == 1. If ends is NULL, these values will
		be computed on demand.
		'''
		cdef:
			double *fa
			double *fb
			double *fc
			unsigned int i
			IntegrableStatus rcode

		if not 1 <= nval <= 32: return INTEGRAND_TOO_MANY_DIMS

		fa = <double *>alloca(3 * nval * sizeof(double));
		if fa == <double *>NULL: return WORK_ALLOCATION_FAILED
		fb = &(fa[nval])
		fc = &(fb[nval])

		# Allocate storage for function evaluations
		if ends == <double *>NULL:
			# Compute the endpoint values
			rcode = self.integrand(fa, 0., ctx)
			if rcode != OK: return rcode
			rcode = self.integrand(fb, 1., ctx)
			if rcode != OK: return rcode
		else:
			# Copy precomputed endpoint values
			for i in range(nval):
				fa[i] = ends[i]
				fb[i] = ends[i + nval]

		# Evaluate at the midpoint, if all other values can be evaluated
		rcode = self.integrand(fc, 0.5, ctx)
		if rcode != OK: return rcode

		# Compute the Simpson integral over the whole interval
		for i in range(nval):
			results[i] = isimpson(fa[i], fb[i], fc[i], 1.)

		return self._simpaux(results, nval, tol, 0., 1., ctx, fa, fb, fc)


	cdef IntegrableStatus integrand(self, double *results, double u, void *ctx) nogil:
		'''
		A dummy integrand that returns False (no value is computed).
		'''
		return NOT_IMPLEMENTED


	cdef IntegrableStatus _simpaux(self, double *results, unsigned int nval,
				double tol, double ua, double ub, void *ctx,
				double *fa, double *fb, double *fc) nogil:
		'''
		A recursive helper for simpson.
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
			IntegrableStatus rcode

		if not 1 <= nval <= 32: return INTEGRAND_TOO_MANY_DIMS

		fd = <double *>alloca(4 * nval * sizeof(double))
		if fd == <double *>NULL: return WORK_ALLOCATION_FAILED
		fe = &(fd[nval])
		sl = &(fe[nval])
		sr = &(sl[nval])

		# Evaluate the function at the left and right midpoints
		rcode = self.integrand(fd, ud, ctx)
		if rcode != OK: return rcode
		rcode = self.integrand(fe, ue, ctx)
		if rcode != OK: return rcode

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
		if h2 <= 2 * DBL_EPSILON or errmax <= 15 * tol: return OK

		# Otherwise, drill down left and right
		tol /= 2
		rcode = self._simpaux(sl, nval, tol, ua, uc, ctx, fa, fc, fd)
		if rcode != OK: return rcode
		rcode = self._simpaux(sr, nval, tol, uc, ub, ctx, fc, fb, fe)
		if rcode != OK: return rcode

		# Merge two-sided integrals
		for i in range(nval): results[i] = sl[i] + sr[i]

		return OK
