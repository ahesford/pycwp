# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from libc.math cimport cos, M_PI, fabs
from libc.float cimport DBL_EPSILON, FLT_EPSILON
from libc.stdlib cimport malloc, free

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
	def __init__(self, double theta, double weight):
		'''
		Initialize the quadrature pair with node cos(theta) and weight.
		'''
		if self.theta < 0 or self.theta > M_PI:
			raise ValueError('Value of theta must be in range [0, pi]')
		self.theta = theta
		self.weight = weight

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
				IntegrableStatus.OK: 'Success',
				IntegrableStatus.GK_WRONG_ORDER: 'Invalid Gauss-Kronrod order',
				IntegrableStatus.GK_NO_CONVERGE: 'Gauss-Kronrod weights failed to converge',
				IntegrableStatus.GK_INVALID_RANGE: 'Gauss-Kronrod interval must satisfy 0 <= ua < ub <= 1',
				IntegrableStatus.INTEGRAND_EVALUATION_FAILED: 'Unable to evaluate integrand',
				IntegrableStatus.INTEGRAND_TOO_MANY_DIMS: 'Integrand has too many dimensions',
				IntegrableStatus.INTEGRAND_MISSING_CONTEXT: 'Integrand requires context variable',
				IntegrableStatus.WORK_ALLOCATION_FAILED: 'Unable to allocate work space',
				IntegrableStatus.NOT_IMPLEMENTED: 'Abstract method not implemented in subclass',
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

		kwts = &(ndwt[N + 1])
		gwts = &(kwts[N + 1])

		eps = min(FLT_EPSILON, max(DBL_EPSILON, 0.001 * tol))
		if not kronrod(N, eps, ndwt, kwts, gwts):
			return IntegrableStatus.GK_NO_CONVERGE
		return IntegrableStatus.OK


	@staticmethod
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

		if order < 1: raise ValueError('Order must be positive')

		nodes = <double *>malloc(3 * (order + 1) * sizeof(double));
		if not nodes:
			raise MemoryError('Could not allocate temporary storage')
		kweights = &(nodes[order + 1])
		gweights = &(kweights[order + 1])


		rcode = Integrable._gkweights(nodes, order, tol)
		if rcode != IntegrableStatus.OK:
			raise ValueError('Cannot evaluate weights: %s' % (Integrable.errmsg(rcode),))

		wtlist = [ ]
		for i in range(order + 1):
			wtlist.append((nodes[i], kweights[i], gweights[i]))

		return wtlist


	cdef IntegrableStatus gausskron(self, double *results,
			unsigned int nval, double atol, double rtol,
			double ua, double ub, int reclimit, void *ctx) nogil:
		'''
		Exactly as self.simpson, except Gauss-Kronrod quadrature of
		order 15 is used in place of Simpson quadrature and the
		integration takes place in the interval [ua, ub], with ua < ub
		and ua, ub in [0, 1].
		'''
		cdef:
			# Find midpoint and interval lengths
			double uc = 0.5 * (ua + ub)
			double h = ub - ua

			# Gauss-Kronrod quadrature of order 15 on (0, 1)
			double *nodes = [ 0.0042723144395937, 0.0254460438286208,
					0.0675677883201155, 0.1292344072003028,
					0.2069563822661544, 0.2970774243113014,
					0.3961075224960507, 0.5000000000000000,
					0.6038924775039493, 0.7029225756886985,
					0.7930436177338456, 0.8707655927996972,
					0.9324322116798845, 0.9745539561713792,
					0.9957276855604063 ]
			double *kwts = [ 0.0114676610052646, 0.0315460463149893,
					0.0523950051611251, 0.0703266298577630,
					0.0845023633196340, 0.0951752890323927,
					0.1022164700376494, 0.1047410705423639,
					0.1022164700376494, 0.0951752890323927,
					0.0845023633196340, 0.0703266298577630,
					0.0523950051611251, 0.0315460463149893,
					0.0114676610052646 ]
			double *gwts = [ 0.0000000000000000, 0.0647424830844349,
					0.0000000000000000, 0.1398526957446383,
					0.0000000000000000, 0.1909150252525595,
					0.0000000000000000, 0.2089795918367347,
					0.0000000000000000, 0.1909150252525595,
					0.0000000000000000, 0.1398526957446383,
					0.0000000000000000, 0.0647424830844349,
					0.0000000000000000 ]

			double *fv, *gint
			double u, err

			bint cnv

			unsigned int i
			IntegrableStatus rcode

		if not 1 <= nval <= 32:
			return IntegrableStatus.INTEGRAND_TOO_MANY_DIMS

		if not 0 <= ua < ub <= 1:
			return IntegrableStatus.GK_INVALID_RANGE

		fv = <double *>alloca(2 * nval * sizeof(double))
		if fv == <double *>NULL:
			return IntegrableStatus.WORK_ALLOCATION_FAILED

		gint = &(fv[nval])

		# Clear the integration results
		for i in range(nval):
			results[i] = 0.
			gint[i] = 0.

		# Compute the integrals
		for i in range(15):
			# Find point in increasing order
			u = ua + h * nodes[i]
			# Update the integrals with the evaluated function
			rcode = self._geval(results, gint, fv, nval, 
						u, h, kwts[i], gwts[i], ctx)
			if rcode != IntegrableStatus.OK: return rcode

		atol = max(atol, DBL_EPSILON)
		rtol = max(rtol, DBL_EPSILON)

		# Check for absolute or relative convergence of all components
		cnv = True
		for i in range(nval):
			err = fabs(results[i] - gint[i])
			if err > atol and err > rtol * results[i]:
				cnv = False
				break

		# Finish on convergence, interval collaps or recursion limit
		if uc <= ua or ub <= uc or reclimit == 0 or cnv:
			return IntegrableStatus.OK

		# Adjust the recursion limit if one exists
		if reclimit > 0: reclimit -= 1

		# Otherwise, drill down left and right
		# Absolute tolerance must be split between sides
		atol /= 2
		rcode = self.gausskron(fv, nval, atol, rtol, ua, uc, reclimit, ctx)
		if rcode != IntegrableStatus.OK: return rcode
		rcode = self.gausskron(gint, nval, atol, rtol, uc, ub, reclimit, ctx)
		if rcode != IntegrableStatus.OK: return rcode

		for i in range(nval): results[i] = fv[i] + gint[i]

		return IntegrableStatus.OK


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
		if rcode != IntegrableStatus.OK: return rcode

		for i in range(nval):
			kres[i] += h * kwt * fv[i]
			gres[i] += h * gwt * fv[i]

		return IntegrableStatus.OK


	cdef IntegrableStatus simpson(self, double *results,
			unsigned int nval, double atol, double rtol,
			int reclimit, void *ctx, double *ends) nogil:
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

		The recursion limit reclimit determines the maximum number of
		times the interval will be recursively bisected to seek
		convergence. A limit of 0 means the the whole interval will
		never be subdivided. If reclimit is negative, it is ignored.

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

		if not 1 <= nval <= 32:
			return IntegrableStatus.INTEGRAND_TOO_MANY_DIMS

		fa = <double *>alloca(3 * nval * sizeof(double));
		if fa == <double *>NULL:
			return IntegrableStatus.WORK_ALLOCATION_FAILED

		fb = &(fa[nval])
		fc = &(fb[nval])

		# Allocate storage for function evaluations
		if ends == <double *>NULL:
			# Compute the endpoint values
			rcode = self.integrand(fa, 0., ctx)
			if rcode != IntegrableStatus.OK: return rcode
			rcode = self.integrand(fb, 1., ctx)
			if rcode != IntegrableStatus.OK: return rcode
		else:
			# Copy precomputed endpoint values
			for i in range(nval):
				fa[i] = ends[i]
				fb[i] = ends[i + nval]

		# Evaluate at the midpoint, if all other values can be evaluated
		rcode = self.integrand(fc, 0.5, ctx)
		if rcode != IntegrableStatus.OK: return rcode

		# Compute the Simpson integral over the whole interval
		for i in range(nval):
			results[i] = isimpson(fa[i], fb[i], fc[i], 1.)

		return self._simpaux(results, nval, atol, rtol,
				0., 1., reclimit, ctx, fa, fb, fc)


	cdef IntegrableStatus integrand(self, double *results, double u, void *ctx) nogil:
		'''
		A dummy integrand that returns False (no value is computed).
		'''
		return IntegrableStatus.NOT_IMPLEMENTED


	cdef IntegrableStatus _simpaux(self, double *results,
			unsigned int nval, double atol, double rtol,
			double ua, double ub, int reclimit, void *ctx,
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

			double scomp, err
			bint cnv

			unsigned int i
			IntegrableStatus rcode

		# Nothing to do if recursion limit has been reached
		# Caller has already set results
		if reclimit == 0: return IntegrableStatus.OK

		if not 1 <= nval <= 32:
			return IntegrableStatus.INTEGRAND_TOO_MANY_DIMS

		fd = <double *>alloca(4 * nval * sizeof(double))
		if fd == <double *>NULL:
			return IntegrableStatus.WORK_ALLOCATION_FAILED

		fe = &(fd[nval])
		sl = &(fe[nval])
		sr = &(sl[nval])

		# Evaluate the function at the left and right midpoints
		rcode = self.integrand(fd, ud, ctx)
		if rcode != IntegrableStatus.OK: return rcode
		rcode = self.integrand(fe, ue, ctx)
		if rcode != IntegrableStatus.OK: return rcode

		# Tolerances below epsilon are problematic
		atol = max(atol, DBL_EPSILON)
		rtol = max(rtol, DBL_EPSILON)

		cnv = True
		for i in range(nval):
			# Evaluate the sub-interval integrals
			sl[i] = isimpson(fa[i], fc[i], fd[i], h2)
			sr[i] = isimpson(fc[i], fb[i], fe[i], h2)
			# Combine to form the whole-interval integrals
			scomp = sl[i] + sr[i]
			# Find the maximum error component
			err = scomp - results[i]
			# Update the best-available solution
			results[i] = scomp + err / 15.
			# Check for convergence of this component
			err = fabs(err)
			if err > 15 * atol and err > 15 * rtol * results[i]:
				cnv = False

		# If converged or interval collapsed, integration is done
		if uc <= ua or ub <= uc or cnv: return IntegrableStatus.OK

		# Otherwise, drill down left and right
		# Absolute tolerance is split between sides
		atol /= 2
		# Adjust recursion limit if necessary
		if reclimit > 0: reclimit -= 1
		rcode = self._simpaux(sl, nval, atol, rtol,
				ua, uc, reclimit, ctx, fa, fc, fd)
		if rcode != IntegrableStatus.OK: return rcode
		rcode = self._simpaux(sr, nval, atol, rtol,
				uc, ub, reclimit, ctx, fc, fb, fe)
		if rcode != IntegrableStatus.OK: return rcode

		# Merge two-sided integrals
		for i in range(nval): results[i] = sl[i] + sr[i]

		return IntegrableStatus.OK
