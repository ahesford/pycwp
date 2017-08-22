# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

ctypedef enum IntegrableStatus:
	OK = 0,
	GK_WRONG_ORDER,
	GK_NO_CONVERGE,
	INTEGRAND_EVALUATION_FAILED,
	INTEGRAND_TOO_MANY_DIMS,
	INTEGRAND_MISSING_CONTEXT,
	WORK_ALLOCATION_FAILED,
	NOT_IMPLEMENTED,
	CUSTOM_RETURN


cdef class QuadPair:
	'''
	A simple class to encapsulate a quadrature node and weight.
	'''
	cdef readonly double theta, weight
	cpdef double x(self)


cdef class Integrable:
	'''
	A class to represent integrable functions, and an adaptive Simpson
	quadrature to integrate them.
	'''
	@staticmethod
	cdef IntegrableStatus _gkweights(double *, unsigned int, double) nogil

	cdef IntegrableStatus gausskron(self, double *, unsigned int,
				unsigned int, double, void *) nogil
	cdef IntegrableStatus _geval(self, double *, double *, double *, unsigned int,
				double, double, double, double, void *) nogil
	cdef IntegrableStatus _gkaux(self, double *, unsigned int, double, double *,
				unsigned int, double, double, void *) nogil

	cdef IntegrableStatus simpson(self, double *, unsigned int, double, void *, double *) nogil
	cdef IntegrableStatus _simpaux(self, double *, unsigned int, double, double,
				double, void *, double*, double *, double *) nogil

	cdef IntegrableStatus integrand(self, double *, double, void *) nogil
