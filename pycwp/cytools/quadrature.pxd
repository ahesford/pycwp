# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

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
	cdef bint _gkweights(double *, unsigned int, double) nogil

	cdef bint gausskron(self, double *, unsigned int,
				unsigned int, double, void *) nogil
	cdef bint _geval(self, double *, double *, double *, unsigned int,
				double, double, double, double, void *) nogil
	cdef bint _gkaux(self, double *, unsigned int, double, double *,
				unsigned int, double, double, void *) nogil

	cdef bint simpson(self, double *, unsigned int, double, void *, double *) nogil
	cdef bint _simpaux(self, double *, unsigned int, double, double,
				double, void *, double*, double *, double *) nogil

	cdef bint integrand(self, double *, double, void *) nogil

