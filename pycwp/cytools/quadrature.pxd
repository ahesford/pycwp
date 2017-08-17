# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

ctypedef bint (*quadfunc)(double *, double, void *) nogil

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
	cdef bint _integrand(self, double *, double, void *) nogil
	cdef bint _simpson(self, double *, unsigned int, double, void *, double *) nogil
	cdef bint _simpaux(self, double *, unsigned int, double, double,
				double, void *, double*, double *, double *) nogil
