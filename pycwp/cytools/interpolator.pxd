'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

import numpy as np
cimport numpy as np

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libc.math cimport fabs, log2

from ptutils cimport point
from quadrature cimport Integrable

cdef class Interpolator3D(Integrable):
	cdef double *coeffs
	cdef unsigned long ncx, ncy, ncz
	cdef bint _usedef
	cdef double _default

	@staticmethod
	cdef bint crdfrac(point *t, long *i, long *j, long *k,
			point p, long nx, long ny, long nz) nogil

	cdef bint _evaluate(self, double *f, point *grad, point p) nogil

	cdef bint _integrand(self, double *, double, void *) nogil


cdef class LagrangeInterpolator3D(Interpolator3D):
	@staticmethod
	cdef void lgwts(double *l, double x) nogil

	@staticmethod
	cdef void dlgwts(double *l, double x) nogil

	@staticmethod
	cdef void adjint(double *t, long *i, unsigned long n) nogil

	cdef bint _evaluate(self, double *f, point *grad, point p) nogil


cdef class HermiteInterpolator3D(Interpolator3D):
	cdef unsigned long nval
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil

	@staticmethod
	cdef int img2coeffs(double[:,:,:,:] coeffs, double[:,:,:] img) except -1

	@staticmethod
	cdef int mhdiffs(double[:] m, double[:] v) except -1

	@staticmethod
	cdef void hermspl(double *w, double t) nogil

	@staticmethod
	cdef void hermdiff(double *w, double t) nogil


cdef class CubicInterpolator3D(Interpolator3D):
	@staticmethod
	cdef int img2coeffs(double[:,:,:] coeffs, double[:,:,:] img) except -1

	@staticmethod
	cdef int nscoeffs(double[:] c, double[:] work) except -1

	@staticmethod
	cdef void bswt(double *w, double t) nogil

	@staticmethod
	cdef void dbswt(double *w, double t) nogil

	cdef bint _evaluate(self, double *f, point *grad, point p) nogil


cdef class LinearInterpolator3D(Interpolator3D):
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil
