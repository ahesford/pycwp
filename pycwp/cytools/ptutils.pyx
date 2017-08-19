# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

#cython: embedsignature=True, cdivision=True

import cython
cimport cython

from libc.math cimport sqrt, fabs, fmax
from ptutils cimport *

cdef double REALEPS = 5.1448789686149945e-12

cdef point axpy(double a, point x, point y) nogil:
	'''
	Return a point struct equal to (a * x + y).
	'''
	cdef point r
	r.x = a * x.x + y.x
	r.y = a * x.y + y.y
	r.z = a * x.z + y.z
	return r

cdef point lintp(double t, point x, point y) nogil:
	'''
	Return the linear interpolation (1 - t) * x + t * y.
	'''
	cdef point r
	cdef double m = 1 - t
	r.x = m * x.x + t * y.x
	r.y = m * x.y + t * y.y
	r.z = m * x.z + t * y.z
	return r

cdef point *iaxpy(double a, point x, point *y) nogil:
	'''
	Store, in y, the value of (a * x + y).
	'''
	y.x += a * x.x
	y.y += a * x.y
	y.z += a * x.z
	return y

cdef point scal(double a, point x) nogil:
	'''
	Scale and return the point x by a.
	'''
	cdef point r
	r.x = x.x * a
	r.y = x.y * a
	r.z = x.z * a
	return r

cdef point *iscal(double a, point *x) nogil:
	'''
	Scale the point x, in place, by a.
	'''
	x.x *= a
	x.y *= a
	x.z *= a
	return x

cdef point *iptmpy(point h, point *x) nogil:
	'''
	Compute, in x, the Hadamard product of points h and x.
	'''
	x.x *= h.x
	x.y *= h.y
	x.z *= h.z
	return x

cdef point *iptdiv(point h, point *x) nogil:
	'''
	Compute, in x, the coordinate-wise ratio x / h.
	'''
	x.x /= h.x
	x.y /= h.y
	x.z /= h.z
	return x

cdef double ptnrm(point x) nogil:
	'''
	The L2-norm of the point x.
	'''
	cdef double ns = x.x * x.x + x.y * x.y + x.z * x.z
	return sqrt(ns)

cdef double ptsqnrm(point x) nogil:
	'''
	The squared L2-norm of the point x.
	'''
	return x.x * x.x + x.y * x.y + x.z * x.z

cdef double ptdst(point x, point y) nogil:
	'''
	The Euclidean distance between points x and y.
	'''
	cdef double dx, dy, dz
	dx = x.x - y.x
	dy = x.y - y.y
	dz = x.z - y.z
	return sqrt(dx * dx + dy * dy + dz * dz)

cdef double dot(point l, point r) nogil:
	'''
	Return the inner product of two point structures.
	'''
	return l.x * r.x + l.y * r.y + l.z * r.z

cdef point cross(point l, point r) nogil:
	'''
	Return the cross product of two Point3D objects.
	'''
	cdef point o
	o.x = l.y * r.z - l.z * r.y
	o.y = l.z * r.x - l.x * r.z
	o.z = l.x * r.y - l.y * r.x
	return o

cpdef int almosteq(double x, double y) nogil:
	'''
	Returns True iff the difference between x and y is less than or
	 * equal to M * eps, where M = max(abs(x), abs(y), 1.0) and eps is the
	 * geometric mean of FLT_EPSILON and DBL_EPSILON.
	 '''
	cdef double mxy = fmax(fabs(x), fmax(fabs(y), 1.0))
	return fabs(x - y) <= REALEPS * mxy

cpdef double infdiv(double a, double b) nogil:
	'''
	Return a / b with special handling of small values:

	1. If |b| <= eps * |a|, return signed infinity,
	2. Otherwise, if |a| <= eps, return 0,

	where eps is the geometric mean of FLT_EPSILON and DBL_EPSILON.
	'''
	cdef double aa = fabs(a), ab = fabs(b)

	if (ab <= REALEPS * aa):
		if ((a >= 0) == (b >= 0)): return (1.0 / 0.0)
		else: return -(1.0 / 0.0)
	elif (aa <= REALEPS): return 0.0

	return a / b
