# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

cdef extern from "ptutils.c":
	ctypedef struct point:
		double x, y, z
	cdef const double realeps
	cdef point axpy(double a, point x, point y) nogil
	cdef point lintp(double t, point x, point y) nogil
	cdef point *iaxpy(double a, point x, point *y) nogil
	cdef point scal(double a, point x) nogil
	cdef point *iscal(double a, point *x) nogil
	cdef point *iptmpy(point h, point *x) nogil
	cdef point *iptdiv(point h, point *x) nogil
	cdef double ptnrm(point x) nogil
	cdef double ptsqnrm(point x) nogil
	cdef double ptdst(point x, point y) nogil
	cdef double dot(point l, point r) nogil
	cdef point cross(point l, point r) nogil
	cdef int almosteq(double x, double y) nogil
	cdef double infdiv(double a, double b) nogil


cdef inline object pt2tup(point a):
	return (a.x, a.y, a.z)

cdef inline int tup2pt(point *pt, object p) except -1:
	cdef double x, y, z
	x, y, z = p

	if pt != <point *>NULL:
		pt.x = x
		pt.y = y
		pt.z = z

	return 0

cdef inline point packpt(double x, double y, double z):
	cdef point r
	r.x = x
	r.y = y
	r.z = z
	return r

cdef inline void pt2arr(double *arr, point pt):
	arr[0] = pt.x
	arr[1] = pt.y
	arr[2] = pt.z
