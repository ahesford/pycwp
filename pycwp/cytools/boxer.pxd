'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

from ptutils cimport point

cdef class Segment3D:
	cdef point _start, _end
	cdef readonly double length

	cdef void setends(self, point start, point end)
	cdef int _ptdist(self, double *results, point pt, bint bounded)

cdef class Triangle3D:
	cdef point _nodes[3]
	cdef point _normal
	cdef point *_qr
	cdef unsigned int _labels[3]

	cdef readonly double offset
	cdef readonly double area

	cdef int setnodes(self, point nodes[3], unsigned int labels[3]) except -1
	cdef point *qrbary(self)
	cdef double _planedist(self, point p)

	@staticmethod
	cdef bint sat_cross_test(point a, point v[3], point hlen) nogil

cdef class Box3D:
	cdef point _lo, _hi, _length, _cell
	cdef unsigned long nx, ny, nz

	cdef int setbounds(self, point lo, point hi) except -1
	cdef point _cart2cell(self, double x, double y, double z) nogil
	cdef point _cell2cart(self, double i, double j, double k) nogil
	cpdef Box3D getCell(self, long i, long j, long k)
	cdef bint _contains(self, point p) nogil
	cdef void _boundsForCell(self, point *lo, point *hi, 
					long i, long j, long k) nogil

	@staticmethod
	cdef bint _intersection(double *t, point l, point h,
				point s, point e, bint halfline=*)

	cdef object _raymarcher(self, point start, point end, double step=*)
	cdef int _chkfzone(self, object hits, Segment3D seg, long *cell,
				double l, double tlen=*, double slen=*) except -1
	cdef void _cellForPoint(self, long *i, long *j, long *k, point p) nogil
