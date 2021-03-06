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

from libc.math cimport sqrt, fabs, acos

from ptutils cimport *
from interpolator cimport *

cdef class Segment3D:
	'''
	A representation of a 3-D line segment.

	Initialize as Segment3D(start, end), where start and end are three-
	element sequences providing the Cartesian coordinates of the respective
	start and end points of the segment.
	'''
	def __init__(self, start, end):
		'''
		Initialize a 3-D line segment that starts and ends at the
		indicated points (3-tuples of float coordinates).
		'''
		cdef point s, e
		tup2pt(&s, start)
		tup2pt(&e, end)
		self.setends(s, e)


	@cython.cdivision(True)
	cdef void setends(self, point start, point end):
		'''
		Initialize a 3-D line segment that starts and ends at the
		indicated points.
		'''
		# Copy the start and end points
		self._start = start
		self._end = end

		# Initialize dependent properties
		self.length = ptnrm(axpy(-1, start, end))


	@property
	def start(self):
		'''The start of the segment, as a 3-tuple of floats'''
		return pt2tup(self._start)

	@property
	def end(self):
		'''The end of the segment, as a 3-tuple of floats'''
		return pt2tup(self._end)

	@property
	def midpoint(self):
		'''The midpoint of the segment, as a 3-tuple of floats'''
		return pt2tup(lintp(0.5, self._start, self._end))

	@cython.cdivision(True)
	@property
	def direction(self):
		'''
		The normalized direction of the segment, as a 3-tuple of
		floats. If the segment length is approximately 0, the direction
		will be (0, 0, 0).
		'''
		cdef point ds = axpy(-1, self._start, self._end)
		if almosteq(self.length, 0.0): return (0, 0, 0)
		iscal(1 / self.length, &ds)
		return pt2tup(ds)

	@cython.cdivision(True)
	cdef int _ptdist(self, double *results, point pt, bint bounded) nogil:
		''' C backer for ptdist. '''
		cdef point ap, apn, n
		cdef double nn, ptd

		# Find vector from start to point
		ap = axpy(-1.0, pt, self._start)
		# Find vector from start to end
		n = axpy(-1.0, self._start, self._end)
		# Find square norm of line direction
		nn = ptsqnrm(n)

		if almosteq(nn, 0.0):
			# For negligible segments, get distance to start
			results[0] = ptnrm(ap)
			results[1] = 0.0
			return 0

		# Compute the fractional projection onto nn
		results[1] = -dot(ap, n) / nn

		if bounded:
			if results[1] < 0:
				# Point is before start; find distance to start
				results[0] = ptnrm(ap)
				return 0
			elif results[1] > 1:
				# Point is after end; find distance to end
				results[0] = ptnrm(axpy(-1.0, self._end, pt))
				return 0

		# Compute the length of the perpendicular component
		results[0] = ptnrm(axpy(results[1], n, ap))
		return 0

	def ptdist(self, p, bint bounded=False):
		'''
		For p, a sequence of 3 floats, calculate and return:

		  1. The perpendicular distance from p to the line coincident
		     with this segment (see note), and

		  2. The projection of the point p onto this segment, as a
		     fraction of self.length.

		If this segment is degenerate (self.direction is 0,0,0), the
		distance will be that from the point to self.start, while the
		projection will be 0.

		NOTE: If bounded is True, and the fractional projection (the
		second return value) is not in the range [0, 1], the computed
		distance (the first return value) will be the distance from p
		to self.start if the projection is negative and the distance
		from p to self.end if the projection is greater than unity.
		'''
		cdef point pt
		cdef double results[2]

		tup2pt(&pt, p)
		self._ptdist(results, pt, bounded)
		return results[0], results[1]

	def cartesian(self, double t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		return pt2tup(lintp(t, self._start, self._end))


	def bbox(self):
		'''
		A Box3D instance that bounds the segment.
		'''
		cdef double lx, ly, lz, hx, hy, hz

		# Populate the min-max values of each coordinate
		lx = min(self._start.x, self._end.x)
		hx = max(self._start.x, self._end.x)
		ly = min(self._start.y, self._end.y)
		hy = max(self._start.y, self._end.y)
		lz = min(self._start.z, self._end.z)
		hz = max(self._start.z, self._end.z)

		return Box3D((lx, ly, lz), (hx, hy, hz))


	def __repr__(self):
		start = pt2tup(self._start)
		end = pt2tup(self._end)
		return '%s(%r, %r)' % (self.__class__.__name__, start, end)


cdef class Triangle3D:
	'''
	A representation of a triangle embedded in 3-D space.

	Initialize as Triangle3D(nodes), where nodes is a sequence of three
	node descriptors, each of which is a three-element sequence which
	provides the Cartesian coordinates of the node.
	'''
	def __cinit__(self, *args, **kwargs):
		'''
		Make sure the QR pointer is NULL for proper management.
		'''
		self._qr = <point *>NULL

	def __init__(self, nodes, labels=None):
		'''
		Initialize a triangle from the sequence of three nodes (each a
		sequence of three floats).

		The argument labels, if provided, should be a sequence of three
		unique, nonnegative integers that label each node in order. If
		labels is omitted, a default labeling (0, 1, 2) will be used.
		'''
		cdef point pnodes[3]
		cdef unsigned int plabels[3]

		if len(nodes) != 3:
			raise ValueError('Length of nodes sequence must be 3')

		tup2pt(&pnodes[0], nodes[0])
		tup2pt(&pnodes[1], nodes[1])
		tup2pt(&pnodes[2], nodes[2])

		if labels is None:
			plabels[0], plabels[1], plabels[2] = 0, 1, 2
		else:
			if len(labels) != 3:
				raise ValueError('Length of labels sequence must be 3')
			if labels[0] < 0 or labels[1] < 0 or labels[2] < 0:
				raise ValueError('Node labels must be nonnegative')
			plabels[0], plabels[1], plabels[2] = labels[0], labels[1], labels[2]

		self.setnodes(pnodes, plabels)


	@cython.cdivision(True)
	cdef int setnodes(self, point nodes[3], unsigned int labels[3]) except -1:
		'''
		Initialize a triangle from the given nodes and integer labels.
		The labels must be nonnegative and unique.
		'''
		self._nodes[0] = nodes[0]
		self._nodes[1] = nodes[1]
		self._nodes[2] = nodes[2]

		self._labels[0] = labels[0]
		self._labels[1] = labels[1]
		self._labels[2] = labels[2]

		if (labels[0] == labels[1] or
				labels[1] == labels[2] or labels[0] == labels[2]):
			raise ValueError('Labels must be unique and nonnegative')

		# Find maximum cross product to determine normal
		cdef point l, r, v
		cdef double mag, vv
		cdef unsigned int j

		mag = -1

		cdef int i
		for i in range(3):
			j = (i + 1) % 3
			l = axpy(-1, self._nodes[i], self._nodes[j])
			j = (i + 2) % 3
			r = axpy(-1, self._nodes[i], self._nodes[j])

			v = cross(l, r)
			vv = ptsqnrm(v)

			if vv > mag:
				self._normal = v
				mag = vv

		# Length of cross product of two sides is twice triangle area
		mag = sqrt(mag)
		self.area = 0.5 * mag
		if almosteq(self.area, 0.0):
			raise ValueError('Triangle must have nonzero area')

		# Store the scaled normal and origin distance
		iscal(1. / mag, &(self._normal))
		self.offset = -dot(self._normal, self._nodes[0])

		if self._qr != <point *>NULL:
			# Make sure any existing QR decomposition is cleared
			PyMem_Free(<void *>self._qr)
			self._qr = <point *>NULL

		return 0

	def __dealloc__(self):
		if self._qr != <point *>NULL:
			PyMem_Free(<void *>self._qr)

	@property
	def nodes(self):
		'''The nodes of the triangle, as a 3-tuple of 3-tuples of floats'''
		return (pt2tup(self._nodes[0]),
				pt2tup(self._nodes[1]), pt2tup(self._nodes[2]))

	@property
	def labels(self):
		'''The nonnegative integer labels of the triangle nodes'''
		return (self._labels[0], self._labels[1], self._labels[2])

	@property
	def normal(self):
		'''The normal of the triangle, as a 3-tuple of floats'''
		return pt2tup(self._normal)

	def __repr__(self):
		return '%s(%r, %r)' % (self.__class__.__name__, self.nodes, self.labels)


	@cython.cdivision(True)
	cdef point *qrbary(self):
		'''
		Returns, as a pointer to an array [q0, q1, r] of point structs,
		a representation of the thin QR factorization of the matrix
		that relates barycentric coordinates l = [l[0], l[1]] to the
		coordinates of p - self.n2 for some 3-D point p. The third
		barycentric coordinate l[2] = 1 - l[1] - l[0].

		The factored matrix Q is represented as Q = [ q0 q1 ], where
		q0 and q1 are the two, three-dimensional, orthonormal column
		vectors spanning the range of the relational matrix.

		The matrix R is represented as r = (r[0], r[1], r[2]), such
		that R = [ r[0], r[1]; 0 r[2] ].

		The system will have no solution if the point p is not in the
		plane of the triangle. In this case, the returned pointer will
		be null but no other error will be raised.
		'''
		if self._qr != <point *>NULL:
			return self._qr

		# Here, self._qr will be NULL until the end
		qr = <point *>PyMem_Malloc(3 * sizeof(point));
		if qr == self._qr:
			# Allocation failed, nothing more to do
			return self._qr

		# Perform Gram-Schmidt orthogonalization to build Q and R
		qr[0] = axpy(-1.0, self._nodes[2], self._nodes[0])
		cdef double r0 = ptnrm(qr[0])
		if almosteq(r0, 0.0):
			# Triangle is degenerate
			PyMem_Free(<void *>qr)
			return self._qr
		iscal(1.0 / r0, &(qr[0]))

		qr[1] = axpy(-1.0, self._nodes[2], self._nodes[1])
		cdef double r1 = dot(qr[0], qr[1])
		iaxpy(-r1, qr[0], &(qr[1]))

		cdef double r2 = ptnrm(qr[1])
		if almosteq(r2, 0.0):
			PyMem_Free(<void *>qr)
			return self._qr
		iscal(1.0 / r2, &(qr[1]))

		qr[2] = packpt(r0, r1, r2)

		self._qr = qr
		return qr


	cdef double _planedist(self, point p):
		'''
		C helper for the self.halfspace.
		'''
		return dot(self._normal, p) + self.offset

	def planedist(self, p):
		'''
		Returns a double representing the distance from a 3-D point p
		to the plane in which this triangle resides. If the distance is
		positive, the point is in the positive half-space of the plane
		(i.e., self.normal points toward the point from the plane); if
		the distance is negative, the point is in the negative
		half-space of the plane.
		'''
		cdef point pp
		tup2pt(&pp, p)
		return self._planedist(pp)


	def bbox(self):
		'''
		A Box3D instance that bounds the triangle.
		'''
		cdef double lx, ly, lz, hx, hy, hz
		cdef point *n = self._nodes

		lx = min(n[0].x, n[1].x, n[2].x)
		hx = max(n[0].x, n[1].x, n[2].x)
		ly = min(n[0].y, n[1].y, n[2].y)
		hy = max(n[0].y, n[1].y, n[2].y)
		lz = min(n[0].z, n[1].z, n[2].z)
		hz = max(n[0].z, n[1].z, n[2].z)

		return Box3D((lx, ly, lz), (hx, hy, hz))


	@cython.cdivision(True)
	def barycentric(self, p, bint project=False):
		'''
		Convert the Cartesian coordinates p (a 3-tuple of floats) into
		barycentric coordinates within the triangle. If project is
		True, the point will be projected onto the plane of the triangle.

		If coordinates cannot be computed (either because the point is
		out of the plane and project is False, or because the QR
		factorization cannot be computed), None will be returned.
		'''
		cdef point pp
		tup2pt(&pp, p)
		cdef point d = axpy(-1.0, self._nodes[2], pp)

		# Make sure the point is in the plane, if necessary
		if not (project or almosteq(dot(d, self._normal), 0.0)):
			return None

		# Make sure the QR factorization exists
		cdef point *qr = self.qrbary()
		if qr == <point *>NULL: return None

		# Invert the orthogonal part of the QR system
		cdef double r2, r1, r0
		r0, r1, r2 = qr[2].x, qr[2].y, qr[2].z

		# Invert the triangular part of the QR system
		cdef double x1 = dot(d, qr[1]) / r2
		cdef double x0 = (dot(d, qr[0]) - r1 * x1) / r0

		return x0, x1, 1 - x0 - x1


	def cartesian(self, p):
		'''
		For a point p in barycentric coordinates (a tuple of 3 floats),
		return the corresponding Cartesian coordinates.
		'''
		cdef double vx, vy, vz, x, y, z
		cdef point *n = self._nodes

		x, y, z = p

		vx = x * n[0].x + y * n[1].x + z * n[2].x
		vy = x * n[0].y + y * n[1].y + z * n[2].y
		vz = x * n[0].z + y * n[1].z + z * n[2].z
		return vx, vy, vz


	@cython.cdivision(True)
	def perpangle(self, unsigned int d, n not None):
		'''
		Return the angle subtended by this triangle with the node at
		index d (0 <= d < 3) when projected perpendicular to a
		direction n = (nx, ny, nz).
		'''
		if d > 2: raise ValueError('Positive integer d must be less than 3')

		cdef point nn, a, b
		tup2pt(&nn, n)

		cdef double nnrm = ptnrm(nn)
		if almosteq(nnrm, 0.0):
			raise ValueError('Direction n must have nonzero norm')
		iscal(nnrm, &nn)

		cdef unsigned int d1, d2
		d1 = (d + 1) % 3
		d2 = (d + 2) % 3

		# Find the directions of the sides
		a = axpy(-1, self._nodes[d], self._nodes[d1])
		b = axpy(-1, self._nodes[d], self._nodes[d2])

		# Subtract components parallel to projection direction
		iaxpy(-dot(a, nn), nn, &a)
		iaxpy(-dot(b, nn), nn, &b)

		cdef double an, bn
		an = ptnrm(a)
		bn = ptnrm(b)

		# No angle if components are essentially parallel to direction
		if almosteq(an, 0.0) or almosteq(bn, 0.0): return 0.0
		# Return the actual angle
		return acos(min(1.0, max(-1.0, dot(a, b) / an / bn)))


	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in this triangle.
		'''
		cdef double x, y, z

		# Make sure to raise an error for incompatible p
		r = self.barycentric(p)

		if r is None: return False

		x, y, z = r

		if x < 0 or 1 < x: return False
		if y < 0 or 1 < y: return False
		if z < 0 or 1 < z: return False

		return True


	@staticmethod
	cdef bint sat_cross_test(point a, point v[3], point hlen) nogil:
		cdef double px, py, pz, r

		px = dot(a, v[0])
		py = dot(a, v[1])
		pz = dot(a, v[2])

		r = hlen.x * fabs(a.x) + hlen.y * fabs(a.y) + hlen.z * fabs(a.z)
		return not(min(px, py, pz) > r or max(px, py, pz) < -r)


	def overlaps(self, Box3D b not None):
		'''
		Returns True iff the Box3D b overlaps with this triangle.

		This algorithm uses the method of Akenine-Moeller (2001) to
		apply the separating axis theorem along the edges of the box
		(this reduces to a bounding-box test), along the normal of the
		facet (which determines whether the plane of the triangle cuts
		the box), and along nine remaining axis formed as the cross
		product of the edges of the box and edges of the triangle.
		'''
		cdef point lo = b._lo
		cdef point hi = b._hi
		cdef point *n = self._nodes

		# Out of bounds if all nodes are outside coordinate limits
		if ((n[0].x < lo.x and n[1].x < lo.x and n[2].x < lo.x) or
			(n[0].x > hi.x and n[1].x > hi.x and n[2].x > hi.x) or
			(n[0].y < lo.y and n[1].y < lo.y and n[2].y < lo.y) or
			(n[0].y > hi.y and n[1].y > hi.y and n[2].y > hi.y) or
			(n[0].z < lo.z and n[1].z < lo.z and n[2].z < lo.z) or
			(n[0].z > hi.z and n[1].z > hi.z and n[2].z > hi.z)):
			return False

		cdef point normal = self._normal

		# Pick the p and n vertices of the cube
		cdef point pv, nv

		if normal.x >= 0:
			pv.x, nv.x = hi.x, lo.x
		else:
			pv.x, nv.x = lo.x, hi.x

		if normal.y >= 0:
			pv.y, nv.y = hi.y, lo.y
		else:
			pv.y, nv.y = lo.y, hi.y

		if normal.z >= 0:
			pv.z, nv.z = hi.z, lo.z
		else:
			pv.z, nv.z = lo.z, hi.z

		# p-vertex is in negative half space of triangle plane
		if self._planedist(pv) < 0: return False
		# n-vertex is in positive half space of triangle plane
		if self._planedist(nv) > 0: return False

		# Perform remaining SAT tests on 9 cross-product axes
		cdef point hlen = scal(0.5, b._length)
		# Shift node origin to box center, regroup by node
		cdef point v[3]
		cdef point midpoint = lintp(0.5, lo, hi)

		v[0] = axpy(-1.0, midpoint, n[0])
		v[1] = axpy(-1.0, midpoint, n[1])
		v[2] = axpy(-1.0, midpoint, n[2])

		cdef unsigned long i, j
		cdef point f, a

		# Build the axes for each triangle edge
		for i in range(3):
			j = (i + 1) % 3
			f = axpy(-1, v[i], v[j])
			# Expand cross products to leverage simple form
			a.x, a.y, a.z = 0, -f.z, f.y
			if not Triangle3D.sat_cross_test(a, v, hlen): return False
			a.x, a.y, a.z = f.z, 0, -f.x
			if not Triangle3D.sat_cross_test(a, v, hlen): return False
			a.x, a.y, a.z = -f.y, f.x, 0
			if not Triangle3D.sat_cross_test(a, v, hlen): return False

		# All tests pass; overlap detected
		return True


	@cython.cdivision(True)
	def intersection(self, Segment3D seg not None, bint halfline=False):
		'''
		Return the intersection of the segment seg with this triangle
		as (l, t, u, v), where l is the length along the segment seg
		and (t, u, v) are the barycentric coordinates in the triangle.

		If the segment and triangle are in the same plane, this method
		raises a NotImplementedError without checking for intersection.

		Otherwise, the method returns the length t along the segment
		that defines the point of intersection. If the segment and
		triangle do not intersect, or the QR factorizaton could not be
		computed, None is returned.
		'''
		cdef point *qr = self.qrbary()
		if qr == <point *>NULL: return None

		cdef double r2, r1, r0
		r0, r1, r2 = qr[2].x, qr[2].y, qr[2].z

		# Extend barycentric QR factorization to a factorization
		# for parametric line-plane intersection

		# The last column of and RHS of the intersection problem
		cdef point y, q2
		cdef double r3, r4, r5
		q2 = axpy(-1.0, seg._end, seg._start)
		y = axpy(-1.0, self._nodes[2], seg._start)

		# Use modified Gram-Schmidt to find the last column
		r3 = dot(qr[0], q2)
		iaxpy(-r3, qr[0], &q2)
		r4 = dot(qr[1], q2)
		iaxpy(-r4, qr[1], &q2)
		r5 = ptnrm(q2)
		if almosteq(r5, 0.0):
			# Line is parallel to facet
			if almosteq(dot(y, self._normal), 0.0):
				raise NotImplementedError('Segment is in plane of facet')
			return None
		iscal(1.0 / r5, &q2)

		# Invert the QR factorization
		cdef point ny
		ny.x, ny.y, ny.z = dot(y, qr[0]), dot(y, qr[1]), dot(y, q2)

		cdef double l, t, u
		l = ny.z / r5
		u = (ny.y - r4 * l) / r2
		t = (ny.x - r3 * l - r1 * u) / r0

		# Check if the line intersection is valid
		cdef bint online = (l >= 0 and (halfline or l <= 1))
		if online and 0 <= t <= 1 and 0 <= u <= 1 and 0 <= t + u <= 1:
			# l is the fraction of segment length
			# t and u are normal barycentric coordinates in triangle
			return l, t, u, 1 - t - u
		else:
			# Intersection is not in segment or triangle
			return None


cdef class Box3D:
	'''
	A representation of an axis-aligned 3-D bounding box.

	Initialize as Box3D(lo, hi), where lo and hi are three-element
	sequences that provide the Cartesian coordinates of the low and high
	corners, respectively.
	'''
	def __init__(self, lo, hi, ncell=None):
		'''
		Initialize a 3-D box with extreme corners lo and hi (each a
		3-tuple of floats). If ncell is not None, it should be a
		3-tuple of integers to which the ncell property will be set.
		'''
		cdef point _lo, _hi
		tup2pt(&_lo, lo)
		tup2pt(&_hi, hi)
		self.setbounds(_lo, _hi)

		if ncell is not None: self.ncell = ncell


	cdef int setbounds(self, point lo, point hi) except -1:
		'''
		Initialize a 3-D box with extreme corners lo and hi.
		'''
		# Copy the extreme corners
		self._lo = lo
		self._hi = hi

		if (self._lo.x > self._hi.x or
				self._lo.y > self._hi.y or self._lo.z > self._hi.z):
			raise ValueError('Coordinates of hi must be no less than those of lo')

		self._length = axpy(-1.0, self._lo, self._hi)

		self.nx = self.ny = self.nz = 1
		self._cell = self._length

		return 0


	@property
	def lo(self):
		'''The lower corner of the box'''
		return pt2tup(self._lo)

	@property
	def hi(self):
		'''The upper corner of the box'''
		return pt2tup(self._hi)

	@property
	def midpoint(self):
		'''The barycenter of the box'''
		return pt2tup(lintp(0.5, self._lo, self._hi))

	@property
	def length(self):
		'''The length of the box edges'''
		return pt2tup(self._length)

	@property
	def cell(self):
		'''The length of each cell into which the box is subdivided'''
		return pt2tup(self._cell)

	@property
	def ncell(self):
		'''The number of cells (nx, ny, nz) that subdivide the box'''
		return self.nx, self.ny, self.nz

	@ncell.setter
	def ncell(self, c):
		cdef long x, y, z
		x, y, z = c

		if x < 1 or y < 1 or z < 1:
			raise ValueError('Grid dimensions must all be positive')

		self.nx = <unsigned long>x
		self.ny = <unsigned long>y
		self.nz = <unsigned long>z

		with cython.cdivision(True):
			self._cell.x = self._length.x / <double>self.nx
			self._cell.y = self._length.y / <double>self.ny
			self._cell.z = self._length.z / <double>self.nz

	def __repr__(self):
		ncell = self.ncell
		lo = self.lo
		hi = self.hi
		cls = self.__class__.__name__
		if any(c != 1 for c in ncell):
			return '%s(%r, %r, %r)' % (cls, lo, hi, ncell)
		return '%s(%r, %r)' % (cls, lo, hi)


	@cython.cdivision(True)
	cdef point _cart2cell(self, double x, double y, double z) nogil:
		'''C backer for cart2cell'''
		cdef point ptcell

		ptcell.x = (x - self._lo.x) / self._cell.x
		ptcell.y = (y - self._lo.y) / self._cell.y
		ptcell.z = (z - self._lo.z) / self._cell.z

		return ptcell

	def cart2cell(self, double x, double y, double z):
		'''
		Convert the 3-D Cartesian coordinates (x, y, z) into grid
		coordinates defined by the box bounds and ncell property. The
		coordinates will always be real-valued and may have a fraction
		part to indicate relative positions within the cell.
		'''
		return pt2tup(self._cart2cell(x, y, z))


	@cython.cdivision
	cdef point _cell2cart(self, double i, double j, double k) nogil:
		cdef point ptcrd

		ptcrd.x = i * self._cell.x + self._lo.x
		ptcrd.y = j * self._cell.y + self._lo.y
		ptcrd.z = k * self._cell.z + self._lo.z

		return ptcrd

	def cell2cart(self, double i, double j, double k):
		'''
		Convert the (possibly fractional) 3-D cell-index coordinates
		(i, j, k), defined by the box bounds and ncell property, into
		Cartesian coordinates.
		'''
		return pt2tup(self._cell2cart(i, j, k))


	cdef void _boundsForCell(self, point *lo,
			point *hi, long i, long j, long k) nogil:
		'''
		Compute, in lo and hi, the respective lower and upper corners
		of the cell at index (i, j, k) in the grid defined for this box.

		The cell does not have to be within the grid limits.
		'''
		lo[0] = self._cell2cart(i, j, k)
		hi[0] = self._cell2cart(i + 1, j + 1, k + 1)


	cpdef Box3D getCell(self, long i, long j, long k):
		'''
		Return a Box3D representing the cell that contains 3-D index c
		based on the grid defined by the ncell property. Any fractional
		part of the coordinates will be truncated.
		'''
		cdef point lo, hi
		self._boundsForCell(&lo, &hi, i, j, k)
		return Box3D(pt2tup(lo), pt2tup(hi))


	def allIndices(self):
		'''
		Return a generator that produces every 3-D cell index within
		the grid defined by the ncell property in row-major order.
		'''
		cdef long i, j, k
		for i in range(self.nx):
			for j in range(self.ny):
				for k in range(self.nz):
					yield i, j, k


	def allCells(self, bint enum=False):
		'''
		Return a generator that produces every cell in the grid defined
		by the ncell property. Generation is done in the same order as
		self.allIndices().

		If enum is True, return a tuple (idx, box), where idx is the
		three-dimensional index of the cell.
		'''
		cdef long i, j, k
		for i, j, k in self.allIndices():
			box = self.getCell(i, j, k)
			if not enum: yield box
			else: yield ((i, j, k), box)

	def overlaps(self, Box3D b not None):
		'''
		Returns True iff the Box3D b overlaps with this box.
		'''
		cdef double alx, aly, alz, ahx, ahy, ahz
		cdef double blx, bly, blz, bhx, bhy, bhz

		alx, aly, alz = self._lo.x, self._lo.y, self._lo.z
		ahx, ahy, ahz = self._hi.x, self._hi.y, self._hi.z
		blx, bly, blz = b._lo.x, b._lo.y, b._lo.z
		bhx, bhy, bhz = b._hi.x, b._hi.y, b._hi.z

		return not (ahx < blx or alx > bhx or ahy < bly or
				aly > bhy or ahz < blz or alz > bhz)


	cdef bint _contains(self, point p) nogil:
		'''C backer for contains'''
		return ((p.x >= self._lo.x) and (p.x <= self._hi.x)
				and (p.y >= self._lo.y) and (p.y <= self._hi.y)
				and (p.z >= self._lo.z) and (p.z <= self._hi.z))


	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in the box.
		'''
		cdef point pp
		tup2pt(&pp, p)
		return self._contains(pp)


	@staticmethod
	cdef bint _intersection(double *t, point l, point h,
			point s, point e, bint halfline=False):
		'''
		Low-level routine to compute intersection of a box, with low
		and high corners l and h, respectively, with a line segment
		that has starting point s and end point e. Optionally, the
		intersection between the box and a half-infinite ray that
		starts at s and passes through e can be found instead.

		If the segment (or ray, when halfline is True) intersects the
		box, True is returned and the values t[0] and t[1] represent,
		respectively, the minimum and maximum lengths along the ray or
		segment (as a multiple of the length of the segment) that
		describe the points of intersection. The value t[0] may be
		negative if the ray or segment starts within the box. The value
		t[1] may exceed unity if a segment ends within the box or
		halfline is True.

		If no intersection is found, False is returned and t is
		untouched.
		'''
		cdef double tmin, tmax, ty1, ty2, tz1, tz2, d

		# Check, in turn, intersections with the x, y and z slabs
		d = e.x - s.x
		tmin = infdiv(l.x - s.x, d)
		tmax = infdiv(h.x - s.x, d)
		if tmax < tmin: tmin, tmax = tmax, tmin
		# Check the y-slab
		d = e.y - s.y
		ty1 = infdiv(l.y - s.y, d)
		ty2 = infdiv(h.y - s.y, d)
		if ty2 < ty1: ty1, ty2 = ty2, ty1
		if ty2 < tmax: tmax = ty2
		if ty1 > tmin: tmin = ty1
		# Check the z-slab
		d = e.z - s.z
		tz1 = infdiv(l.z - s.z, d)
		tz2 = infdiv(h.z - s.z, d)
		if tz2 < tz1: tz1, tz2 = tz2, tz1
		if tz2 < tmax: tmax = tz2
		if tz1 > tmin: tmin = tz1

		if tmax < max(0, tmin) or (not halfline and tmin > 1):
			return False

		t[0] = tmin
		t[1] = tmax
		return True

	def intersection(self, Segment3D seg not None, bint halfline=False):
		'''
		Returns the lengths tmin and tmax along the given Segment3D seg
		at which the segment enters and exits the box, as a multiple of
		the segment length. If the box does not intersect the segment,
		returns None.

		If the segment starts within the box, tmin will be negative. If
		the segment ends within the box, tmax will exceed the segment
		length.

		If halfline is False, the segment and box must overlap for an
		intersection to be returned. If halfline is True, an
		intersection will be returned whenever the half-infinite ray
		that starts at seg.start and passes through seg.end intersects
		the box.
		'''
		cdef double tlims[2]
		if Box3D._intersection(tlims, self._lo,
				self._hi, seg._start, seg._end, halfline):
			return tlims[0], tlims[1]
		else: return None

	def neighborhood(self, p, length):
		'''
		For a single cell index triplet or collection of cell index
		triplets p, compute a set of all cells in the neighborhood of
		p. The neighborhood is defined by length, which must either be
		a scalar or a triplet of floating-point-compatible values, and
		is given by all cells c such that, for some pv in p,

			abs(pv[i] - c[i]) < int(length[i] / self.cell[i])

		for all coordinate indices 0 <= i < 3. (If length is a scalar,
		it is treated as [length, length, length].)

		If the values in p are not integers, they will be converted to
		integers for this method.
		'''
		cdef long[:,:] pts
		cdef long i, j, k, row, lx, hx, ly, hy, lz, hz
		cdef long ncell[3]

		p = np.asarray(p, dtype=np.int64)
		if p.ndim == 1: pts = p[np.newaxis,:]
		elif p.ndim == 2: pts = p
		else: raise ValueError('Argument "p" must be a 1-D or 2-D array')

		if pts.shape[1] != 3:
			raise ValueError('Argument "p" must have shape (N,3)')

		length = np.asarray(length, dtype=np.float64)
		if length.ndim == 0:
			ncell[0] = <long>(length / self._cell.x)
			ncell[1] = <long>(length / self._cell.y)
			ncell[2] = <long>(length / self._cell.z)
		elif length.ndim != 1 or length.shape[0] != 3:
			raise ValueError('Argument "length" must be a scalar or 1-D array of length 3')
		else:
			ncell[0] = <long>(length[0] / self._cell.x)
			ncell[1] = <long>(length[1] / self._cell.y)
			ncell[2] = <long>(length[2] / self._cell.z)

		neighbors = set()
		for row in range(pts.shape[0]):
			cx, cy, cz = pts[row,0], pts[row,1], pts[row,2]
			lx = max(0, pts[row,0] - ncell[0])
			hx = min(pts[row,0] + ncell[0] + 1, self.nx)
			ly = max(0, pts[row,1] - ncell[1])
			hy = min(pts[row,1] + ncell[1] + 1, self.ny)
			lz = max(0, pts[row,2] - ncell[2])
			hz = min(pts[row,2] + ncell[2] + 1, self.nz)

			for i in range(lx, hx):
				for j in range(ly, hy):
					for k in range(lz, hz):
						neighbors.add((i, j, k))

		return neighbors

	cdef void _cellForPoint(self, long *i, long *j, long *k, point p) nogil:
		'''
		Compute, in i, j and k, the coordinates of the cell that
		contains the point p.
		'''
		cdef point c = self._cart2cell(p.x, p.y, p.z)
		i[0] = <long>c.x
		j[0] = <long>c.y
		k[0] = <long>c.z

	def cellForPoint(self, double x, double y, double z):
		'''
		Compute the cell (i, j, k) that contains the point with
		Cartesian coordinates (x, y, z).
		'''
		cdef long i, j, k
		self._cellForPoint(&i, &j, &k, packpt(x, y, z))
		return i, j, k
