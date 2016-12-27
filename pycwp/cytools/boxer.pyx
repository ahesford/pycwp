'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython cimport floating as real

from math import sqrt
from libc.math cimport sqrt, floor

from collections import OrderedDict

from itertools import izip, product as iproduct

cdef extern from "float.h":
	cdef double FLT_EPSILON
	cdef double DBL_EPSILON

cdef double infinity
with cython.cdivision(True):
	infinity = 1.0 / 0.0

cdef double realeps = sqrt(FLT_EPSILON * DBL_EPSILON)

cdef struct point:
	double x, y, z

cdef point axpy(double a, point x, point y):
	'''
	Return a point struct equal to (a * x + y).
	'''
	cdef point r
	r.x = a * x.x + y.x
	r.y = a * x.y + y.y
	r.z = a * x.z + y.z
	return r


cdef point * iaxpy(double a, point x, point *y):
	'''
	Store, in y, the value of (a * x + y).
	'''
	y.x += a * x.x
	y.y += a * x.y
	y.z += a * x.z
	return y


cdef point scal(double a, point x):
	'''
	Scale the point x by a.
	'''
	cdef point r
	r.x = x.x * a
	r.y = x.y * a
	r.z = x.z * a
	return r


cdef point * iscal(double a, point *x):
	'''
	Scale the point x, in place, by a.
	'''
	x.x *= a
	x.y *= a
	x.z *= a
	return x


cdef double ptnrm(point x, bint squared=0):
	cdef double ns = x.x * x.x + x.y * x.y + x.z * x.z
	if squared: return ns
	else: return sqrt(ns)


cdef double dot(point l, point r):
	'''
	Return the inner product of two point structures.
	'''
	return l.x * r.x + l.y * r.y + l.z * r.z


cdef point cross(point l, point r):
	'''
	Return the cross product of two Point3D objects.
	'''
	cdef point o
	o.x = l.y * r.z - l.z * r.y
	o.y = l.z * r.x - l.x * r.z
	o.z = l.x * r.y - l.y * r.x
	return o


cdef bint almosteq(double x, double y, double eps=realeps):
	'''
	Returns True iff the difference between x and y is less than or equal
	to M * eps, where M = max(abs(x), abs(y), 1.0).
	'''
	cdef double mxy = max(abs(x), abs(y), 1.0)
	return abs(x - y) <= eps * mxy


@cython.cdivision(True)
cdef double infdiv(double a, double b, double eps=realeps):
	'''
	Return a / b with special handling of small values:

	1. If |b| <= eps * |a|, return signed infinity,
	2. Otherwise, if |a| <= eps, return 0.
	'''
	cdef double aa, bb
	aa = abs(a)
	ab = abs(b)

	if ab <= eps * aa:
		if (a >= 0) == (b >= 0):
			return infinity
		else: return -infinity
	elif aa <= eps:
		return 0.0

	return a / b


cdef point packpt(double x, double y, double z):
	cdef point r
	r.x = x
	r.y = y
	r.z = z
	return r


cdef object pt2tup(point a):
	return (a.x, a.y, a.z)


cdef bint tup2pt(point *pt, object p) except -1:
	cdef double x, y, z
	x, y, z = p

	if pt != <point *>NULL:
		pt.x = x
		pt.y = y
		pt.z = z

	return 0


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned int grad(point *result, real[:,:,:] f, unsigned long i,
		unsigned long j, unsigned long k, point h):
	'''
	Computes, in result, the gradient of the field f, using central
	differencing away from boundaries and one-sided differencing at the
	boundaries. The return value is a bit field b, with value

		b = (gz << 2) | (gy << 1) | gx,

	where gx, gy, and gz are 0 if the respective x, y, or z components of
	the gradient were computed and 1 otherwise (i.e., if the index is out
	of bounds or the dimension of the field along that axis is unity).
	'''
	cdef unsigned long nx, ny, nz
	cdef bint has_left, has_right
	cdef unsigned int gdirs = 0
	cdef real lval, rval, step

	nx = f.shape[0]
	ny = f.shape[1]
	nz = f.shape[2]

	# Ensure the point is in bounds or indicate no computation
	if i >= nx or j >= ny or k >= ny: return 7

	# Find x derivative
	has_left = (i > 1)
	if has_left:
		lval = f[i - 1, j, k]
		step = h.x
	else:
		# No step to the left
		lval = f[i, j, k]
		step = 0

	has_right = (i < (nx - 1))
	if has_right:
		rval = f[i + 1, j, k]
		step += h.x
	else:
		# No step to the right
		rval = f[i, j, k]

	if has_left or has_right:
		result.x = (rval - lval) / step
	else: gdirs = 1

	# Find y derivative
	has_left = (j > 1)
	if has_left:
		lval = f[i, j - 1, k]
		step = h.y
	else:
		# No step to the left
		lval = f[i, j, k]
		step = 0

	has_right = (j < (ny - 1))
	if has_right:
		rval = f[i, j + 1, k]
		step += h.y
	else:
		# No step to the right
		rval = f[i, j, k]

	if has_left or has_right:
		result.y = (rval - lval) / step
	else: gdirs |= (1 << 1)

	# Find z derivative
	has_left = (k > 1)
	if has_left:
		lval = f[i, j, k - 1]
		step = h.z
	else:
		# No step to the left
		lval = f[i, j, k]
		step = 0

	has_right = (j < (nz - 1))
	if has_right:
		rval = f[i, j, k + 1]
		step += h.z
	else:
		# No step to the right
		rval = f[i, j, k]

	if has_left or has_right:
		result.z = (rval - lval) / step
	else: gdirs |= (1 << 2)

	return gdirs


cdef class Segment3D:
	'''
	A representation of a 3-D line segment.
	'''
	# Intrinsic properties of the segment
	cdef point _start, _end

	# Dependent properties (accessible from Python)
	cdef readonly double length
	cdef readonly unsigned int majorAxis

	# Dependent properties (hidden from Python)
	cdef point _midpoint, _direction

	@cython.cdivision(True)
	def __cinit__(self, start, end):
		'''
		Initialize a 3-D line segment that starts and ends at the
		indicated points (3-tuples of float coordinates).
		'''
		tup2pt(&self._start, start)
		tup2pt(&self._end, end)

		# Initialize dependent properties
		self._direction = axpy(-1.0, self._start, self._end)
		self.length = ptnrm(self._direction)

		if not almosteq(self.length, 0.0):
			iscal(1. / self.length, &(self._direction))

		self._midpoint = axpy(1.0, self._start, self._end)
		iscal(0.5, &(self._midpoint))

		cdef double ma, ca
		self.majorAxis = 0
		ma = abs(self._direction.x)

		ca = abs(self._direction.y)
		if ca > ma:
			ma = ca
			self.majorAxis = 1

		ca = abs(self._direction.z)
		if ca > ma:
			ma = ca
			self.majorAxis = 2


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
		return pt2tup(self._midpoint)

	@property
	def direction(self):
		'''The direction of the segment, as a 3-tuple of floats'''
		return pt2tup(self._direction)


	@cython.embedsignature(True)
	def cartesian(self, double t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		return pt2tup(axpy(t, self._direction, self._start))


	@cython.embedsignature(True)
	cpdef double lengthToAxisPlane(self, double c, unsigned int axis) except? -1:
		'''
		Return the signed distance along the segment from the start to
		the plane defined by a constant value c in the specified axis.

		If the segment and plane are parallel, the result will be
		signed infinity if the plane does not contain the segment. If
		the plane contains the segment, the length will be 0.
		'''
		if axis == 0:
			return infdiv(c - self._start.x, self._direction.x)
		elif axis == 1:
			return infdiv(c - self._start.y, self._direction.y)
		elif axis == 2:
			return infdiv(c - self._start.z, self._direction.z)
		else:
			raise IndexError('Argument axis must be 0, 1 or 2')


	@cython.embedsignature(True)
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
	'''
	cdef point _nodes[3]
	cdef point _normal
	cdef point *_qr

	cdef readonly double offset
	cdef readonly double area

	@cython.cdivision(True)
	def __cinit__(self, nodes):
		'''
		Initialize a triangle from nodes in the given sequence.
		'''
		if len(nodes) != 3:
			raise ValueError('Length of nodes sequence must be 3')

		cdef unsigned int i
		for i in range(3):
			tup2pt(&(self._nodes[i]), nodes[i])

		# Find maximum cross product to determine normal
		cdef point l, r, v
		cdef double mag, vv
		cdef unsigned int j

		mag = -1

		for i in range(3):
			j = (i + 1) % 3
			l = axpy(-1, self._nodes[i], self._nodes[j])
			j = (i + 2) % 3
			r = axpy(-1, self._nodes[i], self._nodes[j])

			v = cross(l, r)
			vv = ptnrm(v, 1)

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

		self._qr = <point *>NULL

	def __dealloc__(self):
		if self._qr != <point *>NULL:
			PyMem_Free(<void *>self._qr)

	@property
	def nodes(self):
		'''The nodes of the triangle, as a 3-tuple of 3-tuples of floats'''
		return (pt2tup(self._nodes[0]),
				pt2tup(self._nodes[1]), pt2tup(self._nodes[2]))

	@property
	def normal(self):
		'''The normal of the triangle, as a 3-tuple of floats'''
		return pt2tup(self._normal)

	def __repr__(self):
		return '%s(%r)' % (self.__class__.__name__, self.nodes)


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
		if qr == self._qr: return self._qr

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


	@cython.embedsignature(True)
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


	@cython.embedsignature(True)
	def cartesian(self, p):
		'''
		For a point p in barycentric coordinates (a tuple of 3 floats),
		return the corresponding Cartesian coordinates.
		'''
		cdef double vx, vy, vz, x, y, z
		cdef point *n = self._nodes

		x, y, z = p

		vx = x * n[0].x + y * n[1].x + z * n.x
		vy = x * n[0].y + y * n[1].y + z * n.y
		vz = x * n[0].z + y * n[1].z + z * n.z
		return vx, vy, vz


	@cython.embedsignature(True)
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
	cdef bint sat_cross_test(point a, point v[3], point hlen):
		cdef double px, py, pz, r

		px = dot(a, v[0])
		py = dot(a, v[1])
		pz = dot(a, v[2])

		r = hlen.x * abs(a.x) + hlen.y * abs(a.y) + hlen.z * abs(a.z)
		return not(min(px, py, pz) > r or max(px, py, pz) < -r)


	@cython.embedsignature(True)
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
		if dot(normal, pv) < -self.offset: return False
		# n-vertex is in positive half space of triangle plane
		if dot(normal, nv) > -self.offset: return False

		# Perform remaining SAT tests on 9 cross-product axes
		cdef point hlen = scal(0.5, b._length)
		# Shift node origin to box center, regroup by node
		cdef point v[3]

		v[0] = axpy(-1.0, b._midpoint, n[0])
		v[1] = axpy(-1.0, b._midpoint, n[1])
		v[2] = axpy(-1.0, b._midpoint, n[2])

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
	@cython.embedsignature(True)
	def intersection(self, Segment3D seg not None):
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
				raise NotImplementedError('Segment seg is in plane of facet')
			return None
		iscal(1.0 / r5, &q2)

		# Invert the QR factorization
		cdef point ny
		ny.x, ny.y, ny.z = dot(y, qr[0]), dot(y, qr[1]), dot(y, q2)

		cdef double t, u, v
		v = ny.z / r5
		u = (ny.y - r4 * v) / r2
		t = (ny.x - r3 * v - r1 * u) / r0

		if ((0 <= v <= 1 and 0 <= u <= 1 and 0 <= t <= 1) and 0 <= u + t <= 1):
			# v is the fraction of segment length
			# t and u are normal barycentric coordinates in triangle
			return v * seg.length, t, u, 1 - t - u
		else:
			# Intersection is not in segment or triangle
			return None


cdef class Box3D:
	'''
	A representation of an axis-aligned 3-D bounding box.
	'''
	cdef point _lo, _hi, _midpoint, _length, _cell
	cdef unsigned long nx, ny, nz

	def __cinit__(self, lo, hi):
		'''
		Initialize a 3-D box with extreme corners lo and hi.
		'''
		tup2pt(&self._lo, lo)
		tup2pt(&self._hi, hi)

		if (self._lo.x > self._hi.x or
				self._lo.y > self._hi.y or self._lo.z > self._hi.z):
			raise ValueError('Coordinates of hi must be no less than those of lo')

		# Compute some dependent properties
		self._midpoint = axpy(1.0, self._lo, self._hi)
		iscal(0.5, &(self._midpoint))

		self._length = axpy(-1.0, self._lo, self._hi)

		self.nx = self.ny = self.nz = 1
		self._cell = self._length


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
		return pt2tup(self._midpoint)

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
		return '%s(%r, %r)' % (self.__class__.__name__, self.lo, self.hi)


	@cython.cdivision(True)
	cdef point _cart2cell(self, double x, double y, double z):
		'''C backer for cart2cell'''
		cdef point ptcell

		ptcell.x = (x - self._lo.x) / self._cell.x
		ptcell.y = (y - self._lo.y) / self._cell.y
		ptcell.z = (z - self._lo.z) / self._cell.z

		return ptcell

	@cython.embedsignature(True)
	def cart2cell(self, double x, double y, double z):
		'''
		Convert the 3-D Cartesian coordinates (x, y, z) into grid
		coordinates defined by the box bounds and ncell property. The
		coordinates will always be real-valued and may have a fraction
		part to indicate relative positions within the cell.
		'''
		return pt2tup(self._cart2cell(x, y, z))


	@cython.cdivision
	cdef point _cell2cart(self, double i, double j, double k):
		cdef point ptcrd

		ptcrd.x = i * self._cell.x + self._lo.x
		ptcrd.y = j * self._cell.y + self._lo.y
		ptcrd.z = k * self._cell.z + self._lo.z

		return ptcrd

	@cython.embedsignature(True)
	def cell2cart(self, double i, double j, double k):
		'''
		Convert the (possibly fractional) 3-D cell-index coordinates
		(i, j, k), defined by the box bounds and ncell property, into
		Cartesian coordinates.
		'''
		return pt2tup(self._cell2cart(i, j, k))


	cdef void _boundsForCell(self, point *lo, point *hi, long i, long j, long k):
		'''
		Compute, in lo and hi, the respective lower and upper corners
		of the cell at index (i, j, k) in the grid defined for this box.

		The cell does not have to be within the grid limits.
		'''
		lo[0] = self._cell2cart(i, j, k)
		hi[0] = self._cell2cart(i + 1, j + 1, k + 1)


	@cython.embedsignature(True)
	cpdef Box3D getCell(self, long i, long j, long k):
		'''
		Return a Box3D representing the cell that contains 3-D index c
		based on the grid defined by the ncell property. Any fractional
		part of the coordinates will be truncated.
		'''
		cdef point lo, hi
		self._boundsForCell(&lo, &hi, i, j, k)
		return Box3D(pt2tup(lo), pt2tup(hi))


	@cython.embedsignature(True)
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


	@cython.embedsignature(True)
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

	@cython.embedsignature(True)
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


	cdef bint _contains(self, point p):
		'''C backer for contains'''
		return ((p.x >= self._lo.x) and (p.x <= self._hi.x)
				and (p.y >= self._lo.y) and (p.y <= self._hi.y)
				and (p.z >= self._lo.z) and (p.z <= self._hi.z))


	@cython.embedsignature(True)
	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in the box.
		'''
		cdef point pp
		tup2pt(&pp, p)
		return self._contains(pp)


	@staticmethod
	cdef bint _intersection(double *t, point l, point h, point s, point d, double sl):
		'''
		Low-level routine to compute intersection of a box, with low
		and high corners l and h, respectively, with a ray or segment
		that has starting point s, direction d, and a length sl (for
		ray intersections, sl should be negative or infinite).

		If the segment or ray intersects the box, True is returned and
		the values t[0] and t[1] represent, respectively, the minimum
		and maximum lengths along the ray or segment (in units of the
		magnitude of d) that describe the points of intersection. The
		value t[0] may be negative if the ray or segment starts within
		the box. The value t[1] may exceed the length if a segment ends
		within the box.

		If no intersection is found, False is returned.
		'''
		cdef double tmin, tmax, ty1, ty2, tz1, tz2

		# Check, in turn, intersections with the x, y and z slabs
		tmin = infdiv(l.x - s.x, d.x)
		tmax = infdiv(h.x - s.x, d.x)
		if tmax < tmin: tmin, tmax = tmax, tmin
		# Check the y-slab
		ty1 = infdiv(l.y - s.y, d.y)
		ty2 = infdiv(h.y - s.y, d.y)
		if ty2 < ty1: ty1, ty2 = ty2, ty1
		if ty2 < tmax: tmax = ty2
		if ty1 > tmin: tmin = ty1
		# Check the z-slab
		tz1 = infdiv(l.z - s.z, d.z)
		tz2 = infdiv(h.z - s.z, d.z)
		if tz2 < tz1: tz1, tz2 = tz2, tz1
		if tz2 < tmax: tmax = tz2
		if tz1 > tmin: tmin = tz1

		if tmax < max(0, tmin) or (sl >= 0 and tmin > sl):
			return False

		t[0] = tmin
		t[1] = tmax
		return True

	@cython.embedsignature(True)
	def intersection(self, Segment3D seg not None):
		'''
		Returns the lengths tmin and tmax along the given Segment3D seg
		at which the segment enters and exits the box. If the box does
		not intersect the segment, returns None.

		If the segment starts within the box, tmin will be negative. If
		the segment ends within the box, tmax will exceed the segment
		length.
		'''
		cdef double tlims[2]
		if Box3D._intersection(tlims, self._lo, self._hi,
				seg._start, seg._direction, seg.length):
			return tlims[0], tlims[1]
		else: return None


	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.embedsignature(True)
	def descent(self, p, t, real[:,:,:] f not None,
			bint report=False, double step=realeps, double rtol=1e-6):
		'''
		Starting at a point p = (x, y, z), where x, y, z are Cartesian
		coordinates within the bounds of this box, perform a
		steepest-descent walk (against the gradient) through the given
		scalar field f. The walk ends when one of:

		1. The cell containing the destination point t is reached,
		2. The path runs off the edge of the grid,
		3. The path returns to a previously-encountered cell, or
		4. The value |grad(f)| <= rtol * |f| in an encountered cell.

		The return value is an OrderedDict mapping (i, j, k) cell
		indices to the start and end points of the path through that
		cell. If p is outside the grid, the mapping will be empty.

		If the argument report is True, a second return value, a
		string with the value 'destination', 'boundary', 'stationary',
		'cycle' or 'stationary', corresponding to each of the above
		four reasons for termination, will indicate the reason for
		terminating the walk.

		The argument step is interpreted as in Box3D.raymarcher; the
		step is used (in adaptively increasing increments) to ensure
		that the segment in each cell advances to a neighboring cell.

		A ValueError will be raised if f is of the wrong shape.
		'''
		if (f.shape[0] != self.nx or
				f.shape[1] != self.ny or f.shape[2] != self.nz):
			raise ValueError('Shape of field f must be %s' % (self.ncell,))

		# Convert the start and end to points
		cdef point pp, tp
		tup2pt(&pp, p)
		tup2pt(&tp, t)

		# The maximum step through any cell cannot exceed cell diagonal
		cdef double hmax
		hmax = sqrt(self._cell.x**2 + self._cell.y**2 + self._cell.z**2)

		cdef point lo, hi, gf, np
		cdef double mgf
		cdef double tlims[2]
		cdef long i, j, k, ti, tj, tk

		# Find cell containing starting point
		self._cellForPoint(&i, &j, &k, pp)

		# Find cell containing ending point (may be out of bounds)
		self._cellForPoint(&ti, &tj, &tk, tp)

		# For convenience
		cdef long nx, ny, nz
		nx, ny, nz = self.nx, self.ny, self.nz

		# Record the paths encountered in the walk
		hits = OrderedDict()
		reason = None

		while True:
			if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
				# Walk has left the bounds of the box
				reason = 'boundary'
				break

			key = i, j, k

			if key in hits:
				# Encountered a cell previously encountered
				reason = 'cycle'
				break

			if i == ti and j == tj and k == tk:
				# Boundary cell has been encountered
				# Walk ends at destination point
				hits[key] = (pp.x, pp.y, pp.z), (tp.x, tp.y, tp.z)
				reason = 'destination'
				break

			# Find the boundaries of the current cell
			self._boundsForCell(&lo, &hi, i, j, k)

			# Find the direction of travel in this cell
			if grad(&gf, f, i, j, k, self._cell) != 0:
				raise ValueError('Gradient does not exist at %r' % (key,))
			mgf = ptnrm(gf)
			if mgf < rtol * abs(f[i,j,k]):
				# Stationary point has been encountered
				# Walk ends, start and end points are the same
				hits[key] = (pp.x, pp.y, pp.z), (pp.x, pp.y, pp.z)
				reason = 'stationary'
				break
			iscal(-1.0 / mgf, &gf)

			# Cast a ray through the cell (negative length is ignored)
			if not Box3D._intersection(tlims, lo, hi, pp, gf, -1):
				raise ValueError('Segment fails to intersect cell %s' % (key,))

			# Record start and end points in this cell
			np = axpy(tlims[1], gf, pp)

			hits[key] = (pp.x, pp.y, pp.z), (np.x, np.y, np.z)

			# Advance the point and find the next cell
			pp = np
			# Step into next box should never exceed cell diagonal
			self._advance(&i, &j, &k, 0.0, step, hmax, pp, gf)

		if report: return hits, reason
		else: return hits


	@cython.embedsignature(True)
	def raymarcher(self, Segment3D seg not None, double step=realeps):
		'''
		Marches along the given Segment3D seg to identify cells in the
		grid (defined by the ncell property) that intersect the
		segment. Returns a map (i, j, k) -> (tmin, tmax), where the key
		(i, j, k) is an index of a cell in the grid and the values tmin
		and tmax are the lengths along the segment of the entry and
		exit points, respectively, through the cell.

		As the segment exits each encountered cell, a step along the
		segment is taken to advance into another intersecting cell. The
		length of the step will be

			step * sum(2**i for i in range(q)),

		where the q is chosen at each step as the minimum nonnegative
		integer that guarantees advancement to another cell. Because
		this step may be nonzero, cells which intersect the segment
		over a total length less than step may be excluded from the
		intersection map.

		If the segment begins or ends in a cell, tmin or tmax for that
		cell may fall outside the range [0, seg.length].
		'''
		# Capture segment parameters for convenience
		cdef point sst = seg._start
		cdef point sdr = seg._direction
		cdef double sl = seg.length

		# Try to grab the intersection lengths, if they exist
		cdef double tlims[2]
		if not Box3D._intersection(tlims, self._lo, self._hi, sst, sdr, sl):
			return { }

		if step <= 0: raise ValueError('Length step must be positive')

		intersections = { }

		# Keep track of accumulated and max length
		cdef double t = max(0, tlims[0])
		cdef double tmax = min(tlims[1], seg.length)

		cdef point lo, hi
		cdef long i, j, k

		# Find the cell that contains the current test point
		self._cellForPoint(&i, &j, &k, axpy(t, seg._direction, seg._start))

		while t < tmax:
			self._boundsForCell(&lo, &hi, i, j, k)
			if not Box3D._intersection(tlims, lo, hi, sst, sdr, sl):
				raise ValueError('Segment fails to intersect cell %s' % ((i,j,k),))

			if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
				# Record a hit inside the grid
				key = i, j, k
				val = tlims[0], tlims[1]
				intersections[key] = val

			# Advance to next cell or run off end of segment
			t = self._advance(&i, &j, &k, tlims[1], step,
					tmax, seg._start, seg._direction)

		return intersections


	cdef void _cellForPoint(self, long *i, long *j, long *k, point p):
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


	cdef double _advance(self, long *i, long *j, long *k, double t,
			double step, double tmax, point start, point direction):
		'''
		For a point with coordinates (start + t * direction), compute a
		new distance tp such that (start + tp * direction) belongs to a
		cell distinct from the provided (i, j, k), the distance

		    tp = t + step * sum(2**i for i in range(q))

		where q is the smallest nonnegative integer that guarantees
		that the point will reside in a distinct cell, and tp < tmax.

		The indices of the cell containing the advanced point will be
		stored in i, j and k and the new value tp will be returned.

		If no tp satisfies all three criteria, the values of i, j and k
		will remain unchanged, while some t >= tmax will be returned.
		'''
		cdef long ni, nj, nk

		while t < tmax:
			# Find the coordinates of the advanced point
			self._cellForPoint(&ni, &nj, &nk, axpy(t, direction, start))

			if ni != i[0] or nj != j[0] or nk != k[0]:
				# Cell is different, terminate advancement
				i[0] = ni
				j[0] = nj
				k[0] = nk
				return t
			# Cell is the same, take another step
			t += step
			# Increase step for next time
			step *= 2

		# No neighbor cell was found, just return too-large t
		return t
