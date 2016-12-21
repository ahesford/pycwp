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

from math import sqrt
from libc.math cimport sqrt, floor

cdef extern from "float.h":
	cdef double DBL_EPSILON

cdef double infinity
with cython.cdivision(True):
	infinity = 1.0 / 0.0

from itertools import izip, product as iproduct

cdef struct point:
	double x, y, z

cdef struct grid:
	long x, y, z


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


cdef double ptnrm(point x, int squared=0):
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


cdef bint almosteq(double x, double y):
	'''
	Returns True iff the difference between x and y is less than or equal
	to M * EPS, where M = max(abs(x), abs(y), 1.0).

	The value EPS is the value of DBL_EPSILON in the float.h.
	'''
	cdef double mxy = max(abs(x), abs(y), 1.0)
	return abs(x - y) <= DBL_EPSILON * mxy


@cython.cdivision(True)
cdef double infdiv(double a, double b):
	'''
	Return a / b with special handling of small values:

	1. If |b| < epsilon * |a| for machine epsilon, return signed infinity,
	2. Otherwise, if |a| < epsilon, return 0.
	'''
	cdef double aa, bb
	aa = abs(a)
	ab = abs(b)

	if ab <= DBL_EPSILON * aa:
		if (a >= 0) == (b >= 0):
			return infinity
		else: return -infinity
	elif aa <= DBL_EPSILON:
		return 0.0

	return a / b


cdef point topoint(double x, double y, double z):
	cdef point r
	r.x = x
	r.y = y
	r.z = z
	return r


cdef object frompoint(point a):
	return (a.x, a.y, a.z)


cdef class Segment3D:
	'''
	A representation of a 3-D line segment.
	'''
	# Intrinsic properties of the segment
	cdef point _start, _end

	# Dependent properties (accessible from Python)
	cdef readonly double length
	cdef readonly int majorAxis

	# Dependent properties (hidden from Python)
	cdef point _midpoint, _direction

	@cython.cdivision(True)
	@cython.embedsignature(True)
	def __cinit__(self, start, end):
		'''
		Initialize a 3-D line segment that starts and ends at the
		indicated points (3-tuples of float coordinates).
		'''
		cdef double x, y, z
		x, y, z = start
		self._start = topoint(x, y, z)
		x, y, z = end
		self._end = topoint(x, y, z)

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
		return frompoint(self._start)

	@property
	def end(self):
		'''The end of the segment, as a 3-tuple of floats'''
		return frompoint(self._end)

	@property
	def midpoint(self):
		'''The midpoint of the segment, as a 3-tuple of floats'''
		return frompoint(self._midpoint)

	@property
	def direction(self):
		'''The direction of the segment, as a 3-tuple of floats'''
		return frompoint(self._direction)


	cdef point _cartesian(self, double t):
		'''C backer for Segment3D.cartesian'''
		return axpy(t, self._direction, self._start)


	def cartesian(self, double t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		return frompoint(self._cartesian(t))


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


	@property
	def bbox(self):
		'''
		A Box3D instance that bounds the segment.
		'''
		cdef double lx, ly, lz, hx, hy, hz
		cdef int i

		# Populate the min-max values of each coordinate
		lx = min(self._start.x, self._end.x)
		hx = max(self._start.x, self._end.x)
		ly = min(self._start.y, self._end.y)
		hy = max(self._start.y, self._end.y)
		lz = min(self._start.z, self._end.z)
		hz = max(self._start.z, self._end.z)

		return Box3D((lx, ly, lz), (hx, hy, hz))


	def __repr__(self):
		start = frompoint(self._start)
		end = frompoint(self._end)
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
	@cython.embedsignature(True)
	def __cinit__(self, nodes):
		'''
		Initialize a triangle from nodes in the given sequence.
		'''
		if len(nodes) != 3:
			raise TypeError('Length of nodes sequence must be 3')

		cdef double x, y, z
		cdef unsigned int i

		for i in range(3):
			x, y, z = nodes[i]
			self._nodes[i] = topoint(x, y, z)

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
		return (frompoint(self._nodes[0]), 
				frompoint(self._nodes[1]), frompoint(self._nodes[2]))

	@property
	def normal(self):
		'''The normal of the triangle, as a 3-tuple of floats'''
		return frompoint(self._normal)

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
		cdef double r1 = (qr[0], qr[1])
		iaxpy(-1.0 / r1, qr[0], &(qr[1]))

		cdef double r2 = ptnrm(qr[1])
		if almosteq(r2, 0.0):
			PyMem_Free(<void *>qr)
			return self._qr
		iscal(1.0 / r2, &(qr[1]))

		qr[2] = topoint(r0, r1, r2)

		self._qr = qr
		return qr


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
		cdef double x, y, z
		x, y, z = p
		cdef point d = axpy(-1.0, self._nodes[2], topoint(x, y, z))

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

		vx = x * n[0].x + y * n[1].x + z * n.x
		vy = x * n[0].y + y * n[1].y + z * n.y
		vz = x * n[0].z + y * n[1].z + z * n.z
		return vx, vy, vz


	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in this triangle.
		'''
		cdef double x, y, z

		try: x, y, z = self.barycentric(p)
		except TypeError: return False

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


	cpdef bint overlaps(self, Box3D b):
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
	def intersection(self, Segment3D seg):
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
		cpdef point *qr = self.qrbary()
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
		v = y.z / r5
		u = (y.y - r4 * v) / r2
		t = (y.x - r3 * v - r1 * u) / r0

		if ((0 <= v <= 1 and 0 <= u <= 1 and 0 <= t <= 1) and 0 <= u + t <= 1):
			# v is the fraction of segment length
			# t and u are normal barycentric coordinates in triangle
			return v * seg._length, t, u, 1 - t - u
		else:
			# Intersection is not in segment or triangle
			return None


cdef class Box3D:
	'''
	A representation of an axis-aligned 3-D bounding box.
	'''
	cdef point _lo, _hi, _midpoint, _length, _cell
	cdef grid _ncell

	def __cinit__(self, lo, hi):
		'''
		Initialize a 3-D box with extreme corners lo and hi.
		'''
		cdef double x, y, z
		x, y, z = lo
		self._lo = topoint(x, y, z)
		x, y, z = hi
		self._hi = topoint(x, y, z)

		if (self._lo.x > self._hi.x
				or self._lo.y > self._hi.y
				or self._lo.z > self._hi.z):
			raise ValueError('Coordinates of hi must be no less than those of lo')

		# Compute some dependent properties
		self._midpoint = axpy(1.0, self._lo, self._hi)
		iscal(0.5, &(self._midpoint))

		self._length = axpy(-1.0, self._lo, self._hi)

		self._ncell.x = self._ncell.y = self._ncell.z = 1
		self._cell = self._length


	@property
	def lo(self):
		'''The lower corner of the box'''
		return frompoint(self._lo)

	@property
	def hi(self):
		'''The upper corner of the box'''
		return frompoint(self._hi)

	@property
	def midpoint(self):
		'''The barycenter of the box'''
		return frompoint(self._midpoint)

	@property
	def length(self):
		'''The length of the box edges'''
		return frompoint(self._length)

	@property
	def cell(self):
		'''The length of each cell into which the box is subdivided'''
		return frompoint(self._cell)

	@property
	def ncell(self):
		'''The number of cells (nx, ny, nz) that subdivide the box'''
		return self._ncell.x, self._ncell.y, self._ncell.z

	@ncell.setter
	def ncell(self, c):
		cdef long x, y, z
		x, y, z = c

		if x < 1 or y < 1 or z < 1:
			raise ValueError('Grid dimensions must all be positive')

		self._ncell.x = <long>x
		self._ncell.y = <long>y
		self._ncell.z = <long>z

		with cython.cdivision(True):
			self._cell.x = self._length.x / <double>self._ncell.x
			self._cell.y = self._length.y / <double>self._ncell.y
			self._cell.z = self._length.z / <double>self._ncell.z

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

	def cart2cell(self, double x, double y, double z):
		'''
		Convert the 3-D Cartesian coordinates (x, y, z) into grid
		coordinates defined by the box bounds and ncell property. The
		coordinates will always be real-valued and may have a fraction
		part to indicate relative positions within the cell.
		'''
		return frompoint(self._cart2cell(x, y, z))


	@cython.cdivision
	cdef point _cell2cart(self, double i, double j, double k):
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
		return frompoint(self._cart2cell(i, j, k))

	def getCell(self, c):
		'''
		Return a Box3D representing the cell that contains 3-D index c
		based on the grid defined by the ncell property. Any fractional
		part of the coordinates will be truncated.
		'''
		cdef double i, j, k
		i, j, k = c

		cdef double li, lj, lk
		li = floor(i)
		lj = floor(j)
		lk = floor(k)

		lo = self.cell2cart(li, lj, lk)
		hi = self.cell2cart(li + 1, lj + 1, lk + 1)

		return Box3D(lo, hi)

	def allIndices(self):
		'''
		Return a generator that produces every 3-D cell index within
		the grid defined by the ncell property in row-major order.
		'''
		return iproduct(*(xrange(n) for n in self.ncell))

	def allCells(self, enum=False):
		'''
		Return a generator that produces every cell in the grid defined
		by the ncell property. Generation is done in the same order as
		self.allIndices().

		If enum is True, return a tuple (idx, box), where idx is the
		three-dimensional index of the cell.
		'''
		for idx in self.allIndices():
			box = self.getCell(idx)
			if not enum: yield box
			else: yield (idx, box)

	cpdef bint overlaps(self, Box3D b):
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


	cpdef bint contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in the box.
		'''
		cdef double x, y, z
		x, y, z = p
		return ((x >= self._lo.x) and (x <= self._hi.x)
				and (y >= self._lo.y) and (y <= self._hi.y)
				and (z >= self._lo.z) and (z <= self._hi.z))


	cdef bint _intersection(self, double *t, Segment3D segment):
		'''C backer for intersection'''
		cdef double sx, sy, sz, dx, dy, dz, seglen

		sx, sy, sz = segment._start.x, segment._start.y, segment._start.z
		dx, dy, dz = segment._direction.x, segment._direction.y, segment._direction.z
		seglen = segment.length

		cdef double lx, ly, lz, hx, hy, hz

		lx, ly, lz = self._lo.x, self._lo.y, self._lo.z
		hx, hy, hz = self._hi.x, self._hi.y, self._hi.z

		cdef double tmin, tmax, ty1, ty2, tz1, tz2

		# Check, in turn, intersections with the x, y and z slabs
		tmin = infdiv(lx - sx, dx)
		tmax = infdiv(hx - sx, dx)
		if tmax < tmin: tmin, tmax = tmax, tmin
		# Check the y-slab
		ty1 = infdiv(ly - sy, dy)
		ty2 = infdiv(hy - sy, dy)
		if ty2 < ty1: ty1, ty2 = ty2, ty1
		if ty2 < tmax: tmax = ty2
		if ty1 > tmin: tmin = ty1
		# Check the z-slab
		tz1 = infdiv(lz - sz, dz)
		tz2 = infdiv(hz - sz, dz)
		if tz2 < tz1: tz1, tz2 = tz2, tz1
		if tz2 < tmax: tmax = tz2
		if tz1 > tmin: tmin = tz1

		if tmax < max(0, tmin) or tmin > seglen: 
			return False

		t[0] = tmin
		t[1] = tmax
		return True

	def intersection(self, Segment3D segment):
		'''
		Returns the lengths tmin and tmax along the given line segment
		(like Segment3D) at which the segment enters and exits the box.
		If the box does not intersect the segment, returns None.

		If the segment starts within the box, tmin will be negative. If
		the segment ends within the box, tmax will exceed the segment
		length.
		'''
		cdef double tlims[2]
		if self._intersection(tlims, segment):
			return tlims[0], tlims[1]
		else: return None


#	def raymarcher(self, segment):
#		'''
#		Marches along the given 3-D line segment (like Segment3D) to
#		identify all cells in the grid (defined by the ncell property)
#		that intersect the segment. Returns a list of tuples of the
#		form (i, j, k, tmin, tmax), where (i, j, k) is a grid index of
#		a cell, and (tmin, tmax) are the lengths along the segment of
#		the entry and exit points, respectively, through the cell.
#
#		If the segment begins or ends in a cell, tmin or tmax may fall
#		outside the range [0, segment.length].
#		'''
#		# March along the major axis
#		axis = segment.majorAxis
#		# Record the transverse axes as tx, ty
#		tx, ty = (axis + 1) % 3, (axis + 2) % 3
#
#		# Pull grid parameters for efficiency
#		ncell = self.ncell
#		lo = self.lo[axis]
#		dslab = self.cell[axis]
#
#		nax, ntx, nty = ncell[axis], ncell[tx], ncell[ty]
#
#		# Try to grab the intersection lengths, if they exist
#		try: tmin, tmax = self.intersection(segment)
#		except TypeError: return []
#
#		# Find the intersection points
#		pmin = segment.cartesian(max(0., tmin))
#		pmax = segment.cartesian(min(segment.length, tmax))
#
#		# Find the minimum and maximum slabs; ensure proper ordering
#		slabmin = int((pmin[axis] - lo) / dslab)
#		slabmax = int((pmax[axis] - lo) / dslab)
#		if slabmin > slabmax: slabmin, slabmax = slabmax, slabmin
#		# Ensure the slabs don't run off the end
#		if slabmin < 0: slabmin = 0
#		if slabmax >= nax: slabmax = nax - 1
#
#		# Compute the starting position of the first slab
#		esc = lo + dslab * slabmin
#
#		# Build a list of slab indices and segment entry points
#		# Add an extra slab to capture exit from the final slab
#		slabs = []
#		for slab in range(slabmin, slabmax + 2):
#			# Figure the segment length at the edge of the slab
#			t = segment.lengthToAxisPlane(esc, axis)
#			# Shift the edge coordinate to the next slab
#			esc += dslab
#			# Find the coordinates of entry cell
#			px, py, pz = segment.cartesian(t)
#			idx = self.cart2cell(px, py, pz, False)
#			# Add the cell coordinates to the list
#			# Override axial index to correct rounding errors
#			slabs.append(idx[:axis] + (slab,) + idx[axis+1:])
#
#		intersections = []
#
#		# Now enumerate all intersecting cells
#		for entry, exit in izip(slabs, slabs[1:]):
#			ranges = [(e,) for e in entry]
#			# Check all cells in range spanned by transverse indices
#			enx, eny = entry[tx], entry[ty]
#			exx, exy = exit[tx], exit[ty]
#			if enx != exx:
#				mn, mx = (enx, exx) if (enx < exx) else (exx, enx)
#				ranges[tx] = xrange(max(0, mn), min(ntx, mx + 1))
#			if eny != exy:
#				mn, mx = (eny, exy) if (eny < exy) else (exy, eny)
#				ranges[ty] = xrange(max(0, mn), min(nty, mx + 1))
#
#			for i in ranges[0]:
#				for j in ranges[1]:
#					for k in ranges[2]:
#						t = self.getCell((i, j, k)).intersection(segment)
#						if t is not None:
#							intersections.append((i, j, k, t[0], t[1]))
#		return intersections
