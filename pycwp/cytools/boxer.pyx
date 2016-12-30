'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

import numpy as np
cimport numpy as np

try:
	from scipy.linalg import solve_banded
except ImportError:
	solve_banded = None


from cpython.mem cimport PyMem_Malloc, PyMem_Free

from math import sqrt
from libc.math cimport sqrt, floor, fabs

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

cdef point axpy(double a, point x, point y) nogil:
	'''
	Return a point struct equal to (a * x + y).
	'''
	cdef point r
	r.x = a * x.x + y.x
	r.y = a * x.y + y.y
	r.z = a * x.z + y.z
	return r


cdef point * iaxpy(double a, point x, point *y) nogil:
	'''
	Store, in y, the value of (a * x + y).
	'''
	y.x += a * x.x
	y.y += a * x.y
	y.z += a * x.z
	return y


cdef point scal(double a, point x) nogil:
	'''
	Scale the point x by a.
	'''
	cdef point r
	r.x = x.x * a
	r.y = x.y * a
	r.z = x.z * a
	return r


cdef point * iscal(double a, point *x) nogil:
	'''
	Scale the point x, in place, by a.
	'''
	x.x *= a
	x.y *= a
	x.z *= a
	return x


cdef double ptnrm(point x, bint squared=0) nogil:
	cdef double ns = x.x * x.x + x.y * x.y + x.z * x.z
	if squared: return ns
	else: return sqrt(ns)


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


cdef bint almosteq(double x, double y, double eps=realeps) nogil:
	'''
	Returns True iff the difference between x and y is less than or equal
	to M * eps, where M = max(abs(x), abs(y), 1.0).
	'''
	cdef double mxy = max(fabs(x), fabs(y), 1.0)
	return fabs(x - y) <= eps * mxy


@cython.cdivision(True)
cdef double infdiv(double a, double b, double eps=realeps) nogil:
	'''
	Return a / b with special handling of small values:

	1. If |b| <= eps * |a|, return signed infinity,
	2. Otherwise, if |a| <= eps, return 0.
	'''
	cdef double aa, bb
	aa = fabs(a)
	ab = fabs(b)

	if ab <= eps * aa:
		if (a >= 0) == (b >= 0):
			return infinity
		else: return -infinity
	elif aa <= eps:
		return 0.0

	return a / b


cdef point packpt(double x, double y, double z) nogil:
	cdef point r
	r.x = x
	r.y = y
	r.z = z
	return r


cdef object pt2tup(point a):
	return (a.x, a.y, a.z)


cdef int tup2pt(point *pt, object p) except -1:
	cdef double x, y, z
	x, y, z = p

	if pt != <point *>NULL:
		pt.x = x
		pt.y = y
		pt.z = z

	return 0


cdef class Interpolator3D:
	'''
	A class to manage a function sampled on a 3-D grid and provide routines
	for interpolating the function and the gradient.
	'''
	cdef double[:,:,:] coeffs

	def __init__(self, image):
		'''
		Create an interpolator instance capable of providing scalar
		function values and the gradient of the given image function at
		arbitrary points.
		'''
		# Make sure the image is an array of doubles
		image = np.asarray(image, dtype=np.float64)
		if image.ndim != 3:
			raise ValueError('Argument "image" must be a 3-D array or None')

		nx, ny, nz = image.shape

		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')

		# Allocate the coefficient storage
		cfs = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64, order='C')
		# Pre-fill the coefficients with the image values
		cfs[:nx,:ny,:nz] = image

		# Precompute tridiagonal matrix for largest dimension
		# Elements are unity off the main diagonal, 4 on the main
		ab = np.ones((3, max(nx,ny,nz) - 2), dtype=np.float64)
		ab[1,:] = 4.

		# Collapse y and z to find coefficients in x
		cx = cfs.reshape((nx + 2, -1), order='C')
		self._rowcoeffs(cx, ab)

		# In case cx was copied instead of a view on cfs
		cfs = cx.reshape(cfs.shape, order='C')

		# Now find y coefficients slab-by-slab
		for cx in cfs: self._rowcoeffs(cx, ab)

		# Collapse x and y to find coefficients in z
		cx = cfs.reshape((-1, nz + 2), order='C').T
		self._rowcoeffs(cx, ab)

		# Capture the memory view
		self.coeffs = cx.T.reshape(cfs.shape, order='C')


	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		return (self.coeffs.shape[0] - 2,
				self.coeffs.shape[1] - 2, self.coeffs.shape[2] - 2)


	@staticmethod
	def _rowcoeffs(np.ndarray[np.float64_t, ndim=2] c not None,
			np.ndarray[np.float64_t, ndim=2] ab=None):
		'''
		Compute, along the rows of c, the coefficients for a 1-D
		natural cubic b-spline interpolation. Coefficients will be
		computed in place.

		The shape of c should be (nx + 2, ny), where interpolation is
		in the interval [0, nx - 1]. Coefficients in row i correspond
		to the cubic b-spline centered at sample i for 0 <= i <= nx.
		The coefficients corresponding to the spline centered at sample
		-1 are placed in row (nx + 1).

		If ab is provided, it should be a compressed banded matrix of
		shape (3, nx - 2) whose first and last row are all ones and
		whose middle row is all fours; this is the tridiagonal matrix
		used to determine natural cubic spline coefficients. The number
		of columns in ab may be greater than nx - 2; extra columns will
		be ignored.

		If ab is None, it will be computed on demand.
		'''
		if not solve_banded:
			raise ImportError('Interpolate3D depends on scipy.linalg.solve_banded')

		nx2, ny = c.shape[0], c.shape[1]

		if nx2 < 4: raise ValueError('Array "c" must have at least 4 rows')

		nx = nx2 - 2
		nl = nx - 1
		nll = nx - 2

		if ab is None:
			ab = np.ones((3, nll), dtype=np.float64)
			ab[1,:] = 4.
		elif ab.shape[0] != 3 or ab.shape[1] < nll:
			raise ValueError('Array "ab" must be None or have shape (3, %d)' % (nll,))

		# Fix endpoint coefficients
		c[0,:] /= 6.
		c[nl,:] /= 6.

		# Adjust first and last interior points in RHS
		c[1,:] -= c[0,:]
		c[nll,:] -= c[nl,:]

		# Solve the tridiagonal system
		c[1:nl,:] = solve_banded((1,1), ab[:,:nll], c[1:nl,:])

		# Fill in the out-of-bounds coefficients for natural splines
		# Rightmost edge
		c[nx,:] = 2 * c[nl,:] - c[nll,:]
		# Leftmost edge, wrapped around
		c[nx+1,:] = 2 * c[0,:] - c[1,:]


	@cython.embedsignature(True)
	def getcoeffs(self):
		'''
		Return a copy of the 3-D cubic b-spline coefficients for this
		interpolator.
		'''
		return np.array(self.coeffs)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@staticmethod
	cdef bint _evaluate(double *f, point *grad, double[:,:,:] c, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad, its gradient at a
		point p given cubic b-spline coefficients c. This method
		returns False if the coordinates are out of bounds and True
		otherwise.
		'''
		cdef long nx, ny, nz

		# Coefficient shape is (nx + 2, ny + 2, nz + 2)
		nx = c.shape[0] - 2
		ny = c.shape[1] - 2
		nz = c.shape[2] - 2

		# Find the fractional and integer parts of the coordinates
		cdef:
			# Maximum interval indices are to left of last pixel
			long i = max(0, min(<long>p.x, nx - 2))
			long j = max(0, min(<long>p.y, ny - 2))
			long k = max(0, min(<long>p.z, nz - 2))

			double t = p.x - <double>i
			double u = p.y - <double>j
			double v = p.z - <double>k

		if not (0 <= t <= 1.0 and 0 <= u <= 1.0 and 0 <= v <= 1.0):
			return False

		cdef:
			double tw[4]
			double uw[4]
			double vw[4]
			double dt[4]
			double du[4]
			double dv[4]

			double tt, dtt, uu, duu, vv, cv
			long ii, jj, kk, si, sj, sk

		# Find the interpolating weights for the interval
		Interpolator3D.bswt(&(tw[0]), t)
		Interpolator3D.bswt(&(uw[0]), u)
		Interpolator3D.bswt(&(vw[0]), v)
		# Find the derivative weights
		Interpolator3D.dbswt(&(dt[0]), t)
		Interpolator3D.dbswt(&(du[0]), u)
		Interpolator3D.dbswt(&(dv[0]), v)

		f[0] = 0.0
		grad[0].x = grad[0].y = grad[0].z = 0.0

		ii = 0
		si = i - 1 if i > 0 else nx + 1
		while ii < 4:
			tt = tw[ii]
			dtt = dt[ii]

			jj = 0
			sj = j - 1 if j > 0 else ny + 1
			while jj < 4:
				uu = uw[jj]
				duu = du[jj]

				kk = 0
				sk = k - 1 if k > 0 else nz + 1
				while kk < 4:
					cv = c[si,sj,sk]
					vv = vw[kk]
					f[0] += cv * tt * uu * vv
					grad[0].x += cv * dtt * uu * vv
					grad[0].y += cv * tt * duu * vv
					grad[0].z += cv * tt * uu * dv[kk]

					kk += 1
					sk = k + kk - 1
				jj += 1
				sj = j + jj - 1
			ii += 1
			si = i + ii - 1

		return True


	@cython.embedsignature(True)
	def evaluate(self, double x, double y, double z):
		'''
		Evaluate and return the value of the image and its gradient at
		grid coordinates (x, y, z). If the coordinates are out of
		bounds, a ValueError will be raised.
		'''
		cdef double f
		cdef point g

		if not Interpolator3D._evaluate(&f, &g, self.coeffs, packpt(x, y, z)):
			raise ValueError('Coordinates are out of bounds')
		return f, (g.x, g.y, g.z)


	@staticmethod
	cdef void bswt(double *w, double t) nogil:
		'''
		Compute, in the four-element array w, the weights for cubic
		b-spline interpolation for a normalized fractional coordinate
		t. These coordinates will be invalid unless 0 <= t <= 1.

		If Phi is the standard unnormalized cubic b-spline [i.e.,
		Phi(0) = 4 and Phi(1) = Phi(-1) = 1], the weights have values

		  w[0] = Phi(-t - 1), w[1] = Phi(-t),
		  w[2] = Phi(1 - t), w[3] = Phi(2 - t).

		If a point x falls in the interval [x_i, x_{i+1}) of a
		uniformly spaced grid, then the normalized fractional
		coordinate is

		  t = (x - x_i) / (x_{i+1} - x_i)

		and the value at x of an interpolated function f with spline
		coefficients c[i] is

		  f(x) = c[i-1] * w[0] + c[i] * w[1]
				+ c[i+1] * w[2] + c[i+2]*w[3].
		'''
		cdef double ts
		ts = 1 - t
		w[0] = ts * ts * ts
		w[1] = 4 - 3 * t * t * (2 - t)
		w[2] = 4 - 3 * ts * ts * (1 + t)
		w[3] = t * t * t


	@staticmethod
	cdef void dbswt(double *w, double t) nogil:
		'''
		Compute, in the four-element array w, the weights for the
		derivative of a cubic b-spline interpolation for a normalized
		fractional coordinate t. The weights in w are the derivatives,
		with respect to t, of the weights from bswt(); thus

		  w[0] = Phi'(-t - 1), w[1] = Phi'(-t),
		  w[2] = Phi'(1 - t), w[3] = Phi'(2 - t).

		As with bswt, the weights will be invalid unless 0 <= t <= 1.
		'''
		cdef double ts
		ts = 1 - t
		w[0] = -3 * ts * ts
		w[1] = 3 * t * (3 * t - 4)
		w[2] = 3 * ts * (3 * t + 1)
		w[3] = 3 * t * t


cdef class Segment3D:
	'''
	A representation of a 3-D line segment.

	Initialize as Segment3D(start, end), where start and end are three-
	element sequences providing the Cartesian coordinates of the respective
	start and end points of the segment.
	'''
	# Intrinsic properties of the segment
	cdef point _start, _end

	# Dependent properties (accessible from Python)
	cdef readonly double length

	# Dependent properties (hidden from Python)
	cdef point _midpoint, _direction

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
		self._direction = axpy(-1.0, self._start, self._end)
		self.length = ptnrm(self._direction)

		if not almosteq(self.length, 0.0):
			iscal(1. / self.length, &(self._direction))

		self._midpoint = axpy(1.0, self._start, self._end)
		iscal(0.5, &(self._midpoint))


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

	Initialize as Triangle3D(nodes), where nodes is a sequence of three
	node descriptors, each of which is a three-element sequence which
	provides the Cartesian coordinates of the node.
	'''
	cdef point _nodes[3]
	cdef point _normal
	cdef point *_qr

	cdef readonly double offset
	cdef readonly double area

	def __cinit__(self, *args, **kwargs):
		'''
		Make sure the QR pointer is NULL for proper management.
		'''
		self._qr = <point *>NULL

	def __init__(self, nodes):
		'''
		Initialize a triangle from the sequence of three nodes (each a
		sequence of three floats).
		'''
		cdef point n1, n2, n3
		if len(nodes) != 3:
			raise ValueError('Length of nodes sequence must be 3')

		tup2pt(&n1, nodes[0])
		tup2pt(&n2, nodes[1])
		tup2pt(&n3, nodes[2])

		self.setnodes(n1, n2, n3)


	@cython.cdivision(True)
	cdef int setnodes(self, point n1, point n2, point n3) except -1:
		'''
		Initialize a triangle from the given nodes.
		'''
		self._nodes[0] = n1
		self._nodes[1] = n2
		self._nodes[2] = n3

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
	cdef bint sat_cross_test(point a, point v[3], point hlen) nogil:
		cdef double px, py, pz, r

		px = dot(a, v[0])
		py = dot(a, v[1])
		pz = dot(a, v[2])

		r = hlen.x * fabs(a.x) + hlen.y * fabs(a.y) + hlen.z * fabs(a.z)
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

	Initialize as Box3D(lo, hi), where lo and hi are three-element
	sequences that provide the Cartesian coordinates of the low and high
	corners, respectively.
	'''
	cdef point _lo, _hi, _midpoint, _length, _cell
	cdef unsigned long nx, ny, nz

	def __init__(self, lo, hi):
		'''
		Initialize a 3-D box with extreme corners lo and hi (each a
		3-tuple of floats).
		'''
		cdef point _lo, _hi
		tup2pt(&_lo, lo)
		tup2pt(&_hi, hi)
		self.setbounds(_lo, _hi)


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

		# Compute some dependent properties
		self._midpoint = axpy(1.0, self._lo, self._hi)
		iscal(0.5, &(self._midpoint))

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
	cdef point _cart2cell(self, double x, double y, double z) nogil:
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
	cdef point _cell2cart(self, double i, double j, double k) nogil:
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


	cdef void _boundsForCell(self, point *lo,
			point *hi, long i, long j, long k) nogil:
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


	cdef bint _contains(self, point p) nogil:
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
	cdef bint _intersection(double *t, point l, point h, point s, point d, double sl) nogil:
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
	@cython.cdivision(True)
	@cython.embedsignature(True)
	def descent(self, p, t, Interpolator3D field, unsigned int cycles=1,
			bint report=False, double step=realeps, double rtol=1e-6):
		'''
		Starting at a point p = (x, y, z), where x, y, z are Cartesian
		coordinates within the bounds of this box, perform a
		steepest-descent walk (against the gradient) through the given
		scalar field, encapsulated in an Interpolator3D instance
		capable of evaluating the field, f, and its gradient at
		arbitrary points within the box. The walk ends when one of:

		1. The cell containing the destination point t is reached,
		2. The path runs off the edge of the grid,
		3. The value |grad(f)| <= rtol * |f| in an encountered cell, or
		4. Any cell is encountered more than cycles times.

		The return value is an OrderedDict mapping (i, j, k) cell
		indices to the start and end points of the path through that
		cell. If p is outside the grid, the mapping will be empty.

		If the argument report is True, a second return value, a
		string with the value 'destination', 'boundary', 'stationary',
		or 'cycle', corresponding to each of the above four reasons for
		termination, will indicate the reason for terminating the walk.

		The argument step is interpreted as in Box3D.raymarcher; the
		step is used (in adaptively increasing increments) to ensure
		that the segment in each cell advances to a neighboring cell.

		A ValueError will be raised if the field has the wrong shape.
		'''
		# Make sure the field is double-array compatible
		cdef long nx, ny, nz
		nx, ny, nz = field.shape
		if (nx != self.nx or ny != self.ny or nz != self.nz):
			raise ValueError('Shape of field must be %s' % (self.ncell,))

		# Convert the start and end to points
		cdef point pp, tp
		tup2pt(&pp, p)
		tup2pt(&tp, t)

		# The maximum step through any cell cannot exceed cell diagonal
		cdef double hmax, hx, hy, hz
		hx, hy, hz = self._cell.x, self._cell.y, self._cell.z
		hmax = sqrt(hx * hx + hy * hy + hz * hz)

		cdef point lo, hi, gf, ep, cp
		cdef double tlims[2]
		cdef double mgf, f
		cdef long i, j, k, ti, tj, tk

		# Find cell containing starting point (may be out of bounds)
		self._cellForPoint(&i, &j, &k, pp)

		# Find cell containing ending point (may be out of bounds)
		self._cellForPoint(&ti, &tj, &tk, tp)

		# For convenience
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

			# Grab (or create) a hitlist for this cell
			if key in hits:
				hitlist = hits[key]

				if len(hitlist) >= cycles:
					# Too many hits in this cell
					reason = 'cycle'
					break
			else:
				hitlist = []
				hits[key] = hitlist

			if i == ti and j == tj and k == tk:
				# Boundary cell has been encountered
				# Walk ends at destination point
				hitlist.append((pt2tup(pp), pt2tup(tp)))
				reason = 'destination'
				break

			# Find the boundaries of the current cell
			self._boundsForCell(&lo, &hi, i, j, k)

			# Find the direction of travel in this cell
			cp = self._cart2cell(pp.x, pp.y, pp.z)
			if not Interpolator3D._evaluate(&f, &gf, field.coeffs, cp):
				raise ValueError('Cannot evaluate gradient at %s' % ((cp.x, cp.y, cp.z),))
			mgf = ptnrm(gf)
			if mgf < rtol * fabs(f):
				# Stationary point has been encountered
				# Walk ends, start and end points are the same
				hitlist.append((pt2tup(pp), pt2tup(pp)))
				reason = 'stationary'
				break
			iscal(-1.0 / mgf, &gf)

			# Cast a ray through the cell (negative length is ignored)
			if not Box3D._intersection(tlims, lo, hi, pp, gf, -1):
				raise ValueError('Segment fails to intersect cell %s' % (key,))

			# Record start and end points in this cell
			ep = axpy(tlims[1], gf, pp)
			hitlist.append((pt2tup(pp), pt2tup(ep)))

			# Advance the point and find the next cell
			pp = ep
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


	cdef double _advance(self, long *i, long *j, long *k, double t,
			double step, double tmax, point start, point direction) nogil:
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
