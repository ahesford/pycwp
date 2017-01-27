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


cdef double ptdst(point x, point y) nogil:
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


# Forward declaration
cdef class Interpolator3D


cdef class LagrangeInterpolator3D(Interpolator3D):
	'''
	An Interpolator3D that implements piecewise Lagrange interpolation of
	third degree along each axis of a 3-D function.
	'''
	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image):
		'''
		Construct a piecewise Lagrange interpolator of degree 3 for the
		given 3-D floating-point image.
		'''
		cdef double [:,:,:] img = np.asarray(image, dtype=np.float64)
		cdef unsigned long nx, ny, nz
		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]

		if nx < 4 or ny < 4 or nz < 4:
			raise ValueError('Size of image must be at least (4, 4, 4)')

		self.ncx, self.ncy, self.ncz = nx, ny, nz

		# Allocate coefficients and capture a view
		self.coeffs = <double *>PyMem_Malloc(nx * ny * nz * sizeof(double))
		if self.coeffs == <double *>NULL:
			raise MemoryError('Unable to allocate storage for coefficients')

		# Capture a view on the coefficients
		cdef double[:,:,:] coeffs
		cdef unsigned long ix, iy, iz

		try:
			coeffs = <double[:nx,:ny,:nz:1]>self.coeffs

			# Just copy the interpolation coefficients
			for ix in range(nx):
				for iy in range(ny):
					for iz in range(nz):
						coeffs[ix,iy,iz] = img[ix,iy,iz]
		except Exception:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL


	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		return (self.ncx, self.ncy, self.ncz)


	@cython.cdivision(True)
	@staticmethod
	cdef void lgwts(double *l, double x) nogil:
		'''
		Compute in l[0]...l[3] the values of the third-degree Lagrange
		polynomials 0...3 for an argument 0 <= x <= 3. The bounds of x
		are not checked.
		'''
		cdef:
			double x1 = x - 1
			double x2 = x - 2
			double x3 = x - 3
			double xx3 = 0.5 * x * x3
			double x12 = x1 * x2 / 6.0

		l[0] = -x12 * x3
		l[1] = xx3 * x2
		l[2] = -xx3 * x1
		l[3] = x * x12


	@cython.cdivision(True)
	@staticmethod
	cdef void dlgwts(double *l, double x) nogil:
		'''
		Compute in l[0]...l[3] the values of the derivatives of the
		third-degree Lagrange polynomials 0...3 for 0 <= x <= 3. The
		bounds of x are not checked.
		'''
		cdef:
			double x1 = x - 1
			double x2 = x - 2
			double x3 = x - 3

			double x23 = x2 * x3
			double x12 = x1 * x2
			double x13 = x1 * x3

			double xx3 = x * x3
			double xx2 = x * x2
			double xx1 = x * x1


		l[0] = -(x23 + x13 + x12) / 6.0
		l[1] = 0.5 * (x23 + xx3 + xx2)
		l[2] = -0.5 * (x13 + xx3 + xx1)
		l[3] = (x12 + xx2 + xx1) / 6.0


	@staticmethod
	cdef void adjint(double *t, long *i, unsigned long n) nogil:
		'''
		Adjust the fractional and interval coordiantes t and i,
		respectively, according to the rule:

			0 < i < n - 2: i -= 1, t += 1.0
			i == n - 2: i -= 2, t += 2.0
			Otherwise: no change
		'''
		if 0 < i[0] < n - 2:
			i[0] -= 1
			t[0] += 1.0
		elif i[0] == n - 2:
			i[0] -= 2
			t[0] += 2


	@cython.wraparound(False)
	@cython.boundscheck(False)
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad if grad is not NULL,
		its gradient at a point p.

		If the coordinates are out of bounds, evaluation will be
		attempted by Interpolator3D._evaluate().

		The method will always return True if evaluation was done
		locally, or the return value of Interpolator3D._evaluate() if
		the evaluation was delegated.
		'''
		cdef long nx, ny, nz
		cdef bint dograd = (grad != <point *>NULL)

		if self.coeffs == <double *>NULL:
			return Interpolator3D._evaluate(self, f, grad, p)

		# Hermite coefficient shape is (nx, ny, nz, self.nval)
		nx, ny, nz = self.ncx, self.ncy, self.ncz

		# Find fractional and integer parts of the coordinates
		cdef long i, j, k
		cdef point t

		if not Interpolator3D.crdfrac(&t, &i, &j, &k, p, nx, ny, nz):
			return Interpolator3D._evaluate(self, f, grad, p)

		# Adjust intervals at endpoints
		LagrangeInterpolator3D.adjint(&(t.x), &i, nx)
		LagrangeInterpolator3D.adjint(&(t.y), &j, ny)
		LagrangeInterpolator3D.adjint(&(t.z), &k, nz)

		# Check for sanity (this should be assured)
		if not (0 <= i <= nx - 4 and 0 <= j <= ny - 4 and 0 <= k <= nz - 4):
			return Interpolator3D._evaluate(self, f, grad, p)

		cdef:
			double lx[4]
			double ly[4]
			double lz[4]
			double dlx[4]
			double dly[4]
			double dlz[4]

			unsigned long ii, jj, kk, soi, soij, soijk
			double cv, lxv, lyv, lzv, dlxv, dlyv, dlzv

		# Initialize the function value
		f[0] = 0.0

		# Find the interpolator values in the interval
		LagrangeInterpolator3D.lgwts(lx, t.x)
		LagrangeInterpolator3D.lgwts(ly, t.y)
		LagrangeInterpolator3D.lgwts(lz, t.z)

		if dograd:
			# Initialize the gradient
			grad.x = grad.y = grad.z = 0.0

			# Find derivative values
			LagrangeInterpolator3D.dlgwts(dlx, t.x)
			LagrangeInterpolator3D.dlgwts(dly, t.y)
			LagrangeInterpolator3D.dlgwts(dlz, t.z)

		for ii in range(4):
			soi = self.ncy * (i + ii)
			lxv = lx[ii]
			dlxv = dlx[ii]

			for jj in range(4):
				soij = self.ncz * (j + jj + soi)
				lyv = ly[jj]
				dlyv = dly[jj]

				for kk in range(4):
					soijk = k + kk + soij
					lzv = lz[kk]
					dlzv = dlz[kk]

					# Add contribution from function value
					cv = self.coeffs[soijk]
					f[0] += cv * lxv * lyv * lzv
					if dograd:
						# Add contribution to gradient
						grad.x += cv * dlxv * lyv * lzv
						grad.y += cv * lxv * dlyv * lzv
						grad.z += cv * lxv * lyv * dlzv
		return True


cdef class HermiteInterpolator3D(Interpolator3D):
	'''
	An Interpolator3D that implements local interpolation with cubic
	Hermite splines. A degree of monotonicity is enforced by forcing first
	derivatives at each sample to match the smaller of the left and right
	differences (if they are of the same sign; zero if they are of
	different sizes). All cross derivatives (xy, xz, yz, xyz) are forced to
	zero.

	*** NOTE: These constraints do not guarantee global monotonicity. ***
	'''
	# THe number of values per sample (f, fx, fy, fz, ...)
	cdef unsigned long nval

	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image):
		'''
		Construct a constrained Hermite cubic interpolator for the
		given 3-D floating-point image.
		'''
		cdef double [:,:,:] img = np.asarray(image, dtype=np.float64)
		cdef unsigned long nx, ny, nz
		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]

		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')

		self.ncx, self.ncy, self.ncz = nx, ny, nz
		# Each sample gets four values: function and first-order derivatives
		self.nval = 4

		# Allocate coefficients
		self.coeffs = <double *>PyMem_Malloc(nx * ny * nz * self.nval * sizeof(double))
		if self.coeffs == <double *>NULL:
			raise MemoryError('Unable to allocate storage for coefficients')

		# Capture a convenient view on the coefficient storage
		cdef double[:,:,:,:] coeffs

		try:
			# Try to populate the coefficient matrix
			coeffs = <double[:nx,:ny,:nz,:self.nval:1]>self.coeffs
			HermiteInterpolator3D.img2coeffs(coeffs, img)
		except Exception:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL


	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		return (self.ncx, self.ncy, self.ncz)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad if grad is not NULL,
		its gradient at a point p.

		If the coordinates are out of bounds, evaluation will be
		attempted by Interpolator3D._evaluate().

		The method will always return True if evaluation was done
		locally, or the return value of Interpolator3D._evaluate() if
		the evaluation was delegated.
		'''
		cdef long nx, ny, nz
		cdef bint dograd = (grad != <point *>NULL)

		if self.coeffs == <double *>NULL:
			return Interpolator3D._evaluate(self, f, grad, p)

		# Hermite coefficient shape is (nx, ny, nz, self.nval)
		nx, ny, nz = self.ncx, self.ncy, self.ncz

		# Find fractional and integer parts of the coordinates
		cdef long i, j, k
		cdef point t

		if not Interpolator3D.crdfrac(&t, &i, &j, &k, p, nx, ny, nz):
			return Interpolator3D._evaluate(self, f, grad, p)

		cdef:
			double hx[4]
			double hy[4]
			double hz[4]
			double dhx[4]
			double dhy[4]
			double dhz[4]

			unsigned long ii, jj, kk, dd, soi, soij, soijk
			double cv, hxv, dhxv, hyv, dhyv, hzv, dhzv
			double hxvp, dhxvp, hyvp, dhyvp, hzvp, dhzvp

		# Initialize the function value
		f[0] = 0.0

		# Find the interpolator values in the interval
		HermiteInterpolator3D.hermspl(hx, t.x)
		HermiteInterpolator3D.hermspl(hy, t.y)
		HermiteInterpolator3D.hermspl(hz, t.z)

		if dograd:
			# Initialize the gradient
			grad.x = grad.y = grad.z = 0.0

			# Find derivative values
			HermiteInterpolator3D.hermdiff(dhx, t.x)
			HermiteInterpolator3D.hermdiff(dhy, t.y)
			HermiteInterpolator3D.hermdiff(dhz, t.z)

		for ii in range(2):
			soi = self.ncy * (i + ii)
			hxv = hx[ii]
			dhxv = hx[ii + 2]
			hxvp = dhx[ii]
			dhxvp = dhx[ii + 2]

			for jj in range(2):
				soij = self.ncz * (j + jj + soi)
				hyv = hy[jj]
				dhyv = hy[jj + 2]
				hyvp = dhy[jj]
				dhyvp = dhy[jj + 2]

				for kk in range(2):
					soijk = self.nval * (k + kk + soij)
					hzv = hz[kk]
					dhzv = hz[kk + 2]
					hzvp = dhz[kk]
					dhzvp = dhz[kk + 2]

					# Add contribution from function value
					cv = self.coeffs[soijk]
					f[0] += cv * hxv * hyv * hzv
					if dograd:
						# Add contribution to gradient
						grad.x += cv * hxvp * hyv * hzv
						grad.y += cv * hxv * hyvp * hzv
						grad.z += cv * hxv * hyv * hzvp

					# Add contribution from x derivative
					cv = self.coeffs[soijk + 1]
					f[0] += cv * dhxv * hyv * hzv
					if dograd:
						# Add contribution to gradient
						grad.x += cv * dhxvp * hyv * hzv
						grad.y += cv * dhxv * hyvp * hzv
						grad.z += cv * dhxv * hyv * hzvp

					# Add contribution from y derivative
					cv = self.coeffs[soijk + 2]
					f[0] += cv * hxv * dhyv * hzv
					if dograd:
						# Add contribution to gradient
						grad.x += cv * hxvp * dhyv * hzv
						grad.y += cv * hxv * dhyvp * hzv
						grad.z += cv * hxv * dhyv * hzvp

					# Add contribution from z derivative
					cv = self.coeffs[soijk + 3]
					f[0] += cv * hxv * hyv * dhzv
					if dograd:
						# Add contribution to gradient
						grad.x += cv * hxvp * hyv * dhzv
						grad.y += cv * hxv * hyvp * dhzv
						grad.z += cv * hxv * hyv * dhzvp

		return True


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@staticmethod
	cdef int img2coeffs(double[:,:,:,:] coeffs, double[:,:,:] img) except -1:
		'''
		Compute, in coeffs, the cubic Hermite spline coefficients for
		the image img. If img has shape (nx, ny, nz), the coefficient
		array should have shape at least (nx, ny, nz, 4).

		The coefficients for sample (i, j, k) are:

			coeffs[i,j,k,0] = img[i,j,k]
			coeffs[i,j,k,1] = d(img[i,j,k]) / dx
			coeffs[i,j,k,1] = d(img[i,j,k]) / dy
			coeffs[i,j,k,2] = d(img[i,j,k]) / dz

		The derivative is approximated as the smaller of the left- and
		right-side finite differences if both have the same sign; if
		the signs differ, the derivative is zero.

		Unspecified, independent cross derivatives (dxdy, dxdz, dydz,
		and dxdydz) are implicitly zero.
		'''
		cdef unsigned long nx, ny, nz, ix, iy, iz
		cdef unsigned long nval = 4

		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]
		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')
		elif (coeffs.shape[0] < nx or coeffs.shape[1] < ny or
				coeffs.shape[2] < nz or coeffs.shape[3] < nval):
			raise ValueError('Shape of coeffs must be at least %s' % ((nx, ny, nz, nval),))

		for ix in range(nx):
			for iy in range(ny):
				# Populate image and find slopes along z
				for iz in range(nz):
					coeffs[ix,iy,iz,0] = img[ix,iy,iz]
				HermiteInterpolator3D.mhdiffs(coeffs[ix,iy,:,3], img[ix,iy,:])

			for iz in range(nz):
				# Find slopes along y
				HermiteInterpolator3D.mhdiffs(coeffs[ix,:,iz,2], img[ix,:,iz])

		# Find slopes along x
		for iy in range(ny):
			for iz in range(nz):
				HermiteInterpolator3D.mhdiffs(coeffs[:,iy,iz,1], img[:,iy,iz])

		return 0


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	@staticmethod
	cdef int mhdiffs(double[:] m, double[:] v) except -1:
		'''
		Given a set of function values v with length N, store in m the
		central differences at each of the interior points, and the
		one-sided differences at the end points.

		The differences are adjusted according to the Fritsch-Carlson
		(1980) method to ensure monotonicity of the cubic Hermite
		spline interpolant using these values and derivatives.
		'''
		cdef unsigned long n, i
		cdef double lh, rh, mm, beta, alpha

		n = v.shape[0]
		if n < 2:
			raise ValueError('Array v must have at least 2 elements')
		elif m.shape[0] < n:
			raise ValueError('Array m must hold at least %d elements' % (n,))

		# Assign the first and last differences
		m[0] = v[1] - v[0]
		m[n - 1] = v[n - 1] - v[n - 2]

		# Initialize the left difference
		lh = m[0]

		# Compute the slopes subject to Fritsch-Carlson criterion
		for i in range(1, n - 1):
			# Compute the right difference
			rh = v[i + 1] - v[i]

			if (lh >= 0) != (rh >= 0):
				# Signs are different; slope is 0
				m[i] = 0.0
			elif almosteq(lh, 0.0) or almosteq(rh, 0.0):
				# Data doesn't change; slope must be flat
				m[i] = 0.0
			else:
				# Use central differencing
				mm = 0.5 * (lh + rh)
				# Limit slope
				beta = max(0, min(mm / lh, 3))
				alpha = max(0, min(mm / rh, 3))
				if beta > alpha: m[i] = alpha * rh
				else: m[i] = beta * lh

			# Move to next point
			lh = rh

		return 0


	@staticmethod
	cdef void hermspl(double *w, double t) nogil:
		'''
		Compute, in the four-elemenet array w, the values of the four
		Hermite cubic polynomials:

			w[0] = (1 + 2 * t) * (1 - t)**2
			w[1] = t**2 * (3 - 2 * t)
			w[2] = t * (1 - t)**2
			w[3] = t**2 * (t - 1)
		'''
		cdef:
			double tsq = t * t
			double mt1 = 1 - t
			double mt1sq = mt1 * mt1
			double t2 = 2 * t

		w[0] = (1 + t2) * mt1sq
		w[1] = tsq * (3 - t2)
		w[2] = t * mt1sq
		w[3] = tsq * mt1

	@staticmethod
	cdef void hermdiff(double *w, double t) nogil:
		'''
		Compute, in the four-element array w, the values of the
		derivatives of the four Hermite cubic polynomials:

			w[0] = 6 * t * (t - 1)
			w[1] = -w[0]
			w[2] = t * (3 * t - 4) + 1
			w[3] = t * (3 * t - 2)
		'''
		cdef double t3 = 3 * t
		w[0] = 6 * t * (t - 1)
		w[1] = -w[0]
		w[2] = t * (t3 - 4) + 1
		w[3] = t * (t3 - 2)


cdef class CubicInterpolator3D(Interpolator3D):
	'''
	An Interpolator3D that implements tricubic interpolation with
	b-splines.
	'''
	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image):
		'''
		Construct a tricubic interpolator for the given 3-D
		floating-point image.
		'''
		cdef double[:,:,:] img = np.asarray(image, dtype=np.float64)
		cdef unsigned long nx, ny, nz, nx2, ny2, nz2
		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]

		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')

		# Need two more coefficients than samples per dimension
		nx2, ny2, nz2 = nx + 2, ny + 2, nz + 2
		self.ncx, self.ncy, self.ncz = nx2, ny2, nz2

		# Allocate coefficients
		self.coeffs = <double *>PyMem_Malloc(nx2 * ny2 * nz2 * sizeof(double))
		if self.coeffs == <double *>NULL:
			raise MemoryError('Unable to allocate storage for coefficients')

		# Capture a convenient view on the coefficient storage
		cdef double[:,:,:] coeffs

		try:
			# Try to populate the coefficient matrix
			coeffs = <double[:nx2,:ny2,:nz2:1]>self.coeffs
			CubicInterpolator3D.img2coeffs(coeffs, img)
		except Exception:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@staticmethod
	cdef int img2coeffs(double[:,:,:] coeffs, double[:,:,:] img) except -1:
		'''
		Attempt to populate the compute, in coeffs, the cubic b-spline
		coefficients of the image img. If img has shape (nx, ny, nz),
		coeffs must have shape at least (nx + 2, ny + 2, nz + 2).
		'''
		cdef unsigned long nx, ny, nz, nx2, ny2, nz2
		cdef double[:] work
		cdef unsigned long ix, iy, iz

		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]
		nx2, ny2, nz2 = nx + 2, ny + 2, nz + 2

		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')

		if (coeffs.shape[0] < nx2 or
				coeffs.shape[1] < ny2 or coeffs.shape[2] < nz2):
			raise ValueError('Shape of coeffs must be at least %s' % ((nx2, ny2, nz2),))

		# Allocate a work array for coefficient evaluation
		work = np.empty((max(nx, ny, nz) - 2,), dtype=np.float64, order='C')

		for ix in range(nx):
			for iy in range(ny):
				# Populate the image and transform along z
				for iz in range(nz):
					coeffs[ix,iy,iz] = img[ix,iy,iz]
				CubicInterpolator3D.nscoeffs(coeffs[ix,iy,:], work)

			# Transform along y, including out-of-bounds z coefficients
			for iz in range(nz2):
				CubicInterpolator3D.nscoeffs(coeffs[ix,:,iz], work)

		# Transform along x axis, including out-of-bounds y and z coefficients
		for iy in range(ny2):
			for iz in range(nz2):
				CubicInterpolator3D.nscoeffs(coeffs[:,iy,iz], work)

		return 0


	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		return (self.ncx - 2, self.ncy - 2, self.ncz - 2)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	@staticmethod
	cdef int nscoeffs(double[:] c, double[:] work) except -1:
		'''
		Compute, in place, the coefficients for a 1-D natural cubic
		b-spline interpolation of a function sampled in c.

		The length of c is (nx + 2), where c contains samples of the
		function on the interval [0, nx - 1] on input. On output, the
		coefficient at sample i corresponds to the cubic b-spline
		centered on the same sample for 0 <= i <= nx. The coefficient
		for the spline centered at sample -1 wraps to index (nx + 1).

		The array work should have a length of at least (nx - 2). The
		contents of this array will be destroyed. A ValueError will be
		raised if work does not have sufficient capacity.
		'''
		cdef unsigned long nx2, nx, nl, nll, i
		cdef double den

		nx2 = c.shape[0]
		nx = nx2 - 2
		nl, nll = nx - 1, nx - 2

		if nx2 < 2:
			raise ValueError('Coefficient length must be at least 2')
		elif work.shape[0] < nll:
			raise ValueError('Array "work" must have length >= %d' % (nll,))

		# Fix the endpoint coefficients
		c[0] /= 6.
		c[nl] /= 6.

		# Adjust first and last interior points in RHS
		c[1] -= c[0]
		c[nll] -= c[nl]

		# Solve the tridiagonal system using the Thomas algorithm

		# First pass: eliminate lower diagonals
		# REMEMBER: shift c indices up by one to skip fixed first sample

		# Scale first super-diagonal element [0] and RHS sample [1]
		work[0] = 1. / 4.
		c[1] /= 4.

		# Eliminate lower diagonals from remaining equations
		for i in range(1, nll):
			den = (4. - work[i - 1])
			# Adjust super-diagonal coefficients
			work[i] = 1. / den
			# Adjust RHS
			c[i+1] -= c[i]
			c[i+1] /= den

		# Second pass: back substitution
		# Output sample c[nll] is already correct
		for i in range(nll - 1, 0, -1):
			c[i] -= work[i - 1] * c[i + 1]

		# Rightward out-of-bounds coefficient
		c[nx] = 2 * c[nl] - c[nll]
		# Leftward out-of-bounds coefficient (wrapped around)
		c[nx+1] = 2 * c[0] - c[1]

		return 0


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


	@cython.wraparound(False)
	@cython.boundscheck(False)
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad, its gradient at a
		point p. If grad is NULL, the gradient is not evaluated.

		If the coordinates are out of bounds, evaluation will be
		attempted by Interpolator3D._evaluate().

		The method will always return True if evaluation was done
		locally, or the return value of Interpolator3D._evaluate() if
		the evaluation was delegated.
		'''
		cdef long nx, ny, nz
		cdef bint dograd = (grad != <point *>NULL)

		if self.coeffs == <double *>NULL:
			return Interpolator3D._evaluate(self, f, grad, p)

		# Cubic coefficient shape is (nx + 2, ny + 2, nz + 2)
		nx, ny, nz = self.ncx - 2, self.ncy - 2, self.ncz - 2

		# Find the fractional and integer parts of the coordinates
		cdef long i, j, k
		cdef point t

		if not Interpolator3D.crdfrac(&t, &i, &j, &k, p, nx, ny, nz):
			return Interpolator3D._evaluate(self, f, grad, p)

		cdef:
			double tw[4]
			double uw[4]
			double vw[4]
			double dt[4]
			double du[4]
			double dv[4]

			double tt, dtt, uu, duu, vv, cv
			long ii, jj, kk, si, sj, sk, soi, soij

		# Initialize the function value
		f[0] = 0.0

		# Find the interpolating weights for the interval
		CubicInterpolator3D.bswt(&(tw[0]), t.x)
		CubicInterpolator3D.bswt(&(uw[0]), t.y)
		CubicInterpolator3D.bswt(&(vw[0]), t.z)

		if dograd:
			# Initialize gradient
			grad.x = grad.y = grad.z = 0.0

			# Find the derivative weights
			CubicInterpolator3D.dbswt(&(dt[0]), t.x)
			CubicInterpolator3D.dbswt(&(du[0]), t.y)
			CubicInterpolator3D.dbswt(&(dv[0]), t.z)

		ii = 0
		si = i - 1 if i > 0 else nx + 1
		while ii < 4:
			tt = tw[ii]
			dtt = dt[ii]

			# Compute the coefficient offset in x
			soi = self.ncy * si

			jj = 0
			sj = j - 1 if j > 0 else ny + 1
			while jj < 4:
				uu = uw[jj]
				duu = du[jj]

				# Compute the coefficient offset in x and y
				soij = self.ncz * (sj + soi)

				kk = 0
				sk = k - 1 if k > 0 else nz + 1
				while kk < 4:
					cv = self.coeffs[soij + sk]
					vv = vw[kk]
					f[0] += cv * tt * uu * vv
					if dograd:
						grad.x += cv * dtt * uu * vv
						grad.y += cv * tt * duu * vv
						grad.z += cv * tt * uu * dv[kk]

					kk += 1
					sk = k + kk - 1
				jj += 1
				sj = j + jj - 1
			ii += 1
			si = i + ii - 1

		return True


cdef class LinearInterpolator3D(Interpolator3D):
	'''
	An Interpolator3D that implements trilinear interpolation.
	'''
	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image):
		'''
		Construct a trilinear interpolator for the given 3-D
		floating-point image.
		'''
		cdef double[:,:,:] img = np.asarray(image, dtype=np.float64)
		cdef unsigned long nx, ny, nz
		nx, ny, nz = img.shape[0], img.shape[1], img.shape[2]

		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Size of image must be at least (2, 2, 2)')

		self.ncx, self.ncy, self.ncz = nx, ny, nz

		# Allocate coefficients and prepopulate image
		self.coeffs = <double *>PyMem_Malloc(nx * ny * nz * sizeof(double))
		if self.coeffs == <double *>NULL:
			raise MemoryError('Unable to allocate storage for coefficients')

		cdef double[:,:,:] coeffs
		cdef unsigned long ix, iy, iz

		try:
			# For convenience, treat the buffer as a 3-D array
			coeffs = <double[:nx,:ny,:nz:1]>self.coeffs

			# Just copy the linear interpolation coefficients
			for ix in range(nx):
				for iy in range(ny):
					for iz in range(nz):
						coeffs[ix,iy,iz] = img[ix,iy,iz]
		except Exception:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL
			raise

	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		return (self.ncx, self.ncy, self.ncz)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad, its gradient at a
		point p. If grad is NULL, the gradient is not evaluated.

		If the coordinates are out of bounds, evaluation will be
		attempted by Interpolator3D._evaluate().

		The method will always return True if evaluation was done
		locally, or the return value of Interpolator3D._evaluate() if
		the evaluation was delegated.
		'''
		cdef long nx, ny, nz
		cdef bint dograd = (grad != <point *>NULL)

		if self.coeffs == <double *>NULL:
			return Interpolator3D._evaluate(self, f, grad, p)

		# Linear coefficient shape is (nx, ny, nz)
		nx, ny, nz = self.ncx, self.ncy, self.ncz

		# Find the fractional and integer parts of the coordinates
		cdef long i, j, k
		cdef point t

		if not Interpolator3D.crdfrac(&t, &i, &j, &k, p, nx, ny, nz):
			return Interpolator3D._evaluate(self, f, grad, p)

		cdef:
			double tw[4]
			double uw[2]
			double dt[4]
			double du[2]
			double dv[2]

			double tt, uu, vv
			long ii, jj

		# Initialize the outputs
		f[0] = 0.0

		if dograd:
			grad.x = grad.y = grad.z = 0.0

		# Interpolate the x face
		ii = (i * self.ncy + j) * self.ncz + k
		jj = ii + self.ncz * self.ncy
		tt = 1.0 - t.x
		tw[0] = tt * self.coeffs[ii] + t.x * self.coeffs[jj]
		tw[1] = tt * self.coeffs[ii + 1] + t.x * self.coeffs[jj + 1]
		if dograd:
			dt[0] = self.coeffs[jj] - self.coeffs[ii]
			dt[1] = self.coeffs[jj + 1] - self.coeffs[ii + 1]
		ii += self.ncz
		jj += self.ncz
		tw[2] = tt * self.coeffs[ii] + t.x * self.coeffs[jj]
		tw[3] = tt * self.coeffs[ii + 1] + t.x * self.coeffs[jj + 1]
		if dograd:
			dt[2] = self.coeffs[jj] - self.coeffs[ii]
			dt[3] = self.coeffs[jj + 1] - self.coeffs[ii + 1]
		# Interpolate the y line
		uu = 1.0 - t.y
		uw[0] = uu * tw[0] + t.y * tw[2]
		uw[1] = uu * tw[1] + t.y * tw[3]
		if dograd:
			# These are the y derivatives at the line ends
			du[0] = tw[2] - tw[0]
			du[1] = tw[3] - tw[1]
			# Interpolate the x derivatives at the line ends
			dv[0] = uu * dt[0] + t.y * dt[2]
			dv[1] = uu * dt[1] + t.y * dt[3]

		# Interpolate the z value
		vv = 1.0 - t.z
		f[0] = vv * uw[0] + t.z * uw[1]

		if dograd:
			grad.z = uw[1] - uw[0]
			grad.y = vv * du[0] + t.z * du[1]
			grad.x = vv * dv[0] + t.z * dv[1]

		return True


cdef class Interpolator3D:
	'''
	An abstract class to manage a function, sampled on a 3-D grid, that can
	provide interpolated values of the function and its gradient anywhere
	in the bounds of the grid.
	'''
	cdef double *coeffs
	cdef unsigned long ncx, ncy, ncz
	cdef bint _usedef
	cdef double _default

	def __cinit__(self, *args, **kwargs):
		'''
		By default, clear the default value and the coefficient
		pointer.
		'''
		self._usedef = False
		self._default = 0.0
		self.ncx = self.ncy = self.ncz = 0
		self.coeffs = <double *>NULL


	def __init__(self, *args, **kwargs):
		'''
		Descendants must implement __init__ and not call this.
		'''
		raise NotImplementedError('Unable to initialize abstract Interpolator3D')


	def __dealloc__(self):
		'''
		Clear coefficient storage.
		'''
		if self.coeffs != <double *>NULL:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL


	@property
	def shape(self):
		'''
		A grid (nx, ny, nz) representing the shape of the image being
		interpolated.
		'''
		raise NotImplementedError('Abstract Interpolator3D has no shape')


	@property
	def default(self):
		'''
		The default value to return when attempting to evaluate the
		interpolated fumction outside the function bounds, or None if
		out-of-bounds evaluations should raise an exception.
		'''
		if self._usedef: return self._default
		else: return None

	@default.setter
	def default(self, val):
		if val is None:
			self._usedef = False
			self._default = 0.0
		self._default = float(val)
		self._usedef = True


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	@cython.embedsignature(True)
	def pathgrad(self, double[:,:] points not None, double hmax):
		'''
		Given control points specified as rows of an N-by-3 array of
		grid coordinates, evaluate the gradient of the path integral as
		computed in self.pathint. The output is an array with the same
		shape as points, where the entry output[i,j] is the derivative
		of the path integral with respect to coordinate j of control
		point i. By convention, the gradient for the first and last
		control points are 0.

		Gradients are evaluated analytically. However, because the
		gradient is of an integral which is calculated using the
		composite Simpson's rule, the the parameter hmax is used as in
		self.pathint to control the accuracy of the integrals.

		If the second dimension of points does not have length three,
		or if any control points fall outside the interpolation grid, a
		ValueError will be raised.
		'''
		cdef unsigned long npts = points.shape[0]
		if points.shape[1] != 3:
			raise ValueError('Length of second dimension of points must be 3')
		if npts < 3:
			raise ValueError('At least 3 control points are required for a gradient')

		cdef double[:,:] grad = np.zeros((npts, 3), dtype=np.float64, order='C')

		cdef unsigned long i, im1
		cdef point l, r, lg, rg

		# Make sure hmax has positive sign
		if hmax < 0: hmax = -hmax

		# Initialize the left point
		l = packpt(points[0,0], points[0,1], points[0,2])

		# Compute contributions to gradients segment by segment
		for i in range(1, npts):
			# Evaluate the right point
			r = packpt(points[i,0], points[i,1], points[i,2])

			# Contribute contributions at l and c for segment lr
			if not self.segrad(&lg, &rg, l, r, hmax):
				raise ValueError('Could not compute path gradient in segment %s -> %s' % (pt2tup(l), pt2tup(r)))

			# Add left and right gradient contributions
			im1 = i - 1
			grad[im1,0] += lg.x
			grad[im1,1] += lg.y
			grad[im1,2] += lg.z

			grad[i,0] += rg.x
			grad[i,1] += rg.y
			grad[i,2] += rg.z

			# Cycle the points for the next round
			l = r

		# Remove contributions at path endpoints
		grad[0,0] = grad[0,1] = grad[0,2] = 0.0
		im1 = npts - 1
		grad[im1,0] = grad[im1,1] = grad[im1,2] = 0.0

		return np.asarray(grad)


	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.cdivision(True)
	@cython.embedsignature(True)
	def pathint(self, double[:,:] points not None, double hmax):
		'''
		Given control points specified as rows of an N-by-3 array of
		grid coordinates, use the composite Simpson's rule to integrate
		the image associated with this interpolator instance along the
		piecewise linear path between the points.

		Each segment of the path will be subdivided into a minimum
		number of equal-length pieces such that the length of each
		piece does not exceed hmax.

		If the second dimension of points does not have length three,
		or if any control point falls outside the interpolation grid,
		a ValueError will be raised.
		'''
		cdef unsigned long npts = points.shape[0]
		if points.shape[1] != 3:
			raise ValueError('Length of second dimension of points must be 3')

		cdef unsigned long i
		cdef point ps, pe
		cdef double ival = 0.0, sval = 0.0

		# Make sure hmax has positive sign
		if hmax < 0: hmax = -hmax

		# Initialize the left point and its value
		ps = packpt(points[0,0], points[0,1], points[0,2])
		if not self._evaluate(&sval, NULL, ps):
			raise ValueError('Start point %s is out of bounds' % (pt2tup(ps),))

		for i in range(1, npts):
			# Pull the end point
			pe = packpt(points[i,0], points[i,1], points[i,2])
			# Add the contribution from this segment
			# This uses, then updates, the function value in sval
			if not self.segint(&ival, ps, pe, hmax, &sval):
				raise ValueError('Path %s -> %s is out of bounds' % (pt2tup(ps), pt2tup(pe)))
			# The next start point is the current end point
			ps = pe

		return ival


	@cython.cdivision(True)
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef bint segrad(self, point *sgrad, point *egrad,
			point s, point e, double hmax) nogil:
		'''
		Store, in lgrad and rgrad, the gradients (with respect to
		points s and e, respectively) of the integral of the this
		interpolated function along the line from point s to point e,
		each in grid coordinates.

		As in segint, the integral is computed using the composite
		Simpson's rule by dividing the segment into the smallest even
		number of equal-length pieces that each have a length no longer
		than hmax. The sign of hmax is ignored.

		Returns True if the integral was evaluated and False if any
		point along the line cannot be evaluated.

		If False is returned, neither sgrad or egrad will be modified.
		'''
		cdef point dp, p, agf, bgf, sgf
		cdef double l, h, fval = 0.0, sval = 0.0, sc = 0.0, ifn = 0.0
		cdef unsigned long i, n

		cdef double mpy[2]
		mpy[0], mpy[1] = 2.0, 4.0

		if hmax < 0: hmax = -hmax

		# Calculate the length of the segment
		dp = axpy(-1, s, e)
		l = ptnrm(dp)

		# Subdivide the step for the composite rule
		n = max(<unsigned long>(l / hmax), 2)
		# Step size must be less than h
		if n * hmax < l: n += 1
		# The step count must be even
		if n % 2: n += 1

		# Scale the walk direction and step size
		h = l / n
		iscal(h / l, &dp)

		# Evaluate the function and gradient at the start
		# Gradient at start fully contributes to path grad wrt a
		if not self._evaluate(&fval, &agf, s): return False
		# Gradient at start does not contribute to path grad wrt b
		bgf.x = bgf.y = bgf.z = 0.0

		# Include the interior points
		for i in range(1, n):
			p = axpy(i, dp, s)
			if not self._evaluate(&sval, &sgf, p): return False
			sc = mpy[i % 2]
			# Update the common function sum
			fval += sc * sval
			# Update contributions to a and b gradients
			ifn = <double>i / <double>n
			# Contribution to a gradient decreases with increasing i
			iaxpy(sc * (1.0 - ifn), sgf, &agf)
			# Contribution to b gradient increases with increasing i
			iaxpy(sc * ifn, sgf, &bgf)

		# Evaluate the endpoint
		if not self._evaluate(&sval, &sgf, e): return False

		# Include endpoint contributions
		fval += sval
		# Gradient at end does not contribute to path grad wrt a
		# Gradient at end fully contributes to path grad wrt b
		iaxpy(1.0, sgf, &bgf)

		# Scale the scalar term
		fval /= (3.0 * n * l)

		# First term in path gradients is gradient of integral step
		sgrad[0] = axpy(-1, e, s)
		iscal(fval, sgrad)
		egrad[0] = axpy(-1, s, e)
		iscal(fval, egrad)

		# Second term is gradient of function
		iaxpy(h / 3.0, agf, sgrad)
		iaxpy(h / 3.0, bgf, egrad)

		return True


	@cython.cdivision(True)
	@cython.boundscheck(False)
	@cython.wraparound(False)
	cdef bint segint(self, double *ival, point s,
			point e, double hmax, double *f) nogil:
		'''
		Add to the value stored in ival the integral of the function
		represented by this interpolator along the line from point s to
		point e, each in grid coordinates. The integral is computed
		using the composite Simpson's rule by dividing the segment into
		the smallest even number of equal-length pieces that each have
		a length no longer than hmax. The sign of hmax is ignored.

		If f is not NULL, it must hold the result of a call to

			self._evaluate(f, NULL, s).

		In this case, initial evaluation of function will not be
		attempted. On output, the value of a call to

			self._evaluate(f, NULL, e)

		will be stored in f. This allows calls to segint to be chained
		over multiple segments that share endpoints without multiple
		evaluations of the function at the same point.

		If f is NULL, the function will be evaluated everywhere.

		Returns True if the integral was evaluated and False if any
		point along the line cannot be evaluated.

		If False is returned, neither ival nor f will have changed.
		'''
		cdef point dp, p
		cdef double l, h, fval = 0.0, sval = 0.0
		cdef unsigned long i, n

		cdef double mpy[2]
		mpy[0], mpy[1] = 2.0, 4.0

		if hmax < 0: hmax = -hmax

		# Calculate the length of the segment
		dp = axpy(-1, s, e)
		l = ptnrm(dp)

		# Subdivide the step for the composite rule
		n = max(<unsigned long>(l / hmax), 2)
		# Step size must be less than h
		if n * hmax < l: n += 1
		# The step count must be even
		if n % 2: n += 1

		# Scale the walk direction and step size
		h = l / n
		iscal(h / l, &dp)

		# Copy a pre-evaluated starting point or evaluate
		if f != <double *>NULL: fval = f[0]
		elif not self._evaluate(&fval, NULL, s): return False

		# Include the interior points
		for i in range(1, n):
			p = axpy(i, dp, s)
			if not self._evaluate(&sval, NULL, p): return False
			fval += mpy[i % 2] * sval

		# Evaluate the endpoint
		if not self._evaluate(&sval, NULL, e): return False

		# Update f with the endpoint value
		if f != <double *>NULL: f[0] = sval

		# Include endpoint contribution and scale the sum
		fval += sval
		fval *= h / 3.0

		# Augment the output
		ival[0] += fval

		return True


	@staticmethod
	cdef bint crdfrac(point *t, long *i, long *j, long *k,
			point p, long nx, long ny, long nz) nogil:
		'''
		In i, j and k, stores the integer portions of p.x, p.y, and
		p.z, respectively, clipped to the grid

			[0, nx - 2] x [0, ny - 2], [0, nz - 2].

		In t.x, t.y, and t.z, stores the remainders

			t.x = p.x - i,
			t.y = p.y - j,
			t.z = p.z - k.

		Returns True if t.x, t.y and t.z are all in [0, 1] and False
		otherwise.
		'''
		i[0] = max(0, min(<long>p.x, nx - 2))
		j[0] = max(0, min(<long>p.y, ny - 2))
		k[0] = max(0, min(<long>p.z, nz - 2))

		t.x = p.x - <double>(i[0])
		t.y = p.y - <double>(j[0])
		t.z = p.z - <double>(k[0])

		return 0 <= t.x <= 1.0 and 0 <= t.y <= 1.0 and 0 <= t.z <= 1.0


	@cython.wraparound(False)
	@cython.boundscheck(False)
	cdef bint _evaluate(self, double *f, point *grad, point p) nogil:
		'''
		Evaluate, in f, a function and, in grad, its gradient at a
		point p given cubic b-spline coefficients c. If grad is NULL,
		the gradient is not evaluated.

		If the coordinates are out of bounds, self._default will be
		substituted if self._usedef is True. The gradient will be zero
		in this case.

		This method returns False if the coordinates are out of bounds
		and self._usedef is False, True otherwise.
		'''
		cdef long nx, ny, nz
		cdef bint dograd = (grad != <point *>NULL)

		if self._usedef:
			f[0] = self._default
			if dograd: grad.x = grad.y = grad.z = 0.0
			return True

		return False


	@cython.embedsignature(True)
	def evaluate(self, double x, double y, double z, bint grad=True):
		'''
		Evaluate and return the value of the image and, if grad is
		True, its gradient at grid coordinates (x, y, z).

		If coordinates are out of bounds, a ValueError will be raised
		unless self.default is not None.
		'''
		cdef double f
		cdef point g
		cdef point *gp

		if grad: gp = &g
		else: gp = <point *>NULL

		if not self._evaluate(&f, gp, packpt(x, y, z)):
			raise ValueError('Coordinates are out of bounds')
		if grad: return f, pt2tup(g)
		else: return f


	@cython.embedsignature(True)
	def gridimage(self, x, y, z):
		'''
		Evalute and return the interpolated image on the grid defined
		as the product of the floating-point grid coordinates in 1-D
		arrays x, y, and z.
		'''
		cdef double[:] cx, cy, cz
		cx = np.asarray(x, dtype=np.float64)
		cy = np.asarray(y, dtype=np.float64)
		cz = np.asarray(z, dtype=np.float64)

		cdef unsigned long nx, ny, nz, ix, iy, iz
		nx = cx.shape[0]
		ny = cy.shape[0]
		nz = cz.shape[0]

		cdef np.ndarray[np.float64_t, ndim=3] out
		out = np.empty((nx, ny, nz), dtype=np.float64, order='C')

		cdef double f = 0.0, xx, yy, zz
		cdef point crd

		for ix in range(nx):
			crd.x = cx[ix]
			for iy in range(ny):
				crd.y = cy[iy]
				for iz in range(nz):
					crd.z = cz[iz]
					if not self._evaluate(&f, NULL, crd):
						raise ValueError('Cannot evaluate image at %s' % (pt2tup(crd),))
					out[ix,iy,iz] = f

		return out


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


	@cython.embedsignature(True)
	def cartesian(self, double t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		return pt2tup(lintp(t, self._start, self._end))


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
			return v, t, u, 1 - t - u
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
	cdef point _lo, _hi, _length, _cell
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
	cdef bint _intersection(double *t, point l, point h, point s, point e):
		'''
		Low-level routine to compute intersection of a box, with low
		and high corners l and h, respectively, with a line segment
		that has starting point s and end point e.

		If the segment intersects the box, True is returned and the
		values t[0] and t[1] represent, respectively, the minimum and
		maximum lengths along the ray or segment (as a multiple of the
		length of the segment) that describe the points of
		intersection. The value t[0] may be negative if the ray or
		segment starts within the box. The value t[1] may exceed unity
		if a segment ends within the box.

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

		if tmax < max(0, tmin) or (tmin > 1):
			return False

		t[0] = tmin
		t[1] = tmax
		return True

	@cython.embedsignature(True)
	def intersection(self, Segment3D seg not None):
		'''
		Returns the lengths tmin and tmax along the given Segment3D seg
		at which the segment enters and exits the box, as a multiple of
		the segment length. If the box does not intersect the segment,
		returns None.

		If the segment starts within the box, tmin will be negative. If
		the segment ends within the box, tmax will exceed the segment
		length.
		'''
		cdef double tlims[2]
		if Box3D._intersection(tlims, self._lo, self._hi, seg._start, seg._end):
			return tlims[0], tlims[1]
		else: return None


	@cython.boundscheck(False)
	@cython.wraparound(False)
	@cython.cdivision(True)
	@cython.embedsignature(True)
	def descent(self, start, end, Interpolator3D field,
			unsigned long cycles = 1,
			double step=1.0, double tol=1e-3,
			double c=0.5, double tau=0.5, bint report=False):
		'''
		Perform a steepest-descent walk from a point p through the
		given field as an Interpolator3D instance capable of evaluating
		both the field, f, and its gradient at arbitrary points within
		this box.

		The walk proceeds from some point p to another point q by
		performing an Armijo backtracking line search along the
		negative gradient of the field. Let h = norm(self.cell) be the
		diameter of a cell in this box, g = grad(f)(p) be the gradent
		of f at p, and m = |g| be the norm of the gradient. The search
		will select the next point q = (p - alpha * g / m), where the
		factor alpha = tau**k * step * h for the smallest integer k
		such that

			f(p - alpha * g / m) - f(p) <= -alpha * c * m.

		The walk terminates when one of:

		1. A point q of the path comes within h of the point end,
		2. The path runs off the edge of the grid,
		3. The factor tau**k < tol when finding the next step, or
		4. The same grid cell is encountered more than cycles times.

		If the argument "report" is True, a second return value, a
		string with value 'destination', 'boundary', 'stationary', or
		'cycle' will be returned to indicate the relevant termination
		criterion (1, 2, 3 or 4, respectively).

		A ValueError will be raised if the field has the wrong shape.
		'''
		# Make sure the field is double-array compatible
		cdef long nx, ny, nz
		nx, ny, nz = field.shape
		if (nx != self.nx or ny != self.ny or nz != self.nz):
			raise ValueError('Shape of field must be %s' % (self.ncell,))

		cdef point p, t, gf, pd, np
		cdef long i, j, k, ti, tj, tk
		cdef double fv, tv, m, lim, alpha
		cdef unsigned long hc

		# Make sure the provided start and end points are valid
		tup2pt(&p, start)
		tup2pt(&t, end)

		# Convert start and end points to grid coordinates
		p = self._cart2cell(p.x, p.y, p.z)
		t = self._cart2cell(t.x, t.y, t.z)

		# Maximum permissible grid coordinates
		nx, ny, nz = self.nx - 1, self.ny - 1, self.nz - 1

		# Include the start point in the hit list
		hits = [ self.cell2cart(p.x, p.y, p.z) ]
		reason = None

		# Keep track of encountered cells for cycle breaks
		hitct = { }

		# Find cell for target points (may be out of bounds)
		ti, tj, tk = <long>t.x, <long>t.y, <long>t.z

		while True:
			# Find the cell for the current test point
			i, j, k = <long>p.x, <long>p.y, <long>p.z

			# Increment and check the cycle counter
			hc = hitct.get((i,j,k), 0) + 1
			if hc > cycles:
				reason = 'cycle'
				break
			hitct[i,j,k] = hc

			if ptdst(p, t) <= 1.0 or (ti == i and tj == j and tk == k):
				# Close enough to destination to make a beeline
				hits.append(self.cell2cart(t.x, t.y, t.z))
				reason = 'destination'
				break

			# Find the function and gradient at the current point
			if not field._evaluate(&fv, &gf, p):
				# Point is out of bounds
				reason = 'boundary'
				break

			# Find the magnitude of the gradient and the search direction
			m = ptnrm(gf)
			pd = scal(-1.0 / m, gf)

			# Establish the baseline Armijo bound
			lim = c * m
			alpha = step

			while alpha >= tol:
				np = axpy(alpha, pd, p)
				# Find the next value
				if field._evaluate(&tv, NULL, np):
					# Stop if Armijo condition is satisfied
					if tv - fv <= -alpha * lim: break
				# Test point out of bounds or failed to satisfy Armijo
				alpha *= tau

			if alpha < tol:
				# Could not find suitable point
				reason = 'stationary'
				break

			# Advance to and record the satisfactory test point
			p = np
			hits.append(self.cell2cart(p.x, p.y, p.z))

		if report: return hits, reason
		else: return hits


	@cython.embedsignature(True)
	cdef object _raymarcher(self, point start, point end, double step=realeps):
		'''
		Helper for Box3D.raymarcher that performs a single march for a
		segment from point start to point end.
		'''
		# Make sure the segment intersects this box
		cdef double tlims[2]

		intersections = { }
		if not Box3D._intersection(tlims, self._lo, self._hi, start, end):
			return intersections

		if step <= 0: step = -step

		# Keep track of accumulated and max length
		cdef double t = max(0, tlims[0])
		cdef double tmax = min(tlims[1], 1)
		# This is a dynamically grown step into the next cell
		cdef double cstep

		cdef point lo, hi
		cdef long i, j, k, ni, nj, nk

		# Find the cell that contains the current test point
		self._cellForPoint(&i, &j, &k, lintp(t, start, end))

		while t < tmax:
			self._boundsForCell(&lo, &hi, i, j, k)
			if not Box3D._intersection(tlims, lo, hi, start, end):
				stt = pt2tup(start)
				edt = pt2tup(end)
				raise ValueError('Segment %s -> %s fails to intersect cell %s' % (stt, edt, (i,j,k)))

			if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
				# Record a hit inside the grid
				key = i, j, k
				val = max(0, tlims[0]), min(tlims[1], 1)
				intersections[i,j,k] = val

			# Advance t; make sure it lands in another cell
			t = tlims[1]
			cstep = step
			while t < tmax:
				# Find the cell containing the point
				self._cellForPoint(&ni, &nj, &nk, lintp(t, start, end))
				if i != ni or j != nj or k != nk:
					# Found a new cell; record and move on
					i, j, k = ni, nj, nk
					break
				# Otherwise, stuck in same cell; bump t
				t += cstep
				# Increase step for next time
				cstep *= 2

		return intersections

	@cython.embedsignature(True)
	def raymarcher(self, p, double step=realeps):
		'''
		Marches along the given p, which is either a single Segment3D
		instance or a 2-D array of shape (N, 3), where N >= 2, that
		defines control points of a piecewise linear curve.

		A march of a single linear segment accumulates a map of the
		form (i, j, k) -> (tmin, tmax), where (i, j, k) is the index of
		a cell that intersects the segment and (tmin, tmax) are the
		minimum and maximum lengths (as fractions of the segment
		length) along which the segment and cell intersect.

		If p is a Segment3D instance or an array of shape (2, 3), a
		single map will be returned. If p is an array of shape (N, 3)
		for N > 2, a list of (N - 1) of maps will be returned, with
		maps[i] providing the intersection map for the segment from
		p[i] to p[i+1].
 
		As a segment exits each encountered cell, a step along the
		segment is taken to advance into another intersecting cell. The
		length of the step will be, in units of the segment length,

			step * sum(2**i for i in range(q)),

		where the q is chosen at each step as the minimum nonnegative
		integer that guarantees advancement to another cell. Because
		this step may be nonzero, cells which intersect the segment
		over a total length less than step may be excluded from the
		intersection map.
		'''
		cdef double[:,:] pts
		cdef Segment3D seg
		cdef point s, e

		if isinstance(p, Segment3D):
			seg = <Segment3D>p
			return self._raymarcher(seg._start, seg._end, step)

		pts = np.asarray(p, dtype=np.float64)
		if pts.shape[0] < 2 or pts.shape[1] != 3:
			raise ValueError('Argument "p" must have shape (N,3), N >= 2')

		# Capture the start of the first segment
		s = packpt(pts[0,0], pts[0,1], pts[0,2])

		if pts.shape[0] == 2:
			# Special case: a single segment in array form
			e = packpt(pts[1,0], pts[1,1], pts[1,2])
			return self._raymarcher(s, e, step)

		# Accumulate results for multiple segments
		results = [ ]

		cdef unsigned long i
		for i in range(1, pts.shape[0]):
			# Build current segment and march
			e = packpt(pts[i,0], pts[i,1], pts[i,2])
			results.append(self._raymarcher(s, e, step))
			# Move end to start for next round
			s = e

		return results

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
