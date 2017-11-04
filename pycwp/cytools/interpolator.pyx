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

import itertools

from cpython.mem cimport PyMem_Malloc, PyMem_Free

from ptutils cimport *
from interpolator cimport *

def savgol(n, order=2):
	'''
	Compute a Savitzky-Golay filter for a cubic tile of half-width n with
	the given interpolating order. The tile size n and order must satisfy

		2 * n + 1 > order.

	The return value is a four-tuple of filter stencils (b, bx, by, bz),
	where convolution of an image with b yields the filtered image and
	convolution of the image with bx, by or bz yields the x, y or z
	derivatives of the filtered image, respectively.
	'''
	if 2 * n + 1 <= order:
		raise ValueError('Tile size n must satisfy 2 * n + 1 > order')

	a = [ ]

	for x, y, z in itertools.product(range(-n, n+1), repeat=3):
		ar = [ ]
		for i, j, k in itertools.product(range(order+1), repeat=3):
			ar.append(x**i * y**j * z**k)
		a.append(ar)

	b = np.linalg.pinv(a)

	# Row indices of the function and first-derivative filters
	dfidx = [ 0, (order + 1)**2, (order + 1), 1 ]
	return tuple(b[r].reshape((2 * n + 1,)*3, order='C') for r in dfidx)


cdef class LagrangeInterpolator3D(Interpolator3D):
	'''
	An Interpolator3D that implements piecewise Lagrange interpolation of
	third degree along each axis of a 3-D function.
	'''
	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image, default=None):
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
			raise

		self.default = default


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
	@cython.wraparound(False)
	@cython.boundscheck(False)
	@cython.embedsignature(True)
	def __init__(self, image, default=None, sgfilt=None):
		'''
		Construct a constrained Hermite cubic interpolator for the
		given 3-D floating-point image.

		If sgfilt is not None, it should be a tuple (n, order) passed
		as arguments to the savgol filter function. The function and
		derivative filters returned by savgol(*sgfilt) will be used to
		produce the values and derivatives used to define the Hermite
		interpolation.
		'''
		# Make sure the image is a valid Numpy array
		image = np.asarray(image, dtype=np.float64)

		cdef double [:,:,:] img = image
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
		cdef double[:,:,:] cimg
		cdef unsigned long lx, ly, lz, i, j, k, l, ii, jj, kk

		# Try to build the coefficient matrix
		try:
			coeffs = <double[:nx,:ny,:nz,:self.nval:1]>self.coeffs
			if not sgfilt:
				# Use finite differences for the derivatives
				HermiteInterpolator3D.img2coeffs(coeffs, img)
			else:
				try:
					import pyfftw
				except ImportError:
					from scipy.fftpack import fftn, ifftn
					empty = np.empty
				else:
					fftn = pyfftw.interfaces.scipy_fftpack.fftn
					ifftn = pyfftw.interfaces.scipy_fftpack.ifftn
					empty = pyfftw.empty_aligned

					pyfftw.interfaces.cache.enable()
					pyfftw.interfaces.cache.set_keepalive_time(5.)

				from scipy.signal import correlate

				# Use unfiltered image near boundary,
				# but let derivatives roll off smoothly
				coeffs[:,:,:,0] = img
				coeffs[:,:,:,1:4] = 0.

				l = 0
				for ba in savgol(*sgfilt):
					cimg = correlate(image, ba, mode='valid')
					lx = (coeffs.shape[0] - cimg.shape[0]) // 2
					ly = (coeffs.shape[1] - cimg.shape[1]) // 2
					lz = (coeffs.shape[2] - cimg.shape[2]) // 2
					for i in range(cimg.shape[0]):
						ii = i + lx
						for j in range(cimg.shape[1]):
							jj = j + ly
							for k in range(cimg.shape[2]):
								kk = k + lz
								coeffs[ii,jj,kk,l] = cimg[i,j,k]
					l += 1
		except Exception:
			PyMem_Free(self.coeffs)
			self.coeffs = <double *>NULL
			raise

		self.default = default


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
	def __init__(self, image, default=None):
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
			raise

		self.default = default


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
	def __init__(self, image, default=None):
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

		self.default = default

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
			return
		self._default = float(val)
		self._usedef = True


	@staticmethod
	cdef bint crdfrac(point *t, long *i, long *j, long *k,
			point p, long nx, long ny, long nz) nogil:
		'''
		In i, j and k, stores the integer portions of p.x, p.y, and
		p.z, respectively, clipped to the grid

			[0, nx - 2] x [0, ny - 2] x [0, nz - 2].

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
	def gridimage(self, x, y, z, bint grad=False):
		'''
		Evalute and return the interpolated image on the grid defined
		as the product of the floating-point grid coordinates in 1-D
		arrays x, y, and z.

		If grad is True, an array of shape (nx, ny, nz, 3) will be
		returned along with the interpolated grid, where the gradient
		at each (x, y, z) sample is provided along the ultimate axis.
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

		# Space for the gradient, if needed
		cdef np.ndarray[np.float64_t, ndim=4] grout
		if grad: grout = np.empty((nx, ny, nz, 3), dtype=np.float64, order='C')

		cdef double f = 0.0, xx, yy, zz
		cdef point crd
		cdef point gpoint
		cdef point *gfp = <point *>NULL

		if grad: gfp = &gpoint

		for ix in range(nx):
			crd.x = cx[ix]
			for iy in range(ny):
				crd.y = cy[iy]
				for iz in range(nz):
					crd.z = cz[iz]
					if not self._evaluate(&f, gfp, crd):
						raise ValueError('Cannot evaluate image at %s' % (pt2tup(crd),))
					out[ix,iy,iz] = f
					if grad:
						grout[ix,iy,iz,0] = gfp.x
						grout[ix,iy,iz,1] = gfp.y
						grout[ix,iy,iz,2] = gfp.z

		if not grad: return out
		return out, grout
