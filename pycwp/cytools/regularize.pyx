import cython
cimport cython

from cython cimport floating as real

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def totvar(img, fwd=False, eps=1e-12):
	'''
	For a three-dimensional scalar image img with shape (nx, ny, nz),
	compute the total variation norm

		TV(img) = Sum(|grad(img)(i,j,k)|)

	for all (i, j, k) in [0, 0, 0] x [nx - 1, ny - 1, nz - 1]. The gradient
	of the discrete norm will also be computed at every point on the image
	grid.

	All derivatives are approximated using backward (if fwd is False) or
	forward (if fwd is True). The gradient is computed analytically based
	on the discrete, approximate representation of the TV norm.

	For simplicity, it is assumed that grad(img) == 0 and grad(TV)(img) on
	the boundaries of the image grid (i.e., i, j, k == 0, i == nx - 1,
	j == ny - 1 or k == nz - 1).

	The image will be converted to a 64-bit floating-point array for
	computation.

	A ValueError will be raised if any of nx, ny, or nz is less than 3.
	'''
	cdef double[:,:,:] cimg = np.asarray(img, dtype=np.float64)

	cdef unsigned long nx, ny, nz
	nx, ny, nz = cimg.shape[0], cimg.shape[1], cimg.shape[2]

	if min(nx, ny, nz) < 3:
		raise ValueError('Minimum size of img is (3, 3, 3)')

	cdef double[:,:,:] gimg = np.zeros((nx, ny, nz), dtype=np.float64)

	cdef unsigned long i, j, k, ip, im, jp, jm, kp, km
	cdef double cv, dxf, dyf, dzf, dxp, dyp, dzp
	cdef double tvnorm = 0.0, gnorm, ctrm, heps = 0.5 * eps

	cdef long fdir = 1 if not fwd else -1

	for i in range(1, nx - 1):
		ip = i + fdir
		im = i - fdir
		for j in range(1, ny - 1):
			jp = j + fdir
			jm = j - fdir
			for k in range(1, nz - 1):
				kp = k + fdir
				km = k - fdir

				cv = cimg[i,j,k]

				# If not fwd, this is the backward difference
				# Otherwise, it is the negative forward difference
				dxf = cv - cimg[im,j,k]
				dyf = cv - cimg[i,jm,k]
				dzf = cv - cimg[i,j,km]

				# If not fwd, this is the forward difference
				# Otherwise, it is the negative backward difference
				dxp = cimg[ip,j,k] - cv
				dyp = cimg[i,jp,k] - cv
				dzp = cimg[i,j,kp] - cv

				# Compute norm of this gradient
				gnorm = sqrt(dxf * dxf + dyf * dyf + dzf * dzf)

				# Add this norm to the total
				tvnorm += gnorm

				# Offset norm away from zero
				gnorm += heps

				# Compute the central term of the gradient
				# Sign must be adjusted for forward differencing
				gimg[i,j,k] = (dxf + dyf + dzf) / gnorm

				# Backward differences are only needed squared
				# Shift square differences away from zero
				dxf = dxf * dxf + heps
				dyf = dyf * dyf + heps
				dzf = dzf * dzf + heps

				# Half epsilon in two "f" terms combine
				# to make full epsilon under square root
				gimg[i,j,k] -= dxp / sqrt(dxp * dxp + dyf + dzf)
				gimg[i,j,k] -= dyp / sqrt(dxf + dyp * dyp + dzf)
				gimg[i,j,k] -= dzp / sqrt(dxf + dyf + dzp * dzp)

	# Return the norm and its gradient
	return tvnorm, np.asarray(gimg)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def tikhonov(img):
	'''
	For a three-dimensional scalar image img with shape (nx, ny, nz),
	compute the Tikhonov regularization term 0.5 * ||img||^2 and its
	gradient (which is just img).

	The returned gradient may be a view on img.
	'''
	cdef double[:,:,:] cimg = np.asarray(img, dtype=np.float64)

	cdef unsigned long nx, ny, nz
	nx, ny, nz = cimg.shape[0], cimg.shape[1], cimg.shape[2]

	if min(nx, ny, nz) < 3:
		raise ValueError('Minimum image size is (3, 3, 3)')

	cdef unsigned long i, j, k
	cdef double tnorm = 0.0, cv

	for i in range(nx):
		for j in range(ny):
			for k in range(nz):
				cv = cimg[i,j,k]
				tnorm += 0.5 * cv * cv

	return tnorm, np.asarray(cimg)
