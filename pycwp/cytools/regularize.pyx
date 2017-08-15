# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

@cython.embedsignature(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def epr(img, fwd=False):
	'''
	For a 3-D scalar image img with shape (nx, ny, nz), compute the EPR
	norm [Zhang and Zhang, J. Appl. Geophys. 138 (2017), pp. 143--153]


	  EPR = Sum_{ijk} (Dx^2 / (Dx^2 + 1) +
	  			Dy^2 / (Dy^2 + 1) + Dz^2 / (Dz^2 + 1)),

	for all (i, j, k) in the grid [0, 0, 0] x [nx - 1, ny - 1, nz - 1],
	where Dx, Dy and Dz are finite-difference approximations to the x, y
	and z derivatives of img, respectively. If fwd is False, backward
	differencing is used; otherwise, forward differencing is used.

	The (nx * ny * nz)-dimensional gradient of the discrete EPR norm, with
	respect to changes in image values, is also computed and returned.

	For simplicity, it is assumed that both grad(img)(i,j,k) and the
	partial derivative of TV(img) with respect to img[i,j,k] vanish when
	point (i, j, k) resides on the boundary of the image grid.

	A ValueError will be raised if any of nx, ny, or nz is less than 3.
	'''
	cdef double[:,:,:] cimg = np.asarray(img, dtype=np.float64)

	cdef unsigned long nx, ny, nz
	nx, ny, nz = cimg.shape[0], cimg.shape[1], cimg.shape[2]

	if min(nx, ny, nz) < 3:
		raise ValueError('Minimum size of img is (3, 3, 3)')

	cdef double[:,:,:] gimg = np.zeros((nx, ny, nz), dtype=np.float64)

	cdef unsigned long i, j, k, im, jm, km
	cdef double cv, dxf, dyf, dzf, dxxf, dyyf, dzzf
	cdef double eprnorm = 0.0, gnorm, gv

	cdef long fdir = 1 if not fwd else -1

	for i in range(1, nx - 1):
		im = i - fdir
		for j in range(1, ny - 1):
			jm = j - fdir
			for k in range(1, nz - 1):
				km = k - fdir

				cv = cimg[i,j,k]

				# If not fwd, this is the backward difference
				# Otherwise, it is the negative forward difference
				dxf = cv - cimg[im,j,k]
				dyf = cv - cimg[i,jm,k]
				dzf = cv - cimg[i,j,km]

				dxxf = dxf * dxf
				dyyf = dyf * dyf
				dzzf = dzf * dzf

				# Compute norm
				gnorm = dxxf / (dxxf + 1.0)
				gnorm += dyyf / (dyyf + 1.0)
				gnorm += dzzf / (dzzf + 1.0)

				# Add this norm to the total
				eprnorm += gnorm

				# Finish denominators for gradient terms
				dxxf += 1.0
				dyyf += 1.0
				dzzf += 1.0

				# Accumulate the gradient terms (first difference)
				gimg[i,j,k] += 2 * dxf / (dxxf * dxxf)
				gimg[i,j,k] += 2 * dyf / (dyyf * dyyf)
				gimg[i,j,k] += 2 * dzf / (dzzf * dzzf)

				# Accumulate shifted gradient terms
				gimg[im,j,k] -= 2 * dxf / (dxxf * dxxf)
				gimg[i,jm,k] -= 2 * dyf / (dyyf * dyyf)
				gimg[i,j,km] -= 2 * dzf / (dzzf * dzzf)

	# Return the norm and its gradient
	return eprnorm, np.asarray(gimg)


@cython.embedsignature(True)
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def totvar(img, fwd=False, eps=0.01):
	'''
	For a three-dimensional scalar image img with shape (nx, ny, nz),
	compute and return the total-variation (TV) norm

	  TV(img) = Sum_{ijk} |grad(img)(i,j,k)|

	for all (i, j, k) in the grid [0, 0, 0] x [nx - 1, ny - 1, nz - 1]. The
	3-D gradient of the image at (i, j, k) is approximated using forward
	differences if fwd is True; otherwise, backward differences are used.

	The (nx * ny * nz)-dimensional gradient of the discrete TV norm, with
	respect to changes in image values, is also computed and returned.
	Terms of the gradient contain denominators like the magnitude of the
	three-dimensional gradient of the image; to avoid singularities, the
	gradient magnitude is approximated as

	  |grad(img)| -> |grad(img)| + eps / 2.

	For simplicity, it is assumed that both grad(img)(i,j,k) and the
	partial derivative of TV(img) with respect to img[i,j,k] vanish when
	point (i, j, k) resides on the boundary of the image grid.

	A ValueError will be raised if any of nx, ny, or nz is less than 3.
	'''
	cdef double[:,:,:] cimg = np.asarray(img, dtype=np.float64)

	cdef unsigned long nx, ny, nz
	nx, ny, nz = cimg.shape[0], cimg.shape[1], cimg.shape[2]

	if min(nx, ny, nz) < 3:
		raise ValueError('Minimum size of img is (3, 3, 3)')

	cdef double[:,:,:] gimg = np.zeros((nx, ny, nz), dtype=np.float64)

	cdef unsigned long i, j, k, im, jm, km
	cdef double cv, dxf, dyf, dzf
	cdef double tvnorm = 0.0, gnorm, heps = 0.5 * eps, gv

	cdef long fdir = 1 if not fwd else -1

	for i in range(1, nx - 1):
		im = i - fdir
		for j in range(1, ny - 1):
			jm = j - fdir
			for k in range(1, nz - 1):
				km = k - fdir

				cv = cimg[i,j,k]

				# If not fwd, this is the backward difference
				# Otherwise, it is the negative forward difference
				dxf = cv - cimg[im,j,k]
				dyf = cv - cimg[i,jm,k]
				dzf = cv - cimg[i,j,km]

				# Compute norm of this gradient
				gnorm = sqrt(dxf * dxf + dyf * dyf + dzf * dzf)

				# Add this norm to the total
				tvnorm += gnorm

				# Offset norm away from zero
				gnorm += heps

				# Compute the central term of the gradient
				gimg[i,j,k] += (dxf + dyf + dzf) / gnorm

				# Compute shifted terms
				gimg[im,j,k] -= dxf / gnorm
				gimg[i,jm,k] -= dyf / gnorm
				gimg[i,j,km] -= dzf / gnorm

	# Return the norm and its gradient
	return tvnorm, np.asarray(gimg)


@cython.embedsignature(True)
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
